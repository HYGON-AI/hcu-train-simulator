# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""GLM-5 DSA compute lowering registrations."""

from hcu_train_simulator.modeling.architecture import (
    attention_score_dim,
    value_head_dim,
)
from hcu_train_simulator.modeling.compute_registry import (
    register_attention_lowering,
    register_gemm_shape,
)
from hcu_train_simulator.modeling.spec import DSASelfAttention


@register_gemm_shape("dsa_index_q")
def dsa_index_q_shape(model):
    return (
        model.transformer_compute_count(),
        model.index_n_heads * model.index_head_dim,
        model.q_lora_rank,
        model.token_count(),
    )


@register_gemm_shape("dsa_index_k")
def dsa_index_k_shape(model):
    return (
        model.transformer_compute_count(),
        model.index_head_dim,
        model.hidden_size,
        model.token_count(),
    )


@register_gemm_shape("dsa_index_weights")
def dsa_index_weights_shape(model):
    return (
        model.transformer_compute_count(),
        model.index_n_heads,
        model.hidden_size,
        model.token_count(),
    )


@register_gemm_shape("dsa_index_score")
def dsa_index_score_shape(model):
    return (
        model.transformer_compute_count() * model.micro_batch_size * model.index_n_heads,
        model.local_seq_length(),
        model.index_head_dim,
        model.seq_length,
    )


def _add_sparse_attention(model, result, spec, module_path, compute_count):
    # A CP rank owns local queries and attends to the globally selected keys.
    local_seq = model.local_seq_length()
    sparse_k = max(1.0, min(model.index_topk, model.seq_length))
    heads = model.num_attention_heads / model.tp_size
    dimensions = attention_score_dim(model) + value_head_dim(model)
    base = model.micro_batch_size * local_seq * sparse_k * heads * dimensions
    forward_flops = 2 * base
    backward_flops = 5 * base
    attention_tflops = model.compute_tflops(use_fp8=model.use_fp8_training)
    forward_ms = forward_flops * compute_count / (
        attention_tflops * model.gemm_efficiency
    ) / 1e9
    backward_ms = backward_flops * compute_count / (
        attention_tflops * model.gemm_efficiency * 0.88
    ) / 1e9
    row = {
        "model_part": "dsa_sparse_attn",
        "b": model.micro_batch_size,
        "m": "/",
        "n": "/",
        "k": "/",
        "shape": (
            f"batch={model.micro_batch_size}, seq={local_seq}, "
            f"topk={int(sparse_k)}, heads={int(heads)}, "
            f"head_dim={attention_score_dim(model)}"
        ),
        "compute_count": compute_count,
        "seq": local_seq,
        "kv_seq": int(sparse_k),
        "heads": int(heads),
        "head_dim": attention_score_dim(model),
        "value_head_dim": value_head_dim(model),
        "causal": True,
        "forward_ms": forward_ms,
        "backward_ms": backward_ms,
        "op_type": "flash_attention",
    }
    model._attach_spec_metadata(row, spec, module_path)
    result.append(row)


@register_attention_lowering(DSASelfAttention, first_gemm_part="mla_q_down")
def lower_dsa_attention(model, result, attention, module_path, compute_count, count_scale):
    submodules = attention.submodules
    attention_path = f"{module_path}.self_attention"
    for part, part_spec, part_name in (
        ("mla_q_down", submodules.linear_q_down_proj, "linear_q_down_proj"),
        ("mla_q_up", submodules.linear_q_up_proj, "linear_q_up_proj"),
        ("mla_kv_down", submodules.linear_kv_down_proj, "linear_kv_down_proj"),
        ("mla_kv_up", submodules.linear_kv_up_proj, "linear_kv_up_proj"),
    ):
        model.add_gemm_part(
            result,
            part,
            count_scale=count_scale,
            module_spec=part_spec,
            module_path=f"{attention_path}.{part_name}",
        )

    indexer = submodules.indexer
    if indexer is not None:
        indexer_path = f"{attention_path}.indexer"
        index = indexer.submodules
        for part, part_spec, part_name in (
            ("dsa_index_q", index.linear_q_proj, "linear_q_proj"),
            ("dsa_index_k", index.linear_k_proj, "linear_k_proj"),
            ("dsa_index_weights", index.weights_proj, "weights_proj"),
            ("dsa_index_score", index.score, "score"),
        ):
            model.add_gemm_part(
                result,
                part,
                count_scale=count_scale,
                module_spec=part_spec,
                module_path=f"{indexer_path}.{part_name}",
            )
        model.add_non_gemm_part(
            result,
            "dsa_index_k_norm",
            model.token_count() * model.index_head_dim,
            compute_count,
            "layernorm",
            module_spec=index.k_layernorm,
            module_path=f"{indexer_path}.k_layernorm",
        )
        index_score_elements = (
            model.micro_batch_size
            * model.local_seq_length()
            * model.seq_length
        )
        model.add_non_gemm_part(
            result,
            "dsa_index_score_reduce",
            index_score_elements * model.index_n_heads,
            compute_count,
            "dsa_score_reduce",
            covered_by_measured="dsa_index_score",
            module_spec=index.score,
            module_path=f"{indexer_path}.score_reduce",
        )
        model.add_non_gemm_part(
            result,
            "dsa_topk",
            index_score_elements,
            compute_count,
            "dsa_topk",
            module_spec=index.topk,
            module_path=f"{indexer_path}.topk",
        )
        model.add_non_gemm_part(
            result,
            "dsa_index_rope",
            model.token_count()
            * (model.index_n_heads + 1)
            * model.qk_rope_head_dim,
            compute_count,
            "rope",
            module_spec=indexer,
            module_path=f"{indexer_path}.rope",
        )

    _add_sparse_attention(
        model,
        result,
        submodules.core_attention,
        f"{attention_path}.core_attention",
        compute_count,
    )
    model.add_non_gemm_part(
        result,
        "rope",
        model.rope_element_count(),
        compute_count,
        "rope",
        module_spec=attention,
        module_path=f"{attention_path}.rope",
    )
    sparse_softmax_elements = (
        model.micro_batch_size
        * model.num_attention_heads
        / model.tp_size
        * model.local_seq_length()
        * max(1.0, min(model.index_topk, model.seq_length))
    )
    model.add_non_gemm_part(
        result,
        "dsa_attention_softmax",
        sparse_softmax_elements,
        compute_count,
        "attention_softmax",
        covered_by_measured="dsa_sparse_attn",
        module_spec=submodules.core_attention,
        module_path=f"{attention_path}.core_attention",
    )
    model.add_gemm_part(
        result,
        "attn_proj",
        count_scale=count_scale,
        module_spec=submodules.linear_proj,
        module_path=f"{attention_path}.linear_proj",
    )
