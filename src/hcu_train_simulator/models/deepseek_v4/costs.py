# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Lower DeepSeek V4-family specs to memory estimates, compute rows and FLOPs."""

from hcu_train_simulator.modeling.compute_registry import (
    register_attention_lowering,
    register_component_lowering,
)
from hcu_train_simulator.modeling.cost_registry import (
    register_activation_cost,
    register_parameter_cost,
)
from hcu_train_simulator.modeling.module_cost import ActivationEstimate, ParameterEstimate
from hcu_train_simulator.modeling.statistics_registry import (
    register_attention_flops,
    register_component_flops,
)

from .spec import (
    CompressedAttention,
    DeepSeekVision,
    EngramTable,
    HyperConnection,
    Projection,
    TokenOperation,
)


def _weights(p):
    return p["input_width"] * p["output_width"] / p["groups"]


def _token_count(seq, batch, divisor):
    return seq // divisor * batch


@register_parameter_cost(module=Projection)
def projection_parameters(model, spec, sharded):
    p = spec.params
    tp = model.tp_size if sharded and p["tp"] and not p["replicate_weights"] else 1
    return ParameterEstimate(((_weights(p) + p["tp_extras"]) / tp + p["extras"]) * p["copies"])


@register_parameter_cost(module=EngramTable)
def engram_parameters(model, spec, sharded):
    # Use the MoE rank group as a proxy for the dedicated Engram group.
    # EDP therefore synchronizes replicas of the same table partition.
    ranks = model.ep_size * model.etp_size if sharded else 1
    elements = spec.params["rows"] * spec.params["head_dim"] / ranks
    return ParameterEstimate(elements, elements)


def _parallel_divisor(model, sequence_parallel):
    return model.tp_size if sequence_parallel and model.sequence_parallel else 1


@register_activation_cost(module=Projection)
def projection_activations(model, spec, layer_uses_moe):
    p = spec.params
    tp = model.tp_size if p["tp"] else 1
    if p.get("vision"):
        seq, images = _vision_dimensions(model)
        # The vision stage is independent of language context sharding.
        tokens = seq / p["token_divisor"] * images * model.cp_size
    else:
        tokens = _token_count(model.seq_length, model.micro_batch_size, p["token_divisor"])
        tokens /= _parallel_divisor(model, p["sequence_parallel"])
    input_shard = tp if p["row_parallel"] else 1
    output_shard = 1 if p["row_parallel"] and p["groups"] == 1 else tp
    width = p["input_width"] / input_shard + p["output_width"] / output_shard
    return ActivationEstimate(tokens * width * p["copies"])


@register_activation_cost(module=EngramTable)
def engram_activations(model, spec, layer_uses_moe):
    return ActivationEstimate(model._tokens() * spec.params["lookup_width"])


def causal_pairs(seq, ratio=1, limit=None):
    """Sum min(floor((q + 1) / ratio), limit) over causal query positions."""
    cap = min(limit, seq // ratio) if limit is not None else seq // ratio
    return ratio * cap * (cap - 1) // 2 + cap * (seq - cap * ratio + 1)


def _attention_pairs(model, spec, seq):
    p = spec.params
    pairs = causal_pairs(seq, limit=model.config.transformer.architecture_options["window"])
    if p["ratio"]:
        pairs += causal_pairs(seq, p["ratio"], model.index_topk if p["indexed"] else None)
    return pairs


def _index_pairs(spec, seq):
    p = spec.params
    return causal_pairs(seq, p["ratio"], p["candidate_limit"] or None) if p["runs_indexer"] else 0


@register_activation_cost(module=CompressedAttention)
def attention_activations(model, spec, layer_uses_moe):
    p = spec.params
    tokens = model._tokens()
    local_heads = model.num_attention_heads / model.tp_size
    # Flash attention retains log-sum-exp per head and a single shared KV.
    saved = tokens * (2 * local_heads + model.kv_channels)
    workspace = 0
    if p["ratio"] and model.cp_size > 1:
        compressed_width = model.kv_channels + model.index_head_dim * p["indexed"]
        # CP gathers compressed latents before attention, while the common
        # memory estimator divides the local activation estimate by CP.
        workspace = model.micro_batch_size * (model.seq_length // p["ratio"]) * compressed_width
        workspace *= model.cp_size
    if p["owns_kv"]:
        compressed = model.micro_batch_size * (model.seq_length // p["ratio"])
        saved += compressed * (model.kv_channels + model.index_head_dim * p["indexed"])
    if p["runs_indexer"]:
        keys = model.seq_length // p["ratio"]
        saved += tokens * 2 * min(model.index_topk, keys)
        if p["candidate_source"]:
            options = model.config.transformer.architecture_options
            block = options["candidate_block_size"]
            saved += tokens * 2 * min(options["candidate_blocks"], (keys + block - 1) // block)
        # A fused indexer streams score tiles. Bound the FP32 score workspace
        # by one tile of query rows and the visible key/candidate domain.
        workspace += 2 * model.micro_batch_size * min(model.seq_length // model.cp_size, 128) * min(
            keys, p["candidate_limit"] or keys,
        ) * model.cp_size
    return ActivationEstimate(saved, workspace)


def _vision_dimensions(model):
    seq = model.config.parallel.vision_seq_length or model.config.vision.num_position_embeddings
    images = model.micro_batch_size * model.config.parallel.vision_num_images
    return seq, images


def _add_operations(model, result, spec, path, count, seq, batch, *, vision=False):
    def token_count(divisor):
        if vision:
            return seq / divisor * batch
        # Average completed compression groups across the CP ranks.
        return _token_count(model.seq_length, batch, divisor) / model.cp_size

    for i, child in enumerate(spec.submodules.operations):
        p = child.params
        part = child.metainfo["role"]
        child_path = f"{path}.operations[{i}]"
        if child.module is Projection:
            tp = model.tp_size if p["tp"] else 1
            tokens = token_count(p["token_divisor"])
            tokens /= _parallel_divisor(model, p["sequence_parallel"])
            if not tokens:
                continue
            row_parallel = p["row_parallel"] and p["groups"] == 1
            m = p["output_width"] / (1 if row_parallel else tp)
            k = p["input_width"] / (tp if row_parallel else p["groups"])
            model.add_gemm_part(
                result, part, module_spec=child, module_path=child_path,
                gemm_shape=(count * p["copies"], m, k, tokens),
            )
        elif child.module is TokenOperation:
            tokens = token_count(p["token_divisor"])
            tokens /= _parallel_divisor(model, p["sequence_parallel"])
            tp = model.tp_size if p.get("tp") else 1
            model.add_non_gemm_part(
                result, part, tokens * p["width"] / tp, count, p["ops"],
                module_spec=child, module_path=child_path,
            )


def _add_attention_kernel(model, result, spec, path, count, *, pairs, heads, head_dim, causal, batch, seq):
    part = spec.metainfo.get("role", "deepseek_vision_attention")
    # QK and PV have the same latent dimension, without a KV head expansion.
    base = batch * pairs * heads * 2 * head_dim
    peak = model.compute_tflops(use_fp8=model.use_fp8_training and causal)
    row = {
        "model_part": part,
        "b": batch,
        "m": "/",
        "n": "/",
        "k": "/",
        "shape": f"seq={seq}, mean_keys={pairs / seq:.2f}, heads={heads}, head_dim={head_dim}",
        "compute_count": count,
        "seq": seq,
        "kv_seq": pairs / seq,
        "heads": heads,
        "head_dim": head_dim,
        "value_head_dim": head_dim,
        "causal": causal,
        "forward_ms": 2 * base * count / (peak * model.gemm_efficiency) / 1e9,
        "backward_ms": 5 * base * count / (peak * model.gemm_efficiency * 0.88) / 1e9,
        "op_type": "flash_attention",
    }
    model._attach_spec_metadata(row, spec, path)
    result.append(row)
    model.add_non_gemm_part(
        result, f"{part}_softmax", batch * pairs * heads, count, "attention_softmax",
        module_spec=spec, module_path=path,
    )


@register_attention_lowering(CompressedAttention, first_gemm_part="deepseek_q_down")
def lower_attention(model, result, spec, path, count, count_scale):
    path = f"{path}.self_attention"
    _add_operations(model, result, spec, path, count, model.local_seq_length(), model.micro_batch_size)
    pairs = _attention_pairs(model, spec, model.seq_length) / model.cp_size
    _add_attention_kernel(
        model, result, spec, path, count, pairs=pairs, heads=model.num_attention_heads // model.tp_size,
        head_dim=model.kv_channels, causal=True, batch=model.micro_batch_size, seq=model.local_seq_length(),
    )
    index_pairs = _index_pairs(spec, model.seq_length) / model.cp_size
    if index_pairs:
        model.add_gemm_part(
            result, "deepseek_index_score", module_spec=spec, module_path=path,
            gemm_shape=(count, model.index_n_heads, model.index_head_dim, model.micro_batch_size * index_pairs),
        )
        model.add_non_gemm_part(
            result, "deepseek_index_reduce", model.micro_batch_size * index_pairs * model.index_n_heads,
            count, "dsa_score_reduce", module_spec=spec, module_path=path,
        )
        model.add_non_gemm_part(
            result, "deepseek_index_topk", model.micro_batch_size * index_pairs,
            count, "dsa_topk", module_spec=spec, module_path=path,
        )
        if spec.params["candidate_source"]:
            model.add_non_gemm_part(
                result, "deepseek_index_candidate_blocks", model.micro_batch_size * index_pairs,
                count, "dsa_topk", module_spec=spec, module_path=path,
            )


def _projection_work(spec, seq, *, vision=False):
    return sum(
        _weights(child.params)
        * child.params["copies"]
        * (seq / child.params["token_divisor"] if vision else seq // child.params["token_divisor"])
        for child in spec.submodules.operations
        if child.module is Projection
    )


@register_attention_flops(CompressedAttention)
def attention_flops(model, spec, seq, train_matmul):
    work = _projection_work(spec, seq)
    work += _attention_pairs(model, spec, seq) * model.num_attention_heads * 2 * model.kv_channels
    work += _index_pairs(spec, seq) * model.index_n_heads * model.index_head_dim
    return train_matmul * work / seq


@register_component_lowering(HyperConnection)
def lower_hyper_connection(model, result, spec, path):
    _add_operations(
        model, result, spec, path, model.microbatch_compute_count(),
        model.local_seq_length(), model.micro_batch_size,
    )


@register_component_flops(HyperConnection)
def hyper_connection_flops(model, spec, train_matmul):
    return train_matmul * _projection_work(spec, model.seq_length) * model.global_batch_size


@register_component_lowering(DeepSeekVision)
def lower_vision(model, result, spec, path):
    v = model.config.vision
    seq, images = _vision_dimensions(model)
    count = model.microbatch_compute_count()
    _add_operations(model, result, spec, path, count, seq, images, vision=True)
    _add_attention_kernel(
        model, result, spec, path, count * v.num_layers, pairs=seq ** 2,
        heads=v.num_attention_heads // model.tp_size, head_dim=v.hidden_size // v.num_attention_heads,
        causal=False, batch=images, seq=seq,
    )
    for name, width, ops in (
        ("norm", v.hidden_size * 2, "rmsnorm"),
        ("rope", v.hidden_size * 2, "rope"),
        ("swiglu", v.ffn_hidden_size, "swiglu"),
    ):
        model.add_non_gemm_part(
            result, f"deepseek_vision_{name}", images * seq * width / model.tp_size,
            count * v.num_layers, ops, module_spec=spec, module_path=path,
        )


@register_component_flops(DeepSeekVision)
def vision_flops(model, spec, train_matmul):
    v = model.config.vision
    seq, _ = _vision_dimensions(model)
    work = _projection_work(spec, seq, vision=True) + v.num_layers * seq ** 2 * 2 * v.hidden_size
    return train_matmul * work * model.global_batch_size * model.config.parallel.vision_num_images
