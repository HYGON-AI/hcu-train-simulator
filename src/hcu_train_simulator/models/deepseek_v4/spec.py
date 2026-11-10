# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Module specifications for DeepSeek's shared-latent attention and residual stream."""

from dataclasses import replace
from types import SimpleNamespace

from hcu_train_simulator.modeling.spec import (
    ModuleSpec,
    TENorm,
    VisionModel,
)
from hcu_train_simulator.modeling.te_spec import (
    build_transformer_engine_model_spec,
    get_gpt_layer_with_transformer_engine_spec,
)


class CompressedAttention:
    pass


class Projection:
    pass


class TokenOperation:
    pass


class EngramTable:
    pass


class HyperConnection:
    pass


class DeepSeekVision(VisionModel):
    pass


def _spec(module, role, **params):
    return ModuleSpec(module, params=params, metainfo={"role": role})


def _projection(role, input_width, output_width, *, groups=1, tp=False,
                sequence_parallel=False, token_divisor=1, extras=0, copies=1,
                row_parallel=False, tp_extras=0, replicate_weights=False):
    return _spec(
        Projection, role, input_width=input_width, output_width=output_width,
        groups=groups, tp=tp, sequence_parallel=sequence_parallel,
        token_divisor=token_divisor, extras=extras, copies=copies,
        row_parallel=row_parallel, tp_extras=tp_extras, replicate_weights=replicate_weights,
    )


def _operation(role, ops, width, *, sequence_parallel=False, token_divisor=1, tp=False):
    return _spec(
        TokenOperation, role, ops=ops, width=width,
        sequence_parallel=sequence_parallel, token_divisor=token_divisor, tp=tp,
    )


def _hyper_connection(t, *, output=False):
    options = t.architecture_options
    streams = options["hc_mult"]
    width = streams * t.hidden_size
    mix = streams if output else streams * (streams + 2)
    role = "deepseek_mhc_head" if output else "deepseek_mhc"
    nodes = []
    if not (output and options["csa2"]):
        nodes.append(_projection(
            role, width, mix, sequence_parallel=True, extras=mix + (1 if output else 3),
        ))
    nodes.append(_operation(
        f"{role}_mix", "residual_dropout_add", width * (1 if output else streams + 2),
        sequence_parallel=True,
    ))
    if not output:
        nodes.append(_operation(
            f"{role}_sinkhorn", "attention_softmax", streams ** 2 * options["hc_iters"],
            sequence_parallel=True,
        ))
    return nodes


def _compressor(role, hidden, width, ratio, *, overlap=False, positional=False):
    if ratio == 1:
        return [
            _projection(f"{role}_kv", hidden, width, extras=width),
            _operation(f"{role}_norm", "rmsnorm", width),
        ]
    projection_width = width * (2 if overlap else 1)
    return [
        _projection(f"{role}_kv", hidden, projection_width, extras=width),
        _projection(f"{role}_gate", hidden, projection_width,
                    extras=ratio * projection_width if positional else 0),
        _operation(f"{role}_pool", "attention_softmax", projection_width),
        _operation(f"{role}_norm", "rmsnorm", width, token_divisor=ratio),
    ]


def build_attention(t, layer_id):
    options = t.architecture_options
    csa2 = options["csa2"]
    is_mtp = layer_id >= t.num_layers
    ratio = 0 if is_mtp else options["ratios"][layer_id]
    owns_kv = ratio > 0 and (not csa2 or layer_id in options["kv_sources"])
    runs_indexer = ratio > 0 and (
        layer_id in options["index_sources"] if csa2 else ratio == options["csa_ratio"]
    )
    index_mode = (
        ("full" if owns_kv else "reindex" if runs_indexer else "reuse") if csa2
        else ("csa" if runs_indexer else "hca")
    ) if ratio else "swa"

    h = t.hidden_size
    head = t.kv_channels
    q = t.q_lora_rank
    heads = t.num_attention_heads
    groups = options["o_groups"]
    o_rank = options["o_rank"]
    rope = t.qk_rope_head_dim or 0
    nodes = [
        _projection("deepseek_q_down", h, q, extras=q),
        _projection("deepseek_q_up", q, heads * head, tp=True, tp_extras=heads),
        _projection("deepseek_swa_kv", h, head, extras=head),
        _operation("deepseek_q_down_norm", "rmsnorm", q),
        _operation("deepseek_q_norm", "qk_norm", heads * head, tp=True),
        _operation("deepseek_swa_kv_norm", "rmsnorm", head),
        _operation("deepseek_swa_kv_rope", "rope", rope),
    ]
    if owns_kv:
        nodes += _compressor("deepseek_compress", h, head, ratio,
                             overlap=not csa2 and runs_indexer, positional=not csa2)
        nodes.append(_operation("deepseek_compress_rope", "rope", rope, token_divisor=ratio))
    if runs_indexer:
        index_head = t.index_head_dim
        index_heads = t.index_n_heads
        nodes += [
            _projection("deepseek_index_q", q, index_heads * index_head),
            _projection("deepseek_index_weights", h, index_heads),
            _operation("deepseek_index_q_rope", "rope", index_heads * rope),
        ]
        if not csa2:
            nodes += _compressor("deepseek_index_compress", h, index_head, ratio,
                                 overlap=True, positional=True)
        elif owns_kv:
            nodes += [
                _projection("deepseek_index_k", head, index_head,
                            token_divisor=ratio, extras=index_head),
                _operation("deepseek_index_k_norm", "rmsnorm", index_head, token_divisor=ratio),
            ]
        if not csa2 or owns_kv:
            nodes.append(_operation("deepseek_index_k_rope", "rope", rope, token_divisor=ratio))

    nodes += [
        # Megatron's V4 bridge replicates the full wo_a parameter, even when
        # a rank computes only the output groups corresponding to its heads.
        _projection("deepseek_o_group", heads * head, groups * o_rank, groups=groups,
                    tp=True, row_parallel=True, replicate_weights=not csa2),
        _projection("deepseek_o_up", groups * o_rank, h, tp=True, row_parallel=True),
    ]
    # Q normalization and Q/output RoPE operate on the local heads.
    nodes += [_operation("deepseek_qo_rope", "rope", 2 * heads * rope, tp=True)]
    if is_mtp:
        nodes += _hyper_connection(t, output=True)
        for name in ("embedding", "hidden"):
            nodes.append(_projection(f"deepseek_mtp_{name}_proj", h, h, tp=True, extras=h))
        nodes.append(_operation("deepseek_mtp_norms", "rmsnorm", 3 * h, sequence_parallel=True))

    rows = options.get("engram_rows", {}).get(layer_id, 0)
    if rows:
        lookup = options["engram_lookup_heads"] * options["engram_head_dim"]
        streams = options["hc_mult"]
        nodes += [
            _spec(EngramTable, "deepseek_engram_table", rows=rows, head_dim=options["engram_head_dim"],
                  lookup_width=lookup),
            _projection("deepseek_engram_kv", lookup, (streams + 1) * h, extras=2 * streams * h),
            _operation("deepseek_engram_gate", "gated_rmsnorm", streams * h),
        ]

    return ModuleSpec(
        CompressedAttention,
        params={
            "ratio": ratio,
            "owns_kv": owns_kv,
            "runs_indexer": runs_indexer,
            "indexed": runs_indexer or (csa2 and ratio > 0),
            "candidate_source": layer_id == options.get("candidate_source"),
            "candidate_limit": (
                options["candidate_blocks"] * options["candidate_block_size"]
                if csa2 and 0 <= options["candidate_source"] < layer_id and runs_indexer else 0
            ),
            "layer_id": layer_id,
        },
        submodules=SimpleNamespace(operations=nodes),
        metainfo={"attention_variant": index_mode, "role": f"deepseek_{index_mode}_attention"},
    )


def _vision_spec(vision, hidden):
    h = vision.hidden_size
    layers = vision.num_layers
    merge = vision.spatial_merge_size ** 2
    projections = [
        _projection("deepseek_vision_patch", vision.in_channels * vision.patch_size ** 2, h,
                    tp=True, extras=h),
        _projection("deepseek_vision_qkv", h, 3 * h, tp=True, extras=3 * h, copies=layers),
        _projection("deepseek_vision_o", h, h, tp=True, row_parallel=True, extras=h, copies=layers),
        _projection("deepseek_vision_gate_up", h, 2 * vision.ffn_hidden_size, tp=True, copies=layers),
        _projection("deepseek_vision_down", vision.ffn_hidden_size, h, tp=True, row_parallel=True, extras=2 * h, copies=layers),
        _projection("deepseek_vision_aligner_in", h * merge, hidden, tp=True,
                    extras=hidden + h, token_divisor=merge),
        _projection("deepseek_vision_aligner_out", hidden, hidden, tp=True, row_parallel=True,
                    extras=4 * hidden, token_divisor=merge),
    ]
    return ModuleSpec(
        DeepSeekVision,
        submodules=SimpleNamespace(operations=[
            replace(node, params={**node.params, "vision": True}) for node in projections
        ]),
    )


def build_model_spec(config):
    t = config.transformer
    options = t.architecture_options
    backbone_attention = [build_attention(t, i) for i in range(t.num_layers)]
    grouped = config.parallel.moe_grouped_gemm
    model = build_transformer_engine_model_spec(
        replace(t, mtp_num_layers=0),
        config.model,
        layer_attention_specs=backbone_attention,
        vision_model_spec=_vision_spec(config.vision, t.hidden_size) if config.vision.enabled else None,
        moe_grouped_gemm=grouped,
    )
    block = model.submodules.decoder.submodules
    for i in range(t.mtp_num_layers):
        block.layer_specs.append(get_gpt_layer_with_transformer_engine_spec(
            t, num_experts=t.num_moe_experts, moe_grouped_gemm=grouped,
            is_mtp_layer=True, attention_spec=build_attention(t, t.num_layers + i),
        ))
    for i, layer in enumerate(block.layer_specs):
        sub = layer.submodules
        router = sub.mlp.submodules.router
        if i in options["hash_layers"]:
            router = replace(router, params={**router.params, "hash_routing": True})
        mlp = replace(sub.mlp, submodules=replace(sub.mlp.submodules, router=router))
        connection = ModuleSpec(
            HyperConnection,
            submodules=SimpleNamespace(operations=_hyper_connection(t)),
        )
        block.layer_specs[i] = replace(layer, submodules=replace(
            sub,
            input_layernorm=ModuleSpec(TENorm, metainfo={"role": "attention_rmsnorm"}),
            mlp=mlp,
            self_attn_bda=connection,
            mlp_bda=connection,
        ))

    final = ModuleSpec(
        HyperConnection,
        submodules=SimpleNamespace(norm=block.layer_norm, operations=_hyper_connection(t, output=True)),
    )
    decoder = replace(model.submodules.decoder, submodules=replace(block, layer_norm=final))
    return replace(model, submodules=replace(model.submodules, decoder=decoder),
                   metainfo={**model.metainfo, "replicated_kv": True})
