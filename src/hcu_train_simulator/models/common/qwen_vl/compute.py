# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Compute lowerings shared by Qwen visual encoders."""

from hcu_train_simulator.modeling.compute_registry import (
    register_component_lowering,
    register_gemm_shape,
)
from hcu_train_simulator.modeling.spec import VisionModel, module_type_name
from hcu_train_simulator.parallelism import get_num_microbatches


def _vision_sequence_length(model):
    return (
        model.config.parallel.vision_seq_length
        or model.config.vision.num_position_embeddings
    )


def _vision_token_count(model):
    return (
        model.micro_batch_size
        * model.config.parallel.vision_num_images
        * _vision_sequence_length(model)
    )


def _vision_compute_count(model):
    return get_num_microbatches(
        model.global_batch_size,
        model.micro_batch_size,
        model.dp_size,
    )


@register_gemm_shape("vision_patch_embed")
def vision_patch_embed_shape(model):
    vision = model.config.vision
    kernel = (
        vision.in_channels
        * vision.temporal_patch_size
        * vision.patch_size**2
    )
    return (
        _vision_compute_count(model),
        vision.hidden_size // model.tp_size,
        kernel,
        _vision_token_count(model),
    )


@register_gemm_shape("vision_qkv")
def vision_qkv_shape(model):
    vision = model.config.vision
    return (
        _vision_compute_count(model),
        3 * vision.hidden_size // model.tp_size,
        vision.hidden_size,
        _vision_token_count(model),
    )


@register_gemm_shape("vision_attn_proj")
def vision_attn_proj_shape(model):
    vision = model.config.vision
    return (
        _vision_compute_count(model),
        vision.hidden_size,
        vision.hidden_size // model.tp_size,
        _vision_token_count(model),
    )


@register_gemm_shape("vision_fc1")
def vision_fc1_shape(model):
    vision = model.config.vision
    return (
        _vision_compute_count(model),
        vision.ffn_hidden_size // model.tp_size,
        vision.hidden_size,
        _vision_token_count(model),
    )


@register_gemm_shape("vision_fc2")
def vision_fc2_shape(model):
    vision = model.config.vision
    return (
        _vision_compute_count(model),
        vision.hidden_size,
        vision.ffn_hidden_size // model.tp_size,
        _vision_token_count(model),
    )


@register_gemm_shape("vision_merger_fc1")
def vision_merger_fc1_shape(model):
    vision = model.config.vision
    merged = vision.hidden_size * vision.spatial_merge_size**2
    tokens = _vision_token_count(model) // max(
        1,
        vision.spatial_merge_size**2,
    )
    return (
        _vision_compute_count(model),
        merged // model.tp_size,
        merged,
        tokens,
    )


@register_gemm_shape("vision_merger_fc2")
def vision_merger_fc2_shape(model):
    vision = model.config.vision
    merged = vision.hidden_size * vision.spatial_merge_size**2
    tokens = _vision_token_count(model) // max(
        1,
        vision.spatial_merge_size**2,
    )
    return (
        _vision_compute_count(model),
        vision.output_hidden_size,
        merged // model.tp_size,
        tokens,
    )


register_gemm_shape("vision_deepstack_merger_fc1")(
    vision_merger_fc1_shape
)
register_gemm_shape("vision_deepstack_merger_fc2")(
    vision_merger_fc2_shape
)


def _lower_vision_layer(model, result, layer_spec, module_path):
    submodules = layer_spec.submodules
    attention = submodules.self_attention
    mlp = submodules.mlp
    compute_count = _vision_compute_count(model)
    tokens = _vision_token_count(model)
    elements = (
        tokens * model.config.vision.hidden_size / model.tp_size
    )

    for name, spec in (
        ("vision_norm1", submodules.input_layernorm),
        ("vision_norm2", submodules.pre_mlp_layernorm),
    ):
        model.add_non_gemm_part(
            result,
            name,
            elements,
            compute_count,
            "layernorm",
            module_spec=spec,
            module_path=f"{module_path}.{name}",
        )
    model.add_gemm_part(
        result,
        "vision_qkv",
        module_spec=attention.submodules.linear_qkv,
        module_path=f"{module_path}.self_attention.linear_qkv",
    )
    model.add_non_gemm_part(
        result,
        "vision_rope",
        2 * elements,
        compute_count,
        "rope",
        module_spec=attention,
        module_path=f"{module_path}.self_attention.rope",
    )

    vision = model.config.vision
    seq = _vision_sequence_length(model)
    heads = vision.num_attention_heads // model.tp_size
    head_dim = vision.hidden_size // vision.num_attention_heads
    batch = (
        model.micro_batch_size * model.config.parallel.vision_num_images
    )
    forward_flops = 4 * batch * heads * seq * seq * head_dim
    backward_flops = forward_flops * 2.5
    result.append(
        {
            "model_part": "vision_flash_attn",
            "b": batch,
            "m": "/",
            "n": "/",
            "k": "/",
            "shape": (
                f"batch={batch}, seq={seq}, heads={heads}, "
                f"head_dim={head_dim}"
            ),
            "elements": int(batch * heads * seq * seq),
            "compute_count": compute_count,
            "seq": seq,
            "heads": heads,
            "head_dim": head_dim,
            "causal": False,
            "forward_ms": (
                forward_flops
                * compute_count
                / (model.fp16_tflops * model.gemm_efficiency)
                / 1e9
            ),
            "backward_ms": (
                backward_flops
                * compute_count
                / (
                    model.fp16_tflops
                    * model.gemm_efficiency
                    * 0.5
                )
                / 1e9
            ),
            "op_type": "flash_attention",
            "module_type": module_type_name(
                attention.submodules.core_attention
            ),
            "module_role": (
                attention.submodules.core_attention.metainfo.get("role")
            ),
            "module_path": (
                f"{module_path}.self_attention.core_attention"
            ),
        }
    )
    model.add_gemm_part(
        result,
        "vision_attn_proj",
        module_spec=attention.submodules.linear_proj,
        module_path=f"{module_path}.self_attention.linear_proj",
    )
    model.add_gemm_part(
        result,
        "vision_fc1",
        module_spec=mlp.submodules.linear_fc1,
        module_path=f"{module_path}.mlp.linear_fc1",
    )
    model.add_non_gemm_part(
        result,
        "vision_gelu",
        tokens * vision.ffn_hidden_size / model.tp_size,
        compute_count,
        "gelu",
        module_spec=mlp.submodules.activation_func,
        module_path=f"{module_path}.mlp.activation_func",
    )
    model.add_gemm_part(
        result,
        "vision_fc2",
        module_spec=mlp.submodules.linear_fc2,
        module_path=f"{module_path}.mlp.linear_fc2",
    )
    for bda_name in ("self_attn_bda", "mlp_bda"):
        model.add_non_gemm_part(
            result,
            "vision_residual_add",
            elements,
            compute_count,
            "residual_dropout_add",
            module_spec=getattr(submodules, bda_name),
            module_path=f"{module_path}.{bda_name}",
        )


@register_component_lowering(VisionModel)
def lower_vision_model(model, result, vision_spec, module_path):
    submodules = vision_spec.submodules
    compute_count = _vision_compute_count(model)
    tokens = _vision_token_count(model)
    model.add_gemm_part(
        result,
        "vision_patch_embed",
        module_spec=submodules.patch_embedding,
        module_path=f"{module_path}.patch_embedding",
    )
    model.add_non_gemm_part(
        result,
        "vision_position_add",
        tokens * model.config.vision.hidden_size / model.tp_size,
        compute_count,
        "position_add",
        module_spec=submodules.position_embedding,
        module_path=f"{module_path}.position_embedding",
    )
    for index, layer_spec in enumerate(
        submodules.decoder.submodules.layer_specs
    ):
        _lower_vision_layer(
            model,
            result,
            layer_spec,
            f"{module_path}.decoder.layers[{index}]",
        )

    _lower_vision_merger(
        model,
        result,
        submodules.merger,
        f"{module_path}.merger",
        prefix="vision_merger",
        use_postshuffle_norm=False,
    )
    for index, merger in enumerate(submodules.deepstack_mergers):
        _lower_vision_merger(
            model,
            result,
            merger,
            f"{module_path}.deepstack_mergers[{index}]",
            prefix="vision_deepstack_merger",
            use_postshuffle_norm=True,
        )


def _lower_vision_merger(
    model,
    result,
    merger,
    module_path,
    *,
    prefix,
    use_postshuffle_norm,
):
    compute_count = _vision_compute_count(model)
    tokens = _vision_token_count(model)
    vision = model.config.vision
    merger_tokens = tokens // max(1, vision.spatial_merge_size**2)
    merged_width = vision.hidden_size * vision.spatial_merge_size**2
    norm_width = (
        merged_width if use_postshuffle_norm else vision.hidden_size
    )
    norm_elements = (
        (merger_tokens if use_postshuffle_norm else tokens)
        * norm_width
        / model.tp_size
    )
    merger_elements = merger_tokens * merged_width / model.tp_size
    model.add_non_gemm_part(
        result,
        f"{prefix}_norm",
        norm_elements,
        compute_count,
        "layernorm",
        module_spec=merger,
        module_path=f"{module_path}.norm",
    )
    model.add_gemm_part(
        result,
        f"{prefix}_fc1",
        module_spec=merger,
        module_path=f"{module_path}.linear_fc1",
    )
    model.add_non_gemm_part(
        result,
        f"{prefix}_gelu",
        merger_elements,
        compute_count,
        "gelu",
        module_spec=merger,
        module_path=f"{module_path}.activation",
    )
    model.add_gemm_part(
        result,
        f"{prefix}_fc2",
        module_spec=merger,
        module_path=f"{module_path}.linear_fc2",
    )
    if use_postshuffle_norm:
        model.add_non_gemm_part(
            result,
            "vision_deepstack_add",
            (
                merger_tokens
                * vision.output_hidden_size
                / model.tp_size
            ),
            compute_count,
            "residual_dropout_add",
            module_spec=merger,
            module_path=f"{module_path}.language_model_add",
        )
