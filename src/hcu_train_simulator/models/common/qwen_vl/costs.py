# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Parameter and activation costs shared by Qwen visual encoders."""

from hcu_train_simulator.modeling.cost_registry import (
    register_activation_cost,
    register_parameter_cost,
)
from hcu_train_simulator.modeling.module_cost import (
    ActivationEstimate,
    ParameterEstimate,
)
from hcu_train_simulator.modeling.spec import (
    VisionMLP,
    VisionPatchEmbedding,
    VisionPatchMerger,
    VisionPositionEmbedding,
    VisionSelfAttention,
)


def _parameter(elements=0):
    return ParameterEstimate(elements, 0)


def _activation(saved=0, workspace=0):
    return ActivationEstimate(saved, workspace)


@register_parameter_cost(role="vision_patch_embed")
def vision_patch_parameters(model, spec, sharded):
    vision = model.config.vision
    tp = model.tp_size if sharded else 1
    kernel = (
        vision.in_channels
        * vision.temporal_patch_size
        * vision.patch_size**2
    )
    return _parameter(vision.hidden_size / tp * (kernel + 1))


@register_parameter_cost(role="vision_position_embedding")
def vision_position_parameters(model, spec, sharded):
    vision = model.config.vision
    tp = model.tp_size if sharded else 1
    return _parameter(
        vision.num_position_embeddings * vision.hidden_size / tp
    )


@register_parameter_cost(role="vision_qkv")
def vision_qkv_parameters(model, spec, sharded):
    vision = model.config.vision
    tp = model.tp_size if sharded else 1
    return _parameter(
        3 * vision.hidden_size * (vision.hidden_size + 1) / tp
    )


@register_parameter_cost(role="vision_attn_proj")
def vision_proj_parameters(model, spec, sharded):
    vision = model.config.vision
    tp = model.tp_size if sharded else 1
    return _parameter(
        vision.hidden_size * (vision.hidden_size + 1) / tp
    )


@register_parameter_cost(role="vision_fc1")
def vision_fc1_parameters(model, spec, sharded):
    vision = model.config.vision
    tp = model.tp_size if sharded else 1
    return _parameter(
        vision.ffn_hidden_size * (vision.hidden_size + 1) / tp
    )


@register_parameter_cost(role="vision_fc2")
def vision_fc2_parameters(model, spec, sharded):
    vision = model.config.vision
    tp = model.tp_size if sharded else 1
    return _parameter(
        vision.hidden_size * (vision.ffn_hidden_size + 1) / tp
    )


@register_parameter_cost(role="vision_patch_merger")
def vision_merger_parameters(model, spec, sharded):
    vision = model.config.vision
    tp = model.tp_size if sharded else 1
    merged = vision.hidden_size * vision.spatial_merge_size**2
    norm_width = (
        merged
        if spec.params.get("use_postshuffle_norm", False)
        else vision.hidden_size
    )
    norms = 2 * norm_width
    fc1 = merged * (merged + 1)
    fc2 = vision.output_hidden_size * (merged + 1)
    return _parameter(norms + (fc1 + fc2) / tp)


@register_activation_cost(module=VisionPatchEmbedding)
@register_activation_cost(module=VisionPositionEmbedding)
def vision_input_activations(model, spec, layer_uses_moe):
    vision = model.config.vision
    return _activation(
        model._vision_tokens() * vision.hidden_size / model.tp_size
    )


@register_activation_cost(module=VisionSelfAttention)
def vision_attention_activations(model, spec, layer_uses_moe):
    vision = model.config.vision
    tokens = model._vision_tokens()
    batch = (
        model.micro_batch_size * model.config.parallel.vision_num_images
    )
    saved = tokens * vision.hidden_size * 8 / model.tp_size
    # Flash Attention materializes per-row statistics, not an SxS matrix.
    workspace = (
        batch
        * vision.num_attention_heads
        / model.tp_size
        * (tokens / max(1, batch))
    )
    return _activation(saved, workspace)


@register_activation_cost(module=VisionMLP)
def vision_mlp_activations(model, spec, layer_uses_moe):
    vision = model.config.vision
    return _activation(
        model._vision_tokens()
        * vision.ffn_hidden_size
        * 2
        / model.tp_size
    )


@register_activation_cost(module=VisionPatchMerger)
def vision_merger_activations(model, spec, layer_uses_moe):
    vision = model.config.vision
    merged_tokens = model._vision_tokens() / max(
        1,
        vision.spatial_merge_size**2,
    )
    merged_width = vision.hidden_size * vision.spatial_merge_size**2
    return _activation(
        merged_tokens
        * (merged_width + vision.output_hidden_size)
        / model.tp_size
    )


@register_activation_cost(role="vision_residual_add")
def vision_residual_activations(model, spec, layer_uses_moe):
    vision = model.config.vision
    return _activation(
        model._vision_tokens() * vision.hidden_size / model.tp_size
    )
