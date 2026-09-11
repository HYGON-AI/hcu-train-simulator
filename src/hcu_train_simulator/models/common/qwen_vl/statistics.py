# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Analytical FLOPs shared by Qwen visual encoders."""

from hcu_train_simulator.modeling.spec import VisionModel
from hcu_train_simulator.modeling.statistics_registry import (
    register_component_flops,
)


@register_component_flops(VisionModel)
def vision_model_flops(model, component, train_matmul):
    vision = model.config.vision
    vision_seq = model.vision_seq_length or vision.num_position_embeddings
    vision_tokens = (
        model.global_batch_size * model.vision_num_images * vision_seq
    )
    patch_kernel = (
        vision.in_channels
        * vision.temporal_patch_size
        * vision.patch_size**2
    )
    total = (
        train_matmul * vision_tokens * patch_kernel * vision.hidden_size
    )
    block_weights = (
        4 * vision.hidden_size**2
        + 2 * vision.hidden_size * vision.ffn_hidden_size
    )
    block_attention = (
        2 * train_matmul * vision.hidden_size * vision_seq
    )
    total += (
        vision.num_layers
        * vision_tokens
        * (train_matmul * block_weights + block_attention)
    )
    merged = vision.hidden_size * vision.spatial_merge_size**2
    merged_tokens = vision_tokens / vision.spatial_merge_size**2
    merger_count = 1 + len(vision.deepstack_visual_indexes)
    total += (
        merger_count
        * train_matmul
        * merged_tokens
        * (merged**2 + merged * vision.output_hidden_size)
    )
    return total
