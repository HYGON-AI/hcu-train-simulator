# Copyright (c) 2026 Hygon Information Technology Co., Ltd.

"""Communication-overlap parts shared by Qwen visual encoders."""

from hcu_train_simulator.estimators.overlap_registry import (
    register_overlap_parts,
)


VISION_ATTENTION_PARTS = {
    "vision_qkv",
    "vision_rope",
    "vision_flash_attn",
    "vision_attn_proj",
}
VISION_PARTS = {
    "vision_patch_embed",
    "vision_position_add",
    "vision_norm1",
    "vision_norm2",
    *VISION_ATTENTION_PARTS,
    "vision_fc1",
    "vision_gelu",
    "vision_fc2",
    "vision_residual_add",
    "vision_merger_norm",
    "vision_merger_fc1",
    "vision_merger_gelu",
    "vision_merger_fc2",
    "vision_deepstack_merger_norm",
    "vision_deepstack_merger_fc1",
    "vision_deepstack_merger_gelu",
    "vision_deepstack_merger_fc2",
    "vision_deepstack_add",
}

register_overlap_parts(
    transformer=VISION_PARTS,
    attention=VISION_ATTENTION_PARTS,
)
