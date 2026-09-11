# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""ModuleSpec composition shared by Qwen vision-language models."""

from hcu_train_simulator.modeling.spec import (
    BiasDropoutAdd,
    GELU,
    MLPSubmodules,
    ModuleSpec,
    TEColumnParallelLinear,
    TEDotProductAttention,
    TERowParallelLinear,
    VisionAttentionSubmodules,
    VisionBlockSubmodules,
    VisionLayerSubmodules,
    VisionMLP,
    VisionModel,
    VisionModelSubmodules,
    VisionPatchEmbedding,
    VisionPatchMerger,
    VisionPositionEmbedding,
    VisionSelfAttention,
    VisionTransformerBlock,
    VisionTransformerLayer,
)
from hcu_train_simulator.modeling.te_spec import _spec, _te_layernorm


def _build_patch_merger_spec(*, use_postshuffle_norm=False):
    return _spec(
        VisionPatchMerger,
        role="vision_patch_merger",
        use_postshuffle_norm=use_postshuffle_norm,
    )


def build_vision_model_spec(vision_config, *, model_type):
    attention = ModuleSpec(
        module=VisionSelfAttention,
        submodules=VisionAttentionSubmodules(
            linear_qkv=_spec(TEColumnParallelLinear, role="vision_qkv"),
            core_attention=_spec(
                TEDotProductAttention,
                role="vision_flash_attn",
            ),
            linear_proj=_spec(
                TERowParallelLinear,
                role="vision_attn_proj",
            ),
        ),
        metainfo={"backend": "transformer_engine", "causal": False},
    )
    mlp = ModuleSpec(
        module=VisionMLP,
        submodules=MLPSubmodules(
            linear_fc1=_spec(TEColumnParallelLinear, role="vision_fc1"),
            activation_func=_spec(GELU, role="vision_gelu"),
            linear_fc2=_spec(TERowParallelLinear, role="vision_fc2"),
        ),
        metainfo={"backend": "transformer_engine"},
    )
    layer = ModuleSpec(
        module=VisionTransformerLayer,
        submodules=VisionLayerSubmodules(
            input_layernorm=_te_layernorm(
                "vision_hidden_size",
                role="vision_norm1",
            ),
            self_attention=attention,
            self_attn_bda=_spec(
                BiasDropoutAdd,
                role="vision_residual_add",
            ),
            pre_mlp_layernorm=_te_layernorm(
                "vision_hidden_size",
                role="vision_norm2",
            ),
            mlp=mlp,
            mlp_bda=_spec(BiasDropoutAdd, role="vision_residual_add"),
        ),
        metainfo={"backend": "transformer_engine", "is_vision_layer": True},
    )
    decoder = ModuleSpec(
        module=VisionTransformerBlock,
        submodules=VisionBlockSubmodules(
            layer_specs=[layer for _ in range(vision_config.num_layers)]
        ),
        metainfo={"backend": "transformer_engine"},
    )
    return ModuleSpec(
        module=VisionModel,
        submodules=VisionModelSubmodules(
            patch_embedding=_spec(
                VisionPatchEmbedding,
                role="vision_patch_embed",
            ),
            position_embedding=_spec(
                VisionPositionEmbedding,
                role="vision_position_embedding",
            ),
            decoder=decoder,
            merger=_build_patch_merger_spec(),
            deepstack_mergers=[
                _build_patch_merger_spec(use_postshuffle_norm=True)
                for _ in vision_config.deepstack_visual_indexes
            ],
        ),
        metainfo={"backend": "transformer_engine", "model_type": model_type},
    )
