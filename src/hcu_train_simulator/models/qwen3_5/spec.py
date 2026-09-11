# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Qwen3.5-specific ModuleSpec composition."""

from hcu_train_simulator.modeling.spec import (
    CausalConv1d,
    GatedDeltaNet,
    GatedDeltaNetSubmodules,
    GatedDeltaRule,
    GatedRMSNorm,
    GatedSelfAttention,
    GatedSelfAttentionSubmodules,
    ModuleSpec,
    SigmoidGate,
    TEColumnParallelLinear,
    TEDotProductAttention,
    TELayerNormColumnParallelLinear,
    TERowParallelLinear,
)
from hcu_train_simulator.modeling.te_spec import (
    _spec,
    _te_norm,
    build_multi_token_predictor_spec,
)


def build_gated_attention_spec():
    return ModuleSpec(
        module=GatedSelfAttention,
        params={"attn_mask_type": "causal", "output_gate": True},
        submodules=GatedSelfAttentionSubmodules(
            linear_q=_spec(
                TELayerNormColumnParallelLinear,
                role="qwen35_q_proj_gate",
                fused_norm_scope="hidden_size",
            ),
            linear_k=_spec(TEColumnParallelLinear, role="qwen35_k_proj"),
            linear_v=_spec(TEColumnParallelLinear, role="qwen35_v_proj"),
            core_attention=_spec(TEDotProductAttention, role="flash_attn"),
            linear_proj=_spec(TERowParallelLinear, role="attn_proj"),
            q_layernorm=_te_norm("head_dim", role="q_layernorm"),
            k_layernorm=_te_norm("head_dim", role="k_layernorm"),
            output_gate=_spec(SigmoidGate, role="attention_output_gate"),
        ),
        metainfo={"backend": "transformer_engine", "attention_variant": "gated_softmax"},
    )


def build_gated_delta_net_spec():
    return ModuleSpec(
        module=GatedDeltaNet,
        submodules=GatedDeltaNetSubmodules(
            in_proj_qkv=_spec(
                TELayerNormColumnParallelLinear,
                role="gdn_in_proj_qkv",
                fused_norm_scope="hidden_size",
            ),
            in_proj_z=_spec(TEColumnParallelLinear, role="gdn_in_proj_z"),
            in_proj_b=_spec(TEColumnParallelLinear, role="gdn_in_proj_b"),
            in_proj_a=_spec(TEColumnParallelLinear, role="gdn_in_proj_a"),
            conv1d=_spec(CausalConv1d, role="gdn_causal_conv1d"),
            core=_spec(GatedDeltaRule, role="gated_delta_rule"),
            gated_norm=_spec(GatedRMSNorm, role="gdn_gated_rmsnorm"),
            out_proj=_spec(TERowParallelLinear, role="gdn_out_proj"),
        ),
        metainfo={"backend": "transformer_engine", "attention_variant": "gated_delta_net"},
    )
