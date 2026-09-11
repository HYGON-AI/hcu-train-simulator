# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Registry of compute rows eligible to cover communication domains."""

_TRANSFORMER_PARTS = {
    "attention_rmsnorm", "qk_norm", "rope", "qkv_weight",
    "mla_q_down", "mla_q_up", "mla_kv_down", "mla_kv_up",
    "flash_attn", "attention_softmax", "attn_proj", "mlp_rmsnorm",
    "linear_fc1", "swiglu_activation", "linear_fc2", "residual_dropout_add",
    "shared_expert_fc1", "shared_swiglu_activation", "shared_expert_fc2",
    "shared_expert_gate",
}
_ATTENTION_PARTS = {
    "attention_rmsnorm", "qk_norm", "rope", "qkv_weight",
    "mla_q_down", "mla_q_up", "mla_kv_down", "mla_kv_up",
    "flash_attn", "attention_softmax", "attn_proj",
}
_EXPERT_PARTS = {
    "mlp_rmsnorm", "topk_router", "router_select", "moe_linear_fc1",
    "moe_swiglu_activation", "moe_linear_fc2", "shared_expert_fc1",
    "shared_swiglu_activation", "shared_expert_fc2",
}


def register_overlap_parts(*, transformer=(), attention=(), expert=()):
    _TRANSFORMER_PARTS.update(transformer)
    _ATTENTION_PARTS.update(attention)
    _EXPERT_PARTS.update(expert)


def get_overlap_part_sets():
    from hcu_train_simulator.models.registry import registered_model_adapters

    registered_model_adapters()
    return set(_TRANSFORMER_PARTS), set(_ATTENTION_PARTS), set(_EXPERT_PARTS)
