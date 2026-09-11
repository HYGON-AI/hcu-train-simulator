# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Qwen3.5 analytical FLOP registrations."""

from hcu_train_simulator.modeling.architecture import (
    attention_output_dim,
    attention_score_dim,
    query_projection_dim,
    value_head_dim,
)
from hcu_train_simulator.modeling.spec import (
    GatedDeltaNet,
    GatedSelfAttention,
    MultiTokenPredictor,
)
from hcu_train_simulator.modeling.statistics_registry import register_attention_flops, register_component_flops


@register_attention_flops(GatedDeltaNet)
def gated_delta_net_flops(model, attention, seq, train_matmul):
    hidden = model.hidden_size
    key_dim = model.linear_num_key_heads * model.linear_key_head_dim
    value_dim = model.linear_num_value_heads * model.linear_value_head_dim
    projection_weights = hidden * (
        2 * key_dim + 2 * value_dim + 2 * model.linear_num_value_heads
    ) + value_dim * hidden
    return (
        train_matmul * projection_weights
        + 36 * model.linear_num_value_heads * model.linear_key_head_dim * model.linear_value_head_dim
    )


@register_attention_flops(GatedSelfAttention)
def gated_attention_flops(model, attention, seq, train_matmul):
    hidden = model.hidden_size
    query_dim = query_projection_dim(model)
    kv_dim = model.num_query_groups * model.kv_channels
    weights = hidden * (2 * query_dim + 2 * kv_dim) + attention_output_dim(model) * hidden
    return (
        train_matmul * weights
        + train_matmul
        * model.num_attention_heads
        * (attention_score_dim(model) + value_head_dim(model))
        * seq
    )


@register_component_flops(MultiTokenPredictor)
def mtp_flops(model, component, train_matmul):
    tokens = model.global_batch_size * model.seq_length
    return train_matmul * tokens * 2 * model.hidden_size ** 2
