# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Qwen3.5-only parameter and activation cost registrations."""

from hcu_train_simulator.modeling.architecture import query_projection_dim
from hcu_train_simulator.modeling.cost_registry import register_activation_cost, register_parameter_cost
from hcu_train_simulator.modeling.spec import (
    GatedDeltaNet,
    MultiTokenPredictor,
)


def _parameter(elements=0, expert_elements=0):
    from hcu_train_simulator.modeling.module_cost import ParameterEstimate

    return ParameterEstimate(elements, expert_elements)


def _activation(saved=0, workspace=0):
    from hcu_train_simulator.modeling.module_cost import ActivationEstimate

    return ActivationEstimate(saved, workspace)


@register_parameter_cost(role="qwen35_q_proj_gate")
def qwen35_q_proj_gate_parameters(model, spec, sharded):
    tp = model.tp_size if sharded else 1
    fused_norm = model._scope_width(spec.params.get("fused_norm_scope"))
    return _parameter(fused_norm + 2 * query_projection_dim(model) * model.hidden_size / tp)


def _qwen35_kv_parameters(model, spec, sharded):
    tp = model.tp_size if sharded else 1
    return _parameter(model.num_query_groups * model.kv_channels * model.hidden_size / tp)


register_parameter_cost(role="qwen35_k_proj")(_qwen35_kv_parameters)
register_parameter_cost(role="qwen35_v_proj")(_qwen35_kv_parameters)


def _gdn_dimensions(model):
    key = model.linear_num_key_heads * model.linear_key_head_dim
    value = model.linear_num_value_heads * model.linear_value_head_dim
    return key, value


@register_parameter_cost(role="gdn_in_proj_qkv")
def gdn_in_proj_qkv_parameters(model, spec, sharded):
    tp = model.tp_size if sharded else 1
    key, value = _gdn_dimensions(model)
    fused_norm = model._scope_width(spec.params.get("fused_norm_scope"))
    return _parameter(fused_norm + model.hidden_size * (2 * key + value) / tp)


@register_parameter_cost(role="gdn_in_proj_z")
def gdn_in_proj_z_parameters(model, spec, sharded):
    tp = model.tp_size if sharded else 1
    _, value = _gdn_dimensions(model)
    return _parameter(model.hidden_size * value / tp)


def _gdn_ab_parameters(model, spec, sharded):
    tp = model.tp_size if sharded else 1
    return _parameter(model.hidden_size * model.linear_num_value_heads / tp)


register_parameter_cost(role="gdn_in_proj_a")(_gdn_ab_parameters)
register_parameter_cost(role="gdn_in_proj_b")(_gdn_ab_parameters)


@register_parameter_cost(role="gdn_causal_conv1d")
def gdn_conv_parameters(model, spec, sharded):
    tp = model.tp_size if sharded else 1
    key, value = _gdn_dimensions(model)
    channels = (2 * key + value) / tp
    # Qwen3.5 uses a depthwise Conv1d with bias=False.
    return _parameter(channels * model.linear_conv_kernel_dim)


@register_parameter_cost(role="mtp_fc")
def mtp_fc_parameters(model, spec, sharded):
    tp = model.tp_size if sharded else 1
    return _parameter(2 * model.hidden_size ** 2 / tp)


@register_parameter_cost(role="mtp_embedding")
def mtp_embedding_parameters(model, spec, sharded):
    if not model.mtp_use_dedicated_embeddings:
        return _parameter()
    tp = model.tp_size if sharded else 1
    return _parameter(model.vocab_size * model.hidden_size / tp)


@register_parameter_cost(role="gated_delta_rule")
def gated_delta_rule_parameters(model, spec, sharded):
    tp = model.tp_size if sharded else 1
    return _parameter(2 * model.linear_num_value_heads / tp)


@register_parameter_cost(role="gdn_gated_rmsnorm")
def gdn_norm_parameters(model, spec, sharded):
    return _parameter(model.linear_value_head_dim)


@register_parameter_cost(role="gdn_out_proj")
def gdn_out_proj_parameters(model, spec, sharded):
    tp = model.tp_size if sharded else 1
    _, value = _gdn_dimensions(model)
    return _parameter(value * model.hidden_size / tp)


@register_activation_cost(module=GatedDeltaNet)
def gdn_activations(model, spec, layer_uses_moe):
    tokens = model._tokens()
    key, value = _gdn_dimensions(model)
    saved = tokens * (2 * key + 3 * value + 2 * model.linear_num_value_heads) / model.tp_size
    state = (
        model.micro_batch_size
        * model.linear_num_value_heads
        * model.linear_key_head_dim
        * model.linear_value_head_dim
        / model.tp_size
    )
    return _activation(saved, state)


@register_activation_cost(module=MultiTokenPredictor)
def mtp_activations(model, spec, layer_uses_moe):
    # The embedding output is already represented by the first normalized H
    # input, so the child embedding node contributes parameters only.
    return _activation(model._tokens() * model.hidden_size * 4)


@register_activation_cost(role="mtp_embedding")
def mtp_embedding_activations(model, spec, layer_uses_moe):
    return _activation()
