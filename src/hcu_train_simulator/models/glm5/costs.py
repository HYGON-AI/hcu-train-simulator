# Copyright (c) 2026 Hygon Information Technology Co., Ltd.

"""GLM-5 DSA parameter and activation cost registrations."""

from hcu_train_simulator.modeling.architecture import kv_cache_dim
from hcu_train_simulator.modeling.cost_registry import (
    register_activation_cost,
    register_parameter_cost,
)
from hcu_train_simulator.modeling.spec import DSAIndexer, DSASelfAttention


def _parameter(elements=0):
    from hcu_train_simulator.modeling.module_cost import ParameterEstimate

    return ParameterEstimate(elements)


def _activation(saved=0, workspace=0):
    from hcu_train_simulator.modeling.module_cost import ActivationEstimate

    return ActivationEstimate(saved, workspace)


@register_parameter_cost(role="dsa_index_q")
def dsa_index_q_parameters(model, spec, sharded):
    # The released GLM/DeepSeek implementations replicate the lightning
    # indexer on TP ranks and broadcast/verify the selected token indices.
    return _parameter(model.q_lora_rank * model.index_n_heads * model.index_head_dim)


@register_parameter_cost(role="dsa_index_k")
def dsa_index_k_parameters(model, spec, sharded):
    return _parameter(model.hidden_size * model.index_head_dim)


@register_parameter_cost(role="dsa_index_k_norm")
def dsa_index_k_norm_parameters(model, spec, sharded):
    # The indexer uses LayerNorm (scale and bias), not RMSNorm.
    return _parameter(2 * model.index_head_dim)


@register_parameter_cost(role="dsa_index_weights")
def dsa_index_weights_parameters(model, spec, sharded):
    return _parameter(model.hidden_size * model.index_n_heads)


@register_activation_cost(module=DSASelfAttention)
def dsa_attention_activations(model, spec, layer_uses_moe):
    tokens = model._tokens()
    if not layer_uses_moe:
        return _activation(tokens * model.hidden_size * 16 / model.tp_size)
    core = tokens * kv_cache_dim(model) / model.tp_size * 2
    if model.cp_size > 1:
        core *= 2
    projection = tokens * model.hidden_size / model.tp_size * 8
    return _activation(core + projection)


@register_activation_cost(module=DSAIndexer)
def dsa_indexer_activations(model, spec, layer_uses_moe):
    tokens = model._tokens()
    selected = min(model.index_topk, model.seq_length)
    saved = tokens * (
        model.index_n_heads * model.index_head_dim
        + model.index_head_dim
        + model.index_n_heads
        + 2 * selected  # int32 top-k indices expressed as BF16-sized elements
    )
    # Fused indexer kernels need at most the aggregated FP32 [B, S, S]
    # score surface; per-head scores can be streamed/recomputed.
    workspace = 2 * model.micro_batch_size * model.seq_length * model.seq_length
    return _activation(saved, workspace)
