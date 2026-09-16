# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Shared architecture helpers for model-specific estimator formulas."""

from hcu_train_simulator.modeling.spec import ModuleSpec, spec_has_qk_norm, spec_uses_mla, spec_uses_moe


def _transformer(config):
    return getattr(config, "transformer", config)


def round_up(value, multiple):
    if not multiple or multiple <= 1:
        return value
    return ((value + multiple - 1) // multiple) * multiple


def uses_mla(model):
    spec = model if isinstance(model, ModuleSpec) else getattr(model, "model_spec", None)
    if spec is None:
        raise TypeError("architecture queries require a resolved model_spec")
    if not isinstance(model, ModuleSpec) and hasattr(model, "_uses_mla_cache"):
        return model._uses_mla_cache
    result = spec_uses_mla(spec)
    if not isinstance(model, ModuleSpec):
        try:
            model._uses_mla_cache = result
        except (AttributeError, TypeError):
            pass
    return result


def uses_moe(model):
    spec = model if isinstance(model, ModuleSpec) else getattr(model, "model_spec", None)
    if spec is None:
        raise TypeError("architecture queries require a resolved model_spec")
    if not isinstance(model, ModuleSpec) and hasattr(model, "_uses_moe_cache"):
        return model._uses_moe_cache
    result = spec_uses_moe(spec)
    if not isinstance(model, ModuleSpec):
        try:
            model._uses_moe_cache = result
        except (AttributeError, TypeError):
            pass
    return result


def has_qk_norm(model):
    spec = model if isinstance(model, ModuleSpec) else getattr(model, "model_spec", None)
    if spec is None:
        raise TypeError("architecture queries require a resolved model_spec")
    if not isinstance(model, ModuleSpec) and hasattr(model, "_has_qk_norm_cache"):
        return model._has_qk_norm_cache
    result = spec_has_qk_norm(spec)
    if not isinstance(model, ModuleSpec):
        try:
            model._has_qk_norm_cache = result
        except (AttributeError, TypeError):
            pass
    return result


def qk_head_dim(model):
    mla = uses_mla(model)
    model = _transformer(model)
    if mla:
        return model.qk_nope_head_dim + model.qk_rope_head_dim
    return model.kv_channels


def value_head_dim(model):
    model = _transformer(model)
    return getattr(model, "v_head_dim", None) or model.kv_channels


def query_projection_dim(model):
    head_dim = qk_head_dim(model)
    model = _transformer(model)
    return model.num_attention_heads * head_dim


def attention_output_dim(model):
    value_dim = value_head_dim(model)
    model = _transformer(model)
    return model.num_attention_heads * value_dim


def kv_projection_dim(model):
    mla = uses_mla(model)
    value_dim = value_head_dim(model)
    model = _transformer(model)
    if mla:
        return model.num_attention_heads * (model.qk_nope_head_dim + value_dim)
    return model.num_query_groups * model.kv_channels * 2


def kv_cache_dim(model):
    mla = uses_mla(model)
    model = _transformer(model)
    if mla:
        return model.kv_lora_rank + model.qk_rope_head_dim
    return model.num_query_groups * model.kv_channels


def attention_score_dim(model):
    return qk_head_dim(model)


def active_transformer_layers(model):
    model = _transformer(model)
    return model.num_layers + (getattr(model, "mtp_num_layers", 0) or 0)


def layer_uses_moe(model, layer_index):
    model = _transformer(model)
    if not getattr(model, "num_moe_experts", None):
        return False
    if layer_index in (getattr(model, "moe_dense_layer_indices", ()) or ()):
        return False
    first_dense = getattr(model, "first_k_dense_replace", 0) or 0
    frequency = getattr(model, "moe_layer_freq", 1) or 1
    offset = getattr(model, "moe_layer_offset", 0) or 0
    return layer_index >= first_dense and (layer_index - offset) % frequency == 0


def output_loss_factor(model):
    model = _transformer(model)
    return 1 + (getattr(model, "mtp_num_layers", 0) or 0)


def shared_expert_intermediate_total(model):
    model = _transformer(model)
    num_shared = getattr(model, "num_shared_experts", None) or 0
    if num_shared <= 0:
        return 0
    shared_size = (
        getattr(model, "shared_expert_ffn_hidden_size", None)
        or getattr(model, "moe_ffn_hidden_size", None)
        or getattr(model, "ffn_hidden_size", 0)
    )
    return num_shared * shared_size
