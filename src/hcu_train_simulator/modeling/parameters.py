# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility facade for ModuleSpec-driven parameter estimation."""

from hcu_train_simulator.context import get_config
from hcu_train_simulator.modeling.module_cost import ModuleSpecMemoryModel
from hcu_train_simulator.modeling.spec import (
    MoELayer,
    get_attention_spec,
    get_mlp_spec,
    get_representative_layer_spec,
    module_is,
)


class ParameterModel:
    def __init__(self):
        self.config = get_config()
        self.model_spec = self.config.model_spec
        self.estimator = ModuleSpecMemoryModel(self.config)

    def estimate(self, spec=None):
        return self.estimator.parameter_estimate(spec)

    def memory_rows(self, spec=None, path="model"):
        return self.estimator.memory_rows(spec, path=path)

    def input_embed(self):
        spec = self.model_spec.submodules.embedding
        return self.estimate(spec).total_elements

    def vision(self):
        spec = getattr(self.model_spec.submodules, "vision_model", None)
        return self.estimate(spec).dense_elements if spec is not None else 0

    def moe_params(self):
        mlp = get_mlp_spec(get_representative_layer_spec(self.model_spec))
        if not module_is(mlp, MoELayer):
            return 0
        return self.estimate(mlp.submodules.experts).expert_elements

    def shared_expert_params(self):
        mlp = get_mlp_spec(get_representative_layer_spec(self.model_spec))
        if not module_is(mlp, MoELayer) or mlp.submodules.shared_experts is None:
            return 0
        return self.estimate(mlp.submodules.shared_experts).total_elements

    def dense_params(self):
        mlp = get_mlp_spec(get_representative_layer_spec(self.model_spec))
        if module_is(mlp, MoELayer):
            return 0
        # The fused pre-MLP norm belongs to linear_fc1 in the spec.  Keep this
        # legacy helper focused on the three MLP matrices.
        total = self.estimate(mlp).total_elements
        fused_norm = mlp.submodules.linear_fc1.params.get("fused_norm_scope")
        if fused_norm:
            total -= self.config.transformer.hidden_size
        return total

    def attention_params(self):
        attention = get_attention_spec(get_representative_layer_spec(self.model_spec))
        total = self.estimate(attention).total_elements
        fused_norm = None
        if hasattr(attention.submodules, "linear_qkv"):
            fused_norm = attention.submodules.linear_qkv.params.get("fused_norm_scope")
        elif hasattr(attention.submodules, "linear_q"):
            fused_norm = attention.submodules.linear_q.params.get("fused_norm_scope")
        elif hasattr(attention.submodules, "in_proj_qkv"):
            fused_norm = attention.submodules.in_proj_qkv.params.get("fused_norm_scope")
        elif attention.submodules.linear_q_down_proj is not None:
            fused_norm = attention.submodules.linear_q_down_proj.params.get("fused_norm_scope")
        elif attention.submodules.linear_q_proj is not None:
            fused_norm = attention.submodules.linear_q_proj.params.get("fused_norm_scope")
        if fused_norm:
            total -= self.config.transformer.hidden_size
        return total

    def transformer_block(self):
        layer = get_representative_layer_spec(self.model_spec)
        return self.estimate(layer).total_elements

    def output_layer(self):
        decoder_norm = self.model_spec.submodules.decoder.submodules.layer_norm
        output = self.model_spec.submodules.output_layer
        return self.estimate(decoder_norm).total_elements + self.estimate(output).total_elements

    def mtp(self):
        spec = getattr(self.model_spec.submodules, "mtp", None)
        return self.estimate(spec) if spec is not None else None
