# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility facade for ModuleSpec-driven activation estimation."""

from hcu_train_simulator.context import get_config
from hcu_train_simulator.modeling.module_cost import ModuleSpecMemoryModel
from hcu_train_simulator.modeling.spec import get_representative_layer_spec


class ActivationModel:
    def __init__(self):
        self.config = get_config()
        self.model_spec = self.config.model_spec
        self.estimator = ModuleSpecMemoryModel(self.config)

    def estimate(self, spec=None):
        return self.estimator.activation_estimate(spec)

    def memory_rows(self, spec=None, path="model"):
        return self.estimator.memory_rows(spec, path=path)

    def input_embed(self):
        return self.estimate(self.model_spec.submodules.embedding).total_elements

    def transformer_block(self):
        return self.estimate(get_representative_layer_spec(self.model_spec)).total_elements

    def final_norm(self):
        spec = self.model_spec.submodules.decoder.submodules.layer_norm
        return self.estimate(spec).total_elements

    def lm_head(self):
        return self.estimate(self.model_spec.submodules.output_layer).total_elements
