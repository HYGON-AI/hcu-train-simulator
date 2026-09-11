# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from hcu_train_simulator.modeling.te_spec import build_transformer_engine_model_spec
from hcu_train_simulator.models.base import (
    ModelAdapter,
    NormalizedModelInput,
    with_rmsnorm_defaults,
)
from hcu_train_simulator.models.registry import register_model_adapter


class Qwen3Adapter(ModelAdapter):
    name = "qwen3"
    model_types = ("qwen3", "qwen3_moe")
    model_type_prefixes = ("qwen3_",)
    architecture_prefixes = ("Qwen3For", "Qwen3MoeFor")

    def normalize_config(self, raw_config):
        return NormalizedModelInput(language=with_rmsnorm_defaults(raw_config))

    def build_spec(self, simulation_config):
        return build_transformer_engine_model_spec(
            simulation_config.transformer,
            simulation_config.model,
            qk_layernorm=True,
            moe_grouped_gemm=simulation_config.parallel.moe_grouped_gemm,
        )


register_model_adapter(Qwen3Adapter())
