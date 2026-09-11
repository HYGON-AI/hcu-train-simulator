# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from hcu_train_simulator.modeling.te_spec import (
    build_multi_token_predictor_spec,
    build_transformer_engine_model_spec,
)
from hcu_train_simulator.models.base import (
    ModelAdapter,
    NormalizedModelInput,
    merge_language_metadata,
    with_rmsnorm_defaults,
)
from hcu_train_simulator.models.common.qwen_vl import build_vision_model_spec
from hcu_train_simulator.models.registry import register_model_adapter
from hcu_train_simulator.models.qwen3_5.spec import (
    build_gated_attention_spec,
    build_gated_delta_net_spec,
)


class Qwen35Adapter(ModelAdapter):
    name = "qwen3_5"
    model_types = ("qwen3_5", "qwen3_5_moe", "qwen3_5_text", "qwen3_5_moe_text")
    model_type_prefixes = ("qwen3_5",)
    architecture_prefixes = ("Qwen3_5",)

    def normalize_config(self, raw_config):
        nested = raw_config.get("text_config")
        language = with_rmsnorm_defaults(
            merge_language_metadata(
                nested if isinstance(nested, dict) else raw_config,
                raw_config,
            )
        )
        vision = raw_config.get("vision_config")
        return NormalizedModelInput(
            language=language,
            vision=dict(vision) if isinstance(vision, dict) else None,
        )

    def build_spec(self, simulation_config):
        transformer = simulation_config.transformer
        layer_types = transformer.layer_types
        if not layer_types:
            interval = transformer.full_attention_interval
            layer_types = tuple(
                "full_attention"
                if (index + 1) % interval == 0
                else "linear_attention"
                for index in range(transformer.num_layers)
            )
        attention_specs = tuple(
            build_gated_delta_net_spec() if layer_type == "linear_attention" else build_gated_attention_spec()
            for layer_type in layer_types
        )
        vision_spec = None
        if simulation_config.vision.enabled:
            vision_spec = build_vision_model_spec(
                simulation_config.vision,
                model_type="qwen3_5_vision",
            )
        return build_transformer_engine_model_spec(
            transformer,
            simulation_config.model,
            qk_layernorm=True,
            layer_attention_specs=attention_specs,
            mtp_attention_spec=build_gated_attention_spec(),
            mtp_model_spec=(
                build_multi_token_predictor_spec()
                if transformer.mtp_num_layers
                else None
            ),
            vision_model_spec=vision_spec,
            moe_grouped_gemm=simulation_config.parallel.moe_grouped_gemm,
        )

    def validate(self, simulation_config) -> None:
        transformer = simulation_config.transformer
        if transformer.full_attention_interval <= 0:
            raise AssertionError("full_attention_interval must be positive.")
        if transformer.layer_types and len(transformer.layer_types) != transformer.num_layers:
            raise AssertionError("layer_types must contain exactly num_hidden_layers entries.")


register_model_adapter(Qwen35Adapter())

# Import registrations after the adapter class is defined.  These modules only
# extend generic registries; estimators remain unaware of the model family.
from hcu_train_simulator.models.common.qwen_vl.benchmarks import (  # noqa: E402
    register_vision_benchmark_mapping,
)

register_vision_benchmark_mapping(Qwen35Adapter.name)

from hcu_train_simulator.models.qwen3_5 import costs as _costs  # noqa: E402,F401
from hcu_train_simulator.models.qwen3_5 import compute as _compute  # noqa: E402,F401
from hcu_train_simulator.models.qwen3_5 import benchmarks as _benchmarks  # noqa: E402,F401
from hcu_train_simulator.models.qwen3_5 import statistics as _statistics  # noqa: E402,F401
from hcu_train_simulator.models.qwen3_5 import communication as _communication  # noqa: E402,F401
