# Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from hcu_train_simulator.modeling.te_spec import (
    build_multi_token_predictor_spec,
    build_transformer_engine_model_spec,
)
from hcu_train_simulator.models.base import (
    ModelAdapter,
    NormalizedModelInput,
    with_rmsnorm_defaults,
)
from hcu_train_simulator.models.registry import register_model_adapter


class DeepSeekV3Adapter(ModelAdapter):
    name = "deepseek_v3"
    model_types = ("deepseek_v3", "deepseek_v2")
    model_type_prefixes = ("deepseek",)
    architecture_prefixes = ("Deepseek", "DeepSeek")

    def normalize_config(self, raw_config):
        return NormalizedModelInput(language=with_rmsnorm_defaults(raw_config))

    def build_spec(self, simulation_config):
        transformer = simulation_config.transformer
        uses_mla = bool(
            transformer.kv_lora_rank
            and transformer.qk_nope_head_dim
            and transformer.qk_rope_head_dim
        )
        return build_transformer_engine_model_spec(
            transformer,
            simulation_config.model,
            multi_latent_attention=uses_mla,
            mtp_model_spec=(
                build_multi_token_predictor_spec()
                if transformer.mtp_num_layers
                else None
            ),
            moe_grouped_gemm=simulation_config.parallel.moe_grouped_gemm,
        )

    def validate(self, simulation_config) -> None:
        transformer = simulation_config.transformer
        if transformer.moe_layer_freq <= 0:
            raise AssertionError("moe_layer_freq must be positive.")
        if not 0 <= transformer.first_k_dense_replace <= transformer.num_layers:
            raise AssertionError(
                "first_k_dense_replace must be between zero and num_hidden_layers."
            )


register_model_adapter(DeepSeekV3Adapter())

# Register DeepSeek-specific measured operator plans after the adapter exists.
from hcu_train_simulator.models.deepseek_v3 import benchmarks as _benchmarks  # noqa: E402,F401
