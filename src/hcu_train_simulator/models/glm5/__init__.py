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
from hcu_train_simulator.models.glm5.spec import build_dsa_attention_spec
from hcu_train_simulator.models.registry import register_model_adapter


class GLM5Adapter(ModelAdapter):
    name = "glm5"
    model_types = ("glm_moe_dsa",)
    model_type_prefixes = ("glm_moe_dsa",)
    architecture_prefixes = ("GlmMoeDsa",)

    def normalize_config(self, raw_config):
        return NormalizedModelInput(language=with_rmsnorm_defaults(raw_config))

    def build_spec(self, simulation_config):
        transformer = simulation_config.transformer
        indexer_types = transformer.indexer_types or ("full",) * transformer.num_layers
        attention_specs = tuple(
            build_dsa_attention_spec(include_indexer=indexer_type == "full")
            for indexer_type in indexer_types
        )
        return build_transformer_engine_model_spec(
            transformer,
            simulation_config.model,
            layer_attention_specs=attention_specs,
            mtp_attention_spec=build_dsa_attention_spec(include_indexer=True),
            mtp_model_spec=(
                build_multi_token_predictor_spec()
                if transformer.mtp_num_layers
                else None
            ),
            moe_grouped_gemm=simulation_config.parallel.moe_grouped_gemm,
        )

    def validate(self, simulation_config) -> None:
        transformer = simulation_config.transformer
        if transformer.q_lora_rank is None or transformer.q_lora_rank <= 0:
            raise AssertionError("GLM-5 DSA requires a positive q_lora_rank.")
        if transformer.index_n_heads <= 0:
            raise AssertionError("index_n_heads must be positive for GLM-5 DSA.")
        if transformer.index_head_dim <= 0:
            raise AssertionError("index_head_dim must be positive for GLM-5 DSA.")
        if transformer.index_topk <= 0:
            raise AssertionError("index_topk must be positive for GLM-5 DSA.")
        if transformer.index_head_dim < transformer.qk_rope_head_dim:
            raise AssertionError("index_head_dim must be at least qk_rope_head_dim.")
        if transformer.indexer_types:
            if len(transformer.indexer_types) != transformer.num_layers:
                raise AssertionError(
                    "indexer_types must contain exactly num_hidden_layers entries."
                )
            if transformer.indexer_types[0] != "full":
                raise AssertionError("the first DSA layer must use a full indexer.")
            if any(mode not in {"full", "shared"} for mode in transformer.indexer_types):
                raise AssertionError("indexer_types entries must be 'full' or 'shared'.")
        if transformer.moe_layer_freq <= 0:
            raise AssertionError("moe_layer_freq must be positive.")
        if not 0 <= transformer.first_k_dense_replace <= transformer.num_layers:
            raise AssertionError(
                "first_k_dense_replace must be between zero and num_hidden_layers."
            )


register_model_adapter(GLM5Adapter())

from hcu_train_simulator.models.glm5 import costs as _costs  # noqa: E402,F401
from hcu_train_simulator.models.glm5 import compute as _compute  # noqa: E402,F401
from hcu_train_simulator.models.glm5 import statistics as _statistics  # noqa: E402,F401
from hcu_train_simulator.models.glm5 import benchmarks as _benchmarks  # noqa: E402,F401
from hcu_train_simulator.models.glm5 import communication as _communication  # noqa: E402,F401
