# Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from hcu_train_simulator.modeling.te_spec import build_transformer_engine_model_spec
from hcu_train_simulator.models.base import ModelAdapter
from hcu_train_simulator.models.registry import register_model_adapter


class GenericGPTAdapter(ModelAdapter):
    name = "generic_gpt"
    fallback = True

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
            moe_grouped_gemm=simulation_config.parallel.moe_grouped_gemm,
        )


register_model_adapter(GenericGPTAdapter())
