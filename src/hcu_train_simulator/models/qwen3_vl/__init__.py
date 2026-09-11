# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Qwen3-VL dense and MoE model adapters."""

from hcu_train_simulator.modeling.te_spec import build_transformer_engine_model_spec
from hcu_train_simulator.models.base import (
    ModelAdapter,
    NormalizedModelInput,
    merge_language_metadata,
    with_rmsnorm_defaults,
)
from hcu_train_simulator.models.common.qwen_vl import build_vision_model_spec
from hcu_train_simulator.models.registry import register_model_adapter


class Qwen3VLAdapter(ModelAdapter):
    name = "qwen3_vl"
    model_types = (
        "qwen3_vl",
        "qwen3_vl_moe",
        "qwen3_vl_text",
        "qwen3_vl_moe_text",
    )
    model_type_prefixes = ("qwen3_vl",)
    architecture_prefixes = ("Qwen3VL",)

    def normalize_config(self, raw_config):
        nested = raw_config.get("text_config")
        language = with_rmsnorm_defaults(
            merge_language_metadata(
                nested if isinstance(nested, dict) else raw_config,
                raw_config,
            )
        )

        # Hugging Face selects a sparse layer when
        # (layer_index + 1) % decoder_sparse_step == 0, except for the
        # explicitly dense ``mlp_only_layers``.
        sparse_step = int(language.get("decoder_sparse_step", 1) or 1)
        language.setdefault("moe_layer_freq", sparse_step)
        language.setdefault("moe_layer_offset", sparse_step - 1)
        language.setdefault(
            "moe_dense_layer_indices",
            tuple(language.get("mlp_only_layers") or ()),
        )

        vision = raw_config.get("vision_config")
        return NormalizedModelInput(
            language=language,
            vision=dict(vision) if isinstance(vision, dict) else None,
        )

    def build_spec(self, simulation_config):
        vision_spec = None
        if simulation_config.vision.enabled:
            vision_spec = build_vision_model_spec(
                simulation_config.vision,
                model_type="qwen3_vl_vision",
            )
        return build_transformer_engine_model_spec(
            simulation_config.transformer,
            simulation_config.model,
            qk_layernorm=True,
            vision_model_spec=vision_spec,
            moe_grouped_gemm=simulation_config.parallel.moe_grouped_gemm,
        )

    def validate(self, simulation_config) -> None:
        transformer = simulation_config.transformer
        if transformer.moe_layer_freq <= 0:
            raise AssertionError("decoder_sparse_step must be positive.")
        dense_indexes = transformer.moe_dense_layer_indices
        if len(set(dense_indexes)) != len(dense_indexes):
            raise AssertionError("mlp_only_layers must contain unique indexes.")
        if any(index < 0 or index >= transformer.num_layers for index in dense_indexes):
            raise AssertionError("mlp_only_layers must refer to existing text layers.")

        vision = simulation_config.vision
        if not vision.enabled:
            return
        if vision.output_hidden_size != transformer.hidden_size:
            raise AssertionError(
                "Qwen3-VL vision out_hidden_size must match text hidden_size."
            )
        indexes = vision.deepstack_visual_indexes
        if len(set(indexes)) != len(indexes):
            raise AssertionError("deepstack_visual_indexes must be unique.")
        if any(index < 0 or index >= vision.num_layers for index in indexes):
            raise AssertionError(
                "deepstack_visual_indexes must refer to existing vision layers."
            )
        if len(indexes) > transformer.num_layers:
            raise AssertionError(
                "Qwen3-VL needs at least one text layer per DeepStack feature."
            )


register_model_adapter(Qwen3VLAdapter())

from hcu_train_simulator.models.common.qwen_vl.benchmarks import (  # noqa: E402
    register_vision_benchmark_mapping,
)

register_vision_benchmark_mapping(Qwen3VLAdapter.name)
