# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import unittest
from dataclasses import replace

from hcu_train_simulator.config.models import SimulationConfig
from hcu_train_simulator.context import reset_config, set_config
from hcu_train_simulator.modeling import ComputeModel, ModuleSpecMemoryModel
from hcu_train_simulator.modeling.parameters import ParameterModel
from hcu_train_simulator.modeling.spec import get_representative_layer_spec
from hcu_train_simulator.modeling.spec import (
    GatedDeltaNet,
    GatedSelfAttention,
    MultiTokenPredictor,
    VisionModel,
    get_attention_spec,
    get_layer_specs,
    module_is,
)
from hcu_train_simulator.benchmarks.operators import (
    build_benchmark_config,
    filtered_plan,
    moe_tokens_per_local_expert as benchmark_moe_tokens_per_local_expert,
)
from hcu_train_simulator.benchmarks.profile import OperatorProfileStore
from hcu_train_simulator.benchmarks.registry import get_benchmark_case
from hcu_train_simulator.simulator import get_benchmark_module_map
from hcu_train_simulator.config.loader import validate_config
from hcu_train_simulator.estimators import estimate_communication, estimate_compute, estimate_memory
from hcu_train_simulator.estimators.communication import moe_etp_activation_payload_bytes
from hcu_train_simulator.modeling.statistics import ModelStatistics
from hcu_train_simulator.modeling.compute_registry import (
    get_attention_lowering,
    get_component_lowering,
)
from hcu_train_simulator.models.registry import resolve_model_adapter
from hcu_train_simulator.parallelism import build_pipeline_layer_layout


def build_config(*, moe=False):
    model = {
        "architectures": ["Qwen3ForCausalLM"],
        "vocab_size": 32000,
        "hidden_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "intermediate_size": 512,
        "head_dim": 32,
        "tie_word_embeddings": False,
    }
    if moe:
        model.update(
            {
                "moe_intermediate_size": 64,
                "num_experts": 8,
                "num_experts_per_tok": 2,
            }
        )
    estimator = {
        "parallel_config": {
            "tp_size": 1,
            "cp_size": 1,
            "ep_size": 2 if moe else 1,
            "etp_size": 1,
            "pp_size": 1,
            "num_layers_per_vp_stage": None,
            "use_distributed_optimizer": True,
            "full_recompute": False,
            "decoder_first_pipeline_num_layers": None,
            "decoder_last_pipeline_num_layers": None,
            "micro_batch_size": 1,
            "global_batch_size": 2,
            "seq_length": 16,
            "num_gpus": 2,
            "tp_overlap_ratio": 0,
            "dp_overlap_ratio": 0,
            "ep_overlap_ratio": 0,
            "pp_overlap_ratio": 0,
        },
        "hardware_config": {
            "fp16_tflops": 480,
            "fp8_tflops": 960,
            "gpus_per_node": 2,
            "hbm_gib": 64,
            "intra_bw_gbps": 448,
            "inter_bw_gbps": 128,
            "gemm_efficiency": 0.4,
            "p2p_intra_efficiency": 0.8,
            "collective_intra_efficiency": 0.7,
            "collective_inter_efficiency": 0.8,
        },
    }
    return SimulationConfig.from_dict(model, estimator)


def build_qwen35_config(*, moe=False):
    text = {
        "model_type": "qwen3_5_moe_text" if moe else "qwen3_5_text",
        "vocab_size": 2048,
        "hidden_size": 128,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 32,
        "intermediate_size": 256,
        "hidden_act": "silu",
        "rms_norm_eps": 1e-6,
        "attn_output_gate": True,
        "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 32,
        "linear_value_head_dim": 32,
        "linear_num_key_heads": 4,
        "linear_num_value_heads": 4,
        "mtp_num_hidden_layers": 1,
        "rope_parameters": {"partial_rotary_factor": 0.25},
    }
    if moe:
        text.update(
            {
                "num_experts": 8,
                "num_experts_per_tok": 2,
                "moe_intermediate_size": 64,
                "shared_expert_intermediate_size": 64,
            }
        )
    model = {
        "architectures": ["Qwen3_5MoeForConditionalGeneration" if moe else "Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5_moe" if moe else "qwen3_5",
        "text_config": text,
        "vision_config": {
            "depth": 2,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_heads": 4,
            "in_channels": 3,
            "patch_size": 2,
            "temporal_patch_size": 2,
            "spatial_merge_size": 2,
            "out_hidden_size": 128,
            "num_position_embeddings": 16,
        },
        "tie_word_embeddings": False,
    }
    estimator = {
        "parallel_config": {
            "tp_size": 1,
            "cp_size": 1,
            "ep_size": 2 if moe else 1,
            "etp_size": 1,
            "pp_size": 1,
            "num_layers_per_vp_stage": None,
            "use_distributed_optimizer": True,
            "full_recompute": False,
            "decoder_first_pipeline_num_layers": None,
            "decoder_last_pipeline_num_layers": None,
            "micro_batch_size": 1,
            "global_batch_size": 2,
            "seq_length": 16,
            "vision_seq_length": 16,
            "vision_num_images": 1,
            "num_gpus": 2,
            "tp_overlap_ratio": 0,
            "dp_overlap_ratio": 0,
            "ep_overlap_ratio": 0,
            "pp_overlap_ratio": 0,
        },
        "hardware_config": {
            "fp16_tflops": 480,
            "fp8_tflops": 960,
            "gpus_per_node": 2,
            "hbm_gib": 64,
            "intra_bw_gbps": 448,
            "inter_bw_gbps": 128,
            "gemm_efficiency": 0.4,
            "p2p_intra_efficiency": 0.8,
            "collective_intra_efficiency": 0.7,
            "collective_inter_efficiency": 0.8,
        },
    }
    return SimulationConfig.from_dict(model, estimator)


class ModuleSpecEstimatorTest(unittest.TestCase):
    def tearDown(self):
        reset_config()

    def test_memory_tree_sums_concrete_module_costs(self):
        config = build_config()
        model = ModuleSpecMemoryModel(config)
        layer = get_representative_layer_spec(config.model_spec)

        parameters = model.parameter_estimate(layer)
        activations = model.activation_estimate(layer)
        rows = model.memory_rows(layer, path="layer")

        self.assertEqual(parameters.total_elements, sum(row["param_elems"] for row in rows))
        self.assertEqual(activations.saved_elements, sum(row["saved_act_elems"] for row in rows))
        self.assertIn("qkv_weight", {row["role"] for row in rows})
        self.assertIn("linear_fc1", {row["role"] for row in rows})

    def test_compute_rows_are_traceable_to_module_specs(self):
        config = build_config()
        set_config(config)

        result = ComputeModel().spec_summary()
        qkv = next(row for row in result if row["model_part"] == "qkv_weight")

        self.assertEqual(qkv["module_role"], "qkv_weight")
        self.assertEqual(qkv["module_type"], "TELayerNormColumnParallelLinear")
        self.assertIn("self_attention.linear_qkv", qkv["module_path"])
        self.assertEqual(qkv["module_count"], config.transformer.num_layers)

    def test_moe_expert_parameters_remain_separate_from_dense_parameters(self):
        config = build_config(moe=True)
        model = ModuleSpecMemoryModel(config)
        layer = get_representative_layer_spec(config.model_spec)

        parameters = model.parameter_estimate(layer)
        rows = model.memory_rows(layer, path="layer")

        self.assertGreater(parameters.expert_elements, 0)
        self.assertGreater(parameters.dense_elements, 0)
        self.assertEqual(
            parameters.expert_elements,
            sum(row["expert_param_elems"] for row in rows),
        )
        self.assertIn("topk_router", {row["role"] for row in rows})

    def test_etp_preserves_expert_flops_and_updates_benchmark_shape(self):
        configs = []
        for etp_size in (1, 2):
            config = build_config(moe=True)
            config.parallel.num_gpus = 4
            config.parallel.dp_size = 4
            config.parallel.etp_size = etp_size
            configs.append(config)

        shapes = []
        benchmark_tokens = []
        swiglu_elements = []
        for config in configs:
            set_config(config)
            compute = ComputeModel()
            shapes.append((compute.moe_linear_fc1(), compute.moe_linear_fc2()))
            swiglu_elements.append(compute.moe_swiglu_element_count())
            benchmark_config = build_benchmark_config(config)
            benchmark_tokens.append(
                benchmark_moe_tokens_per_local_expert(benchmark_config)
            )
            reset_config()

        (fc1_etp1, fc2_etp1), (fc1_etp2, fc2_etp2) = shapes
        self.assertEqual(fc1_etp2[2], 2 * fc1_etp1[2])
        self.assertEqual(fc1_etp2[1], fc1_etp1[1] // 2)
        self.assertEqual(fc2_etp2[2], 2 * fc2_etp1[2])
        self.assertEqual(fc2_etp2[3], fc2_etp1[3] // 2)
        self.assertEqual(
            fc1_etp1[0] * fc1_etp1[1] * fc1_etp1[2] * fc1_etp1[3],
            fc1_etp2[0] * fc1_etp2[1] * fc1_etp2[2] * fc1_etp2[3],
        )
        self.assertEqual(
            fc2_etp1[0] * fc2_etp1[1] * fc2_etp1[2] * fc2_etp1[3],
            fc2_etp2[0] * fc2_etp2[1] * fc2_etp2[2] * fc2_etp2[3],
        )
        self.assertEqual(swiglu_elements[0], swiglu_elements[1])
        self.assertEqual(benchmark_tokens[1], 2 * benchmark_tokens[0])

    def test_etp_payload_uses_full_collective_tensor(self):
        config = build_config(moe=True)
        config.parallel.etp_size = 2
        config.parallel.num_gpus = 4
        config.parallel.dp_size = 4

        expected = (
            2
            * config.parallel.micro_batch_size
            * config.parallel.seq_length
            * config.transformer.hidden_size
            * config.transformer.moe_router_topk
            / config.parallel.cp_size
            / config.parallel.tp_size
            * config.parallel.etp_size
        )
        self.assertEqual(moe_etp_activation_payload_bytes(config), expected)

    def test_optimizer_compute_shards_dense_and_expert_params_separately(self):
        config = build_config(moe=True)
        config.parallel.num_gpus = 8
        config.parallel.dp_size = 4
        config.parallel.cp_size = 2
        config.parallel.etp_size = 1
        set_config(config)

        params = ParameterModel().estimate(config.model_spec)
        edp_size = (
            config.parallel.num_gpus
            / config.parallel.pp_size
            / config.parallel.ep_size
            / config.parallel.etp_size
        )
        expected = (
            params.dense_elements / (config.parallel.dp_size * config.parallel.cp_size)
            + params.expert_elements / edp_size
        )
        self.assertEqual(ComputeModel().optimizer_param_elements(), expected)

    def test_sequence_parallel_only_shards_sequence_owned_operations(self):
        config = build_config(moe=True)
        config.parallel.tp_size = 2
        config.parallel.dp_size = 1
        config.parallel.num_gpus = 2
        config.parallel.sequence_parallel = True
        set_config(config)
        sp_compute = ComputeModel()
        sp_router = sp_compute.topk_router()
        sp_norm = sp_compute.transformer_element_count(sequence_sharded=True)
        sp_qk_norm = sp_compute.qk_norm_element_count()
        reset_config()

        config.parallel.sequence_parallel = False
        set_config(config)
        no_sp_compute = ComputeModel()
        no_sp_router = no_sp_compute.topk_router()
        no_sp_norm = no_sp_compute.transformer_element_count(sequence_sharded=True)
        no_sp_qk_norm = no_sp_compute.qk_norm_element_count()

        self.assertEqual(no_sp_router[3], 2 * sp_router[3])
        self.assertEqual(no_sp_norm, 2 * sp_norm)
        self.assertEqual(no_sp_qk_norm, sp_qk_norm)

        sp_config = build_benchmark_config(config, {"sequence_parallel": True})
        no_sp_config = build_benchmark_config(config, {"sequence_parallel": False})
        self.assertEqual(no_sp_config["sequence_tokens"], 2 * sp_config["sequence_tokens"])
        self.assertEqual(no_sp_config["tokens"], sp_config["tokens"])

    def test_flash_attention_profile_key_includes_kv_sequence_length(self):
        config = build_config()
        store = OperatorProfileStore(None, config, data={})
        local_kv_key = store.flash_attention_key(
            "bf16",
            {"b": 1, "seq": 8, "kv_seq": 8, "heads": 2, "head_dim": 32},
        )
        global_kv_key = store.flash_attention_key(
            "bf16",
            {"b": 1, "seq": 8, "kv_seq": 16, "heads": 2, "head_dim": 32},
        )

        self.assertNotEqual(local_kv_key, global_kv_key)
        self.assertIn("kv_seq=8", local_kv_key)
        self.assertIn("kv_seq=16", global_kv_key)

    def test_validation_rejects_invalid_expert_parallel_factorization(self):
        config = build_config(moe=True)
        config.parallel.etp_size = 2
        set_config(config)

        with self.assertRaisesRegex(
            AssertionError,
            "pp_size \\* ep_size \\* etp_size",
        ):
            validate_config()

    def test_pipeline_layout_uses_each_concrete_layer_spec(self):
        dense_config = build_config()
        moe_config = build_config(moe=True)
        dense_layer = get_representative_layer_spec(dense_config.model_spec)
        moe_layer = get_representative_layer_spec(moe_config.model_spec)

        decoder = moe_config.model_spec.submodules.decoder
        decoder = replace(
            decoder,
            submodules=replace(decoder.submodules, layer_specs=[dense_layer, moe_layer]),
        )
        moe_config.model_spec = replace(
            moe_config.model_spec,
            submodules=replace(moe_config.model_spec.submodules, decoder=decoder),
        )
        moe_config.parallel.pp_size = 2
        layout = build_pipeline_layer_layout(moe_config.model_spec, moe_config.parallel)
        model = ModuleSpecMemoryModel(moe_config)

        first_stage = model.parameter_estimate(layout[0][0][0])
        second_stage = model.parameter_estimate(layout[1][0][0])

        self.assertEqual(first_stage.expert_elements, 0)
        self.assertGreater(second_stage.expert_elements, 0)

    def test_qwen35_nested_vlm_config_builds_hybrid_and_vision_specs(self):
        config = build_qwen35_config()
        layers = get_layer_specs(config.model_spec, include_mtp=False)
        attention_types = [get_attention_spec(layer).module for layer in layers]

        self.assertEqual(attention_types[:3], [GatedDeltaNet] * 3)
        self.assertIs(attention_types[3], GatedSelfAttention)
        self.assertTrue(module_is(config.model_spec.submodules.vision_model, VisionModel))
        self.assertTrue(module_is(config.model_spec.submodules.mtp, MultiTokenPredictor))
        self.assertEqual(config.model_adapter, "qwen3_5")
        self.assertEqual(config.transformer.mtp_num_layers, 1)
        self.assertEqual(config.transformer.partial_rotary_factor, 0.25)

    def test_model_adapter_registry_routes_known_and_fallback_families(self):
        cases = (
            ({"model_type": "qwen3"}, "qwen3"),
            ({"model_type": "deepseek_v3"}, "deepseek_v3"),
            ({"model_type": "qwen3_5"}, "qwen3_5"),
            ({"model_type": "future_transformer"}, "generic_gpt"),
        )
        for raw_config, expected in cases:
            with self.subTest(model_type=raw_config["model_type"]):
                self.assertEqual(resolve_model_adapter(raw_config).name, expected)

    def test_qwen35_structural_nodes_have_registered_compute_lowerings(self):
        config = build_qwen35_config()
        layers = get_layer_specs(config.model_spec, include_mtp=False)
        attention_specs = [get_attention_spec(layer) for layer in layers]

        self.assertTrue(all(get_attention_lowering(spec) for spec in attention_specs))
        self.assertIsNotNone(
            get_component_lowering(config.model_spec.submodules.vision_model)
        )
        self.assertIsNotNone(get_component_lowering(config.model_spec.submodules.mtp))

    def test_qwen35_model_size_excludes_mtp_but_training_estimates_include_it(self):
        config = build_qwen35_config()
        set_config(config)
        model = ModuleSpecMemoryModel(config)

        training_params = model.global_parameter_estimate()
        model_params = model.model_parameter_estimate()
        mtp_layer_params = sum(
            model.global_parameter_estimate(layer).total_elements
            for layer in get_layer_specs(config.model_spec)
            if layer.metainfo.get("is_mtp_layer", False)
        )
        mtp_aux_params = model.global_parameter_estimate(
            config.model_spec.submodules.mtp
        ).total_elements

        self.assertEqual(
            training_params.total_elements - model_params.total_elements,
            mtp_layer_params + mtp_aux_params,
        )
        self.assertEqual(ModelStatistics().compute_model_size(), model_params.total_elements)
        self.assertEqual(
            mtp_aux_params,
            2 * config.transformer.hidden_size ** 2
            + 3 * config.transformer.hidden_size,
        )

        memory = estimate_memory()
        compute = ComputeModel().spec_summary()
        self.assertTrue(
            any(
                row["module_path"].startswith("model.mtp")
                for row in memory[-1]["module_breakdown"]
            )
        )
        self.assertTrue(any(row["model_part"] == "mtp_fc" for row in compute))

    def test_qwen35_gdn_conv_parameter_count_has_no_bias(self):
        config = build_qwen35_config()
        model = ModuleSpecMemoryModel(config)
        gdn_layer = get_layer_specs(config.model_spec, include_mtp=False)[0]
        attention = get_attention_spec(gdn_layer)
        conv_row = next(
            row
            for row in model.memory_rows(attention, path="attention")
            if row["role"] == "gdn_causal_conv1d"
        )
        key_dim = (
            config.transformer.linear_num_key_heads
            * config.transformer.linear_key_head_dim
        )
        value_dim = (
            config.transformer.linear_num_value_heads
            * config.transformer.linear_value_head_dim
        )
        expected = (2 * key_dim + value_dim) * config.transformer.linear_conv_kernel_dim

        self.assertEqual(conv_row["param_elems"], expected)

    def test_qwen35_memory_covers_gdn_vision_and_shared_expert_gate(self):
        config = build_qwen35_config(moe=True)
        model = ModuleSpecMemoryModel(config)
        rows = model.memory_rows()
        roles = {row["role"] for row in rows}

        self.assertIn("gdn_in_proj_qkv", roles)
        self.assertIn("vision_patch_embed", roles)
        self.assertIn("vision_patch_merger", roles)
        self.assertIn("shared_experts", roles)
        self.assertEqual(
            model.parameter_estimate().total_elements,
            sum(row["param_elems"] for row in rows),
        )

    def test_qwen35_compute_gemm_and_non_gemm_rows_have_benchmark_cases(self):
        config = build_qwen35_config()
        set_config(config)
        compute_rows = ComputeModel().spec_summary()
        module_map = get_benchmark_module_map(config)
        benchmark_config = build_benchmark_config(config, {"ops": ["all"]})
        benchmark_names = {plan.name for plan in filtered_plan(benchmark_config)}

        required_parts = {
            row["model_part"]
            for row in compute_rows
            if row["op_type"] in {"gemm", "non_gemm", "flash_attention"}
        }
        self.assertTrue({"gdn_in_proj_qkv", "gated_delta_rule", "vision_qkv", "vision_gelu"} <= required_parts)
        self.assertEqual(required_parts - set(module_map), set())
        for part in required_parts:
            mapped = module_map[part]
            mapped = [mapped] if isinstance(mapped, str) else mapped
            self.assertTrue(set(mapped) & benchmark_names, msg=f"missing benchmark case for {part}: {mapped}")
        plugin_plans = [
            plan
            for plan in filtered_plan(benchmark_config)
            if plan.group in {"qwen35", "gdn", "vision"}
        ]
        self.assertTrue(plugin_plans)
        for plan in plugin_plans:
            self.assertIsNotNone(
                get_benchmark_case(plan.name),
                msg=f"model plugin does not implement benchmark case {plan.name}",
            )

    def test_qwen35_vlm_runs_memory_compute_and_communication_estimators(self):
        config = build_qwen35_config(moe=True)
        set_config(config)
        validate_config()

        memory = estimate_memory()
        compute = estimate_compute()
        communication = estimate_communication(compute)

        self.assertGreater(memory[0]["total_gib"], 0)
        self.assertTrue(any("vision_model" in row["module_path"] for row in memory[0]["module_breakdown"]))
        self.assertTrue(any(row["model_part"] == "gated_delta_rule" for row in compute))
        self.assertTrue(any(row["model_part"] == "vision_flash_attn" for row in compute))
        self.assertGreater(communication["total"], 0)
        self.assertGreater(ModelStatistics().compute_flops(), 0)


if __name__ == "__main__":
    unittest.main()
