# Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from hcu_train_simulator.logging import log_table
from collections import Counter
from hcu_train_simulator.modeling.spec import spec_uses_moe
from hcu_train_simulator.modeling.spec import (
    get_attention_spec,
    get_layer_specs,
    get_mlp_spec,
    spec_has_qk_norm,
)


def _expert_data_parallel_size(config):
    if not spec_uses_moe(config.model_spec):
        return 1
    return config.parallel.num_gpus // (
        config.parallel.pp_size * config.parallel.ep_size * config.parallel.etp_size
    )


def parallel_strategy_rows(config):
    values = {
        "world_size": config.parallel.num_gpus,
        "tensor_model_parallel_size": config.parallel.tp_size,
        "pipeline_model_parallel_size": config.parallel.pp_size,
        "context_parallel_size": config.parallel.cp_size,
        "data_parallel_size": config.parallel.dp_size,
        "expert_model_parallel_size": config.parallel.ep_size,
        "expert_tensor_parallel_size": config.parallel.etp_size,
        "expert_data_parallel_size": _expert_data_parallel_size(config),
        "expert_communication_backend": config.parallel.ep_communication_backend,
        "overlap_mode": config.parallel.overlap_mode,
        "overlap_p2p_comm": config.parallel.overlap_p2p_comm,
        "ep_overlap_enabled": config.parallel.ep_overlap_enabled,
        "virtual_pipeline_model_parallel_size": max(config.parallel.vp_size, 1),
        "micro_batch_size": config.parallel.micro_batch_size,
        "global_batch_size": config.parallel.global_batch_size,
        "seq_length": config.parallel.seq_length,
    }
    if config.vision.enabled:
        values["vision_seq_length"] = (
            config.parallel.vision_seq_length or config.vision.num_position_embeddings
        )
        values["vision_num_images"] = config.parallel.vision_num_images
    return [{"megatron_parameter": key, "value": value} for key, value in values.items()]


def log_parallel_strategy(logger, config):
    log_table(
        logger,
        "parallel strategy",
        parallel_strategy_rows(config),
    )


def model_spec_rows(config):
    layers = get_layer_specs(config.model_spec)
    base_layers = [layer for layer in layers if not layer.metainfo.get("is_mtp_layer", False)]
    first_layer = base_layers[0]
    attention = get_attention_spec(first_layer)
    mlp = get_mlp_spec(first_layer)
    attention_counts = Counter(get_attention_spec(layer).module.__name__ for layer in base_layers)
    values = {
        "model_adapter": config.model_adapter,
        "backend": config.model_spec.metainfo.get("backend", "/"),
        "model_type": config.model.model_type or "/",
        "transformer_layer": first_layer.module.__name__,
        "self_attention": attention.module.__name__,
        "attention_pattern": ", ".join(
            f"{name} x{count}" for name, count in attention_counts.items()
        ),
        "mlp": mlp.module.__name__,
        "qk_layernorm": spec_has_qk_norm(config.model_spec),
        "num_layers": len(base_layers),
        "mtp_num_layers": len(layers) - len(base_layers),
    }
    if config.vision.enabled:
        values.update(
            {
                "vision_encoder": "VisionModel",
                "vision_num_layers": config.vision.num_layers,
                "vision_hidden_size": config.vision.hidden_size,
                "vision_output_hidden_size": config.vision.output_hidden_size,
                "vision_deepstack_mergers": len(
                    config.vision.deepstack_visual_indexes
                ),
                "vision_deepstack_visual_indexes": (
                    list(config.vision.deepstack_visual_indexes)
                    if config.vision.deepstack_visual_indexes
                    else "/"
                ),
                "vision_pipeline_stage": 0,
            }
        )
    return [{"model_spec_parameter": key, "value": value} for key, value in values.items()]


def log_model_spec(logger, config):
    log_table(logger, "resolved model spec", model_spec_rows(config))


def log_measured_compute_environment(logger, env_info):
    log_table(
        logger,
        "compute estimate environment",
        [
            {
                "compute_time_source": "measured operator benchmark",
                "accelerator": env_info.get("accelerator") or "/",
                "gpu_count": env_info.get("gpu_count") or 0,
                "torch": env_info.get("torch_version") or "/",
                "transformer_engine": env_info.get("te_version") or "/",
                "flash_attn": env_info.get("fa_version") or "/",
                "triton": env_info.get("triton_version") or "/",
                "cuda": env_info.get("cuda_version") or "/",
                "driver": env_info.get("driver") or "/",
            }
        ],
    )
