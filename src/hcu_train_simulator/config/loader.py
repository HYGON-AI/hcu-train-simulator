# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import os
import json
import yaml
from pathlib import Path

from hcu_train_simulator.logging import get_logger
from hcu_train_simulator.config.models import SimulationConfig
from hcu_train_simulator.context import get_config, set_config
from hcu_train_simulator.modeling.architecture import (
    attention_output_dim,
    kv_projection_dim,
    query_projection_dim,
    shared_expert_intermediate_total,
    uses_mla,
)
from hcu_train_simulator.modeling.spec import spec_uses_gated_delta_net, spec_uses_vision
from hcu_train_simulator.models.registry import get_model_adapter


logger = get_logger(__name__)


def load_config(yaml_path):
    # prase yaml and model config
    yaml_path = Path(yaml_path).resolve()
    with open(yaml_path, 'r') as stream:
        estimator_config = yaml.safe_load(stream)

    estimator_config["_config_dir"] = str(yaml_path.parent)

    model_path = estimator_config['model_path']
    if not model_path.endswith('.json'):
        model_path = os.path.join(model_path, 'config.json')

    # read model config file
    with open(model_path, 'r') as f:
        model_config = json.load(f)

    return model_config, estimator_config


def validate_config():
    """
    参数校验
    """
    config = get_config()
    parallel_config = config.parallel
    model_config = config.transformer

    profile_mode = str(config.profile.mode or "auto").lower()
    assert profile_mode in {"auto", "builtin", "theoretical"}, \
        "profile_config.mode must be one of: auto, builtin, theoretical."
    config.profile.mode = profile_mode
    if profile_mode == "builtin":
        assert all(
            (
                config.profile.builtin_accelerator,
                config.profile.builtin_torch,
                config.profile.builtin_dtk,
            )
        ), "builtin profile mode requires accelerator, torch, and dtk selectors."

    assert model_config.hidden_size % parallel_config.tp_size == 0, \
        "hidden_size must be divisible by tp_size."
    assert model_config.num_attention_heads % parallel_config.tp_size == 0, \
        "num_attention_heads must be divisible by tp_size."
    if uses_mla(config):
        assert query_projection_dim(config) % parallel_config.tp_size == 0, \
            "MLA query projection dim must be divisible by tp_size."
        assert kv_projection_dim(config) % parallel_config.tp_size == 0, \
            "MLA KV up projection dim must be divisible by tp_size."
        assert attention_output_dim(config) % parallel_config.tp_size == 0, \
            "MLA attention output dim must be divisible by tp_size."
    else:
        assert model_config.num_query_groups % parallel_config.tp_size == 0, \
            "num_query_groups must be divisible by tp_size."

    if spec_uses_gated_delta_net(config.model_spec):
        assert model_config.linear_num_key_heads % parallel_config.tp_size == 0, \
            "linear_num_key_heads must be divisible by tp_size."
        assert model_config.linear_num_value_heads % parallel_config.tp_size == 0, \
            "linear_num_value_heads must be divisible by tp_size."
        assert len(model_config.layer_types) in {0, model_config.num_layers}, \
            "layer_types must contain exactly num_hidden_layers entries."

    if spec_uses_vision(config.model_spec):
        vision = config.vision
        assert vision.hidden_size % parallel_config.tp_size == 0, \
            "vision hidden_size must be divisible by tp_size."
        assert vision.num_attention_heads % parallel_config.tp_size == 0, \
            "vision num_heads must be divisible by tp_size."
        assert vision.ffn_hidden_size % parallel_config.tp_size == 0, \
            "vision intermediate_size must be divisible by tp_size."
        vision_seq_length = parallel_config.vision_seq_length or vision.num_position_embeddings
        assert vision_seq_length > 0, \
            "vision_seq_length or vision_config.num_position_embeddings must be positive."
        assert parallel_config.vision_num_images > 0, \
            "vision_num_images must be positive."

    shared_intermediate = shared_expert_intermediate_total(config)
    if shared_intermediate:
        assert shared_intermediate % parallel_config.tp_size == 0, \
            "shared expert intermediate size must be divisible by tp_size."

    if not parallel_config.decoder_first_pipeline_num_layers and not parallel_config.decoder_last_pipeline_num_layers:
        assert model_config.num_layers % parallel_config.pp_size == 0, \
            "num_layers must be divisible by pp_size."
    else:
        num_layers_first_pp_stage = parallel_config.decoder_first_pipeline_num_layers if parallel_config.decoder_first_pipeline_num_layers else 0
        num_layers_last_pp_stage = parallel_config.decoder_last_pipeline_num_layers if parallel_config.decoder_last_pipeline_num_layers else 0

        # 肯定有一个不是None
        if num_layers_first_pp_stage > 0 and num_layers_last_pp_stage > 0:
            num_pp_stages_spec = 2
        else:
            num_pp_stages_spec = 1
        assert (model_config.num_layers - num_layers_first_pp_stage - num_layers_last_pp_stage) % (
                parallel_config.pp_size - num_pp_stages_spec) == 0, \
            "num_hidden_layers must be divisible by pp_size."

    if parallel_config.num_layers_per_vp_stage:
        assert parallel_config.pp_size > 1, "vpp can only be enabled when pp > 1."
        assert not (
            parallel_config.decoder_first_pipeline_num_layers or parallel_config.decoder_last_pipeline_num_layers
        ), "vpp does not support custom first/last pipeline layer counts."
        num_layers_per_pp_rank = model_config.num_layers // parallel_config.pp_size
        assert num_layers_per_pp_rank % parallel_config.num_layers_per_vp_stage == 0, \
            "num_hidden_layers / pp_size must be divisible by num_layers_per_vp_stage."

    pp_schedule = (parallel_config.pp_schedule or "1f1b").lower()
    assert pp_schedule in {"1f1b", "interleaved", "interleaved_1f1b"}, \
        "pp_schedule must be one of: 1f1b, interleaved, interleaved_1f1b."
    if pp_schedule in {"interleaved", "interleaved_1f1b"}:
        assert parallel_config.num_layers_per_vp_stage, \
            "interleaved pp_schedule requires num_layers_per_vp_stage."
    parallel_config.pp_schedule = pp_schedule

    if parallel_config.overlap_p2p_comm:
        assert pp_schedule in {"interleaved", "interleaved_1f1b"}, \
            "overlap_p2p_comm requires an interleaved VPP schedule."
        assert parallel_config.num_layers_per_vp_stage, \
            "overlap_p2p_comm requires num_layers_per_vp_stage."

    if parallel_config.decoder_first_pipeline_num_layers or parallel_config.decoder_last_pipeline_num_layers:
        assert parallel_config.pp_size > 1, "Specifying the num layers of pp stage is enabled only when pp_size > 1."

    num_parallel_gpus = parallel_config.tp_size * parallel_config.pp_size * parallel_config.cp_size
    assert parallel_config.num_gpus % num_parallel_gpus == 0, \
        "num_gpus must be divisible by the number of GPUs used."

    overlap_mode = (parallel_config.overlap_mode or "auto").lower()
    assert overlap_mode in {"auto", "manual", "none"}, \
        "overlap_mode must be one of: auto, manual, none."
    parallel_config.overlap_mode = overlap_mode

    if parallel_config.pp_size > 1 and not parallel_config.overlap_p2p_comm:
        logger.info(
            "PP P2P communication overlap is disabled; set overlap_p2p_comm=true "
            "only for an interleaved VPP schedule."
        )

    ep_communication_backend = (
        parallel_config.ep_communication_backend or "alltoall"
    ).lower()
    assert ep_communication_backend in {"alltoall", "deepep"}, \
        "ep_communication_backend must be one of: alltoall, deepep."
    parallel_config.ep_communication_backend = ep_communication_backend
    if parallel_config.pp_size == 1:
        logger.info("PP1 disables EP communication overlap in the exposure model.")
    elif overlap_mode == "auto" and not parallel_config.ep_overlap_enabled:
        logger.info(
            "EP communication overlap is disabled; set ep_overlap_enabled=true "
            "only when the training framework overlaps MoE EP dispatch/combine."
        )

    overlap_ratios = [
        parallel_config.tp_overlap_ratio,
        parallel_config.dp_overlap_ratio,
        parallel_config.ep_overlap_ratio,
        parallel_config.pp_overlap_ratio,
        parallel_config.cp_overlap_ratio,
        parallel_config.etp_overlap_ratio,
        parallel_config.edp_overlap_ratio,
    ]
    assert all(ratio is None or 0 <= ratio <= 1 for ratio in overlap_ratios), \
        "overlap ratios must be in [0, 1]."

    get_model_adapter(config.model_adapter).validate(config)

    logger.info("Arguments validation successful")


def initialize_simulation(yaml_path):

    # load yaml
    model_config, estimator_config = load_config(yaml_path)

    # Build the process-local simulation configuration.
    config = SimulationConfig.from_dict(model_config, estimator_config)

    set_config(config)
