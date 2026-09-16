# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from importlib.resources import files

from hcu_train_simulator.communication import Algorithm, CommunicationSimulator, CommType, GPUType, GroupType
from hcu_train_simulator.config import initialize_simulation
from hcu_train_simulator.modeling import ComputeModel, ParameterModel
from hcu_train_simulator.modeling.architecture import (
    kv_cache_dim,
    shared_expert_intermediate_total,
)
from hcu_train_simulator.parallelism import build_pipeline_layer_layout, get_num_microbatches
from hcu_train_simulator.context import get_config
from hcu_train_simulator.logging import get_logger, log_table
from hcu_train_simulator.modeling.spec import (
    MoELayer,
    get_mlp_spec,
    module_is,
    spec_uses_moe,
)
from hcu_train_simulator.estimators.overlap_registry import get_overlap_part_sets
from hcu_train_simulator.estimators.deepep import estimate_deepep_phases


BF16_BYTES = 2
FP8_BYTES = 1
DENSE_TP_ALL_GATHER_PER_LAYER = 6
DENSE_TP_REDUCE_SCATTER_PER_LAYER = 4
DENSE_TP_OUTPUT_ALL_GATHER_PER_MICROBATCH = 3
DENSE_TP_OUTPUT_REDUCE_SCATTER_PER_MICROBATCH = 2
logger = get_logger(__name__)


def communication_dtype_bytes(config):
    """Bytes per element for the FP8-enabled TP/ETP/EP activation paths."""

    if getattr(config.parallel, "use_fp8_training", False):
        return FP8_BYTES
    return BF16_BYTES


def get_pp_p2p_total_communication_count(pp_size, num_microbatches, pp_schedule="1f1b", vp_size=1):
    """Total forward/backward messages across all pipeline boundaries."""

    schedule = (pp_schedule or "1f1b").lower()
    if schedule in {"interleaved", "interleaved_1f1b"}:
        virtual_stages = pp_size * max(vp_size, 1)
    else:
        virtual_stages = pp_size
    return 2 * max(virtual_stages - 1, 0) * num_microbatches


def get_pp_p2p_communication_count(pp_size, num_microbatches, pp_schedule="1f1b", vp_size=1):
    """P2P message count on the busiest physical pipeline rank.

    Iteration time is a per-rank critical-path quantity.  Summing every edge in
    the pipeline serializes communication that actually runs concurrently on
    different ranks.  Count both endpoints (forward and backward) owned by
    each physical rank, then use the maximum rank load.
    """

    if pp_size <= 1:
        return 0
    schedule = (pp_schedule or "1f1b").lower()
    virtual_stages = max(vp_size, 1) if schedule in {"interleaved", "interleaved_1f1b"} else 1
    total_stages = pp_size * virtual_stages
    per_rank_counts = []
    for pp_rank in range(pp_size):
        count_per_microbatch = 0
        for vp_rank in range(virtual_stages):
            stage = vp_rank * pp_size + pp_rank
            if stage > 0:
                count_per_microbatch += 2  # forward receive + backward send
            if stage < total_stages - 1:
                count_per_microbatch += 2  # forward send + backward receive
        per_rank_counts.append(count_per_microbatch * num_microbatches)
    return max(per_rank_counts)


def estimate_pp_rank_param_elements(config, model_parameters, pp_rank):
    params = (
        estimate_pp_rank_data_parallel_param_elements(config, model_parameters, pp_rank)
        + estimate_pp_rank_expert_data_parallel_param_elements(config, model_parameters, pp_rank)
    )
    return params


def expert_data_parallel_size(config):
    if not spec_uses_moe(config.model_spec):
        return 1
    return config.parallel.num_gpus // (
        config.parallel.pp_size * config.parallel.ep_size * config.parallel.etp_size
    )


def sequence_parallel_shard(config):
    if getattr(config.parallel, "sequence_parallel", True):
        return max(1, config.parallel.tp_size)
    return 1


def moe_ep_activation_payload_bytes(config):
    return (
        communication_dtype_bytes(config)
        * config.parallel.micro_batch_size
        * config.parallel.seq_length
        * config.transformer.hidden_size
        * config.transformer.moe_router_topk
        / config.parallel.cp_size
        / sequence_parallel_shard(config)
    )


def pp_activation_payload_bytes(config):
    """Bytes in one pipeline activation/gradient tensor.

    Megatron sends the sequence-parallel shard between pipeline ranks, so TP
    divides the payload when sequence parallelism is enabled.
    """

    return (
        BF16_BYTES
        * config.parallel.micro_batch_size
        * config.parallel.seq_length
        * config.transformer.hidden_size
        / config.parallel.cp_size
        / sequence_parallel_shard(config)
    )


def moe_ep_phase_collective_count(config):
    num_microbatches = get_num_microbatches(
        config.parallel.global_batch_size,
        config.parallel.micro_batch_size,
        config.parallel.dp_size,
    )
    layout = build_pipeline_layer_layout(config.model_spec, config.parallel)
    moe_layers_on_critical_rank = max(
        sum(
            module_is(get_mlp_spec(layer), MoELayer)
            for layer_specs in virtual_stages
            for layer in layer_specs
        )
        for virtual_stages in layout
    )
    return moe_layers_on_critical_rank * num_microbatches


def moe_ep_activation_collective_count(config):
    # Dispatch and combine each execute once in forward and once in backward.
    return 4 * moe_ep_phase_collective_count(config)


def estimate_pp_rank_data_parallel_param_elements(config, model_parameters, pp_rank):
    layout = build_pipeline_layer_layout(config.model_spec, config.parallel)
    params = sum(
        model_parameters.estimate(layer).dense_elements
        for layer_specs in layout[pp_rank]
        for layer in layer_specs
    )

    if pp_rank == 0:
        params += model_parameters.input_embed()
        params += model_parameters.vision()
    if pp_rank == config.parallel.pp_size - 1:
        mtp = model_parameters.mtp()
        if mtp is not None:
            params += mtp.dense_elements
        params += model_parameters.output_layer()
    return params


def estimate_pp_rank_expert_data_parallel_param_elements(config, model_parameters, pp_rank):
    if not spec_uses_moe(config.model_spec):
        return 0
    layout = build_pipeline_layer_layout(config.model_spec, config.parallel)
    return sum(
        model_parameters.estimate(layer).expert_elements
        for layer_specs in layout[pp_rank]
        for layer in layer_specs
    )


def estimate_dp_param_elements(config):
    if config.parallel.dp_size <= 1:
        return 0

    model_parameters = ParameterModel()
    return max(
        estimate_pp_rank_data_parallel_param_elements(config, model_parameters, pp_rank)
        for pp_rank in range(config.parallel.pp_size)
    )


def estimate_edp_param_elements(config):
    if expert_data_parallel_size(config) <= 1:
        return 0

    model_parameters = ParameterModel()
    return max(
        estimate_pp_rank_expert_data_parallel_param_elements(config, model_parameters, pp_rank)
        for pp_rank in range(config.parallel.pp_size)
    )


def get_layers_per_pp_rank(config):
    layout = build_pipeline_layer_layout(config.model_spec, config.parallel)
    return [sum(len(layer_specs) for layer_specs in virtual_stages) for virtual_stages in layout]


def get_moe_layers_per_pp_rank(config):
    layout = build_pipeline_layer_layout(config.model_spec, config.parallel)
    return [
        sum(
            module_is(get_mlp_spec(layer), MoELayer)
            for layer_specs in virtual_stages
            for layer in layer_specs
        )
        for virtual_stages in layout
    ]


def estimate_dense_tp_collective_counts(config, dp_size):
    num_microbatches = get_num_microbatches(
        config.parallel.global_batch_size,
        config.parallel.micro_batch_size,
        dp_size,
    )

    counts = []
    for pp_rank, local_layers in enumerate(get_layers_per_pp_rank(config)):
        all_gather_count = DENSE_TP_ALL_GATHER_PER_LAYER * local_layers * num_microbatches
        reduce_scatter_count = DENSE_TP_REDUCE_SCATTER_PER_LAYER * local_layers * num_microbatches

        if pp_rank == config.parallel.pp_size - 1:
            all_gather_count += DENSE_TP_OUTPUT_ALL_GATHER_PER_MICROBATCH * num_microbatches
            reduce_scatter_count += DENSE_TP_OUTPUT_REDUCE_SCATTER_PER_MICROBATCH * num_microbatches

        counts.append((all_gather_count, reduce_scatter_count))
    return counts


def get_module_time_ms(module):
    benchmark_time = module.get("benchmark_time_ms")
    if isinstance(benchmark_time, (int, float)) and not isinstance(benchmark_time, bool):
        return benchmark_time
    return module.get("all_time_ms", module["forward_ms"] + module["backward_ms"])


def sum_module_time_s(compute_result, names=None, mode="all"):
    if not compute_result:
        return 0.0

    total_ms = 0.0
    for module in compute_result:
        if names is not None and module.get("model_part") not in names:
            continue
        if mode == "backward":
            total_ms += module.get("backward_ms", 0)
        else:
            total_ms += get_module_time_ms(module)
    return total_ms / 1000


def clamp(value, lower=0.0, upper=1.0):
    # 将value限制在0和1之间
    return max(lower, min(value, upper))


OVERLAP_DEFAULTS = {
    # Auto mode is intentionally conservative.  Synchronized training traces
    # show roughly 30-40% aggregate communication overlap, while the previous
    # TP defaults hid about 72% whenever the compute window was large enough.
    "tp": {"eff": 0.80, "floor": 0.55, "max": 0.45, "window": 0.30},
    "cp": {"eff": 0.40, "floor": 0.40, "max": 0.40, "window": 0.30},
    "dp": {"eff": 0.70, "floor": 0.40, "max": 0.60, "window": 0.60},
    "pp": {"eff": 0.45, "floor": 0.60, "max": 0.40, "window": 0.20},
    "ep": {"eff": 0.40, "floor": 0.50, "max": 0.40, "window": 0.30},
    "etp": {"eff": 0.45, "floor": 0.45, "max": 0.50, "window": 0.30},
    "edp": {"eff": 0.70, "floor": 0.40, "max": 0.60, "window": 0.60},
}


def get_overlap_part_names():
    """Return registered model parts that can cover each communication domain."""

    transformer_names = {
        "attention_rmsnorm",
        "qk_norm",
        "rope",
        "qkv_weight",
        "mla_q_down",
        "mla_q_up",
        "mla_kv_down",
        "mla_kv_up",
        "flash_attn",
        "attention_softmax",
        "attn_proj",
        "mlp_rmsnorm",
        "linear_fc1",
        "swiglu_activation",
        "linear_fc2",
        "residual_dropout_add",
        "shared_expert_fc1",
        "shared_swiglu_activation",
        "shared_expert_fc2",
        "shared_expert_gate",
    }
    attention_names = {
        "attention_rmsnorm",
        "qk_norm",
        "rope",
        "qkv_weight",
        "mla_q_down",
        "mla_q_up",
        "mla_kv_down",
        "mla_kv_up",
        "flash_attn",
        "attention_softmax",
        "attn_proj",
    }
    expert_names = {
        "mlp_rmsnorm",
        "topk_router",
        "router_select",
        "moe_linear_fc1",
        "moe_swiglu_activation",
        "moe_linear_fc2",
        "shared_expert_fc1",
        "shared_swiglu_activation",
        "shared_expert_fc2",
    }
    registered_transformer, registered_attention, registered_expert = get_overlap_part_sets()
    transformer_names.update(registered_transformer)
    attention_names.update(registered_attention)
    expert_names.update(registered_expert)
    return transformer_names, attention_names, expert_names


def estimate_overlap_cover_windows(compute_result):
    """
    根据计算明细 compute_result，给每个通信域估一个可用于隐藏通信的计算时间窗口 cover_window
    先从计算时间里估计“通信最多能藏在哪些计算后面、能藏多久”。
    真正的 overlap ratio 是后面用 cover_window * eff 和 raw_comm_time 算出来的。
    """
    transformer_names, attention_names, expert_names = get_overlap_part_names()

    # 聚合计算时间
    # TP/PP 通信通常可以被这些计算的一部分遮盖
    transformer_time = sum_module_time_s(compute_result, transformer_names)
    # CP 通信主要和 attention 上下文切分相关，所以用 attention 时间作为遮盖窗口来源
    attention_time = sum_module_time_s(compute_result, attention_names)
    # EP/ETP 用 expert 计算作为遮盖窗口来源
    expert_time = sum_module_time_s(compute_result, expert_names)
    # DP 梯度同步主要和 backward 过程重叠，所以用 backward 时间
    backward_time = sum_module_time_s(compute_result, mode="backward")

    # window系数表示  这类计算里有多少比例理论上能作为遮盖窗口
    return {
        "tp": transformer_time * OVERLAP_DEFAULTS["tp"]["window"],
        "cp": attention_time * OVERLAP_DEFAULTS["cp"]["window"],
        "dp": backward_time * OVERLAP_DEFAULTS["dp"]["window"],
        "pp": transformer_time * OVERLAP_DEFAULTS["pp"]["window"],
        "ep": expert_time * OVERLAP_DEFAULTS["ep"]["window"],
        "etp": expert_time * OVERLAP_DEFAULTS["etp"]["window"],
        "edp": expert_time * OVERLAP_DEFAULTS["edp"]["window"],
    }


def apply_overlap_for_domain(domain, raw_time, cover_window, manual_ratio=None, mode="auto"):
    """
    给某一个通信域，比如 TP/DP/PP/CP，计算 overlap 之后还暴露在关键路径上的通信时间，以及对应 overlap 比例。

    Args:
        raw_time: float, 原始通信时间
        cover_window: int,
    """
    # 通信时间为0
    if raw_time <= 0:
        return 0.0, 0.0

    # 不隐藏通信
    if mode == "none":
        return raw_time, 0.0

    # 用户手动填写隐藏比例
    if mode == "manual":
        ratio = clamp(manual_ratio or 0.0)
        return raw_time * (1 - ratio), ratio

    defaults = OVERLAP_DEFAULTS[domain]
    # 通信侧的理论可掩盖预算和计算侧的覆盖窗口共同限制 overlap，
    # 两者取小后再乘实现效率，避免自动模式给出过于乐观的结果。
    max_hidden_ratio = min(defaults["max"], 1 - defaults["floor"])
    communication_overlap_budget = raw_time * max_hidden_ratio
    overlappable_time = min(
        communication_overlap_budget,
        max(0.0, cover_window),
    )
    hidden_time = overlappable_time * defaults["eff"]
    # 通信不可能完全隐藏，总有一部分暴露，比如启动开销、依赖边界、尾部 bucket。floor就是下边界，最短通信时间
    exposed_time = max(raw_time * defaults["floor"], raw_time - hidden_time)
    exposed_time = min(raw_time, exposed_time)
    # max为每类通信设置最大 overlap 上限，避免过于乐观
    ratio = clamp(1 - exposed_time / raw_time, 0.0, defaults["max"])
    exposed_time = raw_time * (1 - ratio)
    # 返回最终通信暴露时间（未隐藏通信时间）和隐藏比例
    return exposed_time, ratio


def apply_overlap(raw_comm_result, compute_result, config, domains, domain_overrides=None):
    mode = config.parallel.overlap_mode
    cover_windows = estimate_overlap_cover_windows(compute_result)
    domain_overrides = domain_overrides or {}

    exposed = {}
    ratios = {}
    logger.debug("overlap mode: %s", mode)
    for domain in domains:
        if domain in domain_overrides:
            override = domain_overrides[domain]
            exposed_time = override["exposed_time_s"]
            ratio = override["effective_overlap_ratio"]
            exposed[domain] = exposed_time
            ratios[domain] = ratio
            logger.debug(
                "%s overlap override: raw=%.6fs ratio=%.4f exposed=%.6fs",
                domain.upper(),
                raw_comm_result.get(domain, 0.0),
                ratio,
                exposed_time,
            )
            continue
        manual_ratio = getattr(config.parallel, f"{domain}_overlap_ratio", None)
        domain_mode = mode
        if domain == "pp" and not getattr(
            config.parallel,
            "overlap_p2p_comm",
            False,
        ):
            domain_mode = "none"
        elif domain == "pp" and mode == "manual" and manual_ratio is None:
            # Enabling Megatron P2P overlap must have an effect even when the
            # user did not provide a calibrated manual ratio.  Fall back to
            # the conservative automatic exposure model for this domain only.
            domain_mode = "auto"
        if domain == "ep":
            ep_auto_overlap_enabled = getattr(
                config.parallel,
                "ep_overlap_enabled",
                False,
            )
            if config.parallel.pp_size == 1 or (
                mode == "auto" and not ep_auto_overlap_enabled
            ):
                domain_mode = "none"
        exposed_time, ratio = apply_overlap_for_domain(
            domain,
            raw_comm_result.get(domain, 0.0),
            cover_windows.get(domain, 0.0),
            manual_ratio=manual_ratio,
            mode=domain_mode,
        )
        exposed[domain] = exposed_time
        ratios[domain] = ratio
        logger.debug(
            "%s overlap: raw=%.6fs cover=%.6fs ratio=%.4f exposed=%.6fs",
            domain.upper(),
            raw_comm_result.get(domain, 0.0),
            cover_windows.get(domain, 0.0),
            ratio,
            exposed_time,
        )

    exposed["total"] = sum(exposed[domain] for domain in domains)
    exposed["overlap_ratios"] = ratios
    exposed["raw"] = {domain: raw_comm_result.get(domain, 0.0) for domain in domains}
    exposed["raw"]["total"] = sum(exposed["raw"].values())
    return exposed


def build_default_compute_result(config):
    return ComputeModel().spec_summary()


def build_communication_simulator(config):
    use_bandwidth_table = config.hardware.use_bandwidth_table
    if use_bandwidth_table:
        logger.info(
            "Communication bandwidth mode: built-in measured bandwidth table; "
            "configured bandwidth and efficiency values are not applied."
        )
    else:
        logger.info(
            "Communication bandwidth mode: configured one-direction GB/s; "
            "communication efficiencies are applied."
        )
        logger.info(
            "Configured communication bandwidths: intra=%.3f GB/s, "
            "inter=%.3f GB/s; efficiencies: p2p_intra=%.3f, "
            "collective_intra=%.3f, inter=%.3f.",
            config.hardware.intra_bw_gbps,
            config.hardware.inter_bw_gbps,
            config.hardware.p2p_intra_efficiency,
            config.hardware.collective_intra_efficiency,
            config.hardware.collective_inter_efficiency,
        )

    simulator = CommunicationSimulator()
    simulator.initialize_parallelism(
        num_gpus=config.parallel.num_gpus,
        tp_size=config.parallel.tp_size,
        cp_size=config.parallel.cp_size,
        pp_size=config.parallel.pp_size,
        ep_size=config.parallel.ep_size,
        etp_size=config.parallel.etp_size,
        gpus_per_node=config.hardware.gpus_per_node,
        gpu_model=GPUType.BW1000,
        intra_node_bandwidth_gbps=config.hardware.intra_bw_gbps,
        inter_node_bandwidth_gbps=config.hardware.inter_bw_gbps,
        p2p_intra_efficiency=config.hardware.p2p_intra_efficiency,
        collective_intra_efficiency=config.hardware.collective_intra_efficiency,
        collective_inter_efficiency=config.hardware.collective_inter_efficiency,
        bandwidth_table_dir=(
            files("hcu_train_simulator.communication.bandwidth")
            if use_bandwidth_table
            else None
        ),
    )
    return simulator


def estimate_ag_rs_communication_time(simulator, group_type, data_size, count):
    if count <= 0:
        return 0.0
    all_gather_time = simulator.get_communication_time(
        com_type=CommType.ALL_GATHER,
        algorithm=Algorithm.RING,
        group_type=group_type,
        data_size=data_size,
        rank=0,
    )
    reduce_scatter_time = simulator.get_communication_time(
        com_type=CommType.REDUCE_SCATTER,
        algorithm=Algorithm.RING,
        group_type=group_type,
        data_size=data_size,
        rank=0,
    )
    return count * (all_gather_time + reduce_scatter_time)


def estimate_param_sync_communication_time(simulator, config, group_type, group_size, param_elements, label):
    if group_size <= 1 or param_elements <= 0:
        return 0.0

    if config.parallel.use_distributed_optimizer:
        grad_rs_bytes = param_elements * BF16_BYTES
        param_ag_bytes = param_elements * BF16_BYTES
        reduce_scatter_time = simulator.get_communication_time(
            com_type=CommType.REDUCE_SCATTER,
            algorithm=Algorithm.RING,
            group_type=group_type,
            data_size=grad_rs_bytes,
            rank=0,
        )
        all_gather_time = simulator.get_communication_time(
            com_type=CommType.ALL_GATHER,
            algorithm=Algorithm.RING,
            group_type=group_type,
            data_size=param_ag_bytes,
            rank=0,
        )
        logger.debug(
            "%s distributed optimizer: param_elements=%s grad_rs_bytes=%s param_ag_bytes=%s",
            label,
            int(param_elements),
            int(grad_rs_bytes),
            int(param_ag_bytes),
        )
        logger.debug("%s reduce_scatter raw time: %.6fs", label, reduce_scatter_time)
        logger.debug("%s all_gather raw time: %.6fs", label, all_gather_time)
        return reduce_scatter_time + all_gather_time

    grad_all_reduce_bytes = param_elements * BF16_BYTES
    all_reduce_time = simulator.get_communication_time(
        com_type=CommType.ALL_REDUCE,
        algorithm=Algorithm.RING,
        group_type=group_type,
        data_size=grad_all_reduce_bytes,
        rank=0,
    )
    logger.debug(
        "%s all_reduce: param_elements=%s grad_all_reduce_bytes=%s raw_time=%.6fs",
        label,
        int(param_elements),
        int(grad_all_reduce_bytes),
        all_reduce_time,
    )
    return all_reduce_time


def estimate_dp_communication_time(simulator, config, dp_size):
    return estimate_param_sync_communication_time(
        simulator,
        config,
        GroupType.DP,
        dp_size,
        estimate_dp_param_elements(config),
        "DP",
    )


def estimate_edp_communication_time(simulator, config):
    edp_size = expert_data_parallel_size(config)
    return estimate_param_sync_communication_time(
        simulator,
        config,
        GroupType.EDP,
        edp_size,
        estimate_edp_param_elements(config),
        "EDP",
    )


def estimate_dense_tp_communication_time(simulator, config, dp_size):
    if config.parallel.tp_size <= 1:
        return 0.0

    activation_bytes = (
        communication_dtype_bytes(config)
        * config.parallel.micro_batch_size
        * config.parallel.seq_length
        * config.transformer.hidden_size
        / config.parallel.cp_size
    )
    all_gather_time = simulator.get_communication_time(
        com_type=CommType.ALL_GATHER,
        algorithm=Algorithm.RING,
        group_type=GroupType.TP,
        data_size=activation_bytes,
        rank=0,
    )
    reduce_scatter_time = simulator.get_communication_time(
        com_type=CommType.REDUCE_SCATTER,
        algorithm=Algorithm.RING,
        group_type=GroupType.TP,
        data_size=activation_bytes,
        rank=0,
    )

    count_candidates = estimate_dense_tp_collective_counts(config, dp_size)
    all_gather_count, reduce_scatter_count = max(
        count_candidates,
        key=lambda counts: counts[0] * all_gather_time + counts[1] * reduce_scatter_time,
    )
    total_all_gather_time = all_gather_count * all_gather_time
    total_reduce_scatter_time = reduce_scatter_count * reduce_scatter_time
    logger.debug(
        "TP dense collectives: activation_bytes=%s all_gather_count=%g reduce_scatter_count=%g",
        int(activation_bytes),
        all_gather_count,
        reduce_scatter_count,
    )
    logger.debug("TP all_gather raw time: %.6fs", total_all_gather_time)
    logger.debug("TP reduce_scatter raw time: %.6fs", total_reduce_scatter_time)
    return total_all_gather_time + total_reduce_scatter_time


def estimate_vision_tp_communication_time(simulator, config, dp_size):
    """Estimate stage-0 TP traffic for the visual encoder and mergers."""
    if config.parallel.tp_size <= 1 or not config.vision.enabled:
        return 0.0

    num_microbatches = get_num_microbatches(
        config.parallel.global_batch_size,
        config.parallel.micro_batch_size,
        dp_size,
    )
    activation_bytes = (
        communication_dtype_bytes(config)
        * config.parallel.micro_batch_size
        * config.parallel.vision_num_images
        * (
            config.parallel.vision_seq_length
            or config.vision.num_position_embeddings
        )
        * config.vision.hidden_size
    )
    # Conv3D patch projection, four projections per visual block, and two
    # projections in the main merger plus every DeepStack merger.
    sharded_projections = (
        1
        + 4 * config.vision.num_layers
        + 2 * (1 + len(config.vision.deepstack_visual_indexes))
    )
    return estimate_ag_rs_communication_time(
        simulator,
        GroupType.TP,
        activation_bytes,
        sharded_projections * num_microbatches,
    )


def _log_communication_result(result, domains):
    rows = []
    raw = result.get("raw", {})
    ratios = result.get("overlap_ratios", {})
    for domain in domains:
        rows.append(
            {
                "domain": domain.upper(),
                "raw_s": round(raw.get(domain, 0.0), 6),
                "overlap_ratio": round(ratios.get(domain, 0.0), 4),
                "exposed_s": round(result.get(domain, 0.0), 6),
            }
        )
    rows.append(
        {
            "domain": "TOTAL",
            "raw_s": round(raw.get("total", 0.0), 6),
            "overlap_ratio": "",
            "exposed_s": round(result.get("total", 0.0), 6),
        }
    )
    log_table(logger, "communication estimate", rows)


def _log_deepep_result(details):
    if not details:
        return
    rows = [
        {
            "phase": phase["phase"],
            "calls": phase["calls"],
            "raw_ms": round(phase["raw_time_s"] * 1000, 4),
            "overlap_ratio": round(phase["effective_overlap_ratio"], 4),
            "hidden_ms": round(phase["hidden_time_s"] * 1000, 4),
            "exposed_ms": round(phase["exposed_time_s"] * 1000, 4),
        }
        for phase in details.get("phases", [])
    ]
    log_table(logger, "DeepEP heuristic phase estimate", rows)


def estimate_communication(compute_result=None, log_result=False):
    config = get_config()
    if compute_result is None:
        compute_result = build_default_compute_result(config)

    micro_batch_size = config.parallel.micro_batch_size
    seq_length = config.parallel.seq_length
    hidden_size = config.transformer.hidden_size
    num_experts = config.transformer.num_moe_experts
    top_k = config.transformer.moe_router_topk
    kv_dim = kv_cache_dim(config)
    layers_per_pp_rank = get_layers_per_pp_rank(config)
    moe_layers_per_pp_rank = get_moe_layers_per_pp_rank(config)
    dense_layers_per_pp_rank = [
        layer_count - moe_count
        for layer_count, moe_count in zip(
            layers_per_pp_rank,
            moe_layers_per_pp_rank,
        )
    ]
    layers = max(layers_per_pp_rank)
    moe_layers = max(moe_layers_per_pp_rank)
    tp = config.parallel.tp_size
    pp = config.parallel.pp_size
    cp = config.parallel.cp_size
    ep = config.parallel.ep_size
    etp = config.parallel.etp_size
    vp = max(config.parallel.vp_size, 1)
    pp_schedule = config.parallel.pp_schedule
    dp = config.parallel.num_gpus / tp / cp / pp
    num_microbatches = get_num_microbatches(config.parallel.global_batch_size, micro_batch_size, dp)
    simulator = build_communication_simulator(config)
    raw_result = {}
    comm_bytes = communication_dtype_bytes(config)

    if num_experts is not None:
        # `number` is the number of collective invocations.  The communication
        # simulator already accounts for the ring's (group_size - 1) / group_size
        # traffic factor, so it must not be folded into the invocation count.
        data_size = comm_bytes * micro_batch_size * seq_length * hidden_size
        hidden_pair_time = estimate_ag_rs_communication_time(
            simulator,
            GroupType.TP,
            data_size,
            1,
        )

        data_size = comm_bytes * micro_batch_size * seq_length * num_experts
        router_pair_time = estimate_ag_rs_communication_time(
            simulator,
            GroupType.ETP,
            data_size,
            1,
        )
        has_shared_experts = bool(shared_expert_intermediate_total(config))
        tp_stage_components = [
            (
                2 * layer_count * num_microbatches * hidden_pair_time,
                moe_count * num_microbatches * router_pair_time,
                (
                    2 * moe_count * num_microbatches * hidden_pair_time
                    if has_shared_experts
                    else 0.0
                ),
                2 * dense_count * num_microbatches * hidden_pair_time,
            )
            for layer_count, moe_count, dense_count in zip(
                layers_per_pp_rank,
                moe_layers_per_pp_rank,
                dense_layers_per_pp_rank,
            )
        ]
        critical_tp_components = max(
            tp_stage_components,
            key=sum,
        )
        tp1_time, tp2_time, shared_tp_time, dense_mlp_tp_time = (
            critical_tp_components
        )
        vision_tp_time = estimate_vision_tp_communication_time(
            simulator,
            config,
            dp,
        )
        raw_result["tp"] = sum(critical_tp_components) + vision_tp_time
        logger.debug("TP1 raw time: %.6fs", tp1_time)
        logger.debug("TP2 raw time: %.6fs", tp2_time)
        logger.debug("TP shared expert raw time: %.6fs", shared_tp_time)
        logger.debug("TP dense MLP raw time: %.6fs", dense_mlp_tp_time)
        logger.debug("TP vision raw time: %.6fs", vision_tp_time)
        if shared_expert_intermediate_total(config):
            logger.info("TP_shared_expert_raw_time: %.6fs", shared_tp_time)

        data_size = pp_activation_payload_bytes(config)
        number = get_pp_p2p_communication_count(pp, num_microbatches, pp_schedule, vp)
        pp1_time = number * simulator.get_communication_time(
            com_type=CommType.P2P,
            algorithm=Algorithm.RING,
            group_type=GroupType.PP,
            data_size=data_size,
            rank=0,
        )
        raw_result["pp"] = pp1_time
        logger.debug("PP1 raw time: %.6fs", pp1_time)

        data_size = 2 * micro_batch_size * seq_length * kv_dim
        number = 2 * layers * num_microbatches
        cp1_time = number * simulator.get_communication_time(
            com_type=CommType.ALL_GATHER,
            algorithm=Algorithm.RING,
            group_type=GroupType.CP,
            data_size=data_size,
            rank=0,
        )

        data_size = 2 * micro_batch_size * seq_length * kv_dim
        cp2_time = number * simulator.get_communication_time(
            com_type=CommType.REDUCE_SCATTER,
            algorithm=Algorithm.RING,
            group_type=GroupType.CP,
            data_size=data_size,
            rank=0,
        )
        raw_result["cp"] = cp1_time + cp2_time
        logger.debug("CP1 raw time: %.6fs", cp1_time)
        logger.debug("CP2 raw time: %.6fs", cp2_time)

        dp_time = estimate_dp_communication_time(simulator, config, dp)
        raw_result["dp"] = dp_time
        logger.debug("DP raw time: %.6fs", dp_time)

        edp_time = estimate_edp_communication_time(simulator, config)
        raw_result["edp"] = edp_time
        logger.debug("EDP raw time: %.6fs", edp_time)

        data_size = moe_ep_activation_payload_bytes(config)
        single_ep_time = simulator.get_communication_time(
            com_type=CommType.ALL_TO_ALL,
            algorithm=Algorithm.RING,
            group_type=GroupType.EP,
            data_size=data_size,
            rank=0,
        )
        deepep_details = None
        domain_overrides = None
        if config.parallel.ep_communication_backend == "deepep":
            ep_group = simulator.engine.get_group_info(0, GroupType.EP)
            _, _, expert_names = get_overlap_part_names()
            deepep_details = estimate_deepep_phases(
                single_collective_time_s=single_ep_time,
                phase_count=moe_ep_phase_collective_count(config),
                payload_bytes=data_size,
                compute_result=compute_result,
                expert_names=expert_names,
                config=config,
                ep_nodes=(ep_group.n_nodes if ep_group is not None else None),
            )
            raw_result["ep"] = deepep_details["raw_time_s"]
            domain_overrides = {"ep": deepep_details}
            logger.debug(
                "DeepEP raw time: %.6fs exposed time: %.6fs",
                deepep_details["raw_time_s"],
                deepep_details["exposed_time_s"],
            )
        else:
            number = moe_ep_activation_collective_count(config)
            raw_result["ep"] = number * single_ep_time
            logger.debug("EP raw time: %.6fs", raw_result["ep"])

        data_size = comm_bytes * micro_batch_size * seq_length * hidden_size * top_k / ep
        number = 3 * moe_layers * num_microbatches
        etp_time = estimate_ag_rs_communication_time(
            simulator,
            GroupType.ETP,
            data_size,
            number,
        )
        raw_result["etp"] = etp_time
        logger.debug("ETP raw time: %.6fs", etp_time)

        domains = ["tp", "cp", "ep", "dp", "edp", "pp", "etp"]
        result = apply_overlap(
            raw_result,
            compute_result,
            config,
            domains,
            domain_overrides=domain_overrides,
        )
        if deepep_details is not None:
            result["ep_details"] = deepep_details
        if log_result:
            _log_communication_result(result, domains)
            _log_deepep_result(deepep_details)
        return result
    else:
        tp_time = (
            estimate_dense_tp_communication_time(simulator, config, dp)
            + estimate_vision_tp_communication_time(simulator, config, dp)
        )
        raw_result["tp"] = tp_time
        logger.debug("TP raw time: %.6fs", tp_time)

        data_size = pp_activation_payload_bytes(config)
        number = get_pp_p2p_communication_count(pp, num_microbatches, pp_schedule, vp)
        pp1_time = number * simulator.get_communication_time(
            com_type=CommType.P2P,
            algorithm=Algorithm.RING,
            group_type=GroupType.PP,
            data_size=data_size,
            rank=0,
        )
        raw_result["pp"] = pp1_time
        logger.debug("PP1 raw time: %.6fs", pp1_time)

        data_size = 2 * micro_batch_size * seq_length * kv_dim
        number = 2 * layers * num_microbatches
        cp1_time = number * simulator.get_communication_time(
            com_type=CommType.ALL_GATHER,
            algorithm=Algorithm.RING,
            group_type=GroupType.CP,
            data_size=data_size,
            rank=0,
        )

        data_size = 2 * micro_batch_size * seq_length * kv_dim
        cp2_time = number * simulator.get_communication_time(
            com_type=CommType.REDUCE_SCATTER,
            algorithm=Algorithm.RING,
            group_type=GroupType.CP,
            data_size=data_size,
            rank=0,
        )
        raw_result["cp"] = cp1_time + cp2_time
        logger.debug("CP1 raw time: %.6fs", cp1_time)
        logger.debug("CP2 raw time: %.6fs", cp2_time)

        dp_time = estimate_dp_communication_time(simulator, config, dp)
        raw_result["dp"] = dp_time
        logger.debug("DP raw time: %.6fs", dp_time)

        domains = ["tp", "cp", "dp", "pp"]
        result = apply_overlap(raw_result, compute_result, config, domains)
        if log_result:
            _log_communication_result(result, domains)
        return result
if __name__ == "__main__":
    initialize_simulation("templates/config.yaml")
    _ = estimate_communication(log_result=True)
