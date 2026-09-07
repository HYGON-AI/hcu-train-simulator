from dataclasses import asdict

from hcu_train_simulator.config.loader import get_logger, initialize_simulation, validate_config
from hcu_train_simulator.estimators import estimate_communication, estimate_compute, estimate_memory
from hcu_train_simulator.estimators.communication import expert_data_parallel_size
from hcu_train_simulator.logging import log_step, log_table
from hcu_train_simulator.modeling import ModelStatistics
from hcu_train_simulator.modeling.spec import spec_uses_moe
from hcu_train_simulator.reporting import generate_training_report, write_training_report
from hcu_train_simulator.context import get_config
from hcu_train_simulator.runtime_summary import log_measured_compute_environment, log_model_spec, log_parallel_strategy
from hcu_train_simulator.benchmarks import (
    OperatorProfileCatalog,
    apply_profile_to_compute_result,
    build_benchmark_config,
    check_accelerator_occupancy,
    get_env_info,
    run_benchmark,
    train_env_valid,
    update_profile_from_benchmarks,
)
from hcu_train_simulator.benchmarks.registry import benchmark_module_mapping


logger = get_logger(__name__)


def data_format(data, n=2):
    if isinstance(data, float):
        return round(data, n)
    return data


def is_valid_benchmark_time(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def get_simulated_module_time(module):
    module["all_time_ms"] = module.get("all_time_ms", module["forward_ms"] + module["backward_ms"])
    return module["all_time_ms"]


def get_effective_module_time(module):
    benchmark_time = module.get("benchmark_time_ms")
    if is_valid_benchmark_time(benchmark_time):
        return benchmark_time
    return get_simulated_module_time(module)


def get_benchmark_compute_multiplier(module_name, compute_count, config):
    if module_name not in {"moe_linear_fc1", "moe_linear_fc2", "moe_swiglu_activation"}:
        return compute_count

    num_experts = config.transformer.num_moe_experts
    ep_size = getattr(config.parallel, "ep_size", None)
    if not num_experts or not ep_size:
        return compute_count

    local_experts = max(1, num_experts // ep_size)
    benchmark_local_experts = min(
        int(getattr(config.parallel, "benchmark_max_local_experts", None) or local_experts),
        local_experts,
    )
    return compute_count / max(1, benchmark_local_experts)


def get_benchmark_module_map(config):
    moe_fc1 = (
        "te_moe_grouped_linear_fc1"
        if config.parallel.moe_grouped_gemm
        else "te_moe_linear_fc1"
    )
    moe_fc2 = (
        "te_moe_grouped_linear_fc2"
        if config.parallel.moe_grouped_gemm
        else "te_moe_linear_fc2"
    )
    residual_add = (
        "torch_bias_dropout_add"
        if getattr(config.parallel, "benchmark_dropout_p", 0.0)
        else "torch_residual_add"
    )
    mapping = benchmark_module_mapping(config.model_adapter)
    mapping.update(
        {
            "moe_linear_fc1": moe_fc1,
            "moe_linear_fc2": moe_fc2,
            "residual_dropout_add": residual_add,
        }
    )
    return mapping


def _has_measured_or_profile_time(module):
    source = module.get("benchmark_source")
    return is_valid_benchmark_time(module.get("benchmark_time_ms")) and source not in (None, "/")


def apply_measured_coverage(compute_result):
    available = {
        module.get("model_part")
        for module in compute_result
        if _has_measured_or_profile_time(module)
    }
    if not available:
        return compute_result

    for module in compute_result:
        covered_by = module.get("covered_by_measured")
        if not covered_by:
            continue
        covered_by_parts = covered_by if isinstance(covered_by, list) else [covered_by]
        matched = [part for part in covered_by_parts if part in available]
        if not matched:
            continue
        get_simulated_module_time(module)
        module["benchmark_time_ms"] = 0
        module["benchmark_source"] = "covered by " + "+".join(matched)
    return compute_result


def get_pp_schedule_bubble_ratio(config):
    pp_size = config.parallel.pp_size
    if pp_size <= 1:
        return 0.0

    dp_size = config.parallel.dp_size
    num_microbatches = config.parallel.global_batch_size / config.parallel.micro_batch_size / dp_size
    if num_microbatches <= 0:
        return 0.0

    schedule = (config.parallel.pp_schedule or "1f1b").lower()
    if schedule in {"interleaved", "interleaved_1f1b"}:
        vp_size = max(config.parallel.vp_size, 1)
        return (pp_size - 1) / (num_microbatches * vp_size)
    return (pp_size - 1) / num_microbatches


def estimate_pp_schedule_bubble_time_ms(compute_result, config):
    useful_compute_time = sum(get_effective_module_time(module) for module in compute_result)
    return useful_compute_time * get_pp_schedule_bubble_ratio(config)


def apply_pp_schedule_bubble_time(compute_result, comm_result, config):
    bubble_time_ms = estimate_pp_schedule_bubble_time_ms(compute_result, config)
    bubble_time_s = bubble_time_ms / 1000
    comm_result["pp"] = comm_result.get("pp", 0) + bubble_time_s
    comm_result["total"] = comm_result.get("total", 0) + bubble_time_s
    logger.info(
        "PP schedule bubble added to PP time: schedule=%s, vp_size=%s, bubble_time=%.2fms",
        config.parallel.pp_schedule,
        max(config.parallel.vp_size, 1),
        bubble_time_ms,
    )
    return bubble_time_ms


def build_memory_summary(memory_result, hbm_gib):
    return [
        {
            "pp_rank": stage["pp_rank"],
            "wgo_gib": stage["wgo_gib"],
            "activation_gib": stage["act_gib"],
            "total_gib": stage["total_gib"],
            "hbm_gib": hbm_gib,
            "utilization": f"{round(stage['total_gib'] / hbm_gib * 100, 2)}%",
        }
        for stage in memory_result
    ]


def report_data_processing(memory_result, compute_result, comm_result):
    """整理终端输出与 HTML 报告所需的汇总数据。"""
    config = get_config()

    model_statistics = ModelStatistics()
    model_size = model_statistics.compute_model_size()
    logger.info("Model Size: %.4fB parameters", model_size / 1e9)
    model_flops = model_statistics.compute_flops()

    # 汇总计算和通信时间，报告统一使用毫秒。
    all_compute_time = sum(get_effective_module_time(module) for module in compute_result)

    all_compute_time = data_format(all_compute_time)
    all_comm_time = data_format(comm_result["total"] * 1000)
    iter_time = data_format(all_compute_time + all_comm_time)

    # Dense 与 MoE 模型需要展示的通信域不同。
    if spec_uses_moe(config.model_spec):
        comm_domains = ["tp", "cp", "ep", "dp", "edp", "pp", "etp"]
    else:
        comm_domains = ["tp", "cp", "dp", "pp"]

    comm_ratio, format_comm_result = {}, {}
    comm_domain_summary = []
    raw_comm_result = comm_result.get("raw", {})
    overlap_ratios = comm_result.get("overlap_ratios", {})
    for domain in comm_domains:
        comm_ratio[domain] = data_format(comm_result[domain] / comm_result["total"], 4)
        format_comm_result[domain] = data_format(comm_result[domain] * 1000)
        comm_domain_summary.append(
            {
                "domain": domain.upper(),
                "raw_time_ms": data_format(
                    raw_comm_result.get(domain, comm_result[domain]) * 1000
                ),
                "overlap_percent": data_format(
                    overlap_ratios.get(domain, 0.0) * 100
                ),
                "time_ms": format_comm_result[domain],
                "percent": data_format(comm_ratio[domain] * 100),
            }
        )
    logger.info("communication ratio: %s", comm_ratio)

    # 计算吞吐率与模型 FLOPs 利用率。
    tgs = data_format(
        (config.parallel.global_batch_size * config.parallel.seq_length)
        / config.parallel.num_gpus
        / (iter_time / 1000)
    )
    mfu = data_format(
        model_flops
        / (iter_time / 1000 * 1e12 * config.parallel.num_gpus)
        / config.hardware.fp16_tflops,
        4,
    )

    # 在终端展示核心模拟结果。
    log_table(
        logger,
        "memory summary",
        build_memory_summary(memory_result, config.hardware.hbm_gib),
    )
    log_table(
        logger,
        "simulation summary",
        [
            {
                "compute_ms": all_compute_time,
                "communication_ms": all_comm_time,
                "iteration_ms": iter_time,
                "tgs": tgs,
                "mfu": mfu,
            }
        ],
    )
    log_table(logger, "communication summary", comm_domain_summary)
    ep_details = comm_result.get("ep_details")
    if ep_details:
        logger.info(
            "DeepEP heuristic model: payload_bucket=%s topology=%s policy=%s",
            ep_details.get("payload_bucket"),
            ep_details.get("topology"),
            ep_details.get("overlap_policy"),
        )
        log_table(
            logger,
            "DeepEP phase summary",
            [
                {
                    "phase": phase["phase"],
                    "calls": phase["calls"],
                    "raw_ms": data_format(phase["raw_time_s"] * 1000, 4),
                    "overlap_percent": data_format(
                        phase["effective_overlap_ratio"] * 100,
                        2,
                    ),
                    "hidden_ms": data_format(phase["hidden_time_s"] * 1000, 4),
                    "exposed_ms": data_format(phase["exposed_time_s"] * 1000, 4),
                }
                for phase in ep_details.get("phases", [])
            ],
        )

    return model_size, all_compute_time, all_comm_time, tgs, mfu, comm_domain_summary


def benchmark_data_processing(
    results: list,
    sim_result: list,
    *,
    only_missing: bool = False,
    source: str = "live benchmark",
):
    """将算子实测结果映射到模拟计算模块。"""
    config = get_config()
    map_dict = get_benchmark_module_map(config)

    # 按算子名称建立实测结果索引。
    new_result = {}
    for res in results:
        new_result[res.name] = asdict(res)

    for module in sim_result:
        if only_missing and is_valid_benchmark_time(module.get("benchmark_time_ms")):
            continue
        module_name = module["model_part"]
        num_compute = module["compute_count"]
        get_simulated_module_time(module)
        benchmark_names = map_dict.get(module_name)
        if isinstance(benchmark_names, str):
            benchmark_names = [benchmark_names]
        real_data_detail = None
        for benchmark_name in benchmark_names or []:
            candidate = new_result.get(benchmark_name)
            if candidate and candidate.get("status") == "ok" and candidate.get("mean_ms") is not None:
                real_data_detail = candidate
                break
        if not real_data_detail or real_data_detail.get("status") != "ok" or real_data_detail.get("mean_ms") is None:
            module["benchmark_time_ms"] = "/"
            module["benchmark_source"] = "/"
            continue
        # 单次实测时间乘以该模块在一个训练迭代中的执行次数。
        real_mean_time = real_data_detail["mean_ms"]
        benchmark_multiplier = get_benchmark_compute_multiplier(module_name, num_compute, config)
        if module_name == "qk_norm":
            benchmark_multiplier *= float(getattr(config.parallel, "benchmark_qk_norm_scale", 1.0) or 1.0)
        real_all_time = data_format(real_mean_time * benchmark_multiplier)
        module["benchmark_time_ms"] = real_all_time
        module["benchmark_source"] = source

    return sim_result


def fill_missing_benchmark_result(sim_result: list):
    for module in sim_result:
        module["all_time_ms"] = module.get(
            "all_time_ms", module["forward_ms"] + module["backward_ms"]
        )
        if not is_valid_benchmark_time(module.get("benchmark_time_ms")):
            module["benchmark_time_ms"] = "/"
            module["benchmark_source"] = "theoretical"
    return sim_result


def _module_shape_label(module: dict) -> str:
    op_type = module.get("op_type")
    if op_type == "gemm":
        names = ("m", "n", "k")
    elif op_type == "flash_attention":
        names = ("b", "seq", "heads", "head_dim", "value_head_dim")
    else:
        names = ("elements",)
    fields = [f"{name}={module[name]}" for name in names if module.get(name) is not None]
    return ",".join(fields) or "-"


def _postcheck_candidate_reason(postcheck) -> str:
    if postcheck is None:
        return "post-benchmark accelerator occupancy was not checked"
    reason = postcheck.reason or "post-benchmark accelerator occupancy did not pass"
    if postcheck.foreign_pids:
        return f"postcheck failed: {reason}; pids={list(postcheck.foreign_pids)}"
    return f"postcheck failed: {reason}"


def _not_validated_operator_rows(
    modules: list[dict],
    benchmark_results: list,
    module_to_benchmark: dict,
    *,
    batch_validated: bool,
    batch_reason: str = "",
    benchmark_attempted: bool = True,
) -> list[dict[str, str]]:
    """Explain every requested module shape that was not validated this run."""

    by_name = {result.name: result for result in benchmark_results}
    rows = []
    seen = set()
    for module in modules:
        benchmark_names = module_to_benchmark.get(module.get("model_part"))
        if isinstance(benchmark_names, str):
            benchmark_names = [benchmark_names]
        benchmark_names = list(benchmark_names or [])
        successful = next(
            (
                by_name[name]
                for name in benchmark_names
                if name in by_name
                and by_name[name].status == "ok"
                and by_name[name].mean_ms is not None
            ),
            None,
        )
        if successful is not None and batch_validated:
            continue

        if not benchmark_attempted:
            reason = batch_reason or "operator benchmark was not run"
        elif successful is not None:
            reason = batch_reason or "benchmark batch did not pass profile validation"
        elif not benchmark_names:
            reason = "no benchmark mapping is registered for this model part"
        else:
            failures = []
            for name in benchmark_names:
                result = by_name.get(name)
                if result is None:
                    failures.append(f"{name}: benchmark result was not produced")
                elif result.status != "ok":
                    detail = result.message or "operator benchmark was skipped"
                    failures.append(f"{name}: {result.status}: {detail}")
                else:
                    failures.append(f"{name}: benchmark returned no mean_ms")
            reason = "; ".join(failures)

        row = {
            "operator": str(module.get("model_part") or "unknown"),
            "shape": _module_shape_label(module),
            "benchmark": ",".join(benchmark_names) or "-",
            "reason": reason,
        }
        identity = tuple(row.values())
        if identity not in seen:
            seen.add(identity)
            rows.append(row)
    return rows


def run_simulation(
    config_path="hcu_train_simulator/templates/config.yaml",
    generate_report=True,
    report_dir="training_report",
):
    log_step(logger, "simulation started: config=%s", config_path)
    initialize_simulation(config_path)

    config = get_config()
    # 初始化后统一校验模型、并行策略和 overlap 配置。
    validate_config()
    log_model_spec(logger, config)
    log_parallel_strategy(logger, config)

    memory_result = estimate_memory()
    compute_result = estimate_compute(log_result=True)
    benchmark_env_info = None
    benchmark_details = []

    benchmark_config = build_benchmark_config(config)
    profile_catalog = OperatorProfileCatalog(config)
    mode = config.profile.mode if config.profile.enabled else "theoretical"

    if mode == "theoretical":
        logger.info("operator profile disabled; using theoretical compute estimates")
        compute_result = fill_missing_benchmark_result(compute_result)
    elif mode == "builtin":
        if profile_catalog.select_builtin():
            compute_result = apply_profile_to_compute_result(compute_result, profile_catalog, config)
            compute_result = apply_measured_coverage(compute_result)
        compute_result = fill_missing_benchmark_result(compute_result)
    else:
        # Environment probing can be slow because it imports torch and initializes CUDA/HIP.
        log_step(logger, "checking operator benchmark environment...")
        benchmark_available = benchmark_config.get("allow_cpu") or train_env_valid()
        if not benchmark_available:
            logger.info(
                "training benchmark environment is unavailable; use profile_config.mode=builtin "
                "to select a packaged profile, otherwise theoretical estimates are used"
            )
            compute_result = fill_missing_benchmark_result(compute_result)
        else:
            try:
                env_info = get_env_info()
                log_measured_compute_environment(logger, env_info)
                benchmark_env_info = env_info
                profile_catalog.select_runtime_environment(env_info)
                compute_result = apply_profile_to_compute_result(compute_result, profile_catalog, config)
                compute_result = apply_measured_coverage(compute_result)

                missing_modules = [
                    module
                    for module in compute_result
                    if module.get("op_type") in {"gemm", "flash_attention", "non_gemm"}
                    and not is_valid_benchmark_time(module.get("benchmark_time_ms"))
                    and not str(module.get("benchmark_source", "")).startswith("covered by")
                ]
                for module in missing_modules:
                    module["_profile_missing"] = True

                if not missing_modules:
                    logger.info("all requested operator shapes were resolved from validated profiles")
                else:
                    precheck = None
                    if config.profile.require_idle:
                        precheck = check_accelerator_occupancy()
                        if not precheck.known or not precheck.idle:
                            logger.warning(
                                "operator benchmark skipped; profile not updated: %s, pids=%s",
                                precheck.reason,
                                list(precheck.foreign_pids),
                            )
                            not_validated_rows = _not_validated_operator_rows(
                                missing_modules,
                                [],
                                get_benchmark_module_map(config),
                                batch_validated=False,
                                batch_reason=f"benchmark skipped: {precheck.reason}",
                                benchmark_attempted=False,
                            )
                            logger.warning(
                                "%d operator shape(s) were not validated; details follow",
                                len(not_validated_rows),
                            )
                            log_table(
                                logger,
                                "operators not validated",
                                not_validated_rows,
                            )
                        else:
                            logger.info("accelerator occupancy precheck passed via %s", precheck.command)

                    can_benchmark = (
                        not config.profile.require_idle
                        or (precheck is not None and precheck.known and precheck.idle)
                    )
                    if can_benchmark:
                        bm_results = run_benchmark(benchmark_config)
                        benchmark_details = [asdict(result) for result in bm_results]
                        postcheck = (
                            check_accelerator_occupancy(check_vram=False)
                            if config.profile.require_idle
                            else None
                        )
                        validated = (
                            not config.profile.require_idle
                            or (postcheck is not None and postcheck.known and postcheck.idle)
                        )
                        candidate_reason = (
                            "" if validated else _postcheck_candidate_reason(postcheck)
                        )
                        quality = {
                            "precheck": precheck.as_quality() if precheck else {"required": False},
                            "postcheck": postcheck.as_quality() if postcheck else {"required": False},
                        }

                        not_validated_rows = _not_validated_operator_rows(
                            missing_modules,
                            bm_results,
                            get_benchmark_module_map(config),
                            batch_validated=validated,
                            batch_reason=candidate_reason,
                        )
                        if not_validated_rows:
                            logger.warning(
                                "%d operator shape(s) were not validated; details follow",
                                len(not_validated_rows),
                            )
                            log_table(
                                logger,
                                "operators not validated",
                                not_validated_rows,
                            )

                        if validated:
                            compute_result = benchmark_data_processing(
                                bm_results,
                                compute_result,
                                only_missing=True,
                                source="live benchmark",
                            )
                            compute_result = apply_measured_coverage(compute_result)
                            status = "validated"
                        else:
                            status = "candidate"
                            logger.warning(
                                "operator benchmark result is candidate-only and will not be used: %s",
                                quality,
                            )

                        if config.profile.update_on_benchmark:
                            try:
                                user_store = profile_catalog.prepare_user_store()
                                changed = update_profile_from_benchmarks(
                                    user_store,
                                    compute_result,
                                    bm_results,
                                    get_benchmark_module_map(config),
                                    config,
                                    status=status,
                                    quality=quality,
                                    candidate_reason=candidate_reason or None,
                                    only_marked=True,
                                )
                                if changed:
                                    log_step(
                                        logger,
                                        "operator profile %s data recorded: %s",
                                        status,
                                        user_store.path,
                                    )
                            except Exception as exc:
                                logger.warning("operator profile update failed: %s", exc)

                compute_result = fill_missing_benchmark_result(compute_result)
                for module in compute_result:
                    module.pop("_profile_missing", None)
            except Exception as exc:
                logger.warning("operator benchmark/profile resolution failed; using fallbacks: %s", exc)
                if "missing_modules" in locals() and missing_modules:
                    not_validated_rows = _not_validated_operator_rows(
                        missing_modules,
                        [],
                        get_benchmark_module_map(config),
                        batch_validated=False,
                        batch_reason=f"benchmark failed: {exc}",
                        benchmark_attempted=False,
                    )
                    logger.warning(
                        "%d operator shape(s) were not validated; details follow",
                        len(not_validated_rows),
                    )
                    log_table(
                        logger,
                        "operators not validated",
                        not_validated_rows,
                    )
                compute_result = fill_missing_benchmark_result(compute_result)

    profile_summary = profile_catalog.source_counts(compute_result)
    profile_summary["candidate ignored"] = profile_catalog.candidate_count()
    profile_summary["mode"] = mode
    profile_summary["builtin profile"] = profile_catalog.selected_builtin_id or "/"
    logger.info("operator profile source summary: %s", profile_summary)

    # 将流水线调度气泡计入 PP 域暴露时间。
    communication_result = estimate_communication(compute_result)
    apply_pp_schedule_bubble_time(compute_result, communication_result, config)

    model_size, all_compute_time, all_comm_time, tgs, mfu, comm_domain_summary = report_data_processing(
        memory_result,
        compute_result,
        communication_result,
    )

    parallel_strategy = {}
    if spec_uses_moe(config.model_spec):
        comm_domains = ["tp", "cp", "ep", "dp", "edp", "pp", "etp"]
    else:
        comm_domains = ["tp", "cp", "dp", "pp"]

    for domain in comm_domains:
        if domain == "edp":
            parallel_strategy[domain] = expert_data_parallel_size(config)
        else:
            parallel_strategy[domain] = getattr(config.parallel, f"{domain}_size")

    if not generate_report:
        log_step(logger, "training report generation disabled")
        return {
            "memory": memory_result,
            "compute": compute_result,
            "communication": communication_result,
            "report_path": None,
        }

    # 默认生成 HTML 报告，开发验证时可通过开关关闭。
    html_output = generate_training_report(
        model_b=round(model_size / 1e9, 2),
        total_gpus=config.parallel.num_gpus,
        gpus_per_node=config.hardware.gpus_per_node,
        tgs=tgs,
        mfu=mfu * 100,
        pp_stages=memory_result,
        compute_ms=all_compute_time,
        comm_ms=all_comm_time,
        compute_details=compute_result,
        comm_domains=comm_domain_summary,
        parallel_strategy=parallel_strategy,
        benchmark_env_info=benchmark_env_info,
        benchmark_details=benchmark_details,
        profile_summary=profile_summary,
        ep_details=communication_result.get("ep_details"),
    )
    report_path = write_training_report(html_output, report_dir)
    log_step(logger, "training report generated: %s", report_path)
    return {
        "memory": memory_result,
        "compute": compute_result,
        "communication": communication_result,
        "report_path": report_path,
    }
