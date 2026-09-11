# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import copy
import contextlib
import heapq
import itertools
import io
from itertools import product

from hcu_train_simulator.config.loader import get_logger, load_config, validate_config
from hcu_train_simulator.config.models import SimulationConfig, TransformerConfig
from hcu_train_simulator.context import reset_config, set_config
from hcu_train_simulator.estimators import estimate_communication, estimate_compute, estimate_memory
from hcu_train_simulator.logging import log_step, log_table
from hcu_train_simulator.modeling import ModelStatistics
from hcu_train_simulator.simulator import (
    estimate_pp_schedule_bubble_time_ms,
    get_effective_module_time,
)


logger = get_logger(__name__)

MAX_SEARCH_RESULTS = 10
SEARCH_OBJECTIVE = "tgs"
_TOP_RESULT_SEQUENCE = itertools.count()


def _as_list(value):
    if isinstance(value, list):
        return value
    return [value]


def _iter_candidates(base_parallel_config, candidates):
    if not candidates:
        yield {}
        return

    unknown_keys = sorted(set(candidates) - set(base_parallel_config))
    if unknown_keys:
        raise ValueError(f"Unknown search candidate keys: {unknown_keys}")

    keys = list(candidates.keys())
    value_lists = [_as_list(candidates[key]) for key in keys]
    for values in product(*value_lists):
        yield dict(zip(keys, values))


def _is_dense_model(model_config):
    return not TransformerConfig.from_dict(model_config).num_moe_experts


def _skip_unneeded_dense_candidate(model_config, parallel_config):
    if not _is_dense_model(model_config):
        return False

    ep_size = parallel_config.get("ep_size", 1)
    etp_size = parallel_config.get("etp_size", 1)
    return ep_size != 1 or etp_size not in (None, 1)


def _evaluate_candidate(model_config, estimator_config, candidate):
    candidate_config = copy.deepcopy(estimator_config)
    candidate_config["parallel_config"].update(candidate)

    reset_config()
    try:
        config = SimulationConfig.from_dict(model_config, candidate_config)
        set_config(config)

        with contextlib.redirect_stdout(io.StringIO()):
            validate_config()
            memory_result = estimate_memory()
            compute_result = estimate_compute()
            communication_result = estimate_communication(compute_result)

        bubble_time_ms = estimate_pp_schedule_bubble_time_ms(compute_result, config)
        communication_result["pp"] = communication_result.get("pp", 0.0) + bubble_time_ms / 1000
        communication_result["total"] = communication_result.get("total", 0.0) + bubble_time_ms / 1000

        compute_time_ms = sum(get_effective_module_time(module) for module in compute_result)
        communication_time_ms = communication_result["total"] * 1000
        iteration_time_ms = compute_time_ms + communication_time_ms
        tgs = (
            config.parallel.global_batch_size
            * config.parallel.seq_length
            / config.parallel.num_gpus
            / (iteration_time_ms / 1000)
        )
        model_flops = ModelStatistics().compute_flops()
        mfu = model_flops / (iteration_time_ms / 1000 * 1e12 * config.parallel.num_gpus) / config.hardware.fp16_tflops

        return {
            "candidate": candidate,
            "parallel": copy.deepcopy(candidate_config["parallel_config"]),
            "memory": memory_result,
            "compute_time_ms": compute_time_ms,
            "communication_time_ms": communication_time_ms,
            "iteration_time_ms": iteration_time_ms,
            "tgs": tgs,
            "mfu": mfu,
        }
    finally:
        reset_config()


def _max_memory_gib(result):
    return max(stage["total_gib"] for stage in result["memory"])


def _result_row(rank, result):
    parallel = result["parallel"]
    return {
        "rank": rank,
        "tgs": round(result["tgs"], 2),
        "iter_ms": round(result["iteration_time_ms"], 2),
        "compute_ms": round(result["compute_time_ms"], 2),
        "comm_ms": round(result["communication_time_ms"], 2),
        "max_mem_gib": round(_max_memory_gib(result), 2),
        "mfu": round(result["mfu"], 4),
        "world_size": parallel["num_gpus"],
        "tp": parallel["tp_size"],
        "cp": parallel["cp_size"],
        "pp": parallel["pp_size"],
        "dp": parallel["num_gpus"] // (parallel["tp_size"] * parallel["cp_size"] * parallel["pp_size"]),
        "ep": parallel["ep_size"],
        "etp": parallel["etp_size"],
        "mbs": parallel["micro_batch_size"],
    }


def _push_top_result(top_results, top_k, result):
    item = (result[SEARCH_OBJECTIVE], next(_TOP_RESULT_SEQUENCE), result)
    if len(top_results) < top_k:
        heapq.heappush(top_results, item)
        return
    heapq.heappushpop(top_results, item)


def search_best_perf(config_path):
    model_config, estimator_config = load_config(config_path)
    base_config = SimulationConfig.from_dict(model_config, estimator_config)
    search_config = base_config.search

    if not search_config.enabled:
        raise ValueError("search.enabled must be true to run grid search.")

    requested_top_k = int(search_config.top_k)
    top_k = min(max(requested_top_k, 1), MAX_SEARCH_RESULTS)
    memory_margin_gib = float(search_config.memory_margin_gib)
    candidates = search_config.candidates

    memory_limit_gib = estimator_config["hardware_config"]["hbm_gib"] - memory_margin_gib
    top_results = []
    total_candidates = 0
    valid_candidates = 0
    skipped_candidates = 0

    for candidate in _iter_candidates(estimator_config["parallel_config"], candidates):
        total_candidates += 1
        parallel_config = copy.deepcopy(estimator_config["parallel_config"])
        parallel_config.update(candidate)
        if _skip_unneeded_dense_candidate(model_config, parallel_config):
            skipped_candidates += 1
            continue

        try:
            result = _evaluate_candidate(model_config, estimator_config, candidate)
        except Exception as exc:
            skipped_candidates += 1
            logger.debug("skip invalid search candidate %s: %s", candidate, exc)
            continue

        if _max_memory_gib(result) > memory_limit_gib:
            skipped_candidates += 1
            continue

        valid_candidates += 1
        _push_top_result(top_results, top_k, result)

    results = [item[2] for item in sorted(top_results, key=lambda item: item[0], reverse=True)]

    log_step(
        logger,
        "search finished: objective=%s, total=%s, valid=%s, skipped=%s, memory_limit=%.2fGiB",
        SEARCH_OBJECTIVE,
        total_candidates,
        valid_candidates,
        skipped_candidates,
        memory_limit_gib,
    )

    if not results:
        logger.info("No valid search result found.")
        return []

    rows = [_result_row(rank, result) for rank, result in enumerate(results, start=1)]
    log_table(logger, f"search results (top {len(rows)}, ranked by TGS)", rows)
    return results
