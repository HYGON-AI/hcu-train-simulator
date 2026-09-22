# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Solve practical training configurations from a HuggingFace model config.

The solver deliberately keeps its policy in code.  The existing YAML remains
the single source for the model path, hardware, and training defaults;
users do not need to maintain a second, mostly duplicated solver config.
"""

from __future__ import annotations

import contextlib
import copy
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import yaml

from hcu_train_simulator.commands.search import _evaluate_candidate
from hcu_train_simulator.config.loader import validate_config
from hcu_train_simulator.config.models import SimulationConfig
from hcu_train_simulator.context import reset_config, set_config
from hcu_train_simulator.estimators import estimate_memory
from hcu_train_simulator.logging import get_logger, log_step, log_table
from hcu_train_simulator.modeling import ModuleSpecMemoryModel
from hcu_train_simulator.models.registry import resolve_model_adapter
from hcu_train_simulator.parallelism import get_vpp_layer_candidates


logger = get_logger(__name__)

DEFAULT_SEQ_LENGTH = 4096
DEFAULT_MICRO_BATCH_SIZE = 1
DEFAULT_MAX_BUBBLE_RATIO = 0.15
DEFAULT_MEMORY_MARGIN_GIB = 5.0
PERFORMANCE_WORLD_MULTIPLIER = 2.0
MAX_PERFORMANCE_EVALUATIONS = 160
INITIAL_MINIMUM_WORLD_MULTIPLIER = 2
MAX_MINIMUM_WORLD_MULTIPLIER = 8
MINIMUM_WORLD_EXPANSION_FACTOR = 2


@dataclass(frozen=True)
class SolveOptions:
    seq_length: int | None = None
    micro_batch_size: int | None = None
    global_batch_size: int | None = None
    max_bubble_ratio: float = DEFAULT_MAX_BUBBLE_RATIO
    memory_margin_gib: float | None = None
    model_path: str | None = None


def _powers_of_two(limit: int) -> list[int]:
    values = []
    value = 1
    while value <= max(1, limit):
        values.append(value)
        value *= 2
    return values


def _divisors(value: int, limit: int | None = None) -> list[int]:
    if value <= 0:
        return [1]
    upper = min(value, limit or value)
    return [candidate for candidate in range(1, upper + 1) if value % candidate == 0]


def _first_present(config, *keys, default=None):
    for key in keys:
        value = config.get(key)
        if value is not None:
            return value
    return default


def _prepare_model_config(raw_config: dict) -> tuple[dict, list[str]]:
    """Return a generic-compatible approximation for an unknown model family."""

    adapter = resolve_model_adapter(raw_config)
    if adapter.name != "generic_gpt":
        return raw_config, []

    nested = raw_config.get("text_config")
    language = dict(nested) if isinstance(nested, dict) else dict(raw_config)
    for key in ("architectures", "model_type", "tie_word_embeddings"):
        if key in raw_config:
            language[key] = raw_config[key]

    assumptions = [
        "unsupported model family: using the generic Transformer/MoE approximation",
        "unknown model-specific config fields are ignored",
    ]

    aliases = {
        "num_hidden_layers": ("n_layer", "num_layers"),
        "hidden_size": ("n_embd", "model_dim", "dim"),
        "num_attention_heads": ("n_head", "num_heads"),
        "num_key_value_heads": ("num_kv_heads", "n_kv_heads"),
        "num_experts": ("n_routed_experts", "num_local_experts"),
        "num_experts_per_tok": ("num_experts_per_token", "topk"),
        "moe_intermediate_size": (
            "moe_ffn_hidden_size",
            "expert_intermediate_size",
            "routed_expert_hidden_size",
        ),
        "shared_expert_intermediate_size": (
            "shared_intermediate_size",
            "moe_shared_expert_intermediate_size",
        ),
        "mtp_num_layers": ("num_mtp_modules", "num_mtp_layers", "num_nextn_predict_layers"),
    }
    for target, sources in aliases.items():
        if language.get(target) is None:
            value = _first_present(language, *sources)
            if value is not None:
                language[target] = value

    if language.get("dense_intermediate_size") is not None:
        # Some MoE configs use intermediate_size for each routed expert and a
        # separate dense_intermediate_size for the initial dense layers.
        language.setdefault("moe_intermediate_size", language.get("intermediate_size"))
        language["intermediate_size"] = language["dense_intermediate_size"]

    moe_frequency = language.get("moe_layer_freq")
    if isinstance(moe_frequency, (list, tuple)):
        dense_prefix = 0
        for value in moe_frequency:
            if value:
                break
            dense_prefix += 1
        language["first_k_dense_replace"] = dense_prefix
        language["moe_layer_freq"] = 1
        assumptions.append(
            "per-layer MoE frequency list approximated as a dense prefix followed by MoE layers"
        )

    hidden_size = int(language.get("hidden_size") or 0)
    if not language.get("num_attention_heads") and hidden_size:
        language["num_attention_heads"] = max(1, hidden_size // 128)
        assumptions.append("num_attention_heads missing: inferred with a 128-wide head")
    language.setdefault("num_key_value_heads", language.get("num_attention_heads"))
    if not language.get("intermediate_size") and hidden_size:
        language["intermediate_size"] = 4 * hidden_size
        assumptions.append("intermediate_size missing: assumed 4 * hidden_size")
    language.setdefault("vocab_size", 32_000)

    required = ("hidden_size", "num_hidden_layers", "num_attention_heads", "vocab_size")
    missing = [key for key in required if not language.get(key)]
    if missing:
        raise ValueError(
            "unsupported model config lacks fields needed for a scale estimate: "
            + ", ".join(missing)
        )

    if isinstance(raw_config.get("vision_config"), dict):
        assumptions.append("unsupported vision tower omitted from dynamic-memory modeling")
    return language, assumptions


def _pipeline_layouts(num_layers: int, pp_size: int) -> list[tuple[int | None, int | None]]:
    if pp_size == 1:
        return [(None, None)]
    if num_layers % pp_size == 0:
        return [(None, None)]

    average = num_layers / pp_size
    limit = min(num_layers, max(2, math.ceil(average * 1.75)))
    layouts = []
    # Zero means that endpoint is not special and is therefore part of the
    # equally-sized middle stages, matching build_pp_layer_counts().
    for first in range(0, limit + 1):
        for last in range(0, limit + 1):
            special = int(first > 0) + int(last > 0)
            middle_stages = pp_size - special
            remaining = num_layers - first - last
            if middle_stages <= 0 or remaining <= 0 or remaining % middle_stages:
                continue
            middle = remaining // middle_stages
            counts = ([first] if first else []) + [middle] * middle_stages + ([last] if last else [])
            if len(counts) != pp_size or min(counts) <= 0:
                continue
            spread = max(counts) - min(counts)
            # Prefer balanced layouts, then lighter endpoint stages to leave
            # room for embeddings, output projection, and MTP.
            score = (spread, abs(first - average) + abs(last - average), first + last)
            layouts.append((score, first or None, last or None))
    layouts.sort(key=lambda item: item[0])
    return [(first, last) for _, first, last in layouts[:10]]


def _global_batch_size(
    preferred: int,
    mbs: int,
    dp: int,
    pp: int,
    max_bubble: float,
    vp_size: int = 1,
) -> int:
    required_microbatches = (
        1 if pp <= 1 else math.ceil((pp - 1) / (max_bubble * max(vp_size, 1)))
    )
    quantum = mbs * dp
    required = required_microbatches * quantum
    return max(required, math.ceil(max(preferred, 1) / quantum) * quantum)


def _bubble_ratio(parallel: dict) -> float:
    pp = parallel["pp_size"]
    if pp <= 1:
        return 0.0
    dp = parallel["num_gpus"] // (
        parallel["tp_size"] * parallel["cp_size"] * parallel["pp_size"]
    )
    microbatches = parallel["global_batch_size"] / parallel["micro_batch_size"] / dp
    vp_size = max(int(parallel.get("vp_size", 1)), 1)
    return (pp - 1) / (microbatches * vp_size)


@contextlib.contextmanager
def _quiet_estimators():
    """Keep candidate-internal estimator logs out of the terminal result."""

    previous = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        yield
    finally:
        logging.disable(previous)


def _candidate_config(estimator_config: dict, values: dict) -> dict:
    candidate = copy.deepcopy(estimator_config)
    candidate["parallel_config"].update(values)
    return candidate


def _evaluate_memory(model_config: dict, estimator_config: dict, values: dict):
    candidate_config = _candidate_config(estimator_config, values)
    reset_config()
    try:
        with _quiet_estimators():
            config = SimulationConfig.from_dict(model_config, candidate_config)
            set_config(config)
            validate_config()
            memory = estimate_memory(include_module_breakdown=False)
        return memory
    finally:
        reset_config()


def _peak_memory(memory: list[dict]) -> dict:
    return max(memory, key=lambda stage: stage["total_gib"])


def _model_scale(model_config: dict, estimator_config: dict) -> tuple[float, str]:
    base = copy.deepcopy(estimator_config)
    parallel = base["parallel_config"]
    parallel.update(
        tp_size=1,
        cp_size=1,
        pp_size=1,
        ep_size=1,
        etp_size=1,
        num_gpus=1,
        micro_batch_size=1,
        global_batch_size=1,
        decoder_first_pipeline_num_layers=None,
        decoder_last_pipeline_num_layers=None,
        num_layers_per_vp_stage=None,
        pp_schedule="1f1b",
        overlap_p2p_comm=False,
    )
    reset_config()
    try:
        with _quiet_estimators():
            config = SimulationConfig.from_dict(model_config, base)
            set_config(config)
            parameters = ModuleSpecMemoryModel(config).global_parameter_estimate().total_elements
        return parameters, config.model_adapter
    finally:
        reset_config()


def _iter_parallel_candidates(model_config: dict, estimator_config: dict, world_limit: int):
    probe = SimulationConfig.from_dict(model_config, estimator_config)
    transformer = probe.transformer
    parallel = estimator_config["parallel_config"]
    hardware = estimator_config["hardware_config"]
    seq_length = int(parallel["seq_length"])
    mbs = int(parallel["micro_batch_size"])
    preferred_gbs = int(parallel["global_batch_size"])
    max_bubble = float(estimator_config["_solve_max_bubble_ratio"])
    gpus_per_node = _configured_gpus_per_node(hardware)

    tp_values = _powers_of_two(min(32, transformer.hidden_size, transformer.num_attention_heads))
    cp_values = [value for value in _powers_of_two(min(16, seq_length)) if seq_length % value == 0]
    pp_values = range(1, min(transformer.num_layers, 64) + 1)
    expert_count = int(transformer.num_moe_experts or 0)

    seen = set()
    for tp in tp_values:
        for cp in cp_values:
            for pp in pp_values:
                base_world = tp * cp * pp
                if base_world > world_limit:
                    continue
                # Training deployments overwhelmingly use power-of-two data
                # parallel groups.  Keeping that invariant prevents exotic
                # factorizations from turning a terminal command into an
                # unbounded grid search.
                for dp in _powers_of_two(world_limit // base_world):
                    world = base_world * dp
                    if world % gpus_per_node:
                        continue
                    group = tp * cp * dp
                    ep = 1
                    if expert_count:
                        eligible = [value for value in _divisors(expert_count, min(128, group)) if group % value == 0]
                        ep = max(eligible or [1])
                    layouts = [
                        (first, last, None, 1)
                        for first, last in _pipeline_layouts(transformer.num_layers, pp)
                    ]
                    for vp_layers in get_vpp_layer_candidates(transformer.num_layers, pp):
                        vp_size = transformer.num_layers // pp // vp_layers
                        layouts.append((None, None, vp_layers, vp_size))

                    for first, last, vp_layers, vp_size in layouts:
                        uses_vpp = vp_layers is not None
                        gbs = _global_batch_size(
                            preferred_gbs,
                            mbs,
                            dp,
                            pp,
                            max_bubble,
                            vp_size,
                        )
                        values = {
                            "tp_size": tp,
                            "cp_size": cp,
                            "pp_size": pp,
                            "ep_size": ep,
                            "etp_size": 1,
                            "num_gpus": world,
                            "micro_batch_size": mbs,
                            "global_batch_size": gbs,
                            "decoder_first_pipeline_num_layers": first,
                            "decoder_last_pipeline_num_layers": last,
                            "num_layers_per_vp_stage": vp_layers,
                            "vp_size": vp_size,
                            "pp_schedule": "interleaved_1f1b" if uses_vpp else "1f1b",
                            "overlap_p2p_comm": uses_vpp,
                            "ep_overlap_enabled": bool(expert_count and uses_vpp),
                        }
                        key = tuple(sorted(values.items()))
                        if key not in seen:
                            seen.add(key)
                            yield values


def _find_minimum_solution(model_config, estimator_config, memory_limit, world_floor, world_limit):
    candidates = sorted(
        _iter_parallel_candidates(model_config, estimator_config, world_limit),
        key=lambda item: (
            item["num_gpus"],
            item["num_gpus"] // (item["tp_size"] * item["cp_size"] * item["pp_size"]),
            item["pp_size"],
            item["cp_size"],
            item["tp_size"],
        ),
    )
    best = None
    feasible = []
    evaluated = 0
    for values in candidates:
        world = values["num_gpus"]
        if world < world_floor:
            continue
        if best is not None and world > best["parallel"]["num_gpus"]:
            break
        try:
            memory = _evaluate_memory(model_config, estimator_config, values)
        except Exception:
            continue
        evaluated += 1
        peak = _peak_memory(memory)
        if peak["total_gib"] > memory_limit:
            continue
        result = {"parallel": values, "memory": memory}
        feasible.append(result)
        if best is None or peak["total_gib"] < _peak_memory(best["memory"])["total_gib"]:
            best = result
    return best, evaluated, candidates, feasible


def _find_minimum_solution_with_expansion(
    model_config,
    estimator_config,
    memory_limit,
    world_floor,
    initial_world_limit,
    max_world_limit,
    gpus_per_node,
):
    """Expand the minimum-card search only when the current range has no solution."""

    search_floor = world_floor
    world_limit = initial_world_limit
    total_evaluated = 0
    while True:
        minimum, evaluated, candidates, feasible = _find_minimum_solution(
            model_config,
            estimator_config,
            memory_limit,
            search_floor,
            world_limit,
        )
        total_evaluated += evaluated
        if minimum is not None or world_limit >= max_world_limit:
            return minimum, total_evaluated, candidates, feasible, world_limit

        search_floor = world_limit + gpus_per_node
        expanded_limit = math.ceil(
            world_limit * MINIMUM_WORLD_EXPANSION_FACTOR / gpus_per_node
        ) * gpus_per_node
        world_limit = min(max_world_limit, expanded_limit)
        logger.info(
            "no feasible configuration in the current range; expanding minimum-GPU search to %s GPUs",
            world_limit,
        )


def _find_performance_solution(
    model_config,
    estimator_config,
    memory_limit,
    minimum,
    candidates,
    minimum_world_feasible,
):
    min_world = minimum["parallel"]["num_gpus"]
    max_world = math.ceil(min_world * PERFORMANCE_WORLD_MULTIPLIER)
    transformer = SimulationConfig.from_dict(model_config, estimator_config).transformer
    expert_count = int(transformer.num_moe_experts or 0)
    eligible = []
    seen = set()
    for base in candidates:
        if not min_world <= base["num_gpus"] <= max_world:
            continue
        if _bubble_ratio(base) > estimator_config["_solve_max_bubble_ratio"] + 1e-12:
            continue
        dp = base["num_gpus"] // (base["tp_size"] * base["cp_size"] * base["pp_size"])
        group = base["tp_size"] * base["cp_size"] * dp
        etp_values = [1]
        if expert_count and transformer.moe_ffn_hidden_size:
            etp_values = [
                value
                for value in _powers_of_two(min(8, group))
                if transformer.moe_ffn_hidden_size % value == 0
            ]
        for etp in etp_values:
            item = dict(base)
            item["etp_size"] = etp
            if expert_count:
                ep_values = [
                    value
                    for value in _divisors(expert_count, min(128, group // etp))
                    if group % (value * etp) == 0
                ]
                item["ep_size"] = max(ep_values or [1])
            key = tuple(sorted(item.items()))
            if key not in seen:
                seen.add(key)
                eligible.append(item)
    # Favor a modest PP/CP/TP communication footprint while sampling the full
    # world-size range.  Final ranking still uses simulated TGS.
    eligible.sort(
        key=lambda item: (
            item["pp_size"] * item["cp_size"] * item["tp_size"],
            -item["num_gpus"],
        )
    )
    sampled = []
    per_bucket = max(1, MAX_PERFORMANCE_EVALUATIONS // 8)
    buckets = {}
    for item in eligible:
        bucket = round(item["num_gpus"] / max(min_world, 1) * 8)
        if buckets.get(bucket, 0) >= per_bucket:
            continue
        buckets[bucket] = buckets.get(bucket, 0) + 1
        sampled.append(item)
        if len(sampled) >= MAX_PERFORMANCE_EVALUATIONS:
            break

    best = None
    evaluated = 0

    # First compare all known feasible layouts at the minimum card count.  This
    # guarantees that "higher performance" is not merely the lowest-memory
    # layout relabeled when larger-world heuristics miss a feasible candidate.
    seeded = sorted(
        minimum_world_feasible,
        key=lambda result: (
            result["parallel"]["pp_size"] * result["parallel"]["cp_size"],
            result["parallel"]["tp_size"],
        ),
    )[: MAX_PERFORMANCE_EVALUATIONS // 2]
    for known in seeded:
        try:
            with _quiet_estimators():
                result = _evaluate_candidate(
                    model_config,
                    estimator_config,
                    known["parallel"],
                )
            result["memory"] = known["memory"]
        except Exception:
            continue
        evaluated += 1
        if best is None or result["tgs"] > best["tgs"]:
            best = result

    for values in sampled:
        if evaluated >= MAX_PERFORMANCE_EVALUATIONS:
            break
        try:
            memory = _evaluate_memory(model_config, estimator_config, values)
            if _peak_memory(memory)["total_gib"] > memory_limit:
                continue
            with _quiet_estimators():
                result = _evaluate_candidate(model_config, estimator_config, values)
        except Exception:
            continue
        evaluated += 1
        if best is None or result["tgs"] > best["tgs"]:
            best = result

    if best is None:
        # The minimum result is always useful even if a family-specific FLOP
        # handler cannot evaluate the degraded model's performance.
        return minimum, evaluated
    return best, evaluated


def _attach_performance(model_config, estimator_config, result):
    """Evaluate summary metrics for an already memory-feasible solution."""

    try:
        with _quiet_estimators():
            measured = _evaluate_candidate(
                model_config,
                estimator_config,
                result["parallel"],
            )
        # Preserve the exact memory result used to accept the solution.
        measured["memory"] = result["memory"]
        return measured
    except Exception as exc:
        logger.warning("solve performance estimate unavailable for a selected solution: %s", exc)
        return result


def _configured_gpus_per_node(hardware: dict) -> int:
    """Return the physical node/Scale-Up-1 width for either topology mode."""
    if hardware.get("use_supernode", False):
        value = hardware.get("scale_up_1_num_gpus")
    else:
        value = hardware.get("intra_node_num_gpus", hardware.get("gpus_per_node"))
    if value is None or int(value) <= 0:
        raise ValueError(
            "hardware topology must configure intra_node_num_gpus or "
            "scale_up_1_num_gpus for solve"
        )
    return int(value)


def _result_rows(label: str, result: dict, hardware: dict, memory_limit: float) -> tuple[dict, dict, dict]:
    parallel = result["parallel"]
    peak = _peak_memory(result["memory"])
    world = parallel["num_gpus"]
    dp = world // (parallel["tp_size"] * parallel["cp_size"] * parallel["pp_size"])
    strategy = {
        "solution": label,
        "gpus": world,
        "nodes": math.ceil(world / _configured_gpus_per_node(hardware)),
        "tp": parallel["tp_size"],
        "cp": parallel["cp_size"],
        "pp": parallel["pp_size"],
        "dp": dp,
        "ep": parallel["ep_size"],
        "etp": parallel["etp_size"],
        "vp": parallel.get("vp_size", 1) if parallel.get("num_layers_per_vp_stage") else "-",
        "vp_layers": parallel.get("num_layers_per_vp_stage") or "-",
        "schedule": parallel.get("pp_schedule", "1f1b"),
        "ep_overlap": parallel.get("ep_overlap_enabled", False),
        "first_pp_layers": parallel.get("decoder_first_pipeline_num_layers") or "-",
        "last_pp_layers": parallel.get("decoder_last_pipeline_num_layers") or "-",
        "mbs": parallel["micro_batch_size"],
        "gbs": parallel["global_batch_size"],
        "bubble": f"{_bubble_ratio(parallel) * 100:.2f}%",
    }
    memory = {
        "solution": label,
        "peak_pp_rank": peak["pp_rank"],
        "wgo_gib": peak["wgo_gib"],
        "activation_gib": peak["act_gib"],
        "peak_gib": peak["total_gib"],
        "usable_gib": round(memory_limit, 2),
        "headroom_gib": round(memory_limit - peak["total_gib"], 2),
        "physical_free_gib": round(float(hardware["hbm_gib"]) - peak["total_gib"], 2),
    }
    performance = {
        "solution": label,
        "iteration_ms": round(result.get("iteration_time_ms", float("nan")), 2),
        "mfu": (f"{result['mfu'] * 100:.2f}%" if "mfu" in result else "unavailable"),
        "tgs": (round(result["tgs"], 2) if "tgs" in result else "unavailable"),
    }
    return strategy, memory, performance


def solve_training_config(config_path: str, options: SolveOptions | None = None):
    options = options or SolveOptions()
    config_path_obj = Path(config_path).resolve()
    estimator_config = yaml.safe_load(config_path_obj.read_text())
    estimator_config["_config_dir"] = str(config_path_obj.parent)

    model_path_obj = Path(options.model_path or estimator_config["model_path"])
    if not model_path_obj.is_absolute():
        model_path_obj = config_path_obj.parent / model_path_obj
    if model_path_obj.is_dir():
        model_path_obj /= "config.json"
    model_config = json.loads(model_path_obj.read_text())

    model_config, assumptions = _prepare_model_config(model_config)
    parallel = estimator_config["parallel_config"]
    parallel["seq_length"] = int(options.seq_length or parallel.get("seq_length") or DEFAULT_SEQ_LENGTH)
    parallel["micro_batch_size"] = int(
        options.micro_batch_size or parallel.get("micro_batch_size") or DEFAULT_MICRO_BATCH_SIZE
    )
    if options.global_batch_size is not None:
        parallel["global_batch_size"] = int(options.global_batch_size)
    else:
        parallel["global_batch_size"] = int(parallel.get("global_batch_size") or 256)
    if parallel["seq_length"] <= 0:
        raise ValueError("seq_length must be positive")
    if parallel["micro_batch_size"] <= 0:
        raise ValueError("micro_batch_size must be positive")
    if parallel["global_batch_size"] <= 0:
        raise ValueError("global_batch_size must be positive")

    max_bubble = float(options.max_bubble_ratio)
    if not 0 < max_bubble < 1:
        raise ValueError("max_bubble_ratio must be between 0 and 1")
    estimator_config["_solve_max_bubble_ratio"] = max_bubble
    margin = float(
        options.memory_margin_gib
        if options.memory_margin_gib is not None
        else DEFAULT_MEMORY_MARGIN_GIB
    )
    if margin < 0:
        raise ValueError("memory margin cannot be negative")
    hbm = float(estimator_config["hardware_config"]["hbm_gib"])
    memory_limit = hbm - margin
    if memory_limit <= 0:
        raise ValueError("memory margin must be smaller than hardware_config.hbm_gib")

    parameters, adapter = _model_scale(model_config, estimator_config)
    gpus_per_node = _configured_gpus_per_node(estimator_config["hardware_config"])
    static_18b = math.ceil(parameters * 18 / (memory_limit * 1024**3))
    # At DP=1 the repository's BF16 + FP32 Adam state is 18 bytes per
    # parameter.  Increasing DP only replicates weights/gradients, so it cannot
    # lower the global-card-count bound even with a distributed optimizer.
    world_floor = max(gpus_per_node, math.ceil(static_18b / gpus_per_node) * gpus_per_node)
    world_limit = max(
        world_floor,
        math.ceil(static_18b * INITIAL_MINIMUM_WORLD_MULTIPLIER / gpus_per_node)
        * gpus_per_node,
    )
    max_world_limit = max(
        world_limit,
        math.ceil(static_18b * MAX_MINIMUM_WORLD_MULTIPLIER / gpus_per_node)
        * gpus_per_node,
    )

    log_table(
        logger,
        "solve inputs",
        [{
            "config": str(config_path_obj),
            "model_config": str(model_path_obj.resolve()),
            "adapter": adapter,
            "model_parameters_b": round(parameters / 1e9, 2),
            "seq_length": parallel["seq_length"],
            "mbs": parallel["micro_batch_size"],
            "preferred_gbs": parallel["global_batch_size"],
            "max_bubble": f"{max_bubble * 100:.2f}%",
            "hbm_gib": hbm,
            "safety_margin_gib": margin,
            "distributed_optimizer": parallel["use_distributed_optimizer"],
            "full_recompute": parallel["full_recompute"],
            "configured_profile_mode": (estimator_config.get("profile_config") or {}).get("mode", "auto"),
            "performance_basis": "analytical estimator",
        }],
    )
    log_table(logger, "solve hardware configuration", [estimator_config["hardware_config"]])
    log_table(
        logger,
        "solve policy",
        [{
            "precision": "BF16 mixed precision Adam",
            "memory_margin_gib": margin,
            "max_bubble": f"{max_bubble * 100:.2f}%",
            "tp_limit": 32,
            "cp_limit": 16,
            "pp_limit": 64,
            "ep_limit": 128,
            "etp_perf_limit": 8,
            "dp_candidates": "powers of two",
            "minimum_gpu_search": (
                f"adaptive {INITIAL_MINIMUM_WORLD_MULTIPLIER}x-"
                f"{MAX_MINIMUM_WORLD_MULTIPLIER}x static bound"
            ),
            "performance_gpu_limit": f"{PERFORMANCE_WORLD_MULTIPLIER:.1f}x minimum",
            "sequence_parallel": parallel.get("sequence_parallel", True),
            "overlap_mode": parallel.get("overlap_mode", "auto"),
            "ep_backend": parallel.get("ep_communication_backend", "alltoall"),
        }],
    )
    for assumption in assumptions:
        logger.warning("solve fallback: %s", assumption)

    minimum, memory_evaluated, candidates, minimum_world_feasible, world_limit = (
        _find_minimum_solution_with_expansion(
            model_config,
            estimator_config,
            memory_limit,
            world_floor,
            world_limit,
            max_world_limit,
            gpus_per_node,
        )
    )
    if minimum is None:
        raise RuntimeError(
            f"no memory-feasible configuration found up to {world_limit} GPUs; "
            "increase HBM, enable recompute, or reduce sequence length"
        )
    minimum = _attach_performance(model_config, estimator_config, minimum)
    performance, performance_evaluated = _find_performance_solution(
        model_config,
        estimator_config,
        memory_limit,
        minimum,
        candidates,
        minimum_world_feasible,
    )

    strategy_rows, memory_rows, performance_rows = [], [], []
    for label, result in (("minimum GPUs", minimum), ("higher performance", performance)):
        strategy, memory, perf = _result_rows(
            label, result, estimator_config["hardware_config"], memory_limit
        )
        strategy_rows.append(strategy)
        memory_rows.append(memory)
        performance_rows.append(perf)

    log_table(logger, "solve parallel configurations", strategy_rows)
    log_table(logger, "solve peak memory", memory_rows)
    log_table(logger, "solve performance overview", performance_rows)
    log_step(
        logger,
        "solve finished: memory_candidates=%s, performance_candidates=%s, safety_margin=%.2fGiB",
        memory_evaluated,
        performance_evaluated,
        margin,
    )
    return {"minimum": minimum, "performance": performance, "assumptions": assumptions}
