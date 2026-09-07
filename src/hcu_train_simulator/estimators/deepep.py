# Copyright (c) 2026 Hygon Information Technology Co., Ltd.

"""Heuristic DeepEP training communication model.

The model intentionally reuses the simulator's measured AllToAll baseline and
only estimates how much of four training phases can be hidden.  It is a demo
model, not a replacement for a distributed DeepEP benchmark.
"""

from __future__ import annotations

import math
from typing import Any, Iterable


KIB = 1024
MIB = 1024 * KIB

PAYLOAD_OVERLAP_BUCKETS = (
    (256 * KIB, "<=256KiB", 0.10),
    (1 * MIB, "256KiB-1MiB", 0.20),
    (4 * MIB, "1MiB-4MiB", 0.35),
    (16 * MIB, "4MiB-16MiB", 0.50),
    (64 * MIB, "16MiB-64MiB", 0.60),
    (None, ">64MiB", 0.65),
)

TOPOLOGY_FACTORS = {
    "intra_node": 1.00,
    "inter_node": 0.85,
}

SM_OVERLAP_EFFICIENCY = 0.85
MAX_EFFECTIVE_OVERLAP = 0.60
EXPERT_COMPUTE_WINDOW_RATIO = 0.40

# Dispatch has a stronger dependency on expert compute than combine, so its
# heuristic overlap allowance is deliberately more conservative.
PHASE_SPECS = (
    {
        "name": "forward_dispatch",
        "direction": "forward",
        "comm_factor": 1.00,
        "overlap_factor": 0.75,
        "window_share": 0.40,
    },
    {
        "name": "forward_combine",
        "direction": "forward",
        "comm_factor": 1.00,
        "overlap_factor": 1.00,
        "window_share": 0.60,
    },
    {
        "name": "backward_dispatch",
        "direction": "backward",
        "comm_factor": 1.00,
        "overlap_factor": 0.70,
        "window_share": 0.40,
    },
    {
        "name": "backward_combine",
        "direction": "backward",
        "comm_factor": 1.00,
        "overlap_factor": 0.90,
        "window_share": 0.60,
    },
)


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(value, upper))


def select_payload_overlap(payload_bytes: float) -> tuple[str, float]:
    """Return the embedded payload bucket and its base overlap ratio."""

    size = max(0.0, float(payload_bytes))
    for maximum, label, ratio in PAYLOAD_OVERLAP_BUCKETS:
        if maximum is None or size <= maximum:
            return label, ratio
    raise AssertionError("the final DeepEP payload bucket must be unbounded")


def classify_topology(ep_nodes: int) -> str:
    return "intra_node" if ep_nodes <= 1 else "inter_node"


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _module_direction_time_s(module: dict[str, Any], direction: str) -> float:
    """Split a measured aggregate time using the theoretical FWD/BWD ratio."""

    forward_ms = float(module.get("forward_ms", 0.0) or 0.0)
    backward_ms = float(module.get("backward_ms", 0.0) or 0.0)
    benchmark_ms = module.get("benchmark_time_ms")
    if _is_number(benchmark_ms):
        theoretical_total = forward_ms + backward_ms
        if theoretical_total > 0:
            share = (
                forward_ms / theoretical_total
                if direction == "forward"
                else backward_ms / theoretical_total
            )
        else:
            share = 0.5
        return float(benchmark_ms) * share / 1000
    return (forward_ms if direction == "forward" else backward_ms) / 1000


def estimate_expert_direction_time_s(
    compute_result: list[dict[str, Any]] | None,
    expert_names: Iterable[str],
    direction: str,
) -> float:
    names = set(expert_names)
    return sum(
        _module_direction_time_s(module, direction)
        for module in compute_result or []
        if module.get("model_part") in names
    )


def estimate_deepep_phases(
    *,
    single_collective_time_s: float,
    phase_count: float,
    payload_bytes: float,
    compute_result: list[dict[str, Any]] | None,
    expert_names: Iterable[str],
    config,
    ep_nodes: int | None = None,
) -> dict[str, Any]:
    """Estimate raw and exposed DeepEP time for four training phases."""

    parallel = config.parallel
    if ep_nodes is None:
        ep_nodes = max(
            1,
            math.ceil(parallel.ep_size / max(1, config.hardware.gpus_per_node)),
        )
    topology = classify_topology(ep_nodes)
    topology_factor = TOPOLOGY_FACTORS[topology]
    payload_bucket, base_overlap_ratio = select_payload_overlap(payload_bytes)
    forward_expert_time_s = estimate_expert_direction_time_s(
        compute_result,
        expert_names,
        "forward",
    )
    backward_expert_time_s = estimate_expert_direction_time_s(
        compute_result,
        expert_names,
        "backward",
    )
    direction_times = {
        "forward": forward_expert_time_s,
        "backward": backward_expert_time_s,
    }

    mode = str(parallel.overlap_mode or "auto").lower()
    manual_ratio = clamp(float(parallel.ep_overlap_ratio or 0.0))
    if parallel.pp_size <= 1:
        overlap_policy = "disabled_pp1"
    elif mode == "none":
        overlap_policy = "none"
    elif mode == "manual":
        overlap_policy = "manual"
    elif not parallel.ep_overlap_enabled:
        overlap_policy = "disabled"
    else:
        overlap_policy = "heuristic"

    phases = []
    for spec in PHASE_SPECS:
        raw_time_s = (
            max(0.0, float(single_collective_time_s))
            * max(0.0, float(phase_count))
            * spec["comm_factor"]
        )
        compute_window_s = (
            direction_times[spec["direction"]]
            * EXPERT_COMPUTE_WINDOW_RATIO
            * spec["window_share"]
        )

        if overlap_policy == "manual":
            requested_ratio = manual_ratio
            hidden_time_s = raw_time_s * requested_ratio
        elif overlap_policy == "heuristic":
            requested_ratio = clamp(
                base_overlap_ratio
                * topology_factor
                * SM_OVERLAP_EFFICIENCY
                * spec["overlap_factor"],
                upper=MAX_EFFECTIVE_OVERLAP,
            )
            hidden_time_s = min(
                raw_time_s * requested_ratio,
                compute_window_s,
            )
        else:
            requested_ratio = 0.0
            hidden_time_s = 0.0

        exposed_time_s = max(0.0, raw_time_s - hidden_time_s)
        effective_ratio = hidden_time_s / raw_time_s if raw_time_s > 0 else 0.0
        phases.append(
            {
                "phase": spec["name"],
                "direction": spec["direction"],
                "calls": phase_count,
                "payload_bytes": int(payload_bytes),
                "comm_factor": spec["comm_factor"],
                "phase_overlap_factor": spec["overlap_factor"],
                "window_share": spec["window_share"],
                "compute_window_s": compute_window_s,
                "requested_overlap_ratio": requested_ratio,
                "effective_overlap_ratio": effective_ratio,
                "raw_time_s": raw_time_s,
                "hidden_time_s": hidden_time_s,
                "exposed_time_s": exposed_time_s,
            }
        )

    raw_time_s = sum(phase["raw_time_s"] for phase in phases)
    hidden_time_s = sum(phase["hidden_time_s"] for phase in phases)
    exposed_time_s = sum(phase["exposed_time_s"] for phase in phases)
    effective_overlap_ratio = hidden_time_s / raw_time_s if raw_time_s > 0 else 0.0
    return {
        "backend": "deepep",
        "model": "heuristic_v1",
        "overlap_policy": overlap_policy,
        "payload_bytes": int(payload_bytes),
        "payload_bucket": payload_bucket,
        "base_overlap_ratio": base_overlap_ratio,
        "topology": topology,
        "ep_nodes": ep_nodes,
        "topology_factor": topology_factor,
        "sm_overlap_efficiency": SM_OVERLAP_EFFICIENCY,
        "expert_compute_window_ratio": EXPERT_COMPUTE_WINDOW_RATIO,
        "phase_count": phase_count,
        "forward_expert_time_s": forward_expert_time_s,
        "backward_expert_time_s": backward_expert_time_s,
        "raw_time_s": raw_time_s,
        "hidden_time_s": hidden_time_s,
        "exposed_time_s": exposed_time_s,
        "effective_overlap_ratio": effective_overlap_ratio,
        "phases": phases,
    }
