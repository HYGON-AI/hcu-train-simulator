# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from hcu_train_simulator.logging import get_logger, log_step


logger = get_logger(__name__)
PROFILE_SCHEMA_VERSION = 1
PROFILE_BENCHMARK_VERSION = 1


GEMM_MODULES = {
    "qkv_weight",
    "attn_proj",
    "linear_fc1",
    "linear_fc2",
    "topk_router",
    "moe_linear_fc1",
    "moe_linear_fc2",
    "shared_expert_fc1",
    "shared_expert_fc2",
    "mla_q_down",
    "mla_q_up",
    "mla_kv_down",
    "mla_kv_up",
    "lm_head",
}

MOE_LOCAL_EXPERT_MODULES = {
    "moe_linear_fc1",
    "moe_linear_fc2",
    "moe_swiglu_activation",
}


def parse_profile_key(key: str) -> dict[str, str]:
    fields = {}
    for item in str(key).split(","):
        if "=" not in item:
            continue
        name, value = item.split("=", 1)
        fields[name.strip()] = value.strip()
    return fields


def make_profile_key(fields: dict[str, Any], order: list[str]) -> str:
    return ",".join(f"{name}={fields[name]}" for name in order if fields.get(name) is not None)


def _string_or_empty(value: Any) -> str:
    return "" if value is None else str(value)


def _yaml_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _yaml_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_yaml_safe(item) for item in value]
    return str(value)


def _measurement_entry(
    status: str,
    benchmark_result: dict[str, Any],
    candidate_reason: str | None = None,
    *,
    include_tflops: bool = True,
) -> dict[str, Any]:
    """Return the compact on-disk representation for one operator shape."""

    entry = {
        "status": status,
        "mean_ms": benchmark_result.get("mean_ms"),
    }
    if include_tflops:
        entry["tflops"] = benchmark_result.get("tflops")
    entry["updated_at"] = date.today().isoformat()
    if status == "candidate":
        entry["candidate_reason"] = (
            candidate_reason or "profile validation did not pass"
        )
    return entry


def _compact_measurement_entries(data: dict[str, Any]) -> bool:
    """Drop legacy diagnostic fields whenever a profile is written."""

    changed = False
    for container_name, default_status in (
        ("operators", "validated"),
        ("candidates", "candidate"),
    ):
        container = data.get(container_name)
        if not isinstance(container, dict):
            continue
        for operator_type, entries in container.items():
            if not isinstance(entries, dict):
                continue
            for key, entry in list(entries.items()):
                if not isinstance(entry, dict):
                    continue
                status = str(entry.get("status") or default_status)
                compacted = {
                    "status": status,
                    "mean_ms": entry.get("mean_ms"),
                }
                if operator_type != "non_gemm":
                    compacted["tflops"] = entry.get("tflops")
                compacted["updated_at"] = (
                    entry.get("updated_at") or date.today().isoformat()
                )
                if status == "candidate":
                    compacted["candidate_reason"] = entry.get(
                        "candidate_reason"
                    ) or "legacy candidate; reason was not recorded"
                if compacted != entry:
                    entries[key] = compacted
                    changed = True
    return changed


def _normalize_profile_document(data: dict[str, Any]) -> bool:
    """Normalize legacy fields to the current on-disk profile schema."""

    changed = False
    has_legacy_software = "software" in data
    legacy_software = data.pop("software", None)
    if isinstance(legacy_software, dict):
        environment = data.setdefault("environment", {})
        environment_software = environment.setdefault("software", {})
        for key, value in legacy_software.items():
            canonical_key = "flash_attention" if key == "flash_attn" else key
            if not environment_software.get(canonical_key) and value is not None:
                environment_software[canonical_key] = value
    if has_legacy_software:
        changed = True
    return _compact_measurement_entries(data) or changed


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def normalize_accelerator(value: Any) -> str:
    text = _string_or_empty(value).strip().lower()
    # BW1000 nodes report the board name as ``BW200, UBB BW1000``.  Match the
    # product name before falling back to the first generic BW token, otherwise
    # the same accelerator is incorrectly profiled as ``bw200``.
    product_match = re.search(r"bw\s*[-_]?\s*(1000|1100)\b", text)
    if product_match:
        return f"bw{product_match.group(1)}"
    match = re.search(r"bw\s*[-_]?\s*(\d+)", text)
    return f"bw{match.group(1)}" if match else re.sub(r"[^a-z0-9]+", "", text)


def normalize_torch_version(value: Any) -> str:
    text = _string_or_empty(value).strip()
    match = re.search(r"(\d+)\.(\d+)(?:\.(\d+))?", text)
    if match:
        return "".join(part for part in match.groups() if part is not None)
    return re.sub(r"\D+", "", text)


def normalize_dtk_version(value: Any) -> str:
    text = _string_or_empty(value).strip()
    match = re.search(r"(?:DTK[-_ ]*)?(\d{2})\.(\d{2})", text, re.IGNORECASE)
    if match:
        return "".join(match.groups())
    compact = re.sub(r"\D+", "", text)
    return compact[:4]


def environment_signature(data: dict[str, Any]) -> dict[str, Any]:
    environment = data.get("environment") if isinstance(data, dict) else None
    if isinstance(environment, dict):
        hardware = environment.get("hardware") or {}
        software = environment.get("software") or {}
        return {
            "accelerator": normalize_accelerator(hardware.get("accelerator")),
            "hbm_gib": hardware.get("hbm_gib"),
            "driver": _string_or_empty(hardware.get("driver")),
            "torch": _string_or_empty(software.get("torch")),
            "transformer_engine": _string_or_empty(software.get("transformer_engine")),
            "flash_attention": _string_or_empty(software.get("flash_attention")),
            "triton": _string_or_empty(software.get("triton")),
            "dtk": _string_or_empty(software.get("dtk")),
            "dtype": _string_or_empty(environment.get("dtype") or "bf16"),
        }

    hardware = data.get("hardware", {}) if isinstance(data, dict) else {}
    software = data.get("software", {}) if isinstance(data, dict) else {}
    return {
        "accelerator": normalize_accelerator(hardware.get("accelerator")),
        "hbm_gib": hardware.get("hbm_gib"),
        "driver": _string_or_empty(hardware.get("driver")),
        "torch": _string_or_empty(software.get("torch")),
        "transformer_engine": _string_or_empty(software.get("transformer_engine")),
        "flash_attention": _string_or_empty(
            software.get("flash_attention") or software.get("flash_attn")
        ),
        "triton": _string_or_empty(software.get("triton")),
        "dtk": _string_or_empty(software.get("dtk")),
        "dtype": _string_or_empty(
            (data.get("profile") or {}).get("dtype") or "bf16"
        ),
    }


def environment_fingerprint(data: dict[str, Any]) -> str:
    payload = json.dumps(environment_signature(data), sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]


def profile_filename(data: dict[str, Any]) -> str:
    signature = environment_signature(data)
    accelerator = signature["accelerator"] or "unknown"
    torch_key = normalize_torch_version(signature["torch"]) or "unknown"
    dtk_key = normalize_dtk_version(signature["dtk"]) or "unknown"
    return f"{accelerator}_torch{torch_key}_dtk{dtk_key}_{environment_fingerprint(data)}.yaml"


def entry_is_validated(entry: Any) -> bool:
    return isinstance(entry, dict) and str(entry.get("status", "validated")).lower() == "validated"


def get_compute_multiplier(module: dict[str, Any], config) -> float:
    compute_count = module.get("compute_count", 0)
    module_name = module.get("model_part")
    multiplier = compute_count

    if module_name in MOE_LOCAL_EXPERT_MODULES:
        num_experts = config.transformer.num_moe_experts
        ep_size = getattr(config.parallel, "ep_size", None)
        if num_experts and ep_size:
            local_experts = max(1, num_experts // ep_size)
            benchmark_local_experts = min(
                int(getattr(config.parallel, "benchmark_max_local_experts", None) or local_experts),
                local_experts,
            )
            multiplier = compute_count / max(1, benchmark_local_experts)

    if module_name == "qk_norm":
        multiplier *= float(getattr(config.parallel, "benchmark_qk_norm_scale", 1.0) or 1.0)
    return multiplier


def _driver_version() -> str:
    try:
        completed = subprocess.run(
            ["hy-smi", "--showdriverversion"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
            check=False,
        )
    except Exception:
        return ""
    match = re.search(r"Driver Version:\s*([^\r\n]+)", completed.stdout or "")
    return match.group(1).strip() if match else ""


def build_profile_environment(env_info: dict[str, Any] | None, config) -> dict[str, Any]:
    env_info = env_info or {}
    accelerator_name = _string_or_empty(
        env_info.get("accelerator") or env_info.get("gpu_name")
    )
    accelerator = normalize_accelerator(accelerator_name) or accelerator_name
    dtype = _string_or_empty(getattr(config.parallel, "benchmark_dtype", "bf16") or "bf16")
    hardware = {
        "accelerator": accelerator,
        "hbm_gib": getattr(config.hardware, "hbm_gib", None),
        "driver": _string_or_empty(env_info.get("driver") or env_info.get("driver_version") or _driver_version()),
    }
    software = {
        "torch": _string_or_empty(env_info.get("torch_version")),
        "transformer_engine": _string_or_empty(env_info.get("te_version")),
        "flash_attention": _string_or_empty(env_info.get("fa_version")),
        "triton": _string_or_empty(env_info.get("triton_version")),
        "dtk": _string_or_empty(env_info.get("dtk_version")),
    }
    return {
        "profile": {
            "schema_version": PROFILE_SCHEMA_VERSION,
            "benchmark_version": PROFILE_BENCHMARK_VERSION,
            "name": f"{normalize_accelerator(accelerator) or 'unknown'}_torch{normalize_torch_version(software['torch']) or 'unknown'}_dtk{normalize_dtk_version(software['dtk']) or 'unknown'}",
            "dtype": dtype,
            "updated_at": date.today().isoformat(),
        },
        "environment": {
            "dtype": dtype,
            "hardware": hardware,
            "software": software,
        },
        "operators": {
            "gemm": {},
            "flash_attention": {},
            "non_gemm": {},
        },
        "candidates": {
            "gemm": {},
            "flash_attention": {},
            "non_gemm": {},
        },
    }


class OperatorProfileStore:
    def __init__(
        self,
        path: str | Path | None,
        config,
        env_info: dict[str, Any] | None = None,
        *,
        data: dict[str, Any] | None = None,
        source: str = "user profile hit",
        writable: bool = True,
    ):
        self.path = Path(path) if path else None
        self.config = config
        self.source = source
        self.writable = writable
        self.data = data if data is not None else self._load()
        if env_info is not None:
            self.ensure_environment(env_info)

    def _load(self) -> dict[str, Any]:
        if not self.path or not self.path.exists():
            return {}
        loaded = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
        return loaded if isinstance(loaded, dict) else {}

    def ensure_environment(self, env_info: dict[str, Any] | None = None) -> None:
        base = build_profile_environment(env_info, self.config)
        for section, value in base.items():
            current = self.data.setdefault(section, {})
            if isinstance(value, dict):
                for key, default in value.items():
                    current.setdefault(key, default)
            else:
                self.data.setdefault(section, value)
        operators = self.data.setdefault("operators", {})
        operators.setdefault("gemm", {})
        operators.setdefault("flash_attention", {})
        operators.setdefault("non_gemm", {})
        candidates = self.data.setdefault("candidates", {})
        candidates.setdefault("gemm", {})
        candidates.setdefault("flash_attention", {})
        candidates.setdefault("non_gemm", {})

    def matches_environment(self, env_info: dict[str, Any] | None) -> bool:
        """Return whether the profile was measured in this exact environment."""
        expected = build_profile_environment(env_info, self.config)
        return environment_signature(self.data) == environment_signature(expected)

    def reset_for_environment(self, env_info: dict[str, Any] | None) -> None:
        """Discard measurements from another environment before profiling."""
        self.data = build_profile_environment(env_info, self.config)
        self.ensure_environment(env_info)

    def save(self) -> None:
        if not self.path or not self.writable:
            return
        self.ensure_environment()
        directory_created = not self.path.parent.exists()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if directory_created:
            log_step(logger, "created user profile directory: %s", self.path.parent)
        self.data.setdefault("profile", {})["updated_at"] = date.today().isoformat()
        _normalize_profile_document(self.data)
        self.data = _yaml_safe(self.data)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            yaml.safe_dump(self.data, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        temporary.replace(self.path)
        log_step(logger, "user profile saved: %s", self.path)

    def migrate_measurement_entries(self) -> bool:
        """Persist the compact schema and add reasons to legacy candidates."""

        changed = _normalize_profile_document(self.data)
        if changed:
            self.save()
        return changed

    def find_gemm(self, module: dict[str, Any], dtype: str) -> tuple[dict[str, Any] | None, str]:
        target = [
            _as_int(module.get("m")),
            _as_int(module.get("n")),
            _as_int(module.get("k")),
        ]
        if any(value is None for value in target):
            return None, ""
        target_part = str(module.get("model_part") or "")
        target_grouped = module.get("grouped_gemm")
        target_num_gemms = _as_int(module.get("num_gemms"))
        require_context = target_part in MOE_LOCAL_EXPERT_MODULES
        gemm = self.data.get("operators", {}).get("gemm", {}) or {}
        for key, value in gemm.items():
            fields = parse_profile_key(key)
            if fields.get("dtype") != dtype:
                continue
            stored_part = fields.get("part")
            if stored_part and stored_part != target_part:
                continue
            if require_context and not stored_part:
                continue
            if fields.get("grouped_gemm") is not None:
                if str(fields["grouped_gemm"]).lower() != str(bool(target_grouped)).lower():
                    continue
            stored_num_gemms = _as_int(fields.get("num_gemms"))
            if stored_num_gemms is not None and stored_num_gemms != target_num_gemms:
                continue
            dims = [_as_int(fields.get("m")), _as_int(fields.get("n")), _as_int(fields.get("k"))]
            if any(item is None for item in dims):
                continue
            if dims == target:
                if entry_is_validated(value):
                    return value, self.source
                continue
            if getattr(self.config.profile, "match_permuted_gemm", True) and sorted(dims) == sorted(target):
                if entry_is_validated(value):
                    return value, f"{self.source} (permuted)"
        return None, ""

    def find_flash_attention(self, dtype: str, module: dict[str, Any] | None = None) -> tuple[dict[str, Any] | None, str]:
        key = self.flash_attention_key(dtype, module)
        flash_attention = self.data.get("operators", {}).get("flash_attention", {}) or {}
        item = flash_attention.get(key)
        if entry_is_validated(item):
            return item, self.source
        module = module or {}
        seq = _as_int(module.get("seq"))
        kv_seq = _as_int(module.get("kv_seq"))
        if seq is not None and seq == kv_seq:
            # Profiles written before kv_seq became part of the key are safe
            # only for self-attention where Q and KV have the same length.
            legacy_key = make_profile_key(
                {
                    "batch": module.get("b"),
                    "seq": seq,
                    "heads": module.get("heads"),
                    "head_dim": module.get("head_dim"),
                    "value_head_dim": module.get("value_head_dim", module.get("head_dim")),
                    "dtype": dtype,
                    "causal": str(module.get("causal", True)).lower(),
                },
                ["batch", "seq", "heads", "head_dim", "value_head_dim", "dtype", "causal"],
            )
            item = flash_attention.get(legacy_key)
            if entry_is_validated(item):
                return item, f"{self.source} (legacy equal-Q/KV key)"
        return None, ""

    def find_non_gemm(self, model_part: Any, elements: Any, dtype: str) -> tuple[dict[str, Any] | None, str]:
        key = self.non_gemm_key(model_part, elements, dtype)
        item = (self.data.get("operators", {}).get("non_gemm", {}) or {}).get(key)
        if entry_is_validated(item):
            return item, self.source
        return None, ""

    def gemm_key(self, module: dict[str, Any], dtype: str) -> str:
        return make_profile_key(
            {
                "m": module.get("m"),
                "n": module.get("n"),
                "k": module.get("k"),
                "dtype": dtype,
                "part": module.get("model_part"),
                "grouped_gemm": module.get("grouped_gemm"),
                "num_gemms": module.get("num_gemms"),
            },
            ["part", "m", "n", "k", "dtype", "grouped_gemm", "num_gemms"],
        )

    def flash_attention_key(self, dtype: str, module: dict[str, Any] | None = None) -> str:
        model = self.config.transformer
        parallel = self.config.parallel
        module = module or {}
        return make_profile_key(
            {
                "batch": module.get("b", parallel.micro_batch_size),
                "seq": module.get("seq", parallel.seq_length // max(1, parallel.cp_size)),
                "kv_seq": module.get("kv_seq", parallel.seq_length),
                "heads": module.get("heads", model.num_attention_heads // max(1, parallel.tp_size)),
                "head_dim": module.get("head_dim", model.kv_channels),
                "value_head_dim": module.get(
                    "value_head_dim",
                    model.v_head_dim or model.kv_channels,
                ),
                "dtype": dtype,
                "causal": str(module.get("causal", getattr(parallel, "benchmark_causal", True))).lower(),
            },
            [
                "batch",
                "seq",
                "kv_seq",
                "heads",
                "head_dim",
                "value_head_dim",
                "dtype",
                "causal",
            ],
        )

    def non_gemm_key(self, model_part: Any, elements: Any, dtype: str) -> str:
        return make_profile_key(
            {
                "part": model_part,
                "elements": elements,
                "dtype": dtype,
            },
            ["part", "elements", "dtype"],
        )

    def update_from_benchmark(
        self,
        module: dict[str, Any],
        benchmark_result: dict[str, Any],
        dtype: str,
        samples: int,
        status: str = "validated",
        quality: dict[str, Any] | None = None,
        candidate_reason: str | None = None,
    ) -> bool:
        if benchmark_result.get("status") != "ok" or benchmark_result.get("mean_ms") is None:
            return False
        if str(module.get("benchmark_source", "")).startswith("covered by "):
            return False
        self.ensure_environment()
        if status not in {"validated", "candidate"}:
            raise ValueError(f"unsupported profile status: {status}")
        if status == "candidate":
            return self._insert_candidate(
                module,
                benchmark_result,
                dtype,
                samples,
                quality or {},
                candidate_reason,
            )
        if module.get("op_type") == "flash_attention":
            return self._insert_flash_attention(module, benchmark_result, dtype, samples)
        if module.get("op_type") == "gemm":
            if self.find_gemm(module, dtype)[0]:
                return False
            return self._insert_gemm(module, benchmark_result, dtype, samples)
        if module.get("op_type") == "non_gemm":
            if self.find_non_gemm(module.get("model_part"), module.get("elements"), dtype)[0]:
                return False
            return self._insert_non_gemm(module, benchmark_result, dtype, samples)
        return False

    def _insert_candidate(
        self,
        module: dict[str, Any],
        benchmark_result: dict[str, Any],
        dtype: str,
        samples: int,
        quality: dict[str, Any],
        candidate_reason: str | None,
    ) -> bool:
        op_type = module.get("op_type")
        section = {
            "gemm": "gemm",
            "flash_attention": "flash_attention",
            "non_gemm": "non_gemm",
        }.get(op_type)
        if not section:
            return False
        if section == "gemm":
            key = self.gemm_key(module, dtype)
        elif section == "flash_attention":
            key = self.flash_attention_key(dtype, module)
        else:
            key = self.non_gemm_key(module.get("model_part"), module.get("elements"), dtype)
        entry = _measurement_entry(
            "candidate",
            benchmark_result,
            candidate_reason=candidate_reason,
            include_tflops=section != "non_gemm",
        )
        self.data.setdefault("candidates", {}).setdefault(section, {})[key] = entry
        return True

    def _insert_gemm(self, module: dict[str, Any], benchmark_result: dict[str, Any], dtype: str, samples: int) -> bool:
        gemm = self.data.setdefault("operators", {}).setdefault("gemm", {})
        entry = _measurement_entry("validated", benchmark_result)
        key = self.gemm_key(module, dtype)
        gemm[key] = entry
        self.data.setdefault("candidates", {}).setdefault("gemm", {}).pop(key, None)
        return True

    def _insert_flash_attention(self, module: dict[str, Any], benchmark_result: dict[str, Any], dtype: str, samples: int) -> bool:
        flash_attention = self.data.setdefault("operators", {}).setdefault("flash_attention", {})
        key = self.flash_attention_key(dtype, module)
        if key in flash_attention:
            return False
        entry = _measurement_entry("validated", benchmark_result)
        flash_attention[key] = entry
        self.data.setdefault("candidates", {}).setdefault("flash_attention", {}).pop(key, None)
        return True

    def _insert_non_gemm(self, module: dict[str, Any], benchmark_result: dict[str, Any], dtype: str, samples: int) -> bool:
        non_gemm = self.data.setdefault("operators", {}).setdefault("non_gemm", {})
        key = self.non_gemm_key(module.get("model_part"), module.get("elements"), dtype)
        if key in non_gemm:
            return False
        entry = _measurement_entry(
            "validated",
            benchmark_result,
            include_tflops=False,
        )
        non_gemm[key] = entry
        self.data.setdefault("candidates", {}).setdefault("non_gemm", {}).pop(key, None)
        return True


def apply_profile_to_compute_result(compute_result: list[dict[str, Any]], profile_store: Any, config) -> list[dict[str, Any]]:
    dtype = getattr(config.parallel, "benchmark_dtype", "bf16")
    for module in compute_result:
        module["all_time_ms"] = module.get("all_time_ms", module["forward_ms"] + module["backward_ms"])
        module["benchmark_time_ms"] = "/"
        module["benchmark_source"] = "/"

        if module.get("op_type") == "flash_attention":
            entry, source = profile_store.find_flash_attention(dtype, module)
        elif module.get("op_type") == "gemm":
            entry, source = profile_store.find_gemm(module, dtype)
        elif module.get("op_type") == "non_gemm":
            entry, source = profile_store.find_non_gemm(module.get("model_part"), module.get("elements"), dtype)
        else:
            entry, source = None, ""

        if entry and entry.get("mean_ms") is not None:
            module["benchmark_time_ms"] = round(float(entry["mean_ms"]) * get_compute_multiplier(module, config), 2)
            module["benchmark_source"] = source
    return compute_result


def update_profile_from_benchmarks(
    profile_store: OperatorProfileStore,
    compute_result: list[dict[str, Any]],
    benchmark_results: list[Any],
    module_to_benchmark: dict[str, str],
    config,
    *,
    status: str = "validated",
    quality: dict[str, Any] | None = None,
    candidate_reason: str | None = None,
    only_marked: bool = False,
) -> bool:
    dtype = getattr(config.parallel, "benchmark_dtype", "bf16")
    samples = int(getattr(config.parallel, "benchmark_iters", 1) or 1)
    by_name = {result.name: asdict(result) for result in benchmark_results}
    changed = False
    for module in compute_result:
        if only_marked and not module.get("_profile_missing"):
            continue
        benchmark_names = module_to_benchmark.get(module.get("model_part"))
        if isinstance(benchmark_names, str):
            benchmark_names = [benchmark_names]
        if not benchmark_names:
            continue
        benchmark_result = None
        for benchmark_name in benchmark_names:
            candidate = by_name.get(benchmark_name)
            if candidate and candidate.get("status") == "ok" and candidate.get("mean_ms") is not None:
                benchmark_result = candidate
                break
        if not benchmark_result:
            continue
        changed = profile_store.update_from_benchmark(
            module,
            benchmark_result,
            dtype,
            samples,
            status=status,
            quality=quality,
            candidate_reason=candidate_reason,
        ) or changed
    if changed:
        profile_store.save()
    return changed
