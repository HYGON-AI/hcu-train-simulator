# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml

from hcu_train_simulator.logging import get_logger

from .profile import (
    OperatorProfileStore,
    build_profile_environment,
    environment_signature,
    normalize_accelerator,
    normalize_dtk_version,
    normalize_torch_version,
    profile_filename,
)


logger = get_logger(__name__)


class OperatorProfileCatalog:
    """Resolve validated operator data from user and packaged profile stores."""

    def __init__(self, config):
        self.config = config
        self.target_data: dict[str, Any] | None = None
        self.user_stores: list[OperatorProfileStore] = []
        self.builtin_stores: list[OperatorProfileStore] = []
        self.user_store: OperatorProfileStore | None = None
        self.selected_builtin_id: str | None = None

    @property
    def enabled(self) -> bool:
        return bool(self.config.profile.enabled and self.config.profile.mode != "theoretical")

    def select_runtime_environment(self, env_info: dict[str, Any]) -> None:
        self.target_data = build_profile_environment(env_info, self.config)
        self.selected_builtin_id = None
        self._load_matching_stores()

    def select_builtin(self) -> bool:
        selector = {
            "accelerator": normalize_accelerator(self.config.profile.builtin_accelerator),
            "torch": normalize_torch_version(self.config.profile.builtin_torch),
            "dtk": normalize_dtk_version(self.config.profile.builtin_dtk),
            "dtype": str(getattr(self.config.parallel, "benchmark_dtype", "bf16") or "bf16").lower(),
        }
        if not all(selector.values()):
            logger.warning(
                "builtin profile mode requires accelerator, torch, and dtk selectors"
            )
            return False

        matches = []
        for item, data in self._builtin_documents():
            if str(item.get("status", "validated")).lower() != "validated":
                continue
            if (
                normalize_accelerator(item.get("accelerator")) == selector["accelerator"]
                and normalize_torch_version(item.get("torch")) == selector["torch"]
                and normalize_dtk_version(item.get("dtk")) == selector["dtk"]
                and str(item.get("dtype", "bf16")).lower() == selector["dtype"]
            ):
                matches.append((item, data))

        if not matches:
            logger.warning(
                "builtin operator profile not found: accelerator=%s torch=%s dtk=%s dtype=%s",
                selector["accelerator"],
                selector["torch"],
                selector["dtk"],
                selector["dtype"],
            )
            return False
        if len(matches) > 1:
            raise ValueError(f"ambiguous builtin profile selector: {selector}")

        item, data = matches[0]
        self.target_data = data
        self.selected_builtin_id = str(item.get("id") or item.get("file"))
        self._load_matching_stores(preselected_builtin=(item, data))
        logger.info("selected builtin operator profile: %s", self.selected_builtin_id)
        return True

    def matches_runtime_environment(self, env_info: dict[str, Any]) -> bool:
        if not self.target_data:
            return False
        expected = build_profile_environment(env_info, self.config)
        return environment_signature(self.target_data) == environment_signature(expected)

    def prepare_user_store(self) -> OperatorProfileStore:
        if not self.target_data:
            raise RuntimeError("profile target environment has not been selected")
        if self.user_store is not None:
            return self.user_store

        user_dir = Path(self.config.profile.user_dir)
        path = user_dir / profile_filename(self.target_data)
        self.user_store = OperatorProfileStore(
            path,
            self.config,
            data=self.target_data_copy(),
            source="user profile hit",
            writable=True,
        )
        self.user_stores.insert(0, self.user_store)
        return self.user_store

    def target_data_copy(self) -> dict[str, Any]:
        # YAML round-trip gives us a plain deep copy without sharing builtin data.
        data = yaml.safe_load(yaml.safe_dump(self.target_data, sort_keys=False)) or {}
        data["operators"] = {"gemm": {}, "flash_attention": {}, "non_gemm": {}}
        data["candidates"] = {"gemm": {}, "flash_attention": {}, "non_gemm": {}}
        return data

    def find_gemm(self, module: dict[str, Any], dtype: str):
        return self._find("find_gemm", module, dtype)

    def find_flash_attention(self, dtype: str, module: dict[str, Any] | None = None):
        return self._find("find_flash_attention", dtype, module)

    def find_non_gemm(self, model_part: Any, elements: Any, dtype: str):
        return self._find("find_non_gemm", model_part, elements, dtype)

    def source_counts(self, compute_result: list[dict[str, Any]]) -> dict[str, int]:
        counts = {
            "user profile hit": 0,
            "builtin profile hit": 0,
            "live benchmark": 0,
            "theoretical": 0,
            "covered": 0,
        }
        for module in compute_result:
            source = str(module.get("benchmark_source") or "")
            if source.startswith("user profile hit"):
                counts["user profile hit"] += 1
            elif source.startswith("builtin profile hit"):
                counts["builtin profile hit"] += 1
            elif source == "live benchmark":
                counts["live benchmark"] += 1
            elif source.startswith("covered by"):
                counts["covered"] += 1
            else:
                counts["theoretical"] += 1
        return counts

    def candidate_count(self) -> int:
        count = 0
        for store in self.user_stores:
            for entries in (store.data.get("candidates") or {}).values():
                if isinstance(entries, dict):
                    count += len(entries)
        return count

    def _find(self, method: str, *args):
        for store in [*self.user_stores, *self.builtin_stores]:
            entry, source = getattr(store, method)(*args)
            if entry is not None:
                return entry, source
        return None, ""

    def _load_matching_stores(self, preselected_builtin=None) -> None:
        self.user_stores = []
        self.builtin_stores = []
        self.user_store = None
        if not self.target_data:
            return
        target_signature = environment_signature(self.target_data)

        user_paths = []
        user_dir = Path(self.config.profile.user_dir)
        if user_dir.exists():
            user_paths.extend(sorted(user_dir.glob("*.yaml")))
        legacy_path = self.config.profile.legacy_path
        if legacy_path and Path(legacy_path).exists():
            user_paths.append(Path(legacy_path))

        for path in user_paths:
            try:
                store = OperatorProfileStore(path, self.config, source="user profile hit")
            except Exception as exc:
                logger.warning("failed to load user profile %s: %s", path, exc)
                continue
            if environment_signature(store.data) == target_signature:
                store.migrate_measurement_entries()
                self.user_stores.append(store)
        if self.user_stores:
            self.user_store = self.user_stores[0]

        builtin_documents = (
            [preselected_builtin] if preselected_builtin is not None else self._builtin_documents()
        )
        for item, data in builtin_documents:
            if str(item.get("status", "validated")).lower() != "validated":
                continue
            if environment_signature(data) != target_signature:
                continue
            self.builtin_stores.append(
                OperatorProfileStore(
                    None,
                    self.config,
                    data=data,
                    source="builtin profile hit",
                    writable=False,
                )
            )

    @staticmethod
    def _builtin_documents():
        root = files("hcu_train_simulator").joinpath("profiles", "builtin")
        manifest_resource = root.joinpath("manifest.yaml")
        if not manifest_resource.is_file():
            return []
        manifest = yaml.safe_load(manifest_resource.read_text(encoding="utf-8")) or {}
        documents = []
        for item in manifest.get("profiles") or []:
            resource = root.joinpath(str(item.get("file") or ""))
            if not resource.is_file():
                logger.warning("builtin profile resource is missing: %s", item.get("file"))
                continue
            data = yaml.safe_load(resource.read_text(encoding="utf-8")) or {}
            if isinstance(data, dict):
                documents.append((item, data))
        return documents
