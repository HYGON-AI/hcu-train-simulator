# Copyright (c) 2026 Hygon Information Technology Co., Ltd.

"""Lazy model-adapter registry, similar to Transformers' auto mappings."""

from __future__ import annotations

import importlib
from collections.abc import Mapping

from hcu_train_simulator.models.base import ModelAdapter


_ADAPTERS: dict[str, ModelAdapter] = {}
_BUILTINS_LOADED = False
_BUILTIN_MODULES = (
    "hcu_train_simulator.models.qwen3",
    "hcu_train_simulator.models.qwen3_vl",
    "hcu_train_simulator.models.deepseek_v3",
    "hcu_train_simulator.models.qwen3_5",
    "hcu_train_simulator.models.glm5",
    "hcu_train_simulator.models.generic",
)


def register_model_adapter(adapter: ModelAdapter) -> ModelAdapter:
    existing = _ADAPTERS.get(adapter.name)
    if existing is not None and existing is not adapter:
        raise ValueError(f"model adapter {adapter.name!r} is already registered")
    _ADAPTERS[adapter.name] = adapter
    return adapter


def _load_builtins() -> None:
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    for module_name in _BUILTIN_MODULES:
        importlib.import_module(module_name)
    _BUILTINS_LOADED = True


def get_model_adapter(name: str) -> ModelAdapter:
    _load_builtins()
    try:
        return _ADAPTERS[name]
    except KeyError as exc:
        raise KeyError(f"unknown model adapter {name!r}") from exc


def resolve_model_adapter(raw_config: Mapping) -> ModelAdapter:
    _load_builtins()
    matches = []
    for adapter in _ADAPTERS.values():
        score = adapter.match_score(raw_config)
        if score is not None:
            matches.append((score, adapter.name, adapter))
    if not matches:
        raise ValueError(
            "no model adapter matched model_type="
            f"{raw_config.get('model_type')!r}, architectures={raw_config.get('architectures')!r}"
        )
    return max(matches, key=lambda item: (item[0], item[1]))[2]


def registered_model_adapters() -> tuple[ModelAdapter, ...]:
    _load_builtins()
    return tuple(_ADAPTERS[name] for name in sorted(_ADAPTERS))
