# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Model adapter contracts.

Model adapters own HuggingFace configuration normalization and model-spec
composition.  Estimators consume the normalized configuration and ModuleSpec
tree and therefore do not need to identify model families themselves.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class NormalizedModelInput:
    """Configuration sections consumed by the simulator's common dataclasses."""

    language: dict[str, Any]
    vision: dict[str, Any] | None = None


class ModelAdapter:
    """Base class for a model-family integration."""

    name = "generic"
    model_types: tuple[str, ...] = ()
    model_type_prefixes: tuple[str, ...] = ()
    architecture_prefixes: tuple[str, ...] = ()
    fallback = False

    def match_score(self, raw_config: Mapping[str, Any]) -> int | None:
        model_type = str(raw_config.get("model_type") or "")
        architectures = tuple(str(item) for item in raw_config.get("architectures", ()))
        score: int | None = None
        if model_type in self.model_types:
            score = 10_000 + len(model_type)
        for prefix in self.model_type_prefixes:
            if model_type.startswith(prefix):
                score = max(score or 0, 1_000 + len(prefix))
        for prefix in self.architecture_prefixes:
            if any(name.startswith(prefix) for name in architectures):
                score = max(score or 0, 100 + len(prefix))
        if score is None and self.fallback:
            return 0
        return score

    def normalize_config(self, raw_config: Mapping[str, Any]) -> NormalizedModelInput:
        return NormalizedModelInput(language=dict(raw_config))

    def build_spec(self, simulation_config):
        raise NotImplementedError

    def validate(self, simulation_config) -> None:
        """Perform model-family validation after common validation."""


def merge_language_metadata(
    language_config: Mapping[str, Any],
    raw_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Preserve top-level HuggingFace identity fields on a nested text config."""

    merged = dict(language_config)
    for key in ("architectures", "model_type", "tie_word_embeddings"):
        if key in raw_config:
            merged[key] = raw_config[key]
    return merged


def with_rmsnorm_defaults(language_config: Mapping[str, Any]) -> dict[str, Any]:
    """Apply family-owned defaults without teaching the common config model names."""

    normalized = dict(language_config)
    normalized.setdefault("normalization", "RMSNorm")
    if "rms_norm_eps" not in normalized and "layer_norm_eps" not in normalized:
        normalized["layer_norm_eps"] = 1e-6
    return normalized
