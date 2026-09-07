# Copyright (c) 2026 Hygon Information Technology Co., Ltd.

"""Shared Qwen vision-language encoder support."""

from hcu_train_simulator.models.common.qwen_vl.spec import build_vision_model_spec

# Import common registrations once when either consuming adapter loads this
# package. Benchmark registrations are loaded by each adapter only after that
# adapter has registered itself, avoiding a cycle through the benchmark registry.
from hcu_train_simulator.models.common.qwen_vl import communication as _communication  # noqa: E402,F401
from hcu_train_simulator.models.common.qwen_vl import compute as _compute  # noqa: E402,F401
from hcu_train_simulator.models.common.qwen_vl import costs as _costs  # noqa: E402,F401
from hcu_train_simulator.models.common.qwen_vl import statistics as _statistics  # noqa: E402,F401

__all__ = ["build_vision_model_spec"]
