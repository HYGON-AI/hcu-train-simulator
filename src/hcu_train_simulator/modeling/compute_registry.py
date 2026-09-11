# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Extension points for lowering ModuleSpec nodes into compute rows."""

from __future__ import annotations


_ATTENTION_LOWERINGS: dict[type, object] = {}
_COMPONENT_LOWERINGS: dict[type, object] = {}
_GEMM_SHAPES: dict[str, object] = {}
_FIRST_GEMM_PARTS: dict[type, str] = {}


def _register(target, key, handler):
    existing = target.get(key)
    if existing is not None and existing is not handler:
        raise ValueError(f"compute handler for {key!r} is already registered")
    target[key] = handler
    return handler


def register_attention_lowering(module: type, *, first_gemm_part: str):
    def decorator(handler):
        _FIRST_GEMM_PARTS[module] = first_gemm_part
        return _register(_ATTENTION_LOWERINGS, module, handler)

    return decorator


def register_component_lowering(module: type):
    def decorator(handler):
        return _register(_COMPONENT_LOWERINGS, module, handler)

    return decorator


def register_gemm_shape(model_part: str):
    def decorator(handler):
        return _register(_GEMM_SHAPES, model_part, handler)

    return decorator


def get_attention_lowering(spec):
    return _ATTENTION_LOWERINGS.get(spec.module)


def get_component_lowering(spec):
    return _COMPONENT_LOWERINGS.get(spec.module)


def get_first_gemm_part(spec):
    return _FIRST_GEMM_PARTS.get(spec.module)


def resolve_gemm_shape(model, model_part: str):
    handler = _GEMM_SHAPES.get(model_part)
    if handler is not None:
        return handler(model)
    return getattr(model, model_part)()
