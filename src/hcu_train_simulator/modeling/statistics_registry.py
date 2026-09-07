# Copyright (c) 2026 Hygon Information Technology Co., Ltd.

"""Registries for model-specific analytical FLOP contributions."""

_ATTENTION_FLOPS: dict[type, object] = {}
_COMPONENT_FLOPS: dict[type, object] = {}


def _register(target, module, handler):
    existing = target.get(module)
    if existing is not None and existing is not handler:
        raise ValueError(f"statistics handler for {module!r} is already registered")
    target[module] = handler
    return handler


def register_attention_flops(module):
    def decorator(handler):
        return _register(_ATTENTION_FLOPS, module, handler)

    return decorator


def register_component_flops(module):
    def decorator(handler):
        return _register(_COMPONENT_FLOPS, module, handler)

    return decorator


def get_attention_flops(spec):
    return _ATTENTION_FLOPS.get(spec.module)


def get_component_flops(spec):
    return _COMPONENT_FLOPS.get(spec.module)
