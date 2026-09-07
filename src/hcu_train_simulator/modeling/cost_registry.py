# Copyright (c) 2026 Hygon Information Technology Co., Ltd.

"""Extension registry for ModuleSpec parameter and activation costs."""

from __future__ import annotations

from collections.abc import Callable


_PARAMETER_BY_ROLE: dict[str, Callable] = {}
_PARAMETER_BY_MODULE: dict[type, Callable] = {}
_ACTIVATION_BY_ROLE: dict[str, Callable] = {}
_ACTIVATION_BY_MODULE: dict[type, Callable] = {}


def _register(target, key, handler):
    existing = target.get(key)
    if existing is not None and existing is not handler:
        raise ValueError(f"cost handler for {key!r} is already registered")
    target[key] = handler
    return handler


def register_parameter_cost(*, role: str | None = None, module: type | None = None):
    if (role is None) == (module is None):
        raise ValueError("register_parameter_cost requires exactly one of role or module")

    def decorator(handler):
        return _register(_PARAMETER_BY_ROLE if role is not None else _PARAMETER_BY_MODULE, role or module, handler)

    return decorator


def register_activation_cost(*, role: str | None = None, module: type | None = None):
    if (role is None) == (module is None):
        raise ValueError("register_activation_cost requires exactly one of role or module")

    def decorator(handler):
        return _register(_ACTIVATION_BY_ROLE if role is not None else _ACTIVATION_BY_MODULE, role or module, handler)

    return decorator


def get_parameter_cost_handler(spec):
    role = spec.metainfo.get("role")
    return _PARAMETER_BY_ROLE.get(role) or _PARAMETER_BY_MODULE.get(spec.module)


def get_activation_cost_handler(spec):
    role = spec.metainfo.get("role")
    return _ACTIVATION_BY_ROLE.get(role) or _ACTIVATION_BY_MODULE.get(spec.module)
