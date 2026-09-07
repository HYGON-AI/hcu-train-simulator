# Copyright (c) 2026 Hygon Information Technology Co., Ltd.

_CONFIG = None


def set_config(config):
    global _CONFIG

    if _CONFIG is not None:
        raise RuntimeError("simulation config already initialized")

    _CONFIG = config


def get_config():
    global _CONFIG

    if _CONFIG is None:
        raise RuntimeError("simulation config is not initialized")

    return _CONFIG


def reset_config():
    """Clear process-local configuration, primarily for repeated development runs."""
    global _CONFIG
    _CONFIG = None
