# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import fields, is_dataclass


def flatten_dataclass(obj, prefix=""):
    result = {}

    for field in fields(obj):
        if field.metadata.get("flatten") is False:
            continue
        value = getattr(obj, field.name)
        key = f"{prefix}{field.name}"

        if is_dataclass(value):
            result.update(flatten_dataclass(value, prefix=prefix))
        else:
            result[key] = value

    return result
