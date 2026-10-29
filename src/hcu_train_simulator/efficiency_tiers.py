# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Fixed shape and message-size corrections for analytical estimates.

The ratios approximate the median measured performance relative to the large
shape plateau in the bundled BW1000/BW1100 profiles and bandwidth tables.
"""


def gemm_shape_factor(m, n, k):
    """Return a fixed correction based on the smallest GEMM dimension."""
    smallest = min(m, n, k)
    if smallest <= 512:
        return 0.15
    if smallest <= 1024:
        return 0.40
    if smallest <= 2048:
        return 0.75
    return 1.0


def communication_size_factor(size_bytes):
    """Return a fixed correction for one transferred flow's payload."""
    if size_bytes <= 64 * 1024:
        return 0.02
    if size_bytes <= 1024 * 1024:
        return 0.15
    if size_bytes <= 16 * 1024 * 1024:
        return 0.55
    if size_bytes <= 64 * 1024 * 1024:
        return 0.85
    return 1.0
