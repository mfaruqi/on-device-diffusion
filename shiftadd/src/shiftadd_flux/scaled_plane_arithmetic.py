"""CPU correctness reference for sign planes with arbitrary nonnegative scales."""

from math import gcd

import numpy as np


def grouped_scaled_plane_matmul(integer_x, activation_scale, packed, plane_scales):
    """INT32 signed sums, then FP32 scaling; no dense weight reconstruction.

    plane_scales has shape (planes, output rows, weight groups). This accepts
    upstream BCQ u8/FP16 and power-of-two scales, but is not a GPU kernel.
    """
    x = np.asarray(integer_x)
    a = np.asarray(activation_scale, dtype=np.float32)
    p = np.asarray(packed)
    w = np.asarray(plane_scales, dtype=np.float32)
    if x.dtype != np.int8 or x.ndim != 3:
        raise ValueError("Expected grouped INT8 activation codes")
    if a.shape != x.shape[:2] + (1,) or not np.isfinite(a).all() or np.any(a <= 0):
        raise ValueError("Expected positive finite activation scales")
    if p.dtype != np.uint8 or p.ndim != 4 or w.shape != p.shape[:3]:
        raise ValueError("Invalid plane layout")
    if not np.isfinite(w).all() or np.any(w < 0) or min(p.shape) <= 0:
        raise ValueError("Expected nonnegative finite plane scales and nonempty planes")
    planes, rows, groups, packed_width = p.shape
    width = packed_width * 8
    channels = x.shape[1] * x.shape[2]
    if not channels or channels != groups * width:
        raise ValueError("Activation and weight channel counts disagree")
    step = gcd(width, x.shape[2])
    if step * 128 > np.iinfo(np.int32).max:
        raise ValueError("Integer accumulation could overflow")
    signs = np.unpackbits(p, axis=-1, bitorder="little").astype(np.int32) * 2 - 1
    flat = x.reshape(x.shape[0], channels).astype(np.int32)
    result = np.zeros((x.shape[0], rows), np.float32)
    for start in range(0, channels, step):
        group, offset = divmod(start, width)
        activation_group = start // x.shape[2]
        for plane in range(planes):
            total = flat[:, start : start + step] @ signs[plane, :, group, offset : offset + step].T
            result += total.astype(np.float32) * w[plane, :, group] * a[:, activation_group]
    return result
