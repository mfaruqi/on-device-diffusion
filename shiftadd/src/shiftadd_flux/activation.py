"""Numerical references only; these routines are not performance kernels."""

from math import gcd

import numpy as np


def quantize_per_token(x, bits=8):
    x = np.asarray(x, dtype=np.float32)
    if bits not in (4, 8) or not np.isfinite(x).all():
        raise ValueError("Require finite activations and bits=4 or 8")
    bound = 2 ** (bits - 1) - 1
    maximum = np.max(np.abs(x), axis=-1, keepdims=True)
    scale = np.where(maximum == 0, 1.0, maximum / bound).astype(np.float32)
    return np.rint(x / scale).clip(-bound, bound).astype(np.int8), scale


def quantize_grouped(x, bits=4, group_size=128):
    """Reference quantization with one FP32 scale per token/channel group.

    Returns grouped integers and scales; no packed activation storage is created.
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or group_size <= 0 or x.shape[-1] % group_size:
        raise ValueError("Expected token matrix with channels divisible by group size")
    return quantize_per_token(x.reshape(x.shape[0], -1, group_size), bits)


def signed_plane_matmul(integer_x, activation_scale, packed, exponents):
    """Sum signed int32 activations per group/plane, then rescale into FP32."""
    integer_x = np.asarray(integer_x)
    if integer_x.dtype != np.int8 or integer_x.ndim != 2:
        raise ValueError("Expected token-by-channel int8 matrix")
    bits = np.unpackbits(packed, axis=-1, bitorder="little")
    planes, rows, groups, group_size = bits.shape
    if integer_x.shape[1] != groups * group_size:
        raise ValueError("Activation/weight channels disagree")
    result = np.zeros((integer_x.shape[0], rows), np.float32)
    x = integer_x.reshape(-1, groups, group_size).astype(np.int32)
    for plane in range(planes):
        for group in range(groups):
            sign = bits[plane, :, group].astype(np.int32) * 2 - 1
            integer_sum = x[:, group] @ sign.T
            e = exponents[plane, :, group]
            scale = np.where(e == -128, 0.0, np.exp2(e.astype(np.float32)))
            result += integer_sum * scale
    return result * activation_scale


def grouped_signed_plane_matmul(integer_x, activation_scale, packed, exponents):
    """Integer signed sums over intersecting activation/weight groups.

    Inputs are token x activation-group x channels; scales have a final size 1.
    Weight planes are unpacked to signs, never reconstructed into dense weights.
    This CPU numerical reference is not a performance implementation.
    """
    x = np.asarray(integer_x)
    scales = np.asarray(activation_scale, dtype=np.float32)
    if x.dtype != np.int8 or x.ndim != 3:
        raise ValueError("Expected grouped int8 activations")
    if scales.shape != x.shape[:2] + (1,) or not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError("Expected positive finite scale per token/group")
    if packed.dtype != np.uint8 or packed.ndim != 4 or exponents.shape != packed.shape[:3]:
        raise ValueError("Invalid packed weight shapes or dtype")
    signs = np.unpackbits(packed, axis=-1, bitorder="little").astype(np.int32) * 2 - 1
    planes, rows, groups, weight_group = signs.shape
    activation_group = x.shape[-1]
    channels = x.shape[1] * activation_group
    if channels != groups * weight_group or channels == 0:
        raise ValueError("Activation/weight channels disagree")
    width = gcd(activation_group, weight_group)
    if width * 128 > np.iinfo(np.int32).max:
        raise ValueError("Integer accumulation could overflow int32")
    flat = x.reshape(x.shape[0], channels).astype(np.int32)
    result = np.zeros((x.shape[0], rows), np.float32)
    for start in range(0, channels, width):
        wg, offset = divmod(start, weight_group)
        ag = start // activation_group
        for plane in range(planes):
            total = flat[:, start : start + width] @ signs[plane, :, wg, offset : offset + width].T
            exponent = exponents[plane, :, wg]
            factor = np.where(exponent == -128, 0.0, np.exp2(exponent.astype(np.float32)))
            result += total.astype(np.float32) * factor * scales[:, ag]
    return result
