"""Packed residual sign planes and integer power-of-two exponents."""

import numpy as np


def encode(weight, planes=4, group_size=128):
    """Fit independently per row/group; zero groups use exponent -128 sentinel."""
    w = np.asarray(weight, dtype=np.float32)
    if w.ndim != 2 or w.shape[1] % group_size:
        raise ValueError("Expected matrix with columns divisible by group_size")
    if not 1 <= planes <= 8 or group_size % 8:
        raise ValueError("Require 1..8 planes and byte-aligned groups")
    if not np.isfinite(w).all():
        raise ValueError("Nonfinite weights")
    residual = w.reshape(w.shape[0], -1, group_size).copy()
    packed, exponents = [], []
    for _ in range(planes):
        sign = residual >= 0
        mean = np.abs(residual).mean(axis=-1)
        # Choose the better of adjacent powers of two by squared reconstruction error.
        lower = np.floor(np.log2(np.maximum(mean, np.finfo(np.float32).tiny)))
        low = np.exp2(lower)
        high = low * 2
        scale = np.where(np.abs(mean - low) <= np.abs(mean - high), low, high)
        exponent = np.rint(np.log2(scale)).clip(-126, 126).astype(np.int8)
        exponent[mean == 0] = -128
        scale = np.where(exponent == -128, 0, np.exp2(exponent.astype(np.float32)))
        residual -= np.where(sign, 1.0, -1.0) * scale[..., None]
        packed.append(np.packbits(sign, axis=-1, bitorder="little"))
        exponents.append(exponent)
    return np.stack(packed), np.stack(exponents)


def decode(packed, exponents):
    packed, exponents = np.asarray(packed), np.asarray(exponents)
    if packed.dtype != np.uint8 or exponents.dtype != np.int8:
        raise ValueError("Expected uint8 planes and int8 exponents")
    if packed.shape[:-1] != exponents.shape:
        raise ValueError("Plane/exponent shapes disagree")
    signs = np.unpackbits(packed, axis=-1, bitorder="little").astype(np.float32) * 2 - 1
    scales = np.where(exponents == -128, 0, np.exp2(exponents.astype(np.float32)))
    result = (signs * scales[..., None]).sum(axis=0)
    return result.reshape(result.shape[0], -1)
