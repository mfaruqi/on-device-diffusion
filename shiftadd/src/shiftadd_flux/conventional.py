"""Bounded symmetric uniform weight-quantization reference."""

import numpy as np

from .activation_storage import pack_codes, unpack_codes


def encode_uniform(weight, bits=4, group_size=128):
    w = np.asarray(weight, dtype=np.float32)
    if (
        bits not in (4, 8)
        or w.ndim != 2
        or group_size <= 0
        or w.shape[1] % group_size
        or not np.isfinite(w).all()
    ):
        raise ValueError("Require finite divisible matrix and 4/8 bits")
    grouped = w.reshape(w.shape[0], -1, group_size)
    bound = 2 ** (bits - 1) - 1
    maximum = np.abs(grouped).max(axis=-1, keepdims=True)
    best_error = np.full(maximum.shape, np.inf)
    best_q = np.zeros_like(grouped, dtype=np.int8)
    best_scale = np.ones_like(maximum, dtype=np.float32)
    # Fixed weight-only clipping candidates. No validation inputs select scales.
    for fraction in (1.0, 0.95, 0.9, 0.85, 0.8):
        scale = np.where(maximum == 0, 1.0, maximum * np.float32(fraction / bound)).astype(
            np.float32
        )
        q = np.rint(grouped / scale).clip(-bound, bound).astype(np.int8)
        error = np.square(grouped - q.astype(np.float32) * scale).sum(axis=-1, keepdims=True)
        better = error < best_error
        best_q = np.where(better, q, best_q)
        best_scale = np.where(better, scale, best_scale)
        best_error = np.minimum(best_error, error)
    return pack_codes(best_q, bits), best_scale


def decode_uniform(payload, scales, shape, bits=4):
    q = unpack_codes(payload, tuple(shape), bits)
    if (
        scales.dtype != np.float32
        or scales.ndim != 3
        or scales.shape[0] != shape[0]
        or scales.shape[-1] != 1
    ):
        raise ValueError("Invalid scale layout")
    return (q.reshape(scales.shape[0], scales.shape[1], -1).astype(np.float32) * scales).reshape(
        shape
    )
