"""Joint signed-radix planes with groupwise power-of-two unit search."""

import numpy as np


def encode_radix(weight, planes=4, group_size=128, importance=None):
    w = np.asarray(weight, np.float32)
    if w.ndim != 2 or w.shape[1] % group_size or group_size % 8 or not 1 <= planes <= 8:
        raise ValueError("Unsupported shape or format")
    if not np.isfinite(w).all():
        raise ValueError("Nonfinite weight")
    target = w.reshape(w.shape[0], -1, group_size)
    if importance is not None:
        importance = np.asarray(importance, np.float32)
        if (
            importance.shape != (w.shape[1],)
            or not np.isfinite(importance).all()
            or (importance < 0).any()
        ):
            raise ValueError("Importance must be a finite nonnegative input-channel vector")
        importance = importance.reshape(1, -1, group_size)
    maximum = np.max(np.abs(target), axis=-1)
    limit = 2**planes - 1
    center = np.floor(np.log2(np.maximum(maximum / limit, np.finfo(np.float32).tiny)))
    best = np.full(maximum.shape, np.inf)
    best_code = np.zeros(target.shape, np.uint8)
    best_exp = np.zeros(maximum.shape, np.int8)
    for offset in [-2, -1, 0, 1, 2]:
        exponent = np.clip(center + offset, -126, 119).astype(np.int8)
        unit = np.exp2(exponent.astype(np.float32))
        # Signed digit sums with powers 1,2,... span odd integers [-limit,limit].
        code = np.rint((target / unit[..., None] + limit) / 2).clip(0, limit).astype(np.uint8)
        reconstructed = (2 * code.astype(np.float32) - limit) * unit[..., None]
        squared = np.square(target - reconstructed)
        error = (squared if importance is None else squared * importance).sum(axis=-1)
        update = error < best
        best = np.where(update, error, best)
        best_code = np.where(update[..., None], code, best_code)
        best_exp = np.where(update, exponent, best_exp)
    packed = []
    exponents = []
    for plane in range(planes):
        packed.append(
            np.packbits(((best_code >> plane) & 1).astype(bool), axis=-1, bitorder="little")
        )
        exponents.append(
            np.where(maximum == 0, -128, best_exp.astype(np.int16) + plane).astype(np.int8)
        )
    return np.stack(packed), np.stack(exponents)
