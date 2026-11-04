"""Experimental exact code assignment over a bounded exponent-template search."""

import numpy as np

# Includes the radix candidate, plus repeated/nonuniform exponent spacings.
TEMPLATES = (
    (0, 1, 2, 3),
    (0, 0, 1, 3),
    (0, 1, 1, 3),
    (0, 1, 2, 4),
    (0, 0, 2, 3),
    (0, 1, 3, 4),
)


def encode_nonuniform(weight, group_size=128):
    w = np.asarray(weight, np.float32)
    if w.ndim != 2 or w.shape[1] % group_size or group_size % 8 or not np.isfinite(w).all():
        raise ValueError("Expected finite byte-group-aligned matrix")
    target = w.reshape(w.shape[0], -1, group_size)
    maximum = np.max(np.abs(target), axis=-1)
    best_error = np.full(maximum.shape, np.inf)
    best_code = np.zeros(target.shape, np.uint8)
    best_exponents = np.zeros((4,) + maximum.shape, np.int8)
    for template in TEMPLATES:
        limit = sum(2**k for k in template)
        center = np.floor(np.log2(np.maximum(maximum / limit, np.finfo(np.float32).tiny)))
        levels = np.array(
            [
                sum(
                    (1 if (code >> plane) & 1 else -1) * 2**power
                    for plane, power in enumerate(template)
                )
                for code in range(16)
            ],
            np.float32,
        )
        for offset in (-2, -1, 0, 1, 2):
            exponent = np.clip(center + offset, -126, 119).astype(np.int8)
            unit = np.exp2(exponent.astype(np.float32))
            error = np.full(target.shape, np.inf, np.float32)
            code = np.zeros(target.shape, np.uint8)
            for index, level in enumerate(levels):
                delta = np.square(target - level * unit[..., None])
                better = delta < error
                code = np.where(better, index, code).astype(np.uint8)
                error = np.minimum(delta, error)
            group_error = error.sum(axis=-1)
            update = group_error < best_error
            best_error = np.where(update, group_error, best_error)
            best_code = np.where(update[..., None], code, best_code)
            for plane, power in enumerate(template):
                candidate = np.where(maximum == 0, -128, exponent.astype(np.int16) + power).astype(
                    np.int8
                )
                best_exponents[plane] = np.where(update, candidate, best_exponents[plane])
    packed = np.stack(
        [np.packbits((best_code >> plane) & 1, axis=-1, bitorder="little") for plane in range(4)]
    )
    return packed, best_exponents
