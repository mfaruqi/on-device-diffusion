"""Coordinate refinement of power-of-two residual planes."""

import numpy as np

from shiftadd_flux.format import encode


def refine(weight, planes=4, group_size=128, sweeps=3):
    packed, exponent = encode(weight, planes, group_size)
    signs = np.unpackbits(packed, axis=-1, bitorder="little").astype(np.float32) * 2 - 1
    scales = np.where(exponent == -128, 0, np.exp2(exponent.astype(np.float32)))
    target = np.asarray(weight, np.float32).reshape(signs.shape[1:])
    approximation = (signs * scales[..., None]).sum(axis=0)
    for _ in range(sweeps):
        for i in range(planes):
            residual = target - approximation + signs[i] * scales[i, ..., None]
            new_sign = np.where(residual >= 0, 1.0, -1.0).astype(np.float32)
            mean = np.abs(residual).mean(axis=-1)
            low = np.floor(np.log2(np.maximum(mean, np.finfo(np.float32).tiny)))
            a = np.exp2(low)
            b = a * 2
            new_scale = np.where(np.abs(mean - a) <= np.abs(mean - b), a, b)
            new_scale = np.where(mean == 0, 0, new_scale)
            approximation += new_sign * new_scale[..., None] - signs[i] * scales[i, ..., None]
            signs[i] = new_sign
            scales[i] = new_scale
    exponent = np.where(
        scales == 0,
        -128,
        np.rint(np.log2(np.maximum(scales, np.finfo(np.float32).tiny))),
    ).astype(np.int8)
    return np.packbits(signs > 0, axis=-1, bitorder="little"), exponent
