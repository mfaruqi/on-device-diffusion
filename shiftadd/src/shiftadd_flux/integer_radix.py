"""Exact integer-code bridges for restricted signed radix, not GPU kernels."""

import numpy as np

from .compact_radix import compact_exponents


def to_integer_codes(packed, exponents, target_bits=8):
    """Return signed codes, base exponents and integer affine offset.

    INT8: w = 2**base * code, for up to seven radix planes.
    INT4: w = 2**base * (2*code + 1), for exactly four planes.
    Zero groups retain the -128 scale sentinel. No rounding/refitting occurs.
    """
    if packed.dtype != np.uint8 or packed.ndim != 4 or packed.shape[:-1] != exponents.shape:
        raise ValueError("Invalid packed radix shape")
    base = compact_exponents(exponents)
    planes = packed.shape[0]
    if target_bits == 8 and not 1 <= planes <= 7:
        raise ValueError("Signed INT8 cannot hold these radix levels")
    if target_bits == 4 and planes != 4:
        raise ValueError("INT4 affine bridge requires four planes")
    if target_bits not in (4, 8):
        raise ValueError("Target must be INT4 or INT8")
    bits = np.unpackbits(packed, axis=-1, bitorder="little").astype(np.int16)
    unsigned = sum(bits[p] * (1 << p) for p in range(planes))
    if target_bits == 8:
        return (2 * unsigned - ((1 << planes) - 1)).astype(np.int8), base, 1, 0
    return (unsigned - 8).astype(np.int8), base, 2, 1


def grouped_integer_matmul(x, activation_scales, codes, base, multiplier=1, offset=0):
    """INT32 dot products plus affine group-sum correction; FP32 rescaling."""
    from math import gcd

    if x.dtype != np.int8 or x.ndim != 3 or codes.dtype != np.int8 or codes.ndim != 3:
        raise ValueError("Expected grouped signed integer codes")
    tokens, agroups, ag = x.shape
    rows, wgroups, wg = codes.shape
    if (
        agroups * ag != wgroups * wg
        or base.shape != (rows, wgroups)
        or activation_scales.shape != (tokens, agroups, 1)
    ):
        raise ValueError("Group layout mismatch")
    if (multiplier, offset) not in ((1, 0), (2, 1)):
        raise ValueError("Unknown radix mapping")
    width = gcd(ag, wg)
    if width * 128 * (128 * multiplier + offset) > np.iinfo(np.int32).max:
        raise ValueError("Unsafe accumulation width")
    flat = x.reshape(tokens, -1).astype(np.int32)
    result = np.zeros((tokens, rows), np.float32)
    for start in range(0, flat.shape[1], width):
        group, within = divmod(start, wg)
        activation_group = start // ag
        part = flat[:, start : start + width]
        dot = part @ codes[:, group, within : within + width].astype(np.int32).T
        total = dot * multiplier + offset * part.sum(axis=-1, keepdims=True, dtype=np.int32)
        e = base[:, group]
        scale = np.where(e == -128, 0.0, np.exp2(e.astype(np.float32)))
        result += total.astype(np.float32) * scale * activation_scales[:, activation_group]
    return result
