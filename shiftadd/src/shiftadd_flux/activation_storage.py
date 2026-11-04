"""Byte layout reference for signed activation codes, not a GPU kernel.

A4 uses two's-complement nibbles, first value in the low nibble. An odd final
value is padded with a zero high nibble. Original shape is separate metadata.
"""

import math

import numpy as np


def pack_codes(codes, bits=4):
    codes = np.asarray(codes)
    if codes.dtype != np.int8 or bits not in (4, 8):
        raise ValueError("Expected int8 codes and bits=4 or 8")
    flat = codes.reshape(-1)
    if bits == 8:
        return flat.view(np.uint8).copy()
    if np.any(flat < -8) or np.any(flat > 7):
        raise ValueError("A4 code outside signed nibble range")
    nibbles = flat.astype(np.uint8) & 15
    output = np.zeros((flat.size + 1) // 2, np.uint8)
    output[:] = nibbles[::2]
    output[: flat.size // 2] |= nibbles[1::2] << 4
    return output


def unpack_codes(payload, shape, bits=4):
    payload = np.asarray(payload)
    if payload.dtype != np.uint8 or payload.ndim != 1 or bits not in (4, 8):
        raise ValueError("Expected flat uint8 payload and bits=4 or 8")
    if any(type(s) is not int or s < 0 for s in shape):
        raise ValueError("Invalid activation shape")
    count = math.prod(shape)
    expected = count if bits == 8 else (count + 1) // 2
    if payload.size != expected:
        raise ValueError("Payload size disagrees with activation shape")
    if bits == 8:
        return payload.view(np.int8).reshape(shape).copy()
    if count % 2 and payload[-1] >> 4:
        raise ValueError("Nonzero padding nibble")
    nibbles = np.empty(payload.size * 2, np.uint8)
    nibbles[::2] = payload & 15
    nibbles[1::2] = payload >> 4
    signed = nibbles[:count].astype(np.int8)
    signed[signed >= 8] -= 16
    return signed.reshape(shape)
