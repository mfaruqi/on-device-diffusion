"""Portable integer-activation arithmetic check on a saved real-weight slice."""

import argparse
import json
import platform
from pathlib import Path

import numpy as np

from shiftadd_flux.activation import quantize_per_token, signed_plane_matmul
from shiftadd_flux.format import decode

p = argparse.ArgumentParser()
p.add_argument("--input", required=True)
p.add_argument("--output", required=True)
a = p.parse_args()
f = np.load(a.input, allow_pickle=False)
rows = []
for bits in [4, 8]:
    q, s = quantize_per_token(f["activations"], bits)
    result = signed_plane_matmul(q, s, f["planes"], f["exponents"])
    expected = (q.astype(np.float32) * s) @ decode(f["planes"], f["exponents"]).T
    np.testing.assert_allclose(result, expected, atol=1e-4, rtol=1e-4)
    rows.append(
        {
            "activation_bits": bits,
            "max_absolute_arithmetic_error": float(np.max(np.abs(result - expected))),
        }
    )
Path(a.output).write_text(
    json.dumps(
        {
            "host": platform.node(),
            "machine": platform.machine(),
            "numpy": np.__version__,
            "scope": "numerical CPU reference, not accelerated throughput or energy",
            "results": rows,
        },
        indent=2,
    )
    + "\n"
)
