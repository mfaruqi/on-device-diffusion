"""Verify exact radix bridges on saved real calibration inputs."""

import argparse
import json
from pathlib import Path

import numpy as np

from .activation import quantize_grouped
from .activation_storage import pack_codes, unpack_codes
from .format import decode
from .integer_radix import grouped_integer_matmul, to_integer_codes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    c = json.loads(Path(a.config).read_text())
    out = Path(a.output)
    if out.exists():
        raise FileExistsError(out)
    rows = []
    with np.load(Path(c["capture"]) / "activations.npz", allow_pickle=False) as inputs:
        for module in c["modules"]:
            for method in ("weight_only", "diagonal_activation"):
                with np.load(
                    Path(c["output"]) / (module + "." + method + ".npz"),
                    allow_pickle=False,
                ) as artifact:
                    planes, exponents = artifact["planes"], artifact["exponents"]
                weights = decode(planes, exponents)
                for target in (4, 8):
                    codes, base, multiplier, offset = to_integer_codes(planes, exponents, target)
                    packed = pack_codes(codes, target)
                    codes = unpack_codes(packed, codes.shape, target)
                    scale = np.where(base == -128, 0.0, np.exp2(base.astype(np.float32)))
                    reconstructed = (
                        (codes.astype(np.float32) * multiplier + offset) * scale[..., None]
                    ).reshape(weights.shape)
                    if not np.array_equal(reconstructed, weights):
                        raise ValueError("Integer mapping changed weights")
                    for key in sorted(k for k in inputs if k.startswith(module + ".call_")):
                        x = inputs[key]
                        for abits in (4, 8):
                            for group in (32, 128, 3072):
                                q, s = quantize_grouped(x, abits, group)
                                result = grouped_integer_matmul(
                                    q, s, codes, base, multiplier, offset
                                )
                                reference = (q.astype(np.float32) * s).reshape(x.shape) @ weights.T
                                error = float(
                                    np.linalg.norm(result - reference) / np.linalg.norm(reference)
                                )
                                if not np.isfinite(error) or error > 1e-5:
                                    raise ValueError("Integer arithmetic mismatch")
                                rows.append(
                                    {
                                        "module": module,
                                        "fit": method,
                                        "call": key,
                                        "weight_code_bits": target,
                                        "activation_bits": abits,
                                        "activation_group_size": group,
                                        "relative_l2": error,
                                    }
                                )
    report = {
        "status": "passed",
        "scope": "exact integer representation and CPU arithmetic only; no GPU performance claim",
        "config": c,
        "comparisons": len(rows),
        "max_relative_l2": max(r["relative_l2"] for r in rows),
        "results": rows,
    }
    with out.open("x") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print(report["comparisons"], report["max_relative_l2"])


if __name__ == "__main__":
    main()
