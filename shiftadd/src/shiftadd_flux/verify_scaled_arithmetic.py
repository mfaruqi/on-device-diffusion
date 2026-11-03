"""Verify integer sign-plane arithmetic against decoded upstream artifacts."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .activation import quantize_grouped
from .scaled_plane_arithmetic import grouped_scaled_plane_matmul
from .upstream_loader import decode_rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--artifacts", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--group-size", type=int, default=32)
    args = p.parse_args()
    root = Path(args.artifacts)
    config = json.loads((root / "config.json").read_text())
    assert json.loads((root / "status.json").read_text())["status"] == "completed"
    summary = json.loads((root / "summary.json").read_text())
    checks = []
    with np.load(Path(config["capture"]) / "activations.npz", allow_pickle=False) as inputs:
        for entry in summary["storage"]:
            module = entry["module"]
            path = root / (module + ".npz")
            assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["artifact_sha256"]
            with np.load(path, allow_pickle=False) as data:
                packed = data["planes"]
                planes, rows, groups, _ = packed.shape
                d = data["superblock_scale"].astype(np.float32)
                if config["scale_format"] == "u8":
                    alpha = data["scale_codes"].astype(np.float32) * d[..., None, None]
                elif config["scale_format"] == "pot":
                    payload = data["scale_exponent_nibbles"]
                    e = np.stack((payload & 15, payload >> 4), axis=1).reshape(
                        rows, groups // 2, 2, planes
                    )
                    alpha = np.exp2(-e.astype(np.float32)) * d[..., None, None]
                else:
                    raise ValueError("Unknown scale format")
                factors = alpha.reshape(rows, groups, planes).transpose(2, 0, 1)
                weights = decode_rows(data, config["scale_format"], 0, rows)
                for key in sorted(k for k in inputs if k.startswith(module + ".call_")):
                    x = inputs[key]
                    for bits in (4, 8):
                        q, scale = quantize_grouped(x, bits, args.group_size)
                        actual = grouped_scaled_plane_matmul(q, scale, packed, factors)
                        expected = (q.astype(np.float32) * scale).reshape(x.shape) @ weights.T
                        error = float(
                            np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-20)
                        )
                        if not np.isfinite(actual).all() or error > 1e-5:
                            raise ValueError("Integer arithmetic differs from decoded reference")
                        checks.append(
                            {
                                "module": module,
                                "capture": key,
                                "activation_bits": bits,
                                "relative_l2_disagreement": error,
                            }
                        )
    result = {
        "status": "passed",
        "group_size": args.group_size,
        "scale_format": config["scale_format"],
        "scope": "CPU arithmetic equivalence only; not quality or performance evidence",
        "max_relative_l2_disagreement": max(r["relative_l2_disagreement"] for r in checks),
        "artifact_hashes": {r["module"]: r["artifact_sha256"] for r in summary["storage"]},
        "checks": checks,
    }
    with Path(args.output).open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(len(checks), "checks;", result["max_relative_l2_disagreement"])


if __name__ == "__main__":
    main()
