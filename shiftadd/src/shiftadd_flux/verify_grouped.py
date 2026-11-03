"""Verify saved real-input arithmetic; no speed or quality benchmark."""

import argparse
import hashlib
import json
import traceback
from pathlib import Path

import numpy as np

from .activation import grouped_signed_plane_matmul, quantize_grouped
from .activation_storage import pack_codes, unpack_codes
from .format import decode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)

    def save(name, value):
        (out / name).write_text(json.dumps(value, indent=2) + "\n")

    save("config.json", config)
    save("status.json", {"status": "running"})
    try:
        root = Path(config["output"])
        if json.loads((root / "status.json").read_text())["status"] != "completed":
            raise ValueError("Fitting run is incomplete")
        capture = Path(config["capture"]) / "activations.npz"
        hashes = {str(capture): hashlib.sha256(capture.read_bytes()).hexdigest()}
        results = []
        with np.load(capture, allow_pickle=False) as inputs:
            for module in config["modules"]:
                for method in ("weight_only", "diagonal_activation"):
                    artifact = root / (module + "." + method + ".npz")
                    hashes[str(artifact)] = hashlib.sha256(artifact.read_bytes()).hexdigest()
                    with np.load(artifact, allow_pickle=False) as packed:
                        p, e = packed["planes"], packed["exponents"]
                    weights = decode(p, e)
                    for key in sorted(k for k in inputs if k.startswith(module + ".call_")):
                        x = inputs[key]
                        for bits in (4, 8):
                            q, scale = quantize_grouped(x, bits, config["activation_group_size"])
                            payload = pack_codes(q, bits)
                            restored = unpack_codes(payload, q.shape, bits)
                            if not np.array_equal(q, restored):
                                raise ValueError("Activation code packing changed values")
                            actual = grouped_signed_plane_matmul(restored, scale, p, e)
                            dense = (q.astype(np.float32) * scale).reshape(x.shape) @ weights.T
                            denom = float(np.linalg.norm(dense))
                            error = float(np.linalg.norm(actual - dense)) / max(denom, 1e-30)
                            if not np.isfinite(actual).all() or error > 1e-5:
                                raise ValueError(
                                    f"Arithmetic mismatch: {key}, {method}, A{bits}: {error}"
                                )
                            results.append(
                                {
                                    "module": module,
                                    "method": method,
                                    "input": key,
                                    "activation_bits": bits,
                                    "relative_l2_vs_dense": error,
                                    "packed_activation_bytes": payload.nbytes,
                                    "scale_bytes": scale.nbytes,
                                    "activation_elements": q.size,
                                    "bf16_activation_bytes": q.size * 2,
                                    "max_absolute_error": float(np.abs(actual - dense).max()),
                                }
                            )
        save(
            "summary.json",
            {
                "scope": "arithmetic equivalence only; all captured calibration calls",
                "comparisons": len(results),
                "relative_l2_threshold": 1e-5,
                "max_relative_l2": max(r["relative_l2_vs_dense"] for r in results),
                "artifact_sha256": hashes,
                "results": results,
            },
        )
        save("status.json", {"status": "completed"})
    except BaseException:
        save("status.json", {"status": "failed", "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
