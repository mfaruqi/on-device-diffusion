"""Config-driven, local safetensors-to-packed-planes conversion."""

import argparse
import hashlib
import json
import time
import traceback
from pathlib import Path

import numpy as np
from safetensors import safe_open

from shiftadd_flux.format import decode, encode
from shiftadd_flux.radix import encode_radix


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    out = Path(config["output"])
    out.mkdir(parents=True, exist_ok=False)
    write(out / "config.json", config)
    write(out / "status.json", {"status": "running"})
    try:
        rows = []
        with safe_open(config["source"], framework="pt", device="cpu") as source:
            names = config["tensors"] or [
                k
                for k in source.keys()
                if k.endswith(".weight") and len(source.get_slice(k).get_shape()) == 2
            ]
            for name in names:
                weight = source.get_tensor(name).float().numpy()
                for planes in config["planes"]:
                    start = time.perf_counter()
                    method = config.get("method", "greedy")
                    if method not in ("greedy", "radix"):
                        raise ValueError("Unknown fitting method")
                    fitter = encode if method == "greedy" else encode_radix
                    packed, exponents = fitter(weight, planes, config["group_size"])
                    path = out / (name + f".p{planes}.npz")
                    np.savez(path, planes=packed, exponents=exponents)
                    with np.load(path, allow_pickle=False) as saved:
                        restored = decode(saved["planes"], saved["exponents"])
                    assert restored.shape == weight.shape and np.isfinite(restored).all()
                    rows.append(
                        {
                            "tensor": name,
                            "shape": list(weight.shape),
                            "planes": planes,
                            "bf16_bytes": weight.size * 2,
                            "packed_payload_bytes": packed.nbytes + exponents.nbytes,
                            "artifact_bytes": path.stat().st_size,
                            "artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "relative_l2": float(
                                np.linalg.norm(weight - restored) / np.linalg.norm(weight)
                            ),
                            "max_absolute_error": float(np.max(np.abs(weight - restored))),
                            "conversion_and_verification_s": time.perf_counter() - start,
                        }
                    )
                    write(
                        out / "summary.json",
                        {
                            "results": rows,
                            "scope": "weight reconstruction; no inference performance claim",
                        },
                    )
                    print(json.dumps(rows[-1]), flush=True)
        write(out / "status.json", {"status": "completed"})
    except BaseException:
        write(
            out / "status.json",
            {"status": "failed", "traceback": traceback.format_exc()},
        )
        raise


if __name__ == "__main__":
    main()
