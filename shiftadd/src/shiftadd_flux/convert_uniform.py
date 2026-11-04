"""Local conventional control bundle from the same pinned transformer source."""

import argparse
import hashlib
import json
import time
import traceback
from pathlib import Path

import numpy as np
from safetensors import safe_open
from safetensors.torch import save_file

from .conventional import decode_uniform, encode_uniform


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    c = json.loads(Path(parser.parse_args().config).read_text())
    out = Path(c["output"])
    out.mkdir(parents=True, exist_ok=False)

    def save(name, value):
        (out / name).write_text(json.dumps(value, indent=2) + "\n")

    save("config.json", c)
    save("status.json", {"status": "running"})
    try:
        rows = []
        unchanged = {}
        with safe_open(c["source"], framework="pt") as source:
            for name in source.keys():
                tensor = source.get_tensor(name)
                if not (name.endswith(".weight") and tensor.ndim == 2):
                    unchanged[name] = tensor
                    continue
                started = time.perf_counter()
                w = tensor.float().numpy()
                payload, scales = encode_uniform(w, c["bits"], c["group_size"])
                path = out / (name + ".npz")
                np.savez(path, codes=payload, scales=scales, shape=np.array(w.shape))
                with np.load(path, allow_pickle=False) as f:
                    restored = decode_uniform(
                        f["codes"], f["scales"], f["shape"].tolist(), c["bits"]
                    )
                if restored.shape != w.shape or not np.isfinite(restored).all():
                    raise ValueError("Invalid reconstruction")
                row = {
                    "tensor": name,
                    "shape": list(w.shape),
                    "bits": c["bits"],
                    "bf16_bytes": w.size * 2,
                    "packed_payload_bytes": payload.nbytes + scales.nbytes,
                    "artifact_bytes": path.stat().st_size,
                    "artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "relative_l2": float(np.linalg.norm(w - restored) / np.linalg.norm(w)),
                    "conversion_and_verification_s": time.perf_counter() - started,
                }
                rows.append(row)
                save("summary.json", {"format": "symmetric_uniform", "results": rows})
                print(name, flush=True)
        unchanged_path = out / "unchanged.safetensors"
        save_file(unchanged, str(unchanged_path))
        source_bytes = Path(c["source"]).stat().st_size
        bundle_bytes = sum(r["artifact_bytes"] for r in rows) + unchanged_path.stat().st_size
        save(
            "summary.json",
            {
                "format": "symmetric_uniform",
                "bits": c["bits"],
                "group_size": c["group_size"],
                "results": rows,
                "unchanged_tensors": list(unchanged),
                "source_bytes": source_bytes,
                "unchanged_bytes": unchanged_path.stat().st_size,
                "unchanged_sha256": hashlib.sha256(unchanged_path.read_bytes()).hexdigest(),
                "replacement_bundle_weight_bytes": bundle_bytes,
                "storage_reduction": 1 - bundle_bytes / source_bytes,
                "scope": "weight file storage only; no compressed runtime or quality claim",
            },
        )
        save("status.json", {"status": "completed"})
    except BaseException:
        save("status.json", {"status": "failed", "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
