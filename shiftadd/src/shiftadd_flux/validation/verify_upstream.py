"""Reload upstream-adapter packed artifacts and reproduce recorded output errors."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    a = p.parse_args()
    c = json.loads(Path(a.config).read_text())
    root = Path(c["output"])
    summary = json.loads((root / "summary.json").read_text())
    verified = 0
    with (
        np.load(Path(c["capture"]) / "activations.npz", allow_pickle=False) as inputs,
        safe_open(c["source"], framework="pt") as source,
    ):
        for st in summary["storage"]:
            module = st["module"]
            path = root / (module + ".npz")
            assert hashlib.sha256(path.read_bytes()).hexdigest() == st["artifact_sha256"]
            with np.load(path, allow_pickle=False) as f:
                packed = f["planes"]
                q, rows, groups, bytes_per_group = packed.shape
                signs = np.unpackbits(packed, axis=-1, bitorder="little").astype(np.float32) * 2 - 1
                d = f["superblock_scale"].astype(np.float32)
                if c["scale_format"] == "u8":
                    alpha = f["scale_codes"].astype(np.float32) * d[..., None, None]
                else:
                    payload = f["scale_exponent_nibbles"]
                    e = np.empty(payload.size * 2, np.uint8)
                    e[::2] = payload & 15
                    e[1::2] = payload >> 4
                    alpha = (
                        np.exp2(-e.reshape(rows, groups // 2, 2, q).astype(np.float32))
                        * d[..., None, None]
                    )
                weights = (
                    (signs * alpha.reshape(rows, groups, q).transpose(2, 0, 1)[..., None])
                    .sum(axis=0)
                    .reshape(rows, -1)
                )
                original = (
                    source.get_tensor(module + ".weight")[f["output_row_indices"]].float().numpy()
                )
            for row in (r for r in summary["results"] if r["module"] == module):
                key = sorted(k for k in inputs if k.startswith(module + ".call_"))[row["call"]]
                x = inputs[key]
                reference = x @ original.T
                error = float(np.linalg.norm(x @ weights.T - reference) / np.linalg.norm(reference))
                assert np.isclose(error, row["relative_output_l2"], rtol=1e-6, atol=1e-8)
                verified += 1
    with (root / "reload-verification.json").open("x") as f:
        json.dump(
            {
                "verified_output_comparisons": verified,
                "all_artifact_hashes_match": True,
                "all_metrics_reproduced": True,
            },
            f,
            indent=2,
        )
        f.write("\n")
    print(verified, "saved-artifact comparisons reproduced")


if __name__ == "__main__":
    main()
