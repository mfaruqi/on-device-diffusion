"""Numerical controls for one complete upstream-fitted projection."""

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import numpy as np
from safetensors import safe_open

from .conventional import decode_uniform
from .format import decode


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--bundles", default="shiftadd/artifacts")
    a = p.parse_args()
    c = json.loads(Path(a.config).read_text())
    assert len(c["modules"]) == 1
    module = c["modules"][0]
    controls = {}
    evidence = {}
    for label, folder in [
        ("radix_p4", "transformer-radix-p4-001"),
        ("uniform_int4", "transformer-uniform-int4-001"),
    ]:
        root = Path(a.bundles) / folder
        s = json.loads((root / "summary.json").read_text())
        row = next(r for r in s["results"] if r["tensor"] == module + ".weight")
        path = root / (module + ".weight" + (".p4.npz" if label == "radix_p4" else ".npz"))
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["artifact_sha256"]
        with np.load(path, allow_pickle=False) as f:
            controls[label] = (
                decode(f["planes"], f["exponents"])
                if label == "radix_p4"
                else decode_uniform(f["codes"], f["scales"], f["shape"].tolist(), 4)
            )
        evidence[label] = {
            "artifact": str(path),
            "sha256": row["artifact_sha256"],
            "payload_bytes": row["packed_payload_bytes"],
        }
    rows = []
    with (
        safe_open(c["source"], framework="pt") as source,
        np.load(Path(c["capture"]) / "activations.npz", allow_pickle=False) as inputs,
    ):
        w = source.get_tensor(module + ".weight").float().numpy()
        for call, key in enumerate(sorted(k for k in inputs if k.startswith(module + ".call_"))):
            x = inputs[key]
            ref = x @ w.T
            for label, wq in controls.items():
                rows.append(
                    {
                        "control": label,
                        "call": call,
                        "split": "fit" if call < c["fit_calls"] else "calibration_validation",
                        "relative_output_l2": float(
                            np.linalg.norm(x @ wq.T - ref) / np.linalg.norm(ref)
                        ),
                    }
                )
    agg = {
        label: {
            split: statistics.mean(
                r["relative_output_l2"]
                for r in rows
                if r["control"] == label and r["split"] == split
            )
            for split in ("fit", "calibration_validation")
        }
        for label in controls
    }
    with Path(a.output).open("x") as f:
        json.dump(
            {
                "config": c,
                "module": module,
                "evidence": evidence,
                "aggregate": agg,
                "results": rows,
            },
            f,
            indent=2,
        )
        f.write("\n")
    print(agg)


if __name__ == "__main__":
    main()
