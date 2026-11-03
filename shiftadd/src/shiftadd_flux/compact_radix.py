"""Lossless exponent metadata compression for verified radix bundles only."""

import argparse
import hashlib
import json
import shutil
import traceback
from pathlib import Path

import numpy as np


def expand_exponents(base, planes):
    if base.dtype != np.int8 or base.ndim != 2 or type(planes) is not int or not 1 <= planes <= 8:
        raise ValueError("Invalid radix base layout")
    active = base != -128
    if np.any(base[active].astype(np.int16) + planes - 1 > 127):
        raise ValueError("Exponent overflow")
    return np.stack(
        [np.where(active, base.astype(np.int16) + p, -128).astype(np.int8) for p in range(planes)]
    )


def compact_exponents(exponents):
    if exponents.dtype != np.int8 or exponents.ndim != 3:
        raise ValueError("Invalid exponent layout")
    base = exponents[0].copy()
    if not np.array_equal(expand_exponents(base, len(exponents)), exponents):
        raise ValueError("Exponents do not follow the signed-radix relationship")
    return base


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    src = Path(a.source)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)

    def save(n, v):
        (out / n).write_text(json.dumps(v, indent=2) + "\n")

    save(
        "config.json",
        {
            "source_bundle": str(src),
            "output": str(out),
            "change": "lossless shared radix exponent metadata",
        },
    )
    save("status.json", {"status": "running"})
    try:
        original = json.loads((src / "summary.json").read_text())
        rows = []
        if json.loads((src / "status.json").read_text())["status"] != "completed":
            raise ValueError("Incomplete source")
        for row in original["results"]:
            name = row["tensor"] + f".p{row['planes']}.npz"
            old = src / name
            if hashlib.sha256(old.read_bytes()).hexdigest() != row["artifact_sha256"]:
                raise ValueError("Source hash mismatch")
            with np.load(old, allow_pickle=False) as f:
                packed, exponents = f["planes"], f["exponents"]
                base = compact_exponents(exponents)
            path = out / name
            np.savez(path, planes=packed, base_exponents=base)
            with np.load(path, allow_pickle=False) as f:
                if not np.array_equal(f["planes"], packed) or not np.array_equal(
                    expand_exponents(f["base_exponents"], row["planes"]), exponents
                ):
                    raise ValueError("Lossless round-trip failed")
            rows.append(
                dict(
                    row,
                    representation="compact_radix",
                    packed_payload_bytes=packed.nbytes + base.nbytes,
                    artifact_bytes=path.stat().st_size,
                    artifact_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    source_artifact_sha256=row["artifact_sha256"],
                )
            )
        shutil.copy2(src / "unchanged.safetensors", out / "unchanged.safetensors")
        size = (
            sum(r["artifact_bytes"] for r in rows) + (out / "unchanged.safetensors").stat().st_size
        )
        save(
            "summary.json",
            {
                "results": rows,
                "replacement_bundle_weight_bytes": size,
                "lossless_roundtrip": True,
                "scope": "storage metadata only; identical decoded weights",
            },
        )
        save("status.json", {"status": "completed"})
    except BaseException:
        save("status.json", {"status": "failed", "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
