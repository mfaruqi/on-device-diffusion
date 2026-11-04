"""Reconstruct saved planes into existing model parameters for quality tests."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from shiftadd_flux.format import decode


def apply_artifacts(transformer, artifact_dir, planes=4, names=None, chunk_rows=None):
    if chunk_rows is not None and (not isinstance(chunk_rows, int) or chunk_rows < 1):
        raise ValueError("chunk_rows must be a positive integer or None")
    root = Path(artifact_dir)
    if json.loads((root / "status.json").read_text())["status"] != "completed":
        raise ValueError("Conversion is not complete")
    summary = json.loads((root / "summary.json").read_text())["results"]
    rows = [
        r
        for r in summary
        if (planes is None or r["planes"] == planes) and (names is None or r["tensor"] in names)
    ]
    if not rows or (names is not None and {r["tensor"] for r in rows} != set(names)):
        raise ValueError("Requested tensor set is absent")
    params = dict(transformer.named_parameters())
    # Check all shapes and file hashes before mutating the model.
    for row in rows:
        p = root / (row["tensor"] + f".p{row['planes']}.npz")
        if hashlib.sha256(p.read_bytes()).hexdigest() != row["artifact_sha256"]:
            raise ValueError(f"Artifact hash mismatch: {p}")
        if tuple(params[row["tensor"]].shape) != tuple(row["shape"]):
            raise ValueError("Model shape mismatch")
    with torch.no_grad():
        for row in rows:
            param = params[row["tensor"]]
            with np.load(
                root / (row["tensor"] + f".p{row['planes']}.npz"), allow_pickle=False
            ) as f:
                packed = f["planes"]
                compact = row.get("representation") == "compact_radix"
                if compact:
                    from shiftadd_flux.compact_radix import expand_exponents

                    base = f["base_exponents"]
                else:
                    exponents = f["exponents"]
                step = chunk_rows or param.shape[0]
                for start in range(0, param.shape[0], step):
                    stop = min(start + step, param.shape[0])
                    chunk_exponents = (
                        expand_exponents(base[start:stop], row["planes"])
                        if compact
                        else exponents[:, start:stop]
                    )
                    weight = torch.from_numpy(decode(packed[:, start:stop], chunk_exponents))
                    param[start:stop].copy_(weight.to(device=param.device, dtype=param.dtype))
    return {
        "replaced_tensors": [r["tensor"] for r in rows],
        "runtime_representation": "reconstructed dense parameters",
        "decode_chunk_rows": chunk_rows,
        "packed_artifact_bytes": sum(r["artifact_bytes"] for r in rows),
    }


def apply_uniform_artifacts(transformer, artifact_dir, names=None, chunk_rows=128):
    """Load the explicitly labelled conventional control into dense parameters."""
    from shiftadd_flux.conventional import decode_uniform

    if type(chunk_rows) is not int or chunk_rows < 1:
        raise ValueError("Require a positive integer chunk size")
    root = Path(artifact_dir)
    if json.loads((root / "status.json").read_text())["status"] != "completed":
        raise ValueError("Conversion is incomplete")
    summary = json.loads((root / "summary.json").read_text())
    if summary.get("format") != "symmetric_uniform" or summary["bits"] not in (4, 8):
        raise ValueError("Not a supported conventional bundle")
    rows = [r for r in summary["results"] if names is None or r["tensor"] in names]
    if (
        not rows
        or len({r["tensor"] for r in rows}) != len(rows)
        or (names is not None and {r["tensor"] for r in rows} != set(names))
    ):
        raise ValueError("Requested tensor set is absent or duplicated")
    params = dict(transformer.named_parameters())
    bits = summary["bits"]
    for row in rows:
        path = root / (row["tensor"] + ".npz")
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["artifact_sha256"]:
            raise ValueError("Artifact hash mismatch")
        if tuple(params[row["tensor"]].shape) != tuple(row["shape"]):
            raise ValueError("Model shape mismatch")
        if row["shape"][1] * bits % 8:
            raise ValueError("Row payload is not byte aligned")
    with torch.no_grad():
        for row in rows:
            param = params[row["tensor"]]
            columns = row["shape"][1]
            stride = columns * bits // 8
            with np.load(root / (row["tensor"] + ".npz"), allow_pickle=False) as data:
                codes, scales = data["codes"], data["scales"]
                for start in range(0, param.shape[0], chunk_rows):
                    stop = min(start + chunk_rows, param.shape[0])
                    array = decode_uniform(
                        codes[start * stride : stop * stride],
                        scales[start:stop],
                        (stop - start, columns),
                        bits,
                    )
                    param[start:stop].copy_(
                        torch.from_numpy(array).to(device=param.device, dtype=param.dtype)
                    )
    return {
        "replaced_tensors": [r["tensor"] for r in rows],
        "weight_format": "symmetric_uniform",
        "runtime_representation": "reconstructed dense parameters",
        "decode_chunk_rows": chunk_rows,
        "packed_artifact_bytes": sum(r["artifact_bytes"] for r in rows),
    }
