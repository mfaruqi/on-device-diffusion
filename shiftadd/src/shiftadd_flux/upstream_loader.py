"""Dense reconstruction of the explicitly labelled upstream BCQ artifacts."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def decode_rows(data, scale_format, start, stop):
    packed = data["planes"][:, start:stop]
    q, rows, groups, width = packed.shape
    d = data["superblock_scale"][start:stop].astype(np.float32)
    if scale_format == "u8":
        alpha = data["scale_codes"][start:stop].astype(np.float32) * d[..., None, None]
    elif scale_format == "pot":
        per_row = groups * q
        if per_row % 2:
            raise ValueError("Exponent rows must be nibble aligned")
        payload = data["scale_exponent_nibbles"][start * per_row // 2 : stop * per_row // 2]
        values = np.empty(payload.size * 2, np.uint8)
        values[::2] = payload & 15
        values[1::2] = payload >> 4
        alpha = (
            np.exp2(-values.reshape(rows, groups // 2, 2, q).astype(np.float32))
            * d[..., None, None]
        )
    else:
        raise ValueError("Unknown upstream scale format")
    signs = np.unpackbits(packed, axis=-1, bitorder="little").astype(np.float32) * 2 - 1
    return (
        (signs * alpha.reshape(rows, groups, q).transpose(2, 0, 1)[..., None])
        .sum(axis=0)
        .reshape(rows, groups * width * 8)
    )


def apply_upstream_artifacts(
    transformer, artifact_dir, names=None, chunk_rows=128, cache_packed_arrays=False
):
    if type(chunk_rows) is not int or chunk_rows <= 0:
        raise ValueError("Invalid row chunk")
    if type(cache_packed_arrays) is not bool:
        raise ValueError("cache_packed_arrays must be a bool")
    root = Path(artifact_dir)
    config = json.loads((root / "config.json").read_text())
    if json.loads((root / "status.json").read_text())["status"] != "completed":
        raise ValueError("Incomplete fit")
    if config["group_size"] != 128 or config["scale_format"] not in ("u8", "pot"):
        raise ValueError("This adapter supports group 128 and u8/pot scales only")
    rows = json.loads((root / "summary.json").read_text())["storage"]
    rows = [r for r in rows if names is None or r["module"] + ".weight" in names]
    selected = {r["module"] + ".weight" for r in rows}
    if not rows or len(selected) != len(rows) or (names is not None and selected != set(names)):
        raise ValueError("Missing or duplicate matrix selection")
    params = dict(transformer.named_parameters())
    for row in rows:
        name = row["module"] + ".weight"
        path = root / (row["module"] + ".npz")
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["artifact_sha256"]:
            raise ValueError("Artifact hash mismatch")
        shape = tuple(params[name].shape)
        with np.load(path, allow_pickle=False) as data:
            p = data["planes"]
            indices = data["output_row_indices"]
            if (
                len(shape) != 2
                or p.shape[1] != shape[0]
                or p.shape[2] * p.shape[3] * 8 != shape[1]
                or not np.array_equal(indices, np.arange(shape[0]))
            ):
                raise ValueError("Artifact must cover every output row in original order")
    with torch.no_grad():
        for row in rows:
            param = params[row["module"] + ".weight"]
            with np.load(root / (row["module"] + ".npz"), allow_pickle=False) as data:
                # NpzFile reads a complete array on each lookup. Keeping only the
                # packed arrays avoids repeating that work for every row chunk.
                keys = ["planes", "superblock_scale"]
                keys.append(
                    "scale_codes" if config["scale_format"] == "u8" else "scale_exponent_nibbles"
                )
                decoded_data = {key: data[key] for key in keys} if cache_packed_arrays else data
                for start in range(0, param.shape[0], chunk_rows):
                    stop = min(start + chunk_rows, param.shape[0])
                    array = decode_rows(decoded_data, config["scale_format"], start, stop)
                    if not np.isfinite(array).all():
                        raise ValueError("Nonfinite reconstructed weights")
                    param[start:stop].copy_(
                        torch.from_numpy(array).to(device=param.device, dtype=param.dtype)
                    )
                del decoded_data
    return {
        "replaced_tensors": sorted(selected),
        "weight_format": "upstream_bcq",
        "scale_format": config["scale_format"],
        "decode_chunk_rows": chunk_rows,
        "cache_packed_arrays": cache_packed_arrays,
        "runtime_representation": "reconstructed dense parameters",
        "packed_artifact_bytes": sum(r["file_bytes"] for r in rows),
    }
