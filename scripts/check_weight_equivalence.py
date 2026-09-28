#!/usr/bin/env python
"""Check that two safetensors checkpoints hold the same weights, independent of layout.

Usage:
  python scripts/check_weight_equivalence.py --a <file|index.json> --b <file|index.json> --out report.json

Different packagings of one model rename tensors, fuse them (e.g. q/k/v into one qkv matrix),
split them, or reorder row blocks, so a name-by-name comparison needs a hand-written mapping.
This check avoids that: every tensor is cut into rows (slices along dim 0; 1-D tensors are one
row), each row's raw bytes are hashed, and the two sides' multisets of (row width, dtype, hash)
must be equal. Equal multisets mean every number in A appears in B, grouped the same way within
rows, and vice versa. It does not prove the rows are placed in the same order, which is
the loader's job and is checked end to end by generating images.

If the sides use different float dtypes (e.g. FP32 vs BF16), pass --cast-to bf16. The wider
side is then rounded (round-to-nearest-even, as torch does) before hashing, and the report
says so.

Only the standard library and numpy are used. Files are read via mmap, not loaded into RAM.
"""

import argparse
import collections
import hashlib
import json
import mmap
import struct
from pathlib import Path

import numpy as np

ITEMSIZE = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}


def expand(path):
    """A path is a .safetensors file or a *.safetensors.index.json listing shards."""
    path = Path(path)
    if path.suffix == ".json":
        shards = sorted(set(json.loads(path.read_text())["weight_map"].values()))
        return [path.parent / s for s in shards]
    return [path]


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    return header, 8 + n


def f32_to_bf16_bytes(buf):
    u = np.frombuffer(buf, dtype=np.uint32)
    rounded = u + (0x7FFF + ((u >> 16) & 1))
    return (rounded >> 16).astype(np.uint16).tobytes()


def row_hashes(paths, cast_to):
    rows = collections.Counter()
    info = {"files": [], "tensors": 0, "rows": 0, "params": 0, "dtypes": collections.Counter()}
    for p in paths:
        header, base = read_header(p)
        h_file = hashlib.sha256()
        with open(p, "rb") as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            for name, t in header.items():
                dtype, shape = t["dtype"], t["shape"]
                start, end = t["data_offsets"]
                data = mm[base + start: base + end]
                n_rows = shape[0] if len(shape) > 1 else 1
                row_bytes = len(data) // n_rows if n_rows else 0
                out_dtype = dtype
                if cast_to == "bf16" and dtype == "F32":
                    out_dtype = "BF16"
                for r in range(n_rows):
                    chunk = data[r * row_bytes: (r + 1) * row_bytes]
                    if out_dtype != dtype:
                        chunk = f32_to_bf16_bytes(chunk)
                    width = len(chunk) // ITEMSIZE[out_dtype]
                    rows[(width, out_dtype, hashlib.sha256(chunk).digest())] += 1
                info["tensors"] += 1
                info["rows"] += n_rows
                info["params"] += len(data) // ITEMSIZE[dtype]
                info["dtypes"][dtype] += 1
            mm.close()
        with open(p, "rb") as f:
            for block in iter(lambda: f.read(1 << 24), b""):
                h_file.update(block)
        info["files"].append({"path": str(p), "sha256": h_file.hexdigest(), "size": p.stat().st_size})
    info["dtypes"] = dict(info["dtypes"])
    return rows, info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="reference checkpoint (file or index.json)")
    ap.add_argument("--b", required=True, help="candidate checkpoint (file or index.json)")
    ap.add_argument("--cast-to", choices=["bf16"], default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rows_a, info_a = row_hashes(expand(args.a), args.cast_to)
    rows_b, info_b = row_hashes(expand(args.b), args.cast_to)
    only_a = rows_a - rows_b
    only_b = rows_b - rows_a
    report = {
        "method": "multiset of per-row sha256 over raw tensor bytes (see script docstring)",
        "cast_to": args.cast_to,
        "a": info_a,
        "b": info_b,
        "params_equal": info_a["params"] == info_b["params"],
        "rows_only_in_a": sum(only_a.values()),
        "rows_only_in_b": sum(only_b.values()),
        "equivalent": not only_a and not only_b,
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("a", "b")}, indent=2))
    print(f"A: {info_a['tensors']} tensors, {info_a['params']:,} values {info_a['dtypes']}")
    print(f"B: {info_b['tensors']} tensors, {info_b['params']:,} values {info_b['dtypes']}")


if __name__ == "__main__":
    main()
