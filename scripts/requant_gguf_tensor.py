#!/usr/bin/env python3
"""Rewrite a GGUF file with one tensor re-quantized; every other tensor and all metadata copied unchanged.

Gap campaign use: edge-dit.cpp (97cc585) keeps an embedding table compressed only for F16/Q8_0/Q5_x/Q4_x
(support_get_rows in ggml_extend.hpp) and expands any other type to F32. The Qwen3-4B Q4_K_M encoder stores
token_embd.weight as Q6_K, which edge-dit expands from 304 MiB to 1484 MiB. Re-quantizing only that tensor
to Q8_0 (higher precision than the Q6_K source) keeps it compressed. The output is a separately labelled
compressed checkpoint.

Usage: requant_gguf_tensor.py <in.gguf> <out.gguf> --tensor token_embd.weight --type Q8_0
"""
import argparse
import hashlib
import json
import sys

import numpy as np
from gguf import GGMLQuantizationType, GGUFReader, GGUFValueType, GGUFWriter
from gguf import quants


def copy_metadata(reader: GGUFReader, writer: GGUFWriter) -> None:
    for field in reader.fields.values():
        name = field.name
        if name.startswith("GGUF.") or name == "general.architecture":
            continue  # header fields; the writer emits its own
        vtype = field.types[0]
        if vtype == GGUFValueType.ARRAY:
            sub = field.types[-1]
            if sub == GGUFValueType.STRING:
                val = [bytes(field.parts[i]).decode("utf-8") for i in field.data]
            else:
                val = [field.parts[i].tolist()[0] for i in field.data]
            writer.add_array(name, val)
        elif vtype == GGUFValueType.STRING:
            writer.add_string(name, bytes(field.parts[field.data[0]]).decode("utf-8"))
        else:
            writer.add_key_value(name, field.parts[field.data[0]].tolist()[0], vtype)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--tensor", required=True)
    ap.add_argument("--type", required=True, help="target GGML type name, e.g. Q8_0")
    args = ap.parse_args()
    target = GGMLQuantizationType[args.type]

    reader = GGUFReader(args.src)
    arch = bytes(reader.fields["general.architecture"].parts[-1]).decode("utf-8")
    writer = GGUFWriter(args.dst, arch)
    copy_metadata(reader, writer)

    changed = None
    for t in reader.tensors:
        shape = [int(x) for x in reversed(t.shape)]  # numpy order (rows, cols)
        if t.name == args.tensor:
            # Row chunks keep peak memory low on the 8 GB Jetson (the full table is 1.5 GB as F32).
            parts, max_abs, sq_err, sq_ref = [], 0.0, 0.0, 0.0
            for r0 in range(0, t.data.shape[0], 8192):
                f32 = quants.dequantize(t.data[r0:r0 + 8192], t.tensor_type).astype(np.float32)
                q = quants.quantize(f32, target)
                back = quants.dequantize(q, target)
                max_abs = max(max_abs, float(np.max(np.abs(back - f32))))
                sq_err += float(np.sum((back - f32).astype(np.float64) ** 2))
                sq_ref += float(np.sum(f32.astype(np.float64) ** 2))
                parts.append(q)
            q = np.concatenate(parts, axis=0)
            writer.add_tensor(t.name, q, raw_dtype=target)
            changed = {"tensor": t.name, "from": t.tensor_type.name, "to": target.name, "shape": shape,
                       "bytes_from": int(t.n_bytes), "bytes_to": int(q.nbytes),
                       "max_abs_diff_vs_source": max_abs,
                       "rel_rms_diff_vs_source": (sq_err / sq_ref) ** 0.5}
        else:
            writer.add_tensor(t.name, t.data, raw_shape=t.data.shape, raw_dtype=t.tensor_type)
    if changed is None:
        print(f"tensor {args.tensor} not found", file=sys.stderr)
        return 1

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    h = hashlib.sha256()
    with open(args.dst, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    changed["output_sha256"] = h.hexdigest()
    print(json.dumps(changed, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
