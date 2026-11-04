#!/usr/bin/env python
"""Quantize klein's Qwen3 text encoder once and save it, so later runs reload the quantized weights.

Used by the Jetson PyTorch runs with stage_residency.text_encoder = "release_after_encode": the encoder
is reloaded every generation, and reading ~2.5 GB of NF4 weights is much cheaper than re-quantizing the
7.5 GB BF16 checkpoint each time. Quantization happens while loading (device_map), so the BF16 encoder is
never resident. The output directory gets a receipt with the source revision and file hashes.

Usage: save_quantized_text_encoder.py <klein snapshot dir> <out dir> --method bnb_nf4
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
import transformers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshot", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--method", choices=["bnb_nf4"], required=True)
    args = ap.parse_args()
    qc = transformers.BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                         bnb_4bit_compute_dtype=torch.bfloat16)
    t = time.time()
    model = transformers.Qwen3ForCausalLM.from_pretrained(args.snapshot, subfolder="text_encoder", quantization_config=qc,
                                                          torch_dtype=torch.bfloat16, device_map={"": "cuda"})
    load_s = time.time() - t
    model.save_pretrained(args.out)
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.out.glob("*.safetensors"))}
    receipt = {"source_snapshot": str(args.snapshot), "source_revision": args.snapshot.name, "method": args.method,
               "quantization_config": qc.to_dict(), "transformers": transformers.__version__,
               "footprint_mb": model.get_memory_footprint() / 2**20, "quantize_load_s": load_s,
               "safetensors_sha256": files}
    (args.out / "quantization-receipt.json").write_text(json.dumps(receipt, indent=1, default=str) + "\n")
    print(json.dumps(receipt, indent=1, default=str))


if __name__ == "__main__":
    main()
