"""Weight-residual SVD slice diagnostic; not the SVDQuant algorithm."""

import argparse
import json
import traceback
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .activation import quantize_grouped, quantize_per_token
from .format import decode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    c = json.loads(Path(args.config).read_text())
    out = Path(c["output"])
    out.mkdir(parents=True, exist_ok=False)

    def save(name, value):
        (out / name).write_text(json.dumps(value, indent=2) + "\n")

    save("config.json", c)
    save("status.json", {"status": "running"})
    try:
        rows = []
        storage = []
        with (
            np.load(Path(c["capture"]) / "activations.npz", allow_pickle=False) as inputs,
            safe_open(c["source"], framework="pt") as source,
        ):
            for module in c["modules"]:
                artifact = Path(c["weights"]) / (module + ".weight_only.npz")
                with np.load(artifact, allow_pickle=False) as data:
                    p, e, indices = (
                        data["planes"],
                        data["exponents"],
                        data["output_row_indices"],
                    )
                w = source.get_tensor(module + ".weight")[indices].float().numpy()
                wq = decode(p, e)
                rank = c["rank"]
                if not 0 <= rank < min(w.shape):
                    raise ValueError("Require bounded rank below full slice rank")
                branch_bytes = 0
                file_bytes = 0
                if rank:
                    u, s, vh = np.linalg.svd(w - wq, full_matrices=False)
                    left = torch.from_numpy(u[:, :rank] * s[:rank]).to(torch.bfloat16)
                    right = torch.from_numpy(vh[:rank]).to(torch.bfloat16)
                    branch_bytes = (left.numel() + right.numel()) * 2
                    path = out / (module + ".correction.safetensors")
                    save_file(
                        {"left": left.contiguous(), "right": right.contiguous()},
                        str(path),
                    )
                    file_bytes = path.stat().st_size
                    left, right = left.float().numpy(), right.float().numpy()
                storage.append(
                    {
                        "module": module,
                        "shape": list(w.shape),
                        "packed_weight_payload_bytes": p.nbytes + e.nbytes,
                        "packed_weight_file_bytes": artifact.stat().st_size,
                        "bf16_slice_bytes": w.size * 2,
                        "correction_payload_bytes": branch_bytes,
                        "correction_file_bytes": file_bytes,
                    }
                )
                for key in sorted(k for k in inputs if k.startswith(module + ".call_")):
                    call = int(key.rsplit("_", 1)[-1])
                    x = inputs[key]
                    ref = x @ w.T
                    correction = (x @ right.T) @ left.T if rank else np.zeros_like(ref)
                    for activation in ("original", "a8_token", "a4_g32"):
                        xq = x
                        if activation == "a8_token":
                            q, scale = quantize_per_token(x, 8)
                            xq = q.astype(np.float32) * scale
                        if activation == "a4_g32":
                            q, scale = quantize_grouped(x, 4, 32)
                            xq = (q.astype(np.float32) * scale).reshape(x.shape)
                        result = xq @ wq.T + correction
                        error = float(np.linalg.norm(result - ref) / np.linalg.norm(ref))
                        if not np.isfinite(error):
                            raise ValueError("Nonfinite output error")
                        rows.append(
                            {
                                "module": module,
                                "call": call,
                                "split": "fit"
                                if call < c["fit_calls"]
                                else "calibration_validation",
                                "activation": activation,
                                "relative_output_l2": error,
                            }
                        )
        save(
            "summary.json",
            {
                "scope": "32 spread output rows; FP32 products with BF16-stored residual factors; correction uses original inputs",
                "rank": c["rank"],
                "storage": storage,
                "results": rows,
            },
        )
        save("status.json", {"status": "completed"})
    except BaseException:
        save("status.json", {"status": "failed", "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
