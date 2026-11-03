"""Bounded application of pinned ShiftAddQwen's upstream BCQ fitter to FLUX slices."""

import argparse
import contextlib
import hashlib
import io
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    a = p.parse_args()
    c = json.loads(Path(a.config).read_text())
    out = Path(c["output"])
    out.mkdir(parents=True, exist_ok=False)

    def save(name, obj):
        (out / name).write_text(json.dumps(obj, indent=2) + "\n")

    save("config.json", c)
    save("status.json", {"status": "running"})
    try:
        vendor = Path(c["vendor"])
        provenance = json.loads((vendor / "PROVENANCE.json").read_text())
        for file, digest in provenance["files_sha256"].items():
            if hashlib.sha256((vendor / file).read_bytes()).hexdigest() != digest:
                raise ValueError("Vendor source changed")
        sys.path.insert(0, str(vendor.resolve()))
        from saq.bcq import bits_per_weight
        from saq.gptq_bcq import quantize

        results = []
        storage = []
        with (
            np.load(Path(c["capture"]) / "activations.npz", allow_pickle=False) as inputs,
            safe_open(c["source"], framework="pt") as source,
        ):
            for module in c["modules"]:
                keys = sorted(k for k in inputs if k.startswith(module + ".call_"))
                if len(keys) != 16:
                    raise ValueError("Expected 16 calibration calls")
                fit = torch.from_numpy(np.concatenate([inputs[k] for k in keys[: c["fit_calls"]]]))
                H = (fit.T @ fit) / fit.shape[0]
                if c.get("row_policy", "saved_slice") == "all":
                    indices = np.arange(source.get_slice(module + ".weight").get_shape()[0])
                elif c.get("row_policy", "saved_slice") == "saved_slice":
                    with np.load(
                        Path(c["row_source"]) / (module + ".weight_only.npz"),
                        allow_pickle=False,
                    ) as old:
                        indices = old["output_row_indices"]
                else:
                    raise ValueError("Unknown output-row policy")
                w = source.get_tensor(module + ".weight")[indices].float()
                log = io.StringIO()
                with contextlib.redirect_stdout(log):
                    result = quantize(
                        w,
                        H,
                        q=c["planes"],
                        G=c["group_size"],
                        fmt=c["scale_format"],
                        mu_on=False,
                        SB=256,
                        blocksize=128,
                        percdamp=c["damping"],
                        fit_iters=c["fit_iters"],
                        rounds=1,
                        refit=True,
                        act_order=True,
                    )
                (out / (module + ".fit-log.txt")).write_text(log.getvalue())
                if "cholesky failed" in log.getvalue():
                    raise ValueError(
                        "Upstream increased damping beyond configured value; retain as failed"
                    )
                idx = result["idx"].numpy().astype(np.uint8)
                rows, columns = idx.shape
                q = c["planes"]
                g = c["group_size"]
                packed = np.stack(
                    [
                        np.packbits(
                            ((idx.reshape(rows, -1, g) >> p) & 1).astype(bool),
                            axis=-1,
                            bitorder="little",
                        )
                        for p in range(q)
                    ]
                )
                st = result["store"]["alpha_store"]
                d = st["d"].numpy().astype(np.float16)
                arrays = {
                    "planes": packed,
                    "superblock_scale": d,
                    "output_row_indices": indices,
                }
                if c["scale_format"] == "u8":
                    code = st["code"].numpy()
                    arrays["scale_codes"] = code
                    alpha = code.astype(np.float32) * d[..., None, None].astype(np.float32)
                else:
                    e = st["e"].numpy()
                    flat = e.reshape(-1)
                    arrays["scale_exponent_nibbles"] = flat[::2] | (flat[1::2] << 4)
                    alpha = np.exp2(-e.astype(np.float32)) * d[..., None, None].astype(np.float32)
                signs = np.unpackbits(packed, axis=-1, bitorder="little").astype(np.float32) * 2 - 1
                scales = alpha.reshape(rows, -1, q).transpose(2, 0, 1)
                decoded = (signs * scales[..., None]).sum(axis=0).reshape(rows, columns)
                if not np.allclose(decoded, result["Q"].numpy(), rtol=1e-6, atol=1e-8):
                    raise ValueError("Packed reconstruction disagrees with upstream")
                if not np.isfinite(decoded).all():
                    raise ValueError("Nonfinite reconstruction")
                path = out / (module + ".npz")
                np.savez(path, **arrays)
                payload = sum(v.nbytes for k, v in arrays.items() if k != "output_row_indices")
                storage.append(
                    {
                        "module": module,
                        "payload_bytes": payload,
                        "file_bytes": path.stat().st_size,
                        "artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "bits_per_weight": bits_per_weight(q, g, c["scale_format"], False),
                        "fit_samples": fit.shape[0],
                        "channels": columns,
                        "output_rows": rows,
                        "relative_weight_l2": float(
                            np.linalg.norm(w.numpy() - decoded) / np.linalg.norm(w.numpy())
                        ),
                    }
                )
                for call, key in enumerate(keys):
                    x = inputs[key]
                    reference = x @ w.numpy().T
                    error = float(
                        np.linalg.norm(x @ decoded.T - reference) / np.linalg.norm(reference)
                    )
                    results.append(
                        {
                            "module": module,
                            "call": call,
                            "split": "fit" if call < c["fit_calls"] else "calibration_validation",
                            "relative_output_l2": error,
                        }
                    )
                save(
                    "summary.json",
                    {
                        "results": results,
                        "storage": storage,
                        "scope": "configured output rows; upstream GPTQ/BCQ with calibration inputs, not full-model reproduction",
                        "upstream": provenance,
                    },
                )
        save("status.json", {"status": "completed"})
    except BaseException:
        save("status.json", {"status": "failed", "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
