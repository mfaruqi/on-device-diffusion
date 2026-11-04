"""Fit a bounded weight slice using calibration moments; validate actual outputs."""

import argparse
import json
import traceback
from pathlib import Path

import numpy as np
from safetensors import safe_open

from shiftadd_flux.activation import quantize_grouped, quantize_per_token
from shiftadd_flux.format import decode
from shiftadd_flux.radix import encode_radix


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    c = json.loads(Path(args.config).read_text())
    out = Path(c["output"])
    out.mkdir(parents=True, exist_ok=False)

    def write(name, value):
        (out / name).write_text(json.dumps(value, indent=2) + "\n")

    write("config.json", c)
    write("status.json", {"status": "running"})
    try:
        capture = Path(c["capture"])
        if json.loads((capture / "status.json").read_text())["status"] != "completed":
            raise ValueError("Capture incomplete")
        rows = []
        selected_rows = {}
        with (
            np.load(capture / "activations.npz", allow_pickle=False) as samples,
            safe_open(c["source"], framework="pt") as source,
        ):
            for module in c["modules"]:
                keys = sorted(k for k in samples if k.startswith(module + ".call_"))
                if len(keys) != 16:
                    raise ValueError("Expected four prompts by four steps")
                x = [samples[k] for k in keys]
                fit = np.concatenate(x[: c["fit_calls"]], axis=0)
                moment = np.square(fit.astype(np.float64)).mean(axis=0).astype(np.float32)
                moment /= max(float(moment.mean()), 1e-30)
                full = source.get_tensor(module + ".weight").float().numpy()
                policy = c.get("row_selection", "first")
                count = min(c["output_rows"], len(full))
                if policy not in ("first", "evenly_spaced"):
                    raise ValueError("Unknown row selection")
                indices = (
                    np.arange(count) if policy == "first" else np.arange(count) * len(full) // count
                )
                selected_rows[module] = indices.tolist()
                w = full[indices]
                for method, importance in [
                    ("weight_only", None),
                    ("diagonal_activation", moment),
                ]:
                    packed, e = encode_radix(
                        w,
                        planes=c["planes"],
                        group_size=c["group_size"],
                        importance=importance,
                    )
                    np.savez(
                        out / (module + "." + method + ".npz"),
                        planes=packed,
                        exponents=e,
                        output_row_indices=indices,
                    )
                    reconstructed = decode(packed, e)
                    for call, activation in enumerate(x):
                        reference = activation @ w.T
                        for bits in [None, 8, 4]:
                            inputs = activation
                            if bits is not None:
                                q, scale = quantize_per_token(activation, bits)
                                inputs = q.astype(np.float32) * scale
                                if bits == 4 and c.get("activation_group_size"):
                                    q, scale = quantize_grouped(
                                        activation, bits, c["activation_group_size"]
                                    )
                                    inputs = (q.astype(np.float32) * scale).reshape(
                                        activation.shape
                                    )
                            result = inputs @ reconstructed.T
                            rows.append(
                                {
                                    "module": module,
                                    "method": method,
                                    "call": call,
                                    "sample_index": call // 4,
                                    "step": call % 4,
                                    "split": "fit"
                                    if call < c["fit_calls"]
                                    else "calibration_validation",
                                    "activation_bits": bits,
                                    "relative_output_l2": float(
                                        np.linalg.norm(result - reference)
                                        / np.linalg.norm(reference)
                                    ),
                                }
                            )
        write(
            "summary.json",
            {
                "scope": "selected output rows only; calibration split, not project held-out data",
                "selected_output_rows": selected_rows,
                "results": rows,
            },
        )
        write("status.json", {"status": "completed"})
    except BaseException:
        write("status.json", {"status": "failed", "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
