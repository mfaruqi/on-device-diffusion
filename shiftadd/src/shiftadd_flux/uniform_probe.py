"""Same-source symmetric uniform quantization slice control."""

import argparse
import json
import traceback
from pathlib import Path

import numpy as np
from safetensors import safe_open

from .activation import quantize_grouped, quantize_per_token
from .conventional import decode_uniform, encode_uniform


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    c = json.loads(Path(parser.parse_args().config).read_text())
    out = Path(c["output"])
    out.mkdir(parents=True, exist_ok=False)

    def save(name, value):
        (out / name).write_text(json.dumps(value, indent=2) + "\n")

    save("config.json", c)
    save("status.json", {"status": "running"})
    try:
        results = []
        storage = []
        with (
            safe_open(c["source"], framework="pt") as source,
            np.load(Path(c["capture"]) / "activations.npz", allow_pickle=False) as inputs,
        ):
            for module in c["modules"]:
                with np.load(
                    Path(c["row_source"]) / (module + ".weight_only.npz"),
                    allow_pickle=False,
                ) as old:
                    indices = old["output_row_indices"]
                w = source.get_tensor(module + ".weight")[indices].float().numpy()
                payload, scales = encode_uniform(w, c["bits"], c["group_size"])
                path = out / (module + ".npz")
                np.savez(
                    path,
                    codes=payload,
                    scales=scales,
                    shape=np.array(w.shape),
                    output_row_indices=indices,
                )
                with np.load(path, allow_pickle=False) as saved:
                    wq = decode_uniform(
                        saved["codes"],
                        saved["scales"],
                        saved["shape"].tolist(),
                        c["bits"],
                    )
                storage.append(
                    {
                        "module": module,
                        "shape": list(w.shape),
                        "payload_bytes": payload.nbytes + scales.nbytes,
                        "file_bytes": path.stat().st_size,
                        "bf16_bytes": w.size * 2,
                        "relative_weight_l2": float(np.linalg.norm(w - wq) / np.linalg.norm(w)),
                    }
                )
                for key in sorted(k for k in inputs if k.startswith(module + ".call_")):
                    call = int(key.rsplit("_", 1)[-1])
                    x = inputs[key]
                    ref = x @ w.T
                    for activation in ("original", "a8_token", "a4_g32"):
                        xq = x
                        if activation == "a8_token":
                            q, s = quantize_per_token(x, 8)
                            xq = q.astype(np.float32) * s
                        if activation == "a4_g32":
                            q, s = quantize_grouped(x, 4, 32)
                            xq = (q.astype(np.float32) * s).reshape(x.shape)
                        error = float(np.linalg.norm(xq @ wq.T - ref) / np.linalg.norm(ref))
                        if not np.isfinite(error):
                            raise ValueError("Nonfinite output error")
                        results.append(
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
                "scope": "same-source 32 spread rows; FP32 products; weight-only scale selection",
                "storage": storage,
                "results": results,
            },
        )
        save("status.json", {"status": "completed"})
    except BaseException:
        save("status.json", {"status": "failed", "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
