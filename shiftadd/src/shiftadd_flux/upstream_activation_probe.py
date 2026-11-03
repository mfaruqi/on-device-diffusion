"""Full-row activation-error diagnostics for saved upstream BCQ weights."""

import argparse
import hashlib
import json
import platform
import statistics
import traceback
from pathlib import Path

import numpy as np
from safetensors import safe_open

from .activation import quantize_grouped
from .upstream_loader import decode_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    config = json.loads(Path(parser.parse_args().config).read_text())
    out = Path(config["output"])
    out.mkdir(parents=True, exist_ok=False)

    def save(name, value):
        (out / name).write_text(json.dumps(value, indent=2) + "\n")

    save("config.json", config)
    save("status.json", {"status": "running"})
    save("environment.json", {"platform": platform.platform(), "numpy": np.__version__})
    try:
        root = Path(config["artifacts"])
        fit = json.loads((root / "config.json").read_text())
        summary = json.loads((root / "summary.json").read_text())
        if json.loads((root / "status.json").read_text())["status"] != "completed":
            raise ValueError("Require completed upstream fit")
        if config["activation_bits"] != [4, 8]:
            raise ValueError("This diagnostic requires separate A4 and A8 cases")
        records, hashes = [], {}
        capture = Path(fit["capture"]) / "activations.npz"
        with (
            safe_open(fit["source"], framework="pt") as source,
            np.load(capture, allow_pickle=False) as inputs,
        ):
            for entry in summary["storage"]:
                module = entry["module"]
                path = root / (module + ".npz")
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if digest != entry["artifact_sha256"]:
                    raise ValueError("Artifact hash mismatch")
                hashes[module] = digest
                original = source.get_tensor(module + ".weight").float().numpy()
                with np.load(path, allow_pickle=False) as packed:
                    indices = packed["output_row_indices"]
                    if not np.array_equal(indices, np.arange(original.shape[0])):
                        raise ValueError("Require all output rows in original order")
                    weights = decode_rows(packed, fit["scale_format"], 0, original.shape[0])
                keys = sorted(k for k in inputs if k.startswith(module + ".call_"))
                if len(keys) != config["expected_calls"]:
                    raise ValueError("Unexpected capture coverage")
                for call, key in enumerate(keys):
                    x = inputs[key]
                    reference = x @ original.T
                    denominator = np.linalg.norm(reference)
                    if not np.isfinite(denominator) or denominator <= 0:
                        raise ValueError("Invalid reference norm")
                    activations = {"original": x}
                    for bits in config["activation_bits"]:
                        q, scale = quantize_grouped(x, bits, config["activation_group_size"])
                        activations[f"a{bits}"] = (q.astype(np.float32) * scale).reshape(x.shape)
                    for weight_label, w in (("bf16", original), ("upstream_u8", weights)):
                        for activation_label, xq in activations.items():
                            error = float(np.linalg.norm(xq @ w.T - reference) / denominator)
                            if not np.isfinite(error):
                                raise ValueError("Nonfinite output error")
                            records.append(
                                {
                                    "module": module,
                                    "call": call,
                                    "capture": key,
                                    "split": "fit"
                                    if call < fit["fit_calls"]
                                    else "calibration_validation",
                                    "weights": weight_label,
                                    "activation": activation_label,
                                    "relative_output_l2": error,
                                }
                            )
        groups = sorted({(r["module"], r["weights"], r["activation"], r["split"]) for r in records})
        aggregates = []
        for module, weights, activation, split in groups:
            values = [
                r["relative_output_l2"]
                for r in records
                if (r["module"], r["weights"], r["activation"], r["split"])
                == (module, weights, activation, split)
            ]
            aggregates.append(
                dict(
                    module=module,
                    weights=weights,
                    activation=activation,
                    split=split,
                    mean_relative_output_l2=statistics.mean(values),
                    calls=len(values),
                )
            )
        save(
            "summary.json",
            {
                "scope": "CPU projection-error diagnostic; no performance or image-quality measurement",
                "artifact_hashes": hashes,
                "capture_sha256": hashlib.sha256(capture.read_bytes()).hexdigest(),
                "fit_config": fit,
                "aggregate": aggregates,
                "results": records,
            },
        )
        save("status.json", {"status": "completed", "comparisons": len(records)})
        print(f"Completed {len(records)} projection-error comparisons")
    except BaseException:
        save("status.json", {"status": "failed", "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
