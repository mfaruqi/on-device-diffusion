"""Fresh-process CPU loader-memory diagnostic; no inference benchmark."""

import argparse
import hashlib
import json
import platform
import resource
import subprocess
import sys
import traceback
from pathlib import Path


def child(config):
    import torch

    from shiftadd_flux.loader import apply_artifacts

    if sys.platform not in ("darwin", "linux"):
        raise ValueError("Unspecified ru_maxrss units")
    row = next(
        r
        for r in json.loads((Path(config["bundle"]) / "summary.json").read_text())["results"]
        if r["tensor"] == config["tensor"]
    )

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(row["shape"], dtype=torch.bfloat16))

        def named_parameters(self, *args, **kwargs):
            yield config["tensor"], self.weight

    model = Model()
    audit = apply_artifacts(
        model,
        config["bundle"],
        planes=None,
        names=[config["tensor"]],
        chunk_rows=config["chunk_rows"],
    )
    digest = hashlib.sha256(
        memoryview(model.weight.detach().view(torch.uint16).numpy())
    ).hexdigest()
    multiplier = 1 if sys.platform == "darwin" else 1024
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * multiplier
    return {
        "process_peak_rss_gib": peak / 2**30,
        "parameter_bytes": model.weight.numel() * 2,
        "parameter_sha256": digest,
        "shape": row["shape"],
        "audit": audit,
        "torch": torch.__version__,
        "platform": platform.platform(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--child", action="store_true")
    a = parser.parse_args()
    c = json.loads(Path(a.config).read_text())
    if a.child:
        print(json.dumps(child(c)))
        return
    out = Path(c["output"])
    out.mkdir(parents=True, exist_ok=False)

    def save(name, value):
        (out / name).write_text(json.dumps(value, indent=2) + "\n")

    save("config.json", c)
    save("status.json", {"status": "running"})
    try:
        rows = []
        for i in range(c["fresh_processes"]):
            process = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "shiftadd_flux.diagnostics.load_memory_probe",
                    "--config",
                    a.config,
                    "--child",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            (out / f"child-{i}.stderr.txt").write_text(process.stderr)
            if process.returncode:
                raise RuntimeError(f"Child {i} failed: {process.stderr}")
            row = json.loads(process.stdout)
            row.update(index=i, phase="first" if i == 0 else "measured")
            rows.append(row)
            save("children.json", rows)
        if len({r["parameter_sha256"] for r in rows}) != 1:
            raise ValueError("Loader output differs between processes")
        import statistics

        values = [r["process_peak_rss_gib"] for r in rows if r["phase"] == "measured"]
        save(
            "summary.json",
            {
                "scope": "fresh CPU process lifetime high-water RSS including imports and one BF16 matrix; not GPU memory",
                "first_process_peak_rss_gib": rows[0]["process_peak_rss_gib"],
                "measured_process_peak_rss_gib": {
                    "median": statistics.median(values),
                    "min": min(values),
                    "max": max(values),
                },
                "parameter_sha256": rows[0]["parameter_sha256"],
                "parameter_bytes": rows[0]["parameter_bytes"],
                "unavailable_metrics": ["inference_latency", "device_memory", "energy"],
            },
        )
        save(
            "environment.json",
            {
                "platform": rows[0]["platform"],
                "torch": rows[0]["torch"],
                "python": sys.version,
                "memory_source": "resource.ru_maxrss; Darwin bytes / Linux KiB",
            },
        )
        save("status.json", {"status": "completed"})
    except BaseException:
        save("status.json", {"status": "failed", "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
