"""Fresh-process upstream loader timing and memory; no inference benchmark."""

import argparse
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
import traceback
from pathlib import Path


def child(config):
    os.environ["OMP_NUM_THREADS"] = str(config["cpu_threads"])
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(config["cpu_threads"])
    import torch

    from .upstream_loader import apply_upstream_artifacts

    torch.set_num_threads(config["cpu_threads"])

    if sys.platform not in ("darwin", "linux"):
        raise ValueError("Unspecified ru_maxrss units")
    row = next(
        r
        for r in json.loads((Path(config["bundle"]) / "summary.json").read_text())["storage"]
        if r["module"] + ".weight" == config["tensor"]
    )

    shape = [row["output_rows"], row["channels"]]

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(shape, dtype=torch.bfloat16))

        def named_parameters(self, *args, **kwargs):
            yield config["tensor"], self.weight

    model = Model()
    start = time.perf_counter()
    audit = apply_upstream_artifacts(
        model,
        config["bundle"],
        names=[config["tensor"]],
        chunk_rows=config["chunk_rows"],
        cache_packed_arrays=config["cache_packed_arrays"],
    )
    elapsed = time.perf_counter() - start
    digest = hashlib.sha256(
        memoryview(model.weight.detach().view(torch.uint16).numpy())
    ).hexdigest()
    multiplier = 1 if sys.platform == "darwin" else 1024
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * multiplier
    return {
        "process_peak_rss_gib": peak / 2**30,
        "loader_wall_s": elapsed,
        "parameter_bytes": model.weight.numel() * 2,
        "parameter_sha256": digest,
        "shape": shape,
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
        if c["warmup_processes"] < 1 or c["measured_processes"] < 3:
            raise ValueError("Require at least one discarded warmup and three measured children")
        for i in range(1 + c["warmup_processes"] + c["measured_processes"]):
            process = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "shiftadd_flux.upstream_load_probe",
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
            row.update(
                index=i,
                phase="first"
                if i == 0
                else ("warmup" if i <= c["warmup_processes"] else "measured"),
            )
            rows.append(row)
            save("children.json", rows)
        if len({r["parameter_sha256"] for r in rows}) != 1:
            raise ValueError("Loader output differs between processes")
        import statistics

        values = [r["process_peak_rss_gib"] for r in rows if r["phase"] == "measured"]
        times = [r["loader_wall_s"] for r in rows if r["phase"] == "measured"]
        save(
            "summary.json",
            {
                "scope": "fresh CPU process lifetime high-water RSS including imports and one BF16 matrix; not GPU memory",
                "first_process_peak_rss_gib": rows[0]["process_peak_rss_gib"],
                "first_loader_wall_s": rows[0]["loader_wall_s"],
                "measured_loader_wall_s": {
                    "median": statistics.median(times),
                    "min": min(times),
                    "max": max(times),
                },
                "timing_scope": "apply_upstream_artifacts including file validation, packed array reads, reconstruction and parameter copies; excludes imports, model allocation and output hashing; filesystem cache uncontrolled",
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
                "cpu_threads": c["cpu_threads"],
            },
        )
        save("status.json", {"status": "completed"})
    except BaseException:
        save("status.json", {"status": "failed", "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
