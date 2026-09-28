#!/usr/bin/env python3
"""Extract single-CLI diagnostics, not benchmark metrics, from imported Jetson logs.

Definitions: wiki/methods/baseline-metrics.md#jetson-cli-feasibility-diagnostics
Does not run inference, modify inputs, or align un-timestamped stages to RAM samples.
"""

import argparse
import csv
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path


def analyze(run_dir):
    names = ["generation.log", "tegrastats.log", "command.json", "status-original.json",
             "power-mode.txt", "memory-before.txt", "memory-after.txt"]
    inputs = {name: (run_dir / name).read_bytes() for name in names}
    log = inputs["generation.log"].decode()
    rows = []
    for line_no, line in enumerate(inputs["tegrastats.log"].decode().splitlines(), 1):
        match = re.search(r"RAM (\d+)/(\d+)MB .*?SWAP (\d+)/(\d+)MB", line)
        if not match:
            raise ValueError(f"Unrecognized tegrastats sample at line {line_no}")
        stamp = datetime.strptime(line[:19], "%m-%d-%Y %H:%M:%S")
        ram, total, swap, swap_total = map(int, match.groups())
        gpu = re.search(r"gpu@([\d.]+)C", line)
        rows.append(dict(source_line=line_no, timestamp_local=stamp.isoformat(),
                         ram_reported_mb=ram, ram_total_reported_mb=total,
                         swap_reported_mb=swap, swap_total_reported_mb=swap_total,
                         ram_gib=ram / 1024, swap_gib=swap / 1024,
                         gpu_temperature_c=float(gpu[1]) if gpu else None))
    if not rows:
        raise ValueError("No memory samples")
    if any(a["timestamp_local"] >= b["timestamp_local"] for a, b in zip(rows, rows[1:])):
        raise ValueError("Non-increasing sample timestamps")
    patterns = {
        "text_encode_seconds": r"get_learned_condition completed, taking ([\d.]+)s",
        "sampling_seconds": r"sampling completed, taking ([\d.]+)s",
        "decode_first_stage_seconds": r"decode_first_stage completed, taking ([\d.]+)s",
        "generate_image_seconds": r"generate_image completed in ([\d.]+)s",
    }
    times = {}
    for key, pattern in patterns.items():
        matches = re.findall(pattern, log)
        if len(matches) != 1:
            raise ValueError(f"Expected one {key} diagnostic, got {len(matches)}")
        times[key] = float(matches[0])
    generation_start = log.index("generate_image 512x512")
    loads = re.findall(r"loading tensors completed, taking ([\d.]+)s", log[:generation_start])
    if len(loads) != 1:
        raise ValueError("Expected one initial tensor-loading diagnostic")
    times["initial_tensor_loading_seconds"] = float(loads[0])
    stages_sum = sum(times[k] for k in ("text_encode_seconds", "sampling_seconds",
                                      "decode_first_stage_seconds"))
    times["unattributed_seconds_from_rounded_logs"] = round(
        times["generate_image_seconds"] - stages_sum, 2)
    segments = {name: int(count) for name, count in re.findall(
        r"(qwen3|flux|vae) compute buffer size: .*?peak across (\d+) segments?", log)}
    peak = max(rows, key=lambda x: x["ram_reported_mb"])
    temperatures = [x["gpu_temperature_c"] for x in rows if x["gpu_temperature_c"] is not None]
    report = {
        "scope": "single CLI feasibility attempt; no measured-run medians",
        "input_sha256": {name: hashlib.sha256(data).hexdigest() for name, data in inputs.items()},
        "engine_log_diagnostics": times,
        "segments_reported": segments,
        "monitor_window": {
            "samples": len(rows), "first_sample": rows[0]["timestamp_local"],
            "last_sample": rows[-1]["timestamp_local"], "requested_interval_ms": 1000,
            "ram_peak_gib": peak["ram_gib"], "ram_total_gib": peak["ram_total_reported_mb"] / 1024,
            "ram_peak_source_line": peak["source_line"], "ram_peak_timestamp": peak["timestamp_local"],
            "swap_min_gib": min(x["swap_gib"] for x in rows),
            "swap_max_gib": max(x["swap_gib"] for x in rows),
            "gpu_temperature_max_c": max(temperatures) if temperatures else None,
        },
        "unit_convention": "Tegrastats printed MB treated as MiB (matching reported RAM and swap totals); divide by 1024 for approximate GiB; raw integers retained in CSV.",
        "limitations": [
            "One-second samples can miss short memory peaks; entire capture window, including setup/idle.",
            "RAM is system-wide reported usage, not process allocation or MemAvailable.",
            "No stage-to-memory alignment: generation log has no wall-clock timestamps.",
            "Constant swap occupancy does not establish zero swap I/O.",
            "Initial tensor load excludes some startup work; engine stage times include on-demand loading.",
            "No measured process wall time, warm-inference result, physical disk I/O or throttling diagnosis.",
        ],
    }
    with (run_dir / "tegrastats-samples.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (run_dir / "feasibility-analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.run_dir), indent=2))
