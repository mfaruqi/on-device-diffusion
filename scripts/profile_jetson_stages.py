#!/usr/bin/env python3
"""Single NVTX harness capture; not a repeated baseline runner. Standard library only.

Read capture_profile() for the sequence; jetson_profile.py holds validation,
jetson_device.py owns system monitoring, and measurement.py records run status.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess

from benchlib import write_json
from jetson_device import TegrastatsMonitor, memory_snapshot
from jetson_profile import check_config, collect_environment, profile_command, save_profile_results, verified_model_paths
from measurement import create_run_directory, run_status, sampling
from measurement_events import export_events


def capture_profile(config, run_dir, engine, binary, models):
    check_config(config)
    collect_environment(config, engine, binary, run_dir)
    paths = verified_model_paths(config, models)
    command = profile_command(config, binary, paths, run_dir)
    write_json(run_dir / "command.json", {
        "argv": command,
        "cache_condition": "All weight files SHA256-read immediately before capture; no cache flush.",
    })
    memory_snapshot(run_dir / "memory-before.txt")
    monitor = TegrastatsMonitor(run_dir / "tegrastats.log", config["protocol"]["tegrastats_interval_ms"])
    with sampling(monitor):
        with (run_dir / "profile.log").open("w") as log:
            code = subprocess.call(command, stdout=log, stderr=subprocess.STDOUT,
                                   env={**os.environ, "HF_HUB_OFFLINE": "1"})
        if code:
            raise RuntimeError(f"Profiler/harness exit code {code}; see {run_dir / 'profile.log'}")
        save_profile_results(run_dir, config["workload"]["steps"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--engine", type=Path, default=Path.home() / "tools/stable-diffusion.cpp")
    parser.add_argument("--binary", type=Path, default=Path.home() / "tools/sd-bench-build/sd-bench-nvtx")
    parser.add_argument("--models", type=Path, default=Path.home() / "models/flux2-klein")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    run_dir = create_run_directory(config, Path.home() / "results/runs", "profile")
    try:
        with run_status(run_dir) as status:
            capture_profile(config, run_dir, args.engine, args.binary, args.models)
            export_events(run_dir, "sdcpp")
            status["trace_timeline_review_pending"] = True
    finally:
        memory_snapshot(run_dir / "memory-after.txt")
        print("Artifacts: " + str(run_dir), flush=True)


if __name__ == "__main__":
    main()
