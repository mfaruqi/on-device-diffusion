#!/usr/bin/env python
"""FLUX.2 [klein] baseline runner: end-to-end + per-stage latency and memory.

Usage:
  # 1. On the login node (no GPU): resolve the revision and download weights.
  python scripts/run_flux.py --config configs/a100-flux-klein-bf16.json --prepare-only

  # 2. On a GPU node (via scripts/gilbreth.slurm): run the baseline.
  python scripts/run_flux.py --config configs/a100-flux-klein-bf16.resolved.json [--profile]

Metric definitions are documented in wiki/methods/baseline-metrics.md.
"""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

from benchlib import GIB, REPO_ROOT, DeviceMemorySampler, peak_rss_gib, sh, write_json
from flux_engine import check_config, collect_environment, generate_once, load_pipeline, prepare, run_profile
from measurement import create_run_directory, phase_sequence, run_fields, run_status, sampling, summarize_runs
from measurement_events import export_events
from torch_stages import StageRecorder


def benchmark(cfg, run_dir, do_profile):
    """Read top to bottom: validate → load → measure → summarize → optional trace."""
    import psutil
    import torch

    check_config(cfg)
    phases = phase_sequence(cfg["protocol"])
    assert torch.cuda.is_available(), "No CUDA device visible. Run this on a GPU compute node."
    uuid = "GPU-" + str(torch.cuda.get_device_properties(0).uuid)
    sampler = DeviceMemorySampler(cfg["protocol"]["device_memory_sample_ms"], uuid)
    proc = psutil.Process()

    with sampling(sampler):
        collect_environment(run_dir, sampler)
        pipe, load = load_pipeline(cfg, run_dir, sampler, proc)
        recorder = StageRecorder(torch)
        recorder.install(pipe)

        def generate():
            return generate_once(pipe, recorder, cfg["workload"])

        rows = measure_generations(generate, phases, cfg["workload"]["num_inference_steps"],
                                   run_dir, sampler, proc)
        (run_dir / "nvidia-smi.txt").write_text(sh(["nvidia-smi"]) + "\n")
        summary = build_summary(cfg, rows, load)
        write_json(run_dir / "summary.json", summary)
        if do_profile:
            run_profile(generate, run_dir)

    print(json.dumps(summary["measured"]["wall_ms"], indent=2))


def result_row(run_index, phase, image, result, steps, sampler, process):
    """Turn CUDA stage observations into one runs.csv row; do no timing here."""
    stage_gpu_ms = {}
    for stage in result["stages"]:
        stage_gpu_ms.setdefault(stage["stage"], 0.0)
        stage_gpu_ms[stage["stage"]] += stage["gpu_ms"]
    calls = Counter(stage["stage"] for stage in result["stages"])
    per_step = result["calls_per_step"]
    assert per_step in (1, 2), f"unsupported klein transformer calls per step: {per_step}"
    assert result["scheduler_steps"] == steps, f"expected {steps} completed scheduler steps, saw {result['scheduler_steps']}"
    expected = {f"denoise_step_{k}": per_step for k in range(steps)}
    observed = {name: n for name, n in calls.items() if name.startswith("denoise_step_")}
    assert observed == expected, f"transformer calls per logical step: expected {expected}, saw {observed}"
    assert calls["text_encode"] == per_step, f"expected {per_step} klein text encodes, saw {calls['text_encode']}"
    denoise = sum(stage_gpu_ms[f"denoise_step_{k}"] for k in range(steps))
    named = stage_gpu_ms["text_encode"] + denoise + stage_gpu_ms["vae_decode"] + stage_gpu_ms["postprocess"]
    row = {
        "run_index": run_index,
        "phase": phase,
        "wall_ms": result["wall_ms"],
        "gpu_span_ms": result["gpu_span_ms"],
        "text_encode_ms": stage_gpu_ms["text_encode"],
        "denoise_ms": denoise,
        **{f"denoise_step_{k}_ms": stage_gpu_ms[f"denoise_step_{k}"] for k in range(steps)},
        "vae_decode_ms": stage_gpu_ms["vae_decode"],
        "postprocess_ms": stage_gpu_ms["postprocess"],
        "other_ms": result["gpu_span_ms"] - named,
        "peak_alloc_gib": result["peak_alloc_gib"],
        "peak_reserved_gib": result["peak_reserved_gib"],
        "device_used_peak_gib": sampler.peak_between(result["t0"], result["t1"]),
        "host_rss_gib": process.memory_info().rss / GIB,
        "image_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
    }
    return row


def measure_generations(generate, phases, steps, run_dir, sampler, process):
    """Run the protocol and flush each finished generation before starting another."""
    (run_dir / "images").mkdir(exist_ok=True)
    fields = run_fields(steps)
    stage_fields = ["run_index", "phase", "stage", "gpu_ms", "host_ms", "alloc_start_gib", "alloc_end_gib",
                    "peak_alloc_gib", "peak_reserved_gib"]
    rows = []
    segment_rows = []
    with open(run_dir / "runs.csv", "w", newline="") as runs_file, open(run_dir / "stages.csv", "w", newline="") as stages_file:
        run_writer = csv.DictWriter(runs_file, fieldnames=fields)
        run_writer.writeheader()
        stage_writer = csv.DictWriter(stages_file, fieldnames=stage_fields)
        stage_writer.writeheader()
        measured_idx = 0
        for run_index, phase in enumerate(phases):
            image, result = generate()
            row = result_row(run_index, phase, image, result, steps, sampler, process)
            for stage in result["stages"]:
                stage_writer.writerow({"run_index": run_index, "phase": phase, **stage})
            run_writer.writerow(row)
            runs_file.flush()
            stages_file.flush()
            rows.append(row)
            segment_rows.append({"run_index": run_index, "phase": phase, "segments": result["segments"]})
            if phase == "first":
                image.save(run_dir / "images" / "first.png")
            elif phase == "measured" and measured_idx == 0:
                image.save(run_dir / "images" / "measured-0.png")
            measured_idx += phase == "measured"
            print(f"[{run_index:2d} {phase:8s}] wall {row['wall_ms']:8.1f} ms  text {row['text_encode_ms']:7.1f}  "
                  f"denoise {row['denoise_ms']:8.1f}  vae {row['vae_decode_ms']:7.1f}  "
                  f"peak_alloc {row['peak_alloc_gib']:.2f} GiB", flush=True)
    write_json(run_dir / "memory_segments.json", segment_rows)

    return rows


def build_summary(cfg, rows, load):
    import torch

    summary = {
        "id": cfg["id"],
        "device_label": cfg["device_label"],
        "gpu_name": torch.cuda.get_device_name(0),
        "guidance_scale": cfg["workload"]["guidance_scale"],
        "transformer_calls_per_step": 2 if cfg["workload"]["guidance_scale"] > 1 and not load["is_distilled"] else 1,
        **summarize_runs(rows),
        "load": {k: load[k] for k in ["load_total_s", "allocated_after_load_gib", "peak_alloc_during_load_gib"]},
        "weights_gib": {k: v["weights_gib"] for k, v in load["components"].items()},
        "host_peak_rss_gib": peak_rss_gib(),
        "units": {"time": "ms (load in s)", "memory": "GiB (2^30 bytes)"},
    }
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--prepare-only", action="store_true", help="resolve revision + download; no GPU")
    ap.add_argument("--profile", action="store_true", help="add one torch.profiler generation after the baseline")
    ap.add_argument("--kind", choices=["baseline", "profile", "repeat", "attempt"], default=None,
                    help="run kind in the directory name (default: profile with the profiler flag, else baseline)")
    ap.add_argument("--out-root", type=Path, default=REPO_ROOT / "results" / "runs")
    args = ap.parse_args()

    cfg = json.loads(args.config.read_text())
    if args.prepare_only:
        prepare(cfg, args.config)
        return

    kind = args.kind or ("profile" if args.profile else "baseline")
    run_dir = create_run_directory(cfg, args.out_root, kind)
    with run_status(run_dir):
        benchmark(cfg, run_dir, args.profile)
        export_events(run_dir, "pytorch")


if __name__ == "__main__":
    main()
