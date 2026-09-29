#!/usr/bin/env python
"""stable-diffusion.cpp runner: same protocol and output files as run_flux.py.

Usage:
  # 1. On the login node (no GPU): download the single-file diffusion model at the pinned
  #    revision and write the resolved config.
  python scripts/run_sdcpp.py --config configs/a100-sdcpp-flux-klein-bf16.json --prepare-only

  # 2. On a GPU node (via scripts/gilbreth-sdcpp.slurm): run the baseline.
  python scripts/run_sdcpp.py --config configs/a100-sdcpp-flux-klein-bf16.resolved.json [--nsys]

The model is loaded once by the C++ harness (engines/sdcpp/bench.cpp), which then runs
first + warm-up + measured generations and reports host timestamps at sd.cpp's stage callbacks.
This wrapper never imports torch or creates a CUDA context, so the NVML device-wide memory it
samples belongs to sd.cpp alone. Metric definitions, including how sd.cpp stage times differ
from the PyTorch runner's CUDA-event times: wiki/methods/baseline-metrics.md.
"""

import argparse
import hashlib
import json
import resource
import shutil
import subprocess
from pathlib import Path

from benchlib import GIB, REPO_ROOT, DeviceMemorySampler, host_environment, sh, visible_gpu_uuid, write_json
from measurement import create_run_directory, phase_sequence, run_status, sampling, summarize_runs, write_csv
from measurement_events import export_events
from sdcpp_engine import (
    audit_log, check_config, check_files, expand, harness_args, prepare,
    rows_from_results, snapshot_dir,
)


def benchmark(cfg, run_dir, nsys):
    """Validate → execute the C++ protocol → decode observations → audit and summarize."""
    phases = phase_sequence(cfg["protocol"])
    engine_git = check_config(cfg)
    snap = snapshot_dir(cfg)
    check_files(cfg, snap)
    build = expand(cfg["engine"]["bench_build_dir"])
    binary = build / ("sd-bench-nvtx" if nsys else "sd-bench")
    assert binary.exists(), f"{binary} missing: build engines/sdcpp (wiki/methods/sdcpp-howto.md)"

    uuid = visible_gpu_uuid()
    sampler = DeviceMemorySampler(cfg["protocol"]["device_memory_sample_ms"], uuid)
    env = collect_environment(cfg, run_dir, sampler, engine_git, binary, snap)
    eng_dir, load, gens, child_peak_rss = run_harness(cfg, run_dir, binary, snap, phases, sampler, nsys)
    save_images(run_dir, eng_dir, gens, phases)
    rows, stage_rows = rows_from_results(gens, phases, cfg["workload"]["num_inference_steps"], sampler)
    write_csv(run_dir / "runs.csv", rows)
    write_csv(run_dir / "stages.csv", stage_rows)
    save_summary(cfg, run_dir, env, eng_dir, load, gens, rows, sampler, child_peak_rss)


def collect_environment(cfg, run_dir, sampler, engine_git, binary, snap):
    env = host_environment(extra_env_vars=["GGML_CUDA_CUBLAS_COMPUTE_TYPE", "LD_LIBRARY_PATH"])
    env.update({
        "device_memory_sampler": sampler.note,
        "engine": {**cfg["engine"], **engine_git, "binary": str(binary),
                   "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest()},
        "nvcc": sh(["nvcc", "--version"]).splitlines()[-1:],
        "ldd": sh(["ldd", str(binary)]).splitlines(),
        "snapshot_dir": str(snap),
    })
    write_json(run_dir / "environment.json", env)

    return env


def run_harness(cfg, run_dir, binary, snap, phases, sampler, nsys):
    """One child process loads the model once and performs the entire protocol."""
    eng_dir = run_dir / "engine"
    eng_dir.mkdir()
    cmd = [str(binary), *harness_args(cfg, snap, eng_dir, len(phases))]
    if nsys:
        (run_dir / "profile").mkdir()
        cmd = ["nsys", "profile", "--trace=cuda,nvtx", "--sample=none", "--cpuctxsw=none",
               "--force-overwrite=true", "-o", str(run_dir / "profile" / "trace"), *cmd]
    write_json(eng_dir / "command.json", cmd)
    print("Running:", " ".join(cmd), flush=True)

    with sampling(sampler):
        with open(eng_dir / "stdout.txt", "w") as out:
            proc = subprocess.run(cmd, stdout=out, stderr=subprocess.STDOUT)
    child_peak_rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * 1024 / GIB

    # A loader failure can happen before the harness creates any result files.
    # Report the child error first, rather than hiding it behind FileNotFoundError.
    if proc.returncode != 0:
        detail = (eng_dir / "stdout.txt").read_text(errors="replace")[-4096:]
        raise RuntimeError(f"harness exited with {proc.returncode}; see {eng_dir}/stdout.txt\n{detail}")
    results_path = eng_dir / "results.jsonl"
    if not results_path.is_file():
        raise RuntimeError(f"harness exited successfully but did not write {results_path}; see {eng_dir}/stdout.txt")
    results = [json.loads(ln) for ln in results_path.read_text().splitlines() if ln.strip()]
    load = next((r for r in results if r["event"] == "load"), None)
    gens = [r for r in results if r["event"] == "generate"]
    assert load and load["ok"], "model load failed"
    assert len(gens) == len(phases), f"expected {len(phases)} generations, got {len(gens)}"

    return eng_dir, load, gens, child_peak_rss


def save_images(run_dir, eng_dir, gens, phases):
    # Hash every image; keep PNGs of the first and first measured run, then drop the raw bytes.
    from PIL import Image

    (run_dir / "images").mkdir()
    first_measured = phases.index("measured")
    for g in gens:
        raw = eng_dir / "raw" / f"run-{g['run_index']}.rgb"
        data = raw.read_bytes()
        g["image_sha256"] = hashlib.sha256(data).hexdigest()
        name = {0: "first.png", first_measured: "measured-0.png"}.get(g["run_index"])
        if name:
            Image.frombytes("RGB", (g["width"], g["height"]), data).save(run_dir / "images" / name)
    shutil.rmtree(eng_dir / "raw")



def save_summary(cfg, run_dir, env, eng_dir, load, gens, rows, sampler, child_peak_rss):
    load_rec = {
        "load_total_s": load["t_end"] - load["t_start"],
        "note": "new_sd_ctx with eager load: file read + placement on the GPU. Page cache may hold the files; not a cold read.",
        "device_used_after_load_gib": sampler.peak_between(load["t_end"] - 0.5, load["t_end"] + 0.5),
        "device_used_peak_during_load_gib": sampler.peak_between(load["t_start"], load["t_end"]),
        "model_version": load.get("model_version"),
    }
    write_json(run_dir / "load.json", load_rec)
    (run_dir / "nvidia-smi.txt").write_text(sh(["nvidia-smi"]) + "\n")

    audit = audit_log((eng_dir / "sdcpp.log").read_text(errors="replace"), cfg["engine_settings"], gens[-1]["t_end"])
    summary = {
        "id": cfg["id"],
        "device_label": cfg["device_label"],
        "engine": f"stable-diffusion.cpp {cfg['engine']['tag']}",
        "gpu": env["nvidia_smi_gpu"],
        **summarize_runs(rows),
        "load": load_rec,
        "host_peak_rss_gib": child_peak_rss,
        "timing_method": "host steady_clock at sd.cpp log/progress callbacks (ggml graph compute is synchronous)",
        "unavailable_metrics": {
            "gpu_span_ms": "no CUDA events inside sd.cpp; wall_ms is the end-to-end time",
            "postprocess_ms": "sd.cpp returns uint8 RGB from decode; no separate postprocess stage",
            "peak_alloc_gib/peak_reserved_gib": "PyTorch allocator counters; ggml has no equivalent",
            "host_rss_gib": "per-run RSS not sampled; host_peak_rss_gib is the harness process peak",
        },
        "engine_audit": audit,
        "units": {"time": "ms (load in s)", "memory": "GiB (2^30 bytes)"},
    }
    write_json(run_dir / "summary.json", summary)
    assert not audit["problems"], f"engine audit failed: {audit['problems']}"

    print(json.dumps(summary["measured"]["wall_ms"], indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--prepare-only", action="store_true", help="download + pin file hashes; no GPU")
    ap.add_argument("--nsys", action="store_true", help="run the harness under Nsight Systems (separate profile run)")
    ap.add_argument("--kind", choices=["baseline", "profile", "repeat", "attempt"], default=None,
                    help="run kind in the directory name (default: profile with the profiler flag, else baseline)")
    ap.add_argument("--out-root", type=Path, default=REPO_ROOT / "results" / "runs")
    args = ap.parse_args()

    cfg = json.loads(args.config.read_text())
    if args.prepare_only:
        prepare(cfg, args.config)
        return

    kind = args.kind or ("profile" if args.nsys else "baseline")
    run_dir = create_run_directory(cfg, args.out_root, kind)
    with run_status(run_dir):
        benchmark(cfg, run_dir, args.nsys)
        export_events(run_dir, "sdcpp")


if __name__ == "__main__":
    main()
