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
import copy
import csv
import hashlib
import json
import os
import re
import resource
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchlib import (  # noqa: E402
    GIB,
    REPO_ROOT,
    DeviceMemorySampler,
    host_environment,
    now_utc,
    sh,
    summarize,
    visible_gpu_uuid,
    write_json,
)

HEX40 = re.compile(r"[0-9a-f]{40}")
# The reference configuration may not use any of these; the log audit fails the run if it does.
FORBIDDEN_LOG_PATTERNS = {
    "auto_fit_plan": r"auto-fit plan:",
    "conditioning_cache_hit": r"conditioning cache hit",
    "graph_cut_layer_split": r"graph-cut layer split",
}
# Weights leaving the params backend is only a problem while generations are still running
# (sd.cpp also logs a release when the context is freed at exit).
RELEASE_PATTERN = r"model manager releas"


def expand(p):
    return Path(os.path.expanduser(p))


def snapshot_dir(cfg):
    """The pinned snapshot in the local HF cache (only the files in model.files are present)."""
    from huggingface_hub.constants import HF_HUB_CACHE

    m = cfg["model"]
    snap = Path(HF_HUB_CACHE) / f"models--{m['repo_id'].replace('/', '--')}" / "snapshots" / m["revision"]
    assert snap.is_dir(), f"{snap} missing: run --prepare-only on the login node"
    return snap


# --------------------------------------------------------------------------- prepare


def prepare(cfg, config_path):
    """Download the model files at the pinned revision; record content hashes. No GPU used."""
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download

    m = cfg["model"]
    assert HEX40.fullmatch(m["revision"]), "model.revision must be a commit hash"
    info = HfApi().model_info(m["repo_id"], revision=m["revision"], files_metadata=True)
    lfs = {s.rfilename: s.lfs.sha256 for s in info.siblings if s.lfs}
    patterns = []
    for rel in m["files"].values():
        patterns += [rel, f"{rel}/*"]
    snapshot_download(m["repo_id"], revision=m["revision"], allow_patterns=patterns)
    for rel in m["files"].values():
        if rel.endswith(".safetensors"):
            hf_hub_download(m["repo_id"], rel, revision=m["revision"])
    resolved = copy.deepcopy(cfg)
    resolved["model"]["sha256"] = {
        f: h for f, h in sorted(lfs.items()) if any(f == r or f.startswith(r + "/") for r in m["files"].values())
    }
    resolved["model"]["resolved_at_utc"] = now_utc()
    out = config_path.with_name(config_path.stem + ".resolved.json")
    write_json(out, resolved)
    print(f"Wrote {out}")


# --------------------------------------------------------------------------- checks


def check_config(cfg):
    e, m, s = cfg["engine"], cfg["model"], cfg["engine_settings"]
    assert e["name"] == "stable-diffusion.cpp", e["name"]
    assert HEX40.fullmatch(e["commit"]) and HEX40.fullmatch(e["ggml_commit"]), "engine commits must be hashes"
    assert HEX40.fullmatch(m["revision"]), "model.revision must be a commit hash"
    assert "resolved_at_utc" in m, "use the *.resolved.json config (run --prepare-only first)"
    assert not any(cfg["optimizations"].values()), f"reference runner: optimizations must be off, got {cfg['optimizations']}"
    assert cfg["precision"] == "bfloat16"
    assert cfg["workload"]["batch_size"] == 1
    # Reference placement: everything on one GPU, nothing automatic.
    assert s["backend"] == s["params_backend"] and s["backend"].startswith("cuda"), s
    assert s["auto_fit"] is False and s["eager_load"] is True and s["disable_segmented_compute"] is True, s
    assert s["conditioning_cache_size"] == 0, "conditioning cache would skip text encoding in warm runs"
    assert os.environ.get("GGML_CUDA_CUBLAS_COMPUTE_TYPE") is None, "GGML_CUDA_CUBLAS_COMPUTE_TYPE overrides GEMM precision"

    src = expand(e["source_dir"])
    head = sh(["git", "-C", str(src), "rev-parse", "HEAD"])
    ggml = sh(["git", "-C", str(src / "ggml"), "rev-parse", "HEAD"])
    assert head == e["commit"], f"sd.cpp checkout is at {head}, config pins {e['commit']}"
    assert ggml == e["ggml_commit"], f"ggml checkout is at {ggml}, config pins {e['ggml_commit']}"
    return {"sdcpp_head": head, "ggml_head": ggml, "sdcpp_status": sh(["git", "-C", str(src), "status", "--porcelain"])}


def check_files(cfg, snap):
    """Confirm the snapshot files are the pinned ones (HF cache blobs are named by sha256)."""
    for rel, sha in cfg["model"]["sha256"].items():
        blob = Path(os.path.realpath(snap / rel)).name
        assert blob == sha, f"{rel}: cached blob {blob} does not match pinned sha256 {sha}"


def audit_log(log_text, settings, t_last_generation_end):
    """Extract what sd.cpp actually did, and fail on anything the reference config forbids.

    Log lines from the harness start with a steady_clock timestamp (s).
    """
    lines = log_text.splitlines()
    found = {k: [ln for ln in lines if re.search(p, ln)] for k, p in FORBIDDEN_LOG_PATTERNS.items()}

    def ts(ln):
        m = re.match(r"(\d+\.\d+) ", ln)
        return float(m.group(1)) if m else None

    early_release = [ln for ln in lines if re.search(RELEASE_PATTERN, ln) and (ts(ln) or 0) < t_last_generation_end]
    if early_release:
        found["params_released_during_generation"] = early_release
    segments = [int(x) for x in re.findall(r"(?:using|across) (\d+) segments?", log_text)]
    placements = [ln for ln in lines if "prepared params backend buffers" in ln]
    audit = {
        "weight_type_stats": [ln.split("] ", 1)[-1] for ln in lines if "eight type stat" in ln],
        "flash_attention": [ln.split("] ", 1)[-1] for ln in lines if "Using flash attention" in ln],
        "params_placement": [ln.split("] ", 1)[-1] for ln in placements],
        "max_graph_segments": max(segments) if segments else None,
        "sampler_lines": [ln.split("] ", 1)[-1] for ln in lines if re.search(r"sampl(e|ing) method|scheduler|sigmas", ln)][:10],
        "version_lines": [ln.split("] ", 1)[-1] for ln in lines if re.search(r"Version:|model version", ln)][:3],
        "forbidden": {k: v[:5] for k, v in found.items() if v},
    }
    problems = list(audit["forbidden"])
    if segments and max(segments) > 1:
        problems.append(f"graph ran in {max(segments)} segments")
    backend = settings["params_backend"].upper()
    off_gpu = [p for p in placements if not re.search(rf"ON {backend}\b", p.upper())]
    if off_gpu:
        problems.append(f"params placed off {settings['params_backend']}: {off_gpu[:3]}")
    if settings["diffusion_fa"] and not audit["flash_attention"]:
        problems.append("diffusion flash attention requested but not reported by sd.cpp")
    audit["problems"] = problems
    return audit


# --------------------------------------------------------------------------- benchmark


def harness_args(cfg, snap, out_dir, runs):
    w, s, f = cfg["workload"], cfg["engine_settings"], cfg["model"]["files"]
    b = lambda x: "1" if x else "0"  # noqa: E731
    args = {
        "diffusion-model": snap / f["diffusion_model"],
        "llm": snap / f["llm"],
        "vae": snap / f["vae"],
        "prompt": w["prompt"],
        "width": w["width"],
        "height": w["height"],
        "steps": w["num_inference_steps"],
        "cfg-scale": w["guidance_scale"],
        "seed": w["seed"],
        "runs": runs,
        "out-dir": out_dir,
        "threads": s["threads"],
        "mmap": b(s["mmap"]),
        "fa": b(s["fa"]),
        "diffusion-fa": b(s["diffusion_fa"]),
        "backend": s["backend"],
        "params-backend": s["params_backend"],
        "auto-fit": b(s["auto_fit"]),
        "eager-load": b(s["eager_load"]),
        "disable-segmented-compute": b(s["disable_segmented_compute"]),
        "disable-prefetch": b(s["disable_prefetch"]),
        "conditioning-cache-size": s["conditioning_cache_size"],
    }
    return [x for k, v in args.items() for x in (f"--{k}", str(v))]


def rows_from_results(results, phases, steps, sampler):
    rows, stage_rows = [], []
    for g in results:
        i = g["run_index"]
        ms = lambda a, b: (b - a) * 1000.0  # noqa: E731
        step_ms = []
        prev = g["t_sampling_start"]
        for t in g["t_step_end"]:
            step_ms.append(ms(prev, t))
            prev = t
        assert g["ok"], f"generation {i} failed; see engine/sdcpp.log"
        assert len(step_ms) == steps, f"run {i}: expected {steps} denoise steps, saw {len(step_ms)}"
        assert g["cond_cache_hits"] == 0, f"run {i}: conditioning cache hit"
        wall = ms(g["t_start"], g["t_end"])
        stages = {
            "text_encode": ms(g["t_start"], g["t_cond"]),
            **{f"denoise_step_{k}": v for k, v in enumerate(step_ms)},
            "vae_decode": ms(g["t_sampling_end"], g["t_decode_end"]),
        }
        named = sum(stages.values())
        row = {
            "run_index": i,
            "phase": phases[i],
            "wall_ms": wall,
            "gpu_span_ms": None,
            "text_encode_ms": stages["text_encode"],
            "denoise_ms": sum(step_ms),
            **{f"denoise_step_{k}_ms": v for k, v in enumerate(step_ms)},
            "vae_decode_ms": stages["vae_decode"],
            "postprocess_ms": None,
            "other_ms": wall - named,
            "peak_alloc_gib": None,
            "peak_reserved_gib": None,
            "device_used_peak_gib": sampler.peak_between(g["t_start"], g["t_end"]),
            "host_rss_gib": None,
            "image_sha256": g.get("image_sha256"),
        }
        rows.append(row)
        starts = {"text_encode": g["t_start"], "vae_decode": g["t_sampling_end"]}
        t = g["t_sampling_start"]
        for k, v in enumerate(step_ms):
            starts[f"denoise_step_{k}"] = t
            t += v / 1000.0
        for name, v in stages.items():
            stage_rows.append({
                "run_index": i, "phase": phases[i], "stage": name, "gpu_ms": None, "host_ms": v,
                "alloc_start_gib": None, "alloc_end_gib": None,
                "device_used_peak_gib": sampler.peak_between(starts[name], starts[name] + v / 1000.0),
            })
    return rows, stage_rows


def benchmark(cfg, run_dir, nsys):
    status = {"status": "running", "started_utc": now_utc()}
    write_json(run_dir / "status.json", status)

    engine_git = check_config(cfg)
    snap = snapshot_dir(cfg)
    check_files(cfg, snap)
    build = expand(cfg["engine"]["bench_build_dir"])
    binary = build / ("sd-bench-nvtx" if nsys else "sd-bench")
    assert binary.exists(), f"{binary} missing: build engines/sdcpp (wiki/methods/sdcpp-howto.md)"

    uuid = visible_gpu_uuid()
    sampler = DeviceMemorySampler(cfg["protocol"]["device_memory_sample_ms"], uuid)
    env = host_environment(extra_env_vars=["GGML_CUDA_CUBLAS_COMPUTE_TYPE"])
    env.update({
        "device_memory_sampler": sampler.note,
        "engine": {**cfg["engine"], **engine_git, "binary": str(binary),
                   "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest()},
        "nvcc": sh(["nvcc", "--version"]).splitlines()[-1:],
        "ldd": sh(["ldd", str(binary)]).splitlines(),
        "snapshot_dir": str(snap),
    })
    write_json(run_dir / "environment.json", env)

    p = cfg["protocol"]
    phases = ["first"] * p["first_runs"] + ["warmup"] * p["warmup_runs"] + ["measured"] * p["measured_runs"]
    steps = cfg["workload"]["num_inference_steps"]
    eng_dir = run_dir / "engine"
    eng_dir.mkdir()
    cmd = [str(binary), *harness_args(cfg, snap, eng_dir, len(phases))]
    if nsys:
        (run_dir / "profile").mkdir()
        cmd = ["nsys", "profile", "--trace=cuda,nvtx", "--sample=none", "--cpuctxsw=none",
               "--force-overwrite=true", "-o", str(run_dir / "profile" / "trace"), *cmd]
    write_json(eng_dir / "command.json", cmd)
    print("Running:", " ".join(cmd), flush=True)

    sampler.start()
    with open(eng_dir / "stdout.txt", "w") as out:
        proc = subprocess.run(cmd, stdout=out, stderr=subprocess.STDOUT)
    sampler.stop()
    child_peak_rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * 1024 / GIB

    results = [json.loads(ln) for ln in (eng_dir / "results.jsonl").read_text().splitlines() if ln.strip()]
    load = next((r for r in results if r["event"] == "load"), None)
    gens = [r for r in results if r["event"] == "generate"]
    assert proc.returncode == 0, f"harness exited with {proc.returncode}; see {eng_dir}/stdout.txt and sdcpp.log"
    assert load and load["ok"], "model load failed"
    assert len(gens) == len(phases), f"expected {len(phases)} generations, got {len(gens)}"

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

    rows, stage_rows = rows_from_results(gens, phases, steps, sampler)
    fields = list(rows[0])
    with open(run_dir / "runs.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        wr.writerows(rows)
    with open(run_dir / "stages.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(stage_rows[0]))
        wr.writeheader()
        wr.writerows(stage_rows)

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
    measured = [r for r in rows if r["phase"] == "measured"]
    metric_cols = [c for c in fields if c.endswith("_ms") or c.endswith("_gib")]
    summary = {
        "id": cfg["id"],
        "device_label": cfg["device_label"],
        "engine": f"stable-diffusion.cpp {cfg['engine']['tag']}",
        "gpu": env["nvidia_smi_gpu"],
        "measured_runs": len(measured),
        "first_run": {c: rows[0][c] for c in metric_cols},
        "measured": {c: summarize([r[c] for r in measured]) for c in metric_cols},
        "load": load_rec,
        "deterministic_output": len({r["image_sha256"] for r in measured}) == 1,
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

    status.update({"status": "complete", "finished_utc": now_utc()})
    write_json(run_dir / "status.json", status)
    print(json.dumps(summary["measured"]["wall_ms"], indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--prepare-only", action="store_true", help="download + pin file hashes; no GPU")
    ap.add_argument("--nsys", action="store_true", help="run the harness under Nsight Systems (separate profile run)")
    ap.add_argument("--out-root", type=Path, default=REPO_ROOT / "results" / "runs")
    args = ap.parse_args()

    cfg = json.loads(args.config.read_text())
    if args.prepare_only:
        prepare(cfg, args.config)
        return

    import datetime as dt

    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = args.out_root / f"{cfg['id']}-{stamp}{'-profile' if args.nsys else ''}"
    run_dir.mkdir(parents=True)
    write_json(run_dir / "config.json", cfg)
    print(f"Run directory: {run_dir}", flush=True)
    try:
        benchmark(cfg, run_dir, args.nsys)
    except BaseException as e:
        write_json(run_dir / "status.json", {"status": "failed", "finished_utc": now_utc(), "error": repr(e),
                                             "traceback": traceback.format_exc()})
        print(f"FAILED; partial results kept in {run_dir}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
