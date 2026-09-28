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
import copy
import csv
import datetime as dt
import hashlib
import json
import os
import platform
import re
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchlib import GIB, REPO_ROOT, DeviceMemorySampler, now_utc, sh, summarize, write_json  # noqa: E402
from benchlib import peak_rss_gib as _peak_rss_gib  # noqa: E402


# --------------------------------------------------------------------------- prepare


def prepare(cfg, config_path):
    """Resolve the model revision to an immutable commit and download it. No GPU used."""
    from huggingface_hub import HfApi, snapshot_download

    model = cfg["model"]
    requested = model["revision"]
    sha = HfApi().model_info(model["repo_id"], revision=requested).sha
    print(f"Resolved {model['repo_id']}@{requested} -> {sha}")
    local_dir = snapshot_download(model["repo_id"], revision=sha, allow_patterns=model["allow_patterns"])
    print(f"Cached at {local_dir}")

    resolved = copy.deepcopy(cfg)
    resolved["model"]["requested_revision"] = requested
    resolved["model"]["revision"] = sha
    resolved["model"]["resolved_at_utc"] = now_utc()
    out = config_path.with_name(config_path.stem + ".resolved.json")
    write_json(out, resolved)
    print(f"Wrote {out}")


# --------------------------------------------------------------------------- stage instrumentation


class StageRecorder:
    """Marks pipeline stages with CUDA events, profiler labels and allocator peaks.

    Stages are delimited by wrapping pipe.encode_prompt, each transformer forward,
    pipe.vae.decode and pipe.image_processor.postprocess. The pipeline code itself
    is not modified. CUDA events are recorded on the current stream without
    synchronizing, so the instrumentation does not change the baseline's execution.

    Allocator peaks are tracked per segment: the peak counter is reset at every stage
    boundary, so both stages and the gaps between them ("between:a->b") get a peak,
    and the run peak is the max over all segments.
    """

    def __init__(self, torch):
        self.torch = torch
        self.reset()

    def reset(self):
        self.stages = []  # dicts: name, ev0, ev1, host0, host1, peak_alloc, peak_reserved, alloc0, alloc1
        self.segments = []  # dicts: name, peak_alloc, peak_reserved
        self._open = None
        self._last = "start"
        self._step = 0
        self.torch.cuda.reset_peak_memory_stats()

    def _close_segment(self, name):
        c = self.torch.cuda
        self.segments.append(
            {"name": name, "peak_alloc": c.max_memory_allocated() / GIB, "peak_reserved": c.max_memory_reserved() / GIB}
        )
        c.reset_peak_memory_stats()

    def start(self, name):
        from torch.profiler import record_function

        assert self._open is None, f"nested stage {name} inside {self._open['name']}"
        self._close_segment(f"between:{self._last}->{name}")
        ev0 = self.torch.cuda.Event(enable_timing=True)
        ev0.record()
        rf = record_function(name)
        rf.__enter__()
        self._open = {
            "name": name,
            "ev0": ev0,
            "host0": time.perf_counter(),
            "rf": rf,
            "alloc0": self.torch.cuda.memory_allocated() / GIB,
        }

    def end(self):
        s = self._open
        s["ev1"] = self.torch.cuda.Event(enable_timing=True)
        s["ev1"].record()
        s["host1"] = time.perf_counter()
        s["rf"].__exit__(None, None, None)
        s["alloc1"] = self.torch.cuda.memory_allocated() / GIB
        c = self.torch.cuda
        s["peak_alloc"] = c.max_memory_allocated() / GIB
        s["peak_reserved"] = c.max_memory_reserved() / GIB
        c.reset_peak_memory_stats()
        self.segments.append({"name": s["name"], "peak_alloc": s["peak_alloc"], "peak_reserved": s["peak_reserved"]})
        self.stages.append(s)
        self._last = s["name"]
        self._open = None

    def finish(self):
        """Call after torch.cuda.synchronize(). Returns per-stage rows and run peaks."""
        self._close_segment(f"between:{self._last}->end")
        rows = []
        for s in self.stages:
            rows.append(
                {
                    "stage": s["name"],
                    "gpu_ms": s["ev0"].elapsed_time(s["ev1"]),
                    "host_ms": (s["host1"] - s["host0"]) * 1000,
                    "alloc_start_gib": s["alloc0"],
                    "alloc_end_gib": s["alloc1"],
                    "peak_alloc_gib": s["peak_alloc"],
                    "peak_reserved_gib": s["peak_reserved"],
                }
            )
        peak_alloc = max(seg["peak_alloc"] for seg in self.segments)
        peak_reserved = max(seg["peak_reserved"] for seg in self.segments)
        return rows, peak_alloc, peak_reserved, list(self.segments)

    def next_step_name(self):
        name = f"denoise_step_{self._step}"
        self._step += 1
        return name

    def install(self, pipe):
        rec = self

        def wrap(obj, attr, name):
            orig = getattr(obj, attr)

            def wrapped(*a, **kw):
                rec.start(name)
                try:
                    return orig(*a, **kw)
                finally:
                    rec.end()

            setattr(obj, attr, wrapped)

        wrap(pipe, "encode_prompt", "text_encode")
        wrap(pipe.vae, "decode", "vae_decode")
        wrap(pipe.image_processor, "postprocess", "postprocess")
        pipe.transformer.register_forward_pre_hook(lambda m, a: rec.start(rec.next_step_name()))
        pipe.transformer.register_forward_hook(lambda m, a, o: rec.end())


# --------------------------------------------------------------------------- environment


def collect_environment(run_dir, sampler):
    import diffusers
    import torch
    import transformers

    props = torch.cuda.get_device_properties(0)
    env = {
        "timestamp_utc": now_utc(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "diffusers": diffusers.__version__,
        "transformers": transformers.__version__,
        "gpu_name": props.name,
        "gpu_total_memory_gib": props.total_memory / GIB,
        "gpu_uuid": str(props.uuid),
        "gpu_sm_count": props.multi_processor_count,
        "gpu_compute_capability": f"{props.major}.{props.minor}",
        "nvidia_driver": sh(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]),
        "sdp_backends_enabled": {
            "flash": torch.backends.cuda.flash_sdp_enabled(),
            "mem_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
            "math": torch.backends.cuda.math_sdp_enabled(),
        },
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "device_memory_sampler": sampler.note,
        "git_commit": sh(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"]),
        "git_dirty_files": sh(["git", "-C", str(REPO_ROOT), "status", "--porcelain"]).splitlines(),
        "env_vars": {
            k: os.environ.get(k)
            for k in [
                "SLURM_JOB_ID",
                "SLURM_JOB_PARTITION",
                "SLURM_JOB_ACCOUNT",
                "SLURM_JOB_NODELIST",
                "SLURM_CPUS_PER_TASK",
                "CUDA_VISIBLE_DEVICES",
                "HF_HOME",
                "HF_HUB_OFFLINE",
            ]
        },
    }
    write_json(run_dir / "environment.json", env)
    (run_dir / "environment-freeze.txt").write_text(sh([sys.executable, "-m", "pip", "freeze"]) + "\n")
    return env


# --------------------------------------------------------------------------- benchmark


def benchmark(cfg, run_dir, do_profile):
    import torch
    import diffusers

    status = {"status": "running", "started_utc": now_utc()}
    write_json(run_dir / "status.json", status)

    assert torch.cuda.is_available(), "No CUDA device visible. Run this on a GPU compute node."
    opt = cfg["optimizations"]
    assert not any(opt.values()), f"This runner implements the unoptimized baseline only; got {opt}"
    assert cfg["precision"] == "bfloat16"
    rev = cfg["model"]["revision"]
    assert re.fullmatch(r"[0-9a-f]{40}", rev), "Config revision is not a commit hash. Run --prepare-only first."

    uuid = "GPU-" + str(torch.cuda.get_device_properties(0).uuid)
    sampler = DeviceMemorySampler(cfg["protocol"]["device_memory_sample_ms"], uuid)
    sampler.start()
    collect_environment(run_dir, sampler)

    # ---- load
    import psutil

    proc = psutil.Process()
    torch.cuda.reset_peak_memory_stats()
    pipe_cls = getattr(diffusers, cfg["model"]["pipeline_class"])
    t0 = time.perf_counter()
    pipe = pipe_cls.from_pretrained(
        cfg["model"]["repo_id"], revision=rev, torch_dtype=torch.bfloat16, local_files_only=True
    )
    t1 = time.perf_counter()
    pipe = pipe.to("cuda")
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    pipe.set_progress_bar_config(disable=True)

    components = {}
    for name, comp in pipe.components.items():
        if isinstance(comp, torch.nn.Module):
            params = list(comp.parameters()) + list(comp.buffers())
            components[name] = {
                "class": type(comp).__name__,
                "num_params": sum(p.numel() for p in comp.parameters()),
                "weights_gib": sum(p.numel() * p.element_size() for p in params) / GIB,
                "dtypes": sorted({str(p.dtype) for p in params}),
                "device": str(next(comp.parameters()).device),
            }
    load = {
        "from_pretrained_s": t1 - t0,
        "to_cuda_s": t2 - t1,
        "load_total_s": t2 - t0,
        "note": "Cached load + placement; the OS page cache may already hold the files. Not a cold read.",
        "allocated_after_load_gib": torch.cuda.memory_allocated() / GIB,
        "reserved_after_load_gib": torch.cuda.memory_reserved() / GIB,
        "peak_alloc_during_load_gib": torch.cuda.max_memory_allocated() / GIB,
        "device_used_after_load_gib": sampler.read() / GIB if sampler.available else None,
        "host_rss_after_load_gib": proc.memory_info().rss / GIB,
        "components": components,
    }
    write_json(run_dir / "load.json", load)
    write_json(run_dir / "scheduler.json", {"class": type(pipe.scheduler).__name__, "config": dict(pipe.scheduler.config)})
    print(json.dumps(load, indent=2, default=str), flush=True)

    recorder = StageRecorder(torch)
    recorder.install(pipe)
    w = cfg["workload"]
    assert w["batch_size"] == 1

    def generate():
        from torch.profiler import record_function

        generator = torch.Generator(device="cuda").manual_seed(w["seed"])
        torch.cuda.synchronize()
        recorder.reset()
        ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        h0 = time.perf_counter()
        ev0.record()
        with record_function("generate"):
            image = pipe(
                prompt=w["prompt"],
                height=w["height"],
                width=w["width"],
                num_inference_steps=w["num_inference_steps"],
                guidance_scale=w["guidance_scale"],
                max_sequence_length=w["max_sequence_length"],
                generator=generator,
                output_type="pil",
            ).images[0]
        ev1.record()
        torch.cuda.synchronize()
        h1 = time.perf_counter()
        stages, peak_alloc, peak_reserved, segments = recorder.finish()
        return image, {
            "wall_ms": (h1 - h0) * 1000,
            "gpu_span_ms": ev0.elapsed_time(ev1),
            "t0": h0,
            "t1": h1,
            "stages": stages,
            "segments": segments,
            "peak_alloc_gib": peak_alloc,
            "peak_reserved_gib": peak_reserved,
        }

    p = cfg["protocol"]
    phases = ["first"] * p["first_runs"] + ["warmup"] * p["warmup_runs"] + ["measured"] * p["measured_runs"]
    (run_dir / "images").mkdir(exist_ok=True)
    steps = w["num_inference_steps"]
    step_cols = [f"denoise_step_{i}_ms" for i in range(steps)]
    fields = (
        ["run_index", "phase", "wall_ms", "gpu_span_ms", "text_encode_ms", "denoise_ms"]
        + step_cols
        + ["vae_decode_ms", "postprocess_ms", "other_ms", "peak_alloc_gib", "peak_reserved_gib",
           "device_used_peak_gib", "host_rss_gib", "image_sha256"]
    )
    stage_fields = ["run_index", "phase", "stage", "gpu_ms", "host_ms", "alloc_start_gib", "alloc_end_gib",
                    "peak_alloc_gib", "peak_reserved_gib"]
    rows = []
    seg_rows = []
    with open(run_dir / "runs.csv", "w", newline="") as f_runs, open(run_dir / "stages.csv", "w", newline="") as f_st:
        runs_w = csv.DictWriter(f_runs, fieldnames=fields)
        runs_w.writeheader()
        st_w = csv.DictWriter(f_st, fieldnames=stage_fields)
        st_w.writeheader()
        measured_idx = 0
        for i, phase in enumerate(phases):
            image, r = generate()
            by = {}
            for s in r["stages"]:
                by.setdefault(s["stage"], 0.0)
                by[s["stage"]] += s["gpu_ms"]
                st_w.writerow({"run_index": i, "phase": phase, **s})
            n_calls = sum(1 for s in r["stages"] if s["stage"].startswith("denoise_step_"))
            assert n_calls == steps, f"expected {steps} transformer calls, saw {n_calls} (CFG active?)"
            denoise = sum(by[f"denoise_step_{k}"] for k in range(steps))
            named = by["text_encode"] + denoise + by["vae_decode"] + by["postprocess"]
            row = {
                "run_index": i,
                "phase": phase,
                "wall_ms": r["wall_ms"],
                "gpu_span_ms": r["gpu_span_ms"],
                "text_encode_ms": by["text_encode"],
                "denoise_ms": denoise,
                **{f"denoise_step_{k}_ms": by[f"denoise_step_{k}"] for k in range(steps)},
                "vae_decode_ms": by["vae_decode"],
                "postprocess_ms": by["postprocess"],
                "other_ms": r["gpu_span_ms"] - named,
                "peak_alloc_gib": r["peak_alloc_gib"],
                "peak_reserved_gib": r["peak_reserved_gib"],
                "device_used_peak_gib": sampler.peak_between(r["t0"], r["t1"]),
                "host_rss_gib": proc.memory_info().rss / GIB,
                "image_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
            }
            runs_w.writerow(row)
            f_runs.flush()
            f_st.flush()
            rows.append(row)
            seg_rows.append({"run_index": i, "phase": phase, "segments": r["segments"]})
            if phase == "first":
                image.save(run_dir / "images" / "first.png")
            elif phase == "measured" and measured_idx == 0:
                image.save(run_dir / "images" / "measured-0.png")
            measured_idx += phase == "measured"
            print(f"[{i:2d} {phase:8s}] wall {row['wall_ms']:8.1f} ms  text {row['text_encode_ms']:7.1f}  "
                  f"denoise {row['denoise_ms']:8.1f}  vae {row['vae_decode_ms']:7.1f}  "
                  f"peak_alloc {row['peak_alloc_gib']:.2f} GiB", flush=True)
    write_json(run_dir / "memory_segments.json", seg_rows)
    (run_dir / "nvidia-smi.txt").write_text(sh(["nvidia-smi"]) + "\n")

    measured = [r for r in rows if r["phase"] == "measured"]
    metric_cols = [c for c in fields if c.endswith("_ms") or c.endswith("_gib")]
    summary = {
        "id": cfg["id"],
        "device_label": cfg["device_label"],
        "gpu_name": torch.cuda.get_device_name(0),
        "measured_runs": len(measured),
        "first_run": {c: rows[0][c] for c in metric_cols},
        "measured": {c: summarize([r[c] for r in measured]) for c in metric_cols},
        "load": {k: load[k] for k in ["load_total_s", "allocated_after_load_gib", "peak_alloc_during_load_gib"]},
        "weights_gib": {k: v["weights_gib"] for k, v in components.items()},
        "deterministic_output": len({r["image_sha256"] for r in measured}) == 1,
        "host_peak_rss_gib": _peak_rss_gib(),
        "units": {"time": "ms (load in s)", "memory": "GiB (2^30 bytes)"},
    }
    write_json(run_dir / "summary.json", summary)

    if do_profile:
        run_profile(generate, run_dir)

    sampler.stop()
    status.update({"status": "complete", "finished_utc": now_utc()})
    write_json(run_dir / "status.json", status)
    print(json.dumps(summary["measured"]["wall_ms"], indent=2))


def run_profile(generate, run_dir):
    """One extra generation under torch.profiler, after (and excluded from) the baseline stats."""
    from torch.profiler import ProfilerActivity, profile

    prof_dir = run_dir / "profile"
    prof_dir.mkdir(exist_ok=True)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True,
                 profile_memory=True) as prof:
        _, r = generate()
    prof.export_chrome_trace(str(prof_dir / "trace.json"))
    table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=40)
    (prof_dir / "op_table.txt").write_text(table + "\n")
    write_json(prof_dir / "profiled_run.json", {k: v for k, v in r.items() if k not in ("t0", "t1")})
    print(table)


# --------------------------------------------------------------------------- main


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

    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    # Run directory naming (wiki/methods/baseline-metrics.md#run-naming): <experiment-id>__<kind>__<stamp>
    kind = args.kind or ("profile" if args.profile else "baseline")
    run_dir = args.out_root / f"{cfg['id']}__{kind}__{stamp}"
    run_dir.mkdir(parents=True)
    write_json(run_dir / "config.json", cfg)
    print(f"Run directory: {run_dir}", flush=True)
    try:
        benchmark(cfg, run_dir, args.profile)
    except BaseException as e:
        write_json(run_dir / "status.json", {"status": "failed", "finished_utc": now_utc(), "error": repr(e),
                                             "traceback": traceback.format_exc()})
        print(f"FAILED; partial results kept in {run_dir}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
