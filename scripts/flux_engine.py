"""PyTorch/Diffusers operations used by run_flux.py.

Preparation may download weights. Loading and generation use the local cache only.
Timing and placement are explicit here; shared bookkeeping lives in measurement.py.
"""

import copy
import json
import os
import platform
import re
import socket
import sys
import time

from benchlib import GIB, REPO_ROOT, now_utc, sh, write_json
from measurement import check_optimizations


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


def check_config(cfg):
    """Reject unsupported requests before loading weights."""
    check_optimizations(cfg, {"quantization", "cpu_offload", "vae_tiling", "vae_slicing", "torch_compile"})
    assert cfg["precision"] == "bfloat16"
    assert cfg["workload"]["batch_size"] == 1
    assert cfg["model"]["pipeline_class"] == "Flux2KleinPipeline", "Stage accounting supports Flux2KleinPipeline only"
    w = cfg["workload"]
    assert type(w["num_inference_steps"]) is int and w["num_inference_steps"] > 0
    if w["guidance_scale"] > 1:
        assert "negative_prompt" in w, 'CFG requires an explicit negative_prompt: ""'
    # Diffusers 0.40.0 klein encodes "" internally; it has no negative_prompt
    # keyword. Do not precompute embeddings here and move encoding outside timing.
    assert w.get("negative_prompt", "") == "", "Only the pipeline's empty negative prompt is supported"
    assert re.fullmatch(r"[0-9a-f]{40}", cfg["model"]["revision"]), (
        "Config revision is not a commit hash. Run --prepare-only first."
    )


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


def load_pipeline(cfg, run_dir, sampler, proc):
    """Read cached BF16 weights, place on CUDA, and record load costs separately."""
    import torch
    import diffusers

    torch.cuda.reset_peak_memory_stats()
    pipe_cls = getattr(diffusers, cfg["model"]["pipeline_class"])
    t0 = time.perf_counter()
    pipe = pipe_cls.from_pretrained(
        cfg["model"]["repo_id"], revision=cfg["model"]["revision"], torch_dtype=torch.bfloat16, local_files_only=True
    )
    is_distilled = pipe.config.is_distilled
    assert type(is_distilled) is bool, "Pipeline must declare is_distilled"
    assert not (is_distilled and cfg["workload"]["guidance_scale"] > 1), "Distilled klein ignores CFG; use guidance 1.0"
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
        "is_distilled": is_distilled,
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

    return pipe, load


def generate_once(pipe, recorder, w):
    """One seeded generation; host timing includes the final CUDA synchronization."""
    import torch
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
        "calls_per_step": 2 if pipe.do_classifier_free_guidance else 1,
        "scheduler_steps": recorder.scheduler_steps,
        "wall_ms": (h1 - h0) * 1000,
        "gpu_span_ms": ev0.elapsed_time(ev1),
        "t0": h0,
        "t1": h1,
        "stages": stages,
        "segments": segments,
        "peak_alloc_gib": peak_alloc,
        "peak_reserved_gib": peak_reserved,
    }



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
