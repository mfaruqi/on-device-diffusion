"""sd.cpp adapter: prepare/check weights, build argv, decode callback timestamps.

The C++ harness owns model execution and its CUDA context. This module must never
import torch: a second context would contaminate device-wide memory measurements.
"""

import copy
import hashlib
import math
import shutil
import os
import re
from pathlib import Path

from benchlib import now_utc, sh, write_json
from measurement import check_optimizations


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


def check_config(cfg):
    e, m, s = cfg["engine"], cfg["model"], cfg["engine_settings"]
    assert e["name"] == "stable-diffusion.cpp", e["name"]
    assert HEX40.fullmatch(e["commit"]) and HEX40.fullmatch(e["ggml_commit"]), "engine commits must be hashes"
    assert HEX40.fullmatch(m["revision"]), "model.revision must be a commit hash"
    assert "resolved_at_utc" in m, "use the *.resolved.json config (run --prepare-only first)"
    check_optimizations(cfg, {"quantization", "cpu_offload", "vae_tiling", "step_cache", "auto_fit",
                              "conditioning_cache_size", "prefetch", "mmap"},
                        {"step_cache", "conditioning_cache_size", "prefetch", "mmap"})
    cache_arguments(cfg["optimizations"].get("step_cache"))
    check_execution_options(cfg)
    assert cfg["precision"] == "bfloat16"
    assert cfg["workload"]["batch_size"] == 1
    # Reference placement: everything on one GPU, nothing automatic.
    assert s["backend"] == s["params_backend"] and s["backend"].startswith("cuda"), s
    assert s["auto_fit"] is False and s["eager_load"] is True and s["disable_segmented_compute"] is True, s
    assert os.environ.get("GGML_CUDA_CUBLAS_COMPUTE_TYPE") is None, "GGML_CUDA_CUBLAS_COMPUTE_TYPE overrides GEMM precision"

    src = expand(e["source_dir"])
    head = sh(["git", "-C", str(src), "rev-parse", "HEAD"])
    ggml = sh(["git", "-C", str(src / "ggml"), "rev-parse", "HEAD"])
    assert head == e["commit"], f"sd.cpp checkout is at {head}, config pins {e['commit']}"
    assert ggml == e["ggml_commit"], f"ggml checkout is at {ggml}, config pins {e['ggml_commit']}"
    return {"sdcpp_head": head, "ggml_head": ggml, "sdcpp_status": sh(["git", "-C", str(src), "status", "--porcelain"])}


def check_execution_options(cfg):
    options, settings = cfg['optimizations'], cfg['engine_settings']
    entries = options.get('conditioning_cache_size', 0)
    needed = 2 if cfg['workload']['guidance_scale'] > 1 else 1
    assert type(entries) is int and entries in (0, needed), 'Invalid conditioning cache capacity'
    assert settings['conditioning_cache_size'] == entries, 'Conditioning cache differs from labelled policy'
    for key, field, invert, default in [('prefetch', 'disable_prefetch', True, True),
                                        ('mmap', 'mmap', False, False)]:
        value = options.get(key, default)
        assert type(value) is bool and settings[field] == (not value if invert else value), f'{key} differs from labelled policy'


def conditioning_hits(cfg, count):
    return [0] + [cfg['optimizations'].get('conditioning_cache_size', 0)] * (count - 1)


def check_files(cfg, snap):
    """Confirm the snapshot files are the pinned ones (HF cache blobs are named by sha256)."""
    for rel, sha in cfg["model"]["sha256"].items():
        blob = Path(os.path.realpath(snap / rel)).name
        assert blob == sha, f"{rel}: cached blob {blob} does not match pinned sha256 {sha}"


def cache_arguments(cache):
    """Expose one verified DiT policy; refuse unsupported modes and hidden defaults."""
    if cache is None:
        return {}
    assert isinstance(cache, dict) and set(cache) == {"mode", "reuse_threshold", "start_percent", "end_percent"}, "Explicit EasyCache parameters required"
    assert cache["mode"] == "easycache", "Only EasyCache is currently supported by this adapter"
    for key in ("reuse_threshold", "start_percent", "end_percent"):
        assert type(cache[key]) in (int, float) and math.isfinite(cache[key]), f"Invalid {key}"
    assert cache["reuse_threshold"] >= 0
    assert 0 <= cache["start_percent"] < cache["end_percent"] <= 1
    return {"cache-mode": "easycache", "cache-threshold": cache["reuse_threshold"],
            "cache-start": cache["start_percent"], "cache-end": cache["end_percent"]}


def audit_cache(log, cache, generations):
    enabled = re.findall(r"EasyCache enabled - threshold: ([\d.]+), start: ([\d.]+), end: ([\d.]+)", log)
    outcomes = re.findall(r"EasyCache (?:skipped (\d+)/(\d+) steps|completed without skipping steps)", log)
    if cache is None:
        assert not enabled, "Unexpected EasyCache in no-reuse reference"
        return {"requested": None, "enabled_generations": 0}
    cache_arguments(cache)
    assert len(enabled) == generations and len(outcomes) == generations, "EasyCache initialization/outcome missing; inspect unsupported/disabled warnings"
    expected = (f"{cache['reuse_threshold']:.3f}", f"{cache['start_percent']:.2f}", f"{cache['end_percent']:.2f}")
    assert all(values == expected for values in enabled), "Engine-reported cache parameters differ"
    return {"requested": cache, "enabled_generations": len(enabled),
            "steps_skipped": [int(skipped) if skipped else 0 for skipped, total in outcomes],
            "quality_scope": "Approximate execution; image diagnostics required, no quality eligibility implied."}


def audit_log(log_text, settings, t_last_generation_end, *, expected_conditioning_hits=0):
    """Extract what sd.cpp actually did, and fail on anything the reference config forbids.

    Log lines from the harness start with a steady_clock timestamp (s).
    """
    lines = log_text.splitlines()
    found = {k: [ln for ln in lines if re.search(p, ln)] for k, p in FORBIDDEN_LOG_PATTERNS.items()}
    actual_hits = len(found['conditioning_cache_hit'])
    if actual_hits == expected_conditioning_hits:
        found['conditioning_cache_hit'] = []

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
    if actual_hits != expected_conditioning_hits:
        problems.append(f'conditioning hits: expected {expected_conditioning_hits}, saw {actual_hits}')
    audit['conditioning_cache_hits'] = actual_hits
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
        **cache_arguments(cfg["optimizations"].get("step_cache")),
    }
    return [x for k, v in args.items() for x in (f"--{k}", str(v))]


def rows_from_results(results, phases, steps, sampler, *, expected_conditioning_hits=None):
    expected_conditioning_hits = ([0] * len(results) if expected_conditioning_hits is None
                                  else expected_conditioning_hits)
    assert len(expected_conditioning_hits) == len(results), "conditioning-hit expectations must cover every generation"
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
        assert g["cond_cache_hits"] == expected_conditioning_hits[i], f"run {i}: unexpected conditioning cache hits"
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
