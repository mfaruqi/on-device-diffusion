#!/usr/bin/env python
"""Portable FLUX.2 [klein] PyTorch runner: one script for A100 (cuda), Jetson (jetson) and M5 (mps).

Same protocol, stage accounting and run files as run_flux.py; what differs per device lives in
torch_backends.py. Every change from the BF16 reference is one labelled entry in `optimizations`:

  quantization.transformer   {"method": "gguf", "file": path, "sha256": hex}  (diffusers GGUF loader, BF16 compute)
  quantization.text_encoder  {"method": "bnb_nf4"} or {"method": "torchao_int4", "group_size": 128}
                             (quantized while loading, so the BF16 encoder never has to fit in memory)
                             A pre-quantized encoder saved by save_quantized_text_encoder.py is given as
                             {"method": "bnb_nf4", "path": dir, "sha256": hash of model.safetensors}.
  stage_residency            {"text_encoder": "release_after_encode"[, "transformer": "release_during_encode"]}:
                             the encoder is loaded right before each text encoding and freed right after; with
                             the transformer entry, the transformer is also freed before encoding and reloaded
                             after it, so only one large component is resident at a time (sequential residency).
                             Reloads run between the timed stages: they are inside wall_ms and other_ms, not
                             text_encode, and each is recorded in stage-residency.json. Needs a pre-quantized
                             encoder path.
  vae_tiling                 true: diffusers' tiled VAE decode, with a check that the tiled path ran;
                             or {"tile_sample_px": int, "overlap": float} to set the tile edge (image pixels)
                             and overlap factor explicitly instead of the VAE config's sample_size default
  attention_backend          null (PyTorch picks the SDPA kernel) | "math" | "flash" | "efficient": every
                             generation runs inside torch.nn.attention.sdpa_kernel([that backend]); SDPA calls
                             are counted so a restriction that never applies is caught
  conditioning_reuse         true: exact prompt-encoding reuse (scripts/mps_conditioning.py) wrapped outside the stage
                             residency swaps, so a hit skips the encoder load and the transformer release/reload.
                             Receipt in conditioning-cache.json.
  step_cache                 null | {"method": "first_block_cache", "threshold": float}
                             | {"method": "taylorseer", "cache_interval": int, "disable_cache_before_step": int,
                                "max_order": int}: diffusers CacheMixin.enable_cache on the transformer (re-applied
                             after any reload). Full transformer computations per generation are counted on the last
                             single block (step-cache.json).
  compiler                   null | {"name": "torch_compile", "mode": "default"}
                             | {"name": "tensorrt", "precision": "bfloat16", "require_full_compilation": bool,
                                "min_block_size": int}
                             Applied to the transformer forward only; compilation happens lazily in the
                             first generation, which is reported separately. TensorRT is CUDA-only.

Usage:
  python scripts/run_flux_portable.py --config configs/<id>.json [--kind attempt|baseline] [--profile]
--profile (CUDA/Jetson only) adds one generation under torch.profiler after every measured generation and
receipt, with the device monitor already stopped: profile/trace.json (Chrome trace, opens in Perfetto),
profile/op_table.txt and profile/profiled_run.json. It is excluded from runs.csv and summary.json.
On the Jetson, run it through scripts/jetson_memguard.sh. Metric definitions: wiki/methods/baseline-metrics.md.
"""

import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import time

from benchlib import GIB, REPO_ROOT, host_environment, peak_rss_gib, sh, write_json
from measurement import create_run_directory, phase_sequence, run_status, sampling, summarize_runs
from torch_backends import check_backend, make_device

TE_METHODS = ("bnb_nf4", "torchao_int4")
NEW_KEYS = ("compiler", "stage_residency", "attention_backend", "conditioning_reuse", "step_cache")
SDPA_BACKENDS = (None, "math", "flash", "efficient")


def check_config(cfg):
    """Validate every option before importing torch or reading weights."""
    from flux_engine import check_config as check_reference

    check_backend(cfg)
    opt = cfg["optimizations"]
    q = opt.get("quantization") or {}
    assert set(q) <= {"transformer", "text_encoder"}, f"unknown quantization targets: {sorted(q)}"
    if "transformer" in q:
        t = q["transformer"]
        assert t["method"] == "gguf" and t["file"] and len(t["sha256"]) == 64, "GGUF transformer needs file and sha256"
    if "text_encoder" in q:
        assert q["text_encoder"]["method"] in TE_METHODS, f"text encoder method must be one of {TE_METHODS}"
        if "path" in q["text_encoder"]:
            assert len(q["text_encoder"]["sha256"]) == 64, "pre-quantized encoder needs the model.safetensors sha256"
    residency = opt.get("stage_residency")
    if residency:
        assert residency in ({"text_encoder": "release_after_encode"},
                             {"text_encoder": "release_after_encode", "transformer": "release_during_encode"}), (
            f"unsupported stage_residency: {residency}")
        if "transformer" in residency:
            assert "transformer" in q, "transformer release/reload is wired for the GGUF transformer only"
            assert not opt.get("compiler") or opt.get("conditioning_reuse"), (
                "with transformer reloads, compile only together with conditioning_reuse (one reload, then recompiled once)")
        assert "path" in q.get("text_encoder", {}), "release_after_encode reloads a pre-quantized encoder; give its path"
    tiling = opt.get("vae_tiling", False)
    assert type(tiling) is bool or (type(tiling) is dict and set(tiling) == {"tile_sample_px", "overlap"}
                                    and type(tiling["tile_sample_px"]) is int and 0 < tiling["overlap"] < 1), (
        "vae_tiling must be true, false or {tile_sample_px, overlap}")
    assert opt.get("attention_backend") in SDPA_BACKENDS, f"attention_backend must be one of {SDPA_BACKENDS}"
    assert type(opt.get("conditioning_reuse", False)) is bool, "conditioning_reuse must be true or false"
    cache = opt.get("step_cache")
    if cache is not None:
        assert cache.get("method") in ("first_block_cache", "taylorseer"), f"unsupported step_cache: {cache}"
        keys = {"first_block_cache": {"method", "threshold"},
                "taylorseer": {"method", "cache_interval", "disable_cache_before_step", "max_order"}}[cache["method"]]
        assert set(cache) == keys, f"step_cache {cache['method']} needs exactly {sorted(keys)}"
    for off in ("cpu_offload", "vae_slicing", "torch_compile"):
        assert not opt.get(off), f"{off} is not supported by this runner; use 'compiler' for compilation"
    compiler = opt.get("compiler")
    if compiler and compiler["name"] == "tensorrt":
        assert compiler["precision"] == "bfloat16", "TensorRT precision must be stated; only bfloat16 is wired"
        assert type(compiler["require_full_compilation"]) is bool, "state require_full_compilation explicitly"
    if compiler and compiler["name"] == "torch_compile":
        assert compiler["mode"] in ("default", "reduce-overhead", "max-autotune"), "unknown torch.compile mode"
    if cfg["backend"] == "jetson":
        assert type(cfg["protocol"].get("tegrastats_interval_ms")) is int, "Jetson configs need tegrastats_interval_ms"
    # The reference checks (precision, pipeline, revision pinning, workload) apply unchanged.
    reference = copy.deepcopy(cfg)
    for key in NEW_KEYS:
        reference["optimizations"].pop(key, None)
    reference["optimizations"]["quantization"] = None
    reference["optimizations"]["vae_tiling"] = False
    check_reference(reference)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def load_pipeline(cfg, device, run_dir):
    """Load each component as configured, place everything on the device, and record what was loaded."""
    import diffusers
    import torch
    import transformers
    from huggingface_hub import snapshot_download

    model, q = cfg["model"], cfg["optimizations"].get("quantization") or {}
    snapshot = snapshot_download(model["repo_id"], revision=model["revision"], local_files_only=True,
                                 allow_patterns=model["allow_patterns"])
    dev = device.torch_device
    device.reset_peaks()
    t0 = time.perf_counter()
    kwargs, receipt = {}, {}

    loaders = {}
    if "transformer" in q:
        t = q["transformer"]
        digest = sha256_file(t["file"])  # hashed once here; reloads read the same verified file
        assert digest == t["sha256"], f"GGUF sha256 mismatch: {digest}"

        decoded_bf16 = []

        def transformer_loader():
            m = diffusers.Flux2Transformer2DModel.from_single_file(
                t["file"], quantization_config=diffusers.GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
                config=snapshot, subfolder="transformer", torch_dtype=torch.bfloat16)
            # diffusers 0.40.0 keeps every GGUF tensor as a uint8 GGUFParameter and decodes only inside GGUFLinear.
            # The file's BF16 tensors outside linear layers (RMSNorm scales etc.) would reach their ops as raw
            # bytes, and ModelMixin.dtype would report uint8 (the klein pipeline casts latents to it). Decode those
            # BF16 tensors once into ordinary BF16 parameters: same values, quantized linear layers untouched.
            from diffusers.quantizers.gguf.utils import GGUFLinear, GGUFParameter, dequantize_gguf_tensor
            decoded = []
            for mod_name, mod in m.named_modules():
                if isinstance(mod, GGUFLinear):
                    continue
                for p_name, p in list(mod._parameters.items()):
                    if isinstance(p, GGUFParameter) and p.quant_type.name == "BF16":
                        mod._parameters[p_name] = torch.nn.Parameter(
                            dequantize_gguf_tensor(p).to(torch.bfloat16), requires_grad=False)
                        decoded.append(f"{mod_name}.{p_name}")
            assert m.dtype == torch.bfloat16, f"GGUF transformer reports dtype {m.dtype}"
            decoded_bf16.clear()
            decoded_bf16.extend(decoded)
            # Move before loading anything else: a CPU copy kept until the final placement coexists with the
            # device copy during the move (~2.4 GB extra), which exceeded the Jetson's memory on 2026-10-12.
            return m.to(dev)

        loaders["transformer"] = transformer_loader
        kwargs["transformer"] = transformer_loader()
        receipt["transformer"] = {"method": "gguf", "file": t["file"], "sha256": digest, "compute_dtype": "bfloat16",
                                  "decoded_non_linear_bf16_tensors": list(decoded_bf16),
                                  "note": "BF16 tensors outside GGUFLinear decoded to BF16 parameters (lossless); diffusers 0.40.0 leaves them as raw bytes"}

    if "text_encoder" in q and "path" in q["text_encoder"]:
        te = q["text_encoder"]
        digest = sha256_file(Path(te["path"]) / "model.safetensors")
        assert digest == te["sha256"], f"pre-quantized encoder sha256 mismatch: {digest}"

        def te_loader():
            # The quantization config is stored with the weights; this reads the NF4 tensors directly.
            return transformers.Qwen3ForCausalLM.from_pretrained(te["path"], torch_dtype=torch.bfloat16,
                                                                  device_map={"": dev})

        loaders["text_encoder"] = te_loader
        residency = (cfg["optimizations"].get("stage_residency") or {}).get("text_encoder")
        kwargs["text_encoder"] = None if residency == "release_after_encode" else te_loader()
        receipt["text_encoder"] = dict(te, sha256_verified=digest, resident=residency != "release_after_encode")
    elif "text_encoder" in q:
        te = q["text_encoder"]
        if te["method"] == "bnb_nf4":
            qc = transformers.BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                 bnb_4bit_compute_dtype=torch.bfloat16)
        else:
            from torchao.quantization import Int4WeightOnlyConfig
            qc = transformers.TorchAoConfig(quant_type=Int4WeightOnlyConfig(group_size=te.get("group_size", 128)))
        # device_map places each layer on the device as it is quantized; the BF16 encoder is never resident.
        kwargs["text_encoder"] = transformers.Qwen3ForCausalLM.from_pretrained(
            snapshot, subfolder="text_encoder", quantization_config=qc, torch_dtype=torch.bfloat16,
            device_map={"": dev})
        receipt["text_encoder"] = dict(te)

    pipe = getattr(diffusers, model["pipeline_class"]).from_pretrained(
        snapshot, torch_dtype=torch.bfloat16, local_files_only=True, **kwargs)
    t1 = time.perf_counter()
    for name, comp in pipe.components.items():
        if isinstance(comp, torch.nn.Module) and not (name == "text_encoder" and "text_encoder" in q):
            comp.to(dev)
    device.synchronize()
    t2 = time.perf_counter()
    pipe.set_progress_bar_config(disable=True)

    components = {}
    for name, comp in pipe.components.items():
        if isinstance(comp, torch.nn.Module):
            params = list(comp.parameters()) + list(comp.buffers())
            devices = sorted({p.device.type for p in params})
            assert devices == [torch.device(dev).type], f"{name} is not fully on {dev}: {devices}"
            components[name] = {"class": type(comp).__name__, "weights_gib": sum(p.numel() * p.element_size() for p in params) / GIB,
                                "dtypes": sorted({str(p.dtype) for p in params}), "device": devices[0]}
    load = {"is_distilled": pipe.config.is_distilled, "from_pretrained_s": t1 - t0, "to_device_s": t2 - t1,
            "load_total_s": t2 - t0, **device.allocator_after_load(), "components": components,
            "quantization": receipt, "snapshot": snapshot,
            "note": "Cached load and placement; the OS page cache may already hold the files. Not a cold read."}
    write_json(run_dir / "load.json", load)
    write_json(run_dir / "scheduler.json", {"class": type(pipe.scheduler).__name__, "config": dict(pipe.scheduler.config)})
    return pipe, load, loaders


def apply_stage_residency(pipe, residency, loaders, device, on_reload=None):
    """Swap components in and out around encode_prompt. Install AFTER the stage recorder, so this wraps the
    recorder's timed text_encode stage and every load/free falls between timed stages (other_ms)."""
    import gc

    calls = []
    if not residency:
        return calls
    release_transformer = residency.get("transformer") == "release_during_encode"
    timed_encode = pipe.encode_prompt

    def free():
        gc.collect()
        if device.torch_device == "cuda":
            device.torch.cuda.empty_cache()
        device.synchronize()

    def encode_with_swaps(*args, **kwargs):
        rec, t = {}, time.perf_counter()
        if release_transformer:
            old = pipe.transformer
            # The stage recorder's step hooks live on the module; carry them to each reloaded transformer.
            hooks = (old._forward_pre_hooks, old._forward_hooks)
            pipe.transformer = None
            del old
            free()
            rec["transformer_free_s"] = time.perf_counter() - t
            t = time.perf_counter()
        pipe.text_encoder = loaders["text_encoder"]()
        device.synchronize()
        rec["text_encoder_load_s"] = time.perf_counter() - t
        try:
            return timed_encode(*args, **kwargs)
        finally:
            t = time.perf_counter()
            pipe.text_encoder = None
            free()
            rec["text_encoder_free_s"] = time.perf_counter() - t
            if release_transformer:
                t = time.perf_counter()
                new = loaders["transformer"]()
                new._forward_pre_hooks, new._forward_hooks = hooks
                if on_reload:
                    on_reload(new)  # re-apply compile / step cache to the reloaded module
                pipe.transformer = new
                device.synchronize()
                rec["transformer_load_s"] = time.perf_counter() - t
            calls.append(rec)

    pipe.encode_prompt = encode_with_swaps
    return calls


def apply_vae_tiling(pipe, enabled):
    """Enable tiled decode and count tiled calls, so a request that never takes effect is caught."""
    calls = []
    if enabled:
        pipe.vae.enable_tiling()
        if isinstance(enabled, dict):
            factor = 2 ** (len(pipe.vae.config.block_out_channels) - 1)
            pipe.vae.tile_sample_min_size = enabled["tile_sample_px"]
            pipe.vae.tile_latent_min_size = enabled["tile_sample_px"] // factor
            pipe.vae.tile_overlap_factor = enabled["overlap"]
        original = pipe.vae.tiled_decode

        def tiled_decode(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)

        pipe.vae.tiled_decode = tiled_decode
    return calls


def apply_compiler(pipe, compiler, transformer=None):
    """Compile the transformer forward only. Stage hooks stay outside the compiled region."""
    import torch

    transformer = transformer or pipe.transformer
    if not compiler:
        return {"name": None}
    if compiler["name"] == "torch_compile":
        transformer.forward = torch.compile(transformer.forward, mode=compiler["mode"])
        return {"name": "torch_compile", "mode": compiler["mode"], "torch": torch.__version__}
    import torch_tensorrt

    options = {"enabled_precisions": {torch.bfloat16},
               "require_full_compilation": compiler["require_full_compilation"],
               "min_block_size": compiler.get("min_block_size", 5),
               "truncate_double": True}
    transformer.forward = torch.compile(transformer.forward, backend="torch_tensorrt", options=options)
    return {"name": "tensorrt", "torch_tensorrt": torch_tensorrt.__version__,
            "options": {k: (sorted(str(x) for x in v) if isinstance(v, set) else v) for k, v in options.items()}}


def apply_step_cache(transformer, cache):
    """diffusers CacheMixin step cache on one transformer instance."""
    if not cache:
        return
    from diffusers.hooks import FirstBlockCacheConfig, TaylorSeerCacheConfig
    if cache["method"] == "first_block_cache":
        transformer.enable_cache(FirstBlockCacheConfig(threshold=cache["threshold"]))
    else:
        transformer.enable_cache(TaylorSeerCacheConfig(cache_interval=cache["cache_interval"],
                                                       disable_cache_before_step=cache["disable_cache_before_step"],
                                                       max_order=cache["max_order"]))


def count_full_computations(transformer, counts):
    """Count calls of the last single-stream block; a skipped/predicted step never reaches it."""
    block = transformer.single_transformer_blocks[-1]
    return block.register_forward_pre_hook(lambda *_: counts.__setitem__(-1, counts[-1] + 1) if counts else None)


def restrict_attention(generate, backend):
    """Run each generation under one SDPA backend and count SDPA calls made inside it."""
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel

    calls = []
    if backend is None:
        return generate, calls
    kind = {"math": SDPBackend.MATH, "flash": SDPBackend.FLASH_ATTENTION,
            "efficient": SDPBackend.EFFICIENT_ATTENTION}[backend]
    original = torch.nn.functional.scaled_dot_product_attention

    def counted(*args, **kwargs):
        calls[-1] += 1
        return original(*args, **kwargs)

    def restricted():
        calls.append(0)
        torch.nn.functional.scaled_dot_product_attention = counted
        try:
            with sdpa_kernel([kind]):
                return generate()
        finally:
            torch.nn.functional.scaled_dot_product_attention = original

    return restricted, calls


def environment(device):
    import diffusers
    import torch
    import transformers

    env = host_environment(extra_env_vars=["HF_HOME", "HF_HUB_OFFLINE", "PYTORCH_ENABLE_MPS_FALLBACK", "TORCHINDUCTOR_COMPILE_THREADS"])
    env.update(device.environment())
    env.update(torch=torch.__version__, diffusers=diffusers.__version__, transformers=transformers.__version__)
    for pkg in ("bitsandbytes", "torchao", "gguf", "torch_tensorrt", "tensorrt", "triton"):
        try:
            env[pkg] = __import__(pkg).__version__
        except Exception as error:  # noqa: BLE001
            env[pkg] = f"<not importable: {type(error).__name__}>"
    return env


def benchmark(cfg, run_dir, profile=False):
    import psutil
    from run_flux import measure_generations

    check_config(cfg)
    assert os.environ.get("HF_HUB_OFFLINE") == "1", "Inference must run offline (HF_HUB_OFFLINE=1)"
    phases = phase_sequence(cfg["protocol"])
    device = make_device(cfg, run_dir)
    write_json(run_dir / "environment.json", environment(device))
    sources = ["scripts/run_flux_portable.py", "scripts/torch_backends.py", "scripts/run_flux.py",
               "scripts/flux_engine.py", "scripts/torch_stages.py", "scripts/mps_conditioning.py"]
    write_json(run_dir / "source-provenance.json",
               {name: hashlib.sha256((REPO_ROOT / name).read_bytes()).hexdigest() for name in sources})
    if hasattr(device, "start_monitor"):
        device.start_monitor()
    try:
        with sampling(device.sampler):
            pipe, load, loaders = load_pipeline(cfg, device, run_dir)
            tiled = apply_vae_tiling(pipe, cfg["optimizations"].get("vae_tiling", False))
            opt = cfg["optimizations"]
            compiler = apply_compiler(pipe, opt.get("compiler"))
            apply_step_cache(pipe.transformer, opt.get("step_cache"))
            full_steps = []  # full transformer computations per generation (step-cache receipt)
            counter = [count_full_computations(pipe.transformer, full_steps)] if opt.get("step_cache") else []
            compiles = [1] if opt.get("compiler") else []

            def on_reload(new):
                if opt.get("compiler"):
                    apply_compiler(pipe, opt["compiler"], new)
                    compiles.append(1)
                apply_step_cache(new, opt.get("step_cache"))
                if counter:
                    counter.append(count_full_computations(new, full_steps))

            write_json(run_dir / "compiler.json", compiler)
            recorder = device.recorder()
            conditioning = None
            if opt.get("conditioning_reuse"):
                # Order: residency swaps (innermost) -> exact conditioning cache -> stage recorder (outermost), so a hit
                # skips every swap yet is still timed as text_encode (retrieval). The single miss (first generation)
                # times its swaps inside text_encode; that generation is reported separately.
                from mps_conditioning import ConditioningCache
                reloads = apply_stage_residency(pipe, opt.get("stage_residency"), loaders, device, on_reload)
                conditioning = ConditioningCache(pipe, 2 if cfg["workload"]["guidance_scale"] > 1 else 1)
                recorder.install(pipe)
            else:
                recorder.install(pipe)
                reloads = apply_stage_residency(pipe, opt.get("stage_residency"), loaders, device, on_reload)
            backend = opt.get("attention_backend")

            def generate_once():
                full_steps.append(0)
                return device.generate(pipe, recorder, cfg["workload"])

            generate, sdpa_calls = restrict_attention(generate_once, backend)
            rows = measure_generations(generate, phases,
                                       cfg["workload"]["num_inference_steps"], run_dir, device.sampler, psutil.Process())
            if conditioning:
                assert conditioning.misses == conditioning.capacity and conditioning.hits == (len(phases) - 1) * conditioning.capacity, conditioning.receipt()
                write_json(run_dir / "conditioning-cache.json", conditioning.receipt())
            if opt.get("step_cache"):
                assert len(full_steps) == len(phases)
                write_json(run_dir / "step-cache.json", {"config": opt["step_cache"], "steps": cfg["workload"]["num_inference_steps"],
                                                         "full_transformer_computations_per_generation": full_steps})
            if opt.get("compiler"):
                write_json(run_dir / "compiler.json", {**compiler, "compilations_applied": len(compiles)})
            if cfg["optimizations"].get("stage_residency"):
                # The pipeline calls encode_prompt once per prompt: twice per generation under CFG (prompt, negative).
                encodes = 2 if cfg["workload"]["guidance_scale"] > 1 else 1
                expected = conditioning.capacity if conditioning else encodes * len(phases)
                assert len(reloads) == expected, f"encoder reloads {len(reloads)} != {expected}"
                write_json(run_dir / "stage-residency.json", {"policy": cfg["optimizations"]["stage_residency"],
                                                               "per_generation": reloads})
            if backend is not None:
                assert len(sdpa_calls) == len(phases) and min(sdpa_calls) > 0, f"SDPA restriction not exercised: {sdpa_calls}"
                write_json(run_dir / "attention.json", {"attention_backend": backend, "sdpa_calls_per_generation": sdpa_calls})
            if cfg["optimizations"].get("vae_tiling"):
                assert len(tiled) == len(phases), f"VAE tiling requested; tiled decode ran {len(tiled)}/{len(phases)} times"
            if profile:
                assert device.torch_device == "cuda", "--profile needs CUDA (torch.profiler CUPTI activity)"
                if hasattr(device, "stop_monitor"):
                    device.stop_monitor()  # keep the profiled generation out of the tegrastats capture
                from flux_engine import run_profile
                run_profile(generate, run_dir)
    finally:
        if hasattr(device, "stop_monitor"):
            device.stop_monitor()
    summary = {"id": cfg["id"], "device_label": cfg["device_label"], "backend": cfg["backend"],
               "engine": "PyTorch/Diffusers", **summarize_runs(rows),
               "load": {k: load[k] for k in ("load_total_s", "from_pretrained_s", "to_device_s")},
               "weights_gib": {k: v["weights_gib"] for k, v in load["components"].items()},
               "quantization": load["quantization"], "compiler": compiler,
               "first_run_note": "Includes lazy compilation when a compiler is configured.",
               "system_memory": device.system_memory_summary(run_dir),
               "unavailable_metrics": device.unavailable_metrics(),
               "host_peak_rss_gib": peak_rss_gib(), "units": {"time": "ms (load in s)", "memory": "GiB (2^30 bytes)"}}
    with (run_dir / "stages.csv").open() as stream:  # CFG runs call the transformer twice per step
        first = [r for r in csv.DictReader(stream) if r["run_index"] == "0" and r["stage"] == "denoise_step_0"]
    summary["transformer_calls_per_step"] = len(first)
    write_json(run_dir / "summary.json", summary)
    print(json.dumps(summary["measured"]["wall_ms"], indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--kind", choices=["baseline", "repeat", "attempt", "profile"], default=None,
                    help="run kind in the directory name (default: profile with --profile, else attempt)")
    ap.add_argument("--profile", action="store_true", help="add one torch.profiler generation after the protocol")
    ap.add_argument("--out-root", type=Path, default=REPO_ROOT / "results" / "runs")
    ap.add_argument("--check-only", action="store_true", help="validate the config and exit (no torch import)")
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())
    check_config(cfg)
    if args.check_only:
        print("config ok")
        return
    run_dir = create_run_directory(cfg, args.out_root, args.kind or ("profile" if args.profile else "attempt"))
    with run_status(run_dir):
        benchmark(cfg, run_dir, args.profile)


if __name__ == "__main__":
    main()
