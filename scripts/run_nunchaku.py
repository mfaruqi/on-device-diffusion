#!/usr/bin/env python3
"""Candidate SVDQuant/Klein timer. Shares the CUDA protocol with run_flux.py.

--check-support is metadata-only and does not load a model or import CUDA.
Only the pinned A100 community INT4/distilled path is admitted for smoke testing.
Jetson and M5 are explicit blockers, never BF16 or CPU fallbacks.
"""
import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import time

from benchlib import GIB, REPO_ROOT, write_json
from flux_engine import check_config as check_reference, collect_environment, generate_once
from measurement import create_run_directory, phase_sequence, run_status, sampling
from measurement_events import export_events
from run_flux import build_summary, measure_generations
from torch_stages import StageRecorder

SUPPORT = {
    "a100": {"status": "candidate_not_gpu_validated", "backend": "cuda", "sm": "80",
             "reason": "Pinned community Klein 4B INT4/r32 adapter; needs A100 smoke validation."},
    "jetson": {"status": "blocked", "backend": "cuda", "sm": "87",
               "reason": "Orin sm_87 is absent from the reviewed Nunchaku build targets; ARM64 Spark wheels do not establish Jetson support."},
    "m5": {"status": "blocked", "backend": "metal",
           "reason": "Reviewed Nunchaku kernels require CUDA; no Metal/MPS implementation. SVDQuant is not a device-switch flag."},
}


def check_config(cfg):
    q = cfg["optimizations"]["quantization"]
    assert q == {"method": "svdquant", "component": "transformer", "weights_bits": 4,
                 "activations_bits": 4, "low_rank": 32, "format": "int4"}, "Only labelled W4A4 INT4/r32 is implemented"
    reference = copy.deepcopy(cfg)
    reference["optimizations"]["quantization"] = None
    check_reference(reference)  # Same no-cache/no-offload/no-compile and workload guards.
    assert cfg["device_label"] == "gilbreth-a100-40gb"
    assert cfg["model"]["repo_id"] == "black-forest-labs/FLUX.2-klein-4B", "Base checkpoint support is not established"
    assert cfg["workload"]["guidance_scale"] == 1.0
    assert cfg["engine"]["name"] == "nunchaku"
    for source in [cfg["quantized_transformer"], cfg["engine"]["wheel"]]:
        assert re.fullmatch(r"[0-9a-f]{40}", source["revision"]), "An immutable revision is required"
        assert re.fullmatch(r"[0-9a-f]{64}", source["sha256"]), "A content SHA256 is required"
    phase_sequence(cfg["protocol"])


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def cached_file(source, offline):
    from huggingface_hub import hf_hub_download
    p = Path(hf_hub_download(source["repo_id"], filename=source["filename"],
                            revision=source["revision"], local_files_only=offline))
    assert file_sha256(p) == source["sha256"], f"Content hash mismatch: {p.name}"
    return p


def validate_smoke(cfg, directory):
    assert directory is not None, "Full timing requires --smoke-run pointing to a completed matching smoke"
    status = json.loads((directory/"status.json").read_text())
    smoke = json.loads((directory/"config.json").read_text())
    summary = json.loads((directory/"summary.json").read_text())
    audit = json.loads((directory/"quantization-audit.json").read_text())
    assert status["status"] == "complete" and summary["measured_runs"] == 1
    expected = copy.deepcopy(cfg)
    expected["protocol"].update(first_runs=1, warmup_runs=0, measured_runs=1)
    assert smoke == expected, "Smoke configuration differs from requested full run"
    assert audit["quantized_modules"] and audit["checkpoint_sha256"] == cfg["quantized_transformer"]["sha256"]
    from PIL import Image
    for name in ("first.png", "measured-0.png"):
        with Image.open(directory/"images"/name) as image:
            assert image.size == (cfg["workload"]["width"], cfg["workload"]["height"])
            assert any(low != high for low, high in image.convert("RGB").getextrema()), "Flat smoke image"


def prepare(cfg):
    """Download only pinned inputs, outside measurement. Does not install the wheel."""
    from huggingface_hub import snapshot_download
    check_config(cfg)
    model = cfg["model"]
    base = snapshot_download(model["repo_id"], revision=model["revision"],
                             allow_patterns=model["allow_patterns"])
    quantized = cached_file(cfg["quantized_transformer"], offline=False)
    wheel = cached_file(cfg["engine"]["wheel"], offline=False)
    print(json.dumps({"pipeline_cache": base, "transformer": str(quantized), "wheel": str(wheel)}, indent=2))


def check_runtime(cfg, torch):
    assert torch.cuda.is_available(), "Run on an A100 compute node, not a login node or MPS."
    props = torch.cuda.get_device_properties(0)
    assert (props.major, props.minor) == (8, 0) and "A100" in props.name
    assert 39 <= props.total_memory / GIB <= 41, "This configuration is labelled A100 40 GB"
    assert tuple(torch.__version__.split(".")[:2]) == ("2", "11"), "Pinned wheel requires PyTorch 2.11"
    assert torch.version.cuda == "12.8", "Pinned wheel requires cu128"
    engine = cfg["engine"]
    assert importlib.metadata.version("nunchaku") == engine["version"], "Nunchaku version differs from pinned build"
    provenance = json.loads(importlib.metadata.distribution("nunchaku").read_text("direct_url.json") or "{}")
    archive = provenance.get("archive_info", {})
    sha = archive.get("hashes", {}).get("sha256") or archive.get("hash", "").removeprefix("sha256=")
    assert sha == engine["wheel"]["sha256"], "Install the pinned wheel by local path/URL; its hash must match"
    return provenance


def validate_component_dtypes(component, entries):
    """Keep weights BF16; permit only Qwen3's nonpersistent FP32 RoPE buffers."""
    for entry in entries:
        if not entry["floating"] or entry["dtype"] == "torch.bfloat16":
            continue
        rope_buffer = (component == "text_encoder" and entry["kind"] == "buffer"
                       and entry["owner_class"] == "Qwen3RotaryEmbedding"
                       and entry["leaf_name"] in {"inv_freq", "original_inv_freq"}
                       and not entry["persistent"] and entry["dtype"] == "torch.float32")
        assert rope_buffer, f"Unexpected dtype: {component}.{entry['name']} = {entry['dtype']}"


def component_tensor_audit(module):
    owners = dict(module.named_modules())
    entries = []
    for kind, iterator in (("parameter", module.named_parameters()), ("buffer", module.named_buffers())):
        for name, tensor in iterator:
            parent, _, leaf = name.rpartition(".")
            owner = owners[parent]
            entries.append({"name": name, "kind": kind, "leaf_name": leaf,
                            "owner_class": type(owner).__name__,
                            "persistent": kind == "parameter" or leaf not in owner._non_persistent_buffers_set,
                            "dtype": str(tensor.dtype), "device": str(tensor.device),
                            "shape": list(tensor.shape), "floating": tensor.is_floating_point()})
    return entries


def load_pipeline(cfg, run_dir, quantized):
    import torch
    from diffusers import Flux2KleinPipeline
    from huggingface_hub import hf_hub_download
    from nunchaku import NunchakuFlux2Transformer2DModel
    from nunchaku.models.linear import SVDQW4A4Linear

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    transformer, metadata = NunchakuFlux2Transformer2DModel.from_pretrained(
        str(quantized), device="cuda", torch_dtype=torch.bfloat16, offload=False, return_metadata=True)
    modules = {name: m for name, m in transformer.named_modules() if isinstance(m, SVDQW4A4Linear)}
    assert modules, "No SVDQuant layers loaded; refusing an unquantized fallback"
    assert all(m.precision == "int4" and m.rank == 32 for m in modules.values())
    assert not transformer.offload
    model = cfg["model"]
    reference_config = json.loads(Path(hf_hub_download(model["repo_id"], "transformer/config.json",
                                 revision=model["revision"], local_files_only=True)).read_text())
    for key in ("num_layers", "num_single_layers", "num_attention_heads", "attention_head_dim", "in_channels", "joint_attention_dim", "guidance_embeds"):
        assert key in reference_config and transformer.config[key] == reference_config[key], f"Transformer architecture mismatch: {key}"
    pipe = Flux2KleinPipeline.from_pretrained(model["repo_id"], revision=model["revision"],
                transformer=transformer, torch_dtype=torch.bfloat16, local_files_only=True).to("cuda")
    assert pipe.config.is_distilled is True, "Only the distilled checkpoint is screened"
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    pipe.set_progress_bar_config(disable=True)
    components = {}
    tensor_audit = {name: component_tensor_audit(getattr(pipe, name))
                    for name in ("transformer", "text_encoder", "vae")}
    write_json(run_dir/"component-tensor-audit.json", tensor_audit)
    for name in ("transformer", "text_encoder", "vae"):
        module = getattr(pipe, name)
        tensors = list(module.parameters()) + list(module.buffers())
        assert tensors and all(t.device.type == "cuda" for t in tensors), f"Unexpected placement: {name}"
        if name != "transformer":
            validate_component_dtypes(name, tensor_audit[name])
        components[name] = {"class": type(module).__name__, "weights_gib": sum(t.numel()*t.element_size() for t in tensors)/GIB,
                            "dtypes": sorted({str(t.dtype) for t in tensors}), "device": "cuda"}
    load = {"is_distilled": True, "load_total_s": elapsed,
            "allocated_after_load_gib": torch.cuda.memory_allocated()/GIB,
            "peak_alloc_during_load_gib": torch.cuda.max_memory_allocated()/GIB, "components": components,
            "note": "Cached load and CUDA placement; SHA256 validation occurred before load timing. Not cold-storage latency."}
    write_json(run_dir/"load.json", load)
    write_json(run_dir/"scheduler.json", {"class": type(pipe.scheduler).__name__, "config": dict(pipe.scheduler.config)})
    write_json(run_dir/"quantization-audit.json", {"method": "svdquant", "format": "int4", "rank": 32,
        "quantized_modules": sorted(modules), "metadata": metadata, "checkpoint_sha256": cfg["quantized_transformer"]["sha256"],
        "formal_quality_eligible": False, "source_revision_equivalence": "Community checkpoint ancestry is not proven to equal our pinned BF16 transformer revision."})
    return pipe, load


def benchmark(cfg, run_dir):
    import psutil
    import torch
    from benchlib import DeviceMemorySampler

    check_config(cfg)
    provenance = check_runtime(cfg, torch)
    # No downloads during execution. Hashing is outside model-load/generation timing.
    quantized = cached_file(cfg["quantized_transformer"], offline=True)
    uuid = str(torch.cuda.get_device_properties(0).uuid)
    if not uuid.startswith("GPU-"):
        uuid = "GPU-" + uuid
    sampler = DeviceMemorySampler(cfg["protocol"]["device_memory_sample_ms"], uuid)
    proc = psutil.Process()
    with sampling(sampler):
        env = collect_environment(run_dir, sampler)
        env.update(nunchaku=importlib.metadata.version("nunchaku"), nunchaku_install=provenance)
        source_manifest = REPO_ROOT / "source-manifest.json"
        if source_manifest.exists():
            env["source_snapshot"] = json.loads(source_manifest.read_text())
        write_json(run_dir/"environment.json", env)
        pipe, load = load_pipeline(cfg, run_dir, quantized)
        recorder = StageRecorder(torch)
        recorder.install(pipe)
        def generate():
            return generate_once(pipe, recorder, cfg["workload"])
        rows = measure_generations(generate, phase_sequence(cfg["protocol"]),
                    cfg["workload"]["num_inference_steps"], run_dir, sampler, proc)
        summary = build_summary(cfg, rows, load)
        summary.update(engine="nunchaku-community", timing_method="CUDA events at pipeline stage boundaries; synchronized host total",
                       formal_quality_eligible=False, memory_scope="PyTorch allocator peaks exclude allocations outside its allocator; NVML device-wide samples include context and other GPU processes.")
        write_json(run_dir/"summary.json", summary)
        export_events(run_dir, "pytorch")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", required=True, choices=SUPPORT)
    ap.add_argument("--config", type=Path)
    ap.add_argument("--check-support", action="store_true")
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--smoke-run", type=Path, help="Required for a full protocol; same configuration, completed 1+0+1 smoke")
    ap.add_argument("--run-dir-file", type=Path, help="Write the created run directory for a gated Slurm continuation")
    ap.add_argument("--out-root", type=Path, default=REPO_ROOT/"results/runs")
    args = ap.parse_args()
    if args.check_support:
        print(json.dumps({"device": args.device, **SUPPORT[args.device]}, indent=2))
        return
    if SUPPORT[args.device]["status"] == "blocked":
        ap.error(SUPPORT[args.device]["reason"])
    if args.config is None:
        ap.error("--config is required for preparation or timing")
    cfg = json.loads(args.config.read_text())
    if args.prepare_only:
        prepare(cfg)
        return
    os.environ["HF_HUB_OFFLINE"] = "1"
    kind = "attempt" if cfg["protocol"]["measured_runs"] < 10 else "baseline"
    run_dir = create_run_directory(cfg, args.out_root, kind)
    with run_status(run_dir):
        if args.run_dir_file:
            with args.run_dir_file.open("x") as stream:
                stream.write(str(run_dir.resolve()) + "\n")
        if kind == "baseline":
            validate_smoke(cfg, args.smoke_run)
        benchmark(cfg, run_dir)


if __name__ == "__main__":
    main()
