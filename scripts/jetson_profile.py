"""Shared Jetson preparation plus single-profile validation; no model execution here."""

import hashlib
import json
import os
import sqlite3
import subprocess

from benchlib import write_json
from sdcpp_engine import cache_arguments


def check_execution_config(config):
    """Validate shared execution settings; entry points validate their own protocol."""
    workload = config["workload"]
    arguments = config["harness_arguments"]
    options = config["optimizations"]
    unknown = set(options) - {"quantization", "cpu_offload", "vae_tiling", "step_cache",
                              "graph_segmentation", "parameter_storage", "auto_fit", "eager_loading", "prefetch",
                              "conditioning_cache_size", "mmap"}
    if unknown:
        raise ValueError(f"Unknown optimizations: {sorted(unknown)}")
    if options.get("cpu_offload") or options.get("vae_tiling"):
        raise ValueError("CPU offload and VAE tiling are unsupported by this capture")
    mmap = options.get("mmap", False)
    if type(mmap) is not bool or arguments["mmap"] != int(mmap):
        raise ValueError("mmap must be a boolean matching the harness argument")
    prefetch = options.get("prefetch", True)
    if type(prefetch) is not bool or arguments["disable-prefetch"] != int(not prefetch):
        raise ValueError("prefetch must be a boolean matching disable-prefetch")
    entries = options.get("conditioning_cache_size", 0)
    needed = 2 if workload["cfg_scale"] > 1 else 1
    if type(entries) is not int or entries not in (0, needed) or arguments["conditioning-cache-size"] != entries:
        raise ValueError("conditioning_cache_size must match the explicit complete prompt cache (0 or CFG-dependent size)")
    cache_args = cache_arguments(options.get("step_cache"))
    supplied_cache = {k: v for k, v in arguments.items() if k.startswith("cache-")}
    if supplied_cache != cache_args:
        raise ValueError("Cache harness arguments differ from the explicit step_cache policy")
    for name, value in config["engine_settings"].items():
        if value != arguments[name.replace("_", "-")]:
            raise ValueError(f"engine_settings.{name} differs from harness_arguments")
    if options["parameter_storage"] != arguments["params-backend"]:
        raise ValueError("parameter_storage differs from params-backend")
    if bool(options.get("auto_fit", False)) != bool(arguments["auto-fit"]):
        raise ValueError("auto_fit differs from harness_arguments")
    if arguments["auto-fit"] and arguments["params-backend"] != "unset":
        raise ValueError("auto-fit requires explicitly unset parameter placement")
    if "eager_loading" in options:
        if type(options["eager_loading"]) is not bool or options["eager_loading"] != bool(arguments["eager-load"]):
            raise ValueError("eager_loading must be a boolean matching eager-load")
    if options["graph_segmentation"] != (not arguments["disable-segmented-compute"]):
        raise ValueError("graph_segmentation differs from disable-segmented-compute")
    for key in ["prompt", "width", "height", "steps", "seed", "scheduler"]:
        if workload[key] != arguments[key]:
            raise ValueError(f"workload.{key} differs from harness_arguments.{key}")
    for workload_key, argument_key in [("cfg_scale", "cfg-scale"), ("sampling_method", "sampling-method")]:
        if workload[workload_key] != arguments[argument_key]:
            raise ValueError(f"workload.{workload_key} differs from harness_arguments.{argument_key}")
    if type(workload["steps"]) is not int or workload["steps"] <= 0:
        raise ValueError("steps must be a positive integer")
    if len(config["model"]["components"]) != 3:
        raise ValueError("Expected diffusion model, text encoder and VAE, in that order")


def check_config(config):
    check_execution_config(config)
    arguments, workload, protocol = config["harness_arguments"], config["workload"], config["protocol"]
    if arguments["runs"] != 1 or workload["batch_count"] != 1:
        raise ValueError("This capture supports one generation only")
    if (protocol["attempted_generations"], protocol["warmups"], protocol["measured_generations"]) != (1, 0, 0):
        raise ValueError("This capture is a single profile, not a repeated baseline")


def collect_environment(config, engine, binary, run_dir, *, profile=True):
    commit = subprocess.check_output(["git", "-C", str(engine), "rev-parse", "HEAD"], text=True).strip()
    if commit != config["engine"]["commit"]:
        raise RuntimeError("Engine commit differs from config")
    version = subprocess.check_output(["nsys", "--version"], text=True) if profile else None
    power = subprocess.check_output(["sudo", "-n", "nvpmodel", "-q"], text=True)
    if "NV Power Mode: " + config["required_power_mode"] not in power:
        raise RuntimeError("Power mode differs from config")
    write_json(run_dir / "environment.json", {
        "engine_commit": commit, "nsys": version, "power_mode": power,
        "platform": list(os.uname()), "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
    })


def verified_model_paths(config, models_dir):
    """Read and hash each pinned file. This warms the filesystem cache."""
    paths = []
    for component in config["model"]["components"]:
        path = models_dir / component["local_file"]
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != component["sha256"]:
            raise RuntimeError("Weight hash mismatch: " + str(path))
        paths.append(str(path))
    return paths


def harness_command(config, binary, model_paths, run_dir):
    command = [str(binary), "--out-dir", str(run_dir), "--diffusion-model", model_paths[0],
               "--llm", model_paths[1], "--vae", model_paths[2]]
    for name, value in config["harness_arguments"].items():
        command.extend(["--" + name, str(value)])
    return command


def profile_command(config, binary, model_paths, run_dir):
    command = harness_command(config, binary, model_paths, run_dir)
    return ["nsys", "profile", "--trace=cuda,nvtx,osrt", "--sample=none", "--cpuctxsw=none",
            "--output=" + str(run_dir / "trace"), *command]


def require_profile_binary(binary):
    """Reject a stale/plain harness before starting an expensive capture."""
    try:
        capabilities = json.loads(subprocess.check_output([str(binary), "--capabilities", "1"], text=True, timeout=10))
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        raise RuntimeError("Rebuild sd-bench-nvtx from this bundle; capability check failed") from error
    if capabilities.get("profile_generation") is not True:
        raise RuntimeError("Rebuild sd-bench-nvtx from this bundle before profiling")


def full_profile_command(command, binary, profile_dir):
    """Trace loading and every generation, matching the A100 sd.cpp capture scope."""
    require_profile_binary(binary)
    profile_dir.mkdir()
    return ["nsys", "profile", "--trace=cuda,nvtx,osrt", "--sample=none", "--cpuctxsw=none",
            "--output=" + str(profile_dir / "trace"), *command]


def targeted_profile_command(command, binary, profile_dir, generation):
    """Reject stale harnesses, then arm Nsight for one registered NVTX range."""
    require_profile_binary(binary)
    profile_dir.mkdir()
    return ["nsys", "profile", "--trace=cuda,nvtx,osrt", "--sample=none", "--cpuctxsw=none",
            "--capture-range=nvtx", "--nvtx-capture=profile_capture", "--capture-range-end=stop",
            "--output=" + str(profile_dir / "trace"), *command,
            "--profile-generation", str(generation)]


def validate_targeted_trace(profile_dir, steps):
    """Stats exports SQLite; check the trace contains exactly one full generation."""
    validate_profile_trace(profile_dir, steps, generations=1, include_load=False)


def validate_profile_trace(profile_dir, steps, *, generations, include_load):
    """Require the declared capture count for every stage and initial-load scope."""
    validate_trace_labels(profile_dir, steps)
    database = profile_dir / "trace.sqlite"
    connection = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        ranges = connection.execute(
            "SELECT COALESCE(n.text, s.value) FROM NVTX_EVENTS n "
            "LEFT JOIN StringIds s ON n.textId = s.id WHERE n.end IS NOT NULL").fetchall()
        names = [row[0] for row in ranges]
        for name in ["generate", "text_encode", "vae_decode"] + [f"denoise_step_{i}" for i in range(steps)]:
            if names.count(name) != generations:
                raise RuntimeError(f"Expected {generations} captured NVTX ranges: {name}")
        if names.count("load") != int(include_load):
            raise RuntimeError("Initial model loading capture differs from the configured scope")
        if not connection.execute("SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0]:
            raise RuntimeError("Trace contains no CUDA kernels")
    finally:
        connection.close()


def expected_conditioning_hits(config, index):
    """A fixed prompt fills the cache on the first image, then reuses each condition."""
    return config['optimizations'].get('conditioning_cache_size', 0) if index > 0 else 0


def validate_callbacks(rows, steps, conditioning_cache_hits=0):
    generations = [row for row in rows if row["event"] == "generate"]
    if len(generations) != 1 or not generations[0]["ok"]:
        raise RuntimeError("Expected one successful generation")
    generation = generations[0]
    if (len(generation["t_step_end"]) != steps or generation["progress_calls"] != steps + 1
            or generation["cond_cache_hits"] != conditioning_cache_hits):
        raise RuntimeError("Unexpected step callbacks or conditioning reuse")
    times = [generation["t_start"], generation["t_cond"], generation["t_sampling_start"],
             *generation["t_step_end"], generation["t_sampling_end"], generation["t_decode_end"], generation["t_end"]]
    if any(value <= 0 for value in times) or times != sorted(times):
        raise RuntimeError("Missing or unordered stage boundaries")
    loads = [row for row in rows if row["event"] == "load"]
    if len(loads) != 1 or not loads[0]["ok"]:
        raise RuntimeError("Expected one successful model load")
    return loads[0]


def validate_trace_labels(run_dir, steps):
    with (run_dir / "nsys-stats.txt").open("w") as stream:
        subprocess.run(["nsys", "stats", "--report",
                        "nvtx_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_api_sum",
                        str(run_dir / "trace.nsys-rep")],
                       stdout=stream, stderr=subprocess.STDOUT, check=True)
    stats = (run_dir / "nsys-stats.txt").read_text()
    for name in ["text_encode", "vae_decode"] + [f"denoise_step_{i}" for i in range(steps)]:
        if name not in stats:
            raise RuntimeError("Missing NVTX label: " + name)


def save_profile_results(run_dir, steps):
    rows = [json.loads(line) for line in (run_dir / "results.jsonl").read_text().splitlines()]
    load = validate_callbacks(rows, steps)
    validate_trace_labels(run_dir, steps)
    write_json(run_dir / "load.json", load)
    write_json(run_dir / "summary.json", {
        "record_kind": "single stage-labelled profile", "measured": None,
        "stage_callbacks_validated": True, "nvtx_labels_present": True,
        "unavailable_metrics": {
            "baseline": "No repeated unprofiled generations",
            "memory": "Raw whole-system tegrastats only; no NVML or per-stage peak derived",
            "gpu_stage_time": "Timeline analysis pending",
        },
    })
    # Preserve the existing profile schema: no baseline rows or invented timings.
    (run_dir / "runs.csv").write_text("run_index,wall_ms\n")
    (run_dir / "stages.csv").write_text("run_index,stage,wall_ms\n")
