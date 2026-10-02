"""Preparation and validation for one Jetson sd.cpp profile; no model execution here."""

import hashlib
import json
import os
import subprocess

from benchlib import write_json


def check_config(config):
    """Refuse ambiguous workload copies or a protocol this capture cannot report."""
    workload = config["workload"]
    arguments = config["harness_arguments"]
    protocol = config["protocol"]
    options = config["optimizations"]
    unknown = set(options) - {"quantization", "cpu_offload", "vae_tiling", "step_cache",
                              "graph_segmentation", "parameter_storage"}
    if unknown:
        raise ValueError(f"Unknown optimizations: {sorted(unknown)}")
    if options.get("cpu_offload") or options.get("vae_tiling") or options.get("step_cache"):
        raise ValueError("CPU offload, VAE tiling and step caching are unsupported by this capture")
    for name, value in config["engine_settings"].items():
        if value != arguments[name.replace("_", "-")]:
            raise ValueError(f"engine_settings.{name} differs from harness_arguments")
    if options["parameter_storage"] != arguments["params-backend"]:
        raise ValueError("parameter_storage differs from params-backend")
    if options["graph_segmentation"] != (not arguments["disable-segmented-compute"]):
        raise ValueError("graph_segmentation differs from disable-segmented-compute")
    for key in ["prompt", "width", "height", "steps", "seed", "scheduler"]:
        if workload[key] != arguments[key]:
            raise ValueError(f"workload.{key} differs from harness_arguments.{key}")
    for workload_key, argument_key in [("cfg_scale", "cfg-scale"), ("sampling_method", "sampling-method")]:
        if workload[workload_key] != arguments[argument_key]:
            raise ValueError(f"workload.{workload_key} differs from harness_arguments.{argument_key}")
    if arguments["runs"] != 1 or workload["batch_count"] != 1:
        raise ValueError("This capture supports one generation only")
    if (protocol["attempted_generations"], protocol["warmups"], protocol["measured_generations"]) != (1, 0, 0):
        raise ValueError("This capture is a single profile, not a repeated baseline")
    if type(workload["steps"]) is not int or workload["steps"] <= 0:
        raise ValueError("steps must be a positive integer")
    if len(config["model"]["components"]) != 3:
        raise ValueError("Expected diffusion model, text encoder and VAE, in that order")


def collect_environment(config, engine, binary, run_dir):
    commit = subprocess.check_output(["git", "-C", str(engine), "rev-parse", "HEAD"], text=True).strip()
    if commit != config["engine"]["commit"]:
        raise RuntimeError("Engine commit differs from config")
    version = subprocess.check_output(["nsys", "--version"], text=True)
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


def profile_command(config, binary, model_paths, run_dir):
    command = [str(binary), "--out-dir", str(run_dir), "--diffusion-model", model_paths[0],
               "--llm", model_paths[1], "--vae", model_paths[2]]
    for name, value in config["harness_arguments"].items():
        command.extend(["--" + name, str(value)])
    return ["nsys", "profile", "--trace=cuda,nvtx,osrt", "--sample=none", "--cpuctxsw=none",
            "--output=" + str(run_dir / "trace"), *command]


def validate_callbacks(rows, steps):
    generations = [row for row in rows if row["event"] == "generate"]
    if len(generations) != 1 or not generations[0]["ok"]:
        raise RuntimeError("Expected one successful generation")
    generation = generations[0]
    if (len(generation["t_step_end"]) != steps or generation["progress_calls"] != steps + 1
            or generation["cond_cache_hits"]):
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
