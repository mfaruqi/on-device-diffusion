#!/usr/bin/env python
"""Export benchmark run directories to Weights & Biases, for viewing only.

Usage (login node, which has internet; uses its own venv so the benchmark env is unchanged):
  ~/.venvs/wandb/bin/wandb login                      # once, with your own API key
  ~/.venvs/wandb/bin/python scripts/export_wandb.py --dry-run
  ~/.venvs/wandb/bin/python scripts/export_wandb.py   # exports runs not yet in the project
  ~/.venvs/wandb/bin/python scripts/export_wandb.py --only a100-sdcpp --force

The run directories under results/runs/ stay the record. W&B gets a copy: one W&B run per
run directory, named after it, so re-running the export never duplicates anything.
Existing W&B runs are matched by id or config.run_dir. --update changes their config and
summary in place and skips runs not yet uploaded. --force deletes and re-creates runs
under new ids (W&B never reuses a deleted run's id).

What each W&B run contains:
- config:  engine, device, GPU variant, node, workload, precision, optimizations, versions
- history: one step per generation (first, warm-up, measured) with stage times and memory
- summary: medians / min / max of measured runs, first run, load, determinism (same keys for all engines)
- tables:  generations (runs.csv), stages (stages.csv), kernel groups (profile runs)
- media:   images/first.png, images/measured-0.png
- artifact: the run directory's small files (json/csv/md), so the numbers are traceable
Metric definitions: wiki/methods/baseline-metrics.md.
"""

import argparse
import csv
import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS = REPO_ROOT / "results" / "runs"
STAGE_METRICS = ["wall_ms", "text_encode_ms", "denoise_ms", "vae_decode_ms", "postprocess_ms", "other_ms"]
MEMORY_METRICS = ["peak_alloc_gib", "peak_reserved_gib", "device_used_peak_gib"]
SMALL_SUFFIXES = {".json", ".csv", ".md", ".txt", ".jsonl"}


def read_json(p):
    return json.loads(p.read_text()) if p.exists() else {}


def read_csv(p):
    if not p.exists():
        return []
    with p.open() as stream:
        return list(csv.DictReader(stream))


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


KINDS = ("baseline", "profile", "repeat", "attempt")


def model_storage(env, device):
    """Where the model files were read from, as recorded by the run (or its device's setup notes)."""
    if env.get("storage"):
        return env["storage"]
    hf = (env.get("env_vars") or {}).get("HF_HOME")
    if hf:
        return f"HF cache on {hf}"
    if "jetson" in device:
        return ("microSD /dev/mmcblk0p1 (from raw/jetson-setup-2026-09-28.md; "
                "not recorded in this run)")
    return None


def parse_name(name):
    """<experiment-id>__<kind>__<YYYYMMDD-HHMMSS> (wiki/methods/baseline-metrics.md#run-naming)."""
    parts = name.split("__")
    if len(parts) == 3 and parts[1] in KINDS:
        return parts[0], parts[1], parts[2]
    raise ValueError(f"run directory {name!r} doesn't follow <experiment-id>__<kind>__<stamp>")


def timing_labels(kind, is_sdcpp, summ, stat, cfg, rows):
    """Label the measured protocol separately from an optional, extra PyTorch trace."""
    measured = (summ.get("measured") or {}).get("wall_ms") or {}
    scope = "profiled" if kind == "profile" and is_sdcpp else "unprofiled"
    if kind == "profile" and not is_sdcpp:
        note = ("Generation aggregates and load precede the separate torch.profiler generation. "
                "profile/* metrics describe the extra traced generation.")
    elif kind == "profile" and (cfg.get('profiling') or {}).get('generation') is not None:
        note = (f"Nsight traces only generation {cfg['profiling']['generation']} after first/warm-up generations. "
                "It is attached throughout; measured timing is a single profiled diagnostic, not a baseline.")
    elif kind == "profile":
        note = "Nsight is attached during loading and generation; timings include profiling conditions."
    else:
        note = "Generation and load were recorded without a trace profiler attached."
    count = measured.get("n", 0)
    protocol = cfg.get("protocol", {})
    phase_counts = {phase: sum(row.get("phase") == phase for row in rows)
                    for phase in ("first", "warmup", "measured")}
    protocol_matches = all(phase_counts[phase] == protocol.get(key) for phase, key in
                           [("first", "first_runs"), ("warmup", "warmup_runs"),
                            ("measured", "measured_runs")])
    eligible = (stat.get("status") == "complete" and scope == "unprofiled"
                and kind in ("baseline", "repeat", "profile") and count > 1
                and count == phase_counts["measured"] and protocol_matches)
    return scope, note, eligible


def describe(run_dir):
    """Build the W&B payload for one run directory. Pure function of the files; no wandb calls."""
    cfg = read_json(run_dir / "config.json")
    env = read_json(run_dir / "environment.json")
    summ = read_json(run_dir / "summary.json")
    stat = read_json(run_dir / "status.json")
    status = stat.get("status", "unknown")
    failed = status == "failed"
    experiment, kind, stamp = parse_name(run_dir.name)
    engine = cfg.get("engine") if isinstance(cfg.get("engine"), dict) else {}
    engine_name = engine.get("name") or "pytorch-diffusers"
    is_sdcpp = engine.get("name") == "stable-diffusion.cpp"
    is_native = engine_name in {"stable-diffusion.cpp", "edge-dit.cpp"}
    # Requested tool, including failed attempts; status/trace evidence establish success.
    profiler = ("nsight-systems" if is_sdcpp else "torch.profiler") if kind == "profile" else "none"
    device = cfg.get("device_label", "")
    gpu = env.get("gpu_name") or env.get("nvidia_smi_gpu", "").split(",")[0].strip()
    if not gpu and "jetson" in device:
        gpu = f"NVIDIA Jetson Orin Nano (sm_{engine.get('cuda_architecture', '?')})"
    variant = ("SXM4" if "SXM4" in gpu else "PCIE" if "PCIE" in gpu else
               "jetson-orin-nano" if "jetson" in device else "unknown")
    opt = cfg.get("optimizations") or {}
    quant = opt.get("quantization")

    config = {
        "run_dir": run_dir.name,
        "experiment_id": experiment,
        "kind": kind,
        "profiler": profiler,
        "profiling": cfg.get("profiling"),
        "outcome": "FAILED" if failed else "OK",
        "benchmark_started": stamp,
        "engine": engine_name,
        "engine_version": engine.get("tag") or (engine.get("commit") or "")[:12] if is_native
                          else f"diffusers {env.get('diffusers')} / torch {env.get('torch')}",
        "engine_commit": engine.get("commit") if is_native else None,
        "device_label": device,
        "gpu": gpu,
        "gpu_variant": variant,
        "node": env.get("hostname", "").split(".")[0] or device,
        "slurm_job": env.get("env_vars", {}).get("SLURM_JOB_ID"),
        "repo_commit": (env.get("git_commit") or "")[:12],
        "record_kind": summ.get("record_kind") or cfg.get("record_kind"),
        "status": status,
        "error": stat.get("error"),
        "failed_phase": stat.get("phase"),
        "model": cfg.get("model"),
        "precision": cfg.get("precision") or ("quantized" if quant else None),
        "workload": cfg.get("workload"),
        "protocol": cfg.get("protocol"),
        "optimizations": opt,
        "engine_settings": cfg.get("engine_settings"),
        "power_mode": (env.get("power_mode") or cfg.get("power_mode_observed_before_run")
                       or cfg.get("required_power_mode")),
        "model_storage": model_storage(env, device),
    }

    rows = read_csv(run_dir / "runs.csv")
    history = []
    for r in rows:
        step = {"generation": int(r["run_index"]), "phase": r["phase"]}
        for k, v in r.items():
            if k.endswith("_ms") or k.endswith("_gib"):
                if num(v) is not None:
                    step[f"gen/{k}"] = num(v)
        history.append(step)

    summary = {"status": status, "outcome": config["outcome"]}
    measured = summ.get("measured") or {}
    for k, v in measured.items():
        if v:
            for st in ("median", "min", "max", "stdev"):
                summary[f"{st}/{k}"] = v[st]
    for k, v in (summ.get("first_run") or {}).items():
        if v is not None:
            summary[f"first/{k}"] = v
    load = summ.get("load") or read_json(run_dir / "load.json")
    if "load_total_s" in load:
        summary["load_total_s"] = load["load_total_s"]
    if "deterministic_output" in summ:
        summary["deterministic_output"] = summ["deterministic_output"]
    for k, v in (summ.get("engine_log_diagnostics") or {}).items():
        summary[f"diag/{k}"] = v   # single-attempt engine-log durations; not protocol medians
    mw = (read_json(run_dir / "feasibility-analysis.json").get("monitor_window")
          or summ.get("monitor_window") or {})
    if "ram_peak_gib" in mw:
        summary["diag/system_ram_peak_gib"] = mw["ram_peak_gib"]   # tegrastats, 1 s sampling
        summary["diag/system_ram_total_gib"] = mw.get("ram_total_gib")
    # Keep timing values and label their conditions. Run kind alone is insufficient:
    # PyTorch profiles an extra generation; sd.cpp profiles the entire protocol.
    diag = summ.get("engine_log_diagnostics") or {}
    td = read_json(run_dir / "timing-diagnostics.json") if kind == "profile" else {}
    scope, scope_note, eligible = timing_labels(kind, is_sdcpp, summ, stat, cfg, rows)
    if kind == "profile":
        for k in ("load_seconds", "generation_seconds", "text_encode_seconds", "vae_decode_seconds"):
            if isinstance(td.get(k), (int, float)):
                summary[f"diag/profile_{k}"] = td[k]
        for i, v in enumerate(td.get("denoise_step_seconds") or []):
            summary[f"diag/profile_denoise_step_{i}_seconds"] = v
        if td.get("scope"):
            summary["diag/profile_scope"] = td["scope"]
    basis = "unavailable"
    if measured.get("wall_ms"):
        basis = "measured_median"
        summary["timing/generate_s"] = measured["wall_ms"]["median"] / 1000
        summary["timing/generate_basis"] = f"median of {measured['wall_ms']['n']} measured generations"
        summary["timing/generate_samples"] = measured["wall_ms"]["n"]
        summary["timing/generate_source"] = "summary.json:measured.wall_ms.median"
        if kind == 'profile' and (cfg.get('profiling') or {}).get('generation') is not None:
            basis = 'single_profile_callback'
            summary['timing/generate_basis'] = (f"single profiled generation {cfg['profiling']['generation']}, "
                                                "host callbacks (not a repeated baseline)")
    elif isinstance(td.get("generation_seconds"), (int, float)):
        basis = "single_profile_callback"
        summary["timing/generate_s"] = td["generation_seconds"]
        summary["timing/generate_basis"] = "single profiled generation, host callbacks (not a protocol median)"
        summary["timing/generate_samples"] = 1
        summary["timing/generate_source"] = "timing-diagnostics.json:generation_seconds"
    elif "generate_image_seconds" in diag:
        basis = "single_engine_log"
        summary["timing/generate_s"] = diag["generate_image_seconds"]
        summary["timing/generate_basis"] = "single attempt, engine-log duration (not a protocol median)"
        summary["timing/generate_samples"] = 1
        summary["timing/generate_source"] = "summary.json:engine_log_diagnostics.generate_image_seconds"
    if "load_total_s" in load:
        summary["timing/load_s"] = load["load_total_s"]
    elif isinstance(td.get("load_seconds"), (int, float)):
        summary["timing/load_s"] = td["load_seconds"]
    elif "initial_tensor_loading_seconds" in diag:
        summary["timing/load_s"] = diag["initial_tensor_loading_seconds"]
    summary["timing/load_scope"] = scope if "timing/load_s" in summary else "unavailable"
    if kind == 'profile' and (cfg.get('profiling') or {}).get('generation') is not None and 'timing/load_s' in summary:
        summary['timing/load_scope'] = 'profiler_attached_untraced'
    generation_scope = scope if basis != "unavailable" else "unavailable"
    config.update(measurement_scope=generation_scope, measurement_basis=basis,
                  baseline_comparison_eligible=eligible, measurement_scope_note=scope_note)
    summary.update({"profiler": profiler, "timing/measurement_scope": generation_scope,
                    "timing/scope_note": scope_note,
                    "timing/baseline_comparison_eligible": eligible})
    t0, t1 = stat.get("started_utc"), stat.get("finished_utc")
    if t0 and t1:
        import datetime as dt
        f = lambda t: dt.datetime.fromisoformat(t.replace("Z", "+00:00"))  # noqa: E731
        summary["timing/job_wall_s"] = (f(t1) - f(t0)).total_seconds()
        summary["timing/job_wall_scope"] = ("whole runner, including loading and profiling" if kind == "profile"
                                             else "whole runner, including loading and all protocol phases")
    if failed:
        summary["error"] = stat.get("error")
    if measured.get("wall_ms"):
        med = measured["wall_ms"]["median"]
        for k in STAGE_METRICS[1:]:
            if measured.get(k):
                summary[f"share/{k}"] = measured[k]["median"] / med

    tables = {}
    if rows:
        tables["generations"] = (list(rows[0]), [list(r.values()) for r in rows])
    stages = read_csv(run_dir / "stages.csv")
    if stages:
        tables["stages"] = (list(stages[0]), [list(r.values()) for r in stages])
    teg = read_csv(run_dir / "tegrastats-samples.csv")
    if teg:
        tables["tegrastats"] = (list(teg[0]), [list(r.values()) for r in teg])
    for name in ("denoise_kernels", "other_stages"):
        kj = read_json(run_dir / "profile" / f"{name}.json")
        if kj:
            data = [[stage, group, v["ms"], v["kernels"]]
                    for stage, st in kj["stages"].items() for group, v in st["by_group"].items()]
            tables[f"kernels_{name}"] = (["stage", "group", "ms", "kernels"], data)
            for stage, st in kj["stages"].items():
                summary[f"profile/{stage}/busy_ms"] = st["busy_ms"]
                summary[f"profile/{stage}/idle_ms"] = st["idle_ms"]

    images = [p for p in (run_dir / "images" / "first.png", run_dir / "images" / "measured-0.png",
                          run_dir / "output.png") if p.exists()]
    files = [p for p in run_dir.rglob("*") if p.is_file() and p.suffix in SMALL_SUFFIXES | {".log"}
             and "raw" not in p.parts and p.stat().st_size < 5_000_000]
    tags = [config["engine"], variant, kind, "failed" if failed else "complete"]
    if quant:
        tags.append("quantized")
    return {
        "id": run_dir.name[:128],
        "name": f"{experiment} · {kind} · {config['outcome']}",
        "run_dir": run_dir.name,
        "failed": failed,
        "group": experiment,
        "job_type": kind,
        "tags": tags,
        "config": config,
        "history": history,
        "summary": summary,
        "tables": tables,
        "images": images,
        "files": files,
        "notes": cfg.get("description", ""),
    }


def export(p, project, entity):
    import wandb

    # W&B's automatic metadata (GPU, CPU, host, git, runtime, system stats) would describe the
    # login node running this export, not the benchmark node. Turn it off; the benchmark's own
    # hardware and software are in `config` (gpu, gpu_variant, node, versions) from environment.json.
    settings = wandb.Settings(x_disable_meta=True, x_disable_stats=True, x_disable_machine_info=True,
                              disable_git=True, disable_code=True, disable_job_creation=True)
    run = wandb.init(project=project, entity=entity, id=p["id"], name=p["name"], group=p["group"],
                     job_type=p["job_type"], tags=p["tags"], config=p["config"], notes=p["notes"],
                     resume="never", reinit="finish_previous", settings=settings)
    # x-axis of per-generation charts is the generation index (0 = first, then warm-ups, then measured)
    run.define_metric("generation")
    run.define_metric("gen/*", step_metric="generation")
    for step in p["history"]:
        run.log({k: v for k, v in step.items() if k != "phase"})
    for name, (cols, data) in p["tables"].items():
        run.log({f"table/{name}": wandb.Table(columns=cols, data=data)})
    if p["images"]:
        run.log({"images": [wandb.Image(str(i), caption=i.stem) for i in p["images"]]})
    art = wandb.Artifact(name=f"run-{p['id']}"[:128], type="benchmark-run")
    for f in p["files"]:
        art.add_file(str(f), name=str(f.relative_to(RUNS / p["run_dir"])))
    run.log_artifact(art)
    run.summary.update(p["summary"])
    url = run.url
    run.finish(exit_code=1 if p["failed"] else 0)   # failed runs show W&B's red "Failed" state
    return url


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", default="on-device-diffusion")
    ap.add_argument("--entity", default=None, help="W&B user or team (default: your default entity)")
    ap.add_argument("--only", default="", help="export only run directories whose name contains this")
    ap.add_argument("--force", action="store_true", help="delete and re-create runs that already exist in W&B")
    ap.add_argument("--dry-run", action="store_true", help="print what would be exported; no network")
    ap.add_argument("--update", action="store_true",
                    help="update existing runs in place; skip missing runs (no uploads or deletion)")
    args = ap.parse_args()
    os.environ.setdefault("WANDB_DIR", f"/scratch/gilbreth/{os.environ.get('USER', 'user')}/wandb")
    Path(os.environ["WANDB_DIR"]).mkdir(parents=True, exist_ok=True)

    dirs = sorted(d for d in RUNS.iterdir() if d.is_dir() and args.only in d.name)
    payloads = [describe(d) for d in dirs]
    if args.dry_run:
        for p in payloads:
            print(f"{p['run_dir']} [{p['name']}]: group={p['group']} tags={p['tags']} steps={len(p['history'])} "
                  f"tables={list(p['tables'])} images={len(p['images'])} files={len(p['files'])}")
            print("   summary:", {k: round(v, 2) if isinstance(v, float) else v for k, v in p["summary"].items()
                                  if k.startswith(("median/wall", "median/device", "status", "deterministic"))})
            print("   measurement:", {k: p["config"][k] for k in
                                      ("measurement_scope", "measurement_basis", "baseline_comparison_eligible")})
        return

    import wandb

    api = wandb.Api()
    entity = args.entity or api.default_entity
    project_path = f"{entity}/{args.project}"
    try:
        existing = {r.id: r for r in api.runs(project_path)}
    except Exception:  # noqa: BLE001  (project doesn't exist yet)
        if args.update:
            raise  # Never interpret an authentication/network failure as an empty project.
        existing = {}
    for p in payloads:
        # W&B run id = run directory name. W&B never reuses a deleted id, so --force re-creates a run
        # under the directory name plus an export timestamp.
        old = existing.get(p["id"]) or next((r for r in existing.values() if r.config.get("run_dir") == p["run_dir"]), None)
        if args.update:
            if old is None:
                print(f"skip {p['run_dir']} (not uploaded; --update only changes existing runs)")
                continue
            for k, v in p["config"].items():
                old.config[k] = v
            for k, v in p["summary"].items():
                old.summary[k] = v
            old.update()
            print(f"update {p['run_dir']} -> {old.url}")
            continue
        if old and not args.force:
            print(f"skip {p['run_dir']} (already in W&B: {old.url})")
            continue
        if old:
            old.delete(delete_artifacts=True)
            import datetime as dt
            p["id"] = f"{p['run_dir']}__x{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%S}"[:128]
        print(f"export {p['run_dir']} -> {export(p, args.project, entity)}")

if __name__ == "__main__":
    main()
