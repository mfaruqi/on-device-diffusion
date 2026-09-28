#!/usr/bin/env python
"""Export benchmark run directories to Weights & Biases, for viewing only.

Usage (login node, which has internet; uses its own venv so the benchmark env is unchanged):
  ~/.venvs/wandb/bin/wandb login                      # once, with your own API key
  ~/.venvs/wandb/bin/python scripts/export_wandb.py --dry-run
  ~/.venvs/wandb/bin/python scripts/export_wandb.py   # exports runs not yet in the project
  ~/.venvs/wandb/bin/python scripts/export_wandb.py --only a100-sdcpp --force

The run directories under results/runs/ stay the record. W&B gets a copy: one W&B run per
run directory, with id = directory name, so re-running the export never duplicates anything.
Existing W&B runs are skipped unless --force (which deletes and re-creates them).

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
    return list(csv.DictReader(open(p))) if p.exists() else []


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def describe(run_dir):
    """Build the W&B payload for one run directory. Pure function of the files; no wandb calls."""
    cfg = read_json(run_dir / "config.json")
    env = read_json(run_dir / "environment.json")
    summ = read_json(run_dir / "summary.json")
    status = read_json(run_dir / "status.json").get("status", "unknown")
    is_sdcpp = "engine" in cfg
    profiled = run_dir.name.endswith("-profile")
    gpu = env.get("gpu_name") or env.get("nvidia_smi_gpu", "").split(",")[0].strip()
    variant = "SXM4" if "SXM4" in gpu else "PCIE" if "PCIE" in gpu else "unknown"
    node = env.get("hostname", "").split(".")[0]

    config = {
        "experiment_id": cfg.get("id"),
        "engine": "stable-diffusion.cpp" if is_sdcpp else "pytorch-diffusers",
        "engine_version": cfg["engine"]["tag"] if is_sdcpp else f"diffusers {env.get('diffusers')} / torch {env.get('torch')}",
        "engine_commit": cfg["engine"]["commit"] if is_sdcpp else None,
        "device_label": cfg.get("device_label"),
        "gpu": gpu,
        "gpu_variant": variant,
        "node": node,
        "slurm_job": env.get("env_vars", {}).get("SLURM_JOB_ID"),
        "repo_commit": (env.get("git_commit") or "")[:12],
        "repo_dirty": bool(env.get("git_dirty_files")),
        "profiled": profiled,
        "status": status,
        "model_repo": cfg.get("model", {}).get("repo_id"),
        "model_revision": cfg.get("model", {}).get("revision"),
        "precision": cfg.get("precision"),
        "workload": cfg.get("workload"),
        "protocol": cfg.get("protocol"),
        "optimizations": cfg.get("optimizations"),
        "engine_settings": cfg.get("engine_settings"),
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

    summary = {"status": status}
    measured = summ.get("measured", {})
    for k, v in measured.items():
        if v:
            for stat in ("median", "min", "max", "stdev"):
                summary[f"{stat}/{k}"] = v[stat]
    for k, v in (summ.get("first_run") or {}).items():
        if v is not None:
            summary[f"first/{k}"] = v
    load = summ.get("load") or read_json(run_dir / "load.json")
    if "load_total_s" in load:
        summary["load_total_s"] = load["load_total_s"]
    if "deterministic_output" in summ:
        summary["deterministic_output"] = summ["deterministic_output"]
    if status == "failed":
        summary["error"] = read_json(run_dir / "status.json").get("error")
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
    for name in ("denoise_kernels", "other_stages"):
        kj = read_json(run_dir / "profile" / f"{name}.json")
        if kj:
            data = [[stage, group, v["ms"], v["kernels"]]
                    for stage, st in kj["stages"].items() for group, v in st["by_group"].items()]
            tables[f"kernels_{name}"] = (["stage", "group", "ms", "kernels"], data)
            for stage, st in kj["stages"].items():
                summary[f"profile/{stage}/busy_ms"] = st["busy_ms"]
                summary[f"profile/{stage}/idle_ms"] = st["idle_ms"]

    images = [p for p in (run_dir / "images" / "first.png", run_dir / "images" / "measured-0.png") if p.exists()]
    files = [p for p in run_dir.rglob("*") if p.is_file() and p.suffix in SMALL_SUFFIXES
             and "raw" not in p.parts and p.stat().st_size < 5_000_000]
    tags = [config["engine"], variant, status] + (["profile"] if profiled else ["baseline"])
    return {
        "id": run_dir.name[:128],
        "name": run_dir.name,
        "group": cfg.get("id"),
        "job_type": "profile" if profiled else "baseline",
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

    run = wandb.init(project=project, entity=entity, id=p["id"], name=p["name"], group=p["group"],
                     job_type=p["job_type"], tags=p["tags"], config=p["config"], notes=p["notes"],
                     resume="never", reinit="finish_previous")
    for step in p["history"]:
        run.log({k: v for k, v in step.items() if k != "generation"}, step=step["generation"])
    for name, (cols, data) in p["tables"].items():
        run.log({f"table/{name}": wandb.Table(columns=cols, data=data)})
    if p["images"]:
        run.log({"images": [wandb.Image(str(i), caption=i.stem) for i in p["images"]]})
    art = wandb.Artifact(name=f"run-{p['id']}"[:128], type="benchmark-run")
    for f in p["files"]:
        art.add_file(str(f), name=str(f.relative_to(RUNS / p["name"])))
    run.log_artifact(art)
    run.summary.update(p["summary"])
    url = run.url
    run.finish()
    return url


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", default="on-device-diffusion")
    ap.add_argument("--entity", default=None, help="W&B user or team (default: your default entity)")
    ap.add_argument("--only", default="", help="export only run directories whose name contains this")
    ap.add_argument("--force", action="store_true", help="delete and re-create runs that already exist in W&B")
    ap.add_argument("--dry-run", action="store_true", help="print what would be exported; no network")
    args = ap.parse_args()
    os.environ.setdefault("WANDB_DIR", f"/scratch/gilbreth/{os.environ.get('USER', 'user')}/wandb")
    Path(os.environ["WANDB_DIR"]).mkdir(parents=True, exist_ok=True)

    dirs = sorted(d for d in RUNS.iterdir() if d.is_dir() and args.only in d.name)
    payloads = [describe(d) for d in dirs]
    if args.dry_run:
        for p in payloads:
            print(f"{p['name']}: group={p['group']} tags={p['tags']} steps={len(p['history'])} "
                  f"tables={list(p['tables'])} images={len(p['images'])} files={len(p['files'])}")
            print("   summary:", {k: round(v, 2) if isinstance(v, float) else v for k, v in p["summary"].items()
                                  if k.startswith(("median/wall", "median/device", "status", "deterministic"))})
        return

    import wandb

    api = wandb.Api()
    entity = args.entity or api.default_entity
    for p in payloads:
        path = f"{entity}/{args.project}/{p['id']}"
        try:
            existing = api.run(path)
        except Exception:  # noqa: BLE001  (wandb raises CommError when the run doesn't exist)
            existing = None
        if existing and not args.force:
            print(f"skip {p['name']} (already in W&B: {existing.url})")
            continue
        if existing:
            existing.delete(delete_artifacts=True)
        print(f"export {p['name']} -> {export(p, args.project, entity)}")


if __name__ == "__main__":
    main()
