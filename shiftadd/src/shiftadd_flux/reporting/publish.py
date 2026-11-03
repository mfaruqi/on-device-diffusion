"""Publish diagnostic payloads through the parent repository's W&B exporter."""

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[4]
    spec = importlib.util.spec_from_file_location("parent_export", root / "scripts/export_wandb.py")
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    run = root / "results/runs" / args.run
    if "__smoke__" in run.name:
        original_parse = exporter.parse_name
        exporter.parse_name = lambda name: original_parse(name.replace("__smoke__", "__attempt__"))
    payload = exporter.describe(run)
    evidence = json.loads((run / "summary.json").read_text())
    env = json.loads((run / "environment.json").read_text())
    if json.loads((run / "status.json").read_text())["status"] != "completed":
        raise ValueError("Only completed diagnostic runs are supported")
    payload.update(
        name="A100 SXM4 · ShiftAdd reconstruction · quality smoke",
        group="A100 SXM4 40GB",
        tags=["shiftadd", "reconstructed-bf16", "quality-smoke", "single-prompt"],
        notes="Packed weights reconstructed to dense BF16. No packed-kernel speed, memory or energy claim. Activation scope and group size are declared per case in the saved configuration.",
    )
    payload["config"].update(
        gpu=env["device"],
        gpu_variant="SXM4",
        accelerator="A100 SXM4 40GB",
        precision="packed sign planes; dense BF16 execution",
        baseline_comparison_eligible=False,
        measurement_basis="single-prompt quality diagnostic",
    )
    payload["config"]["calibration_samples"] = json.loads((run / "config.json").read_text()).get(
        "samples"
    )
    payload["config"]["diagnostic_cases"] = json.loads((run / "config.json").read_text()).get(
        "cases"
    )
    if any(
        c.get("weight_format") == "symmetric_uniform"
        for c in payload["config"]["diagnostic_cases"] or []
    ):
        payload.update(
            name="A100 SXM4 · Same-source quantization control · quality smoke",
            notes="BF16, radix sign planes and conventional symmetric INT4 reconstructed to dense BF16. Single-prompt diagnostic; no packed execution performance claim.",
        )
        payload["tags"].append("conventional-int4-control")
        payload["config"]["precision"] = (
            "BF16 reference; packed radix and symmetric INT4 candidates; dense BF16 execution"
        )
    if any(
        c.get("weight_format") == "upstream_bcq"
        for c in payload["config"]["diagnostic_cases"] or []
    ):
        payload.update(
            name="A100 SXM4 · Upstream BCQ projection control · quality smoke",
            notes="Selected projection comparison with upstream BCQ candidates. All execute dense BF16; no packed performance claim. See saved diagnostic_cases for exact weight formats, activation quantization and tensor scope.",
        )
        payload["tags"].append("upstream-bcq-control")
        payload["config"]["precision"] = (
            "BF16 reference and configured upstream BCQ projection controls; dense BF16 execution"
        )
    if payload["config"]["calibration_samples"]:
        payload["tags"] = [tag for tag in payload["tags"] if tag != "single-prompt"]
        payload["tags"].append("calibration-prompt-set")
        payload["config"]["measurement_basis"] = "small calibration prompt-set diagnostic"
    payload["images"] = sorted(Path(args.images).glob("*.png"))
    rows = evidence["results"]
    columns = list(rows[0])
    payload["tables"]["quality_cases"] = (
        columns,
        [[r[k] for k in columns] for r in rows],
    )
    for row in rows:
        case_key = (row["sample_id"] + "/" if "sample_id" in row else "") + row["case"]
        payload["summary"]["quality/" + case_key + "/relative_latent_l2"] = row[
            "relative_latent_l2"
        ]
        payload["summary"]["quality/" + case_key + "/finite"] = row["finite"]
    payload["summary"].update(
        quality_scope=evidence["quality_scope"], packed_execution_measured=False
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "name": payload["name"],
                    "images": len(payload["images"]),
                    "summary": payload["summary"],
                },
                indent=2,
            )
        )
        return
    os.environ.setdefault("WANDB_DIR", str(root / "output/shiftadd-prep/wandb-cache"))
    import wandb

    api = wandb.Api()
    project = "mfaruqi-purdue-university/on-device-diffusion"
    # Authentication/listing failures propagate rather than becoming an empty project.
    existing = [
        r for r in api.runs(project) if r.id == payload["id"] or r.config.get("run_dir") == args.run
    ]
    if existing:
        print("Existing run; preserved:", existing[0].url)
        remote = existing[0]
    else:
        exporter.export(payload, "on-device-diffusion", "mfaruqi-purdue-university")
        remote = api.run(project + "/" + payload["id"])
    assert remote.state == "finished", remote.state
    for key, value in payload["summary"].items():
        if key.startswith("quality/"):
            actual = remote.summary.get(key)
            # JSON services may round the last binary64 decimal digit.
            matches = (
                isinstance(actual, (int, float))
                and not isinstance(actual, bool)
                and math.isclose(actual, value, rel_tol=1e-14, abs_tol=1e-15)
                if isinstance(value, float)
                else actual == value
            )
            assert matches, (key, actual, value)
    artifacts = list(remote.logged_artifacts())
    assert artifacts, "Missing evidence artifact"
    assert remote.summary.get("images"), "Missing image media"
    receipt = {
        "url": remote.url,
        "state": remote.state,
        "verified_quality_metrics": True,
        "images_verified": True,
        "image_count": remote.summary.get("images").get("count"),
        "artifacts": [a.name for a in artifacts],
    }
    (run / "wandb-verification.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(remote.url)


if __name__ == "__main__":
    main()
