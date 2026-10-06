#!/usr/bin/env python3
"""Plot saved Week 1 baselines; optionally upload a separate W&B analysis run.

Dependencies: plotly, wandb (only for --upload).
Example: python scripts/plot_stage_breakdown.py --output output/stage-breakdown --upload
No model execution. --data-json reuses an exported dataset on another host.
"""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import statistics
import uuid


ROOT = Path(__file__).resolve().parents[1]
BASELINES = [
    ("a100-flux-klein-001__baseline__20260923-221530", "PyTorch / Diffusers",
     "A100-PCIE-40GB", "BF16 · 1024 × 1024"),
    ("a100-sdcpp-flux-klein-001__baseline__20260925-114328", "stable-diffusion.cpp",
     "A100-PCIE-40GB", "BF16 · 1024 × 1024"),
    ("jetson-flux-klein-003__baseline__20260930-193208", "stable-diffusion.cpp",
     "Jetson Orin Nano", "Q4_0 transformer / Q4_K_M text · 512 × 512 · disk-backed"),
]
STAGES = {
    "Text encoding": "#3C8DAD",
    "Denoising (4 steps)": "#6652A2",
    "VAE decoding": "#DB9650",
    "Output / remaining time": "#B4BBC5",
}
METRICS = ["text_encode_ms", "denoise_ms", "vae_decode_ms"]
METHOD = (
    "Means across measured generations only; first and warmups excluded. "
    "Remaining = wall - encoding - denoising - decoding; includes postprocessing "
    "where separately recorded and host/event boundary differences. "
    "PyTorch stages use CUDA events; sd.cpp uses host callbacks and includes "
    "in-stage reloads. sd.cpp decode includes output conversion. Initial loading excluded. "
    "Different precision, resolution and residency across devices; not a matched GPU "
    "speed comparison. Cross-engine quality equivalence is untested."
)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def collect(root):
    runs = []
    for name, engine, gpu, workload in BASELINES:
        directory = root / "results" / "runs" / name
        summary = json.loads((directory / "summary.json").read_text())
        status = json.loads((directory / "status.json").read_text())
        if status["status"] != "complete":
            raise ValueError(f"Incomplete baseline: {name}")
        with (directory / "runs.csv").open() as stream:
            measured = [r for r in csv.DictReader(stream) if r["phase"] == "measured"]
        if len(measured) != summary["measured_runs"] or len(measured) != 10:
            raise ValueError(f"Expected ten measured generations: {name}")
        means = {}
        for key in ["wall_ms", *METRICS]:
            means[key] = statistics.mean(float(r[key]) for r in measured) / 1000
            if not math.isclose(means[key], summary["measured"][key]["mean"] / 1000,
                                rel_tol=1e-7, abs_tol=1e-7):
                raise ValueError(f"CSV/summary mean mismatch: {name}: {key}")
        values = [means[key] for key in METRICS]
        values.append(means["wall_ms"] - sum(values))
        if any(not math.isfinite(v) or v < 0 for v in values):
            raise ValueError(f"Invalid or negative stage duration: {name}")
        runs.append({
            "run_dir": name, "engine": engine, "gpu": gpu, "workload": workload,
            "measured_count": len(measured), "stages_s": dict(zip(STAGES, values)),
            "mean_s": means["wall_ms"],
            "median_s": summary["measured"]["wall_ms"]["median"] / 1000,
            "min_s": summary["measured"]["wall_ms"]["min"] / 1000,
            "max_s": summary["measured"]["wall_ms"]["max"] / 1000,
            "config": json.loads((directory / "config.json").read_text()),
            "source_sha256": {f: hashlib.sha256((directory / f).read_bytes()).hexdigest()
                              for f in ["summary.json", "runs.csv", "config.json"]},
        })
    return {"schema_version": 1, "method": METHOD, "runs": runs}


def figure(runs, title):
    import plotly.graph_objects as go

    labels = [f"{r['engine']}<br>{r['gpu']}<br>{r['workload']}" for r in runs]
    fig = go.Figure()
    for stage, color in STAGES.items():
        fig.add_trace(go.Bar(
            name=stage, x=labels, y=[r["stages_s"][stage] for r in runs],
            marker={"color": color, "line": {"color": "white", "width": 1}},
            texttemplate="%{y:.3f} s", textposition="inside",
            hovertemplate="%{x}<br>%{y:.4f} s<extra>%{fullData.name}</extra>",
        ))
    for label, run in zip(labels, runs):
        fig.add_annotation(x=label, y=run["mean_s"], yshift=22, showarrow=False,
                           text=f"{run['mean_s']:.3f} s mean<br>{run['median_s']:.3f} s median")
    fig.update_layout(
        title={"text": title + "<br><sup>FLUX.2 klein 4B · 10 measured generations · means</sup>"},
        barmode="stack", template="plotly_white", height=720,
        yaxis={"title": "Generation time (seconds)", "range": [0, max(r["mean_s"] for r in runs) * 1.22]},
        xaxis={"title": "Engine / GPU / workload", "tickfont": {"size": 11}},
        legend={"orientation": "h", "y": 1.12, "x": 0},
        margin={"t": 145, "b": 220}, uniformtext={"minsize": 10, "mode": "hide"},
    )
    fig.add_annotation(
        x=0, y=-0.47, xref="paper", yref="paper", showarrow=False, xanchor="left",
        align="left", font={"size": 11},
        text="Stage timing basis differs by engine; reloads included, initial loading excluded."
             "<br>Jetson uses a different workload/residency policy. Quality equivalence is untested.",
    )
    return fig


def upload(data, figures, output, args):
    import wandb

    receipt = output / "wandb-upload.json"
    identity = json.loads(receipt.read_text()) if receipt.exists() else {
        "id": uuid.uuid4().hex[:16], "entity": args.entity, "project": args.project,
    }
    if (identity["entity"], identity["project"]) != (args.entity, args.project):
        raise ValueError("Output directory belongs to a different W&B destination")
    write_json(receipt, identity)
    settings = wandb.Settings(x_disable_meta=True, x_disable_stats=True,
                              x_disable_machine_info=True, disable_git=True,
                              disable_code=True, disable_job_creation=True)
    with wandb.init(entity=args.entity, project=args.project, id=identity["id"],
                    resume="allow", name="baseline-stage-comparison", job_type="analysis",
                    tags=["analysis", "stage-breakdown", "unprofiled-source-data"],
                    config={"measurement_basis": "mean", "method": data["method"],
                            "source_runs": [r["run_dir"] for r in data["runs"]],
                            "packages": {p: importlib.metadata.version(p) for p in ["plotly", "wandb"]}},
                    settings=settings, dir=str(output)) as run:
        table = wandb.Table(columns=["source_run", "engine", "gpu", "workload", "stage", "mean_seconds"])
        for record in data["runs"]:
            for stage, seconds in record["stages_s"].items():
                table.add_data(record["run_dir"], record["engine"], record["gpu"],
                               record["workload"], stage, seconds)
        run.log({**{name: wandb.Plotly(fig) for name, fig in figures.items()}, "stage_data": table})
        artifact = wandb.Artifact("baseline-stage-comparison", type="analysis")
        for name in ["chart-data.json", "overview.html", "a100-detail.html"]:
            artifact.add_file(str(output / name), name=name)
        artifact.add_file(str(Path(__file__).resolve()), name="plot_stage_breakdown.py")
        run.log_artifact(artifact)
        identity["url"] = run.url
    identity["status"] = "uploaded"
    write_json(receipt, identity)
    print(identity["url"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "stage-breakdown")
    parser.add_argument("--data-json", type=Path)
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--entity", default="mfaruqi-purdue-university")
    parser.add_argument("--project", default="on-device-diffusion")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = json.loads(args.data_json.read_text()) if args.data_json else collect(args.root)
    write_json(args.output / "chart-data.json", data)
    figures = {
        "overview": figure(data["runs"], "Generation time by pipeline stage"),
        "a100-detail": figure([r for r in data["runs"] if r["gpu"] == "A100-PCIE-40GB"],
                               "A100: generation time by pipeline stage"),
    }
    for name, fig in figures.items():
        fig.write_html(args.output / f"{name}.html", include_plotlyjs=True)
    print(f"Validated {len(data['runs'])} baselines; wrote interactive charts to {args.output}")
    if args.upload:
        upload(data, figures, args.output, args)


if __name__ == "__main__":
    main()
