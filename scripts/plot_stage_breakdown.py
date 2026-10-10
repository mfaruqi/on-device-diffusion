#!/usr/bin/env python3
"""Plot saved completed baselines; optionally upload a separate W&B analysis run.

Dependencies: plotly, wandb (only for --upload), pillow (uploading PNG panels).
Example: python scripts/plot_stage_breakdown.py --output output/stage-breakdown --upload
No model execution. --data-json reuses an exported dataset on another host.
--manifest selects labelled runs and chart groups; without it, plot Week 1 baselines.
"""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import statistics
import textwrap
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
    "Text conditioning": "#3C8DAD",
    "Encoding + setup (edge-dit)": "#279D91",
    "Denoising": "#6652A2",
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
    " Conditioning hits include retrieval/setup; EasyCache is approximate and carries no "
    "formal quality acceptance. Edge-dit encoding includes latent/schedule setup and uses "
    "host system-clock stages with millisecond-rounded steady-clock totals. "
    "Whiskers show observed total min-max, not confidence intervals."
)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def collect(root, manifest=None):
    runs = []
    selections = manifest["runs"] if manifest else [
        dict(run_dir=name, engine=engine, gpu=gpu, workload=workload,
             label=engine) for name, engine, gpu, workload in BASELINES
    ]
    if len({r['run_dir'] for r in selections}) != len(selections):
        raise ValueError("Duplicate source run in manifest")
    for selection in selections:
        name = selection["run_dir"]
        engine, gpu, workload = (selection[k] for k in ["engine", "gpu", "workload"])
        directory = root / "results" / "runs" / name
        summary = json.loads((directory / "summary.json").read_text())
        status = json.loads((directory / "status.json").read_text())
        if status["status"] != "complete":
            raise ValueError(f"Incomplete baseline: {name}")
        if not any(f"__{kind}__" in name for kind in ["baseline", "repeat"]):
            raise ValueError(f"Not an unprofiled baseline/repeat: {name}")
        with (directory / "runs.csv").open() as stream:
            measured = [r for r in csv.DictReader(stream) if r["phase"] == "measured"]
        if len(measured) != summary["measured_runs"] or len(measured) != 10:
            raise ValueError(f"Expected ten measured generations: {name}")
        edge = engine == "edge-dit.cpp"
        metrics = METRICS[1:] if edge else METRICS
        means = {}
        for key in ["wall_ms", *metrics]:
            means[key] = statistics.mean(float(r[key]) for r in measured) / 1000
            if not math.isclose(means[key], summary["measured"][key]["mean"] / 1000,
                                rel_tol=1e-7, abs_tol=1e-7):
                raise ValueError(f"CSV/summary mean mismatch: {name}: {key}")
        encode = {}
        if edge:
            with (directory / "stages.csv").open() as stream:
                for row in csv.DictReader(stream):
                    if row["phase"] == "measured" and row["stage"] == "encode_setup":
                        if row['run_index'] in encode:
                            raise ValueError(f"Duplicate encode_setup stage: {name}")
                        encode[row['run_index']] = float(row['host_ms'])
            if set(encode) != {r['run_index'] for r in measured}:
                raise ValueError(f"Incomplete encode_setup stages: {name}")
        remaining = []
        for row in measured:
            encoding = encode[row['run_index']] if edge else float(row['text_encode_ms'])
            values = [float(row[k]) for k in ['wall_ms', 'denoise_ms', 'vae_decode_ms']]
            residual = values[0] - encoding - values[1] - values[2]
            if any(not math.isfinite(v) or v < 0 for v in [*values, encoding, residual]):
                raise ValueError(f"Invalid per-generation stage duration: {name}")
            remaining.append(residual / 1000)
        stages = {
            "Encoding + setup (edge-dit)" if edge else "Text conditioning":
                statistics.mean(encode.values()) / 1000 if edge else means['text_encode_ms'],
            "Denoising": means['denoise_ms'], "VAE decoding": means['vae_decode_ms'],
            "Output / remaining time": statistics.mean(remaining),
        }
        if not math.isclose(sum(stages.values()), means['wall_ms'], abs_tol=1e-7):
            raise ValueError(f"Invalid or negative stage duration: {name}")
        runs.append({
            "run_dir": name, "engine": engine, "gpu": gpu, "workload": workload,
            "label": selection.get('label', engine),
            "measured_count": len(measured), "stages_s": stages,
            "mean_s": means["wall_ms"],
            "median_s": summary["measured"]["wall_ms"]["median"] / 1000,
            "min_s": summary["measured"]["wall_ms"]["min"] / 1000,
            "max_s": summary["measured"]["wall_ms"]["max"] / 1000,
            "config": json.loads((directory / "config.json").read_text()),
            "source_sha256": {f: hashlib.sha256((directory / f).read_bytes()).hexdigest()
                              for f in ["summary.json", "runs.csv", "config.json", "stages.csv", "status.json"]},
        })
    return {"schema_version": 2, "method": METHOD, "runs": runs,
            "charts": manifest.get('charts') if manifest else None,
            "exclusions": manifest.get('exclusions', []) if manifest else []}


def figure(runs, title, note=None):
    import plotly.graph_objects as go

    labels = [f"{r.get('label', r['engine'])}<br>"
              f"{r['engine'].replace('stable-diffusion.cpp', 'sd.cpp')}<br>{r['gpu']}" for r in runs]
    if len(set(labels)) != len(labels):
        raise ValueError("Chart labels must distinguish every source run")
    fig = go.Figure()
    for stage, color in STAGES.items():
        if not any(stage in r['stages_s'] for r in runs):
            continue
        fig.add_trace(go.Bar(
            name=stage, x=labels, y=[r["stages_s"].get(stage, 0) for r in runs],
            customdata=[[r['run_dir'], r['workload']] for r in runs],
            marker={"color": color, "line": {"color": "white", "width": 1}},
            texttemplate="%{y:.3f} s", textposition="inside",
            hovertemplate="%{x}<br>%{y:.4f} s<br>%{customdata[1]}<br>%{customdata[0]}"
                          "<extra>%{fullData.name}</extra>",
        ))
    fig.add_trace(go.Scatter(
        name='Total median / min–max', x=labels, y=[r['median_s'] for r in runs],
        mode='markers', marker={'color': '#202936', 'symbol': 'diamond', 'size': 7},
        error_y={'type': 'data', 'symmetric': False,
                 'array': [r['max_s'] - r['median_s'] for r in runs],
                 'arrayminus': [r['median_s'] - r['min_s'] for r in runs],
                 'color': '#202936', 'thickness': 1.3, 'width': 5},
        hovertemplate='%{x}<br>Median %{y:.4f} s<extra>Total min–max whisker</extra>',
    ))
    for label, run in zip(labels, runs):
        fig.add_annotation(x=label, y=run["max_s"], yshift=22, showarrow=False,
                           text=f"{run['mean_s']:.3f} s mean<br>{run['median_s']:.3f} s median")
    fig.update_layout(
        title={"text": title + "<br><sup>10 measured generations · stacked means · diamonds: total medians</sup>"},
        barmode="stack", template="plotly_white", height=760, width=max(1100, len(runs)*190),
        yaxis={"title": "Generation time (seconds)", "range": [0, max(r["max_s"] for r in runs) * 1.22]},
        xaxis={"title": None, "tickfont": {"size": 11}, 'tickangle': 0},
        legend={"orientation": "h", "y": 1.12, "x": 0, 'font': {'size': 11}, 'traceorder': 'normal'},
        margin={"t": 150, "b": 185, 'l': 75, 'r': 30}, uniformtext={"minsize": 10, "mode": "hide"},
    )
    footer = note or (
        "Stage timing basis differs by engine; reloads included, initial loading excluded."
        "<br>Different device workloads/residency; quality equivalence untested."
    )
    footer = '<br>'.join(part for line in footer.split('<br>')
                          for part in textwrap.wrap(line, width=150))
    fig.add_annotation(
        x=0, y=-0.25, xref="paper", yref="paper", showarrow=False, xanchor="left", yanchor='top',
        align="left", font={"size": 11},
        text=footer,
    )
    return fig


def wandb_figure(fig):
    """W&B supplies a panel title and resizes Plotly; avoid paper-positioned headings."""
    import plotly.graph_objects as go

    compact = go.Figure(fig)
    compact.layout.title = None
    compact.layout.annotations = ()
    labels = list(compact.data[0].x)
    # The panel identifies the device/workload. Only repeat the differing labels on x.
    engines = {label.split('<br>')[-2] for label in labels}
    ticks = [label.split('<br>')[-2] if len(engines) > 1
             else '<br>'.join(label.split('<br>')[:-2]) for label in labels]
    if len(engines) == 1:
        ticks = [text.split('<br>')[0] if 'reference<br>' in text.lower() else text
                 for text in ticks]
        ticks = [text.replace('Initial reference', 'Initial ref.')
                 .replace('Repeat reference', 'Repeat ref.')
                 .replace('Original reference', 'Original ref.')
                 .replace('No reuse reference', 'Reference')
                 .replace('EasyCache<br>+ Conditioning reuse', 'EasyCache<br>+ conditioning')
                 .replace('Conditioning reuse', 'Conditioning<br>reuse') for text in ticks]
    compact.update_traces(textangle=0, selector={'type': 'bar'})
    compact.update_layout(
        width=None, height=None, autosize=True,
        margin={'t': 60, 'b': 100, 'l': 55, 'r': 15},
        legend={'orientation': 'h', 'x': 0, 'y': 1.02, 'yanchor': 'bottom',
                'font': {'size': 10}, 'traceorder': 'normal'},
        xaxis={'tickvals': labels, 'ticktext': ticks, 'tickangle': -45 if len(labels) > 4 else 0,
               'tickfont': {'size': 10}, 'automargin': True},
        yaxis={'title': {'text': 'Time (s)', 'font': {'size': 11}},
               'tickfont': {'size': 10}, 'automargin': True},
        font={'size': 11}, uniformtext={'minsize': 9, 'mode': 'hide'},
    )
    return compact


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
                    resume="allow", name=args.name, job_type="analysis",
                    tags=["analysis", "stage-breakdown", "unprofiled-source-data"],
                    config={"measurement_basis": "mean", "method": data["method"],
                            "source_runs": [r["run_dir"] for r in data["runs"]],
                            "packages": {p: importlib.metadata.version(p) for p in ["plotly", "wandb"]}},
                    settings=settings, dir=str(output)) as run:
        table = wandb.Table(columns=["source_run", "configuration", "engine", "gpu", "workload", "stage", "mean_seconds"])
        for record in data["runs"]:
            for stage, seconds in record["stages_s"].items():
                table.add_data(record["run_dir"], record.get('label', record['engine']).replace('<br>', ' '), record["engine"], record["gpu"],
                               record["workload"], stage, seconds)
        payload = {name: wandb.Plotly(wandb_figure(fig)) for name, fig in figures.items()}
        # Images preserve the publication layout when a W&B media panel is very short.
        for name in figures:
            png = output / f'{name}.png'
            if png.exists():
                payload[f'figures/{name}'] = wandb.Image(str(png), caption=name)
        run.log({**payload, "stage_data": table})
        artifact = wandb.Artifact(args.name, type="analysis")
        for name in ["chart-data.json", *[f'{key}.html' for key in figures]]:
            artifact.add_file(str(output / name), name=name)
        for key in figures:
            for extension in ['png', 'pdf']:
                path = output / f'{key}.{extension}'
                if path.exists():
                    artifact.add_file(str(path), name=path.name)
        if args.manifest:
            artifact.add_file(str(args.manifest), name='manifest.json')
        artifact.add_file(str(Path(__file__).resolve()), name="plot_stage_breakdown.py")
        run.log_artifact(artifact)
        run.summary['source_run_count'] = len(data['runs'])
        run.summary['chart_count'] = len(figures)
        run.summary['exclusions'] = data.get('exclusions', [])
        identity["url"] = run.url
    identity["status"] = "uploaded"
    write_json(receipt, identity)
    print(identity["url"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "stage-breakdown")
    parser.add_argument("--data-json", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--name", default="baseline-stage-comparison")
    parser.add_argument("--static", action="store_true", help="Also export PNG/PDF (requires kaleido and Chrome)")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--entity", default="mfaruqi-purdue-university")
    parser.add_argument("--project", default="on-device-diffusion")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.data_json and args.manifest:
        parser.error('Choose --manifest or --data-json, not both')
    manifest = json.loads(args.manifest.read_text()) if args.manifest else None
    data = json.loads(args.data_json.read_text()) if args.data_json else collect(args.root, manifest)
    # Older exported Week 1 datasets used these stage labels.
    for record in data['runs']:
        for old, new in [('Text encoding', 'Text conditioning'), ('Denoising (4 steps)', 'Denoising')]:
            if old in record['stages_s']:
                record['stages_s'][new] = record['stages_s'].pop(old)
    write_json(args.output / "chart-data.json", data)
    figures = {} if data.get('charts') else {
        "overview": figure(data["runs"], "Generation time by pipeline stage"),
        "a100-detail": figure([r for r in data["runs"] if r["gpu"] == "A100-PCIE-40GB"],
                               "A100: generation time by pipeline stage"),
    }
    lookup = {r['run_dir']: r for r in data['runs']}
    for chart in data.get('charts') or []:
        figures[chart['key']] = figure([lookup[name] for name in chart['runs']], chart['title'], chart['note'])
    for name, fig in figures.items():
        fig.write_html(args.output / f"{name}.html", include_plotlyjs=True)
        if args.static:
            fig.write_image(args.output / f'{name}.png', scale=2)
            fig.write_image(args.output / f'{name}.pdf')
    print(f"Validated {len(data['runs'])} baselines; wrote interactive charts to {args.output}")
    if args.upload:
        upload(data, figures, args.output, args)


if __name__ == "__main__":
    main()
