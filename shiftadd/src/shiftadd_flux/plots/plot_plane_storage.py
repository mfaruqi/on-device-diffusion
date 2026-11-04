"""Compare recorded plane-allocation storage with calculated dense BF16 weight payload."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[4]
    runs = root / "results/runs"
    sources = {}

    def read(run, filename="summary.json"):
        directory = runs / run
        if json.loads((directory / "status.json").read_text())["status"] != "completed":
            raise ValueError(f"Incomplete source: {run}")
        path = directory / filename
        sources[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())

    dual = read("m5-flux-klein-shiftadd-015__attempt__20261103-all-dual")
    single_id = "m5-flux-klein-shiftadd-016__attempt__20261103-all-single"
    single = read(single_id)
    inventory = read(single_id, "source-checkpoint-inventory.json")
    p3 = read("m5-flux-klein-shiftadd-022__attempt__20261103-mixed-fused")
    p2 = read("m5-flux-klein-shiftadd-023__attempt__20261103-selected-p2", "aggregate.json")
    p1 = read("m5-flux-klein-shiftadd-024__attempt__20261103-selected-p1", "aggregate.json")
    package = read("m5-flux-klein-shiftadd-020__attempt__20261103-package")
    int4_id = "m5-flux-klein-uniform-001__conversion__20261102"
    int4 = read(int4_id)
    int4_config = read(int4_id, "config.json")
    assert int4_config["source_revision"] == inventory["source_revision"]
    assert int4["bits"] == 4 and int4["group_size"] == 128
    storage = dual["storage"] + single["storage"]
    selected_names = {r["module"] + ".weight" for r in storage}
    int4_rows = [r for r in int4["results"] if r["tensor"] in selected_names]
    assert len(int4_rows) == 50 and {r["tensor"] for r in int4_rows} == selected_names
    for row in int4_rows:
        assert row["shape"] == inventory["tensors"][row["tensor"]]["shape"]
    int4_bytes = sum(r["artifact_bytes"] for r in int4_rows)
    assert len(storage) == 50 and len({r["module"] for r in storage}) == 50
    selected_bf16 = sum(r["output_rows"] * r["channels"] * 2 for r in storage)
    total_bf16 = inventory["source_payload_bytes"]
    remainder = total_bf16 - selected_bf16
    packed = [
        sum(r["file_bytes"] for r in storage),
        p3["assembly"]["packed_file_bytes"],
        p2["assembled_fifty_file_bytes"],
        p1["assembled_fifty_file_bytes"],
    ]
    rows = [
        {"case": "BF16", "selected_bytes": selected_bf16},
        {"case": "symmetric_INT4", "selected_bytes": int4_bytes},
    ]
    rows += [
        {"case": name, "selected_bytes": size}
        for name, size in zip(["p4", "p3p4", "p2p4", "p1p4"], packed)
    ]
    for row in rows:
        row.update(
            unchanged_bf16_payload_bytes=remainder,
            transformer_weight_accounting_bytes=remainder + row["selected_bytes"],
            dense_runtime_weight_payload_bytes=total_bf16,
        )
    args.output.mkdir(parents=True, exist_ok=True)
    data = {
        "scope": "transformer only; excludes text encoder, VAE, activations and workspaces",
        "storage_basis": "BF16 tensor payload; packed NPZ file bytes include headers and scales",
        "whole_transformer_basis": "remainder tensor payload plus selected files; excludes remainder container header and package metadata",
        "runtime_basis": "calculated persistent dense BF16 parameter payload, not measured peak GPU memory",
        "int4_control": "same 50 matrices; symmetric signed INT4 codes, FP32 scales, group 128; dense BF16 image execution, not SVDQuant/Nunchaku or GGUF Q4",
        "int4_selected_tensors": sorted(selected_names),
        "actual_saved_p4_package_bytes": package["package_total_bytes"],
        "rows": rows,
        "source_sha256": sources,
    }
    (args.output / "data.json").write_text(json.dumps(data, indent=2) + "\n")
    with (args.output / "sizes.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    colors = ["#718096", "#7c3aed", "#2563eb", "#0d9488", "#d97706", "#dc2626"]
    labels = [
        "Original BF16",
        "Symmetric INT4",
        "4 planes",
        "Mixed 3 / 4",
        "Mixed 2 / 4",
        "Mixed 1 / 4",
    ]
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    values = [row["selected_bytes"] / 2**30 for row in rows]
    bars = axes[0].barh(labels, values, color=colors)
    axes[0].bar_label(bars, labels=[f"{v:.3f}" for v in values], padding=5)
    axes[0].set_xlim(0, 4.2)
    axes[0].set_title("Selected 50 matrices: stored size")
    axes[0].set_xlabel("GiB (BF16 payload / packed files)")
    axes[0].invert_yaxis()
    unchanged = remainder / 2**30
    axes[1].barh(
        labels, [unchanged] * len(rows), color="#d1d5db", label="Unchanged BF16 tensor payload"
    )
    bars = axes[1].barh(labels, values, left=unchanged, color=colors)
    totals = [v + unchanged for v in values]
    axes[1].bar_label(bars, labels=[f"{v:.3f}" for v in totals], padding=5)
    axes[1].set_xlim(0, 8)
    axes[1].set_title("Whole transformer: weight-size accounting")
    axes[1].set_xlabel("GiB; excludes package metadata / remainder header")
    axes[1].invert_yaxis()
    axes[1].legend(loc="lower right", fontsize=9)
    fig.suptitle("Stored weights: conventional INT4 and ShiftAdd allocations", fontsize=17)
    fig.text(
        0.5,
        0.025,
        "Mixed cases change the same 10 fused matrices; the other 40 stay at four planes. Disk sizes are not GPU memory measurements.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    fig.savefig(args.output / "stored-weight-sizes.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(14, 6))
    values = [total_bf16 / 2**30] * len(rows)
    bars = ax.barh(labels, values, color=colors)
    ax.bar_label(bars, labels=[f"{v:.3f} GiB" for v in values], padding=5)
    ax.set_xlim(0, 8.6)
    ax.invert_yaxis()
    ax.set_title("Current image runner: identical dense BF16 weight footprint", pad=16)
    ax.set_xlabel("Calculated transformer parameter payload (GiB)")
    fig.text(
        0.5,
        0.025,
        "Not measured peak memory. Text encoder, VAE, activations, buffers and loading temporaries add further memory.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(args.output / "dense-runtime-weight-footprint.png", dpi=160)
    plt.close(fig)
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
