"""Plot a saved mixed-plane selection diagnostic, not image-quality scores."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root, output = Path(args.run), Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    if json.loads((root / "status.json").read_text())["status"] != "completed":
        raise ValueError("Require a completed candidate selection")
    summary = json.loads((root / "summary.json").read_text())
    curve, candidate = summary["allocation_curve"], summary["candidate_proxy"]
    fixed_bytes = (
        summary["assembly"]["packed_file_bytes"] - candidate["file_bytes_for_variable_matrices"]
    )
    sizes = [(r["file_bytes_for_variable_matrices"] + fixed_bytes) / 2**30 for r in curve]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), constrained_layout=False)
    for ax, split, title, color in zip(
        axes,
        ("fit", "calibration_validation"),
        ("Fitting calls · used for selection", "Calibration validation · evaluation only"),
        ("#245AA8", "#C15A19"),
    ):
        key = split + "_mean_projection_error_proxy"
        values = [r[key] for r in curve]
        ax.plot(sizes, values, color=color, marker="o", markersize=3, linewidth=1.8)
        k = candidate["lower_count"]
        ax.scatter(
            sizes[k],
            values[k],
            color="#111827",
            marker="D",
            s=48,
            zorder=3,
            label=f"Candidate: {k} matrices at lower precision",
        )
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Packed selected-matrix files (GiB)")
        ax.set_ylabel("Mean local projection-error proxy")
        ax.grid(alpha=0.22)
        ax.legend(loc="best", fontsize=8, frameon=False)
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    fig.suptitle("Mixed planes: subset storage versus local projection error", fontsize=14)
    fig.text(
        0.5,
        0.035,
        "BF16 remainder excluded from storage. Local projection error is not image quality.\n"
        "Curves combine independent projection diagnostics; mixed-model images are untested.",
        ha="center",
        fontsize=9,
        color="#4B5563",
    )
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.25, top=0.83, wspace=0.3)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
