"""Standalone storage figure from a verified bundle inventory."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--audit", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    out = Path(a.output)
    if out.exists():
        raise FileExistsError(out)
    data = json.loads(Path(a.audit).read_text())
    bundles = data["bundles"]
    names = [
        "BF16 source",
        "Conventional INT4",
        "Radix p4",
        "Compact radix p4",
        "Compact mixed p4/p5",
    ]
    keys = [
        "transformer-uniform-int4-001",
        "transformer-radix-p4-001",
        "transformer-radix-compact-p4-001",
        "transformer-radix-compact-mixed-001",
    ]
    sizes = [data["source_bytes"] / 2**30] + [bundles[k]["gib"] for k in keys]
    reductions = [0] + [100 * bundles[k]["reduction_vs_source"] for k in keys]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    fig.subplots_adjust(left=0.26, right=0.98, top=0.88, bottom=0.19)
    ax.barh(names, sizes, color=["#666666", "#4477aa", "#aa3377", "#228833", "#ccbb44"])
    ax.invert_yaxis()
    ax.set_xlim(0, 8.4)
    ax.set_xlabel("Transformer weight files (GiB)")
    ax.set_title("Distilled Klein: saved transformer storage", loc="left", weight="bold")
    for i, (size, reduction) in enumerate(zip(sizes, reductions)):
        ax.text(
            size + 0.08,
            i,
            f"{size:.3f} GiB" + (f"  (−{reduction:.2f}%)" if i else ""),
            va="center",
            fontsize=10,
        )
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(
        0.01,
        0.01,
        "File sizes include unchanged transformer tensors. Quality is not matched; live inference memory is not measured.",
        fontsize=9,
    )
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
