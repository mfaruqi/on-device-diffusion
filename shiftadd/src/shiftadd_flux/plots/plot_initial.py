"""Static evidence figure from saved images and weight-file reports."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[3]
out = root / "runs/figures"
out.mkdir(parents=True, exist_ok=True)
store = json.loads((root / "artifacts/transformer-p4-001/storage-report.json").read_text())
mixed = json.loads((root / "artifacts/transformer-radix-mixed-001/storage-report.json").read_text())
cases = [
    ("BF16 reference", "a100-greedy-paired-001/bf16.png", store["source_file_bytes"]),
    (
        "Greedy 4 planes",
        "a100-greedy-paired-001/all-matrices-p4.png",
        store["replacement_bundle_weight_bytes"],
    ),
    (
        "Joint radix 4 planes",
        "a100-radix-paired-001/radix-p4.png",
        store["replacement_bundle_weight_bytes"],
    ),
    (
        "Mixed 4 / 5 planes",
        "a100-mixed-paired-001/mixed-p4-p5.png",
        mixed["replacement_bundle_bytes"],
    ),
]
fig, axes = plt.subplots(1, 4, figsize=(15, 4.8))
for ax, (label, image, bytes_) in zip(axes, cases):
    ax.imshow(plt.imread(root / "runs" / image))
    ax.set_axis_off()
    ax.set_title(
        label + "\n" + f"{bytes_ / 2**30:.3f} GiB stored transformer weights",
        fontsize=11,
    )
fig.suptitle(
    "Distilled Klein: initial weight conversion and reconstructed-image smoke",
    fontsize=16,
    y=0.98,
)
fig.text(
    0.5,
    0.08,
    "Same prompt and seed • 512 × 512 • 4 steps • A100 SXM4 • One calibration image per case",
    ha="center",
    fontsize=11,
)
fig.text(
    0.5,
    0.035,
    "Dense BF16 execution after reconstruction. Storage savings do not establish inference-memory, speed, energy or matched-quality gains.",
    ha="center",
    fontsize=10,
)
fig.subplots_adjust(left=0.015, right=0.985, top=0.80, bottom=0.14, wspace=0.04)
fig.savefig(out / "initial-comparison.png", dpi=180)
print(out / "initial-comparison.png")
