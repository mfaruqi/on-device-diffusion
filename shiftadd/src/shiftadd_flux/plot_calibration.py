"""Render all available calibration image cases without altering the images."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[2]
run = root / "runs/a100-calibration-001"
config = json.loads((run / "config.json").read_text())
fig, axes = plt.subplots(len(config["samples"]), len(config["cases"]), figsize=(10, 13))
for row, sample in enumerate(config["samples"]):
    for col, case in enumerate(config["cases"]):
        ax = axes[row, col]
        path = run / (sample["id"] + "--" + case["name"] + ".png")
        if path.exists():
            ax.imshow(plt.imread(path))
        else:
            ax.text(0.5, 0.5, "Pending", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
        if row == 0:
            ax.set_title(case["name"], fontsize=12)
        if col == 0:
            ax.set_ylabel(sample["id"], fontsize=12)
fig.suptitle("Klein calibration-only smoke: matched prompts and seeds", fontsize=15)
fig.text(
    0.5,
    0.015,
    "Dense reconstructed BF16 execution • Visual diagnostic only • No held-out quality or speed claim",
    ha="center",
    fontsize=10,
)
fig.subplots_adjust(left=0.06, right=0.98, top=0.94, bottom=0.04, hspace=0.04, wspace=0.04)
out = root / "runs/figures"
out.mkdir(exist_ok=True)
fig.savefig(out / "calibration-comparison.png", dpi=150)
