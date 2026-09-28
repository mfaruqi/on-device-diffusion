---
type: project
summary: Registry of experiments — one row each, linking to the full record and the run directories. Details never live here.
status: active
updated: 2026-09-28
---

# Experiment registry

One row per experiment ([SCHEMA.md](SCHEMA.md#anti-flood-rules)). Numbers and discussion live in the
record. The viewer column will hold a link once a results viewer is chosen
([D-007](project/decisions.md#d-007-results-stay-in-run-directories-viewer-undecided)). The lint checks
that every record and every run directory appears here.

| Experiment | Device | Engine | Question | Status | Record | Run directories | Viewer |
|---|---|---|---|---|---|---|---|
| a100-flux-klein-001 | A100-PCIE-40GB | pytorch-diffusers | Reference latency, stage split and memory of FLUX.2 klein (BF16, 1024², 4 steps) | complete | [record](../experiments/a100-flux-klein-001.md) | [baseline](../results/runs/a100-flux-klein-001__baseline__20260923-221530/), [profile](../results/runs/a100-flux-klein-001__profile__20260923-221717/), [repeat](../results/runs/a100-flux-klein-001__repeat__20260925-123908/) | [W&B](https://wandb.ai/mfaruqi-purdue-university/on-device-diffusion/groups/a100-flux-klein-001) |
| a100-sdcpp-flux-klein-001 | A100-PCIE-40GB | stable-diffusion.cpp | Same workload and weights with the sd.cpp engine | complete | [record](../experiments/a100-sdcpp-flux-klein-001.md) | [baseline](../results/runs/a100-sdcpp-flux-klein-001__baseline__20260925-114328/), [profile](../results/runs/a100-sdcpp-flux-klein-001__profile__20260925-124703/), [SXM4 run](../results/runs/a100-sdcpp-flux-klein-001__baseline__20260925-114138/), [failed profile](../results/runs/a100-sdcpp-flux-klein-001__profile__20260925-124021/) | [W&B](https://wandb.ai/mfaruqi-purdue-university/on-device-diffusion/groups/a100-sdcpp-flux-klein-001) |

| Experiment | Device | Engine | Question | Status | Record | Run directories | Viewer |
|---|---|---|---|---|---|---|---|
| jetson-flux-klein-001 | Jetson Orin Nano | stable-diffusion.cpp | Quantized 512×512 all-resident feasibility | failed; partial evidence import | [record](../experiments/jetson-flux-klein-001.md) | [attempt](../results/runs/jetson-flux-klein-001__attempt__20260928-183618/) | [W&B](https://wandb.ai/mfaruqi-purdue-university/on-device-diffusion/groups/jetson-flux-klein-001) |
| jetson-flux-klein-002 | Jetson Orin Nano | stable-diffusion.cpp | Quantized 512×512 segmented feasibility | failed; partial evidence import | [record](../experiments/jetson-flux-klein-002.md) | [attempt](../results/runs/jetson-flux-klein-002__attempt__20260928-184410/) | [W&B](https://wandb.ai/mfaruqi-purdue-university/on-device-diffusion/groups/jetson-flux-klein-002) |
| jetson-flux-klein-003 | Jetson Orin Nano | stable-diffusion.cpp | Quantized 512×512 disk-backed feasibility | complete; visual smoke check passed | [record](../experiments/jetson-flux-klein-003.md) | [attempt](../results/runs/jetson-flux-klein-003__attempt__20260928-185009/) | [W&B](https://wandb.ai/mfaruqi-purdue-university/on-device-diffusion/groups/jetson-flux-klein-003) |

## Comparisons

| Comparison | Runs | Record |
|---|---|---|
| PyTorch vs sd.cpp, FLUX.2 klein, A100-PCIE | a100-flux-klein-001, a100-sdcpp-flux-klein-001 | [record](../experiments/compare-a100-pytorch-vs-sdcpp-flux-klein.md) |
