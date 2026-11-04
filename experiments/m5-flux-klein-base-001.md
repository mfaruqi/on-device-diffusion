---
type: experiment-record
id: m5-flux-klein-base-001
status: complete
device: macbook-air-m5
engine: pytorch-diffusers
runs: [m5-flux-klein-base-001__attempt__20261010-183713, m5-flux-klein-base-001__baseline__20261010-184312]
updated: 2026-10-10
---

# m5-flux-klein-base-001

## Question

Week 2 Apple baseline work supporting RQ1: can this labelled configuration complete the full repeated protocol on the M5?

## Setup

M5 PyTorch/Diffusers BF16 reference, Base 50 steps CFG 4, 512x512, no cache or offload.

[Configuration](../configs/m5-flux-klein-base-bf16-512.resolved.json). One first image, three discarded warm-ups and ten measured generations for the full protocol. Smoke gates use one first and one measured image. Parameters, model revisions, backend and precision are explicit in each execution config.

[M5 procedure](../wiki/methods/m5-howto.md); [metric definitions](../wiki/methods/baseline-metrics.md).

## Results

### m5-flux-klein-base-001__attempt__20261010-183713

[Status](../results/runs/m5-flux-klein-base-001__attempt__20261010-183713/status.json), [execution config](../results/runs/m5-flux-klein-base-001__attempt__20261010-183713/config.json). Status: complete.

[Summary](../results/runs/m5-flux-klein-base-001__attempt__20261010-183713/summary.json), [validation](../results/runs/m5-flux-klein-base-001__attempt__20261010-183713/validation.json).

| Metric | Median (s) | Min–max (s) |
|---|---:|---:|
| End to end | 169.888 | 169.888–169.888 |
| Text encoding | 1.743 | 1.743–1.743 |
| Denoising | 167.620 | 167.620–167.620 |
| VAE decoding | 0.504 | 0.504–0.504 |

Measured generations: 1; first image: 170.131 s; cached load: 16.014 s.
Output hash repeatability: True. This is not a quality assessment.

Whole-system available minimum: 0.016 GiB; whole-system swap start/peak: 4.781/9.046 GiB. Harness/process peak RSS: 0.496 GiB. These must not be interpreted as GPU allocation peaks.

### m5-flux-klein-base-001__baseline__20261010-184312

[Status](../results/runs/m5-flux-klein-base-001__baseline__20261010-184312/status.json), [execution config](../results/runs/m5-flux-klein-base-001__baseline__20261010-184312/config.json). Status: complete.

[Summary](../results/runs/m5-flux-klein-base-001__baseline__20261010-184312/summary.json), [validation](../results/runs/m5-flux-klein-base-001__baseline__20261010-184312/validation.json).

| Metric | Median (s) | Min–max (s) |
|---|---:|---:|
| End to end | 167.101 | 165.326–182.825 |
| Text encoding | 2.682 | 1.804–14.414 |
| Denoising | 163.212 | 161.646–167.096 |
| VAE decoding | 0.570 | 0.474–3.770 |

Measured generations: 10; first image: 170.825 s; cached load: 16.711 s.
Output hash repeatability: True. This is not a quality assessment.

Whole-system available minimum: 0.032 GiB; whole-system swap start/peak: 4.700/8.692 GiB. Harness/process peak RSS: 0.498 GiB. These must not be interpreted as GPU allocation peaks.

## Interpretation

A completed full protocol establishes execution and repeated timing under the captured system state. User applications, system swap and background preparation were not isolated. CPU-side and GPU-side timing boundaries follow the linked metric definitions. No formal quality eligibility or controlled performance claim is established.

## Next experiment

Repeat this exact configuration under an isolated system state if a controlled performance comparison is required.
