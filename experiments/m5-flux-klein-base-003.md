---
type: experiment-record
id: m5-flux-klein-base-003
status: complete
device: macbook-air-m5
engine: pytorch-diffusers
runs: [m5-flux-klein-base-003__attempt__20261010-214141, m5-flux-klein-base-003__baseline__20261010-214732]
updated: 2026-10-10
---

# m5-flux-klein-base-003

## Question

Week 2 Apple baseline work supporting RQ1: can this labelled configuration complete the full repeated protocol on the M5?

## Setup

M5 BF16 MPS flux-klein-base: VAE tiling only against m5-flux-klein-base-001; 256-pixel tiles and 25 percent overlap. Same checkpoint and sampling; tiled decode may change image pixels.

[Configuration](../configs/m5-flux-klein-base-bf16-512-vae-tiled256.resolved.json). One first image, three discarded warm-ups and ten measured generations for the full protocol. Smoke gates use one first and one measured image. Parameters, model revisions, backend and precision are explicit in each execution config.

[M5 procedure](../wiki/methods/m5-howto.md); [metric definitions](../wiki/methods/baseline-metrics.md).

## Results

### m5-flux-klein-base-003__attempt__20261010-214141

[Status](../results/runs/m5-flux-klein-base-003__attempt__20261010-214141/status.json), [execution config](../results/runs/m5-flux-klein-base-003__attempt__20261010-214141/config.json). Status: complete.

[Summary](../results/runs/m5-flux-klein-base-003__attempt__20261010-214141/summary.json), [validation](../results/runs/m5-flux-klein-base-003__attempt__20261010-214141/validation.json).

| Metric | Median (s) | Min–max (s) |
|---|---:|---:|
| End to end | 164.187 | 164.187–164.187 |
| Text encoding | 2.786 | 2.786–2.786 |
| Denoising | 160.566 | 160.566–160.566 |
| VAE decoding | 0.794 | 0.794–0.794 |

Measured generations: 1; first image: 158.672 s; cached load: 24.427 s.
Output hash repeatability: True. This is not a quality assessment.

Whole-system available minimum: 0.024 GiB; whole-system swap start/peak: 4.117/12.452 GiB. Harness/process peak RSS: 0.497 GiB. These must not be interpreted as GPU allocation peaks.

### m5-flux-klein-base-003__baseline__20261010-214732

[Status](../results/runs/m5-flux-klein-base-003__baseline__20261010-214732/status.json), [execution config](../results/runs/m5-flux-klein-base-003__baseline__20261010-214732/config.json). Status: complete.

[Summary](../results/runs/m5-flux-klein-base-003__baseline__20261010-214732/summary.json), [validation](../results/runs/m5-flux-klein-base-003__baseline__20261010-214732/validation.json).

| Metric | Median (s) | Min–max (s) |
|---|---:|---:|
| End to end | 146.049 | 131.420–158.936 |
| Text encoding | 1.514 | 1.320–2.271 |
| Denoising | 143.696 | 129.514–156.699 |
| VAE decoding | 0.623 | 0.562–1.018 |

Measured generations: 10; first image: 172.797 s; cached load: 17.892 s.
Output hash repeatability: True. This is not a quality assessment.

Whole-system available minimum: 0.082 GiB; whole-system swap start/peak: 7.436/11.681 GiB. Harness/process peak RSS: 0.497 GiB. These must not be interpreted as GPU allocation peaks.

## Interpretation

A completed full protocol establishes execution and repeated timing under the captured system state. User applications, system swap and background preparation were not isolated. CPU-side and GPU-side timing boundaries follow the linked metric definitions. No formal quality eligibility or controlled performance claim is established.

## Next experiment

Repeat this exact configuration under an isolated system state if a controlled performance comparison is required.

## Reference evidence

[Same-engine reference](../experiments/m5-flux-klein-base-001.md). Image differences are diagnostic only; see each run’s `reference-image-check.json`.

## Execution and image audit

[Protocol validation](../results/runs/m5-flux-klein-base-003__baseline__20261010-214732/validation.json) confirms 1+3+10 generations and timing continuity. [Tiling receipt](../results/runs/m5-flux-klein-base-003__baseline__20261010-214732/vae-tiling.json) confirms tiled decoding on all 14 generations. The retained image differs from the untiled reference ([pixel diagnostic](../results/runs/m5-flux-klein-base-003__baseline__20261010-214732/reference-image-check.json)); no formal quality eligibility is established.
