---
type: experiment-record
id: m5-flux-klein-005
status: complete
device: macbook-air-m5
engine: pytorch-diffusers
runs: [m5-flux-klein-005__attempt__20261015-132946, m5-flux-klein-005__baseline__20261015-133226]
updated: 2026-10-15
---

# m5-flux-klein-005

## Question

Week 2–3 Apple individual-option work supporting RQ1: can this labelled configuration complete the full repeated protocol on the M5?

## Setup

M5 BF16 Klein distilled: taylorseer only versus configs/m5-flux-klein-bf16-512.resolved.json; no conditioning reuse, offload or tiling.

[Configuration](../configs/m5-flux-klein-bf16-512-taylorseer.resolved.json). One first image, three discarded warm-ups and ten measured generations for the full protocol. Smoke gates use one first and one measured image. Parameters, model revisions, backend and precision are explicit in each execution config.

[M5 procedure](../wiki/methods/m5-howto.md); [metric definitions](../wiki/methods/baseline-metrics.md).

## Results

### m5-flux-klein-005__attempt__20261015-132946

[Status](../results/runs/m5-flux-klein-005__attempt__20261015-132946/status.json), [execution config](../results/runs/m5-flux-klein-005__attempt__20261015-132946/config.json). Status: complete.

[Summary](../results/runs/m5-flux-klein-005__attempt__20261015-132946/summary.json), [validation](../results/runs/m5-flux-klein-005__attempt__20261015-132946/validation.json).

| Metric | Median (s) | Min–max (s) |
|---|---:|---:|
| End to end | 64.455 | 64.455–64.455 |
| Text encoding | 34.627 | 34.627–34.627 |
| Denoising | 28.823 | 28.823–28.823 |
| VAE decoding | 0.966 | 0.966–0.966 |

Measured generations: 1; first image: 53.725 s; cached load: 35.633 s.
Output hash repeatability: True. This is not a quality assessment.

Whole-system available minimum: 0.090 GiB; whole-system swap start/peak: 17.768/26.066 GiB. Harness/process peak RSS: 0.497 GiB. These must not be interpreted as GPU allocation peaks.

### m5-flux-klein-005__baseline__20261015-133226

[Status](../results/runs/m5-flux-klein-005__baseline__20261015-133226/status.json), [execution config](../results/runs/m5-flux-klein-005__baseline__20261015-133226/config.json). Status: complete.

[Summary](../results/runs/m5-flux-klein-005__baseline__20261015-133226/summary.json), [validation](../results/runs/m5-flux-klein-005__baseline__20261015-133226/validation.json).

| Metric | Median (s) | Min–max (s) |
|---|---:|---:|
| End to end | 57.556 | 41.794–69.497 |
| Text encoding | 27.492 | 25.035–36.664 |
| Denoising | 17.818 | 12.136–34.886 |
| VAE decoding | 4.900 | 4.300–14.139 |

Measured generations: 10; first image: 62.757 s; cached load: 27.089 s.
Output hash repeatability: True. This is not a quality assessment.

Whole-system available minimum: 0.034 GiB; whole-system swap start/peak: 17.771/25.123 GiB. Harness/process peak RSS: 0.497 GiB. These must not be interpreted as GPU allocation peaks.

## Interpretation

A completed full protocol establishes execution and repeated timing under the captured system state. User applications, system swap and background preparation were not isolated. CPU-side and GPU-side timing boundaries follow the linked metric definitions. No formal quality eligibility or controlled performance claim is established.

## Next experiment

Repeat this exact configuration under an isolated system state if a controlled performance comparison is required.

## Reference and option audit

[Reference](m5-flux-klein-001.md). TaylorSeer predicts attention outputs in 25 attention modules. Each of the 14 generations executed 75 computed and 25 predicted attention calls, with per-image state reset. These are attention calls, not whole-transformer step skips. No formal quality approval. Significant whole-system swap is a performance confound.

[Execution audit](../results/runs/m5-flux-klein-005__baseline__20261015-133226/step-cache.json); [reference-image diagnostic](../results/runs/m5-flux-klein-005__baseline__20261015-133226/reference-image-check.json).
