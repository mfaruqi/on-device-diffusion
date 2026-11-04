---
type: experiment-record
id: m5-flux-klein-base-005
status: complete
device: macbook-air-m5
engine: pytorch-diffusers
runs: [m5-flux-klein-base-005__attempt__20261015-143525, m5-flux-klein-base-005__baseline__20261015-143814]
updated: 2026-10-15
---

# m5-flux-klein-base-005

## Question

Week 2–3 Apple individual-option work supporting RQ1: can this labelled configuration complete the full repeated protocol on the M5?

## Setup

M5 BF16 Klein-base: taylorseer only versus configs/m5-flux-klein-base-bf16-512.resolved.json; no conditioning reuse, offload or tiling.

[Configuration](../configs/m5-flux-klein-base-bf16-512-taylorseer.resolved.json). One first image, three discarded warm-ups and ten measured generations for the full protocol. Smoke gates use one first and one measured image. Parameters, model revisions, backend and precision are explicit in each execution config.

[M5 procedure](../wiki/methods/m5-howto.md); [metric definitions](../wiki/methods/baseline-metrics.md).

## Results

### m5-flux-klein-base-005__attempt__20261015-143525

[Status](../results/runs/m5-flux-klein-base-005__attempt__20261015-143525/status.json), [execution config](../results/runs/m5-flux-klein-base-005__attempt__20261015-143525/config.json). Status: complete.

[Summary](../results/runs/m5-flux-klein-base-005__attempt__20261015-143525/summary.json), [validation](../results/runs/m5-flux-klein-base-005__attempt__20261015-143525/validation.json).

| Metric | Median (s) | Min–max (s) |
|---|---:|---:|
| End to end | 82.752 | 82.752–82.752 |
| Text encoding | 14.231 | 14.231–14.231 |
| Denoising | 65.450 | 65.450–65.450 |
| VAE decoding | 3.037 | 3.037–3.037 |

Measured generations: 1; first image: 57.083 s; cached load: 23.885 s.
Output hash repeatability: True. This is not a quality assessment.

Whole-system available minimum: 0.028 GiB; whole-system swap start/peak: 13.543/20.031 GiB. Harness/process peak RSS: 0.498 GiB. These must not be interpreted as GPU allocation peaks.

### m5-flux-klein-base-005__baseline__20261015-143814

[Status](../results/runs/m5-flux-klein-base-005__baseline__20261015-143814/status.json), [execution config](../results/runs/m5-flux-klein-base-005__baseline__20261015-143814/config.json). Status: complete.

[Summary](../results/runs/m5-flux-klein-base-005__baseline__20261015-143814/summary.json), [validation](../results/runs/m5-flux-klein-base-005__baseline__20261015-143814/validation.json).

| Metric | Median (s) | Min–max (s) |
|---|---:|---:|
| End to end | 92.474 | 83.386–100.159 |
| Text encoding | 21.319 | 14.442–26.882 |
| Denoising | 69.631 | 66.433–72.872 |
| VAE decoding | 0.524 | 0.364–3.081 |

Measured generations: 10; first image: 69.681 s; cached load: 21.525 s.
Output hash repeatability: True. This is not a quality assessment.

Whole-system available minimum: 0.026 GiB; whole-system swap start/peak: 14.272/20.154 GiB. Harness/process peak RSS: 0.497 GiB. These must not be interpreted as GPU allocation peaks.

## Interpretation

A completed full protocol establishes execution and repeated timing under the captured system state. User applications, system swap and background preparation were not isolated. CPU-side and GPU-side timing boundaries follow the linked metric definitions. No formal quality eligibility or controlled performance claim is established.

## Next experiment

Repeat this exact configuration under an isolated system state if a controlled performance comparison is required.

## Reference and option audit

[Reference](m5-flux-klein-base-001.md). TaylorSeer executed 650 computed and 1,850 predicted attention calls per image across 25 attention modules, 50 steps and separate conditional/unconditional contexts. Cache state resets each generation. These are attention predictions, not whole-transformer step skips. The inspected sign reads “HELLO WORLD”, but the image differs from the uncached reference; no formal quality approval is established. Significant whole-system swap prevents a controlled speedup claim.

[Execution receipt](../results/runs/m5-flux-klein-base-005__baseline__20261015-143814/step-cache.json); [image diagnostic](../results/runs/m5-flux-klein-base-005__baseline__20261015-143814/reference-image-check.json).
