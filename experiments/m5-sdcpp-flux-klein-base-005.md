---
type: experiment-record
id: m5-sdcpp-flux-klein-base-005
status: complete
device: macbook-air-m5
engine: stable-diffusion-cpp
runs: [m5-sdcpp-flux-klein-base-005__attempt__20261011-025051, m5-sdcpp-flux-klein-base-005__baseline__20261011-030358]
updated: 2026-10-10
---

# m5-sdcpp-flux-klein-base-005

## Question

Week 2 Apple baseline work supporting RQ1: can this labelled configuration complete the full repeated protocol on the M5?

## Setup

M5 flux-klein-base: mmap only against m5-sdcpp-flux-klein-base-001; same mixed-Q4 weights, 512x512, residency and sampling. 

[Configuration](../configs/m5-sdcpp-flux-klein-base-q4-512-mmap.resolved.json). One first image, three discarded warm-ups and ten measured generations for the full protocol. Smoke gates use one first and one measured image. Parameters, model revisions, backend and precision are explicit in each execution config.

[M5 procedure](../wiki/methods/m5-howto.md); [metric definitions](../wiki/methods/baseline-metrics.md).

## Results

### m5-sdcpp-flux-klein-base-005__attempt__20261011-025051

[Status](../results/runs/m5-sdcpp-flux-klein-base-005__attempt__20261011-025051/status.json), [execution config](../results/runs/m5-sdcpp-flux-klein-base-005__attempt__20261011-025051/config.json). Status: complete.

[Summary](../results/runs/m5-sdcpp-flux-klein-base-005__attempt__20261011-025051/summary.json), [validation](../results/runs/m5-sdcpp-flux-klein-base-005__attempt__20261011-025051/validation.json).

| Metric | Median (s) | Min–max (s) |
|---|---:|---:|
| End to end | 392.819 | 392.819–392.819 |
| Text encoding | 1.072 | 1.072–1.072 |
| Denoising | 389.046 | 389.046–389.046 |
| VAE decoding | 2.699 | 2.699–2.699 |

Measured generations: 1; first image: 389.152 s; cached load: 2.418 s.
Output hash repeatability: True. This is not a quality assessment.

Whole-system available minimum: 3.923 GiB; whole-system swap start/peak: 3.793/3.793 GiB. Harness/process peak RSS: 0.518 GiB. These must not be interpreted as GPU allocation peaks.

### m5-sdcpp-flux-klein-base-005__baseline__20261011-030358

[Status](../results/runs/m5-sdcpp-flux-klein-base-005__baseline__20261011-030358/status.json), [execution config](../results/runs/m5-sdcpp-flux-klein-base-005__baseline__20261011-030358/config.json). Status: complete.

[Summary](../results/runs/m5-sdcpp-flux-klein-base-005__baseline__20261011-030358/summary.json), [validation](../results/runs/m5-sdcpp-flux-klein-base-005__baseline__20261011-030358/validation.json).

| Metric | Median (s) | Min–max (s) |
|---|---:|---:|
| End to end | 389.533 | 387.493–395.481 |
| Text encoding | 1.099 | 1.065–1.183 |
| Denoising | 385.716 | 383.748–391.666 |
| VAE decoding | 2.689 | 2.620–2.971 |

Measured generations: 10; first image: 391.810 s; cached load: 2.433 s.
Output hash repeatability: True. This is not a quality assessment.

Whole-system available minimum: 3.598 GiB; whole-system swap start/peak: 3.793/3.793 GiB. Harness/process peak RSS: 0.534 GiB. These must not be interpreted as GPU allocation peaks.

## Interpretation

A completed full protocol establishes execution and repeated timing under the captured system state. User applications, system swap and background preparation were not isolated. CPU-side and GPU-side timing boundaries follow the linked metric definitions. No formal quality eligibility or controlled performance claim is established.

## Next experiment

Repeat this exact configuration under an isolated system state if a controlled performance comparison is required.

## Reference evidence

[Same-engine reference](../experiments/m5-sdcpp-flux-klein-base-001.md). Image differences are diagnostic only; see each run’s `reference-image-check.json`.

## Execution and image audit

The full 1+3+10 protocol passed functional, protocol and timing-continuity checks ([validation](../results/runs/m5-sdcpp-flux-klein-base-005__baseline__20261011-030358/validation.json)). The engine confirmed mmap for the transformer, text encoder and VAE weight files, with no placement or fallback problems ([summary](../results/runs/m5-sdcpp-flux-klein-base-005__baseline__20261011-030358/summary.json)). The retained measured image is pixel-identical to the same-engine reference ([diagnostic](../results/runs/m5-sdcpp-flux-klein-base-005__baseline__20261011-030358/reference-image-check.json)); this single prompt/seed check does not establish general quality or causal speedup.
