---
type: experiment-record
id: m5-sdcpp-flux-klein-005
status: complete
device: macbook-air-m5
engine: stable-diffusion-cpp
runs: [m5-sdcpp-flux-klein-005__attempt__20261010-205853, m5-sdcpp-flux-klein-005__baseline__20261010-205929]
updated: 2026-10-10
---

# m5-sdcpp-flux-klein-005

## Question

Week 2 Apple baseline work supporting RQ1: can this labelled configuration complete the full repeated protocol on the M5?

## Setup

M5 flux-klein: mmap only against m5-sdcpp-flux-klein-001; same mixed-Q4 weights, 512x512, residency and sampling. 

[Configuration](../configs/m5-sdcpp-flux-klein-q4-512-mmap.resolved.json). One first image, three discarded warm-ups and ten measured generations for the full protocol. Smoke gates use one first and one measured image. Parameters, model revisions, backend and precision are explicit in each execution config.

[M5 procedure](../wiki/methods/m5-howto.md); [metric definitions](../wiki/methods/baseline-metrics.md).

## Results

### m5-sdcpp-flux-klein-005__attempt__20261010-205853

[Status](../results/runs/m5-sdcpp-flux-klein-005__attempt__20261010-205853/status.json), [execution config](../results/runs/m5-sdcpp-flux-klein-005__attempt__20261010-205853/config.json). Status: complete.

[Summary](../results/runs/m5-sdcpp-flux-klein-005__attempt__20261010-205853/summary.json), [validation](../results/runs/m5-sdcpp-flux-klein-005__attempt__20261010-205853/validation.json).

| Metric | Median (s) | Min–max (s) |
|---|---:|---:|
| End to end | 17.446 | 17.446–17.446 |
| Text encoding | 0.455 | 0.455–0.455 |
| Denoising | 14.568 | 14.568–14.568 |
| VAE decoding | 2.421 | 2.421–2.421 |

Measured generations: 1; first image: 16.804 s; cached load: 1.081 s.
Output hash repeatability: True. This is not a quality assessment.

Whole-system available minimum: 7.565 GiB; whole-system swap start/peak: 4.970/4.970 GiB. Harness/process peak RSS: 0.583 GiB. These must not be interpreted as GPU allocation peaks.

### m5-sdcpp-flux-klein-005__baseline__20261010-205929

[Status](../results/runs/m5-sdcpp-flux-klein-005__baseline__20261010-205929/status.json), [execution config](../results/runs/m5-sdcpp-flux-klein-005__baseline__20261010-205929/config.json). Status: complete.

[Summary](../results/runs/m5-sdcpp-flux-klein-005__baseline__20261010-205929/summary.json), [validation](../results/runs/m5-sdcpp-flux-klein-005__baseline__20261010-205929/validation.json).

| Metric | Median (s) | Min–max (s) |
|---|---:|---:|
| End to end | 17.624 | 17.209–18.150 |
| Text encoding | 0.506 | 0.460–0.565 |
| Denoising | 14.592 | 14.289–15.093 |
| VAE decoding | 2.465 | 2.421–2.564 |

Measured generations: 10; first image: 16.876 s; cached load: 1.084 s.
Output hash repeatability: True. This is not a quality assessment.

Whole-system available minimum: 6.774 GiB; whole-system swap start/peak: 4.947/4.947 GiB. Harness/process peak RSS: 0.584 GiB. These must not be interpreted as GPU allocation peaks.

## Interpretation

A completed full protocol establishes execution and repeated timing under the captured system state. User applications, system swap and background preparation were not isolated. CPU-side and GPU-side timing boundaries follow the linked metric definitions. No formal quality eligibility or controlled performance claim is established.

## Next experiment

Repeat this exact configuration under an isolated system state if a controlled performance comparison is required.

## Reference evidence

[Same-engine reference](../experiments/m5-sdcpp-flux-klein-001.md). Image differences are diagnostic only; see each run’s `reference-image-check.json`.

## Execution and image audit

[Protocol validation](../results/runs/m5-sdcpp-flux-klein-005__baseline__20261010-205929/validation.json) confirms a complete 1+3+10 protocol and timing continuity. The [engine audit](../results/runs/m5-sdcpp-flux-klein-005__baseline__20261010-205929/summary.json) confirms mmap file selection. The retained measured image is pixel-identical to the same-engine reference ([diagnostic](../results/runs/m5-sdcpp-flux-klein-005__baseline__20261010-205929/reference-image-check.json)); this is one prompt/seed, not formal quality evidence.
