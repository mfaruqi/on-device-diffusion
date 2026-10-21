---
type: experiment-record
id: m5-sdcpp-flux-klein-003
status: complete
device: macbook-air-m5
engine: stable-diffusion-cpp
runs: [m5-sdcpp-flux-klein-003__attempt__20261010-204709, m5-sdcpp-flux-klein-003__baseline__20261010-205038]
updated: 2026-10-10
---

# m5-sdcpp-flux-klein-003

## Question

Week 2 Apple baseline work supporting RQ1: can this labelled configuration complete the full repeated protocol on the M5?

## Setup

M5 flux-klein: conditioning only against m5-sdcpp-flux-klein-001; same mixed-Q4 weights, 512x512, residency and sampling. Exact repeated identical prompt conditioning; first generation includes cache population.

[Configuration](../configs/m5-sdcpp-flux-klein-q4-512-conditioning.resolved.json). One first image, three discarded warm-ups and ten measured generations for the full protocol. Smoke gates use one first and one measured image. Parameters, model revisions, backend and precision are explicit in each execution config.

[M5 procedure](../wiki/methods/m5-howto.md); [metric definitions](../wiki/methods/baseline-metrics.md).

## Results

### m5-sdcpp-flux-klein-003__attempt__20261010-204709

[Status](../results/runs/m5-sdcpp-flux-klein-003__attempt__20261010-204709/status.json), [execution config](../results/runs/m5-sdcpp-flux-klein-003__attempt__20261010-204709/config.json). Status: complete.

[Summary](../results/runs/m5-sdcpp-flux-klein-003__attempt__20261010-204709/summary.json), [validation](../results/runs/m5-sdcpp-flux-klein-003__attempt__20261010-204709/validation.json).

| Metric | Median (s) | Min–max (s) |
|---|---:|---:|
| End to end | 14.839 | 14.839–14.839 |
| Text encoding | 0.003 | 0.003–0.003 |
| Denoising | 12.664 | 12.664–12.664 |
| VAE decoding | 2.170 | 2.170–2.170 |

Measured generations: 1; first image: 15.158 s; cached load: 1.266 s.
Output hash repeatability: True. This is not a quality assessment.

Whole-system available minimum: 8.147 GiB; whole-system swap start/peak: 5.440/5.440 GiB. Harness/process peak RSS: 5.738 GiB. These must not be interpreted as GPU allocation peaks.

### m5-sdcpp-flux-klein-003__baseline__20261010-205038

[Status](../results/runs/m5-sdcpp-flux-klein-003__baseline__20261010-205038/status.json), [execution config](../results/runs/m5-sdcpp-flux-klein-003__baseline__20261010-205038/config.json). Status: complete.

[Summary](../results/runs/m5-sdcpp-flux-klein-003__baseline__20261010-205038/summary.json), [validation](../results/runs/m5-sdcpp-flux-klein-003__baseline__20261010-205038/validation.json).

| Metric | Median (s) | Min–max (s) |
|---|---:|---:|
| End to end | 15.299 | 14.800–15.468 |
| Text encoding | 0.002 | 0.002–0.003 |
| Denoising | 13.048 | 12.616–13.148 |
| VAE decoding | 2.230 | 2.180–2.316 |

Measured generations: 10; first image: 15.203 s; cached load: 1.257 s.
Output hash repeatability: True. This is not a quality assessment.

Whole-system available minimum: 8.073 GiB; whole-system swap start/peak: 5.330/5.330 GiB. Harness/process peak RSS: 5.756 GiB. These must not be interpreted as GPU allocation peaks.

## Interpretation

A completed full protocol establishes execution and repeated timing under the captured system state. User applications, system swap and background preparation were not isolated. CPU-side and GPU-side timing boundaries follow the linked metric definitions. No formal quality eligibility or controlled performance claim is established.

## Next experiment

Repeat this exact configuration under an isolated system state if a controlled performance comparison is required.

## Reference evidence

[Same-engine reference](../experiments/m5-sdcpp-flux-klein-001.md). Image differences are diagnostic only; see each run’s `reference-image-check.json`.

## Execution and image audit

[Image diagnostic](../results/runs/m5-sdcpp-flux-klein-003__baseline__20261010-205038/reference-image-check.json), [protocol validation](../results/runs/m5-sdcpp-flux-klein-003__baseline__20261010-205038/validation.json). These diagnostics cover one development prompt and seed only.
