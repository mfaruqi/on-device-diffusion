---
type: experiment-record
id: m5-sdcpp-flux-klein-004
status: complete
device: macbook-air-m5
engine: stable-diffusion-cpp
runs: [m5-sdcpp-flux-klein-004__attempt__20261010-205422, m5-sdcpp-flux-klein-004__baseline__20261010-205458]
updated: 2026-10-10
---

# m5-sdcpp-flux-klein-004

## Question

Week 2 Apple baseline work supporting RQ1: can this labelled configuration complete the full repeated protocol on the M5?

## Setup

M5 flux-klein: prefetch-off only against m5-sdcpp-flux-klein-001; same mixed-Q4 weights, 512x512, residency and sampling. Eager GPU residency is unchanged; this option may be inactive for this placement.

[Configuration](../configs/m5-sdcpp-flux-klein-q4-512-prefetch-off.resolved.json). One first image, three discarded warm-ups and ten measured generations for the full protocol. Smoke gates use one first and one measured image. Parameters, model revisions, backend and precision are explicit in each execution config.

[M5 procedure](../wiki/methods/m5-howto.md); [metric definitions](../wiki/methods/baseline-metrics.md).

## Results

### m5-sdcpp-flux-klein-004__attempt__20261010-205422

[Status](../results/runs/m5-sdcpp-flux-klein-004__attempt__20261010-205422/status.json), [execution config](../results/runs/m5-sdcpp-flux-klein-004__attempt__20261010-205422/config.json). Status: complete.

[Summary](../results/runs/m5-sdcpp-flux-klein-004__attempt__20261010-205422/summary.json), [validation](../results/runs/m5-sdcpp-flux-klein-004__attempt__20261010-205422/validation.json).

| Metric | Median (s) | Min–max (s) |
|---|---:|---:|
| End to end | 16.730 | 16.730–16.730 |
| Text encoding | 0.471 | 0.471–0.471 |
| Denoising | 13.910 | 13.910–13.910 |
| VAE decoding | 2.347 | 2.347–2.347 |

Measured generations: 1; first image: 17.509 s; cached load: 1.248 s.
Output hash repeatability: True. This is not a quality assessment.

Whole-system available minimum: 7.682 GiB; whole-system swap start/peak: 5.174/5.174 GiB. Harness/process peak RSS: 5.749 GiB. These must not be interpreted as GPU allocation peaks.

### m5-sdcpp-flux-klein-004__baseline__20261010-205458

[Status](../results/runs/m5-sdcpp-flux-klein-004__baseline__20261010-205458/status.json), [execution config](../results/runs/m5-sdcpp-flux-klein-004__baseline__20261010-205458/config.json). Status: complete.

[Summary](../results/runs/m5-sdcpp-flux-klein-004__baseline__20261010-205458/summary.json), [validation](../results/runs/m5-sdcpp-flux-klein-004__baseline__20261010-205458/validation.json).

| Metric | Median (s) | Min–max (s) |
|---|---:|---:|
| End to end | 16.815 | 16.465–17.010 |
| Text encoding | 0.475 | 0.441–0.513 |
| Denoising | 13.945 | 13.642–14.125 |
| VAE decoding | 2.377 | 2.339–2.445 |

Measured generations: 10; first image: 16.170 s; cached load: 1.248 s.
Output hash repeatability: True. This is not a quality assessment.

Whole-system available minimum: 7.389 GiB; whole-system swap start/peak: 5.158/5.158 GiB. Harness/process peak RSS: 5.712 GiB. These must not be interpreted as GPU allocation peaks.

## Interpretation

A completed full protocol establishes execution and repeated timing under the captured system state. User applications, system swap and background preparation were not isolated. CPU-side and GPU-side timing boundaries follow the linked metric definitions. No formal quality eligibility or controlled performance claim is established.

## Next experiment

Repeat this exact configuration under an isolated system state if a controlled performance comparison is required.

## Reference evidence

[Same-engine reference](../experiments/m5-sdcpp-flux-klein-001.md). Image differences are diagnostic only; see each run’s `reference-image-check.json`.

## Execution and image audit

[Image diagnostic](../results/runs/m5-sdcpp-flux-klein-004__baseline__20261010-205458/reference-image-check.json), [protocol validation](../results/runs/m5-sdcpp-flux-klein-004__baseline__20261010-205458/validation.json). These diagnostics cover one development prompt and seed only.
