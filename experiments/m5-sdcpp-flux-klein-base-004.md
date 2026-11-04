---
type: experiment-record
id: m5-sdcpp-flux-klein-base-004
status: complete
device: macbook-air-m5
engine: stable-diffusion-cpp
runs: [m5-sdcpp-flux-klein-base-004__attempt__20261011-010657, m5-sdcpp-flux-klein-base-004__baseline__20261011-012000]
updated: 2026-10-10
---

# m5-sdcpp-flux-klein-base-004

## Question

Week 2 Apple baseline work supporting RQ1: can this labelled configuration complete the full repeated protocol on the M5?

## Setup

M5 flux-klein-base: prefetch-off only against m5-sdcpp-flux-klein-base-001; same mixed-Q4 weights, 512x512, residency and sampling. Eager GPU residency is unchanged; this option may be inactive for this placement.

[Configuration](../configs/m5-sdcpp-flux-klein-base-q4-512-prefetch-off.resolved.json). One first image, three discarded warm-ups and ten measured generations for the full protocol. Smoke gates use one first and one measured image. Parameters, model revisions, backend and precision are explicit in each execution config.

[M5 procedure](../wiki/methods/m5-howto.md); [metric definitions](../wiki/methods/baseline-metrics.md).

## Results

### m5-sdcpp-flux-klein-base-004__attempt__20261011-010657

[Status](../results/runs/m5-sdcpp-flux-klein-base-004__attempt__20261011-010657/status.json), [execution config](../results/runs/m5-sdcpp-flux-klein-base-004__attempt__20261011-010657/config.json). Status: complete.

[Summary](../results/runs/m5-sdcpp-flux-klein-base-004__attempt__20261011-010657/summary.json), [validation](../results/runs/m5-sdcpp-flux-klein-base-004__attempt__20261011-010657/validation.json).

| Metric | Median (s) | Min–max (s) |
|---|---:|---:|
| End to end | 386.290 | 386.290–386.290 |
| Text encoding | 1.070 | 1.070–1.070 |
| Denoising | 382.551 | 382.551–382.551 |
| VAE decoding | 2.667 | 2.667–2.667 |

Measured generations: 1; first image: 387.320 s; cached load: 7.189 s.
Output hash repeatability: True. This is not a quality assessment.

Whole-system available minimum: 3.993 GiB; whole-system swap start/peak: 3.817/3.817 GiB. Harness/process peak RSS: 5.745 GiB. These must not be interpreted as GPU allocation peaks.

### m5-sdcpp-flux-klein-base-004__baseline__20261011-012000

[Status](../results/runs/m5-sdcpp-flux-klein-base-004__baseline__20261011-012000/status.json), [execution config](../results/runs/m5-sdcpp-flux-klein-base-004__baseline__20261011-012000/config.json). Status: complete.

[Summary](../results/runs/m5-sdcpp-flux-klein-base-004__baseline__20261011-012000/summary.json), [validation](../results/runs/m5-sdcpp-flux-klein-base-004__baseline__20261011-012000/validation.json).

| Metric | Median (s) | Min–max (s) |
|---|---:|---:|
| End to end | 386.820 | 385.211–398.317 |
| Text encoding | 1.112 | 1.049–1.192 |
| Denoising | 383.085 | 381.393–394.404 |
| VAE decoding | 2.707 | 2.615–2.739 |

Measured generations: 10; first image: 387.683 s; cached load: 7.072 s.
Output hash repeatability: True. This is not a quality assessment.

Whole-system available minimum: 4.005 GiB; whole-system swap start/peak: 3.801/3.801 GiB. Harness/process peak RSS: 5.742 GiB. These must not be interpreted as GPU allocation peaks.

## Interpretation

A completed full protocol establishes execution and repeated timing under the captured system state. User applications, system swap and background preparation were not isolated. CPU-side and GPU-side timing boundaries follow the linked metric definitions. No formal quality eligibility or controlled performance claim is established.

## Next experiment

Repeat this exact configuration under an isolated system state if a controlled performance comparison is required.

## Reference evidence

[Same-engine reference](../experiments/m5-sdcpp-flux-klein-base-001.md). Image differences are diagnostic only; see each run’s `reference-image-check.json`.

## Execution and image audit

The full protocol contains one first image, three warm-ups and ten measured generations; functional, protocol and timing-continuity checks passed ([validation](../results/runs/m5-sdcpp-flux-klein-base-004__baseline__20261011-012000/validation.json)). The engine ran with prefetch disabled and eager Metal parameter residency; its audit reports no unexpected placement or fallback ([command](../results/runs/m5-sdcpp-flux-klein-base-004__baseline__20261011-012000/engine/command.json), [summary](../results/runs/m5-sdcpp-flux-klein-base-004__baseline__20261011-012000/summary.json)).

The retained measured image is pixel-identical to the same-engine reference ([diagnostic](../results/runs/m5-sdcpp-flux-klein-base-004__baseline__20261011-012000/reference-image-check.json)). Setting verification does not establish that prefetch was on the critical path under eager residency; this run alone does not establish a causal performance benefit.
