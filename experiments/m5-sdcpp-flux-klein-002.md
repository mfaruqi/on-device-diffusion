---
type: experiment-record
id: m5-sdcpp-flux-klein-002
status: complete
device: macbook-air-m5
engine: stable-diffusion-cpp
runs: [m5-sdcpp-flux-klein-002__attempt__20261010-204248, m5-sdcpp-flux-klein-002__baseline__20261010-204326]
updated: 2026-10-10
---

# m5-sdcpp-flux-klein-002

## Question

Week 2 Apple baseline work supporting RQ1: can this labelled configuration complete the full repeated protocol on the M5?

## Setup

M5 flux-klein: easycache only against m5-sdcpp-flux-klein-001; same mixed-Q4 weights, 512x512, residency and sampling. Approximate step reuse; no quality eligibility implied.

[Configuration](../configs/m5-sdcpp-flux-klein-q4-512-easycache.resolved.json). One first image, three discarded warm-ups and ten measured generations for the full protocol. Smoke gates use one first and one measured image. Parameters, model revisions, backend and precision are explicit in each execution config.

[M5 procedure](../wiki/methods/m5-howto.md); [metric definitions](../wiki/methods/baseline-metrics.md).

## Results

### m5-sdcpp-flux-klein-002__attempt__20261010-204248

[Status](../results/runs/m5-sdcpp-flux-klein-002__attempt__20261010-204248/status.json), [execution config](../results/runs/m5-sdcpp-flux-klein-002__attempt__20261010-204248/config.json). Status: complete.

[Summary](../results/runs/m5-sdcpp-flux-klein-002__attempt__20261010-204248/summary.json), [validation](../results/runs/m5-sdcpp-flux-klein-002__attempt__20261010-204248/validation.json).

| Metric | Median (s) | Min–max (s) |
|---|---:|---:|
| End to end | 15.127 | 15.127–15.127 |
| Text encoding | 0.436 | 0.436–0.436 |
| Denoising | 12.511 | 12.511–12.511 |
| VAE decoding | 2.179 | 2.179–2.179 |

Measured generations: 1; first image: 15.137 s; cached load: 7.020 s.
Output hash repeatability: True. This is not a quality assessment.

Whole-system available minimum: 9.746 GiB; whole-system swap start/peak: 6.026/6.026 GiB. Harness/process peak RSS: 5.735 GiB. These must not be interpreted as GPU allocation peaks.

### m5-sdcpp-flux-klein-002__baseline__20261010-204326

[Status](../results/runs/m5-sdcpp-flux-klein-002__baseline__20261010-204326/status.json), [execution config](../results/runs/m5-sdcpp-flux-klein-002__baseline__20261010-204326/config.json). Status: complete.

[Summary](../results/runs/m5-sdcpp-flux-klein-002__baseline__20261010-204326/summary.json), [validation](../results/runs/m5-sdcpp-flux-klein-002__baseline__20261010-204326/validation.json).

| Metric | Median (s) | Min–max (s) |
|---|---:|---:|
| End to end | 15.807 | 15.558–15.870 |
| Text encoding | 0.442 | 0.440–0.447 |
| Denoising | 13.098 | 12.881–13.157 |
| VAE decoding | 2.261 | 2.228–2.275 |

Measured generations: 10; first image: 15.357 s; cached load: 1.247 s.
Output hash repeatability: True. This is not a quality assessment.

Whole-system available minimum: 9.388 GiB; whole-system swap start/peak: 5.737/5.737 GiB. Harness/process peak RSS: 5.749 GiB. These must not be interpreted as GPU allocation peaks.

## Interpretation

A completed full protocol establishes execution and repeated timing under the captured system state. User applications, system swap and background preparation were not isolated. CPU-side and GPU-side timing boundaries follow the linked metric definitions. No formal quality eligibility or controlled performance claim is established.

## Next experiment

Repeat this exact configuration under an isolated system state if a controlled performance comparison is required.

## Reference evidence

[Same-engine reference](../experiments/m5-sdcpp-flux-klein-001.md). Image differences are diagnostic only; see each run’s `reference-image-check.json`.

## Execution and image audit

[Image diagnostic](../results/runs/m5-sdcpp-flux-klein-002__baseline__20261010-204326/reference-image-check.json), [protocol validation](../results/runs/m5-sdcpp-flux-klein-002__baseline__20261010-204326/validation.json). [Cache audit](../results/runs/m5-sdcpp-flux-klein-002__baseline__20261010-204326/summary.json) confirms the requested EasyCache parameters but zero skipped steps across all generations. These diagnostics cover one development prompt and seed only.
