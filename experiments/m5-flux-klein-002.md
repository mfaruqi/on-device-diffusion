---
type: experiment-record
id: m5-flux-klein-002
status: complete
device: macbook-air-m5
engine: pytorch-diffusers
runs: [m5-flux-klein-002__attempt__20261010-203159, m5-flux-klein-002__baseline__20261010-203805]
updated: 2026-10-10
---

# m5-flux-klein-002

## Question

Week 2 Apple baseline work supporting RQ1: can this labelled configuration complete the full repeated protocol on the M5?

## Setup

M5 BF16 MPS flux-klein: exact repeated prompt encoding reuse only against m5-flux-klein-001; positive and CFG negative entries cached within one pipeline; first image populates the cache.

[Configuration](../configs/m5-flux-klein-bf16-512-conditioning.resolved.json). One first image, three discarded warm-ups and ten measured generations for the full protocol. Smoke gates use one first and one measured image. Parameters, model revisions, backend and precision are explicit in each execution config.

[M5 procedure](../wiki/methods/m5-howto.md); [metric definitions](../wiki/methods/baseline-metrics.md).

## Results

### m5-flux-klein-002__attempt__20261010-203159

[Status](../results/runs/m5-flux-klein-002__attempt__20261010-203159/status.json), [execution config](../results/runs/m5-flux-klein-002__attempt__20261010-203159/config.json). Status: complete.

[Summary](../results/runs/m5-flux-klein-002__attempt__20261010-203159/summary.json), [validation](../results/runs/m5-flux-klein-002__attempt__20261010-203159/validation.json).

| Metric | Median (s) | Min–max (s) |
|---|---:|---:|
| End to end | 4.360 | 4.360–4.360 |
| Text encoding | 0.000 | 0.000–0.000 |
| Denoising | 4.088 | 4.088–4.088 |
| VAE decoding | 0.264 | 0.264–0.264 |

Measured generations: 1; first image: 18.844 s; cached load: 26.791 s.
Output hash repeatability: True. This is not a quality assessment.

Whole-system available minimum: 0.031 GiB; whole-system swap start/peak: 2.566/11.174 GiB. Harness/process peak RSS: 0.496 GiB. These must not be interpreted as GPU allocation peaks.

### m5-flux-klein-002__baseline__20261010-203805

[Status](../results/runs/m5-flux-klein-002__baseline__20261010-203805/status.json), [execution config](../results/runs/m5-flux-klein-002__baseline__20261010-203805/config.json). Status: complete.

[Summary](../results/runs/m5-flux-klein-002__baseline__20261010-203805/summary.json), [validation](../results/runs/m5-flux-klein-002__baseline__20261010-203805/validation.json).

| Metric | Median (s) | Min–max (s) |
|---|---:|---:|
| End to end | 4.289 | 4.257–4.409 |
| Text encoding | 0.000 | 0.000–0.000 |
| Denoising | 4.015 | 3.986–4.131 |
| VAE decoding | 0.266 | 0.265–0.272 |

Measured generations: 10; first image: 16.167 s; cached load: 17.849 s.
Output hash repeatability: True. This is not a quality assessment.

Whole-system available minimum: 0.038 GiB; whole-system swap start/peak: 5.865/11.373 GiB. Harness/process peak RSS: 0.496 GiB. These must not be interpreted as GPU allocation peaks.

## Interpretation

A completed full protocol establishes execution and repeated timing under the captured system state. User applications, system swap and background preparation were not isolated. CPU-side and GPU-side timing boundaries follow the linked metric definitions. No formal quality eligibility or controlled performance claim is established.

## Next experiment

Repeat this exact configuration under an isolated system state if a controlled performance comparison is required.

## Reference evidence

[Same-engine reference](../experiments/m5-flux-klein-001.md). Image differences are diagnostic only; see each run’s `reference-image-check.json`.

## Execution and image audit

[Image diagnostic](../results/runs/m5-flux-klein-002__baseline__20261010-203805/reference-image-check.json), [protocol validation](../results/runs/m5-flux-klein-002__baseline__20261010-203805/validation.json). [Cache receipt](../results/runs/m5-flux-klein-002__baseline__20261010-203805/conditioning-cache.json) confirms first-generation population and subsequent exact encoding reuse. These diagnostics cover one development prompt and seed only.
