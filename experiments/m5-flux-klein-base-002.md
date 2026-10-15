---
type: experiment-record
id: m5-flux-klein-base-002
status: complete
device: macbook-air-m5
engine: pytorch-diffusers
runs: [m5-flux-klein-base-002__attempt__20261010-210338, m5-flux-klein-base-002__baseline__20261010-210907]
updated: 2026-10-10
---

# m5-flux-klein-base-002

## Question

Week 2 Apple baseline work supporting RQ1: can this labelled configuration complete the full repeated protocol on the M5?

## Setup

M5 BF16 MPS flux-klein-base: exact repeated prompt encoding reuse only against m5-flux-klein-base-001; positive and CFG negative entries cached within one pipeline; first image populates the cache.

[Configuration](../configs/m5-flux-klein-base-bf16-512-conditioning.resolved.json). One first image, three discarded warm-ups and ten measured generations for the full protocol. Smoke gates use one first and one measured image. Parameters, model revisions, backend and precision are explicit in each execution config.

[M5 procedure](../wiki/methods/m5-howto.md); [metric definitions](../wiki/methods/baseline-metrics.md).

## Results

### m5-flux-klein-base-002__attempt__20261010-210338

[Status](../results/runs/m5-flux-klein-base-002__attempt__20261010-210338/status.json), [execution config](../results/runs/m5-flux-klein-base-002__attempt__20261010-210338/config.json). Status: complete.

[Summary](../results/runs/m5-flux-klein-base-002__attempt__20261010-210338/summary.json), [validation](../results/runs/m5-flux-klein-base-002__attempt__20261010-210338/validation.json).

| Metric | Median (s) | Min–max (s) |
|---|---:|---:|
| End to end | 131.725 | 131.725–131.725 |
| Text encoding | 0.000 | 0.000–0.000 |
| Denoising | 131.309 | 131.309–131.309 |
| VAE decoding | 0.394 | 0.394–0.394 |

Measured generations: 1; first image: 138.588 s; cached load: 17.849 s.
Output hash repeatability: True. This is not a quality assessment.

Whole-system available minimum: 0.020 GiB; whole-system swap start/peak: 4.814/10.891 GiB. Harness/process peak RSS: 0.497 GiB. These must not be interpreted as GPU allocation peaks.

### m5-flux-klein-base-002__baseline__20261010-210907

[Status](../results/runs/m5-flux-klein-base-002__baseline__20261010-210907/status.json), [execution config](../results/runs/m5-flux-klein-base-002__baseline__20261010-210907/config.json). Status: complete.

[Summary](../results/runs/m5-flux-klein-base-002__baseline__20261010-210907/summary.json), [validation](../results/runs/m5-flux-klein-base-002__baseline__20261010-210907/validation.json).

| Metric | Median (s) | Min–max (s) |
|---|---:|---:|
| End to end | 136.075 | 129.359–164.207 |
| Text encoding | 0.000 | 0.000–0.000 |
| Denoising | 135.654 | 128.959–163.705 |
| VAE decoding | 0.408 | 0.342–0.482 |

Measured generations: 10; first image: 122.132 s; cached load: 17.645 s.
Output hash repeatability: True. This is not a quality assessment.

Whole-system available minimum: 0.021 GiB; whole-system swap start/peak: 6.765/11.594 GiB. Harness/process peak RSS: 0.496 GiB. These must not be interpreted as GPU allocation peaks.

## Interpretation

A completed full protocol establishes execution and repeated timing under the captured system state. User applications, system swap and background preparation were not isolated. CPU-side and GPU-side timing boundaries follow the linked metric definitions. No formal quality eligibility or controlled performance claim is established.

## Next experiment

Repeat this exact configuration under an isolated system state if a controlled performance comparison is required.

## Reference evidence

[Same-engine reference](../experiments/m5-flux-klein-base-001.md). Image differences are diagnostic only; see each run’s `reference-image-check.json`.

## Execution and image audit

[Cache receipt](../results/runs/m5-flux-klein-base-002__baseline__20261010-210907/conditioning-cache.json) records two first-generation misses and 26 hits across subsequent generations for separate positive and negative conditioning. [Protocol validation](../results/runs/m5-flux-klein-base-002__baseline__20261010-210907/validation.json) confirms 1+3+10 generations and timing continuity. The retained measured image matches the reference pixels exactly ([diagnostic](../results/runs/m5-flux-klein-base-002__baseline__20261010-210907/reference-image-check.json)); this is not formal quality evidence.
