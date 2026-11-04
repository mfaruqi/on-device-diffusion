---
type: experiment-record
id: m5-sdcpp-flux-klein-base-002
status: complete
device: macbook-air-m5
engine: stable-diffusion-cpp
runs: [m5-sdcpp-flux-klein-base-002__attempt__20261010-222736, m5-sdcpp-flux-klein-base-002__baseline__20261010-223355]
updated: 2026-10-10
---

# m5-sdcpp-flux-klein-base-002

## Question

Week 2 Apple baseline work supporting RQ1: can this labelled configuration complete the full repeated protocol on the M5?

## Setup

M5 flux-klein-base: easycache only against m5-sdcpp-flux-klein-base-001; same mixed-Q4 weights, 512x512, residency and sampling. Approximate step reuse; no quality eligibility implied.

[Configuration](../configs/m5-sdcpp-flux-klein-base-q4-512-easycache.resolved.json). One first image, three discarded warm-ups and ten measured generations for the full protocol. Smoke gates use one first and one measured image. Parameters, model revisions, backend and precision are explicit in each execution config.

[M5 procedure](../wiki/methods/m5-howto.md); [metric definitions](../wiki/methods/baseline-metrics.md).

## Results

### m5-sdcpp-flux-klein-base-002__attempt__20261010-222736

[Status](../results/runs/m5-sdcpp-flux-klein-base-002__attempt__20261010-222736/status.json), [execution config](../results/runs/m5-sdcpp-flux-klein-base-002__attempt__20261010-222736/config.json). Status: complete.

[Summary](../results/runs/m5-sdcpp-flux-klein-base-002__attempt__20261010-222736/summary.json), [validation](../results/runs/m5-sdcpp-flux-klein-base-002__attempt__20261010-222736/validation.json).

| Metric | Median (s) | Min–max (s) |
|---|---:|---:|
| End to end | 197.684 | 197.684–197.684 |
| Text encoding | 1.028 | 1.028–1.028 |
| Denoising | 194.067 | 194.067–194.067 |
| VAE decoding | 2.587 | 2.587–2.587 |

Measured generations: 1; first image: 170.998 s; cached load: 8.832 s.
Output hash repeatability: True. This is not a quality assessment.

Whole-system available minimum: 6.159 GiB; whole-system swap start/peak: 6.443/6.443 GiB. Harness/process peak RSS: 5.757 GiB. These must not be interpreted as GPU allocation peaks.

### m5-sdcpp-flux-klein-base-002__baseline__20261010-223355

[Status](../results/runs/m5-sdcpp-flux-klein-base-002__baseline__20261010-223355/status.json), [execution config](../results/runs/m5-sdcpp-flux-klein-base-002__baseline__20261010-223355/config.json). Status: complete.

[Summary](../results/runs/m5-sdcpp-flux-klein-base-002__baseline__20261010-223355/summary.json), [validation](../results/runs/m5-sdcpp-flux-klein-base-002__baseline__20261010-223355/validation.json).

| Metric | Median (s) | Min–max (s) |
|---|---:|---:|
| End to end | 200.305 | 194.841–220.093 |
| Text encoding | 1.156 | 1.084–1.350 |
| Denoising | 196.394 | 191.079–215.834 |
| VAE decoding | 2.735 | 2.648–3.200 |

Measured generations: 10; first image: 200.545 s; cached load: 7.093 s.
Output hash repeatability: True. This is not a quality assessment.

Whole-system available minimum: 4.951 GiB; whole-system swap start/peak: 4.620/4.620 GiB. Harness/process peak RSS: 5.751 GiB. These must not be interpreted as GPU allocation peaks.

## Interpretation

A completed full protocol establishes execution and repeated timing under the captured system state. User applications, system swap and background preparation were not isolated. CPU-side and GPU-side timing boundaries follow the linked metric definitions. No formal quality eligibility or controlled performance claim is established.

## Next experiment

Repeat this exact configuration under an isolated system state if a controlled performance comparison is required.

## Reference evidence

[Same-engine reference](../experiments/m5-sdcpp-flux-klein-base-001.md). Image differences are diagnostic only; see each run’s `reference-image-check.json`.

## Execution and image audit

The full protocol contains one first generation, three warm-ups and ten measured generations; functional, protocol and timing-continuity checks passed ([validation](../results/runs/m5-sdcpp-flux-klein-base-002__baseline__20261010-223355/validation.json)). EasyCache was enabled for all 14 generations, with 25 of 50 denoising steps skipped per generation; the engine audit reports no placement or fallback problems ([summary](../results/runs/m5-sdcpp-flux-klein-base-002__baseline__20261010-223355/summary.json)).

The retained measured image differs from the same-engine no-cache reference: RGB mean absolute differences are 3.369, 2.870 and 2.698 on the 0–255 pixel scale ([diagnostic](../results/runs/m5-sdcpp-flux-klein-base-002__baseline__20261010-223355/reference-image-check.json)). This single development prompt/seed comparison does not establish perceptual quality eligibility.
