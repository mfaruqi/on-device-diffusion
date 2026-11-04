---
type: experiment-record
id: m5-flux-klein-001
status: complete
device: macbook-air-m5
engine: pytorch-diffusers
runs: [m5-flux-klein-001__attempt__20261010-183352, m5-flux-klein-001__baseline__20261010-183446]
updated: 2026-10-10
---

# m5-flux-klein-001

## Question

Week 2 Apple baseline work supporting RQ1: can this labelled configuration complete the full repeated protocol on the M5?

## Setup

M5 PyTorch/Diffusers BF16 reference, distilled four steps CFG 1, 512x512, no cache or offload.

[Configuration](../configs/m5-flux-klein-bf16-512.resolved.json). One first image, three discarded warm-ups and ten measured generations for the full protocol. Smoke gates use one first and one measured image. Parameters, model revisions, backend and precision are explicit in each execution config.

[M5 procedure](../wiki/methods/m5-howto.md); [metric definitions](../wiki/methods/baseline-metrics.md).

## Results

### m5-flux-klein-001__attempt__20261010-183352

[Status](../results/runs/m5-flux-klein-001__attempt__20261010-183352/status.json), [execution config](../results/runs/m5-flux-klein-001__attempt__20261010-183352/config.json). Status: complete.

[Summary](../results/runs/m5-flux-klein-001__attempt__20261010-183352/summary.json), [validation](../results/runs/m5-flux-klein-001__attempt__20261010-183352/validation.json).

| Metric | Median (s) | Min–max (s) |
|---|---:|---:|
| End to end | 7.598 | 7.598–7.598 |
| Text encoding | 0.600 | 0.600–0.600 |
| Denoising | 6.516 | 6.516–6.516 |
| VAE decoding | 0.472 | 0.472–0.472 |

Measured generations: 1; first image: 21.906 s; cached load: 20.664 s.
Output hash repeatability: True. This is not a quality assessment.

Whole-system available minimum: 0.024 GiB; whole-system swap start/peak: 1.132/9.864 GiB. Harness/process peak RSS: 0.498 GiB. These must not be interpreted as GPU allocation peaks.

### m5-flux-klein-001__baseline__20261010-183446

[Status](../results/runs/m5-flux-klein-001__baseline__20261010-183446/status.json), [execution config](../results/runs/m5-flux-klein-001__baseline__20261010-183446/config.json). Status: complete.

[Summary](../results/runs/m5-flux-klein-001__baseline__20261010-183446/summary.json), [validation](../results/runs/m5-flux-klein-001__baseline__20261010-183446/validation.json).

| Metric | Median (s) | Min–max (s) |
|---|---:|---:|
| End to end | 8.278 | 7.636–8.856 |
| Text encoding | 0.944 | 0.653–1.476 |
| Denoising | 6.862 | 6.538–7.119 |
| VAE decoding | 0.481 | 0.419–0.539 |

Measured generations: 10; first image: 13.584 s; cached load: 15.466 s.
Output hash repeatability: True. This is not a quality assessment.

Whole-system available minimum: 0.018 GiB; whole-system swap start/peak: 4.807/8.981 GiB. Harness/process peak RSS: 0.496 GiB. These must not be interpreted as GPU allocation peaks.

## Interpretation

A completed full protocol establishes execution and repeated timing under the captured system state. User applications, system swap and background preparation were not isolated. CPU-side and GPU-side timing boundaries follow the linked metric definitions. No formal quality eligibility or controlled performance claim is established.

## Next experiment

Repeat this exact configuration under an isolated system state if a controlled performance comparison is required.
