---
type: experiment-record
id: m5-sdcpp-flux-klein-base-001
status: complete
device: macbook-air-m5
engine: stable-diffusion-cpp
runs: [m5-sdcpp-flux-klein-base-001__baseline__20261010-165813]
updated: 2026-10-10
---

# m5-sdcpp-flux-klein-base-001

## Question

Week 2 Apple baseline work supporting RQ1: can this labelled configuration complete the full repeated protocol on the M5?

## Setup

M5 Metal non-distilled klein Base mixed-Q4 full protocol: 512x512, 50 steps, CFG 4, no reuse. Separate workload from distilled klein and the BF16 reference.

[Configuration](../configs/m5-sdcpp-flux-klein-base-q4-512.json). One first image, three discarded warm-ups and ten measured generations for the full protocol. Smoke gates use one first and one measured image. Parameters, model revisions, backend and precision are explicit in each execution config.

[M5 procedure](../wiki/methods/m5-howto.md); [metric definitions](../wiki/methods/baseline-metrics.md).

## Results

### m5-sdcpp-flux-klein-base-001__baseline__20261010-165813

[Status](../results/runs/m5-sdcpp-flux-klein-base-001__baseline__20261010-165813/status.json), [execution config](../results/runs/m5-sdcpp-flux-klein-base-001__baseline__20261010-165813/config.json). Status: complete.

[Summary](../results/runs/m5-sdcpp-flux-klein-base-001__baseline__20261010-165813/summary.json), [validation](../results/runs/m5-sdcpp-flux-klein-base-001__baseline__20261010-165813/validation.json).

| Metric | Median (s) | Min–max (s) |
|---|---:|---:|
| End to end | 407.547 | 400.276–443.388 |
| Text encoding | 1.203 | 1.143–1.323 |
| Denoising | 403.543 | 396.363–439.176 |
| VAE decoding | 2.799 | 2.650–3.003 |

Measured generations: 10; first image: 383.262 s; cached load: 7.048 s.
Output hash repeatability: True. This is not a quality assessment.

Whole-system available minimum: 4.249 GiB; whole-system swap start/peak: 1.849/1.849 GiB. Harness/process peak RSS: 5.750 GiB. These must not be interpreted as GPU allocation peaks.

## Interpretation

A completed full protocol establishes execution and repeated timing under the captured system state. User applications, system swap and background preparation were not isolated. CPU-side and GPU-side timing boundaries follow the linked metric definitions. No formal quality eligibility or controlled performance claim is established.

## Next experiment

Repeat this exact configuration under an isolated system state if a controlled performance comparison is required.
