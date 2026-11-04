---
type: experiment-record
id: m5-sdcpp-flux-klein-001
status: complete
device: macbook-air-m5
engine: stable-diffusion-cpp
runs: [m5-sdcpp-flux-klein-001__attempt__20261010-140940, m5-sdcpp-flux-klein-001__baseline__20261010-152141, m5-sdcpp-flux-klein-001__baseline__20261010-165421]
updated: 2026-10-10
---

# m5-sdcpp-flux-klein-001

## Question

Week 2 Apple baseline work supporting RQ1: can this labelled configuration complete the full repeated protocol on the M5?

## Setup

M5 Metal distilled klein mixed-Q4 full protocol: 512x512, four steps, no reuse; same execution settings as the feasibility smoke. Separate quantized workload, not BF16 reference.

[Configuration](../configs/m5-sdcpp-flux-klein-q4-512.json). One first image, three discarded warm-ups and ten measured generations for the full protocol. Smoke gates use one first and one measured image. Parameters, model revisions, backend and precision are explicit in each execution config.

[M5 procedure](../wiki/methods/m5-howto.md); [metric definitions](../wiki/methods/baseline-metrics.md).

## Results

### m5-sdcpp-flux-klein-001__attempt__20261010-140940

[Status](../results/runs/m5-sdcpp-flux-klein-001__attempt__20261010-140940/status.json), [execution config](../results/runs/m5-sdcpp-flux-klein-001__attempt__20261010-140940/config.json). Status: complete.

[Summary](../results/runs/m5-sdcpp-flux-klein-001__attempt__20261010-140940/summary.json), [validation](../results/runs/m5-sdcpp-flux-klein-001__attempt__20261010-140940/validation.json).

| Metric | Median (s) | Min–max (s) |
|---|---:|---:|
| End to end | 14.710 | 14.710–14.710 |
| Text encoding | 0.488 | 0.488–0.488 |
| Denoising | 11.876 | 11.876–11.876 |
| VAE decoding | 2.343 | 2.343–2.343 |

Measured generations: 1; first image: 15.018 s; cached load: 8.050 s.
Output hash repeatability: True. This is not a quality assessment.

Whole-system available minimum: 2.826 GiB; whole-system swap start/peak: 23.253/24.967 GiB. Harness/process peak RSS: 5.652 GiB. These must not be interpreted as GPU allocation peaks.

### m5-sdcpp-flux-klein-001__baseline__20261010-152141

[Status](../results/runs/m5-sdcpp-flux-klein-001__baseline__20261010-152141/status.json), [execution config](../results/runs/m5-sdcpp-flux-klein-001__baseline__20261010-152141/config.json). Status: complete.

[Summary](../results/runs/m5-sdcpp-flux-klein-001__baseline__20261010-152141/summary.json), [validation](../results/runs/m5-sdcpp-flux-klein-001__baseline__20261010-152141/validation.json).

| Metric | Median (s) | Min–max (s) |
|---|---:|---:|
| End to end | 17.198 | 15.279–17.814 |
| Text encoding | 0.490 | 0.456–0.599 |
| Denoising | 14.038 | 12.368–14.641 |
| VAE decoding | 2.605 | 2.452–2.743 |

Measured generations: 10; first image: 15.066 s; cached load: 8.955 s.
Output hash repeatability: True. This is not a quality assessment.

Whole-system available minimum: 2.636 GiB; whole-system swap start/peak: 21.483/25.860 GiB. Harness/process peak RSS: 5.710 GiB. These must not be interpreted as GPU allocation peaks.

### m5-sdcpp-flux-klein-001__baseline__20261010-165421

[Status](../results/runs/m5-sdcpp-flux-klein-001__baseline__20261010-165421/status.json), [execution config](../results/runs/m5-sdcpp-flux-klein-001__baseline__20261010-165421/config.json). Status: complete.

[Summary](../results/runs/m5-sdcpp-flux-klein-001__baseline__20261010-165421/summary.json), [validation](../results/runs/m5-sdcpp-flux-klein-001__baseline__20261010-165421/validation.json).

| Metric | Median (s) | Min–max (s) |
|---|---:|---:|
| End to end | 16.211 | 15.217–17.712 |
| Text encoding | 0.456 | 0.442–0.526 |
| Denoising | 13.401 | 12.577–14.670 |
| VAE decoding | 2.342 | 2.196–2.536 |

Measured generations: 10; first image: 15.241 s; cached load: 6.932 s.
Output hash repeatability: True. This is not a quality assessment.

Whole-system available minimum: 5.916 GiB; whole-system swap start/peak: 1.849/1.849 GiB. Harness/process peak RSS: 5.737 GiB. These must not be interpreted as GPU allocation peaks.

## Interpretation

A completed full protocol establishes execution and repeated timing under the captured system state. User applications, system swap and background preparation were not isolated. CPU-side and GPU-side timing boundaries follow the linked metric definitions. No formal quality eligibility or controlled performance claim is established.

## Next experiment

Repeat this exact configuration under an isolated system state if a controlled performance comparison is required.
