---
type: experiment-record
id: m5-sdcpp-flux-klein-base-003
status: complete
device: macbook-air-m5
engine: stable-diffusion-cpp
runs: [m5-sdcpp-flux-klein-base-003__attempt__20261010-232130, m5-sdcpp-flux-klein-base-003__baseline__20261010-233405]
updated: 2026-10-10
---

# m5-sdcpp-flux-klein-base-003

## Question

Week 2 Apple baseline work supporting RQ1: can this labelled configuration complete the full repeated protocol on the M5?

## Setup

M5 flux-klein-base: conditioning only against m5-sdcpp-flux-klein-base-001; same mixed-Q4 weights, 512x512, residency and sampling. Exact repeated identical prompt conditioning; first generation includes cache population.

[Configuration](../configs/m5-sdcpp-flux-klein-base-q4-512-conditioning.resolved.json). One first image, three discarded warm-ups and ten measured generations for the full protocol. Smoke gates use one first and one measured image. Parameters, model revisions, backend and precision are explicit in each execution config.

[M5 procedure](../wiki/methods/m5-howto.md); [metric definitions](../wiki/methods/baseline-metrics.md).

## Results

### m5-sdcpp-flux-klein-base-003__attempt__20261010-232130

[Status](../results/runs/m5-sdcpp-flux-klein-base-003__attempt__20261010-232130/status.json), [execution config](../results/runs/m5-sdcpp-flux-klein-base-003__attempt__20261010-232130/config.json). Status: complete.

[Summary](../results/runs/m5-sdcpp-flux-klein-base-003__attempt__20261010-232130/summary.json), [validation](../results/runs/m5-sdcpp-flux-klein-base-003__attempt__20261010-232130/validation.json).

| Metric | Median (s) | Min–max (s) |
|---|---:|---:|
| End to end | 365.645 | 365.645–365.645 |
| Text encoding | 0.006 | 0.006–0.006 |
| Denoising | 363.054 | 363.054–363.054 |
| VAE decoding | 2.583 | 2.583–2.583 |

Measured generations: 1; first image: 380.314 s; cached load: 7.588 s.
Output hash repeatability: True. This is not a quality assessment.

Whole-system available minimum: 5.262 GiB; whole-system swap start/peak: 4.020/4.020 GiB. Harness/process peak RSS: 5.766 GiB. These must not be interpreted as GPU allocation peaks.

### m5-sdcpp-flux-klein-base-003__baseline__20261010-233405

[Status](../results/runs/m5-sdcpp-flux-klein-base-003__baseline__20261010-233405/status.json), [execution config](../results/runs/m5-sdcpp-flux-klein-base-003__baseline__20261010-233405/config.json). Status: complete.

[Summary](../results/runs/m5-sdcpp-flux-klein-base-003__baseline__20261010-233405/summary.json), [validation](../results/runs/m5-sdcpp-flux-klein-base-003__baseline__20261010-233405/validation.json).

| Metric | Median (s) | Min–max (s) |
|---|---:|---:|
| End to end | 400.141 | 389.498–440.085 |
| Text encoding | 0.007 | 0.003–0.016 |
| Denoising | 397.391 | 386.743–437.145 |
| VAE decoding | 2.760 | 2.587–3.046 |

Measured generations: 10; first image: 360.915 s; cached load: 7.086 s.
Output hash repeatability: True. This is not a quality assessment.

Whole-system available minimum: 3.067 GiB; whole-system swap start/peak: 3.973/3.973 GiB. Harness/process peak RSS: 5.765 GiB. These must not be interpreted as GPU allocation peaks.

## Interpretation

A completed full protocol establishes execution and repeated timing under the captured system state. User applications, system swap and background preparation were not isolated. CPU-side and GPU-side timing boundaries follow the linked metric definitions. No formal quality eligibility or controlled performance claim is established.

## Next experiment

Repeat this exact configuration under an isolated system state if a controlled performance comparison is required.

## Reference evidence

[Same-engine reference](../experiments/m5-sdcpp-flux-klein-base-001.md). Image differences are diagnostic only; see each run’s `reference-image-check.json`.

## Execution and image audit

The full protocol contains one first image, three warm-ups and ten measured generations; functional, protocol and timing-continuity checks passed ([validation](../results/runs/m5-sdcpp-flux-klein-base-003__baseline__20261010-233405/validation.json)). The first generation has zero conditioning-cache hits; every subsequent generation has two hits, for 26 hits in total. Measured text-conditioning latency is 7.380 ms median; denoising remains 397.391 s median ([summary](../results/runs/m5-sdcpp-flux-klein-base-003__baseline__20261010-233405/summary.json)).

The retained measured image is pixel-identical to the same-engine reference ([diagnostic](../results/runs/m5-sdcpp-flux-klein-base-003__baseline__20261010-233405/reference-image-check.json)). This validates the fixed development prompt/seed check, not general quality or a controlled speedup.
