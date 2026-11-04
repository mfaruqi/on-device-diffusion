---
type: experiment-record
id: m5-flux-klein-003
status: complete
device: macbook-air-m5
engine: pytorch-diffusers
runs: [m5-flux-klein-003__attempt__20261010-204022, m5-flux-klein-003__baseline__20261010-204109]
updated: 2026-10-10
---

# m5-flux-klein-003

## Question

Week 2 Apple baseline work supporting RQ1: can this labelled configuration complete the full repeated protocol on the M5?

## Setup

M5 BF16 MPS flux-klein: VAE tiling only against m5-flux-klein-001; 256-pixel tiles and 25 percent overlap. Same checkpoint and sampling; tiled decode may change image pixels.

[Configuration](../configs/m5-flux-klein-bf16-512-vae-tiled256.resolved.json). One first image, three discarded warm-ups and ten measured generations for the full protocol. Smoke gates use one first and one measured image. Parameters, model revisions, backend and precision are explicit in each execution config.

[M5 procedure](../wiki/methods/m5-howto.md); [metric definitions](../wiki/methods/baseline-metrics.md).

## Results

### m5-flux-klein-003__attempt__20261010-204022

[Status](../results/runs/m5-flux-klein-003__attempt__20261010-204022/status.json), [execution config](../results/runs/m5-flux-klein-003__attempt__20261010-204022/config.json). Status: complete.

[Summary](../results/runs/m5-flux-klein-003__attempt__20261010-204022/summary.json), [validation](../results/runs/m5-flux-klein-003__attempt__20261010-204022/validation.json).

| Metric | Median (s) | Min–max (s) |
|---|---:|---:|
| End to end | 4.871 | 4.871–4.871 |
| Text encoding | 0.451 | 0.451–0.451 |
| Denoising | 3.976 | 3.976–3.976 |
| VAE decoding | 0.439 | 0.439–0.439 |

Measured generations: 1; first image: 18.439 s; cached load: 19.424 s.
Output hash repeatability: True. This is not a quality assessment.

Whole-system available minimum: 0.023 GiB; whole-system swap start/peak: 6.843/11.968 GiB. Harness/process peak RSS: 0.497 GiB. These must not be interpreted as GPU allocation peaks.

### m5-flux-klein-003__baseline__20261010-204109

[Status](../results/runs/m5-flux-klein-003__baseline__20261010-204109/status.json), [execution config](../results/runs/m5-flux-klein-003__baseline__20261010-204109/config.json). Status: complete.

[Summary](../results/runs/m5-flux-klein-003__baseline__20261010-204109/summary.json), [validation](../results/runs/m5-flux-klein-003__baseline__20261010-204109/validation.json).

| Metric | Median (s) | Min–max (s) |
|---|---:|---:|
| End to end | 5.030 | 4.929–5.224 |
| Text encoding | 0.461 | 0.452–0.478 |
| Denoising | 4.125 | 4.029–4.297 |
| VAE decoding | 0.441 | 0.439–0.444 |

Measured generations: 10; first image: 13.034 s; cached load: 17.215 s.
Output hash repeatability: True. This is not a quality assessment.

Whole-system available minimum: 0.030 GiB; whole-system swap start/peak: 6.004/10.886 GiB. Harness/process peak RSS: 0.496 GiB. These must not be interpreted as GPU allocation peaks.

## Interpretation

A completed full protocol establishes execution and repeated timing under the captured system state. User applications, system swap and background preparation were not isolated. CPU-side and GPU-side timing boundaries follow the linked metric definitions. No formal quality eligibility or controlled performance claim is established.

## Next experiment

Repeat this exact configuration under an isolated system state if a controlled performance comparison is required.

## Reference evidence

[Same-engine reference](../experiments/m5-flux-klein-001.md). Image differences are diagnostic only; see each run’s `reference-image-check.json`.

## Execution and image audit

[Image diagnostic](../results/runs/m5-flux-klein-003__baseline__20261010-204109/reference-image-check.json), [protocol validation](../results/runs/m5-flux-klein-003__baseline__20261010-204109/validation.json). [Tiling receipt](../results/runs/m5-flux-klein-003__baseline__20261010-204109/vae-tiling.json) confirms tiled decoding on every generation. Tiling changes the retained output pixels; no quality eligibility is established. These diagnostics cover one development prompt and seed only.
