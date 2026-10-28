---
type: experiment-record
id: m5-mflux-flux-klein-006
status: complete
device: macbook-air-m5
engine: mflux
runs: [m5-mflux-flux-klein-006__attempt__20261015-135027, m5-mflux-flux-klein-006__baseline__20261015-135054]
updated: 2026-10-15
---

# m5-mflux-flux-klein-006

## Question

Week 2–3 Apple individual-option work supporting RQ1: can this labelled configuration complete the full repeated protocol on the M5?

## Setup

M5 MFLUX Klein distilled q8: vae-tiled256 only versus configs/m5-mflux-flux-klein-q8-512.resolved.json; native MLX compilation retained.

[Configuration](../configs/m5-mflux-flux-klein-q8-512-vae-tiled256.resolved.json). One first image, three discarded warm-ups and ten measured generations for the full protocol. Smoke gates use one first and one measured image. Parameters, model revisions, backend and precision are explicit in each execution config.

[M5 procedure](../wiki/methods/m5-howto.md); [metric definitions](../wiki/methods/baseline-metrics.md).

## Results

### m5-mflux-flux-klein-006__attempt__20261015-135027

[Status](../results/runs/m5-mflux-flux-klein-006__attempt__20261015-135027/status.json), [execution config](../results/runs/m5-mflux-flux-klein-006__attempt__20261015-135027/config.json). Status: complete.

[Summary](../results/runs/m5-mflux-flux-klein-006__attempt__20261015-135027/summary.json), [validation](../results/runs/m5-mflux-flux-klein-006__attempt__20261015-135027/validation.json).

| Metric | Median (s) | Min–max (s) |
|---|---:|---:|
| End to end | 8.548 | 8.548–8.548 |
| Text encoding | 1.000 | 1.000–1.000 |
| Denoising | 6.545 | 6.545–6.545 |
| VAE decoding | 0.999 | 0.999–0.999 |

Measured generations: 1; first image: 10.495 s; cached load: 3.426 s.
Output hash repeatability: True. This is not a quality assessment.

Whole-system available minimum: 1.841 GiB; whole-system swap start/peak: 16.248/21.173 GiB. Harness/process peak RSS: 4.572 GiB. These must not be interpreted as GPU allocation peaks.

### m5-mflux-flux-klein-006__baseline__20261015-135054

[Status](../results/runs/m5-mflux-flux-klein-006__baseline__20261015-135054/status.json), [execution config](../results/runs/m5-mflux-flux-klein-006__baseline__20261015-135054/config.json). Status: complete.

[Summary](../results/runs/m5-mflux-flux-klein-006__baseline__20261015-135054/summary.json), [validation](../results/runs/m5-mflux-flux-klein-006__baseline__20261015-135054/validation.json).

| Metric | Median (s) | Min–max (s) |
|---|---:|---:|
| End to end | 8.607 | 8.121–10.166 |
| Text encoding | 0.520 | 0.431–1.529 |
| Denoising | 6.836 | 6.326–7.473 |
| VAE decoding | 1.152 | 0.960–2.122 |

Measured generations: 10; first image: 9.771 s; cached load: 3.335 s.
Output hash repeatability: True. This is not a quality assessment.

Whole-system available minimum: 1.963 GiB; whole-system swap start/peak: 15.253/19.642 GiB. Harness/process peak RSS: 4.659 GiB. These must not be interpreted as GPU allocation peaks.

## Interpretation

A completed full protocol establishes execution and repeated timing under the captured system state. User applications, system swap and background preparation were not isolated. CPU-side and GPU-side timing boundaries follow the linked metric definitions. No formal quality eligibility or controlled performance claim is established.

## Next experiment

Repeat this exact configuration under an isolated system state if a controlled performance comparison is required.

## Reference and option audit

[Reference](m5-mflux-flux-klein-002.md). Explicit 256-pixel VAE tiling with 64-pixel overlap executed nine tile decodes in each of 14 generations. Tile-local normalization may change colors; image differences are retained. The inspected 8-bit sign reads “hello world”. One development image is not formal quality evidence.

[Execution audit](../results/runs/m5-mflux-flux-klein-006__baseline__20261015-135054/execution-audit.json); [reference-image diagnostic](../results/runs/m5-mflux-flux-klein-006__baseline__20261015-135054/reference-image-check.json).
