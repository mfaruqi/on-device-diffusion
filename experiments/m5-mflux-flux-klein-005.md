---
type: experiment-record
id: m5-mflux-flux-klein-005
status: complete
device: macbook-air-m5
engine: mflux
runs: [m5-mflux-flux-klein-005__attempt__20261015-134805, m5-mflux-flux-klein-005__baseline__20261015-134829]
updated: 2026-10-15
---

# m5-mflux-flux-klein-005

## Question

Week 2–3 Apple individual-option work supporting RQ1: can this labelled configuration complete the full repeated protocol on the M5?

## Setup

M5 MFLUX Klein distilled q4: vae-tiled256 only versus configs/m5-mflux-flux-klein-q4-512.resolved.json; native MLX compilation retained.

[Configuration](../configs/m5-mflux-flux-klein-q4-512-vae-tiled256.resolved.json). One first image, three discarded warm-ups and ten measured generations for the full protocol. Smoke gates use one first and one measured image. Parameters, model revisions, backend and precision are explicit in each execution config.

[M5 procedure](../wiki/methods/m5-howto.md); [metric definitions](../wiki/methods/baseline-metrics.md).

## Results

### m5-mflux-flux-klein-005__attempt__20261015-134805

[Status](../results/runs/m5-mflux-flux-klein-005__attempt__20261015-134805/status.json), [execution config](../results/runs/m5-mflux-flux-klein-005__attempt__20261015-134805/config.json). Status: complete.

[Summary](../results/runs/m5-mflux-flux-klein-005__attempt__20261015-134805/summary.json), [validation](../results/runs/m5-mflux-flux-klein-005__attempt__20261015-134805/validation.json).

| Metric | Median (s) | Min–max (s) |
|---|---:|---:|
| End to end | 8.622 | 8.622–8.622 |
| Text encoding | 0.534 | 0.534–0.534 |
| Denoising | 6.898 | 6.898–6.898 |
| VAE decoding | 1.187 | 1.187–1.187 |

Measured generations: 1; first image: 7.088 s; cached load: 3.208 s.
Output hash repeatability: True. This is not a quality assessment.

Whole-system available minimum: 2.514 GiB; whole-system swap start/peak: 14.880/16.448 GiB. Harness/process peak RSS: 4.716 GiB. These must not be interpreted as GPU allocation peaks.

### m5-mflux-flux-klein-005__baseline__20261015-134829

[Status](../results/runs/m5-mflux-flux-klein-005__baseline__20261015-134829/status.json), [execution config](../results/runs/m5-mflux-flux-klein-005__baseline__20261015-134829/config.json). Status: complete.

[Summary](../results/runs/m5-mflux-flux-klein-005__baseline__20261015-134829/summary.json), [validation](../results/runs/m5-mflux-flux-klein-005__baseline__20261015-134829/validation.json).

| Metric | Median (s) | Min–max (s) |
|---|---:|---:|
| End to end | 7.662 | 7.400–9.081 |
| Text encoding | 0.416 | 0.397–0.726 |
| Denoising | 6.279 | 6.065–6.961 |
| VAE decoding | 0.971 | 0.914–1.715 |

Measured generations: 10; first image: 8.120 s; cached load: 3.185 s.
Output hash repeatability: True. This is not a quality assessment.

Whole-system available minimum: 1.904 GiB; whole-system swap start/peak: 15.941/18.387 GiB. Harness/process peak RSS: 4.817 GiB. These must not be interpreted as GPU allocation peaks.

## Interpretation

A completed full protocol establishes execution and repeated timing under the captured system state. User applications, system swap and background preparation were not isolated. CPU-side and GPU-side timing boundaries follow the linked metric definitions. No formal quality eligibility or controlled performance claim is established.

## Next experiment

Repeat this exact configuration under an isolated system state if a controlled performance comparison is required.

## Reference and option audit

[Reference](m5-mflux-flux-klein-003.md). Explicit 256-pixel VAE tiling with 64-pixel overlap executed nine tile decodes in each of 14 generations. Tile-local normalization may change colors; image differences are retained. The inspected 4-bit sign retains a text-generation error. One development image is not formal quality evidence.

[Execution audit](../results/runs/m5-mflux-flux-klein-005__baseline__20261015-134829/execution-audit.json); [reference-image diagnostic](../results/runs/m5-mflux-flux-klein-005__baseline__20261015-134829/reference-image-check.json).
