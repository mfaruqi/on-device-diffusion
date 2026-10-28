---
type: experiment-record
id: m5-mflux-flux-klein-base-004
status: complete
device: macbook-air-m5
engine: mflux
runs: [m5-mflux-flux-klein-base-004__attempt__20261015-150106, m5-mflux-flux-klein-base-004__baseline__20261015-150452]
updated: 2026-10-15
---

# m5-mflux-flux-klein-base-004

## Question

Week 2–3 Apple individual-option work supporting RQ1: can this labelled configuration complete the full repeated protocol on the M5?

## Setup

M5 MFLUX Klein-base q4: conditioning only versus configs/m5-mflux-flux-klein-base-q4-512.resolved.json; native MLX compilation retained.

[Configuration](../configs/m5-mflux-flux-klein-base-q4-512-conditioning.resolved.json). One first image, three discarded warm-ups and ten measured generations for the full protocol. Smoke gates use one first and one measured image. Parameters, model revisions, backend and precision are explicit in each execution config.

[M5 procedure](../wiki/methods/m5-howto.md); [metric definitions](../wiki/methods/baseline-metrics.md).

## Results

### m5-mflux-flux-klein-base-004__attempt__20261015-150106

[Status](../results/runs/m5-mflux-flux-klein-base-004__attempt__20261015-150106/status.json), [execution config](../results/runs/m5-mflux-flux-klein-base-004__attempt__20261015-150106/config.json). Status: complete.

[Summary](../results/runs/m5-mflux-flux-klein-base-004__attempt__20261015-150106/summary.json), [validation](../results/runs/m5-mflux-flux-klein-base-004__attempt__20261015-150106/validation.json).

| Metric | Median (s) | Min–max (s) |
|---|---:|---:|
| End to end | 126.387 | 126.387–126.387 |
| Text encoding | 0.001 | 0.001–0.001 |
| Denoising | 125.274 | 125.274–125.274 |
| VAE decoding | 1.100 | 1.100–1.100 |

Measured generations: 1; first image: 91.739 s; cached load: 3.129 s.
Output hash repeatability: True. This is not a quality assessment.

Whole-system available minimum: 0.940 GiB; whole-system swap start/peak: 11.223/15.812 GiB. Harness/process peak RSS: 4.730 GiB. These must not be interpreted as GPU allocation peaks.

### m5-mflux-flux-klein-base-004__baseline__20261015-150452

[Status](../results/runs/m5-mflux-flux-klein-base-004__baseline__20261015-150452/status.json), [execution config](../results/runs/m5-mflux-flux-klein-base-004__baseline__20261015-150452/config.json). Status: complete.

[Summary](../results/runs/m5-mflux-flux-klein-base-004__baseline__20261015-150452/summary.json), [validation](../results/runs/m5-mflux-flux-klein-base-004__baseline__20261015-150452/validation.json).

| Metric | Median (s) | Min–max (s) |
|---|---:|---:|
| End to end | 128.670 | 121.897–139.802 |
| Text encoding | 0.000 | 0.000–0.000 |
| Denoising | 127.590 | 120.848–138.647 |
| VAE decoding | 1.075 | 1.041–3.362 |

Measured generations: 10; first image: 125.323 s; cached load: 3.162 s.
Output hash repeatability: True. This is not a quality assessment.

Whole-system available minimum: 0.868 GiB; whole-system swap start/peak: 11.107/17.630 GiB. Harness/process peak RSS: 4.806 GiB. These must not be interpreted as GPU allocation peaks.

## Interpretation

A completed full protocol establishes execution and repeated timing under the captured system state. User applications, system swap and background preparation were not isolated. CPU-side and GPU-side timing boundaries follow the linked metric definitions. No formal quality eligibility or controlled performance claim is established.

## Next experiment

Repeat this exact configuration under an isolated system state if a controlled performance comparison is required.

## Reference and option audit

[Reference](m5-mflux-flux-klein-base-003.md). The prompt-pair cache recorded one population and 13 hits across 14 generations, preserving both conditional and empty negative conditioning with two CFG passes per step. The measured image matches the same-engine 4-bit reference pixels. Lookup time is positive but rounds to 0.000 s in the table. This verifies exact reuse for the repeated conditioning, not arbitrary new prompts or formal quality approval. Whole-system memory pressure remains uncontrolled.

[Execution audit](../results/runs/m5-mflux-flux-klein-base-004__baseline__20261015-150452/execution-audit.json); [image diagnostic](../results/runs/m5-mflux-flux-klein-base-004__baseline__20261015-150452/reference-image-check.json); [W&B verification](../results/runs/m5-mflux-flux-klein-base-004__baseline__20261015-150452/wandb-followup-verification.json).
