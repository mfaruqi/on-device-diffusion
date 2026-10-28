---
type: experiment-record
id: m5-flux-klein-base-004
status: complete
device: macbook-air-m5
engine: pytorch-diffusers
runs: [m5-flux-klein-base-004__attempt__20261015-135305, m5-flux-klein-base-004__baseline__20261015-135837]
updated: 2026-10-15
---

# m5-flux-klein-base-004

## Question

Week 2–3 Apple individual-option work supporting RQ1: can this labelled configuration complete the full repeated protocol on the M5?

## Setup

M5 BF16 Klein-base: compile only versus configs/m5-flux-klein-base-bf16-512.resolved.json; no conditioning reuse, offload or tiling.

[Configuration](../configs/m5-flux-klein-base-bf16-512-compile.resolved.json). One first image, three discarded warm-ups and ten measured generations for the full protocol. Smoke gates use one first and one measured image. Parameters, model revisions, backend and precision are explicit in each execution config.

[M5 procedure](../wiki/methods/m5-howto.md); [metric definitions](../wiki/methods/baseline-metrics.md).

## Results

### m5-flux-klein-base-004__attempt__20261015-135305

[Status](../results/runs/m5-flux-klein-base-004__attempt__20261015-135305/status.json), [execution config](../results/runs/m5-flux-klein-base-004__attempt__20261015-135305/config.json). Status: complete.

[Summary](../results/runs/m5-flux-klein-base-004__attempt__20261015-135305/summary.json), [validation](../results/runs/m5-flux-klein-base-004__attempt__20261015-135305/validation.json).

| Metric | Median (s) | Min–max (s) |
|---|---:|---:|
| End to end | 154.718 | 154.718–154.718 |
| Text encoding | 22.154 | 22.154–22.154 |
| Denoising | 131.384 | 131.384–131.384 |
| VAE decoding | 1.094 | 1.094–1.094 |

Measured generations: 1; first image: 145.998 s; cached load: 23.856 s.
Output hash repeatability: True. This is not a quality assessment.

Whole-system available minimum: 0.025 GiB; whole-system swap start/peak: 19.266/26.223 GiB. Harness/process peak RSS: 0.556 GiB. These must not be interpreted as GPU allocation peaks.

### m5-flux-klein-base-004__baseline__20261015-135837

[Status](../results/runs/m5-flux-klein-base-004__baseline__20261015-135837/status.json), [execution config](../results/runs/m5-flux-klein-base-004__baseline__20261015-135837/config.json). Status: complete.

[Summary](../results/runs/m5-flux-klein-base-004__baseline__20261015-135837/summary.json), [validation](../results/runs/m5-flux-klein-base-004__baseline__20261015-135837/validation.json).

| Metric | Median (s) | Min–max (s) |
|---|---:|---:|
| End to end | 143.877 | 134.498–158.681 |
| Text encoding | 14.960 | 6.932–21.453 |
| Denoising | 127.931 | 121.877–136.618 |
| VAE decoding | 0.781 | 0.421–4.053 |

Measured generations: 10; first image: 158.945 s; cached load: 25.367 s.
Output hash repeatability: True. This is not a quality assessment.

Whole-system available minimum: 0.025 GiB; whole-system swap start/peak: 17.301/24.219 GiB. Harness/process peak RSS: 0.558 GiB. These must not be interpreted as GPU allocation peaks.

## Interpretation

A completed full protocol establishes execution and repeated timing under the captured system state. User applications, system swap and background preparation were not isolated. CPU-side and GPU-side timing boundaries follow the linked metric definitions. No formal quality eligibility or controlled performance claim is established.

## Next experiment

Repeat this exact configuration under an isolated system state if a controlled performance comparison is required.

## Reference and option audit

[Reference](m5-flux-klein-base-001.md). Transformer-only fullgraph Inductor execution recorded one graph and 144 generated kernels, with no graph breaks or eager error fallback. The smoke first-forward check passed at relative RMSE 0.00982543 against the 0.01 engineering limit; the smoke first image includes an additional eager forward, while the full protocol does not. Generated code was reused from the compiler cache: first-image duration is not a standalone compiler-build cost. The full output differs from the eager reference. The inspected sign reads “HELLO WORLD”; this is not formal image-quality approval. Significant whole-system swap prevents a controlled speedup claim.

[Execution receipt](../results/runs/m5-flux-klein-base-004__baseline__20261015-135837/compiler.json); [image diagnostic](../results/runs/m5-flux-klein-base-004__baseline__20261015-135837/reference-image-check.json).
