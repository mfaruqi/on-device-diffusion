---
type: experiment-record
id: m5-mflux-flux-klein-004
status: complete
device: macbook-air-m5
engine: mflux
runs: [m5-mflux-flux-klein-004__attempt__20261015-134618, m5-mflux-flux-klein-004__baseline__20261015-134636]
updated: 2026-10-15
---

# m5-mflux-flux-klein-004

## Question

Week 2–3 Apple individual-option work supporting RQ1: can this labelled configuration complete the full repeated protocol on the M5?

## Setup

M5 MFLUX Klein distilled q4: conditioning only versus configs/m5-mflux-flux-klein-q4-512.resolved.json; native MLX compilation retained.

[Configuration](../configs/m5-mflux-flux-klein-q4-512-conditioning.resolved.json). One first image, three discarded warm-ups and ten measured generations for the full protocol. Smoke gates use one first and one measured image. Parameters, model revisions, backend and precision are explicit in each execution config.

[M5 procedure](../wiki/methods/m5-howto.md); [metric definitions](../wiki/methods/baseline-metrics.md).

## Results

### m5-mflux-flux-klein-004__attempt__20261015-134618

[Status](../results/runs/m5-mflux-flux-klein-004__attempt__20261015-134618/status.json), [execution config](../results/runs/m5-mflux-flux-klein-004__attempt__20261015-134618/config.json). Status: complete.

[Summary](../results/runs/m5-mflux-flux-klein-004__attempt__20261015-134618/summary.json), [validation](../results/runs/m5-mflux-flux-klein-004__attempt__20261015-134618/validation.json).

| Metric | Median (s) | Min–max (s) |
|---|---:|---:|
| End to end | 5.021 | 5.021–5.021 |
| Text encoding | 0.000 | 0.000–0.000 |
| Denoising | 4.188 | 4.188–4.188 |
| VAE decoding | 0.828 | 0.828–0.828 |

Measured generations: 1; first image: 5.863 s; cached load: 3.145 s.
Output hash repeatability: True. This is not a quality assessment.

Whole-system available minimum: 2.245 GiB; whole-system swap start/peak: 17.616/19.257 GiB. Harness/process peak RSS: 5.411 GiB. These must not be interpreted as GPU allocation peaks.

### m5-mflux-flux-klein-004__baseline__20261015-134636

[Status](../results/runs/m5-mflux-flux-klein-004__baseline__20261015-134636/status.json), [execution config](../results/runs/m5-mflux-flux-klein-004__baseline__20261015-134636/config.json). Status: complete.

[Summary](../results/runs/m5-mflux-flux-klein-004__baseline__20261015-134636/summary.json), [validation](../results/runs/m5-mflux-flux-klein-004__baseline__20261015-134636/validation.json).

| Metric | Median (s) | Min–max (s) |
|---|---:|---:|
| End to end | 5.691 | 5.276–7.622 |
| Text encoding | 0.000 | 0.000–0.000 |
| Denoising | 4.803 | 4.328–6.217 |
| VAE decoding | 0.884 | 0.465–1.725 |

Measured generations: 10; first image: 5.844 s; cached load: 3.142 s.
Output hash repeatability: True. This is not a quality assessment.

Whole-system available minimum: 1.418 GiB; whole-system swap start/peak: 19.175/21.147 GiB. Harness/process peak RSS: 5.591 GiB. These must not be interpreted as GPU allocation peaks.

## Interpretation

A completed full protocol establishes execution and repeated timing under the captured system state. User applications, system swap and background preparation were not isolated. CPU-side and GPU-side timing boundaries follow the linked metric definitions. No formal quality eligibility or controlled performance claim is established.

## Next experiment

Repeat this exact configuration under an isolated system state if a controlled performance comparison is required.

## Reference and option audit

[Reference](m5-mflux-flux-klein-003.md). Exact prompt-pair reuse recorded one cache population and 13 subsequent hits; the first measured image is pixel-identical to the same-engine 4-bit reference. Lookup and synchronization remain in the text stage: the sub-millisecond duration rounds to 0.000 s in the table, but is not zero work. The inspected sign still has the existing 4-bit text-generation error; this is not formal quality approval.

[Execution audit](../results/runs/m5-mflux-flux-klein-004__baseline__20261015-134636/execution-audit.json); [reference-image diagnostic](../results/runs/m5-mflux-flux-klein-004__baseline__20261015-134636/reference-image-check.json).
