---
type: experiment-record
id: m5-mflux-flux-klein-base-006
status: complete
device: macbook-air-m5
engine: mflux
runs: [m5-mflux-flux-klein-base-006__attempt__20261015-163359, m5-mflux-flux-klein-base-006__attempt__20261015-232458, m5-mflux-flux-klein-base-006__baseline__20261015-232911]
updated: 2026-10-16
---

# m5-mflux-flux-klein-base-006

## Question

Week 2–3 Apple individual-option work supporting RQ1: can this labelled configuration complete the full repeated protocol on the M5?

## Setup

M5 MFLUX Klein-base q8: vae-tiled256 only versus configs/m5-mflux-flux-klein-base-q8-512.resolved.json; native MLX compilation retained.

[Configuration](../configs/m5-mflux-flux-klein-base-q8-512-vae-tiled256.resolved.json). One first image, three discarded warm-ups and ten measured generations for the full protocol. Smoke gates use one first and one measured image. Parameters, model revisions, backend and precision are explicit in each execution config.

[M5 procedure](../wiki/methods/m5-howto.md); [metric definitions](../wiki/methods/baseline-metrics.md).

## Results

### m5-mflux-flux-klein-base-006__attempt__20261015-163359

[Status](../results/runs/m5-mflux-flux-klein-base-006__attempt__20261015-163359/status.json), [execution config](../results/runs/m5-mflux-flux-klein-base-006__attempt__20261015-163359/config.json). Status: failed.

The AC-power preflight stopped this attempt before model loading or inference. It is not an engine/tiling failure and supplies no timing result.

```text
AssertionError('AC power required')
```

### m5-mflux-flux-klein-base-006__attempt__20261015-232458

[Status](../results/runs/m5-mflux-flux-klein-base-006__attempt__20261015-232458/status.json), [execution config](../results/runs/m5-mflux-flux-klein-base-006__attempt__20261015-232458/config.json). Status: complete.

[Summary](../results/runs/m5-mflux-flux-klein-base-006__attempt__20261015-232458/summary.json), [validation](../results/runs/m5-mflux-flux-klein-base-006__attempt__20261015-232458/validation.json).

| Metric | Median (s) | Min–max (s) |
|---|---:|---:|
| End to end | 139.790 | 139.790–139.790 |
| Text encoding | 0.749 | 0.749–0.749 |
| Denoising | 137.666 | 137.666–137.666 |
| VAE decoding | 1.368 | 1.368–1.368 |

Measured generations: 1; first image: 103.644 s; cached load: 3.942 s.
Output hash repeatability: True. This is not a quality assessment.

Whole-system available minimum: 1.677 GiB; whole-system swap start/peak: 12.780/20.321 GiB. Harness/process peak RSS: 3.738 GiB. These must not be interpreted as GPU allocation peaks.

### m5-mflux-flux-klein-base-006__baseline__20261015-232911

[Status](../results/runs/m5-mflux-flux-klein-base-006__baseline__20261015-232911/status.json), [execution config](../results/runs/m5-mflux-flux-klein-base-006__baseline__20261015-232911/config.json). Status: complete.

[Summary](../results/runs/m5-mflux-flux-klein-base-006__baseline__20261015-232911/summary.json), [validation](../results/runs/m5-mflux-flux-klein-base-006__baseline__20261015-232911/validation.json).

| Metric | Median (s) | Min–max (s) |
|---|---:|---:|
| End to end | 128.976 | 127.111–136.041 |
| Text encoding | 0.559 | 0.513–0.611 |
| Denoising | 127.246 | 125.076–133.877 |
| VAE decoding | 1.239 | 1.099–1.662 |

Measured generations: 10; first image: 148.287 s; cached load: 3.181 s.
Output hash repeatability: True. This is not a quality assessment.

Whole-system available minimum: 2.373 GiB; whole-system swap start/peak: 14.833/20.640 GiB. Harness/process peak RSS: 5.175 GiB. These must not be interpreted as GPU allocation peaks.

## Interpretation

A completed full protocol establishes execution and repeated timing under the captured system state. User applications, system swap and background preparation were not isolated. CPU-side and GPU-side timing boundaries follow the linked metric definitions. No formal quality eligibility or controlled performance claim is established.

## Next experiment

Repeat this exact configuration under an isolated system state if a controlled performance comparison is required.

## Reference and option audit

[Reference](../experiments/m5-mflux-flux-klein-base-002.md). Only VAE tiling changes: 256-pixel tiles with 64-pixel overlap; 8-bit affine quantization and native MLX compilation stay fixed. The [execution audit](../results/runs/m5-mflux-flux-klein-base-006__baseline__20261015-232911/execution-audit.json) confirms nine tile decodes per image, 50 denoising steps and two CFG passes per step. Stage times are synchronized host durations. The full protocol passed clock-continuity validation. The inspected measured image has a legible HELLO WORLD sign, but [reference-image diagnostics](../results/runs/m5-mflux-flux-klein-base-006__baseline__20261015-232911/reference-image-check.json) show changed pixels. Tile-local normalization is approximate; no formal quality approval or controlled speedup claim.

Smoke and full W&B exports are verified in their per-run receipts and the [batch verification](../output/m5-followup-20261015/final-upload-verification.json). The validated full is included in the [Klein charts](../output/stage-breakdown-20261007/followup-upload-verification.json).
