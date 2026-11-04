---
type: experiment-record
id: m5-mflux-flux-klein-base-005
status: failed
device: macbook-air-m5
engine: mflux
runs: [m5-mflux-flux-klein-base-005__attempt__20261015-153625, m5-mflux-flux-klein-base-005__baseline__20261015-154117]
updated: 2026-10-15
---

# m5-mflux-flux-klein-base-005

## Question

Week 2–3 Apple individual-option work supporting RQ1: can this labelled configuration complete the full repeated protocol on the M5?

## Setup

M5 MFLUX Klein-base q4: vae-tiled256 only versus configs/m5-mflux-flux-klein-base-q4-512.resolved.json; native MLX compilation retained.

[Configuration](../configs/m5-mflux-flux-klein-base-q4-512-vae-tiled256.resolved.json). One first image, three discarded warm-ups and ten measured generations for the full protocol. Smoke gates use one first and one measured image. Parameters, model revisions, backend and precision are explicit in each execution config.

[M5 procedure](../wiki/methods/m5-howto.md); [metric definitions](../wiki/methods/baseline-metrics.md).

## Results

### m5-mflux-flux-klein-base-005__attempt__20261015-153625

[Status](../results/runs/m5-mflux-flux-klein-base-005__attempt__20261015-153625/status.json), [execution config](../results/runs/m5-mflux-flux-klein-base-005__attempt__20261015-153625/config.json). Status: complete.

[Summary](../results/runs/m5-mflux-flux-klein-base-005__attempt__20261015-153625/summary.json), [validation](../results/runs/m5-mflux-flux-klein-base-005__attempt__20261015-153625/validation.json).

| Metric | Median (s) | Min–max (s) |
|---|---:|---:|
| End to end | 154.682 | 154.682–154.682 |
| Text encoding | 0.588 | 0.588–0.588 |
| Denoising | 152.686 | 152.686–152.686 |
| VAE decoding | 1.401 | 1.401–1.401 |

Measured generations: 1; first image: 128.311 s; cached load: 3.408 s.
Output hash repeatability: True. This is not a quality assessment.

Whole-system available minimum: 2.633 GiB; whole-system swap start/peak: 10.226/14.718 GiB. Harness/process peak RSS: 4.291 GiB. These must not be interpreted as GPU allocation peaks.

### m5-mflux-flux-klein-base-005__baseline__20261015-154117

[Status](../results/runs/m5-mflux-flux-klein-base-005__baseline__20261015-154117/status.json), [execution config](../results/runs/m5-mflux-flux-klein-base-005__baseline__20261015-154117/config.json). Status: failed.

Execution or validation failed; the error and partial measurements are retained. No performance claim.

```text
StopImageGenerationException('Stopping image generation at step 27/50')
```

## Interpretation

The smoke passed, but the full protocol is invalid: macOS entered clamshell sleep at 16:06:07 and woke at 16:30:59 local time. The sampler recorded a 1299.187-second wall/monotonic gap. The verified benchmark process was interrupted to preserve partial outputs; the recorded stop exception is a consequence, not the initiating cause ([power evidence](../results/runs/m5-mflux-flux-klein-base-005__baseline__20261015-154117/sleep-event-evidence.txt), [interruption receipt](../results/runs/m5-mflux-flux-klein-base-005__baseline__20261015-154117/sleep-interruption.json), [clock summary](../results/runs/m5-mflux-flux-klein-base-005__baseline__20261015-154117/system-memory-summary.json)). An external W&B update also overlapped this attempt ([observation](../results/runs/m5-mflux-flux-klein-base-005__baseline__20261015-154117/external-activity-observation.json)). No latency comparison is valid for this partial full run.

## Next experiment

Recover only the unchanged full protocol on AC power with the lid open, reusing the valid smoke gate.

## Reference and option audit

[Reference](../experiments/m5-mflux-flux-klein-base-003.md). Only VAE tiling changes: 256-pixel tiles with 64-pixel overlap; native MLX compilation and quantization stay fixed. Tile-local normalization can change pixels and colors; no formal quality approval.
