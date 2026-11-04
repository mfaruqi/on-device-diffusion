---
type: experiment-record
id: m5-mflux-flux-klein-base-005
status: complete
device: macbook-air-m5
engine: mflux
runs: [m5-mflux-flux-klein-base-005__attempt__20261015-153625, m5-mflux-flux-klein-base-005__baseline__20261015-154117, m5-mflux-flux-klein-base-005__baseline__20261015-214940, m5-mflux-flux-klein-base-005__baseline__20261015-225903]
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

Clamshell sleep invalidated this attempt: the recorded wall/monotonic gap was 1299.187 seconds. The runner was then interrupted; the exception below is a consequence, not the root cause. [Sleep evidence](../results/runs/m5-mflux-flux-klein-base-005__baseline__20261015-154117/sleep-interruption.json) and [external upload overlap observation](../results/runs/m5-mflux-flux-klein-base-005__baseline__20261015-154117/external-activity-observation.json) are retained. Timings are excluded.

```text
StopImageGenerationException('Stopping image generation at step 27/50')
```

### m5-mflux-flux-klein-base-005__baseline__20261015-214940

[Status](../results/runs/m5-mflux-flux-klein-base-005__baseline__20261015-214940/status.json), [execution config](../results/runs/m5-mflux-flux-klein-base-005__baseline__20261015-214940/config.json). Status: failed.

Stopped on request during the first image, before any completed timing sample. [Interruption receipt](../results/runs/m5-mflux-flux-klein-base-005__baseline__20261015-214940/interruption.json). No engine failure or performance claim is inferred.

```text
StopImageGenerationException('Stopping image generation at step 17/50')
```

### m5-mflux-flux-klein-base-005__baseline__20261015-225903

[Status](../results/runs/m5-mflux-flux-klein-base-005__baseline__20261015-225903/status.json), [execution config](../results/runs/m5-mflux-flux-klein-base-005__baseline__20261015-225903/config.json). Status: complete.

[Summary](../results/runs/m5-mflux-flux-klein-base-005__baseline__20261015-225903/summary.json), [validation](../results/runs/m5-mflux-flux-klein-base-005__baseline__20261015-225903/validation.json).

| Metric | Median (s) | Min–max (s) |
|---|---:|---:|
| End to end | 111.959 | 105.587–113.790 |
| Text encoding | 0.410 | 0.383–0.457 |
| Denoising | 110.409 | 104.050–112.219 |
| VAE decoding | 1.134 | 1.119–1.207 |

Measured generations: 10; first image: 89.815 s; cached load: 3.945 s.
Output hash repeatability: True. This is not a quality assessment.

Whole-system available minimum: 1.245 GiB; whole-system swap start/peak: 10.937/19.415 GiB. Harness/process peak RSS: 3.628 GiB. These must not be interpreted as GPU allocation peaks.

## Interpretation

A completed full protocol establishes execution and repeated timing under the captured system state. User applications, system swap and background preparation were not isolated. CPU-side and GPU-side timing boundaries follow the linked metric definitions. No formal quality eligibility or controlled performance claim is established.

## Next experiment

Repeat this exact configuration under an isolated system state if a controlled performance comparison is required.

## Reference and option audit

[Reference](../experiments/m5-mflux-flux-klein-base-003.md). Only VAE tiling changes: 256-pixel tiles with 64-pixel overlap; quantization and native MLX compilation stay fixed. The [execution audit](../results/runs/m5-mflux-flux-klein-base-005__baseline__20261015-225903/execution-audit.json) records nine tile decodes per image, 50 denoising steps and two CFG passes per step. Stage timings are synchronized host times. Tile-local normalization changes pixels; the reviewed measured image has incorrect sign text, and formal quality remains unapproved. The full run passed clock-continuity validation. Its [W&B upload](../results/runs/m5-mflux-flux-klein-base-005__baseline__20261015-225903/wandb-followup-verification.json) is verified; the validated recovery is included in the [Klein charts](../output/stage-breakdown-20261007/followup-upload-verification.json).
