---
type: experiment-record
id: m5-mflux-flux-klein-003
status: complete
device: macbook-air-m5
engine: mflux
runs: [m5-mflux-flux-klein-003__attempt__20261012-161526, m5-mflux-flux-klein-003__baseline__20261012-161546]
updated: 2026-10-12
---

# m5-mflux-flux-klein-003

## Question

Week 2 engine coverage carried into Week 3, RQ1: measure the M5 Klein distilled workload with 4-bit affine eligible-layer weights.

## Setup

M5 MFLUX Klein distilled, 4-bit affine weight quantization of all eligible layers; reference m5-mflux-flux-klein-001, 512x512, no reuse/offload.

The [procedure](../wiki/methods/mflux-howto.md) specifies the pinned MFLUX/MLX implementation,
checkpoints and gates. Native prediction compilation remains enabled. Quantization applies to
eligible layers across components; actual counts and tensor dtypes are recorded per run.
The native FP32 rotary frequency buffer is preserved; other floating parameters must be BF16.

[Metric definitions](../wiki/methods/baseline-metrics.md#mflux--mlx-on-m5): synchronized host stages,
including denoising scheduler/CFG work, and separate MLX allocator/system memory. These are not GPU event times.
First and warm-ups are separate from ten measured generations in full protocols. Smoke tests have
one first and one measured generation. Filesystem writes are outside generation timing.

## Results

### m5-mflux-flux-klein-003__attempt__20261012-161526

[Status](../results/runs/m5-mflux-flux-klein-003__attempt__20261012-161526/status.json): complete. [Configuration](../results/runs/m5-mflux-flux-klein-003__attempt__20261012-161526/config.json).

[Summary](../results/runs/m5-mflux-flux-klein-003__attempt__20261012-161526/summary.json), [environment](../results/runs/m5-mflux-flux-klein-003__attempt__20261012-161526/environment.json), [layer audit](../results/runs/m5-mflux-flux-klein-003__attempt__20261012-161526/quantization-audit.json).

| Metric | Median (s) | Min–max (s) |
|---|---:|---:|
| End to end | 7.462 | 7.462–7.462 |
| Text encoding | 0.410 | 0.410–0.410 |
| Denoising + scheduler | 6.230 | 6.230–6.230 |
| VAE decoding | 0.819 | 0.819–0.819 |

Measured generations: 1; first image: 7.638 s; cached load/quantization: 3.282 s.

MLX peak active allocation median: 6.673 GiB. Whole-system available minimum: 1.667 GiB; swap start/peak: 18.255/20.742 GiB. Scopes overlap and must not be added.

Measured image hash repeatability: True. Clock continuity: True. Formal quality eligibility: false.

[Functional/protocol validation](../results/runs/m5-mflux-flux-klein-003__attempt__20261012-161526/validation.json): True.

[Per-generation data](../results/runs/m5-mflux-flux-klein-003__attempt__20261012-161526/runs.csv).

### m5-mflux-flux-klein-003__baseline__20261012-161546

[Status](../results/runs/m5-mflux-flux-klein-003__baseline__20261012-161546/status.json): complete. [Configuration](../results/runs/m5-mflux-flux-klein-003__baseline__20261012-161546/config.json).

[Summary](../results/runs/m5-mflux-flux-klein-003__baseline__20261012-161546/summary.json), [environment](../results/runs/m5-mflux-flux-klein-003__baseline__20261012-161546/environment.json), [layer audit](../results/runs/m5-mflux-flux-klein-003__baseline__20261012-161546/quantization-audit.json).

| Metric | Median (s) | Min–max (s) |
|---|---:|---:|
| End to end | 10.296 | 7.598–13.513 |
| Text encoding | 0.650 | 0.411–2.908 |
| Denoising + scheduler | 8.316 | 6.675–10.443 |
| VAE decoding | 1.033 | 0.493–1.776 |

Measured generations: 10; first image: 7.456 s; cached load/quantization: 3.178 s.

MLX peak active allocation median: 6.673 GiB. Whole-system available minimum: 1.442 GiB; swap start/peak: 14.287/17.959 GiB. Scopes overlap and must not be added.

Measured image hash repeatability: True. Clock continuity: True. Formal quality eligibility: false.

[Visual review](../results/runs/m5-mflux-flux-klein-003__baseline__20261012-161546/visual-review.json): Coherent cat and sign, but sign lettering does not correctly render the requested hello world phrase. This is a visible text-fidelity limitation in this fixed-prompt sample, not a formal quality-set evaluation.

[Functional/protocol validation](../results/runs/m5-mflux-flux-klein-003__baseline__20261012-161546/validation.json): True.

[Per-generation data](../results/runs/m5-mflux-flux-klein-003__baseline__20261012-161546/runs.csv).

## Observations and limits

Functional gates check phase counts, image dimensions/hashes/nonflat pixels, timings and clock continuity.
They do not assess prompt alignment or establish quantization quality. Background applications, system
swap and thermal conditions are not isolated; latency is descriptive of the recorded state. Native MLX
noise is not shared with other frameworks. No formal quality approval or isolated engine speedup is claimed.

## Next experiment

Complete or review the matching gated full protocol; no additional variants implied.
