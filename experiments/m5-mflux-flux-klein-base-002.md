---
type: experiment-record
id: m5-mflux-flux-klein-base-002
status: complete
device: macbook-air-m5
engine: mflux
runs: [m5-mflux-flux-klein-base-002__attempt__20261012-161817, m5-mflux-flux-klein-base-002__baseline__20261012-162447]
updated: 2026-10-12
---

# m5-mflux-flux-klein-base-002

## Question

Week 2 engine coverage carried into Week 3, RQ1: measure the M5 Klein Base workload with 8-bit affine eligible-layer weights.

## Setup

M5 MFLUX Klein Base, 8-bit affine weight quantization of all eligible layers; reference m5-mflux-flux-klein-base-001, 512x512, no reuse/offload.

The [procedure](../wiki/methods/mflux-howto.md) specifies the pinned MFLUX/MLX implementation,
checkpoints and gates. Native prediction compilation remains enabled. Quantization applies to
eligible layers across components; actual counts and tensor dtypes are recorded per run.
The native FP32 rotary frequency buffer is preserved; other floating parameters must be BF16.

[Metric definitions](../wiki/methods/baseline-metrics.md#mflux--mlx-on-m5): synchronized host stages,
including denoising scheduler/CFG work, and separate MLX allocator/system memory. These are not GPU event times.
First and warm-ups are separate from ten measured generations in full protocols. Smoke tests have
one first and one measured generation. Filesystem writes are outside generation timing.

## Results

### m5-mflux-flux-klein-base-002__attempt__20261012-161817

[Status](../results/runs/m5-mflux-flux-klein-base-002__attempt__20261012-161817/status.json): complete. [Configuration](../results/runs/m5-mflux-flux-klein-base-002__attempt__20261012-161817/config.json).

[Summary](../results/runs/m5-mflux-flux-klein-base-002__attempt__20261012-161817/summary.json), [environment](../results/runs/m5-mflux-flux-klein-base-002__attempt__20261012-161817/environment.json), [layer audit](../results/runs/m5-mflux-flux-klein-base-002__attempt__20261012-161817/quantization-audit.json).

| Metric | Median (s) | Min–max (s) |
|---|---:|---:|
| End to end | 212.972 | 212.972–212.972 |
| Text encoding | 17.732 | 17.732–17.732 |
| Denoising + scheduler | 177.543 | 177.543–177.543 |
| VAE decoding | 17.676 | 17.676–17.676 |

Measured generations: 1; first image: 172.135 s; cached load/quantization: 3.143 s.

MLX peak active allocation median: 10.351 GiB. Whole-system available minimum: 1.036 GiB; swap start/peak: 13.421/19.417 GiB. Scopes overlap and must not be added.

Measured image hash repeatability: True. Clock continuity: True. Formal quality eligibility: false.

[Visual review](../results/runs/m5-mflux-flux-klein-base-002__attempt__20261012-161817/visual-review.json): Stylized orange cat holding a clearly readable HELLO WORLD sign. Functional visual inspection only; formal quality eligibility is not established.

[Functional/protocol validation](../results/runs/m5-mflux-flux-klein-base-002__attempt__20261012-161817/validation.json): True.

[Per-generation data](../results/runs/m5-mflux-flux-klein-base-002__attempt__20261012-161817/runs.csv).

### m5-mflux-flux-klein-base-002__baseline__20261012-162447

[Status](../results/runs/m5-mflux-flux-klein-base-002__baseline__20261012-162447/status.json): complete. [Configuration](../results/runs/m5-mflux-flux-klein-base-002__baseline__20261012-162447/config.json).

[Summary](../results/runs/m5-mflux-flux-klein-base-002__baseline__20261012-162447/summary.json), [environment](../results/runs/m5-mflux-flux-klein-base-002__baseline__20261012-162447/environment.json), [layer audit](../results/runs/m5-mflux-flux-klein-base-002__baseline__20261012-162447/quantization-audit.json).

| Metric | Median (s) | Min–max (s) |
|---|---:|---:|
| End to end | 206.175 | 188.161–249.298 |
| Text encoding | 13.408 | 11.966–17.287 |
| Denoising + scheduler | 168.837 | 152.604–213.062 |
| VAE decoding | 23.977 | 20.139–30.403 |

Measured generations: 10; first image: 181.496 s; cached load/quantization: 3.131 s.

MLX peak active allocation median: 10.351 GiB. Whole-system available minimum: 1.792 GiB; swap start/peak: 15.385/20.385 GiB. Scopes overlap and must not be added.

Measured image hash repeatability: True. Clock continuity: True. Formal quality eligibility: false.

[Visual review](../results/runs/m5-mflux-flux-klein-base-002__baseline__20261012-162447/visual-review.json): Stylized orange cat and clearly readable HELLO WORLD sign. Qualitative inspection only, not formal quality evaluation.

[Functional/protocol validation](../results/runs/m5-mflux-flux-klein-base-002__baseline__20261012-162447/validation.json): True.

[Per-generation data](../results/runs/m5-mflux-flux-klein-base-002__baseline__20261012-162447/runs.csv).

## Observations and limits

Functional gates check phase counts, image dimensions/hashes/nonflat pixels, timings and clock continuity.
They do not assess prompt alignment or establish quantization quality. Background applications, system
swap and thermal conditions are not isolated; latency is descriptive of the recorded state. Native MLX
noise is not shared with other frameworks. No formal quality approval or isolated engine speedup is claimed.

## Next experiment

Complete or review the matching gated full protocol; no additional variants implied.
