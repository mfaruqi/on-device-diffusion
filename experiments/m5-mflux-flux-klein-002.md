---
type: experiment-record
id: m5-mflux-flux-klein-002
status: complete
device: macbook-air-m5
engine: mflux
runs: [m5-mflux-flux-klein-002__attempt__20261012-155918, m5-mflux-flux-klein-002__baseline__20261012-160000]
updated: 2026-10-12
---

# m5-mflux-flux-klein-002

## Question

Week 2 engine coverage carried into Week 3, RQ1: measure the M5 Klein distilled workload with 8-bit affine eligible-layer weights.

## Setup

M5 MFLUX Klein distilled, 8-bit affine weight quantization of all eligible layers; reference m5-mflux-flux-klein-001, 512x512, no reuse/offload.

The [procedure](../wiki/methods/mflux-howto.md) specifies the pinned MFLUX/MLX implementation,
checkpoints and gates. Native prediction compilation remains enabled. Quantization applies to
eligible layers across components; actual counts and tensor dtypes are recorded per run.
The native FP32 rotary frequency buffer is preserved; other floating parameters must be BF16.

[Metric definitions](../wiki/methods/baseline-metrics.md#mflux--mlx-on-m5): synchronized host stages,
including denoising scheduler/CFG work, and separate MLX allocator/system memory. These are not GPU event times.
First and warm-ups are separate from ten measured generations in full protocols. Smoke tests have
one first and one measured generation. Filesystem writes are outside generation timing.

## Results

### m5-mflux-flux-klein-002__attempt__20261012-155918

[Status](../results/runs/m5-mflux-flux-klein-002__attempt__20261012-155918/status.json): complete. [Configuration](../results/runs/m5-mflux-flux-klein-002__attempt__20261012-155918/config.json).

[Summary](../results/runs/m5-mflux-flux-klein-002__attempt__20261012-155918/summary.json), [environment](../results/runs/m5-mflux-flux-klein-002__attempt__20261012-155918/environment.json), [layer audit](../results/runs/m5-mflux-flux-klein-002__attempt__20261012-155918/quantization-audit.json).

| Metric | Median (s) | Min–max (s) |
|---|---:|---:|
| End to end | 21.434 | 21.434–21.434 |
| Text encoding | 5.369 | 5.369–5.369 |
| Denoising + scheduler | 9.496 | 9.496–9.496 |
| VAE decoding | 6.560 | 6.560–6.560 |

Measured generations: 1; first image: 12.571 s; cached load/quantization: 5.336 s.

MLX peak active allocation median: 10.351 GiB. Whole-system available minimum: 1.976 GiB; swap start/peak: 13.171/18.677 GiB. Scopes overlap and must not be added.

Measured image hash repeatability: True. Clock continuity: True. Formal quality eligibility: false.

[Functional/protocol validation](../results/runs/m5-mflux-flux-klein-002__attempt__20261012-155918/validation.json): True.

[Per-generation data](../results/runs/m5-mflux-flux-klein-002__attempt__20261012-155918/runs.csv).

### m5-mflux-flux-klein-002__baseline__20261012-160000

[Status](../results/runs/m5-mflux-flux-klein-002__baseline__20261012-160000/status.json): complete. [Configuration](../results/runs/m5-mflux-flux-klein-002__baseline__20261012-160000/config.json).

[Summary](../results/runs/m5-mflux-flux-klein-002__baseline__20261012-160000/summary.json), [environment](../results/runs/m5-mflux-flux-klein-002__baseline__20261012-160000/environment.json), [layer audit](../results/runs/m5-mflux-flux-klein-002__baseline__20261012-160000/quantization-audit.json).

| Metric | Median (s) | Min–max (s) |
|---|---:|---:|
| End to end | 71.173 | 66.166–91.023 |
| Text encoding | 17.209 | 15.411–27.236 |
| Denoising + scheduler | 26.831 | 22.963–35.633 |
| VAE decoding | 26.510 | 23.616–37.317 |

Measured generations: 10; first image: 14.245 s; cached load/quantization: 3.184 s.

MLX peak active allocation median: 10.351 GiB. Whole-system available minimum: 2.333 GiB; swap start/peak: 18.582/23.662 GiB. Scopes overlap and must not be added.

Measured image hash repeatability: True. Clock continuity: True. Formal quality eligibility: false.

[Functional/protocol validation](../results/runs/m5-mflux-flux-klein-002__baseline__20261012-160000/validation.json): True.

[Per-generation data](../results/runs/m5-mflux-flux-klein-002__baseline__20261012-160000/runs.csv).

## Observations and limits

Functional gates check phase counts, image dimensions/hashes/nonflat pixels, timings and clock continuity.
They do not assess prompt alignment or establish quantization quality. Background applications, system
swap and thermal conditions are not isolated; latency is descriptive of the recorded state. Native MLX
noise is not shared with other frameworks. No formal quality approval or isolated engine speedup is claimed.

## Next experiment

Complete or review the matching gated full protocol; no additional variants implied.
