---
type: experiment-record
id: m5-mflux-flux-klein-001
status: complete
device: macbook-air-m5
engine: mflux
runs: [m5-mflux-flux-klein-001__attempt__20261012-153912, m5-mflux-flux-klein-001__attempt__20261012-154009, m5-mflux-flux-klein-001__baseline__20261012-154228]
updated: 2026-10-12
---

# m5-mflux-flux-klein-001

## Question

Week 2 engine coverage carried into Week 3, RQ1: measure the M5 Klein distilled workload with unquantized BF16 reference weights.

## Setup

M5 MFLUX Klein distilled, BF16 unquantized reference, 512x512, no reuse/offload.

The [procedure](../wiki/methods/mflux-howto.md) specifies the pinned MFLUX/MLX implementation,
checkpoints and gates. Native prediction compilation remains enabled. Quantization applies to
eligible layers across components; actual counts and tensor dtypes are recorded per run.
The native FP32 rotary frequency buffer is preserved; other floating parameters must be BF16.

[Metric definitions](../wiki/methods/baseline-metrics.md#mflux--mlx-on-m5): synchronized host stages,
including denoising scheduler/CFG work, and separate MLX allocator/system memory. These are not GPU event times.
First and warm-ups are separate from ten measured generations in full protocols. Smoke tests have
one first and one measured generation. Filesystem writes are outside generation timing.

## Results

### m5-mflux-flux-klein-001__attempt__20261012-153912

[Status](../results/runs/m5-mflux-flux-klein-001__attempt__20261012-153912/status.json): failed. [Configuration](../results/runs/m5-mflux-flux-klein-001__attempt__20261012-153912/config.json).

Error: `AssertionError('text_encoder: unexpected floating parameter dtype')`. Failure evidence is retained.

### m5-mflux-flux-klein-001__attempt__20261012-154009

[Status](../results/runs/m5-mflux-flux-klein-001__attempt__20261012-154009/status.json): complete. [Configuration](../results/runs/m5-mflux-flux-klein-001__attempt__20261012-154009/config.json).

[Summary](../results/runs/m5-mflux-flux-klein-001__attempt__20261012-154009/summary.json), [environment](../results/runs/m5-mflux-flux-klein-001__attempt__20261012-154009/environment.json), [layer audit](../results/runs/m5-mflux-flux-klein-001__attempt__20261012-154009/quantization-audit.json).

| Metric | Median (s) | Min–max (s) |
|---|---:|---:|
| End to end | 66.055 | 66.055–66.055 |
| Text encoding | 31.942 | 31.942–31.942 |
| Denoising + scheduler | 33.359 | 33.359–33.359 |
| VAE decoding | 0.747 | 0.747–0.747 |

Measured generations: 1; first image: 44.323 s; cached load/quantization: 5.271 s.

MLX peak active allocation median: 17.247 GiB. Whole-system available minimum: 0.415 GiB; swap start/peak: 25.528/32.836 GiB. Scopes overlap and must not be added.

Measured image hash repeatability: True. Clock continuity: True. Formal quality eligibility: false.

[Functional/protocol validation](../results/runs/m5-mflux-flux-klein-001__attempt__20261012-154009/validation.json): True.

[Per-generation data](../results/runs/m5-mflux-flux-klein-001__attempt__20261012-154009/runs.csv).

### m5-mflux-flux-klein-001__baseline__20261012-154228

[Status](../results/runs/m5-mflux-flux-klein-001__baseline__20261012-154228/status.json): complete. [Configuration](../results/runs/m5-mflux-flux-klein-001__baseline__20261012-154228/config.json).

[Summary](../results/runs/m5-mflux-flux-klein-001__baseline__20261012-154228/summary.json), [environment](../results/runs/m5-mflux-flux-klein-001__baseline__20261012-154228/environment.json), [layer audit](../results/runs/m5-mflux-flux-klein-001__baseline__20261012-154228/quantization-audit.json).

| Metric | Median (s) | Min–max (s) |
|---|---:|---:|
| End to end | 80.377 | 64.334–105.953 |
| Text encoding | 32.542 | 29.063–50.645 |
| Denoising + scheduler | 48.035 | 32.916–64.968 |
| VAE decoding | 1.275 | 0.769–2.377 |

Measured generations: 10; first image: 7.931 s; cached load/quantization: 4.610 s.

MLX peak active allocation median: 17.247 GiB. Whole-system available minimum: 0.509 GiB; swap start/peak: 26.202/33.719 GiB. Scopes overlap and must not be added.

Measured image hash repeatability: True. Clock continuity: True. Formal quality eligibility: false.

[Functional/protocol validation](../results/runs/m5-mflux-flux-klein-001__baseline__20261012-154228/validation.json): True.

[Per-generation data](../results/runs/m5-mflux-flux-klein-001__baseline__20261012-154228/runs.csv).

## Observations and limits

Functional gates check phase counts, image dimensions/hashes/nonflat pixels, timings and clock continuity.
They do not assess prompt alignment or establish quantization quality. Background applications, system
swap and thermal conditions are not isolated; latency is descriptive of the recorded state. Native MLX
noise is not shared with other frameworks. No formal quality approval or isolated engine speedup is claimed.

## Next experiment

Complete or review the matching gated full protocol; no additional variants implied.
