---
type: experiment-record
id: m5-mflux-flux-klein-base-003
status: complete
device: macbook-air-m5
engine: mflux
runs: [m5-mflux-flux-klein-base-003__attempt__20261012-171421, m5-mflux-flux-klein-base-003__baseline__20261012-171934]
updated: 2026-10-12
---

# m5-mflux-flux-klein-base-003

## Question

Week 2 engine coverage carried into Week 3, RQ1: measure the M5 Klein Base workload with 4-bit affine eligible-layer weights.

## Setup

M5 MFLUX Klein Base, 4-bit affine weight quantization of all eligible layers; reference m5-mflux-flux-klein-base-001, 512x512, no reuse/offload.

The [procedure](../wiki/methods/mflux-howto.md) specifies the pinned MFLUX/MLX implementation,
checkpoints and gates. Native prediction compilation remains enabled. Quantization applies to
eligible layers across components; actual counts and tensor dtypes are recorded per run.
The native FP32 rotary frequency buffer is preserved; other floating parameters must be BF16.

[Metric definitions](../wiki/methods/baseline-metrics.md#mflux--mlx-on-m5): synchronized host stages,
including denoising scheduler/CFG work, and separate MLX allocator/system memory. These are not GPU event times.
First and warm-ups are separate from ten measured generations in full protocols. Smoke tests have
one first and one measured generation. Filesystem writes are outside generation timing.

## Results

### m5-mflux-flux-klein-base-003__attempt__20261012-171421

[Status](../results/runs/m5-mflux-flux-klein-base-003__attempt__20261012-171421/status.json): complete. [Configuration](../results/runs/m5-mflux-flux-klein-base-003__attempt__20261012-171421/config.json).

[Summary](../results/runs/m5-mflux-flux-klein-base-003__attempt__20261012-171421/summary.json), [environment](../results/runs/m5-mflux-flux-klein-base-003__attempt__20261012-171421/environment.json), [layer audit](../results/runs/m5-mflux-flux-klein-base-003__attempt__20261012-171421/quantization-audit.json).

| Metric | Median (s) | Min–max (s) |
|---|---:|---:|
| End to end | 155.086 | 155.086–155.086 |
| Text encoding | 0.645 | 0.645–0.645 |
| Denoising + scheduler | 152.804 | 152.804–152.804 |
| VAE decoding | 1.627 | 1.627–1.627 |

Measured generations: 1; first image: 151.321 s; cached load/quantization: 4.266 s.

MLX peak active allocation median: 6.673 GiB. Whole-system available minimum: 0.610 GiB; swap start/peak: 12.254/18.246 GiB. Scopes overlap and must not be added.

Measured image hash repeatability: True. Clock continuity: True. Formal quality eligibility: false.

[Visual review](../results/runs/m5-mflux-flux-klein-base-003__attempt__20261012-171421/visual-review.json): Coherent orange cat and sign, but sign reads HELHOTE WORLD instead of the requested hello world. This fixed-prompt sample demonstrates a text-fidelity limitation; no formal quality eligibility is implied.

[Functional/protocol validation](../results/runs/m5-mflux-flux-klein-base-003__attempt__20261012-171421/validation.json): True.

[Per-generation data](../results/runs/m5-mflux-flux-klein-base-003__attempt__20261012-171421/runs.csv).

### m5-mflux-flux-klein-base-003__baseline__20261012-171934

[Status](../results/runs/m5-mflux-flux-klein-base-003__baseline__20261012-171934/status.json): complete. [Configuration](../results/runs/m5-mflux-flux-klein-base-003__baseline__20261012-171934/config.json).

[Summary](../results/runs/m5-mflux-flux-klein-base-003__baseline__20261012-171934/summary.json), [environment](../results/runs/m5-mflux-flux-klein-base-003__baseline__20261012-171934/environment.json), [layer audit](../results/runs/m5-mflux-flux-klein-base-003__baseline__20261012-171934/quantization-audit.json).

| Metric | Median (s) | Min–max (s) |
|---|---:|---:|
| End to end | 151.007 | 137.067–178.486 |
| Text encoding | 0.671 | 0.417–1.230 |
| Denoising + scheduler | 146.880 | 134.999–166.207 |
| VAE decoding | 3.169 | 1.236–11.321 |

Measured generations: 10; first image: 158.271 s; cached load/quantization: 3.353 s.

MLX peak active allocation median: 6.673 GiB. Whole-system available minimum: 0.854 GiB; swap start/peak: 12.915/18.148 GiB. Scopes overlap and must not be added.

Measured image hash repeatability: True. Clock continuity: True. Formal quality eligibility: false.

[Visual review](../results/runs/m5-mflux-flux-klein-base-003__baseline__20261012-171934/visual-review.json): Coherent cat and sign, but lettering reads HELHOTE WORLD instead of hello world; the same visible error appears in the smoke sample. Fixed-prompt diagnostic only, not formal quality evaluation.

[Functional/protocol validation](../results/runs/m5-mflux-flux-klein-base-003__baseline__20261012-171934/validation.json): True.

[Per-generation data](../results/runs/m5-mflux-flux-klein-base-003__baseline__20261012-171934/runs.csv).

## Observations and limits

Functional gates check phase counts, image dimensions/hashes/nonflat pixels, timings and clock continuity.
They do not assess prompt alignment or establish quantization quality. Background applications, system
swap and thermal conditions are not isolated; latency is descriptive of the recorded state. Native MLX
noise is not shared with other frameworks. No formal quality approval or isolated engine speedup is claimed.

## Next experiment

Complete or review the matching gated full protocol; no additional variants implied.
