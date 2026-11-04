---
type: experiment-record
id: m5-mflux-flux-klein-base-001
status: failed
device: macbook-air-m5
engine: mflux
runs: [m5-mflux-flux-klein-base-001__attempt__20261012-161808]
updated: 2026-10-12
---

# m5-mflux-flux-klein-base-001

## Question

Week 2 engine coverage carried into Week 3, RQ1: measure the M5 Klein Base workload with unquantized BF16 reference weights.

## Setup

M5 MFLUX Klein Base, BF16 unquantized reference, 512x512, no reuse/offload.

The [procedure](../wiki/methods/mflux-howto.md) specifies the pinned MFLUX/MLX implementation,
checkpoints and gates. Native prediction compilation remains enabled. Quantization applies to
eligible layers across components; actual counts and tensor dtypes are recorded per run.
The native FP32 rotary frequency buffer is preserved; other floating parameters must be BF16.

[Metric definitions](../wiki/methods/baseline-metrics.md#mflux--mlx-on-m5): synchronized host stages,
including denoising scheduler/CFG work, and separate MLX allocator/system memory. These are not GPU event times.
First and warm-ups are separate from ten measured generations in full protocols. Smoke tests have
one first and one measured generation. Filesystem writes are outside generation timing.

## Results

### m5-mflux-flux-klein-base-001__attempt__20261012-161808

[Status](../results/runs/m5-mflux-flux-klein-base-001__attempt__20261012-161808/status.json): failed. [Configuration](../results/runs/m5-mflux-flux-klein-base-001__attempt__20261012-161808/config.json).

Error: `SIGSEGV in MLX Metal CommandEncoder::set_input_array during Compiled::eval_gpu; macOS crash report matched PID 2594 and queue parent 98727.`. Failure evidence is retained.

[Native crash evidence](../results/runs/m5-mflux-flux-klein-base-001__attempt__20261012-161808/crash-evidence.json): matched process identity and Metal compiled-evaluation stack; root cause remains unproven.

## Observations and limits

Functional gates check phase counts, image dimensions/hashes/nonflat pixels, timings and clock continuity.
They do not assess prompt alignment or establish quantization quality. Background applications, system
swap and thermal conditions are not isolated; latency is descriptive of the recorded state. Native MLX
noise is not shared with other frameworks. No formal quality approval or isolated engine speedup is claimed.

## Next experiment

Complete or review the matching gated full protocol; no additional variants implied.
