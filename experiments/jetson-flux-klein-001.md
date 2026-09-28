---
type: experiment-record
id: jetson-flux-klein-001
status: failed
device: jetson-orin-nano
engine: stable-diffusion-cpp
runs: [jetson-flux-klein-001__attempt__20260928-183618]
updated: 2026-09-28
---

# jetson-flux-klein-001: quantized 512×512 all-resident feasibility attempt

Status: **failed**. Evidence import is partial; full logs remain on the Jetson.

## Question

Can the quantized FLUX.2 klein 4B pipeline generate one image with all weights resident
on CUDA and monolithic graph execution on this Jetson? This establishes feasibility
for a separately labelled quantized, lower-resolution workload; it is not a performance
baseline or an optimization comparison. Supports Week 1's “A100 and Jetson image baselines”
and RQ1 ([proposal overview](../wiki/project/overview.md)).

## Setup

- [Reconstructed configuration](../configs/jetson-flux-klein-q4-512-all-resident.json):
  pinned sd.cpp revision, transformer Q4_0, text encoder Q4_K_M, original VAE;
  component revisions and expected hashes recorded. Checksum success was operator-confirmed.
- Image 512×512, four Euler steps with flux2 scheduler, CFG 1.0, seed 0, batch one;
  prompt “A cat holding a sign that says hello world”.
- CUDA compute and parameter placement, eager loading, automatic fitting and graph
  segmentation disabled, diffusion flash attention enabled; no VAE tiling or step caching.
- One attempted generation, no warm-ups or measured repetitions. Tegrastats sampled
  at one-second intervals. No structured per-stage instrumentation or profiling.
- Power mode observed before the attempt: 25W. CUDA vectorAdd passed and sd.cpp detected
  Orin compute capability 8.7 ([terminal evidence](../results/runs/jetson-flux-klein-001__attempt__20260928-183618/terminal-excerpts.json)).
- Shared [metric definitions](../wiki/methods/baseline-metrics.md) apply to future benchmark
  reporting; this attempt has no validated benchmark metrics.

## Results

[Run directory](../results/runs/jetson-flux-klein-001__attempt__20260928-183618/),
[status](../results/runs/jetson-flux-klein-001__attempt__20260928-183618/status.json),
[summary](../results/runs/jetson-flux-klein-001__attempt__20260928-183618/summary.json).

The process exited with code 1. The engine could not make enough memory available on
CUDA0 and reported “flux segment 1/1 (graph) failed during weight preparation”, followed
by diffusion sampling and generation failures. It then released its tracked parameter
buffers ([error excerpt](../results/runs/jetson-flux-klein-001__attempt__20260928-183618/terminal-excerpts.json)).

No completed image, repeatability check, quality result, or end-to-end latency is available.
The short sampling-failure diagnostic is not a generation-time measurement. The imported
tegrastats lines are post-failure samples, not peak-memory observations. Full generation
and sampling logs await import from the original directory identified in the
[run README](../results/runs/jetson-flux-klein-001__attempt__20260928-183618/README.md).

## Observations and interpretation

The observed failure is insufficient available memory during denoising weight preparation,
before VAE decoding. It establishes a limitation of this configuration under the recorded
conditions, not general FLUX infeasibility on the device. The post-failure memory readings
do not establish the execution peak, a leak, or thermal throttling.

## Next experiment

- Candidate `jetson-flux-klein-q4-512-segmented.json`: enable graph segmentation only;
  not yet created or run, and not guaranteed to resolve residency pressure.
