---
type: experiment-record
id: jetson-flux-klein-002
status: failed
device: jetson-orin-nano
engine: stable-diffusion-cpp
runs: [jetson-flux-klein-002__attempt__20260928-184410]
updated: 2026-09-28
---

# jetson-flux-klein-002: segmented quantized 512×512 feasibility attempt

Status: **failed**; partial evidence import, full originals remain on the Jetson.

## Question

Can automatic graph segmentation enable one image with the quantized 512×512
workload while parameters remain eagerly resident on CUDA? This supports Week 1's
“A100 and Jetson image baselines” and RQ1 ([overview](../wiki/project/overview.md)).

## Setup

[Configuration](../configs/jetson-flux-klein-q4-512-segmented.json) changes only graph
segmentation relative to [jetson-flux-klein-001](jetson-flux-klein-001.md).
The pinned sd.cpp revision and model components, Q4_0 transformer, Q4_K_M text encoder,
original VAE, prompt, seed, resolution, four Euler steps and flux2 scheduler are retained.
Compute and parameters are assigned CUDA0, eager loading remains enabled, auto-fit
is off, and VAE tiling is off. The intended power mode is 25W; this attempt's captured
power-mode file awaits import.

One attempted image, no warm-ups, repeats or profiling. Tegrastats sampling interval
is one second. [Metric definitions](../wiki/methods/baseline-metrics.md) apply to future
benchmark reporting; no validated benchmark metric is reported here.

## Results

[Run directory](../results/runs/jetson-flux-klein-002__attempt__20260928-184410/),
[status](../results/runs/jetson-flux-klein-002__attempt__20260928-184410/status.json),
[summary](../results/runs/jetson-flux-klein-002__attempt__20260928-184410/summary.json).

The engine split Qwen3 execution into segments, but reported
“qwen3 segment 2/29 (llm.text.layers.0) failed during allocated capacity check”.
The preceding diagnostic reports insufficient available CUDA0 memory; prompt encoding
and generation then failed, with exit code 1. The engine released its parameter buffers
([terminal excerpts](../results/runs/jetson-flux-klein-002__attempt__20260928-184410/terminal-excerpts.json)).

No image, end-to-end latency, peak memory, repeatability or quality result is available.
Full command, environment snapshots and sampling logs remain on the Jetson; the
[run README](../results/runs/jetson-flux-klein-002__attempt__20260928-184410/README.md) lists the missing originals.

## Interpretation

Segmentation was exercised, but was insufficient with this parameter residency policy
under the observed system conditions. A different failure location does not by itself
establish a latency or memory regression: complete traces and matched starting conditions
are unavailable. Hypothesis: reducing persistent parameter residency may leave more room
for segment execution buffers.

## Next experiment

- `jetson-flux-klein-q4-512-segmented-disk.json`: change parameter storage to disk only;
  retain CUDA computation and segmentation. Not yet run.
