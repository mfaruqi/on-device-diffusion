---
type: experiment-record
id: jetson-flux-klein-003
status: complete
device: jetson-orin-nano
engine: stable-diffusion-cpp
runs: [jetson-flux-klein-003__attempt__20260928-185009]
updated: 2026-09-28
---

# jetson-flux-klein-003: disk-backed quantized 512×512 feasibility success

Status: **single CLI attempt complete**; visual smoke check passed; repeated benchmarking pending.
Full supplied CLI logs, command, system snapshots and image have been imported.

## Question

Can disk-backed parameter storage enable one image with CUDA computation and graph
segmentation for the quantized 512×512 workload? Supports Week 1's “A100 and Jetson
image baselines” and RQ1 ([overview](../wiki/project/overview.md)).

## Setup

[Configuration](../configs/jetson-flux-klein-q4-512-segmented-disk.json) changes parameter
storage from CUDA0 to disk relative to [jetson-flux-klein-002](jetson-flux-klein-002.md).
The pinned engine and weights, Q4_0 transformer, Q4_K_M text encoder, original VAE,
512×512 size, four Euler steps, flux2 scheduler, CFG 1.0, seed 0 and prompt are retained.
CUDA computation, graph segmentation, eager loading and diffusion flash attention remain
configured; auto-fit, VAE tiling and step caching remain off. The imported power-mode snapshot confirms 25W. Command and verbose settings were audited
([audit](../results/runs/jetson-flux-klein-003__attempt__20260928-185009/engine-audit.json)).

One attempted image, no warm-ups, repeated measurements or profiling. Tegrastats sampled
once per second. Shared [metric definitions](../wiki/methods/baseline-metrics.md) remain
unchanged; the diagnostics below are not validated benchmark metrics.

## Results

[Run directory](../results/runs/jetson-flux-klein-003__attempt__20260928-185009/),
[status](../results/runs/jetson-flux-klein-003__attempt__20260928-185009/status.json),
[summary](../results/runs/jetson-flux-klein-003__attempt__20260928-185009/summary.json).

### Single-invocation timing diagnostics

| Engine-reported boundary | Seconds |
|---|---:|
| Initial tensor loading (before generation) | 64.11 |
| Text encoding (`get_learned_condition`) | 13.61 |
| Sampling | 20.63 |
| Decode (`decode_first_stage`) | 4.95 |
| Generation total (`generate_image`) | 39.42 |

[Extracted diagnostics and input hashes](../results/runs/jetson-flux-klein-003__attempt__20260928-185009/feasibility-analysis.json).
The generation total includes the three stage durations and an unassigned residual;
initial tensor loading is separate. These are printed engine timings, not process wall
clock or warm-run medians. Stages include on-demand weight loading, and no total physical
disk-read time is established.

### Whole-window sampled memory

- Highest reported system RAM: approximately **7.00 GiB** of
  **7.44 GiB**, at local time 2026-09-28T18:50:54.
- Swap occupancy remained approximately **0.416 GiB** across all samples.
- Maximum sampled GPU temperature: **49.09 °C**.
- 114 samples cover 2026-09-28T18:50:10 through 2026-09-28T18:52:05.

Source: [analysis](../results/runs/jetson-flux-klein-003__attempt__20260928-185009/feasibility-analysis.json),
[sample CSV](../results/runs/jetson-flux-klein-003__attempt__20260928-185009/tegrastats-samples.csv).
Memory unit convention and capture scope are defined in the
[Jetson diagnostics method](../wiki/methods/baseline-metrics.md#jetson-cli-feasibility-diagnostics).
The one-second monitor can miss short peaks. No synchronized stage timestamps exist;
the peak cannot be assigned to a stage from these logs. Constant swap occupancy does not
prove zero swap I/O. The temperature alone does not diagnose throttling.

## Observations and interpretation

The log reports Qwen3 execution across 29 segments, and one segment each for diffusion
and VAE. Although `--params-backend disk` was requested, `--eager-load` initially prepared
all parameter buffers on CUDA. Later entries show parameter-buffer release and on-demand
reloads, including release between pipeline stages
([audited log evidence](../results/runs/jetson-flux-klein-003__attempt__20260928-185009/engine-audit.json)).
Thus the successful configuration should not be described as avoiding all-resident
weights throughout startup. The exact physical storage traffic is not measured.

The process exited successfully and saved one image. The imported 512×512 PNG shows a
coherent cat holding a sign with legible “hello world” text and no obvious gross corruption
([inspection and hash](../results/runs/jetson-flux-klein-003__attempt__20260928-185009/image-inspection.json)). This is one visual
smoke check, not a formal quality or repeatability assessment.

## Next experiment

- Instrument repeated runs and a separate stage-labelled Nsight Systems profile of
  `jetson-flux-klein-q4-512-segmented-disk.json` before reporting a baseline.
