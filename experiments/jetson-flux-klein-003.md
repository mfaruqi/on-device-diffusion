---
type: experiment-record
id: jetson-flux-klein-003
status: complete
device: jetson-orin-nano
engine: stable-diffusion-cpp
runs: [jetson-flux-klein-003__baseline__20261005-105911, jetson-flux-klein-003__attempt__20261005-104044, jetson-flux-klein-003__profile__20260930-210728, jetson-flux-klein-003__profile__20260930-201956, jetson-flux-klein-003__baseline__20260930-193208, jetson-flux-klein-003__attempt__20260930-192732, jetson-flux-klein-003__profile__20260928-195625, jetson-flux-klein-003__profile__20260928-194910, jetson-flux-klein-003__attempt__20260928-185009, jetson-flux-klein-003__profile__20260928-191418]
updated: 2026-10-05
---

# jetson-flux-klein-003: disk-backed quantized 512×512 baseline

Status: **repeated unprofiled baseline complete**; ten measured generations after first/warm-up phases.
Targeted and full-protocol profiling are also complete; earlier CLI, smoke and profile evidence remain preserved.
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

## Separate CLI profile: capture reported, validation pending

Nsight Systems wrote a report and the application saved an image. Supplied stats confirm
CUDA kernel, memory-operation and API event presence; complete coverage, NVTX coverage and
profiler exit status still require the original artifacts ([profile record](../results/runs/jetson-flux-klein-003__profile__20260928-191418/)).
The supplied command requests CUDA, NVTX and OS-runtime tracing, with CPU sampling and
context-switch tracing disabled. No harness stage markers were added to this CLI.
The recorded engine timings are profiler diagnostics, not baseline results; any change
in elapsed time cannot be attributed to profiling or caching from this single capture.

The [selected stats rows](../results/runs/jetson-flux-klein-003__profile__20260928-191418/nsys-stats-excerpts.json)
show `mul_mat_q` type 2 accounting for 39.3% of summed kernel duration, followed by
`im2col_kernel` at 13.2% and the listed flash-attention specialization at 9.4%.
Host-to-device copies total 4.50 s of GPU event duration; `cudaStreamSynchronize`
accounts for 11.97 s of CPU API duration and `cudaMalloc` for 4.01 s.
These are whole-capture aggregates with different denominators and overlapping intervals;
they cannot be added into generation latency or attributed to stages from these tables.
The selected rows are transcribed evidence; the complete stats file and trace remain on Jetson.

## Harness profile validation failure

The [194910 capture](../results/runs/jetson-flux-klein-003__profile__20260928-194910/)
saved one image but failed callback validation: tensor-loading updates were counted as
sampling progress. Its stage labels and denoise timings are invalid; generation did not
fail. The corrected harness filters progress totals and checks the sampling sequence.
A corrected capture subsequently passed the callback and label-presence checks (below).

## Corrected harness profile

The [195625 capture](../results/runs/jetson-flux-klein-003__profile__20260928-195625/) reports successful generation,
five sampling callbacks (start plus four completions), 37 excluded loading callbacks,
and no conditioning-cache hits. NVTX label-presence checks and imported SQLite timeline review passed ([summary](../results/runs/jetson-flux-klein-003__profile__20260928-195625/summary.json)).
Host callback durations with the profiler attached are preserved in
[timing diagnostics](../results/runs/jetson-flux-klein-003__profile__20260928-195625/timing-diagnostics.json).
These include weight-loading work inside stages and are not repeated baseline measurements.

The imported [trace review](../results/runs/jetson-flux-klein-003__profile__20260928-195625/trace-review.json)
confirms ordered, nested generation stages and no captured GPU events crossing their ends.
Qwen3, diffusion and VAE each report one segment in this capture. Whole-window sampled RAM
peaked at 6.824 GiB; swap occupancy ranged from 0.876 to 1.135 GiB. These are system totals,
not per-stage memory or swap-I/O measurements.

The [denoising breakdown](../results/runs/jetson-flux-klein-003__profile__20260928-195625/profile/denoise_kernels.md)
shows substantial time outside captured GPU activity in the first step; this is not a measured
physical disk-read duration. [Other stages](../results/runs/jetson-flux-klein-003__profile__20260928-195625/profile/other_stages.md)
are recorded separately. The losslessly converted output passes a
[visual smoke check](../results/runs/jetson-flux-klein-003__profile__20260928-195625/image-inspection.json).
The [run README](../results/runs/jetson-flux-klein-003__profile__20260928-195625/README.md)
describes reproduction and artifact provenance. Exported to W&B as a single profile
([`jetson-flux-klein-003 · profile · OK`](https://wandb.ai/mfaruqi-purdue-university/on-device-diffusion/runs/jetson-flux-klein-003__profile__20260928-195625)); no baseline medians. Viewer timing fields retain explicit profile/single-capture labels
([definitions](../wiki/methods/baseline-metrics.md#wb-timing-labels)).

## Repeated-run integration smoke check (2026-09-30)

Run: [`jetson-flux-klein-003__attempt__20260930-192732`](../results/runs/jetson-flux-klein-003__attempt__20260930-192732/). The [status](../results/runs/jetson-flux-klein-003__attempt__20260930-192732/status.json) is complete.
The [config](../results/runs/jetson-flux-klein-003__attempt__20260930-192732/config.json) retains Jetson 003's pinned Q4_0 transformer,
Q4_K_M text encoder, VAE, 512×512 four-step workload and disk-backed parameter storage.
Protocol: one first, zero warm-ups, one measured generation, in one context and without Nsight.
The [environment](../results/runs/jetson-flux-klein-003__attempt__20260930-192732/environment.json) records 25 W mode, CUDA 12.6, pinned engine commit,
a clean reported engine working tree and an active Xorg/GNOME session. Model files were
SHA256-read before capture, warming the filesystem cache; this is not a cold-disk test.

The [import review](../results/runs/jetson-flux-klein-003__attempt__20260930-192732/smoke-review.json) reproduces both generation rows, all twelve
stage rows and the summary from raw callbacks, replays the engine audit and verifies image hashes.

| Host callback duration (s) | First generation | Single measured generation |
|---|---|---|
| Generation | 25.897 | 59.929 |
| Text encoding | 2.085 | 23.246 |
| Denoising step 0 | 11.578 | 23.737 |
| Denoising step 1 | 2.730 | 2.700 |
| Denoising step 2 | 2.729 | 2.700 |
| Denoising step 3 | 2.729 | 2.700 |
| VAE decode | 3.535 | 4.568 |

Source: [summary](../results/runs/jetson-flux-klein-003__attempt__20260930-192732/summary.json), [callback records](../results/runs/jetson-flux-klein-003__attempt__20260930-192732/engine/results.jsonl).
Model-context loading took 61.927 s. There is only one measured sample;
its zero standard deviation is not evidence of stable performance.

**Observation:** the second generation is slower, with about 98% of the increase in text
encoding and denoising step 0. Steps 1–3 remain close to 2.7 s. The engine reports parameter
buffer releases and subsequent tensor-loading intervals: 21.87 s during the second text
encoding and 20.19 s during its first denoising step. These are engine-reported loading
intervals, not independent physical-storage measurements; source lines and input hash are
preserved in the [review](../results/runs/jetson-flux-klein-003__attempt__20260930-192732/smoke-review.json).

**Interpretation:** repeated disk-backed generation includes substantial component reload
costs. A first generation is not necessarily the slowest under this residency policy.
Filesystem-cache eviction or storage latency may contribute, but this capture does not
measure physical I/O or isolate their causes. Two generations do not establish steady state.

Whole-capture sampled system RAM peaked at 6.227 GiB;
swap occupancy ranged from 0.000 to 0.114 GiB,
and maximum sampled GPU temperature was 51.062 °C.
These 148 samples cover loading and both generations, not separate
stage peaks ([samples](../results/runs/jetson-flux-klein-003__attempt__20260930-192732/tegrastats-samples.csv),
[definitions](../wiki/methods/baseline-metrics.md#jetson-repeated-baseline-memory)).
Swap occupancy does not quantify swap I/O; temperature alone does not establish throttling.

The first and measured PNGs decode to identical RGB bytes matching both recorded hashes.
Visual inspection shows a cat holding a legible “hello world” sign. This is a visual smoke
check, not a quality evaluation. `deterministic_output=true` alone is insufficient here
because the summary computes that flag over the single measured image; the independent
first-versus-measured hash check is recorded in the [review](../results/runs/jetson-flux-klein-003__attempt__20260930-192732/smoke-review.json).

## Repeated unprofiled baseline (2026-09-30)

Run: [`jetson-flux-klein-003__baseline__20260930-193208`](../results/runs/jetson-flux-klein-003__baseline__20260930-193208/); [W&B baseline](https://wandb.ai/mfaruqi-purdue-university/on-device-diffusion/runs/jetson-flux-klein-003__baseline__20260930-193208).
[Status](../results/runs/jetson-flux-klein-003__baseline__20260930-193208/status.json): complete. The [config](../results/runs/jetson-flux-klein-003__baseline__20260930-193208/config.json) preserves the
pinned component files, 512×512 four-step quantized disk-backed workload and 25 W power requirement.
The plain harness runs once, with one first generation, three discarded warm-ups and ten measured
generations. No profiler is attached. [Environment](../results/runs/jetson-flux-klein-003__baseline__20260930-193208/environment.json) records the engine
commit/binary hash, CUDA linkage and desktop processes. SHA256 checks read all weights before
loading, so the load result is not a cold-storage measurement.

The [import review](../results/runs/jetson-flux-klein-003__baseline__20260930-193208/baseline-review.json) verifies all 14 generation and 84 stage
rows against raw callbacks, replays the summary and engine audit, and checks retained image pixels
against recorded hashes. The W&B readback matched all 14 generation rows and expected config/summary
fields; the run is tagged as an unprofiled measured median with baseline comparison eligibility.
Eligibility does not establish matched workload, precision or quality across other configurations.

| Host callback duration (s) | Measured median | Measured min–max |
|---|---|---|
| Generation | 66.117 | 54.830–69.043 |
| Text encoding | 23.109 | 14.616–24.468 |
| Denoising total | 38.773 | 31.373–40.613 |
| Denoising step 0 | 30.622 | 23.245–32.487 |
| Denoising step 1 | 2.714 | 2.703–2.722 |
| Denoising step 2 | 2.708 | 2.702–2.719 |
| Denoising step 3 | 2.716 | 2.706–2.726 |
| VAE decode | 4.547 | 3.610–4.645 |

Source: [summary](../results/runs/jetson-flux-klein-003__baseline__20260930-193208/summary.json). Generation standard deviation is
4.888 s; initial context load is 58.069 s and the
first generation is 25.785 s. Stage medians are computed independently
and need not sum exactly to the generation median. Warm-ups are excluded from all measured statistics.

### Observation: repeated component loading dominates the slow generations

Every measured generation contains three engine-reported tensor-loading intervals attributed to
text encoding, denoising step 0 and VAE decode by containment in callback boundaries. Summed per
generation, their median is 49.200 s, with a
38.260–51.880 s range.
Subtracting these rounded intervals from each generation leaves
16.570–17.175 s; this residual is not pure GPU compute time.
Source lines, engine-reported read durations and log hashes are retained in the
[review](../results/runs/jetson-flux-klein-003__baseline__20260930-193208/baseline-review.json). These diagnostics do not measure physical SD-card I/O.

Later denoising steps remain near 2.7 s while text encoding and step 0 account for much of the
latency and variation. The fast first generation is not representative of the repeated protocol.
**Interpretation:** repeated component loading is a concrete target for residency/cache analysis.
Which costs come from physical storage, filesystem cache behavior, transfers or allocation remains
unresolved. A single initial-generation profile cannot explain all later-generation costs.

### Memory, repeatability and quality scope

The whole-capture RAM peak is 6.078 GiB across
882 one-second samples; swap occupancy ranges from
0.014 to 0.906 GiB.
Maximum sampled GPU temperature is 53.375 °C
([summary](../results/runs/jetson-flux-klein-003__baseline__20260930-193208/summary.json), [samples](../results/runs/jetson-flux-klein-003__baseline__20260930-193208/tegrastats-samples.csv)). These are system-wide
capture diagnostics, not generation/stage allocation peaks; occupancy does not quantify swap I/O
and temperature does not prove or exclude throttling.

All 14 recorded RGB hashes match. The retained first and first-measured PNGs independently verify
against their hashes; other raw images were removed by the runner after hashing. Visual inspection
of the retained measured image shows a recognizable cat with a legible “hello world” sign.
This is a repeatability/visual smoke check, not held-out quality evaluation.

## Targeted later-generation profile (2026-09-30)

Run: [`jetson-flux-klein-003__profile__20260930-201956`](../results/runs/jetson-flux-klein-003__profile__20260930-201956/),
[status](../results/runs/jetson-flux-klein-003__profile__20260930-201956/status.json), [config](../results/runs/jetson-flux-klein-003__profile__20260930-201956/config.json).
The pinned model/workload/engine/execution settings match the repeated baseline. Only the
measurement protocol changes: first + three warm-ups + one traced generation in one context.
Nsight Systems 2024.5.4 records generation index 4; earlier generations and initial loading
are untraced but run with the profiler attached. This is a single diagnostic observation,
not a ten-sample baseline or evidence of a speedup. Supports Week 1/RQ1.

The [import review](../results/runs/jetson-flux-klein-003__profile__20260930-201956/profile-review.json) replays all five generation rows and 30 stage
rows from raw callbacks, reproduces the summary/audit, verifies both retained PNG pixel hashes,
and checks the SQLite trace. Exactly one complete generation and all six stage ranges are
present, with 5574 CUDA kernels; no initial `load` range is captured.
The [capture receipt](../results/runs/jetson-flux-klein-003__profile__20260930-201956/profile/capture.json) and raw per-generation flags select only index 4.

| Host callback duration | Single profiled value (s) |
|---|---|
| Generation | 60.916 |
| Text encoding | 18.910 |
| Denoising total | 37.087 |
| Denoising step 0 | 28.818 |
| Denoising step 1 | 2.756 |
| Denoising step 2 | 2.758 |
| Denoising step 3 | 2.756 |
| VAE decode | 4.637 |

Source: [summary](../results/runs/jetson-flux-klein-003__profile__20260930-201956/summary.json). Context load is 63.037 s,
reported separately; file hash checks warmed filesystem cache before loading. The one-sample
min, max and median coincide and stdev is zero by construction, not evidence of timing stability.

### Timeline and loading observations

| Stage | NVTX span (s) | Captured GPU activity union (s) | Uncovered span (s) |
|---|---|---|---|
| `text_encode` | 18.901 | 2.199 | 16.701 |
| `vae_decode` | 4.636 | 2.980 | 1.656 |
| `denoise_step_0` | 28.809 | 4.631 | 24.178 |
| `denoise_step_1` | 2.755 | 2.729 | 0.026 |
| `denoise_step_2` | 2.757 | 2.730 | 0.027 |
| `denoise_step_3` | 2.755 | 2.729 | 0.026 |

Sources: [denoising analysis](../results/runs/jetson-flux-klein-003__profile__20260930-201956/profile/denoise_kernels.json),
[text/VAE analysis](../results/runs/jetson-flux-klein-003__profile__20260930-201956/profile/other_stages.json).
GPU activity union includes captured kernels, memory copies and memory sets; uncovered span
is not proof of device-wide idleness or physical disk I/O. NVTX and host callback clocks differ
slightly; retain their separately reported values rather than mixing boundaries.

The engine reports three tensor-loading intervals in the captured generation:
text encoding 17.38 s (read 16.57 s), denoising step 0 25.16 s (read 24.23 s), and
VAE decode 1.20 s (read 1.04 s). Their sum is 43.74 s.
These are rounded engine-defined durations with source lines preserved in the
[review](../results/runs/jetson-flux-klein-003__profile__20260930-201956/profile-review.json); they overlap stage time and must not be added to it.
The engine's `read` label does not distinguish filesystem-cache access from physical storage.

**Interpretation:** step 0's long latency is primarily outside captured GPU activity;
later denoising steps are almost continuously busy. Loading diagnostics align with the long
text-encoding and step-0 intervals. This supports investigating component reload/residency
costs before treating step 0 as a much more expensive transformer computation. Attribution
between storage, page cache, CPU work and synchronization remains unresolved.

### Memory, repeatability and viewer labels

Whole-capture RAM peaks at 6.087 GiB across
346 one-second samples; swap occupancy remains
0.310 GiB and maximum sampled GPU temperature is
52.406 °C
([summary](../results/runs/jetson-flux-klein-003__profile__20260930-201956/summary.json)). These describe the whole five-generation capture,
not the profiled generation alone; constant swap occupancy does not exclude swap I/O.
All five recorded RGB hashes match and both retained PNGs verify against their hashes.
No held-out quality evaluation is performed.

W&B export uses `profiler=nsight-systems`, `measurement_scope=profiled`,
`measurement_basis=single_profile_callback`, and `baseline_comparison_eligible=false`.
Five `gen/*` history rows and one median-only chart row preserve their separate meanings;
profile stage activity/gap metrics and kernel-group tables are included.
[W&B run](https://wandb.ai/mfaruqi-purdue-university/on-device-diffusion/runs/jetson-flux-klein-003__profile__20260930-201956); post-upload readback verified five generation rows, one median chart row,
all expected config/summary values, finished state and uploaded artifacts
([readback receipt](../results/runs/jetson-flux-klein-003__profile__20260930-201956/wandb-export-verification.json)).

## Full-protocol profile (2026-09-30)

Run: [`jetson-flux-klein-003__profile__20260930-210728`](../results/runs/jetson-flux-klein-003__profile__20260930-210728/),
[status](../results/runs/jetson-flux-klein-003__profile__20260930-210728/status.json), [config](../results/runs/jetson-flux-klein-003__profile__20260930-210728/config.json).
The full-profile config preserves the repeated baseline's weights, workload, execution settings
and first + three warm-ups + ten measured protocol. Nsight Systems traces initial loading and
all 14 generations in one model context. This validates the full-protocol mode for Week 1/RQ1;
its timings include profiling conditions and are not a clean-baseline replacement.

The [import review](../results/runs/jetson-flux-klein-003__profile__20260930-210728/profile-review.json) verifies 14 CSV generation rows and 84 stage rows
against raw callbacks, replays the summary and engine audit, and checks both retained PNG pixel
hashes. The SQLite trace contains one initial-load range, 14 generation ranges and 84 stage ranges;
every generation contains its six stages and CUDA kernels. There are 78,036 captured
CUDA kernels across the report. The [capture receipt](../results/runs/jetson-flux-klein-003__profile__20260930-210728/profile/capture.json) identifies
traced indices 0–13 and measured indices 4–13.

| Host callback duration (s) | Measured median | Measured min–max |
|---|---|---|
| Generation | 59.319 | 52.521–66.051 |
| Text encoding | 20.203 | 11.455–23.814 |
| Denoising total | 37.629 | 28.291–40.136 |
| Denoising step 0 | 29.343 | 19.987–31.808 |
| Denoising step 1 | 2.765 | 2.751–2.776 |
| Denoising step 2 | 2.766 | 2.757–2.776 |
| Denoising step 3 | 2.765 | 2.759–2.777 |
| VAE decode | 4.654 | 3.967–4.874 |

Source: [summary](../results/runs/jetson-flux-klein-003__profile__20260930-210728/summary.json). There are ten measured observations; generation
standard deviation is 4.387 s. First generation is
26.076 s and context load is 62.774 s, both
separate from the measured median. Stage medians are independent and need not sum to the
end-to-end median. Initial loading is profiled and preceded by model-file hash reads.

Every measured generation has three engine-reported component-loading intervals; their summed
per-generation median is 42.295 s, range
35.500–48.620 s.
Source lines and interval containment are preserved in the [review](../results/runs/jetson-flux-klein-003__profile__20260930-210728/profile-review.json).
These rounded loading intervals overlap the stage durations and do not establish physical
storage I/O. Later denoising steps stay near 2.76 s while text encoding and step 0 vary more.
Interpretation: component loading remains a relevant latency source; the trace does not alone
separate storage/cache, host work and synchronization costs.

[Kernel breakdown](../results/runs/jetson-flux-klein-003__profile__20260930-210728/profile/denoise_kernels.json) and
[text/VAE breakdown](../results/runs/jetson-flux-klein-003__profile__20260930-210728/profile/other_stages.json) analyze **generation 13 only**, the last
measured generation. Their `profile/*` activity metrics are single-generation diagnostics,
not medians across ten traces. All 14 generations remain available in the preserved trace.

Whole-capture RAM peaks at 6.114 GiB across
878 one-second samples; swap occupancy is
0.310–0.372 GiB and maximum
sampled GPU temperature is 54.031 °C
([summary](../results/runs/jetson-flux-klein-003__profile__20260930-210728/summary.json)). These are whole-system capture diagnostics, not per-stage
allocation peaks or swap-I/O measurements. All 14 recorded RGB hashes match; both retained
PNGs independently verify against their hashes. No held-out quality evaluation is performed.

W&B uses `profiler=nsight-systems`, `profiling.scope=full-process`,
`measurement_scope=profiled`, `measurement_basis=measured_median`, and
`baseline_comparison_eligible=false`. All 14 generation history/table rows are exported;
medians are Summary-only, without an additional median history row.

[W&B run](https://wandb.ai/mfaruqi-purdue-university/on-device-diffusion/runs/jetson-flux-klein-003__profile__20260930-210728)
readback verifies finished status, all 14 generation rows, expected config/summary fields,
and zero median-only history rows ([export receipt](../results/runs/jetson-flux-klein-003__profile__20260930-210728/wandb-export-verification.json)).

## Next experiment

- Measure storage/page-cache and process-I/O behavior for the same disk-backed workload before selecting a residency or prefetch variant.

## Process-restricted read-call analysis (2026-10-05)

The saved captures were reanalyzed with the [read/GPU coverage method](../wiki/methods/baseline-metrics.md#captured-read-call--gpu-coverage). These are stage diagnostics from one selected generation in each capture, not baseline medians. The original traces and previous reports remain unchanged.

| Capture / stage | Stage span (ms) | GPU-covered (ms) | Read-only-covered (ms) | Neither-covered (ms) |
|---|---:|---:|---:|---:|
| [Single startup capture](../results/runs/jetson-flux-klein-003__profile__20260928-195625/profile/osrt-analysis/stage_kernels_osrt.json) / text_encode | 2714.991 | 1004.462 | 0.009 | 1710.520 |
| [Single startup capture](../results/runs/jetson-flux-klein-003__profile__20260928-195625/profile/osrt-analysis/stage_kernels_osrt.json) / denoise_step_0 | 11808.091 | 3372.812 | 7523.418 | 911.861 |
| [Single startup capture](../results/runs/jetson-flux-klein-003__profile__20260928-195625/profile/osrt-analysis/stage_kernels_osrt.json) / denoise_step_1 | 2752.614 | 2725.482 | 0.000 | 27.132 |
| [Single startup capture](../results/runs/jetson-flux-klein-003__profile__20260928-195625/profile/osrt-analysis/stage_kernels_osrt.json) / vae_decode | 4003.939 | 2905.442 | 223.018 | 875.479 |
| [Targeted generation 4](../results/runs/jetson-flux-klein-003__profile__20260930-201956/profile/osrt-analysis/stage_kernels_osrt.json) / text_encode | 18900.647 | 2199.481 | 16023.904 | 677.262 |
| [Targeted generation 4](../results/runs/jetson-flux-klein-003__profile__20260930-201956/profile/osrt-analysis/stage_kernels_osrt.json) / denoise_step_0 | 28808.888 | 4631.078 | 23368.073 | 809.736 |
| [Targeted generation 4](../results/runs/jetson-flux-klein-003__profile__20260930-201956/profile/osrt-analysis/stage_kernels_osrt.json) / denoise_step_1 | 2755.129 | 2728.795 | 0.000 | 26.334 |
| [Targeted generation 4](../results/runs/jetson-flux-klein-003__profile__20260930-201956/profile/osrt-analysis/stage_kernels_osrt.json) / vae_decode | 4636.104 | 2980.390 | 1146.364 | 509.350 |
| [Full capture, generation 13](../results/runs/jetson-flux-klein-003__profile__20260930-210728/profile/osrt-analysis/stage_kernels_osrt.json) / text_encode | 19253.964 | 1917.075 | 16887.875 | 449.015 |
| [Full capture, generation 13](../results/runs/jetson-flux-klein-003__profile__20260930-210728/profile/osrt-analysis/stage_kernels_osrt.json) / denoise_step_0 | 19987.943 | 4187.652 | 15100.911 | 699.379 |
| [Full capture, generation 13](../results/runs/jetson-flux-klein-003__profile__20260930-210728/profile/osrt-analysis/stage_kernels_osrt.json) / denoise_step_1 | 2769.046 | 2741.008 | 0.000 | 28.038 |
| [Full capture, generation 13](../results/runs/jetson-flux-klein-003__profile__20260930-210728/profile/osrt-analysis/stage_kernels_osrt.json) / vae_decode | 4687.640 | 2994.339 | 991.160 | 702.141 |

Read-only coverage is substantial in text encoding and the first denoising step of the later-generation captures. Subsequent denoising steps have almost complete captured GPU coverage. This is consistent with residency/loading contributing to stage latency, but read-call intervals do not establish physical storage wait or bandwidth. The remaining uncovered time can include host work, synchronization and uncaptured short calls.

The earlier CLI capture has no NVTX stage markers; stage analysis is therefore unavailable ([analysis status](../results/runs/jetson-flux-klein-003__profile__20260928-191418/profile/osrt-analysis/analysis-status.json)). No stage values were inferred for it.

The derived coverage table and reports are available in the separate [W&B analysis run](https://wandb.ai/mfaruqi-purdue-university/on-device-diffusion/runs/jetson-osrt-analysis-20261005).

## Two-generation correctness attempt: jetson-flux-klein-003__attempt__20261005-104044

[Config](../results/runs/jetson-flux-klein-003__attempt__20261005-104044/config.json), [summary](../results/runs/jetson-flux-klein-003__attempt__20261005-104044/summary.json), [status](../results/runs/jetson-flux-klein-003__attempt__20261005-104044/status.json), [environment](../results/runs/jetson-flux-klein-003__attempt__20261005-104044/environment.json), [validation](../results/runs/jetson-flux-klein-003__attempt__20261005-104044/validation.json).

Pinned sd.cpp19bbbca; Jetson Orin Nano25W; distilled klein transformer Q4_0, Qwen3 Q4_K_M and BF16 VAE;512×512,4Euler/flux2steps,CFG1,seed0,batch1. Fixed prompt: “A cat holding a sign that says hello world”. One context, first generation then one measured observation; no discarded warm-up. The single option is **no-cache control on the cache-capable harness**. Engine snapshot31f3be71d07adda2fb122de0ef75a056c27ec7b1; four CPU threads, eager disk-backed segmented CUDA execution. Requested settings and engine log audit are preserved in the summary; they do not establish continuous residency or physical I/O.

[Metric definitions](../wiki/methods/baseline-metrics.md): host callback times include on-demand loading; no CUDA-event span or allocator peak is available.

| Metric | First, ms | Second observation, ms |
|---|---:|---:|
| wall_ms | 25441.067 | 60099.405 |
| text_encode_ms | 1945.158 | 22875.571 |
| denoise_ms | 19564.932 | 32301.220 |
| vae_decode_ms | 3571.255 | 4649.695 |
| other_ms | 359.722 | 272.919 |

Context creation 62.319213s after hash checks warmed the file cache. Whole-capture sampled RAM peak 6.005859GiB, swap 0.316406–0.319336GiB, 148samples; includes model loading and both generations.

Callback, settings, phase, aggregate and image-hash validation passed. Conditioning hits per generation: [0, 0]; actual transformer passes: [4, 4]. Both saved images are512×512RGB. Reference image equality is recorded in validation; no formal quality eligibility or held-out evaluation.

Interpretation: functional validation passed; one measured observation cannot qualify an option under the full-protocol speed-range rule.

Next experiment: unchanged full1+3+10protocol using `configs/jetson-flux-klein-stage-profile.json`.

## Cache-capable harness full control (2026-10-05)

[Run summary](../results/runs/jetson-flux-klein-003__baseline__20261005-105911/summary.json), [config](../results/runs/jetson-flux-klein-003__baseline__20261005-105911/config.json), [environment](../results/runs/jetson-flux-klein-003__baseline__20261005-105911/environment.json), [status](../results/runs/jetson-flux-klein-003__baseline__20261005-105911/status.json). Completed14:59:11–15:14:28UTC on Jetson Orin Nano25W. Snapshot31f3be71d07adda2fb122de0ef75a056c27ec7b1; pinned sd.cpp19bbbca; binary SHA256 `2d66fa04053c02b441d61a84d24412537a16f9099d6cb31b80dcca3c8cc1f896`.

This is the no-cache reference for the current four-step option tests: distilled klein Q4_0 transformer,Qwen3 Q4_K_M,BF16 VAE,512²,4Euler/flux2steps,CFG1,seed0,batch1,fixed development prompt. One context; first generation separate,three warm-ups discarded,ten measured generations. Eager disk-backed segmented CUDA execution,four CPU threads,prefetch enabled,mmap disabled,conditioning cache0,step cache off. Hash checks warm the file cache; context creation is not a cold-storage measurement. [Metric definitions](../wiki/methods/baseline-metrics.md).

| Metric | Median, ms | Min–max, ms |
|---|---:|---:|
| wall_ms | 60935.979 | 46863.060–70348.506 |
| text_encode_ms | 22139.018 | 12379.430–24546.671 |
| denoise_ms | 36899.651 | 30073.790–41110.784 |
| vae_decode_ms | 4563.792 | 3610.729–4658.949 |
| other_ms | 289.341 | 264.250–311.416 |

First generation25722.318ms; context creation58.405591s. Whole-capture system RAM peak5.997070GiB,swap0.327148–0.329102GiB,873samples; includes loading and all phases,not a per-stage allocation peak.

[Independent validation](../results/runs/jetson-flux-klein-003__baseline__20261005-105911/validation.json) passed:1+3+10phases,four transformer passes per generation,zero conditioning hits,no approximate caching,all measured summary aggregates and saved PNG hashes checked. All14reported RGB hashes agree; first and first-measured PNGs are retained,other raw images were hashed before deletion by the runner. Saved measured pixels match the same-binary control attempt. [Paired diagnostic](../results/runs/jetson-flux-klein-003__baseline__20261005-105911/quality-diagnostics.json) records MSE0,infinite PSNR and LPIPS0; one development prompt/seed,not formal quality eligibility.

Observation: measured wall time ranges from46863.060to70348.506ms. This spread must be retained in later option comparisons. Interpretation: this is a validated reference distribution for the current harness; timing spread alone does not identify storage,thermal or memory-pressure causes.

Next experiment: finish the unchanged four-step EasyCache full protocol.
