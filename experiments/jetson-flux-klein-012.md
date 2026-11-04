---
type: experiment-record
id: jetson-flux-klein-012
status: failed
device: jetson-orin-nano
engine: stable-diffusion-cpp
runs: [jetson-flux-klein-012__baseline__20261015-115718, jetson-flux-klein-012__attempt__20261015-115414]
updated: 2026-10-15
---

# jetson-flux-klein-012: all weights resident on CUDA0 with VAE tiling, quantized 512×512

Status: **failed** (2026-10-15). The two-generation correctness attempt completed. The full
1+3+10 protocol failed in generation index 4 (the first measured generation): the first
generation and all three warm-ups completed, and no measured generation exists. No median is reported.

## Question

Does keeping all weights resident on CUDA0, instead of reloading them from disk per stage, run the
quantized 512×512 workload with VAE tiling, and how does it change stage times and memory
diagnostics? The reference is the disk-backed tiled configuration
[jetson-flux-klein-011](jetson-flux-klein-011.md). Week 3 “verify target feasibility” and RQ1:
“Which diffusion optimization choices transfer across devices and workloads?”
([proposal overview](../wiki/project/overview.md)).

## Setup

- [Configuration](../configs/jetson-flux-klein-q4-512-resident-vaetile.json) and
  [attempt configuration](../configs/jetson-flux-klein-q4-512-resident-vaetile-smoke.json). One change
  against [the 011 configuration](../configs/jetson-flux-klein-q4-512-disk-vaetile.json): parameter
  storage disk → CUDA0 (`optimizations.parameter_storage: "CUDA0"`, `engine_settings.params_backend:
  "CUDA0"`, harness `--params-backend CUDA0`). The run configs differ from 011's only in these fields,
  `id`, `description` and `reference_config`.
- Workload: pinned sd.cpp `19bbbca`; FLUX.2 klein 4B transformer Q4_0, Qwen3-4B Q4_K_M text
  encoder, BF16 VAE (revisions and SHA256 in the config); 512×512, 4 Euler steps, flux2 scheduler,
  CFG 1.0, seed 0, batch 1, prompt “A cat holding a sign that says hello world”.
- Execution: CUDA0 compute, CUDA0 parameters, eager loading, segmented compute allowed, diffusion
  flash attention on, prefetch on, mmap off, conditioning cache 0, step cache off, VAE tiling on
  (128 px tiles, 0.25 target overlap), auto-fit off, 4 CPU threads, 25 W mode.
- Harness: tiling-capable `sd-bench` (`~/tools/sd-bench-build-tiling-20261014`), binary SHA256
  `f9ed5788425f5ce53bf6da86182fdaeba2fa00e7c1c73a2c309f2f63808607f4`, the same binary as 011
  ([environment](../results/runs/jetson-flux-klein-012__baseline__20261015-115718/environment.json)).
  Runner `scripts/run_jetson_sdcpp.py`, which accepts `parameter_storage` `disk` or `CUDA0`.
- Protocol: one model context; full run 1 first + 3 discarded warm-ups + 10 measured; attempt
  1 first + 1 measured. All weight files were SHA256-read before each run (warm filesystem cache, no
  flush; [command](../results/runs/jetson-flux-klein-012__baseline__20261015-115718/engine/command.json)).
- Memory guard (outside the config): kill if MemAvailable < 400 MB or swap growth > 768 MB, polled
  every 0.1 s.
- Metrics: [baseline-metrics.md, stable-diffusion.cpp runs](../wiki/methods/baseline-metrics.md#stable-diffusioncpp-runs).
  Condition time, first-step interval, steady per-step time (`(t_step_end[3] − t_step_end[0]) / 3`,
  i.e. steps 1–3) and decode are the derived quantities defined in
  [jetson-flux-klein-010](jetson-flux-klein-010.md#derived-quantities-used-in-this-record). Because the
  full run wrote no `summary.json`, all of its timings below are computed from the harness callbacks in
  [`engine/results.jsonl`](../results/runs/jetson-flux-klein-012__baseline__20261015-115718/engine/results.jsonl)
  (`wall` = `t_end − t_start`).

## Results

### Full protocol (failed): jetson-flux-klein-012__baseline__20261015-115718

[Run directory](../results/runs/jetson-flux-klein-012__baseline__20261015-115718/),
[status](../results/runs/jetson-flux-klein-012__baseline__20261015-115718/status.json) failed,
15:57:18–16:00:41 UTC, [engine log](../results/runs/jetson-flux-klein-012__baseline__20261015-115718/engine/sdcpp.log),
[memguard](../results/runs/jetson-flux-klein-012__baseline__20261015-115718/memguard.json). Power mode 25 W; CUDA 12.6.

**Error.** Generation index 4 failed during its first transformer call. The engine log reports, in order:

- `model manager memory on CUDA0: reported free 872.71 MB / total 7619.96 MB, tracked weights 4882.75 MB / other runtime 0.00 MB / current runtime 18.01 MB`
- `model manager cannot make enough memory available on CUDA0: need 873.58 MB device / 379.59 MB budget, available 872.71 MB device / unlimited budget`
- `flux segment 7/27 (flux.single_blocks.0) failed during weight preparation`, then
  `diffusion model compute failed`, `Diffusion model sampling failed`,
  `sampling for image 1/1 failed after 0.74s`.

The harness recorded generation 4 as `ok: false` (condition 884.708 ms, failure 724.274 ms after
sampling start, 1624.835 ms in total) and released the 4882.75 MB parameter buffers. The runner then
raised `RuntimeError('Expected one load and exactly the configured number of generations')`; no
`summary.json` was written and `runs.csv`/`stages.csv` contain headers only. The memory guard did not
trigger (process exit code 1).

**Partial timings** (ms; first generation and the three discarded warm-ups; none of these enter a median):

| Generation | wall | condition | first-step interval | steps 1 / 2 / 3 | steady per-step | decode | remainder |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 (first) | 18625.561 | 2199.298 | 3629.236 | 2713.039 / 2709.211 / 2707.000 | 2709.750 | 4629.385 | 38.392 |
| 1 (warm-up) | 16381.493 | 885.506 | 3050.094 | 2717.425 / 2716.178 / 2716.193 | 2716.599 | 4246.000 | 50.097 |
| 2 (warm-up) | 16333.069 | 882.836 | 3001.272 | 2726.841 / 2722.204 / 2720.054 | 2723.033 | 4246.427 | 33.435 |
| 3 (warm-up) | 16277.326 | 871.907 | 2981.845 | 2711.034 / 2710.946 / 2710.708 | 2710.896 | 4249.976 | 40.910 |
| 4 (measured) | failed | 884.708 | – | – | – | – | – |

Remainder = wall − condition − first-step interval − steps 1–3 − decode (latent setup and graph setup
between stages; corresponds to `other_ms`). Model context creation (load event) took 65.045 s; the engine
reports tensor loading 61.61 s (read 59.15 s, copy to backend 1.07 s).

**Engine-printed buffers** ([engine log](../results/runs/jetson-flux-klein-012__baseline__20261015-115718/engine/sdcpp.log)):

- Parameters: `prepared params backend buffers (4882.75 MB, 795 tensors, 5 blocks, VRAM) on CUDA0`
  once at load; `total params memory size = 4882.75MB (VRAM 4882.75MB, RAM 0.00MB)`: text encoders
  2375.91 MB, diffusion model 2346.40 MB, VAE 160.43 MB. No further parameter preparation or release
  is logged until the release after the failure.
- qwen3 compute buffer 65.00 MB (1 segment) and VAE compute buffer 104.03 MB (1 segment) in every generation.
- Flux compute buffer: generation 0 ran every step as one graph (`flux compute buffer size: 329.00 MB(VRAM)
  on CUDA0 (peak across 1 segment)`). In generations 1, 2 and 3 the engine logged `flux using 27 segments`
  for the first transformer call (prelude, 5 double blocks, 20 single blocks, final; each segment
  adds an 18.00–18.01 MB `graph cut cache` entry), reported `flux compute buffer size: 343.58 MB(VRAM) on
  CUDA0 (peak across 27 segments)`, then logged `flux using 1 segment` for the next call and
  `343.58 MB ... (peak across 1 segment)`. Steps 1–3 ran as one graph. Generation 4 again chose 27
  segments and failed at segment 7.
- The engine's last available-memory probe before the first transformer call of each generation
  (`ggml_backend_cuda_get_available_uma_memory`): 1109.34 MB (gen 0), 962.03 MB (gen 1), 884.17 MB (gen 2),
  891.48 MB (gen 3), 874.14 MB (gen 4). The lowest probe in the log is 872.39 MB, in generation 4.

**Tiling:** every decode logs `VAE decode tile size: 128x128 pixels (8x8 latent)`, `num tiles : 5, 5`,
`optimal overlap : 0.250000, 0.250000` and `processing 25 tiles`.

**System memory and output:**

- Memory guard not triggered: MemAvailable 5912 MB at start, minimum 862 MB; swap 307 → 308 MB.
- tegrastats (135 one-second samples, whole capture including load): RAM peak 6593 MB (6.438 GiB) of
  7620 MB, at the 12:00:38 local sample, two seconds before the last sample; swap 307–308 MB; maximum GPU
  temperature 57.687 °C. Not stage-aligned and not an allocator peak.
- The four completed raw outputs (`engine/raw/run-0..3.rgb`) all have SHA256 `252b6cb0…e858`.

### Correctness attempt (complete): jetson-flux-klein-012__attempt__20261015-115414

[Run](../results/runs/jetson-flux-klein-012__attempt__20261015-115414/),
[status](../results/runs/jetson-flux-klein-012__attempt__20261015-115414/status.json) complete,
15:54:14–15:56:47 UTC, [summary](../results/runs/jetson-flux-klein-012__attempt__20261015-115414/summary.json),
[engine log](../results/runs/jetson-flux-klein-012__attempt__20261015-115414/engine/sdcpp.log).

| Generation | wall | condition | first-step interval | steps 1 / 2 / 3 | steady per-step | decode | other |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 (first) | 18999.560 | 2335.874 | 3767.139 | 2706.382 / 2709.879 / 2706.941 | 2707.734 | 4723.190 | 50.155 |
| 1 (measured observation) | 16147.911 | 904.082 | 2833.712 | 2698.400 / 2703.204 / 2702.864 | 2701.489 | 4263.439 | 42.210 |

- Context creation 67.388 s. Engine audit: parameter storage CUDA0, settings confirmed, no auto-fit,
  diffusion flash attention in use, zero conditioning-cache hits, 2 tiled decodes, callbacks validated.
- Same 4882.75 MB CUDA0 parameter preparation as the full run. Both generations ran the transformer as
  one graph: `flux compute buffer size: 329.00 MB ... (peak across 1 segment)`; no `flux using 27
  segments` line appears.
- Memory guard not triggered: MemAvailable 6282 MB at start, minimum 838 MB; swap 305 → 307 MB.
  tegrastats RAM peak 6.409 GiB (103 samples); maximum GPU temperature 53.343 °C.
- Both images have SHA256 `252b6cb0…e858` (`deterministic_output: true`). One measured observation does
  not qualify a speed result.

### Delta vs reference (011), first generation and warm-ups only

The 012 full run has no measured generations, so the comparison uses the generations both full runs
completed: the first generation and warm-ups 1–3 (discarded in both protocols). The 011 values are from
[its callbacks](../results/runs/jetson-flux-klein-011__baseline__20261014-185639/engine/results.jsonl);
011 ran on 2026-10-14, 012 on 2026-10-15, so session conditions differ. Values in seconds.

| Quantity | 011 first | 012 first | 011 warm-ups 1–3 (range) | 012 warm-ups 1–3 (range) |
|---|---:|---:|---:|---:|
| wall | 28.597 | 18.626 | 62.975–72.135 | 16.277–16.381 |
| condition | 2.441 | 2.199 | 24.603–25.399 | 0.872–0.886 |
| first-step interval | 12.740 | 3.629 | 23.926–32.925 | 2.982–3.050 |
| steady per-step | 2.713 | 2.710 | 2.707–2.721 | 2.711–2.723 |
| decode | 4.867 | 4.629 | 5.409–5.751 | 4.246–4.250 |

| Memory diagnostic | 011 full run | 012 full run |
|---|---:|---:|
| Context creation (s) | 66.212 | 65.045 |
| tegrastats RAM peak (GiB) | 5.967 (1000 samples) | 6.438 (135 samples) |
| MemAvailable minimum (MB) | 1269 | 862 |
| Flux compute buffer (MB, engine-printed) | 329.00, 1 segment throughout | 329.00 (gen 0), then 343.58 with 27-segment first calls |
| Output hash | `252b6cb0…e858` | `252b6cb0…e858` |

## Observations

- With resident weights, warm-up wall time is 16.28–16.38 s and the first generation 18.63 s (smoke:
  19.00 s first, 16.15 s measured observation).
- Condition time falls to 0.87–0.89 s and the first-step interval to 2.98–3.05 s in warm-ups; in 011's
  warm-ups these were 24.6–25.4 s and 23.9–32.9 s.
- Steady per-step time (2.71–2.72 s) is the same as in 011.
- Decode is 4.25 s in warm-ups, 1.16–1.50 s below 011's warm-ups.
- The engine prepares the 4882.75 MB of parameters on CUDA0 once and logs no further parameter loading
  before the failure.
- From generation 1 onwards in the full run, the first transformer call of each generation ran as 27
  segments with a 343.58 MB compute buffer; the smoke run's generation 1 ran as one graph with 329.00 MB.
- The engine's available-memory probe before the first transformer call fell from 1109.34 MB (gen 0)
  to 874.14 MB (gen 4). Generation 4 failed with a 0.87 MB shortfall (need 873.58 MB, available 872.71 MB).
- The 27-segment first-call interval (2.98–3.05 s) is 0.15–0.22 s longer than the smoke run's
  one-graph generation-1 first-step interval (2.83 s).
- System MemAvailable minimum is 862 MB (full) and 838 MB (smoke), against 1269 MB in 011; the
  tegrastats RAM peak is 0.47 GiB higher than 011's.
- Output pixels are identical to 011 (same hash): the storage change does not alter numerics.

## Interpretation

- Removing per-stage reloads removes nearly all of 011's condition-time and first-step reload cost and
  about 1.2–1.5 s of decode time per warm-up generation; per-step transformer compute is unchanged. The
  warm-up wall time of about 16.3 s is a lower bound on what this configuration costs once loaded, but no
  measured generation exists, so no median or speed claim is made.
- The configuration leaves under 1 GB of engine-reported free device memory once all 4882.75 MB of
  weights are resident, and that headroom shrank across successive generations until a
  segmented transformer call could not prepare memory for its first single block.
- **Hypothesis A (allocation limit / fragmentation):** memory used outside tracked weights grows
  slightly with each generation (for example compute-buffer growth from 329.00 to 343.58 MB, graph-cut
  cache entries, or allocator fragmentation), so the margin falls below what a segment needs. Not tested;
  no allocator trace or per-allocation record exists.
- **Hypothesis B (segmentation fallback):** the engine switches the first transformer call to 27
  segments when its available-memory probe is low, and the segmented path requests more device memory per
  segment than the one-graph path needs, which turns a small margin into a failure. The smoke
  generation 1 ran as one graph with a probe of 923.60 MB while full-run generation 1 segmented with
  962.03 MB, so the switching rule is not a simple threshold on this probe value. Not tested.
- Neither hypothesis is established by these data; the run shows only that the configuration is
  infeasible for a 14-generation protocol under the recorded memory conditions.

## Next experiment

- Repeat `jetson-flux-klein-q4-512-resident-vaetile.json` with the engine's per-generation available
  memory and segment decision recorded, to test whether the failure point is reproducible.
- Labelled variant of `jetson-flux-klein-q4-512-resident-vaetile.json` with segmented compute disabled,
  to separate Hypothesis B from Hypothesis A.

Both runs are uploaded to W&B (`mfaruqi-purdue-university/on-device-diffusion`):
[full (failed)](https://wandb.ai/mfaruqi-purdue-university/on-device-diffusion/runs/jetson-flux-klein-012__baseline__20261015-115718),
[attempt](https://wandb.ai/mfaruqi-purdue-university/on-device-diffusion/runs/jetson-flux-klein-012__attempt__20261015-115414).
