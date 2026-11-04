---
type: experiment-record
id: jetson-flux-klein-010
status: complete
device: jetson-orin-nano
engine: stable-diffusion-cpp
runs: [jetson-flux-klein-010__baseline__20261014-183501, jetson-flux-klein-010__attempt__20261014-183050, jetson-flux-klein-010__attempt__20261014-165043]
updated: 2026-10-14
---

# jetson-flux-klein-010: diffusion flash attention off, disk-backed quantized 512×512

Status: **complete** (2026-10-14); full unprofiled 1+3+10 protocol and a two-generation
correctness attempt completed. An earlier attempt failed after generation because of a config error.

## Question

How does turning off flash attention in the diffusion model change stage latency, output
and memory diagnostics for the disk-backed quantized 512×512 workload of
[jetson-flux-klein-003](jetson-flux-klein-003.md)? Week 3 “verify target feasibility” and RQ1:
“Which diffusion optimization choices transfer across devices and workloads?”
([proposal overview](../wiki/project/overview.md)). The reference is the same-session 003 repeat
[`jetson-flux-klein-003__repeat__20261014-163411`](../results/runs/jetson-flux-klein-003__repeat__20261014-163411/),
run with the same harness binary.

## Setup

- [Configuration](../configs/jetson-flux-klein-q4-512-disk-noflash.json) and
  [attempt configuration](../configs/jetson-flux-klein-q4-512-disk-noflash-smoke.json). One change
  against [the 003 reference](../configs/jetson-flux-klein-q4-512-disk-baseline.json):
  `engine_settings.diffusion_fa: false` (harness `--diffusion-fa 0`), named in the `optimizations`
  block as `diffusion_flash_attention: false`. Text-encoder flash attention (`--fa 0`) is off in both.
- Workload: pinned sd.cpp `19bbbca`; FLUX.2 klein 4B transformer Q4_0, Qwen3-4B Q4_K_M text
  encoder, BF16 VAE (revisions and SHA256 in the config); 512×512, 4 Euler steps, flux2 scheduler,
  CFG 1.0, seed 0, batch 1, prompt “A cat holding a sign that says hello world”.
- Execution: CUDA0 compute, disk-backed parameters, eager loading, segmented compute allowed,
  prefetch on, mmap off, conditioning cache 0, step cache off, VAE tiling off, 4 CPU threads, 25 W mode.
- Harness: `sd-bench` built from `engines/sdcpp/bench.cpp` of this branch (adds optional
  `--vae-tiling`, inactive here). Binary SHA256 `f9ed5788425f5ce53bf6da86182fdaeba2fa00e7c1c73a2c309f2f63808607f4`,
  identical to the reference run ([environment](../results/runs/jetson-flux-klein-010__baseline__20261014-183501/environment.json)).
- Protocol: one model context; baseline 1 first + 3 discarded warm-ups + 10 measured; attempt
  1 first + 1 measured. All weight files were SHA256-read before each run, so the filesystem cache
  was warm; no cache flush ([command](../results/runs/jetson-flux-klein-010__baseline__20261014-183501/engine/command.json)).
- Memory guard (outside the config): the run is killed if MemAvailable falls below 400 MB or swap
  use grows by more than 768 MB, polled every 0.1 s ([memguard](../results/runs/jetson-flux-klein-010__baseline__20261014-183501/memguard.json)).
- Metrics: [baseline-metrics.md, stable-diffusion.cpp runs](../wiki/methods/baseline-metrics.md#stable-diffusioncpp-runs).
  Host callback times include on-demand weight loading. Derived quantities are defined below.

### Derived quantities used in this record

Computed from the harness callback timestamps in `engine/results.jsonl` for measured generations
(run indices 4–13 for the baseline), then the median and min–max over generations:

| Name | Per-generation definition | Relation to standard metrics |
|---|---|---|
| condition time | `t_cond − t_start` | equals `text_encode_ms`; includes text-encoder weight reload |
| first-step interval | `t_step_end[0] − t_sampling_start` | equals `denoise_step_0_ms`; includes transformer weight reload |
| steady per-step time | `(t_step_end[3] − t_step_end[0]) / 3` | mean of `denoise_step_1..3_ms`; no engine loading message falls inside this window in any generation of the three 2026-10-14 full runs |
| decode | `t_decode_end − t_sampling_end` | equals `vae_decode_ms` |

The steady per-step time is the derived quantity that separates per-step transformer compute
from weight reloading in this disk-backed configuration. A definition is proposed for
`wiki/methods/baseline-metrics.md`; it is not yet part of that file.

## Results

### Full protocol: jetson-flux-klein-010__baseline__20261014-183501

[Run directory](../results/runs/jetson-flux-klein-010__baseline__20261014-183501/),
[status](../results/runs/jetson-flux-klein-010__baseline__20261014-183501/status.json) complete,
22:35:01–22:52:05 UTC, [summary](../results/runs/jetson-flux-klein-010__baseline__20261014-183501/summary.json),
[config](../results/runs/jetson-flux-klein-010__baseline__20261014-183501/config.json),
[callbacks](../results/runs/jetson-flux-klein-010__baseline__20261014-183501/engine/results.jsonl),
[engine log](../results/runs/jetson-flux-klein-010__baseline__20261014-183501/engine/sdcpp.log). Power mode 25 W;
CUDA 12.6; sd.cpp `19bbbca`.

| Metric | Median (ms) | Min–max (ms) |
|---|---:|---:|
| wall_ms | 69739.370 | 61022.373–74612.769 |
| text_encode_ms | 24480.497 | 21657.043–26336.946 |
| denoise_ms | 40811.902 | 31834.426–44622.165 |
| denoise_step_0_ms | 30215.140 | 21227.589–34013.625 |
| denoise_step_1_ms | 3532.143 | 3528.744–3533.633 |
| denoise_step_2_ms | 3532.697 | 3527.837–3543.852 |
| denoise_step_3_ms | 3542.552 | 3530.079–3545.223 |
| vae_decode_ms | 4797.034 | 4172.581–5110.941 |
| other_ms | 287.933 | 267.137–303.714 |

Wall standard deviation 4246.699 ms. First generation (separate): 30288.331 ms, of which
text_encode 2261.626 ms, denoise_step_0 13388.099 ms, vae_decode 3701.740 ms. Model-context
creation 65.988 s ([load](../results/runs/jetson-flux-klein-010__baseline__20261014-183501/load.json)); not a cold-storage
measurement. Stage medians are independent and need not sum to the wall median.

Derived quantities (s, median and min–max over the 10 measured generations):

| Quantity | Median | Min–max |
|---|---:|---:|
| condition time | 24.480 | 21.657–26.337 |
| first-step interval | 30.215 | 21.228–34.014 |
| steady per-step time | 3.535 | 3.531–3.541 |
| decode | 4.797 | 4.173–5.111 |

Engine-reported tensor-loading intervals (`loading tensors completed, taking …` in `sdcpp.log`)
inside each measured generation sum to a median of 49.57 s (40.83–53.97 s); 22.98 s median falls
inside the condition window and 25.78 s inside the first-step interval. These are rounded engine
diagnostics that overlap stage time; they do not measure physical storage I/O.

### Memory diagnostics

- Whole-capture system RAM peak 5.990 GiB of 7.441 GiB (968 one-second tegrastats samples);
  swap occupancy 0.314–0.327 GiB; maximum sampled GPU temperature 54.218 °C
  ([summary](../results/runs/jetson-flux-klein-010__baseline__20261014-183501/summary.json)). These cover loading and all
  14 generations; they are not per-stage or allocator peaks, and occupancy does not quantify swap I/O.
- Memory guard: not triggered. MemAvailable minimum 1268 MB; swap use 326 → 334 MB
  ([memguard](../results/runs/jetson-flux-klein-010__baseline__20261014-183501/memguard.json)).
- Engine-printed compute buffers: flux 491.00 MB (≈0.479 GiB), qwen3 65.00 MB, vae 1664.50 MB
  (≈1.625 GiB), each on CUDA0, once per generation ([engine log](../results/runs/jetson-flux-klein-010__baseline__20261014-183501/engine/sdcpp.log);
  printed MB treated as MiB).

### Repeatability and output

- Engine audit: `diffusion_flash_attn: false` in the context settings; the line “Using flash attention
  in the diffusion model” is absent. Settings confirmed, no auto-fit, zero conditioning-cache hits,
  no tiled decodes, callbacks validated. Each component executes as one graph segment per call.
- All 14 generations have RGB hash `275c50ae…40b6`. This differs from the reference hash
  `2d487c94…517e`: turning flash attention off changes output pixels. No paired image diagnostic
  (MSE/PSNR/LPIPS) and no visual inspection were recorded for this run; no quality claim is made.

### Delta vs reference (same-session repeat)

Reference: [`jetson-flux-klein-003__repeat__20261014-163411`](../results/runs/jetson-flux-klein-003__repeat__20261014-163411/summary.json)
(same binary, 20:34–20:50 UTC). This run started 1 h 45 min after the reference ended, after other
workloads had run on the device.

| Metric (median) | Reference (ms) | FA off (ms) | Δ (ms) | Δ % |
|---|---:|---:|---:|---:|
| wall_ms | 60640.928 | 69739.370 | +9098.442 | +15.0 |
| text_encode_ms | 22563.202 | 24480.497 | +1917.294 | +8.5 |
| denoise_ms | 36009.000 | 40811.902 | +4802.902 | +13.3 |
| denoise_step_0_ms | 27890.725 | 30215.140 | +2324.415 | +8.3 |
| denoise_step_1_ms | 2710.426 | 3532.143 | +821.717 | +30.3 |
| denoise_step_2_ms | 2713.217 | 3532.697 | +819.481 | +30.2 |
| denoise_step_3_ms | 2716.341 | 3542.552 | +826.211 | +30.4 |
| vae_decode_ms | 4638.280 | 4797.034 | +158.755 | +3.4 |
| other_ms | 281.891 | 287.933 | +6.043 | +2.1 |

| Derived (median, s) | Reference | FA off | Δ |
|---|---:|---:|---:|
| steady per-step time | 2.713 (2.701–2.725) | 3.535 (3.531–3.541) | +0.822 (+30.3 %) |
| decode | 4.638 (3.833–4.862) | 4.797 (4.173–5.111) | +0.159 |
| condition time | 22.563 (11.849–24.804) | 24.480 (21.657–26.337) | +1.917 |
| first-step interval | 27.891 (21.213–33.428) | 30.215 (21.228–34.014) | +2.324 |
| engine-reported loading per generation | 43.62 (38.43–50.49) | 49.57 (40.83–53.97) | +5.95 |

First generations (no reload before the step; separate from medians): reference 26738.272 ms,
FA off 30288.331 ms (+3550.059 ms); denoise_step_0 12300.532 → 13388.099 ms.

### Correctness attempt: jetson-flux-klein-010__attempt__20261014-183050

[Run](../results/runs/jetson-flux-klein-010__attempt__20261014-183050/),
[status](../results/runs/jetson-flux-klein-010__attempt__20261014-183050/status.json) complete,
[summary](../results/runs/jetson-flux-klein-010__attempt__20261014-183050/summary.json). First + one measured generation.

| Metric | First (ms) | Measured observation (ms) |
|---|---:|---:|
| wall_ms | 30187.551 | 67019.497 |
| text_encode_ms | 2285.006 | 24898.088 |
| denoise_step_0_ms | 13117.804 | 26522.797 |
| steady per-step time (s) | 3.524 | 3.526 |
| vae_decode_ms | 3771.986 | 4731.203 |

Context creation 65.456 s. RAM peak 6.006 GiB (163 samples); guard not triggered (MemAvailable
minimum 1300 MB, swap 326 → 329 MB). Both images have hash `275c50ae…40b6`. One measured
observation does not qualify a speed result.

### Failed attempt: jetson-flux-klein-010__attempt__20261014-165043

[Status](../results/runs/jetson-flux-klein-010__attempt__20261014-165043/status.json): failed, `KeyError('record_kind')` in
`run_jetson_sdcpp.py` while building `summary.json`. Both generations completed and were recorded
([runs.csv](../results/runs/jetson-flux-klein-010__attempt__20261014-165043/runs.csv),
[callbacks](../results/runs/jetson-flux-klein-010__attempt__20261014-165043/engine/results.jsonl)); no summary or event
records exist. Cause: the attempt config lacked the `record_kind` field the runner requires; this is
a config error, fixed by restoring `record_kind` before the complete attempt above. Partial
measurements (not used for any median): first wall 30367.001 ms, measured wall 69437.126 ms,
steady per-step time 3.523 s (first) and 3.527 s (measured). Guard not triggered (exit code 1 is the
runner failure; swap 287 → 342 MB). Image hash `275c50ae…40b6`.

## Observations

- Steady per-step time rises from 2.713 s to 3.535 s (+30.3 %); the per-generation range
  (3.531–3.541 s) does not overlap the reference range (2.701–2.725 s). The attempts give 3.523–3.527 s.
- Steps 1–3 contain no engine tensor-loading messages in either run.
- The flux compute buffer printed by the engine grows from 329.00 MB to 491.00 MB; whole-capture RAM
  peak is 5.990 GiB versus 6.000 GiB in the reference.
- Condition time and the first-step interval are higher by 1.917 s and 2.324 s in median, and the
  engine-reported loading sum per generation is higher by 5.95 s.
- Decode changes by +0.159 s; VAE settings are identical in both runs.
- Output pixels differ from the reference.

## Interpretation

- The steady per-step time isolates the effect of the option on transformer compute: without
  diffusion flash attention each denoising step costs about 0.82 s more, roughly 2.47 s per
  generation for steps 1–3. Step 0 of the first generation, which has no reload before it, is
  1.088 s longer, so the compute effect per generation is about 3.5 s, consistent with the
  first-generation wall delta (+3.55 s).
- The wall-time delta (+9.10 s) is larger than that compute effect. Most of the remainder sits in
  condition time and the first-step interval, which in this configuration include weight reloading,
  and the engine-reported loading sum is also higher. **Interpretation:** the wall-time delta mixes the
  option's compute effect with reload drift between the two runs. **Hypothesis:** the drift comes from
  page-cache / storage I/O state that changed during the 1 h 45 min between runs; it is not measured here.
- Flash attention is the better setting for this configuration on both per-step time and the
  diffusion compute buffer. Turning it off is not a quality-neutral change (pixels differ).

## Next experiment

- Back-to-back interleaved reference/FA-off runs (`jetson-flux-klein-q4-512-disk-baseline.json`,
  `jetson-flux-klein-q4-512-disk-noflash.json`) with a page-cache state recorded before each run.
- Paired image diagnostic (MSE/PSNR/LPIPS) of the FA-off output against the reference image.
