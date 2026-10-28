---
type: experiment-record
id: jetson-flux-klein-011
status: complete
device: jetson-orin-nano
engine: stable-diffusion-cpp
runs: [jetson-flux-klein-011__baseline__20261014-185639, jetson-flux-klein-011__attempt__20261014-185236, jetson-flux-klein-011__attempt__20261014-165450]
updated: 2026-10-14
---

# jetson-flux-klein-011: VAE tiling (128 px tiles, 0.25 overlap), disk-backed quantized 512×512

Status: **complete** (2026-10-14); full unprofiled 1+3+10 protocol and a two-generation
correctness attempt completed. An earlier attempt failed after generation because of a config error.

## Question

How does tiled VAE decoding with 128-pixel tiles and a 0.25 target overlap change decode latency,
the VAE compute buffer, output and memory diagnostics for the disk-backed quantized 512×512 workload
of [jetson-flux-klein-003](jetson-flux-klein-003.md)? Week 3 “verify target feasibility” and RQ1:
“Which diffusion optimization choices transfer across devices and workloads?”
([proposal overview](../wiki/project/overview.md)). The reference is the same-session 003 repeat
[`jetson-flux-klein-003__repeat__20261014-163411`](../results/runs/jetson-flux-klein-003__repeat__20261014-163411/),
run with the same harness binary.

## Setup

- [Configuration](../configs/jetson-flux-klein-q4-512-disk-vaetile.json) and
  [attempt configuration](../configs/jetson-flux-klein-q4-512-disk-vaetile-smoke.json). One change
  against [the 003 reference](../configs/jetson-flux-klein-q4-512-disk-baseline.json): VAE tiling on
  (`optimizations.vae_tiling: true`; `engine_settings.vae_tile_size: 128`, `vae_tile_overlap: 0.25`;
  harness `--vae-tiling 1 --vae-tile-size 128 --vae-tile-overlap 0.25`). Tile size and overlap are
  passed explicitly so the run does not depend on library defaults.
- Workload: pinned sd.cpp `19bbbca`; FLUX.2 klein 4B transformer Q4_0, Qwen3-4B Q4_K_M text
  encoder, BF16 VAE (revisions and SHA256 in the config); 512×512, 4 Euler steps, flux2 scheduler,
  CFG 1.0, seed 0, batch 1, prompt “A cat holding a sign that says hello world”.
- Execution: CUDA0 compute, disk-backed parameters, eager loading, segmented compute allowed,
  diffusion flash attention on, prefetch on, mmap off, conditioning cache 0, step cache off,
  4 CPU threads, 25 W mode.
- Harness: `sd-bench` built from `engines/sdcpp/bench.cpp` of this branch, which adds the optional
  `--vae-tiling/--vae-tile-size/--vae-tile-overlap` flags. Binary SHA256
  `f9ed5788425f5ce53bf6da86182fdaeba2fa00e7c1c73a2c309f2f63808607f4`, identical to the reference run
  ([environment](../results/runs/jetson-flux-klein-011__baseline__20261014-185639/environment.json)).
- Protocol: one model context; baseline 1 first + 3 discarded warm-ups + 10 measured; attempt
  1 first + 1 measured. All weight files were SHA256-read before each run, so the filesystem cache
  was warm; no cache flush ([command](../results/runs/jetson-flux-klein-011__baseline__20261014-185639/engine/command.json)).
- Memory guard (outside the config): the run is killed if MemAvailable falls below 400 MB or swap
  use grows by more than 768 MB, polled every 0.1 s ([memguard](../results/runs/jetson-flux-klein-011__baseline__20261014-185639/memguard.json)).
- Metrics: [baseline-metrics.md, stable-diffusion.cpp runs](../wiki/methods/baseline-metrics.md#stable-diffusioncpp-runs).
  Host callback times include on-demand weight loading. Condition time, first-step interval,
  steady per-step time and decode are the derived quantities defined in
  [jetson-flux-klein-010](jetson-flux-klein-010.md#derived-quantities-used-in-this-record),
  computed from `engine/results.jsonl` for measured generations.

## Results

### Full protocol: jetson-flux-klein-011__baseline__20261014-185639

[Run directory](../results/runs/jetson-flux-klein-011__baseline__20261014-185639/),
[status](../results/runs/jetson-flux-klein-011__baseline__20261014-185639/status.json) complete,
22:56:39–23:14:20 UTC, [summary](../results/runs/jetson-flux-klein-011__baseline__20261014-185639/summary.json),
[config](../results/runs/jetson-flux-klein-011__baseline__20261014-185639/config.json),
[callbacks](../results/runs/jetson-flux-klein-011__baseline__20261014-185639/engine/results.jsonl),
[engine log](../results/runs/jetson-flux-klein-011__baseline__20261014-185639/engine/sdcpp.log). Power mode 25 W;
CUDA 12.6; sd.cpp `19bbbca`.

| Metric | Median (ms) | Min–max (ms) |
|---|---:|---:|
| wall_ms | 71633.709 | 66979.990–71965.948 |
| text_encode_ms | 24324.393 | 21756.121–24904.149 |
| denoise_ms | 40981.023 | 38958.030–41382.924 |
| denoise_step_0_ms | 32841.800 | 30805.728–33236.878 |
| denoise_step_1_ms | 2714.309 | 2704.026–2722.194 |
| denoise_step_2_ms | 2714.059 | 2706.728–2722.584 |
| denoise_step_3_ms | 2713.226 | 2706.490–2725.064 |
| vae_decode_ms | 5857.549 | 5747.395–6206.326 |
| other_ms | 275.328 | 261.687–302.518 |

Wall standard deviation 1524.005 ms. First generation (separate): 28597.009 ms, of which
text_encode 2440.672 ms, denoise_step_0 12740.459 ms, vae_decode 4867.381 ms. Model-context
creation 66.212 s ([load](../results/runs/jetson-flux-klein-011__baseline__20261014-185639/load.json)); not a cold-storage
measurement. Stage medians are independent and need not sum to the wall median.

Derived quantities (s, median and min–max over the 10 measured generations):

| Quantity | Median | Min–max |
|---|---:|---:|
| condition time | 24.324 | 21.756–24.904 |
| first-step interval | 32.842 | 30.806–33.237 |
| steady per-step time | 2.713 | 2.706–2.721 |
| decode | 5.858 | 5.747–6.206 |

Engine-reported tensor-loading intervals inside each measured generation sum to a median of
53.68 s (49.19–53.98 s): 22.84 s median inside the condition window, 29.10 s inside the first-step
interval and 1.40 s inside decode. These rounded engine diagnostics overlap stage time and do not
measure physical storage I/O.

### Tiling evidence from the engine log

Each of the 14 decodes in the baseline logs the same sequence
([engine log](../results/runs/jetson-flux-klein-011__baseline__20261014-185639/engine/sdcpp.log)):

- `VAE decode tile size: 128x128 pixels (8x8 latent)`
- `num tiles : 5, 5`
- `optimal overlap : 0.250000, 0.250000 (targeting 0.250000)`
- `processing 25 tiles`

The context settings print `VAE tiling: true`; the summary audit counts 14 tiled decodes
(2 in each attempt, which log the same lines). The VAE graph executes 25 times per decode
(350 VAE segment executions over 14 generations, versus 14 in the reference).

### Memory diagnostics

- Engine-printed VAE compute buffer: 104.03 MB (≈0.102 GiB) per decode, versus 1664.50 MB
  (≈1.625 GiB) in the reference. Flux 329.00 MB and qwen3 65.00 MB are unchanged (printed MB
  treated as MiB).
- Whole-capture system RAM peak 5.967 GiB of 7.441 GiB (1000 one-second tegrastats samples);
  swap occupancy 0.305–0.384 GiB; maximum sampled GPU temperature 53.531 °C
  ([summary](../results/runs/jetson-flux-klein-011__baseline__20261014-185639/summary.json)). These cover loading and all
  14 generations; they are not per-stage or allocator peaks, and occupancy does not quantify swap I/O.
- Memory guard: not triggered. MemAvailable minimum 1269 MB; swap use 312 → 392 MB
  ([memguard](../results/runs/jetson-flux-klein-011__baseline__20261014-185639/memguard.json)).

### Repeatability and output

- Engine audit: diffusion flash attention in use, settings confirmed, no auto-fit, zero
  conditioning-cache hits, callbacks validated.
- All 14 generations have RGB hash `252b6cb0…e858`. This differs from the reference hash
  `2d487c94…517e`: tiled decoding changes output pixels. No paired image diagnostic
  (MSE/PSNR/LPIPS) and no visual inspection were recorded for this run; no quality claim is made.

### Delta vs reference (same-session repeat)

Reference: [`jetson-flux-klein-003__repeat__20261014-163411`](../results/runs/jetson-flux-klein-003__repeat__20261014-163411/summary.json)
(same binary, 20:34–20:50 UTC). This run started 2 h 06 min after the reference ended, after other
workloads had run on the device.

| Metric (median) | Reference (ms) | Tiling (ms) | Δ (ms) | Δ % |
|---|---:|---:|---:|---:|
| wall_ms | 60640.928 | 71633.709 | +10992.781 | +18.1 |
| text_encode_ms | 22563.202 | 24324.393 | +1761.190 | +7.8 |
| denoise_ms | 36009.000 | 40981.023 | +4972.023 | +13.8 |
| denoise_step_0_ms | 27890.725 | 32841.800 | +4951.075 | +17.8 |
| denoise_step_1_ms | 2710.426 | 2714.309 | +3.883 | +0.1 |
| denoise_step_2_ms | 2713.217 | 2714.059 | +0.842 | +0.0 |
| denoise_step_3_ms | 2716.341 | 2713.226 | −3.115 | −0.1 |
| vae_decode_ms | 4638.280 | 5857.549 | +1219.269 | +26.3 |
| other_ms | 281.891 | 275.328 | −6.562 | −2.3 |

| Derived (median, s) | Reference | Tiling | Δ |
|---|---:|---:|---:|
| steady per-step time | 2.713 (2.701–2.725) | 2.713 (2.706–2.721) | 0.000 |
| decode | 4.638 (3.833–4.862) | 5.858 (5.747–6.206) | +1.220 (+26.3 %) |
| condition time | 22.563 (11.849–24.804) | 24.324 (21.756–24.904) | +1.761 |
| first-step interval | 27.891 (21.213–33.428) | 32.842 (30.806–33.237) | +4.951 |
| engine-reported loading per generation | 43.62 (38.43–50.49) | 53.68 (49.19–53.98) | +10.06 |

Decode-window engine loading is 1.30 s (reference) and 1.40 s (tiling) in median. First generations
(separate from medians): reference 26738.272 ms, tiling 28597.009 ms (+1858.737 ms); vae_decode
3650.802 → 4867.381 ms (+1216.579 ms).

### Correctness attempt: jetson-flux-klein-011__attempt__20261014-185236

[Run](../results/runs/jetson-flux-klein-011__attempt__20261014-185236/),
[status](../results/runs/jetson-flux-klein-011__attempt__20261014-185236/status.json) complete,
[summary](../results/runs/jetson-flux-klein-011__attempt__20261014-185236/summary.json). First + one measured generation.

| Metric | First | Measured observation |
|---|---:|---:|
| wall_ms | 27963.146 | 65813.796 |
| text_encode_ms | 2221.006 | 25101.088 |
| denoise_step_0_ms | 12356.214 | 26535.251 |
| steady per-step time (s) | 2.712 | 2.715 |
| vae_decode_ms | 4766.759 | 5747.988 |

Context creation 65.526 s. RAM peak 5.995 GiB (161 samples); guard not triggered (MemAvailable
minimum 1262 MB, swap 312 → 312 MB). Both images have hash `252b6cb0…e858`. One measured
observation does not qualify a speed result.

### Failed attempt: jetson-flux-klein-011__attempt__20261014-165450

[Status](../results/runs/jetson-flux-klein-011__attempt__20261014-165450/status.json): failed, `KeyError('record_kind')` in
`run_jetson_sdcpp.py` while building `summary.json`. Both generations completed and were recorded
([runs.csv](../results/runs/jetson-flux-klein-011__attempt__20261014-165450/runs.csv),
[callbacks](../results/runs/jetson-flux-klein-011__attempt__20261014-165450/engine/results.jsonl)); no summary or event
records exist. Cause: the attempt config lacked the `record_kind` field the runner requires; this is
a config error, fixed by restoring `record_kind` before the complete attempt above. Partial
measurements (not used for any median): first wall 28086.234 ms, measured wall 63360.717 ms,
vae_decode 4765.307 ms (first) and 5809.198 ms (measured); the log shows the same 128×128-pixel,
25-tile decode. Guard not triggered (exit code 1 is the runner failure; swap 296 → 502 MB, below the
768 MB growth limit). Image hash `252b6cb0…e858`.

## Observations

- Every decode used 128×128-pixel tiles (8×8 latent), a 5×5 grid, 0.25 realised overlap and 25 tiles.
- The engine-printed VAE compute buffer falls from 1664.50 MB to 104.03 MB; the whole-capture RAM
  peak is 5.967 GiB versus 6.000 GiB in the reference.
- Decode rises by 1.220 s (+26.3 %) in median; the first-generation decode rises by 1.217 s.
  Decode-window engine loading differs by 0.10 s.
- Steady per-step time is unchanged (2.713 s in both).
- Condition time and the first-step interval are higher by 1.761 s and 4.951 s in median; the
  engine-reported loading sum per generation is higher by 10.06 s.
- Output pixels differ from the reference.

## Interpretation

- The compute cost of tiling in this configuration is in decode: about 1.2 s per generation,
  measured both in the median and in the first generation. Transformer per-step time is unaffected,
  as expected for a decode-only change.
- The wall-time delta (+10.99 s) is mostly outside decode. It sits in condition time and the
  first-step interval, which include weight reloading, and matches the higher engine-reported loading
  sum. **Interpretation:** the wall-time delta mixes the decode cost with reload drift between the two
  runs. **Hypothesis:** the drift comes from page-cache / storage I/O state that changed during the
  2 h between runs; it is not measured here.
- Tiling reduces the VAE's engine-printed compute buffer by about 1.5 GiB but does not lower the
  whole-capture RAM peak. **Hypothesis:** that peak is set outside the VAE decode window (for example
  during loading or text encoding); the one-second monitor is not stage-aligned, so this is not tested.

## Next experiment

- Back-to-back interleaved reference/tiling runs (`jetson-flux-klein-q4-512-disk-baseline.json`,
  `jetson-flux-klein-q4-512-disk-vaetile.json`) with a page-cache state recorded before each run.
- Paired image diagnostic (MSE/PSNR/LPIPS) of the tiled output against the reference image.
