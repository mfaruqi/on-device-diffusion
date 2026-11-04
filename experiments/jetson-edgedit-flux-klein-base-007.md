---
type: experiment-record
id: jetson-edgedit-flux-klein-base-007
status: complete
device: jetson-orin-nano
engine: edge-dit-cpp
runs: [jetson-edgedit-flux-klein-base-007__baseline__20261016-005119, jetson-edgedit-flux-klein-base-007__attempt__20261016-004603, jetson-edgedit-flux-klein-base-007__attempt__20261015-164019]
updated: 2026-10-16
---

# jetson-edgedit-flux-klein-base-007: edge-dit.cpp step cache `magcache` (engine defaults), klein Base Q4_0 512², 50 steps CFG 4; 70 of 100 transformer passes reused, wall −68.6%

Status: **complete** (2026-10-16; full run 04:51:19–05:10:43 UTC, smoke 04:46:03–04:50:49 UTC). Full 1+3+10 protocol; no memory-guard intervention. A first attempt on 2026-10-15 failed in the runner, not the engine (Setup).

## Question

Does edge-dit.cpp's `magcache` step cache, at its default parameters, skip or reuse transformer work in the 50-step CFG klein Base placement [jetson-edgedit-flux-klein-base-001](jetson-edgedit-flux-klein-base-001.md), and what does it change in latency, memory and output? Single change against [jetson-edgedit-flux-klein-base-q4-embq8-512-vaetile.json](../configs/jetson-edgedit-flux-klein-base-q4-embq8-512-vaetile.json). Step-reuse lever under a quality requirement (RQ2).

## Setup

- **Config**: [jetson-edgedit-flux-klein-base-q4-embq8-512-vaetile-magcache.json](../configs/jetson-edgedit-flux-klein-base-q4-embq8-512-vaetile-magcache.json); smoke [jetson-edgedit-flux-klein-base-q4-embq8-512-vaetile-magcache-smoke.json](../configs/jetson-edgedit-flux-klein-base-q4-embq8-512-vaetile-magcache-smoke.json) (first + one measured). `reference_config` is the base-001 config.
- **Change vs reference (only one)**: `optimizations.step_cache: {"method": "magcache"}`, mapped by the runner to `--cache_method magcache` with no tuning flags, so the engine's per-method defaults apply ([command.json](../results/runs/jetson-edgedit-flux-klein-base-007__baseline__20261016-005119/command.json)). The engine log does not print the default parameter values.
- Everything else as in the [base-001 setup](jetson-edgedit-flux-klein-base-001.md#setup): edge-dit.cpp `97cc585`, same `ed-sample` binary; Base Q4_0 transformer, Q4_K_M encoder with Q8_0 token embedding (compressed, separately labelled), BF16 VAE, all weights resident; flash attention on; VAE tiling on; 512×512, 50 Euler steps, guidance 4.0, CFG 4.0, seed 0, one prompt; 25W.
- **Protocol**: one `ed-sample` process, one load, `--repeat 14` (first, 3 warm-ups, 10 measured); smoke `--repeat 2`. [jetson_gap_queue_20261016f.sh](../scripts/jetson_gap_queue_20261016f.sh) runs the full protocol only when the smoke's `step-cache.json` reports skipped/reused work (here [70, 70]); each job under [jetson_memguard.sh](../scripts/jetson_memguard.sh) (400 MB floor, 768 MB swap-growth limit).
- **Cache counts are per transformer pass.** With CFG each step runs two transformer passes (the engine reports `variants=2`), 100 per image. The engine line `<Method> reused|skipped k/50 steps` uses the step count as denominator, but k counts passes (MagCache prints k = 70 > 50 on this workload); `step-cache.json → skipped_steps_per_generation` copies k unchanged.
- **First attempt** [jetson-edgedit-flux-klein-base-007__attempt__20261015-164019](../results/runs/jetson-edgedit-flux-klein-base-007__attempt__20261015-164019/) (2026-10-15): `failed`, `KeyError('denoise_step_4_ms')`. The runner's log parser reused the step-count variable name inside its stage loop, so a 50-step run looked up `denoise_step_4_ms`; corrected in [run_edgedit_jetson.py](../scripts/run_edgedit_jetson.py) (sha256 `153d3ea9…` for these runs) before the rerun. Not a measurement.
- Metric derivation: [Jetson edge-dit runner](../wiki/methods/baseline-metrics.md#jetson-edge-dit-runner) (wall from `ed-sample` stdout at 1 ms rounding; stages from the engine breakdown at 10 ms rounding; no per-step times; whole-system tegrastats only; only the final pass's image retained).

## Results

Full run [jetson-edgedit-flux-klein-base-007__baseline__20261016-005119](../results/runs/jetson-edgedit-flux-klein-base-007__baseline__20261016-005119/): [status.json](../results/runs/jetson-edgedit-flux-klein-base-007__baseline__20261016-005119/status.json) `complete`; [memguard.json](../results/runs/jetson-edgedit-flux-klein-base-007__baseline__20261016-005119/memguard.json) not killed, exit 0.

Cache receipt ([engine.log](../results/runs/jetson-edgedit-flux-klein-base-007__baseline__20261016-005119/engine.log), [step-cache.json](../results/runs/jetson-edgedit-flux-klein-base-007__baseline__20261016-005119/step-cache.json)): every pass logs `cache enabled: mode=MagCache model=flux2-klein granularity=feature matched_sites=1 variants=2 slots=1` and `MagCache reused 70/50 steps` (14 of 14 passes; smoke 2 of 2; source `magcache_policy.cpp`). The log does not say which passes were affected.

### Latency (measured n = 10; [summary.json](../results/runs/jetson-edgedit-flux-klein-base-007__baseline__20261016-005119/summary.json), [runs.csv](../results/runs/jetson-edgedit-flux-klein-base-007__baseline__20261016-005119/runs.csv), [stages.csv](../results/runs/jetson-edgedit-flux-klein-base-007__baseline__20261016-005119/stages.csv))

| Stage | Median (ms) | Min–max (ms) | Share of median wall | base-001 median (ms) | Δ |
|---|---:|---:|---:|---:|---:|
| wall (1 ms rounding) | **73108.0** | 72931.0–73275.0 | 100% | 232499.0 | −159391.0 (−68.6%) |
| text_encode | 920.0 | 910.0–930.0 | 1.3% | 930.0 | −10.0 |
| denoise (engine `sampling`, 50 steps) | 69410.0 | 69200.0–69580.0 | 94.9% | 228770.0 | −159360.0 (−69.7%) |
| vae_decode (tiled) | 1745.0 | 1730.0–1760.0 | 2.4% | 1750.0 | −5.0 |
| other | 1040.0 | 1030.0–1050.0 | 1.4% | 1050.0 | −10.0 |
| First generation (wall) | 77934.0 | text 1820.0 / denoise 70680.0 / vae 4390.0 / other 1050.0 | – | 238765.0 | −160831.0 (−67.4%) |
| Model load (s) | 63.033 | – | – | 63.132 | – |

Wall stdev 111.2 ms; warm-ups 73182.0, 73188.0, 73075.0 ms. Denoise per computed pass (derived): 69410.0 / (100 − 70) = 2313.7 ms; reference 228770.0 / 100 = 2287.7 ms. Measured denoise ratio 0.303; computed-pass fraction (100 − 70)/100 = 0.30.

### Memory (whole-system tegrastats at 200 ms, whole capture including load)

| Quantity | 007 | base-001 |
|---|---|---|
| RAM peak (tegrastats) | 6.772 GiB of 7.44 GiB | 6.731 GiB |
| Swap (tegrastats) min–max | 0.308–0.308 GiB | 0.311–0.311 GiB |
| MemAvailable (guard), start → minimum | 5895 → 531 MB | 5996 → 568 MB |
| GPU temperature max | 64.7 °C | 65.2 °C |

### Smoke run

[jetson-edgedit-flux-klein-base-007__attempt__20261016-004603](../results/runs/jetson-edgedit-flux-klein-base-007__attempt__20261016-004603/): complete, guard not triggered; MagCache 70 in both passes. First 78591.0 ms (text 2340.0 / denoise 70860.0 / vae 4330.0 / other 1060.0); one measured 73101.0 ms. Load 63.155 s. RAM peak 6.727 GiB; MemAvailable minimum 494 MB. Not a baseline (n = 1).

### Repeatability and functional check (visual, not a quality metric)

Final-pass image (run index 13) RGB SHA256 `69200372c447da14…`; the smoke's final image has the same hash. It differs from the reference final image (`d6f0a13ed741a535…`): mean absolute RGB difference 6.5 levels; 19.8% of pixels differ by more than 8 levels in any channel (computed from the two `output.png` files).

Visual check of `output.png` against the reference's: same cat, sign and background as the reference, no blur, but the letters of "world" are deformed: the "o" and "r" merge into one shape. "hello" is legible. Text content is visibly changed.

## Observations

- The engine reused 70 of 100 transformer passes in every generation; denoise fell from 228770.0 to 69410.0 ms (−69.7%) and wall from 232499.0 to 73108.0 ms (−68.6%).
- Measured denoise ratio 0.303 against a computed-pass fraction of 0.30; time per computed pass 2313.7 ms against 2287.7 ms in the reference (+1.1%).
- Text encode, VAE decode and other medians are within 10 ms of the reference.
- RAM peak 6.772 GiB (+0.041 GiB vs base-001); MemAvailable minimum 531 MB (−37 MB); swap did not grow.
- Output deterministic across the smoke and full processes (final-image hash); 6.5 mean RGB levels from the reference image.

## Interpretation

- Sampling is 98.4% of the reference wall, so the wall saving follows the number of passes the cache removes; the per-pass cost stays within 3% of the reference, so the cache's own bookkeeping is small on this device.
- Hypothesis: MagCache's default magnitude-ratio threshold allows long reuse runs on this schedule; 70 of 100 passes reused leaves too few computed passes to keep fine glyph structure, while coarse content survives.
- Whether the changed pixels meet a declared quality requirement needs a quality metric over more than one prompt; this record has one image.

## Next experiment

- Copy of `jetson-edgedit-flux-klein-base-q4-embq8-512-vaetile-magcache.json` with a paired saved-image quality metric against base-001 (calibration prompts only).
