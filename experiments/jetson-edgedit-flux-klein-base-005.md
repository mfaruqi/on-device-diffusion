---
type: experiment-record
id: jetson-edgedit-flux-klein-base-005
status: complete
device: jetson-orin-nano
engine: edge-dit-cpp
runs: [jetson-edgedit-flux-klein-base-005__baseline__20261015-231521, jetson-edgedit-flux-klein-base-005__attempt__20261015-230740, jetson-edgedit-flux-klein-base-005__attempt__20261015-162349]
updated: 2026-10-16
---

# jetson-edgedit-flux-klein-base-005: edge-dit.cpp step cache `taylorseer` (engine defaults), klein Base Q4_0 512², 50 steps CFG 4; 38 of 100 transformer passes reused, wall −37.1%

Status: **complete** (2026-10-16; full run 03:15:21–03:51:49 UTC, smoke 03:07:40–03:14:51 UTC). Full 1+3+10 protocol; no memory-guard intervention. A first attempt on 2026-10-15 failed in the runner, not the engine (Setup).

## Question

Does edge-dit.cpp's `taylorseer` step cache, at its default parameters, skip or reuse transformer work in the 50-step CFG klein Base placement [jetson-edgedit-flux-klein-base-001](jetson-edgedit-flux-klein-base-001.md), and what does it change in latency, memory and output? Single change against [jetson-edgedit-flux-klein-base-q4-embq8-512-vaetile.json](../configs/jetson-edgedit-flux-klein-base-q4-embq8-512-vaetile.json). Step-reuse lever under a quality requirement (RQ2).

## Setup

- **Config**: [jetson-edgedit-flux-klein-base-q4-embq8-512-vaetile-taylorseer.json](../configs/jetson-edgedit-flux-klein-base-q4-embq8-512-vaetile-taylorseer.json); smoke [jetson-edgedit-flux-klein-base-q4-embq8-512-vaetile-taylorseer-smoke.json](../configs/jetson-edgedit-flux-klein-base-q4-embq8-512-vaetile-taylorseer-smoke.json) (first + one measured). `reference_config` is the base-001 config.
- **Change vs reference (only one)**: `optimizations.step_cache: {"method": "taylorseer"}`, mapped by the runner to `--cache_method taylorseer` with no tuning flags, so the engine's per-method defaults apply ([command.json](../results/runs/jetson-edgedit-flux-klein-base-005__baseline__20261015-231521/command.json)). The engine log does not print the default parameter values.
- Everything else as in the [base-001 setup](jetson-edgedit-flux-klein-base-001.md#setup): edge-dit.cpp `97cc585`, same `ed-sample` binary; Base Q4_0 transformer, Q4_K_M encoder with Q8_0 token embedding (compressed, separately labelled), BF16 VAE, all weights resident; flash attention on; VAE tiling on; 512×512, 50 Euler steps, guidance 4.0, CFG 4.0, seed 0, one prompt; 25W.
- **Protocol**: one `ed-sample` process, one load, `--repeat 14` (first, 3 warm-ups, 10 measured); smoke `--repeat 2`. [jetson_gap_queue_20261016f.sh](../scripts/jetson_gap_queue_20261016f.sh) runs the full protocol only when the smoke's `step-cache.json` reports skipped/reused work (here [38, 38]); each job under [jetson_memguard.sh](../scripts/jetson_memguard.sh) (400 MB floor, 768 MB swap-growth limit).
- **Cache counts are per transformer pass.** With CFG each step runs two transformer passes (the engine reports `variants=2`), 100 per image. The engine line `<Method> reused|skipped k/50 steps` uses the step count as denominator, but k counts passes (MagCache prints k = 70 > 50 on this workload); `step-cache.json → skipped_steps_per_generation` copies k unchanged.
- **First attempt** [jetson-edgedit-flux-klein-base-005__attempt__20261015-162349](../results/runs/jetson-edgedit-flux-klein-base-005__attempt__20261015-162349/) (2026-10-15): `failed`, `KeyError('denoise_step_4_ms')`. The runner's log parser reused the step-count variable name inside its stage loop, so a 50-step run looked up `denoise_step_4_ms`; corrected in [run_edgedit_jetson.py](../scripts/run_edgedit_jetson.py) (sha256 `153d3ea9…` for these runs) before the rerun. Not a measurement.
- Metric derivation: [Jetson edge-dit runner](../wiki/methods/baseline-metrics.md#jetson-edge-dit-runner) (wall from `ed-sample` stdout at 1 ms rounding; stages from the engine breakdown at 10 ms rounding; no per-step times; whole-system tegrastats only; only the final pass's image retained).

## Results

Full run [jetson-edgedit-flux-klein-base-005__baseline__20261015-231521](../results/runs/jetson-edgedit-flux-klein-base-005__baseline__20261015-231521/): [status.json](../results/runs/jetson-edgedit-flux-klein-base-005__baseline__20261015-231521/status.json) `complete`; [memguard.json](../results/runs/jetson-edgedit-flux-klein-base-005__baseline__20261015-231521/memguard.json) not killed, exit 0.

Cache receipt ([engine.log](../results/runs/jetson-edgedit-flux-klein-base-005__baseline__20261015-231521/engine.log), [step-cache.json](../results/runs/jetson-edgedit-flux-klein-base-005__baseline__20261015-231521/step-cache.json)): every pass logs `cache enabled: mode=TaylorSeer model=flux2-klein granularity=feature matched_sites=1 variants=2 slots=1` and `TaylorSeer reused 38/50 steps (4.17x)` (14 of 14 passes; smoke 2 of 2; source `taylorseer_policy.cpp`). The log does not say which passes were affected.

### Latency (measured n = 10; [summary.json](../results/runs/jetson-edgedit-flux-klein-base-005__baseline__20261015-231521/summary.json), [runs.csv](../results/runs/jetson-edgedit-flux-klein-base-005__baseline__20261015-231521/runs.csv), [stages.csv](../results/runs/jetson-edgedit-flux-klein-base-005__baseline__20261015-231521/stages.csv))

| Stage | Median (ms) | Min–max (ms) | Share of median wall | base-001 median (ms) | Δ |
|---|---:|---:|---:|---:|---:|
| wall (1 ms rounding) | **146273.5** | 145955.0–146530.0 | 100% | 232499.0 | −86225.5 (−37.1%) |
| text_encode | 930.0 | 910.0–940.0 | 0.6% | 930.0 | +0.0 |
| denoise (engine `sampling`, 50 steps) | 142555.0 | 142260.0–142840.0 | 97.5% | 228770.0 | −86215.0 (−37.7%) |
| vae_decode (tiled) | 1740.0 | 1720.0–1750.0 | 1.2% | 1750.0 | −10.0 |
| other | 1040.0 | 1030.0–1060.0 | 0.7% | 1050.0 | −10.0 |
| First generation (wall) | 151041.0 | text 1960.0 / denoise 143850.0 / vae 4140.0 / other 1090.0 | – | 238765.0 | −87724.0 (−36.7%) |
| Model load (s) | 63.373 | – | – | 63.132 | – |

Wall stdev 196.3 ms; warm-ups 146188.0, 146886.0, 146321.0 ms. Denoise per computed pass (derived): 142555.0 / (100 − 38) = 2299.3 ms; reference 228770.0 / 100 = 2287.7 ms. Measured denoise ratio 0.623; computed-pass fraction (100 − 38)/100 = 0.62.

### Memory (whole-system tegrastats at 200 ms, whole capture including load)

| Quantity | 005 | base-001 |
|---|---|---|
| RAM peak (tegrastats) | 6.851 GiB of 7.44 GiB | 6.731 GiB |
| Swap (tegrastats) min–max | 0.309–0.309 GiB | 0.311–0.311 GiB |
| MemAvailable (guard), start → minimum | 5985 → 442 MB | 5996 → 568 MB |
| GPU temperature max | 64.9 °C | 65.2 °C |

### Smoke run

[jetson-edgedit-flux-klein-base-005__attempt__20261015-230740](../results/runs/jetson-edgedit-flux-klein-base-005__attempt__20261015-230740/): complete, guard not triggered; TaylorSeer 38 in both passes. First 150745.0 ms (text 2040.0 / denoise 143310.0 / vae 4300.0 / other 1090.0); one measured 146188.0 ms. Load 63.357 s. RAM peak 6.747 GiB; MemAvailable minimum 551 MB. Not a baseline (n = 1).

### Repeatability and functional check (visual, not a quality metric)

Final-pass image (run index 13) RGB SHA256 `61eeecd2015a5566…`; the smoke's final image has the same hash. It differs from the reference final image (`d6f0a13ed741a535…`): mean absolute RGB difference 1.6 levels; 3.7% of pixels differ by more than 8 levels in any channel (computed from the two `output.png` files).

Visual check of `output.png` against the reference's: near-identical to the reference at full resolution: same cat, sign, script lettering and background. The sign reads "hello world". No stray mark, no blur.

## Observations

- The engine reused 38 of 100 transformer passes in every generation; denoise fell from 228770.0 to 142555.0 ms (−37.7%) and wall from 232499.0 to 146273.5 ms (−37.1%).
- Measured denoise ratio 0.623 against a computed-pass fraction of 0.62; time per computed pass 2299.3 ms against 2287.7 ms in the reference (+0.5%).
- Text encode, VAE decode and other medians are within 10 ms of the reference.
- RAM peak 6.851 GiB (+0.119 GiB vs base-001); MemAvailable minimum 442 MB (−126 MB); swap did not grow.
- Output deterministic across the smoke and full processes (final-image hash); 1.6 mean RGB levels from the reference image.

## Interpretation

- Sampling is 98.4% of the reference wall, so the wall saving follows the number of passes the cache removes; the per-pass cost stays within 3% of the reference, so the cache's own bookkeeping is small on this device.
- Hypothesis: predicting features with a Taylor expansion (`granularity=feature`) keeps the output closer to the reference than reusing whole outputs, which would explain the smallest pixel difference among the caches despite the largest count of reused passes after MagCache; a single prompt cannot establish it.
- Whether the changed pixels meet a declared quality requirement needs a quality metric over more than one prompt; this record has one image.

## Next experiment

- Copy of `jetson-edgedit-flux-klein-base-q4-embq8-512-vaetile-taylorseer.json` with a paired saved-image quality metric against base-001 (calibration prompts only).
