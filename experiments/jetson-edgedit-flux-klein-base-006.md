---
type: experiment-record
id: jetson-edgedit-flux-klein-base-006
status: complete
device: jetson-orin-nano
engine: edge-dit-cpp
runs: [jetson-edgedit-flux-klein-base-006__baseline__20261016-000110, jetson-edgedit-flux-klein-base-006__attempt__20261015-235220, jetson-edgedit-flux-klein-base-006__attempt__20261015-163130]
updated: 2026-10-16
---

# jetson-edgedit-flux-klein-base-006: edge-dit.cpp step cache `cache-dit` (engine defaults), klein Base Q4_0 512², 50 steps CFG 4; 23 of 100 transformer passes skipped, wall −22.5%

Status: **complete** (2026-10-16; full run 04:01:10–04:45:33 UTC, smoke 03:52:20–04:00:39 UTC). Full 1+3+10 protocol; no memory-guard intervention. A first attempt on 2026-10-15 failed in the runner, not the engine (Setup).

## Question

Does edge-dit.cpp's `cache-dit` step cache, at its default parameters, skip or reuse transformer work in the 50-step CFG klein Base placement [jetson-edgedit-flux-klein-base-001](jetson-edgedit-flux-klein-base-001.md), and what does it change in latency, memory and output? Single change against [jetson-edgedit-flux-klein-base-q4-embq8-512-vaetile.json](../configs/jetson-edgedit-flux-klein-base-q4-embq8-512-vaetile.json). Step-reuse lever under a quality requirement (RQ2).

## Setup

- **Config**: [jetson-edgedit-flux-klein-base-q4-embq8-512-vaetile-cache-dit.json](../configs/jetson-edgedit-flux-klein-base-q4-embq8-512-vaetile-cache-dit.json); smoke [jetson-edgedit-flux-klein-base-q4-embq8-512-vaetile-cache-dit-smoke.json](../configs/jetson-edgedit-flux-klein-base-q4-embq8-512-vaetile-cache-dit-smoke.json) (first + one measured). `reference_config` is the base-001 config.
- **Change vs reference (only one)**: `optimizations.step_cache: {"method": "cache-dit"}`, mapped by the runner to `--cache_method cache-dit` with no tuning flags, so the engine's per-method defaults apply ([command.json](../results/runs/jetson-edgedit-flux-klein-base-006__baseline__20261016-000110/command.json)). The engine log does not print the default parameter values.
- Everything else as in the [base-001 setup](jetson-edgedit-flux-klein-base-001.md#setup): edge-dit.cpp `97cc585`, same `ed-sample` binary; Base Q4_0 transformer, Q4_K_M encoder with Q8_0 token embedding (compressed, separately labelled), BF16 VAE, all weights resident; flash attention on; VAE tiling on; 512×512, 50 Euler steps, guidance 4.0, CFG 4.0, seed 0, one prompt; 25W.
- **Protocol**: one `ed-sample` process, one load, `--repeat 14` (first, 3 warm-ups, 10 measured); smoke `--repeat 2`. [jetson_gap_queue_20261016f.sh](../scripts/jetson_gap_queue_20261016f.sh) runs the full protocol only when the smoke's `step-cache.json` reports skipped/reused work (here [23, 23]); each job under [jetson_memguard.sh](../scripts/jetson_memguard.sh) (400 MB floor, 768 MB swap-growth limit).
- **Cache counts are per transformer pass.** With CFG each step runs two transformer passes (the engine reports `variants=2`), 100 per image. The engine line `<Method> reused|skipped k/50 steps` uses the step count as denominator, but k counts passes (MagCache prints k = 70 > 50 on this workload); `step-cache.json → skipped_steps_per_generation` copies k unchanged.
- **First attempt** [jetson-edgedit-flux-klein-base-006__attempt__20261015-163130](../results/runs/jetson-edgedit-flux-klein-base-006__attempt__20261015-163130/) (2026-10-15): `failed`, `KeyError('denoise_step_4_ms')`. The runner's log parser reused the step-count variable name inside its stage loop, so a 50-step run looked up `denoise_step_4_ms`; corrected in [run_edgedit_jetson.py](../scripts/run_edgedit_jetson.py) (sha256 `153d3ea9…` for these runs) before the rerun. Not a measurement.
- Metric derivation: [Jetson edge-dit runner](../wiki/methods/baseline-metrics.md#jetson-edge-dit-runner) (wall from `ed-sample` stdout at 1 ms rounding; stages from the engine breakdown at 10 ms rounding; no per-step times; whole-system tegrastats only; only the final pass's image retained).

## Results

Full run [jetson-edgedit-flux-klein-base-006__baseline__20261016-000110](../results/runs/jetson-edgedit-flux-klein-base-006__baseline__20261016-000110/): [status.json](../results/runs/jetson-edgedit-flux-klein-base-006__baseline__20261016-000110/status.json) `complete`; [memguard.json](../results/runs/jetson-edgedit-flux-klein-base-006__baseline__20261016-000110/memguard.json) not killed, exit 0.

Cache receipt ([engine.log](../results/runs/jetson-edgedit-flux-klein-base-006__baseline__20261016-000110/engine.log), [step-cache.json](../results/runs/jetson-edgedit-flux-klein-base-006__baseline__20261016-000110/step-cache.json)): every pass logs `cache enabled: mode=CacheDiT model=flux2-klein granularity=output matched_sites=1 variants=2 slots=1` and `CacheDiT skipped 23/50 steps (1.85x estimated speedup), accum_diff=1.0802` (14 of 14 passes; smoke 2 of 2; source `condition_policy.cpp`). The log does not say which passes were affected.

### Latency (measured n = 10; [summary.json](../results/runs/jetson-edgedit-flux-klein-base-006__baseline__20261016-000110/summary.json), [runs.csv](../results/runs/jetson-edgedit-flux-klein-base-006__baseline__20261016-000110/runs.csv), [stages.csv](../results/runs/jetson-edgedit-flux-klein-base-006__baseline__20261016-000110/stages.csv))

| Stage | Median (ms) | Min–max (ms) | Share of median wall | base-001 median (ms) | Δ |
|---|---:|---:|---:|---:|---:|
| wall (1 ms rounding) | **180237.5** | 179905.0–180386.0 | 100% | 232499.0 | −52261.5 (−22.5%) |
| text_encode | 920.0 | 900.0–930.0 | 0.5% | 930.0 | −10.0 |
| denoise (engine `sampling`, 50 steps) | 176535.0 | 176200.0–176690.0 | 97.9% | 228770.0 | −52235.0 (−22.8%) |
| vae_decode (tiled) | 1750.0 | 1730.0–1760.0 | 1.0% | 1750.0 | +0.0 |
| other | 1040.0 | 1040.0–1050.0 | 0.6% | 1050.0 | −10.0 |
| First generation (wall) | 185363.0 | text 2330.0 / denoise 177640.0 / vae 4330.0 / other 1070.0 | – | 238765.0 | −53402.0 (−22.4%) |
| Model load (s) | 63.194 | – | – | 63.132 | – |

Wall stdev 184.2 ms; warm-ups 180122.0, 180059.0, 180489.0 ms. Denoise per computed pass (derived): 176535.0 / (100 − 23) = 2292.7 ms; reference 228770.0 / 100 = 2287.7 ms. Measured denoise ratio 0.772; computed-pass fraction (100 − 23)/100 = 0.77.

### Memory (whole-system tegrastats at 200 ms, whole capture including load)

| Quantity | 006 | base-001 |
|---|---|---|
| RAM peak (tegrastats) | 6.729 GiB of 7.44 GiB | 6.731 GiB |
| Swap (tegrastats) min–max | 0.308–0.309 GiB | 0.311–0.311 GiB |
| MemAvailable (guard), start → minimum | 5938 → 552 MB | 5996 → 568 MB |
| GPU temperature max | 65.0 °C | 65.2 °C |

### Smoke run

[jetson-edgedit-flux-klein-base-006__attempt__20261015-235220](../results/runs/jetson-edgedit-flux-klein-base-006__attempt__20261015-235220/): complete, guard not triggered; CacheDiT 23 in both passes. First 186358.0 ms (text 2340.0 / denoise 178710.0 / vae 4210.0 / other 1090.0); one measured 179873.0 ms. Load 63.125 s. RAM peak 6.625 GiB; MemAvailable minimum 632 MB. Not a baseline (n = 1).

### Repeatability and functional check (visual, not a quality metric)

Final-pass image (run index 13) RGB SHA256 `2fe71ab3ef720f94…`; the smoke's final image has the same hash. It differs from the reference final image (`d6f0a13ed741a535…`): mean absolute RGB difference 9.2 levels; 37.3% of pixels differ by more than 8 levels in any channel (computed from the two `output.png` files).

Visual check of `output.png` against the reference's: hash-identical to [jetson-edgedit-flux-klein-base-004](jetson-edgedit-flux-klein-base-004.md) (DBCache): stray stroke beside the second "l" of "hello"; the sign reads "hello world". No blur.

## Observations

- The engine skipped 23 of 100 transformer passes in every generation; denoise fell from 228770.0 to 176535.0 ms (−22.8%) and wall from 232499.0 to 180237.5 ms (−22.5%).
- Measured denoise ratio 0.772 against a computed-pass fraction of 0.77; time per computed pass 2292.7 ms against 2287.7 ms in the reference (+0.2%).
- Text encode, VAE decode and other medians are within 10 ms of the reference.
- RAM peak 6.729 GiB (−0.003 GiB vs base-001); MemAvailable minimum 552 MB (−16 MB); swap did not grow.
- Output deterministic across the smoke and full processes (final-image hash); 9.2 mean RGB levels from the reference image.

## Interpretation

- Sampling is 98.4% of the reference wall, so the wall saving follows the number of passes the cache removes; the per-pass cost stays within 3% of the reference, so the cache's own bookkeeping is small on this device.
- Hypothesis: cache-dit and DBCache share one engine policy (`condition_policy.cpp`) with the same defaults; identical skip counts and final-image hash are consistent with that.
- Whether the changed pixels meet a declared quality requirement needs a quality metric over more than one prompt; this record has one image.

## Next experiment

- Copy of `jetson-edgedit-flux-klein-base-q4-embq8-512-vaetile-cache-dit.json` with a paired saved-image quality metric against base-001 (calibration prompts only).
