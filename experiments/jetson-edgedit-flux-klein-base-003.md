---
type: experiment-record
id: jetson-edgedit-flux-klein-base-003
status: complete
device: jetson-orin-nano
engine: edge-dit-cpp
runs: [jetson-edgedit-flux-klein-base-003__baseline__20261015-213038, jetson-edgedit-flux-klein-base-003__attempt__20261015-212203, jetson-edgedit-flux-klein-base-003__attempt__20261015-160624]
updated: 2026-10-16
---

# jetson-edgedit-flux-klein-base-003: edge-dit.cpp step cache `ucache` (engine defaults), klein Base Q4_0 512², 50 steps CFG 4; 26 of 100 transformer passes skipped, wall −25.3%

Status: **complete** (2026-10-16; full run 01:30:38–02:13:30 UTC, smoke 01:22:03–01:30:07 UTC). Full 1+3+10 protocol; no memory-guard intervention. A first attempt on 2026-10-15 failed in the runner, not the engine (Setup).

## Question

Does edge-dit.cpp's `ucache` step cache, at its default parameters, skip or reuse transformer work in the 50-step CFG klein Base placement [jetson-edgedit-flux-klein-base-001](jetson-edgedit-flux-klein-base-001.md), and what does it change in latency, memory and output? Single change against [jetson-edgedit-flux-klein-base-q4-embq8-512-vaetile.json](../configs/jetson-edgedit-flux-klein-base-q4-embq8-512-vaetile.json). Step-reuse lever under a quality requirement (RQ2).

## Setup

- **Config**: [jetson-edgedit-flux-klein-base-q4-embq8-512-vaetile-ucache.json](../configs/jetson-edgedit-flux-klein-base-q4-embq8-512-vaetile-ucache.json); smoke [jetson-edgedit-flux-klein-base-q4-embq8-512-vaetile-ucache-smoke.json](../configs/jetson-edgedit-flux-klein-base-q4-embq8-512-vaetile-ucache-smoke.json) (first + one measured). `reference_config` is the base-001 config.
- **Change vs reference (only one)**: `optimizations.step_cache: {"method": "ucache"}`, mapped by the runner to `--cache_method ucache` with no tuning flags, so the engine's per-method defaults apply ([command.json](../results/runs/jetson-edgedit-flux-klein-base-003__baseline__20261015-213038/command.json)). The engine log does not print the default parameter values.
- Everything else as in the [base-001 setup](jetson-edgedit-flux-klein-base-001.md#setup): edge-dit.cpp `97cc585`, same `ed-sample` binary; Base Q4_0 transformer, Q4_K_M encoder with Q8_0 token embedding (compressed, separately labelled), BF16 VAE, all weights resident; flash attention on; VAE tiling on; 512×512, 50 Euler steps, guidance 4.0, CFG 4.0, seed 0, one prompt; 25W.
- **Protocol**: one `ed-sample` process, one load, `--repeat 14` (first, 3 warm-ups, 10 measured); smoke `--repeat 2`. [jetson_gap_queue_20261016f.sh](../scripts/jetson_gap_queue_20261016f.sh) runs the full protocol only when the smoke's `step-cache.json` reports skipped/reused work (here [26, 26]); each job under [jetson_memguard.sh](../scripts/jetson_memguard.sh) (400 MB floor, 768 MB swap-growth limit).
- **Cache counts are per transformer pass.** With CFG each step runs two transformer passes (the engine reports `variants=2`), 100 per image. The engine line `<Method> reused|skipped k/50 steps` uses the step count as denominator, but k counts passes (MagCache prints k = 70 > 50 on this workload); `step-cache.json → skipped_steps_per_generation` copies k unchanged.
- **First attempt** [jetson-edgedit-flux-klein-base-003__attempt__20261015-160624](../results/runs/jetson-edgedit-flux-klein-base-003__attempt__20261015-160624/) (2026-10-15): `failed`, `KeyError('denoise_step_4_ms')`. The runner's log parser reused the step-count variable name inside its stage loop, so a 50-step run looked up `denoise_step_4_ms`; corrected in [run_edgedit_jetson.py](../scripts/run_edgedit_jetson.py) (sha256 `153d3ea9…` for these runs) before the rerun. Not a measurement.
- Metric derivation: [Jetson edge-dit runner](../wiki/methods/baseline-metrics.md#jetson-edge-dit-runner) (wall from `ed-sample` stdout at 1 ms rounding; stages from the engine breakdown at 10 ms rounding; no per-step times; whole-system tegrastats only; only the final pass's image retained).

## Results

Full run [jetson-edgedit-flux-klein-base-003__baseline__20261015-213038](../results/runs/jetson-edgedit-flux-klein-base-003__baseline__20261015-213038/): [status.json](../results/runs/jetson-edgedit-flux-klein-base-003__baseline__20261015-213038/status.json) `complete`; [memguard.json](../results/runs/jetson-edgedit-flux-klein-base-003__baseline__20261015-213038/memguard.json) not killed, exit 0.

Cache receipt ([engine.log](../results/runs/jetson-edgedit-flux-klein-base-003__baseline__20261015-213038/engine.log), [step-cache.json](../results/runs/jetson-edgedit-flux-klein-base-003__baseline__20261015-213038/step-cache.json)): every pass logs `cache enabled: mode=UCache model=flux2-klein granularity=output matched_sites=1 variants=2 slots=1` and `UCache skipped 26/50 steps (2.08x estimated speedup)` (14 of 14 passes; smoke 2 of 2; source `ucache_policy.cpp`). The log does not say which passes were affected.

### Latency (measured n = 10; [summary.json](../results/runs/jetson-edgedit-flux-klein-base-003__baseline__20261015-213038/summary.json), [runs.csv](../results/runs/jetson-edgedit-flux-klein-base-003__baseline__20261015-213038/runs.csv), [stages.csv](../results/runs/jetson-edgedit-flux-klein-base-003__baseline__20261015-213038/stages.csv))

| Stage | Median (ms) | Min–max (ms) | Share of median wall | base-001 median (ms) | Δ |
|---|---:|---:|---:|---:|---:|
| wall (1 ms rounding) | **173627.0** | 173262.0–174007.0 | 100% | 232499.0 | −58872.0 (−25.3%) |
| text_encode | 925.0 | 920.0–930.0 | 0.5% | 930.0 | −5.0 |
| denoise (engine `sampling`, 50 steps) | 169910.0 | 169570.0–170310.0 | 97.9% | 228770.0 | −58860.0 (−25.7%) |
| vae_decode (tiled) | 1740.0 | 1730.0–1750.0 | 1.0% | 1750.0 | −10.0 |
| other | 1040.0 | 1030.0–1050.0 | 0.6% | 1050.0 | −10.0 |
| First generation (wall) | 179553.0 | text 2730.0 / denoise 171420.0 / vae 4300.0 / other 1100.0 | – | 238765.0 | −59212.0 (−24.8%) |
| Model load (s) | 63.452 | – | – | 63.132 | – |

Wall stdev 267.2 ms; warm-ups 173275.0, 173635.0, 173378.0 ms. Denoise per computed pass (derived): 169910.0 / (100 − 26) = 2296.1 ms; reference 228770.0 / 100 = 2287.7 ms. Measured denoise ratio 0.743; computed-pass fraction (100 − 26)/100 = 0.74.

### Memory (whole-system tegrastats at 200 ms, whole capture including load)

| Quantity | 003 | base-001 |
|---|---|---|
| RAM peak (tegrastats) | 6.725 GiB of 7.44 GiB | 6.731 GiB |
| Swap (tegrastats) min–max | 0.309–0.318 GiB | 0.311–0.311 GiB |
| MemAvailable (guard), start → minimum | 5942 → 568 MB | 5996 → 568 MB |
| GPU temperature max | 65.0 °C | 65.2 °C |

### Smoke run

[jetson-edgedit-flux-klein-base-003__attempt__20261015-212203](../results/runs/jetson-edgedit-flux-klein-base-003__attempt__20261015-212203/): complete, guard not triggered; UCache 26 in both passes. First 178632.0 ms (text 2180.0 / denoise 170920.0 / vae 4430.0 / other 1080.0); one measured 173173.0 ms. Load 63.410 s. RAM peak 6.608 GiB; MemAvailable minimum 647 MB. Not a baseline (n = 1).

### Repeatability and functional check (visual, not a quality metric)

Final-pass image (run index 13) RGB SHA256 `3e2b21b0656a28e4…`; the smoke's final image has the same hash. It differs from the reference final image (`d6f0a13ed741a535…`): mean absolute RGB difference 7.8 levels; 35.3% of pixels differ by more than 8 levels in any channel (computed from the two `output.png` files).

Visual check of `output.png` against the reference's: same cat, sign, script lettering and background as the reference; the face differs in small details. The sign reads "hello world". No stray mark, no blur.

## Observations

- The engine skipped 26 of 100 transformer passes in every generation; denoise fell from 228770.0 to 169910.0 ms (−25.7%) and wall from 232499.0 to 173627.0 ms (−25.3%).
- Measured denoise ratio 0.743 against a computed-pass fraction of 0.74; time per computed pass 2296.1 ms against 2287.7 ms in the reference (+0.4%).
- Text encode, VAE decode and other medians are within 10 ms of the reference.
- RAM peak 6.725 GiB (−0.007 GiB vs base-001); MemAvailable minimum 568 MB (+0 MB); swap did not grow.
- Output deterministic across the smoke and full processes (final-image hash); 7.8 mean RGB levels from the reference image.

## Interpretation

- Sampling is 98.4% of the reference wall, so the wall saving follows the number of passes the cache removes; the per-pass cost stays within 3% of the reference, so the cache's own bookkeeping is small on this device.
- Hypothesis: UCache's default criterion is stricter than EasyCache's on this schedule (26 vs 29 skipped passes, smaller pixel difference); the default thresholds are not logged.
- Whether the changed pixels meet a declared quality requirement needs a quality metric over more than one prompt; this record has one image.

## Next experiment

- Copy of `jetson-edgedit-flux-klein-base-q4-embq8-512-vaetile-ucache.json` with a paired saved-image quality metric against base-001 (calibration prompts only).
