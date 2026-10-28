---
type: experiment-record
id: jetson-edgedit-flux-klein-base-002
status: complete
device: jetson-orin-nano
engine: edge-dit-cpp
runs: [jetson-edgedit-flux-klein-base-002__baseline__20261015-204021, jetson-edgedit-flux-klein-base-002__attempt__20261015-203159, jetson-edgedit-flux-klein-base-002__attempt__20261015-155801]
updated: 2026-10-16
---

# jetson-edgedit-flux-klein-base-002: edge-dit.cpp step cache `easycache` (engine defaults), klein Base Q4_0 512², 50 steps CFG 4; 29 of 100 transformer passes skipped, wall −28.3%

Status: **complete** (2026-10-16; full run 00:40:21–01:21:33 UTC, smoke 00:31:59–00:39:51 UTC). Full 1+3+10 protocol; no memory-guard intervention. A first attempt on 2026-10-15 failed in the runner, not the engine (Setup).

## Question

Does edge-dit.cpp's `easycache` step cache, at its default parameters, skip or reuse transformer work in the 50-step CFG klein Base placement [jetson-edgedit-flux-klein-base-001](jetson-edgedit-flux-klein-base-001.md), and what does it change in latency, memory and output? Single change against [jetson-edgedit-flux-klein-base-q4-embq8-512-vaetile.json](../configs/jetson-edgedit-flux-klein-base-q4-embq8-512-vaetile.json). Step-reuse lever under a quality requirement (RQ2).

## Setup

- **Config**: [jetson-edgedit-flux-klein-base-q4-embq8-512-vaetile-easycache.json](../configs/jetson-edgedit-flux-klein-base-q4-embq8-512-vaetile-easycache.json); smoke [jetson-edgedit-flux-klein-base-q4-embq8-512-vaetile-easycache-smoke.json](../configs/jetson-edgedit-flux-klein-base-q4-embq8-512-vaetile-easycache-smoke.json) (first + one measured). `reference_config` is the base-001 config.
- **Change vs reference (only one)**: `optimizations.step_cache: {"method": "easycache"}`, mapped by the runner to `--cache_method easycache` with no tuning flags, so the engine's per-method defaults apply ([command.json](../results/runs/jetson-edgedit-flux-klein-base-002__baseline__20261015-204021/command.json)). The engine log does not print the default parameter values.
- Everything else as in the [base-001 setup](jetson-edgedit-flux-klein-base-001.md#setup): edge-dit.cpp `97cc585`, same `ed-sample` binary; Base Q4_0 transformer, Q4_K_M encoder with Q8_0 token embedding (compressed, separately labelled), BF16 VAE, all weights resident; flash attention on; VAE tiling on; 512×512, 50 Euler steps, guidance 4.0, CFG 4.0, seed 0, one prompt; 25W.
- **Protocol**: one `ed-sample` process, one load, `--repeat 14` (first, 3 warm-ups, 10 measured); smoke `--repeat 2`. [jetson_gap_queue_20261016f.sh](../scripts/jetson_gap_queue_20261016f.sh) runs the full protocol only when the smoke's `step-cache.json` reports skipped/reused work (here [29, 29]); each job under [jetson_memguard.sh](../scripts/jetson_memguard.sh) (400 MB floor, 768 MB swap-growth limit).
- **Cache counts are per transformer pass.** With CFG each step runs two transformer passes (the engine reports `variants=2`), 100 per image. The engine line `<Method> reused|skipped k/50 steps` uses the step count as denominator, but k counts passes (MagCache prints k = 70 > 50 on this workload); `step-cache.json → skipped_steps_per_generation` copies k unchanged.
- **First attempt** [jetson-edgedit-flux-klein-base-002__attempt__20261015-155801](../results/runs/jetson-edgedit-flux-klein-base-002__attempt__20261015-155801/) (2026-10-15): `failed`, `KeyError('denoise_step_4_ms')`. The runner's log parser reused the step-count variable name inside its stage loop, so a 50-step run looked up `denoise_step_4_ms`; corrected in [run_edgedit_jetson.py](../scripts/run_edgedit_jetson.py) (sha256 `153d3ea9…` for these runs) before the rerun. Not a measurement.
- Metric derivation: [Jetson edge-dit runner](../wiki/methods/baseline-metrics.md#jetson-edge-dit-runner) (wall from `ed-sample` stdout at 1 ms rounding; stages from the engine breakdown at 10 ms rounding; no per-step times; whole-system tegrastats only; only the final pass's image retained).

## Results

Full run [jetson-edgedit-flux-klein-base-002__baseline__20261015-204021](../results/runs/jetson-edgedit-flux-klein-base-002__baseline__20261015-204021/): [status.json](../results/runs/jetson-edgedit-flux-klein-base-002__baseline__20261015-204021/status.json) `complete`; [memguard.json](../results/runs/jetson-edgedit-flux-klein-base-002__baseline__20261015-204021/memguard.json) not killed, exit 0.

Cache receipt ([engine.log](../results/runs/jetson-edgedit-flux-klein-base-002__baseline__20261015-204021/engine.log), [step-cache.json](../results/runs/jetson-edgedit-flux-klein-base-002__baseline__20261015-204021/step-cache.json)): every pass logs `cache enabled: mode=EasyCache model=flux2-klein granularity=output matched_sites=1 variants=2 slots=1` and `EasyCache skipped 29/50 steps (2.38x estimated speedup)` (14 of 14 passes; smoke 2 of 2; source `easycache_policy.cpp`). The log does not say which passes were affected.

### Latency (measured n = 10; [summary.json](../results/runs/jetson-edgedit-flux-klein-base-002__baseline__20261015-204021/summary.json), [runs.csv](../results/runs/jetson-edgedit-flux-klein-base-002__baseline__20261015-204021/runs.csv), [stages.csv](../results/runs/jetson-edgedit-flux-klein-base-002__baseline__20261015-204021/stages.csv))

| Stage | Median (ms) | Min–max (ms) | Share of median wall | base-001 median (ms) | Δ |
|---|---:|---:|---:|---:|---:|
| wall (1 ms rounding) | **166730.5** | 166240.0–167220.0 | 100% | 232499.0 | −65768.5 (−28.3%) |
| text_encode | 920.0 | 910.0–930.0 | 0.6% | 930.0 | −10.0 |
| denoise (engine `sampling`, 50 steps) | 163025.0 | 162540.0–163540.0 | 97.8% | 228770.0 | −65745.0 (−28.7%) |
| vae_decode (tiled) | 1750.0 | 1730.0–1760.0 | 1.0% | 1750.0 | +0.0 |
| other | 1040.0 | 1040.0–1050.0 | 0.6% | 1050.0 | −10.0 |
| First generation (wall) | 171297.0 | text 1600.0 / denoise 164090.0 / vae 4550.0 / other 1060.0 | – | 238765.0 | −67468.0 (−28.3%) |
| Model load (s) | 62.960 | – | – | 63.132 | – |

Wall stdev 338.9 ms; warm-ups 166377.0, 166288.0, 166334.0 ms. Denoise per computed pass (derived): 163025.0 / (100 − 29) = 2296.1 ms; reference 228770.0 / 100 = 2287.7 ms. Measured denoise ratio 0.713; computed-pass fraction (100 − 29)/100 = 0.71.

### Memory (whole-system tegrastats at 200 ms, whole capture including load)

| Quantity | 002 | base-001 |
|---|---|---|
| RAM peak (tegrastats) | 6.708 GiB of 7.44 GiB | 6.731 GiB |
| Swap (tegrastats) min–max | 0.309–0.309 GiB | 0.311–0.311 GiB |
| MemAvailable (guard), start → minimum | 5989 → 581 MB | 5996 → 568 MB |
| GPU temperature max | 65.2 °C | 65.2 °C |

### Smoke run

[jetson-edgedit-flux-klein-base-002__attempt__20261015-203159](../results/runs/jetson-edgedit-flux-klein-base-002__attempt__20261015-203159/): complete, guard not triggered; EasyCache 29 in both passes. First 171683.0 ms (text 1610.0 / denoise 164300.0 / vae 4720.0 / other 1050.0); one measured 166399.0 ms. Load 63.098 s. RAM peak 6.650 GiB; MemAvailable minimum 635 MB. Not a baseline (n = 1).

### Repeatability and functional check (visual, not a quality metric)

Final-pass image (run index 13) RGB SHA256 `26e9921f452cb5a7…`; the smoke's final image has the same hash. It differs from the reference final image (`d6f0a13ed741a535…`): mean absolute RGB difference 11.9 levels; 46.9% of pixels differ by more than 8 levels in any channel (computed from the two `output.png` files).

Visual check of `output.png` against the reference's: same cat, cardboard sign, script lettering and grey background as the reference; the face and fur differ in small details, and a short stray stroke sits between the second "l" and the "o" of "hello". The sign reads "hello world". No blur.

## Observations

- The engine skipped 29 of 100 transformer passes in every generation; denoise fell from 228770.0 to 163025.0 ms (−28.7%) and wall from 232499.0 to 166730.5 ms (−28.3%).
- Measured denoise ratio 0.713 against a computed-pass fraction of 0.71; time per computed pass 2296.1 ms against 2287.7 ms in the reference (+0.4%).
- Text encode, VAE decode and other medians are within 10 ms of the reference.
- RAM peak 6.708 GiB (−0.023 GiB vs base-001); MemAvailable minimum 581 MB (+13 MB); swap did not grow.
- Output deterministic across the smoke and full processes (final-image hash); 11.9 mean RGB levels from the reference image.

## Interpretation

- Sampling is 98.4% of the reference wall, so the wall saving follows the number of passes the cache removes; the per-pass cost stays within 3% of the reference, so the cache's own bookkeeping is small on this device.
- Hypothesis: the stray stroke and the larger pixel difference (mean 11.9 levels, the largest of the six caches that kept the text) come from reusing output-level residuals (`granularity=output`) across consecutive passes; the log does not say which passes were skipped, so where the reuse happens is not known.
- Whether the changed pixels meet a declared quality requirement needs a quality metric over more than one prompt; this record has one image.

## Next experiment

- Copy of `jetson-edgedit-flux-klein-base-q4-embq8-512-vaetile-easycache.json` with a paired saved-image quality metric against base-001 (calibration prompts only).
