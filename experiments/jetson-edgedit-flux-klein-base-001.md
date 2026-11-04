---
type: experiment-record
id: jetson-edgedit-flux-klein-base-001
status: complete
device: jetson-orin-nano
engine: edge-dit-cpp
runs: [jetson-edgedit-flux-klein-base-001__baseline__20261015-193452, jetson-edgedit-flux-klein-base-001__attempt__20261015-192415, jetson-edgedit-flux-klein-base-001__attempt__20261015-154751]
updated: 2026-10-16
---

# jetson-edgedit-flux-klein-base-001: edge-dit.cpp klein Base reference, Q4_0 512², 50 steps CFG 4, all weights resident with VAE tiling; 232.5 s per image

Status: **complete** (2026-10-15/16; full run 2026-10-15 23:34:52 – 2026-10-16 00:31:29 UTC, smoke 2026-10-15 23:24:15–23:34:21 UTC). Full 1+3+10 protocol; no memory-guard intervention. A first attempt on 2026-10-15 failed in the runner, not the engine (Setup).

## Question

What does edge-dit.cpp cost per image on the Jetson for the klein Base workload (50 steps, CFG 4) with all weights resident, flash attention on and VAE tiling, with no step cache? Reference for the Base step-cache screens [base-002](jetson-edgedit-flux-klein-base-002.md) … [base-008](jetson-edgedit-flux-klein-base-008.md). Engine baseline for RQ1; reference for the step-reuse lever (RQ2).

## Setup

- **Config**: [jetson-edgedit-flux-klein-base-q4-embq8-512-vaetile.json](../configs/jetson-edgedit-flux-klein-base-q4-embq8-512-vaetile.json); smoke [jetson-edgedit-flux-klein-base-q4-embq8-512-vaetile-smoke.json](../configs/jetson-edgedit-flux-klein-base-q4-embq8-512-vaetile-smoke.json) (first + one measured). Reference configuration: no `reference_config`.
- **Engine**: edge-dit.cpp `97cc585`, CUDA performance build for sm_87 ([build evidence](../results/runs/jetson-edgedit-build-20261011/)), `ed-sample` sha256 `5936674c…`; `--backend cuda --type preserve --cache_method off --vae-tiling on --threads 4` ([command.json](../results/runs/jetson-edgedit-flux-klein-base-001__baseline__20261015-193452/command.json)).
- **Weights (compressed, separately labelled checkpoint)**: `leejet/FLUX.2-klein-base-4B-GGUF` Q4_0 transformer at `d1267112…` (weight-type stat `q4_0:80|bf16:69`); Qwen3-4B Q4_K_M encoder with the token embedding requantized Q6_K → Q8_0 ([conversion log](../results/runs/jetson-edgedit-convert-20261011/requant.log)); BF16 VAE from `black-forest-labs/FLUX.2-klein-4B` at `e7b7dc27…`. All weights resident on the GPU; FP32 activations.
- **Workload**: 512×512, 50 Euler steps, guidance 4.0, CFG 4.0 (two transformer passes per step, 100 per image), seed 0, batch 1, one prompt (`A cat holding a sign that says hello world`). VAE decode tiled (`processing 25 tiles` in every pass); flash attention on (command-line flag only; the engine log does not report it).
- **Protocol**: one `ed-sample` process, one load, `--repeat 14` (first, 3 warm-ups, 10 measured); smoke `--repeat 2`. Launched by [jetson_gap_queue_20261016f.sh](../scripts/jetson_gap_queue_20261016f.sh) under [jetson_memguard.sh](../scripts/jetson_memguard.sh) (400 MB MemAvailable floor, 768 MB swap-growth limit, 0.1 s poll); nothing else ran on the board. 25W power mode asserted by the runner.
- **Runner**: [run_edgedit_jetson.py](../scripts/run_edgedit_jetson.py) sha256 `153d3ea9…` ([source-provenance.json](../results/runs/jetson-edgedit-flux-klein-base-001__baseline__20261015-193452/source-provenance.json)).
- **First attempt** [jetson-edgedit-flux-klein-base-001__attempt__20261015-154751](../results/runs/jetson-edgedit-flux-klein-base-001__attempt__20261015-154751/): `failed`, `KeyError('denoise_step_4_ms')`. The runner's log parser reused the step-count variable name inside its stage loop, so a 50-step run looked up the 4-step key; corrected before the rerun. Not a measurement.
- Metric derivation: [Jetson edge-dit runner](../wiki/methods/baseline-metrics.md#jetson-edge-dit-runner) (wall from `ed-sample` stdout at 1 ms rounding; text / sampling / VAE / other from the engine `flux generate breakdown` line at 10 ms rounding; raw `[[phase]]` intervals in `stages.csv`; no per-step times; whole-system tegrastats at 200 ms; only the final pass's image retained).

## Results

Full run [jetson-edgedit-flux-klein-base-001__baseline__20261015-193452](../results/runs/jetson-edgedit-flux-klein-base-001__baseline__20261015-193452/): [status.json](../results/runs/jetson-edgedit-flux-klein-base-001__baseline__20261015-193452/status.json) `complete`; [memguard.json](../results/runs/jetson-edgedit-flux-klein-base-001__baseline__20261015-193452/memguard.json) not killed, exit 0. The engine log reports `cache mode   : original` and 700 `flux step` lines (50 × 14).

### Latency (measured n = 10; [summary.json](../results/runs/jetson-edgedit-flux-klein-base-001__baseline__20261015-193452/summary.json), [runs.csv](../results/runs/jetson-edgedit-flux-klein-base-001__baseline__20261015-193452/runs.csv), [stages.csv](../results/runs/jetson-edgedit-flux-klein-base-001__baseline__20261015-193452/stages.csv))

| Stage | Median (ms) | Min–max (ms) | Share of median wall |
|---|---:|---:|---:|
| wall (1 ms rounding) | **232499.0** | 232023.0–233674.0 (stdev 543.6) | 100% |
| text_encode (engine breakdown) | 930.0 | 910.0–930.0 | 0.4% |
| denoise (engine `sampling`, 50 steps, 100 passes) | 228770.0 | 228290.0–229950.0 | 98.4% |
| vae_decode (tiled) | 1750.0 | 1730.0–1750.0 | 0.8% |
| other | 1050.0 | 1030.0–1050.0 | 0.5% |
| `[[phase]]` markers: encode_setup / denoise / vae_decode | 1955.0 / 228770.4 / 1747.3 | 1922.4–1962.5 / 228291.0–229947.0 / 1728.6–1751.9 | – |
| First generation: wall | 238765.0 | – | text 1690.0 / denoise 231280.0 / vae 4730.0 / other 1060.0 |
| Warm-ups (3, discarded): wall | – | 232239.0–232700.0 | – |
| Model load (s) | 63.132 | – | page cache warm (files hashed just before launch) |

Derived: denoise per transformer pass 228770 / 100 = 2287.7 ms (per step 4575.4 ms).

### Memory (whole-system tegrastats, 16189 samples at 200 ms, whole capture including load)

| Quantity | Value |
|---|---:|
| RAM peak (tegrastats) | 6.731 GiB of 7.44 GiB |
| Swap (tegrastats) | 0.311 GiB throughout |
| MemAvailable (guard), start → minimum | 5996 → 568 MB |
| GPU temperature max | 65.2 °C |

Allocator metrics, per-generation device peaks and host RSS are unavailable for this runner.

### Smoke run

[jetson-edgedit-flux-klein-base-001__attempt__20261015-192415](../results/runs/jetson-edgedit-flux-klein-base-001__attempt__20261015-192415/): complete, guard not triggered (MemAvailable minimum 685 MB). First 238996.0 ms (text 2550.0 / denoise 230710.0 / vae 4610.0 / other 1110.0); one measured 232355.0 ms (930.0 / 228630.0 / 1750.0 / 1040.0). Load 65.080 s. RAM peak 6.630 GiB. Not a baseline (n = 1).

### Repeatability and functional check (visual, not a quality metric)

Final-pass image (run index 13) RGB SHA256 `d6f0a13ed741a535…`; the smoke's final image has the same hash, so the output reproduces across two processes. Visual check of `output.png`: a tabby cat holding a cardboard sign with script lettering reading "hello world", grey background; no artefact. No quality metric.

## Observations

- Sampling is 98.4% of the median wall; text encode, tiled VAE decode and other together are 3.7 s.
- The measured wall range is 1651 ms (0.7% of the median); the first generation is 6.3 s slower than the median, 4.3 s of it in sampling and 3.0 s in VAE decode.
- MemAvailable fell to 568 MB at its minimum (168 MB above the guard floor); swap did not grow.
- Output deterministic across the smoke and full processes.

## Interpretation

- With all weights resident and the prompt re-encoded each pass, this configuration is compute-bound in the transformer: 100 passes at about 2.29 s each set the per-image time, and only step-level levers (fewer computed passes, cheaper passes) can move it materially.
- The 168 MB margin above the guard floor leaves little room for any option that adds per-pass state (caches that store features or probes).

## Next experiment

- Copy of `jetson-edgedit-flux-klein-base-q4-embq8-512-vaetile.json` with a paired saved-image quality metric over the calibration prompts, as the reference for the cache screens.
