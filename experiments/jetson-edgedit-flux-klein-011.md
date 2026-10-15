---
type: experiment-record
id: jetson-edgedit-flux-klein-011
status: complete
device: jetson-orin-nano
engine: edge-dit-cpp
runs: [jetson-edgedit-flux-klein-011__baseline__20261015-133915, jetson-edgedit-flux-klein-011__attempt__20261015-133558]
updated: 2026-10-15
---

# jetson-edgedit-flux-klein-011: edge-dit.cpp step cache `taylorseer` (engine defaults); 0 of 4 steps reused, latency and output unchanged

Status: **complete** (2026-10-15, full run 17:39:15–17:44:27 UTC; smoke 17:35:58–17:38:45 UTC). Full 1+3+10 protocol; no memory-guard intervention. The engine reused no step in any of the 16 passes.

## Question

Does edge-dit.cpp's TaylorSeer step cache, at its default parameters, reuse any of the 4 denoising steps of the [006](jetson-edgedit-flux-klein-006.md) placement, and what does it change in latency, memory and output? Single change against [jetson-edgedit-flux-klein-q4-embq8-512-vaetile.json](../configs/jetson-edgedit-flux-klein-q4-embq8-512-vaetile.json). Step-reuse lever (RQ2).

## Setup

- **Config**: [jetson-edgedit-flux-klein-q4-embq8-512-vaetile-taylorseer.json](../configs/jetson-edgedit-flux-klein-q4-embq8-512-vaetile-taylorseer.json); smoke [jetson-edgedit-flux-klein-q4-embq8-512-vaetile-taylorseer-smoke.json](../configs/jetson-edgedit-flux-klein-q4-embq8-512-vaetile-taylorseer-smoke.json) (first + one measured). The run-directory `config.json` copies equal these files; `reference_config` is the 006 config.
- **Change vs reference (only one)**: `optimizations.step_cache: {"method": "taylorseer"}`, mapped by the runner to `--cache_method taylorseer` with no tuning flags, so the engine's per-method defaults apply ([command.json](../results/runs/jetson-edgedit-flux-klein-011__baseline__20261015-133915/command.json)). The engine log does not print the default parameter values.
- Everything else as in [006 setup](jetson-edgedit-flux-klein-006.md#setup): edge-dit.cpp `97cc585`, same `ed-sample` binary hash; Q4_0 transformer, **Q4_K_M encoder with Q8_0 token embedding (compressed, separately labelled checkpoint)**, BF16 VAE, all weights resident; flash attention on; VAE tiling on (25 tiles every pass); 512×512, 4 Euler steps (same sigmas as 006 in every pass), guidance 1.0, seed 0, one prompt; 25W.
- **Protocol**: one `ed-sample` process, one load, `--repeat 14` (first, 3 warm-ups, 10 measured); smoke `--repeat 2`.
- **Runner and guard**: [run_edgedit_jetson.py](../scripts/run_edgedit_jetson.py) at sha256 `fcd45913…` ([source-provenance.json](../results/runs/jetson-edgedit-flux-klein-011__baseline__20261015-133915/source-provenance.json)), launched by [jetson_gap_queue_20261015d.sh](../scripts/jetson_gap_queue_20261015d.sh) under [jetson_memguard.sh](../scripts/jetson_memguard.sh) (400 MB floor, 768 MB swap-growth limit, 0.1 s poll). The queue's `cache_screen` runs the full protocol unless the smoke's `step-cache.json` sums to exactly 0 skipped steps; a null count also triggers it.
- **Step-cache receipt (runner note)**: the engine prints `<Method> reused k/4 steps` for this method. At run time the runner's regex matched only `<method> skipped k/4 steps`, so `step-cache.json` reports `skipped_steps_per_generation: null` in both runs, and that null is what sent the method to the full protocol. The counts below are read from `engine.log`. The runner regex was widened to `skipped|reused` on 2026-10-15 (current [run_edgedit_jetson.py](../scripts/run_edgedit_jetson.py)); these runs predate the change.
- Metric derivation (wall from `ed-sample` stdout at 1 ms rounding; stages from the engine `flux generate breakdown` line at 10 ms rounding; raw `[[phase]]` intervals in `stages.csv`; whole-system tegrastats only; only the final pass's image retained): [006 setup](jetson-edgedit-flux-klein-006.md#setup), [edge-dit adapter](../wiki/methods/baseline-metrics.md#edge-dit-ed-sample-adapter), [Jetson repeated baseline memory](../wiki/methods/baseline-metrics.md#jetson-repeated-baseline-memory).

## Results

Full run [jetson-edgedit-flux-klein-011__baseline__20261015-133915](../results/runs/jetson-edgedit-flux-klein-011__baseline__20261015-133915/): [status.json](../results/runs/jetson-edgedit-flux-klein-011__baseline__20261015-133915/status.json) `complete`; [memguard.json](../results/runs/jetson-edgedit-flux-klein-011__baseline__20261015-133915/memguard.json) not killed, exit 0.

Cache receipt ([engine.log](../results/runs/jetson-edgedit-flux-klein-011__baseline__20261015-133915/engine.log)): every pass logs `cache enabled: mode=TaylorSeer model=flux2-klein granularity=feature matched_sites=1 variants=2 slots=1` and `TaylorSeer reused 0/4 steps` (14 of 14 passes; smoke 2 of 2).

### Latency (measured n = 10; [summary.json](../results/runs/jetson-edgedit-flux-klein-011__baseline__20261015-133915/summary.json), [runs.csv](../results/runs/jetson-edgedit-flux-klein-011__baseline__20261015-133915/runs.csv))

| Stage | Median (ms) | Min–max (ms) | Share of median wall | 006 median (ms) | Δ |
|---|---:|---:|---:|---:|---:|
| wall (1 ms rounding) | 11973.0 | 11946.0–12003.0 | 100% | 11962.5 | +10.5 (+0.1%) |
| text_encode (10 ms rounding) | 920.0 | 910.0–940.0 | 7.7% | 930.0 | −10.0 |
| denoise (engine `sampling`, 4 steps) | 9200.0 | 9180.0–9220.0 | 76.8% | 9185.0 | +15.0 (+0.2%) |
| vae_decode (tiled) | 1730.0 | 1730.0–1750.0 | 14.4% | 1740.0 | −10.0 |
| other | 120.0 | 110.0–120.0 | 1.0% | 120.0 | 0 |
| First generation (wall) | 17583.0 | text 2340.0 / denoise 10650.0 / vae 4440.0 / other 150.0 | – | 17395.0 | +188.0 (+1.1%) |
| Model load (s) | 65.215 | – | – | 64.706 | – |

Wall stdev 22.0 ms; warm-ups 12056.0, 12029.0, 11999.0 ms. Raw marker medians ([stages.csv](../results/runs/jetson-edgedit-flux-klein-011__baseline__20261015-133915/stages.csv)): encode_setup 1013.4 (1006.8–1028.3), denoise 9201.4 (9182.4–9221.7), vae_decode 1729.4 (1726.0–1750.7) ms. Per-step times are not available from this engine (steady per-step ≈ denoise / 4 = 2300 ms).

### Memory (whole-system tegrastats, 1174 samples at 200 ms, whole capture including load)

| Quantity | 011 | 006 |
|---|---|---|
| RAM peak (tegrastats) | 6930 MB = 6.768 GiB of 7.44 GiB | 6881 MB = 6.720 GiB |
| Swap (tegrastats) | 314 MB throughout (0.307 GiB) | 297 MB throughout |
| MemAvailable (guard), start → minimum | 5982 → 540 MB | 5973 → 584 MB |
| GPU temperature max | 62.3 °C | 61.5 °C |

### Smoke run

[jetson-edgedit-flux-klein-011__attempt__20261015-133558](../results/runs/jetson-edgedit-flux-klein-011__attempt__20261015-133558/): complete, guard not triggered. First 17667.0 ms (2350.0 / 10740.0 / 4410.0 / 170.0); one measured 12069.0 ms (930.0 / 9270.0 / 1750.0 / 120.0). Load 65.279 s. RAM peak 6.683 GiB; MemAvailable minimum 625 MB; swap 314 MB.

### Repeatability and functional check

Final-pass image (run index 13) RGB SHA256 `7c7a84f3fe9988c8f43f76d656273ceb102b893cee89d4c14615116eeeaabd32`, identical to the 006 final image; the smoke's final image has the same hash. Visual check of `output.png` (not a quality metric): identical to 006's ("helllo world" sign, as in 006). No quality metric.

## Observations

- TaylorSeer was enabled (feature granularity, one matched site) and reused 0 of 4 steps in every pass; output is bit-identical to 006.
- Latency medians are within 15 ms of 006 in every stage; first generation +188 ms.
- RAM peak is 49 MB higher and the MemAvailable minimum 44 MB lower than in 006 (single runs each).

## Interpretation

- At default parameters, TaylorSeer is inert for this 4-step schedule; this configuration is 006 plus cache bookkeeping, with no measurable latency cost.
- Hypothesis: the default policy needs more computed steps (to fit the Taylor terms) than a 4-step schedule provides before it predicts; the engine's default skip/order settings (`--cache-taylor-skip`, `--cache-taylor-order`) were not logged.
- Hypothesis: the small memory difference is cache state allocated per matched site; with two single runs it is not separable from run-to-run variation.

## Next experiment

- Copy of `jetson-edgedit-flux-klein-q4-embq8-512-vaetile-taylorseer.json` with explicit `--cache-taylor-skip`/`--cache-taylor-order` settings that allow one predicted step (needs a runner option; separately labelled).
