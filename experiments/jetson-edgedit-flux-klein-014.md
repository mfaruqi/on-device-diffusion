---
type: experiment-record
id: jetson-edgedit-flux-klein-014
status: complete
device: jetson-orin-nano
engine: edge-dit-cpp
runs: [jetson-edgedit-flux-klein-014__baseline__20261015-135841, jetson-edgedit-flux-klein-014__attempt__20261015-135524]
updated: 2026-10-15
---

# jetson-edgedit-flux-klein-014: edge-dit.cpp step cache `dicache` (engine defaults); 0 of 4 steps reused, denoise +2.3%

Status: **complete** (2026-10-15, full run 17:58:41–18:03:57 UTC; smoke 17:55:24–17:58:10 UTC). Full 1+3+10 protocol; no memory-guard intervention (lowest MemAvailable 430 MB, 30 MB above the floor). The engine reused no step in any of the 16 passes.

## Question

Does edge-dit.cpp's DiCache step cache, at its default parameters, reuse any of the 4 denoising steps of the [006](jetson-edgedit-flux-klein-006.md) placement, and what does it change in latency, memory and output? Single change against [jetson-edgedit-flux-klein-q4-embq8-512-vaetile.json](../configs/jetson-edgedit-flux-klein-q4-embq8-512-vaetile.json). Step-reuse lever (RQ2).

## Setup

- **Config**: [jetson-edgedit-flux-klein-q4-embq8-512-vaetile-dicache.json](../configs/jetson-edgedit-flux-klein-q4-embq8-512-vaetile-dicache.json); smoke [jetson-edgedit-flux-klein-q4-embq8-512-vaetile-dicache-smoke.json](../configs/jetson-edgedit-flux-klein-q4-embq8-512-vaetile-dicache-smoke.json) (first + one measured). The run-directory `config.json` copies equal these files; `reference_config` is the 006 config.
- **Change vs reference (only one)**: `optimizations.step_cache: {"method": "dicache"}`, mapped by the runner to `--cache_method dicache` with no tuning flags, so the engine's per-method defaults apply ([command.json](../results/runs/jetson-edgedit-flux-klein-014__baseline__20261015-135841/command.json)). The engine log does not print the default parameter values.
- Everything else as in [006 setup](jetson-edgedit-flux-klein-006.md#setup): edge-dit.cpp `97cc585`, same `ed-sample` binary hash; Q4_0 transformer, **Q4_K_M encoder with Q8_0 token embedding (compressed, separately labelled checkpoint)**, BF16 VAE, all weights resident; flash attention on; VAE tiling on (25 tiles every pass); 512×512, 4 Euler steps (same sigmas as 006 in every pass), guidance 1.0, seed 0, one prompt; 25W.
- **Protocol**: one `ed-sample` process, one load, `--repeat 14` (first, 3 warm-ups, 10 measured); smoke `--repeat 2`.
- **Runner and guard**: [run_edgedit_jetson.py](../scripts/run_edgedit_jetson.py) at sha256 `fcd45913…` ([source-provenance.json](../results/runs/jetson-edgedit-flux-klein-014__baseline__20261015-135841/source-provenance.json)), launched by [jetson_gap_queue_20261015d.sh](../scripts/jetson_gap_queue_20261015d.sh) under [jetson_memguard.sh](../scripts/jetson_memguard.sh) (400 MB floor, 768 MB swap-growth limit, 0.1 s poll). The queue's `cache_screen` runs the full protocol unless the smoke's `step-cache.json` sums to exactly 0 skipped steps; a null count also triggers it.
- **Step-cache receipt (runner note)**: the engine prints `<Method> reused k/4 steps` for this method. At run time the runner's regex matched only `<method> skipped k/4 steps`, so `step-cache.json` reports `skipped_steps_per_generation: null` in both runs, and that null is what sent the method to the full protocol. The counts below are read from `engine.log`. The runner regex was widened to `skipped|reused` on 2026-10-15 (current [run_edgedit_jetson.py](../scripts/run_edgedit_jetson.py)); these runs predate the change.
- Metric derivation (wall from `ed-sample` stdout at 1 ms rounding; stages from the engine `flux generate breakdown` line at 10 ms rounding; raw `[[phase]]` intervals in `stages.csv`; whole-system tegrastats only; only the final pass's image retained): [006 setup](jetson-edgedit-flux-klein-006.md#setup), [edge-dit adapter](../wiki/methods/baseline-metrics.md#edge-dit-ed-sample-adapter), [Jetson repeated baseline memory](../wiki/methods/baseline-metrics.md#jetson-repeated-baseline-memory).

## Results

Full run [jetson-edgedit-flux-klein-014__baseline__20261015-135841](../results/runs/jetson-edgedit-flux-klein-014__baseline__20261015-135841/): [status.json](../results/runs/jetson-edgedit-flux-klein-014__baseline__20261015-135841/status.json) `complete`; [memguard.json](../results/runs/jetson-edgedit-flux-klein-014__baseline__20261015-135841/memguard.json) not killed, exit 0.

Cache receipt ([engine.log](../results/runs/jetson-edgedit-flux-klein-014__baseline__20261015-135841/engine.log)): every pass logs `cache enabled: mode=DiCache model=flux2-klein granularity=probe matched_sites=2 variants=3 slots=4` and `DiCache reused 0/4 steps` (14 of 14 passes; smoke 2 of 2).

### Latency (measured n = 10; [summary.json](../results/runs/jetson-edgedit-flux-klein-014__baseline__20261015-135841/summary.json), [runs.csv](../results/runs/jetson-edgedit-flux-klein-014__baseline__20261015-135841/runs.csv))

| Stage | Median (ms) | Min–max (ms) | Share of median wall | 006 median (ms) | Δ |
|---|---:|---:|---:|---:|---:|
| wall (1 ms rounding) | 12174.5 | 12135.0–12204.0 | 100% | 11962.5 | +212.0 (+1.8%) |
| text_encode (10 ms rounding) | 920.0 | 910.0–930.0 | 7.6% | 930.0 | −10.0 |
| denoise (engine `sampling`, 4 steps) | 9395.0 | 9370.0–9430.0 | 77.2% | 9185.0 | +210.0 (+2.3%) |
| vae_decode (tiled) | 1730.0 | 1720.0–1760.0 | 14.2% | 1740.0 | −10.0 |
| other | 120.0 | 110.0–120.0 | 1.0% | 120.0 | 0 |
| First generation (wall) | 18037.0 | text 2420.0 / denoise 10880.0 / vae 4570.0 / other 140.0 | – | 17395.0 | +642.0 (+3.7%) |
| Model load (s) | 65.951 | – | – | 64.706 | – |

Wall stdev 23.6 ms; warm-ups 12277.0, 12219.0, 12198.0 ms. The measured denoise ranges of 014 (9370–9430 ms) and 006 (9160–9230 ms) do not overlap. Raw marker medians ([stages.csv](../results/runs/jetson-edgedit-flux-klein-014__baseline__20261015-135841/stages.csv)): encode_setup 1012.4 (1006.0–1027.8), denoise 9392.9 (9373.0–9426.2), vae_decode 1728.4 (1718.3–1755.1) ms. Per-step times are not available from this engine (denoise / 4 = 2349 ms).

### Memory (whole-system tegrastats, 1191 samples at 200 ms, whole capture including load)

| Quantity | 014 | 006 |
|---|---|---|
| RAM peak (tegrastats) | 7039 MB = 6.874 GiB of 7.44 GiB | 6881 MB = 6.720 GiB |
| Swap (tegrastats) | 316 MB throughout (0.309 GiB) | 297 MB throughout |
| MemAvailable (guard), start → minimum | 5967 → **430 MB** | 5973 → 584 MB |
| GPU temperature max | 62.6 °C | 61.5 °C |

### Smoke run

[jetson-edgedit-flux-klein-014__attempt__20261015-135524](../results/runs/jetson-edgedit-flux-klein-014__attempt__20261015-135524/): complete, guard not triggered. First 18064.0 ms (2520.0 / 10820.0 / 4540.0 / 180.0); one measured 12233.0 ms (940.0 / 9440.0 / 1730.0 / 120.0). Load 65.038 s. RAM peak 6.805 GiB; MemAvailable minimum 481 MB; swap 316–317 MB.

### Repeatability and functional check

Final-pass image (run index 13) RGB SHA256 `7c7a84f3fe9988c8f43f76d656273ceb102b893cee89d4c14615116eeeaabd32`, identical to the 006 final image; the smoke's final image has the same hash. No quality metric.

## Observations

- DiCache was enabled (probe granularity, two matched sites, four slots) and reused 0 of 4 steps in every pass; output is bit-identical to 006.
- Denoising is 210 ms (+2.3%) slower than in 006, with non-overlapping ranges; the other stages are unchanged within rounding.
- RAM peak is 158 MB higher than in 006 and the MemAvailable minimum is 430 MB, 154 MB lower than 006's and 30 MB above the guard floor.

## Interpretation

- At default parameters DiCache costs time and memory without reusing a step on this schedule: its probe computation and cache slots run every step, while the reuse decision never fires.
- Hypothesis: the +210 ms per generation (about 52 ms per step) is the probe evaluation at the two matched sites, and the extra ~150 MB is the four cache slots; the engine does not report buffer sizes, so neither is confirmed.
- With a 30 MB margin to the guard floor, this configuration is at the edge of what this placement admits under the stated guard.

## Next experiment

- None planned at default parameters; a tuned DiCache would be a new labelled copy of `jetson-edgedit-flux-klein-q4-embq8-512-vaetile-dicache.json` with its threshold stated.
