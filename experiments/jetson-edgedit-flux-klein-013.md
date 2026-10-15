---
type: experiment-record
id: jetson-edgedit-flux-klein-013
status: complete
device: jetson-orin-nano
engine: edge-dit-cpp
runs: [jetson-edgedit-flux-klein-013__baseline__20261015-135118, jetson-edgedit-flux-klein-013__attempt__20261015-134813]
updated: 2026-10-15
---

# jetson-edgedit-flux-klein-013: edge-dit.cpp step cache `magcache` (engine defaults); 3 of 4 steps reused, denoise −74.5%, output visibly degraded

Status: **complete** (2026-10-15, full run 17:51:18–17:54:54 UTC; smoke 17:48:13–17:50:47 UTC). Full 1+3+10 protocol; no memory-guard intervention. The engine reused 3 of 4 steps in every pass; the output image is visibly degraded.

## Question

Does edge-dit.cpp's MagCache step cache, at its default parameters, reuse any of the 4 denoising steps of the [006](jetson-edgedit-flux-klein-006.md) placement, and what does it change in latency, memory and output? Single change against [jetson-edgedit-flux-klein-q4-embq8-512-vaetile.json](../configs/jetson-edgedit-flux-klein-q4-embq8-512-vaetile.json). Step-reuse lever under a quality requirement (RQ2).

## Setup

- **Config**: [jetson-edgedit-flux-klein-q4-embq8-512-vaetile-magcache.json](../configs/jetson-edgedit-flux-klein-q4-embq8-512-vaetile-magcache.json); smoke [jetson-edgedit-flux-klein-q4-embq8-512-vaetile-magcache-smoke.json](../configs/jetson-edgedit-flux-klein-q4-embq8-512-vaetile-magcache-smoke.json) (first + one measured). The run-directory `config.json` copies equal these files; `reference_config` is the 006 config.
- **Change vs reference (only one)**: `optimizations.step_cache: {"method": "magcache"}`, mapped by the runner to `--cache_method magcache` with no tuning flags, so the engine's per-method defaults apply ([command.json](../results/runs/jetson-edgedit-flux-klein-013__baseline__20261015-135118/command.json)). The engine log does not print the default parameter values.
- Everything else as in [006 setup](jetson-edgedit-flux-klein-006.md#setup): edge-dit.cpp `97cc585`, same `ed-sample` binary hash; Q4_0 transformer, **Q4_K_M encoder with Q8_0 token embedding (compressed, separately labelled checkpoint)**, BF16 VAE, all weights resident; flash attention on; VAE tiling on (25 tiles every pass); 512×512, 4 Euler steps (same sigmas as 006 in every pass), guidance 1.0, seed 0, one prompt; 25W.
- **Protocol**: one `ed-sample` process, one load, `--repeat 14` (first, 3 warm-ups, 10 measured); smoke `--repeat 2`.
- **Runner and guard**: [run_edgedit_jetson.py](../scripts/run_edgedit_jetson.py) at sha256 `fcd45913…` ([source-provenance.json](../results/runs/jetson-edgedit-flux-klein-013__baseline__20261015-135118/source-provenance.json)), launched by [jetson_gap_queue_20261015d.sh](../scripts/jetson_gap_queue_20261015d.sh) under [jetson_memguard.sh](../scripts/jetson_memguard.sh) (400 MB floor, 768 MB swap-growth limit, 0.1 s poll). The queue's `cache_screen` runs the full protocol unless the smoke's `step-cache.json` sums to exactly 0 skipped steps; a null count also triggers it.
- **Step-cache receipt (runner note)**: the engine prints `<Method> reused k/4 steps` for this method. At run time the runner's regex matched only `<method> skipped k/4 steps`, so `step-cache.json` reports `skipped_steps_per_generation: null` in both runs, and that null is what sent the method to the full protocol. The counts below are read from `engine.log`. The runner regex was widened to `skipped|reused` on 2026-10-15 (current [run_edgedit_jetson.py](../scripts/run_edgedit_jetson.py)); these runs predate the change.
- Metric derivation (wall from `ed-sample` stdout at 1 ms rounding; stages from the engine `flux generate breakdown` line at 10 ms rounding; raw `[[phase]]` intervals in `stages.csv`; whole-system tegrastats only; only the final pass's image retained): [006 setup](jetson-edgedit-flux-klein-006.md#setup), [edge-dit adapter](../wiki/methods/baseline-metrics.md#edge-dit-ed-sample-adapter), [Jetson repeated baseline memory](../wiki/methods/baseline-metrics.md#jetson-repeated-baseline-memory).

## Results

Full run [jetson-edgedit-flux-klein-013__baseline__20261015-135118](../results/runs/jetson-edgedit-flux-klein-013__baseline__20261015-135118/): [status.json](../results/runs/jetson-edgedit-flux-klein-013__baseline__20261015-135118/status.json) `complete`; [memguard.json](../results/runs/jetson-edgedit-flux-klein-013__baseline__20261015-135118/memguard.json) not killed, exit 0.

Cache receipt ([engine.log](../results/runs/jetson-edgedit-flux-klein-013__baseline__20261015-135118/engine.log)): every pass logs `cache enabled: mode=MagCache model=flux2-klein granularity=feature matched_sites=1 variants=2 slots=1` and `MagCache reused 3/4 steps (4.00x)` (14 of 14 passes; smoke 2 of 2). The engine still prints `flux step 1/4` … `4/4` with the unchanged sigma schedule in every pass. The log does not say which three steps were reused; the "(4.00x)" figure is the engine's own ratio, not a measured speed-up.

### Latency (measured n = 10; [summary.json](../results/runs/jetson-edgedit-flux-klein-013__baseline__20261015-135118/summary.json), [runs.csv](../results/runs/jetson-edgedit-flux-klein-013__baseline__20261015-135118/runs.csv))

| Stage | Median (ms) | Min–max (ms) | Share of median wall | 006 median (ms) | Δ |
|---|---:|---:|---:|---:|---:|
| wall (1 ms rounding) | 5099.5 | 5086.0–5139.0 | 100% | 11962.5 | −6863.0 (−57.4%) |
| text_encode (10 ms rounding) | 910.0 | 910.0–930.0 | 17.8% | 930.0 | −20.0 |
| denoise (engine `sampling`, 4 steps) | 2340.0 | 2330.0–2350.0 | 45.9% | 9185.0 | −6845.0 (−74.5%) |
| vae_decode (tiled) | 1735.0 | 1720.0–1750.0 | 34.0% | 1740.0 | −5.0 |
| other | 120.0 | 110.0–120.0 | 2.4% | 120.0 | 0 |
| First generation (wall) | 10728.0 | text 2440.0 / denoise 3760.0 / vae 4370.0 / other 140.0 | – | 17395.0 | −6667.0 (−38.3%) |
| Model load (s) | 64.946 | – | – | 64.706 | – |

Wall stdev 19.7 ms; warm-ups 5215.0, 5155.0, 5141.0 ms. Raw marker medians ([stages.csv](../results/runs/jetson-edgedit-flux-klein-013__baseline__20261015-135118/stages.csv)): encode_setup 1007.7 (1000.6–1028.0), denoise 2337.0 (2329.1–2347.8), vae_decode 1731.7 (1720.3–1752.7) ms. Per-step times are not available from this engine; 2340 ms is close to one 006 step (9185 / 4 = 2296 ms).

### Memory (whole-system tegrastats, 708 samples at 200 ms, whole capture including load)

| Quantity | 013 | 006 |
|---|---|---|
| RAM peak (tegrastats) | 6933 MB = 6.771 GiB of 7.44 GiB | 6881 MB = 6.720 GiB |
| Swap (tegrastats) | 316 MB throughout (0.309 GiB) | 297 MB throughout |
| MemAvailable (guard), start → minimum | 5952 → 493 MB | 5973 → 584 MB |
| GPU temperature max | 59.0 °C | 61.5 °C |

### Smoke run

[jetson-edgedit-flux-klein-013__attempt__20261015-134813](../results/runs/jetson-edgedit-flux-klein-013__attempt__20261015-134813/): complete, guard not triggered. First 10732.0 ms (2480.0 / 3760.0 / 4320.0 / 170.0); one measured 5222.0 ms (940.0 / 2410.0 / 1750.0 / 120.0). Load 65.068 s. RAM peak 6.654 GiB; MemAvailable minimum 610 MB; swap 315 → 377 MB (guard).

### Repeatability and functional check (visual, not a quality metric)

Final-pass image (run index 13) RGB SHA256 `efe3a47297eed586b099bb95ac2bdd3ae104de39ad649416a53c976e7fc8a246`; the smoke's final image has the same hash, so the output reproduces across two processes. It differs from the 006 final image (`7c7a84f3…`): mean absolute RGB difference 33.3 levels (computed from the two `output.png` files).

Visual check of `output.png` against 006's `output.png`: same scene type (tabby cat behind a white sign held in both paws), but the cat is lighter and differently marked, with soft, smeared fur and blurred eyes and face; the paws are blurry; the background is a featureless blur. The sign text is garbled: thick, smeared, overlapping strokes reading roughly "hullowa", not "hello world" (006: a sharp "helllo world"). The image is visibly degraded.

## Observations

- MagCache reused 3 of 4 steps in every pass; denoising fell from 9185 to 2340 ms (−74.5%) and wall from 11962.5 to 5099.5 ms (−57.4%).
- The remaining denoise time is about one full 006 step (2296 ms) plus about 44 ms.
- Text encode and tiled VAE decode are unchanged; tiled decode is now 34.0% of wall.
- The output is visibly degraded (blur, garbled sign) and differs from 006 by 33.3 mean RGB levels.
- RAM peak is 52 MB higher and the MemAvailable minimum 91 MB lower than in 006 (single runs each).

## Interpretation

- With only 4 steps, reusing 3 leaves a single computed transformer evaluation; the latency saving is close to the maximum possible for denoising, and the image shows that one evaluation plus three magnitude-based reuses does not reproduce the reference output at this step count.
- Under any quality requirement that includes legible text or sharp detail, this default configuration does not qualify; it remains a latency bound for step reuse on this placement.
- Hypothesis: MagCache's default magnitude-ratio threshold is calibrated for longer schedules, where consecutive steps differ less; on a 4-step distilled schedule the per-step changes are large, so the reuse error is large.
- Hypothesis: the small memory difference is the cached feature slot; it is not separable from run-to-run variation with single runs.

## Next experiment

- Copy of `jetson-edgedit-flux-klein-q4-embq8-512-vaetile-magcache.json` with a stricter `--cache-threshold` that reuses at most one step (needs a runner option; separately labelled), with a saved-image comparison against 006.
