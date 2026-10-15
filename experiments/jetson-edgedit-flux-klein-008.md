---
type: experiment-record
id: jetson-edgedit-flux-klein-008
status: partial
device: jetson-orin-nano
engine: edge-dit-cpp
runs: [jetson-edgedit-flux-klein-008__attempt__20261015-132605]
updated: 2026-10-15
---

# jetson-edgedit-flux-klein-008: edge-dit.cpp step cache `easycache` (engine defaults); screen only, 0 of 4 steps skipped

Status: **partial** (2026-10-15, 17:26:05–17:28:52 UTC). Two-generation screen (first + one measured) completed; the engine reported 0 skipped steps in both generations, so by the queue's screen rule the full 1+3+10 protocol was not run. Not a baseline.

## Question

Does edge-dit.cpp's `easycache` step cache, at its default parameters, skip any of the 4 denoising steps of the [006](jetson-edgedit-flux-klein-006.md) placement? Screen against the reference config [jetson-edgedit-flux-klein-q4-embq8-512-vaetile.json](../configs/jetson-edgedit-flux-klein-q4-embq8-512-vaetile.json). Step-reuse lever (RQ2).

## Setup

- **Config**: smoke [jetson-edgedit-flux-klein-q4-embq8-512-vaetile-easycache-smoke.json](../configs/jetson-edgedit-flux-klein-q4-embq8-512-vaetile-easycache-smoke.json) (equal to the run-directory `config.json`); full config [jetson-edgedit-flux-klein-q4-embq8-512-vaetile-easycache.json](../configs/jetson-edgedit-flux-klein-q4-embq8-512-vaetile-easycache.json) exists but was not run.
- **Change vs reference (only one)**: `optimizations.step_cache: {"method": "easycache"}`, mapped by the runner to `--cache_method easycache` with no tuning flags, so the engine's per-method defaults apply ([command.json](../results/runs/jetson-edgedit-flux-klein-008__attempt__20261015-132605/command.json)). The engine log does not print the default parameter values.
- Everything else as in [006 setup](jetson-edgedit-flux-klein-006.md#setup): edge-dit.cpp `97cc585`, same `ed-sample` binary hash; Q4_0 transformer, **Q4_K_M encoder with Q8_0 token embedding (compressed, separately labelled checkpoint)**, BF16 VAE, all weights resident; flash attention on; VAE tiling on (25 tiles every pass); 512×512, 4 Euler steps, guidance 1.0, seed 0, one prompt; 25W. One `ed-sample` process, `--repeat 2`.
- Runner [run_edgedit_jetson.py](../scripts/run_edgedit_jetson.py) at sha256 `fcd45913…` ([source-provenance.json](../results/runs/jetson-edgedit-flux-klein-008__attempt__20261015-132605/source-provenance.json)); launched by [jetson_gap_queue_20261015d.sh](../scripts/jetson_gap_queue_20261015d.sh) (`cache_screen`: the full protocol runs unless the smoke's `step-cache.json` sums to exactly 0 skipped steps; a null count also triggers it) under [jetson_memguard.sh](../scripts/jetson_memguard.sh) (400 MB floor, 768 MB swap-growth limit, 0.1 s poll).
- Metric derivation and memory scope: [006 setup](jetson-edgedit-flux-klein-006.md#setup), [edge-dit adapter](../wiki/methods/baseline-metrics.md#edge-dit-ed-sample-adapter), [Jetson repeated baseline memory](../wiki/methods/baseline-metrics.md#jetson-repeated-baseline-memory).

## Results

[Run](../results/runs/jetson-edgedit-flux-klein-008__attempt__20261015-132605/): [status.json](../results/runs/jetson-edgedit-flux-klein-008__attempt__20261015-132605/status.json) `complete`; [memguard.json](../results/runs/jetson-edgedit-flux-klein-008__attempt__20261015-132605/memguard.json) not killed, exit 0.

- Cache receipt: the engine logged `cache enabled: mode=EasyCache model=flux2-klein granularity=output matched_sites=1 variants=2 slots=1` and `EasyCache skipped 0/4 steps` once per pass, 2 of 2 passes ([engine.log](../results/runs/jetson-edgedit-flux-klein-008__attempt__20261015-132605/engine.log)); [step-cache.json](../results/runs/jetson-edgedit-flux-klein-008__attempt__20261015-132605/step-cache.json) `skipped_steps_per_generation: [0, 0]`. The schedule (4 steps, same sigmas as 006) was unchanged.

| Quantity | First generation | Measured (n = 1) | 006 measured median |
|---|---:|---:|---:|
| wall (ms, 1 ms rounding) | 17747.0 | 12042.0 | 11962.5 |
| text_encode / denoise / vae_decode / other (ms, 10 ms rounding) | 2450.0 / 10510.0 / 4620.0 / 140.0 | 930.0 / 9270.0 / 1730.0 / 120.0 | 930.0 / 9185.0 / 1740.0 / 120.0 |

Raw marker intervals of the measured pass ([stages.csv](../results/runs/jetson-edgedit-flux-klein-008__attempt__20261015-132605/stages.csv)): encode_setup 1021.0, denoise 9268.0, vae_decode 1728.6 ms. Model load 65.978 s ([load.json](../results/runs/jetson-edgedit-flux-klein-008__attempt__20261015-132605/load.json)). Tegrastats (469 samples, whole capture): RAM peak 6.598 GiB of 7.44 GiB, swap 0.302–0.322 GiB, GPU temperature max 53.4 °C. Guard: MemAvailable 5865 MB at start, minimum 676 MB; swap 309 → 329 MB.

Functional check: the retained final-pass image (run index 1) has RGB SHA256 `7c7a84f3fe9988c8f43f76d656273ceb102b893cee89d4c14615116eeeaabd32` ([image-retention.json](../results/runs/jetson-edgedit-flux-klein-008__attempt__20261015-132605/image-retention.json), [runs.csv](../results/runs/jetson-edgedit-flux-klein-008__attempt__20261015-132605/runs.csv)), identical to the 006 final image. No quality metric.

## Observations

- The cache was enabled (one matched site) but skipped no step in either generation, and the output is bit-identical to 006.
- The single measured pass (12042.0 ms) is +79.5 ms from the 006 measured median, with denoise +85.0 ms. This is one observation; for scale, the 006 measured range is 11943.0–12026.0 ms and the 006 smoke's single measured pass was 12090.0 ms.

## Interpretation

- At its default parameters this method is inert for a 4-step schedule on this placement: the configuration reduces to 006 with the cache bookkeeping active.
- Hypothesis: the default policy's warm-up or threshold does not allow a skip within 4 steps; whether a tuned threshold would skip a step, and at what output change, is not tested here.

## Next experiment

- Only with explicit tuning flags in a new labelled config (copy of `jetson-edgedit-flux-klein-q4-embq8-512-vaetile-easycache.json` with the threshold stated), if this method is revisited.
