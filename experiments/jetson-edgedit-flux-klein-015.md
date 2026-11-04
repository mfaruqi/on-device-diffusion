---
type: experiment-record
id: jetson-edgedit-flux-klein-015
status: failed
device: jetson-orin-nano
engine: edge-dit-cpp
runs: [jetson-edgedit-flux-klein-015__attempt__20261015-140427]
updated: 2026-10-15
---

# jetson-edgedit-flux-klein-015: edge-dit.cpp step cache `sencache`; refused by the engine without a calibrated profile

Status: **failed** (2026-10-15, smoke 18:04:27–18:05:37 UTC). `ed-sample` exited with code 1 before loading any weights; the full protocol was not run.

## Question

Does edge-dit.cpp's SenCache step cache, at its default parameters, reuse any of the 4 denoising steps of the [006](jetson-edgedit-flux-klein-006.md) placement? Single change against [jetson-edgedit-flux-klein-q4-embq8-512-vaetile.json](../configs/jetson-edgedit-flux-klein-q4-embq8-512-vaetile.json). Step-reuse lever (RQ2).

## Setup

- **Config**: smoke [jetson-edgedit-flux-klein-q4-embq8-512-vaetile-sencache-smoke.json](../configs/jetson-edgedit-flux-klein-q4-embq8-512-vaetile-sencache-smoke.json) (equal to the run-directory `config.json`); the full config [jetson-edgedit-flux-klein-q4-embq8-512-vaetile-sencache.json](../configs/jetson-edgedit-flux-klein-q4-embq8-512-vaetile-sencache.json) was not run.
- **Change vs reference (only one)**: `optimizations.step_cache: {"method": "sencache"}` → `--cache_method sencache`, no other cache flags. Everything else as in [006 setup](jetson-edgedit-flux-klein-006.md#setup) (edge-dit.cpp `97cc585`, same binary hash; compressed Q8_0-embedding encoder checkpoint, separately labelled; 512×512, 4 steps, `--repeat 2`).
- Runner [run_edgedit_jetson.py](../scripts/run_edgedit_jetson.py) sha256 `fcd45913…`; queue [jetson_gap_queue_20261015d.sh](../scripts/jetson_gap_queue_20261015d.sh) under [jetson_memguard.sh](../scripts/jetson_memguard.sh) (400 MB floor, 768 MB swap-growth limit).

## Results

[status.json](../results/runs/jetson-edgedit-flux-klein-015__attempt__20261015-140427/status.json): `failed`, `CalledProcessError(1, [ed-sample … --cache_method sencache …])`. First line of [engine.log](../results/runs/jetson-edgedit-flux-klein-015__attempt__20261015-140427/engine.log):

```
--cache sencache requires a calibrated profile: pass --cache-profile <path>, or produce one first with --cache-calibrate <path>
```

followed by the `ed-sample` usage text, which lists `--cache-calibrate <path>` and `--cache-profile <path>` under "Cache tuning". No model was loaded and no generation, `runs.csv`, `summary.json` or image exists. [memguard.json](../results/runs/jetson-edgedit-flux-klein-015__attempt__20261015-140427/memguard.json): not killed, exit 1; MemAvailable 5990 → minimum 5946 MB; swap 316 MB.

## Observations

- SenCache has no default mode in this engine build: it refuses to run without a profile file, either supplied or produced by a calibration pass.
- The failure is an argument check before loading; it says nothing about the method's latency or output.

## Interpretation

- Running SenCache needs two labelled steps: a calibration run that writes a profile (on a calibration prompt set, kept separate from the held-out set), then a measured run that reads it. The current runner exposes neither flag.

## Next experiment

- Runner option for `--cache-calibrate` / `--cache-profile`, then a copy of `jetson-edgedit-flux-klein-q4-embq8-512-vaetile-sencache.json` naming the profile file and its calibration prompts.
