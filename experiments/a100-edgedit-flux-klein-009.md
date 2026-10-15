---
type: experiment-record
id: a100-edgedit-flux-klein-009
status: failed
device: a100-pcie-40gb
engine: edge-dit-cpp
runs: [a100-edgedit-flux-klein-009__attempt__20261015-152145]
updated: 2026-10-15
---

# a100-edgedit-flux-klein-009: edge-dit.cpp step cache `sencache`, BF16 1024² klein; refused by the engine without a calibrated profile

Status: **failed** (2026-10-15, job 11959806 on gilbreth-g008, smoke 19:21:45 UTC). `ed-sample` exited with code 1
before loading any weights; the full protocol was not run.

## Question

Does edge-dit.cpp's SenCache step cache, at its default parameters, reuse any of the 4 denoising steps of the A100
no-cache configuration [a100-edgedit-flux-klein-001](a100-edgedit-flux-klein-001.md)? Single change against
[a100-edgedit-flux-klein-bf16.json](../configs/a100-edgedit-flux-klein-bf16.json). Step-reuse lever (RQ2).

## Setup

- **Config**: smoke [a100-edgedit-flux-klein-bf16-sencache-smoke.json](../configs/a100-edgedit-flux-klein-bf16-sencache-smoke.json)
  (equal to the run-directory `config.json`); the full config
  [a100-edgedit-flux-klein-bf16-sencache.json](../configs/a100-edgedit-flux-klein-bf16-sencache.json) was not run.
- **Change vs reference (only one)**: `optimizations.step_cache: {"method": "sencache"}` → `--cache_method sencache`,
  no other cache flags ([command.json](../results/runs/a100-edgedit-flux-klein-009__attempt__20261015-152145/command.json)).
  Everything else as in the [reference](a100-edgedit-flux-klein-001.md#full-repeated-baseline-2026-10-06)
  (edge-dit.cpp `97cc585`, same `ed-sample` binary sha256 `fc6b14a7…`; BF16 weights; 1024×1024, 4 steps, seed 0;
  `--repeat 2`).
- Runner [scripts/run_edgedit.py](../scripts/run_edgedit.py); queue
  [scripts/gilbreth-gap-20261015.slurm](../scripts/gilbreth-gap-20261015.slurm).

## Results

[status.json](../results/runs/a100-edgedit-flux-klein-009__attempt__20261015-152145/status.json): `failed`,
`CalledProcessError(1, [ed-sample … --cache_method sencache …])`. First line of
[engine.log](../results/runs/a100-edgedit-flux-klein-009__attempt__20261015-152145/engine.log):

```
--cache sencache requires a calibrated profile: pass --cache-profile <path>, or produce one first with --cache-calibrate <path>
```

followed by the `ed-sample` usage text, which lists `--cache-calibrate <path>` and `--cache-profile <path>` under
"Cache tuning". No model was loaded and no generation, `runs.csv`, `summary.json`, memory capture or image exists.

## Observations

- SenCache has no default mode in this engine build: it refuses to run without a profile file, either supplied or
  produced by a calibration pass.
- The failure is an argument check before loading; it says nothing about the method's latency or output.

## Interpretation

- Running SenCache needs two labelled steps: a calibration run that writes a profile (on a calibration prompt set,
  kept separate from the held-out set), then a measured run that reads it. The current runner exposes neither flag.

## Next experiment

- Runner option for `--cache-calibrate` / `--cache-profile`, then a copy of `a100-edgedit-flux-klein-bf16-sencache.json` naming the profile file and its calibration prompts.
