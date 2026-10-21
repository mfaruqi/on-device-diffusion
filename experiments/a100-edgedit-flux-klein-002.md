---
type: experiment-record
id: a100-edgedit-flux-klein-002
status: partial
device: a100-pcie-40gb
engine: edge-dit-cpp
runs: [a100-edgedit-flux-klein-002__attempt__20261015-152019]
updated: 2026-10-15
---

# a100-edgedit-flux-klein-002: edge-dit.cpp step cache `easycache` (engine defaults), BF16 1024² klein; screen only, 0 of 4 steps skipped

Status: **partial** (2026-10-15, job 11959806 on gilbreth-g008, 19:20:19–19:20:35 UTC). Two-generation screen (first + one measured) completed; the engine reported 0 skipped steps in both generations, so by the queue's screen rule the full 1+3+10 protocol was not run. Not a baseline.

## Question

Does edge-dit.cpp's `easycache` step cache, at its default parameters, skip or reuse any of the 4 denoising steps of the A100 no-cache configuration [a100-edgedit-flux-klein-001](a100-edgedit-flux-klein-001.md)? Screen against [a100-edgedit-flux-klein-bf16.json](../configs/a100-edgedit-flux-klein-bf16.json). Step-reuse lever (RQ2).

## Setup

- **Config**: smoke [a100-edgedit-flux-klein-bf16-easycache-smoke.json](../configs/a100-edgedit-flux-klein-bf16-easycache-smoke.json) (equal to the run-directory `config.json`); the full config [a100-edgedit-flux-klein-bf16-easycache.json](../configs/a100-edgedit-flux-klein-bf16-easycache.json) exists but was not run.
- **Change vs reference (only one)**: `optimizations.step_cache: {"method": "easycache"}`, mapped by [scripts/run_edgedit.py](../scripts/run_edgedit.py) to `--cache_method easycache` with no tuning flags, so the engine's per-method defaults apply ([command.json](../results/runs/a100-edgedit-flux-klein-002__attempt__20261015-152019/command.json)). The engine log does not print the default parameter values.
- Everything else as in the [reference](a100-edgedit-flux-klein-001.md#full-repeated-baseline-2026-10-06): edge-dit.cpp `97cc585`, CUDA performance build, same `ed-sample` binary (sha256 `fc6b14a7…`); BF16 weights with engine-reported FP32 transformer activations; flash attention on; VAE tiling off; 1024×1024, 4 Euler steps, guidance 1.0, CFG 1.0, seed 0, one prompt; four threads. One `ed-sample` process, `--repeat 2`.
- Launched by [scripts/gilbreth-gap-20261015.slurm](../scripts/gilbreth-gap-20261015.slurm): the full protocol runs only if the smoke's `step-cache.json` `skipped_steps_per_generation` sums to more than 0 (a null count also triggers it). `step-cache.json` is parsed from the engine's per-pass `<Method> skipped|reused k/4 steps` line. The runner script hash is not recorded in the run directory.
- The reference baseline ran on 2026-10-05 in a different Slurm job (11891242) on a different node (gilbreth-g007); this screen ran on gilbreth-g008.
- Metric derivation: [edge-dit adapter](../wiki/methods/baseline-metrics.md#edge-dit-ed-sample-adapter) (wall from `ed-sample` stdout at 1 ms rounding; `encode_setup` / `denoise` / `vae_decode` from host `system_clock` phase markers; device memory is one NVML peak over the whole child process).

## Results

[Run](../results/runs/a100-edgedit-flux-klein-002__attempt__20261015-152019/): [status.json](../results/runs/a100-edgedit-flux-klein-002__attempt__20261015-152019/status.json) `complete`.

- Cache receipt: the engine logged `cache enabled: mode=EasyCache model=flux2-klein granularity=output matched_sites=1 variants=2 slots=1` and `EasyCache skipped 0/4 steps` once per pass, 2 of 2 passes ([engine.log](../results/runs/a100-edgedit-flux-klein-002__attempt__20261015-152019/engine.log)); [step-cache.json](../results/runs/a100-edgedit-flux-klein-002__attempt__20261015-152019/step-cache.json) `skipped_steps_per_generation: [0, 0]`. The engine still printed 4 steps per pass.

| Quantity | First generation | Measured (n = 1) | Reference measured median (min–max) |
|---|---:|---:|---:|
| wall (ms, 1 ms rounding) | 3066.0 | 1677.0 | 1641.0 (1637.0–1644.0) |
| encode_setup (ms) | 661.0 | 277.2 | 269.6 (268.1–271.1) |
| denoise (ms) | 1594.7 | 1121.2 | 1096.2 (1092.0–1099.5) |
| vae_decode (ms) | 805.0 | 274.0 | 269.9 (268.8–272.8) |

Model load 7.06 s ([load.json](../results/runs/a100-edgedit-flux-klein-002__attempt__20261015-152019/load.json); filesystem cache may be warm). Whole-capture device-wide NVML peak 21.607 GiB (reference capture 21.607 GiB); host peak RSS 2.728 GiB (reference 2.740 GiB). Both include load.

Functional check: the retained final-pass image (run index 1) has RGB SHA256 `c476fc325f80bc3386922415ed1e8c410a6f44ebca2a1d7cb02ca5899accad55` ([image-retention.json](../results/runs/a100-edgedit-flux-klein-002__attempt__20261015-152019/image-retention.json), [runs.csv](../results/runs/a100-edgedit-flux-klein-002__attempt__20261015-152019/runs.csv)), identical to the final image hash recorded for the reference baseline (`c476fc32…`, [validation.json](../results/runs/a100-edgedit-flux-klein-001__baseline__20261005-233237/validation.json)); the reference PNG itself is not retained in the local run directory, so the comparison is by hash. No quality metric.

## Observations

- The cache was enabled but skipped no step in either generation, and the output is bit-identical to the reference.
- The single measured pass (1677.0 ms) is +36.0 ms from the reference median, with denoise +25.0 ms; this is one observation, from a different job and node than the reference.

## Interpretation

- At its default parameters this method is inert for the 4-step schedule on this configuration: it reduces to the no-cache configuration with the cache bookkeeping active.
- Hypothesis: the default policy's warm-up or threshold does not permit a reuse decision within 4 steps; whether a tuned threshold would skip a step, and at what output change, is not tested here. Whether the denoise difference is cache overhead or node/job variation cannot be separated from one observation.

## Next experiment

- Only with explicit tuning flags in a new labelled config (copy of `a100-edgedit-flux-klein-bf16-easycache.json` with the threshold stated), if this method is revisited.
