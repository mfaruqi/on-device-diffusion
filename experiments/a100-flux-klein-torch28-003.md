---
type: experiment-record
id: a100-flux-klein-torch28-003
status: failed
device: a100-pcie-40gb
engine: pytorch-diffusers
runs: [a100-flux-klein-torch28-003__attempt__20261015-151915]
updated: 2026-10-15
---

# a100-flux-klein-torch28-003: diffusers First-Block Cache (threshold 0.05), BF16 1024² klein; not supported for FLUX.2 blocks in diffusers 0.40.0

Status: **failed** (2026-10-15, job 11959806 on gilbreth-g008; smoke 19:19:15–19:19:29 UTC). The cache could
not be attached to the transformer, so no generation ran. The full protocol was not run (the queue runs it only
after a complete smoke).

## Question

Does diffusers' First-Block Cache with threshold 0.05 skip transformer computation for the 4-step BF16 reference
workload, and at what latency and output change? Single change against
[a100-flux-klein-torch28-001](a100-flux-klein-torch28-001.md). Step-reuse lever (RQ2).

## Setup

- Smoke config [a100-flux-klein-torch28-bf16-fbcache-smoke.json](../configs/a100-flux-klein-torch28-bf16-fbcache-smoke.json)
  (equal to the run-directory `config.json`); the full config
  [a100-flux-klein-torch28-bf16-fbcache.json](../configs/a100-flux-klein-torch28-bf16-fbcache.json) was not run.
- **One change vs [a100-flux-klein-torch28-bf16.json](../configs/a100-flux-klein-torch28-bf16.json):**
  `optimizations.step_cache: {"method": "first_block_cache", "threshold": 0.05}`.
- How it is applied ([scripts/run_flux_portable.py](../scripts/run_flux_portable.py) `apply_step_cache`):
  `pipe.transformer.enable_cache(FirstBlockCacheConfig(threshold=0.05))` right after load.
- Workload and environment as in [torch28-001 setup](a100-flux-klein-torch28-001.md#setup) (torch 2.8.0+cu126,
  **diffusers 0.40.0**, [environment.json](../results/runs/a100-flux-klein-torch28-003__attempt__20261015-151915/environment.json)).
  Runner sha256 `44e44608…`.

## Results

[status.json](../results/runs/a100-flux-klein-torch28-003__attempt__20261015-151915/status.json): `failed`, error

```
ValueError("Model class <class 'diffusers.models.transformers.transformer_flux2.Flux2TransformerBlock'> not registered.")
```

raised from `diffusers/hooks/first_block_cache.py` `initialize_hook` →
`TransformerBlockRegistry.get(Flux2TransformerBlock)` while `enable_cache` installed the head-block hook.
Loading had completed: [load.json](../results/runs/a100-flux-klein-torch28-003__attempt__20261015-151915/load.json)
`load_total_s` 8.75 s, 14.93 GiB allocated after load. No `runs.csv`, `summary.json` or image exists.

## Observations

- diffusers 0.40.0 has no `TransformerBlockRegistry` entry for `Flux2TransformerBlock`, so First-Block Cache
  cannot be enabled on `Flux2Transformer2DModel` in this environment.
- The failure is at configuration time, before any generation; it is not a memory or numerical failure.

## Interpretation

- First-Block Cache is unavailable for this model in the installed diffusers version; this record measures
  nothing about its effect. Hypothesis: enabling it would require registering the block's input/output
  signature with the hook registry (not attempted; that would be a library-level change).

## Next experiment

- Revisit `a100-flux-klein-torch28-bf16-fbcache.json` only with a diffusers version that registers `Flux2TransformerBlock`.
