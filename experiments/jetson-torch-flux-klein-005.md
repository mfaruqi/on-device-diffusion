---
type: experiment-record
id: jetson-torch-flux-klein-005
status: failed
device: jetson-orin-nano
engine: pytorch-diffusers
runs: [jetson-torch-flux-klein-005__attempt__20261015-141745]
updated: 2026-10-15
---

# jetson-torch-flux-klein-005: diffusers First-Block Cache (threshold 0.05), PyTorch/diffusers klein Q4_0 + NF4; not supported for FLUX.2 blocks

Status: **failed** (2026-10-15, smoke `__attempt__20261015-141745`, 18:17:45–18:19:25 UTC). The cache could
not be attached to the transformer, so no generation ran. The full protocol was not run (the queue runs it
only after a complete smoke).

## Question

Does diffusers' First-Block Cache with threshold 0.05 skip transformer computation for the 4-step distilled
klein workload of [jetson-torch-flux-klein-001](jetson-torch-flux-klein-001.md), and at what latency and
output change? Single change against the 001 config. Step-reuse lever (RQ2).

## Setup

- [Configuration](../configs/jetson-torch-flux-klein-q4-nf4-512-fbcache.json) and
  [smoke configuration](../configs/jetson-torch-flux-klein-q4-nf4-512-fbcache-smoke.json); the run-directory
  `config.json` equals the smoke file. Reference config:
  [jetson-torch-flux-klein-q4-nf4-512.json](../configs/jetson-torch-flux-klein-q4-nf4-512.json).
- **One change vs the reference:** `optimizations.step_cache: {"method": "first_block_cache", "threshold": 0.05}`.
- How it is applied ([scripts/run_flux_portable.py](../scripts/run_flux_portable.py) `apply_step_cache`):
  `transformer.enable_cache(FirstBlockCacheConfig(threshold=0.05))` right after load, and again on every
  reloaded transformer.
- Workload, checkpoints (GGUF Q4_0 transformer, NF4 encoder, BF16 VAE; separately labelled), sequential
  residency, protocol (smoke 1 + 1) and device/software stack as in 001; torch 2.8.0, **diffusers 0.40.0**
  ([environment.json](../results/runs/jetson-torch-flux-klein-005__attempt__20261015-141745/environment.json)).
  Runner sha256 `44e44608…`. Launched by [scripts/jetson_gap_queue_20261015d.sh](../scripts/jetson_gap_queue_20261015d.sh)
  under the memory guard (400 MB floor, 1536 MB swap-growth limit).

## Results

[status.json](../results/runs/jetson-torch-flux-klein-005__attempt__20261015-141745/status.json): `failed`,
error

```
ValueError("Model class <class 'diffusers.models.transformers.transformer_flux2.Flux2TransformerBlock'> not registered.")
```

raised from `diffusers/hooks/first_block_cache.py` `initialize_hook` →
`TransformerBlockRegistry.get(Flux2TransformerBlock)` while `enable_cache` installed the head-block hook.
Loading had completed: [load.json](../results/runs/jetson-torch-flux-klein-005__attempt__20261015-141745/load.json)
`load_total_s` 74.94 s, 2.475 GiB allocated after load. No `runs.csv`, `summary.json` or image exists.
[memguard.json](../results/runs/jetson-torch-flux-klein-005__attempt__20261015-141745/memguard.json): not
killed, runner exit 1; MemAvailable 5990 → minimum 3401 MB; swap 321 MB throughout.

## Observations

- diffusers 0.40.0 has no `TransformerBlockRegistry` entry for `Flux2TransformerBlock`, so First-Block Cache
  cannot be enabled on `Flux2Transformer2DModel` in this environment.
- The failure is at configuration time, before any generation; it is not a memory or numerical failure.

## Interpretation

- First-Block Cache is unavailable for this model in the installed diffusers version; this record measures
  nothing about its effect. Registering the block would require knowing the block's input/output
  signature for the hook (hypothesis about what a fix involves; not attempted).

## Next experiment

- Revisit `jetson-torch-flux-klein-q4-nf4-512-fbcache.json` only after a diffusers version that registers `Flux2TransformerBlock`.
