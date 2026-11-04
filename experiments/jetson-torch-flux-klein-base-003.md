---
type: experiment-record
id: jetson-torch-flux-klein-base-003
status: failed
device: jetson-orin-nano
engine: pytorch-diffusers
runs: [jetson-torch-flux-klein-base-003__attempt__20261016-032053, jetson-torch-flux-klein-base-003__attempt__20261015-165212]
updated: 2026-10-16
---

# jetson-torch-flux-klein-base-003: diffusers TaylorSeer on top of exact conditioning reuse, klein Base Q4_0 + NF4 512², 50 steps CFG 4; stopped by the memory guard during the first text-encoder load

Status: **failed** (2026-10-16 07:20:53–07:25:10 UTC). The smoke was stopped by [jetson_memguard.sh](../scripts/jetson_memguard.sh) before the first denoising step; no full protocol was run. An earlier attempt on 2026-10-15 failed before loading (Setup).

## Question

Does diffusers' TaylorSeer step cache, applied on top of exact conditioning reuse ([jetson-torch-flux-klein-base-002](jetson-torch-flux-klein-base-002.md)), reduce the per-image time of the 50-step CFG Base workload, and does it fit in memory? Single change against [jetson-torch-flux-klein-base-q4-nf4-512-condreuse.json](../configs/jetson-torch-flux-klein-base-q4-nf4-512-condreuse.json). Step-reuse lever under a memory budget (RQ2).

## Setup

- **Config**: smoke [jetson-torch-flux-klein-base-q4-nf4-512-condreuse-taylorseer-smoke.json](../configs/jetson-torch-flux-klein-base-q4-nf4-512-condreuse-taylorseer-smoke.json) (first + one measured); full [jetson-torch-flux-klein-base-q4-nf4-512-condreuse-taylorseer.json](../configs/jetson-torch-flux-klein-base-q4-nf4-512-condreuse-taylorseer.json) not run.
- **Change vs reference (only one)**: `optimizations.step_cache: {"method": "taylorseer", "cache_interval": 5, "disable_cache_before_step": 3, "max_order": 1}` (diffusers `TaylorSeerCacheConfig`, applied by [run_flux_portable.py](../scripts/run_flux_portable.py) `apply_step_cache()` to the transformer at load and after every reload).
- Everything else as in [base-002](jetson-torch-flux-klein-base-002.md#setup): Base Q4_0 GGUF transformer (BF16 compute), pre-quantized NF4 Qwen3 encoder, BF16 VAE; sequential stage residency (encoder released after encode, transformer released during encode); exact conditioning reuse; SDPA default; no VAE tiling; 512×512, 50 steps, guidance 4.0 with empty negative prompt, seed 0; torch 2.8.0 / diffusers 0.40.0 (Jetson AI Lab jp6/cu126 stack); 25W.
- **Guard**: 400 MB MemAvailable floor, 1536 MB swap-growth limit, 0.1 s poll; launched by [jetson_gap_queue_20261016f.sh](../scripts/jetson_gap_queue_20261016f.sh). The guard sends SIGKILL, so `status.json` was left at `running` and was corrected afterwards to `failed` from `memguard.json` (`status_correction` field).
- **Earlier attempt** [jetson-torch-flux-klein-base-003__attempt__20261015-165212](../results/runs/jetson-torch-flux-klein-base-003__attempt__20261015-165212/) (2026-10-15): `failed`, `IncompleteSnapshotError` for the pinned Base snapshot `a3b4f484…`, because the config's `allow_patterns` requested transformer and encoder weight files that are not needed on this device; the patterns were narrowed to configs, tokenizer, scheduler and VAE before the rerun. Not a measurement.

## Results

[jetson-torch-flux-klein-base-003__attempt__20261016-032053](../results/runs/jetson-torch-flux-klein-base-003__attempt__20261016-032053/): [status.json](../results/runs/jetson-torch-flux-klein-base-003__attempt__20261016-032053/status.json) `failed`, `killed by scripts/jetson_memguard.sh: MemAvailable 391 MB < 400 MB (exit 137)`.

| Quantity | Value |
|---|---|
| Pipeline load | 74.91 s; allocated after load 2.475 GiB (transformer 2.291 GiB, VAE 0.157 GiB; encoder not resident) ([load.json](../results/runs/jetson-torch-flux-klein-base-003__attempt__20261016-032053/load.json)) |
| Point of failure | first generation, first `encode_prompt` call: the run log ends with the NF4 encoder's weight loading at 398/398 tensors, followed by the guard's kill; `runs.csv` and `stages.csv` are empty |
| MemAvailable (guard), start → minimum | 5991 → 391 MB |
| Swap (guard), start → maximum | 317 → 850 MB |
| Last tegrastats sample (03:25:09 local) | RAM 6981/7620 MB, swap 812 MB |

No generation completed; no image, no summary.json.

## Observations

- The guard fired during the first text-encoder load, before any transformer call, so the TaylorSeer cache never ran a step.
- Under the same policy without the step cache ([base-002](jetson-torch-flux-klein-base-002.md)), the same first encode completed with a MemAvailable minimum of 1472 MB over the whole full run.

## Interpretation

- Hypothesis: with the TaylorSeer hook registered on the transformer, releasing `pipe.transformer` before the encoder load no longer frees the transformer's weights (the hook registry or cache state keeps a reference), so the 2.29 GiB transformer and the NF4 encoder are resident together during the first encode. The data cannot separate this from other allocations; a check of `torch.cuda.memory_allocated()` after the release would test it.

## Next experiment

- Rerun `jetson-torch-flux-klein-base-q4-nf4-512-condreuse-taylorseer-smoke.json` after the runner removes the step-cache hook before each transformer release, with an allocator reading logged after the release.
