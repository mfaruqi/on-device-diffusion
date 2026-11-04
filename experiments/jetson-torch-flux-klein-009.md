---
type: experiment-record
id: jetson-torch-flux-klein-009
status: failed
device: jetson-orin-nano
engine: pytorch-diffusers
runs: [jetson-torch-flux-klein-009__attempt__20261102-164203]
updated: 2026-11-02
---

# jetson-torch-flux-klein-009: text encoder and transformer both resident with tiled VAE decode, PyTorch/diffusers klein Q4_0 + NF4, 512×512

Status: **failed** (2026-11-02, run `__attempt__20261102-164203`, run directly on the device). The model
loaded; the memory guard ended the process during the first generation. No generation completed.

## Question

Does PyTorch/diffusers run the quantized 512×512 workload with the text encoder and the transformer both resident
for the whole run (no stage residency), with the tiled VAE decode of
[jetson-torch-flux-klein-003](jetson-torch-flux-klein-003.md)? Reference: 003 (sequential residency, same tiling).
Memory-residency lever under a memory budget (RQ3).

## Setup

- [Configuration](../configs/jetson-torch-flux-klein-q4-nf4-512-resident-vaetile-check.json). **One change vs
  [003](../configs/jetson-torch-flux-klein-q4-nf4-512-vaetile.json):** `optimizations.stage_residency: null`
  (003: encoder `release_after_encode`, transformer `release_during_encode`), plus the shorter check protocol.
- Workload as 003: FLUX.2 klein 4B (revision `e7b7dc27`), GGUF Q4_0 transformer, pre-quantized NF4 text encoder,
  BF16 VAE, 512×512, 4 steps, guidance 1.0, seed 0, max sequence length 512; VAE tiling 128 px / 0.25.
- Protocol (feasibility check): 1 first + 1 discarded warm-up + 4 measured generations, one process.
  Prompt fixed; conditioning reuse off, so every generation runs the text encoder.
- Queue [`scripts/jetson_residency_check_queue_20261102.sh`](../scripts/jetson_residency_check_queue_20261102.sh),
  one job on the board, desktop session logged out. Memory guard (outside the config): kill if MemAvailable
  < 400 MB or swap growth > 1536 MB, polled every 0.1 s. Power mode 25 W.
- The board ran the runner from its `~/on-device-diffusion-gap` copy, not a git checkout
  ([environment](../results/runs/jetson-torch-flux-klein-009__attempt__20261102-164203/environment.json) has an empty `git_commit`).
- Metrics: [baseline-metrics.md](../wiki/methods/baseline-metrics.md).

## Results

[Run directory](../results/runs/jetson-torch-flux-klein-009__attempt__20261102-164203/), [load](../results/runs/jetson-torch-flux-klein-009__attempt__20261102-164203/load.json), [memguard](../results/runs/jetson-torch-flux-klein-009__attempt__20261102-164203/memguard.json), tegrastats (local-only artifact: `../results/runs/jetson-torch-flux-klein-009__attempt__20261102-164203/tegrastats.log`).

- `status.json` still reads `running`: the guard's SIGKILL (exit code 137) leaves no chance to write `failed`.
  `runs.csv` and `stages.csv` have headers only; no image was written.
- Load: 144.67 s (`from_pretrained` 143.68 s); allocated after load 5.119 GiB, reserved 5.143 GiB, all on CUDA:
  text encoder 2.417 GiB, transformer 2.291 GiB, VAE 0.157 GiB.
- Memory guard: `MemAvailable 382 MB < 400 MB; start 6074 MB; swap 327 → 564 MB`.
- Last tegrastats sample before the kill: RAM 7006 MB of 7620 MB, GR3D_FREQ 99%.
- Timing: process start 16:42:03 local, last tegrastats sample 16:44:53; minus the load time, the process ran
  for at most about 25 s after loading finished (an upper bound: imports precede the load). The
  stage at the kill is not recorded.

## Observations

- Loading with both large components resident completed; the peak allocated during load equals the
  allocation after load.
- System RAM rose from the post-load level to the guard threshold within the first generation.

## Interpretation

- With both components resident, about 5.1 GiB is allocated before generation and the first generation
  needs more system memory than the board has left above the guard threshold. The stage that sets the peak is
  not identified; 003's allocator peak was set by `text_encode`, but this run recorded no stage times.
- **Hypothesis:** the extra memory is the first generation's activations and workspaces plus PyTorch/CUDA
  library memory that is not counted in the allocator figures. Not measured.

## Next experiment

- Same configuration with per-stage memory recorded before each stage starts, to locate the peak.
- Labelled variant with a lower `max_sequence_length`, if text encoding sets the peak.
