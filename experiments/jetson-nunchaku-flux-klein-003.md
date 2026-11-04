---
type: experiment-record
id: jetson-nunchaku-flux-klein-003
status: failed
device: jetson-orin-nano
engine: pytorch-diffusers
runs: [jetson-nunchaku-flux-klein-003__attempt__20261102-164525]
updated: 2026-11-02
---

# jetson-nunchaku-flux-klein-003: Nunchaku INT4 r32 transformer and NF4 text encoder both resident with tiled VAE decode, 512×512

Status: **failed** (2026-11-02, run `__attempt__20261102-164525`, run directly on the device). The model
loaded; the memory guard ended the process during the first generation. No generation completed.

## Question

Does the Nunchaku SVDQuant INT4 (rank 32) transformer, which replaces the GGUF transformer of
[jetson-torch-flux-klein-009](jetson-torch-flux-klein-009.md), run with the text encoder and transformer both resident
and tiled VAE decode? Compressed-checkpoint workload, reported separately from execution-only changes (RQ3).

## Setup

- [Configuration](../configs/jetson-nunchaku-flux-klein-int4-r32-512-resident-vaetile-check.json). Differs from
  [009](../configs/jetson-torch-flux-klein-q4-nf4-512-resident-vaetile-check.json) only in the transformer
  checkpoint (GGUF Q4_0 → Nunchaku `svdq-int4_r32`, sm_87 build at revision `0c8379d` with the two-line
  architecture patch recorded in [load.json](../results/runs/jetson-nunchaku-flux-klein-003__attempt__20261102-164525/load.json)). Against the earlier sequential
  Nunchaku configuration (`jetson-nunchaku-flux-klein-002`) it changes two settings: residency and VAE tiling.
- Workload: FLUX.2 klein 4B (revision `e7b7dc27`), NF4 text encoder, BF16 VAE, 512×512, 4 steps, guidance 1.0,
  seed 0; VAE tiling 128 px / 0.25.
- Protocol (feasibility check): 1 first + 1 discarded warm-up + 4 measured generations, one process.
  Prompt fixed; conditioning reuse off, so every generation runs the text encoder.
- Queue [`scripts/jetson_residency_check_queue_20261102.sh`](../scripts/jetson_residency_check_queue_20261102.sh),
  one job on the board, desktop session logged out. Memory guard (outside the config): kill if MemAvailable
  < 400 MB or swap growth > 1536 MB, polled every 0.1 s. Power mode 25 W.
- The board ran the runner from its `~/on-device-diffusion-gap` copy, not a git checkout
  ([environment](../results/runs/jetson-nunchaku-flux-klein-003__attempt__20261102-164525/environment.json) has an empty `git_commit`).
- Metrics: [baseline-metrics.md](../wiki/methods/baseline-metrics.md).

## Results

[Run directory](../results/runs/jetson-nunchaku-flux-klein-003__attempt__20261102-164525/), [load](../results/runs/jetson-nunchaku-flux-klein-003__attempt__20261102-164525/load.json), [memguard](../results/runs/jetson-nunchaku-flux-klein-003__attempt__20261102-164525/memguard.json), [tegrastats](../results/runs/jetson-nunchaku-flux-klein-003__attempt__20261102-164525/tegrastats.log).

- `status.json` still reads `running`: the guard's SIGKILL (exit code 137) leaves no chance to write `failed`.
  `runs.csv` and `stages.csv` have headers only; no image was written.
- Load: 110.76 s (`from_pretrained` 110.42 s); allocated after load 5.135 GiB, reserved 5.154 GiB, all on CUDA:
  text encoder 2.417 GiB, transformer 2.280 GiB (120 SVDQuant layers), VAE 0.157 GiB.
- Memory guard: `MemAvailable 262 MB < 400 MB; start 6060 MB; swap 328 → 330 MB`.
- Last tegrastats sample before the kill: RAM 7198 MB of 7620 MB, GR3D_FREQ 0%.
- Timing: process start 16:45:25 local, last tegrastats sample 16:47:42; minus the load time, the process ran
  for at most about 26 s after loading finished (an upper bound: imports precede the load). The
  stage at the kill is not recorded.

## Observations

- Loading with both large components resident completed; the peak allocated during load equals the
  allocation after load.
- System RAM rose from the post-load level to the guard threshold within the first generation.

## Interpretation

- The Nunchaku transformer is 0.011 GiB smaller than the GGUF one in this pipeline, so the resident total is
  essentially the same as 009's and the first generation fails the same way. The stage that sets the peak is
  not identified.

## Next experiment

- Same configuration after 009's peak is located, with the same per-stage memory recording.
