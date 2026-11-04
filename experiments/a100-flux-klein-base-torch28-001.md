---
type: experiment-record
id: a100-flux-klein-base-torch28-001
status: complete
device: a100-pcie-40gb
engine: pytorch-diffusers
runs: [a100-flux-klein-base-torch28-001__baseline__20261015-155338, a100-flux-klein-base-torch28-001__attempt__20261015-154758]
updated: 2026-10-15
---

# a100-flux-klein-base-torch28-001: BF16 1024² klein Base (50 steps, CFG 4) reference under torch 2.8.0 (no optimizations)

Status: **complete** (2026-10-15, job 11959986 on gilbreth-g009; full run 19:53:38–20:00:05 UTC, smoke
19:47:58–19:53:08 UTC).

## Question

What are the latency, stage split and memory of the unchanged BF16 klein Base workload (50 steps, CFG 4) in the
torch 2.8.0 environment that the A100 Base conditioning-reuse, TaylorSeer and compile variants of 2026-10-15 run
in? This run is the same-environment reference for those variants. Week 2 baseline infrastructure for RQ1.

## Setup

- Config [a100-flux-klein-base-torch28-bf16.json](../configs/a100-flux-klein-base-torch28-bf16.json); smoke
  [a100-flux-klein-base-torch28-bf16-smoke.json](../configs/a100-flux-klein-base-torch28-bf16-smoke.json). The
  run-directory `config.json` copies equal these files. Checkpoint `black-forest-labs/FLUX.2-klein-base-4B` pinned
  in the config to revision `a3b4f4849157f664bdbc776fd7453c2783562f4d` (the snapshot path in
  [load.json](../results/runs/a100-flux-klein-base-torch28-001__baseline__20261015-155338/load.json) matches); no
  separate `*.resolved.json` exists for this config.
- Workload: klein Base 4B, BF16, batch 1, 1024×1024, 50 steps (FlowMatchEulerDiscreteScheduler, dynamic shifting;
  [scheduler.json](../results/runs/a100-flux-klein-base-torch28-001__baseline__20261015-155338/scheduler.json)),
  guidance 4.0 with the empty negative prompt, `max_sequence_length` 512, prompt "A cat holding a sign that says
  hello world", seed 0.
- **Reference configuration**: no quantization, offload, VAE tiling/slicing, compiler, step cache or conditioning
  reuse. The config's `reference_config` names
  [a100-flux-klein-base-bf16.json](../configs/a100-flux-klein-base-bf16.json) (same workload); only the software
  environment differs (below), and this config pins the checkpoint commit directly.
- Protocol: 1 first + 3 warm-up (discarded) + 10 measured, one process, one load; smoke 1 first + 1 measured.
  NVML device memory sampled every 20 ms.
- Environment ([environment.json](../results/runs/a100-flux-klein-base-torch28-001__baseline__20261015-155338/environment.json)):
  NVIDIA A100-PCIE-40GB, driver 595.91.07, 250 W limit; Python 3.10.21 venv `/scratch/gilbreth/mfaruqi/envs/torch28-trt`
  ([scripts/gilbreth_setup_torch28.sh](../scripts/gilbreth_setup_torch28.sh)); torch 2.8.0+cu126 (CUDA runtime 12.6,
  cuDNN 9.10.2), diffusers 0.40.0, transformers 5.16.1, Triton 3.4.0; torch_tensorrt 2.8.0 / TensorRT 10.12.0.36
  installed but not used. `HF_HUB_OFFLINE=1`.
- Runner [scripts/run_flux_portable.py](../scripts/run_flux_portable.py), sha256 `06f0a949…` at run time
  ([source-provenance.json](../results/runs/a100-flux-klein-base-torch28-001__baseline__20261015-155338/source-provenance.json));
  `git_commit` is empty, so provenance is the script hashes. Launched by
  [scripts/gilbreth-gap-base-20261015.slurm](../scripts/gilbreth-gap-base-20261015.slurm) (smoke, then the full
  protocol only if the smoke completed).
- **CFG accounting.** Every generation has 2 `text_encode` calls (positive and empty negative prompt) and 2
  transformer calls per step for all 50 steps (from
  [stages.csv](../results/runs/a100-flux-klein-base-torch28-001__baseline__20261015-155338/stages.csv)). These runs
  predate the runner change that writes `transformer_calls_per_step` into `summary.json` (commit `1480c19`); the field was backfilled into each `summary.json` from `stages.csv` (value 2; the source is noted in the file).
- Metric definitions: [baseline-metrics.md](../wiki/methods/baseline-metrics.md#klein-base-and-classifier-free-guidance).
  `denoise_step_i_ms` sums both CFG calls of step i. Steady per-step = median over measured generations of each
  generation's mean of `denoise_step_1..49_ms` (computed from `runs.csv`).

## Results

[Run](../results/runs/a100-flux-klein-base-torch28-001__baseline__20261015-155338/),
[summary.json](../results/runs/a100-flux-klein-base-torch28-001__baseline__20261015-155338/summary.json),
[runs.csv](../results/runs/a100-flux-klein-base-torch28-001__baseline__20261015-155338/runs.csv).
`status.json` `complete`.

### Latency

| Stage | Median (ms), n = 10 | Min–max (ms) | Share of median wall |
|---|---:|---:|---:|
| text_encode_ms (2 calls) | 93.0 | 91.5–97.1 | 0.4% |
| denoise_ms (100 transformer calls) | 26171.4 | 26076.2–26283.7 | 98.8% |
| Steady per-step (mean of steps 1–49, 2 calls each) | 523.6 | 521.7–525.8 | – |
| denoise_step_0_ms | 517.8 | – | – |
| vae_decode_ms | 170.0 | 169.3–170.7 | 0.6% |
| postprocess_ms | 28.8 | 27.6–41.1 | 0.1% |
| other_ms | 12.9 | 12.3–14.6 | 0.0% |
| **wall_ms** | **26476.6** | 26380.9–26596.7 (stdev 81.7) | 100% |
| gpu_span_ms | 26476.1 | 26380.4–26596.3 | – |
| First generation: wall_ms | 26518.4 | – | text 420.5 / denoise 25730.8 (step 0: 551.3) / VAE 300.0 / post 43.8 / other 22.8 |
| Warm-ups (3, discarded): wall_ms | – | 26339.0–26552.0 | – |
| Load (cached, s) | 9.73 | – | `from_pretrained` 2.80 + `.to("cuda")` 6.92 |

### Memory (GiB)

| Quantity | Value |
|---|---:|
| Weights: text encoder / transformer / VAE | 7.49 / 7.22 / 0.16 (total 14.87) |
| Allocated after load | 14.93 |
| peak_alloc_gib (all 10 measured equal) | **17.33**, set by `vae_decode` (stage peaks: text encode 15.19, denoise steps 15.52, postprocess 14.97) |
| peak_reserved_gib | 19.61 |
| device_used_peak_gib (NVML, 20 ms) | 20.62 |
| host_rss_gib, measured median | 1.69 (1.65–1.69) |
| host_peak_rss_gib (includes load) | 8.07 |

### Repeatability and functional check (visual, not a quality metric)

All 14 generations have `image_sha256` `7357668ff45cc26fdb530a90375447ae761d0243a6972a3248e5c1a01387fa55`
(`deterministic_output: true`); the smoke run has the same hash. Visual check of `images/measured-0.png`: coherent
tabby cat behind a white sign held in both paws, blurred indoor background; the sign reads "HELLLO WORLD" (three Ls)
in thin dark capitals.

### Smoke gate

[`__attempt__20261015-154758`](../results/runs/a100-flux-klein-base-torch28-001__attempt__20261015-154758/):
complete. Load 45.57 s (`from_pretrained` 22.20 s, `.to("cuda")` 23.37 s). First generation 54393.1 ms (text
11605.7 / denoise 27897.1 with step 0 2898.9 / VAE 14016.0 / post 211.5 / other 662.0 ms); one measured 26077.5 ms
(text 145.4 / denoise 25711.5 / VAE 167.9 ms). Memory 17.33 / 19.61 / 20.62 GiB (allocated / reserved / device). Not
a baseline (n = 1).

### Prior reference on a different software stack

[a100-flux-klein-base-001](a100-flux-klein-base-001.md) (same device type and workload, torch 2.5.1+cu121, node
gilbreth-g007, 2026-10-05; its config [a100-flux-klein-base-bf16.json](../configs/a100-flux-klein-base-bf16.json)
requested revision `main`, resolved to the same commit `a3b4f484…` in
[a100-flux-klein-base-bf16.resolved.json](../configs/a100-flux-klein-base-bf16.resolved.json)) reported a median
`wall_ms` of 25147.7 ms and recorded image hash `abd41d67…`. That run comes from a different software stack and is
cited for orientation only; no variant in this campaign is compared against it.

## Observations

- Median wall is 26476.6 ms with a measured range of 215.8 ms (0.8% of the median); the 100 transformer calls are
  98.8% of it, and text encode, VAE decode, postprocess and `other_ms` together are 304.6 ms.
- Steady per-step time (two CFG calls) is 523.6 ms; step 0 has a median of 517.8 ms and steps 1–49 have medians of
  520.1–525.9 ms.
- The first generation of the full process (26518.4 ms) lies inside the measured range; its text encode (420.5 ms)
  and VAE decode (300.0 ms) are higher than the measured medians, its denoise (25730.8 ms) lower.
- VAE decode sets the allocator peak (17.33 GiB, +2.40 GiB over the 14.93 GiB held after load).
- Output is bit-identical across all generations and across the smoke and full processes.
- The smoke process (first in the job) loaded in 45.6 s and its first generation took 54.4 s, with text encode and VAE
  decode each above 11 s; the full process loaded in 9.7 s.

## Interpretation

- This run is the reference for the 2026-10-15 torch 2.8.0 Base variants (same workload, environment, node and job,
  no optimizations). With 50 steps and CFG, every lever that does not change transformer time is bounded by the
  1.2% of wall outside denoising.
- Hypothesis: the slow smoke load and first generation reflect cold state at the start of the job (page cache for the
  checkpoint files, first CUDA/cuDNN/cuBLAS initialisation in the venv); the data do not separate these. They are not
  part of any reported median.

## Next experiment

- Variants against this configuration: [a100-flux-klein-base-torch28-002](a100-flux-klein-base-torch28-002.md),
  [-003](a100-flux-klein-base-torch28-003.md), [-004](a100-flux-klein-base-torch28-004.md).
