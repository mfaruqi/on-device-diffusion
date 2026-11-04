---
type: experiment-record
id: a100-flux-klein-torch28-001
status: complete
device: a100-pcie-40gb
engine: pytorch-diffusers
runs: [a100-flux-klein-torch28-001__profile__20261021-115545, a100-flux-klein-torch28-001__baseline__20261015-151548, a100-flux-klein-torch28-001__attempt__20261015-151216, a100-flux-klein-torch28-001__profile__20261024-141920, a100-flux-klein-torch28-001__profile__20261024-143259, a100-flux-klein-torch28-001__profile__20261024-161538]
updated: 2026-10-24
---

# a100-flux-klein-torch28-001: BF16 1024² klein reference re-established under torch 2.8.0 (no optimizations)

Status: **complete** (2026-10-15, job 11959806 on gilbreth-g008; full run 19:15:48–19:16:22 UTC, smoke
19:12:16–19:15:30 UTC).

## Question

What are the latency, stage split and memory of the unchanged BF16 reference workload in the torch 2.8.0
environment that the A100 compiler and step-cache variants of 2026-10-15 run in? This run is the
same-environment reference for those variants. The existing A100 PyTorch baseline
([a100-flux-klein-001](a100-flux-klein-001.md)) ran in a different software stack (torch 2.5.1, `flux_env`),
so it cannot serve as the reference for variants measured under torch 2.8.0. Week 2 baseline
infrastructure for RQ1.

## Setup

- Config [a100-flux-klein-torch28-bf16.json](../configs/a100-flux-klein-torch28-bf16.json); smoke
  [a100-flux-klein-torch28-bf16-smoke.json](../configs/a100-flux-klein-torch28-bf16-smoke.json). The
  run-directory `config.json` copies equal these files. The checkpoint revision is pinned in the config
  (`e7b7dc27f91deacad38e78976d1f2b499d76a294`); no separate `*.resolved.json` exists for this config.
- Workload: distilled FLUX.2 [klein] 4B, BF16, batch 1, 1024×1024, 4 steps, guidance 1.0,
  `max_sequence_length` 512, prompt "A cat holding a sign that says hello world", seed 0.
- **Change vs [a100-flux-klein-bf16.resolved.json](../configs/a100-flux-klein-bf16.resolved.json):** none in
  the workload or optimizations; only the software environment differs (below). This is a reference
  configuration, not a variant. No quantization, offload, VAE tiling, compiler, step cache or conditioning reuse.
- Protocol: 1 first + 3 warm-up (discarded) + 10 measured, one process, one load; smoke 1 first + 1 measured.
  NVML device memory sampled every 20 ms.
- Environment ([environment.json](../results/runs/a100-flux-klein-torch28-001__baseline__20261015-151548/environment.json)):
  NVIDIA A100-PCIE-40GB, driver 595.91.07, 250 W limit; Python 3.10.21 venv
  `/scratch/gilbreth/mfaruqi/envs/torch28-trt` created 2026-10-15 by
  [scripts/gilbreth_setup_torch28.sh](../scripts/gilbreth_setup_torch28.sh) (torch from the PyTorch cu126 index
  with PyPI as extra index; the script's default `/depot/you139/mfaruqi` location was not writable, so the venv
  is on scratch, which is purged after 60 days without use). torch 2.8.0+cu126 (CUDA runtime 12.6, cuDNN
  9.10.2), diffusers 0.40.0, transformers 5.16.1, torch_tensorrt 2.8.0 / TensorRT 10.12.0.36 installed but not
  used; Triton 3.4.0 (not recorded in `environment.json`). `HF_HUB_OFFLINE=1`.
- Runner [scripts/run_flux_portable.py](../scripts/run_flux_portable.py), sha256 `44e44608…` at run time
  ([source-provenance.json](../results/runs/a100-flux-klein-torch28-001__baseline__20261015-151548/source-provenance.json));
  `git_commit` is empty, so provenance is the script hashes. Launched by
  [scripts/gilbreth-gap-20261015.slurm](../scripts/gilbreth-gap-20261015.slurm) (smoke, then the full protocol
  only if the smoke completed).
- Metric definitions: [baseline-metrics.md](../wiki/methods/baseline-metrics.md). Steady per-step = median over
  measured generations of each generation's mean of `denoise_step_1..3_ms` (computed from `runs.csv`).

## Results

[Run](../results/runs/a100-flux-klein-torch28-001__baseline__20261015-151548/),
[summary.json](../results/runs/a100-flux-klein-torch28-001__baseline__20261015-151548/summary.json),
[runs.csv](../results/runs/a100-flux-klein-torch28-001__baseline__20261015-151548/runs.csv),
[stages.csv](../results/runs/a100-flux-klein-torch28-001__baseline__20261015-151548/stages.csv),
[load.json](../results/runs/a100-flux-klein-torch28-001__baseline__20261015-151548/load.json).
`status.json` `complete`.

### Latency

| Stage | Median (ms), n = 10 | Min–max (ms) | Share of median wall |
|---|---:|---:|---:|
| text_encode_ms | 46.2 | 45.9–48.4 | 3.6% |
| denoise_ms (4 transformer calls) | 1037.8 | 1032.3–1041.6 | 80.4% |
| denoise_step_0 / 1 / 2 / 3_ms | 255.6 / 257.1 / 258.4 / 266.0 | 251.5–268.2 (all four) | – |
| Steady per-step (mean of steps 1–3) | 260.3 | 260.1–261.7 | – |
| vae_decode_ms | 168.2 | 166.8–168.8 | 13.0% |
| postprocess_ms | 29.8 | 27.1–34.6 | 2.3% |
| other_ms | 7.5 | 7.3–8.0 | 0.6% |
| **wall_ms** | **1290.5** | 1283.3–1294.4 (stdev 4.0) | 100% |
| gpu_span_ms | 1290.4 | 1283.2–1294.4 | – |
| First generation: wall_ms | 1830.9 | – | text 380.2 / denoise 1080.3 (step 0: 307.8) / VAE 305.8 / post 45.8 / other 18.6 |
| Warm-ups (3, discarded): wall_ms | – | 1278.7–1292.9 | – |
| Load (cached, s) | 8.80 | – | `from_pretrained` 2.77 + `.to("cuda")` 6.03 |

### Memory (GiB)

| Quantity | Value |
|---|---:|
| Weights: text encoder / transformer / VAE | 7.49 / 7.22 / 0.16 (total 14.87) |
| Allocated after load | 14.93 |
| peak_alloc_gib (all 10 measured equal) | **17.32**, set by `vae_decode` (stage peaks: text encode 15.19, each denoise step 15.51) |
| peak_reserved_gib | 19.61 |
| device_used_peak_gib (NVML, 20 ms) | 20.62 |
| host_rss_gib, measured median | 1.69 (1.65–1.69) |
| host_peak_rss_gib (includes load) | 8.07 |

### Repeatability and functional check (visual, not a quality metric)

All 14 generations have `image_sha256` `f11220f630cce576da295977e5f575a1c1b391c8d117dc5db263cf5ab04624dd`
(`deterministic_output: true`); the smoke run has the same hash. Visual check of
`images/measured-0.png`: coherent tabby cat holding a white sign in both paws, indoor background; the sign
reads "hellword", not "hello world".

### Smoke gate

[`__attempt__20261015-151216`](../results/runs/a100-flux-klein-torch28-001__attempt__20261015-151216/):
complete. First generation 29970.1 ms (text_encode 12771.4, denoise 3767.3 with step 0 2997.1, VAE 12455.6 ms);
one measured 1337.4 ms (text 72.4 / denoise 1054.3 / VAE 169.4 ms). Load 39.17 s (`from_pretrained` 18.12 s,
`.to("cuda")` 21.05 s). Memory 17.32 / 19.61 / 20.62 GiB (allocated / reserved / device). Not a baseline (n = 1).

### Prior reference on a different software stack

[a100-flux-klein-001](a100-flux-klein-001.md) (same device type and workload, torch 2.5.1+cu121, driver
590.48.01, 2026-09-23) reported a median `wall_ms` of 1240.6 ms. That figure comes from a different software
stack and is cited for orientation only; no variant in this campaign is compared against it.

## Observations

- Median wall is 1290.5 ms with a measured range of 11.2 ms (0.9% of the median); denoising is 80.4% of it.
- Per-step time rises from step 0 to step 3 (255.6 → 266.0 ms) in the measured generations.
- VAE decode sets the allocator peak (17.32 GiB, +2.40 GiB over the 14.93 GiB held after load).
- Output is bit-identical across all generations and across the smoke and full processes.
- The smoke process (first in the job) loaded in 39.2 s and its first generation took 30.0 s, with text encode
  and VAE decode each above 12 s; the full process, started 18 s later, loaded in 8.8 s and its first generation
  took 1.8 s.

## Interpretation

- This run is the reference for the 2026-10-15 torch 2.8.0 variants: the same workload, environment, node and
  job, with no optimizations.
- Hypothesis: the slow smoke load and first generation reflect cold state at the start of the job (page cache
  for the checkpoint files, first CUDA/cuDNN/cuBLAS initialisation in a new venv); the data do not separate
  these. They are not part of any reported median.

## Hardware-counter attempt (2026-10-21)

Slurm **12001684** on **gilbreth-g001** failed before model loading/image generation. The original BF16 configuration and image settings were unchanged; [capture plan](../configs/a100-flux-klein-torch28-ncu.json). [Status](../results/runs/a100-flux-klein-torch28-001__profile__20261021-115545/status.json) records `ValueError('No captured kernels in NCU export')`.

The probe log (local-only artifact: `../results/runs/a100-flux-klein-torch28-001__profile__20261021-115545/profile/probe.log`) and [raw CSV](../results/runs/a100-flux-klein-torch28-001__profile__20261021-115545/profile/probe.csv) show that Nsight Compute captured one BF16 GEMM over four replay passes. The [environment gate](../results/runs/a100-flux-klein-torch28-001__profile__20261021-115545/environment-gate.json) passed. Failure was in CSV parsing: NCU 2026.1.1 emits a wide metric table with a units row (`ns`), whereas the initial parser expected long metric/value rows (`nsecond`). This was not a counter-permission denial.

The parser now accepts both explicit layouts and nanosecond spellings, retaining strict missing/nonfinite/unit checks. [Post-hoc probe review](../results/runs/a100-flux-klein-torch28-001__profile__20261021-115545/profile/probe-review.json) validates positive BF16 operation and HBM traffic counters against the preserved CSV; a regression fixture in [tests](../tests/test_a100_counters.py) reproduces the actual export. Original failure status/raw evidence remain unchanged. No full-image counters, images, or new latency measurements were produced; roofline plots and W&B results remain unchanged. Corrected resubmissions followed on 2026-10-24 (below). [Procedure](../docs/a100-hardware-counters.md).

## Hardware-counter capture (2026-10-24)

Three resubmissions from fresh isolated snapshots (`/scratch/gilbreth/mfaruqi/a100-ncu-20261024`, `…b`, `…c`; each `RESUBMISSION-NOTE.txt` lists its one change):

| Job | Node | Change vs previous | Outcome |
|---|---|---|---|
| 12014733 | g000 | repaired wide-CSV parser | [failed](../results/runs/a100-flux-klein-torch28-001__profile__20261024-141920/status.json): counter probe exceeded its 180 s limit before profiling its GEMM (the 2026-10-21 probe took 137 s on g001) |
| 12014773 | g005 | `probe_timeout_s` 180 → 600 | [failed](../results/runs/a100-flux-klein-torch28-001__profile__20261024-143259/status.json): probe and image gates passed; the full-image replay exceeded `capture_timeout_s` 5400 after about 4,160 of 8,719 kernels |
| 12015059 | g000 | `capture_timeout_s` 5400 → 16200; Slurm time 2 → 6 h | capture **complete** ([run](../results/runs/a100-flux-klein-torch28-001__profile__20261024-161538/)); post-processing then rejected the stage labels, because NVTX-renamed kernel names are `<stage>/<kernel>`. [profile_a100_counters.py](../scripts/profile_a100_counters.py) now takes the stage before the first `/` (tested) and its post-processing is re-runnable with `--finalize`; the saved exports were finalized offline with no new GPU work. [validation.json](../results/runs/a100-flux-klein-torch28-001__profile__20261024-161538/validation.json): passed, reference image hash matched, all stages present. |

Scope: one warmed 1024² image of this configuration under Nsight Compute kernel replay (clocks and caches uncontrolled); counted BF16/FP16/TF32 dense Tensor Core operations plus FP32 add/mul/FMA; DRAM read + write bytes. Summed kernel durations come from replay and are not generation latency ([counter-summary.json](../results/runs/a100-flux-klein-torch28-001__profile__20261024-161538/profile/counter-summary.json), [counter-kernels.json.gz](../results/runs/a100-flux-klein-torch28-001__profile__20261024-161538/profile/counter-kernels.json.gz)).

| Stage | Kernels | Summed kernel time (s) | HBM (GB) | Counted ops (T) | Intensity (FLOP/B) | TFLOP/s | HBM GB/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| text_encode | 2166 | 0.044 | 18.7 | 4.28 | 229.4 | 98.4 | 429 |
| denoise_step_0 … 3 (each) | 1543 | 0.228–0.229 | 143.5–143.7 | 34.95 | 243.2–243.5 | 152.5–153.1 | 626–630 |
| vae_decode | 336 | 0.159 | 69.6 | 10.63 | 152.7 | 66.8 | 438 |

Kernel time within `denoise_step_1` (share; intensity; TFLOP/s): BF16 GEMM 128×256 tiles 33.4% (671; 258), flash attention 14.1% (2109; 204), BF16 GEMM 256×128 tiles 13.9% (728; 265), elementwise/copy/concatenation kernels about 36% together (intensity 0.0–0.7; ≤ 0.8 TFLOP/s).

Observations:

- Counted operations per denoising step (34.95 T) match the roofline script's analytic estimate (34.82 TFLOP) within 0.4%; measured HBM traffic (143.6 GB) is 6.7× its analytic bytes (21.5 GB), so measured intensity is 243 FLOP/B rather than the analytic 1616.
- The denoising step sits just right of the A100 PCIe BF16 ridge (312 TFLOP/s ÷ 1555 GB/s = 201 FLOP/B) at 153 TFLOP/s (49% of the dense BF16 specification) over summed kernel time; its GEMMs run at 258–265 TFLOP/s (83–85%) and flash attention at 204.
- About a third of each step's kernel time is elementwise, copy and concatenation kernels near zero intensity; they account for most of the traffic above the analytic estimate.
- Summed kernel time per step (229 ms) is 88% of this configuration's measured denoise time per step (1037.8 ms ÷ 4 = 259 ms, [baseline](../results/runs/a100-flux-klein-torch28-001__baseline__20261015-151548/summary.json)); replay serialises kernels, so the remainder is not attributed here.

Interpretation (hypothesis): torch.compile's −21% ([a100-flux-klein-compile-001](a100-flux-klein-compile-001.md)) is consistent with fusing the low-intensity elementwise kernels; a matching capture of the compiled configuration would test it.

## Next experiment

- Variants against this configuration: [a100-flux-klein-compile-001](a100-flux-klein-compile-001.md),
  [a100-flux-klein-torch28-002](a100-flux-klein-torch28-002.md), [-003](a100-flux-klein-torch28-003.md),
  [-004](a100-flux-klein-torch28-004.md).
