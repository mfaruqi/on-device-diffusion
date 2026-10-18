---
type: experiment-record
id: jetson-torch-flux-klein-007
status: complete
device: jetson-orin-nano
engine: pytorch-diffusers
runs: [jetson-torch-flux-klein-007__baseline__20261015-145234, jetson-torch-flux-klein-007__attempt__20261015-144541, jetson-torch-flux-klein-007__profile__20261017-220257, jetson-torch-flux-klein-007__profile__20261017-223538]
updated: 2026-10-18
---

# jetson-torch-flux-klein-007: torch.compile (default mode) on the transformer, with conditioning reuse, PyTorch/diffusers klein Q4_0 + NF4, 512×512

Status: **complete** (2026-10-15, run `__baseline__20261015-145234`, 18:52:34–18:59:23 UTC; no Slurm job).
Smoke gate `__attempt__20261015-144541` complete. The first generation includes compilation and is
reported separately.

## Question

How much does `torch.compile` (default mode, Inductor) of the transformer forward reduce steady denoising
time for the conditioning-reuse configuration [jetson-torch-flux-klein-004](jetson-torch-flux-klein-004.md),
and what does compilation cost in the first generation and in memory? Reference: the 004 baseline
[`__baseline__20261015-141028`](../results/runs/jetson-torch-flux-klein-004__baseline__20261015-141028/).
004 is the reference because conditioning reuse keeps the transformer resident after the first
generation, so a compiled module is reused instead of being discarded at every reload. Compiler lever (RQ1).

## Setup

- [Configuration](../configs/jetson-torch-flux-klein-q4-nf4-512-condreuse-compile.json) and
  [smoke configuration](../configs/jetson-torch-flux-klein-q4-nf4-512-condreuse-compile-smoke.json); the
  run-directory `config.json` copies equal these files. Reference config:
  [jetson-torch-flux-klein-q4-nf4-512-condreuse.json](../configs/jetson-torch-flux-klein-q4-nf4-512-condreuse.json).
- **One change vs the reference:** `optimizations.compiler: {"name": "torch_compile", "mode": "default"}`.
  The config otherwise differs only in `id`, `description` and `reference_config`.
- How it is applied ([scripts/run_flux_portable.py](../scripts/run_flux_portable.py) `apply_compiler`,
  `benchmark()`): `transformer.forward = torch.compile(transformer.forward, mode="default")`; the stage hooks
  stay outside the compiled region. Compilation is lazy. The runner wraps the transformer loaded at start-up
  and, through the residency `on_reload` callback, the transformer reloaded during the first generation's
  (single) residency swap. [compiler.json](../results/runs/jetson-torch-flux-klein-007__baseline__20261015-145234/compiler.json)
  records `compilations_applied: 2` for these two wrappings. The start-up module is released inside the
  first `encode_prompt` call, before any transformer forward, so only the reloaded module is ever executed
  and compiled. `check_config` allows a compiler with transformer reloads only together with conditioning reuse.
- Compile environment: `TORCHINDUCTOR_COMPILE_THREADS=1` (set by
  [scripts/jetson_gap_queue_20261015d.sh](../scripts/jetson_gap_queue_20261015d.sh) and recorded in
  `environment.json` `env_vars`), so Inductor uses one compile worker. Triton 3.4.0 was installed into
  `~/venvs/torch-jp6-ailab` on 2026-10-15 for these runs; the version appears only in the config
  `description`, because `environment.json` does not record Triton. Other versions as in 004: torch 2.8.0,
  diffusers 0.40.0, transformers 5.16.1, bitsandbytes 0.48.0.dev0+ff389db
  ([environment.json](../results/runs/jetson-torch-flux-klein-007__baseline__20261015-145234/environment.json)).
- Unchanged from 004: workload (distilled klein `e7b7dc27…`, 512×512, 4 steps, guidance 1.0, seed 0),
  compressed checkpoints (GGUF Q4_0 transformer, NF4 encoder, BF16 VAE; separately labelled), sequential
  residency with exact conditioning reuse (1 miss, 13 hits), protocol 1 + 3 + 10 (smoke 1 + 1), tegrastats at
  200 ms, memory guard (400 MB floor, 1536 MB swap-growth limit). Runner sha256 `44e44608…`.
- Metric definitions: [baseline-metrics.md](../wiki/methods/baseline-metrics.md); `text_encode_ms` semantics
  under conditioning reuse as in [004](jetson-torch-flux-klein-004.md#setup).

## Results

[Run](../results/runs/jetson-torch-flux-klein-007__baseline__20261015-145234/),
[summary.json](../results/runs/jetson-torch-flux-klein-007__baseline__20261015-145234/summary.json),
[runs.csv](../results/runs/jetson-torch-flux-klein-007__baseline__20261015-145234/runs.csv),
[stages.csv](../results/runs/jetson-torch-flux-klein-007__baseline__20261015-145234/stages.csv),
[conditioning-cache.json](../results/runs/jetson-torch-flux-klein-007__baseline__20261015-145234/conditioning-cache.json)
(hits 13, misses 1),
[stage-residency.json](../results/runs/jetson-torch-flux-klein-007__baseline__20261015-145234/stage-residency.json)
(one swap: transformer free 0.851 s, encoder load 34.742 s, encoder free 0.484 s, transformer reload 33.149 s),
[memguard.json](../results/runs/jetson-torch-flux-klein-007__baseline__20261015-145234/memguard.json).
`status.json` `complete`; guard not triggered (exit 0).

### Latency

| Metric | Median (ms), n = 10 | Min–max (ms) | Share of median wall |
|---|---:|---:|---:|
| wall_ms | 7783.6 | 7755.5–7796.8 | – |
| gpu_span_ms | 7783.1 | 7755.1–7796.2 | – |
| text_encode_ms (cache retrieval) | 0.8 | 0.5–0.8 | 0.0% |
| denoise_ms | 7249.5 | 7229.6–7260.4 | 93.1% |
| denoise_step_0_ms | 1816.2 | 1811.6–1820.8 | – |
| denoise_step_1 / 2 / 3_ms | 1813.5 / 1812.6 / 1809.0 | 1803.6–1818.5 (all three) | – |
| Steady per-step (mean of steps 1–3) | 1811.7 | 1806.0–1814.5 | – |
| vae_decode_ms | 485.3 | 484.1–485.9 | 6.2% |
| postprocess_ms | 17.4 | 12.2–18.0 | 0.2% |
| other_ms | 31.9 | 24.6–33.8 | 0.4% |
| **First generation (includes the residency swap and compilation): wall_ms** | 202655.5 | – | – |
| First generation: text_encode (includes swap, 69.227 s) / denoise / vae_decode / postprocess / other | 73937.7 / 123954.8 / 4072.5 / 115.9 / 563.1 | – | – |
| First generation: denoise_step_0 (includes compilation) / steps 1–3 | 118520.2 / 1816.1, 1809.7, 1808.8 | – | – |
| Warm-ups (3, discarded): wall_ms | – | 7752.5–7860.3 | – |
| Cold load: load_total_s | 81.19 s (`from_pretrained` 80.79 s); cached, not a cold read | – | – |

Measured `wall_ms` stdev 12.5 ms. The first generation's step 0 exceeds a steady step by about 116.7 s;
this is the compilation cost of this run at one compile worker (it also contains the first-call execution).
In every measured generation step 0 is within 10 ms of steps 1–3; the first warm-up's step 0 (1898.1 ms) is
about 85 ms above its other steps, and the smoke run's second generation shows the same (1919.3 ms).

### Memory

| Quantity | Value | Source |
|---|---:|---|
| peak_alloc_gib, measured (all 10 equal) | 3.083 GiB, set by `vae_decode` | summary.json, stages.csv |
| peak_reserved_gib, measured | 3.725 GiB | summary.json |
| Denoise-step peak (each step) | 2.890 GiB | stages.csv |
| First generation peak alloc / reserved | 3.083 / 3.723 GiB | summary.json |
| host_rss_gib, measured | 2.137 (2.136–2.137) GiB | summary.json |
| host_peak_rss_gib (includes load) | 5.504 GiB | summary.json |
| System RAM peak, whole capture (tegrastats, 1859 samples) | 6.552 / 7.441 GiB | summary.json `system_memory` |
| Swap occupancy, whole capture (tegrastats) | 0.316 → 0.504 GiB | summary.json `system_memory` |
| Guard: MemAvailable start → minimum | 5999 → 700 MB | memguard.json |
| Guard: swap start → maximum | 325 → 584 MB (growth 259 MB) | memguard.json |

### Repeatability and functional check (visual, not a quality metric)

All 14 generations have `image_sha256` `544a3bf947e7f5293933d83e352d18ef0bcf75e9f695ab9bdd0b9653fed9d3cb`
(`deterministic_output: true`). This differs from 004/001 (`fe94904f…`) and from the smoke run's hash
(`13f6174f78cee725e35b13be8881a38e34de01cbf86509911da6f69fcea5c57e`, identical across its two generations):
the compiled output is deterministic within a process but differed between the two processes. Mean
absolute RGB difference from 004's retained image is 3.5 levels (4.8% of pixels differ by more than 8);
between the smoke and full images it is 1.9 levels (computed from the `measured-0.png` files).

Visual check of `images/measured-0.png` (full and smoke) against 004's: composition, cat, sign and background
are near-identical. The sign reads "HOLLO WORD" (004: "HOLLO WORLD"); the second word's "L" is missing.
No other visible difference was found.

### Smoke gate

[`__attempt__20261015-144541`](../results/runs/jetson-torch-flux-klein-007__attempt__20261015-144541/):
complete; `compilations_applied` 2, cache hits 1, misses 1. First 256190.2 ms (text_encode 70616.3 including
a 65.099 s swap; step 0 176017.9 ms); one measured 7915.5 ms (denoise 7387.9 ms, steady per-step 1822.9 ms,
VAE 484.8 ms). Load 77.47 s. Peak alloc 3.083 / reserved 3.725 GiB. Tegrastats RAM peak 6.483 GiB; guard
MemAvailable minimum 749 MB. Not a baseline (n = 1).

### Delta vs reference (004 baseline)

| Metric | 004 median (ms) | 007 median (ms) | Δ |
|---|---:|---:|---:|
| wall_ms | 14911.4 | 7783.6 | −7127.8 (−47.8%) |
| denoise_ms | 14377.3 | 7249.5 | −7127.8 (−49.6%) |
| Steady per-step | 3593.1 | 1811.7 | −1781.4 (−49.6%) |
| denoise_step_0_ms | 3599.6 | 1816.2 | −1783.4 (−49.5%) |
| vae_decode_ms | 485.7 | 485.3 | −0.4 |
| other_ms | 30.6 | 31.9 | +1.3 |
| First generation wall_ms | 96501.7 | 202655.5 | +106153.8 (+110.0%) |
| peak_alloc_gib / peak_reserved_gib | 3.083 / 3.426 | 3.083 / 3.725 | 0.000 / +0.299 |
| host_rss_gib | 1.834 | 2.137 | +0.303 GiB |
| System RAM peak (tegrastats) | 5.982 GiB | 6.552 GiB | +0.570 GiB |
| Guard MemAvailable minimum | 1234 MB | 700 MB | −534 MB |

## Kernel profile (2026-10-17/18)

Queues [jetson_gap_queue_20261017h.sh](../scripts/jetson_gap_queue_20261017h.sh) (as the user) and [jetson_gap_queue_20261018i.sh](../scripts/jetson_gap_queue_20261018i.sh) (as root via `sudo`, after the desktop session was stopped; MemAvailable 5949 MB at start). [run_flux_portable.py](../scripts/run_flux_portable.py) `--profile`: the unchanged config's full 1+3+10 protocol, then one extra generation under `torch.profiler` (CPU + CUDA activities, shapes, memory) with the tegrastats monitor already stopped; it is excluded from `runs.csv` and `summary.json`. Trace `profile/trace.json` (Chrome format, opens in Perfetto; git-ignored, kept locally); kernel tables from [analyze_profile.py](../scripts/analyze_profile.py) in `profile/stage_kernels.md`. Kernel times are from a profiled generation and are reported as shares and per-step means, not as baseline latency.

- **Without root, desktop session active** ([jetson-torch-flux-klein-007__profile__20261017-220257](../results/runs/jetson-torch-flux-klein-007__profile__20261017-220257/), queue H): `failed`, killed by the memory guard (MemAvailable 391 MB < 400 MB) during the first generation's compilation; MemAvailable was 5243 MB at start (desktop session) against 5999 MB for the full run above.
- **As root** ([jetson-torch-flux-klein-007__profile__20261017-223538](../results/runs/jetson-torch-flux-klein-007__profile__20261017-223538/), queue I): `complete`; 1909 kernel events. Protocol repeat: wall median 7846.6 ms (7835.0–7855.3; +0.8% vs the full run above), first 256211.9 ms; profiled generation 7910.0 ms.

Per denoise step (mean of 4 steps, [stage_kernels.md](../results/runs/jetson-torch-flux-klein-007__profile__20261017-223538/profile/stage_kernels.md)), against the eager profile of [004](jetson-torch-flux-klein-004.md#kernel-profile-2026-10-1718):

| Kernel group | 007 ms / step | 004 ms / step | Δ |
|---|---:|---:|---:|
| GPU span | 1839.2 | 3691.8 | −1852.6 (−50.2%) |
| gemm (cuBLAS/cutlass BF16) | 1377.8 | 1374.4 | +3.4 |
| weight unpack / rescale / cast (eager: copy/cast + elementwise mul + other; compiled: fused Triton `triton_poi_fused___rshift____to_copy_bitwise_and_…`) | 313.9 | 2087.1 | −1773.2 (−85.0%) |
| attention | 105.1 | 79.2 | +25.9 |
| Kernels / step | 460 | 2347 | −80% |

GPU busy is 99–100% in both, so the saving is not launch overhead: Inductor fuses the Q4_0 unpack (shift, mask, subtract, scale) and the BF16 cast into one kernel per weight instead of several full-size passes; the BF16 matmuls are unchanged.

## Observations

- Compiled denoising runs at 1.81 s per step against 3.59 s, a factor of 1.98; the whole measured generation
  takes 7.78 s.
- The saving is entirely in the transformer steps; VAE decode, retrieval and `other_ms` are unchanged.
- Compilation adds about 117 s to the first generation (step 0: 118.5 s); load (81.2 s) plus first generation
  (202.7 s, including the 69.2 s swap) total about 284 s before the first image.
- After compilation, measured step times are flat (range about 10 ms per step position); the generation
  after the compiling one has a step 0 about 0.1 s longer, then nothing further.
- Allocated peaks are unchanged; reserved memory (+0.30 GiB), host RSS (+0.30 GiB) and whole-system RAM
  (+0.57 GiB) rise, and the guard's MemAvailable minimum falls to 700 MB.
- Output pixels change slightly and the sign text loses a letter; the compiled output is not bit-identical
  across the two processes.

## Interpretation

- For this fixed-prompt configuration, compiling the transformer halves steady generation latency at the
  cost of about two minutes of one-time compilation; it pays back after roughly 16 generations at the
  measured medians (116.7 s / 7.13 s), not counting the compile memory risk.
- Hypothesis: the gain comes from Inductor fusing the BF16 elementwise and normalization work and the GGUF
  dequantize-then-matmul path into fewer kernels; a kernel profile of one compiled and one eager step would
  test it.
- Hypothesis: the extra reserved memory and host RSS are Inductor/Triton compiled-kernel state and
  autotuning/codegen residue held by the process; the data do not attribute them.
- Hypothesis: the output change and its cross-process variation come from different floating-point
  reduction order in generated kernels (and possibly process-dependent kernel choices); this is a numerical
  change, not a precision change, but it is visible in fine text.

## Next experiment

- Copy of `jetson-torch-flux-klein-q4-nf4-512-condreuse-compile.json` with `mode: "max-autotune"` (separately labelled; first generation reported separately).
- Runner: record the Triton version in `environment.json`.
