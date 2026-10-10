---
type: experiment-record
id: a100-sdcpp-flux-klein-001
status: complete
device: a100-pcie-40gb
engine: stable-diffusion-cpp
runs: [a100-sdcpp-flux-klein-001__baseline__20260925-114328, a100-sdcpp-flux-klein-001__profile__20260925-124703, a100-sdcpp-flux-klein-001__baseline__20260925-114138, a100-sdcpp-flux-klein-001__profile__20260925-124021, a100-sdcpp-flux-klein-001__attempt__20260929-001722, a100-sdcpp-flux-klein-001__attempt__20260929-005352, a100-sdcpp-flux-klein-001__attempt__20261005-193229, a100-sdcpp-flux-klein-001__baseline__20261005-220435]
updated: 2026-10-06
---

# a100-sdcpp-flux-klein-001: FLUX.2 [klein] 4B with stable-diffusion.cpp on one A100-PCIE-40GB

Status: **baseline complete** (2026-09-25, Slurm job 11817906). Profiler run complete (job 11818087).

## Question

How long does one FLUX.2 [klein] 4B generation take with stable-diffusion.cpp on the A100, with
the same weights, workload and protocol as the reference configuration, and how is the time and
device memory split between text encoding, denoising and VAE decoding? This is an engine baseline
and a functional check. It is **not** a quality benchmark.

## Setup

- Config: `configs/a100-sdcpp-flux-klein-bf16.json` → `configs/a100-sdcpp-flux-klein-bf16.resolved.json`
- Engine: stable-diffusion.cpp tag `master-919-19bbbca` (commit `19bbbca1`), ggml `4bf5f600`, built with
  `-DSD_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=80 -DCMAKE_BUILD_TYPE=Release` (CUDA 12.6, gcc 11.5).
  Driven by the harness `engines/sdcpp/bench.cpp` (load once, then generate). How to build and run it:
  [sdcpp-howto.md](../wiki/methods/sdcpp-howto.md).
- Weights: `black-forest-labs/FLUX.2-klein-4B` @ `e7b7dc27f91deacad38e78976d1f2b499d76a294`
  - diffusion model: `flux-2-klein-4b.safetensors`, value-identical to the diffusers `transformer/`
    (3,875,544,576 BF16 values; `configs/sdcpp-weights.check.json`);
  - text encoder: `text_encoder/` (2 shards via the safetensors index);
  - VAE: `vae/diffusion_pytorch_model.safetensors`.
- Workload: BF16 weights, batch 1, 1024×1024, 4 steps, cfg 1.0, prompt
  "A cat holding a sign that says hello world", seed 0. sd.cpp's RNG (`cuda`, sd-webui compatible) differs
  from torch's, so the initial noise is not the same as in PyTorch runs with the same seed.
- Engine settings: all weights and compute on `cuda0`; auto-fit off; graph segmentation disabled;
  eager weight load; conditioning cache off; flash attention in the diffusion model only; no
  offload, VAE tiling or step caching. The reasons for each are in [sdcpp-howto.md](../wiki/methods/sdcpp-howto.md).
- Protocol: 1 first + 3 warm-up + 10 measured generations in one process.
- Device: Gilbreth `a100-40gb` partition, `--constraint=G` (A100-PCIE-40GB, 250 W), account `you139`.
- Metric definitions, including how sd.cpp stage times are measured:
  [baseline-metrics.md → stable-diffusion.cpp runs](../wiki/methods/baseline-metrics.md#stable-diffusioncpp-runs)

## Results

Run directory: `results/runs/a100-sdcpp-flux-klein-001__baseline__20260925-114328`   Slurm job: `11817906`   Node: `gilbreth-g004`
GPU: NVIDIA A100-PCIE-40GB, driver 590.48.01, 250 W limit.
Repo commit at run time: `771641e`. The sd.cpp kit files were not yet committed; they are listed in
`environment.json → git_dirty_files`.

### Engine audit (from sd.cpp's log, `summary.json → engine_audit`)

| Check | Result |
|---|---|
| Model detected | `Flux.2 klein` |
| Weight types | text encoder 398 × BF16, diffusion model 149 × BF16, VAE 250 × BF16 + 1 × I32 |
| Parameter placement | 15,225 MB, all on CUDA0 (0 MB in RAM) |
| Graph segments | 1 per model (no graph cutting) |
| Flash attention | "Using flash attention in the diffusion model" |
| Sampler / scheduler | Euler, Flux2 scheduler (`image_seq_len=4096, steps=4, mu=2.291`) |
| Prompt | 21 tokens with the chat template (log); sd.cpp pads FLUX.2 klein prompts to a minimum of 512 tokens (`min_length = 512` in `src/conditioning/conditioner.hpp`, not printed in the log) |
| Problems | none |

### Latency (measured runs, n = 10)

| Stage | Median (ms) | Min–max (ms) | Share of total |
|---|---|---|---|
| Text encode (Qwen3, incl. tokenization) | 64.7 | 63.8–69.9 | 2.7% |
| Denoise (4 transformer calls + Euler updates) | 1725.4 | 1717.5–1731.0 | 71.2% |
|   per step (steps 0/1/2/3) | 441.0 / 426.9 / 427.5 / 427.3 | 424.6–449.4 | 17.6–18.2% each |
| VAE decode (incl. conversion to uint8) | 603.5 | 601.8–608.8 | 24.9% |
| Other (noise setup, graph setup) | 28.7 | 28.5–29.4 | 1.2% |
| **End-to-end `wall_ms`** | **2423.3** | 2420.5–2428.5 (sd 2.9) | 100% |
| First run (not in median) | 2604.5 | – | text 189.7, denoise 1760.2 (step 0: 485.5), VAE 627.4 |
| Load (`new_sd_ctx`, eager, s) | 5.17 | – | – |

### Memory (device-wide, NVML, GiB)

| Metric | GiB |
|---|---|
| Weights on the GPU (sd.cpp's report) | 15,225 MB = 14.87 GiB (text encoder 7.49, diffusion model 7.22, VAE 0.16) |
| Device used after load | 17.02 |
| Peak during text encode / each denoise step / VAE decode | 16.44 / 17.42 / **22.88** |
| **Peak device-wide during a generation** | **22.88** (set by VAE decode) |
| sd.cpp compute buffers (log) | Qwen3 65 MB, transformer 1,065.5 MB, **VAE 6,658 MB** |
| Harness process peak RSS | 1.19 |

Allocator metrics (`peak_alloc_gib`, `peak_reserved_gib`) don't exist for ggml; see the metrics definitions.

### Repeatability

- Outputs bit-identical across all measured runs: **yes** (`deterministic_output: true`).
- Warm-up: the first warm-up (2425 ms) is already at the measured level. Measured spread (max−min)/median: **0.33%**.
- Functional check: `images/measured-0.png` shows a cat holding a sign that reads "hello world".
- Same configuration on an A100-**SXM4**-40GB node (job 11817893, `gilbreth-n001`, 400 W;
  `results/runs/a100-sdcpp-flux-klein-001__baseline__20260925-114138`): 2294.8 ms (range 2293.1–2297.4), per step
  402–418 ms, VAE 585.4 ms, the same 22.88 GiB peak. That run is on different hardware and is
  not this experiment's result. It is why the Slurm scripts now pin `--constraint=G`.

### Profiler run (diagnostic, Nsight Systems, job 11818087)

Run directory: `results/runs/a100-sdcpp-flux-klein-001__profile__20260925-124703` (node `gilbreth-g005`,
A100-PCIE-40GB). The same configuration and protocol run under `nsys profile --trace=cuda,nvtx`, using the
harness build with NVTX stage ranges. Its 10 measured runs had a median of 2510.3 ms (+3.6% profiler
overhead), and the audit was clean. The report (`profile/trace.nsys-rep`, 4.5 MB) is git-ignored.
Breakdowns come from `scripts/analyze_profile.py` and cover the last (measured) generation:
`profile/denoise_kernels.md` and `profile/other_stages.md` (text encode + VAE decode).

An earlier attempt (job 11817920, `gilbreth-g006`) failed at nsys startup: "Failed to probe the process
(sync). Timeout: 2 sec". It didn't reproduce: on g005, nsys started first time with default, `/tmp` and
`/dev/shm` temp directories.

**Per denoise step** (mean of 4 steps; GPU kernel time):

| Per step | Value |
|---|---|
| NVTX span / GPU busy / idle (ms) | 446.9 / 423.5 / 23.4. The GPU is idle for **5%** of the step |
| Kernels | 1133 in every step, median 120 µs |

| Group (mean per step) | ms | Share | Kernels | Main kernels |
|---|---|---|---|---|
| Flash attention | 148.3 | 35.0% | 25 | `flash_attn_ext_f16<128,128,64,1>` (ggml's tensor-core F16 kernel) |
| GEMM | 116.0 | 27.4% | 89 | cuBLAS `ampere_s16816gemm_bf16_*` (BF16 inputs, FP32 accumulate/output) |
| Copy / dtype conversion | 56.7 | 13.4% | 310 | `cpy_scalar`, `convert_unary`, `cpy_scalar_contiguous` |
| Elementwise `mul` / `add` / other | 24.5 / 17.0 / 14.4 | 13.2% | 519 | `k_bin_bcast`, `unary_op_kernel` |
| Norm | 19.2 | 4.5% | 101 | `rms_norm_f32`, `norm_f32` |
| `cat` | 15.6 | 3.7% | 36 | `concat_cont` |
| memcpy / memset | 11.8 | 2.8% | 52 | host↔device copies; 18.1 ms in step 0, 9.8 ms in steps 1–3 |

- Attention: 25 calls per step over 4608 tokens (B1, H24, D128) reach **~44 TFLOP/s** (4·B·H·S²·D per call).
- The GPU is idle for ~23 ms per step: gaps between graph launches while the host prepares the step.

**Text encode and VAE decode** (same generation):

| Stage | NVTX span (ms) | GPU busy (ms) | Largest components |
|---|---|---|---|
| Text encode | 66.4 | 37.6 | GEMM 18.8 ms; **28.8 ms idle** (host-side tokenization and graph setup) |
| VAE decode | 630.5 | 606.8 | **`im2col_kernel` 403.7 ms (66.5%)**, `group_norm_f32` 56.2, FP16 GEMMs (`ampere_h16816gemm_*`) ~50, `k_bin_bcast` 43.2 |

- The VAE decoder's convolutions run as im2col followed by FP16 GEMMs. The im2col step, which unrolls
  input patches into a matrix, takes two thirds of the decode time. The VAE GEMMs run in **FP16**,
  not BF16.

## Observations

- **Denoising is 71% of warm latency**, at ~427 ms per transformer call. Step 0 is consistently
  ~14 ms slower than steps 1–3.
- **VAE decode is 25% of warm latency (604 ms) and sets the memory peak.** sd.cpp reserves a 6.66 GB
  compute buffer for the decoder at 1024², and device-wide use rises from 17.4 to 22.9 GiB during decode.
- **Text encoding is 65 ms** for 512 padded tokens.
- **The first run costs only 7% more than a warm run** (2604 vs 2423 ms). Most of the extra is in text
  encoding (+125 ms) and denoise step 0 (+45 ms).
- **Eager load takes 5.2 s** and places all 14.87 GiB of weights on the GPU. sd.cpp reports 4.2 s of
  file reads in the smoke-test log.
- `other_ms` is 28.7 ms; it includes graph construction between stages. sd.cpp logs compute-buffer
  allocation for each model in every generation.

## Interpretation

- The engine runs the reference configuration faithfully: same weights (verified value-identical), BF16
  everywhere, everything resident on the GPU, no automatic fitting, graph cutting or caching, and a
  deterministic output.
- The VAE decoder is the largest lever for peak memory and the second largest for time in this engine.
  Its convolutions are lowered to im2col + GEMM. The im2col step takes 404 of ~607 ms of GPU time, and
  its unrolled matrices are the likely reason for the 6.66 GB compute buffer (hypothesis; the buffer
  contents aren't itemized in the log). `--vae-conv-direct` (direct convolution) and `--vae-tiling`
  are the engine's own options aimed at exactly this. Each would be a separately labelled configuration.
- In the transformer, BF16 GEMMs take 116 ms per step. The largest single cost is flash attention at
  148 ms per step (~44 TFLOP/s). The dtype conversions around ggml's FP32 activations (57 ms), host
  gaps (23 ms) and per-step host↔device copies (12 ms) make up most of the rest.

## Next experiment

- One-change variants of this config: `vae_conv_direct`, `vae_tiling`, and quantized diffusion model
  weights (GGUF Q8_0/Q4_0, reported as a changed checkpoint).
- Cross-engine comparison with the PyTorch reference: [compare-a100-pytorch-vs-sdcpp-flux-klein.md](compare-a100-pytorch-vs-sdcpp-flux-klein.md).

## Runner refactor launch failure (2026-09-29)

[Failed run artifacts](../results/runs/a100-sdcpp-flux-klein-001__attempt__20260929-001722/), Slurm job `11839376`, account `you139`,
node `gilbreth-g005`. The [saved config](../results/runs/a100-sdcpp-flux-klein-001__attempt__20260929-001722/config.json) preserves
the pinned BF16 workload and explicitly uses **1 first + 0 warmup + 1 measured**
generation for a functional smoke check, with no profiling or downloads.

The child process failed before model loading: the retained local `engine/stdout.txt`
reports that `libcudart.so.12` could not be found (raw logs are not committed). No generation timestamps or timing
summary exist. The [status traceback](../results/runs/a100-sdcpp-flux-klein-001__attempt__20260929-001722/status.json) instead
shows `FileNotFoundError` for `engine/results.jsonl`, because the wrapper tried to
parse output before checking the child exit code.

The launch scripts now explicitly expose the CUDA 12.6 toolkit's `lib64` directory
through `LD_LIBRARY_PATH`. The wrapper now checks the child exit code before parsing
results and includes the child error in failure status. CPU regression tests cover
this launch failure and a successful exit that fails to create results. A small
sd.cpp-only retry was submitted as job `11839470`; its completed outcome is recorded below. The failed
artifacts are retained unchanged.

## Runner refactor retry passed (2026-09-29)

[Run artifacts](../results/runs/a100-sdcpp-flux-klein-001__attempt__20260929-005352/), Slurm job `11839470`, account `you139`,
node `gilbreth-g010`, A100-PCIE-40GB. The same reduced smoke configuration was used: one first
and one measured generation, no warmups, profiling or downloads. The CUDA 12.6 library
path is explicit in `environment.json`; pinned engine/weights and workload are unchanged.

[Status](../results/runs/a100-sdcpp-flux-klein-001__attempt__20260929-005352/status.json) is complete.
[runs.csv](../results/runs/a100-sdcpp-flux-klein-001__attempt__20260929-005352/runs.csv) has two rows, and
[stages.csv](../results/runs/a100-sdcpp-flux-klein-001__attempt__20260929-005352/stages.csv) has six stages per generation
(text, four denoising steps and VAE). The [summary](../results/runs/a100-sdcpp-flux-klein-001__attempt__20260929-005352/summary.json)
has no engine-audit problems. Load, environment, images and summary artifacts were produced.
The Slurm job completed with exit code 0 in 25 seconds. This verifies the launch fix and
refactored runner's functional path. One measured sample is not a latency distribution or
quality evaluation, and the single-sample determinism flag is not evidence of repeatability.

## Four-step no-cache control: two-generation validation, 2026-10-05

[Run and raw outputs](../results/runs/a100-sdcpp-flux-klein-001__attempt__20261005-193229/), [saved config](../results/runs/a100-sdcpp-flux-klein-001__attempt__20261005-193229/config.json), [environment](../results/runs/a100-sdcpp-flux-klein-001__attempt__20261005-193229/environment.json), [summary](../results/runs/a100-sdcpp-flux-klein-001__attempt__20261005-193229/summary.json), [independent validation](../results/runs/a100-sdcpp-flux-klein-001__attempt__20261005-193229/validation.json). Slurm 11883783 on gilbreth-g000.rcac.purdue.edu; repository snapshot a4b4028e0c13f9ac62b751886bccf7ce0013a617. Measurement definitions: [baseline metrics](../wiki/methods/baseline-metrics.md).

BF16 weights,1024×1024,4 scheduler steps,CFG 1.0,seed0,one development cat/sign prompt. One context,first image and one measured observation,zero discarded warm-ups. Engine and checkpoint revisions, all file hashes, fixed CUDA residency and all option values are preserved in the config. Cache-capable binary SHA256 `d5efff548b7e7b3bd7739c6a94d8b6df99a97aa9f961b6cff565bd3430a8f6cc`. Full benchmark queued as Slurm 11890006; these two observations are a functional gate.

| Metric | Second image, ms |
|---|---:|
| wall_ms | 2411.007000 |
| text_encode_ms | 64.491000 |
| denoise_ms | 1711.380000 |
| vae_decode_ms | 605.955000 |
| other_ms | 29.181000 |

First image 2648.748000ms. Context load 10.696340s (filesystem cache may be warm). Sampled generation device-wide peak 22.880310GiB; harness host peakRSS 1.206726GiB. GPU allocator counters unavailable.

Actual transformer passes per image: [4, 4]; conditioning hits: [0, 0]. Approximate cache skips: [0, 0]. Confirmed mmap filenames: []. Settings, callback order, actual CFG execution counts, phases, CSV arithmetic and summary statistics passed independent checks. Both retained image files were rehashed; saved pixels equal same-binary reference: **True**.

Interpretation: this verifies execution and saved-output behavior for this option. A single later observation cannot establish a latency improvement.

[W&B run](https://wandb.ai/mfaruqi-purdue-university/on-device-diffusion/runs/a100-sdcpp-flux-klein-001__attempt__20261005-193229).

## Full protocol a100-sdcpp-flux-klein-001__baseline__20261005-220435

[Summary](../results/runs/a100-sdcpp-flux-klein-001__baseline__20261005-220435/summary.json), [config](../results/runs/a100-sdcpp-flux-klein-001__baseline__20261005-220435/config.json), [independent validation](../results/runs/a100-sdcpp-flux-klein-001__baseline__20261005-220435/validation.json), [environment](../results/runs/a100-sdcpp-flux-klein-001__baseline__20261005-220435/environment.json). Slurm 11890006, gilbreth-g007.rcac.purdue.edu, 2026-10-06T02:04:35Z to 2026-10-06T02:05:22Z; repo snapshot `a4b4028e0c13f9ac62b751886bccf7ce0013a617`. One context; one first generation, three discarded warm-ups and ten measured generations. Same pinned BF16 workload as the functional gate; all requested settings retained. [Metric definitions](../wiki/methods/baseline-metrics.md).

| Metric | Median, ms | Measured min–max, ms |
|---|---:|---:|
| wall_ms | 2429.042500 | 2416.243000–2439.335000 |
| text_encode_ms | 65.255500 | 63.690000–68.030000 |
| denoise_ms | 1725.524000 | 1717.460000–1735.248000 |
| vae_decode_ms | 605.722500 | 604.704000–615.596000 |
| other_ms | 28.993500 | 28.504000–29.653000 |

First generation 2655.648000ms; context load 10.384540s (filesystem cache may be warm). Sampled generation device-wide peak 22.880310GiB; child host peak RSS 1.223545GiB. No allocator or per-stage RSS measurement.

All14 generations passed callback/actual transformer-call accounting, phase, engine-setting and CSV/summary checks. Actual transformer passes per generation: [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]. Conditioning hits: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]. Approximate steps skipped: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]. Mapped files: []. All14 runner-reported pixel hashes agree; both retained PNGs independently rehashed. Retained RGB equals the same-binary no-cache reference: True.

Interpretation: this establishes a repeated timing distribution for the declared configuration. Selection and deltas belong in the matched comparison; sequential capture conditions were not randomized.

[W&B](https://wandb.ai/mfaruqi-purdue-university/on-device-diffusion/runs/a100-sdcpp-flux-klein-001__baseline__20261005-220435).
