# a100-flux-klein-001: FLUX.2 [klein] 4B reference baseline on one A100-40GB

Status: **baseline complete** (2026-09-23, Slurm job 11807390). Profiler run complete (job 11807396).

## Question

How long does one complete FLUX.2 [klein] 4B generation take on the A100, and how is
that time and memory split between text encoding, denoising and VAE decoding? This is
a performance baseline and functional smoke check. It is **not** a quality benchmark.

## Setup

- Config: `configs/a100-flux-klein-bf16.json` → pinned in `configs/a100-flux-klein-bf16.resolved.json`
- Model: `black-forest-labs/FLUX.2-klein-4B` @ `e7b7dc27f91deacad38e78976d1f2b499d76a294`
- Workload: BF16, batch 1, 1024×1024, 4 steps, guidance 1.0, prompt
  "A cat holding a sign that says hello world", seed 0
- No quantization, offloading, VAE tiling or `torch.compile`
- Protocol: 1 first + 3 warm-up + 10 measured generations (+1 optional profiler generation)
- Device: Gilbreth `a100-40gb` partition, account `you139`
- Environment: see `environment.json` / `environment-freeze.txt` in the run directory
- Metric definitions: [baseline-metrics.md](baseline-metrics.md)
- How to run: [gilbreth-howto.md](gilbreth-howto.md)

## Results

Run directory: `results/runs/a100-flux-klein-001-20260923-221530`   Slurm job: `11807390`   Node: `gilbreth-g008`
GPU: NVIDIA A100-PCIE-40GB, driver 590.48.01 · torch 2.5.1+cu121 · diffusers 0.40.0 · transformers 5.16.1
Repo commit at run time: `f91e2c8` (kit files were still uncommitted; see `environment.json`)

### Latency (measured runs, n = 10)

| Stage | Median (ms) | Min–max (ms) | Share of total |
|---|---|---|---|
| Text encode (Qwen3, incl. tokenization) | 47.0 | 46.2–48.2 | 3.8% |
| Denoise (4 transformer calls) | 980.8 | 976.9–984.0 | 79.1% |
|   per step (steps 0/1/2/3) | 242.3 / 245.6 / 245.0 / 247.5 | 241.3–249.5 | ~19.8% each |
| VAE decode | 165.3 | 164.1–166.8 | 13.3% |
| Postprocess (→ CPU PIL) | 39.3 | 33.1–45.0 | 3.2% |
| Other (setup, scheduler, unpack) | 7.8 | 7.4–8.6 | 0.6% |
| **End-to-end `wall_ms`** | **1240.6** | 1233.1–1244.9 (sd 3.4) | 100% |
| First run (not in median) | 3163.2 | – | text 996, denoise 1202 (step 0: 469), VAE 862 |
| Load (cached, s) | 13.7 | – | `from_pretrained` 1.9 s + `.to("cuda")` 11.8 s |

### Memory

| Metric | GiB |
|---|---|
| Weights: text encoder / transformer / VAE | 7.49 / 7.22 / 0.16 (total 14.87) |
| Allocated after load | 14.93 |
| Peak allocated (generation) | **17.32** |
| Peak reserved (generation) | 19.60 |
| Stage that sets the peak | **VAE decode** (+2.4 GiB over weights). Denoise peaks at 15.51 (+0.57), text encode at 15.18 |
| Device-wide used peak (NVML) | 20.61 (≈1 GiB CUDA context/workspaces over reserved) |
| Host RSS: peak (during load) / steady | 7.99 / 1.56 |

### Repeatability

- Outputs bit-identical across all measured runs: **yes** (`deterministic_output: true`)
- Warm-up stabilised by run 1 (the first warm-up); measured spread (max−min)/median: **0.9%**
- Functional check: `images/measured-0.png` shows a coherent cat holding a sign. The text
  reads "hellword" rather than "hello world" (text rendering, not a pipeline error).

### Profiler run (diagnostic, job 11807396)

Run directory: `results/runs/a100-flux-klein-001-20260923-221717-profile`. The 10 baseline runs
in this job had a median of 1236.8 ms (within 0.3% of job 11807390) and bit-identical outputs.
The trace (`profile/trace.json`, 34 MB, git-ignored) is on scratch; open it in https://ui.perfetto.dev.

Self-CUDA time over the whole profiled generation (1.184 s of kernel time; profiler overhead
inflates it, so compare shares only):

| Kernel group | Self CUDA (ms) | Share | Notes |
|---|---|---|---|
| `aten::mm` (BF16 tensor-core GEMM) | 517 | 43.7% | mostly transformer linears (text encoder too) |
| Flash attention (`_flash_attention_forward`) | 149 | 12.6% | SDPA picked the flash backend |
| Elementwise `mul` / `add` | 117 / 48 | 9.9% / 4.0% | unfused modulation / residual ops |
| `group_norm` / conv (cuDNN) | 75 / 58 | 6.3% / 4.9% | VAE decoder |
| `copy_`/`to`, `cat`, `rms_norm` | 59 / 58 / 57 | ~5% each | dtype casts, RoPE/concat, norms |

Takeaway: GEMMs are under half of GPU time. Roughly a third goes to memory-bound
elementwise, norm, concat and copy kernels. Those are what fusion or compilation would
remove, while low-bit GEMM kernels would only speed up the ~44% share. The per-stage split for
these groups is in the trace; it has not been extracted yet.

## Observations

- **Denoising is 79% of warm latency**, at about 245 ms per transformer call. With 4 steps, the
  transformer is the only place where speedups scale with the step count.
- **The VAE is small but expensive**: 0.16 GiB of weights, yet 13% of the time and the source of
  the memory peak (about 2.4 GiB of activations at 1024²). Tiled decoding would target exactly
  this peak.
- **The text encoder is half the weights (7.5 GiB) for 4% of the time.** Its weights sit
  resident through denoise and decode without being used.
- **The first run costs 2.5× a warm run.** Most of the extra is in text encode (+950 ms) and VAE decode
  (+700 ms). These are one-time costs, probably cuBLAS/cuDNN setup and kernel loading. Any
  "cold request" figure for on-device use should report this separately.
- **Load time is dominated by `.to("cuda")` (11.8 s).** The weights are memory-mapped, so the actual
  file read happens during the transfer, not in `from_pretrained`.
- Postprocess (GPU→CPU copy + PIL) varies the most (33–45 ms), since it is host-bound.

## Interpretation

- At BF16 the weights alone (14.9 GiB) are nearly twice the Jetson Orin Nano's 8 GB of *shared*
  memory, and the peak (17.3 GiB allocated, 20.6 GiB device) is larger again. A Jetson run needs
  either quantized weights or staged residency, most likely both.
- The measured split points to three independent levers:
  1. **Transformer compute/precision** (79% of time): low-bit weights with fast kernels, or compilation.
  2. **Text-encoder residency** (50% of weights, 4% of time): encode, then release or offload
     before denoising. Its memory is never needed alongside the VAE peak.
  3. **VAE activation peak** (sets peak memory): tiled decode, or free transformer memory
     before decoding.
- Levers 2 and 3 are memory-management and scheduling questions, which fit RQ3
  (buffer lifetime / stage residency).

## Next experiment

- Extract the kernel mix *within* one denoise step from the trace (the table above covers the whole generation).
- Jetson Orin Nano: find a configuration that fits, then compare the stage split with the A100's.
- Separately labelled A100 configs, one change each: sequential stage residency
  (release the text encoder after encoding), VAE tiling, and a quantized transformer. Record
  quality alongside for the quantized one.
