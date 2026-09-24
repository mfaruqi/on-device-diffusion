# Baseline metrics: what we measure and how

This file defines every number that `scripts/run_flux.py` reports. It applies to
all FLUX baseline runs. Experiment notes (e.g. `a100-flux-klein-001.md`) refer to
it rather than repeating the definitions. If the runner changes how a metric is
measured, update this file in the same commit.

Units: time in **ms** (load time in **s**), memory in **GiB** (2^30 bytes).

## Workload (held fixed per experiment)

| Setting | Baseline value | Why |
|---|---|---|
| Model | FLUX.2 [klein] 4B, revision pinned to a commit hash | proposal target; pinning keeps later runs comparable |
| Precision | BF16 weights and activations | reference precision, no quantization |
| Resolution | 1024×1024 | model card reference setting |
| Steps | 4 | distilled 4-step reference configuration |
| Guidance | 1.0 (the distilled model has no CFG, so 1 transformer call per step) | the runner checks for exactly 4 calls |
| Batch | 1 | on-device use case |
| Prompt / seed | one fixed prompt, seed 0, fresh CUDA generator per run | repeatability, not quality |
| Optimizations | none: no offload, tiling, compile, quantization | this is the reference point every later change is compared against |

## Run protocol

1. **Load**: `from_pretrained` from the local cache, then `.to("cuda")`.
2. **First** generation (1): includes one-time costs such as cuDNN/cuBLAS autotuning,
   kernel loading and allocator growth. Reported separately.
3. **Warm-up** generations (3): discarded; check `runs.csv` that times have stabilised.
4. **Measured** generations (10): the only rows used for the medians and ranges in `summary.json`.
5. **Profile** (optional, `--profile`): one extra generation under `torch.profiler`.
   It is excluded from all statistics above.

## Latency

| Metric | Definition | Includes | Excludes |
|---|---|---|---|
| `wall_ms` | Host wall-clock time from just before `pipe(...)` to after a final `cuda.synchronize()` | prompt encoding, denoising, VAE decode, GPU→CPU copy, PIL conversion | model load, downloads, saving the PNG |
| `gpu_span_ms` | CUDA-event time on the stream across the same call | as above | same; should be within ~1% of `wall_ms` |
| `text_encode_ms` | CUDA-event time around `pipe.encode_prompt` | tokenization (CPU) and Qwen3 text encoder forward pass | |
| `denoise_step_i_ms` | CUDA-event time around the i-th transformer forward | one transformer call | scheduler step (goes into `other_ms`) |
| `denoise_ms` | sum of the `denoise_step_i_ms` values | all transformer calls | |
| `vae_decode_ms` | CUDA-event time around `pipe.vae.decode` | VAE decoder | latent unpacking / de-normalisation |
| `postprocess_ms` | CUDA-event time around `image_processor.postprocess` | denormalise, GPU→CPU copy, PIL image | |
| `other_ms` | `gpu_span_ms` minus the named stages | latent/noise setup, timestep schedule, scheduler steps, latent unpack, Python overhead | |
| `load_total_s` | `from_pretrained` + `.to("cuda")` + sync | reading cached files (probably from the OS page cache) and moving weights to the GPU | process start, imports, downloads. **This is not a cold-disk read.** |

Stage boundaries are CUDA events recorded on the stream **without synchronizing**, so
the instrumentation does not change execution. An event time is the time on the
GPU stream between two markers. If the GPU sits idle waiting for the CPU (for example
during tokenization), that idle time counts toward the stage, which matches what a
user would wait for. Host-side durations are also saved in `stages.csv` (`host_ms`),
but they measure how long the CPU took to launch the work, not how long it ran on
the GPU. Do not report them as stage latency.

Report the **median** and the **min–max range** of measured runs. Do not average the
first run into these.

## Memory

| Metric | Source | Meaning |
|---|---|---|
| `peak_alloc_gib` | `torch.cuda.max_memory_allocated` | peak bytes held by PyTorch tensors during the generation |
| `peak_reserved_gib` | `torch.cuda.max_memory_reserved` | peak size of PyTorch's caching-allocator pool, which is ≥ allocated |
| per-stage `peak_alloc_gib` (in `stages.csv`) | peak counter reset at every stage boundary | which stage sets the peak |
| `alloc_start/end_gib` (in `stages.csv`) | `memory_allocated` at stage boundaries | memory still held across stages, e.g. weights plus embeddings |
| `device_used_peak_gib` | NVML device-wide used memory, sampled every 20 ms in a background thread | what the GPU reports as used: allocator pool + CUDA context + cuBLAS workspaces + any other process. It can miss spikes shorter than 20 ms |
| `allocated_after_load_gib` (in `load.json`) | `memory_allocated` after `.to("cuda")` | roughly the resident weights |
| `weights_gib` per component (in `load.json`) | sum of parameter + buffer bytes | text encoder / transformer / VAE sizes |
| `host_rss_gib`, `host_peak_rss_gib` | psutil RSS, `getrusage` max RSS | CPU RAM of the process. The peak includes loading |

Rules:
- **Allocated and reserved are different views of the same memory, so never add them.**
- Allocator numbers **are not** device-wide usage. For comparisons against C++ engines
  (stable-diffusion.cpp, edge-dit.cpp), use `device_used_peak_gib` or an external
  monitor measured the same way for both.
- On Jetson (shared CPU/GPU memory), GPU and host memory come from the same physical
  pool. Report their sum as system usage, and don't double-count mapped memory.
- `nvidia-smi.txt` is a single snapshot taken after the measured runs. It is not a peak.

## Repeatability checks

- `image_sha256` per run: with a fixed seed on one GPU the output should be
  bit-identical. `summary.json → deterministic_output` flags this.
- Look for drift across measured runs, for example from thermal or clock changes or
  other tenants on the node. The node has one exclusive GPU, but the CPU and filesystem
  are shared.
- Seeds are **not** portable across frameworks: the same integer seed in another
  engine does not give the same initial noise.

## Profiling (diagnostic only)

`--profile` writes `profile/trace.json` (large, git-ignored) and `profile/op_table.txt`.
The trace contains labels `generate`, `text_encode`, `denoise_step_i`, `vae_decode`,
and `postprocess`. Open it in Perfetto (https://ui.perfetto.dev) and read **GPU**
lanes for stage durations. The CPU label durations are launch times. The operator
table's nested times overlap, so don't sum them to get end-to-end latency.
Profiler overhead makes that run slower. Never report it as a baseline number.
