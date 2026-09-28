---
type: method
summary: Definition of every reported metric (latency, memory, repeatability, profiling), for the PyTorch and stable-diffusion.cpp runners.
status: active
updated: 2026-09-28
---

# Baseline metrics: what we measure and how

This file defines every number that `scripts/run_flux.py` (PyTorch/diffusers) and
`scripts/run_sdcpp.py` (stable-diffusion.cpp) report. Both write the same files and column
names; where an engine measures a column differently or can't produce it, the
[stable-diffusion.cpp runs](#stable-diffusioncpp-runs) section says so. It applies to
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
- On Jetson (shared CPU/GPU memory), use a whole-system physical-memory source.
  Do not add host RSS and GPU allocations: they can overlap in the same physical pool.
- `nvidia-smi.txt` is a single snapshot taken after the measured runs. It is not a peak.

## Jetson CLI feasibility diagnostics

`scripts/analyze_jetson_feasibility.py` extracts diagnostics from a single successful
512×512 CLI invocation and its imported `tegrastats` capture. This is not the repeated
benchmark runner and does not populate its measured-run timing or NVML columns.

- `feasibility-analysis.json → engine_log_diagnostics`: durations printed by the engine
  for initial tensor loading, `get_learned_condition`, `sampling`, `decode_first_stage`,
  and `generate_image`, in seconds. Stage durations include any on-demand weight loading
  inside their engine-defined boundaries; initial tensor loading is not total process
  startup. The difference between generation and summed stage durations is a residual
  of rounded diagnostics, not a separately measured stage.
- `monitor_window.ram_peak_gib`: maximum tegrastats `RAM X/Y` numerator across the entire
  capture window, including initial loading and any idle time. This is reported
  system-wide RAM usage, not process allocation, free memory, or per-stage memory.
  Samples requested every 1000 ms can miss short peaks. Generation logs have no wall-clock
  timestamps, so this analysis does not assign the memory peak to a stage.
- `swap_min_gib`, `swap_max_gib`: min/max reported swap occupancy in that window.
  Constant occupancy does not establish zero swap reads/writes. Do not add swap to RAM.
- Tegrastats prints `MB`; this import treats those values as MiB, consistent with the
  device's reported memory totals, and divides by 1024 for approximate GiB. Raw printed
  integers, local timestamps, and source line numbers remain in `tegrastats-samples.csv`.
- `gpu_temperature_max_c`: maximum sampled `gpu@...C`; this alone does not establish
  whether throttling occurred. Inputs are hashed in the analysis JSON for provenance.

NVIDIA documents the fields in [tegrastats](https://docs.nvidia.com/jetson/archives/r36.4.3/DeveloperGuide/AT/JetsonLinuxDevelopmentTools/TegrastatsUtility.html).
These diagnostics must not be compared as though they were repeated-run medians or
20 ms NVML peaks. Existing benchmark metrics retain their definitions.

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

Per-stage kernel breakdowns come from `scripts/analyze_profile.py --run-dir <profile run>`
(add `--stages` to pick stage labels by regex). A kernel belongs to a stage if it starts inside
that stage's GPU-side annotation span (`gpu_user_annotation`). It is attributed to the aten op
that launched it, found through the CUDA launch's correlation id. "Busy" is the union of kernel
intervals, and "idle" is the span minus busy. GEMM/attention TFLOP/s use the recorded input
shapes (2·M·N·K for GEMMs, 4·B·H·Sq·Sk·D for attention forward).

## Hardware label

Gilbreth's `a100-40gb` partition has two node types: `gilbreth-g*` nodes with **A100-PCIE-40GB**
(250 W power limit, Slurm feature `G`) and `gilbreth-n*` nodes with **A100-SXM4-40GB** (400 W,
feature `N`). The same workload runs measurably faster on SXM4. Runs of different node types are
never compared as the same device. The Slurm scripts request `--constraint=G`, and every run
records the GPU name and power limit in `environment.json` / `summary.json`.

## stable-diffusion.cpp runs

`run_sdcpp.py` runs the C++ harness `engines/sdcpp/bench.cpp`, which loads the model once and calls
sd.cpp's `generate_image` for the same 1 first + 3 warm-up + 10 measured protocol. How to build and
run it: [sdcpp-howto.md](sdcpp-howto.md).

| Column | sd.cpp definition | Difference from the PyTorch runner |
|---|---|---|
| `wall_ms` | host time around `generate_image` | same scope: text encode through uint8 RGB image in host memory; excludes load and saving |
| `text_encode_ms` | generate start → sd.cpp's `get_learned_condition completed` log line | includes tokenization and the Qwen3 forward, as in PyTorch |
| `denoise_step_i_ms` | between consecutive progress callbacks (step 0 fires just before the first transformer call) | includes the Euler update (tiny); in PyTorch that goes to `other_ms` |
| `vae_decode_ms` | `sampling completed` → `decode_first_stage completed` | includes latent unpacking and conversion to uint8 on the host |
| `other_ms` | `wall_ms` minus the named stages | noise/latent setup, graph setup between stages |
| `postprocess_ms`, `gpu_span_ms` | empty | sd.cpp has no separate postprocess stage and no CUDA events |
| `peak_alloc_gib`, `peak_reserved_gib` | empty | no PyTorch allocator. Use `device_used_peak_gib` |
| `device_used_peak_gib` | NVML device-wide used memory, 20 ms sampling (same sampler) | comparable across engines. The wrapper never creates a CUDA context, so the number is sd.cpp's alone |
| per-stage `device_used_peak_gib` (in `stages.csv`) | NVML peak inside each stage's time window | replaces the per-stage allocator peaks |
| `load_total_s` | `new_sd_ctx` with eager loading | reads the files and places all weights on the GPU |

Timing primitive: host `steady_clock` timestamps taken when sd.cpp's log/progress callbacks fire.
ggml computes each graph synchronously, so a callback fires after that stage's GPU work has
finished. The stage times are therefore end-to-end stage latencies, like the PyTorch CUDA-event
times, but they are measured on the host and include any host work inside the stage.

Engine audit: `summary.json → engine_audit` records what sd.cpp's log reports (weight types per
model, flash-attention use, parameter placement, graph segments, sampler/scheduler). The run fails if
any of these happened: auto-fit placement, graph cuts, conditioning-cache hits, weights released
during generation, or requested flash attention not in use.

Profiling: `--nsys` runs the harness build with NVTX ranges (`text_encode`, `denoise_step_i`,
`vae_decode`) under Nsight Systems. `analyze_profile.py` assigns each kernel to the NVTX range that
contains its GPU start time and groups ggml kernels by name (see `categorize_ggml`). There is no
launching-op table for ggml.

## Run naming

Run directories are named `<experiment-id>__<kind>__<YYYYMMDD-HHMMSS>`. Kind `baseline` is a clean
timing run under the full protocol; `repeat` re-runs a baseline to check reproducibility; `profile`
has a profiler attached, so its latency is never quoted; `attempt` is a single feasibility try outside
the protocol, with no medians. Old names and their mapping are in [results/README.md](../../results/README.md#run-directory-names).

### Jetson stage-labelled profile capture

`scripts/profile_jetson_stages.py` uses the same sd.cpp callback/NVTX boundaries defined
above, with the disk-backed quantized configuration. Stage intervals include on-demand
weight loading. It captures one generation, writes no baseline measurement rows, and
retains raw timestamps in `results.jsonl`. Label presence and callback ordering are checked;
GPU timeline attribution requires subsequent analysis. Whole-system tegrastats stays at
one-second sampling, without NVML or derived per-stage memory peaks. Model SHA256 checks
read all files before profiling; this cache condition is recorded in `command.json`.

The Jetson disk-backed capture `20260928-194910` exposed shared tensor-loading and sampling
progress callbacks. The harness now filters callbacks by the configured sampling-step total
and requires the sequence 0 through N; other totals increment `ignored_progress_calls`.
This is scoped to the pinned, untiled single-image configuration; it is not a general event-type
identifier for arbitrary models. Loading remains inside the active stage interval.
The affected capture's denoise timings and labels are invalid
([failure evidence](../../results/runs/jetson-flux-klein-003__profile__20260928-194910/status.json));
do not compare them with corrected captures. Existing captures need callback validation.
