---
type: method
summary: Definition of every reported metric (latency, memory, repeatability, profiling), for the PyTorch and stable-diffusion.cpp runners.
status: active
updated: 2026-10-04
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
| `text_encode_ms` | Sum of CUDA-event intervals around `pipe.encode_prompt` | positive encoding and, with CFG, the empty negative-prompt encoding | gaps between encoding calls |
| `denoise_step_i_ms` | Sum of CUDA-event intervals around transformer forwards before scheduler update i | one call without CFG; conditional and unconditional calls with klein Base CFG | guidance combination, gaps between calls and scheduler update (in `other_ms`) |
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

### Klein Base and classifier-free guidance

The [Base configuration](../../configs/a100-flux-klein-base-bf16.json) defines a
separate workload: Base 4B, 50 steps, guidance 4.0, empty negative prompt. The
four-step distilled reference above remains unchanged. Diffusers 0.40.0's
[klein pipeline](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/flux2/pipeline_flux2_klein.py)
encodes the empty negative prompt internally and does not accept a `negative_prompt`
keyword. The runner records that explicit policy in the config, rejects nonempty
values, and does not precompute embeddings outside the measured call.

[StageRecorder](../../scripts/torch_stages.py) advances its logical step counter
after each successful scheduler update. Both CFG forwards receive the same step
name; `stages.csv` preserves the separate calls in execution order.
[The runner](../../scripts/run_flux.py) validates completed scheduler steps, calls
per step and text-encoding counts for this pipeline. `summary.json` records
`guidance_scale` and `transformer_calls_per_step`; `load.json` records `is_distilled`.
The old single-call distilled timing definitions and CSV columns are unchanged;
this does not make Base and distilled workloads interchangeable comparisons.

[The PyTorch trace reader](../../scripts/profile_readers.py) sums disjoint windows
with the same label and unions captured GPU activity within them. Gaps between
those windows are excluded. Overlapping selected windows or GPU activity crossing
a selected window's end are rejected rather than assigned an ambiguous duration.
Existing reports with unique windows are unaffected; reports with repeated names
must be regenerated from the original trace to obtain corrected spans.

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

### Jetson repeated baseline memory

The [repeated runner](../../scripts/run_jetson_sdcpp.py) uses the same sd.cpp callback
timing boundaries and phase summaries as the A100 wrapper, with disk-backed loading
included inside the active stage. One first, three warm-ups and ten measured generations
produce timing medians/ranges; the two-generation smoke config does not establish a baseline.
The earlier single-capture artifacts retain their original definitions.

[Tegrastats capture](../../scripts/jetson_device.py) retains the one-second interval and
printed-MB-as-MiB conversion above. `monitor_window` summarizes the entire harness window,
including model load, generations and untimed raw-image writes. It is not aligned to
generation/stage boundaries and can miss short peaks. RAM, swap occupancy and GPU
temperature are diagnostics, not per-process allocations, swap I/O or proof of throttling.
`device_used_peak_gib` and PyTorch allocator columns stay empty with explanations.
Raw lines and source line numbers remain available; an empty or unrecognized capture fails
explicitly. [CPU tests](../../tests/test_jetson_baseline.py) check this separation.

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

## Derived event records

[measurement_events.py](../../scripts/measurement_events.py) derives `events.jsonl`
after successful capture in the three measurement entry points. This additive artifact
does not change the timing boundaries, CSVs or summary definitions above. It adds no
CUDA synchronization or instrumentation inside timed regions. Historical runs can be
replayed into a separate output directory without rewriting their evidence.

Schema version 1 records `event`, `name`, `run_index`, `phase`, `start_s`, `end_s`,
`duration_ms`, `clock`, `timing_method`, `source` and `details`. The source identifies
the file and line or JSON key. `events-metadata.json` records input SHA256 hashes,
event counts and interpretation limits. Missing values are null, not zero.

| Observation | Evidence and timing |
|---|---|
| PyTorch load/generation | `load.json` / `runs.csv` host durations; absolute timestamps unavailable |
| PyTorch stage | `stages.csv` CUDA-event duration; host launch duration retained in details; absolute timestamps unavailable |
| sd.cpp load/generation/stage | Exact harness host callback intervals from `results.jsonl`, using the existing boundaries above |
| sd.cpp parameter buffers prepared/released | Explicit engine log messages only; backend, printed MB, tensor/block counts and raw message retained |
| sd.cpp tensor load complete | Rounded engine diagnostic duration; not a substitute for harness load time or a measurement of physical disk reads |

Timestamped harness logs and callback records use that process's `steady_clock`.
Untimestamped logs retain null timestamps. Do not align separate processes, runs or
Nsight clocks by their raw values. Records are grouped by source rather than globally
sorted. Parameter-buffer amounts retain the engine's printed MB label; they are not
converted into an inferred allocation peak. Components and destinations are unknown.
A buffer release does not establish CPU offload or a write to storage. Missing log
messages do not prove that no release happened. PyTorch's existing artifacts expose
no weight-release observations. [Regression tests](../../tests/test_measurement_events.py)
cover these distinctions and replay saved A100/Jetson evidence.

## Run naming

Run directories are named `<experiment-id>__<kind>__<YYYYMMDD-HHMMSS>`. Kind `baseline` is a clean
timing run under the full protocol; `repeat` re-runs a baseline to check reproducibility; `profile`
contains a profiler capture. In PyTorch, the extra traced generation follows the unprofiled
protocol; in sd.cpp, Nsight attaches to the process, with either whole-process or explicitly
selected-generation tracing. Profiled durations are diagnostics,
not clean baseline timings. `attempt` marks a feasibility or smoke check outside the full
baseline protocol; a smoke check can contain a one-sample aggregate.
See the [PyTorch runner](../../scripts/run_flux.py), [sd.cpp launcher](../../scripts/gilbreth-sdcpp.slurm)
and [run directory index](../../results/README.md#run-directory-names).

### W&B timing labels

The [exporter](../../scripts/export_wandb.py) preserves timing values and identifies their
measurement conditions. This is viewer metadata; the saved measurements and their boundaries
are unchanged. [Regression checks](../../tests/test_export_wandb.py) replay the saved runs.

| Field | Meaning |
|---|---|
| Config and summary `profiler` | `none`, `torch.profiler`, or `nsight-systems`. Identifies the requested tool, including failed captures; it does not certify successful tracing. Historical values derive from engine and run kind. |
| Config `measurement_scope`; summary `timing/measurement_scope` | `unprofiled`, `profiled`, or `unavailable`, for the generation statistic shown in `timing/generate_s` |
| Config `measurement_basis` | `measured_median`, `single_engine_log`, `single_profile_callback`, or `unavailable` |
| `timing/generate_s` | Measured `wall_ms` median divided by 1000, or an explicitly labelled single engine-log/callback duration |
| `timing/generate_basis`, `timing/generate_samples`, `timing/generate_source` | Human-readable basis, sample count and source file/field |
| `timing/load_scope` | Profiling conditions for `timing/load_s`, independently of generation availability; targeted Jetson capture uses `profiler_attached_untraced` |
| `timing/scope_note` (also config `measurement_scope_note`) | Explains which part of the run was profiled |
| Config `baseline_comparison_eligible`; summary `timing/baseline_comparison_eligible` | True only for a completed, unprofiled, multiple-measured-generation baseline/repeat/profile protocol with CSV phase counts matching config and summary |
| `timing/job_wall_scope` | Explains that job wall time includes loading and all runner work, including profiling where applicable |

A PyTorch profile run's ordinary generation aggregates can therefore be eligible, while
its extra `profile/*` diagnostics remain profiled. sd.cpp profile timings remain visible
with `measurement_scope=profiled` and eligibility false. Single Jetson attempts and captures
are not eligible baseline medians. Eligibility is a protocol filter, not proof of matched
hardware, workload, checkpoint, precision or quality; those conditions must still be matched.
`median/*`, `first/*`, `gen/*`, `profile/*` and `diag/profile_*` retain their existing values.
The W&B built-in Runtime reflects the export process; use the labelled benchmark fields instead.

The [exporter](../../scripts/export_wandb.py) writes `median/*` only to Summary;
[regression tests](../../tests/test_export_wandb.py) verify that no median history rows
are added and all generation history is preserved. The earlier targeted Jetson upload
retains one median-only row ([receipt](../../results/runs/jetson-flux-klein-003__profile__20260930-201956/wandb-export-verification.json));
the exporter no longer creates these automatic panels. Existing history/layouts stay unchanged.
`gen/*` charts contain all generations, indexed from zero; their Summary entries retain the
last logged values. Count generation rows by `gen/*` data, not total W&B history length,
which can also include tables, images and the historical median-only row.

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

### Jetson full-protocol capture

The [full-profile config](../../configs/jetson-flux-klein-q4-512-disk-profile-full.json) preserves
the baseline's first + three warm-ups + ten measured protocol and execution settings, while
Nsight traces initial loading and all 14 generations. This matches the A100 sd.cpp capture
scope. `summary.json.measured` summarizes only generations 4–13 (`n=10`); first and warm-up
rows stay separate. W&B reports a profiled `measured_median` in `median/*` summary fields.
Load is also profiled and baseline comparison eligibility stays false.

Host callback timing definitions, stage boundaries and one-second whole-capture tegrastats
sampling are unchanged. Trace validation requires the declared counts for initial load,
generation and every stage, plus CUDA kernels. See the [runner](../../scripts/run_jetson_sdcpp.py)
and [validation](../../scripts/jetson_profile.py). The prior five-generation capture remains
a single targeted diagnostic; its results are never relabelled as a ten-sample measurement.

### Jetson targeted later-generation capture

The [targeted config](../../configs/jetson-flux-klein-q4-512-disk-profile.json) uses the
[repeated runner](../../scripts/run_jetson_sdcpp.py) and unchanged baseline execution settings.
Its diagnostic protocol is one first generation, three warm-ups, then one profiled generation,
all in the same model context. Only index 4 is inside the registered `profile_capture` NVTX
range. Nsight is attached throughout but tracing starts and stops at that range; the prefix
and load are not claimed as independent clean baseline measurements. The total generation
count is reduced from 14 to 5 to bound capture cost; this is a profiling protocol change,
not a new optimization or replacement baseline.

The existing callback clock and stage boundaries are unchanged. `runs.csv` retains five rows;
`summary.json.measured` contains only the captured row (`n=1`). W&B labels it
`single_profile_callback`, `profiled`, and baseline-ineligible. The registered trigger starts
before the `generate` range and its host timer; collection stops after both close, before RGB
file writes. The per-generation `profile_capture` boolean confirms harness selection;
`profile/capture.json` is written only after checking one complete NVTX generation and CUDA
kernels in the trace. See the [validation code](../../scripts/jetson_profile.py).

Tegrastats still covers the entire five-generation process, including load, at one-second
sampling. Its peak is not the captured generation's peak. Nsight trace generation index 0
corresponds to harness generation index 4. The original one-generation startup profile and
this later-generation trace have different contexts and must be labelled separately.

Imported stage profiles can be reviewed with `scripts/review_jetson_profile.py`. It uses the
existing `analyze_profile.py` GPU-start-in-NVTX-window rule and rejects captured activities
crossing a stage end. Kernel-table busy time is the union of captured GPU activity intervals;
uncovered time is not a measurement of disk I/O. Tegrastats samples retain the existing
whole-window shared-memory convention, without per-stage alignment. Profile host diagnostics
remain separate from repeated unprofiled baseline metrics.

### Captured read-call / GPU coverage

`analyze_profile.py --out-dir <separate-directory>` additionally reports `osrt_coverage`
when an Nsight SQLite export contains OSRT calls and process identifiers. It selects the
process owning the chosen `generate` range, includes its worker threads, and clips captured
`read` and `pread64` intervals to each NVTX stage. GPU activities are also process-restricted;
activities crossing a stage boundary fail validation. Process keys follow NVIDIA's
[serialized identifier definition](https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html#serialized-process-and-thread-identifiers).

For each stage, `gpu_covered_ms` is the union of captured kernel, memcpy and memset intervals;
`read_only_covered_ms` is the union of reads and GPU intervals minus GPU coverage;
`neither_covered_ms` is the stage span minus their combined union. These three terms sum to
the stage span without double-counting concurrent threads or GPU/read overlap. Time inside
read calls is not necessarily physical storage wait; neither-covered time is not necessarily
idle. Capture thresholds can omit short OSRT calls. Missing OSRT data produces no coverage
fields, not zero read time. These are new profile diagnostics; baseline CSV timing definitions
and previously saved reports are unchanged. Multi-process traces are now filtered explicitly.

### Approximate sd.cpp step reuse

Optional EasyCache uses the same host callback stage boundaries. Each denoise_step_i is
a scheduler step, including cache checking, residual application and any actual transformer
computation; it is not a count of full transformer evaluations. Thus cache overhead stays
inside the measured generation. summary.json.cache_audit records the explicit policy
parameters, initialization count and skipped-step counts from each generation's engine log.
A missing initialization or completion report fails validation. No-cache timing definitions
are unchanged. Image agreement and formal quality eligibility remain separate from speed.

### Paired saved-image diagnostics

`quality-diagnostics.json`, when present, describes a separate evaluation of saved images;
these values never enter generation latency or memory summaries. Decode full-resolution RGB
with no crop or resize. `psnr_db` is 10 log10(255²/MSE), where MSE averages squared differences
over all decoded 8-bit RGB values. Exact pixel equality is reported separately; do not encode
infinity as a JSON number. Image-file and decoded-pixel SHA256 hashes identify the inputs.

`lpips_alex_v0_1` uses the calibrated AlexNet LPIPS network, version0.1, with RGB tensors
scaled to [-1,1], float32 CPU evaluation/inference mode and no spatial averaging override.
Record package versions and both backbone/calibration weight hashes. A same-image distance
check must be approximately zero. These conventions follow the [official LPIPS implementation](https://github.com/richzhang/PerceptualSimilarity).
Lower LPIPS indicates greater perceptual similarity; neither LPIPS nor PSNR establishes
prompt alignment or the proposal's formal quality requirement. A single development prompt
is not a held-out test. Missing evaluators produce an explicit unavailable value, never zero.

### Exact sd.cpp conditioning reuse

A labelled sd.cpp A100 or Jetson variant can set `optimizations.conditioning_cache_size`: one entry without
CFG, two with CFG, or zero for the unchanged no-cache reference. The pinned engine caches
positive and negative conditions separately. With a repeated fixed prompt, the first generation
must report zero hits and each later generation must report the configured number;
`summary.json.conditioning_cache_hits_per_generation` contains those validated counts.
The engine log's total is checked independently by `engine_audit.conditioning_cache_hits`.

`text_encode_ms` retains the same host boundary: generation start through the engine's
`get_learned_condition completed` log. A hit therefore measures conditioning retrieval and
associated setup, not a fresh encoder forward. It is not set to zero or removed from wall time.
Compare generation latency across policies, while describing this stage as conditioning time
for cache-enabled runs. First-run encoding remains separate. Old no-cache measurements are
unchanged. See [callback validation](../../scripts/jetson_profile.py),
[runner audit](../../scripts/run_jetson_sdcpp.py) and [host timing](../../engines/sdcpp/bench.cpp).

### sd.cpp mapped-I/O receipt

For explicitly labelled sd.cpp A100 or Jetson mmap variants, `summary.json.engine_audit.mmap_io` records
`requested` and `confirmed_files`. Confirmation requires the pinned engine's successful
file-mapping log for each model component and no mapping fallback. This is an execution
audit, not a memory or latency metric; stage boundaries and sampled system-memory definitions
remain unchanged. File mapping does not prove zero-copy GPU access, continuous residency,
or reduced physical disk traffic. See [audit implementation](../../scripts/run_jetson_sdcpp.py)
and [pinned source excerpt](../../raw/engine-evidence/mmap-source-evidence.txt).

## edge-dit ed-sample adapter

`scripts/run_edgedit.py` wraps the pinned, validated four-step no-cache `ed-sample`
interface. It requests zero upstream warm-ups and `first + warmup + measured`
repeats in one loaded context, then labels and filters those observations through
our shared protocol helpers. No engine source changes or additional GPU syncs are
introduced. PNG saving occurs between repetitions, outside generation timing.

- `wall_ms`: upstream `steady_clock` duration around `ed_generate_image`, parsed
  from per-pass stdout rounded to 1 ms. Includes the returned image conversion,
  excludes PNG encoding and context loading. Medians/min–max use measured rows only.
- `load_total_s`: upstream `model_load_seconds` around context creation. Filesystem
  cache may already be warm; this is not a claim of cold disk loading.
- `stages.csv` `host_ms`: differences between upstream `system_clock` phase markers,
  not GPU event durations. `denoise` spans the entire sampling loop; `vae_decode`
  spans VAE decode and excludes final tensor-to-image conversion. `encode_setup`
  includes prompt conditioning and latent/schedule preparation. It is not isolated
  text encoding: `text_encode_ms` stays unavailable. `other_ms`, `postprocess_ms`
  and per-step timings also stay unavailable. Marker order and containment in the
  rounded generation interval are checked; undetected wall-clock adjustments remain
  a limitation. Do not align those markers to the monotonic NVML sample clock.
- `capture_device_used_peak_gib`: device-wide NVML peak over the entire child
  process, including load and between-image PNG handling. Per-generation
  `device_used_peak_gib` stays unavailable. Raw sampler observations are retained.
- `host_peak_rss_gib`: maximum child RSS (`RUSAGE_CHILDREN`, Linux KiB converted
  to GiB); not per-stage memory. Allocator metrics are unavailable.
- Upstream overwrites each prompt's PNG on repeat. `output.png` is the final
  measured image, with its actual index in `image-retention.json`; only that row has
  a pixel hash. `deterministic_output` is null, since earlier images are unavailable.
- The validated build reports BF16 weights and FP32 transformer activations.
  Preserve that distinction in engine comparisons. `timing.json`'s
  `time_wo_decoding` and `time_with_decoding` both duplicate its full e2e total;
  neither is a stage metric. Full e2e aggregate also includes the first and warm-up
  observations requested by this adapter, so it is not the measured median.

Source semantics and functional evidence: [edge-dit record](../../experiments/a100-edgedit-flux-klein-001.md).
