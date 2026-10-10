# Reading and changing the measurement scripts

Start with `benchmark()` in [run_flux.py](run_flux.py), then follow only the helper
that answers your question. It shows the experiment in execution order. The sd.cpp
equivalent is `benchmark()` in [run_sdcpp.py](run_sdcpp.py).

This is Week 1 baseline infrastructure supporting **RQ1**: “Which diffusion
optimization choices transfer across devices and workloads?”
([proposal overview](../wiki/project/overview.md#research-questions)).
It is not yet the planner or an automatic device-selection system.

## W&B measurement labels

[export_wandb.py](export_wandb.py) copies saved results to the viewer; it does not run a
benchmark. Timing values remain visible with explicit `measurement_scope` and
`measurement_basis` config fields. In W&B, add these columns beside `timing/generate_s`.
For clean repeated timing comparisons, filter config `baseline_comparison_eligible = true`,
then match hardware, workload, checkpoint and execution settings. This filter alone does
not establish comparable quality or identical configurations.

The PyTorch profile run's timing median precedes its extra traced generation; the sd.cpp
profile median includes Nsight conditions. Single Jetson attempts/captures are labelled
separately. See [field definitions](../wiki/methods/baseline-metrics.md#wb-timing-labels).

`median/*` values are exported only to Summary, for tables and manually configured
comparison charts. They are not logged as history rows or used to trigger automatic
median panels. `gen/*` history contains every generation; only its Summary entry is
the last logged value. The X-axis `generation` is the zero-based image index.
Missing metrics stay absent and profile medians keep their scope/eligibility labels.
The earlier targeted Jetson upload retains its one median-only history row and any
existing workspace panels; this change does not delete remote history or layouts.
`--update` remains metadata-only, and ordinary export skips existing runs.

To refresh **existing** runs in place from Gilbreth's login node:

```sh
~/.venvs/wandb/bin/python scripts/export_wandb.py --update --entity mfaruqi-purdue-university
```

`--update` skips directories that have not been uploaded, preserves other summary fields,
and leaves run IDs, histories and artifacts in place. Without `--update`, the exporter can
upload new runs. Use `--dry-run` to inspect local payloads without connecting to W&B.

## Stacked stage charts

[plot_stage_breakdown.py](plot_stage_breakdown.py) renders the three saved Week 1
unprofiled baselines as interactive Plotly charts, with a separate A100 detail view.
It verifies stage means against the measured CSV rows and writes source hashes and
configuration metadata to `chart-data.json`. Stacks use means so they add to mean wall
time; median totals are annotated separately. The Jetson workload differs in precision,
resolution and residency, so the overview is not a matched GPU-speed comparison.

Requires `plotly`; uploading also requires `wandb` and an authenticated account:

```sh
python scripts/plot_stage_breakdown.py --output output/stage-breakdown
python scripts/plot_stage_breakdown.py --output output/stage-breakdown --upload
```

`--upload` creates a separate analysis run in `mfaruqi-purdue-university/on-device-diffusion`.
The output directory retains its run ID for retries; use a new output directory for a new
analysis run. `--data-json` accepts a previously exported dataset for upload from another
host. No benchmarks are executed or modified. Dependencies are recorded in the analysis run.

## Which file does what?

| File | Responsibility | Read when you want to… |
|---|---|---|
| [run_jetson_sdcpp.py](run_jetson_sdcpp.py) | Run repeated Jetson measurements or one later-generation trace | Start with `benchmark()`, then `read_generations()` and the shared sd.cpp parser |
| [run_flux.py](run_flux.py) | Coordinate the PyTorch experiment; write generation rows | Follow the full experiment |
| [flux_engine.py](flux_engine.py) | Download, validate, load and invoke Diffusers; capture a trace | Understand the actual model call and GPU placement |
| [torch_stages.py](torch_stages.py) | Wrap pipeline stages with CUDA events and allocator counters | Understand timing boundaries and memory peaks |
| [run_sdcpp.py](run_sdcpp.py) | Coordinate the C++ process, collect images, write and audit results | Follow the sd.cpp experiment |
| [sdcpp_engine.py](sdcpp_engine.py) | Validate the engine/weights, construct arguments, interpret callbacks | Understand sd.cpp flags and stage timestamps |
| [measurement.py](measurement.py) | Phase order, summary statistics, artifact naming, status and monitor cleanup | Understand shared experiment rules |
| [measurement_events.py](measurement_events.py) | Normalize saved load, generation, stage and buffer observations | Find the evidence behind an event |
| [benchlib.py](benchlib.py) | NVML memory sampler, environment queries, JSON and numerical helpers | Understand where device memory readings come from |
| [profile_jetson_stages.py](profile_jetson_stages.py) | Coordinate one Jetson profile capture | Follow the Jetson experiment |
| [jetson_profile.py](jetson_profile.py) | Check settings and hashes, build profiler command, validate callbacks | Understand Jetson capture preconditions |
| [jetson_device.py](jetson_device.py) | Start/stop tegrastats and save system memory snapshots | Understand the Jetson monitor |
| [analyze_profile.py](analyze_profile.py) | Read a trace, summarize each stage, write reports | Follow general GPU profile analysis |
| [profile_readers.py](profile_readers.py) | Convert PyTorch/Nsight records into stage activities | Understand timestamps and stage assignment |
| [profile_metrics.py](profile_metrics.py) | Group kernels and calculate busy time, shapes and throughput | Inspect profile arithmetic |
| [profile_report.py](profile_report.py) | Turn computed values into Markdown tables | Change report presentation |
| [review_jetson_profile.py](review_jetson_profile.py) | Validate one Jetson capture and derive review artifacts | Follow the full Jetson review |

These are ordinary function calls with explicit arguments. There is no plugin
registry, class hierarchy, or automatic fallback hidden behind them. The existing
NVML sampler is the A100 memory adapter; Jetson has different measurement sources.

## Walk through the PyTorch experiment

1. `main()` reads the JSON config. `--prepare-only` resolves/downloads the checkpoint
   and exits before measurement. Otherwise it creates a new directory and enters
   `run_status()`, which records success or failure, including a traceback.
2. `check_config()` rejects unsupported precision, unpinned revisions, batch sizes
   and optimization options. `phase_sequence()` makes the first/warmup/measured list.
3. `DeviceMemorySampler(...)` selects the device and sampling interval. `with
   sampling(sampler)` starts monitoring and guarantees it stops when this block exits.
   `with` here means “acquire a resource, and clean it up even on an exception.”
4. `collect_environment()` records versions and hardware. `load_pipeline()` reads
   cached weights, calls `.to("cuda")`, and saves load costs separately from inference.
5. `StageRecorder.install()` wraps text encoding, denoiser calls, VAE decoding and
   postprocessing. It does not edit Diffusers source.
6. The small `generate()` function binds the pipeline, recorder and workload to
   `generate_once()`. This gives the loop a callable requiring no arguments.
7. `measure_generations()` invokes that callable once per phase. `result_row()`
   converts stage observations to a CSV row. Completed rows are flushed immediately;
   image hashing and saving occur after the timed generation.
8. `build_summary()` adds device and load details to `summarize_runs()`. Only rows
   labelled `measured` enter the measured statistics. First-run values stay separate.
9. If requested, `run_profile()` performs one extra generation after the summary.
   Its instrumentation does not enter the clean baseline rows.
10. Back in `main()`, `export_events()` derives event records from the saved measurements.
    It adds no instrumentation inside the measured generation.

To understand one generation, read `generate_once()` in `flux_engine.py`: create a
fresh seeded generator → finish prior CUDA work → reset counters → record the start
event and host time → call the pipeline → record the end event → wait for completion
→ read the stage observations. Stage boundaries themselves do **not** synchronize.
The [metric definitions](../wiki/methods/baseline-metrics.md) remain authoritative.

## Klein Base development

[a100-flux-klein-base-bf16.json](../configs/a100-flux-klein-base-bf16.json) is a
separate 50-step Base workload (guidance 4.0, empty negative prompt), with the
same first + 3 warmup + 10 measured protocol. It is an unresolved preparation
config; no Base checkpoint has been downloaded or benchmarked by this change.

`torch_stages.py` groups transformer calls by the scheduler update they precede.
`run_flux.py` validates two forwards per step and two text encodes when klein CFG
is active; it still expects one of each for the distilled reference. The scheduler
wrapper only advances a counter, so scheduler/guidance work remains in `other_ms`.
The trace reader aggregates repeated labels without counting the gaps between
their windows. No new runner, dependency or shared engine abstraction is needed.

Diffusers 0.40.0 encodes an empty negative prompt internally. The config records
`negative_prompt: ""`; the runner rejects nonempty values and does not pass the
unsupported keyword to the pipeline. See the
[metric definitions](../wiki/methods/baseline-metrics.md#klein-base-and-classifier-free-guidance).

Validation on October 4: 69 CPU tests pass, including saved distilled-run arithmetic,
CFG grouping/count rejection and repeated trace windows. CUDA execution and the
installed Gilbreth pipeline still require a hardware smoke check before a baseline.
Use a separately labelled attempt config with zero warmups and one measured image
after the first image for that check; the runner requires at least one measured row.

After verifying the installed pipeline API, prepare on the Gilbreth login node:

```sh
HF_HOME=/scratch/gilbreth/mfaruqi/huggingface ~/.conda/envs/2025.06-py313/flux_env/bin/python scripts/run_flux.py --config configs/a100-flux-klein-base-bf16.json --prepare-only
```

This pins/downloads the weights but performs no inference. After a successful GPU
attempt, submit the baseline from the repository root (Slurm allocates the GPU):

```sh
mkdir -p logs
sbatch --export=ALL,CONFIG=configs/a100-flux-klein-base-bf16.resolved.json scripts/gilbreth.slurm
```

## What differs for sd.cpp and Jetson?

The A100 sd.cpp coordinator asks `run_harness()` to start **one C++ process**, which
loads once and executes the full protocol. Python does not call the model once per
image. The harness lives in [bench.cpp](../engines/sdcpp/bench.cpp).
`rows_from_results()` reads its timestamps; `audit_log()` checks that the engine
actually honored the reference settings. The wrapper never imports PyTorch or
creates another CUDA context that would affect the device-wide memory measurement.

PyTorch stage durations use CUDA events. sd.cpp stage durations use host callback
timestamps. Missing GPU-event or allocator metrics stay empty for sd.cpp. The shared
summary helper does not reinterpret host time as GPU time.

`--nsys` profiles the sd.cpp harness process, including its protocol. It produces a
separate profile run; unlike PyTorch's `--profile`, it does not add one extra
generation after a clean baseline. Compare baseline runs separately from trace runs.

[profile_jetson_stages.py](profile_jetson_stages.py) starts with `capture_profile()`:
check config → record environment → verify hashes → construct command → monitor and
capture → validate and save. It is **not a repeated baseline**. Its standard-library
helpers ship in the updated [transfer bundle](../bundles/README.md), which is tested
after extraction outside the repository. Shared `run_status()` adds failure traceback
and timestamps; the existing profile summary and header-only CSVs are preserved.

Both device monitors expose `start()` and `stop()` and use `sampling()` for guaranteed
cleanup. NVML provides A100 device memory samples; tegrastats retains Jetson system
RAM/swap as raw text. This common lifecycle does not equate those memory metrics.
The Jetson interval now comes from `protocol.tegrastats_interval_ms` (1000 in the
existing config), and monitor shutdown is bounded. [review_jetson_profile.py](review_jetson_profile.py) analyzes the
resulting trace; [analyze_profile.py](analyze_profile.py) analyzes A100 profiles.

## Compatibility and checks

The CLI flags, valid reference configs, filenames, CSV columns, timing boundaries
and numerical summary definitions are preserved. Both A100 runners now reject
unknown optimization keys, even when disabled, and invalid phase counts. Failure
status retains its start time and traceback; monitor cleanup runs on failures too.
PyTorch preserves completed CSV rows on failure. sd.cpp retains raw harness artifacts
on failure; conversion to CSV still happens after a successful complete protocol.

Run the CPU-only regression checks from the repository root:

```sh
python3 -m unittest discover -s tests -v
```

[test_measurement.py](../tests/test_measurement.py) includes a small fake generation
function, so the protocol can run without weights, PyTorch, Pillow or a GPU. Its
temporary “images” are dummy bytes, not inference results. The tests also reconstruct
rows and summaries from saved A100 artifacts without modifying them. Summary values
match exactly except for a tiny tolerance on standard deviation across Python versions.
These checks validate bookkeeping; they cannot establish CUDA execution or performance.

Slurm job `11839376` ran under `you139` in `/home/mfaruqi/on-device-diffusion` on
A100 PCIe node `gilbreth-g005`. The [PyTorch smoke check passed](../experiments/a100-flux-klein-001.md#runner-refactor-smoke-check-2026-09-29).
The [sd.cpp launch failed](../experiments/a100-sdcpp-flux-klein-001.md#runner-refactor-launch-failure-2026-09-29)
because the CUDA shared-library path was missing. Both sd.cpp launch scripts now
set that path, and the runner reports child-process failures before looking for
results. The [sd.cpp-only retry](../experiments/a100-sdcpp-flux-klein-001.md#runner-refactor-retry-passed-2026-09-29),
job `11839470`, passed on A100 PCIe node `gilbreth-g010` with an empty engine-audit problem list.

These jobs request one GPU for at most five minutes and use 1 first + 0 warmup +
1 measured generation per engine, without profiling or downloads. They are functional
checks, not baseline timing estimates. Submission details, reduced configs and source
hashes are in `logs/refactor-smoke/` on Gilbreth. The A100 checks passed in Gilbreth's
Python 3.10 environment. That refactor's suite had 40 passing CPU checks locally and on Gilbreth, including
Jetson command/schema replay, bad callbacks, monitor cleanup and archive portability.
Jetson repeated measurements subsequently passed; the targeted trace has also passed
[hardware validation](../experiments/jetson-flux-klein-003.md#targeted-later-generation-profile-2026-09-30). See the [current checks](#targeted-jetson-profiling-and-profiler-labels).

For full baseline/profile validation later, with both existing resolved configs and
cached weights available, submit from the repository root:

```sh
mkdir -p logs
sbatch scripts/gilbreth.slurm
sbatch scripts/gilbreth-sdcpp.slurm
sbatch --export=ALL,PROFILE=1 scripts/gilbreth.slurm
sbatch --export=ALL,PROFILE=1 scripts/gilbreth-sdcpp.slurm
```

`sbatch` requests a GPU compute node; do not run the benchmarks on the login node.
Compare these new runs with the same pinned configurations for output hashes, stage
counts, timing/memory distributions and audits. Variation in live timings is expected.

## Reading the analysis scripts

`analyze_profile.analyze()` selects a trace reader and calls `summarize_stage()` for
each stage. Both readers produce microsecond timestamps and activity records, so the
metric calculations do not branch on engine type. Nsight databases open read-only
and close when analysis finishes. Busy time merges overlapping activity intervals;
summed kernel/activity time retains overlap. Those values have different meanings.

`review_jetson_profile.review_profile()` validates callbacks and NVTX windows, parses
system-memory samples and segmentation diagnostics, then writes reports. Each of
those steps is a separately named function in the same file. The optional `--out-dir`
keeps derived files separate from the input run; writing inside `originals/` is refused.
The input still needs the original SQLite trace, tegrastats log and RGB bytes, which
are ignored by Git and are absent from a small-artifact-only checkout.

Regression checks reproduce all five saved Markdown reports exactly. Synthetic
PyTorch and Nsight traces exercise launch correlation, tensor shapes, overlapping
GPU work, generation selection and stage-boundary validation. A synthetic end-to-end
Jetson review preserves input hashes and the original derived output formats. This
does not claim a fresh replay of raw traces that are unavailable in the checkout.

## Reading the event records

The three capture entry points now write `events.jsonl` and `events-metadata.json`
after successful measurement. Start with `export_events()` in
[measurement_events.py](measurement_events.py), then read the adapter for the engine.
Each record points to its input file and line or JSON key; metadata hashes the inputs.
The [event definitions](../wiki/methods/baseline-metrics.md#derived-event-records)
describe the clocks, units and unavailable observations.

PyTorch records preserve durations; its existing CSVs have no absolute stage timestamps
or weight-release observations. sd.cpp records preserve exact harness callback intervals
and any explicit parameter-buffer messages in the engine log. A release message does
not establish a transfer to CPU or a write to the SD card. Unknown destinations and
timestamps stay null. Records are grouped by source, not sorted into a global timeline.

To normalize an existing run without changing its evidence:

```sh
python3 scripts/measurement_events.py \
  --run-dir results/runs/a100-sdcpp-flux-klein-001__attempt__20260929-005352 \
  --engine sdcpp --out-dir logs/event-review
```

The scoped readability refactor is complete. The repeated Jetson baseline has passed;
the new targeted profile mode has also passed hardware validation. This layer exposes
observations for later planner work; it does not choose precision or residency policies.

## Targeted Jetson profiling and profiler labels

The standard sd.cpp profile now uses the
[full-protocol config](../configs/jetson-flux-klein-q4-512-disk-profile-full.json): one first,
three warm-ups and ten measured generations, with loading and all generations traced.
`full_profile_command()` omits capture selectors, and `validate_profile_trace()` requires
14 occurrences of each stage plus initial loading. Median statistics still use only the
ten measured rows and retain profiled/baseline-ineligible labels. CPU checks and
[Jetson hardware validation](../experiments/jetson-flux-klein-003.md#full-protocol-profile-2026-09-30) pass. See the
[transfer/run procedure](../wiki/methods/jetson-howto.md#full-protocol-profile).
The targeted mode below is retained for bounded diagnostics; existing results are unchanged.

`run_jetson_sdcpp.py` reads optional `profiling` config metadata. The
[targeted config](../configs/jetson-flux-klein-q4-512-disk-profile.json) selects the NVTX
binary and a five-generation diagnostic protocol. In `jetson_profile.py`,
`targeted_profile_command()` checks harness capabilities and arms Nsight for the registered
`profile_capture` range. `bench.cpp` opens that outer range only around generation 4;
the existing callback ranges/timestamps stay unchanged. `read_generations()` checks the
harness's per-generation selection receipt, then `validate_targeted_trace()` checks the
exported SQLite ranges and CUDA kernel presence before the runner summarizes results.

The archive includes the new config. [Transfer/build/run commands](../wiki/methods/jetson-howto.md#profile-a-later-generation)
describe the validated target procedure. The suite now has 58 passing CPU checks,
including fake-engine five-generation integration, stale-binary rejection and malformed
trace rejection. Both plain and NVTX C++ harness syntax are checked against Gilbreth's
installed engine/CUDA headers; this does not exercise GPU execution or Nsight collection.

`export_wandb.describe()` adds `profiler` to config and summary: `none`, `torch.profiler`,
or `nsight-systems`. Existing runs can receive this label without replacing history or
measurements. Failed runs retain the requested tool name; check status and trace evidence
before claiming a successful capture. The [scope definitions](../wiki/methods/baseline-metrics.md#wb-timing-labels)
still determine which durations belong in baseline comparisons.

### Optional sd.cpp EasyCache gate

The A100 adapter now accepts only one explicitly parameterized step-cache policy:
EasyCache with mode, reuse_threshold, start_percent and end_percent in optimizations.step_cache.
The no-cache path emits the same harness arguments as before. Every generation must log
the requested EasyCache initialization and a completion/skip report; otherwise the run fails.
summary.json.cache_audit records skipped steps. Scheduler-step timing still includes cache
checks and reuse work, so a skipped transformer evaluation is not a missing denoising step.
The Base EasyCache config is prepared for a correctness attempt after the Base no-reuse gate.
CPU tests pass; this new policy branch has not yet been validated on a GPU.

The pinned sd.cpp implementation rejects UCache for DiT models. TeaCache and DiCache are not
names accepted by this adapter; do not substitute EasyCache under either name.

Jetson Base EasyCache configurations use the same explicit policy validation and per-generation log audit as the A100 adapter. A cache request requires a separately rebuilt harness advertising EasyCache support; no-cache arguments are unchanged. GPU execution and image diagnostics remain gates, and approximate reuse is not quality eligibility.

Jetson prefetch variants set `optimizations.prefetch` explicitly (`false` disables parameter prefetch) and must match `harness_arguments.disable-prefetch`. Absence preserves the reference with prefetch enabled. The existing requested-context audit verifies the emitted flag; it does not prove physical I/O overlap. Separate labelled smoke/full configs keep eager loading and all other settings fixed.

Jetson exact conditioning reuse uses `optimizations.conditioning_cache_size`, matching the
engine setting and harness argument. Zero preserves the reference; a repeated fixed prompt
requires one entry without CFG or two with CFG (positive and negative conditions). The first
generation must have zero hits, all later generations the expected count, and the independent
engine-log total must agree. This is exact prompt reuse, separate from approximate step caches.
The existing completion callback also fires on cache hits; `text_encode_ms` includes retrieval
and setup and must not be described as encoder execution for cached generations. See the
[metric definition](../wiki/methods/baseline-metrics.md#exact-sdcpp-conditioning-reuse).

Jetson weight-file mapping uses explicit `optimizations.mmap` (default false), mirrored in
the harness argument. A mapped-I/O run must log successful mappings for all three pinned
component files and no mapping fallback; otherwise the run fails and keeps its evidence.
`summary.json.engine_audit.mmap_io` records requested state and confirmed filenames. These
receipts establish engine-reported file mapping, not zero-copy GPU access, continuous weight
residency, or measured physical I/O savings. The Base mmap configs are prepared for a separate
correctness gate; GPU execution is not yet validated. No engine or harness rebuild is needed.

### Original four-step optimization tests

The distilled workload now has separately labelled EasyCache, exact conditioning reuse,
prefetch-disabled and mmap configurations for both A100 BF16 1024² and Jetson Q4 512².
Each has a two-generation correctness config and a full1+3+10config; cache-control configs
retain no reuse on the current harness. Distilled CFG1 needs one conditioning entry; Base
CFG4 needs two. EasyCache retains explicit threshold0.2/start0.15/end0.95; zero skipped steps
is a valid observed result, not a reason to silently change its parameters.

The A100 adapter uses the same strict expected conditioning-hit counts and mapping receipts;
reference defaults stay unchanged. Stage times retain the existing definitions, including
conditioning retrieval inside the text stage. Memory mappings confirm file I/O policy only.
No changes to the PyTorch reference pipeline or C++ engine are required for these tests.

### edge-dit minimal repeated adapter

`run_edgedit.py` uses the existing pinned `ed-sample` repeat interface for the
four-step no-cache A100 configuration. `gilbreth-edgedit.slurm` accepts `CONFIG`
and `KIND` overrides; default is a two-generation adapter smoke attempt. After
that validates, use `configs/a100-edgedit-flux-klein-bf16.json` with `KIND=baseline`
for first + three warm-ups + ten measured images in one context. No upstream C++
changes are required. Unsupported optimizations fail explicitly.

Wall measurements have 1 ms stdout precision. Encode includes setup, per-step
latency and per-generation memory are unavailable, and only the final PNG survives
upstream repeats. See [metric definitions](../wiki/methods/baseline-metrics.md#edge-dit-ed-sample-adapter)
before comparing stages or claiming deterministic images.
