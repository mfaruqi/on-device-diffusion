# Reading and changing the measurement scripts

Start with `benchmark()` in [run_flux.py](run_flux.py), then follow only the helper
that answers your question. It shows the experiment in execution order. The sd.cpp
equivalent is `benchmark()` in [run_sdcpp.py](run_sdcpp.py).

This is Week 1 baseline infrastructure supporting **RQ1**: “Which diffusion
optimization choices transfer across devices and workloads?”
([proposal overview](../wiki/project/overview.md#research-questions)).
It is not yet the planner or an automatic device-selection system.

## Which file does what?

| File | Responsibility | Read when you want to… |
|---|---|---|
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
Python 3.10 environment. The expanded suite has 40 passing CPU checks locally and on Gilbreth, including
Jetson command/schema replay, bad callbacks, monitor cleanup and archive portability.
The refactored Jetson capture is not yet hardware-tested. Full repeated/profiler
validation remains pending and is outside the currently authorized small GPU tests.

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

The scoped readability refactor is complete. Remaining hardware validation is the
Jetson capture and full repeated/profile runs described above; those are separate
from the completed CPU regressions and small A100 execution checks. This layer exposes
observations for later planner work; it does not choose precision or residency policies.
