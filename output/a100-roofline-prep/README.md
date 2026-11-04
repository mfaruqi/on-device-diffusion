# A100 roofline preparation — 2026-10-21

Status: data gathered; estimate-based A100 figures now rendered with the existing script and Matplotlib ([outputs](../roofline-a100-20261021/README.md)). Subsequent hardware-counter probe succeeded, but image capture stopped at a CSV parser error ([procedure](../../docs/a100-hardware-counters.md)); no full-image counters yet. Supports Week 3–4 compiler analysis and RQ1; [project scope](../../wiki/project/overview.md), [milestones](../../wiki/project/milestones.md).

## Gathered inputs

- [inventory.json](inventory.json): 30 completed full benchmark runs selected by the consolidated stage manifest and three historical profile runs; config, environment, workload, precision, optimizations, timing summaries, source hashes and profile-file inventory.
- [runs.csv](runs.csv): compact run/device/software/timing inventory. Times are milliseconds. Profile-associated summary timings remain distinct from the separately instrumented generation.
- [operator-shapes.csv](operator-shapes.csv): 874 recorded stage/shape observations (including repeated shapes across denoising steps and text/VAE stages). Fields are copied from existing analysis tables, not newly measured counters. `reported_analytic_tflops` uses shape-derived FLOPs divided by associated kernel time.
- [tools.json](tools.json): plotting-source provenance, local rendering packages and remotely checked profiling tools.
- Raw evidence remains under [results/runs](../../results/runs/); the inventory points to it rather than duplicating traces. Source-file hashes permit rechecking the selected inputs.

## What is actually available

| Configuration | Full stage timings | Retained kernel profile | Measured HBM bytes |
|---|---|---|---|
| PyTorch eager, original distilled | Yes | 4 denoising stages; 60 GEMM-shape and 4 attention-shape entries | Not found locally |
| PyTorch eager, original Base | Yes | 53 stage entries including text/VAE; 758 GEMM-shape and 52 attention-shape entries | Not found locally |
| sd.cpp distilled | Yes | 4 denoising stages; kernel groups, no explicit GEMM/attention shape entries | Not found locally |
| PyTorch 2.8 eager vs compiled, distilled and Base | Yes, same-version reference pairs | No corresponding retained A100 kernel profiles found | Not found locally |
| edge-dit distilled / Base | Yes | No retained A100 kernel-profile table found | Not found locally |
| Nunchaku distilled | Yes | No retained A100 kernel-profile table found | Not found locally |
| Nunchaku Base | No completed full run in selected inventory | No | No |

This is a local evidence audit, not proof that a file does not exist on an uninspected remote filesystem. No `.ncu-rep` or counter-named file was found under local A100 run directories. Older PyTorch profiles cannot be relabelled as the newer torch 2.8 environment. sd.cpp profiles cannot be assigned another engine's shapes without explicitly labelling an inference.

## Two evidence levels

Existing benchmarks suffice for a planned **analytic roofline overview**: shape-derived work, analytic tensor traffic and saved timing, all labelled as estimates. This requires no new GPU run. The [existing roofline method](../../docs/roofline-plots.md) already provides such formulas; it cannot establish achieved HBM bandwidth or a definitive hardware bottleneck.

A **hardware-counter roofline** is an optional deeper diagnostic when actual traffic/cache/fusion behavior affects the next optimization decision. The capture sequence below is a proposal for that purpose, not a prerequisite for an assumption-labelled overview and not an authorized job submission.

## First proposed comparison

Start with **A100 PCIe 40 GB, distilled Klein 4B, 1024², batch 1, four steps, CFG 1**, and the existing PyTorch 2.8 eager/compiled pair:

- [Eager](../../results/runs/a100-flux-klein-torch28-001__baseline__20261015-151548/config.json), [environment](../../results/runs/a100-flux-klein-torch28-001__baseline__20261015-151548/environment.json).
- [Compiled](../../results/runs/a100-flux-klein-compile-001__baseline__20261015-151740/config.json), [environment](../../results/runs/a100-flux-klein-compile-001__baseline__20261015-151740/environment.json).

Both run directories are recorded as PCIe A100 and torch 2.8.0+cu126. Preserve checkpoint revision, attention backend, dtype, compilation options and residency from the configs. Verify the actual allocated GPU, driver, clocks, power and MIG state again before a future capture. Do not substitute SXM hardware or an updated runtime.

Purpose: locate which kernels improve with compilation and which still dominate. After that, add sd.cpp/edge-dit and the Base eager/compiled pair. Nunchaku is a separately labelled mixed-precision/checkpoint path, not a BF16 utilization comparison. Approximate caches are deferred until executed-work accounting is established; skipped reference FLOPs must not inflate achieved compute.

## Data needed for each roofline point

1. Source run/config fingerprint, engine/version, stage, denoising step, CFG branch and kernel launch identity.
2. Actual tensor dimensions, storage and compute precision, kernel name, launch geometry, and whether fusion/dequantization is included.
3. GPU duration and measured HBM read + write bytes for that same kernel/range and documented replay/cache policy.
4. FLOP basis: executed-instruction accounting where supported and understood, or explicitly labelled mathematical work from verified shapes (GEMM `2*M*N*K`; fused attention needs its own validated work definition). INT4 operations are TOP/s, not interchangeable BF16 TFLOP/s.
5. Separate L2 traffic if using a hierarchical roofline. Never substitute model file size, allocator peak, or PCIe transfers for HBM traffic.

Coordinates, once evidence exists: arithmetic intensity = work / measured bytes; throughput = work / GPU seconds. Use a ceiling appropriate to the compute path. Do not combine wall-clock stage latency with one kernel's traffic. A full-stage point requires complete matched work/traffic coverage and elapsed stage time; do not simply sum overlapping kernel durations.

Existing [roofline tooling](../../docs/roofline-plots.md) uses analytic tensor traffic and estimated stage work. Retain its assumption-labelled historical outputs; do not present those as measured-HBM plots. Its parser supports selected older engine schemas and cache audits, not arbitrary new MFLUX/TaylorSeer runs.

## Toolchain and bounded capture sequence — proposed, not executed

| Tool | Role | Status |
|---|---|---|
| `torch.profiler` + existing portable runner | CPU/CUDA timeline, stage labels, shapes and kernel correlation; compile warm-up before the captured generation | Runner exists; newer A100 eager/compiled profile outputs still needed |
| Nsight Systems | Engine-independent timeline, NVTX regions, copies, launch gaps | Gilbreth CLI 2026.3.2 checked |
| Nsight Compute | GPU duration, HBM/L2 traffic, compute-pipeline counters and Tensor Core roofline sections | Gilbreth 2026.1.1 checked; counter permissions/compatibility not yet tested on a GPU |
| `scripts/analyze_profile.py`, `profile_readers.py`, `profile_metrics.py` | Read PyTorch Chrome / Nsight SQLite profiles and rank dominant kernel groups/shapes | Existing tooling; output is not a DRAM-counter measurement |
| Matplotlib | Static figures, PNG/PDF export | Isolated local environment prepared; estimated A100 figures rendered |
| Plotly / W&B | Interactive charts and published roofline views | Existing exporter retained |

Future capture order:
1. In a Slurm A100 allocation, verify the exact device and run a tiny counter-permission/compatibility check. No GPU workload on the login node; no blanket privilege changes.
2. Capture one warmed generation of the existing eager and compiled configs. Keep compiled first-image cost and profiling overhead separate from the saved ten-image unprofiled latency results.
3. Rank kernels by total time and group repeated shape/kernel signatures. Select representative dominant GEMM, attention and fused/elementwise kernels; record the fraction of runtime covered.
4. Collect targeted Nsight Compute sections/metrics on those launches. Query supported metric names on the allocated GPU rather than assuming a universal list. Candidate fields include `gpu__time_duration.sum`, `dram__bytes_read.sum`, and `dram__bytes_write.sum`; validate availability and units. Tensor Core FLOP accounting must match the A100 instruction path.
5. Save `.ncu-rep`, exported CSV, resolved configs, tool versions and replay/clock/cache controls. Replay can flush caches or serialize work; state that difference explicitly. Do not use profiler wall time as the baseline.
6. Check completeness and counter consistency, then agree the figure layout and only then render. Initially use kernel-level points; add whole-stage points only if the evidence supports them.

No new benchmark or diagnostic job was launched. The later estimate-only rendering is documented in the [plotting instructions](../../docs/roofline-plots.md).

## Reference appearance

The linked [frontier figure](https://github.com/ranery/ShiftAddQwen/blob/main/docs/frontier.png) is a model-size/quality/throughput comparison, not a roofline. Its [Matplotlib script](https://github.com/ranery/ShiftAddQwen/blob/5bf17676f28b0a5411781f1a61798f87cb3e7e05/tools/plot.py) uses STIX/serif type, Okabe–Ito colours, inward ticks, restrained grids, offset labels and PNG/PDF output. The estimated A100 exports use that visual vocabulary; they retain diffusion roofline coordinates and do not reuse its LLM data or axes.

Technical references: [NVIDIA profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html), [Nsight Compute CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html), [PyTorch profiler](https://docs.pytorch.org/docs/stable/profiler.html).
