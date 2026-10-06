---
type: experiment-record
id: a100-edgedit-flux-klein-001
status: complete
device: a100-pcie-40gb
engine: edge-dit-cpp
runs: [a100-edgedit-flux-klein-001__attempt__20261005-073914, a100-edgedit-flux-klein-001__attempt__20261005-191822, a100-edgedit-flux-klein-001__attempt__20261005-221339, a100-edgedit-flux-klein-001__baseline__20261005-233237]
updated: 2026-10-06
---

# A100 edge-dit: distilled klein no-cache baseline

Status: **complete**; full repeated no-cache baseline validated (Slurm 11891242). Earlier gate evidence is retained below.

## Question

What latency and memory evidence does the pinned performance build provide for the distilled klein workload? Week2 image-engine baselines, RQ1. Engine change relative to configs/a100-flux-klein-bf16.resolved.json; no cross-engine speed claim from this gate.

## Setup

[Config](../results/runs/a100-edgedit-flux-klein-001__attempt__20261005-073914/config.json), [command](../results/runs/a100-edgedit-flux-klein-001__attempt__20261005-073914/command.json), [environment](../results/runs/a100-edgedit-flux-klein-001__attempt__20261005-073914/environment.json). Pinned edge-dit97cc585d9dbd9ac5e30b78ef2e2c4a0d8bfbe9f2, CUDA performance build, ggmlf270346d688c7bec7c56bae2ae73343aba6817ad. Pinned distilled checkpoint e7b7dc27f91deacad38e78976d1f2b499d76a294.1024×1024,four Euler steps,CFG1,guidance1,seed0,one cat/sign prompt. Requested BF16 weights, CUDA backend, flash attention, four threads; cache and VAE tiling off, no automatic placement or offload switches. Scheduler auto explicitly requested.

## Results and validation

[Status](../results/runs/a100-edgedit-flux-klein-001__attempt__20261005-073914/status.json), [engine log](../results/runs/a100-edgedit-flux-klein-001__attempt__20261005-073914/engine.log), [validation](../results/runs/a100-edgedit-flux-klein-001__attempt__20261005-073914/validation.json). Completed11:39:14–11:39:30UTC. Output1024×1024; visually coherent cat holding a readable hello world sign. Image hash and original-file hashes are retained in image-check.json and provenance.json.

The engine reports BF16 model weights and **FP32 transformer activations**. BF16 therefore describes weight storage, not uniform activation arithmetic. The log confirms Flux2 scheduling with mu2.291 and four transitions ending at zero. Pinned source flux_pipeline.cpp selects ed_flux2_sigmas for Flux2 when no explicit flow shift is supplied.

Encode/denoise/decode phase markers are present. The CLI ED_WALL line reports generation=0ms despite nonzero stage markers; that field is unusable for latency. No repeated latency or peak-memory result is reported. The image is a functional diagnostic only, not formal quality eligibility.

## Interpretation

This build can produce an image for the pinned workload. A repeated adapter must use the ed-sample steady-clock generation boundaries, retain the engine-specific precision and schedule labels, and verify stage-marker semantics. Pinned sample_main.cpp creates one context before its repeat loop and times ed_generate_image; PNG encoding follows timing. Its repeats overwrite earlier images; only final-image evidence is retained, so cross-repeat determinism remains unavailable.

## Next experiment

- Any cache variant must use a separately labelled config and pass execution and image-validation gates.

[W&B gate](https://wandb.ai/mfaruqi-purdue-university/on-device-diffusion/runs/a100-edgedit-flux-klein-001__attempt__20261005-073914).

## Two-repeat ed-sample interface gate

[Raw run](../results/runs/a100-edgedit-flux-klein-001__attempt__20261005-191822/), [summary](../results/runs/a100-edgedit-flux-klein-001__attempt__20261005-191822/summary.json), [validation](../results/runs/a100-edgedit-flux-klein-001__attempt__20261005-191822/validation.json), [original timing export](../results/runs/a100-edgedit-flux-klein-001__attempt__20261005-191822/engine/timing.json), [provenance](../results/runs/a100-edgedit-flux-klein-001__attempt__20261005-191822/provenance.json). Slurm11883430 completed23:18:22–23:18:38UTC on the pinned performance build. One context,two identical prompt/seed repetitions,1024²,four steps,CFG1; BF16 weights and engine-reported FP32 transformer activations. Final retained RGB SHA256 c476fc325f80bc3386922415ed1e8c410a6f44ebca2a1d7cb02ca5899accad55. Informal image check: coherent cat holding a readable hello world sign.

Per-pass stdout reports2.283s and1.631s from upstream steady_clock around ed_generate_image, rounded to milliseconds; PNG encoding is outside that interval. No1+3+10 benchmark or sampled memory measurement is claimed. Only the final image survives upstream filename overwriting, so no repeatability claim across the two images is possible.

Raw encode/denoise/decode markers are complete and ordered. Source uses system_clock (epoch time), not CUDA events or a monotonic clock. The encode marker includes prompt conditioning plus latent/schedule setup; decode excludes final tensor-to-image conversion. Preserve those semantics before mapping to shared stage names. timing.json's time_wo_decoding and time_with_decoding both equal full e2e and must not be treated as separate stages. [Pinned source inspection](../output/overnight-20261004/edge-phase-source.txt).

Next: a minimal parser/runner using existing ed-sample repeats, explicit timing limits and unavailable fields; no upstream source modification.

## Adapter implementation status

[Runner](../scripts/run_edgedit.py) and [Slurm entry](../scripts/gilbreth-edgedit.slurm) reuse upstream repeats and shared first/warm-up/measured bookkeeping. The saved two-repeat log replays successfully; the CPU suite passes85tests, including incomplete markers, changed schedule and inconsistent clocks. [Metric definitions](../wiki/methods/baseline-metrics.md#edge-dit-ed-sample-adapter) preserve rounding, clock, memory and image-retention limitations. GPU adapter smoke11890064 subsequently passed; it was submitted from isolated snapshot `aae3cb5`; full benchmark requires that gate.

## Adapter smoke validation, 2026-10-06

This directory is a **two-generation adapter attempt**, not a full baseline. [Summary](../results/runs/a100-edgedit-flux-klein-001__attempt__20261005-221339/summary.json), [validation](../results/runs/a100-edgedit-flux-klein-001__attempt__20261005-221339/validation.json), [config](../results/runs/a100-edgedit-flux-klein-001__attempt__20261005-221339/config.json), [environment](../results/runs/a100-edgedit-flux-klein-001__attempt__20261005-221339/environment.json). Slurm11890064 on g007 completed. First wall1975ms; second wall1629ms, rounded to1ms by upstream stdout. One context and complete ordered stages; unchanged four-step schedule, BF16weights/FP32activations. Whole-child device peak21.606873GiB includes loading and between-image handling; per-generation peak unavailable. Load3.774428s, filesystem cache possibly warm.

Raw-log/CSV/summary/events/memory/image checks passed. Final RGB matches the previously inspected interface image; earlier repeats are overwritten and determinism stays null. Encode_setup is not isolated text encoding, and per-step timing is unavailable. Full1+3+10 job11891242 now submitted after this gate; its metrics require independent validation.

## Full repeated baseline, 2026-10-06

[summary.json](../results/runs/a100-edgedit-flux-klein-001__baseline__20261005-233237/summary.json), [config.json](../results/runs/a100-edgedit-flux-klein-001__baseline__20261005-233237/config.json), [environment.json](../results/runs/a100-edgedit-flux-klein-001__baseline__20261005-233237/environment.json), [validation.json](../results/runs/a100-edgedit-flux-klein-001__baseline__20261005-233237/validation.json). Slurm11891242, g007, isolated snapshot `aae3cb5`. One context, one first generation, three discarded warm-ups and ten measured generations; unchanged pinned four-step 1024² workload, BF16 weights with FP32 transformer activations. No cache or offload.

| Metric | Median, ms | Measured min–max, ms |
|---|---:|---:|
| wall_ms | 1641.000000 | 1637.000000–1644.000000 |
| denoise_ms | 1096.152425 | 1092.038870–1099.545956 |
| vae_decode_ms | 269.868374 | 268.759966–272.795916 |

First generation 2066.000ms; context load 28.357095s (filesystem cache may be warm). Whole-child sampled device peak 21.606873GiB; host peak RSS 2.740219GiB. The device peak includes loading and PNG handling; per-generation peaks and allocator counters are unavailable.

Wall timing is upstream steady-clock output rounded to 1ms; phase intervals use host system_clock, not GPU events. Encode_setup includes conditioning and latent/schedule setup and is not isolated text encoding. Per-step timing is unavailable. Definitions and limitations: [baseline metrics](../wiki/methods/baseline-metrics.md#edge-dit-ed-sample-adapter).

Independent raw-log/CSV/summary/event/phase/schedule/image validation passed for all14 generations. The final retained image matches the previously inspected coherent cat/sign output. Earlier images are overwritten upstream, so cross-repeat determinism remains unknown. No formal quality acceptance or causal cross-engine speed claim follows from this baseline.

[W&B](https://wandb.ai/mfaruqi-purdue-university/on-device-diffusion/runs/a100-edgedit-flux-klein-001__baseline__20261005-233237).
