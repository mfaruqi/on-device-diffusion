---
type: experiment-record
id: jetson-flux-klein-004
status: failed
device: jetson-orin-nano
engine: stable-diffusion-cpp
runs: [jetson-flux-klein-004__attempt__20261005-000226]
updated: 2026-10-05
---

# jetson-flux-klein-004: automatic placement fails during text encoding

Status: **failed**; no completed image or benchmark median.

## Question

Does sd.cpp's automatic placement produce a feasible plan for the established quantized
512×512 workload? This tests the automatic-policy baseline against fixed disk-backed
[003](jetson-flux-klein-003.md), supporting Week 1–2 baseline characterization and RQ1.

## Setup

[Attempt configuration](../configs/jetson-flux-klein-q4-512-autofit-smoke.json) changes the
placement policy to auto-fit with an explicitly unset parameter backend. An explicit
fixed backend would disable auto-fit in this engine. The pinned transformer Q4_0,
text encoder Q4_K_M and BF16 VAE, four Euler/flux2 steps, guidance 1, seed 0, 512×512
image, segmentation, eager loading and 25 W mode are retained from 003.
No conditioning or step cache is enabled.

Planned protocol: first + one measured generation, no warm-ups, to establish correctness
before a repeated baseline. [Environment](../results/runs/jetson-flux-klein-004__attempt__20261005-000226/environment.json) records the clean
engine commit, submodule pins and harness hash. This attempt used the isolated campaign
harness with explicit unset-backend support. Metric definitions:
[shared method](../wiki/methods/baseline-metrics.md).

## Results

[Run directory](../results/runs/jetson-flux-klein-004__attempt__20261005-000226/), [status](../results/runs/jetson-flux-klein-004__attempt__20261005-000226/status.json),
[analysis](../results/runs/jetson-flux-klein-004__attempt__20261005-000226/failure-analysis.json), [requested and reported placement](../results/runs/jetson-flux-klein-004__attempt__20261005-000226/engine-audit.json).

- Model context creation completed in 63.118 s; file hashing had warmed the filesystem cache.
- Auto-fit selected CUDA0 parameter residency for the transformer and CPU parameter residency
  for the conditioner and VAE. These are engine-reported decisions, separate from requested settings.
- The first prompt encoding failed at Qwen3 segment 14/29. The engine reported a
  577.93 MiB device-memory request against 557.17 MiB available.
- No generation completed. The wrapper preserved the partial load/failed-generation JSON,
  header-only CSVs and error. Its status error reports the incomplete generation count;
  the underlying memory failure is in the engine log and extracted analysis.
- Whole-window one-second sampled system RAM peaked at
  6.699 GiB. This is not a synchronized allocation peak.
- No image or quality result exists. The full 1+3+10 baseline was not started.

## Interpretation

This auto-fit decision was infeasible under the recorded shared-memory conditions.
**Hypothesis:** simultaneous CPU-resident parameters and GPU buffers leave insufficient
physical-memory headroom during text encoding. The plan log alone does not establish
whether the allocator, memory budget model or background pressure is the primary cause.
No setting was silently changed and this attempt is excluded from baseline comparisons.

## Next experiment

- Evaluate an explicitly labelled memory-policy variant against the fixed disk configuration.

[W&B failed attempt](https://wandb.ai/mfaruqi-purdue-university/on-device-diffusion/runs/jetson-flux-klein-004__attempt__20261005-000226) retains the failure and diagnostic artifacts.
