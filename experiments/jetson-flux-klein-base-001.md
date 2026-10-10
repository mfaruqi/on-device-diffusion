---
type: experiment-record
id: jetson-flux-klein-base-001
status: complete
device: jetson-orin-nano
engine: stable-diffusion-cpp
runs: [jetson-flux-klein-base-001__repeat__20261005-042014, jetson-flux-klein-base-001__profile__20261005-033141, jetson-flux-klein-base-001__attempt__20261005-021728, jetson-flux-klein-base-001__baseline__20261005-005357, jetson-flux-klein-base-001__attempt__20261005-003943]
updated: 2026-10-05
---

# Jetson klein Base: fixed disk-backed Q4 reference

Status: **complete** (2026-10-05); unprofiled 1+3+10 reference finished.

## Question

Can the pinned FLUX.2 klein Base 4B workload execute fifty denoising steps with guidance four under the Jetson memory budget? This is a separately labelled checkpoint/workload reference, serving Week 1–2 image baseline evidence and RQ1.

## Setup

[Attempt config](../configs/jetson-flux-klein-base-q4-512-disk-smoke.json), [full config](../configs/jetson-flux-klein-base-q4-512-disk.json). Transformer Base Q4_0 revision d12671125306ca6b5f6db1b33ed4c80c8511a53f; Qwen3 Q4_K_M and original VAE pinned by revision and SHA256. Files verified before measurement. sd.cpp commit19bbbca1c736bbb9538679fc0ae690cb2b46b492; fixed disk-backed parameters, eager loading, segmentation, prefetch and diffusion flash attention; no step or conditioning reuse, auto-fit or VAE tiling. CUDA computation, 25W mode, four CPU threads.

Workload:512×512, fifty Euler steps with flux2 scheduler, guidance4, seed0, batch1, cat holding hello-world sign. Attempt protocol:one first and one additional observation, no warm-up. The single additional observation is not a warm baseline median. [Metric definitions](../wiki/methods/baseline-metrics.md).

## Results: correctness attempt

[Run](../results/runs/jetson-flux-klein-base-001__attempt__20261005-003943/), [summary](../results/runs/jetson-flux-klein-base-001__attempt__20261005-003943/summary.json), [environment](../results/runs/jetson-flux-klein-base-001__attempt__20261005-003943/environment.json), [status](../results/runs/jetson-flux-klein-base-001__attempt__20261005-003943/status.json).

| Metric | First, ms | Second observation, ms |
|---|---:|---:|
| wall_ms | 287391.903 | 329125.142 |
| text_encode_ms | 2984.582 | 24249.223 |
| denoise_ms | 280235.505 | 299974.532 |
| denoise_step_0_ms | 14389.223 | 34163.124 |
| vae_decode_ms | 3596.113 | 4593.800 |
| other_ms | 575.703 | 307.587 |

Model-context creation:63.883 seconds after file hashing warmed filesystem cache. Whole-capture system RAM peak:6.036GiB, sampled once per second including loading and both images. This is not a per-stage allocator peak.

## Validation and quality scope

Each generation has the expected callback boundaries and fifty denoising steps. Engine logs contain exactly one hundred single-segment transformer executions inside each generation, consistent with two passes per step, and report txt_cfg4.00 and sample_steps50. Requested placement settings, output dimensions, phase counts and summary aggregates passed independent checks. Logged placement transitions do not prove continuous residency.

Both generation RGB hashes agree. The saved image shows a coherent cat holding a legible hello world sign, with no obvious gross corruption. [Inspection receipt](../results/runs/jetson-flux-klein-base-001__attempt__20261005-003943/image-inspection.json). This is functional evidence for one prompt/seed, not a formal quality evaluation.

## Results: repeated reference

[Run](../results/runs/jetson-flux-klein-base-001__baseline__20261005-005357/), [summary](../results/runs/jetson-flux-klein-base-001__baseline__20261005-005357/summary.json), [environment](../results/runs/jetson-flux-klein-base-001__baseline__20261005-005357/environment.json). One first generation, three discarded warm-ups and ten measured generations completed in one model context.

| Metric | Median, ms | Min–max, ms |
|---|---:|---:|
| wall_ms | 325631.833 | 313668.333–330741.966 |
| text_encode_ms | 24357.311 | 15685.698–25337.666 |
| denoise_ms | 298587.185 | 289659.113–301166.262 |
| denoise_step_0_ms | 32847.927 | 23824.512–34732.887 |
| denoise_step_1_ms | 5422.220 | 5419.791–5446.207 |
| vae_decode_ms | 4539.275 | 3598.732–4728.997 |
| other_ms | 292.680 | 275.318–334.149 |

First generation:287017.098ms, reported separately. Context creation:58.437s after hashing warmed file cache. Whole-capture one-second system RAM peak:6.027GiB; observed swap range:0.266–0.274GiB. These include loading and are not per-stage allocation peaks.

All fourteen rows have the same RGB hash, matching the saved PNG pixels. Each generation completed fifty callback steps and one hundred single-segment transformer executions. Recorded guidance, settings, stage boundaries, phase sequence and summary medians/min/max passed independent validation. Visual inspection found a coherent cat and legible sign; no formal quality metric is inferred.

## Cache-capable harness control

[Control run](../results/runs/jetson-flux-klein-base-001__attempt__20261005-021728/), [summary](../results/runs/jetson-flux-klein-base-001__attempt__20261005-021728/summary.json), [validation receipt](../results/runs/jetson-flux-klein-base-001__attempt__20261005-021728/control-validation.json). The separate cache-capable harness retained the no-cache configuration and completed one first plus one measured observation, with no warm-ups. The isolated runner snapshot is92580bb; the original harness and reference were preserved.

First generation:287181.913ms; second observation:321034.480ms. Model-context creation:62.969s; sampled whole-capture RAM peak:6.041GiB. These two observations are a functional control, not repeated timing evidence. Callback boundaries, fifty steps, one hundred transformer passes per generation, settings, CSV aggregates and saved image hashes passed independent validation. Cache audit reports zero enabled generations. Saved control pixels match the no-cache reference image exactly; this checks harness output preservation on this prompt/seed only.

[W&B control](https://wandb.ai/mfaruqi-purdue-university/on-device-diffusion/runs/jetson-flux-klein-base-001__attempt__20261005-021728).

## Targeted Base profile

[Run](../results/runs/jetson-flux-klein-base-001__profile__20261005-033141/), [capture scope](../results/runs/jetson-flux-klein-base-001__profile__20261005-033141/profile/capture.json), [validation](../results/runs/jetson-flux-klein-base-001__profile__20261005-033141/profile-validation.json), [derived process-restricted analysis](../results/runs/jetson-flux-klein-base-001__profile__20261005-033141/profile/osrt-analysis/stage_kernels.json). Nsight Systems traces generation4 after one first and three warm-up generations. The profiler was attached throughout; initial model loading is outside the captured range. The profile is separate from unprofiled baseline timing.

The captured generation took 328.655s by host callbacks. Whole-capture sampled system RAM peaked at 6.089GiB, including loading and untraced generations. All five generations completed fifty steps and one hundred transformer passes; saved RGB pixels match the unprofiled no-cache reference. SQLite integrity, single captured generation, stage label counts and summary accounting passed.

| Captured stage | NVTX span, ms | GPU coverage, ms | Read calls without GPU coverage, ms | Neither covered, ms |
|---|---:|---:|---:|---:|
| text_encode | 18697.086 | 2907.132 | 15200.838 | 589.116 |
| denoise_step_0 | 33647.069 | 7469.463 | 25257.750 | 919.856 |
| denoise_step_1 | 5539.422 | 5490.410 | 0.000 | 49.012 |
| vae_decode | 4504.378 | 2974.582 | 848.114 | 681.682 |

Intervals are restricted to the traced process and clipped to stage windows. The three coverage categories sum to each stage span. Read-call coverage means time inside captured read/pread64 calls without overlapping captured GPU activity; it is not measured physical disk wait. GPU activity coverage is not SM utilization, and the uncovered remainder is not proof of idle hardware.

Interpretation: this capture separates read-call-heavy text encoding and the first denoising step from GPU-covered later denoising. It supports investigating residency transitions independently of later-step compute; it does not establish a single whole-pipeline bandwidth or compute bottleneck.

## Repeated reference on the cache-capable harness

[Run](../results/runs/jetson-flux-klein-base-001__repeat__20261005-042014/), [summary](../results/runs/jetson-flux-klein-base-001__repeat__20261005-042014/summary.json), [validation](../results/runs/jetson-flux-klein-base-001__repeat__20261005-042014/repeat-validation.json). The unchanged no-cache config completed one first generation, three discarded warm-ups and ten measured generations in one context. Binary SHA256 is2d66fa04053c02b441d61a84d24412537a16f9099d6cb31b80dcca3c8cc1f896; earlier reference outputs remain preserved.

| Metric | Median, ms | Min–max, ms |
|---|---:|---:|
| wall_ms | 328719.445 | 313962.972–331504.639 |
| text_encode_ms | 24192.235 | 16156.323–24600.976 |
| denoise_ms | 299599.784 | 289221.908–301971.807 |
| denoise_step_0_ms | 33523.779 | 23392.502–34719.475 |
| denoise_step_1_ms | 5428.281 | 5416.511–5607.596 |
| vae_decode_ms | 4598.830 | 3588.229–4890.237 |
| other_ms | 295.655 | 280.565–306.708 |

First generation:286869.307ms. Context creation:62.845s after file hashing. Whole-capture system RAM peak:6.015GiB; swap range:0.303–0.316GiB from4540one-second samples. These include loading, not per-stage memory peaks.

All fourteen generations completed fifty callback steps and one hundred transformer executions, with no cache enabled. Phase order, settings, summary aggregates and PNG hashes passed independent validation. Saved RGB pixels equal the previously inspected no-cache reference. This establishes a repeated no-cache measurement on the cache-capable binary; policy comparison is in the [comparison record](compare-jetson-base-easycache.md).

[W&B repeat](https://wandb.ai/mfaruqi-purdue-university/on-device-diffusion/runs/jetson-flux-klein-base-001__repeat__20261005-042014).

## Interpretation

The longer Base workload is feasible through the full repeated protocol under this configuration. Denoising dominates generation latency. The first denoising step is substantially longer than later steps; host callback timing includes on-demand loading, so this alone does not isolate GPU compute. No cache was enabled. The repeated run establishes a no-reuse timing reference, not formal quality eligibility.

## Next experiment

- Test configs/jetson-flux-klein-base-q4-512-disk-lazy-smoke.json as a separate loading-policy variant.

[W&B attempt](https://wandb.ai/mfaruqi-purdue-university/on-device-diffusion/runs/jetson-flux-klein-base-001__attempt__20261005-003943).

[W&B repeated reference](https://wandb.ai/mfaruqi-purdue-university/on-device-diffusion/runs/jetson-flux-klein-base-001__baseline__20261005-005357).

[W&B profile](https://wandb.ai/mfaruqi-purdue-university/on-device-diffusion/runs/jetson-flux-klein-base-001__profile__20261005-033141).
