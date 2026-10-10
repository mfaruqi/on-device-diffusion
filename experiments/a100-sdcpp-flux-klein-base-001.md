---
type: experiment-record
id: a100-sdcpp-flux-klein-base-001
status: partial
device: a100-pcie-40gb
engine: stable-diffusion-cpp
runs: [a100-sdcpp-flux-klein-base-001__attempt__20261005-040120, a100-sdcpp-flux-klein-base-001__baseline__20261005-080118, a100-sdcpp-flux-klein-base-001__profile__20261005-081219, a100-sdcpp-flux-klein-base-001__attempt__20261005-082117, a100-sdcpp-flux-klein-base-001__profile__20261005-191522, a100-sdcpp-flux-klein-base-001__baseline__20261005-214624]
updated: 2026-10-06
---

# A100 sd.cpp klein Base: fifty-step BF16 reference

Status: **partial**; correctness attempt completed and validated, job11880759. Full baseline11882493 passed; profile11882497 failed before generation with a Nsight process-probe timeout. Retry11883400 also failed; see the preserved second failure below.

## Question

Can pinned klein Base4B execute at BF16,1024²,50steps and guidance4 with fixed GPU parameter placement and correct CFG callback accounting? This is Week1–2 image baseline evidence for RQ1, a separate checkpoint/workload reference.

## Setup

[Saved resolved config](../results/runs/a100-sdcpp-flux-klein-base-001__attempt__20261005-040120/config.json). Checkpoint black-forest-labs/FLUX.2-klein-base-4B at a3b4f4849157f664bdbc776fd7453c2783562f4d; BF16 transformer/text encoder/VAE, no quantization or reuse. Batch1,1024×1024,50steps, guidance4, seed0, fixed cat/sign prompt. Effective Flux2 scheduler is confirmed in engine logs. Eager CUDA0 parameters and execution, automatic fitting and segmented computation disabled, conditioning cache zero, diffusion flash attention enabled, four CPU threads. Protocol:one first plus one additional observation, no warm-ups; not a repeated baseline.

Node gilbreth-g001.rcac.purdue.edu, NVIDIA A100-PCIE-40GB. sd.cpp 19bbbca1c736bbb9538679fc0ae690cb2b46b492, ggml 4bf5f6000653b7881d00963cd6ddb665ccd62a8d; isolated runner snapshot 1e983ca8986116dd04970740909104072eaebc97. [Environment](../results/runs/a100-sdcpp-flux-klein-base-001__attempt__20261005-040120/environment.json), [metric definitions](../wiki/methods/baseline-metrics.md). Stage timing uses host callbacks around synchronous ggml execution, not CUDA-event stage measurements.

## Results

[Run](../results/runs/a100-sdcpp-flux-klein-base-001__attempt__20261005-040120/), [summary](../results/runs/a100-sdcpp-flux-klein-base-001__attempt__20261005-040120/summary.json), [status](../results/runs/a100-sdcpp-flux-klein-base-001__attempt__20261005-040120/status.json).

| Metric | First, ms | Second observation, ms |
|---|---:|---:|
| wall_ms | 43720.672 | 43850.116 |
| text_encode_ms | 315.652 | 140.632 |
| denoise_ms | 42731.177 | 43070.483 |
| vae_decode_ms | 645.831 | 612.059 |
| other_ms | 28.012 | 26.942 |

Context creation: 11.434s; filesystem page cache may be warm. Device-wide sampled peak during the second generation: 22.880GiB. Harness peak host RSS: 1.205GiB. Device-wide usage is not a PyTorch allocated/reserved metric; those are unavailable here.

## Validation and observations

[Validation receipt](../results/runs/a100-sdcpp-flux-klein-base-001__attempt__20261005-040120/validation.json). Both generations completed50steps with51progress callbacks and100single-segment transformer passes, consistent with CFG. Engine audit found BF16 weights, CUDA0 parameter placement and no forbidden release/offload behavior. Callback boundaries, stage-plus-other accounting, phase order, summary aggregates and PNG hashes were independently checked.

Both1024×1024saved images have identical RGB hashes. Visual inspection found a coherent cat holding a readable hello world sign with no obvious gross corruption. This one-prompt/seed check is functional evidence, not formal quality eligibility.

## Full unprofiled baseline

[Run](../results/runs/a100-sdcpp-flux-klein-base-001__baseline__20261005-080118/), [summary](../results/runs/a100-sdcpp-flux-klein-base-001__baseline__20261005-080118/summary.json), [validation](../results/runs/a100-sdcpp-flux-klein-base-001__baseline__20261005-080118/validation.json), [environment](../results/runs/a100-sdcpp-flux-klein-base-001__baseline__20261005-080118/environment.json). Job11882493 on gilbreth-g007 ran12:01:18–12:11:49UTC. One first generation,three discarded warm-ups,ten measured generations; same pinned workload and no cache/offload/tiling.

| Metric | Median, ms | Measured min–max, ms |
|---|---:|---:|
| wall_ms | 43686.406 | 43637.599–43893.339 |
| text_encode_ms | 119.130 | 116.471–143.104 |
| denoise_ms | 42923.012 | 42880.496–43130.816 |
| vae_decode_ms | 612.247 | 609.133–618.391 |
| other_ms | 28.112 | 26.248–29.689 |

First generation43610.616ms; eager context loading9.087711s (filesystem cache may be warm). Device-wide sampled generation peak22.880310GiB; harness host RSS peak1.226830GiB. Allocator counters are unavailable. All14recorded image hashes agree; both retained1024×1024PNGs independently rehashed. Every image has51progress callbacks and100transformer passes across50scheduler steps. Settings, stage accounting, phase order and all reported measured medians/min/max passed independent checks. This single development prompt/seed is not formal quality evidence.

[W&B baseline](https://wandb.ai/mfaruqi-purdue-university/on-device-diffusion/runs/a100-sdcpp-flux-klein-base-001__baseline__20261005-080118).

## Failed separate profiler attempt

[Run](../results/runs/a100-sdcpp-flux-klein-base-001__profile__20261005-081219/), [status and traceback](../results/runs/a100-sdcpp-flux-klein-base-001__profile__20261005-081219/status.json). Job11882497 on gilbreth-g007 failed12:12:24UTC, before model loading/generation produced any measurements. Nsight Systems2024.4.2 reported `Failed to probe the process (sync). Timeout: 2 sec`. No latency, memory or image result is inferred from this failed attempt. Its config, command and environment remain intact.

Hypothesis: the failure concerns profiler startup/environment; the available message does not establish its cause. The harness shared-library check resolved all dependencies. No model, precision or resolution fallback was attempted.

[W&B failed profile](https://wandb.ai/mfaruqi-purdue-university/on-device-diffusion/runs/a100-sdcpp-flux-klein-base-001__profile__20261005-081219).

## Cache-capable harness no-cache control

[Control](../results/runs/a100-sdcpp-flux-klein-base-001__attempt__20261005-082117/), [validation](../results/runs/a100-sdcpp-flux-klein-base-001__attempt__20261005-082117/validation.json), [environment](../results/runs/a100-sdcpp-flux-klein-base-001__attempt__20261005-082117/environment.json). Job11883351 on gilbreth-g007 completed12:22:59UTC. Same pinned Base workload, no cache and unchanged engine settings, using the separately built cache-capable harness. Binary SHA256 `d5efff548b7e7b3bd7739c6a94d8b6df99a97aa9f961b6cff565bd3430a8f6cc`. Protocol:one first plus one additional observation; this is a correctness control, not a repeated warm baseline.

Both saved images match the existing no-cache reference pixels. All callback/CFG/stage/settings/summary checks passed. The second observation took43648.865ms with sampled device-wide peak22.880310GiB. A full matched-binary reference remains necessary before attributing a cache timing difference solely to reuse.

[W&B control](https://wandb.ai/mfaruqi-purdue-university/on-device-diffusion/runs/a100-sdcpp-flux-klein-base-001__attempt__20261005-082117).

## Interpretation

The specified Base configuration fits and passes the correctness gate. Denoising dominates the observed duration. The full protocol establishes the repeated warm timing distribution for this configuration; kernel profiling remains incomplete because the profiler failed before generation.

## Next experiment

- Investigate Nsight startup separately before any further profiler attempt.

[W&B attempt](https://wandb.ai/mfaruqi-purdue-university/on-device-diffusion/runs/a100-sdcpp-flux-klein-base-001__attempt__20261005-040120).

## Nsight retry failed before inference, 2026-10-05

[Failed run](../results/runs/a100-sdcpp-flux-klein-base-001__profile__20261005-191522/), [status](../results/runs/a100-sdcpp-flux-klein-base-001__profile__20261005-191522/status.json), [environment](../results/runs/a100-sdcpp-flux-klein-base-001__profile__20261005-191522/environment.json). Slurm11883400 on gilbreth-g011 failed with the same `Failed to probe the process (sync). Timeout: 2 sec` error despite excluding the earlier g006/g007 nodes. No load, generation, stage timing or quality result was produced. The bounded retry is exhausted; further node retries are not scheduled. The message identifies a profiler startup failure, not its underlying cause.

## Full protocol a100-sdcpp-flux-klein-base-001__baseline__20261005-214624

[Summary](../results/runs/a100-sdcpp-flux-klein-base-001__baseline__20261005-214624/summary.json), [config](../results/runs/a100-sdcpp-flux-klein-base-001__baseline__20261005-214624/config.json), [independent validation](../results/runs/a100-sdcpp-flux-klein-base-001__baseline__20261005-214624/validation.json), [environment](../results/runs/a100-sdcpp-flux-klein-base-001__baseline__20261005-214624/environment.json). Slurm 11889993, gilbreth-g007.rcac.purdue.edu, 2026-10-06T01:46:24Z to 2026-10-06T01:56:53Z; repo snapshot `d04afb02a492ad8a271afaf2da8a16f9cf2bb11c`. One context; one first generation, three discarded warm-ups and ten measured generations. Same pinned BF16 workload as the functional gate; all requested settings retained. [Metric definitions](../wiki/methods/baseline-metrics.md).

| Metric | Median, ms | Measured min–max, ms |
|---|---:|---:|
| wall_ms | 43976.167000 | 43934.973000–44062.672000 |
| text_encode_ms | 121.095500 | 118.190000–128.693000 |
| denoise_ms | 43209.511000 | 43176.221000–43298.267000 |
| vae_decode_ms | 611.149500 | 608.665000–626.937000 |
| other_ms | 29.260500 | 28.656000–29.511000 |

First generation 43475.825000ms; context load 10.585840s (filesystem cache may be warm). Sampled generation device-wide peak 22.880310GiB; child host peak RSS 1.225285GiB. No allocator or per-stage RSS measurement.

All14 generations passed callback/actual transformer-call accounting, phase, engine-setting and CSV/summary checks. Actual transformer passes per generation: [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]. Conditioning hits: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]. Approximate steps skipped: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]. Mapped files: []. All14 runner-reported pixel hashes agree; both retained PNGs independently rehashed. Retained RGB equals the same-binary no-cache reference: True.

Interpretation: this establishes a repeated timing distribution for the declared configuration. Selection and deltas belong in the matched comparison; sequential capture conditions were not randomized.

[W&B](https://wandb.ai/mfaruqi-purdue-university/on-device-diffusion/runs/a100-sdcpp-flux-klein-base-001__baseline__20261005-214624).
