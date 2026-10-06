---
type: experiment-record
id: a100-sdcpp-flux-klein-base-003
status: partial
device: a100-pcie-40gb
engine: stable-diffusion-cpp
runs: [a100-sdcpp-flux-klein-base-003__attempt__20261005-234246]
updated: 2026-10-06
---

# A100 sd.cpp Base: exact conditioning reuse

Status: **partial**; two-generation correctness gate completed and independently validated (Slurm11891252, g007). Full protocol11893094 queued after validation.

## Question

Can exact positive/negative prompt-conditioning reuse execute correctly for the Base workload? Week2 image-engine baselines, RQ1. Reference: same-binary Base no-cache run214624; performance comparison requires the repeated protocol.

## Setup

[summary.json](../results/runs/a100-sdcpp-flux-klein-base-003__attempt__20261005-234246/summary.json), [config.json](../results/runs/a100-sdcpp-flux-klein-base-003__attempt__20261005-234246/config.json), [environment.json](../results/runs/a100-sdcpp-flux-klein-base-003__attempt__20261005-234246/environment.json), [validation.json](../results/runs/a100-sdcpp-flux-klein-base-003__attempt__20261005-234246/validation.json). Isolated snapshot `2c5f5da3442b50cb90f121725deedb6485df30c0`. One change: exact conditioning cache capacity2. BF16 weights,1024²,50 scheduler steps,CFG4,seed0,one development cat/sign prompt; fixed CUDA residency, no approximate reuse or offload. Pinned checkpoint revisions and component hashes are in config.json. Cache-capable binary SHA256 `d5efff548b7e7b3bd7739c6a94d8b6df99a97aa9f961b6cff565bd3430a8f6cc`.

One context,one first generation and one measured observation,zero warm-ups. [Metric definitions](../wiki/methods/baseline-metrics.md).

## Results

| Metric | Second image, ms |
|---|---:|
| wall_ms | 43698.488000 |
| text_encode_ms | 14.648000 |
| denoise_ms | 43042.353000 |
| vae_decode_ms | 613.112000 |
| other_ms | 28.375000 |

First image 43587.610000ms. Context load 10.387243s (filesystem cache may be warm). Sampled generation device-wide peak 22.880310GiB; child host peak RSS 1.226822GiB. Allocator counters unavailable.

## Observations and quality

Both generations retained100 transformer passes (two CFG passes per scheduler step). Conditioning hits were0 then2. Settings, phase order, callbacks, CSV arithmetic, summary and retained image checks passed. Both retained images equal the same-binary no-cache reference pixels.

[Paired diagnostic](../results/runs/a100-sdcpp-flux-klein-base-003__attempt__20261005-234246/quality-diagnostics.json): identical pixels, PSNR infinity and LPIPS0.0 against Base001 baseline214624, full1024² without resizing. One development prompt/seed; no held-out or formal quality evaluation.

## Interpretation

This verifies exact reuse for repeated identical conditioning inputs. It does not characterize new-prompt latency. One later observation cannot establish a speed improvement.

## Next experiment

- Full1+3+10 protocol using configs/a100-sdcpp-flux-klein-base-conditioning-cache.resolved.json (job11893094).

[W&B](https://wandb.ai/mfaruqi-purdue-university/on-device-diffusion/runs/a100-sdcpp-flux-klein-base-003__attempt__20261005-234246).
