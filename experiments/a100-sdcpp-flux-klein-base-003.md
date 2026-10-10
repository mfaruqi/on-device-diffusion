---
type: experiment-record
id: a100-sdcpp-flux-klein-base-003
status: complete
device: a100-pcie-40gb
engine: stable-diffusion-cpp
runs: [a100-sdcpp-flux-klein-base-003__attempt__20261005-234246, a100-sdcpp-flux-klein-base-003__baseline__20261006-095509]
updated: 2026-10-07
---

# A100 sd.cpp Base: exact conditioning reuse

Status: **complete**; full protocol Slurm11893094 completed on g000 on 2026-10-06. Earlier two-generation gate Slurm11891252 on g007 is retained below.

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

- A paired same-physical-GPU reference and conditioning repeat is required before attributing the small timing difference to the cache.

[W&B](https://wandb.ai/mfaruqi-purdue-university/on-device-diffusion/runs/a100-sdcpp-flux-klein-base-003__attempt__20261005-234246).

## Full repeated baseline, 2026-10-06

[Summary](../results/runs/a100-sdcpp-flux-klein-base-003__baseline__20261006-095509/summary.json), [config](../results/runs/a100-sdcpp-flux-klein-base-003__baseline__20261006-095509/config.json), [environment](../results/runs/a100-sdcpp-flux-klein-base-003__baseline__20261006-095509/environment.json), [status](../results/runs/a100-sdcpp-flux-klein-base-003__baseline__20261006-095509/status.json), [validation receipt](../results/runs/a100-sdcpp-flux-klein-base-003__baseline__20261006-095509/validation.json). One first generation, three discarded warm-ups and ten measured generations in one context; unchanged BF16, 1024², 50-step, CFG4 repeated-prompt workload and cache capacity2. Node `gilbreth-g000.rcac.purdue.edu`, snapshot `2c5f5da3442b50cb90f121725deedb6485df30c0`.

| Metric | Median, ms | Measured min–max, ms |
|---|---:|---:|
| wall_ms | 43767.162000 | 43746.861000–43788.440000 |
| text_encode_ms | 15.111000 | 14.209000–28.589000 |
| denoise_ms | 43110.188000 | 43091.218000–43125.065000 |
| vae_decode_ms | 613.090500 | 610.527000–619.844000 |
| other_ms | 28.182000 | 27.861000–28.760000 |

First generation 43508.466000ms; context load 11.062893s (filesystem cache may be warm). Sampled generation device peak 22.880310GiB; host peak RSS 1.206364GiB. Allocator metrics unavailable.

The saved validation receipt confirms 100 transformer passes in every generation and conditioning hits 0 then 2 for each later generation. [Paired image diagnostics](../results/runs/a100-sdcpp-flux-klein-base-003__baseline__20261006-095509/quality-diagnostics.json) report identical saved reference pixels, MSE0 and LPIPS0; this is one development prompt/seed, not formal quality acceptance.

[Matching receipt](../results/runs/a100-sdcpp-flux-klein-base-003__baseline__20261006-095509/comparison.json): same workload, model, precision, protocol and binary, but a different physical A100 GPU UUID from the reference. The observed median difference is approximately −0.475%; attribution and combination selection remain deferred pending a paired same-GPU repeat.

[W&B source artifact receipt](../results/runs/a100-sdcpp-flux-klein-base-003__baseline__20261006-095509/wandb-import.json): small measurement files were recovered from the existing uploaded benchmark artifact on 2026-10-07. CSV/summary statistics and protocol counts were rechecked locally; prior raw-log and image validation is retained as the uploaded receipt, not repeated image validation on this host.
