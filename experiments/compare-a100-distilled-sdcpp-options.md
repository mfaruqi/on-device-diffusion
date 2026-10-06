---
type: experiment-record
id: compare-a100-distilled-sdcpp-options
status: complete
device: a100-pcie-40gb
engine: stable-diffusion-cpp
runs: [a100-sdcpp-flux-klein-001__baseline__20261005-220435, a100-sdcpp-flux-klein-002__baseline__20261005-220636, a100-sdcpp-flux-klein-003__baseline__20261005-220737, a100-sdcpp-flux-klein-004__baseline__20261005-220938, a100-sdcpp-flux-klein-005__baseline__20261005-221039]
updated: 2026-10-06
---

# Compare: A100 distilled sd.cpp individual options

## Question

Which tested individual option reduces latency under the campaign diagnostic rule? Week2 image-engine baselines, RQ1. This is descriptive configuration evidence, not formal quality-constrained planner evaluation.

## Runs

| Run and saved configuration | Experiment record | Intended difference |
|---|---|---|
| [a100-sdcpp-flux-klein-001__baseline__20261005-220435](../results/runs/a100-sdcpp-flux-klein-001__baseline__20261005-220435/config.json) | [a100-sdcpp-flux-klein-001](a100-sdcpp-flux-klein-001.md) | reference |
| [a100-sdcpp-flux-klein-002__baseline__20261005-220636](../results/runs/a100-sdcpp-flux-klein-002__baseline__20261005-220636/config.json) | [a100-sdcpp-flux-klein-002](a100-sdcpp-flux-klein-002.md) | step_cache |
| [a100-sdcpp-flux-klein-003__baseline__20261005-220737](../results/runs/a100-sdcpp-flux-klein-003__baseline__20261005-220737/config.json) | [a100-sdcpp-flux-klein-003](a100-sdcpp-flux-klein-003.md) | conditioning_cache_size |
| [a100-sdcpp-flux-klein-004__baseline__20261005-220938](../results/runs/a100-sdcpp-flux-klein-004__baseline__20261005-220938/config.json) | [a100-sdcpp-flux-klein-004](a100-sdcpp-flux-klein-004.md) | prefetch |
| [a100-sdcpp-flux-klein-005__baseline__20261005-221039](../results/runs/a100-sdcpp-flux-klein-005__baseline__20261005-221039/config.json) | [a100-sdcpp-flux-klein-005](a100-sdcpp-flux-klein-005.md) | mmap |

## Matching

| Setting | Held fixed |
|---|---|
| Device | Same A100-PCIE-40GB, same host gilbreth-g007.rcac.purdue.edu; GPU UUID recorded in environment |
| Engine | Same sd.cpp/ggml commits, binary `d5efff548b7e7b3bd7739c6a94d8b6df99a97aa9f961b6cff565bd3430a8f6cc` |
| Checkpoint | Identical model object, pinned revisions and all component SHA256 values |
| Precision | BF16 weights; unchanged backend |
| Workload | 1024²,4steps,CFG1.0,same prompt,seed0,batch1 |
| Protocol | One first,three discarded warm-ups,ten measured;20ms NVML samples |
| Residency | Fixed CUDA0 parameters/compute,eager load,one segment,no auto-fit/offload |
| Variable | One labelled optimization per variant; matching receipt lists its engine-setting realization |

Confounds: separate sequential jobs without randomized order or controlled filesystem-cache/thermal/background conditions. Same engine/seed fixes the noise-generation procedure but no explicit initial latent tensor was supplied. Timing ratios are descriptive; range separation is a campaign selection heuristic, not a statistical confidence interval. Memory compares the same sampled device-wide metric.

## Results

All values below come from each linked run's summary.json. All variants have independent validation.json and quality-diagnostics.json; reference images are the saved no-cache outputs.

| Configuration | Wall median [min–max], ms | Delta vs reference, ms | Reference/variant | Device peak, GiB | Speed-range rule |
|---|---:|---:|---:|---:|---|
| reference | 2429.043 [2416.243–2439.335] | 0.000 | 1.000000 | 22.880310 | reference |
| step_cache | 2435.751 [2424.584–2453.261] | 6.708 | 0.997246 | 22.880310 | fail |
| conditioning_cache_size | 2389.445 [2371.651–2406.890] | -39.598 | 1.016572 | 22.880310 | pass |
| prefetch | 2428.830 [2419.048–2440.712] | -0.213 | 1.000087 | 22.880310 | fail |
| mmap | 2439.542 [2430.872–2459.142] | 10.500 | 0.995696 | 22.880310 | fail |

| Configuration | Text median, ms | Denoise median, ms | VAE median, ms |
|---|---:|---:|---:|
| reference | 65.256 | 1725.524 | 605.722 |
| step_cache | 65.117 | 1734.358 | 607.434 |
| conditioning_cache_size | 13.176 | 1732.962 | 614.955 |
| prefetch | 65.039 | 1726.226 | 607.148 |
| mmap | 65.082 | 1735.284 | 609.320 |

## Execution, image evidence and selection

- **step_cache**: actual transformer passes [4]; skipped steps [0]; conditioning hits [0]. PSNR infinity (identical pixels)dB, LPIPS 0.000000000. Qualifies for later combinations under the campaign diagnostic rule: **False** ([receipt](../results/runs/a100-sdcpp-flux-klein-002__baseline__20261005-220636/comparison.json)).
- **conditioning_cache_size**: actual transformer passes [4]; skipped steps [0]; conditioning hits [0, 1]. PSNR infinity (identical pixels)dB, LPIPS 0.000000000. Qualifies for later combinations under the campaign diagnostic rule: **True** ([receipt](../results/runs/a100-sdcpp-flux-klein-003__baseline__20261005-220737/comparison.json)).
- **prefetch**: actual transformer passes [4]; skipped steps [0]; conditioning hits [0]. PSNR infinity (identical pixels)dB, LPIPS 0.000000000. Qualifies for later combinations under the campaign diagnostic rule: **False** ([receipt](../results/runs/a100-sdcpp-flux-klein-004__baseline__20261005-220938/comparison.json)).
- **mmap**: actual transformer passes [4]; skipped steps [0]; conditioning hits [0]. PSNR infinity (identical pixels)dB, LPIPS 0.000000000. Qualifies for later combinations under the campaign diagnostic rule: **False** ([receipt](../results/runs/a100-sdcpp-flux-klein-005__baseline__20261005-221039/comparison.json)).

EasyCache executed all four transformer steps and produced identical pixels; no step reuse benefit is observed. Conditioning reuse is the sole qualifying variant here, with identical pixels; its small end-to-end gain applies to repeated identical prompt inputs. Prefetch-disabled and mmap do not pass range separation. One qualifying option does not form a combination.

## Limits

One development prompt/seed; no alignment benchmark, blinded quality study or held-out prompts. Ten measurements within one loaded context do not characterize every prompt, hardware pressure state or job-to-job variance. Sampled memory can miss brief peaks and is device-wide. No cross-device or cross-engine ranking is established by this record.
