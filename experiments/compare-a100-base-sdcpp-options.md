---
type: experiment-record
id: compare-a100-base-sdcpp-options
status: complete
device: a100-pcie-40gb
engine: stable-diffusion-cpp
runs: [a100-sdcpp-flux-klein-base-001__baseline__20261005-214624, a100-sdcpp-flux-klein-base-002__baseline__20261005-215731]
updated: 2026-10-06
---

# Compare: A100 base sd.cpp individual options

## Question

Which tested individual option reduces latency under the campaign diagnostic rule? Week2 image-engine baselines, RQ1. This is descriptive configuration evidence, not formal quality-constrained planner evaluation.

## Runs

| Run and saved configuration | Experiment record | Intended difference |
|---|---|---|
| [a100-sdcpp-flux-klein-base-001__baseline__20261005-214624](../results/runs/a100-sdcpp-flux-klein-base-001__baseline__20261005-214624/config.json) | [a100-sdcpp-flux-klein-base-001](a100-sdcpp-flux-klein-base-001.md) | reference |
| [a100-sdcpp-flux-klein-base-002__baseline__20261005-215731](../results/runs/a100-sdcpp-flux-klein-base-002__baseline__20261005-215731/config.json) | [a100-sdcpp-flux-klein-base-002](a100-sdcpp-flux-klein-base-002.md) | step_cache |

## Matching

| Setting | Held fixed |
|---|---|
| Device | Same A100-PCIE-40GB, same host gilbreth-g007.rcac.purdue.edu; GPU UUID recorded in environment |
| Engine | Same sd.cpp/ggml commits, binary `d5efff548b7e7b3bd7739c6a94d8b6df99a97aa9f961b6cff565bd3430a8f6cc` |
| Checkpoint | Identical model object, pinned revisions and all component SHA256 values |
| Precision | BF16 weights; unchanged backend |
| Workload | 1024²,50steps,CFG4.0,same prompt,seed0,batch1 |
| Protocol | One first,three discarded warm-ups,ten measured;20ms NVML samples |
| Residency | Fixed CUDA0 parameters/compute,eager load,one segment,no auto-fit/offload |
| Variable | One labelled optimization per variant; matching receipt lists its engine-setting realization |

Confounds: separate sequential jobs without randomized order or controlled filesystem-cache/thermal/background conditions. Same engine/seed fixes the noise-generation procedure but no explicit initial latent tensor was supplied. Timing ratios are descriptive; range separation is a campaign selection heuristic, not a statistical confidence interval. Memory compares the same sampled device-wide metric.

## Results

All values below come from each linked run's summary.json. All variants have independent validation.json and quality-diagnostics.json; reference images are the saved no-cache outputs.

| Configuration | Wall median [min–max], ms | Delta vs reference, ms | Reference/variant | Device peak, GiB | Speed-range rule |
|---|---:|---:|---:|---:|---|
| reference | 43976.167 [43934.973–44062.672] | 0.000 | 1.000000 | 22.880310 | reference |
| step_cache | 24192.530 [24158.457–24244.361] | -19783.637 | 1.817758 | 22.880310 | pass |

| Configuration | Text median, ms | Denoise median, ms | VAE median, ms |
|---|---:|---:|---:|
| reference | 121.096 | 43209.511 | 611.149 |
| step_cache | 118.979 | 23432.650 | 611.300 |

## Execution, image evidence and selection

- **step_cache**: actual transformer passes [54]; skipped steps [23]; conditioning hits [0]. PSNR 26.54929742448861dB, LPIPS 0.016157273. Qualifies for later combinations under the campaign diagnostic rule: **True** ([receipt](../results/runs/a100-sdcpp-flux-klein-base-002__baseline__20261005-215731/comparison.json)).

The observed reduction is concentrated in denoising and accompanies23/50 skipped scheduler steps, reducing actual CFGtransformer passes from100to54. The approximate output differs; the PSNR/LPIPS pair is recorded, not used as formal quality approval. Only EasyCache was tested as a full single-option variant in this Base comparison; a combination needs another independently qualifying option.

## Limits

One development prompt/seed; no alignment benchmark, blinded quality study or held-out prompts. Ten measurements within one loaded context do not characterize every prompt, hardware pressure state or job-to-job variance. Sampled memory can miss brief peaks and is device-wide. No cross-device or cross-engine ranking is established by this record.
