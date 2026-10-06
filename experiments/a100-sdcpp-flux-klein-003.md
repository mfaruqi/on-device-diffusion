---
type: experiment-record
id: a100-sdcpp-flux-klein-003
status: complete
device: a100-pcie-40gb
engine: stable-diffusion-cpp
runs: [a100-sdcpp-flux-klein-003__attempt__20261005-193430, a100-sdcpp-flux-klein-003__baseline__20261005-220737]
updated: 2026-10-06
---

# a100-sdcpp-flux-klein-003: Four-step exact conditioning cache

Status: **complete**; full repeated protocol validated. Earlier functional-gate evidence is retained below.

## Question

Does this individual option execute as configured for the matched A100 workload? Week 2 image-engine baseline work, RQ1. The repeated baseline will measure its effect; a two-generation gate is not a performance conclusion.


## Four-step exact conditioning cache: two-generation validation, 2026-10-05

[Run and raw outputs](../results/runs/a100-sdcpp-flux-klein-003__attempt__20261005-193430/), [saved config](../results/runs/a100-sdcpp-flux-klein-003__attempt__20261005-193430/config.json), [environment](../results/runs/a100-sdcpp-flux-klein-003__attempt__20261005-193430/environment.json), [summary](../results/runs/a100-sdcpp-flux-klein-003__attempt__20261005-193430/summary.json), [independent validation](../results/runs/a100-sdcpp-flux-klein-003__attempt__20261005-193430/validation.json). Slurm 11883785 on gilbreth-g007.rcac.purdue.edu; repository snapshot a4b4028e0c13f9ac62b751886bccf7ce0013a617. Measurement definitions: [baseline metrics](../wiki/methods/baseline-metrics.md).

BF16 weights,1024×1024,4 scheduler steps,CFG 1.0,seed0,one development cat/sign prompt. One context,first image and one measured observation,zero discarded warm-ups. Engine and checkpoint revisions, all file hashes, fixed CUDA residency and all option values are preserved in the config. Cache-capable binary SHA256 `d5efff548b7e7b3bd7739c6a94d8b6df99a97aa9f961b6cff565bd3430a8f6cc`. Full benchmark queued as Slurm 11890008; these two observations are a functional gate.

| Metric | Second image, ms |
|---|---:|
| wall_ms | 2412.678000 |
| text_encode_ms | 13.627000 |
| denoise_ms | 1716.886000 |
| vae_decode_ms | 653.821000 |
| other_ms | 28.344000 |

First image 2529.736000ms. Context load 3.262355s (filesystem cache may be warm). Sampled generation device-wide peak 22.880310GiB; harness host peakRSS 1.208721GiB. GPU allocator counters unavailable.

Actual transformer passes per image: [4, 4]; conditioning hits: [0, 1]. Approximate cache skips: [0, 0]. Confirmed mmap filenames: []. Settings, callback order, actual CFG execution counts, phases, CSV arithmetic and summary statistics passed independent checks. Both retained image files were rehashed; saved pixels equal same-binary reference: **True**.

[Paired image diagnostic](../results/runs/a100-sdcpp-flux-klein-003__attempt__20261005-193430/quality-diagnostics.json): PSNR infinity (identical pixels)dB, LPIPS AlexNet v0.1 0.000000000; reference `a100-sdcpp-flux-klein-001__attempt__20261005-193229`. Full1024² RGB without resizing, one development prompt/seed. Formal quality eligibility remains false.

Interpretation: this verifies execution and saved-output behavior for this option. A single later observation cannot establish a latency improvement.

Conditioning reuse concerns repeated identical prompt inputs; it is not general new-prompt latency.

[W&B run](https://wandb.ai/mfaruqi-purdue-university/on-device-diffusion/runs/a100-sdcpp-flux-klein-003__attempt__20261005-193430).

## Full protocol a100-sdcpp-flux-klein-003__baseline__20261005-220737

[Summary](../results/runs/a100-sdcpp-flux-klein-003__baseline__20261005-220737/summary.json), [config](../results/runs/a100-sdcpp-flux-klein-003__baseline__20261005-220737/config.json), [independent validation](../results/runs/a100-sdcpp-flux-klein-003__baseline__20261005-220737/validation.json), [environment](../results/runs/a100-sdcpp-flux-klein-003__baseline__20261005-220737/environment.json). Slurm 11890008, gilbreth-g007.rcac.purdue.edu, 2026-10-06T02:07:37Z to 2026-10-06T02:08:16Z; repo snapshot `a4b4028e0c13f9ac62b751886bccf7ce0013a617`. One context; one first generation, three discarded warm-ups and ten measured generations. Same pinned BF16 workload as the functional gate; all requested settings retained. [Metric definitions](../wiki/methods/baseline-metrics.md).

| Metric | Median, ms | Measured min–max, ms |
|---|---:|---:|
| wall_ms | 2389.444500 | 2371.651000–2406.890000 |
| text_encode_ms | 13.176000 | 12.916000–13.719000 |
| denoise_ms | 1732.962500 | 1723.710000–1743.298000 |
| vae_decode_ms | 614.954500 | 605.860000–623.307000 |
| other_ms | 28.603000 | 28.301000–28.774000 |

First generation 2542.853000ms; context load 3.239646s (filesystem cache may be warm). Sampled generation device-wide peak 22.880310GiB; child host peak RSS 1.228821GiB. No allocator or per-stage RSS measurement.

All14 generations passed callback/actual transformer-call accounting, phase, engine-setting and CSV/summary checks. Actual transformer passes per generation: [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]. Conditioning hits: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]. Approximate steps skipped: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]. Mapped files: []. All14 runner-reported pixel hashes agree; both retained PNGs independently rehashed. Retained RGB equals the same-binary no-cache reference: True.

[Paired image diagnostic](../results/runs/a100-sdcpp-flux-klein-003__baseline__20261005-220737/quality-diagnostics.json): PSNR infinity (identical pixels)dB; LPIPS AlexNet v0.1 0.000000000. One development prompt/seed, full1024² RGB without resizing; no held-out evaluation or formal quality acceptance.

Interpretation: this establishes a repeated timing distribution for the declared configuration. Selection and deltas belong in the matched comparison; sequential capture conditions were not randomized.

Exact conditioning hits concern repeated identical prompt inputs, not new-prompt latency.

[W&B](https://wandb.ai/mfaruqi-purdue-university/on-device-diffusion/runs/a100-sdcpp-flux-klein-003__baseline__20261005-220737).
