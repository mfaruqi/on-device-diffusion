---
type: experiment-record
id: a100-sdcpp-flux-klein-004
status: complete
device: a100-pcie-40gb
engine: stable-diffusion-cpp
runs: [a100-sdcpp-flux-klein-004__attempt__20261005-193530, a100-sdcpp-flux-klein-004__baseline__20261005-220938]
updated: 2026-10-06
---

# a100-sdcpp-flux-klein-004: Four-step prefetch disabled

Status: **complete**; full repeated protocol validated. Earlier functional-gate evidence is retained below.

## Question

Does this individual option execute as configured for the matched A100 workload? Week 2 image-engine baseline work, RQ1. The repeated baseline will measure its effect; a two-generation gate is not a performance conclusion.


## Four-step prefetch disabled: two-generation validation, 2026-10-05

[Run and raw outputs](../results/runs/a100-sdcpp-flux-klein-004__attempt__20261005-193530/), [saved config](../results/runs/a100-sdcpp-flux-klein-004__attempt__20261005-193530/config.json), [environment](../results/runs/a100-sdcpp-flux-klein-004__attempt__20261005-193530/environment.json), [summary](../results/runs/a100-sdcpp-flux-klein-004__attempt__20261005-193530/summary.json), [independent validation](../results/runs/a100-sdcpp-flux-klein-004__attempt__20261005-193530/validation.json). Slurm 11883786 on gilbreth-g007.rcac.purdue.edu; repository snapshot a4b4028e0c13f9ac62b751886bccf7ce0013a617. Measurement definitions: [baseline metrics](../wiki/methods/baseline-metrics.md).

BF16 weights,1024×1024,4 scheduler steps,CFG 1.0,seed0,one development cat/sign prompt. One context,first image and one measured observation,zero discarded warm-ups. Engine and checkpoint revisions, all file hashes, fixed CUDA residency and all option values are preserved in the config. Cache-capable binary SHA256 `d5efff548b7e7b3bd7739c6a94d8b6df99a97aa9f961b6cff565bd3430a8f6cc`. Full benchmark queued as Slurm 11890009; these two observations are a functional gate.

| Metric | Second image, ms |
|---|---:|
| wall_ms | 2422.701000 |
| text_encode_ms | 66.515000 |
| denoise_ms | 1719.715000 |
| vae_decode_ms | 602.272000 |
| other_ms | 34.199000 |

First image 2540.911000ms. Context load 3.093868s (filesystem cache may be warm). Sampled generation device-wide peak 22.880310GiB; harness host peakRSS 1.207207GiB. GPU allocator counters unavailable.

Actual transformer passes per image: [4, 4]; conditioning hits: [0, 0]. Approximate cache skips: [0, 0]. Confirmed mmap filenames: []. Settings, callback order, actual CFG execution counts, phases, CSV arithmetic and summary statistics passed independent checks. Both retained image files were rehashed; saved pixels equal same-binary reference: **True**.

[Paired image diagnostic](../results/runs/a100-sdcpp-flux-klein-004__attempt__20261005-193530/quality-diagnostics.json): PSNR infinity (identical pixels)dB, LPIPS AlexNet v0.1 0.000000000; reference `a100-sdcpp-flux-klein-001__attempt__20261005-193229`. Full1024² RGB without resizing, one development prompt/seed. Formal quality eligibility remains false.

Interpretation: this verifies execution and saved-output behavior for this option. A single later observation cannot establish a latency improvement.

[W&B run](https://wandb.ai/mfaruqi-purdue-university/on-device-diffusion/runs/a100-sdcpp-flux-klein-004__attempt__20261005-193530).

## Full protocol a100-sdcpp-flux-klein-004__baseline__20261005-220938

[Summary](../results/runs/a100-sdcpp-flux-klein-004__baseline__20261005-220938/summary.json), [config](../results/runs/a100-sdcpp-flux-klein-004__baseline__20261005-220938/config.json), [independent validation](../results/runs/a100-sdcpp-flux-klein-004__baseline__20261005-220938/validation.json), [environment](../results/runs/a100-sdcpp-flux-klein-004__baseline__20261005-220938/environment.json). Slurm 11890009, gilbreth-g007.rcac.purdue.edu, 2026-10-06T02:09:38Z to 2026-10-06T02:10:18Z; repo snapshot `a4b4028e0c13f9ac62b751886bccf7ce0013a617`. One context; one first generation, three discarded warm-ups and ten measured generations. Same pinned BF16 workload as the functional gate; all requested settings retained. [Metric definitions](../wiki/methods/baseline-metrics.md).

| Metric | Median, ms | Measured min–max, ms |
|---|---:|---:|
| wall_ms | 2428.830000 | 2419.048000–2440.712000 |
| text_encode_ms | 65.038500 | 64.128000–67.318000 |
| denoise_ms | 1726.226000 | 1718.472000–1731.622000 |
| vae_decode_ms | 607.147500 | 604.956000–615.415000 |
| other_ms | 29.097500 | 28.612000–29.460000 |

First generation 2552.671000ms; context load 3.266493s (filesystem cache may be warm). Sampled generation device-wide peak 22.880310GiB; child host peak RSS 1.224949GiB. No allocator or per-stage RSS measurement.

All14 generations passed callback/actual transformer-call accounting, phase, engine-setting and CSV/summary checks. Actual transformer passes per generation: [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]. Conditioning hits: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]. Approximate steps skipped: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]. Mapped files: []. All14 runner-reported pixel hashes agree; both retained PNGs independently rehashed. Retained RGB equals the same-binary no-cache reference: True.

[Paired image diagnostic](../results/runs/a100-sdcpp-flux-klein-004__baseline__20261005-220938/quality-diagnostics.json): PSNR infinity (identical pixels)dB; LPIPS AlexNet v0.1 0.000000000. One development prompt/seed, full1024² RGB without resizing; no held-out evaluation or formal quality acceptance.

Interpretation: this establishes a repeated timing distribution for the declared configuration. Selection and deltas belong in the matched comparison; sequential capture conditions were not randomized.

[W&B](https://wandb.ai/mfaruqi-purdue-university/on-device-diffusion/runs/a100-sdcpp-flux-klein-004__baseline__20261005-220938).
