---
type: experiment-record
id: a100-sdcpp-flux-klein-005
status: complete
device: a100-pcie-40gb
engine: stable-diffusion-cpp
runs: [a100-sdcpp-flux-klein-005__attempt__20261005-193630, a100-sdcpp-flux-klein-005__baseline__20261005-221039]
updated: 2026-10-06
---

# a100-sdcpp-flux-klein-005: Four-step mmap weight-file I/O

Status: **complete**; full repeated protocol validated. Earlier functional-gate evidence is retained below.

## Question

Does this individual option execute as configured for the matched A100 workload? Week 2 image-engine baseline work, RQ1. The repeated baseline will measure its effect; a two-generation gate is not a performance conclusion.


## Four-step mmap weight-file I/O: two-generation validation, 2026-10-05

[Run and raw outputs](../results/runs/a100-sdcpp-flux-klein-005__attempt__20261005-193630/), [saved config](../results/runs/a100-sdcpp-flux-klein-005__attempt__20261005-193630/config.json), [environment](../results/runs/a100-sdcpp-flux-klein-005__attempt__20261005-193630/environment.json), [summary](../results/runs/a100-sdcpp-flux-klein-005__attempt__20261005-193630/summary.json), [independent validation](../results/runs/a100-sdcpp-flux-klein-005__attempt__20261005-193630/validation.json). Slurm 11883787 on gilbreth-g007.rcac.purdue.edu; repository snapshot a4b4028e0c13f9ac62b751886bccf7ce0013a617. Measurement definitions: [baseline metrics](../wiki/methods/baseline-metrics.md).

BF16 weights,1024×1024,4 scheduler steps,CFG 1.0,seed0,one development cat/sign prompt. One context,first image and one measured observation,zero discarded warm-ups. Engine and checkpoint revisions, all file hashes, fixed CUDA residency and all option values are preserved in the config. Cache-capable binary SHA256 `d5efff548b7e7b3bd7739c6a94d8b6df99a97aa9f961b6cff565bd3430a8f6cc`. Full benchmark queued as Slurm 11890010; these two observations are a functional gate.

| Metric | Second image, ms |
|---|---:|
| wall_ms | 2413.486000 |
| text_encode_ms | 65.666000 |
| denoise_ms | 1715.218000 |
| vae_decode_ms | 603.767000 |
| other_ms | 28.835000 |

First image 2591.886000ms. Context load 8.112325s (filesystem cache may be warm). Sampled generation device-wide peak 22.880310GiB; harness host peakRSS 15.628212GiB. GPU allocator counters unavailable.

Actual transformer passes per image: [4, 4]; conditioning hits: [0, 0]. Approximate cache skips: [0, 0]. Confirmed mmap filenames: ['diffusion_pytorch_model.safetensors', 'flux-2-klein-4b.safetensors', 'model-00001-of-00002.safetensors', 'model-00002-of-00002.safetensors']. Settings, callback order, actual CFG execution counts, phases, CSV arithmetic and summary statistics passed independent checks. Both retained image files were rehashed; saved pixels equal same-binary reference: **True**.

[Paired image diagnostic](../results/runs/a100-sdcpp-flux-klein-005__attempt__20261005-193630/quality-diagnostics.json): PSNR infinity (identical pixels)dB, LPIPS AlexNet v0.1 0.000000000; reference `a100-sdcpp-flux-klein-001__attempt__20261005-193229`. Full1024² RGB without resizing, one development prompt/seed. Formal quality eligibility remains false.

Interpretation: this verifies execution and saved-output behavior for this option. A single later observation cannot establish a latency improvement.

Mmap confirms weight-file mapping, not zero-copy GPU weights or a changed GPU residency policy.

[W&B run](https://wandb.ai/mfaruqi-purdue-university/on-device-diffusion/runs/a100-sdcpp-flux-klein-005__attempt__20261005-193630).

## Full protocol a100-sdcpp-flux-klein-005__baseline__20261005-221039

[Summary](../results/runs/a100-sdcpp-flux-klein-005__baseline__20261005-221039/summary.json), [config](../results/runs/a100-sdcpp-flux-klein-005__baseline__20261005-221039/config.json), [independent validation](../results/runs/a100-sdcpp-flux-klein-005__baseline__20261005-221039/validation.json), [environment](../results/runs/a100-sdcpp-flux-klein-005__baseline__20261005-221039/environment.json). Slurm 11890010, gilbreth-g007.rcac.purdue.edu, 2026-10-06T02:10:39Z to 2026-10-06T02:11:23Z; repo snapshot `a4b4028e0c13f9ac62b751886bccf7ce0013a617`. One context; one first generation, three discarded warm-ups and ten measured generations. Same pinned BF16 workload as the functional gate; all requested settings retained. [Metric definitions](../wiki/methods/baseline-metrics.md).

| Metric | Median, ms | Measured min–max, ms |
|---|---:|---:|
| wall_ms | 2439.542500 | 2430.872000–2459.142000 |
| text_encode_ms | 65.082000 | 64.793000–70.456000 |
| denoise_ms | 1735.284000 | 1729.218000–1748.773000 |
| vae_decode_ms | 609.320000 | 607.473000–613.719000 |
| other_ms | 29.094000 | 28.718000–29.363000 |

First generation 2550.805000ms; context load 7.650543s (filesystem cache may be warm). Sampled generation device-wide peak 22.880310GiB; child host peak RSS 15.638645GiB. No allocator or per-stage RSS measurement.

All14 generations passed callback/actual transformer-call accounting, phase, engine-setting and CSV/summary checks. Actual transformer passes per generation: [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]. Conditioning hits: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]. Approximate steps skipped: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]. Mapped files: ['diffusion_pytorch_model.safetensors', 'flux-2-klein-4b.safetensors', 'model-00001-of-00002.safetensors', 'model-00002-of-00002.safetensors']. All14 runner-reported pixel hashes agree; both retained PNGs independently rehashed. Retained RGB equals the same-binary no-cache reference: True.

[Paired image diagnostic](../results/runs/a100-sdcpp-flux-klein-005__baseline__20261005-221039/quality-diagnostics.json): PSNR infinity (identical pixels)dB; LPIPS AlexNet v0.1 0.000000000. One development prompt/seed, full1024² RGB without resizing; no held-out evaluation or formal quality acceptance.

Interpretation: this establishes a repeated timing distribution for the declared configuration. Selection and deltas belong in the matched comparison; sequential capture conditions were not randomized.

[W&B](https://wandb.ai/mfaruqi-purdue-university/on-device-diffusion/runs/a100-sdcpp-flux-klein-005__baseline__20261005-221039).
