---
type: experiment-record
id: a100-sdcpp-flux-klein-base-002
status: complete
device: a100-pcie-40gb
engine: stable-diffusion-cpp
runs: [a100-sdcpp-flux-klein-base-002__attempt__20261005-191624, a100-sdcpp-flux-klein-base-002__baseline__20261005-215731]
updated: 2026-10-06
---

# a100-sdcpp-flux-klein-base-002: Base EasyCache BF16 50-step gate

Status: **complete**; full repeated protocol validated. Earlier functional-gate evidence is retained below.

## Question

Does this individual option execute as configured for the matched A100 workload? Week 2 image-engine baseline work, RQ1. The repeated baseline will measure its effect; a two-generation gate is not a performance conclusion.


## Base EasyCache BF16 50-step gate: two-generation validation, 2026-10-05

[Run and raw outputs](../results/runs/a100-sdcpp-flux-klein-base-002__attempt__20261005-191624/), [saved config](../results/runs/a100-sdcpp-flux-klein-base-002__attempt__20261005-191624/config.json), [environment](../results/runs/a100-sdcpp-flux-klein-base-002__attempt__20261005-191624/environment.json), [summary](../results/runs/a100-sdcpp-flux-klein-base-002__attempt__20261005-191624/summary.json), [independent validation](../results/runs/a100-sdcpp-flux-klein-base-002__attempt__20261005-191624/validation.json). Slurm 11883425 on gilbreth-g011.rcac.purdue.edu; repository snapshot d04afb02a492ad8a271afaf2da8a16f9cf2bb11c. Measurement definitions: [baseline metrics](../wiki/methods/baseline-metrics.md).

BF16 weights,1024×1024,50 scheduler steps,CFG 4.0,seed0,one development cat/sign prompt. One context,first image and one measured observation,zero discarded warm-ups. Engine and checkpoint revisions, all file hashes, fixed CUDA residency and all option values are preserved in the config. Cache-capable binary SHA256 `d5efff548b7e7b3bd7739c6a94d8b6df99a97aa9f961b6cff565bd3430a8f6cc`. Full benchmark queued as Slurm 11889994; these two observations are a functional gate.

| Metric | Second image, ms |
|---|---:|
| wall_ms | 23942.406000 |
| text_encode_ms | 117.686000 |
| denoise_ms | 23187.844000 |
| vae_decode_ms | 607.672000 |
| other_ms | 29.204000 |

First image 23974.394000ms. Context load 10.800542s (filesystem cache may be warm). Sampled generation device-wide peak 22.880310GiB; harness host peakRSS 1.205399GiB. GPU allocator counters unavailable.

Actual transformer passes per image: [54, 54]; conditioning hits: [0, 0]. Approximate cache skips: [23, 23]. Confirmed mmap filenames: []. Settings, callback order, actual CFG execution counts, phases, CSV arithmetic and summary statistics passed independent checks. Both retained image files were rehashed; saved pixels equal same-binary reference: **False**.

[Paired image diagnostic](../results/runs/a100-sdcpp-flux-klein-base-002__attempt__20261005-191624/quality-diagnostics.json): PSNR 26.54929742448861dB, LPIPS AlexNet v0.1 0.016157273; reference `a100-sdcpp-flux-klein-base-001__attempt__20261005-082117`. Full1024² RGB without resizing, one development prompt/seed. Formal quality eligibility remains false.

Informal visual check: coherent cat/sign image, with text-shape differences from no-cache. Neither this prompt nor the perceptual diagnostic establishes text accuracy or quality acceptance.

Interpretation: this verifies execution and saved-output behavior for this option. A single later observation cannot establish a latency improvement.

[W&B run](https://wandb.ai/mfaruqi-purdue-university/on-device-diffusion/runs/a100-sdcpp-flux-klein-base-002__attempt__20261005-191624).

## Full protocol a100-sdcpp-flux-klein-base-002__baseline__20261005-215731

[Summary](../results/runs/a100-sdcpp-flux-klein-base-002__baseline__20261005-215731/summary.json), [config](../results/runs/a100-sdcpp-flux-klein-base-002__baseline__20261005-215731/config.json), [independent validation](../results/runs/a100-sdcpp-flux-klein-base-002__baseline__20261005-215731/validation.json), [environment](../results/runs/a100-sdcpp-flux-klein-base-002__baseline__20261005-215731/environment.json). Slurm 11889994, gilbreth-g007.rcac.purdue.edu, 2026-10-06T01:57:31Z to 2026-10-06T02:03:21Z; repo snapshot `d04afb02a492ad8a271afaf2da8a16f9cf2bb11c`. One context; one first generation, three discarded warm-ups and ten measured generations. Same pinned BF16 workload as the functional gate; all requested settings retained. [Metric definitions](../wiki/methods/baseline-metrics.md).

| Metric | Median, ms | Measured min–max, ms |
|---|---:|---:|
| wall_ms | 24192.530000 | 24158.457000–24244.361000 |
| text_encode_ms | 118.979000 | 118.151000–132.241000 |
| denoise_ms | 23432.649500 | 23398.613000–23471.632000 |
| vae_decode_ms | 611.299500 | 608.594000–626.689000 |
| other_ms | 29.251000 | 28.964000–29.732000 |

First generation 24068.096000ms; context load 9.311283s (filesystem cache may be warm). Sampled generation device-wide peak 22.880310GiB; child host peak RSS 1.205910GiB. No allocator or per-stage RSS measurement.

All14 generations passed callback/actual transformer-call accounting, phase, engine-setting and CSV/summary checks. Actual transformer passes per generation: [54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54]. Conditioning hits: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]. Approximate steps skipped: [23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23]. Mapped files: []. All14 runner-reported pixel hashes agree; both retained PNGs independently rehashed. Retained RGB equals the same-binary no-cache reference: False.

[Paired image diagnostic](../results/runs/a100-sdcpp-flux-klein-base-002__baseline__20261005-215731/quality-diagnostics.json): PSNR 26.54929742448861dB; LPIPS AlexNet v0.1 0.016157273. One development prompt/seed, full1024² RGB without resizing; no held-out evaluation or formal quality acceptance.

Interpretation: this establishes a repeated timing distribution for the declared configuration. Selection and deltas belong in the matched comparison; sequential capture conditions were not randomized.

[W&B](https://wandb.ai/mfaruqi-purdue-university/on-device-diffusion/runs/a100-sdcpp-flux-klein-base-002__baseline__20261005-215731).
