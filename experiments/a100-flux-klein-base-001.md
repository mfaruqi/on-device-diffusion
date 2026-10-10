---
type: experiment-record
id: a100-flux-klein-base-001
status: complete
device: a100-pcie-40gb
engine: pytorch-diffusers
runs: [a100-flux-klein-base-001__attempt__20261005-033729, a100-flux-klein-base-001__baseline__20261005-074425, a100-flux-klein-base-001__profile__20261005-075123]
updated: 2026-10-05
---

# A100 PyTorch klein Base: fifty-step BF16 reference

Status: **complete**; correctness attempt11880735, full baseline11882219 and separate profile11882223 completed and independently validated.

## Question

Can the pinned FLUX.2 klein Base4B workload execute at BF16,1024²,50steps and guidance4 with correct per-step CFG accounting? Week1–2 image baselines and RQ1; a separate checkpoint/workload reference, not an execution-only change to distilled klein.

## Setup

[Saved resolved config](../results/runs/a100-flux-klein-base-001__attempt__20261005-033729/config.json). Checkpoint black-forest-labs/FLUX.2-klein-base-4B, revision a3b4f4849157f664bdbc776fd7453c2783562f4d. Batch1,1024×1024,50scheduler steps, guidance4, empty negative prompt, maximum sequence length512, seed0 and fixed cat/sign prompt. Requested BF16 model dtype; no compilation, quantization, offloading or VAE tiling/slicing. GPU execution offline. Protocol:one first generation plus one additional measured observation, no warm-ups; not a warm baseline estimate.

Node gilbreth-g007.rcac.purdue.edu, NVIDIA A100-PCIE-40GB, torch2.5.1+cu121, diffusers0.40.0, transformers5.16.1. [Environment](../results/runs/a100-flux-klein-base-001__attempt__20261005-033729/environment.json); [metric definitions](../wiki/methods/baseline-metrics.md). CUDA events measure stages; synchronized host wall time measures generation.

## Results

[Run](../results/runs/a100-flux-klein-base-001__attempt__20261005-033729/), [summary](../results/runs/a100-flux-klein-base-001__attempt__20261005-033729/summary.json), [status](../results/runs/a100-flux-klein-base-001__attempt__20261005-033729/status.json).

| Metric | First, ms | Second observation, ms |
|---|---:|---:|
| wall_ms | 25325.712 | 24875.904 |
| gpu_span_ms | 25325.418 | 24875.525 |
| text_encode_ms | 538.305 | 109.095 |
| denoise_ms | 24263.609 | 24550.275 |
| vae_decode_ms | 443.899 | 168.008 |
| postprocess_ms | 44.697 | 30.894 |
| other_ms | 34.908 | 17.253 |

Model loading:24.868s. Second-observation PyTorch peak allocated:17.332GiB; peak reserved:19.602GiB. These are separate allocator metrics and are never added. Loading left14.927GiB allocated.

## Validation and observations

Each generation records two text-encoding calls, two transformer calls for every denoise_step_0 through denoise_step_49, one VAE decode and one postprocess. The runner asserts50completed scheduler steps. Independent checks reproduced stage sums, stage-plus-other GPU span, summary aggregates, phase order, image dimensions and saved pixel hashes. [Validation receipt](../results/runs/a100-flux-klein-base-001__attempt__20261005-033729/validation.json).

Both images are1024×1024 and pixel-identical. Visual inspection found a coherent cat holding a readable HELLO WORLD sign, with no obvious gross corruption. One prompt/seed is functional evidence, not a formal quality evaluation.

## Full unprofiled baseline

[Run](../results/runs/a100-flux-klein-base-001__baseline__20261005-074425/), [summary](../results/runs/a100-flux-klein-base-001__baseline__20261005-074425/summary.json), [environment](../results/runs/a100-flux-klein-base-001__baseline__20261005-074425/environment.json), [validation](../results/runs/a100-flux-klein-base-001__baseline__20261005-074425/validation.json). Slurm11882219 on gilbreth-g007;11:44:25–11:50:47UTC. One first generation,three discarded warm-ups,ten measured generations. Same pinned config, no profiler attached.

| Metric | Median, ms | Measured min–max, ms |
|---|---:|---:|
| wall_ms | 25147.679 | 25054.344–25285.104 |
| text_encode_ms | 94.795 | 93.626–103.741 |
| denoise_ms | 24836.578 | 24725.643–24975.987 |
| vae_decode_ms | 167.762 | 166.576–169.206 |
| postprocess_ms | 39.709 | 34.388–48.494 |
| other_ms | 12.982 | 12.225–15.932 |

First generation 24861.913ms. Cached load plus CUDA placement 23.376680s; not a cold filesystem read. Measured peak allocated 17.331809GiB, reserved 19.601562GiB, device-wide sampled peak 20.614685GiB. These are distinct metrics. Component weights: VAE 0.156549GiB,text encoder 7.492431GiB,transformer 7.218764GiB.

All fourteen generation hashes agree. The two retained PNGs (first and measured-0) are1024×1024 and independently match their recorded hashes; remaining images have hash evidence only. Every generation has two text encodes,two transformer calls per each of50steps,one VAE decode and one postprocess. All stage sums and measured median/min/max values were independently checked. One development prompt/seed; no formal quality evaluation.

[W&B baseline](https://wandb.ai/mfaruqi-purdue-university/on-device-diffusion/runs/a100-flux-klein-base-001__baseline__20261005-074425).

## Separate torch.profiler capture

[Profile run](../results/runs/a100-flux-klein-base-001__profile__20261005-075123/), [trace validation](../results/runs/a100-flux-klein-base-001__profile__20261005-075123/profile-validation.json), [derived stage breakdown](../results/runs/a100-flux-klein-base-001__profile__20261005-075123/profile/stage_kernels.md), [operator table](../results/runs/a100-flux-klein-base-001__profile__20261005-075123/profile/op_table.txt). Job11882223 on gilbreth-g007 ran11:51:23–12:00:16UTC. The runner first executes an unprofiled1+3+10 protocol, then captures one extra generation. W&B's ordinary timing aggregates describe those preceding unprofiled observations; `diag/profile_*` and `profile/*` describe the extra capture.

The trace has two CPU and two GPU windows for text encoding and for each of50denoising steps, plus one VAE and one postprocess window. All104stage calls are retained; duplicate CFG labels are aggregated by summing both windows. Trace window sums and GPU coverage accounting passed independent checks. The extra generation took25.313186s synchronized host wall time; this is a single profiled diagnostic. Its image was not saved. The preceding protocol's two retained PNGs and all14recorded hashes passed validation.

Across denoising, the summed captured kernel durations comprise GEMM **50.70%**, attention **15.30%**, elementwise multiplication **11.52%**, and copy/cast **7.98%**. These are disjoint categorized kernel shares, not sums of nested operator totals. They identify major kernel families; they do not establish whether kernels are compute- or bandwidth-limited. The83MBraw trace is retained locally and remotely but excluded from the small W&B artifact; its SHA256 is in the validation receipt.

[W&B profile](https://wandb.ai/mfaruqi-purdue-university/on-device-diffusion/runs/a100-flux-klein-base-001__profile__20261005-075123).

## Interpretation

The Base checkpoint fits the specified A100 configuration and passes the CFG accounting checks. Denoising dominates the observed generation duration. The full unprofiled protocol supplies the repeated warm latency distribution. Profiling remains a separate diagnostic capture.

## Next experiment

- A separately labelled25-step variant of configs/a100-flux-klein-base-bf16.resolved.json would test a shorter Base schedule.

[W&B attempt](https://wandb.ai/mfaruqi-purdue-university/on-device-diffusion/runs/a100-flux-klein-base-001__attempt__20261005-033729).
