---
type: system
kind: model
summary: FLUX.2 [klein] 4B — 4-step distilled text-to-image model; the main image workload.
status: active
aliases: [klein, flux2-klein, flux.2 klein]
updated: 2026-09-28
---

# FLUX.2 [klein] 4B

- **Role**: main image workload ([overview](../project/overview.md#workloads)).
- **Checkpoint**: `black-forest-labs/FLUX.2-klein-4B` @ `e7b7dc27f91deacad38e78976d1f2b499d76a294`, not gated.
- **Components** (BF16): Qwen3 text encoder 7.49 GiB, transformer (5 double-stream + 20 single-stream blocks)
  7.22 GiB, VAE 0.16 GiB ([record](../../experiments/a100-flux-klein-001.md#memory)).
- **Workload shape at 1024²**: latent 128 × 64 × 64, 4096 image tokens + 512 text tokens (prompt padded to 512).
  Four steps, guidance 1.0, so one transformer call per step and no CFG.
- **Sampler**: flow-matching Euler with a dynamic shift ([concept](../concepts/flow-matching-sampler.md)).
- **Packagings**: the diffusers folders and a single-file transformer in the same commit are value-identical
  ([equivalence report](../../configs/sdcpp-weights.check.json)).
- **Base variant** (FLUX.2 klein Base 4B, longer schedule) is planned for reuse experiments ([overview](../project/overview.md#workloads)).
