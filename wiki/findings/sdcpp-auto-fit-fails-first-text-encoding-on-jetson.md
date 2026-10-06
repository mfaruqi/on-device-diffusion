---
type: finding
summary: sd.cpp auto-fit on Jetson placed the transformer on the GPU and the text encoder and VAE in CPU memory, then failed the first text encoding.
status: supported
confidence: medium
rq: [RQ1]
sources: [../../experiments/jetson-flux-klein-004.md]
updated: 2026-10-06
---

# sd.cpp auto-fit fails the first text encoding on the Jetson

**Claim.** With automatic placement, sd.cpp logged an auto-fit plan that kept transformer weights on CUDA0
and put the text-encoder and VAE weights in CPU memory, created the context, and then failed with
"failed to encode prompt" on the first generation.

**Evidence.** [Record](../../experiments/jetson-flux-klein-004.md) and its
[run directory](../../results/runs/jetson-flux-klein-004__attempt__20261005-000226/). The attempt is preserved as failed; no settings were altered
and retried.

**Interpretation.** The engine's automatic policy chooses placement by memory rules and does not
validate the result. On a shared-memory device, CPU placement does not add capacity
([overview](../project/overview.md#targets)). The failure's root cause is not established.

Related: [stable-diffusion.cpp](../systems/stable-diffusion-cpp.md), [stage residency](../concepts/stage-residency.md).
