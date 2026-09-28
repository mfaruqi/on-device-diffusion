---
type: concept
summary: Which pipeline components' weights and buffers are resident on the device during each stage (text encode, denoise, decode) — a memory-planning choice.
status: active
aliases: [residency, sequential stage loading, offload]
updated: 2026-09-28
---

# Stage residency

A text-to-image pipeline runs stages that use different components: the text encoder, then the
denoiser repeatedly, then the VAE decoder. Keeping every component resident for the whole request is
the simplest plan, but the peak is set by whatever overlaps. Residency choices include releasing a
component after its stage, loading the next one ahead of time, and offloading to host memory. These
are part of the proposal's memory family ([overview](../project/overview.md#optimization-families-registry)).

For FLUX.2 klein, measured facts that make residency matter:
- the text encoder is half the weights and is used only at the start ([finding](../findings/text-encoder-half-of-weights-little-of-time.md));
- the VAE decode sets the peak ([finding](../findings/vae-decode-sets-peak-memory.md)).

On shared-memory devices, moving weights between CPU and GPU allocations doesn't by itself reduce the
footprint ([overview](../project/overview.md#targets)).
