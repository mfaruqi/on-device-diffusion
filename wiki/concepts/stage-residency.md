---
type: concept
summary: Which pipeline components' weights and buffers are resident on the device during each stage (text encode, denoise, decode) — a memory-planning choice.
status: active
aliases: [residency, sequential stage loading, offload]
updated: 2026-10-06
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
- on the Jetson with disk-backed weights, reloading each component every generation is most of a 4-step
  image ([finding](../findings/weight-reloading-dominates-disk-backed-jetson-generations.md)), and whether one
  component is re-read changes how fast another reloads ([finding](../findings/exact-conditioning-reuse-removes-most-jetson-reload-time.md));
- sd.cpp's automatic placement on the Jetson failed its first generation ([finding](../findings/sdcpp-auto-fit-fails-first-text-encoding-on-jetson.md)).

A deployment-framework precedent: ExecuTorch plans tensor lifetimes ahead of time into fixed arenas, and
frees large delegate segments after initialization to lower peak memory ([paper](../papers/executorch-mlsys2026.md#mechanisms-relevant-here)).

On shared-memory devices, moving weights between CPU and GPU allocations doesn't by itself reduce the
footprint ([overview](../project/overview.md#targets)).
