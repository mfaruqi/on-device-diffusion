---
type: concept
summary: The starting latent of a diffusion sample; the same integer seed gives different noise in different frameworks, so cross-engine image comparisons need shared noise.
status: active
aliases: [initial latents, seed, rng, philox]
updated: 2026-09-28
---

# Initial noise

Flow-matching sampling starts from Gaussian noise of the latent's shape. For FLUX.2 klein at 1024² that
is 128 × 64 × 64. Both engines use this shape in the same memory order.

- **PyTorch**: `torch.randn` on a CUDA generator, drawn directly in BF16 ([pipeline](../systems/pytorch-diffusers.md)).
- **stable-diffusion.cpp**: its own Philox-4×32 RNG (a CPU port imitating torch's CUDA randn), FP32,
  re-seeded per image. Element *i* uses key = seed and counter = (0, 0, i, 0), then a Box–Muller transform
  (`src/core/rng_philox.hpp` at the pinned tag).

A numpy port of sd.cpp's RNG matched it only to ~1 ulp, because of float `log`/`sin` rounding. The
exact noise therefore has to come from sd.cpp's own C++ code. The proposed approach is in
[D-006](../project/decisions.md#d-006-shared-initial-noise-from-sdcpps-philox-rng-proposed). The
proposal asks for initial noise matched "where supported" ([overview](../project/overview.md#evaluation-rules)).
