---
type: concept
summary: FLUX.2's sampling — Euler steps along a flow-matching trajectory with a resolution-dependent timestep shift (mu); defines what "4 steps" means.
status: active
aliases: [flowmatch euler, flux2 scheduler, dynamic shift]
updated: 2026-09-28
---

# Flow-matching sampler (FLUX.2)

The latent moves from noise (σ = 1) to data (σ = 0) in a fixed number of Euler steps. Each step runs one
transformer call when guidance is 1.0 ([model](../systems/flux2-klein-4b.md)). The σ schedule is
shifted by an amount that depends on the number of image tokens. For 4096 tokens and 4 steps, sd.cpp
logs `mu=2.291` ([record](../../experiments/a100-sdcpp-flux-klein-001.md#engine-audit-from-sdcpps-log-summaryjson--engine_audit)).
Diffusers uses its FlowMatchEuler scheduler with dynamic shifting (`scheduler.json` in each PyTorch run).

The two engines' σ values haven't been compared value by value
([comparison limits](../../experiments/compare-a100-pytorch-vs-sdcpp-flux-klein.md#limits)).
With 4 steps there are only four transformer calls to reuse across, which is why the proposal treats
the 4-step case as a test of when caching overhead outweighs its benefit ([cross-step reuse](cross-step-reuse.md)).
