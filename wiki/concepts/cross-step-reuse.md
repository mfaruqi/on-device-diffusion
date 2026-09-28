---
type: concept
summary: Reusing computation across denoising steps — exact reuse of step-independent values, or approximate caching (TeaCache, DiCache) of slowly changing features.
status: active
aliases: [step caching, feature caching, teacache, dicache]
updated: 2026-09-28
---

# Cross-step reuse

Denoising repeats the same network over many steps, so some results can be reused:
- **exact reuse**: values that are provably step-independent, such as text-only computations that
  don't interact with changing image features. The compiler has to prove invariance.
- **approximate reuse**: skipping or caching blocks whose outputs change slowly (TeaCache, DiCache).
  These need calibration, quality evidence and runtime guards.

These definitions are from the proposal ([overview](../project/overview.md#system-components),
[registry families](../project/overview.md#optimization-families-registry)). No-reuse always stays a candidate, and
caching is rejected when its memory or decision overhead exceeds the saved compute.

In this project nothing has been measured yet. stable-diffusion.cpp ships several approximate
policies (`--cache-mode easycache|dbcache|taylorseer|cache-dit|spectrum`), and none has been run
([engine](../systems/stable-diffusion-cpp.md)). Approximate reuse is planned on longer schedules: Wan, and
FLUX.2 klein Base ([overview](../project/overview.md#workloads)).
