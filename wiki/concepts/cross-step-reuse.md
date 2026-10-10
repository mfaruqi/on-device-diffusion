---
type: concept
summary: Reusing computation across denoising steps — exact reuse of step-independent values, or approximate caching (TeaCache, DiCache) of slowly changing features.
status: active
aliases: [step caching, feature caching, teacache, dicache]
updated: 2026-10-06
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

Measured so far, on the Jetson with stable-diffusion.cpp (one prompt and seed, diagnostics only for quality):
- **approximate reuse:** EasyCache skips half the steps of a 50-step Base generation but none of a 4-step
  one, where it only adds cost ([finding](../findings/step-caching-helps-at-50-steps-but-not-at-4-on-jetson.md));
- **exact reuse across requests:** the conditioning cache removes text encoding for a repeated prompt
  and, indirectly, much of the reload time ([finding](../findings/exact-conditioning-reuse-removes-most-jetson-reload-time.md));
- combined, the savings overlap ([finding](../findings/combined-reuse-savings-overlap-on-jetson-base.md)).
The other sd.cpp cache modes (DBCache, TaylorSeer, cache-dit, Spectrum) remain unmeasured
([engine](../systems/stable-diffusion-cpp.md)). Exact within-generation reuse needs graph analysis, which is
planned in the compiled pipeline ([levels](kernel-graph-plan-search.md)).
