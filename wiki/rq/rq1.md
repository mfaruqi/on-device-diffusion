---
type: rq
summary: RQ1 — which diffusion optimization choices transfer across devices and workloads. Evidence so far — one A100 engine comparison.
status: active
updated: 2026-09-28
---

# RQ1: Which diffusion optimization choices transfer across devices and workloads?

**From the proposal**: "Measure the compute, memory, and quality tradeoffs of precision, reuse, residency,
and stage placement across CUDA, macOS Metal, and iOS Core ML/MLX, including whether caching remains
beneficial for few-step image generation and longer video runs." ([overview](../project/overview.md#research-questions))

## Evidence so far
One device (A100-PCIE), one workload (FLUX.2 klein 1024², 4 steps), two engines with identical weights:
- GEMM time transfers across engines ([finding](../findings/bf16-gemm-time-matches-across-pytorch-and-sdcpp.md)).
- Attention kernel efficiency doesn't ([finding](../findings/ggml-flash-attention-4x-slower-than-sdpa-flash-at-4608-tokens.md)).
- Activation-precision design costs ~70 ms per step in ggml ([finding](../findings/ggml-fp32-activations-add-conversion-and-idle-time.md)).
- The VAE decode is where engines diverge most, and it sets the peak in both ([im2col](../findings/sdcpp-vae-decode-dominated-by-im2col.md), [peak](../findings/vae-decode-sets-peak-memory.md)).
- Cold start transfers poorly: the engines' first-run and load costs differ by 2–3× ([finding](../findings/sdcpp-starts-faster-than-pytorch.md)).
- Hardware variants within "A100" aren't interchangeable ([finding](../findings/a100-sxm4-runs-faster-than-a100-pcie.md)).

## Gaps
- No second device yet: [Jetson Orin Nano](../systems/jetson-orin-nano.md), [M1 MacBook Pro](../systems/m1-macbook-pro.md)
  and [iPhone](../systems/iphone.md) are all planned.
- No second workload yet: [Wan 2.1 T2V-1.3B](../systems/wan2-1-t2v-1-3b.md) and [DreamLite-mobile](../systems/dreamlite-mobile.md) are planned.
- No optimization choice has been varied yet: precision, reuse and residency are all at reference settings.
- No quality measurement. Shared initial noise is needed first ([D-006](../project/decisions.md#d-006-shared-initial-noise-from-sdcpps-philox-rng-proposed)).

## Next
Jetson image baseline (week 1); one-change variants on A100 for precision, residency and VAE tiling; M1 and Wan baselines (week 2).

Related: [RQ2](rq2.md) (selecting plans from these measurements), [RQ3](rq3.md) (joint planning).
