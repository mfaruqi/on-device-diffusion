---
type: rq
summary: RQ1 — one matched A100 engine comparison; Jetson feasibility and repeated baseline evidence exist, but matched cross-device transfer remains untested.
status: active
updated: 2026-10-06
---

# RQ1: Which diffusion optimization choices transfer across devices and workloads?

**From the proposal**: "Measure the compute, memory, and quality tradeoffs of precision, reuse, residency,
and stage placement across CUDA, macOS Metal, and iOS Core ML/MLX, including whether caching remains
beneficial for few-step image generation and longer video runs." ([overview](../project/overview.md#research-questions))

## Evidence so far
Matched engine-comparison evidence covers A100-PCIE, FLUX.2 klein 1024², 4 steps, and two engines with identical weights:
- GEMM time transfers across engines ([finding](../findings/bf16-gemm-time-matches-across-pytorch-and-sdcpp.md)).
- Attention kernel efficiency doesn't ([finding](../findings/ggml-flash-attention-4x-slower-than-sdpa-flash-at-4608-tokens.md)).
- Activation-precision design costs ~70 ms per step in ggml ([finding](../findings/ggml-fp32-activations-add-conversion-and-idle-time.md)).
- The VAE decode is where engines diverge most, and it sets the peak in both ([im2col](../findings/sdcpp-vae-decode-dominated-by-im2col.md), [peak](../findings/vae-decode-sets-peak-memory.md)).
- Cold start transfers poorly: the engines' first-run and load costs differ by 2–3× ([finding](../findings/sdcpp-starts-faster-than-pytorch.md)).
- Hardware variants within "A100" aren't interchangeable ([finding](../findings/a100-sxm4-runs-faster-than-a100-pcie.md)).

Jetson also has disk-backed quantized 512×512 feasibility, repeated baseline and profiling evidence
([record](../../experiments/jetson-flux-klein-003.md), [results viewer](https://wandb.ai/mfaruqi-purdue-university/on-device-diffusion/groups/jetson-flux-klein-003)).
This does not establish a matched cross-device transfer result; that requires a dedicated comparison
under the [evaluation rules](../project/overview.md#evaluation-rules).

## Gaps
- Matched cross-device transfer remains untested. [MacBook Air M5](../systems/macbook-air-m5.md)
  and [iPhone](../systems/iphone.md) measurements remain planned.
- No second workload yet: [Wan 2.1 T2V-1.3B](../systems/wan2-1-t2v-1-3b.md) and [DreamLite-mobile](../systems/dreamlite-mobile.md) are planned.
- Jetson feasibility configurations change memory execution choices ([records](../experiments.md));
  controlled latency/quality comparisons of optimization policies remain to be established. No reuse policy is measured yet ([reuse](../concepts/cross-step-reuse.md)).
- No quality measurement. Shared initial noise is needed first ([D-006](../project/decisions.md#d-006-shared-initial-noise-from-sdcpps-philox-rng-proposed)).

## Next
Deepen profiling, screen edge-dit.cpp and audit existing auto-fit/caching options
([meeting actions](../../raw/meetings/2026-10-01-haoran-you.md#to-do)). Further DreamLite work and FastVideo
are temporarily deferred ([D-009](../project/decisions.md#d-009-prioritize-image-engine-baselines-and-profiling)).
Since [D-010](../project/decisions.md#d-010-deliver-a-tvm-compiled-klein-pipeline-with-the-planner-on-top), transfer is evaluated
across CUDA, Metal and WebGPU on the compiled klein pipeline; Wan is deferred until it runs.

Related: [RQ2](rq2.md) (selecting plans from these measurements), [RQ3](rq3.md) (joint planning).
