---
type: project
summary: Unresolved contradictions and open questions, each with both sides cited. Resolved items move to the Archive section.
status: active
updated: 2026-10-06
---

# Open questions

Each item: the question, the evidence (cited), `Status: Unresolved` until the user resolves it, and then
the resolution with a link. See [SCHEMA.md](SCHEMA.md#writing-rules), rule 4.

## Current

### Why is ggml's flash-attention kernel ~4× slower than PyTorch SDPA-flash at 4608 tokens?
- Evidence: per-step attention time differs by ~4× for the same shape
  ([finding](findings/ggml-flash-attention-4x-slower-than-sdpa-flash-at-4608-tokens.md)).
- Not established: kernel tiling for long non-causal sequences, the FP16 K/V conversion, or the launch configuration.
- Status: Unresolved

### Do both engines produce the same image from the same initial noise?
- Evidence: none yet. The initial noise differs between engines, so today's images differ by construction
  ([concept](concepts/initial-noise.md), [D-006](project/decisions.md#d-006-shared-initial-noise-from-sdcpps-philox-rng-proposed)).
- Status: Unresolved

### Why does each PyTorch denoise step get slightly slower from step 0 to step 3?
- Evidence: GEMM time rises 120 → 126 ms across steps with the same kernels and shapes, and the
  unprofiled baseline shows the same pattern in every run
  ([record](../experiments/a100-flux-klein-001.md#kernel-mix-within-one-denoise-step-profiled-generation-job-11807396)).
- Hypothesis: clock throttling under the A100-PCIE's 250 W limit. Untested; it needs clock and power logging.
- Status: Unresolved

### Should ExecuTorch (CoreML delegate) be an engine baseline on M1/iPhone?
- Evidence: its CoreML delegate matches native CoreML on iPhone 15 Pro for vision models
  ([paper](papers/executorch-mlsys2026.md#their-evaluation)). The proposal's Apple-device baselines are the
  DreamLite Core ML/MLX reference and planner-selected variants within that stack ([overview](project/overview.md#evaluation-rules)).
- Not established: whether it can export the diffusion workloads; its Metal backend is experimental.
- A scope question for the advisor.
- Status: Unresolved

### Which target comes after WebGPU: Android, or AMD/Intel GPUs?
- Evidence: the advisor's overview lists NVIDIA, Intel, AMD and Apple hardware and macOS, Linux, Windows,
  Android and web platforms ([overview](../raw/meetings/2026-10-06-advisor-scope.md#project-overview-as-provided-by-prof-haoran-you)).
  The confirmed scope fixes CUDA, Metal and WebGPU ([D-010](project/decisions.md#d-010-deliver-a-tvm-compiled-klein-pipeline-with-the-planner-on-top));
  the reply did not rank the remaining targets ([answer](../raw/meetings/2026-10-06-advisor-scope.md#scope-question-and-answer)).
- Status: Unresolved

### Is Jetson weight-reload time storage-bound or page-cache-bound?
- Evidence: the same transformer reload is much faster when the text encoder was not re-read in that
  generation ([finding](findings/exact-conditioning-reuse-removes-most-jetson-reload-time.md)), and loading dominates
  4-step generations ([finding](findings/weight-reloading-dominates-disk-backed-jetson-generations.md)).
- Not established: file-cache residency, physical microSD reads, or allocation cost; none were measured
  ([record](../experiments/jetson-flux-klein-003.md)).
- Status: Unresolved

## Archive

(none yet)
