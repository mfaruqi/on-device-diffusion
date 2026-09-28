---
type: concept
summary: Fused attention kernels that avoid materialising the S×S score matrix; implementations differ a lot in speed for long non-causal sequences.
status: active
aliases: [fused attention, sdpa-flash, flash_attn_ext]
updated: 2026-09-28
---

# Flash attention

Computes softmax(QKᵀ)V in tiles without materialising the S×S score matrix, so memory stays linear in
sequence length and the op becomes compute-bound. FLUX.2 klein at 1024² attends over 4608 tokens
(B1, H24, D128) in 25 blocks per denoise step ([model](../systems/flux2-klein-4b.md)).

Implementations seen in this project:
- **PyTorch SDPA-flash** (`_flash_attention_forward`, BF16): ~180 TFLOP/s on A100-PCIE ([record](../../experiments/a100-flux-klein-001.md#kernel-mix-within-one-denoise-step-profiled-generation-job-11807396)).
- **ggml `flash_attn_ext_f16`** (F16 inputs converted from FP32 activations): ~44 TFLOP/s for the same shape
  ([record](../../experiments/a100-sdcpp-flux-klein-001.md#profiler-run-diagnostic-nsight-systems-job-11818087)).

FLOPs per call are counted as 4·B·H·S_q·S_k·D ([metrics](../methods/baseline-metrics.md#profiling-diagnostic-only)).
Kernel choice is one of the registry's kernel family options ([overview](../project/overview.md#optimization-families-registry)).

Attention-kernel choice also dominates cross-framework gaps for LLMs on phones ([ExecuTorch paper](../papers/executorch-mlsys2026.md#their-evaluation)).

See: [finding](../findings/ggml-flash-attention-4x-slower-than-sdpa-flash-at-4608-tokens.md).
