---
type: paper
summary: ExecuTorch (Meta, MLSys 2026) — PyTorch-native ahead-of-time export plus a lean C++ runtime with pluggable hardware backends; LLM and vision results on phones, no diffusion. Design precedent for memory planning and capability-based backend delegation.
status: ingested
ref: Nachin, Desai, Jia et al., "ExecuTorch: A Unified PyTorch Solution to Run AI Models On-Device", MLSys 2026 (Industry Track)
source: ../../raw/papers/executorch-mlsys2026.pdf
updated: 2026-09-28
---

# ExecuTorch: A Unified PyTorch Solution to Run AI Models On-Device (MLSys 2026)

Source: [PDF](../../raw/papers/executorch-mlsys2026.pdf) ([proceedings](https://proceedings.mlsys.org/paper_files/paper/2026/file/236f915dd02af4f11927f67330b21d4b-Paper-Conference.pdf)); code: [pytorch/executorch](https://github.com/pytorch/executorch).

## What it is
- `torch.export` captures a static graph (Export IR), lowered to an **Edge Dialect** of fewer than 300 core
  ATen operators with explicit dtypes and memory layout (§4.1).
- Serialized to a `.pte` file of linear instruction lists run by a C++17 runtime. The runtime does no
  dynamic allocation; all memory is supplied by the caller (§4.5, §5.1).
- 12 hardware backends, including XNNPACK (CPU), Vulkan (mobile GPU), Qualcomm QNN (NPU), CoreML and
  Arm Ethos-U. CUDA and Metal are "under active development" / experimental (§7, §12).

## Mechanisms relevant here
- **Memory planning** ahead of time: tensor sizes and lifetimes are packed into fixed-size arenas with a
  greedy best-fit reuse; custom planners are pluggable (§4.2).
- **Peak-memory control at load**: large delegate segments are freed after initialization; page-aligned
  segments can be memory-mapped without copying (§4.5).
- **Capability-based delegation**: each backend declares the operators it supports; a partitioner routes
  only matching subgraphs to it, with CPU fallback for the rest (§4.4). Quantization is applied according
  to backend-declared capabilities (§1, §4.3).
- **Cost of partial delegation**: for ViT and Swin-T on the mobile GPU, CPU-fallback operators took
  about 22–29% of latency, while the CPU↔GPU copies at graph breaks took only 5–6% (§11).
- **Devtools**: ETRecord (export-time graph and debug metadata) and ETDump (runtime operator latencies,
  memory lifetimes, delegate events) are read through an Inspector API (§8).

## Their evaluation
- Devices: Samsung Galaxy S25 Ultra, Pixel 9 Pro XL, iPhone 15 Pro. Workloads: Qwen3 0.6B, Llama 3.2 1B,
  Phi4 Mini, and four vision models, compared against llama.cpp, ONNX Runtime, LiteRT and CoreML (§11).
- Protocol: a warm-up run, 3 runs with min/max reported, a 60 s cooldown between runs against thermal
  throttling (§11); ±5–10% run-to-run variance expected (artifact appendix).
- The CoreML delegate matches native CoreML on iPhone (§11).
- Operator breakdown against llama.cpp: linear layers are at parity, attention implementations differ by
  1.1–2.6×, and decomposing RMSNorm, RoPE and activations into core ops adds overhead (appendix A).

## What it means for this project
- **Design precedent** for the proposal's registry (implementations declaring their capabilities) and
  memory planning ([overview](../project/overview.md#system-components), [RQ3](../rq/rq3.md)).
- **Independent support, from a different domain**, for our engine-comparison pattern: GEMMs transfer
  while attention and op decomposition don't
  ([GEMM](../findings/bf16-gemm-time-matches-across-pytorch-and-sdcpp.md),
  [attention](../findings/ggml-flash-attention-4x-slower-than-sdpa-flash-at-4608-tokens.md),
  [unfused ops](../findings/pytorch-denoise-step-third-in-unfused-memory-bound-ops.md)).
- **Phone protocol**: cooldown and variance practice to reuse for the [iPhone](../systems/iphone.md) runs.
- **Possible Apple-device path** (CoreML delegate), outside the proposal's current baselines
  ([open question](../open-questions.md#should-executorch-coreml-delegate-be-an-engine-baseline-on-m1iphone)).

## Limits
No diffusion workloads. CUDA and Metal backends aren't production yet. Ahead-of-time artifacts are
hardware-specific, so different Android NPUs need different model files (§12).
