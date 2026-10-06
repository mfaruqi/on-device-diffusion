---
type: concept
summary: Three levels at which diffusion inference can be optimized — kernel search (same math), graph transforms (fusion, memory planning) and plan search (what to compute, under memory and quality limits) — and where existing tools sit.
status: active
aliases: [optimization levels, plan search, autotuning, ML for ML]
updated: 2026-10-06
---

# Kernel, graph and plan search

| Level | What changes | Exact? | Existing tools |
|---|---|---|---|
| **Kernel search** | how one operator runs: tiling, loop order, thread mapping | yes | TVM AutoTVM/Ansor/[MetaSchedule](../papers/metaschedule.md); [TensorRT tactic timing](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/builder-performance.html); [`torch.compile` max-autotune](https://docs.pytorch.org/docs/main/generated/torch.compile.html) |
| **Graph transforms** | fusion, layouts, static memory planning | yes | TVM/[MLC passes](../papers/mlc-compiler-precedent.md); Inductor fusion; ggml graph allocation |
| **Plan search** | what is computed: precision per component, reuse across steps, which weights stay resident | not always | engine flags chosen by hand; rule-based auto-fit in [stable-diffusion.cpp](../systems/stable-diffusion-cpp.md) and [edge-dit.cpp](https://github.com/THU-MIG/edge-dit.cpp) |

**Kernel search** keeps the math fixed and measures candidate implementations, using a learned cost model
to decide which to measure (AutoTVM's "ML for ML"). OctoML productized it with TVM plus other runtimes
from 2019, and its successor OctoAI was acquired by NVIDIA in 2024
([report](https://www.bigdatawire.com/2024/09/30/octoai-snapped-up-by-nvidia)). MLC-LLM mostly uses
rule-based schedules (DLight) instead of measured tuning ([MLC review](../papers/mlc-compiler-precedent.md)).

**Where each level shows up in this project's measurements:**
- kernel: ggml's flash attention is about 4× slower than PyTorch's at the same shape
  ([finding](../findings/ggml-flash-attention-4x-slower-than-sdpa-flash-at-4608-tokens.md));
- graph: about a third of a PyTorch denoising step is unfused memory-bound work
  ([finding](../findings/pytorch-denoise-step-third-in-unfused-memory-bound-ops.md));
- plan: on the Jetson, weight reloading and step caching dominate, and their best setting depends on the
  workload ([reloading](../findings/weight-reloading-dominates-disk-backed-jetson-generations.md),
  [step caching](../findings/step-caching-helps-at-50-steps-but-not-at-4-on-jetson.md),
  [conditioning reuse](../findings/exact-conditioning-reuse-removes-most-jetson-reload-time.md)).

**Plan search is harder than kernel search:**
- it can change the output, so each candidate needs quality evidence;
- each measurement is a whole generation (seconds to minutes), not a kernel call;
- options interact ([finding](../findings/combined-reuse-savings-overlap-on-jetson-base.md)).

Interpretation: a learned cost model over plans, rather than kernels, is the plan-level analogue of
AutoTVM's idea. That is the comparison [RQ2](../rq/rq2.md) asks for. The proposed planner works at the plan
level and reuses TVM for the two levels below ([D-010](../project/decisions.md#d-010-deliver-a-tvm-compiled-klein-pipeline-with-the-planner-on-top)).
No existing engine selects plans by measuring candidates against a quality requirement. Engine
auto-fit uses memory rules (stable-diffusion.cpp: [failure on Jetson](../findings/sdcpp-auto-fit-fails-first-text-encoding-on-jetson.md);
edge-dit.cpp: precision and placement chosen together under `--max-vram`, per its README).

Related: [cross-step reuse](cross-step-reuse.md), [stage residency](stage-residency.md).
