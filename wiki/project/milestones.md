---
type: project
summary: Milestone status per week. Revised schedule per D-010 (compiled klein pipeline on CUDA/Metal/WebGPU + planner); optimistic, with roll-over allowed past Dec 11.
status: active
updated: 2026-10-06
---

# Milestones

The schedule follows [D-010](decisions.md#d-010-deliver-a-tvm-compiled-klein-pipeline-with-the-planner-on-top). It is deliberately
optimistic: a goal that slips rolls over to the next week, and work may continue past Dec 11 ([guidance](../../raw/meetings/2026-10-06-advisor-scope.md#scope-question-and-answer)).
Status is one of: done, partial, not started. Evidence links point to records or pages, never to chat.
Gates are go/no-go checks; a failed gate is recorded and triggers the "Revisit if" of D-010.

## Revised schedule

| Week | Dates | Goal | Gate / deliverable | Status | Evidence / what's left |
|---|---|---|---|---|---|
| 1 | Sep 28–Oct 2 | A100 and Jetson image baselines; review DreamLite | Baselines and scope | **partial** | [PyTorch](../../experiments/a100-flux-klein-001.md), [sd.cpp](../../experiments/a100-sdcpp-flux-klein-001.md), [Jetson](../../experiments/jetson-flux-klein-003.md); [DreamLite review](../papers/dreamlite.md). Left: written planning hypothesis. |
| 2 | Oct 5–9 | Base workload, engine baselines and option sweeps; scope decision; TVM environment; Apple laptop setup | Klein import sandbox runs on CPU; sd.cpp Metal smoke on the M5 Air | **partial** | Base and option campaign in the [registry](../experiments.md); [scope](decisions.md#d-010-deliver-a-tvm-compiled-klein-pipeline-with-the-planner-on-top). Left: TVM sandbox, M5 Air setup. |
| 3 | Oct 12–16 | Import klein's transformer (one denoising step) into TVM Relax; check numerics against PyTorch on CUDA | **Gate:** transformer imports and matches; unsupported ops listed | not started | – |
| 4 | Oct 19–23 | End-to-end compiled klein on A100: text encoder (MLC Qwen3 path if it matches), transformer loop, VAE; static memory plan | Compiled pipeline vs PyTorch/sd.cpp baselines | not started | – |
| 5 | Oct 26–30 | Same pipeline on Jetson (CUDA) and M5 Air (Metal); 4-bit group-quantized variant with fused dequantize | Two edge targets run the compiled pipeline | not started | – |
| 6 | Nov 2–6 | WebGPU build in the browser (512², quantized); first compiled reuse variants (exact invariants, step-cache region) | **Gate:** klein runs in a browser | not started | – |
| 7 | Nov 9–13 | Planner prototype: registry, memory-timeline feasibility, history-based ranking, bounded search over precision × reuse × residency | Planner prototype ([RQ3](../rq/rq3.md)) | not started | – |
| 8 | Nov 16–20 | Runtime guards and a second reuse policy; rules vs lookup vs learned ranking ([RQ2](../rq/rq2.md)) | Adaptive prototype | not started | – |
| 9 | Nov 23–27 | Transfer across CUDA, Metal and WebGPU ([RQ1](../rq/rq1.md)); planner/history/runtime ablations | Core comparisons | not started | – |
| 10 | Nov 30–Dec 4 | Freeze configs; held-out quality tests; repeated timings | Frozen experiments | not started | – |
| 11 | Dec 7–11 | Reproduce key results; compile report | Thesis draft and artifacts | not started | – |

## Roll-over candidates (after Week 11, or earlier if ahead)

Ordered by expected value. Each needs its own go/no-go before starting.
- Compiled Wan 2.1 T2V-1.3B pipeline; Wan engine baselines (sd.cpp supports Wan 2.1) as the cheaper first step.
- Android via Vulkan or OpenCL; AMD/Intel GPUs via Vulkan or ROCm.
- iPhone: Core ML or Core AI path, and DreamLite-mobile as the bounded phone reference.
- klein Base in the compiled pipeline (longer schedules for reuse experiments).

## Original proposal schedule (superseded by D-010)

| Week | Goal (proposal) |
|---|---|
| 1 | Establish A100 and Jetson image baselines; review DreamLite and define the bounded planning hypothesis |
| 2 | Add M1 and A100 Wan baselines; verify iPhone/DreamLite export and device access; specify registry and history records |
| 3 | Compile a denoising region; expose loop dependencies and candidate boundaries; verify target feasibility |
| 4 | Connect stages and one reuse policy; add per-stage provider metadata and iOS measurement capture |
| 5 | Implement bounded joint search using rules and measured costs; prepare compatible execution variants |
| 6 | Add runtime guards and a second policy; compare learned ranking with lookup if data permit |
| 7 | Evaluate transfer across CUDA, macOS Metal, and iOS Core ML/MLX; isolate integration overhead |
| 8 | Run planner, history, and runtime ablations; test resource pressure, iPhone sustained runs, and preparation amortization |
| 9 | Resolve any pending issues and freeze configurations, history, and evaluation scripts |
| 10 | Run held-out quality tests and repeated timings; complete matched-quality comparisons |
| 11 | Reproduce key results and compile a report |

Work done before week 1 (Sep 23–27): the A100 reference kit, the sd.cpp engine baseline, and the
wiki. See the [log](../log.md).
