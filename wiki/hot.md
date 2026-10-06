---
type: project
summary: Current project state, active campaign and blockers. Read first each session.
status: active
updated: 2026-10-06
---

# Hot: current state

**Week 2** (Oct 5–9). **Scope changed** ([D-010](project/decisions.md#d-010-deliver-a-tvm-compiled-klein-pipeline-with-the-planner-on-top)):
the thesis delivers (a) a TVM-compiled FLUX.2 klein pipeline on CUDA, Metal and WebGPU and (b) the joint
reuse/precision/residency planner on top. Hypothesis and RQs unchanged ([overview](project/overview.md#scope-amendments)).
Schedule is optimistic with roll-over allowed ([milestones](project/milestones.md#revised-schedule)).

## Established evidence
- A100 distilled klein baselines (PyTorch, sd.cpp) and comparison; Jetson disk-backed Q4 512² baseline and profiles ([registry](experiments.md)).
- Base workload (50 steps, guidance 4) on A100 and Jetson; engine option sweeps with no-cache controls ([registry](experiments.md)).
- Jetson Base: EasyCache + exact conditioning reuse combined passes the campaign speed rule; no formal quality approval ([comparison](../experiments/compare-jetson-base-combined-reuse.md)).
- Jetson distilled: exact conditioning reuse passes the speed rule; EasyCache, prefetch-off and mmap fail it ([conditioning](../experiments/compare-jetson-distilled-conditioning-reuse.md), [EasyCache](../experiments/compare-jetson-distilled-easycache.md)).
- Jetson auto-fit chose CPU/GPU placement but failed first text encoding ([004](../experiments/jetson-flux-klein-004.md)).
- edge-dit full A100 baseline validated with stated limits ([record](../experiments/a100-edgedit-flux-klein-001.md)); Torch-TensorRT compile gate failed ([record](../experiments/a100-flux-klein-torchtrt-002.md)).

## Campaign (Codex is the single writer while it runs)
- State and handoff: [state](../output/overnight-20261004/state.json), [README](../output/overnight-20261004/README.md). Jetson scope closed; A100 Base conditioning full protocol pending ([record](../experiments/a100-sdcpp-flux-klein-base-003.md)).
- Base engine comparison keeps RNG/timer confounds ([comparison](../experiments/compare-a100-base-pytorch-vs-sdcpp.md)).

## Next (Week 2 → 3)
1. TVM sandbox: import a small randomly initialized FLUX-style transformer on CPU; inspect the IR, fusion and memory planning.
2. Week 3 gate: import klein's transformer step into TVM Relax and match PyTorch numerics on CUDA ([milestones](project/milestones.md#revised-schedule)).
3. MacBook Air M5 setup: sd.cpp Metal smoke ([system](systems/macbook-air-m5.md)).
4. Review the draft planning hypothesis and three candidate contributions ([RQ3](rq/rq3.md)); review CacheQuant and Q&C before any novelty claim ([reading list](papers/reading-list.md)).
5. Approve or revise the TVM approach and engine-level planner fallback ([D-011, proposed](project/decisions.md#d-011-build-the-compiled-pipeline-mlc-style-on-tvm-with-an-engine-level-planner-fallback-proposed)).
6. Shared initial noise for engine comparisons remains proposed ([D-006](project/decisions.md#d-006-shared-initial-noise-from-sdcpps-philox-rng-proposed)).

## Open
- Next target after WebGPU; whether Jetson reload time is storage- or page-cache-bound; four earlier questions ([open questions](open-questions.md)).
- New Jetson findings: [step caching vs schedule length](findings/step-caching-helps-at-50-steps-but-not-at-4-on-jetson.md), [conditioning reuse](findings/exact-conditioning-reuse-removes-most-jetson-reload-time.md), [overlapping savings](findings/combined-reuse-savings-overlap-on-jetson-base.md); framing in [kernel, graph and plan search](concepts/kernel-graph-plan-search.md).
