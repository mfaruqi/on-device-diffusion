---
type: project
summary: Milestone status per week (Sep 28 – Dec 11, 2026). Goals are from the proposal; status and evidence are updated as work lands.
status: active
updated: 2026-09-28
---

# Milestones

Goals and deliverables are quoted from the [overview](overview.md#milestones-sep-28--dec-11-2026). Status
is one of: done, partial, not started. Evidence links point to records or pages, never to chat.

| Week | Dates | Goal (proposal) | Status | Evidence / what's left |
|---|---|---|---|---|
| 1 | Sep 28–Oct 2 | Establish A100 and Jetson image baselines; review DreamLite and define the bounded planning hypothesis | **partial** | A100 done for two engines: [PyTorch](../../experiments/a100-flux-klein-001.md), [sd.cpp](../../experiments/a100-sdcpp-flux-klein-001.md). [DreamLite source review](../papers/dreamlite.md) done. Left: Jetson baseline, written planning hypothesis |
| 2 | Oct 5–9 | Add M1 and A100 Wan baselines; verify iPhone/DreamLite export and device access; specify registry and history records | not started | – |
| 3 | Oct 12–16 | Compile a denoising region; expose loop dependencies and candidate boundaries; verify target feasibility | not started | – |
| 4 | Oct 19–23 | Connect stages and one reuse policy; add per-stage provider metadata and iOS measurement capture | not started | – |
| 5 | Oct 26–30 | Implement bounded joint search using rules and measured costs; prepare compatible execution variants | not started | – |
| 6 | Nov 2–6 | Add runtime guards and a second policy; compare learned ranking with lookup if data permit | not started | – |
| 7 | Nov 9–13 | Evaluate transfer across CUDA, macOS Metal, and iOS Core ML/MLX; isolate integration overhead | not started | – |
| 8 | Nov 16–20 | Run planner, history, and runtime ablations; test resource pressure, iPhone sustained runs, and preparation amortization | not started | – |
| 9 | Nov 23–27 | Resolve any pending issues and freeze configurations, history, and evaluation scripts | not started | – |
| 10 | Nov 30–Dec 4 | Run held-out quality tests and repeated timings; complete matched-quality comparisons | not started | – |
| 11 | Dec 7–11 | Reproduce key results and compile a report | not started | – |

Work done before week 1 (Sep 23–27): the A100 reference kit, the sd.cpp engine baseline, and the
wiki. See the [log](../log.md).
