---
type: project
summary: The research project as proposed — hypothesis, planner objective, components, workloads, targets, RQs, evaluation rules, milestones.
status: active
updated: 2026-09-28
---

# Project overview (condensed from the proposal)

Source: [the proposal PDF](../../proposal/OnDeviceDiffusionProposal.pdf). This file is a faithful condensation for
quick reference. If the two disagree, the PDF wins and this file gets fixed. Update this
file only when the proposal itself changes, not when a session's plans change.

## Hypothesis

Jointly planning **reuse regions, precision and memory residency** can reduce end-to-end
latency under a **fixed memory budget** and a **declared quality requirement**, compared with
fixed configurations or independently selected policies.

The contribution is **diffusion-specific planning inside a compiler/runtime**: represent the
denoising structure, expose compatible optimization choices, and use measured costs and quality
evidence to select and specialize an execution plan. Existing engines (stable-diffusion.cpp,
edge-dit.cpp, TVM, MLC Web SD) are baselines/precedents, not the contribution.

## Planner objective

p* = argmin over compatible plans p of L̂(p), subject to M̂(p) + M_reserve ≤ B and p ∈ P_Q

- B: available memory budget. P_Q: plans that meet the declared quality criteria on calibration data.
- Peak memory estimate covers weights, activations, workspaces, caches and loading transitions.
- Combinations are measured together (costs and errors need not be additive).
- No-reuse execution is always a candidate; caching is rejected when overhead exceeds benefit.
- An infeasible request produces a reported limitation, never a silent model swap.

## System components

1. **Compile-time analysis**: model adapters import stage graphs into TVM Relax and describe the
   timestep loop, conditioning dependencies and sampling config. Finds repeated regions, shapes,
   tensor lifetimes, step-independent computations. Slowly changing features are approximation
   candidates, not proven invariants.
2. **Optimization registry**: each implementation declares architectures, devices, dtypes/layouts,
   regions, state, memory/runtime costs, hooks, constraints. Exact and approximate entries have
   different contracts; approximate ones also record calibration needs, quality evidence and guards.
3. **Deployment-time planner**: enumerate → reject infeasible (unsupported ops, conflicts, over budget)
   → rank using bundled seed measurements + local cache, benchmark a few uncertain candidates →
   select one plan or report infeasibility.
4. **Persistent history**: a versioned seed database (from this study) plus a writable local cache,
   keyed by checkpoint/graph fingerprint, workload, device, compiler/runtime version, policy version.
   Quality eligibility is kept separate from performance records.
5. **Specialization + adaptive runtime**: the execution plan is an explicit contract (per-stage
   precision and provider, reuse regions/thresholds, residency and buffer lifetimes, prevalidated
   fallbacks). At runtime only bounded, plan-authorized decisions are made.

## Optimization families (registry)

| Family | Scope |
|---|---|
| Precision | full/half and low-bit, per component, with compatible kernels (SVDQuant candidate). Low-bit storage alone ≠ faster |
| Reuse | exact reuse of proven step-independent values, no reuse, one approximate policy (TeaCache / DiCache), then a second to test registry independence |
| Memory | buffer reuse, stage residency, selective recomputation, tiled decoding |
| Kernels | existing generated/library kernels, fused attention, low-bit ops; kernel support constrains precision and region boundaries |

Minimum prototype: jointly choose a small set of reuse regions and memory policies across
supported precision configs. One-device results precede cross-device and video transfer.

## Workloads

| Workload | Role |
|---|---|
| FLUX.2 [klein] 4B (4-step distilled), text-to-image | main image workload; tests when caching overhead beats benefit at few steps |
| FLUX.2 [klein] 4B Base (where feasible) | longer denoising for approximate-reuse experiments |
| Wan 2.1 T2V-1.3B, text-to-video | main video workload; start with short 480p clips, record frames/fps/schedule/resolution |
| DreamLite-mobile (0.39B U-Net, 4-step) | bounded iPhone workload and deployment reference, not a third full study |

## Targets

| Device | Path | Role |
|---|---|---|
| NVIDIA A100 (Gilbreth) | CUDA | development and reference platform |
| Jetson Orin Nano (8 GB shared) | CUDA | edge target |
| M1 MacBook Pro | Metal | edge target |
| iPhone 16 Pro / Pro Max (8 GB) or newer | Core ML / MLX (DreamLite export + Swift runtime) | phone target |

Device records: usable memory, enabled compute units, power mode, OS/runtime versions, initial
thermal state. Memory accounting covers the whole pipeline and shared physical memory.

## Research questions

- **RQ1**: Which diffusion optimization choices transfer across devices and workloads?
  (precision, reuse, residency, stage placement across CUDA, Metal, Core ML/MLX; whether caching
  still pays for few-step image vs longer video.)
- **RQ2**: Can structure, hardware information and persistent history select effective plans at
  low cost? (rule-based vs measured lookup vs learned cost model at equal profiling budgets;
  held-out workloads; prediction error, search cost, constraint violations, latency vs best
  evaluated feasible plan.)
- **RQ3**: Does joint planning with bounded runtime adaptation outperform fixed or independently
  tuned policies? (isolate region selection, memory planning, runtime guards under matched
  checkpoints and quality requirements.)

## Evaluation rules

- Baselines: stable-diffusion.cpp, edge-dit.cpp (incl. their auto-fitting), Wan reference, FastVideo;
  DreamLite Core ML/MLX on iPhone. Within each stack: no-reuse plan, tuned fixed config,
  independently tuned policies, joint planner. "Best evaluated feasible plan" comes from a bounded
  offline sweep, not a global oracle.
- Match prompts, initial noise where supported, resolution, schedule, guidance, video settings.
- Report changed/compressed checkpoints and unsupported combinations separately from execution-only changes.
- Quality: separate calibration vs held-out prompts/seeds. Freeze criteria before planner tuning:
  max degradation on GenEval (image), selected VBench dimensions (video), structured blinded review.
  Paired perceptual metrics (LPIPS etc.) diagnose error but do not replace alignment/quality metrics.
- Latency: full generation and per stage; separate compile/export, calibration, planning, cold load,
  warm inference. Include transfers, syncs, probes, cache management. Median, tail, uncertainty.
- Memory: peak device allocations **and** total physical memory on shared-memory targets. Record OOMs,
  fallback frequency, headroom. iPhone: thermal state, sustained runs, package size, energy.
- Ablations: remove history, diffusion-specific analysis, joint selection, runtime adaptation in turn.
  Freeze database and predictor before held-out tests. Keep existing adaptive caching active in
  baselines.

## Milestones (Sep 28 – Dec 11, 2026)

| Week | Dates | Goal | Deliverable |
|---|---|---|---|
| 1 | Sep 28–Oct 2 | A100 and Jetson image baselines; review DreamLite; define the bounded planning hypothesis | Baselines and scope |
| 2 | Oct 5–9 | M1 and A100 Wan baselines; verify iPhone/DreamLite export and access; specify registry and history records | Evaluation protocol |
| 3 | Oct 12–16 | Compile a denoising region; expose loop dependencies and candidate boundaries; verify target feasibility | Analysis prototype |
| 4 | Oct 19–23 | Connect stages and one reuse policy; per-stage provider metadata; iOS measurement capture | Instrumented pipeline |
| 5 | Oct 26–30 | Bounded joint search using rules and measured costs; prepare execution variants | Planner prototype |
| 6 | Nov 2–6 | Runtime guards and a second policy; learned ranking vs lookup if data permit | Adaptive prototype |
| 7 | Nov 9–13 | Transfer across CUDA, Metal, Core ML/MLX; isolate integration overhead | Transfer results |
| 8 | Nov 16–20 | Planner/history/runtime ablations; resource pressure; iPhone sustained runs; amortization | Core comparisons |
| 9 | Nov 23–27 | Resolve issues; freeze configs, history, eval scripts | Frozen experiments |
| 10 | Nov 30–Dec 4 | Held-out quality tests and repeated timings; matched-quality comparisons | Final results |
| 11 | Dec 7–11 | Reproduce key results; compile report | Thesis draft and artifacts |

## Fallbacks and scope control

- If FLUX doesn't fit Jetson or iPhone: try supported quantization, sequential stage loading and
  tiled decoding first. Lower resolution or a smaller/replacement checkpoint is a **separately
  labelled workload**.
- DreamLite-mobile is the phone-feasible baseline and is never presented as an optimized FLUX result.
- Unsupported compiler ops may use decompositions or existing-runtime stages, with the integration cost measured.
- If direct iOS lowering stays blocked, keep the Core ML/MLX artifacts behind the shared plan interface and narrow the claim.
- A learned predictor is an evaluated alternative, not a dependency.
- Out of scope: agent integration and joint LLM/diffusion residency (future work).
