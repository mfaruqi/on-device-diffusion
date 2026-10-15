---
type: project
summary: Current project state, active campaign and blockers. Read first each session.
status: active
updated: 2026-10-15
---

# Hot: current state

**Week 3** (Oct 12–16). **Scope changed** ([D-010](project/decisions.md#d-010-deliver-a-tvm-compiled-klein-pipeline-with-the-planner-on-top)):
the thesis delivers (a) a TVM-compiled FLUX.2 klein pipeline on CUDA, Metal and WebGPU and (b) the joint reuse/precision/residency planner on top. Hypothesis and RQs unchanged ([overview](project/overview.md#scope-amendments)).
Schedule is optimistic with roll-over allowed ([milestones](project/milestones.md#revised-schedule)).
**Priority (Oct 10):** evaluate every existing engine (PyTorch, sd.cpp, edge-dit) completely and holistically on each device before any optimization, planner or TVM work. Gap campaign (A100 + Jetson, M5 with Codex) distilled coverage done Oct 14–15; Base cache/compile queues running Oct 15–16; TVM items below wait on it ([plan](../output/gap-campaign-20261010/PLAN.md)).

## Established evidence
- A100 distilled klein baselines (PyTorch, sd.cpp) and comparison; Jetson disk-backed Q4 512² baseline and profiles ([registry](experiments.md)).
- Base workload (50 steps, guidance 4) on A100 and Jetson; engine option sweeps with no-cache controls ([registry](experiments.md)).
- Jetson Base: EasyCache + exact conditioning reuse combined passes the campaign speed rule; no formal quality approval ([comparison](../experiments/compare-jetson-base-combined-reuse.md)).
- Jetson distilled: exact conditioning reuse passes the speed rule; EasyCache, prefetch-off and mmap fail it ([conditioning](../experiments/compare-jetson-distilled-conditioning-reuse.md), [EasyCache](../experiments/compare-jetson-distilled-easycache.md)).
- edge-dit full A100 baseline validated with stated limits ([record](../experiments/a100-edgedit-flux-klein-001.md)); Torch-TensorRT compile gate failed ([record](../experiments/a100-flux-klein-torchtrt-002.md)).

## Campaign records
- Completed and failed runs are indexed in the [registry](experiments.md). Jetson reopened for engine coverage (PyTorch, edge-dit, untested sd.cpp options); A100 Base conditioning full protocol recovered; different-GPU comparison caveat ([record](../experiments/a100-sdcpp-flux-klein-base-003.md)).
- Gap campaign, distilled (Oct 14–15): all three engines run on the Jetson; edge-dit fits only with VAE tiling and is fastest with weights resident; matched toggles transfer (fused attention off ≈ +30 %/step in sd.cpp and PyTorch; tiling adds decode time only) ([comparison](../experiments/compare-jetson-engines-matched-options.md)). sd.cpp all-resident + tiling ran 4 images at ~16 s then failed by <1 MB ([012](../experiments/jetson-flux-klein-012.md)); auto-fit fails the first text encode with or without tiling ([013](../experiments/jetson-flux-klein-013.md)).
- Jetson PyTorch: exact conditioning reuse removes the per-image reload (80.3 → 14.9 s, bit-identical); + torch.compile 7.8 s, output not bit-identical ([004](../experiments/jetson-torch-flux-klein-004.md), [007](../experiments/jetson-torch-flux-klein-007.md)). A100 torch 2.8: compile −21 %, reuse −2 % ([compile](../experiments/a100-flux-klein-compile-001.md)). At 4 steps most step caches skip nothing; MagCache/TaylorSeer speedups visibly degrade images; diffusers First-Block Cache unsupported for FLUX.2.
- Stage charts: [86 completed runs, twenty-six panels](../output/stage-breakdown-20261007/README.md); means stacked, total medians/ranges shown ([October log](log-2026-10.md)). Roofline panels/table include paper-style exports and responsive W&B plots with explicit assumptions ([instructions](../docs/roofline-plots.md), [verification](../output/roofline/wandb-verification.json)).
## Next (Week 2 → 3)
1. TVM sandbox: import a small randomly initialized FLUX-style transformer on CPU; inspect the IR, fusion and memory planning.
2. Week 3 gate: import klein's transformer step into TVM Relax and match PyTorch numerics on CUDA ([milestones](project/milestones.md#revised-schedule)).
3. M5: twelve earlier individual option protocols validated; [ten missing compile/cache/MFLUX-tiling options](../output/m5-followup-20261015/README.md) now smoke-gated sequentially: seven new full protocols complete; [Base MFLUX conditioning validated/uploaded](../experiments/m5-mflux-flux-klein-base-004.md), [4-bit tiling full invalidated by sleep](../experiments/m5-mflux-flux-klein-base-005.md); [8-bit smoke stopped on battery](../experiments/m5-mflux-flux-klein-base-006.md); both await AC-power recovery; [all 15 valid run uploads](../output/m5-followup-20261015/upload-recovery-verification.json) and [updated charts verified](../output/stage-breakdown-20261007/followup-upload-verification.json); bounded edge-dit investigation ended with a native distilled Metal CUSTOM-op failure and Base preparation blocker. Earlier queue stopped; all 35 retained run records uploaded to W&B, with six M5 panels in the [Klein breakdown](../output/stage-breakdown-20261007/README.md) ([procedure](methods/m5-howto.md), [registry](experiments.md)).
4. Review the draft planning hypothesis and three candidate contributions ([RQ3](rq/rq3.md)); review CacheQuant and Q&C before any novelty claim ([reading list](papers/reading-list.md)).
5. Approve or revise the TVM approach and engine-level planner fallback ([D-011, proposed](project/decisions.md#d-011-build-the-compiled-pipeline-mlc-style-on-tvm-with-an-engine-level-planner-fallback-proposed)).
6. Shared initial noise for engine comparisons remains proposed ([D-006](project/decisions.md#d-006-shared-initial-noise-from-sdcpps-philox-rng-proposed)).
Nunchaku: A100 11929464 smoke + full validated and uploaded to W&B; fixed-seed hashes differ and quality remains unassessed ([record](../experiments/a100-nunchaku-flux-klein-001.md)). Jetson/Metal remain blocked ([how-to](methods/nunchaku-howto.md)).
MFLUX M5: five full protocols validated and all twelve run records uploaded; [Klein charts](../output/stage-breakdown-20261007/README.md) include all completed precisions. Both 4-bit full samples have sign-text errors; [Base BF16](../experiments/m5-mflux-flux-klein-base-001.md) is blocked by a native MLX Metal crash. [Evidence verification](../output/mflux-prep/wandb-evidence-verification.json); swap/timing/quality caveats retained.
## Open
- Base (50 steps) cache/compile queues running: Jetson edge-dit + PyTorch, A100 job 11959986 (configs `*-base-*`); records pending. Codex's 2026-10-15 M5 timings may overlap local chart/W&B work on the same Mac; check before comparing.
- Next target after WebGPU; whether Jetson reload time is storage- or page-cache-bound; four earlier questions ([open questions](open-questions.md)).
- New Jetson findings: [step caching vs schedule length](findings/step-caching-helps-at-50-steps-but-not-at-4-on-jetson.md), [conditioning reuse](findings/exact-conditioning-reuse-removes-most-jetson-reload-time.md), [overlapping savings](findings/combined-reuse-savings-overlap-on-jetson-base.md); framing in [kernel, graph and plan search](concepts/kernel-graph-plan-search.md).
