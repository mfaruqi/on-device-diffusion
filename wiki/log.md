---
type: project
summary: Chronological log of experiments, decisions, ingests and lint passes (newest first, one-liners).
status: active
updated: 2026-10-06
---

# Log

Monthly entries: [October 2026 updates](log-2026-10.md); [September 2026](log-2026-09.md).

## [2026-10-06] ingest | Campaign findings, planner design and prior art
Five findings from the Jetson and campaign records ([step caching](findings/step-caching-helps-at-50-steps-but-not-at-4-on-jetson.md), [conditioning reuse](findings/exact-conditioning-reuse-removes-most-jetson-reload-time.md), [reloading](findings/weight-reloading-dominates-disk-backed-jetson-generations.md), [overlap](findings/combined-reuse-savings-overlap-on-jetson-base.md), [auto-fit](findings/sdcpp-auto-fit-fails-first-text-encoding-on-jetson.md)); [levels concept](concepts/kernel-graph-plan-search.md); [RQ3](rq/rq3.md) draft hypothesis and planner design; [RQ2](rq/rq2.md) update; [D-011 proposed](project/decisions.md#d-011-build-the-compiled-pipeline-mlc-style-on-tvm-with-an-engine-level-planner-fallback-proposed); one [open question](open-questions.md); [reading list](papers/reading-list.md) additions.

## [2026-10-06] schema | Log archived by month to keep log.md within budget
September entries moved unchanged to [log-2026-09.md](log-2026-09.md); [SCHEMA](SCHEMA.md#special-files) lists the `log-YYYY-MM.md` convention; the index includes archive months.

## [2026-10-06] decision | D-010: compiled klein pipeline on CUDA/Metal/WebGPU with the planner on top
[Advisor overview and confirmation](../raw/meetings/2026-10-06-advisor-scope.md) → [D-010](project/decisions.md#d-010-deliver-a-tvm-compiled-klein-pipeline-with-the-planner-on-top), [revised milestones](project/milestones.md) with roll-over, [overview amendment](project/overview.md#scope-amendments), [MacBook Air M5](systems/macbook-air-m5.md) replaces M1, Wan deferred; one [open question](open-questions.md) on the next target.

## [2026-10-06] experiment | Edge full baseline and Base conditioning gate validated
[Edge baseline](../experiments/a100-edgedit-flux-klein-001.md) completed1+3+10; [Base conditioning](../experiments/a100-sdcpp-flux-klein-base-003.md) preserved reference pixels with two prompt-cache hits. Full conditioning protocol11893094 queued.

## [2026-10-06] experiment | A100 repeated options validated
[Base comparison](../experiments/compare-a100-base-sdcpp-options.md) and [four-step comparison](../experiments/compare-a100-distilled-sdcpp-options.md) retain same-binary checks, timing ranges and image diagnostics. [edge adapter gate](../experiments/a100-edgedit-flux-klein-001.md) validated; full protocol11891242 submitted.

## [2026-10-06] experiment | A100 reuse gates completed; repeated protocols queued
[Base EasyCache](../experiments/a100-sdcpp-flux-klein-base-002.md) and four-step [control](../experiments/a100-sdcpp-flux-klein-001.md), [EasyCache](../experiments/a100-sdcpp-flux-klein-002.md), [conditioning](../experiments/a100-sdcpp-flux-klein-003.md), [prefetch](../experiments/a100-sdcpp-flux-klein-004.md), [mmap](../experiments/a100-sdcpp-flux-klein-005.md) gates validated. [edge-dit interface](../experiments/a100-edgedit-flux-klein-001.md) passed; [Nsight retry](../experiments/a100-sdcpp-flux-klein-base-001.md) failed before inference.

## [2026-10-05] experiment | Jetson Base mmap full protocol validated

[Base006](../experiments/jetson-flux-klein-base-006.md) completed; [comparison](../experiments/compare-jetson-base-mmap.md) retains identical pixels, overlapping timing ranges and failed speed-selection rule.

## [2026-10-05] experiment | jetson-flux-klein-base-006: mmap functional gate

[Record](../experiments/jetson-flux-klein-base-006.md): component mappings, CFG/callback checks and identical retained pixels validated; full repeated timing pending.

## [2026-10-05] experiment | Jetson Base combined reuse full protocol validated
[Base007](../experiments/jetson-flux-klein-base-007.md) complete; [matched comparison](../experiments/compare-jetson-base-combined-reuse.md) passes diagnostic speed selection against each individual policy, with EasyCache-only pixels.

## [2026-10-05] experiment | Base combined reuse correctness passed
[jetson-flux-klein-base-007](../experiments/jetson-flux-klein-base-007.md): exact conditioning and EasyCache both activate; functional/paired image diagnostics passed, full protocol running.

## [2026-10-05] experiment | Jetson four-step mmap full protocol
[008](../experiments/jetson-flux-klein-009.md) validated; [matched comparison](../experiments/compare-jetson-distilled-mmap.md) retains identical pixels and overlapping ranges, failing speed selection.

## [2026-10-05] experiment | Jetson four-step prefetch-disabled full protocol
[008](../experiments/jetson-flux-klein-008.md) validated; [matched comparison](../experiments/compare-jetson-distilled-prefetch.md) retains identical pixels and overlapping ranges, failing speed selection.

## [2026-10-05] experiment | Jetson four-step exact conditioning full protocol
[007](../experiments/jetson-flux-klein-007.md) validates exact prompt reuse; [comparison](../experiments/compare-jetson-distilled-conditioning-reuse.md) passes diagnostic speed/fidelity selection; no new combination scheduled.

## [2026-10-05] experiment | Jetson four-step EasyCache full protocol and comparison
[006](../experiments/jetson-flux-klein-006.md) validates enabled caching with zero skipped steps; [matched comparison](../experiments/compare-jetson-distilled-easycache.md) fails the declared speed-selection rule.

## [2026-10-05] experiment | Jetson distilled full no-cache control validated
[003 record](../experiments/jetson-flux-klein-003.md#cache-capable-harness-full-control-2026-10-05): matched harness reference completed with ten measured samples; timing spread and image equality preserved.

## [2026-10-05] experiment | jetson-flux-klein-009: distilled memory-mapped weight-file I/O correctness
[Record](../experiments/jetson-flux-klein-009.md); two-generation callback/settings/image validation passed; full protocol pending.

## [2026-10-05] experiment | jetson-flux-klein-008: distilled prefetch disabled correctness
[Record](../experiments/jetson-flux-klein-008.md); two-generation callback/settings/image validation passed; full protocol pending.

## [2026-10-05] experiment | jetson-flux-klein-007: distilled exact conditioning reuse correctness
[Record](../experiments/jetson-flux-klein-007.md); two-generation callback/settings/image validation passed; full protocol pending.

## [2026-10-05] experiment | jetson-flux-klein-006: distilled EasyCache correctness
[Record](../experiments/jetson-flux-klein-006.md); two-generation callback/settings/image validation passed; full protocol pending.

## [2026-10-05] experiment | jetson-flux-klein-003: distilled no-cache control on the cache-capable harness correctness
[Record](../experiments/jetson-flux-klein-003.md); two-generation callback/settings/image validation passed; full protocol pending.

## [2026-10-05] experiment | Jetson Base conditioning full protocol validated
[Base005](../experiments/jetson-flux-klein-base-005.md) retains identical pixels; [matched comparison](../experiments/compare-jetson-base-conditioning-reuse.md) passes the declared speed/fidelity screen.

## [2026-10-05] experiment | Jetson Base exact conditioning reuse correctness passed
[Base005](../experiments/jetson-flux-klein-base-005.md) confirms both CFG conditioning hits and identical output pixels; full protocol started separately.

## [2026-10-05] experiment | Jetson Base prefetch protocol complete
[Base004](../experiments/jetson-flux-klein-base-004.md) validated; [matched comparison](../experiments/compare-jetson-base-prefetch.md) preserves identical pixels but fails the speed-range selection rule.

## [2026-10-05] experiment | BF16 compiler screen ends; A100 cache control passes
[Torch-TensorRT002](../experiments/a100-flux-klein-torchtrt-002.md) failed scalar-multiply conversion after value-preserving layout copies. [sd.cpp Base001](../experiments/a100-sdcpp-flux-klein-base-001.md) no-cache control passed on the cache-capable harness.

## [2026-10-05] experiment | A100 Base engine comparison
[Comparison](../experiments/compare-a100-base-pytorch-vs-sdcpp.md) records repeated unprofiled timings and device-wide memory; unmatched initial noise, native schedules and stage timing semantics limit causal/quality claims.

## [2026-10-05] experiment | A100 sd.cpp Base baseline validated; profile startup failed
[Base001](../experiments/a100-sdcpp-flux-klein-base-001.md) completed1+3+10 with CFG/settings/image checks. Nsight failure preserved; one unchanged-workload retry queued on a different PCIe node.

## [2026-10-05] experiment | A100 PyTorch Base profile validated
[Base001](../experiments/a100-flux-klein-base-001.md) completed the separate torch.profiler capture; both CFG windows per step and derived kernel shares validated.

## [2026-10-05] experiment | A100 PyTorch Base full baseline validated
[Base001](../experiments/a100-flux-klein-base-001.md) completed1+3+10 with CFG/stage/hash/aggregate checks; separate profile started.

## [2026-10-05] experiment | A100 engine gates and Jetson prefetch correctness
[edge-dit](../experiments/a100-edgedit-flux-klein-001.md) generated an image; [Torch-TensorRT](../experiments/a100-flux-klein-torchtrt-001.md) failed input-layout preparation. [Jetson Base004](../experiments/jetson-flux-klein-base-004.md) correctness passed; full protocol running.

## [2026-10-05] experiment | Jetson Base lazy-loading completed
[Base003](../experiments/jetson-flux-klein-base-003.md) attempt and full protocol validated; [matched comparison](../experiments/compare-jetson-base-eager-vs-lazy.md) records identical pixels, overlapping timing ranges and no speed-selected combination.

## [2026-10-05] experiment | Jetson same-binary reference repeat
[Base reference](../experiments/jetson-flux-klein-base-001.md) repeated and validated on the cache-capable harness; [EasyCache comparison](../experiments/compare-jetson-base-easycache.md) now includes matched binary/protocol evidence. Diagnostic selection remains separate from formal quality eligibility.

## [2026-10-05] experiment | Base profile and sd.cpp correctness validated
[Jetson Base](../experiments/jetson-flux-klein-base-001.md): targeted profile validated and read/GPU coverage derived. [A100 sd.cpp Base](../experiments/a100-sdcpp-flux-klein-base-001.md): two-image correctness gate passed; full baseline and separate profile queued.

## [2026-10-05] experiment | A100 PyTorch Base correctness

[Record](../experiments/a100-flux-klein-base-001.md): pinned BF16 fifty-step guidance-four attempt passed; full reference and separate profile queued. No repeated baseline result yet.

## [2026-10-05] experiment | Jetson Base EasyCache correctness and diagnostic comparison

[Record](../experiments/jetson-flux-klein-base-002.md): two-generation attempt and repeated cache protocol validated. [Comparison](../experiments/compare-jetson-base-easycache.md) retains paired PSNR/LPIPS diagnostics, repeated results and harness-matching limits.

## [2026-10-05] experiment | Jetson klein Base correctness

[Record](../experiments/jetson-flux-klein-base-001.md): full fifty-step guidance-four no-cache1+3+10 reference completed; correctness attempt and cache-capable harness no-cache control retained separately.

## [2026-10-05] experiment | Jetson eager versus lazy loading

[Comparison](../experiments/compare-jetson-eager-vs-lazy-flux-klein.md): overlapping timing ranges and unmatched capture conditions; descriptive evidence only.

## [2026-10-05] experiment | Jetson lazy-loading correctness passed
[Record](../experiments/jetson-flux-klein-005.md): full unprofiled 1+3+10 protocol complete, with matching RGB hashes across fourteen outputs. An accidental terminal interruption is retained as a distinct attempt.

## [2026-10-05] experiment | Jetson automatic placement attempt failed
[Record](../experiments/jetson-flux-klein-004.md): engine selected CPU/GPU residency; first text encoding failed during segment weight preparation. No completed image or baseline median.

## [2026-10-05] experiment | Jetson captured read/GPU coverage
[Existing record](../experiments/jetson-flux-klein-003.md#process-restricted-read-call-analysis-2026-10-05): three saved captures reanalyzed with process filtering and interval unions; the CLI capture lacks stage markers. Baseline numbers unchanged.

## [2026-10-04] lint | 1 fix, 0 open
[Jetson how-to](methods/jetson-howto.md) no longer says the refactored capture is untested on hardware; the [record](../experiments/jetson-flux-klein-003.md) shows Sep 30 runs. Lint skill scope now includes uncommitted pages, and stale-status checks cover method pages.

## [2026-10-02] lint | Status and literature links refreshed
[Milestones](project/milestones.md), [RQ1](rq/rq1.md), [Jetson](systems/jetson-orin-nano.md), [DreamLite](systems/dreamlite-mobile.md) and [reading list](papers/reading-list.md) updated from existing records/reviews; [meeting memo](../raw/meetings/2026-10-01.md) wording retained with GitHub/W&B links.

## [2026-10-02] ingest | October 1 meeting priorities and engine candidates
[Meeting memo](../raw/meetings/2026-10-01.md) → [D-009](project/decisions.md#d-009-prioritize-image-engine-baselines-and-profiling), [next actions](hot.md), and [reading list](papers/reading-list.md): LightX2V/TensorRT screening candidates, engine-option audits and temporary FastVideo/DreamLite deferral; no new benchmark findings.

## [2026-10-01] ingest | Week 1 evidence and status reconciliation review
[Jetson baseline/profiles](../experiments/jetson-flux-klein-003.md) already registered; [DreamLite source review](papers/dreamlite.md) exists with export/device validation pending. Milestone, hot, RQ1 and DreamLite reading-list corrections prepared; substantive updates await approval.
