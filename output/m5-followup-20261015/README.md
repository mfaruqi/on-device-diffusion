# M5 individual follow-up tests — 2026-10-15

Week 2–3 engine coverage supporting RQ1. Existing completed configurations are retained; this batch tests only missing individual options. Claude's Jetson/A100 work is separate.

| Engine | Option | Distilled | Base | Reference |
|---|---|---|---|---|
| PyTorch/Diffusers MPS BF16 | Transformer-only Inductor `torch.compile`, fullgraph, default mode | 004 | base-004 | Same-engine BF16 001 |
| PyTorch/Diffusers MPS BF16 | Upstream TaylorSeer attention-output prediction, interval 5, three initial compute steps, first-order | 005 | base-005 | Same-engine BF16 001 |
| MFLUX / MLX 4-bit | Exact prompt-pair reuse | 004 | base-004 | Same-engine 4-bit 003 |
| MFLUX / MLX 4-bit | VAE tiling, 256 px, 64 px overlap | 005 | base-005 | Same-engine 4-bit 003 |
| MFLUX / MLX 8-bit | VAE tiling, 256 px, 64 px overlap | 006 | base-006 | Same-engine 8-bit 002 |

All configurations preserve pinned checkpoints, 512², CFG1/four steps distilled and CFG4/50 steps Base. Quantization is held fixed within each MFLUX comparison. No combinations or automatic fitting/offload. edge-dit remains blocked at its uncached Metal CUSTOM operation; a cache test cannot bypass that baseline. Existing sd.cpp EasyCache/conditioning/prefetch/mmap and PyTorch conditioning/tiling protocols are not repeated.

## Execution and validation

`state.json` is the queue manifest and attempt ledger. `run_queue.py` uses existing runners sequentially, with the shared M5 continuation lock. Actual process identities and advancing output must agree with state before recovery. Detached queue PID is recorded in state; stale PIDs alone are not evidence. The initial shell-background launch did not survive; the detached process launch did. No duplicate inference was started.

Smoke: first + one measured. Full: first + three discarded warmups + ten measured. First-image timings are separate. Compile smoke additionally compares the first transformer call against eager with identical inputs and BF16 weights (relative RMSE ≤1%); its first image includes an extra eager call and is diagnostic, not ordinary first-image latency. The full protocol omits that diagnostic. Compiler receipts reject graph breaks and eager error fallback.

TaylorSeer audits attention prediction counts, resets per generation, and separates conditional/unconditional histories. This is neither EasyCache nor whole-transformer step skipping. MFLUX prompt-pair reuse retains text-encoder weights and times cache lookup; exact same-engine image identity gates full execution. MFLUX VAE audits actual tile decodes; tile-local GroupNorm can change image colors. Neither approximate method implies formal quality approval.

System swap and available memory are recorded. The first compilation smoke increased swap substantially; background applications are not isolated. The queue aborts its own run if available RAM stays below 0.5 GiB for 15 seconds, or after 20 minutes smoke / 90 minutes full. It never kills user applications or changes precision/residency to fit. Preserve all failed and partial runs. Low available memory is a conservative stop condition, not proof of GPU OOM.

## Publishing and continuation

Validated completed runs are documented and exported between timed protocols, with duplicate checks and W&B verification receipts. Failures remain local evidence. Full protocols are eligible for addition to the existing Klein breakdown only after validation; preserve prior panels and label memory-pressure, precision, timing and quality limits. Chart additions require the monitor's final review; they are not performed inside the timed runner.

Heartbeat `monitor-m5-compile-cache-and-tiling-tests` checks every 30 minutes and finishes validation, uploads, charts and bookkeeping. No commit or push. It must reconcile interrupted entries rather than blindly restart. It deletes itself when this bounded list and publishing are finished or documented blocked.

## Final record review

The reused documentation helper needs a scoped editorial pass after each entry: remove unrelated engine/option descriptions from its generic “Reference and option audit” paragraph, distinguish completed execution from failed validation, and update any index row initially added while running. The first compile record has already been corrected. Keep compiler numerical diagnostics distinct from quality approval. Preserve the monthly log budget while retaining one entry per experiment. No new durable finding is inferred solely from this uncontrolled batch.

## 18:00 UTC review

Four distilled full protocols passed execution/protocol review; Base compilation passed its smoke numerical gate and is active. See `review-20261015T1800.json`. All eight attempted uploads failed before network export because the shared exporter defaulted to Gilbreth scratch. `publish.py` now sets Mac-local W&B directories; its local export dry-run passes. Retry these eight after timed inference ends; do not mark them uploaded yet. The running queue automatically uses the repaired publisher for later entries.

`stage-manifest-pending.json` contains the four validated new full runs while preserving all fifteen existing panels, plus a MFLUX options panel. It is a draft: merge against the latest canonical manifest after remaining results arrive, rather than replacing newer chart edits. No chart upload has run during measurement.

## 19:00 UTC review

Base compile and TaylorSeer completed and passed execution/protocol review; see `review-20261015T1900.json`. The repaired uploader verified Base compile smoke and full receipts; TaylorSeer publishing follows automatically before Base MFLUX begins. Three Base MFLUX configurations remain. The pending chart manifest now contains six new full protocols; no charts have been published from this batch. Eight distilled exports from the initial path failure still require idle-time retry. Experiment notes were narrowed to their own options and running index rows updated; monthly log separator spacing was compacted without removing entries.

## 20:00 UTC observation

Base 4-bit tiling is advancing with nine tile decodes per completed image. An external W&B metadata update from the other worktree was observed during inference; see `monitor-20261015T2000.json` and the run’s `external-activity-observation.json`. Do not attribute latency changes to tiling alone. The queue’s generic final record writer may overwrite the interim observation paragraph, so retain this confound in the final experiment note and chart caveats. No external process was terminated and no upload was launched by this monitor.

## 20:31 UTC interruption and power gate

The Base 4-bit tiling full attempt was invalidated by lid-closed sleep (1299.187-second wall/monotonic gap); its partial outputs and power evidence are retained. Its earlier smoke remains valid and uploaded. The following Base 8-bit tiling smoke stopped before inference because AC power was absent. The queue has exited and must not be relaunched while on battery. See `recovery-needed.json`: after verifying AC power, open lid and no competing inference, set only Base 4-bit tiling to `smoke_passed` and Base 8-bit tiling to `pending`, preserving all run paths; restart the existing sequential queue. Do not repeat other completed configurations. Final uploads/charts can proceed during this idle period.

## Idle-time publishing

The seven valid new full protocols are now in the existing Klein analysis run, with 56 total source runs and 17 panels. W&B latest step is 8. `output/stage-breakdown-20261007/followup-upload-verification.json` verifies all prior panels, the four updated/new option plot payloads and the exact analysis manifest. Excluded sleep/preflight failures remain excluded. All 15 validated smoke/full exports are verified; see `upload-recovery-verification.json`. AC power briefly returned during publishing but reverted to battery at the recovery launch preflight; the guard prevented any new benchmark launch. Both remaining tests still await stable AC power.

## 21:00 UTC power verification

No benchmark or upload process is active. Host power inspection reports battery power, 96%, with `AppleSmartBattery ExternalConnected=No` and no active adapter. A sandboxed power query reports AC inconsistently; use host-level power and adapter checks before recovery. The earlier apparent AC reports do not establish that an adapter was connected or fluctuating. Both unfinished configurations remain waiting; all existing upload/chart receipts are retained.

## User-requested pause after power recovery

Host AC adapter and open lid were confirmed; recovery queue 21805 started Base 4-bit full attempt `20261015-214940`. It was explicitly stopped during first image step 17/50: coordinator terminated first, child interrupted, raw failure preserved. Base 8-bit remains pending. Heartbeat is PAUSED. Do not resume without an agreed time or explicit ready message; then reuse the valid 4-bit smoke and run only the two unfinished configurations.

## Authorized resume, October 15 at 22:59 Indianapolis

Explicit resume received. Host AC adapter and open lid verified; no competing inference/upload. Queue 23947 is active, with Base 4-bit full recovery `m5-mflux-flux-klein-base-005__baseline__20261015-225903` advancing and Base 8-bit smoke/gated full next. The 30-minute heartbeat is ACTIVE again. Earlier pause instructions are superseded; retain all prior interruptions and reuse the valid 4-bit smoke. After completion, review generated notes for option scope, preserve interruption causes, correct any running index rows, and publish only new validated results/chart additions.

## 03:29 UTC review, October 16

Base 4-bit tiling recovery completed, validated and uploaded; its experiment record now preserves both prior interruption causes and the reviewed sign-text quality limitation. Base 8-bit recovery smoke passed and full `20261015-232911` is advancing on host-verified AC power under queue 23947 / child 25250. See `review-20261016T0329.json`. Eight full configurations are complete, distilled compile remains gate-blocked, and one full is active. No additional uploads or chart rendering started during inference. The last two full chart additions remain pending; preserve all 56 current sources and 17 panels when extending the canonical manifest. Do not overwrite live queue state from this monitor.

## Completion, October 16 at 00:04 Indianapolis

All ten authorized entries are terminal: nine validated full protocols and the preserved distilled compile numerical-gate failure. Base tiling recoveries measured 111.959 seconds/image (4-bit) and 128.976 seconds/image (8-bit); ten measured images each, first/warm-ups excluded. All 18 successful smoke/full exports are verified (`final-upload-verification.json`). The existing Klein analysis now includes 58 source runs and all 17 panels; latest step 12, analysis artifact v12, exact manifest and updated Base MFLUX plot verified. Both recovered tiling runs preserve clock continuity; tile-local normalization and uncontrolled swap/background conditions prevent formal quality or isolated speedup claims. Prior failures/interruptions remain local evidence and are excluded from bars. No inference remains active; no new runs, commits or pushes are scheduled. The completed heartbeat was deleted after final verification.
