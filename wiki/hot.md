---
type: project
summary: Semester focus is ShiftAdd allocation, existing-engine packed kernels and measured memory/latency/energy; Nunchaku feasibility remains open.
status: active
updated: 2026-11-04
---

# Hot: current state

**Week 6 (Nov 2–6).** [D-012](project/decisions.md#d-012-prioritize-shiftadd-experiments-and-an-existing-engine-kernel-prototype-this-semester) prioritizes completing ShiftAdd experiments and an existing-engine packed kernel prototype by semester end; Nunchaku is the first candidate to audit. The broader TVM/planner delivery is deferred from semester acceptance, not completed. Proposal hypothesis and RQs remain unchanged ([overview](project/overview.md)); advisor confirmation of broader thesis scope remains open.
See the [meeting memo](../raw/meetings/2026-11-04.md) and [weekly gates](project/milestones.md#semester-completion-plan--2026-11-04).

## Consolidated evidence
- Claude's committed A100/Jetson campaign and local M5 recovery records now share this checkout; source commit `cf48eb9` and reconciliation scope are recorded in the [stage-chart documentation](../output/stage-breakdown-20261007/README.md).
- The canonical [stage manifest](../output/stage-breakdown-20261007/manifest.json) contains 98 full runs and 28 panels; [W&B step 14 verified](../output/stage-breakdown-20261007/reconciliation-verification.json). Failed, interrupted, smoke and profiled captures are separate evidence in the [registry](experiments.md).
- Jetson PyTorch eager/compiled CUDA kernel traces are available; compilation chiefly reduces GGUF unpack/rescale/cast work ([eager](../experiments/jetson-torch-flux-klein-004.md#kernel-profile-2026-10-1718), [compiled](../experiments/jetson-torch-flux-klein-007.md#kernel-profile-2026-10-1718)).
- A100 and Jetson Base engine/cache/compile results are documented ([A100 compile](../experiments/a100-flux-klein-base-torch28-004.md), [Jetson](../experiments/compare-jetson-base-caches-reuse-compile.md)). Approximate output changes are not formal quality approval.
- M5 follow-up: nine validated full configurations, distilled compile numerical gate blocked, final tiling recoveries complete; no active queue ([campaign](../output/m5-followup-20261015/README.md)). Pressure confounds remain ([review](../experiments/compare-accelerator-optimization-review-20261016.md)).
- MFLUX initial Base BF16 failed native Metal execution; M5 edge-dit baseline remains blocked ([MFLUX](../experiments/m5-mflux-flux-klein-base-001.md), [M5 procedure](methods/m5-howto.md)).
- Nunchaku runs on the Jetson from an sm_87 source build (two allowlist lines patched); the layer check matches the A100, and denoise is 5.0× faster than the GGUF Q4_0 transformer; with conditioning reuse an image takes 3.44 s, 4.3× faster than GGUF with reuse ([record](../experiments/jetson-nunchaku-flux-klein-001.md), [reuse](../experiments/jetson-nunchaku-flux-klein-002.md), [compare](../experiments/compare-jetson-nunchaku-vs-gguf-pytorch.md)). M5 Metal remains unsupported ([procedure](methods/nunchaku-howto.md)).

- ShiftAdd: the [complete upstream/BF16 package](../experiments/m5-flux-klein-shiftadd-020.md) covers all 50 attention/fused matrices, with the other 59 matrix weights retained in BF16. [Image controls](../experiments/a100-flux-klein-upstream-all-attention-001.md) and [A8/A4 diagnostics](../experiments/a100-flux-klein-upstream-all-attention-activation-001.md) are verified. A [fit-selected mixed candidate](../experiments/m5-flux-klein-shiftadd-022.md) has a completed [image comparison](../experiments/a100-flux-klein-upstream-mixed-fused-001.md): coherent scenes but greater BF16 latent distance on all four calibration prompts. [CPU loading-cache measurements](../experiments/m5-flux-klein-shiftadd-load-002.md) are separate from inference; no packed GPU latency, runtime-memory or energy advantage is established. A [one/two-plane follow-up](../experiments/a100-flux-klein-upstream-lowplane-fused-001.md) is complete: two-plane scenes remain recognizable; one plane causes semantic failures. [Size charts](../experiments/compare-shiftadd-plane-storage.md) separate stored weights from dense runtime payload. See the [campaign guide](../shiftadd/docs/current-campaign.md) and [conversion walkthrough](../shiftadd/docs/conversion-walkthrough.md); support modules are grouped into diagnostics, validation, plots and reporting.

## Next
1. Build transformer/text-encoder/VAE memory accounting: stored and resident weights, activations, workspaces, metadata, correction branches, caches and loading transitions; compare declared BF16/INT4/SVDQuant controls ([memo](../raw/meetings/2026-11-04.md)).
2. Audit Nunchaku layouts and architecture support; implement and validate one packed projection before expanding scope ([kernel gates](../raw/meetings/2026-11-04.md)).
3. Validate 2/3/4/5/6-plane support and re-fit allocation sensitivity on calibration data; freeze before Week 10 held-out evaluation ([quality plan](../raw/meetings/2026-11-04.md)).
4. Locate `shiftaddllm_energy.py`; separate analytical estimates from timestamped Jetson power integration and measured joules/image ([energy plan](../raw/meetings/2026-11-04.md)).

## Open
- Formal matched-quality and cross-device transfer conclusions remain limited by checkpoint, RNG, residency and timing differences ([RQ1](rq/rq1.md), [evaluation rules](project/overview.md#evaluation-rules)).
- Shared initial noise remains proposed ([D-006](project/decisions.md#d-006-shared-initial-noise-from-sdcpps-philox-rng-proposed)); unresolved questions remain in [open questions](open-questions.md).
