---
type: project
summary: Chronological log of experiments, decisions, ingests and lint passes (newest first, one-liners).
status: active
updated: 2026-09-28
---

# Log

## [2026-09-28] schema | Uniform run names; W&B re-exported with outcome labels
Run directories renamed to `<experiment-id>__<kind>__<stamp>` ([naming](methods/baseline-metrics.md#run-naming),
[rename table](../results/README.md#run-directory-names)). All 10 runs re-exported; W&B names show
`· OK`/`· FAILED` and failed runs use W&B's Failed state.

## [2026-09-28] ingest | Jetson successful run: full CLI logs analyzed
[Record](../experiments/jetson-flux-klein-003.md): load/stage diagnostics and whole-window sampled RAM extracted;
command audited, eager CUDA loading followed by releases/reloads documented. Repeated timing/profile pending.

## [2026-09-28] experiment | jetson-flux-klein-003: disk-backed generation completed
[Record](../experiments/jetson-flux-klein-003.md): one image reported saved, exit 0.
Partial logs preserved; imported image passed visual smoke check (hash recorded). Full logs and benchmark pending.

## [2026-09-28] experiment | jetson-flux-klein-002: segmented feasibility failed
[Record](../experiments/jetson-flux-klein-002.md): Qwen3 segment capacity failure with CUDA-resident weights.
Partial terminal evidence preserved; disk-backed parameter variant prepared, not run.

## [2026-09-28] experiment | jetson-flux-klein-001: quantized feasibility failed
[Record](../experiments/jetson-flux-klein-001.md): insufficient memory during denoising preparation.
Preserved terminal excerpts and reconstructed config/status; full Jetson logs pending import.

Format: `## [YYYY-MM-DD] kind | title`, with at most five lines under each entry. Details live in the
linked record or page. List the last five with `grep "^## \[" wiki/log.md | head -5`.

## [2026-09-28] ingest | Jetson setup and SSH access
[Setup evidence](../raw/jetson-setup-2026-09-28.md): boot, Mac SSH login, and CUDA compiler verified.
Added [access/how-to](methods/jetson-howto.md); updated device and current state. GPU execution and benchmark pending.

## [2026-09-28] experiment | Remaining 6 runs exported to W&B
All 7 run directories are in the W&B project; medians verified against `summary.json`. Registry links
point to each experiment's W&B group.

## [2026-09-28] decision | First run exported to W&B (viewer pilot)
[a100-flux-klein-001 baseline](https://wandb.ai/mfaruqi-purdue-university/on-device-diffusion/groups/a100-flux-klein-001) via `scripts/export_wandb.py`; values match `summary.json`. Run
directories stay the record ([D-007](project/decisions.md#d-007-results-stay-in-run-directories-viewer-undecided)).

## [2026-09-28] ingest | ExecuTorch (MLSys 2026)
Paper page [executorch-mlsys2026](papers/executorch-mlsys2026.md); related-evidence links on three findings; one open question.

## [2026-09-28] schema | Wiki created from notes/
Split `notes/` into `experiments/` (records) and `wiki/` (curated). See [SCHEMA.md](SCHEMA.md). Seeded
project, RQ, finding, system and concept pages from the A100 work.

## [2026-09-26] decision | Shared initial noise from sd.cpp's Philox RNG (proposed)
[D-006](project/decisions.md#d-006-shared-initial-noise-from-sdcpps-philox-rng-proposed). Not implemented yet.

## [2026-09-25] experiment | a100-sdcpp-flux-klein-001: stable-diffusion.cpp baseline + profile
2423.3 ms end-to-end on A100-PCIE, 1.95× the PyTorch reference; record:
[a100-sdcpp-flux-klein-001](../experiments/a100-sdcpp-flux-klein-001.md),
comparison: [compare-a100-pytorch-vs-sdcpp-flux-klein](../experiments/compare-a100-pytorch-vs-sdcpp-flux-klein.md).

## [2026-09-25] decision | Pin Slurm jobs to A100-PCIE nodes
[D-002](project/decisions.md#d-002-pin-gilbreth-jobs-to-a100-pcie-nodes).

## [2026-09-25] experiment | a100-flux-klein-001 repeated after runner refactor
Same image hash, memory identical, GPU stages within 0.9%; record:
[a100-flux-klein-001](../experiments/a100-flux-klein-001.md#repeatability).

## [2026-09-24] experiment | a100-flux-klein-001: kernel mix inside one denoise step
Record: [a100-flux-klein-001](../experiments/a100-flux-klein-001.md#kernel-mix-within-one-denoise-step-profiled-generation-job-11807396).

## [2026-09-23] experiment | a100-flux-klein-001: PyTorch/diffusers reference baseline
1240.6 ms end-to-end on A100-PCIE; record: [a100-flux-klein-001](../experiments/a100-flux-klein-001.md).
