---
type: project
summary: Chronological log of experiments, decisions, ingests and lint passes (newest first, one-liners).
status: active
updated: 2026-09-28
---

# Log

Format: `## [YYYY-MM-DD] kind | title`, with at most five lines under each entry. Details live in the
linked record or page. List the last five with `grep "^## \[" wiki/log.md | head -5`.

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
