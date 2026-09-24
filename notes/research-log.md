# Research log

Dated entries, newest first. Each experiment gets its own note, and this log links to it.

## 2026-09-23: A100 FLUX.2 klein baseline results

- Baseline done ([a100-flux-klein-001.md](a100-flux-klein-001.md)): **1240.6 ms** end-to-end
  (0.9% spread, bit-identical outputs), with denoise 79%, VAE 13% and text encode 4%.
  Peak allocated memory is **17.3 GiB**, set by the VAE decode. Weights are 14.9 GiB, half of that the
  text encoder.
- Profiler: GEMMs are 44% of GPU time, flash attention 13%, and about a third is memory-bound
  elementwise/norm/copy ops.
- Leads: text-encoder residency, the VAE peak, and fusion of the non-GEMM ops. None of it fits the
  8 GB Jetson at BF16.

## 2026-09-23: A100 FLUX.2 klein baseline kit

- Wrote the first experiment kit: `configs/a100-flux-klein-bf16.json`, `scripts/run_flux.py`,
  `scripts/gilbreth.slurm`. Metric definitions are in [baseline-metrics.md](baseline-metrics.md),
  cluster usage in [gilbreth-howto.md](gilbreth-howto.md), and the experiment note in
  [a100-flux-klein-001.md](a100-flux-klein-001.md).
- Earlier exploratory scripts in `flux_test/` used **FLUX.1-schnell** at 512², not the proposal
  target. Keep them only as a sanity reference, not a baseline. `flux_test/flux_trace.json`
  (110 MB, A100-PCIE-40GB) is git-ignored.
- Next: `--prepare-only` download, then submit the baseline and profile jobs.
