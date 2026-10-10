# Klein stage breakdowns — 2026-10-07

[Open the seven interactive charts in W&B](https://wandb.ai/mfaruqi-purdue-university/on-device-diffusion/runs/c6af35f4845e4833).

Week 2 baseline analysis, RQ1. Twenty-seven completed, unprofiled runs; one first generation and three warm-ups excluded, ten measured generations per run. Source selection and labels: [manifest.json](manifest.json). Numeric data, source-file SHA256 hashes and full configurations: [chart-data.json](chart-data.json). Upload verification: [wandb-verification.json](wandb-verification.json).

| Panel | Sources |
|---|---|
| A100 distilled sd.cpp options | No reuse, EasyCache, conditioning reuse, prefetch off, mmap |
| A100 Base sd.cpp options | No reuse, EasyCache, conditioning reuse (different physical GPU) |
| Jetson distilled sd.cpp options | Original and current references, lazy loading, EasyCache, conditioning reuse, prefetch off, mmap |
| Jetson Base sd.cpp options | Initial and repeat references, EasyCache, lazy loading, prefetch off, conditioning reuse, mmap, EasyCache + conditioning |
| A100 distilled engine references | PyTorch/Diffusers and sd.cpp |
| A100 Base engine references | PyTorch/Diffusers and sd.cpp |
| A100 edge-dit phases | Encoding + setup, denoising, decoding, remaining time |

## Reading the charts

Stack heights use arithmetic means so the components sum to the mean wall time. The remaining component is computed for each generation as wall time minus the three named stages, then averaged. Diamonds mark total medians; whiskers span observed total min–max (not confidence intervals). Stage values are not independent medians. Hover to identify source run, workload and duration; click legend entries to inspect smaller components.

Text conditioning includes retrieval/setup when cached. sd.cpp stages include loading within callback boundaries; PyTorch stages use CUDA events. Edge-dit's host-timed encoding phase includes latent/schedule setup, so it has a distinct label and panel. Initial model loading is excluded. Device panels have independent scales and different precision/resolution/residency. Cross-engine quality equivalence and formal EasyCache quality eligibility are not established.

The A100 Base conditioning benchmark was recovered from its W&B artifact and its CSV/summary arithmetic rechecked. Its physical GPU differs from the reference: the small latency difference is descriptive, not an isolated cache effect. Prior validation and quality receipts are retained in its run directory. Smoke tests, failed/incomplete runs, profiler captures and older duplicate captures are excluded.

## W&B display

Interactive panels use the W&B panel heading, a responsive plot, compact x-axis labels and no embedded footer or duplicate title. Full source/workload labels remain in hover text. The `figures/` panels are image versions with fixed typography, complete titles, total annotations and methodology notes; use these for clean side-by-side viewing and screenshots. Refresh the run and choose the latest Step after an update.

## Reproduce

Dependencies: `plotly`, `wandb` for upload and `pillow` for image panels; optional `kaleido` and Chrome for PNG/PDF exports. Versions used are recorded in the W&B analysis configuration.

```sh
python scripts/plot_stage_breakdown.py \
  --manifest output/stage-breakdown-20261007/manifest.json \
  --output output/stage-breakdown-20261007 \
  --name klein-stage-breakdown-20261007 \
  --static --upload
```

The output directory retains an upload ID so repeat uploads resume the same analysis run. PNG, PDF and standalone interactive HTML files are available locally and in its W&B analysis artifact. Generated renders and W&B runtime files are ignored by Git; the manifest and numeric export are retained. No model inference was run for these charts.
