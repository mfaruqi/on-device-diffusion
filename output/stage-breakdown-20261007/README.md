# Klein stage breakdowns — updated 2026-10-15

[Open the seventeen interactive charts in W&B](https://wandb.ai/mfaruqi-purdue-university/on-device-diffusion/runs/c6af35f4845e4833).

Week 2–3 engine and individual-option analysis, RQ1. Fifty-six completed, unprofiled runs; one first generation and three warm-ups excluded, ten measured generations per run. Source selection and labels: [manifest.json](manifest.json). Numeric data, source-file SHA256 hashes and full configurations: [chart-data.json](chart-data.json). Earlier MFLUX chart verification: [mflux-upload-verification.json](mflux-upload-verification.json); prior verification remains [wandb-verification.json](wandb-verification.json).

| Panel | Sources |
|---|---|
| A100 distilled sd.cpp options | No reuse, EasyCache, conditioning reuse, prefetch off, mmap |
| A100 Base sd.cpp options | No reuse, EasyCache, conditioning reuse (different physical GPU) |
| Jetson distilled sd.cpp options | Original and current references, lazy loading, EasyCache, conditioning reuse, prefetch off, mmap |
| Jetson Base sd.cpp options | Initial and repeat references, EasyCache, lazy loading, prefetch off, conditioning reuse, mmap, EasyCache + conditioning |
| A100 distilled engine configurations | PyTorch/Diffusers BF16, sd.cpp BF16 and Nunchaku INT4/r32 |
| A100 Base engine references | PyTorch/Diffusers and sd.cpp |
| A100 edge-dit phases | Encoding + setup, denoising, decoding, remaining time |
| M5 distilled / Base sd.cpp options (two panels) | Fresh mixed-Q4 reference, EasyCache, conditioning reuse, prefetch off, mmap |
| M5 distilled / Base PyTorch options (two panels) | BF16 reference, conditioning reuse, VAE tiling 256 pixels / 25% overlap, TaylorSeer; Base also fullgraph torch.compile |
| M5 distilled / Base engine configurations (two panels) | PyTorch/Diffusers BF16, sd.cpp mixed Q4 and MFLUX BF16/8-bit/4-bit where completed; precision, timing and RNG differ |
| M5 distilled / Base MFLUX precision (two panels) | Distilled BF16, 8-bit, 4-bit; Base 8-bit and 4-bit (BF16 smoke crashed) |

| M5 distilled / Base MFLUX options (two panels) | Same-bit references versus exact conditioning reuse; distilled also 4-bit and 8-bit VAE tiling |

## October 15 additions

Seven validated full protocols were added, preserving all 49 earlier sources and 15 panels. [Remote verification](followup-upload-verification.json) checks the latest four modified/new option panels, all plot/image paths and the analysis artifact manifest. Distilled compilation failed its numerical gate; Base 4-bit tiling was interrupted by clamshell sleep and Base 8-bit tiling stopped at AC-power preflight. These attempts are excluded pending recovery.

TaylorSeer predicts attention outputs with separate CFG histories; it is approximate and has no formal quality approval. Compile smoke uses an extra identical-input eager check; only the subsequent full protocol appears here. System swap, thermals and background load were uncontrolled; differences are descriptive, not isolated optimization speedups. Exact conditioning reuse applies only to repeated identical prompt pairs. Tiling changes pixels through local normalization.

## Reading the charts

Stack heights use arithmetic means so the components sum to the mean wall time. The remaining component is computed for each generation as wall time minus the three named stages, then averaged. Diamonds mark total medians; whiskers span observed total min–max (not confidence intervals). Stage values are not independent medians. Hover to identify source run, workload and duration; click legend entries to inspect smaller components.

Text conditioning includes retrieval/setup when cached. sd.cpp stages include loading within callback boundaries; PyTorch stages use CUDA events on A100 and MPS events on M5. Edge-dit's host-timed encoding phase includes latent/schedule setup, so it has a distinct label and panel. Initial model loading is excluded. Device panels have independent scales and different precision/resolution/residency. Cross-engine quality equivalence and formal EasyCache quality eligibility are not established.

The A100 distilled engine panel includes the completed Nunchaku SVDQuant W4A4/r32 replacement transformer with BF16 text encoder/VAE. Its software environment, checkpoint precision and physical node differ from the BF16 references; checkpoint ancestry and quality equivalence remain unestablished. This is a descriptive configuration comparison, not an isolated engine speedup. Nunchaku stage durations use CUDA events; fixed-seed output hashes differ ([record](../../experiments/a100-nunchaku-flux-klein-001.md)).

The A100 Base conditioning benchmark was recovered from its W&B artifact and its CSV/summary arithmetic rechecked. Its physical GPU differs from the reference: the small latency difference is descriptive, not an isolated cache effect. Prior validation and quality receipts are retained in its run directory. Smoke tests, failed/incomplete runs, profiler captures and older duplicate captures are excluded.

## M5 scope

The preceding sd.cpp/PyTorch/edge-dit batch retained 35 uploaded M5 run records: 17 full protocols (including the earlier valid distilled reference), 15 successful smoke tests and three failed edge-dit attempts. Timings, image counts, tables and evidence artifacts were checked against the local records ([upload verification](m5-upload-verification.json)).

The six original M5 panels use sixteen fresh sd.cpp/PyTorch full protocols (four references and twelve individual options). The older valid sd.cpp distilled baseline remains a separate W&B record; smoke tests and failed edge-dit attempts are also separate records and excluded from these charts. Base native-weight preparation failed before a run existed and is documented in the experiment record, not represented as a measured run.

M5 system load, swap and thermal conditions were not controlled: observed timing differences are descriptive. Exact conditioning reuse applies to repeated identical conditioning. Distilled EasyCache executed zero skipped steps; Base EasyCache skipped 25 of 50 steps. EasyCache and VAE tiling have no formal quality approval. Engine comparisons differ in precision and initial noise, so they do not isolate engine performance.

## MFLUX additions

Five completed MFLUX full protocols are included: distilled BF16, 8-bit and 4-bit, and Base 8-bit and 4-bit. The Base BF16 smoke crashed in native MLX Metal compiled evaluation before any image; it has no timing bar. Five successful smoke runs and two preserved failures are separate W&B records and excluded from these charts. [Run evidence verification](../mflux-prep/wandb-evidence-verification.json).

MFLUX 8/4-bit uses affine weight quantization on eligible text-encoder, transformer and VAE layers, with BF16 remaining parameters. This differs from sd.cpp's mixed-Q4 configuration and PyTorch BF16. MLX stages are synchronized host times including scheduler work, not GPU event measurements. Loading/on-load quantization, first generation and warm-ups are excluded from the bars. System swap, thermal state and background applications were uncontrolled; these measurements do not isolate an engine or quantization speedup.

Both inspected 4-bit full outputs have incorrect sign lettering ([distilled record](../../experiments/m5-mflux-flux-klein-003.md), [Base record](../../experiments/m5-mflux-flux-klein-base-003.md)). Functional protocol validation is not quality approval. No held-out prompt-set assessment was performed.

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
