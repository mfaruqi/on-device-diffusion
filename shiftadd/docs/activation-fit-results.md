# Activation-weighted slice fitting: initial results

Source capture: A100 job 12133795, BF16 Klein, two modules, four calibration
prompts × four denoising steps × 32 sampled tokens; 3072 input channels.
Local evidence: `shiftadd/runs/a100-activation-capture-001/`.

Two prompts fit the diagonal moment weights; the other two form a calibration
validation split. This is not the project's held-out test set. Fitting and
validation multiply saved BF16-origin inputs and weights in FP32.

The first-32-output-row screen (`artifacts/activation-fit-slice-001/`) suggested
improvements in both modules. A second screen spreads 32 output rows across each
entire matrix (`artifacts/activation-fit-spread-001/`) to avoid restricting the
fused projection to its first outputs. Configs and exact selected indices are saved.

Mean per-call relative output L2 on eight calibration validation calls, spread
rows, from `aggregate-validation.json`:

| Module slice | Fit | Unquantized captured inputs | A8 simulation | A4 simulation |
|---|---|---:|---:|---:|
| Query projection | Weight only | 0.081639 | 0.082107 | 0.242219 |
| Query projection | Diagonal activation | 0.045345 | 0.045960 | 0.222394 |
| Fused projection | Weight only | 0.079554 | 0.081215 | 0.318894 |
| Fused projection | Diagonal activation | 0.080051 | 0.081780 | 0.320034 |

Interpretation: the diagonal objective helps the query slice, but not the spread
fused slice. A8 adds relatively little error in these cases; naive A4 adds much
more. These are layer-slice numerical diagnostics, not generated-image quality,
full-layer guarantees, GPU-kernel results or a formal activation-precision ranking.
No full-model activation-weighted conversion is justified by this screen alone.

Next hypotheses: retain layer-specific calibration decisions; examine activation
outliers and correlations before broader A4 deployment. A full covariance/GPTQ
method or low-rank correction would be separately labelled work, with its own
storage and execution costs.
