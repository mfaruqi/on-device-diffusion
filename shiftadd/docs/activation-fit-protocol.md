# Bounded activation-weighted fit

## Hypothesis

Weight-error-only fitting ignores input-channel scale and correlation. Weighting
squared reconstruction error by E[x_i²] may improve output accuracy at fixed
storage. This is a diagonal second-moment approximation; it is not full GPTQ.

## Capture and split

`a100-activation-capture-001.json` specifies two linears and the four calibration
prompts already used for the image screen. Each module contributes four calls per
prompt, at most 32 evenly spaced tokens per call. Calls 0–7 (first two prompts)
fit the scales; calls 8–15 (last two prompts) are calibration validation. This
split is inside calibration data and does not access project held-out data.

The initial fit uses only the first 32 output rows of each source weight matrix.
It compares uniform signed-radix p4 scale search with the same candidate scale
search weighted by mean squared captured activations. Packed byte budgets match.

## Evaluation

Compute FP32 X Wᵀ using the saved BF16-origin weights and inputs as the layer
reference. Compare reconstructed-weight outputs for unquantized captured inputs,
per-token symmetric INT8 inputs, and per-token symmetric INT4 inputs. Report
relative output L2 per prompt and denoising call, separating fit and calibration
validation. This isolates numerical representation error, not actual BF16 GPU
rounding or runtime behavior. It does not measure generated-image quality.

A lower diagonal fitting objective does not guarantee lower real output error,
because cross-channel correlations are omitted. Save adverse results as well.
Do not expand to all matrices merely because the fitting loss decreases.

## Verification

A synthetic end-to-end evaluator fixture is preserved at
`shiftadd/runs/evaluator-fixture-001/`. It produces 96 comparisons: one module,
two fitting methods, sixteen calls and three activation representations. It
checks disjoint fit/validation call assignment and saves reconstructable planes.
These fixture values are code checks only, not model findings.

Actual capture job: 12133795; verify scheduler/status before consuming inputs.
