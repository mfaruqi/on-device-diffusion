# How FLUX weight conversion works

## Relationship to ShiftAddQwen

The current upstream path calls the actual pinned ShiftAddQwen `bcq.py` and
`gptq_bcq.py` implementations. They are not locally rewritten approximations:
[provenance](../vendor/shiftaddqwen/PROVENANCE.json) records the source commit and
hashes, and [the adapter](../src/shiftadd_flux/upstream_probe.py) checks those hashes
before fitting. Early greedy and restricted-radix experiments are separate controls.

The adapter uses activation ordering, GPTQ error compensation and scale refitting.
The current [fused-matrix config](../configs/upstream-bcq-all-single-attention-001.json)
uses four planes, groups of 128, u8 scale codes with shared FP16 scales,
15 fitting iterations, damping 0.01 and complete output rows. Superblocks have
256 weights; the adapter disables the optional mean term.

This is a partial adaptation, not the complete ShiftAddQwen pipeline:

| Component | FLUX implementation |
|---|---|
| Output-aware GPTQ/BCQ fitting | Reuses the pinned upstream fitter |
| Calibration inputs | Captures BF16 FLUX inputs across four denoising steps |
| Sequential quantized-prefix calibration | Not implemented; captures come from the BF16 model |
| Mixed precision | Fixed-count selection by local fitting error, not end-to-end KLD knapsack |
| End-to-end scale distillation | Not implemented |
| Deployment format | Local packed NPZ arrays, not the upstream GGUF runtime types |
| GPU execution | Dense BF16 reconstruction; upstream CUDA kernels are not ported |

See the [reviewed upstream method](../../wiki/papers/shiftaddqwen.md) and
[mixed-candidate record](../../experiments/m5-flux-klein-shiftadd-022.md).

## Conversion, step by step

1. **Capture inputs.** `capture_pipeline.py` runs BF16 FLUX on calibration prompts;
   `capture.py` saves sampled input vectors for selected linear layers.
2. **Fit the approximation.** `upstream_probe.py` forms `H = X.T @ X / N` from
   fitting inputs and calls the upstream fitter. This weights errors by the
   inputs the layer actually sees, rather than weight error alone.
3. **Pack the result.** Each weight is approximated by a sum of signed plane
   scales. Sign choices become packed bits; u8 scale codes and shared FP16 scales
   are stored alongside them. The adapter checks reconstruction against the
   upstream fitter's output before saving.
4. **Verify independently.** `validation/verify_upstream.py` reloads the artifacts
   and checks reconstruction/error evidence. Calibration-validation calls are
   separate from fitting calls; neither is the final held-out evaluation.
5. **Assemble and package.** `assemble_upstream.py` combines selected matrices.
   `package_upstream.py` includes untouched BF16 tensors so the transformer can
   load without reading the original weight file. `select_mixed_planes.py` is
   an optional experimental three/four-plane selection step.
6. **Load and generate.** `upstream_loader.py` reconstructs dense BF16 weights.
   `calibration_smoke.py` compares images from matched cases. Activation hooks
   simulate low-bit rounding but still execute floating-point matrix products.

The [current campaign](current-campaign.md) covers 50 upstream-converted matrices,
not every transformer matrix. Text encoder and VAE conversion are outside scope.

## Which files are necessary?

The conversion/loading modules, vendor source, package metadata and matching
configs form the working path. Tests and validation protect correctness.
Diagnostics and plots are optional for conversion, but support the recorded
experiments. Reporting is optional for local work. None of these supporting
files is a trained weight file or a GPU kernel.

The [code guide](../src/shiftadd_flux/README.md) lists every Python module.
The [configuration guide](../configs/README.md) explains experiment filename families.
