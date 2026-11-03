# Code guide

Start with the conversion flow below. Diagnostics and plots support experiments; they are not required to load converted weights.

## Current upstream conversion flow

1. `capture_pipeline.py` uses `capture.py` to record inputs from BF16 FLUX calibration runs.
2. `upstream_probe.py` builds the input covariance and calls the pinned ShiftAddQwen GPTQ/BCQ fitter in `vendor/shiftaddqwen/saq/`.
3. `validation/verify_upstream.py` independently checks the saved packed representation.
4. `assemble_upstream.py` combines fitted matrices; `select_mixed_planes.py` optionally chooses three/four-plane candidates.
5. `package_upstream.py` adds unchanged BF16 tensors; `upstream_loader.py` reconstructs dense BF16 weights.
6. `calibration_smoke.py` compares generated images. This is not packed GPU execution.

The initial `convert.py` / `radix.py` path is a separate baseline, not the upstream fitter.

## File inventory

Paths below are relative to this directory. `__init__.py` files mark Python packages.

### Core conversion, loading and image evaluation

| File | Purpose |
|---|---|
| [activation.py](activation.py) | Numerical references only; these routines are not performance kernels. |
| [activation_hooks.py](activation_hooks.py) | Explicit fake activation quantization for numerical pipeline experiments. |
| [activation_storage.py](activation_storage.py) | Byte layout reference for signed activation codes, not a GPU kernel. |
| [assemble_upstream.py](assemble_upstream.py) | Assemble verified full-row BCQ fits without refitting or changing values. |
| [calibration_smoke.py](calibration_smoke.py) | Paired image smoke test; dense reconstructed execution, not packed kernels. |
| [capture.py](capture.py) | Bounded calibration activation capture; instrumentation invalidates timing. |
| [capture_pipeline.py](capture_pipeline.py) | Capture selected real Klein activations on calibration prompts only. |
| [case_validation.py](case_validation.py) | Reject paired-case schedules that would retain unintended prior weights. |
| [compact_radix.py](compact_radix.py) | Lossless exponent metadata compression for verified radix bundles only. |
| [conventional.py](conventional.py) | Bounded symmetric uniform weight-quantization reference. |
| [convert.py](convert.py) | Config-driven, local safetensors-to-packed-planes conversion. |
| [convert_uniform.py](convert_uniform.py) | Local conventional control bundle from the same pinned transformer source. |
| [format.py](format.py) | Packed residual sign planes and integer power-of-two exponents. |
| [integer_radix.py](integer_radix.py) | Exact integer-code bridges for restricted signed radix, not GPU kernels. |
| [loader.py](loader.py) | Reconstruct saved planes into existing model parameters for quality tests. |
| [package_upstream.py](package_upstream.py) | Save and load complete transformer weights with an upstream BCQ subset. |
| [pipeline_smoke.py](pipeline_smoke.py) | Paired image smoke test; dense reconstructed execution, not packed kernels. |
| [radix.py](radix.py) | Joint signed-radix planes with groupwise power-of-two unit search. |
| [refine.py](refine.py) | Coordinate refinement of power-of-two residual planes. |
| [scaled_plane_arithmetic.py](scaled_plane_arithmetic.py) | CPU correctness reference for sign planes with arbitrary nonnegative scales. |
| [select_mixed_planes.py](select_mixed_planes.py) | Select a fixed number of smaller fits using fitting-call error only. |
| [upstream_loader.py](upstream_loader.py) | Dense reconstruction of the explicitly labelled upstream BCQ artifacts. |
| [upstream_probe.py](upstream_probe.py) | Bounded application of pinned ShiftAddQwen's upstream BCQ fitter to FLUX slices. |

### Plots

| File | Purpose |
|---|---|
| [plots/plot_calibration.py](plots/plot_calibration.py) | Render all available calibration image cases without altering the images. |
| [plots/plot_initial.py](plots/plot_initial.py) | Static evidence figure from saved images and weight-file reports. |
| [plots/plot_plane_tradeoff.py](plots/plot_plane_tradeoff.py) | Plot a saved mixed-plane selection diagnostic, not image-quality scores. |
| [plots/plot_storage.py](plots/plot_storage.py) | Standalone storage figure from a verified bundle inventory. |

### Diagnostics

| File | Purpose |
|---|---|
| [diagnostics/evaluate_capture.py](diagnostics/evaluate_capture.py) | Fit a bounded weight slice using calibration moments; validate actual outputs. |
| [diagnostics/full_query_controls.py](diagnostics/full_query_controls.py) | Numerical controls for one complete upstream-fitted projection. |
| [diagnostics/load_memory_probe.py](diagnostics/load_memory_probe.py) | Fresh-process CPU loader-memory diagnostic; no inference benchmark. |
| [diagnostics/nonuniform.py](diagnostics/nonuniform.py) | Experimental exact code assignment over a bounded exponent-template search. |
| [diagnostics/residual_probe.py](diagnostics/residual_probe.py) | Weight-residual SVD slice diagnostic; not the SVDQuant algorithm. |
| [diagnostics/uniform_probe.py](diagnostics/uniform_probe.py) | Same-source symmetric uniform quantization slice control. |
| [diagnostics/upstream_activation_probe.py](diagnostics/upstream_activation_probe.py) | Full-row activation-error diagnostics for saved upstream BCQ weights. |
| [diagnostics/upstream_load_probe.py](diagnostics/upstream_load_probe.py) | Fresh-process upstream loader timing and memory; no inference benchmark. |

### Validation

| File | Purpose |
|---|---|
| [validation/arithmetic_check.py](validation/arithmetic_check.py) | Portable integer-activation arithmetic check on a saved real-weight slice. |
| [validation/audit_bundles.py](validation/audit_bundles.py) | Read-only bundle integrity and logical storage inventory. |
| [validation/validate_control.py](validation/validate_control.py) | Validate paired control evidence; this is not a quality acceptance gate. |
| [validation/verify_grouped.py](validation/verify_grouped.py) | Verify saved real-input arithmetic; no speed or quality benchmark. |
| [validation/verify_integer_bridge.py](validation/verify_integer_bridge.py) | Verify exact radix bridges on saved real calibration inputs. |
| [validation/verify_scaled_arithmetic.py](validation/verify_scaled_arithmetic.py) | Verify integer sign-plane arithmetic against decoded upstream artifacts. |
| [validation/verify_upstream.py](validation/verify_upstream.py) | Reload upstream-adapter packed artifacts and reproduce recorded output errors. |

### Reporting

| File | Purpose |
|---|---|
| [reporting/energy.py](reporting/energy.py) | Gross sampled-power integration; no inference or power-sampling implementation. |
| [reporting/publish.py](reporting/publish.py) | Publish diagnostic payloads through the parent repository's W&B exporter. |

## Commands after reorganization

Run from the repository root with the package installed or `PYTHONPATH=shiftadd/src`:

```bash
python -m shiftadd_flux.upstream_probe --config /path/to/fit.json
python -m shiftadd_flux.validation.verify_upstream --config /path/to/fit.json
python -m shiftadd_flux.diagnostics.upstream_load_probe --config /path/to/load.json
python -m shiftadd_flux.plots.plot_plane_tradeoff --help
```

Historical raw run records retain their original module names. For those commands, insert the folder shown in the inventory (for example, `shiftadd_flux.verify_upstream` becomes `shiftadd_flux.validation.verify_upstream`). Old flat utility entry points are not retained. Core conversion and image entry points are unchanged.

Tests remain in `../../tests/`. Labelled configs and Slurm jobs remain in `../../configs/` to preserve their recorded paths. Vendor code, weights and evidence are unchanged.
