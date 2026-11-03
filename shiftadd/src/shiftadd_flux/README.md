# Code guide

Run modules as `python -m shiftadd_flux.<module>` after installing the subproject,
or set `PYTHONPATH=shiftadd/src` from the parent repository. CLI modules support
`--help`. Module-level functions are research interfaces, not a stable public API.

| Area | Modules | Role |
|---|---|---|
| Weight-only fits | `format`, `radix`, `nonuniform`, `conventional` | Encoders, decoders and numerical fit routines |
| Conversion | `convert`, `convert_uniform`, `refine`, `compact_radix` | Create labelled bundles and preserve provenance |
| Calibration | `capture`, `capture_pipeline`, `evaluate_capture`, `upstream_probe` | Capture bounded inputs and fit with calibration data |
| Loading | `loader`, `upstream_loader` | Verify artifact hashes/shapes and reconstruct dense parameters |
| Activation arithmetic | `activation`, `activation_storage`, `integer_radix`, `scaled_plane_arithmetic` | Quantized activation codes, packed layouts and CPU arithmetic references |
| Pipeline tests | `activation_hooks`, `pipeline_smoke`, `calibration_smoke`, `case_validation` | Scoped fake quantization and paired image diagnostics |
| Verification | `verify_*`, `arithmetic_check`, `validate_control`, `audit_bundles` | Reload checks, arithmetic equivalence and evidence audits |
| Diagnostics | `residual_probe`, `uniform_probe`, `full_query_controls`, `upstream_activation_probe`, `load_memory_probe` | Low-rank, conventional and memory-loading controls |
| Reporting | `plot_*`, `publish`, `energy` | Figures, parent W&B export and timestamped-power integration |

`loader.apply_artifacts` handles sign-plane bundles; use
`apply_uniform_artifacts` for conventional INT4 and
`upstream_loader.apply_upstream_artifacts` for full-row upstream BCQ artifacts.
The upstream loader intentionally rejects partial-row slices as model replacements.
All loaders reconstruct dense weights: packing is a storage representation here.

Paired runners reuse a model. `case_validation` rejects a later case that would
leave weights modified by an earlier case outside its declared selection. Each
candidate must overwrite every previously changed parameter; changing to an
unrelated selection requires a fresh model/run.

The exact single-integer-code bridge in `integer_radix` applies only to the
restricted radix family. Arbitrary upstream scales instead use signed sums per
plane followed by scale multiplication in `scaled_plane_arithmetic`.
