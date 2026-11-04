# Plane-allocation size charts

These charts separate logical stored weight sizes from the calculated dense BF16
parameter footprint. They do not report measured peak GPU memory.

Reproduce from the repository root with Matplotlib available:

```bash
PYTHONPATH=shiftadd/src python -m shiftadd_flux.plots.plot_plane_storage --output output/shiftadd-plane-storage-20261104
```

- `stored-weight-sizes.png`: selected fifty matrices and whole-transformer accounting.
- `dense-runtime-weight-footprint.png`: dense parameter payload used by the current runner.
- `data.json`: exact byte values, scope definitions and source SHA256 hashes.
- `sizes.csv`: tabular values in bytes.

[Comparison record](../../experiments/compare-shiftadd-plane-storage.md).
Generated figures remain local; the script and small source-derived data are versioned.
