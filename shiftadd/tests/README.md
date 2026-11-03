# CPU correctness tests

From the repository root after installing `shiftadd`:

```bash
python -m unittest discover -s shiftadd/tests -v
```

Tests cover packing, decoding, grouped scales, integer arithmetic, mixed group
boundaries, loading guards, activation hooks, paired-case selection and analytic
power integration. They use synthetic arrays and temporary files; no downloaded
model, GPU, HPC account or W&B credentials are required.

Passing tests establishes software correctness for these fixtures. Real-artifact
replay and GPU image diagnostics have separate experiment records. Tests do not
establish image quality, speedups, memory savings during inference or energy wins.
