---
type: experiment-record
id: jetson-flux-klein-shiftadd-001
status: complete
device: jetson-orin-nano
engine: numpy-reference
runs: [jetson-flux-klein-shiftadd-001__attempt__20261102-integer-bridge]
updated: 2026-11-02
---

# jetson-flux-klein-shiftadd-001: exact radix-to-integer arithmetic on Jetson CPU

## Setup

Week 5 execution preparation, RQ1/RQ3. NumPy 1.21.5, Python 3.10.12,
aarch64 Linux 5.15.148-tegra. CPU-only numerical reference; no GPU model is loaded
and no throughput/energy measurements are made. Saved 32-row weight slices from
two transformer projections, both weight-only and diagonal-activation fits, and
16 captured calibration calls per module. The source transfer manifest verifies
SHA-256 of every input/weight file before execution.

Four-plane radix weights map to exact INT8 codes or signed INT4 codes plus the
2q+1 affine group correction. The latter adds an input-dependent group sum.
Both weight code formats round-trip through byte packing. Activations use A4/A8
at group sizes 32, 128 and 3072. The dense FP32 reference reconstructs the same
weights and activations before multiplication. No source weights are refitted.

## Results

All **768 comparisons pass** the declared 1e-5 relative-L2
threshold. The maximum relative difference is **2.99121325e-07**.
All weight reconstructions are exact before multiplication; remaining output
differences are compatible with FP32 accumulation-order differences.
[Raw results, environment, configuration and transfer hashes](../results/runs/jetson-flux-klein-shiftadd-001__attempt__20261102-integer-bridge/).

## Limits

This establishes correctness for the tested slices on the Jetson CPU. It does
not establish GPU operator support, whole-model image quality, live compressed
memory, latency or energy. No nominal operation count is presented as a power
saving. Captured calibration inputs are reused; no project held-out set is used.
