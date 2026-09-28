---
type: concept
summary: Lowering a convolution to a matrix multiply by unrolling input patches into a large matrix (im2col); simple, but memory- and bandwidth-heavy at high resolution.
status: active
aliases: [im2col]
updated: 2026-09-28
---

# im2col convolution

A k×k convolution over a C-channel input becomes one GEMM once every input patch is unrolled into a
row of a (H·W) × (C·k²) matrix. The GEMM is fast, but building and storing the unrolled matrix
moves about k² times the input data. At 1024² that dominates a VAE decoder.

In this project, sd.cpp's VAE decode spends two thirds of its GPU time in `im2col_kernel`
([finding](../findings/sdcpp-vae-decode-dominated-by-im2col.md)). PyTorch's decoder uses cuDNN
convolutions instead ([record](../../experiments/a100-flux-klein-001.md#profiler-run-diagnostic-job-11807396)).
The alternatives are direct convolution kernels (sd.cpp `--vae-conv-direct`) and tiled decoding, which
bounds the unrolled matrix size. Tiled decoding is one of the proposal's memory options
([overview](../project/overview.md#optimization-families-registry)).
