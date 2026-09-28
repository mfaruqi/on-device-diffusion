---
type: finding
summary: In sd.cpp, two thirds of FLUX.2 klein's VAE decode GPU time at 1024² is the im2col step of its convolutions; the decode is 3.65× slower than PyTorch's.
status: supported
confidence: high
rq: [RQ1, RQ3]
sources: [../../experiments/a100-sdcpp-flux-klein-001.md, ../../experiments/compare-a100-pytorch-vs-sdcpp-flux-klein.md]
updated: 2026-09-28
---

# sd.cpp's VAE decode is dominated by im2col

**Claim.** `im2col_kernel` takes **403.7 ms**
([`stages.vae_decode.by_leaf_op.im2col_kernel.ms`](../../results/runs/a100-sdcpp-flux-klein-001__profile__20260925-124703/profile/other_stages.json))
of the decode's 606.8 ms of GPU time. End to end, the decode takes **603.5 ms**
([`measured.vae_decode_ms.median`](../../results/runs/a100-sdcpp-flux-klein-001__baseline__20260925-114328/summary.json))
versus 165.3 ms in PyTorch. sd.cpp's figure includes the conversion to uint8, which PyTorch
reports separately as postprocess, so the like-for-like gap is slightly under 3.65×.

**Evidence.** Profile of the sd.cpp reference run ([record](../../experiments/a100-sdcpp-flux-klein-001.md#profiler-run-diagnostic-nsight-systems-job-11818087)).
The decoder GEMMs run in FP16 and take about 50 ms.

**Hypothesis.** The unrolled im2col matrices are the likely reason for sd.cpp's 6.66 GB VAE compute
buffer, which sets its memory peak. `--vae-conv-direct` and `--vae-tiling` are the engine's own
options against this. Each is a separate configuration, not yet run.

Related: [im2col](../concepts/im2col-convolution.md), [VAE sets the peak](vae-decode-sets-peak-memory.md).
