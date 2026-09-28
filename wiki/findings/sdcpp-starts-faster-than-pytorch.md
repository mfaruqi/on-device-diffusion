---
type: finding
summary: sd.cpp loads FLUX.2 klein 2.6× faster than PyTorch and its first generation is only ~7% slower than a warm one, versus 2.55× for PyTorch.
status: supported
confidence: medium
rq: [RQ1]
sources: [../../experiments/compare-a100-pytorch-vs-sdcpp-flux-klein.md]
updated: 2026-09-28
---

# stable-diffusion.cpp has a much smaller cold-start cost than PyTorch

**Claim.** Loading cached weights onto the GPU takes **5.2 s**
([`load.load_total_s`](../../results/runs/a100-sdcpp-flux-klein-001-20260925-114328/summary.json))
with sd.cpp and **13.7 s**
([`load.load_total_s`](../../results/runs/a100-flux-klein-001-20260923-221530/summary.json))
with PyTorch. The first generation costs 1.07× a warm run in sd.cpp and 2.55× in PyTorch.

**Evidence.** [Comparison](../../experiments/compare-a100-pytorch-vs-sdcpp-flux-klein.md#end-to-end-and-stages-median-of-10-measured-runs-ms).
PyTorch's first-run excess sits in text encode and VAE decode, probably one-time cuBLAS/cuDNN setup
([record](../../experiments/a100-flux-klein-001.md#observations)).

**Why it matters.** The proposal reports cold start separately from warm runs, and on-device use
often starts cold ([overview](../project/overview.md#evaluation-rules)).

Related: [RQ1](../rq/rq1.md).
