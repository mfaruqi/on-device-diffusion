---
type: experiment-record
id: m5-flux-klein-shiftadd-004
status: complete
device: macbook-air-m5
engine: pytorch-diffusers
runs: [m5-flux-klein-shiftadd-004__attempt__20261102-r0, m5-flux-klein-shiftadd-004__attempt__20261102-r4, m5-flux-klein-shiftadd-004__attempt__20261102-r8]
updated: 2026-11-02
---

# m5-flux-klein-shiftadd-004: BF16 low-rank residual correction on packed slices

## Question and setup

Week 5 execution variants, RQ1/RQ3. Does a small BF16 correction improve the
output error of four-plane radix slices enough to justify its extra storage?
Reference rank 0 and variants rank 4/8 differ only in correction rank.
Use the same 32 spread rows of two modules and captured calibration inputs as
[the grouped activation diagnostic](m5-flux-klein-shiftadd-003.md).

Fit truncated SVD to the weight residual W−Wq. Save factors UΣ and Vᵀ in BF16.
The correction branch consumes original BF16-origin inputs, whereas the packed
main branch uses original inputs, simulated per-token A8, or grouped A4 with
32-channel groups. Products and accumulation use FP32; this is not a BF16 GPU
execution or a reproduction of SVDQuant. No activation data is used to fit the
residual. The last two prompts (eight calls) are calibration validation, not the
project held-out set. Source revision, configs, and exact source slice artifact
paths are in the saved run files. Correction files remain local.

## Results

Mean per-call relative output L2 on calibration validation. Extra payload is
relative to the original packed slice, not relative to BF16 source storage.

| Residual rank | Module | Original inputs | A8 per token | A4 group 32 | Extra weight payload |
|---|---|---:|---:|---:|---:|
| 0 | transformer_blocks | 0.081639 | 0.082107 | 0.095928 | 0.00% |
| 0 | single_transformer_blocks | 0.079554 | 0.081215 | 0.133765 | 0.00% |
| 4 | transformer_blocks | 0.050403 | 0.050955 | 0.069058 | 47.55% |
| 4 | single_transformer_blocks | 0.070376 | 0.072493 | 0.126341 | 47.55% |
| 8 | transformer_blocks | 0.038033 | 0.038781 | 0.058240 | 95.10% |
| 8 | single_transformer_blocks | 0.061739 | 0.064090 | 0.122779 | 95.10% |

Each slice contains 32×3072 weights: 196,608 BF16 bytes, or 52,224 packed
four-plane bytes. Rank 4 adds 24,832 BF16 factor bytes; rank 8 adds 49,664.
Total slice payload is consequently 77,056 or 101,888 bytes, respectively.
Actual file sizes including headers are reported separately in `summary.json`.

Evidence:
- [m5-flux-klein-shiftadd-004__attempt__20261102-r0](../results/runs/m5-flux-klein-shiftadd-004__attempt__20261102-r0/)
- [m5-flux-klein-shiftadd-004__attempt__20261102-r4](../results/runs/m5-flux-klein-shiftadd-004__attempt__20261102-r4/)
- [m5-flux-klein-shiftadd-004__attempt__20261102-r8](../results/runs/m5-flux-klein-shiftadd-004__attempt__20261102-r8/)

## Interpretation and limits

Correction lowers mean error in both sampled modules, but the query slice
benefits more. A4 error includes activation error that weight-residual correction
does not directly target. The correction adds two matrix products and needs
original activations; no latency, energy, peak-memory or image-quality benefit
is established. The unusually short 32-row slices have a different correction
storage ratio and residual spectrum from full matrices, so these rank choices
and savings must not be extrapolated to whole layers or the transformer.

## Next experiment

- Assess a larger output slice before choosing full-layer correction ranks.
