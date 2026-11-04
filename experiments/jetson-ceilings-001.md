---
type: experiment-record
id: jetson-ceilings-001
status: complete
device: jetson-orin-nano
engine: pytorch-diffusers
runs: [jetson-ceilings-001__ceilings__20261029-105711]
updated: 2026-11-01
---

# jetson-ceilings-001: measured roofline ceilings of the Jetson Orin Nano 8GB (25 W)

Status: **complete** (2026-10-29, Jetson Orin Nano `mfaruqi-desktop`, under `jetson_memguard.sh`).

## Question

What memory bandwidth and dense matrix-multiply throughput does this Jetson Orin Nano sustain per precision in
the 25 W power mode, measured rather than taken from the datasheet? These values are the solid ceilings of the
measured Jetson rooflines. Serves RQ1 (where each engine's denoising stage sits relative to the hardware limits).

## Setup

- [Config](../configs/jetson-ceilings.json), [script](../scripts/measure_ceilings_cuda.py) (same script as
  [a100-ceilings-001](a100-ceilings-001.md); the config's `sampler: tegrastats` selects the Jetson clock and
  power adapter).
- No model is loaded. PyTorch 2.8.0 (CUDA runtime 12.6), driver 540.4.0, power mode 25 W, GPU devfreq maximum
  918 MHz ([environment](../results/runs/jetson-ceilings-001__ceilings__20261029-105711/environment.json)). Clocks
  are not locked; `jetson_clocks` was not applied.
- Bandwidth: copy, read, write and triad over 0.5 GiB FP32 buffers in the shared LPDDR5, counting the minimum bytes
  each kernel must move, in decimal GB/s.
- GEMM: cuBLAS through `torch.matmul` (`torch._int_mm` for INT8) in the `nn.Linear` layout, BF16, FP16, TF32, FP32
  and INT8, at square 2048 and 4096 and at two FLUX.2 klein 512² single-block shapes (M1536 N27648 K3072,
  M1536 N3072 K12288). Work is 2·M·N·K.
- Protocol per test: 5 warm-up calls, then 10 timed blocks of 20 calls (CUDA events); the test value is the median
  block, and a ceiling is the test with the highest median. A 60 s sustained BF16 4096³ GEMM follows, with the GPU
  devfreq clock sampled every 500 ms and `tegrastats` logging power rails and temperatures.
- Datasheet values (dashed roofs) and how the unpublished ones are derived are recorded in the config's `datasheet`
  block: 102 GB/s, BF16/FP16 17 TFLOP/s, INT8 33 TOP/s dense, quoted for the 1020 MHz Super mode.
- Metric definitions: [measured ceilings (Jetson)](../wiki/methods/baseline-metrics.md#measured-ceilings-jetson).

## Results

[summary.json](../results/runs/jetson-ceilings-001__ceilings__20261029-105711/summary.json),
[measurements.csv](../results/runs/jetson-ceilings-001__ceilings__20261029-105711/measurements.csv),
[hardware-override.json](../results/runs/jetson-ceilings-001__ceilings__20261029-105711/hardware-override.json).

| Ceiling | Measured (median) | Test that sets it | Datasheet | Ratio |
|---|---:|---|---:|---:|
| LPDDR5 bandwidth | 97.4 GB/s | triad | 102 | 0.95 |
| BF16 Tensor Core | 9.88 TFLOP/s | square 2048 | 17 | 0.58 |
| FP16 Tensor Core | 9.28 TFLOP/s | square 4096 | 17 | 0.55 |
| TF32 Tensor Core | 5.50 TFLOP/s | klein M1536 N27648 K3072 | 8.5 | 0.65 |
| FP32 | 1.17 TFLOP/s | square 2048 | 2.09 | 0.56 |
| INT8 Tensor Core | 14.57 TOP/s | klein M1536 N3072 K12288 | 33 | 0.44 |

Bandwidth by kernel (median GB/s): copy 79.1, read 77.9, write 81.4, triad 97.4.

BF16 at the klein shapes (median TFLOP/s): 7.81 (M1536 N27648 K3072), 4.79 (M1536 N3072 K12288).

Sustained BF16 4096³ over 60 s: 191 blocks, first-quarter median 8.86, last-quarter median 8.63 TFLOP/s (−2.6 %).
GPU samples (whole run, 500): devfreq clock median 918 MHz (range 306–918), peak VDD_IN 23.2 W, peak GPU temperature
68.4 °C. Memory guard: MemAvailable 6700 MB at start, minimum 4418 MB, no swap.

Measured ridge points: BF16 9.88 TFLOP/s ÷ 97.4 GB/s = 101 FLOP/B; INT8 14.57 TOP/s ÷ 97.4 GB/s = 150 OP/B.

## Observations

- Bandwidth reaches 95 % of the datasheet with triad; single-stream copy, read and write kernels reach 76–80 %.
- Tensor Core throughput reaches 44–65 % of the datasheet; INT8 is lowest at 44 %.
- The klein single-block output projection (K12288, N3072) runs at 48 % of the best BF16 result; the input
  projection at 79 %.
- The GPU clock stayed at its 918 MHz maximum (median) and the sustained GEMM lost 2.6 % over 60 s.

## Interpretation

Hypothesis: part of the gap to the datasheet compute peaks is the clock. The datasheet figures assume the 1020 MHz
Super mode; this board ran at 918 MHz, which scales 17 TFLOP/s to about 15.3. The remaining gap (9.9 vs 15.3) is
not explained by this run. The bandwidth ceiling is the best case (triad, two reads and one write); kernels that
mostly read or mostly write sustained 78–81 GB/s.

## Next experiment

- `configs/jetson-ceilings.json` in the MAXN Super power mode with `jetson_clocks` (run by the board owner, root)
  to separate the clock effect.
