---
type: experiment-record
id: a100-ceilings-001
status: complete
device: a100-pcie-40gb
engine: pytorch-diffusers
runs: [a100-ceilings-001__ceilings__20261029-042704, a100-ceilings-001__ceilings__20261029-011815, a100-ceilings-001__ceilings__20261028-200826]
updated: 2026-10-29
---

# a100-ceilings-001: measured roofline ceilings of the A100 PCIe 40GB

Status: **complete** (2026-10-29, Slurm 12044905 on gilbreth-g011). Two earlier runs failed or are superseded (below).

## Question

What memory bandwidth and dense matrix-multiply throughput does this A100 PCIe 40GB sustain per precision,
measured, rather than taken from the datasheet? These values are the solid ceilings of the measured A100
rooflines ([procedure](../docs/a100-hardware-counters.md#measured-ceilings)). Serves RQ1 (where each engine's
denoising stage sits relative to the hardware limits).

## Setup

- [Config](../configs/a100-ceilings.json), [script](../scripts/measure_ceilings_cuda.py),
  [Slurm script](../scripts/gilbreth-ceilings.slurm) (standby QoS, `constraint=G` PCIe A100 nodes).
- No model is loaded. PyTorch 2.8.0+cu126 (CUDA runtime 12.6), driver 595.91.07, NVIDIA A100-PCIE-40GB
  ([environment](../results/runs/a100-ceilings-001__ceilings__20261029-042704/environment.json)). Clocks are not locked.
- Bandwidth: four kernels over 2 GiB FP32 buffers, counting the minimum bytes each must move, in decimal GB/s:
  copy (read + write), read (sum reduction), write (fill), triad (two reads, one write).
- GEMM: cuBLAS through `torch.matmul` (`torch._int_mm` for INT8) in the `nn.Linear` layout (weight stored
  N × K, used transposed). BF16, FP16, TF32 (`allow_tf32` on), FP32 (`allow_tf32` off) and INT8, each at square
  4096 and 8192 and at three FLUX.2 klein single-block shapes (M4608 N27648 K3072, M4608 N3072 K12288,
  M1536 N27648 K3072). Work is 2·M·N·K.
- Protocol per test: 5 warm-up calls, then 10 timed blocks of 20 calls (CUDA events). Each test reports the
  median block; a ceiling is the test with the highest median. A 60 s sustained BF16 8192³ GEMM follows,
  with `nvidia-smi` sampling clocks, power and throttle reasons every 500 ms.

## Results

[summary.json](../results/runs/a100-ceilings-001__ceilings__20261029-042704/summary.json),
[measurements.csv](../results/runs/a100-ceilings-001__ceilings__20261029-042704/measurements.csv),
[hardware-override.json](../results/runs/a100-ceilings-001__ceilings__20261029-042704/hardware-override.json).

| Ceiling | Measured (median) | Test that sets it | Datasheet | Ratio |
|---|---:|---|---:|---:|
| HBM bandwidth | 1550.5 GB/s | write | 1555 | 1.00 |
| BF16 Tensor Core | 218.6 TFLOP/s | square 8192 | 312 | 0.70 |
| FP16 Tensor Core | 207.0 TFLOP/s | square 8192 | 312 | 0.66 |
| TF32 Tensor Core | 101.2 TFLOP/s | square 4096 | 156 | 0.65 |
| FP32 | 17.5 TFLOP/s | square 8192 | 19.5 | 0.90 |
| INT8 Tensor Core | 395.3 TOP/s | square 4096 | 624 | 0.63 |

Bandwidth by kernel (median GB/s): copy 1381.0, read 1453.4, write 1550.5, triad 1369.4.

BF16 at the klein shapes (median TFLOP/s): 209.1 (M4608 N27648 K3072), 214.4 (M4608 N3072 K12288),
204.9 (M1536 N27648 K3072).

Sustained BF16 8192³ over 60 s: 565 blocks, first-quarter median 210.9, last-quarter median 209.6 TFLOP/s
(−0.6 %). GPU samples: SM clock median 1020 MHz (range 210–1410), memory clock 1215 MHz, peak power 354.1 W,
peak temperature 62 °C, throttle reasons `0x1` (idle) and `0x4` (software power cap).

### Failed and superseded runs

| Run | Job, node | Outcome |
|---|---|---|
| [20261028-200826](../results/runs/a100-ceilings-001__ceilings__20261028-200826/status.json) | 12038580, g006 | failed after the copy test: `torch.sum(a, out=out)` needs a dimension. Copy measured 1381.6 GB/s before the error; partial `measurements.csv` kept. |
| [20261029-011815](../results/runs/a100-ceilings-001__ceilings__20261029-011815/summary.json) | 12042202, g000 | complete, INT8 superseded: the second INT8 operand was row-major (K × N), for which cuBLAS does not select its INT8 Tensor Core kernels (70.5 TOP/s). Other ceilings agree with the final run within 4 %. |
| [20261029-042704](../results/runs/a100-ceilings-001__ceilings__20261029-042704/summary.json) | 12044905, g011 | complete; all precisions in the `nn.Linear` layout. These are the reported ceilings. |

The two later runs did not record the Slurm job in `environment.json`; the script now does.

## Observations

- Tensor Core throughput reaches 63–70 % of the datasheet for every precision; FP32 reaches 90 %.
- The sustained BF16 GEMM holds within 0.6 % over 60 s while `nvidia-smi` reports software power-cap
  throttling and a median SM clock of 1020 MHz, below the 1410 MHz maximum.
- The klein single-block shapes run at 94–98 % of the best square BF16 result.
- The write kernel reaches the datasheet bandwidth; kernels that read reach 88–93 % of it.

## Interpretation

Hypothesis: on this card the attainable BF16 matrix rate under sustained load is set by the power cap, not by
the Tensor Core peak. The clock settles near 1 GHz; scaling the 312 TFLOP/s peak (at 1410 MHz) by 1020/1410 gives
about 226, close to the measured 218.6.
Single kernels profiled in isolation (Nsight Compute kernel replay) can run at boost clock and exceed this
ceiling; the [eager counter capture](a100-flux-klein-torch28-001.md#hardware-counter-capture-2026-10-24)
records GEMMs at 258–265 TFLOP/s. The bandwidth ceiling is the best case (pure write); a kernel that mostly
reads is bounded nearer 1450 GB/s.

## Next experiment

- `configs/a100-ceilings.json` with `nvidia-smi -lgc` locked clocks (needs administrator support) to separate
  the clock effect from the power cap.
