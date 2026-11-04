# Klein roofline analysis

48 source runs; 979 points.

- [a100-roofline](a100-roofline.html)
- [jetson-roofline](jetson-roofline.html)
- [m5-roofline](m5-roofline.html)
- [cross-device-roofline-pct](cross-device-roofline-pct.html)
- [a100-stage-roofline PNG](a100-stage-roofline.png) · [vector PDF](a100-stage-roofline.pdf)
- [a100-kernel-roofline PNG](a100-kernel-roofline.png) · [vector PDF](a100-kernel-roofline.pdf)

Analytic tensor traffic, not PMU DRAM counters. GEMM: 2MNK FLOPs, MK*b_a + KN*b_w + MN*b_a bytes. Flash attention: 4BHSqSkD FLOPs, 2BH(Sq+Sk)D*b_a bytes, no score materialization. Stage numerator is dominant linear + attention FLOPs of a reference klein 4B pass times executed CFG passes; excludes elementwise, normalization, positional encoding, modulation, dequantization and low-rank correction overhead. Stage bytes = one transformer weight read per pass + ideal operator activation reads/writes; assumes no inter-operator cache reuse. Operator throughput uses summed associated kernel durations, not stage host time or GPU busy-union time. sd.cpp inferred attention assumes one flash call per block. Stage effective throughput uses measured-generation median denoising time including host callback reloads for sd.cpp, CUDA/MPS events for PyTorch. Profile samples are separate instrumented generations. First/warmups excluded from baseline medians. Q4 storage does not imply INT4 compute. Mixed Nunchaku stage normalization uses BF16 as a reference ceiling, not an assertion of its mixed-kernel utilization. No-cache and EasyCache use executed work (not skipped reference FLOPs); approximate cache hits still incur uncounted overhead. Cross-device workloads/precision/residency differ; quality equivalence unassessed. Percent of theoretical roof is descriptive, not measured hardware utilization; above 100% is retained as a model discrepancy.

See [building instructions](../../docs/roofline-plots.md).
