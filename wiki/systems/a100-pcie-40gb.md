---
type: system
kind: device
summary: NVIDIA A100-PCIE-40GB on Gilbreth g-nodes (250 W) — the CUDA reference device for all A100 results.
status: active
aliases: [a100, a100-pcie]
updated: 2026-09-28
---

# NVIDIA A100-PCIE-40GB

- **Where**: Gilbreth `gilbreth-g000`–`g011`, two GPUs per node, Slurm feature `G` ([cluster](gilbreth.md)).
- **Specs recorded in runs**: 40,960 MiB, 250 W power limit, compute capability 8.0, driver 590.48.01 (`environment.json` in each run).
- **Role**: development and reference platform ([overview](../project/overview.md#targets)); every Gilbreth job is pinned to it
  ([D-002](../project/decisions.md#d-002-pin-gilbreth-jobs-to-a100-pcie-nodes)).
- **Peak BF16 dense tensor throughput** used for efficiency figures: 312 TFLOP/s (vendor figure).
- **Experiments**: see the [registry](../experiments.md).
