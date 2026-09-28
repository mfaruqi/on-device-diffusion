---
type: system
kind: cluster
summary: Purdue RCAC Gilbreth community GPU cluster — account you139 (one A100-40GB share), Slurm, storage layout.
status: active
updated: 2026-09-28
---

# Gilbreth (Purdue RCAC)

- **Access**: account `you139` owns one A100-40GB share (`normal` QOS). `standby` borrows idle GPUs for up to 4 h
  ([how-to](../methods/gilbreth-howto.md)).
- **Partition `a100-40gb`** mixes [A100-PCIE-40GB](a100-pcie-40gb.md) and [A100-SXM4-40GB](a100-sxm4-40gb.md) nodes.
- **Storage**: home (25 GB, snapshots) holds the repo and envs; scratch (purged after 60 days of no use)
  holds the model cache; depot `/depot/you139` (100 GB, lab-shared) holds long-lived artifacts ([how-to](../methods/gilbreth-howto.md#storage-rules)).
- **Toolchain**: CUDA 12.6 (nvcc, nsys, ncu), gcc 11.5, cmake 3.26 ([sd.cpp how-to](../methods/sdcpp-howto.md)).
- **Login nodes** have no usable GPU; compute nodes run jobs submitted with `sbatch`.
