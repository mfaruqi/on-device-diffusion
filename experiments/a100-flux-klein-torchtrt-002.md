---
type: experiment-record
id: a100-flux-klein-torchtrt-002
status: failed
device: a100-pcie-40gb
engine: pytorch-diffusers
runs: [a100-flux-klein-torchtrt-002__attempt__20261005-081316]
updated: 2026-10-05
---

# A100 Torch-TensorRT: BF16 conversion failure with contiguous inputs

Status: **failed**, job11883257 on gilbreth-g007,12:13:16–12:20:45UTC.

## Question

Can the real distilled klein transformer input compile at unchanged BF16 precision after explicitly making input tensors contiguous? Week2 engine feasibility, RQ1. The only integration change from the [original gate](a100-flux-klein-torchtrt-001.md) is a value-preserving input-layout copy; this is not a baseline or speed experiment.

## Setup

[Config](../results/runs/a100-flux-klein-torchtrt-002__attempt__20261005-081316/config.json), [environment](../results/runs/a100-flux-klein-torchtrt-002__attempt__20261005-081316/environment.json), [input shapes](../results/runs/a100-flux-klein-torchtrt-002__attempt__20261005-081316/inputs.json). Pinned distilled klein revision e7b7dc27f91deacad38e78976d1f2b499d76a294,1024×1024,four steps,guidance1,seed0,BF16. Isolated torch2.5.1+cu124,Torch-TensorRT2.5.0,TensorRT10.3.0. Same-environment eager image precedes export/compilation. Five real transformer inputs are copied to contiguous layout; every copied tensor must equal its original. Original-layout eager forward is the intended numerical reference. Compile settings and precision are unchanged;1200second process bound.

## Results

[Status/traceback](../results/runs/a100-flux-klein-torchtrt-002__attempt__20261005-081316/status.json), [layout receipt](../results/runs/a100-flux-klein-torchtrt-002__attempt__20261005-081316/input-layout.json), [validation](../results/runs/a100-flux-klein-torchtrt-002__attempt__20261005-081316/validation.json), [provenance](../results/runs/a100-flux-klein-torchtrt-002__attempt__20261005-081316/provenance.json).

All five compile inputs are contiguous and value-equal to their originals. Eager image and graph export completed. Pipeline loading took28.990065s and export7.112953s; these are one-off preparation observations, not timing medians.

Compilation reached operator conversion and failed at `timestep.to(hidden_states.dtype) * 1000`. The BF16 scalar multiply converter attempted to map the tensor dtype to NumPy and raised `TypeError: Unsupported numpy dtype`. No compiled transformer forward, module coverage inventory or full compiled image was produced. The eager image remains functional reference evidence only. No precision fallback, operator exclusion or library-source patch was applied.

## Interpretation

This pinned compiler stack does not pass the BF16 klein feasibility gate with these settings. The failure concerns conversion of the scalar multiply, not inference memory capacity. The bounded screen ends here; these results do not establish whether a different compiler release or explicitly labelled mixed partition could work. No compiler speedup or numerical equivalence is claimed.

## Next experiment

- A separately scoped compiler-version or converter-support investigation would precede further inference benchmarking.

[W&B failed gate](https://wandb.ai/mfaruqi-purdue-university/on-device-diffusion/runs/a100-flux-klein-torchtrt-002__attempt__20261005-081316).
