---
type: experiment-record
id: a100-flux-klein-torchtrt-001
status: failed
device: a100-pcie-40gb
engine: pytorch-diffusers
runs: [a100-flux-klein-torchtrt-001__attempt__20261005-074014]
updated: 2026-10-05
---

# A100 Torch-TensorRT: strided-input feasibility failure

Status: **failed**, Slurm11881410;11:40:14–11:43:38UTC.

## Question

Can Torch-TensorRT export, compile and run a real distilled klein transformer input at BF16? Week2 engine screening, RQ1. This is a bounded feasibility gate, not repeated inference timing.

## Setup

[Config](../results/runs/a100-flux-klein-torchtrt-001__attempt__20261005-074014/config.json), [environment](../results/runs/a100-flux-klein-torchtrt-001__attempt__20261005-074014/environment.json). Isolated torch2.5.1+cu124,Torch-TensorRT2.5.0,TensorRT10.3.0. Pinned distilled klein checkpoint e7b7dc27f91deacad38e78976d1f2b499d76a294,BF16,1024×1024,four steps,CFG1,seed0. Same-environment eager image generated first. Five real first-transformer inputs captured; no contiguous-layout conversion in this attempt. Mixed compiler partitions allowed; success requires actual TRT runtime modules and finite same-shape/dtype output. Timeout1200s.

## Results

[Status and traceback](../results/runs/a100-flux-klein-torchtrt-001__attempt__20261005-074014/status.json), [load](../results/runs/a100-flux-klein-torchtrt-001__attempt__20261005-074014/load.json), [inputs](../results/runs/a100-flux-klein-torchtrt-001__attempt__20261005-074014/inputs.json). Eager image and graph export completed. Context/pipeline loading took 16.562869s; export took 6.800670s. These are one-off preparation observations, not medians.

Compilation failed in Input.from_tensor while preparing arguments: `Tensor does not have a supported memory format, supported formats are contiguous or channel_last`. No compiled runtime module inventory, compiled forward or full compiled image was produced. The failed phase is compile; precision and model were not changed. Original output hashes are saved in provenance.json.

## Interpretation

The failure establishes an input-layout incompatibility in this compiler entry point, not that klein operators are unsupported. Installed _Input.py validates contiguous or channels-last input. A separate attempt can make explicit contiguous copies, verify input equality and compare compiled output against the original-layout eager output. Full-pipeline conversion overhead would still require measurement. This environment uses cu124, so any eventual timing comparison requires its own eager control.

## Next experiment

- configs/a100-flux-klein-torchtrt-contiguous-smoke.json: bounded contiguous-input retry, Slurm11883257 after11882497.

[W&B failed gate](https://wandb.ai/mfaruqi-purdue-university/on-device-diffusion/runs/a100-flux-klein-torchtrt-001__attempt__20261005-074014).
