---
type: experiment-record
id: a100-flux-klein-shiftadd-005
status: complete
device: a100-sxm4-40gb
engine: pytorch-diffusers
runs: [a100-flux-klein-shiftadd-005__attempt__20261102-activation-capture]
updated: 2026-11-02
---

# a100-flux-klein-shiftadd-005: bounded BF16 activation capture

Job 12133795 completed on gilbreth-n012 with Torch 2.5.1+cu121. Week 5
precision calibration, RQ1/RQ3. Pinned distilled Klein revision
`e7b7dc27f91deacad38e78976d1f2b499d76a294`, BF16 execution, four calibration
prompts, 512-square, four steps and guidance 1.

[Config and summary](../results/runs/a100-flux-klein-shiftadd-005__attempt__20261102-activation-capture/)
record exact prompts, seeds and modules. Query projection
`transformer_blocks.0.attn.to_q` and fused projection
`single_transformer_blocks.0.attn.to_qkv_mlp_proj` each yielded 16 calls with
32 evenly spaced sampled token vectors and 3072 channels per call.
All sampled values were finite. Four calls correspond to each prompt in order.
The FP32 files preserve copies of BF16-origin inputs.

The capture hook makes CPU transfers, so this run is not inference timing or
energy evidence. No compressed weights or activation quantization were applied.
The activation file remains local with its path, size and hash in
`activation-artifact.json`; it is not committed to git. No held-out prompts used.
