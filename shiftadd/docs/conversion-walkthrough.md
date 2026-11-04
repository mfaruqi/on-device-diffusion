# How FLUX weight conversion works

## Relationship to ShiftAddQwen

The current upstream path calls the actual pinned ShiftAddQwen `bcq.py` and
`gptq_bcq.py` implementations. They are not locally rewritten approximations:
[provenance](../vendor/shiftaddqwen/PROVENANCE.json) records the source commit and
hashes, and [the adapter](../src/shiftadd_flux/upstream_probe.py) checks those hashes
before fitting. Early greedy and restricted-radix experiments are separate controls.

The adapter uses activation ordering, GPTQ error compensation and scale refitting.
The current [fused-matrix config](../configs/upstream-bcq-all-single-attention-001.json)
uses four planes, groups of 128, u8 scale codes with shared FP16 scales,
15 fitting iterations, damping 0.01 and complete output rows. Superblocks have
256 weights; the adapter disables the optional mean term.

This is a partial adaptation, not the complete ShiftAddQwen pipeline:

| Component | FLUX implementation |
|---|---|
| Output-aware GPTQ/BCQ fitting | Reuses the pinned upstream fitter |
| Calibration inputs | Captures BF16 FLUX inputs across four denoising steps |
| Sequential quantized-prefix calibration | Not implemented; captures come from the BF16 model |
| Mixed precision | Fixed-count selection by local fitting error, not end-to-end KLD knapsack |
| End-to-end scale distillation | Not implemented |
| Deployment format | Local packed NPZ arrays, not the upstream GGUF runtime types |
| GPU execution | Dense BF16 reconstruction; upstream CUDA kernels are not ported |

See the [reviewed upstream method](../../wiki/papers/shiftaddqwen.md) and
[mixed-candidate record](../../experiments/m5-flux-klein-shiftadd-022.md).

## Conversion, step by step

1. **Capture inputs.** `capture_pipeline.py` runs BF16 FLUX on calibration prompts;
   `capture.py` saves sampled input vectors for selected linear layers.
2. **Fit the approximation.** `upstream_probe.py` forms `H = X.T @ X / N` from
   fitting inputs and calls the upstream fitter. This weights errors by the
   inputs the layer actually sees, rather than weight error alone.
3. **Pack the result.** Each weight is approximated by a sum of signed plane
   scales. Sign choices become packed bits; u8 scale codes and shared FP16 scales
   are stored alongside them. The adapter checks reconstruction against the
   upstream fitter's output before saving.
4. **Verify independently.** `validation/verify_upstream.py` reloads the artifacts
   and checks reconstruction/error evidence. Calibration-validation calls are
   separate from fitting calls; neither is the final held-out evaluation.
5. **Assemble and package.** `assemble_upstream.py` combines selected matrices.
   `package_upstream.py` includes untouched BF16 tensors so the transformer can
   load without reading the original weight file. `select_mixed_planes.py` is
   an optional experimental three/four-plane selection step.
6. **Load and generate.** `upstream_loader.py` reconstructs dense BF16 weights.
   `calibration_smoke.py` compares images from matched cases. Activation hooks
   simulate low-bit rounding but still execute floating-point matrix products.

The [current campaign](current-campaign.md) covers 50 upstream-converted matrices,
not every transformer matrix. Text encoder and VAE conversion are outside scope.

## Which files are necessary?

The conversion/loading modules, vendor source, package metadata and matching
configs form the working path. Tests and validation protect correctness.
Diagnostics and plots are optional for conversion, but support the recorded
experiments. Reporting is optional for local work. None of these supporting
files is a trained weight file or a GPU kernel.

The [code guide](../src/shiftadd_flux/README.md) lists every Python module.
The [configuration guide](../configs/README.md) explains experiment filename families.

## Exact block selection and names

Indices below are zero-based and refer to the DiT transformer, not the separate
Qwen text encoder. The packaged architecture has five dual-stream blocks and
twenty single-stream blocks.

| Region | Parameters converted | Allocation in latest tests |
|---|---|---|
| Dual-stream blocks 0–4 | `attn.to_q`, `attn.to_k`, `attn.to_v` and `attn.add_q_proj`, `attn.add_k_proj`, `attn.add_v_proj`, each `.weight` | All thirty stay at four planes |
| Single-stream blocks 0–8 and 11 | `attn.to_qkv_mlp_proj.weight` | These ten vary between four, three, two and one plane |
| Single-stream blocks 9, 10, 12–19 | `attn.to_qkv_mlp_proj.weight` | These ten stay at four planes |
| Remaining transformer tensors | Output projections, separate feed-forward weights, embeddings, modulation and normalization parameters | Original BF16 |

In full names, dual-stream parameters start with `transformer_blocks.<index>.`
and single-stream parameters with `single_transformer_blocks.<index>.` Each
single-stream fused matrix has shape 27648×3072 and combines Q, K, V and gated
MLP input projections. The entire matrix is fitted; its MLP rows are included.
Dual-stream Q/K/V matrices have shape 3072×3072. Scope and sizes are recorded in
the [source tensor inventory](../../results/runs/m5-flux-klein-shiftadd-016__attempt__20261103-all-single/source-checkpoint-inventory.json)
and [low-plane config](../configs/a100-upstream-lowplane-fused-001.json).

`p1p4` means the first selected set uses one plane and the other converted
matrices use four. `p2p4` and `p3p4` use the identical selection. They do not
specify activation precision. The current comparison keeps activations BF16.

The selection originated from the fitting-error increase when each fused matrix
changed from four to three planes. Ten was a fixed count, and the ten smallest
increases were selected. The one/two-plane follow-ups retain that selection;
it was not reoptimized for one or two planes. See [selection record](../../experiments/m5-flux-klein-shiftadd-022.md).

## Radix versus the current upstream format

In this project's restricted signed-radix baseline, each group shares a
power-of-two unit `u` and four planes reconstruct as
`u * (s0 + 2*s1 + 4*s2 + 8*s3)`, with each sign `s` equal to -1 or +1.
The binary powers fix the spacing between available values. See
[radix encoder](../src/shiftadd_flux/radix.py).

The latest upstream fits instead use `sum(alpha[p] * sign[p])`, with learned
per-plane/group scales. A group has 128 weights. Scales use u8 codes and a shared
FP16 superblock scale; signs are packed bits. Consequently four planes do not
mean exactly four total bits per weight or a conventional INT4 grid. General
fitted scales need multiplication during reconstruction; the name ShiftAdd does
not make every operation a literal bit shift. See [fitting adapter](../src/shiftadd_flux/upstream_probe.py).

## Actual execution and transfer order

1. Offline fitting runs on the local M5 CPU using PyTorch and the pinned fitter;
   NumPy packs the results into NPZ files.
2. Packed artifacts and code are copied to Gilbreth's filesystem. This is a file
   transfer, not yet a GPU tensor transfer or a public model upload.
3. On the allocated A100, Hugging Face Diffusers loads the original pipeline
   with PyTorch BF16 and moves it to CUDA. The BF16 case is generated first.
4. Before each candidate's four prompts, the loader verifies artifact hashes,
   reconstructs weights using NumPy on the CPU in 128-row chunks, converts each
   chunk to BF16 on CUDA and copies it into the existing parameter tensor.
5. The modified dense weights are reused for that candidate's prompts and
   denoising steps. The runner does not decode sign planes on every step.
6. Before the next candidate, all fifty selected matrices are overwritten to
   prevent values from the previous case leaking into the next comparison.

This path is [PyTorch/Diffusers CUDA](../src/shiftadd_flux/calibration_smoke.py)
with the [CPU reconstruction loader](../src/shiftadd_flux/upstream_loader.py).
These image runs do not execute Nunchaku/SVDQuant, stable-diffusion.cpp, TVM or a
custom packed ShiftAdd GPU kernel. Those are separate baselines or future
integration paths. The complete saved-package CPU loading check is also distinct
from this original-pipeline-and-replacement image runner.

[Size charts and exact accounting](../../experiments/compare-shiftadd-plane-storage.md)
separate smaller stored files from the unchanged calculated dense weight footprint.
