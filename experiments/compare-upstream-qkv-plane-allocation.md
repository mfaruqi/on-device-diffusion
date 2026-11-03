---
type: experiment-record
id: compare-upstream-qkv-plane-allocation
status: complete
device: a100-pcie-40gb
engine: pytorch-diffusers
runs: [a100-flux-klein-upstream-qkv-mixed-001__attempt__20261103-calibration, a100-flux-klein-upstream-qkv-q3-001__attempt__20261103-calibration]
updated: 2026-11-03
---

# Compare: equal-storage first-block Q/K/V plane allocation

## Question

Week 5 precision variants, RQ1/RQ3: at equal packed weight storage, which of Q,
K or V is the more promising matrix to reduce from four to three upstream BCQ
planes in this first-block calibration diagnostic?

## Runs

| Run and executed configuration | Device | Cases |
|---|---|---|
| [K3/V3 evidence](../results/runs/a100-flux-klein-upstream-qkv-mixed-001__attempt__20261103-calibration/), [config](../results/runs/a100-flux-klein-upstream-qkv-mixed-001__attempt__20261103-calibration/config.json) | A100 SXM4, gilbreth-n012 | BF16, Q4/K4/V4, Q4/K3/V4, Q4/K4/V3 |
| [Q3 evidence](../results/runs/a100-flux-klein-upstream-qkv-q3-001__attempt__20261103-calibration/), [config](../results/runs/a100-flux-klein-upstream-qkv-q3-001__attempt__20261103-calibration/config.json) | A100 SXM4, gilbreth-n012 | BF16, Q4/K4/V4, Q3/K4/V4 |

Full records: [K3/V3](a100-flux-klein-upstream-qkv-mixed-001.md),
[Q3](a100-flux-klein-upstream-qkv-q3-001.md).

## Matching

| Setting | Held fixed |
|---|---|
| Checkpoint | Distilled Klein revision e7b7dc27f91deacad38e78976d1f2b499d76a294 |
| Weight scope | Exactly first-block Q/K/V; all other weights unchanged |
| Fit method | Upstream BCQ, u8/shared-FP16 scales, group 128, same calibration split |
| Inputs and execution | BF16 activations; dense BF16 reconstruction |
| Workload | 512×512, four steps, guidance 1, batch one |
| Prompts and RNG | Same four prompts and CUDA seeds 17/23/31/47 |
| Protocol | One image per case/prompt; no repeated performance protocol |
| Runtime | Torch 2.5.1+cu121 on the same GPU model/node |

The allocation of three versus four planes across Q/K/V is the variable.
The shared BF16 and four-plane reference latents reproduce exactly, and their
PNGs match byte-for-byte. Initial-noise tensors were not separately saved;
matching uses the same CUDA RNG setup and exact reference reproduction.

## Results

Relative latent L2 versus BF16, read from the source summaries:

| Prompt | Q4/K4/V4 | Q3/K4/V4 | Q4/K3/V4 | Q4/K4/V3 |
|---|---:|---:|---:|---:|
| cal01 | 0.0354433 | 0.0435952 | 0.0801918 | 0.0669129 |
| cal02 | 0.0898966 | 0.0764791 | 0.0921766 | 0.1750685 |
| cal03 | 0.1460473 | 0.1346028 | 0.1474769 | 0.2248293 |
| cal04 | 0.1423134 | 0.1034672 | 0.1139243 | 0.3000044 |

Each three-plane allocation uses 14,085,204 packed file bytes, saving 1,253,376
bytes (8.17%) against the 15,338,580-byte Q4/K4/V4 bundle. This is the same
three-matrix file-storage metric in each case, not GPU memory. No measured
latency, memory-residency or energy comparison is available.

## Interpretation

Q3/K4/V4 has the lowest BF16 latent distance among the three equal-storage
allocations on all four prompts. It is therefore the leading allocation to
investigate within this bounded candidate set. It also beats the larger Q4/K4/V4
reference on three prompts, demonstrating non-monotonic final latent error.
Two prompts supplied fitting inputs; the other two are calibration-validation.

## Limits

These results cover one block and four prompts, not a general precision policy.
Latent closeness is not image quality admission: all lemon images show two lemons
instead of three. The runs do not measure packed GPU execution and cannot show
speed or energy advantages. No held-out quality tests were performed.
