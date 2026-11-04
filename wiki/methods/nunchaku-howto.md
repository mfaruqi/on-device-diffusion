---
type: method
summary: Candidate Nunchaku SVDQuant timing on A100; pinned community Klein adapter and explicit Jetson/Metal blockers.
status: active
updated: 2026-10-11
---

# SVDQuant / Nunchaku screening and candidate runner — 2026-10-11

Week 2 engine coverage, RQ1; potential precision candidate for the later planner. Job 11929464 passed smoke and full protocol on an A100 PCIe 40 GB after a narrow dtype-validation repair. Outputs and original failure evidence are local; fixed-seed output is not bitwise deterministic and formal quality is unassessed ([record](../../experiments/a100-nunchaku-flux-klein-001.md)). Live paths are in the [queue state](../../output/nunchaku-prep/state.json).

## What is supported

| Target | Reviewed implementation status | Runner action |
|---|---|---|
| A100 40 GB, sm_80 | Community Klein 4B INT4/r32 checkpoint + x86 wheel exist; smoke and full protocol passed | Pinned distilled workload validated; quality unassessed |
| Jetson Orin Nano, sm_87 | Not in the reviewed upstream or community wheel architecture list; Spark ARM64 is not proof of Orin support | Reject before model loading |
| M5 Metal/MPS | Reviewed Nunchaku implementation requires CUDA, no Metal kernels | Reject before model loading |
| Klein Base | Exact matching community checkpoint not screened | Reject; do not substitute distilled weights |

SVDQuant is a quantization method; DeepCompressor prepares compressed weights, and Nunchaku supplies fused inference kernels. The primary path is W4A4 plus a high-precision low-rank branch. Paper §E.6 evaluates 3-bit variants and reports failure below that in its tested settings (including W2A4/W4A2 ternary quantization). This is not a working ternary engine. Three-value weights have log2(3) ≈ 1.585 bits of information, not one bit.

## Pinned evidence

- [Requested fork](https://github.com/dbw6/svdquant): earlier Nunchaku fork; use upstream/community implementation evidence rather than assuming it provides additional hardware backends.
- [SVDQuant paper, §4 and §E.6](https://arxiv.org/html/2411.05007v4).
- [Upstream setup.py at reviewed commit](https://github.com/nunchux-ai/nunchaku/blob/302e0e97024ebd68688fe890e5df83731edf7b54/setup.py): listed architectures omit sm_87. [Installation](https://nunchaku.tech/docs/nunchaku/installation/installation.html).
- [Klein PR #926](https://github.com/nunchux-ai/nunchaku/pull/926) was open/unmerged; [adapter inspected at its head](https://github.com/tonera/nunchaku/blob/0c8379d10b4b9017928d02fd0fdb2c26b991a8b6/nunchaku/models/transformers/transformer_flux2.py).
- [Community runtime and wheel matrix](https://huggingface.co/tonera/vitoom-nunchaku/blob/85bad5510d409570527b018387491fd0c81e4727/README.md). The wheel is a separate community artifact, not claimed identical to the inspected PR source. Its immutable repository revision and SHA256 are in the config; installed archive provenance and API/module checks are enforced.
- [Community distilled 4B checkpoint](https://huggingface.co/tonera/FLUX.2-klein-4B-Nunchaku/tree/8aa64445d8e45edc2ea431743b67ac7e2ca5f739). Exact ancestry to our pinned BF16 transformer revision is not established; label as a replacement checkpoint plus kernel implementation.

## Files and protocol

- [Runner](../../scripts/run_nunchaku.py): `--device a100|jetson|m5`, `--check-support`, `--prepare-only`; no automatic backend/precision selection.
- [Smoke config](../../configs/a100-nunchaku-flux-klein-int4-r32-smoke.json): first + one measured, 1024², four steps, CFG 1, original A100 reference prompt/seed; unchanged BF16 text encoder/VAE.
- [Full config](../../configs/a100-nunchaku-flux-klein-int4-r32.json): first + three warmups + ten measured; requires a matching complete smoke with nonflat saved images. `RUN_FULL_AFTER_SMOKE=1` runs both sequentially in one allocation and stops on any smoke failure.
- [Slurm entrypoint](../../scripts/nunchaku.slurm): one A100 PCIe 40 GB on account you139; no inference on login node.
- [CPU tests](../../tests/test_nunchaku.py): pinned config checks and rejection of unsupported hardware, precision, offload, Base substitution and unpinned inputs.

The runner reuses `StageRecorder`, `generate_once`, `measure_generations`, `build_summary`, phase bookkeeping and event export. CUDA events define stage durations; host total includes final synchronization. Transformer hooks count logical denoising steps. Native allocations may escape PyTorch allocator accounting; NVML samples remain separately labelled. See [metric definitions](baseline-metrics.md).

## Preparation and launch (A100 host)

Use a dedicated Python 3.11 environment with PyTorch 2.11/cu128; keep the BF16 reference environment intact. Dependencies also include Diffusers with Flux2KleinPipeline, transformers, accelerate, huggingface_hub, safetensors, psutil and Pillow. Diffusers 0.40.0 with Transformers 5.19.0 passed the recorded GPU smoke and full protocol. The current BF16 environment must not be upgraded in place.

```sh
# Read-only support checks; these work on the Mac without Torch/Nunchaku.
python3 scripts/run_nunchaku.py --device a100 --check-support
python3 scripts/run_nunchaku.py --device jetson --check-support
python3 scripts/run_nunchaku.py --device m5 --check-support

# On a network-enabled host, using the dedicated environment's interpreter:
"$NUNCHAKU_PYTHON" scripts/run_nunchaku.py --device a100 \
  --config configs/a100-nunchaku-flux-klein-int4-r32-smoke.json --prepare-only
# The command prints the verified wheel path. Install that exact local wheel:
"$NUNCHAKU_PYTHON" -m pip install /path/printed/by/prepare/nunchaku.whl

mkdir -p logs
sbatch --export=ALL,NUNCHAKU_PYTHON="$NUNCHAKU_PYTHON",RUN_FULL_AFTER_SMOKE=1 scripts/nunchaku.slurm

# Alternatively, submit only smoke (omit RUN_FULL_AFTER_SMOKE), then launch full manually:
sbatch --export=ALL,NUNCHAKU_PYTHON="$NUNCHAKU_PYTHON",CONFIG=configs/a100-nunchaku-flux-klein-int4-r32.json,SMOKE_RUN=results/runs/ACTUAL_SMOKE_DIRECTORY scripts/nunchaku.slurm
```

Preparation downloads the pinned quantized transformer (~2.45 GB), pinned wheel (~167 MB) and needed official pipeline components. It skips the original BF16 transformer weights but keeps its architecture config. Jobs use local cache only. Files already present in the cache are reused. No automatic cache, offload, FP4, text-encoder quantization or model substitution is enabled.

Before claiming speedup, repeat the BF16 reference with the compatible software environment and the same A100 GPU class, preserve images, and compare quality separately. Passing a smoke is functional admission, not formal quality acceptance or proven acceleration. Jetson needs a kernel/build port and a separate memory plan; M5 needs a Metal implementation or a separately named quantization backend such as an MLX approach—not a Nunchaku-equivalent result by assumption.

## Portability preflight (2026-10-11)

The pinned source's `get_sm_targets()` function was executed on the Orin without installation, compilation or model inference. It raises `AssertionError: Unsupported SM 87`; the device reports PyTorch 2.8.0/CUDA 12.6 ([receipt](../../output/nunchaku-prep/portability/jetson-build-preflight.txt), [pinned build logic](https://github.com/tonera/nunchaku/blob/0c8379d10b4b9017928d02fd0fdb2c26b991a8b6/setup.py)). This proves rejection by the default build path, not impossibility of an SM87 kernel port. No existing Jetson environment was modified.

The resident A100 configuration is not a feasible physical-memory plan for Orin: the unchanged BF16 text encoder alone occupies 7.49 GiB ([component accounting](../../results/runs/m5-flux-klein-001__baseline__20261010-183446/load.json)), before the quantized transformer, VAE, activations and OS. CPU offload uses the same physical memory on Orin. A Jetson attempt needs separately labelled and validated text-encoder quantization or disk/stage residency in addition to a working kernel build; changing the GPU architecture allowlist alone does not solve this.

The pinned build uses `CUDAExtension` and CUDA source files; no Metal target appears in its build path ([source](https://github.com/tonera/nunchaku/blob/0c8379d10b4b9017928d02fd0fdb2c26b991a8b6/setup.py)). M5 execution would require implementing the low-bit and low-rank operations on Metal and validating checkpoint packing and numerics. No M5 package installation or model run was attempted.
