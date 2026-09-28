---
type: method
summary: How to build stable-diffusion.cpp and its benchmark harness, point it at the pinned weights, and run the protocol on Gilbreth.
status: active
updated: 2026-09-28
---

# Running stable-diffusion.cpp on Gilbreth

How to build stable-diffusion.cpp (sd.cpp), point it at the pinned FLUX.2 [klein] weights, and run
the benchmark protocol on an A100. Cluster basics (Slurm, storage, accounts) are in
[gilbreth-howto.md](gilbreth-howto.md); metric definitions are in [baseline-metrics.md](baseline-metrics.md).

## What gets built, and where

| Item | Location | In git? |
|---|---|---|
| sd.cpp source + build (library, `sd-cli`) | `~/tools/stable-diffusion.cpp`, `build/` | no (upstream repo, pinned by commit) |
| Benchmark harness source | `engines/sdcpp/bench.cpp`, `engines/sdcpp/CMakeLists.txt` | yes |
| Harness binaries `sd-bench`, `sd-bench-nvtx` | `~/tools/sd-bench-build/` | no |
| Weights | `$HF_HOME` cache on scratch, pinned snapshot `e7b7dc27…` | no |

The pinned versions are in `configs/a100-sdcpp-flux-klein-bf16.json → engine`. `run_sdcpp.py` refuses
to run if the checkout's commit (or its ggml submodule's) doesn't match the config.

## 1. Build sd.cpp (login node, no GPU needed, ~25 min)

Compiling CUDA code doesn't need a GPU, only `nvcc`, which is on the login node's PATH.

```bash
mkdir -p ~/tools && cd ~/tools
git clone https://github.com/leejet/stable-diffusion.cpp
cd stable-diffusion.cpp
git checkout master-919-19bbbca              # tag = config engine.tag
git submodule update --init --recursive      # ggml and third-party libraries
cmake -B build -DSD_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=80 -DCMAKE_BUILD_TYPE=Release
cmake --build build -j16
```

`CMAKE_CUDA_ARCHITECTURES=80` compiles kernels for the A100 only (compute capability 8.0), which
keeps the build time down. A binary built this way does not run on other GPU generations.

## 2. Build the harness (~1 min)

`sd-cli` reloads the model on every call and computes the text conditioning once per
`--batch-count`, so it can't run the warm-up/measured protocol. The harness loads the model once
and calls the library's `generate_image` repeatedly. It links the static libraries from step 1,
so it runs the same code as `sd-cli`.

```bash
cd ~/on-device-diffusion
cmake -S engines/sdcpp -B ~/tools/sd-bench-build -DSDCPP_DIR=$HOME/tools/stable-diffusion.cpp -DCMAKE_BUILD_TYPE=Release
cmake --build ~/tools/sd-bench-build -j8
```

## 3. Weights (login node)

sd.cpp reads the same pinned Hugging Face snapshot as the PyTorch runner:

| sd.cpp input | File in the pinned snapshot | Notes |
|---|---|---|
| `--diffusion-model` | `flux-2-klein-4b.safetensors` | single-file transformer. Verified value-identical to `transformer/` (see below) |
| `--llm` | `text_encoder/` (2 shards + `model.safetensors.index.json`) | sd.cpp reads the shard index directly |
| `--vae` | `vae/diffusion_pytorch_model.safetensors` | sd.cpp converts the diffusers tensor names |

```bash
PY=~/.conda/envs/2025.06-py313/flux_env/bin/python
export HF_HOME=/scratch/gilbreth/mfaruqi/huggingface
$PY scripts/run_sdcpp.py --config configs/a100-sdcpp-flux-klein-bf16.json --prepare-only
#   -> downloads flux-2-klein-4b.safetensors (7.2 GiB) and writes *.resolved.json with every file's sha256
```

Equivalence of the single-file transformer with the diffusers one (row-hash multiset check, ~8 min):

```bash
S=$HF_HOME/hub/models--black-forest-labs--FLUX.2-klein-4B/snapshots/e7b7dc27f91deacad38e78976d1f2b499d76a294
$PY scripts/check_weight_equivalence.py --a $S/transformer/diffusion_pytorch_model.safetensors \
    --b $S/flux-2-klein-4b.safetensors --out configs/sdcpp-weights.check.json
```

Result: `equivalent: true`. There are 3,875,544,576 BF16 values on both sides, stored as 169 tensors
(diffusers) and 149 tensors (single file, q/k/v fused).

## 4. Run

```bash
mkdir -p logs
sbatch scripts/gilbreth-sdcpp-smoke.slurm                    # optional: one image with sd-cli, verbose log
sbatch scripts/gilbreth-sdcpp.slurm                          # baseline: 1 first + 3 warm-up + 10 measured
sbatch --export=ALL,PROFILE=1 scripts/gilbreth-sdcpp.slurm   # separate Nsight Systems run
```

The job log prints the run directory. It has the same files as a PyTorch run (see
[results/README.md](../../results/README.md)), plus `engine/`:

| File | Contents |
|---|---|
| `engine/command.json` | exact harness command line |
| `engine/sdcpp.log` | every sd.cpp log line with a timestamp; the source of `summary.json → engine_audit` |
| `engine/results.jsonl` | raw stage timestamps per generation |
| `engine/stdout.txt` | harness stdout/stderr |

## Reference settings and why each one is set

sd.cpp enables several automatic features by default. The reference run turns each one off, and
the wrapper fails the run if the log shows any of them happened.

| Setting | Default | Reference | Why |
|---|---|---|---|
| `--backend` / `--params-backend` | auto | `cuda0` / `cuda0` | all compute and all weights on the GPU |
| `--auto-fit` | on | off | auto-fit places weights by free memory (GPU, RAM or disk) |
| segmented compute | automatic when needed | disabled | fail rather than silently split graphs |
| `--eager-load` | lazy | eager | otherwise the first generation includes the weight transfer |
| `--conditioning-cache-size` | 4 | 0 | the same prompt would skip text encoding in warm runs |
| `--diffusion-fa` | off | on | flash attention in the transformer, as in the PyTorch run |
| `--offload-to-cpu`, `--vae-tiling`, `--cache-mode` | off | off | not part of the reference |
| `GGML_CUDA_CUBLAS_COMPUTE_TYPE` | unset | must be unset | overrides the GEMM compute precision |
