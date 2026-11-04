---
type: method
summary: Prepare the M5 sd.cpp Metal smoke test with pinned mixed-Q4 weights and separate unified-memory accounting.
status: active
updated: 2026-10-11
---

# M5 Metal bring-up

Week 2 Apple setup supporting RQ1 ([milestones](../project/milestones.md#revised-schedule)).
The first [configuration](../../configs/m5-sdcpp-flux-klein-q4-512-smoke.json) is a feasibility gate:
512² distilled klein, four steps, Q4_0 transformer / Q4_K_M text encoder / BF16 VAE.
It is not the BF16 reference or a repeated performance baseline. Metrics follow
[baseline metrics](baseline-metrics.md#macos-metal-sdcpp).

## Environment and build

Run from the repository root, plugged into power. Use one inference process at a time.
The smoke protocol prevents idle/system sleep temporarily with `caffeinate -is`; keep the lid open.
Local source checkouts, builds and weights live under ignored `output/m5-prep/`.
The environment uses Python 3.12 plus `psutil`, `pillow` and `huggingface-hub`;
actual versions are captured in each run's `environment.json`.

```bash
uv venv --python 3.12 .venv-m5
uv pip install --python .venv-m5/bin/python psutil pillow huggingface-hub
mkdir -p output/m5-prep
git clone https://github.com/leejet/stable-diffusion.cpp output/m5-prep/stable-diffusion.cpp
git -C output/m5-prep/stable-diffusion.cpp checkout --detach 19bbbca1c736bbb9538679fc0ae690cb2b46b492
git -C output/m5-prep/stable-diffusion.cpp submodule update --init --recursive
cmake -S output/m5-prep/stable-diffusion.cpp -B output/m5-prep/stable-diffusion.cpp/build -G Ninja \
  -DSD_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON -DGGML_OPENMP=OFF \
  -DSD_BUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_OSX_SYSROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk
cmake --build output/m5-prep/stable-diffusion.cpp/build --target sd-cli -j 6
cmake -S engines/sdcpp -B output/m5-prep/sd-bench-build -G Ninja \
  -DSDCPP_DIR="$PWD/output/m5-prep/stable-diffusion.cpp" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_OSX_SYSROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk
cmake --build output/m5-prep/sd-bench-build -j 4
output/m5-prep/stable-diffusion.cpp/build/bin/sd-cli --list-devices
```

The explicit SDK path avoids the installed default macOS 27 SDK/linker mismatch on the bring-up host.
`GGML_METAL_EMBED_LIBRARY=ON` embeds shader source for runtime Metal compilation in this pinned ggml;
it does not require the standalone `metal` compiler. Shader/pipeline initialization can affect first-use time.
The Apple harness links the upstream shared library; the CUDA build path remains static with NVTX support.
Device enumeration must show `MTL0 Apple M5` before this configuration is used.

## Prepare and execute

Preparation downloads three files at pinned commits, verifies SHA256 bytes and writes a resolved config.
Ordinary HTTPS is selected explicitly because the initial Xet transfer stalled on this host.

```bash
export HF_HOME="$PWD/output/m5-prep/huggingface"
HF_HUB_DISABLE_XET=1 .venv-m5/bin/python scripts/run_sdcpp.py \
  --config configs/m5-sdcpp-flux-klein-q4-512-smoke.json --prepare-only
HF_HUB_OFFLINE=1 .venv-m5/bin/python scripts/run_sdcpp.py \
  --config configs/m5-sdcpp-flux-klein-q4-512-smoke.resolved.json --kind attempt
```

One context runs one first generation and one subsequent measured observation, with no warm-ups.
No cache, automatic placement, segmentation, CPU weight offload or VAE tiling is requested.
The log audit rejects automatic VAE tiling after an allocation failure and unexpected parameter placement.
Keep failed attempts; do not silently change precision, resolution, placement or checkpoint.

A passing smoke permits a separately labelled full protocol. Do not present the single measured
observation's degenerate min/max or determinism flag as established stability. MFLUX requires its
own adapter and model/quantization receipts before an engine comparison.

## Full protocols and PyTorch preparation

The full sd.cpp configurations are [distilled](../../configs/m5-sdcpp-flux-klein-q4-512.json)
and [Base](../../configs/m5-sdcpp-flux-klein-base-q4-512.json). Each keeps one context for one first
image, three discarded warm-ups and ten measured generations. Base uses its own transformer
checkpoint, 50 steps and CFG 4. Prepare each configuration, then run its resolved copy with
`--kind baseline`; never run two inference processes simultaneously.

The [PyTorch MPS runner](../../scripts/run_flux_mps.py) uses a separate `.venv-m5-torch` environment
(torch, diffusers 0.40.0, transformers, accelerate, psutil, pillow, huggingface-hub).
It accepts the BF16 [distilled](../../configs/m5-flux-klein-bf16-512.json) and
[Base](../../configs/m5-flux-klein-base-bf16-512.json) configurations. Use `--prepare-only` online,
then `HF_HUB_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=0` for execution. It retains failures instead of
changing precision, enabling CPU offload or raising the MPS memory limit. Run a separate smoke
before the repeated protocol when bringing up this engine. BF16 and mixed-Q4 are distinct precision
configurations; engine-only speedup claims require matched representations and initial noise.

## Individual optimization queue

The local queue state is `output/m5-prep/optimization-state.json`; its launcher and progress log are `run_individual_options.py` and `optimization-queue.log` in that directory. It shares the prior queue lock and refuses competing inference processes. Each new option runs a two-image smoke before the full 1+3+10 protocol. Failed gates retain evidence and skip the full run. Interrupted entries require process/output reconciliation before resume.

The bounded initial list tests sd.cpp EasyCache (.2/.15/.95), exact conditioning reuse, prefetch-off and mmap separately, plus PyTorch exact conditioning reuse and VAE tiling (256 pixels, 25% overlap), each on distilled and Base. References retain their original no-reuse configuration and precision. Prefetch-off may be inactive under eager GPU residency; setting validation alone does not demonstrate a performance mechanism. PyTorch VAE slicing is not queued because the workload has batch size one. PyTorch EasyCache is not implemented or claimed; compatibility must be established before extending the list.

For edge-dit, the native transformer safetensors file at the same pinned revision is prepared and hash-checked. Earlier Diffusers-layout failures remain recorded. One native smoke retry per workload is allowed, bounded to 15 minutes each; a passing gate is reviewed before a full protocol. This investigation must not block the working engines' option queue indefinitely.

### PyTorch EasyCache support screen (2026-10-10)

The installed Diffusers 0.40.0 source has no `EasyCache`/`easy_cache` implementation. The [official EasyCache repository](https://github.com/H-EmbodVis/EasyCache) currently publishes HunyuanVideo and Wan 2.1/2.2 integrations; it does not document a Flux2Klein MPS integration. A compatible ready-to-run PyTorch path was not established in this screen. The queue therefore leaves PyTorch EasyCache unimplemented and runs its separately named exact-conditioning and VAE-tiling policies. This is a support gap for the pinned stack, not proof that a port is impossible.

### Bounded batch outcome (2026-10-11)

All twelve individual sd.cpp/PyTorch option protocols completed their smoke and full gates; records and raw evidence are in the [registry](../experiments.md). The queue is stopped. The native edge-dit distilled retry passed weight loading but aborted on a Metal `CUSTOM` operation ([record](../../experiments/m5-edgedit-flux-klein-001.md)); Base native preparation remained incomplete after read timeouts ([record](../../experiments/m5-edgedit-flux-klein-base-001.md)). That initial queue scheduled no further retries or option sweeps. Formal quality approval and controlled causal performance comparisons remain outside these validation receipts.


## October 15 missing individual options

The [follow-up manifest and recovery guide](../../output/m5-followup-20261015/README.md) covers transformer compilation and TaylorSeer attention prediction in PyTorch, plus exact prompt-pair reuse and VAE tiling in MFLUX. Completed earlier options are retained without repetition. Each new configuration passes its own smoke gate before the full protocol; the queue shares the existing continuation lock. Compilation diagnostics, actual cache/tile execution, memory pressure and image differences are recorded separately from performance or quality approval.
