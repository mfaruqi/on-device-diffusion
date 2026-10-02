---
type: paper
summary: ExecuTorch — deployment infrastructure with existing diffusion export and calibration recipes; automatic joint recipe selection is unverified, and a research contribution requires a closer prior-work audit.
status: ingested
ref: Nachin, Desai, Jia et al., "ExecuTorch: A Unified PyTorch Solution to Run AI Models On-Device", MLSys 2026 (Industry Track)
source: ../../raw/papers/executorch-mlsys2026.pdf
updated: 2026-09-29
---

# ExecuTorch: A Unified PyTorch Solution to Run AI Models On-Device (MLSys 2026)

Sources: [paper PDF](../../raw/papers/executorch-mlsys2026.pdf) ([proceedings](https://proceedings.mlsys.org/paper_files/paper/2026/file/236f915dd02af4f11927f67330b21d4b-Paper-Conference.pdf)); upstream documentation/code reviewed 2026-09-29 below. Paper-era claims and current implementation evidence are distinguished.

## What it is
- `torch.export` captures computation into Export IR; Edge Dialect restricts it to fewer than 300 Core ATen operators with explicit types/layouts. Decomposition simplifies backend coverage but can add execution overhead ([paper](../../raw/papers/executorch-mlsys2026.pdf), §4.1, appendix A).
- `.pte` contains runtime instructions and serialized data, not application C++ source. The C++17 **core runtime** uses caller-supplied memory; extensions/backends are exempt from its allocation restrictions ([paper](../../raw/papers/executorch-mlsys2026.pdf), §§4.5, 5.1).
- Backend selection is explicit during lowering; supported regions become delegate calls. A delegate call can execute multiple kernels ([export guide](https://docs.pytorch.org/executorch/stable/using-executorch-export.html), [paper](../../raw/papers/executorch-mlsys2026.pdf), §4.4).

## Mechanisms relevant here
- **Memory planning:** tensor lifetimes support greedy best-fit arena reuse and custom planners. Reusing dead storage differs from preserving features across diffusion steps ([paper](../../raw/papers/executorch-mlsys2026.pdf), §4.2; [reuse](../concepts/cross-step-reuse.md)).
- **Loading:** large delegate segments can be freed after initialization; page-aligned segments support memory mapping without extra copies ([paper](../../raw/papers/executorch-mlsys2026.pdf), §4.5).
- **Delegation:** backends declare capabilities; unsupported regions can use CPU fallback. Capability matching alone does not demonstrate globally optimal placement ([paper](../../raw/papers/executorch-mlsys2026.pdf), §4.4).
- **Partial delegation:** CPU fallback accounted for about 22–29% of ViT/Swin-T latency, versus 5–6% for CPU↔GPU copies; these are workload-specific results ([paper](../../raw/papers/executorch-mlsys2026.pdf), §11).
- **Inspection:** ETRecord links export graphs/debug metadata to ETDump operator timings, tensor lifetimes and delegate events ([paper](../../raw/papers/executorch-mlsys2026.pdf), §8).

## Their evaluation
- The paper benchmarks LLMs and image classifiers on Galaxy S25 Ultra, Pixel 9 Pro XL and iPhone 15 Pro, against llama.cpp, ONNX Runtime, LiteRT and CoreML; it does not evaluate diffusion ([paper](../../raw/papers/executorch-mlsys2026.pdf), §11).
- LLM protocol: warm-up, three measured runs with min/max throughput, 60-second cooldowns. Vision: ten warm-ups, 200 iterations, mean/p5/p95 latency ([paper](../../raw/papers/executorch-mlsys2026.pdf), §11).
- CoreML delegation approximately matches native CoreML on the measured iPhone classifiers ([paper](../../raw/papers/executorch-mlsys2026.pdf), §11).
- CPU decode linear layers are near parity with llama.cpp; attention performance depends on prefill versus decode, while decomposition adds overhead. Quantization differences limit matched-quality interpretation ([paper](../../raw/papers/executorch-mlsys2026.pdf), §11, appendix A).

## Current diffusion recipes
- The official OpenVINO LCM example exports separate text-encoder, U-Net and VAE-decoder `.pte` files, with optional component-specific quantization ([README](https://github.com/pytorch/executorch/blob/main/examples/openvino/stable_diffusion/README.md)). This establishes existing diffusion deployment, not feasibility for this project's exact workloads.
- Its exporter explicitly says "Collect UNet calibration inputs from prompts." It runs the pipeline, captures latent/timestep/conditioning inputs for U-Net calibration, and uses weight-only compression for other components ([exporter](https://github.com/pytorch/executorch/blob/main/examples/openvino/stable_diffusion/export_lcm.py)).
- **Review boundary:** the inspected example specifies recipes; it does not implement a search jointly selecting cross-step reuse regions, precision and residency under memory and output-quality constraints. This limited source review is not an exhaustive absence claim about ExecuTorch or its ecosystem ([exporter](https://github.com/pytorch/executorch/blob/main/examples/openvino/stable_diffusion/export_lcm.py), [proposed objective](../project/overview.md#planner-objective)).

## What it means for this project
- **Interpretation:** ExecuTorch is a candidate foundation for custom diffusion transformations and a surrounding planner; generic export, delegation and memory planning can be reused. Exact workload export, cache-state interfaces, fusion/copy overhead and delegate-memory visibility require validation ([paper](../../raw/papers/executorch-mlsys2026.pdf), §§4, 8; [proposal](../project/overview.md#system-components)).
- **Hypothesis:** discovering legal reuse boundaries and jointly selecting compatible precision/residency recipes may improve end-to-end latency at matched memory and quality limits. A larger cache could save compute yet force slower residency choices; compare smaller-cache and no-reuse plans ([proposal objective](../project/overview.md#planner-objective), [RQ3](../rq/rq3.md)).
- **Novelty remains unverified:** importing an existing recipe is integration. CacheQuant already describes "jointly optimizing model caching and quantization techniques"; QuantCache combines caching, quantization and pruning, and Xema describes diffusion memory planning. Their full decision spaces need review ([CacheQuant](https://arxiv.org/abs/2503.01323), [QuantCache](https://arxiv.org/abs/2503.06545), [Xema](https://arxiv.org/abs/2607.11136), [reading list](reading-list.md)).
- **Evidence needed:** identify a specific decision existing methods miss; isolate analysis/selection benefits within the same runtime against tuned fixed and independent policies, including preparation costs, copies, probes and total memory ([RQ2](../rq/rq2.md), [proposal evaluation](../project/overview.md#evaluation-rules)).
- **Scope:** Week 1 bounded hypothesis and Week 2 registry design; RQ2/RQ3. No framework switch is decided: TVM Relax remains the proposal's initial foundation and the Apple baseline question remains open ([overview](../project/overview.md), [open question](../open-questions.md#should-executorch-coreml-delegate-be-an-engine-baseline-on-m1iphone)).
- **Related evidence:** the paper's operator breakdown is an external-domain parallel to the project's [GEMM](../findings/bf16-gemm-time-matches-across-pytorch-and-sdcpp.md), [attention](../findings/ggml-flash-attention-4x-slower-than-sdpa-flash-at-4608-tokens.md) and [unfused-op](../findings/pytorch-denoise-step-third-in-unfused-memory-bound-ops.md) observations; it is not diffusion transfer evidence ([paper](../../raw/papers/executorch-mlsys2026.pdf), appendix A).

## Limits
The paper labels CUDA/Metal experimental and notes hardware-specific artifacts/export limitations ([paper](../../raw/papers/executorch-mlsys2026.pdf), §12). Current documentation describes a CUDA backend; backend availability alone does not establish our workload support ([CUDA guide](https://docs.pytorch.org/executorch/stable/backends/cuda/cuda-overview.html)).
