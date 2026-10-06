---
type: paper
summary: References from the proposal and related work to read. A paper gets its own page only when it is ingested.
status: unread
ref: proposal references [1]–[25]
source: ../../proposal/OnDeviceDiffusionProposal.pdf
updated: 2026-10-06
---

# Reading list

From the proposal's references, grouped by the part of the project they inform. When a paper is
ingested (`/wiki-ingest`), it gets its own page in `papers/` and its line here links to that page.
The proposal's Week 1 includes "review DreamLite" ([milestones](../project/milestones.md));
further DreamLite work and FastVideo are temporarily deferred under the
[October 1 priorities](../project/decisions.md#d-009-prioritize-image-engine-baselines-and-profiling).
LightX2V and TensorRT are suggestions for screening from the [meeting memo](../../raw/meetings/2026-10-01-haoran-you.md#to-do),
not selected baselines or verified FLUX.2 klein/A100/Jetson integrations.

| Group | Reference | Why it matters | Status |
|---|---|---|---|
| Phone workload | DreamLite (Feng et al., 2026), arXiv 2603.28713; DreamLite mobile deployment guide | iPhone workload and Core ML/MLX reference; per-stage backend and precision choices | [source review ingested](dreamlite.md); export/device validation pending, further work temporarily deferred |
| Reuse policies | TeaCache (Liu et al., 2024), arXiv 2411.19108 | approximate cross-step reuse | [source review ingested](teacache.md); target integration unverified |
| Reuse policies | DiCache (Bu et al., 2025), arXiv 2508.17356 | interchangeable second reuse policy | [source review ingested](dicache.md); target integration unverified |
| Precision | SVDQuant (Li et al., ICLR 2025), arXiv 2411.05007 | 4-bit diffusion, quantization matched to kernels | [source review ingested](svdquant.md); target integration unverified |
| Compiler | TVM (Chen et al., OSDI 2018); Relax (Lai et al., arXiv 2311.02103) | initial compiler foundation | unread |
| Compiler | TVM MetaSchedule (Shao et al., NeurIPS 2022, arXiv 2205.13603) | persistent tuning database, precedent for history | [source review ingested](metaschedule.md) |
| Compiler | MLC-LLM compiler passes and packaging; Web Stable Diffusion | pass/dispatch/runtime separation; JIT cache | [source review ingested](mlc-compiler-precedent.md) |
| Engines | stable-diffusion.cpp; [edge-dit.cpp](https://github.com/THU-MIG/edge-dit.cpp) | engine baselines, auto-fitting | stable-diffusion.cpp in use ([engine](../systems/stable-diffusion-cpp.md)); edge-dit.cpp next for feasibility screening ([memo](../../raw/meetings/2026-10-01-haoran-you.md#to-do)) |
| Engine candidates | [LightX2V](https://github.com/ModelTC/LightX2V) | screen an additional inference framework for supported model/device combinations and integration effort; Weeks 1-2 / RQ1 | suggested candidate; detailed review and local validation pending |
| Engine candidates | [TensorRT diffusion pipeline](https://github.com/NVIDIA/TensorRT/tree/release/11.2/demo/Diffusion) | screen NVIDIA inference support | screened: removed in TensorRT 11.3 ([changelog](https://github.com/NVIDIA/TensorRT/blob/main/CHANGELOG.md)); never listed FLUX.2; Torch-TensorRT compile gate failed ([record](../../experiments/a100-flux-klein-torchtrt-002.md)) |
| Engine options | [sd.cpp caching documentation](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/caching.md) | audit EasyCache, DBCache, TaylorSeer, cache-dit and Spectrum against the pinned engine/model; measure speed, cache memory and quality | planned option audit ([memo](../../raw/meetings/2026-10-01-haoran-you.md#to-do)); no new measurements |
| Engines | FastVideo; Wan reference implementation | video baselines | unread; FastVideo temporarily deferred ([decision](../project/decisions.md#d-009-prioritize-image-engine-baselines-and-profiling)) |
| Engines | ExecuTorch (Nachin et al., MLSys 2026), official OpenVINO diffusion example | existing diffusion recipes; candidate infrastructure, automatic joint selection unverified | [ingested](executorch-mlsys2026.md) |
| Joint optimization | [CacheQuant](https://arxiv.org/abs/2503.01323) (Liu et al., CVPR 2025) | caching/quantization coupling, cache scheduling and error correction; audit novelty overlap | screened; full review pending |
| Joint optimization | [QuantCache](https://arxiv.org/abs/2503.06545) (Wu et al., 2025) | hierarchical caching, quantization and pruning for video diffusion | abstract screened; full review pending |
| Joint optimization | [Q&C](https://arxiv.org/abs/2503.02508) (ICLR 2026) | interaction of quantization and caching errors; audit overlap with error-budget allocation ([RQ3](../rq/rq3.md)) | screened; full review pending |
| Joint optimization | [MoDiff](https://arxiv.org/abs/2506.22463) (ICML 2025) | quantizing temporal changes with error compensation | screened; full review pending |
| Serving | [DiFlow](https://github.com/diflow-project/diflow) (SOSP 2026) | micro-serving: schedules text encoder, denoiser and VAE as separate nodes across a GPU cluster; cluster-scale counterpart of stage residency | README screened; paper unread |
| Compiler | AutoTVM, arXiv 1805.08166; Ansor, arXiv 2006.06762 | learned cost models for kernel search ("ML for ML"); analogue for RQ2 at plan level ([levels](../concepts/kernel-graph-plan-search.md)) | unread |
| Compiler history | [OctoML/OctoAI](https://www.bigdatawire.com/2024/09/30/octoai-snapped-up-by-nvidia) | TVM-based optimization service (2019–2024), acquired by NVIDIA | screened |
| Apple | [Core AI models](https://github.com/apple/coreai-models); [mflux](https://github.com/filipstrand/mflux) | Core AI exports distilled klein-4B (macOS 27); mflux supports klein-4B and Base on MLX | README screened; not run |
| Engines | [Olive recipes](https://github.com/microsoft/olive-recipes) (ONNX Runtime) | a FLUX.2-Klein-4B recipe exists (Ryzen AI); ONNX Runtime is a fallback for the web target ([D-011](../project/decisions.md#d-011-build-the-compiled-pipeline-mlc-style-on-tvm-with-an-engine-level-planner-fallback-proposed)) | README screened |
| Memory/planning | [Xema](https://arxiv.org/abs/2607.11136) (Kang et al., 2026) | trace-guided diffusion memory control and constrained serving configuration | abstract screened; full review pending |
| Evaluation | GenEval (Ghosh et al., 2023), arXiv 2310.11513 | image alignment metric | unread |
| Evaluation | VBench (Huang et al., 2023), arXiv 2311.17982 | video quality dimensions | unread |
| Memory | Diffusers "Reduce memory usage" docs; CUDA for Tegra memory notes | offload and tiling options; Jetson shared memory | unread |
| Models | FLUX.2 [klein] 4B and Base model cards; Wan 2.1 T2V-1.3B model card | workloads | used ([model](../systems/flux2-klein-4b.md)) |
| Apple | Core ML docs; MLX Swift | iPhone and Mac paths | unread |
