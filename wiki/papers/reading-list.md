---
type: paper
summary: References from the proposal and related work to read. A paper gets its own page only when it is ingested.
status: unread
ref: proposal references [1]–[25]
source: ../../proposal/OnDeviceDiffusionProposal.pdf
updated: 2026-10-02
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
| Phone workload | DreamLite (Feng et al., 2026), arXiv 2603.28713; DreamLite mobile deployment guide | iPhone/Core ML/MLX reference; cached-prompt Android timing versus fresh-prompt protocol | [source review ingested](dreamlite.md); export/device validation pending, further work temporarily deferred |
| Reuse policies | TeaCache (Liu et al., 2024), arXiv 2411.19108 | approximate cross-step reuse; polynomial calibration and threshold cost | [source review ingested](teacache.md); target integration unverified |
| Reuse policies | DiCache (Bu et al., 2025), arXiv 2508.17356 | online probe and trajectory alignment; second policy | [source review ingested](dicache.md); target integration unverified |
| Precision | SVDQuant (Li et al., ICLR 2025), arXiv 2411.05007 | low-bit diffusion with fused low-rank kernels | [source review ingested](svdquant.md); target integration unverified |
| Compiler | TVM (Chen et al., OSDI 2018); Relax (Lai et al., arXiv 2311.02103) | initial compiler foundation | unread |
| Compiler | TVM MetaSchedule (Shao et al., NeurIPS 2022, arXiv 2205.13603) | composable search, measured-cost ranking and tuning history | [source review ingested](metaschedule.md) |
| Compiler | MLC-LLM compiler passes and packaging; Web Stable Diffusion | compilation, tuning history and memory planning; joint-planner distinction | [source review ingested](mlc-compiler-precedent.md); detailed pass/packaging audit pending |
| Engines | stable-diffusion.cpp; [edge-dit.cpp](https://github.com/THU-MIG/edge-dit.cpp) | engine baselines, auto-fitting | stable-diffusion.cpp in use ([engine](../systems/stable-diffusion-cpp.md)); edge-dit.cpp next for feasibility screening ([memo](../../raw/meetings/2026-10-01-haoran-you.md#to-do)) |
| Engine candidates | [LightX2V](https://github.com/ModelTC/LightX2V) | screen an additional inference framework for supported model/device combinations and integration effort; Weeks 1-2 / RQ1 | suggested candidate; detailed review and local validation pending |
| Engine candidates | [TensorRT diffusion pipeline](https://github.com/NVIDIA/TensorRT/tree/main/demo/Diffusion) | screen NVIDIA inference support, export/build requirements, and device compatibility; Weeks 1-2 / RQ1 | suggested candidate; exact FLUX.2 klein and Jetson support unverified |
| Engine options | [sd.cpp caching documentation](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/caching.md) | audit EasyCache, DBCache, TaylorSeer, cache-dit and Spectrum against the pinned engine/model; measure speed, cache memory and quality | planned option audit ([memo](../../raw/meetings/2026-10-01-haoran-you.md#to-do)); no new measurements |
| Engines | FastVideo; Wan reference implementation | video baselines | unread; FastVideo temporarily deferred ([decision](../project/decisions.md#d-009-prioritize-image-engine-baselines-and-profiling)) |
| Engines | ExecuTorch (Nachin et al., MLSys 2026), official OpenVINO diffusion example | existing diffusion recipes; candidate infrastructure, automatic joint selection unverified | [ingested](executorch-mlsys2026.md) |
| Joint optimization | [CacheQuant](https://arxiv.org/abs/2503.01323) (Liu et al., CVPR 2025) | caching/quantization coupling, cache scheduling and error correction; audit novelty overlap | screened; full review pending |
| Joint optimization | [QuantCache](https://arxiv.org/abs/2503.06545) (Wu et al., 2025) | hierarchical caching, quantization and pruning for video diffusion | abstract screened; full review pending |
| Memory/planning | [Xema](https://arxiv.org/abs/2607.11136) (Kang et al., 2026) | trace-guided diffusion memory control and constrained serving configuration | abstract screened; full review pending |
| Evaluation | GenEval (Ghosh et al., 2023), arXiv 2310.11513 | image alignment metric | unread |
| Evaluation | VBench (Huang et al., 2023), arXiv 2311.17982 | video quality dimensions | unread |
| Memory | Diffusers "Reduce memory usage" docs; CUDA for Tegra memory notes | offload and tiling options; Jetson shared memory | unread |
| Models | FLUX.2 [klein] 4B and Base model cards; Wan 2.1 T2V-1.3B model card | workloads | used ([model](../systems/flux2-klein-4b.md)) |
| Apple | Core ML docs; MLX Swift | iPhone and M1 paths | unread |
