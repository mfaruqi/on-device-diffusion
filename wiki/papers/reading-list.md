---
type: paper
summary: References from the proposal and related work to read. A paper gets its own page only when it is ingested.
status: unread
ref: proposal references [1]–[25]
source: ../../proposal/OnDeviceDiffusionProposal.pdf
updated: 2026-09-28
---

# Reading list

From the proposal's references, grouped by the part of the project they inform. When a paper is
ingested (`/wiki-ingest`), it gets its own page in `papers/` and its line here links to that page.
Week 1 includes "review DreamLite" ([milestones](../project/milestones.md)).

| Group | Reference | Why it matters | Status |
|---|---|---|---|
| Phone workload | DreamLite (Feng et al., 2026), arXiv 2603.28713; DreamLite mobile deployment guide | iPhone workload; cached-prompt Android timing versus fresh-prompt protocol | [source review ingested](dreamlite.md); export/access verification pending |
| Reuse policies | TeaCache (Liu et al., 2024), arXiv 2411.19108 | approximate cross-step reuse; polynomial calibration and threshold cost | [ingested](teacache.md); integration untested |
| Reuse policies | DiCache (Bu et al., 2025), arXiv 2508.17356 | online probe and trajectory alignment; second policy | [ingested](dicache.md); integration untested |
| Precision | SVDQuant (Li et al., ICLR 2025), arXiv 2411.05007 | low-bit diffusion with fused low-rank kernels | [ingested](svdquant.md); target support unverified |
| Compiler | TVM (Chen et al., OSDI 2018); Relax (Lai et al., arXiv 2311.02103) | initial compiler foundation | unread |
| Compiler | TVM MetaSchedule (Shao et al., NeurIPS 2022, arXiv 2205.13603) | composable search, measured-cost ranking and tuning history | [ingested](metaschedule.md) |
| Compiler | MLC-LLM compiler passes and packaging; Web Stable Diffusion | compilation, tuning history and memory planning; joint-planner distinction | [documentation ingested](mlc-compiler-precedent.md); detailed pass/packaging audit pending |
| Engines | stable-diffusion.cpp; edge-dit.cpp | engine baselines, auto-fitting | stable-diffusion.cpp in use ([engine](../systems/stable-diffusion-cpp.md)) |
| Engines | FastVideo; Wan reference implementation | video baselines | unread |
| Engines | ExecuTorch (Nachin et al., MLSys 2026) | PyTorch-native AOT export and runtime; memory planning and backend-delegation precedent | [ingested](executorch-mlsys2026.md) |
| Evaluation | GenEval (Ghosh et al., 2023), arXiv 2310.11513 | image alignment metric | unread |
| Evaluation | VBench (Huang et al., 2023), arXiv 2311.17982 | video quality dimensions | unread |
| Memory | Diffusers "Reduce memory usage" docs; CUDA for Tegra memory notes | offload and tiling options; Jetson shared memory | unread |
| Models | FLUX.2 [klein] 4B and Base model cards; Wan 2.1 T2V-1.3B model card | workloads | used ([model](../systems/flux2-klein-4b.md)) |
| Apple | Core ML docs; MLX Swift | iPhone and M1 paths | unread |
