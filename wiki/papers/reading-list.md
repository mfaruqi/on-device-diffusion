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
| Phone workload | DreamLite (Feng et al., 2026), arXiv 2603.28713; DreamLite mobile deployment guide | iPhone workload and Core ML/MLX reference; per-stage backend and precision choices | unread (**week 1**) |
| Reuse policies | TeaCache (Liu et al., 2024), arXiv 2411.19108 | approximate cross-step reuse | unread |
| Reuse policies | DiCache (Bu et al., 2025), arXiv 2508.17356 | interchangeable second reuse policy | unread |
| Precision | SVDQuant (Li et al., ICLR 2025), arXiv 2411.05007 | 4-bit diffusion, quantization matched to kernels | unread |
| Compiler | TVM (Chen et al., OSDI 2018); Relax (Lai et al., arXiv 2311.02103) | initial compiler foundation | unread |
| Compiler | TVM MetaSchedule (Shao et al., NeurIPS 2022, arXiv 2205.13603) | persistent tuning database, precedent for history | unread |
| Compiler | MLC-LLM compiler passes and packaging; Web Stable Diffusion | pass/dispatch/runtime separation; JIT cache | unread |
| Engines | stable-diffusion.cpp; edge-dit.cpp | engine baselines, auto-fitting | stable-diffusion.cpp in use ([engine](../systems/stable-diffusion-cpp.md)) |
| Engines | FastVideo; Wan reference implementation | video baselines | unread |
| Engines | ExecuTorch (Nachin et al., MLSys 2026) | PyTorch-native AOT export and runtime; memory planning and backend-delegation precedent | [ingested](executorch-mlsys2026.md) |
| Evaluation | GenEval (Ghosh et al., 2023), arXiv 2310.11513 | image alignment metric | unread |
| Evaluation | VBench (Huang et al., 2023), arXiv 2311.17982 | video quality dimensions | unread |
| Memory | Diffusers "Reduce memory usage" docs; CUDA for Tegra memory notes | offload and tiling options; Jetson shared memory | unread |
| Models | FLUX.2 [klein] 4B and Base model cards; Wan 2.1 T2V-1.3B model card | workloads | used ([model](../systems/flux2-klein-4b.md)) |
| Apple | Core ML docs; MLX Swift | iPhone and M1 paths | unread |
