---
type: paper
summary: HeteroMosaic — causal LLM micro-batching and trace-guided GPU/NPU scheduling; related-work precedent for timeline costs, with diffusion and backend transfer unverified.
status: ingested
ref: Jun et al., HeteroMosaic, arXiv 2607.12839v1, MICRO 2026 forthcoming
source: https://arxiv.org/html/2607.12839v1
updated: 2026-10-15
---

# HeteroMosaic

Source: [paper v1, July 14, 2026](https://arxiv.org/html/2607.12839v1),
[PDF](https://arxiv.org/pdf/2607.12839v1). Literature review; no project experiment.

## Mechanism and reported evidence

- Decoder-only LLM prefill is split into causal token micro-batches. Once earlier KV writes finish,
  later attention can overlap earlier post-attention work (§5.2.1, [paper](https://arxiv.org/html/2607.12839v1#S5.SS2.SSS1)).
- A shared-bandwidth roofline bounds GPU/NPU tensor splitting; traces guide allocation, chunk sizes
  and deliberate delays around critical intervals (§§4–5, [paper](https://arxiv.org/html/2607.12839v1)).
- Compatible W4A16 layouts, shared buffers and NPU request coalescing enable the schedule (§5.3,
  [paper](https://arxiv.org/html/2607.12839v1#S5.SS3)).
- Three AMD Ryzen AI systems show average prefill speedups of 2.30×, 1.50× and 1.17× against the
  authors' iGPU baseline; decode is mostly near parity (§6.2, [paper](https://arxiv.org/html/2607.12839v1#S6.SS2)).
- Search takes roughly eight hours offline per model/device configuration. HeteroInfer is recreated;
  llama.cpp uses Q4_K_S rather than identical AWQ (§§5.2.4, 6.2,
  [paper](https://arxiv.org/html/2607.12839v1)).

## Interpretation for this project

- Interpretation: schedule-dependent costs deserve explicit treatment in the planner. The project's
  evidence remains the [Jetson option comparison](../../experiments/compare-jetson-base-combined-reuse.md),
  not the paper's AMD measurements; no cross-device performance finding follows.
- Inference: causal chunking cannot simply be applied to Klein's non-causal image attention; any
  exact decomposition must retain all required keys/values, and successive denoising steps retain
  latent dependencies. Check graph legality before proposing overlap ([causal premise](https://arxiv.org/html/2607.12839v1#S2.SS3),
  [project loop analysis](../project/overview.md#system-components)).
- Hardware context: M5 is a SoC; Jetson's SoM contains a SoC and memory, attached to a carrier board.
  Orin Nano has no DLA ([NVIDIA specifications](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/),
  [module and carrier](https://blogs.nvidia.com/blog/jetson-generative-ai-supercomputer/)).
- Inference: CPU offload on either shared-memory target is not a second physical memory tier.
  Freeing allocations, avoiding copies and controlling file-cache pressure remain distinct choices
  ([Tegra memory guide](https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/index.html#memory-management),
  [M5 target](../systems/macbook-air-m5.md), [residency](../concepts/stage-residency.md)).
- Hypothesis: M5 GPU/Neural Engine overlap would need a supported runtime path and measured handoff
  costs; a Metal build alone does not establish ANE execution ([Core ML compute units](https://developer.apple.com/documentation/coreml/mlcomputeunits),
  [compiled target](../project/overview.md#scope-amendments)).
- Scope: Week 5 edge portability, Weeks 7–8 planning/search budgets, Week 9 transfer; RQ1–RQ3.
  These are design implications, not milestone completion or evidence of diffusion quality
  ([milestones](../project/milestones.md#revised-schedule), [objective](../project/overview.md#planner-objective)).
