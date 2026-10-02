---
type: paper
summary: MLC-LLM and Web Stable Diffusion documentation — compilation, tuning history and memory-planning precedents; the proposed joint diffusion planner remains an untested contribution.
status: ingested
ref: MLC AI, MLC-LLM Compile Model Libraries and Web Stable Diffusion project documentation
source: https://github.com/mlc-ai/web-stable-diffusion
updated: 2026-09-28
---

# MLC: compiler and runtime precedent

Sources reviewed on 2026-09-28: [MLC-LLM compilation documentation](https://llm.mlc.ai/docs/compilation/compile_models.html)
and [Web Stable Diffusion README](https://github.com/mlc-ai/web-stable-diffusion#how).
This is a documentation review, not a benchmark or an exhaustive audit of either implementation.

## Established capabilities

- **MLC-LLM** compiles a model library containing inference logic for a target platform using TVM.
  Architecture, quantization, platform and metadata affecting memory planning specify the library;
  the Python API can JIT-compile it when no library is supplied ([compilation docs](https://llm.mlc.ai/docs/compilation/compile_models.html)).
- **Web Stable Diffusion** imports Stable Diffusion v1.5 components into TVM IRModule, transforms them
  and generates code for native GPU or WebGPU execution ([README](https://github.com/mlc-ai/web-stable-diffusion#how)).
- It uses TensorIR and MetaSchedule for device-tuned programs, preserves transformations in a database
  so later builds need not retune, and implements static memory planning to reuse storage across layers
  ([README](https://github.com/mlc-ai/web-stable-diffusion#how)).
- The source explicitly describes "static memory planning optimizations" and "a database that records
  these transformations"; these are existing precedents, not proposed contributions here
  ([README](https://github.com/mlc-ai/web-stable-diffusion#how)).

## Interpretation for this proposal

- The import → transform → specialize → runtime architecture is a useful precedent for the proposed
  TVM Relax pipeline. Diffusion compilation, target specialization, tuning history and memory reuse
  alone do not distinguish this project ([MLC source](https://github.com/mlc-ai/web-stable-diffusion#how),
  [proposed components](../project/overview.md#system-components)).
- The proposed distinction is joint selection of diffusion reuse regions, supported precision
  configurations and stage residency, minimizing latency within a memory budget and a quality-eligible
  plan set. Estimates include caches, workspaces and loading transitions; combinations require measured
  cost and quality evidence ([planner objective](../project/overview.md#planner-objective)).
- **Hypothesis:** a reuse choice that saves compute can consume enough cache memory to force costly
  offloading; selecting reuse, precision and residency together may avoid that tradeoff. No reuse must
  remain a candidate when its alternatives cost more than they save
  ([proposal](../project/overview.md#planner-objective), [current evidence](../rq/rq3.md#evidence-so-far)).
- The reviewed MLC sources establish compiler and memory-planning precedents; they do not establish
  the proposal's quality-constrained joint search. This limited review does not prove that all MLC/TVM
  work lacks such mechanisms ([MLC-LLM docs](https://llm.mlc.ai/docs/compilation/compile_models.html),
  [Web SD source](https://github.com/mlc-ai/web-stable-diffusion#how)).

## Evidence still required

- **RQ2:** compare rules, measured lookup and learned ranking at equal profiling budgets; account for
  selection cost and constraint violations ([RQ2](../rq/rq2.md)).
- **RQ3:** demonstrate a benefit over tuned fixed and independently tuned policies, retaining existing
  engine auto-fitting and adaptive caching in the relevant baselines. The proposal calls for identifying
  "where saved computation outweighs cache storage, probe costs, and runtime overhead"
  ([RQ3](../rq/rq3.md), [evaluation rules](../project/overview.md#evaluation-rules)).
- This review informs the **Week 1 bounded planning hypothesis** and subsequent registry/planner design;
  it adds no measured performance result or completed planner milestone
  ([milestones](../project/milestones.md), [RQ3 evidence](../rq/rq3.md#evidence-so-far)).
