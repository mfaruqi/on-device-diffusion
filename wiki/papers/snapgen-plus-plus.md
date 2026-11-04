---
type: paper
summary: SnapGen++ — compact three-stage DiT with sparse attention and few-step distillation, exported via Core ML to iPhones; a phone-scale DiT reference, not a FLUX.2 klein optimization.
status: ingested
ref: Hu et al. (Snap Inc.), SnapGen++ — Unleashing Diffusion Transformers for Efficient High-Fidelity Image Generation on Edge Devices, arXiv 2601.08303v3
source: https://arxiv.org/html/2601.08303v3
updated: 2026-10-10
---

# SnapGen++

Sources: [paper v3](https://arxiv.org/html/2601.08303v3).
Review date: 2026-10-10; literature evidence only, no local export or device validation.

## Mechanism and evidence

- A three-stage DiT (Down, Middle, Up) runs its Middle stage on a 2×2-downsampled latent, so most
  blocks see a quarter of the tokens at 1024² (§3.1, [paper](https://arxiv.org/html/2601.08303v3)).
- Adaptive sparse self-attention replaces full attention at high resolution: a global branch with
  convolutionally compressed keys/values plus a blockwise neighbourhood branch, chosen for mobile
  kernel support (§3.1, [paper](https://arxiv.org/html/2601.08303v3)).
- Elastic training slices one 1.6B supernetwork into 0.3B and 0.4B subnetworks; few-step quality comes
  from teacher distillation and a DMD variant (K-DMD) (§3.2–3.3, [paper](https://arxiv.org/html/2601.08303v3)).
- Deployment: Core ML export with attention rewritten as split einsum; the 1.6B model runs only after
  mixed 4/8-bit k-means weight quantization. Per-step latency is reported on iPhone 16 Pro Max,
  iPhone 14 Pro Max and two Samsung phones; Flux.1-dev is OOM on mobile at 1024² (Tables 2–3, App. C,
  [paper](https://arxiv.org/html/2601.08303v3)).

## Interpretation and integration checks

- This is a new, retrained architecture. Like [DreamLite](dreamlite.md), it is a separate phone
  workload and never an optimized FLUX result ([scope](../project/overview.md#fallbacks-and-scope-control)).
- Reported timings are per denoise step plus a VAE estimate; on-device memory, text-encoder time and
  quantization quality loss are not reported. Measure full fresh-prompt pipelines before comparing
  ([paper](https://arxiv.org/html/2601.08303v3), [evaluation rules](../project/overview.md#evaluation-rules)).
- Interpretation: its Core ML attention rewrite and the 4-bit requirement for the large model are
  evidence that kernel support and weight precision constrain what fits on an iPhone, the coupling the
  [registry](../project/overview.md#optimization-families-registry) is meant to record.
- Public weights and export scripts were not verified in this review.
- Scope: phone-target precedent for [RQ1](../rq/rq1.md) stage placement; candidate alternative or
  complement to DreamLite-mobile on the [iPhone](../systems/iphone.md).
