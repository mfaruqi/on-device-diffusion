---
type: finding
summary: In FLUX.2 klein the Qwen3 text encoder is half of the BF16 weights (7.49 of 14.87 GiB) but ~4% of warm latency, and is idle during denoise and decode.
status: supported
confidence: high
rq: [RQ3]
sources: [../../experiments/a100-flux-klein-001.md]
updated: 2026-09-28
---

# The text encoder is half the weights for a few percent of the time

**Claim.** The text encoder's weights are **7.49 GiB**
([`components.text_encoder.weights_gib`](../../results/runs/a100-flux-klein-001-20260923-221530/load.json))
of 14.87 GiB. Encoding takes **47.0 ms**
([`measured.text_encode_ms.median`](../../results/runs/a100-flux-klein-001-20260923-221530/summary.json))
of a 1240.6 ms generation.

**Evidence.** [PyTorch reference record](../../experiments/a100-flux-klein-001.md#memory). The weights stay resident
through denoise and decode without being used.

**Interpretation.** Releasing or offloading the encoder after encoding is a memory-residency lever that
costs only a reload. Its memory is never needed at the same time as the VAE peak. This is a candidate
for the joint planner's residency choices (RQ3).

Related: [stage residency](../concepts/stage-residency.md), [RQ3](../rq/rq3.md).
