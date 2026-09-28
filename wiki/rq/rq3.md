---
type: rq
summary: RQ3 — does joint planning with bounded runtime adaptation beat fixed or independently tuned policies. Candidate levers identified, nothing tested.
status: active
updated: 2026-09-28
---

# RQ3: Does joint planning with bounded runtime adaptation outperform fixed or independently tuned policies?

**From the proposal**: "Isolate region selection, memory planning, and runtime guards under matched
checkpoints and quality requirements. Identify where saved computation outweighs cache storage, probe
costs, and runtime overhead." ([overview](../project/overview.md#research-questions))

## Evidence so far
None for the question itself. Measured facts that identify candidate levers for a memory-budgeted plan:
- text-encoder residency: half the weights for a few percent of the time ([finding](../findings/text-encoder-half-of-weights-little-of-time.md));
- VAE decode sets the peak memory ([finding](../findings/vae-decode-sets-peak-memory.md));
- about a third of each denoise step is fusable memory-bound work ([finding](../findings/pytorch-denoise-step-third-in-unfused-memory-bound-ops.md)).

## Gaps
- No reuse policy measured yet ([cross-step reuse](../concepts/cross-step-reuse.md)).
- No memory budget defined per target device.
- The bounded planning hypothesis isn't written yet (week 1 deliverable, [milestones](../project/milestones.md)).

## Next
Write the bounded planning hypothesis. Run single-lever A100 variants (release the text encoder, tiled VAE, `torch.compile`) so their interactions can be measured together later.

Related: [RQ1](rq1.md), [RQ2](rq2.md).
