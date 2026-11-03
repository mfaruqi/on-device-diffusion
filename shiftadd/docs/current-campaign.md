# Current campaign evidence

All tests below use the pinned distilled Klein checkpoint. Image tests reconstruct
dense BF16 weights. None establishes packed GPU memory, speed or energy savings.
Only calibration prompts have been used; no formal held-out evaluation is complete.

## Saved weights

All 109 transformer matrices have greedy, radix, mixed radix and conventional
INT4 bundles. Greedy full-transformer images fail and are retained as a negative
baseline. Compact radix stores the same decoded weights with less exponent metadata.
See the [verified storage comparison](../../experiments/compare-shiftadd-transformer-storage.md).

Upstream ShiftAddQwen fitting additionally covers two sampled slices and complete
query/fused projections. Its u8/shared-FP16 scales differ from the restricted radix
family. See the [slice](../../experiments/m5-flux-klein-shiftadd-008.md),
[query](../../experiments/m5-flux-klein-shiftadd-009.md) and
[fused](../../experiments/m5-flux-klein-shiftadd-010.md) fitting records.
Complete first-block K/V projections are also saved and independently verified
([K/V record](../../experiments/m5-flux-klein-shiftadd-011.md)). This is not a full-transformer upstream conversion.

## Completed image diagnostics

| Experiment | Evidence |
|---|---|
| Greedy single/full replacements | [001](../../experiments/a100-flux-klein-shiftadd-001.md): full-transformer failure |
| Radix and single-query activation hooks | [002](../../experiments/a100-flux-klein-shiftadd-002.md) |
| Mixed four/five-plane weights | [003](../../experiments/a100-flux-klein-shiftadd-003.md) |
| Four calibration prompts | [004](../../experiments/a100-flux-klein-shiftadd-004.md): mixed closer to BF16 on three of four prompts, without establishing prompt-quality superiority |
| Same-source conventional INT4 control | [uniform](../../experiments/a100-flux-klein-uniform-001.md): competitive with radix |
| All-linear grouped A8/A4 | [006](../../experiments/a100-flux-klein-shiftadd-006.md): 109 modules, four calls each; recognizable one-prompt outputs |
| Upstream query projection | [query](../../experiments/a100-flux-klein-upstream-query-001.md): lower BF16 latent distance than the two single-projection controls |
| Upstream fused projection | [fused](../../experiments/a100-flux-klein-upstream-fused-001.md): higher BF16 latent distance than both controls; all four images retain readable sign text |

## Completed calibration follow-up

Job `12135879` completed all 16 outputs and passed paired numerical validation.
The [four-prompt record](../../experiments/a100-flux-klein-upstream-fused-calibration-001.md)
documents prompt-specific latent rankings and the lemon-count difference. Two
prompts supplied fitting inputs and two are calibration-validation prompts;
none belongs to the reserved held-out evaluation.

## Pending first-block Q/K/V comparison

Job `12138121` was submitted with a five-minute limit using
`configs/a100-upstream-qkv-calibration-001.json`. Each candidate changes all three
Q/K/V matrices in transformer block 0. Four existing calibration prompts compare
BF16, radix, conventional INT4 and upstream BCQ. The upstream bundle assembles
previously verified Q/K/V files without refitting; all three hashes passed transfer.
Submission is a snapshot, not a live scheduler view.

## Execution preparation

- Hash-checked chunked loaders and independent reconstruction checks.
- Grouped A4/A8 CPU sign-plane arithmetic, including arbitrary upstream scales.
- Exact INT8/INT4 mappings for the restricted radix family only.
- Scoped fake-quantization hooks and paired-case replacement guards.
- Low-rank residual diagnostics; these are not a SVDQuant reproduction.
- Timestamped power integration with analytic tests; device capture and
  synchronized inference energy measurements remain unimplemented.

Images, weights and activation captures stay local; small evidence is preserved
under the parent `results/runs/` directory. W&B publications contain diagnostic
results/images, not model weights. See the experiment records for verified links.
