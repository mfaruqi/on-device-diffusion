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

The [complete upstream/BF16 transformer package](../../experiments/m5-flux-klein-shiftadd-020.md)
is saved locally and loads without the original checkpoint. It includes all
169 tensors, with 50 packed and the remainder unchanged. See the
[package guide](upstream-package.md); loading still reconstructs dense weights.

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

## Completed first-block Q/K/V follow-ups

- [Four-plane controls](../../experiments/a100-flux-klein-upstream-qkv-calibration-001.md):
  16 images compare BF16, radix, conventional INT4 and upstream BCQ across four
  prompts. Upstream is closest to BF16 on each; this is not formal quality superiority.
- [Grouped A8](../../experiments/a100-flux-klein-upstream-qkv-a8-001.md):
  12 images and 48 verified hook calls. Two outputs move closer to BF16 and two
  farther away. Only first-block Q/K/V inputs are quantized.
- [Mixed planes](../../experiments/a100-flux-klein-upstream-qkv-mixed-001.md):
  reducing K or V to three planes saves 8.17% of the three-matrix bundle’s file
  size. K3 is the more promising candidate on the two calibration-validation
  prompts; all lemon images still show two lemons rather than three.
- Three-plane [K/V](../../experiments/m5-flux-klein-shiftadd-013.md) and
  [Q](../../experiments/m5-flux-klein-shiftadd-014.md) artifacts have independent
  reload and quantized-activation arithmetic checks. Their per-matrix files are
  24.51% smaller than the four-plane versions.

[Equal-storage Q3/K3/V3 comparison](../../experiments/compare-upstream-qkv-plane-allocation.md)
compares the completed matched runs; Q3 has the lowest latent distance among
those three candidates on all four prompts, without formal quality admission.

## Broader dual-stream results

All 30 image/text Q/K/V matrices across the five dual-stream blocks are now
[fitted and independently verified](../../experiments/m5-flux-klein-shiftadd-015.md).
The [16-image comparison](../../experiments/a100-flux-klein-upstream-all-dual-001.md)
changes the same 30 matrices for radix, INT4 and upstream BCQ. Upstream has lower
BF16 latent distance on all four prompts, with different bit usage and no formal
quality admission. This does not include the 20 single-stream fused projections.

[Grouped A4/A8 projection checks](../../experiments/m5-flux-klein-shiftadd-017.md)
cover all 30 matrices; A8 adds much less local error than A4. The
[five image-query Q3 fits](../../experiments/m5-flux-klein-shiftadd-018.md) are
saved and verified, offering 4.09% lower file storage when combined with the
other 25 four-plane dual-stream matrices. The
[mixed-query image comparison](../../experiments/a100-flux-klein-upstream-all-dual-q3-001.md)
and [all-dual A8 follow-up](../../experiments/a100-flux-klein-upstream-all-dual-a8-001.md)
are complete and verified. Neither establishes a consistent quality advantage.
The A8 lemon image satisfies the requested count despite greater BF16 latent
distance, reinforcing the need for prompt-quality criteria.

[All 20 fused-projection A4/A8 checks](../../experiments/m5-flux-klein-shiftadd-019.md)
are also complete: the A8 error increase is small, while A4 is larger. These
are captured-input numerical diagnostics, not combined-model quality evidence.

## Prepared follow-ups

The [Q3 image comparison](../../experiments/a100-flux-klein-upstream-qkv-q3-001.md)
is complete: all 12 images validated, with lower BF16 latent distance on three
of four prompts than Q4/K4/V4. This is not formal quality superiority. The
[all-attention capture](../../experiments/a100-flux-klein-shiftadd-activation-capture-002.md)
is complete: 800 finite arrays, with exact reproduction of the earlier 32 arrays.
It covers 30 dual-stream image/text Q/K/V matrices and 20 single-stream fused
QKV/MLP projections. The [30 dual-stream fits](../../experiments/m5-flux-klein-shiftadd-015.md) are complete
and independently verified. [All 20 single-stream fits](../../experiments/m5-flux-klein-shiftadd-016.md) are complete and verified. No image comparison combining both the
dual-stream and single-stream upstream bundles exists yet; packed execution
remains unmeasured. Existing four-prompt calibration
data remain separate from formal held-out evaluation.

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
