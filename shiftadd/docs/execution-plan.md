# Initial execution campaign

Scope: Week 5 compatible quantized variants, RQ1/RQ3. Source is distilled
FLUX.2 Klein 4B revision e7b7dc27f91deacad38e78976d1f2b499d76a294.

1. Convert and verify all 109 matrix weights with four residual sign planes,
   groups of 128, one int8 exponent per group/plane. Retain all other tensors.
2. Paired CUDA latent smoke: identical prompt and seeded initial noise,
   BF16 reference, one reconstructed attention projection, then all matrices.
   Cases accumulate only because the all-matrix case replaces every matrix
   changed by the one-projection case. This is not a timing baseline.
3. Decode images and examine calibration-only diagnostics before expanding.
4. Compare groupwise conventional 4-bit fitting and an improved ShiftAdd fit.
   The initial greedy fit is not GPTQ or output-aware calibration.
5. Activation simulation: symmetric per-token INT8, then INT4, with explicit
   clipping and scales. Compare dense fake-quantized execution numerically;
   do not label it integer-kernel throughput.
6. Arithmetic reference: unpack signs, accumulate signed integer activations,
   apply power-of-two exponents and activation scales. Compare against dense
   reconstructed multiplication. This establishes algebra, not performance.

## Storage versus runtime

Packed payload per weight is Q * (1 + 8/128) bits with int8 exponents.
Four planes therefore use 4.25 bits/weight, before file headers and unchanged
weights. A runtime keeping both packed and decoded weights can use more memory
than BF16. Report packed disk size, resident packed weights, decoded buffers,
activations and workspaces separately.

## Energy and timing

No speed or energy saving follows from storage ratios. A100 sampling must include
power timestamps over repeated steady generations; Jetson must capture board
power and shared physical memory with fixed power mode. Report joules/image and
measurement boundary. Conversion and calibration are outside inference timing.
Reconstructed BF16 execution is a quality tool; custom packed-kernel work is not
required in this initial campaign.

## Implemented preparation

`activation_hooks.ActivationQuantization` scopes per-token symmetric INT4/INT8
fake quantization to an explicit list of `torch.nn.Linear` modules, with call
counts and hook removal on exit. It retains floating matrix multiplication and
adds quantization overhead; its timing is not evidence for integer-kernel speed.
`activation.signed_plane_matmul` independently accumulates int32 signed groups
before applying exponent and token scales. Small real-weight slices pass on
both M5 and Jetson CPU.

`radix.encode_radix` is a separate fit candidate: groupwise search over five
power-of-two units and nearest signed-radix levels. Its equal-storage weight
error on the screened projection is much lower than the greedy fit. This
restricted radix construction is a conventional-quantization-like baseline;
it does not reproduce output-aware ShiftAddQwen calibration.

## Activation-aware fitting preparation

`capture_pipeline.py` and `configs/a100-activation-capture-001.json` capture two
named transformer linears on the four calibration prompts. Each call retains at
most 32 evenly spaced token vectors; each module is capped at 16 calls (four
samples × four steps). Inputs are FP32 copies of actual BF16 computation inputs.
The hook preserves outputs, fails if the call budget is exceeded, rejects nonfinite
samples, and removes itself on exit. A unit test verifies selection, noninterference
and budget enforcement. CPU transfers invalidate inference timing. The resulting
NPZ can evaluate `||(W-Wq)X||` and per-step sensitivity without another model run.
This capture does not use held-out prompts or change existing weights.

## Bounded reconstruction buffers

`loader.apply_artifacts(..., chunk_rows=128)` reconstructs rows in bounded chunks,
then copies them into existing parameters. A pipeline config may set
`decode_chunk_rows`; omission preserves whole-matrix decoding for prior configs.
The audit records this setting. A unit test checks exact reconstruction across
chunk boundaries and rejection of corrupted artifact bytes. This reduces the
shape of temporary decoded arrays; a separate [single-matrix CPU diagnostic](../../experiments/m5-flux-klein-shiftadd-load-001.md) now measures its peak-RSS effect. Full-model and GPU peaks remain unmeasured.
Packed NPZ arrays and the destination dense parameters are still resident. This
is loading preparation, not a compressed inference kernel.

## Grouped activation arithmetic

`activation.grouped_signed_plane_matmul` now handles different activation and
weight group sizes by accumulating over their intersections. Signed int32 sums
are scaled per weight plane and per activation group. It supports A4/A8 codes
in int8 containers; activation nibble packing is provided by `activation_storage.py`; a GPU
implementation remains unimplemented. Weight signs are unpacked into an int32 reference array, so this
is not a memory-efficient runtime.

`python -m shiftadd_flux.verify_grouped --config
shiftadd/configs/activation-group-g32-001.json --output <new-directory>` checks
real saved inputs against independent dense reconstruction. Run as one command
with `PYTHONPATH=shiftadd/src`. Group sizes 32, 128 and 3072 passed 384 total
comparisons; evidence is in experiment `m5-flux-klein-shiftadd-003`.

## Prepared grouped activation pipeline checks

`ActivationQuantization(..., group_size=32)` supports arbitrary leading batch/token
dimensions, FP32 scale computation, and reconstruction to the original activation
dtype. It rejects group sizes incompatible with any selected linear before
installing hooks. Both pipeline runners read `activation_quantization.group_size`;
omission retains the previous per-token behavior. The saved config records the
group size; hook call counts still verify execution.

Prepared configurations `a100-grouped-a4-g32-prepared-001.json` and
`a100-grouped-a4-g128-prepared-001.json` each pair BF16, radix weight-only,
per-token A4, and grouped A4 on the same query projection and prompt.
They have distinct output paths and have **not been submitted**.
The batched BF16 hook test agrees exactly with the independent NumPy reference
for A4/A8 and groups 32/128, and verifies hook removal and invalid-group rejection.
These two prepared configurations remain unsubmitted. A separate
[full-transformer A4/A8 group-32 diagnostic](../../experiments/a100-flux-klein-shiftadd-006.md)
has since completed with all 109 linear modules exercised. This remains floating
GEMM with fake quantization; it cannot establish integer-kernel speed or energy.

## Activation storage layout

`activation_storage.pack_codes` stores A4 codes as signed two's-complement
nibbles, low nibble first, with zero high-nibble padding for odd element counts.
A8 is a byte-preserving copy. Shape and scales are separate metadata.
`unpack_codes` validates payload size and padding before reconstruction.
`verify_grouped` now round-trips the codes before packed sign-plane multiplication
and reports actual code and scale array sizes. These are payload measurements,
not end-to-end peak memory. Existing verification outputs are preserved.

## Same-source conventional full-transformer control

`convert_uniform.py` converts all two-dimensional weight matrices to symmetric
INT4 with FP32 group scales, copies unchanged tensors, reloads every packed file,
and records payload bytes, file bytes and hashes. It refuses existing output
directories. `loader.apply_uniform_artifacts` verifies all selected file hashes
and shapes before copying reconstructed rows into dense parameters.
Both pipeline runners require explicit `weight_format: symmetric_uniform` for
this bundle; unknown formats fail. Prepared config
`a100-uniform-control-prepared-001.json` pairs BF16, radix p4 and conventional
INT4 with the same source and image settings. The remote conventional bundle
was copied and verified before the
[completed conventional INT4 comparison](../../experiments/a100-flux-klein-uniform-001.md).
The prepared configuration is retained as preparation history; the experiment
record links the actual executed configuration and validation evidence.

## Exact integer mapping for the restricted radix fit

The saved radix p4 levels are odd integers −15, −13, …, 15 times a
power-of-two group scale. They map exactly to signed INT8 codes with no
additional quantization. Five-plane levels −31…31 also fit INT8; the reference
supports up to seven planes. This does not apply to arbitrary independent
ShiftAdd exponents. `integer_radix.to_integer_codes` verifies the radix exponent
relationship before conversion.

For four planes, let unsigned nibble c ∈ [0,15], signed INT4 q=c−8, and group
unit u=2**base. Then w=u(2c−15)=u(2q+1). Thus for each intersecting weight/activation
group, the output contribution is u·activation_scale·(2·Σ(qx·q)+Σqx).
The extra sum depends on the current inputs: it is not a fixed linear bias.
It must be computed with group boundaries and per-output-row weight scales.
The reference verifies integer dot products and this correction independently
against the original decoded weights.

`verify_integer_bridge.py` passed 768 real-input comparisons across both saved
weight fits, two modules, all 16 calls, A4/A8 activations, groups 32/128/3072, and
INT4/INT8 weight-code targets. Maximum relative difference from dense FP32
reference was about 1.01e-6. Both weight-code formats round-trip through actual
byte packing. Evidence supplements experiment `m5-flux-klein-shiftadd-003`.

These mappings remove a numerical requantization requirement for this radix
candidate, but not the implementation work: group scales, correction, packing
and accumulation must still fit a GPU kernel. A fully expanded INT8 buffer uses
8 bits per code instead of the compact four-plane bundle's 4 bits; tilewise
decoding would require an actual operator and workspace accounting. There is
no demonstrated tensor-core speed or energy result.

## Paired-case isolation preflight

Both image runners now validate case schedules before allocating the GPU model.
The first case must be unmodified BF16, case names and calibration sample IDs
must be unique, and each candidate must overwrite every parameter modified by
the preceding candidate. Non-nested partial replacements or a return to BF16
after mutation require a separate model reload and are rejected by these runners.
This prevents silently retaining a previous candidate's quantized parameters.
Artifact completion, tensor selection and representation are checked explicitly.

The retrospective `shiftadd/runs/completed-case-sequence-audit.json` verifies all
five completed image campaigns against their saved configurations and local
artifact manifests. The greedy sequence is 0→1→109 replaced matrices; the other
sequences use a BF16 reference followed by full 109-matrix replacements.
They satisfy the guard, so this finding does not invalidate their results.
Future runs save `case-sequence-audit.json` before model loading.

The exact integer bridge also passes the saved-slice numerical checks on
[Jetson CPU](../../experiments/jetson-flux-klein-shiftadd-001.md). This is not
a Jetson GPU kernel benchmark. Input/weight transfer hashes are archived.


## Staged upstream query image diagnostic

The pinned code and both complete upstream-fitted projection artifacts are staged
at `/scratch/gilbreth/mfaruqi/shiftadd-initial/upstream-image-stage-001` on Gilbreth.
All 52 transferred source/config/artifact files passed SHA-256 verification;
remote loader imports and case-selection preflight passed. Evidence is retained
locally in `shiftadd/runs/upstream-image-stage-001/stage-verification.json`.
Job 12134140 completed and passed paired validation; see the
[record](../../experiments/a100-flux-klein-upstream-query-001.md). It followed validated full-activation job 12133980. The original stage-verification receipt predates submission.

`prepared-job.slurm` uses the frozen `src/` snapshot and `prepared-config.json`.
Its four cases are BF16, radix p4, conventional INT4, and upstream u8 BCQ, each
changing only `transformer_blocks.0.attn.to_q.weight`. The fused artifact is staged
but is not selected by this config. All execution reconstructs dense BF16 weights;
these cases cannot establish packed-kernel latency, resident memory or energy.
The ten-minute job is pinned to the same A100 node as earlier paired checks.
The full-activation diagnostic is documented separately in
[its record](../../experiments/a100-flux-klein-shiftadd-006.md).


## Upstream scale arithmetic reference

`scaled_plane_arithmetic.grouped_scaled_plane_matmul` accepts explicit nonnegative
plane scales, so it covers both upstream u8/shared-FP16 and pot formats. It uses
INT32 activation/sign sums and FP32 scale products without forming dense weights.
It unpacks sign bits on the CPU and is not a fast inference kernel. The existing
exact single-INT8/INT4 radix bridge still applies only to the restricted radix
family; this reference does not extend that bridge to arbitrary BCQ scales.

`verify_scaled_arithmetic` replays all captured inputs with grouped A4/A8 against
decoded-weight matmul. The [slice record](../../experiments/m5-flux-klein-shiftadd-008.md)
links all 128 comparisons. The frozen Gilbreth image snapshot predates this
CPU-only addition and does not depend on it.

The separately labelled fused-projection comparison uses the same frozen source
and staged artifacts. Config `a100-upstream-fused-001.json` selects only
`single_transformer_blocks.0.attn.to_qkv_mlp_proj.weight`; job 12134626 was
submitted after the query comparison passed validation. It is a ten-minute
single-prompt diagnostic, not a performance benchmark.
