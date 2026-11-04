# Baseline inventory and next comparison gates

This is a preparation document, not a performance comparison. It maps existing
records to the controls needed for the ShiftAdd-format study. Scope: week 5
execution variants and RQ1/RQ3; the proposal's precision family states:
**“Low-bit storage alone ≠ faster”**
([proposal overview](../../wiki/project/overview.md#optimization-families-registry)).

## Existing controls

| Control | Evidence | What is established | Limitation for this study |
|---|---|---|---|
| A100 BF16 eager, distilled Klein | [reference](../../experiments/a100-flux-klein-001.md) | Full latency and memory protocol, pinned source | Original workload is 1024² on A100 PCIe; current ShiftAdd images are 512² on SXM4 |
| A100 Nunchaku SVDQuant W4A4/r32 | [record](../../experiments/a100-nunchaku-flux-klein-001.md) | Completed fused-kernel pipeline and memory measurements | Community replacement checkpoint; ancestry and matched quality not established |
| Jetson GGUF Q4_0 transformer, BF16 computation | [record](../../experiments/jetson-torch-flux-klein-001.md) | Completed pipeline with staged residency | A weight-storage/dequantization control, not integer activation throughput |
| Jetson Nunchaku W4A4/r32 | [record](../../experiments/jetson-nunchaku-flux-klein-001.md) | Completed native sm_87 pipeline | Different checkpoint/kernel; quality unassessed |
| Jetson exact conditioning reuse controls | [existing comparison](../../experiments/compare-jetson-nunchaku-vs-gguf-pytorch.md) | Separate completed no-reuse/reuse configurations | Reuse assumes repeated identical conditioning; not interchangeable with fresh-prompt latency |
| ShiftAdd-format greedy, radix and mixed weights | [initial comparison](../../experiments/compare-a100-shiftadd-initial.md), [calibration](../../experiments/a100-flux-klein-shiftadd-004.md) | Saved full-transformer weights and paired reconstructed-BF16 images | Storage and calibration diagnostics only; greedy full-transformer visual failure |
| Same-source conventional INT4 | [all 109 matrices](../../experiments/a100-flux-klein-uniform-001.md), [50-matrix controls](../../experiments/a100-flux-klein-upstream-all-attention-001.md) | One-prompt full-transformer control and four-prompt matched-subset control | Floating reconstruction; upstream fitting uses a different calibration objective and slightly different bit budget |
| Upstream BCQ with A8/A4 inputs | [50-matrix images](../../experiments/a100-flux-klein-upstream-all-attention-activation-001.md) | Fixed weights, separate activation modes, exact reference reproduction and call-count audits | Fake quantization; neither executable integer W4A4 nor formal quality admission |
| Mixed planes at equal subset storage | [first-block Q/K/V allocation](../../experiments/compare-upstream-qkv-plane-allocation.md) | Matched Q3/K3/V3 alternatives and exact reference reproduction | One block and four calibration prompts; not a general allocation policy |
| Activation and correction diagnostics | [groups](../../experiments/m5-flux-klein-shiftadd-003.md), [residual](../../experiments/m5-flux-klein-shiftadd-004.md) | Slice error, actual activation payload, packed arithmetic, correction storage | Selected rows/modules; neither full-layer nor image-quality guarantees |

The project registry does not establish a matched conventional W8A8 control for
this ShiftAdd campaign. Existing Nunchaku W4A4 is the relevant implemented
low-bit execution baseline, but it is not a same-source quality control.

## Controls still needed

1. **Expand and match conventional weight fitting.** The [initial slice control](../../experiments/m5-flux-klein-shiftadd-005.md), full-transformer one-prompt INT4 control and four-prompt 50-matrix control are complete.
   Full-transformer INT4 calibration coverage and a same-source W8A8 execution
   baseline remain gaps. To isolate the upstream fitter, also control the
   calibration objective and scale format at matched storage. Compare decoded
   output error and total payload including scales. The radix candidate
   is itself a restricted conventional-like quantizer; do not call an advantage
   over greedy fitting a ShiftAdd advantage over conventional quantization.
2. **Turn activation diagnostics into executable controls.** Original inputs,
   grouped A8 and grouped A4 are now compared across all 50 upstream attention
   matrices. The other 59 matrix weights remain BF16. Extend scope only with
   separately labelled configs and call-count audits, and select a supported
   packed backend before claiming W4A4/W8A8 speed or memory results.
3. **Mixed precision at matched storage.** Compare a fixed-format candidate and
   a mixed candidate under a common byte budget, including exponents, unchanged
   tensors and correction factors. The equal-storage first-block Q3/K3/V3
   comparison is complete; broader layer allocation remains a candidate.
   Earlier mixed p4/p5 intentionally costs more than uniform p4, so its
   improvement does not isolate allocation quality.
4. **Correction ablation.** Include no correction and fixed ranks, counting both
   BF16 factors and the execution branch. The slice residual-SVD probe is not
   SVDQuant and must not borrow its kernel-efficiency claims.
5. **Execution and energy.** Compare executable implementations on the same device,
   workload and residency policy only after calibration admission. Reconstructed
   BF16 plus fake quantization is a quality control, not a low-bit speed control.

## Fixed evaluation contract for the next bounded run

- Pin source checkpoint, conversion config, artifact hashes and runtime versions.
- Use the same 512² distilled workload, four steps, prompt, seed, initial noise
  where supported, text encoder, VAE and residency for each paired diagnostic.
- Keep calibration and project held-out prompts separate. Existing calibration
  observations are exploratory; formal acceptance criteria must be frozen before
  held-out evaluation. BF16 latent closeness does not establish prompt correctness.
- For timed runs use the parent protocol: first run separately, discard warmups,
  then report medians and min–max. Include preprocessing, quantization, decoding,
  transfers and corrections within their declared execution boundaries.
- Report deployment weight bytes separately from peak allocations, reserved memory,
  device-wide usage and shared physical memory. Never sum allocated and reserved.
- For energy, integrate timestamped measured power over the same synchronized
  repeated-generation interval. Report total joules divided by completed images
  and power measurement scope. Gross board/device energy is the primary quantity;
  any idle-subtracted estimate must state the separately measured idle baseline.
  Do not substitute TDP times latency or nominal shift/add operation counts.
- Pin device power mode; record temperature, clocks and competing load. A100 device
  power and Jetson board power have different boundaries and cannot be treated as
  the same system-energy metric. A measured energy result requires adequate sample
  coverage and a separate repeat, not a single short generation.

No jobs are submitted by this document. The current bounded campaign has no
measured packed-kernel speed or energy advantage.
