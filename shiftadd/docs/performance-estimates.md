# What the storage result can predict

Four planes and an int8 exponent per group of 128 use 4.25 bits/weight.
Measured transformer weight files shrink from 7,751,109,744 bytes to
2,058,956,988 bytes (including untouched tensors). See the artifact storage
reports and the conversion experiment record for exact checksums.

## Conditional bounds, not measured speedups

If a stage were entirely limited by reading these weights, read them once per
call, used no additional buffers or metadata traffic, and decoded for free,
the weight-traffic-only speedup ceiling would be about 3.76x. This is an
optimistic conditional bound, not a prediction for Klein or either GPU.
If only a fraction f of baseline time benefits, a corresponding ideal bound
is 1 / ((1-f) + f/3.76). At f=0.1 it is about 1.08x; at f=0.5 about 1.58x.
Decode, activation quantization and extra transfers can erase the gain.

Expanding the full transformer into dense BF16 before inference changes loading
and disk size but preserves dense matrix compute and parameter storage. Keeping
both forms adds about 1.92 GiB to BF16 weights; layerwise expansion instead needs
a bounded decode buffer and an actual operator implementation. Text encoder,
VAE, activations, caches and workspaces remain additional memory costs.

## Timing preparation

Separate disk load, artifact decoding, model installation, first generation,
discarded warmups and repeated generations. Keep GPU class, software, checkpoint,
noise, prompts, resolution and schedule matched. Use no fake-quantization timing
as a surrogate for a low-bit GPU kernel. Record failures without changing format.

## Energy preparation

Record timestamped power and integration boundaries for repeated steady runs.
Joules per image equals integrated power over that interval divided by image
count; report board versus GPU boundary and any idle subtraction explicitly.
A100 NVML and Jetson board telemetry have different scopes. Fix power mode,
record clocks and thermal state, and evaluate warm repeated generations. Do not
infer lower energy merely from weight compression or shift/add operation counts.

A validated reduction utility and the required future capture schema are in
[the energy contract](energy-contract.md). Existing inspected baseline exports
lack the aligned power/window data required for per-image energy. No energy
number has been derived from them.
