# Bounded nonuniform exponent-template screen

Hypothesis: independently spaced power-of-two planes may reduce weight error at
fixed four-plane storage compared with signed radix.

The first 32 output rows from six matrices were tested: one transformer query
projection, context embedding, output projection, fused single-stream input
projection, single-stream output projection and image modulation projection.
Groups contain 128 input-channel weights. The search enumerates six exponent
templates, five unit exponents and all 16 sign combinations per weight, including
the signed-radix candidate. This is a bounded weight-error screen, not GPTQ or
activation-aware fitting.

Evidence: `shiftadd/artifacts/nonuniform-slice-001/summary.json` and
`multi-matrix-summary.json`. None of these six slices improves over signed radix;
reported relative L2 values are identical. The nonuniform search takes more work.
This does not rule out other templates, learned scales or activation-aware
objectives. It does not justify a full-model conversion of this candidate.

Implementation: `src/shiftadd_flux/nonuniform.py`. Original artifacts are retained.
