# Energy measurement preparation

No ShiftAdd energy result has been measured. The numerical utility
`shiftadd_flux.reporting.energy` prepares reduction of a future timestamped power trace;
it does not sample a device or turn diagnostic inference times into energy.

## Existing evidence limitation

The inspected [Jetson Nunchaku telemetry CSV](../../results/runs/jetson-nunchaku-flux-klein-001__baseline__20261102-120036/tegrastats-samples.csv)
contains memory and temperature fields, without power or sample timestamps.
The [A100 Nunchaku generation events](../../results/runs/a100-nunchaku-flux-klein-001__baseline__20261011-124123/events.jsonl)
contain durations but null start/end times. These exports alone cannot establish
power integrated over individual generation boundaries. This does not claim
that no raw power log exists elsewhere; it identifies the missing information
in the inspected baseline evidence.

## Required future capture

Power CSV columns:

- `timestamp_s`: monotonic-clock seconds, strictly increasing.
- `power_w`: measured watts, with documented sensor/update semantics.
- `clock_id`: shared host/boot/clock identity for samples and boundaries.
- `power_scope`: physical measurement boundary, such as one GPU or whole board.

Window JSON fields: `start_s`, `end_s`, `images` (completed images), `clock_id`,
`power_scope`, and `max_gap_s` (declared acceptable sample gap). The runner must
synchronize work before opening and closing the interval, record all completions,
and use the same clock as the sampler. Keep first generation, warmup and load
outside a warm-generation interval; record separate intervals if those costs
are being studied. Include all quantization, decoding, correction, transfers
and synchronization required by the selected execution policy.

A minimum of three actual samples inside an interval is a validity floor,
not a sufficient precision guarantee. Prefer a repeated-generation window long
enough to cover many sensor updates; repeat the measurement independently.
Do not assume faster polling makes a sensor update faster. Record device power
mode, temperature, clocks, competing load, sensor name and sampling settings.

## Reduction

Run from the repository root:

```bash
PYTHONPATH=shiftadd/src python -m shiftadd_flux.reporting.energy \
  --samples <power.csv> --window <window.json> --output <new-result.json>
```

The reducer integrates watts over seconds with trapezoids, interpolates only
between bracketing samples at the endpoints, and divides gross joules by completed
images. It rejects gaps above the declared limit, missing temporal coverage,
clock/scope mismatches, nonfinite values and inadequate sample counts. It refuses
to overwrite its output. `gross_energy_j` and `gross_j_per_image` are sampled
estimates; they do not include an uncertainty estimate or recover missing transients.
No idle subtraction is performed.

The analytic unit test integrates a known linear power curve over fractional
boundaries. It is a software correctness fixture, not device evidence. Existing
full-pipeline runners still need a synchronized sampler/window integration before
this utility can produce any measured inference energy result.

A100 GPU energy and Jetson board energy describe different physical boundaries.
Report them separately. Do not multiply TDP by latency, infer joules from storage
savings, or compare unlike power scopes as if they were total system energy.
