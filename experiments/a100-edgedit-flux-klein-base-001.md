---
type: experiment-record
id: a100-edgedit-flux-klein-base-001
status: complete
device: a100-pcie-40gb
engine: edge-dit-cpp
runs: [a100-edgedit-flux-klein-base-001__baseline__20261015-165052, a100-edgedit-flux-klein-base-001__attempt__20261015-164901, a100-edgedit-flux-klein-base-001__attempt__20261015-161752]
updated: 2026-10-15
---

# a100-edgedit-flux-klein-base-001: edge-dit.cpp klein Base (50 steps, CFG 4) from GGUF components (Q4_0 transformer, Q4_K_M encoder, BF16 VAE), 1024², no cache

Status: **complete** (2026-10-15). Full 1+3+10 protocol in job 11960173 on gilbreth-g002 (20:50:52–21:02:34 UTC)
after a smoke in the same job. A first attempt in job 11959986 on gilbreth-g009 generated both images but **failed**
in the runner's log parsing (runner bug, fixed the same day; see below).

## Question

What are the latency, stage split and device memory of klein Base (50 Euler steps, CFG 4, 1024²) in edge-dit.cpp on
the A100, from GGUF component files with weights resident and no step cache? This run is the reference for the
edge-dit Base step-cache variants [-002](a100-edgedit-flux-klein-base-002.md) to
[-008](a100-edgedit-flux-klein-base-008.md). It is a **compressed-checkpoint workload** (4-bit transformer and text
encoder), labelled and reported separately from BF16 references. RQ1 baseline infrastructure; RQ2 reuse reference.

## Setup

- **Config**: [a100-edgedit-flux-klein-base-q4-1024.json](../configs/a100-edgedit-flux-klein-base-q4-1024.json);
  smoke [a100-edgedit-flux-klein-base-q4-1024-smoke.json](../configs/a100-edgedit-flux-klein-base-q4-1024-smoke.json)
  (first + one measured). The run-directory `config.json` copies equal these files; `reference_config` is null.
- **Model files** (each pinned by revision and checked by SHA256 before launch):
  - transformer `leejet/FLUX.2-klein-base-4B-GGUF` @ `d12671125306ca6b5f6db1b33ed4c80c8511a53f`,
    `flux-2-klein-base-4b-Q4_0.gguf` (sha256 `c3a28545…`); engine weight-type stat `q4_0:80|bf16:69`;
  - text encoder `unsloth/Qwen3-4B-GGUF` @ `22c9fc8a8c7700b76a1789366280a6a5a1ad1120`, `Qwen3-4B-Q4_K_M.gguf`
    (sha256 `f6f85177…`); stat `f32:145|q4_K:216|q6_K:37`;
  - VAE `black-forest-labs/FLUX.2-klein-4B` @ `e7b7dc27f91deacad38e78976d1f2b499d76a294`,
    `vae/diffusion_pytorch_model.safetensors` (BF16; stat `bf16:248`).
  The engine log reports `flux activation dtype: f32` and VRAM parameter buffers of 2346.40 MB (flux), 3555.38 MB
  (qwen3) and 94.72 MB (VAE) ([engine.log](../results/runs/a100-edgedit-flux-klein-base-001__baseline__20261015-165052/engine.log)).
- **Workload**: 1024×1024, 50 steps, Euler sampler, scheduler `auto` (engine log: `flux2_mu=2.023`), guidance 4.0,
  `cfg_scale` 4.0, seed 0, batch 1, prompt "A cat holding a sign that says hello world". Flash attention on (command-line
  flag only; the engine log does not report it), VAE tiling off, no step cache (`cache mode : original`), four threads.
- **Engine**: edge-dit.cpp `97cc585`, CUDA performance build, `ed-sample` sha256 `fc6b14a7…`; source tree clean and
  commit verified by the runner ([environment.json](../results/runs/a100-edgedit-flux-klein-base-001__baseline__20261015-165052/environment.json)).
  NVIDIA A100-PCIE-40GB, driver 595.91.07, 250 W. Command in
  [command.json](../results/runs/a100-edgedit-flux-klein-base-001__baseline__20261015-165052/command.json).
- **Runner**: [scripts/run_edgedit_jetson.py](../scripts/run_edgedit_jetson.py) with `device_label a100-pcie-40gb`
  (requires a Slurm allocation; device memory by NVML), sha256 `153d3ea9…`
  ([source-provenance.json](../results/runs/a100-edgedit-flux-klein-base-001__baseline__20261015-165052/source-provenance.json)).
  Launched by [scripts/gilbreth-gap-base-edgedit-20261015.slurm](../scripts/gilbreth-gap-base-edgedit-20261015.slurm)
  (smoke, then the full protocol only if the smoke completed).
- **Protocol**: one `ed-sample` process, one load, `--warmup 0 --repeat 14` (first, 3 warm-ups, 10 measured); smoke
  `--repeat 2`.
- **Metric derivation** ([Jetson edge-dit runner](../wiki/methods/baseline-metrics.md#jetson-edge-dit-runner),
  [edge-dit adapter](../wiki/methods/baseline-metrics.md#edge-dit-ed-sample-adapter)): `wall_ms` from `ed-sample`
  per-pass stdout (1 ms rounding); `text_encode_ms`, `denoise_ms` (engine `sampling`), `vae_decode_ms` and `other_ms`
  from the engine's `flux generate breakdown` line (10 ms rounding); raw `[[phase]]` markers in `stages.csv`. No
  per-step times. Device memory is one NVML peak over the whole child process including load
  (`summary.json → device_used_peak_gib`; 20 ms sampling); per-generation memory, allocator metrics and host RSS are
  unavailable. The `unavailable_metrics` note in `summary.json` names tegrastats (Jetson wording); on the A100 the
  sampler is NVML. Only the final pass's image is retained.

## Results

Full run [a100-edgedit-flux-klein-base-001__baseline__20261015-165052](../results/runs/a100-edgedit-flux-klein-base-001__baseline__20261015-165052/):
[status.json](../results/runs/a100-edgedit-flux-klein-base-001__baseline__20261015-165052/status.json) `complete`;
[summary.json](../results/runs/a100-edgedit-flux-klein-base-001__baseline__20261015-165052/summary.json),
[runs.csv](../results/runs/a100-edgedit-flux-klein-base-001__baseline__20261015-165052/runs.csv),
[stages.csv](../results/runs/a100-edgedit-flux-klein-base-001__baseline__20261015-165052/stages.csv).
The engine printed 50 `flux step` lines per pass (700 in all) and one CFG txt2img line per pass with `cfg=4.00`.

### Latency (measured n = 10)

| Stage | Median (ms) | Min–max (ms) | Share of median wall |
|---|---:|---:|---:|
| wall (1 ms rounding) | **49428.5** | 49408.0–49444.0 (stdev 13.0) | 100% |
| text_encode (engine breakdown) | 120 | 120–120 | 0.2% |
| denoise (engine `sampling`, 50 steps × 2 CFG passes) | 48740 | 48730–48760 | 98.6% |
| vae_decode | 280 | 270–280 | 0.6% |
| other | 290 | 290–290 | 0.6% |
| `[[phase]]` markers: encode_setup / denoise / vae_decode | 404.4 / 48743.4 / 275.6 | 401.7–405.9 / 48725.7–48758.2 / 274.1–276.6 | – |
| First generation: wall | 49476.0 | – | text 200 / denoise 48590 / vae 380 / other 310 |
| Warm-ups (3, discarded): wall | – | 49329.0–49423.0 | – |
| Model load (s) | 2.15 | – | files hashed just before launch, page cache warm |

Mean time per transformer pass: 48740 / 100 = 487.4 ms (derived).

### Memory (GiB; whole child process, includes load)

| Quantity | Value |
|---|---:|
| device_used_peak_gib (NVML, 20 ms; whole capture) | 11.732 |

### Repeatability and functional check (visual, not a quality metric)

The final-pass image (run index 13) has RGB SHA256 `dd6b02c16d941b76…`
([image-retention.json](../results/runs/a100-edgedit-flux-klein-base-001__baseline__20261015-165052/image-retention.json));
the smoke's final image and the failed attempt's final image have the same hash, so the output reproduces across three
processes on two nodes. Earlier passes' images are overwritten (`deterministic_output` null). Visual check of
`output.png`: coherent tabby cat holding a cardboard sign in both paws against a plain pink background; the sign
reads "hello world" in black script lettering.

### Smoke run

[`__attempt__20261015-164901`](../results/runs/a100-edgedit-flux-klein-base-001__attempt__20261015-164901/):
complete (runner `153d3ea9…`). First 50024.0 ms (text 490 / denoise 48410 / vae 810 / other 320); one measured
49129.0 ms (120 / 48430 / 280 / 300). Load 2.34 s. Device peak 11.732 GiB. Not a baseline (n = 1).

### Failed first attempt (runner bug)

[`__attempt__20261015-161752`](../results/runs/a100-edgedit-flux-klein-base-001__attempt__20261015-161752/)
(job 11959986, gilbreth-g009, 20:17:52–20:19:45 UTC, smoke config, runner sha256 `5e6d296a…`): `status.json` `failed`
with `KeyError('denoise_step_4_ms')`, raised in `summarize_runs` (`scripts/measurement.py`) called from `benchmark()`.
The engine itself completed both passes: [engine.log](../results/runs/a100-edgedit-flux-klein-base-001__attempt__20261015-161752/engine.log)
shows `pass 2/2` and both breakdown lines, and `runs.csv`, `stages.csv`, `load.json` and `output.png` were written
before the failure (first pass wall 50811.0 ms, denoise 48820 ms; second 48873.0 ms, denoise 48180 ms; load 2.69 s;
final image hash `dd6b02c1…`, the same as the later runs). No `summary.json` was written and the NVML samples were not
saved.

Cause, as reconstructed from the traceback and the fixed runner: in `parse_log` the loop over the phase-marker offsets
(0, 2, 4) reused the variable that held the step count, so after the first row it held 4 and later rows were created
with only `denoise_step_0..3_ms` columns; `summarize_runs` takes the metric list from the first row (50 step columns)
and failed on the missing `denoise_step_4_ms`. The runner was fixed the same day (sha256 `153d3ea9…`) and the queue
was rerun as job 11960173. The attempt is kept as evidence; its timings are not used.

## Observations

- Median wall is 49428.5 ms with a measured range of 36.0 ms (0.07% of the median); the sampling loop (100 transformer
  passes) is 98.6% of it, about 487 ms per pass.
- Text encode, VAE decode and other are each 0.2–0.6% of wall (10 ms resolution).
- The first generation of the full run is within 50 ms of the measured median; the smoke's first generation has a
  slower text encode (490 ms) and VAE decode (810 ms).
- The whole-process device peak is 11.732 GiB in both the smoke and full runs.
- The output is identical across the failed attempt, the smoke and the full run.

## Interpretation

- For this compressed workload, latency is set almost entirely by the 100 transformer passes; only levers that reduce
  per-pass time or the number of passes can change end-to-end latency materially.
- The device peak includes about 5.9 GB of resident parameter buffers (engine log); the remainder is CUDA context,
  activations and workspaces, which the single whole-process NVML peak does not separate.
- The failed attempt is a runner defect in post-processing, not an engine or workload failure.

## Next experiment

- Step-cache variants against this configuration: [-002](a100-edgedit-flux-klein-base-002.md) to [-008](a100-edgedit-flux-klein-base-008.md).
- Runner: replace the Jetson wording in `summary.json → unavailable_metrics` for A100 runs.
