# Building roofline plots from saved Klein runs

This is a Week 2 baseline-analysis tool supporting RQ1: “Which diffusion optimization choices transfer across devices and workloads?” ([proposal overview](../wiki/project/overview.md#research-questions), [milestones](../wiki/project/milestones.md#revised-schedule)). It executes no models. Existing run artifacts remain immutable. Outputs, including W&B state, go only to the output directory.

## Run and publish

Dependencies: Python 3.10+, `plotly`; `wandb` and `pillow` for upload with image panels; `matplotlib` for paper PNG/PDF exports. The optional legacy Plotly renderer needs `kaleido` and Chrome. Install them in a dedicated environment outside timed benchmark execution.

## Paper-style presentation

The plot is constructed from coordinates: arithmetic intensity on X, effective TFLOP/s on Y. The reference curve is `min(peak, bandwidth × intensity)`. Its bend is the ridge point `peak / bandwidth`. Both axes are logarithmic; the horizontal red guide marks the primary compute ceiling, and the vertical green guide marks its ridge. Distinct colours, outlined markers, compact ranges, powers-of-ten ticks, dotted minor grids and an inset legend reproduce common paper-figure styling. A reference screenshot cannot establish which software its authors used. Interactive charts use Plotly. Paper PNG/PDF exports now use Matplotlib, following the verified [ShiftAddQwen plotting source](https://github.com/ranery/ShiftAddQwen/blob/5bf17676f28b0a5411781f1a61798f87cb3e7e05/tools/plot.py): STIX serif fonts, Okabe–Ito colours, inward ticks and light grids. The reference itself is a model-size/quality/throughput frontier, not a roofline.

The legacy `--static --static-backend plotly` additionally writes `m5-paper-roofline.{html,png,pdf}`: side-by-side distilled and Base panels with shared bounds, per-option marker styling and separate legends. All saved M5 stage coordinates are retained exactly, without jitter or averaging repeated source runs. Filled markers denote MPS BF16 and open markers denote sd.cpp mixed Q4; the assumed M5 ceiling remains explicit. The W&B upload adds `roofline_images/*` image panels (including `roofline_images/m5-paper`) as well as responsive interactive `roofline/*` panels. Use the image panels when a fixed paper layout matters. W&B interactive panels use automatic sizing, concise axis titles and a compact scope note to fit dashboard tiles. Full scope notes remain in exports and the data artifact. No benchmark measurement or coordinate formula changes with the styling.

```sh
python scripts/plot_roofline.py --output output/roofline --upload
python scripts/plot_roofline.py --output output/roofline --static
python3 -m unittest discover -s tests -v
```

The default discovers completed baseline/repeat/profile directories under `results/runs/`. Attempts, smoke runs, incomplete profiles and non-PCIe A100 samples are excluded explicitly in `roofline-data.json`. Full baselines require ten measured generations with CSV/summary agreement. Repeated baselines remain separate points. A profile uses one enriched OSRT kernel file when available, avoiding duplicate points from parallel exports of the same trace. All available consulted kernel files are hashed.

Four HTML charts are self-contained: `a100-roofline`, `jetson-roofline`, `m5-roofline`, `cross-device-roofline-pct`. PNG/PDF exports are optional. `roofline-data.json` and `roofline-points.csv` contain coordinates, duration scopes, estimates, exclusions and SHA256 provenance. The W&B run has `job_type="analysis"`, four `roofline/*` Plotly panels, `roofline_points` Table, and an evidence artifact containing the dataset, charts and script. Reusing the same output directory resumes its analysis run ID. Upload verification downloads and compares the remote table and evidence JSON; only a successful comparison marks `wandb-upload.json` verified. Static figures are archived when available; the interactive panels are the primary visualization.

An exported dataset can be rendered/uploaded on another host:

```sh
python scripts/plot_roofline.py --data-json output/roofline/roofline-data.json --output output/roofline-copy
```

## Matplotlib A100 exports (October 21)

`--static` now defaults to Matplotlib. `--static-device a100` limits paper exports to A100; interactive Plotly output still contains all devices in the supplied dataset. Two files separate timer scopes: `a100-stage-roofline.{png,pdf}` compares denoising stages; `a100-kernel-roofline.{png,pdf}` shows historical profiled GEMM/attention observations. Each has distilled and Base panels, shared limits and legends below the axes. Coordinates are unchanged by rendering. The floating-point axis no longer includes an INT4 TOP/s curve.

[Input/tool inventory](../output/a100-roofline-prep/README.md) and [curated A100 selections](../output/roofline-a100-20261021/a100-manifest.json) distinguish nine unprofiled full runs from three historical profiles. The selected sd.cpp and edge-dit stages explicitly estimate four-byte activations; their BF16 reference roof is context for a mixed execution path, not a hardware-utilization claim. Nunchaku's low-rank corrections/dequantization remain excluded from the reference-work estimate. Approximate step caches are excluded from this overview pending executed-work accounting. Other devices' historical coordinates are retained unchanged in the combined dataset.

```sh
MPLCONFIGDIR=/tmp/odd-matplotlib python scripts/plot_roofline.py \
  --data-json output/roofline-a100-20261021/roofline-data.json \
  --output output/roofline-a100-20261021 --static --static-device a100 --upload
```

The saved upload receipt resumes the existing roofline analysis. Images appear under `roofline_images/a100-stage` and `roofline_images/a100-kernel`; interactive panels remain under `roofline/*`. Upload validation compares the remote table and data JSON with local coordinates/provenance and verifies exported PNG/PDF artifact digests. No new GPU measurement is performed by this command. Actual HBM counter collection remains deferred; these are analytic estimates with measured timings.

## Measured Jetson rooflines (November 1)

[plot_roofline_jetson.py](../scripts/plot_roofline_jetson.py) builds the Jetson Orin Nano figures from measured inputs only: ceilings from [jetson-ceilings-001](../experiments/jetson-ceilings-001.md) and Nsight Compute LPDDR5 bytes per denoising pass from the engine counter captures ([definitions](../wiki/methods/baseline-metrics.md#nsight-compute-engine-counter-captures-jetson)). It plots the fastest per-pass configuration of each engine plus the PyTorch eager reference; edge-dit, which has no capture, is drawn hollow at its bandwidth lower bound. Outputs in `output/roofline-jetson-measured/`: `jetson-engine-roofline-measured`, `jetson-kernel-bandwidth-measured`, `jetson-traffic-breakdown` (PNG + PDF) and `plot-data.json`. Results: [comparison](../experiments/compare-jetson-engine-roofline-measured-20261101.md).

```sh
python scripts/plot_roofline_jetson.py [--upload]
```

## Select runs and override assumptions

`--manifest` accepts a JSON object with `runs` selections and optional `hardware_specs` and `exclusions`. Each selection requires `run_dir`; optional fields are `label`, `weight_gib`, `activation_bytes`, and `ceiling`. Explicit selections fail on incomplete or unsupported runs. Selections must be direct children of `results/runs/`. `--hardware-specs` accepts just the device override dictionary. Overrides must be positive and are preserved in the dataset; control files and script are also hashed. `--data-json` cannot be combined with new selections or ceiling overrides.

```json
{
  "runs": [{
    "run_dir": "a100-flux-klein-001__baseline__20260923-221530",
    "label": "PyTorch BF16 reference"
  }],
  "hardware_specs": {
    "gilbreth-a100-40gb": {
      "bw_gbs": 1555.0,
      "ceilings": {"bf16_tensor": 312.0},
      "note": "Dense theoretical specification ceiling"
    }
  }
}
```

The A100 dictionary uses dense BF16/FP16 312 TFLOP/s, FP32 19.5 TFLOP/s and INT4 **1248 TOP/s**; 624 is the dense INT8 peak, not INT4. Bandwidth is 1555 GB/s. Sources: [NVIDIA integer/compute specification](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet.pdf), [PCIe bandwidth specification](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf).

Jetson uses Orin Nano **Super** reference ceilings of **102 GB/s**, **17 FP16 TFLOP/s**, and 33 dense INT8 TOP/s. The original 68 GB/s, 20/40 TOP/s numbers refer to different mode/sparsity assumptions. These are specification upper bounds, not observed sustained clock or bandwidth measurements. Q4_0 weight storage is compared against floating-point compute, not an integer roof. Source: [NVIDIA original/Super comparison](https://developer.nvidia.com/blog/nvidia-jetpack-6-2-brings-super-mode-to-nvidia-jetson-orin-nano-and-jetson-orin-nx-modules/).

M5 bandwidth is **153 GB/s**, as [Apple publishes](https://www.apple.com/macbook-air/specs/). The **18 TFLOP/s** compute roof is a requested illustrative reference assumption, not an Apple specification or a measured peak. Every M5 percentage depends on this assumption. Hardware dictionaries and chart footers retain this qualification.

## Coordinates and scope

For GEMM `M m N n K k`, reference FLOPs are `2*m*n*k`; bytes are `(m*k+m*n)*b_a + k*n*b_w`. BF16 uses two-byte activations and weights; GGUF Q4_0 uses 0.5625 bytes/weight (including block scales), and INT4 uses 0.5. Mixed precision is explicitly labelled.

For flash attention `B b H h Sq sq Sk sk D d`, reference FLOPs are `4*b*h*sq*sk*d`; ideal bytes are `2*b*h*(sq+sk)*d*b_a`, with two-byte Q/K/V/O and no score matrix materialization. Recorded shape/call/kernel durations must agree with saved TFLOP/s. Where sd.cpp lacks shapes, only a group entirely attributable to `flash_attn_ext_f16` is used; shape is inferred from the reference architecture and labelled `operator-inferred`. Other kernels without shape evidence are not assigned invented GEMM coordinates.

Operator duration is the summed associated kernel duration divided by calls. It is distinct from GPU busy-union time and host callback stage duration. Profiles retain their own source run and generation scope; they are not unprofiled benchmark medians.

Stage estimates use the pinned [Klein 4B architecture](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/blob/e7b7dc27f91deacad38e78976d1f2b499d76a294/transformer/config.json): hidden width 3072, 24 heads × 128 dimensions, five dual blocks, twenty single blocks, SwiGLU ratio 3, 128 input/output channels, 7680 text input width. Sequence length is image tokens `(W/16)*(H/16)` plus configured text length (default 512). Each dual block counts QKV, attention output, and two SwiGLU projections on both streams; each single block counts fused `D→9D` and `4D→D` projections. Input/output projections and all 25 attention blocks are included. This is **dominant reference work**, excluding elementwise/normalization/RoPE, timestep/modulation, dequantization and Nunchaku low-rank correction costs; it is not a hardware instruction count.

Stage bytes are one read of the transformer footprint per pass plus the operator activation reads/writes. Recorded `summary.weights_gib.transformer` or `load.components.transformer.weights_gib` takes precedence. Runs without a recorded component footprint use an explicit labelled estimate: BF16 7.218764305114746 GiB or mixed Q4_0 2.15 GiB. The latter approximates mixed storage, not pure 4-bit weights; it can be replaced by a manifest estimate. The Nunchaku run uses its recorded mixed footprint. Byte estimates assume ideal tensor traffic without inter-operator cache reuse; no PMU DRAM traffic has been measured. Actual traffic may be higher or lower.

Executed passes are steps × CFG multiplicity (one at guidance ≤1, two at guidance >1), subtracting saved per-generation EasyCache skipped steps before multiplication. Missing skip audits fail. For fixed executed pass counts, stage throughput is total reference FLOPs / median denoising duration. If skip counts vary, it is the median **per-generation** executed reference FLOPs / measured denoising duration; the displayed duration and total work are separate medians if skip counts vary. Skip count indexing follows `run_index` into the complete generation audit. It is not a speedup score; skipped work is not credited as executed work. Exact conditioning reuse may also alter smaller reference operations, so these remain approximate coordinates.

Jetson no-cache references get paired `denoise_step_0` and steady `denoise_step_1..3` points using the same baseline's measured `stages.csv`. Steady duration is the median of each generation's mean of steps 1–3. A dashed downward connector shows the observed rate difference including the first-step reload stall. Attribution to read/pread64 comes from the separately saved OSRT profiles ([distilled](../experiments/jetson-flux-klein-003.md), [Base](../experiments/jetson-flux-klein-base-001.md)); the connector alone cannot prove storage causality or distinguish storage from page-cache effects. No illustrative ~10× ratio or fixed latency from task prose is substituted for saved data.

Normalization is `100*effective_tflops/min(ceiling, bw_gbs/1000*AI)`, using each point's stated reference ceiling. Nunchaku's mixed stage uses the BF16 reference denominator: neither pure INT4 nor pure BF16 utilization is inferred. Integer specifications remain labelled TOP/s in the metadata and are not drawn on the floating-point axes. Values over 100% remain visible as an analytic-model discrepancy. These percentages are not matched-quality comparisons or measured hardware utilization. Different workloads, precision, residency, engine timers and profile instrumentation limit cross-device interpretation ([metric definitions](../wiki/methods/baseline-metrics.md)).

## Counter-based A100 eager and compiled views

`plot_roofline_measured.py` renders the five-engine distilled 1024² overview plus matching eager and
compiled kernel/bandwidth figures. It reads the eager NVTX-labelled `denoise_step_1` and the compiled
capture's validated `profile/pass-kernels.json`, checking the compiled launch count, bytes and duration
against `counter-summary.json`. It executes no models. Reproduce with:

```sh
MPLCONFIGDIR=/tmp/odd-matplotlib python3 scripts/plot_roofline_measured.py \
  --ceilings results/runs/a100-ceilings-001__ceilings__20261029-042704 \
  --output output/roofline-a100-measured
```

The new `a100-kernel-roofline-compiled.{png,pdf}` shares axes and units with
`a100-kernel-roofline-measured.{png,pdf}` (eager). Kernel groups follow names, not exact operator
attribution: Triton fusions remain in the elementwise/reduction group even when their names mention
attention or normalization. Shares use summed replay kernel durations, never unprofiled generation
latency. Missing counted instruction classes remain explicit. Measured microbenchmark ceilings are
references, not universal bounds. The five-engine overview uses reference-model work for effective
throughput; this is distinct from executed hardware operations, especially for mixed INT4/BF16.

Methodology references: [Williams et al., CACM (2009)](https://doi.org/10.1145/1498765.1498785)
for the roofline model; [Yang et al., hierarchical deep-learning analysis](https://arxiv.org/abs/2009.05257)
for empirical machine characterization, per-kernel counters and framework comparison; and
[Wang et al., time-based analysis](https://arxiv.org/abs/2009.04598) for retaining runtime alongside
throughput when implementations perform different work. The latter two are proceedings papers,
not journal articles. [Malenza et al., IEEE TPDS (2025)](https://doi.org/10.1109/TPDS.2025.3591452)
provides a journal precedent for comparing GPU programming frameworks with kernel rooflines;
its workload is an astrometric solver, not neural inference.


## A100 three-panel preview

`scripts/plot_a100_three_panel.py` produces roofline, warm generation latency and sampled device-wide
GPU-memory panels for the five selected distilled 1024² configurations. It reads the counter-based
roofline dataset and original baseline JSON/CSVs; no model execution or W&B upload is implemented.

```sh
MPLCONFIGDIR=/tmp/odd-matplotlib python3 scripts/plot_a100_three_panel.py
```

Outputs: `output/a100-three-panel/a100-three-panel.{png,pdf,svg}` and `figure-data.json` with source
hashes, exact coordinates, first-image latency and measurement scopes. The roof uses the dense BF16
specification (312 TFLOP/s, 1555 GB/s, ridge 200.6 FLOP/byte). Mixed-precision Nunchaku is hollow and
labelled as a compressed replacement with unverified ancestry/quality, not BF16 hardware utilization.

Latency totals are medians of ten measured wall times, with min–max whiskers. To preserve additivity,
stack segments average the corresponding denoise and VAE durations of the two middle wall-time
observations; remaining time is their wall-time residual. They are not independent stage medians.
This grouping avoids relabelling edge-dit's combined encoding/setup time as pure text encoding.
Memory uses median measured-generation NVML peaks for four engines; edge-dit has only a whole-child
capture peak including loading and first/warm-up generations, explicitly hatched and footnoted.
No allocator or host memory is substituted or added. These views reproduce existing measurements;
they do not establish a matched-quality engine ranking.


## Jetson three-panel preview

`scripts/plot_jetson_three_panel.py` renders the three-panel layout for distilled Klein at 512² on
Orin Nano with conditioning reuse disabled throughout. Four existing complete 1+3+10 runs are used:
PyTorch eager (001), resident tiled sd.cpp (012 completed repeat), resident edge-dit with tiled VAE (006), and
Nunchaku with sequential residency (001). PyTorch compiled is omitted because no completed
no-conditioning-reuse baseline is available; its existing cache-hit result is not substituted.
Different checkpoints, residency, software dates and tile policies prevent a controlled engine-only
or matched-quality ranking. All runs use the fixed development prompt recomputed every image;
this measures a no-cache request path rather than a varied-prompt quality/workload evaluation.

```sh
MPLCONFIGDIR=/tmp/odd-matplotlib python3 scripts/plot_jetson_three_panel.py
```

Outputs: `output/jetson-three-panel/jetson-three-panel.{png,pdf,svg}`, `figure-data.json` and README.
No upload or model execution is performed. The original two-panel outputs remain unchanged.

Roofline traffic is analytic, unlike the A100 counter-based preview. The common reference-op model
uses four-byte intermediates for the C++ FP32-activation paths and two-byte intermediates for
PyTorch (a BF16 reference for mixed Nunchaku). Those widths do not assert all tensors or kernels
use one precision. Packed footprint uses recorded PyTorch/Nunchaku component sizes; C++ mixed-Q4
uses the existing explicit 2.15 GiB estimate. Unpacking, low-rank, normalization and fusion-specific
traffic are not modeled. All roofline markers are hollow to denote estimated traffic. The
17 TFLOP/s, 102 GB/s roof is a specification reference, not a measured sustained limit or mixed
INT4 utilization. C++ x coordinates differ from the older common two-byte approximation. The
stage-rate numerator is reference work and the denominator includes sd.cpp's first-step reload;
this application-level effective rate must not be interpreted as kernel hardware utilization.

Latency totals are median wall times with min–max intervals and INCLUDE per-generation encoder
and transformer reloads. Stack segments use corresponding two middle wall-time observations.
The residual includes text encoding, setup and reloads outside engine denoise/VAE timers. One-time
initial model load and the separately reported first generation are excluded from the warm-session
bars; they remain in the data artifact. No cache-hit timing is used.

Memory is whole-system tegrastats RAM peak across the entire capture, including loading, first
image and warm-ups; it is never relabelled GPU allocations. Sampling differs (sd.cpp 1 s; others
200 ms), and background activity is included. Swap extrema remain in figure-data.json.


## Eager-normalized engine speedup previews

[Local output and method](../output/engine-speedups/README.md) provides grouped bars for
Klein Distilled and Klein Base, separately on A100 and Jetson. Each device has denoising-stage
and end-to-end speedup figures. PyTorch eager is 1× independently per workload. Reproduce
with `MPLCONFIGDIR=/tmp/odd-matplotlib python3 scripts/plot_engine_speedups.py`.
The source manifest and absolute timings are retained in `figure-data.json` and `speedups.csv`.
Missing no-reuse combinations are labelled N/M; precision/checkpoint and stage-boundary
confounds are printed on the figures. These are local previews, not matched-quality rankings.


## Jetson latency, memory and residency preview

[Local figure and method](../output/jetson-residency/README.md) combines no-reuse end-to-end
latency, whole-capture tegrastats RAM and a schematic stage residency policy for four engines.
Reproduce using `scripts/plot_jetson_residency.py`; `--sdcpp-run` selects a completed full-run
replacement. Memory elapsed time is nominal from the sampling interval, not aligned to
individual generation stages. The sd.cpp source is the completed resident+tiled full repeat
`jetson-flux-klein-012__baseline__20261102-165643`. No upload performed.

### Jetson sd.cpp full-repeat refresh (2026-11-02)

The current Jetson roofline/latency/memory, residency, engine speedup and detailed roofline
previews use the [completed resident+tiled full repeat](../experiments/jetson-flux-klein-012.md#completed-full-repeat-2026-11-02)
for Distilled sd.cpp. Base results retain their existing source. Charts are local; the existing
W&B publication has not been updated. Historical campaign records retain their original runs.
