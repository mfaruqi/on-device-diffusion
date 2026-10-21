# Building roofline plots from saved Klein runs

This is a Week 2 baseline-analysis tool supporting RQ1: “Which diffusion optimization choices transfer across devices and workloads?” ([proposal overview](../wiki/project/overview.md#research-questions), [milestones](../wiki/project/milestones.md#revised-schedule)). It executes no models. Existing run artifacts remain immutable. Outputs, including W&B state, go only to the output directory.

## Run and publish

Dependencies: Python 3.10+, `plotly`; `wandb` and `pillow` for upload with image panels; `kaleido` and Chrome for optional PNG/PDF exports. Install them in a dedicated environment outside timed benchmark execution.

## Paper-style presentation

The plot is constructed from coordinates: arithmetic intensity on X, effective TFLOP/s on Y. The reference curve is `min(peak, bandwidth × intensity)`. Its bend is the ridge point `peak / bandwidth`. Both axes are logarithmic; the horizontal red guide marks the primary compute ceiling, and the vertical green guide marks its ridge. Distinct colours, outlined markers, compact ranges, powers-of-ten ticks, dotted minor grids and an inset legend reproduce common paper-figure styling. A reference screenshot cannot establish which software its authors used. This implementation uses Plotly for both interactive charts and vector PDF exports; Matplotlib or LaTeX/PGFPlots can produce similar graphics.

`--static` additionally writes `m5-paper-roofline.{html,png,pdf}`: side-by-side distilled and Base panels with shared bounds, per-option marker styling and separate legends. All saved M5 stage coordinates are retained exactly, without jitter or averaging repeated source runs. Filled markers denote MPS BF16 and open markers denote sd.cpp mixed Q4; the assumed M5 ceiling remains explicit. The W&B upload adds `roofline_images/*` image panels (including `roofline_images/m5-paper`) as well as responsive interactive `roofline/*` panels. Use the image panels when a fixed paper layout matters. W&B interactive panels use automatic sizing, concise axis titles and a compact scope note to fit dashboard tiles. Full scope notes remain in exports and the data artifact. No benchmark measurement or coordinate formula changes with the styling.

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

Normalization is `100*effective_tflops/min(ceiling, bw_gbs/1000*AI)`, using each point's stated reference ceiling. Nunchaku's mixed stage uses the BF16 reference denominator: neither pure INT4 nor pure BF16 utilization is inferred. Integer roofs are labelled TOP/s and are contextual reference lines. Values over 100% remain visible as an analytic-model discrepancy. These percentages are not matched-quality comparisons or measured hardware utilization. Different workloads, precision, residency, engine timers and profile instrumentation limit cross-device interpretation ([metric definitions](../wiki/methods/baseline-metrics.md)).
