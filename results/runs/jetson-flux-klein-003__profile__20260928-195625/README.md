# Jetson 003: reviewed stage-labelled profile

Status: complete; one profiled generation, no baseline medians.

## Organization

- `originals/`: unchanged files transferred from Jetson; hashes in `artifact-manifest.json`.
- Root config/environment/command/results: copies of originals for standard tooling.
- `status.json`, `summary.json`, `load.json`: curated reviewed metadata; original versions retained.
- `trace-review.json`: SQLite integrity, NVTX boundaries, event counts, actual segmentation and memory summary.
- `timing-diagnostics.json`: host callback durations; includes loading within stages.
- `profile/denoise_kernels.{json,md}` and `profile/other_stages.{json,md}`: existing analyzer's stage GPU breakdowns.
- `tegrastats-samples.csv`: 100 whole-system samples; no synchronized stage-memory attribution.
- `output.png`: lossless serialization of `originals/raw/run-0.rgb`; smoke check in `image-inspection.json`.

The JSON/JSONL excerpts previously transcribed from terminal output agree with imported timestamps
and checks. The imported configuration matches the planned config. All eight expected NVTX
ranges appear once. Generation stages are ordered and nested within `generate`; captured GPU
activities do not cross their stage ends. This checks the captured timeline, not absence of
all possible profiler losses. Qwen, diffusion and VAE each report one graph segment.

## Reproduce review

From the repository root (standard-library Python; no GPU required):

```bash
python3 scripts/review_jetson_profile.py --run-dir results/runs/jetson-flux-klein-003__profile__20260928-195625
```

## W&B handoff

Dry-run of the existing exporter succeeds with one image, tegrastats, and both kernel tables.
No upload was performed. Claude can export this exact run using `--only` with its full directory
name. Keep kind `profile`, quantized 512-square workload, `measured: null`, and no baseline history.
The exporter sees zero generation-history rows because runs.csv intentionally has headers only.
Host diagnostics remain in timing-diagnostics.json rather than median timing fields.

The original `.nsys-rep`, `.sqlite` and `.rgb` files exceed the exporter's selected file types;
upload them explicitly as separate artifacts if desired. Do not commit these binaries.
The exporter includes small logs and JSON/CSV/Markdown files as artifacts.

## Interpretation limits

Kernel-table `busy_ms` means union of captured GPU activities (kernels and copies), not SM
utilization. `idle_ms` is the uncovered part of the stage window, not measured disk latency.
NVTX summary percentages overlap because `generate` encloses stage ranges; do not sum them.
The first step includes weight loading. Whole-window sampled RAM peaks include startup.
Swap occupancy increased; occupancy alone does not measure swap I/O volume.
