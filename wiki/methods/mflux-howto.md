---
type: method
summary: Pinned MFLUX/MLX Klein on M5, with separate BF16, 8-bit and 4-bit smoke-gated protocols.
status: active
updated: 2026-10-12
---

# MFLUX / MLX on M5

Week 2 engine coverage carried into Week 3, RQ1. The bounded set is distilled and Base Klein 4B,
each at BF16 (unquantized), 8-bit and 4-bit. Each configuration must pass one first plus one measured
smoke before its one first, three warm-ups and ten measured full protocol. Operational state and
attempt paths are in [state.json](../../output/mflux-prep/state.json).

## Implementation and pins

[Runner](../../scripts/run_mflux.py), [adapter](../../scripts/mflux_engine.py),
[contract tests](../../tests/test_mflux.py). MFLUX 0.22.0 is installed from commit
`f0bf9cf0506a2ae15115af1c72e3dc0093e41597` in a separate `.venv-m5-mflux` environment.
The adapter verifies installed Python sources against that checkout before loading. Source and
package provenance are saved per run. MLX 0.32.3 detects Apple M5 Metal.

The same pinned original BF16 checkpoints as the existing M5 PyTorch references are used:
[distilled config](../../configs/m5-mflux-flux-klein-bf16-512.resolved.json) and
[Base config](../../configs/m5-mflux-flux-klein-base-bf16-512.resolved.json). Preparation saves
file SHA256/size inventories and loads by the exact local snapshot directory; generation is offline.
All configurations are 512², batch one, seed zero and the existing cat/sign prompt. Distilled is
four steps/CFG 1; Base is fifty steps/CFG 4. The negative prompt is explicitly empty.

`quantize=8` or `4` uses the pinned engine's affine weight quantization with group size 64 across
eligible component layers. It is not Nunchaku/SVDQuant. `quantization-audit.json` records actual
layer bit widths, group sizes and tensor dtypes. Other floating weights remain BF16; the native
FP32 `text_encoder.rotary_emb.inv_freq` buffer (64 values) is explicitly allowed. No text reuse,
VAE tiling, residency change or alternate precision fallback is enabled. Native prediction
compilation remains enabled in all configurations.

## Timing and validation

Follow [MFLUX metric definitions](baseline-metrics.md#mflux--mlx-on-m5). Stages are synchronized
host times, including lazy array evaluation; they must not be labelled GPU events. Denoising
includes CFG and scheduler updates. Initial load/on-load quantization is separate, first-use
compilation remains in the first generation, and PNG writes are outside generation timing.
Mac system memory/swap and wall-versus-monotonic continuity are sampled separately from MLX
allocator peaks. Background applications are not controlled, so comparisons remain descriptive.

The smoke gate checks exact workload/quantization identity, phase counts, complete status,
clock continuity, finite positive timings and saved image dimensions/hashes/nonflat pixels.
Functional admission does not establish formal quality eligibility. Save all images; within-engine
image differences are diagnostics, and cross-engine seeds do not imply matched initial noise.

```sh
# Preparation only; repeat for each distinct pinned source checkpoint.
HF_HUB_DISABLE_XET=1 .venv-m5-mflux/bin/python scripts/run_mflux.py \
  --config configs/m5-mflux-flux-klein-bf16-512.json --prepare-only

# Run only after verifying no other inference is active; keep AC connected and lid open.
HF_HUB_OFFLINE=1 .venv-m5-mflux/bin/python scripts/run_mflux.py \
  --config configs/m5-mflux-flux-klein-bf16-512-smoke.resolved.json
HF_HUB_OFFLINE=1 .venv-m5-mflux/bin/python scripts/run_mflux.py \
  --config configs/m5-mflux-flux-klein-bf16-512.resolved.json \
  --smoke-run results/runs/ACTUAL_MATCHING_SMOKE
```

The local one-off queue in `output/mflux-prep/run_queue.py` executes only the six recorded
configurations sequentially. It preserves completed work, stops full admission after a failed
smoke and refuses concurrent inference. Reconcile state, real processes and outputs before any
restart; a stale running status alone does not authorize relaunching. Documentation and W&B
uploads happen outside timed inference, then validated full runs are added to the existing
[Klein charts](../../output/stage-breakdown-20261007/README.md).

The bounded queue finished on 2026-10-12: five full protocols completed and Base BF16 was blocked by a native MLX Metal crash before generation. Twelve records were uploaded, including both preserved failures ([evidence verification](../../output/mflux-prep/wandb-evidence-verification.json)); the five completed full protocols were added to the existing charts. No further runs are queued. The local
`output/mflux-prep/document_runs.py` refreshes only its experiment/index/registry entries.
After the queue has stopped and actual processes confirm no inference, the prepared
`output/mflux-prep/finalize.py` validates completed runs, uploads through the existing exporter,
adds completed sources to the existing M5 engine panels plus two MFLUX precision panels,
and verifies remote timings/history and the chart-data artifact. It must be run with
`.venv-m5-mflux/bin/python`; static plots use the isolated chart environment. Rendered figures and chart README, hot page and log were reviewed and updated at completion.
Do not infer publishing completion from queue completion alone; inspect the upload receipts.
