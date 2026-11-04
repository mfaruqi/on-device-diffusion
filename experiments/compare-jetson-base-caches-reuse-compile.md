---
type: experiment-record
id: compare-jetson-base-caches-reuse-compile
status: complete
device: jetson-orin-nano
engine: [edge-dit-cpp, pytorch-diffusers, stable-diffusion-cpp]
runs: [jetson-edgedit-flux-klein-base-001__baseline__20261015-193452, jetson-edgedit-flux-klein-base-002__baseline__20261015-204021, jetson-edgedit-flux-klein-base-003__baseline__20261015-213038, jetson-edgedit-flux-klein-base-004__baseline__20261015-222249, jetson-edgedit-flux-klein-base-005__baseline__20261015-231521, jetson-edgedit-flux-klein-base-006__baseline__20261016-000110, jetson-edgedit-flux-klein-base-007__baseline__20261016-005119, jetson-torch-flux-klein-base-001__baseline__20261016-050136, jetson-torch-flux-klein-base-002__baseline__20261016-015207, jetson-torch-flux-klein-base-004__baseline__20261016-033850, jetson-edgedit-flux-klein-006__baseline__20261014-170149, jetson-edgedit-flux-klein-013__baseline__20261015-135118, jetson-torch-flux-klein-001__baseline__20261014-171410, jetson-torch-flux-klein-004__baseline__20261015-141028, jetson-torch-flux-klein-007__baseline__20261015-145234, jetson-flux-klein-base-001__baseline__20261005-005357, jetson-flux-klein-base-002__baseline__20261005-024556, jetson-flux-klein-base-007__baseline__20261005-121955]
updated: 2026-10-16
---

# Compare: Jetson klein Base (50 steps, CFG 4) step caches, conditioning reuse and torch.compile across edge-dit and PyTorch, and against the 4-step distilled workload

## Question

On the Jetson Orin Nano, which execution levers move the 50-step CFG Base workload, do they act the same way in edge-dit.cpp and PyTorch/diffusers, and does each lever's effect transfer from the 4-step distilled workload? RQ2 (reuse, precision and residency levers under a memory budget); engine context for RQ1.

## Runs

| Run | Engine | Workload | What differs |
|---|---|---|---|
| [jetson-edgedit-flux-klein-base-001](jetson-edgedit-flux-klein-base-001.md) | edge-dit | Base | reference: all resident, VAE tiling, no cache |
| [base-002](jetson-edgedit-flux-klein-base-002.md) … [base-007](jetson-edgedit-flux-klein-base-007.md) | edge-dit | Base | one step cache each (EasyCache, UCache, DBCache, TaylorSeer, cache-dit, MagCache); [base-008](jetson-edgedit-flux-klein-base-008.md) DiCache failed (memory guard) |
| [jetson-torch-flux-klein-base-001](jetson-torch-flux-klein-base-001.md) | PyTorch | Base | reference: sequential residency |
| [base-002](jetson-torch-flux-klein-base-002.md) | PyTorch | Base | + exact conditioning reuse |
| [base-004](jetson-torch-flux-klein-base-004.md) | PyTorch | Base | + conditioning reuse + torch.compile; [base-003](jetson-torch-flux-klein-base-003.md) TaylorSeer failed (memory guard) |
| [jetson-flux-klein-base-001](jetson-flux-klein-base-001.md), [-002](jetson-flux-klein-base-002.md), [-007](jetson-flux-klein-base-007.md) | sd.cpp | Base | disk-backed reference, EasyCache, EasyCache + conditioning reuse (2026-10-05) |
| [jetson-edgedit-flux-klein-006](jetson-edgedit-flux-klein-006.md), [-013](jetson-edgedit-flux-klein-013.md) | edge-dit | distilled | reference; MagCache |
| [jetson-torch-flux-klein-001](jetson-torch-flux-klein-001.md), [-004](jetson-torch-flux-klein-004.md), [-007](jetson-torch-flux-klein-007.md) | PyTorch | distilled | reference; reuse; reuse + compile |

## Matching

| Setting | edge-dit Base | PyTorch Base | sd.cpp Base |
|---|---|---|---|
| Transformer | Base Q4_0 GGUF (same file) | Base Q4_0 GGUF (same file), diffusers GGUF loader, BF16 compute | Base Q4_0 GGUF (same file) |
| Text encoder | Qwen3 Q4_K_M + Q8_0 embedding (GGUF) | Qwen3 NF4 (bitsandbytes) | Qwen3 Q4_K_M (GGUF) |
| Residency | all resident | sequential (encoder only to encode; transformer released meanwhile) | disk-backed parameters |
| VAE | BF16, tiled 128 px / 0.25 | BF16, untiled | BF16, untiled |
| Workload | 512², 50 steps, guidance 4, CFG 4, seed 0, one prompt | same (empty negative prompt) | same |
| Protocol | 1 + 3 + 10, 25W, memory guard | same | same, earlier session (2026-10-05) |

**Confounds**: encoder quantization, residency, VAE tiling and RNG differ between engines, and sd.cpp ran in a different session; **engine-to-engine wall times are qualitative only**. Within-engine comparisons (each lever against its own reference, same queue) are matched except for the one named change. Distilled ↔ Base comparisons change the workload (steps, CFG) and are stated as ratios, not absolute times.

## Results

Numbers from each run's `summary.json` (medians of 10 measured generations).

### Within-engine levers, Base

| Engine | Lever | Wall (s) | Δ vs own reference | Transformer passes computed | Output (visual) |
|---|---|---:|---:|---:|---|
| edge-dit | reference | 232.5 | – | 100 | sign "hello world" |
| edge-dit | TaylorSeer | 146.3 | −37.1% | 62 | closest to reference (3.7% px > 8 levels) |
| edge-dit | EasyCache | 166.7 | −28.3% | 71 | legible, stray stroke |
| edge-dit | UCache | 173.6 | −25.3% | 74 | legible |
| edge-dit | DBCache / cache-dit | 180.1 / 180.2 | −22.5% / −22.5% | 77 | legible, stray stroke (identical outputs) |
| edge-dit | MagCache | 73.1 | −68.6% | 30 | "world" deformed |
| edge-dit | DiCache | failed | – | (54 reused in the one completed pass) | memory guard |
| PyTorch | reference (sequential, 2 swaps / image) | 494.6 | – | 100 | sign "HELLO WORLD" |
| PyTorch | conditioning reuse | 359.4 | −27.3% | 100 | bit-identical |
| PyTorch | reuse + torch.compile | 183.9 | −62.8% (−48.8% vs reuse) | 100 | pixels differ, visually same |
| PyTorch | reuse + TaylorSeer | failed | – | – | memory guard at first encoder load |
| sd.cpp | disk-backed reference | 325.6 | – | – | – |
| sd.cpp | EasyCache | 184.2 | −43.4% | – | see record |
| sd.cpp | EasyCache + conditioning reuse | 141.0 | −56.7% | – | see record |

edge-dit time per computed pass stays at 2.29–2.33 s in every completed cache run (reference 2.29 s), so wall follows the computed-pass count.

### Lever transfer, distilled (4 steps, CFG 1) → Base (50 steps, CFG 4)

| Lever | Distilled effect | Base effect |
|---|---|---|
| edge-dit step caches at defaults | 0 of 4 steps reused for TaylorSeer, DiCache and four others; MagCache 3/4 reused (11.96 → 5.10 s), output degraded | 23–70 of 100 passes reused for six methods; −22% to −69% wall; MagCache again degrades text |
| PyTorch conditioning reuse | 80.3 → 14.9 s (−81.4%), bit-identical | 494.6 → 359.4 s (−27.3%), bit-identical |
| PyTorch torch.compile (on reuse) | steady step 3593 → 1812 ms (−49.6%); wall 14.9 → 7.8 s | steady step 7177 → 3668 ms (−48.9%); wall 359.4 → 183.9 s |
| Memory-guard failures | — | DiCache (edge-dit) and TaylorSeer (PyTorch) both exceed the 400 MB MemAvailable floor |

## Findings

- **Step caches need schedule length.** At 50 CFG steps edge-dit's caches remove 23–70% of transformer passes and the saving is proportional to that count; at 4 steps the same defaults remove nothing except MagCache. This extends [step caching helps at 50 steps but not at 4 on Jetson](../wiki/findings/step-caching-helps-at-50-steps-but-not-at-4-on-jetson.md) from sd.cpp EasyCache to six edge-dit methods.
- **Conditioning reuse's value depends on what it skips.** Its saving is the residency swap cost: about 65 s per `encode_prompt` call in PyTorch, which is 81% of a 4-step image but 27% of a 50-step image. Bit-identical in both.
- **torch.compile transfers unchanged.** It halves the PyTorch transformer step at both schedule lengths on this device (−49.6% and −48.9%), so its relative wall gain grows as the transformer share grows; it costs about 0.6 GiB of headroom at Base.
- **Engine ordering at Base (qualitative).** Fastest text-preserving configurations: sd.cpp EasyCache + reuse 141.0 s and edge-dit TaylorSeer 146.3 s (both approximate caches), then PyTorch reuse + compile 183.9 s (pixels change only through compilation), then edge-dit's exact reference 232.5 s. MagCache (73.1 s) is faster but deforms text. The confounds above make this ordering indicative only.
- **Memory headroom is the binding constraint for stateful levers.** The two failures are both state-adding options (DiCache probes; TaylorSeer hook) on placements whose reference MemAvailable minimum is 568 MB (edge-dit) or whose release path may keep the transformer alive (PyTorch, hypothesis in the record).

## Limits

- One prompt, one seed; quality judged by eye and pixel difference only. No lever has formal quality approval.
- Engine-to-engine numbers carry the encoder, residency, tiling and session confounds listed above.
- Cache parameters are engine defaults; no tuning sweep.
- sd.cpp Base runs are from an earlier session; their per-pass counts are not logged in the same form.
