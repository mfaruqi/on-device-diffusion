---
type: experiment-record
id: jetson-edgedit-flux-klein-base-008
status: failed
device: jetson-orin-nano
engine: edge-dit-cpp
runs: [jetson-edgedit-flux-klein-base-008__attempt__20261016-011113, jetson-edgedit-flux-klein-base-008__attempt__20261015-164533]
updated: 2026-10-16
---

# jetson-edgedit-flux-klein-base-008: edge-dit.cpp step cache `dicache` (engine defaults), klein Base Q4_0 512², 50 steps CFG 4; stopped by the memory guard at the start of the second pass, twice

Status: **failed** (2026-10-16 05:11:13–05:15:33 UTC; first attempt 2026-10-15 20:45:33 UTC). Both smoke attempts were stopped by [jetson_memguard.sh](../scripts/jetson_memguard.sh) after one completed generation; no full protocol was run.

## Question

Does edge-dit.cpp's `dicache` step cache, at its default parameters, skip or reuse transformer work in the 50-step CFG klein Base placement [jetson-edgedit-flux-klein-base-001](jetson-edgedit-flux-klein-base-001.md), and does it fit in memory? Single change against [jetson-edgedit-flux-klein-base-q4-embq8-512-vaetile.json](../configs/jetson-edgedit-flux-klein-base-q4-embq8-512-vaetile.json). Step-reuse lever under a memory budget (RQ2).

## Setup

- **Config**: smoke [jetson-edgedit-flux-klein-base-q4-embq8-512-vaetile-dicache-smoke.json](../configs/jetson-edgedit-flux-klein-base-q4-embq8-512-vaetile-dicache-smoke.json) (first + one measured); full [jetson-edgedit-flux-klein-base-q4-embq8-512-vaetile-dicache.json](../configs/jetson-edgedit-flux-klein-base-q4-embq8-512-vaetile-dicache.json) not run.
- **Change vs reference (only one)**: `optimizations.step_cache: {"method": "dicache"}` → `--cache_method dicache`, engine defaults. Everything else as in the [base-001 setup](jetson-edgedit-flux-klein-base-001.md#setup).
- **Guard**: 400 MB MemAvailable floor, 768 MB swap-growth limit, 0.1 s poll; the guard sends SIGKILL, so `status.json` was left at `running` and was corrected afterwards to `failed` from `memguard.json` (`status_correction` field in each run's [status.json](../results/runs/jetson-edgedit-flux-klein-base-008__attempt__20261016-011113/status.json)).
- The 2026-10-15 attempt ran in the first Base queue, whose runner had the parser defect described in [base-001](jetson-edgedit-flux-klein-base-001.md#setup); the guard stopped it before the parser ran, so the defect does not affect this outcome.

## Results

| Attempt | Guard reason | MemAvailable start → min | Swap | Completed generations |
|---|---|---|---|---|
| [20261016-011113](../results/runs/jetson-edgedit-flux-klein-base-008__attempt__20261016-011113/) | `MemAvailable 371 MB < 400 MB`, exit 137 | 5982 → 371 MB | 315 MB, no growth | 1 (first) |
| [20261015-164533](../results/runs/jetson-edgedit-flux-klein-base-008__attempt__20261015-164533/) | `MemAvailable 390 MB < 400 MB`, exit 137 | 5995 → 390 MB | 320 MB, no growth | 1 (first) |

Partial measurements from [engine.log](../results/runs/jetson-edgedit-flux-klein-base-008__attempt__20261016-011113/engine.log) (first generation only, not a baseline):

- Cache receipt: `cache enabled: mode=DiCache model=flux2-klein granularity=probe matched_sites=2 variants=3 slots=4`; `DiCache reused 54/50 steps` (k counts transformer passes out of 100).
- First generation breakdown: total 121.60 s (text_encode 1.77 / sampling 114.10 / vae_decode 4.65 / other 1.07 s); the 2026-10-15 attempt: 121.35 s (1.71 / 114.25 / 4.31 / 1.07 s). Reference first generation: 238.765 s (sampling 231.28 s).
- In both attempts the guard fired during the second pass: the log ends at `flux step 1/50` of pass 2 (51 `flux step` lines in all). The last tegrastats sample reads RAM 7038/7620 MB.

No summary.json, no measured run, no image retained.

## Observations

- DiCache reused 54 of 100 passes in the one completed generation, and that generation's sampling took 114.1 s against 231.3 s for the reference first generation.
- Both attempts were stopped at the same point (start of the second pass) with MemAvailable 371 and 390 MB, below the 400 MB floor; swap did not grow.
- The engine reports `variants=3` and `slots=4` for DiCache, against `variants=2 slots=1` for every other cache method on this workload.

## Interpretation

- Hypothesis: DiCache's probe state (two matched sites, three variants, four slots) adds enough device memory on top of the resident placement, whose reference minimum MemAvailable is 568 MB, to cross the 400 MB floor once a second pass starts; the guard cannot say which allocation crossed it.
- The guard floor is a fixed safety limit for the campaign, not part of the configuration; lowering it to let this run continue would change the risk to the board, not the placement under test.

## Next experiment

- Copy of `jetson-edgedit-flux-klein-base-q4-embq8-512-vaetile-dicache.json` with the text encoder released after encode (frees the encoder's resident weights), if the engine exposes that option.
