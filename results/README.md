# Results index

One row per run directory under `results/runs/`. Numbers are medians of measured runs. stable-diffusion.cpp runs have no allocator metric; their device-wide peak is shown instead. See
`wiki/methods/baseline-metrics.md` for definitions. Images and profiler traces in each run
directory are git-ignored (kept locally, not in git).

| Run directory | Experiment note | Device | Config | End-to-end (ms) | Text / Denoise / VAE (ms) | Peak alloc (GiB) | Status |
|---|---|---|---|---|---|---|---|
| `a100-nunchaku-flux-klein-001__attempt__20261011-123802` | [a100-nunchaku-flux-klein-001](../experiments/a100-nunchaku-flux-klein-001.md) | gilbreth-a100-40gb (PCIe, g009) | a100-nunchaku-flux-klein-int4-r32 | 904.8 | 48.9 / 658.6 / 164.4 | 12.52 | complete (job 11929464): smoke; quality unassessed |
| `a100-nunchaku-flux-klein-001__baseline__20261011-124123` | [a100-nunchaku-flux-klein-001](../experiments/a100-nunchaku-flux-klein-001.md) | gilbreth-a100-40gb (PCIe, g009) | a100-nunchaku-flux-klein-int4-r32 | 897.8 | 47.1 / 663.2 / 164.8 | 12.52 | complete (job 11929464): full protocol; quality unassessed |
| `a100-nunchaku-flux-klein-001__attempt__20261011-113031` | [a100-nunchaku-flux-klein-001](../experiments/a100-nunchaku-flux-klein-001.md) | gilbreth-a100-40gb (PCIe, g005) | a100-nunchaku-flux-klein-int4-r32-smoke | – | – | – | failed (job 11929279): text-encoder dtype guard before generation |
| `a100-flux-klein-001__baseline__20260923-221530` | [a100-flux-klein-001](../experiments/a100-flux-klein-001.md) | gilbreth-a100-40gb (PCIe, g008) | a100-flux-klein-bf16 | 1240.6 | 47.0 / 980.8 / 165.3 | 17.32 | complete (job 11807390) |
| `a100-flux-klein-001__profile__20260923-221717` | [a100-flux-klein-001](../experiments/a100-flux-klein-001.md) | gilbreth-a100-40gb (PCIe, g008) | a100-flux-klein-bf16 + `--profile` | 1236.8 | 47 / 981 / 165 | 17.32 | complete (job 11807396) |
| `a100-flux-klein-001__repeat__20260925-123908` | [a100-flux-klein-001](../experiments/a100-flux-klein-001.md) | gilbreth-a100-40gb (PCIe, g006) | a100-flux-klein-bf16 (repeat after runner refactor) | 1229.6 | 47.5 / 979.5 / 165.9 | 17.32 | complete (job 11817919) |
| `a100-sdcpp-flux-klein-001__baseline__20260925-114138` | [a100-sdcpp-flux-klein-001](../experiments/a100-sdcpp-flux-klein-001.md) | gilbreth-a100-40gb (**SXM4**, n001) | a100-sdcpp-flux-klein-bf16 | 2294.8 | 60.7 / 1625.6 / 585.4 | – (device 22.88) | complete (job 11817893); SXM4 node, not comparable with PCIe runs |
| `a100-sdcpp-flux-klein-001__baseline__20260925-114328` | [a100-sdcpp-flux-klein-001](../experiments/a100-sdcpp-flux-klein-001.md) | gilbreth-a100-40gb (PCIe, g004) | a100-sdcpp-flux-klein-bf16 | 2423.3 | 64.7 / 1725.4 / 603.5 | – (device 22.88) | complete (job 11817906) |
| `a100-sdcpp-flux-klein-001__profile__20260925-124021` | [a100-sdcpp-flux-klein-001](../experiments/a100-sdcpp-flux-klein-001.md) | gilbreth-a100-40gb (PCIe, g006) | a100-sdcpp-flux-klein-bf16 + `--nsys` | – | – | – | failed (job 11817920): nsys could not attach to the harness |
| `a100-sdcpp-flux-klein-001__profile__20260925-124703` | [a100-sdcpp-flux-klein-001](../experiments/a100-sdcpp-flux-klein-001.md) | gilbreth-a100-40gb (PCIe, g005) | a100-sdcpp-flux-klein-bf16 + `--nsys` | 2510.3 | 66.4 / 1785.1 / 629.2 | – (device 22.88) | complete (job 11818087); profiled, not a baseline number |

Jetson feasibility attempt (partial evidence import; no benchmark medians):

| Run directory | Experiment note | Device | Config | End-to-end (ms) | Text / Denoise / VAE (ms) | Peak alloc (GiB) | Status |
|---|---|---|---|---|---|---|---|
| `jetson-flux-klein-001__attempt__20260928-183618` | [jetson-flux-klein-001](../experiments/jetson-flux-klein-001.md) | jetson-orin-nano | jetson-flux-klein-q4-512-all-resident | – | – | – | failed (exit 1): insufficient memory during denoising preparation; full logs pending |
| `jetson-flux-klein-002__attempt__20260928-184410` | [jetson-flux-klein-002](../experiments/jetson-flux-klein-002.md) | jetson-orin-nano | jetson-flux-klein-q4-512-segmented | – | – | – | failed (exit 1): text-encoding memory capacity; full logs pending |
| `jetson-flux-klein-003__attempt__20260928-185009` | [jetson-flux-klein-003](../experiments/jetson-flux-klein-003.md) | jetson-orin-nano | jetson-flux-klein-q4-512-segmented-disk | – | – | – | complete (exit 0): one image inspected; full CLI logs analyzed, feasibility only |
| `jetson-flux-klein-003__profile__20260928-195625` | [jetson-flux-klein-003](../experiments/jetson-flux-klein-003.md) | jetson-orin-nano | jetson-flux-klein-stage-profile | – | – | – | complete; imported timeline reviewed; single profile |
| `jetson-flux-klein-003__profile__20260928-194910` | [jetson-flux-klein-003](../experiments/jetson-flux-klein-003.md) | jetson-orin-nano | jetson-flux-klein-stage-profile | – | – | – | generation succeeded; stage validation failed |
| `jetson-flux-klein-003__profile__20260928-191418` | [jetson-flux-klein-003](../experiments/jetson-flux-klein-003.md) | jetson-orin-nano | jetson-flux-klein-q4-512-segmented-disk-profile | – | – | – | CUDA events confirmed in supplied stats; original trace import pending |

## Jetson repeated baseline

| Run directory | Experiment note | Device | Config | End-to-end (ms) | Text / Denoise / VAE (ms) | Peak alloc (GiB) | Status |
|---|---|---|---|---|---|---|---|
| `jetson-flux-klein-003__baseline__20260930-193208` | [jetson-flux-klein-003](../experiments/jetson-flux-klein-003.md#repeated-unprofiled-baseline-2026-09-30) | Jetson Orin Nano, 25 W | jetson-flux-klein-q4-512-disk-baseline | 66117.0 | 23109.3 / 38773.0 / 4547.1 | – (system RAM 6.078, whole capture) | complete; 10 measured; [W&B](https://wandb.ai/mfaruqi-purdue-university/on-device-diffusion/runs/jetson-flux-klein-003__baseline__20260930-193208) |

## Jetson targeted profile

Single traced generation after first + three warm-ups; diagnostic, not a repeated baseline.

| Run directory | Experiment note | Device | Config | End-to-end (ms) | Text / Denoise / VAE (ms) | Peak alloc (GiB) | Status |
|---|---|---|---|---|---|---|---|
| `jetson-flux-klein-003__profile__20260930-201956` | [jetson-flux-klein-003](../experiments/jetson-flux-klein-003.md#targeted-later-generation-profile-2026-09-30) | Jetson Orin Nano, 25 W | jetson-flux-klein-q4-512-disk-profile | 60916.3 | 18910.1 / 37087.0 / 4637.0 | – (system RAM 6.087, whole capture) | complete; one profiled generation, baseline-ineligible; [W&B](https://wandb.ai/mfaruqi-purdue-university/on-device-diffusion/runs/jetson-flux-klein-003__profile__20260930-201956) |

## Jetson full-protocol profile

Loading and all 14 generations traced; statistics use ten measured generations only.

| Run directory | Experiment note | Device | Config | End-to-end (ms) | Text / Denoise / VAE (ms) | Peak alloc (GiB) | Status |
|---|---|---|---|---|---|---|---|
| `jetson-flux-klein-003__profile__20260930-210728` | [jetson-flux-klein-003](../experiments/jetson-flux-klein-003.md#full-protocol-profile-2026-09-30) | Jetson Orin Nano, 25 W | jetson-flux-klein-q4-512-disk-profile-full | 59318.9 | 20202.6 / 37629.4 / 4654.5 | – (system RAM 6.114, whole capture) | complete; 10 measured, all 14 profiled; baseline-ineligible; [W&B verified](https://wandb.ai/mfaruqi-purdue-university/on-device-diffusion/runs/jetson-flux-klein-003__profile__20260930-210728) |

## Runner refactor smoke checks

Reduced protocol (1 first, no warmup, 1 measured); functional checks, not baseline estimates.

| Run directory | Experiment note | Device | Config | End-to-end (ms) | Text / Denoise / VAE (ms) | Peak alloc (GiB) | Status |
|---|---|---|---|---|---|---|---|
| `a100-flux-klein-001__attempt__20260929-001645` | [a100-flux-klein-001](../experiments/a100-flux-klein-001.md#runner-refactor-smoke-check-2026-09-29) | A100 PCIe, g005 | BF16, reduced smoke protocol | – | – | – | complete (job 11839376); functional only |
| `a100-sdcpp-flux-klein-001__attempt__20260929-001722` | [a100-sdcpp-flux-klein-001](../experiments/a100-sdcpp-flux-klein-001.md#runner-refactor-launch-failure-2026-09-29) | A100 PCIe, g005 | BF16, reduced smoke protocol | – | – | – | failed (job 11839376); CUDA shared library missing before load |
| `a100-sdcpp-flux-klein-001__attempt__20260929-005352` | [a100-sdcpp-flux-klein-001](../experiments/a100-sdcpp-flux-klein-001.md#runner-refactor-retry-passed-2026-09-29) | A100 PCIe, g010 | BF16, reduced smoke protocol | – | – | – | complete (job 11839470); launch fix verified, functional only |
| `jetson-flux-klein-003__attempt__20260930-192732` | [jetson-flux-klein-003](../experiments/jetson-flux-klein-003.md#repeated-run-integration-smoke-check-2026-09-30) | Jetson Orin Nano, 25 W | Q4 512×512 disk-backed, reduced smoke protocol | – | – | – | complete; two images hash-identical, functional only |

## Run directory names

`<experiment-id>__<kind>__<YYYYMMDD-HHMMSS>`, where kind is `baseline` (clean timing run), `profile`
(profiler attached; not quoted as latency), `repeat` (a baseline re-run to check reproducibility) or
`attempt` (a single feasibility try, not the full protocol). The runners create these names (`--kind`
overrides the default). W&B runs use the directory name as their id.

Renamed on 2026-09-28. Files inside a run (`environment.json`, `engine/command.json`, READMEs) still
show the old path from when they were written:

| Old directory name | New directory name |
|---|---|
| `a100-flux-klein-001-20260923-221530` | `a100-flux-klein-001__baseline__20260923-221530` |
| `a100-flux-klein-001-20260923-221717-profile` | `a100-flux-klein-001__profile__20260923-221717` |
| `a100-flux-klein-001-20260925-123908` | `a100-flux-klein-001__repeat__20260925-123908` |
| `a100-sdcpp-flux-klein-001-20260925-114138` | `a100-sdcpp-flux-klein-001__baseline__20260925-114138` |
| `a100-sdcpp-flux-klein-001-20260925-114328` | `a100-sdcpp-flux-klein-001__baseline__20260925-114328` |
| `a100-sdcpp-flux-klein-001-20260925-124021-profile` | `a100-sdcpp-flux-klein-001__profile__20260925-124021` |
| `a100-sdcpp-flux-klein-001-20260925-124703-profile` | `a100-sdcpp-flux-klein-001__profile__20260925-124703` |
| `jetson-flux-klein-q4-512-20260928-183618` | `jetson-flux-klein-001__attempt__20260928-183618` |
| `jetson-flux-klein-002-20260928-184410` | `jetson-flux-klein-002__attempt__20260928-184410` |
| `jetson-flux-klein-003-20260928-185009` | `jetson-flux-klein-003__attempt__20260928-185009` |

## Files in each run directory

| File | Purpose |
|---|---|
| `config.json` | Exact workload and checkpoint revision |
| `environment.json` | GPU, driver, packages, CUDA runtime, git state, Slurm job |
| `environment-freeze.txt` | `pip freeze` |
| `scheduler.json` | Scheduler config from the loaded pipeline |
| `load.json` | Cached load/placement time, memory after load, per-component weight sizes |
| `runs.csv` | One row per generation (first / warmup / measured), with stage times and memory |
| `stages.csv` | Long format: every stage of every run, GPU and host time, allocator memory |
| `memory_segments.json` | Allocator peaks for every stage and every gap between stages |
| `summary.json` | Median / min / max / stdev over measured runs, first run, determinism flag |
| `nvidia-smi.txt` | Snapshot after the measured runs (context only, not a peak) |
| `images/` | `first.png`, `measured-0.png` for a functional check (git-ignored) |
| `profile/` | `op_table.txt` (committed), `trace.json` (git-ignored), only with `--profile`; `denoise_kernels.md/.json` from `scripts/analyze_profile.py` |
| `engine/` | stable-diffusion.cpp runs only: harness command, sd.cpp log, raw stage timestamps (see [sdcpp-howto.md](../wiki/methods/sdcpp-howto.md)) |
| `status.json` | running / complete / failed (+ traceback) |
| `events.jsonl` | New successful captures: derived load/stage observations and explicit engine buffer messages; unknown timestamps/destinations stay null |
| `events-metadata.json` | Event schema version, source hashes and limitations; see [definitions](../wiki/methods/baseline-metrics.md#derived-event-records). Historical runs are not rewritten |

## Automatic placement attempt

| Run directory | Experiment note | Device | Config | End-to-end (ms) | Text / Denoise / VAE (ms) | Peak alloc (GiB) | Status |
|---|---|---|---|---|---|---|---|
| `jetson-flux-klein-004__attempt__20261005-000226` | [jetson-flux-klein-004](../experiments/jetson-flux-klein-004.md) | Jetson Orin Nano, 25 W | jetson-flux-klein-q4-512-autofit-smoke | – | – | – | failed: text-encoding memory capacity; no completed image |

## Lazy-loading measurements

| Run directory | Experiment note | Device | Config | End-to-end (ms) | Text / Denoise / VAE (ms) | Peak alloc (GiB) | Status |
|---|---|---|---|---|---|---|---|
| `jetson-flux-klein-005__attempt__20261005-000734` | [jetson-flux-klein-005](../experiments/jetson-flux-klein-005.md) | Jetson Orin Nano, 25W | jetson-flux-klein-q4-512-disk-lazy-smoke | – | – | – | interrupted by user; not an engine failure |
| `jetson-flux-klein-005__attempt__20261005-000956` | [jetson-flux-klein-005](../experiments/jetson-flux-klein-005.md) | Jetson Orin Nano, 25W | jetson-flux-klein-q4-512-disk-lazy-smoke | – | – | – | complete; first + one observation, not a baseline |
| `jetson-flux-klein-005__baseline__20261005-001522` | [jetson-flux-klein-005](../experiments/jetson-flux-klein-005.md) | Jetson Orin Nano, 25W | jetson-flux-klein-q4-512-disk-lazy | 60922.1 | 21211.0 / 35970.2 / 4578.5 | – | complete; unprofiled 1+3+10 |

## Jetson Base reference

| Run directory | Experiment note | Device | Config | End-to-end (ms) | Text / Denoise / VAE (ms) | Peak alloc (GiB) | Status |
|---|---|---|---|---|---|---|---|
| `jetson-flux-klein-base-001__attempt__20261005-003943` | [jetson-flux-klein-base-001](../experiments/jetson-flux-klein-base-001.md) | Jetson Orin Nano25W | jetson-flux-klein-base-q4-512-disk-smoke | – | – | – | complete; two-observation correctness test, not baseline |
| `jetson-flux-klein-base-001__baseline__20261005-005357` | [jetson-flux-klein-base-001](../experiments/jetson-flux-klein-base-001.md) | Jetson Orin Nano, 25W | jetson-flux-klein-base-q4-512-disk | 325631.8 | 24357.3 / 298587.2 / 4539.3 | – | complete; unprofiled 1+3+10 |
| `jetson-flux-klein-base-001__attempt__20261005-021728` | [jetson-flux-klein-base-001](../experiments/jetson-flux-klein-base-001.md) | Jetson Orin Nano, 25W | jetson-flux-klein-base-q4-512-disk-smoke | – | – | – | complete; two-observation no-cache harness control |
| `jetson-flux-klein-base-002__attempt__20261005-023028` | [jetson-flux-klein-base-002](../experiments/jetson-flux-klein-base-002.md) | Jetson Orin Nano, 25W | jetson-flux-klein-base-q4-512-easycache-smoke | – | – | – | complete; two-observation cache correctness attempt |
| `jetson-flux-klein-base-002__baseline__20261005-024556` | [jetson-flux-klein-base-002](../experiments/jetson-flux-klein-base-002.md) | Jetson Orin Nano, 25W | jetson-flux-klein-base-q4-512-easycache | 184192.8 | 22132.5 / 158403.9 / 4454.5 | – | complete; unprofiled1+3+10 |

| `jetson-flux-klein-base-001__profile__20261005-033141` | [jetson-flux-klein-base-001](../experiments/jetson-flux-klein-base-001.md) | Jetson Orin Nano,25W | jetson-flux-klein-base-q4-512-disk-profile | – | – | – | complete; targeted generation4 profile, separate from baseline |

| `jetson-flux-klein-base-001__repeat__20261005-042014` | [jetson-flux-klein-base-001](../experiments/jetson-flux-klein-base-001.md) | Jetson Orin Nano,25W | jetson-flux-klein-base-q4-512-disk | 328719.4 | 24192.2 / 299599.8 / 4598.8 | – | complete; unprofiled1+3+10 on cache-capable binary |

| `jetson-flux-klein-base-003__attempt__20261005-054530` | [jetson-flux-klein-base-003](../experiments/jetson-flux-klein-base-003.md) | Jetson Orin Nano,25W | jetson-flux-klein-base-q4-512-disk-lazy-smoke | – | – | – | complete; two-observation lazy-loading correctness attempt |

## A100 Base reference

| Run directory | Experiment note | Device | Config | End-to-end (ms) | Text / Denoise / VAE (ms) | Peak alloc (GiB) | Status |
|---|---|---|---|---|---|---|---|
| `a100-flux-klein-base-001__attempt__20261005-033729` | [a100-flux-klein-base-001](../experiments/a100-flux-klein-base-001.md) | A100-PCIE-40GB | a100-flux-klein-base-bf16-smoke | – | – | – | complete; job11880735, two-observation correctness attempt |

| `a100-sdcpp-flux-klein-base-001__attempt__20261005-040120` | [a100-sdcpp-flux-klein-base-001](../experiments/a100-sdcpp-flux-klein-base-001.md) | A100-PCIE-40GB | a100-sdcpp-flux-klein-base-bf16-smoke | – | – | – | complete; job11880759, two-observation correctness attempt |

| `jetson-flux-klein-base-003__baseline__20261005-060028` | [jetson-flux-klein-base-003](../experiments/jetson-flux-klein-base-003.md) | Jetson Orin Nano,25W | jetson-flux-klein-base-q4-512-disk-lazy | 330683.062 | 24536.232 / 301083.412 / 4643.861 | – (system 4.157) | complete; validated1+3+10 lazy-loading protocol |

| `a100-edgedit-flux-klein-001__attempt__20261005-073914` | [a100-edgedit-flux-klein-001](../experiments/a100-edgedit-flux-klein-001.md) | A100 PCIe | a100-edgedit-flux-klein-bf16-smoke | – | – | – | complete functional gate11881347; repeated timing unavailable |

| `a100-flux-klein-torchtrt-001__attempt__20261005-074014` | [a100-flux-klein-torchtrt-001](../experiments/a100-flux-klein-torchtrt-001.md) | A100 PCIe | a100-flux-klein-torchtrt-smoke | – | – | – | failed11881410; compiler input-layout rejection |

| `jetson-flux-klein-base-004__attempt__20261005-073439` | [jetson-flux-klein-base-004](../experiments/jetson-flux-klein-base-004.md) | Jetson Orin Nano,25W | jetson-flux-klein-base-q4-512-no-prefetch-smoke | – | – | – | complete; two-observation correctness attempt |

| `a100-flux-klein-base-001__baseline__20261005-074425` | [a100-flux-klein-base-001](../experiments/a100-flux-klein-base-001.md) | A100 PCIe,g007 | a100-flux-klein-base-bf16 | 25147.679 | 94.795 / 24836.578 / 167.762 | 17.332 | complete11882219; validated1+3+10 |
| `a100-flux-klein-base-001__profile__20261005-075123` | [a100-flux-klein-base-001](../experiments/a100-flux-klein-base-001.md) | A100 PCIe,g007 | a100-flux-klein-base-bf16 | – | – | – | complete11882223; separate extra torch.profiler capture, preceding protocol retained |
| `a100-sdcpp-flux-klein-base-001__baseline__20261005-080118` | [a100-sdcpp-flux-klein-base-001](../experiments/a100-sdcpp-flux-klein-base-001.md) | A100 PCIe,g007 | a100-sdcpp-flux-klein-base-bf16 | 43686.406 | 119.130 / 42923.012 / 612.247 | – (device 22.880) | complete11882493; validated1+3+10 |
| `a100-sdcpp-flux-klein-base-001__profile__20261005-081219` | [a100-sdcpp-flux-klein-base-001](../experiments/a100-sdcpp-flux-klein-base-001.md) | A100 PCIe,g007 | a100-sdcpp-flux-klein-base-bf16 + Nsight | – | – | – | failed11882497; Nsight process-probe timeout before generation |
| `a100-flux-klein-torchtrt-002__attempt__20261005-081316` | [a100-flux-klein-torchtrt-002](../experiments/a100-flux-klein-torchtrt-002.md) | A100 PCIe,g007 | torchtrt contiguous BF16 gate | – | – | – | failed11883257; BF16 scalar-multiply conversion |
| `a100-sdcpp-flux-klein-base-001__attempt__20261005-082117` | [a100-sdcpp-flux-klein-base-001](../experiments/a100-sdcpp-flux-klein-base-001.md) | A100 PCIe,g007 | cache-capable no-cache correctness control | – | – | – | complete11883351; two images, same reference pixels |

| `jetson-flux-klein-base-004__baseline__20261005-074852` | [jetson-flux-klein-base-004](../experiments/jetson-flux-klein-base-004.md) | Jetson Orin Nano,25W | Base prefetch disabled | 320877.059 | 21929.209 / 295679.244 / 4367.119 | – (system 6.046) | complete; validated1+3+10 |

| `jetson-flux-klein-base-005__attempt__20261005-091031` | [jetson-flux-klein-base-005](../experiments/jetson-flux-klein-base-005.md) | Jetson Orin Nano,25W | Base exact conditioning reuse | – | – | – | complete; two-observation correctness attempt |

| `jetson-flux-klein-base-005__baseline__20261005-093239` | [jetson-flux-klein-base-005](../experiments/jetson-flux-klein-base-005.md) | Jetson Orin Nano,25W | Base exact conditioning reuse | 279605.293 | 13.342 / 276089.382 / 3334.923 | – (system 6.022) | complete; validated1+3+10 |
| `jetson-flux-klein-003__attempt__20261005-104044` | [jetson-flux-klein-003](../experiments/jetson-flux-klein-003.md) | Jetson Orin Nano,25W | distilled no-cache control on the cache-capable harness | – | – | – | complete; two-generation functional attempt, full timing pending |
| `jetson-flux-klein-006__attempt__20261005-104417` | [jetson-flux-klein-006](../experiments/jetson-flux-klein-006.md) | Jetson Orin Nano,25W | distilled EasyCache | – | – | – | complete; two-generation functional attempt, full timing pending |
| `jetson-flux-klein-007__attempt__20261005-104736` | [jetson-flux-klein-007](../experiments/jetson-flux-klein-007.md) | Jetson Orin Nano,25W | distilled exact conditioning reuse | – | – | – | complete; two-generation functional attempt, full timing pending |
| `jetson-flux-klein-008__attempt__20261005-105027` | [jetson-flux-klein-008](../experiments/jetson-flux-klein-008.md) | Jetson Orin Nano,25W | distilled prefetch disabled | – | – | – | complete; two-generation functional attempt, full timing pending |
| `jetson-flux-klein-009__attempt__20261005-105344` | [jetson-flux-klein-009](../experiments/jetson-flux-klein-009.md) | Jetson Orin Nano,25W | distilled memory-mapped weight-file I/O | – | – | – | complete; two-generation functional attempt, full timing pending |
| `jetson-flux-klein-003__baseline__20261005-105911` | [jetson-flux-klein-003](../experiments/jetson-flux-klein-003.md) | Jetson Orin Nano,25W | four-step same-harness no-cache control | 60935.979 | 22139.018 / 36899.651 / 4563.792 | – (system 5.997) | complete; validated1+3+10 |
| `jetson-flux-klein-006__baseline__20261005-111428` | [jetson-flux-klein-006](../experiments/jetson-flux-klein-006.md) | Jetson Orin Nano,25W | four-step EasyCache | 68241.927 | 23499.517 / 39763.747 / 4587.891 | – (system 5.991) | complete; validated1+3+10, zero steps skipped |
| `jetson-flux-klein-007__baseline__20261005-113106` | [jetson-flux-klein-007](../experiments/jetson-flux-klein-007.md) | Jetson Orin Nano,25W | four-step exact conditioning reuse | 15734.541 | 11.550 / 12186.432 / 3336.363 | – (system 6.011) | complete; validated1+3+10 |
| `jetson-flux-klein-008__baseline__20261005-113721` | [jetson-flux-klein-008](../experiments/jetson-flux-klein-008.md) | Jetson Orin Nano,25W | four-step prefetch disabled | 60503.164 | 21274.854 / 33938.654 / 4488.260 | – (system 5.997) | complete; validated1+3+10 |
| `jetson-flux-klein-009__baseline__20261005-115227` | [jetson-flux-klein-009](../experiments/jetson-flux-klein-009.md) | Jetson Orin Nano,25W | four-step mmap | 67758.838 | 23245.910 / 39497.698 / 4641.656 | – (system 5.953) | complete; validated1+3+10 |
| `jetson-flux-klein-base-007__attempt__20261005-120925` | [jetson-flux-klein-base-007](../experiments/jetson-flux-klein-base-007.md) | Jetson Orin Nano,25W | Base EasyCache + exact conditioning | – | – | – | complete; two-generation correctness gate, full pending |
| `jetson-flux-klein-base-007__baseline__20261005-121955` | [jetson-flux-klein-base-007](../experiments/jetson-flux-klein-base-007.md) | Jetson Orin Nano,25W | Base EasyCache + conditioning reuse | 141012.347 | 13.349 / 137497.409 / 3323.317 | – (system 6.003) | complete; validated1+3+10 |
| `jetson-flux-klein-base-006__attempt__20261005-125535` | [jetson-flux-klein-base-006](../experiments/jetson-flux-klein-base-006.md) | Jetson Orin Nano,25W | Base memory-mapped weight-file I/O | – | – | – | complete; two-generation correctness gate |
| `jetson-flux-klein-base-006__baseline__20261005-131024` | [jetson-flux-klein-base-006](../experiments/jetson-flux-klein-base-006.md) | Jetson Orin Nano,25W | Base mmap | 329425.282 | 24306.888 / 300190.007 / 4624.989 | – (system 5.993) | complete; validated1+3+10 |
| `a100-sdcpp-flux-klein-base-002__attempt__20261005-191624` | [a100-sdcpp-flux-klein-base-002](../experiments/a100-sdcpp-flux-klein-base-002.md) | A100 PCIe | Base EasyCache BF16 50-step gate | 23942.406 | 117.686 / 23187.844 / 607.672 | – (device 22.880) | complete 11883425; attempt, one measured observation |
| `a100-sdcpp-flux-klein-001__attempt__20261005-193229` | [a100-sdcpp-flux-klein-001](../experiments/a100-sdcpp-flux-klein-001.md) | A100 PCIe | Four-step no-cache control | 2411.007 | 64.491 / 1711.380 / 605.955 | – (device 22.880) | complete 11883783; attempt, one measured observation |
| `a100-sdcpp-flux-klein-002__attempt__20261005-193329` | [a100-sdcpp-flux-klein-002](../experiments/a100-sdcpp-flux-klein-002.md) | A100 PCIe | Four-step EasyCache | 2420.973 | 71.173 / 1714.139 / 606.994 | – (device 22.880) | complete 11883784; attempt, one measured observation |
| `a100-sdcpp-flux-klein-003__attempt__20261005-193430` | [a100-sdcpp-flux-klein-003](../experiments/a100-sdcpp-flux-klein-003.md) | A100 PCIe | Four-step exact conditioning cache | 2412.678 | 13.627 / 1716.886 / 653.821 | – (device 22.880) | complete 11883785; attempt, one measured observation |
| `a100-sdcpp-flux-klein-004__attempt__20261005-193530` | [a100-sdcpp-flux-klein-004](../experiments/a100-sdcpp-flux-klein-004.md) | A100 PCIe | Four-step prefetch disabled | 2422.701 | 66.515 / 1719.715 / 602.272 | – (device 22.880) | complete 11883786; attempt, one measured observation |
| `a100-sdcpp-flux-klein-005__attempt__20261005-193630` | [a100-sdcpp-flux-klein-005](../experiments/a100-sdcpp-flux-klein-005.md) | A100 PCIe | Four-step mmap weight-file I/O | 2413.486 | 65.666 / 1715.218 / 603.767 | – (device 22.880) | complete 11883787; attempt, one measured observation |
| `a100-sdcpp-flux-klein-base-001__profile__20261005-191522` | [a100-sdcpp-flux-klein-base-001](../experiments/a100-sdcpp-flux-klein-base-001.md) | A100 PCIe,g011 | Base BF16 + Nsight | – | – | – | failed11883400; repeated Nsight startup timeout |
| `a100-edgedit-flux-klein-001__attempt__20261005-191822` | [a100-edgedit-flux-klein-001](../experiments/a100-edgedit-flux-klein-001.md) | A100 PCIe | ed-sample BF16 weights/FP32 activations | – | – | – | complete11883430; two-repeat interface gate, no full benchmark |
| `a100-sdcpp-flux-klein-001__baseline__20261005-220435` | [a100-sdcpp-flux-klein-001](../experiments/a100-sdcpp-flux-klein-001.md) | A100 PCIe,g007 | a100-sdcpp-flux-klein-001 saved config | 2429.043 | 65.256 / 1725.524 / 605.722 | – (device 22.880) | complete11890006; validated1+3+10 |
| `a100-sdcpp-flux-klein-002__baseline__20261005-220636` | [a100-sdcpp-flux-klein-002](../experiments/a100-sdcpp-flux-klein-002.md) | A100 PCIe,g007 | a100-sdcpp-flux-klein-002 saved config | 2435.751 | 65.117 / 1734.358 / 607.434 | – (device 22.880) | complete11890007; validated1+3+10 |
| `a100-sdcpp-flux-klein-003__baseline__20261005-220737` | [a100-sdcpp-flux-klein-003](../experiments/a100-sdcpp-flux-klein-003.md) | A100 PCIe,g007 | a100-sdcpp-flux-klein-003 saved config | 2389.445 | 13.176 / 1732.962 / 614.955 | – (device 22.880) | complete11890008; validated1+3+10 |
| `a100-sdcpp-flux-klein-004__baseline__20261005-220938` | [a100-sdcpp-flux-klein-004](../experiments/a100-sdcpp-flux-klein-004.md) | A100 PCIe,g007 | a100-sdcpp-flux-klein-004 saved config | 2428.830 | 65.039 / 1726.226 / 607.148 | – (device 22.880) | complete11890009; validated1+3+10 |
| `a100-sdcpp-flux-klein-005__baseline__20261005-221039` | [a100-sdcpp-flux-klein-005](../experiments/a100-sdcpp-flux-klein-005.md) | A100 PCIe,g007 | a100-sdcpp-flux-klein-005 saved config | 2439.542 | 65.082 / 1735.284 / 609.320 | – (device 22.880) | complete11890010; validated1+3+10 |
| `a100-sdcpp-flux-klein-base-001__baseline__20261005-214624` | [a100-sdcpp-flux-klein-base-001](../experiments/a100-sdcpp-flux-klein-base-001.md) | A100 PCIe,g007 | a100-sdcpp-flux-klein-base-001 saved config | 43976.167 | 121.096 / 43209.511 / 611.149 | – (device 22.880) | complete11889993; validated1+3+10 |
| `a100-sdcpp-flux-klein-base-002__baseline__20261005-215731` | [a100-sdcpp-flux-klein-base-002](../experiments/a100-sdcpp-flux-klein-base-002.md) | A100 PCIe,g007 | a100-sdcpp-flux-klein-base-002 saved config | 24192.530 | 118.979 / 23432.650 / 611.300 | – (device 22.880) | complete11889994; validated1+3+10 |
| `a100-edgedit-flux-klein-001__attempt__20261005-221339` | [a100-edgedit-flux-klein-001](../experiments/a100-edgedit-flux-klein-001.md) | A100 PCIe,g007 | edge adapter smoke | 1629 | unavailable / 1091.122 / 267.402 | – (whole-child21.607) | complete11890064; one measured observation |

| `a100-edgedit-flux-klein-001__baseline__20261005-233237` | [a100-edgedit-flux-klein-001](../experiments/a100-edgedit-flux-klein-001.md) | A100 PCIe40GB, g007 | edge-dit no-cache BF16 weights/FP32 activations | 1641.000000 | unavailable / 1096.152425 / 269.868374 | unavailable | complete job11891242; full1+3+10; whole-child device peak21.606873GiB |

| `a100-sdcpp-flux-klein-base-003__attempt__20261005-234246` | [a100-sdcpp-flux-klein-base-003](../experiments/a100-sdcpp-flux-klein-base-003.md) | A100 PCIe40GB, g007 | Base exact conditioning cache capacity2 | 43698.488000 | 14.648000 / 43042.353000 / 613.112000 | unavailable | complete job11891252; two-generation gate, not full benchmark; device peak22.880310GiB |
| `a100-sdcpp-flux-klein-base-003__baseline__20261006-095509` | [a100-sdcpp-flux-klein-base-003](../experiments/a100-sdcpp-flux-klein-base-003.md) | A100 PCIe40GB, g000 | Base exact conditioning cache capacity2 | 43767.162000 | 15.111000 / 43110.188000 / 613.090500 | unavailable | complete job11893094; 1+3+10; different physical GPU from reference; device peak22.880310GiB |

M5 Metal bring-up (functional smoke, not a repeated baseline):

| Run directory | Experiment note | Device | Config | End-to-end (ms) | Text / Denoise / VAE (ms) | Peak alloc (GiB) | Status |
|---|---|---|---|---|---|---|---|
| `m5-sdcpp-flux-klein-001__attempt__20261010-140940` | [m5-sdcpp-flux-klein-001](../experiments/m5-sdcpp-flux-klein-001.md) | MacBook Air M5 | mixed-Q4 512² smoke, sleep guard | 14709.992 | 488.476 / 11876.472 / 2342.870 | unavailable | complete smoke; one measured observation |

| `m5-sdcpp-flux-klein-001__baseline__20261010-152141` | [m5-sdcpp-flux-klein-001](../experiments/m5-sdcpp-flux-klein-001.md) | MacBook Air M5 | m5-sdcpp-flux-klein-q4-512 | 17198.3 | 490.1 / 14038.4 / 2604.5 | — | complete; 1+3+10, background preparation caveat |



| `m5-sdcpp-flux-klein-001__baseline__20261010-165421` | [m5-sdcpp-flux-klein-001](../experiments/m5-sdcpp-flux-klein-001.md) | MacBook Air M5 | m5-sdcpp-flux-klein-q4-512 | 16211.1 | 456.0 / 13401.1 / 2342.2 | — | complete; see validation and environment limits |

| `m5-sdcpp-flux-klein-base-001__baseline__20261010-165813` | [m5-sdcpp-flux-klein-base-001](../experiments/m5-sdcpp-flux-klein-base-001.md) | MacBook Air M5 | m5-sdcpp-flux-klein-base-q4-512 | 407547.2 | 1202.9 / 403543.3 / 2798.8 | — | complete; see validation and environment limits |

| `m5-flux-klein-001__attempt__20261010-183352` | [m5-flux-klein-001](../experiments/m5-flux-klein-001.md) | MacBook Air M5 | m5-flux-klein-bf16-512.resolved | 7597.7 | 600.4 / 6515.8 / 472.4 | — | complete; see validation and environment limits |

| `m5-flux-klein-001__baseline__20261010-183446` | [m5-flux-klein-001](../experiments/m5-flux-klein-001.md) | MacBook Air M5 | m5-flux-klein-bf16-512.resolved | 8277.9 | 944.1 / 6861.6 / 480.8 | — | complete; see validation and environment limits |

| `m5-edgedit-flux-klein-001__attempt__20261010-183705` | [m5-edgedit-flux-klein-001](../experiments/m5-edgedit-flux-klein-001.md) | MacBook Air M5 | m5-edgedit-flux-klein-bf16-512 | — | — | — | failed; see validation and environment limits |

| `m5-flux-klein-base-001__attempt__20261010-183713` | [m5-flux-klein-base-001](../experiments/m5-flux-klein-base-001.md) | MacBook Air M5 | m5-flux-klein-base-bf16-512.resolved | 169887.5 | 1743.2 / 167620.2 / 503.8 | — | complete; see validation and environment limits |

| `m5-flux-klein-base-001__baseline__20261010-184312` | [m5-flux-klein-base-001](../experiments/m5-flux-klein-base-001.md) | MacBook Air M5 | m5-flux-klein-base-bf16-512.resolved | 167100.9 | 2682.0 / 163212.1 / 570.0 | — | complete; see validation and environment limits |

| `m5-edgedit-flux-klein-base-001__attempt__20261010-192312` | [m5-edgedit-flux-klein-base-001](../experiments/m5-edgedit-flux-klein-base-001.md) | MacBook Air M5 | m5-edgedit-flux-klein-base-bf16-512 | — | — | — | failed; see validation and environment limits |

| `m5-flux-klein-002__attempt__20261010-203159` | [m5-flux-klein-002](../experiments/m5-flux-klein-002.md) | MacBook Air M5 | m5-flux-klein-bf16-512-conditioning.resolved | 4359.6 | 0.1 / 4088.4 / 264.4 | — | complete; see validation and environment limits |

| `m5-flux-klein-002__baseline__20261010-203805` | [m5-flux-klein-002](../experiments/m5-flux-klein-002.md) | MacBook Air M5 | m5-flux-klein-bf16-512-conditioning.resolved | 4288.6 | 0.0 / 4015.4 / 266.3 | — | complete; see validation and environment limits |

| `m5-flux-klein-003__attempt__20261010-204022` | [m5-flux-klein-003](../experiments/m5-flux-klein-003.md) | MacBook Air M5 | m5-flux-klein-bf16-512-vae-tiled256.resolved | 4871.0 | 450.7 / 3975.7 / 438.6 | — | complete; see validation and environment limits |

| `m5-flux-klein-003__baseline__20261010-204109` | [m5-flux-klein-003](../experiments/m5-flux-klein-003.md) | MacBook Air M5 | m5-flux-klein-bf16-512-vae-tiled256.resolved | 5029.9 | 461.1 / 4125.2 / 440.6 | — | complete; see validation and environment limits |

| `m5-sdcpp-flux-klein-002__attempt__20261010-204248` | [m5-sdcpp-flux-klein-002](../experiments/m5-sdcpp-flux-klein-002.md) | MacBook Air M5 | m5-sdcpp-flux-klein-q4-512-easycache.resolved | 15127.5 | 435.6 / 12510.9 / 2179.0 | — | complete; see validation and environment limits |

| `m5-sdcpp-flux-klein-002__baseline__20261010-204326` | [m5-sdcpp-flux-klein-002](../experiments/m5-sdcpp-flux-klein-002.md) | MacBook Air M5 | m5-sdcpp-flux-klein-q4-512-easycache.resolved | 15807.4 | 441.7 / 13098.4 / 2260.8 | — | complete; see validation and environment limits |

| `m5-sdcpp-flux-klein-003__attempt__20261010-204709` | [m5-sdcpp-flux-klein-003](../experiments/m5-sdcpp-flux-klein-003.md) | MacBook Air M5 | m5-sdcpp-flux-klein-q4-512-conditioning.resolved | 14839.0 | 2.8 / 12664.3 / 2170.1 | — | complete; see validation and environment limits |

| `m5-sdcpp-flux-klein-003__baseline__20261010-205038` | [m5-sdcpp-flux-klein-003](../experiments/m5-sdcpp-flux-klein-003.md) | MacBook Air M5 | m5-sdcpp-flux-klein-q4-512-conditioning.resolved | 15298.6 | 2.4 / 13048.4 / 2230.1 | — | complete; see validation and environment limits |

| `m5-sdcpp-flux-klein-004__attempt__20261010-205422` | [m5-sdcpp-flux-klein-004](../experiments/m5-sdcpp-flux-klein-004.md) | MacBook Air M5 | m5-sdcpp-flux-klein-q4-512-prefetch-off.resolved | 16729.6 | 470.7 / 13910.1 / 2346.8 | — | complete; see validation and environment limits |

| `m5-sdcpp-flux-klein-004__baseline__20261010-205458` | [m5-sdcpp-flux-klein-004](../experiments/m5-sdcpp-flux-klein-004.md) | MacBook Air M5 | m5-sdcpp-flux-klein-q4-512-prefetch-off.resolved | 16814.6 | 475.4 / 13945.1 / 2377.2 | — | complete; see validation and environment limits |

| `m5-sdcpp-flux-klein-005__attempt__20261010-205853` | [m5-sdcpp-flux-klein-005](../experiments/m5-sdcpp-flux-klein-005.md) | MacBook Air M5 | m5-sdcpp-flux-klein-q4-512-mmap.resolved | 17446.0 | 455.1 / 14568.2 / 2420.6 | — | complete; see validation and environment limits |

| `m5-sdcpp-flux-klein-005__baseline__20261010-205929` | [m5-sdcpp-flux-klein-005](../experiments/m5-sdcpp-flux-klein-005.md) | MacBook Air M5 | m5-sdcpp-flux-klein-q4-512-mmap.resolved | 17624.1 | 505.6 / 14592.5 / 2465.3 | — | complete; see validation and environment limits |

| `m5-flux-klein-base-002__attempt__20261010-210338` | [m5-flux-klein-base-002](../experiments/m5-flux-klein-base-002.md) | MacBook Air M5 | m5-flux-klein-base-bf16-512-conditioning.resolved | 131725.1 | 0.1 / 131308.6 / 394.1 | — | complete; see validation and environment limits |

| `m5-flux-klein-base-002__baseline__20261010-210907` | [m5-flux-klein-base-002](../experiments/m5-flux-klein-base-002.md) | MacBook Air M5 | m5-flux-klein-base-bf16-512-conditioning.resolved | 136075.5 | 0.1 / 135653.7 / 407.8 | — | complete; see validation and environment limits |

| `m5-flux-klein-base-003__attempt__20261010-214141` | [m5-flux-klein-base-003](../experiments/m5-flux-klein-base-003.md) | MacBook Air M5 | m5-flux-klein-base-bf16-512-vae-tiled256.resolved | 164186.9 | 2786.1 / 160565.6 / 794.4 | — | complete; see validation and environment limits |

| `m5-flux-klein-base-003__baseline__20261010-214732` | [m5-flux-klein-base-003](../experiments/m5-flux-klein-base-003.md) | MacBook Air M5 | m5-flux-klein-base-bf16-512-vae-tiled256.resolved | 146049.4 | 1513.6 / 143696.0 / 623.0 | — | complete; see validation and environment limits |

| `m5-sdcpp-flux-klein-base-002__attempt__20261010-222736` | [m5-sdcpp-flux-klein-base-002](../experiments/m5-sdcpp-flux-klein-base-002.md) | MacBook Air M5 | m5-sdcpp-flux-klein-base-q4-512-easycache.resolved | 197683.9 | 1027.7 / 194067.2 / 2586.8 | — | complete; see validation and environment limits |

| `m5-sdcpp-flux-klein-base-002__baseline__20261010-223355` | [m5-sdcpp-flux-klein-base-002](../experiments/m5-sdcpp-flux-klein-base-002.md) | MacBook Air M5 | m5-sdcpp-flux-klein-base-q4-512-easycache.resolved | 200305.2 | 1155.7 / 196393.7 / 2735.2 | — | complete; see validation and environment limits |

| `m5-sdcpp-flux-klein-base-003__attempt__20261010-232130` | [m5-sdcpp-flux-klein-base-003](../experiments/m5-sdcpp-flux-klein-base-003.md) | MacBook Air M5 | m5-sdcpp-flux-klein-base-q4-512-conditioning.resolved | 365645.4 | 6.1 / 363054.2 / 2583.2 | — | complete; see validation and environment limits |

| `m5-sdcpp-flux-klein-base-003__baseline__20261010-233405` | [m5-sdcpp-flux-klein-base-003](../experiments/m5-sdcpp-flux-klein-base-003.md) | MacBook Air M5 | m5-sdcpp-flux-klein-base-q4-512-conditioning.resolved | 400140.9 | 7.4 / 397391.1 / 2760.5 | — | complete; see validation and environment limits |

| `m5-sdcpp-flux-klein-base-004__attempt__20261011-010657` | [m5-sdcpp-flux-klein-base-004](../experiments/m5-sdcpp-flux-klein-base-004.md) | MacBook Air M5 | m5-sdcpp-flux-klein-base-q4-512-prefetch-off.resolved | 386289.9 | 1070.1 / 382551.1 / 2666.6 | — | complete; see validation and environment limits |

| `m5-sdcpp-flux-klein-base-004__baseline__20261011-012000` | [m5-sdcpp-flux-klein-base-004](../experiments/m5-sdcpp-flux-klein-base-004.md) | MacBook Air M5 | m5-sdcpp-flux-klein-base-q4-512-prefetch-off.resolved | 386819.5 | 1112.4 / 383085.2 / 2707.1 | — | complete; see validation and environment limits |

| `m5-sdcpp-flux-klein-base-005__attempt__20261011-025051` | [m5-sdcpp-flux-klein-base-005](../experiments/m5-sdcpp-flux-klein-base-005.md) | MacBook Air M5 | m5-sdcpp-flux-klein-base-q4-512-mmap.resolved | 392819.0 | 1072.1 / 389045.8 / 2699.1 | — | complete; see validation and environment limits |

| `m5-sdcpp-flux-klein-base-005__baseline__20261011-030358` | [m5-sdcpp-flux-klein-base-005](../experiments/m5-sdcpp-flux-klein-base-005.md) | MacBook Air M5 | m5-sdcpp-flux-klein-base-q4-512-mmap.resolved | 389532.8 | 1099.4 / 385716.1 / 2688.9 | — | complete; see validation and environment limits |

| `m5-edgedit-flux-klein-001__attempt__20261011-043503` | [m5-edgedit-flux-klein-001](../experiments/m5-edgedit-flux-klein-001.md) | MacBook Air M5 | m5-edgedit-flux-klein-bf16-512-native | — | — | — | failed; see validation and environment limits |

## MFLUX on M5

| Run directory | Experiment note | Device | Config | End-to-end (ms) | Text / Denoise / VAE (ms) | Peak alloc (GiB) | Status |
|---|---|---|---|---|---|---|---|
| `m5-mflux-flux-klein-001__attempt__20261012-153912` | [m5-mflux-flux-klein-001](../experiments/m5-mflux-flux-klein-001.md) | macbook-air-m5 | m5-mflux-flux-klein-001 | – | – / – / – | – | failed; smoke attempt; synchronized host stages |
| `m5-mflux-flux-klein-001__attempt__20261012-154009` | [m5-mflux-flux-klein-001](../experiments/m5-mflux-flux-klein-001.md) | macbook-air-m5 | m5-mflux-flux-klein-001 | 66054.5 | 31942.3 / 33358.9 / 747.4 | 17.247348265722394 | complete; smoke attempt; synchronized host stages |
| `m5-mflux-flux-klein-001__baseline__20261012-154228` | [m5-mflux-flux-klein-001](../experiments/m5-mflux-flux-klein-001.md) | macbook-air-m5 | m5-mflux-flux-klein-001 | 80376.7 | 32541.9 / 48035.0 / 1275.5 | 17.24734838306904 | complete; full protocol; synchronized host stages |
| `m5-mflux-flux-klein-002__attempt__20261012-155918` | [m5-mflux-flux-klein-002](../experiments/m5-mflux-flux-klein-002.md) | macbook-air-m5 | m5-mflux-flux-klein-002 | 21434.1 | 5369.5 / 9495.5 / 6559.6 | 10.351489501073956 | complete; smoke attempt; synchronized host stages |
| `m5-mflux-flux-klein-002__baseline__20261012-160000` | [m5-mflux-flux-klein-002](../experiments/m5-mflux-flux-klein-002.md) | macbook-air-m5 | m5-mflux-flux-klein-002 | 71173.4 | 17209.2 / 26830.7 / 26509.8 | 10.351489618420601 | complete; full protocol; synchronized host stages |
| `m5-mflux-flux-klein-003__attempt__20261012-161526` | [m5-mflux-flux-klein-003](../experiments/m5-mflux-flux-klein-003.md) | macbook-air-m5 | m5-mflux-flux-klein-003 | 7461.9 | 409.7 / 6230.4 / 818.5 | 6.6728090811520815 | complete; smoke attempt; synchronized host stages |
| `m5-mflux-flux-klein-003__baseline__20261012-161546` | [m5-mflux-flux-klein-003](../experiments/m5-mflux-flux-klein-003.md) | macbook-air-m5 | m5-mflux-flux-klein-003 | 10296.2 | 650.1 / 8316.2 / 1033.0 | 6.6728091817349195 | complete; full protocol; synchronized host stages |
| `m5-mflux-flux-klein-base-001__attempt__20261012-161808` | [m5-mflux-flux-klein-base-001](../experiments/m5-mflux-flux-klein-base-001.md) | macbook-air-m5 | m5-mflux-flux-klein-base-001 | – | – / – / – | – | failed; smoke attempt; synchronized host stages |
| `m5-mflux-flux-klein-base-002__attempt__20261012-161817` | [m5-mflux-flux-klein-base-002](../experiments/m5-mflux-flux-klein-base-002.md) | macbook-air-m5 | m5-mflux-flux-klein-base-002 | 212971.5 | 17732.3 / 177542.6 / 17676.3 | 10.351490186527371 | complete; smoke attempt; synchronized host stages |
| `m5-mflux-flux-klein-base-002__baseline__20261012-162447` | [m5-mflux-flux-klein-base-002](../experiments/m5-mflux-flux-klein-base-002.md) | macbook-air-m5 | m5-mflux-flux-klein-base-002 | 206174.7 | 13407.8 / 168837.0 / 23976.5 | 10.351491067558527 | complete; full protocol; synchronized host stages |
| `m5-mflux-flux-klein-base-003__attempt__20261012-171421` | [m5-mflux-flux-klein-base-003](../experiments/m5-mflux-flux-klein-base-003.md) | macbook-air-m5 | m5-mflux-flux-klein-base-003 | 155085.6 | 645.3 / 152803.6 / 1627.3 | 6.672809766605496 | complete; smoke attempt; synchronized host stages |
| `m5-mflux-flux-klein-base-003__baseline__20261012-171934` | [m5-mflux-flux-klein-base-003](../experiments/m5-mflux-flux-klein-base-003.md) | macbook-air-m5 | m5-mflux-flux-klein-base-003 | 151007.4 | 671.3 / 146880.3 / 3169.3 | 6.672810457646847 | complete; full protocol; synchronized host stages |

| `m5-flux-klein-004__attempt__20261015-131752` | [m5-flux-klein-004](../experiments/m5-flux-klein-004.md) | MacBook Air M5 | m5-flux-klein-bf16-512-compile.resolved | — | — | — | complete; see validation and environment limits |

| `m5-flux-klein-004__attempt__20261015-132823` | [m5-flux-klein-004](../experiments/m5-flux-klein-004.md) | MacBook Air M5 | m5-flux-klein-bf16-512-compile.resolved | — | — | — | failed; see validation and environment limits |

| `m5-flux-klein-005__attempt__20261015-132946` | [m5-flux-klein-005](../experiments/m5-flux-klein-005.md) | MacBook Air M5 | m5-flux-klein-bf16-512-taylorseer.resolved | 64455.1 | 34626.9 / 28823.3 / 965.6 | — | Complete; validated protocol, uncontrolled memory pressure; no quality approval |

| `m5-flux-klein-005__baseline__20261015-133226` | [m5-flux-klein-005](../experiments/m5-flux-klein-005.md) | MacBook Air M5 | m5-flux-klein-bf16-512-taylorseer.resolved | 57556.4 | 27491.7 / 17817.6 / 4900.0 | — | Complete; validated protocol, uncontrolled memory pressure; no quality approval |

| `m5-mflux-flux-klein-004__attempt__20261015-134618` | [m5-mflux-flux-klein-004](../experiments/m5-mflux-flux-klein-004.md) | MacBook Air M5 | m5-mflux-flux-klein-q4-512-conditioning.resolved | 5021.5 | 0.0 / 4188.2 / 828.4 | — | Complete; validated protocol, uncontrolled memory pressure; no quality approval |

| `m5-mflux-flux-klein-004__baseline__20261015-134636` | [m5-mflux-flux-klein-004](../experiments/m5-mflux-flux-klein-004.md) | MacBook Air M5 | m5-mflux-flux-klein-q4-512-conditioning.resolved | 5691.3 | 0.0 / 4803.0 / 884.1 | — | Complete; validated protocol, uncontrolled memory pressure; no quality approval |

| `m5-mflux-flux-klein-005__attempt__20261015-134805` | [m5-mflux-flux-klein-005](../experiments/m5-mflux-flux-klein-005.md) | MacBook Air M5 | m5-mflux-flux-klein-q4-512-vae-tiled256.resolved | 8622.5 | 533.9 / 6898.2 / 1187.2 | — | Complete; validated protocol, uncontrolled memory pressure; no quality approval |

| `m5-mflux-flux-klein-005__baseline__20261015-134829` | [m5-mflux-flux-klein-005](../experiments/m5-mflux-flux-klein-005.md) | MacBook Air M5 | m5-mflux-flux-klein-q4-512-vae-tiled256.resolved | 7661.5 | 416.3 / 6279.2 / 971.3 | — | Complete; validated protocol, uncontrolled memory pressure; no quality approval |

| `m5-mflux-flux-klein-006__attempt__20261015-135027` | [m5-mflux-flux-klein-006](../experiments/m5-mflux-flux-klein-006.md) | MacBook Air M5 | m5-mflux-flux-klein-q8-512-vae-tiled256.resolved | 8547.7 | 1000.3 / 6545.2 / 999.2 | — | Complete; validated protocol, uncontrolled memory pressure; no quality approval |

| `m5-mflux-flux-klein-006__baseline__20261015-135054` | [m5-mflux-flux-klein-006](../experiments/m5-mflux-flux-klein-006.md) | MacBook Air M5 | m5-mflux-flux-klein-q8-512-vae-tiled256.resolved | 8606.5 | 519.8 / 6835.6 / 1152.2 | — | Complete; validated protocol, uncontrolled memory pressure; no quality approval |

| `m5-flux-klein-base-004__attempt__20261015-135305` | [m5-flux-klein-base-004](../experiments/m5-flux-klein-base-004.md) | MacBook Air M5 | m5-flux-klein-base-bf16-512-compile.resolved | 154718.1 | 22153.7 / 131383.6 / 1094.1 | — | Complete; validated protocol, uncontrolled swap; no formal quality approval |

| `m5-flux-klein-base-004__baseline__20261015-135837` | [m5-flux-klein-base-004](../experiments/m5-flux-klein-base-004.md) | MacBook Air M5 | m5-flux-klein-base-bf16-512-compile.resolved | 143876.9 | 14959.6 / 127930.9 / 781.1 | — | Complete; validated protocol, uncontrolled swap; no formal quality approval |

| `m5-flux-klein-base-005__attempt__20261015-143525` | [m5-flux-klein-base-005](../experiments/m5-flux-klein-base-005.md) | MacBook Air M5 | m5-flux-klein-base-bf16-512-taylorseer.resolved | 82751.9 | 14231.3 / 65449.7 / 3036.6 | — | Complete; validated protocol, uncontrolled swap; no formal quality approval |

| `m5-flux-klein-base-005__baseline__20261015-143814` | [m5-flux-klein-base-005](../experiments/m5-flux-klein-base-005.md) | MacBook Air M5 | m5-flux-klein-base-bf16-512-taylorseer.resolved | 92473.8 | 21319.0 / 69631.0 / 524.2 | — | Complete; validated protocol, uncontrolled swap; no formal quality approval |

| `m5-mflux-flux-klein-base-004__attempt__20261015-150106` | [m5-mflux-flux-klein-base-004](../experiments/m5-mflux-flux-klein-base-004.md) | MacBook Air M5 | m5-mflux-flux-klein-base-q4-512-conditioning.resolved | — | — | — | Running smoke; no validated full result |

| `m5-mflux-flux-klein-base-004__baseline__20261015-150452` | [m5-mflux-flux-klein-base-004](../experiments/m5-mflux-flux-klein-base-004.md) | MacBook Air M5 | m5-mflux-flux-klein-base-q4-512-conditioning.resolved | — | — | — | Smoke gate passed; full protocol running |

| `m5-mflux-flux-klein-base-005__attempt__20261015-153625` | [m5-mflux-flux-klein-base-005](../experiments/m5-mflux-flux-klein-base-005.md) | MacBook Air M5 | m5-mflux-flux-klein-base-q4-512-vae-tiled256.resolved | 154681.6 | 588.3 / 152686.1 / 1400.9 | — | complete; see validation and environment limits |

| `m5-mflux-flux-klein-base-005__baseline__20261015-154117` | [m5-mflux-flux-klein-base-005](../experiments/m5-mflux-flux-klein-base-005.md) | MacBook Air M5 | m5-mflux-flux-klein-base-q4-512-vae-tiled256.resolved | — | — | — | Invalid full: clamshell sleep; partial evidence retained; recovery pending |

| `m5-mflux-flux-klein-base-006__attempt__20261015-163359` | [m5-mflux-flux-klein-base-006](../experiments/m5-mflux-flux-klein-base-006.md) | MacBook Air M5 | m5-mflux-flux-klein-base-q8-512-vae-tiled256.resolved | — | — | — | Failed preflight: AC power required; no inference |
## Jetson PyTorch/diffusers (portable runner)

Distilled klein 512², GGUF Q4_0 transformer + NF4 encoder; `scripts/run_flux_portable.py`. Two-generation smokes are functional checks, not baselines.

| Run directory | Experiment note | Device | Config | End-to-end (ms) | Text / Denoise / VAE (ms) | Peak alloc (GiB) | Status |
|---|---|---|---|---|---|---|---|
| `jetson-torch-env-t0__attempt__20261011` | [jetson-torch-flux-klein-001](../experiments/jetson-torch-flux-klein-001.md) | Jetson Orin Nano, 25W | environment gate T0 (JetPack 6.1 torch 2.5.0a0 wheel) | – | – | – | failed: wheel lacks torch.distributed; diffusers Flux2/GGUF/torchao imports fail; bitsandbytes NF4 CUDA error |
| `jetson-torch-env-t0__attempt__20261011-ailab` | [jetson-torch-flux-klein-001](../experiments/jetson-torch-flux-klein-001.md) | Jetson Orin Nano, 25W | environment gate T0 (Jetson AI Lab stack, torch 2.8.0) | – | – | – | complete; all environment checks pass |
| `jetson-torch-flux-klein-001__attempt__20261012-122056` | [jetson-torch-flux-klein-001](../experiments/jetson-torch-flux-klein-001.md) | Jetson Orin Nano, 25W | jetson-torch-flux-klein-q4-nf4-512-smoke (all resident, earlier runner) | – | – | – | failed: memory guard during load (runner bug: transformer CPU+GPU copies coexisted); status.json stale |
| `jetson-torch-flux-klein-001__attempt__20261012-123054` | [jetson-torch-flux-klein-001](../experiments/jetson-torch-flux-klein-001.md) | Jetson Orin Nano, 25W | jetson-torch-flux-klein-q4-nf4-512-smoke (all resident) | – | – | – | failed: memory guard during first text encode (MemAvailable 398 MB); status.json stale |
| `jetson-torch-flux-klein-001__attempt__20261012-143714` | [jetson-torch-flux-klein-001](../experiments/jetson-torch-flux-klein-001.md) | Jetson Orin Nano, 25W | jetson-torch-flux-klein-q4-nf4-512-smoke (encoder released, transformer resident) | – | – | – | failed (inferred): ended during first generation at ~7.0 GB system RAM; no guard record; status.json stale |
| `jetson-torch-flux-klein-001__attempt__20261012-152208` | [jetson-torch-flux-klein-001](../experiments/jetson-torch-flux-klein-001.md) | Jetson Orin Nano, 25W | jetson-torch-flux-klein-q4-nf4-512-smoke (sequential residency) | – | – | – | failed: GGUF linear dtype mismatch (uint8 vs BF16) in first transformer call |
| `jetson-torch-flux-klein-001__attempt__20261012-153146` | [jetson-torch-flux-klein-001](../experiments/jetson-torch-flux-klein-001.md) | Jetson Orin Nano, 25W | jetson-torch-flux-klein-q4-nf4-512-smoke (sequential residency) | – | – | – | failed: norm_q weight shape [256] vs [128] (BF16 norms left as raw bytes) |
| `jetson-torch-flux-klein-001__attempt__20261012-153448` | [jetson-torch-flux-klein-001](../experiments/jetson-torch-flux-klein-001.md) | Jetson Orin Nano, 25W | jetson-torch-flux-klein-q4-nf4-512-smoke | – | – | – | complete; first + one observation, not a baseline |
| `jetson-torch-flux-klein-001__baseline__20261012-153954` | [jetson-torch-flux-klein-001](../experiments/jetson-torch-flux-klein-001.md) | Jetson Orin Nano, 25W | jetson-torch-flux-klein-q4-nf4-512 | 79293.2 | 1279.2 / 14758.0 / 492.7 | 3.08 | failed: memory guard (swap growth 774 MB > 768 MB) at generation 12; partial medians, n = 8 of 10 measured; status.json stale |
| `jetson-torch-t1-memory-diag-20261012` | [jetson-torch-flux-klein-001](../experiments/jetson-torch-flux-klein-001.md) | Jetson Orin Nano, 25W | memory diagnostics (ad hoc scripts) | – | – | – | complete; diagnostics only, not a benchmark |
| `jetson-torch-flux-klein-001__baseline__20261014-161555` | [jetson-torch-flux-klein-001](../experiments/jetson-torch-flux-klein-001.md) | Jetson Orin Nano, 25W | jetson-torch-flux-klein-q4-nf4-512 | – | – | – | failed: refused before load (HF_HUB_OFFLINE not set); no measurement |
| `jetson-torch-flux-klein-001__baseline__20261014-161622` | [jetson-torch-flux-klein-001](../experiments/jetson-torch-flux-klein-001.md) | Jetson Orin Nano, 25W | jetson-torch-flux-klein-q4-nf4-512 | – | – | – | failed: HF cache not found at load (HF_HOME not set); no measurement |
| `jetson-torch-flux-klein-001__baseline__20261014-161717` | [jetson-torch-flux-klein-001](../experiments/jetson-torch-flux-klein-001.md) | Jetson Orin Nano, 25W | jetson-torch-flux-klein-q4-nf4-512 | – | – | – | failed: stopped by operator after first generation (concurrent ~5 GB sha256sum on the same microSD); contaminated, not reported; status.json stale |
| `jetson-torch-flux-klein-001__baseline__20261014-171410` | [jetson-torch-flux-klein-001](../experiments/jetson-torch-flux-klein-001.md) | Jetson Orin Nano, 25W | jetson-torch-flux-klein-q4-nf4-512 | 80314.9 | 1241.3 / 14795.5 / 491.4 | 3.08 | complete; 10 measured; other_ms 63343.9 (per-generation encoder load + transformer reload); guard swap growth 957 MB (limit 1536), min MemAvailable 1206 MB |
| `jetson-torch-flux-klein-002__attempt__20261014-173551` | [jetson-torch-flux-klein-002](../experiments/jetson-torch-flux-klein-002.md) | Jetson Orin Nano, 25W | jetson-torch-flux-klein-q4-nf4-512-sdpa-math-smoke | – | – | 3.15 | complete; smoke, first + 1 measured, not a baseline; 137 SDPA calls per generation |
| `jetson-torch-flux-klein-002__baseline__20261014-174108` | [jetson-torch-flux-klein-002](../experiments/jetson-torch-flux-klein-002.md) | Jetson Orin Nano, 25W | jetson-torch-flux-klein-q4-nf4-512-sdpa-math | 84221.5 | 1720.4 / 19297.6 / 511.2 | 3.15 | complete; 10 measured; steady per-step 4694.6 ms (+30.7% vs 001); 137 SDPA calls per generation; output hash differs from 001 |
| `jetson-torch-flux-klein-003__attempt__20261014-180328` | [jetson-torch-flux-klein-003](../experiments/jetson-torch-flux-klein-003.md) | Jetson Orin Nano, 25W | jetson-torch-flux-klein-q4-nf4-512-vaetile-smoke | – | – | 3.07 | complete; smoke, first + 1 measured, not a baseline; tiled decode 2/2 |
| `jetson-torch-flux-klein-003__baseline__20261014-180843` | [jetson-torch-flux-klein-003](../experiments/jetson-torch-flux-klein-003.md) | Jetson Orin Nano, 25W | jetson-torch-flux-klein-q4-nf4-512-vaetile | 82991.0 | 1450.2 / 14748.4 / 1850.4 | 3.07 | complete; 10 measured; tiled decode 14/14; VAE-decode stage peak 2.527 GiB (001: 3.081), peak now set by text_encode; output hash differs from 001 |
| `jetson-torch-flux-klein-004__attempt__20261015-140608` | [jetson-torch-flux-klein-004](../experiments/jetson-torch-flux-klein-004.md) | Jetson Orin Nano, 25W | jetson-torch-flux-klein-q4-nf4-512-condreuse-smoke | – | – | 3.08 | complete; smoke, first + 1 measured, not a baseline; cache hits 1 / misses 1 |
| `jetson-torch-flux-klein-004__baseline__20261015-141028` | [jetson-torch-flux-klein-004](../experiments/jetson-torch-flux-klein-004.md) | Jetson Orin Nano, 25W | jetson-torch-flux-klein-q4-nf4-512-condreuse | 14911.4 | 0.9 (retrieval) / 14377.3 / 485.7 | 3.08 | complete; 10 measured; steady per-step 3593.1 ms; other_ms 30.6 (001: 63343.9); first generation 96501.7 ms incl. the only residency swap (70.29 s); cache hits 13 / misses 1; output hash identical to 001 |
| `jetson-torch-flux-klein-005__attempt__20261015-141745` | [jetson-torch-flux-klein-005](../experiments/jetson-torch-flux-klein-005.md) | Jetson Orin Nano, 25W | jetson-torch-flux-klein-q4-nf4-512-fbcache-smoke | – | – | – | failed: `ValueError` "Model class Flux2TransformerBlock not registered" in diffusers 0.40.0 `first_block_cache` while enabling the cache after load; no generation; full protocol not run |
| `jetson-torch-flux-klein-006__attempt__20261015-142002` | [jetson-torch-flux-klein-006](../experiments/jetson-torch-flux-klein-006.md) | Jetson Orin Nano, 25W | jetson-torch-flux-klein-q4-nf4-512-taylorseer-smoke | – | – | 3.52 | complete; smoke, first + 1 measured, not a baseline |
| `jetson-torch-flux-klein-006__baseline__20261015-142456` | [jetson-torch-flux-klein-006](../experiments/jetson-torch-flux-klein-006.md) | Jetson Orin Nano, 25W | jetson-torch-flux-klein-q4-nf4-512-taylorseer | 77644.2 | 1414.9 / 9155.6 / 489.1 | 3.52 | complete; 10 measured; steps 0–3 4060.1 / 810.4 / 3603.6 / 684.4 ms (steps 1, 3 predicted); output hash differs from 001, sign text visibly degraded (visual check) |
| `jetson-torch-flux-klein-007__attempt__20261015-144541` | [jetson-torch-flux-klein-007](../experiments/jetson-torch-flux-klein-007.md) | Jetson Orin Nano, 25W | jetson-torch-flux-klein-q4-nf4-512-condreuse-compile-smoke | – | – | 3.08 | complete; smoke, first + 1 measured, not a baseline; first generation 256190.2 ms incl. compilation; output hash 13f6174f… (differs from the full run) |
| `jetson-torch-flux-klein-007__baseline__20261015-145234` | [jetson-torch-flux-klein-007](../experiments/jetson-torch-flux-klein-007.md) | Jetson Orin Nano, 25W | jetson-torch-flux-klein-q4-nf4-512-condreuse-compile | 7783.6 | 0.8 (retrieval) / 7249.5 / 485.3 | 3.08 | complete; 10 measured; steady per-step 1811.7 ms (004: 3593.1); first generation 202655.5 ms (step 0 118520.2 ms incl. compilation); compilations_applied 2; reserved 3.72 GiB; output hash differs from 004 ("HOLLO WORD", visual check) |

## Jetson edge-dit.cpp

Distilled klein 512²; `scripts/run_edgedit_jetson.py` (006 onward). Memory is whole-system tegrastats; stage times from the engine breakdown line (10 ms rounding).

| Run directory | Experiment note | Device | Config | End-to-end (ms) | Text / Denoise / VAE (ms) | Peak alloc (GiB) | Status |
|---|---|---|---|---|---|---|---|
| `jetson-edgedit-build-20261011` | [jetson-edgedit-flux-klein-001](../experiments/jetson-edgedit-flux-klein-001.md) | Jetson Orin Nano | edge-dit.cpp 97cc585 CUDA performance build, sm_87 (jetson_build_edgedit.sh) | – | – | – | complete; build evidence only (BUILD_OK) |
| `jetson-edgedit-flux-klein-001__attempt__20261011-062302` | [jetson-edgedit-flux-klein-001](../experiments/jetson-edgedit-flux-klein-001.md) | Jetson Orin Nano, 25W | edge-dit all-GPU explicit placement, Q4_0 / Q4_K_M / BF16 VAE, 512² (run.sh; no config JSON) | – | – | – (tegrastats RAM 7424/7620 MB) | failed: kernel OOM kill during weight loading, exit 137 |
| `jetson-edgedit-flux-klein-002__attempt__20261011-autofit` | [jetson-edgedit-flux-klein-002](../experiments/jetson-edgedit-flux-klein-002.md) | Jetson Orin Nano, 25W | edge-dit `--auto-fit`, Q4_0 / Q4_K_M / BF16 VAE, 512² (no config JSON) | – | – | – (tegrastats RAM 7478/7620 MB) | failed: board stopped responding during text encoding and rebooted; no exit code |
| `jetson-edgedit-flux-klein-003__attempt__20261011-autoalloc` | [jetson-edgedit-flux-klein-003](../experiments/jetson-edgedit-flux-klein-003.md) | Jetson Orin Nano, 25W | edge-dit `--auto-allocate`, Q4_0 / Q4_K_M / BF16 VAE, 512² (no config JSON) | – | – | – (tegrastats RAM 6975/7620 MB) | failed: memory guard (MemAvailable 399 MB < 400 MB) during weight loading, exit 137 |
| `jetson-edgedit-convert-20261011` | [jetson-edgedit-flux-klein-004](../experiments/jetson-edgedit-flux-klein-004.md) | Jetson Orin Nano | requant_gguf_tensor.py: Qwen3-4B Q4_K_M, token_embd Q6_K → Q8_0 | – | – | – | complete; conversion evidence only (rel. RMS diff 0.00529 on token_embd); a conversion attempt with weight type `preserve` failed (command not recorded) |
| `jetson-edgedit-flux-klein-004__attempt__20261011-embq8` | [jetson-edgedit-flux-klein-004](../experiments/jetson-edgedit-flux-klein-004.md) | Jetson Orin Nano, 25W | edge-dit all-GPU, Q4_0 / **Q4_K_M + Q8_0 token embedding** / BF16 VAE, 512² (no config JSON) | – | – | – (tegrastats RAM 6837/7620 MB) | failed: memory guard (MemAvailable 345 MB) at start of VAE decode, exit 137; encode and 4 denoise steps completed |
| `jetson-edgedit-flux-klein-005__attempt__20261011-embq8-vaetile` | [jetson-edgedit-flux-klein-005](../experiments/jetson-edgedit-flux-klein-005.md) | Jetson Orin Nano, 25W | edge-dit all-GPU, Q4_0 / **Q4_K_M + Q8_0 token embedding** / BF16 VAE, `--vae-tiling on`, 512² (no config JSON) | – | – | – (tegrastats RAM 6701/7620 MB) | partial: functional admission, one generation, exit 0; host-clock markers encode_setup 2564 / denoise 10630 / tiled decode 4378 ms (single first generation, not medians); image coherent, sign reads "helllo world" |
| `jetson-edgedit-flux-klein-006__attempt__20261014-165850` | [jetson-edgedit-flux-klein-006](../experiments/jetson-edgedit-flux-klein-006.md) | Jetson Orin Nano, 25W | jetson-edgedit-flux-klein-q4-embq8-512-vaetile-smoke | – | – | – (tegrastats RAM 6757/7620 MB) | complete; first + one observation, not a baseline |
| `jetson-edgedit-flux-klein-006__baseline__20261014-170149` | [jetson-edgedit-flux-klein-006](../experiments/jetson-edgedit-flux-klein-006.md) | Jetson Orin Nano, 25W | jetson-edgedit-flux-klein-q4-embq8-512-vaetile | 11962.5 | 930.0 / 9185.0 / 1740.0 | – (tegrastats RAM 6881/7620 MB) | complete; 1+3+10, one load (64.7 s); first 17395.0 ms; engine-breakdown stages (10 ms rounding) |
| `jetson-edgedit-flux-klein-007__attempt__20261014-170729` | [jetson-edgedit-flux-klein-007](../experiments/jetson-edgedit-flux-klein-007.md) | Jetson Orin Nano, 25W | jetson-edgedit-flux-klein-q4-embq8-512-vaetile-noflash-smoke | – | – | – (tegrastats RAM 6852/7620 MB) | complete; first + one observation (measured 15473.0 ms), not a baseline; MemAvailable min 615 MB |
| `jetson-edgedit-flux-klein-007__baseline__20261014-171051` | [jetson-edgedit-flux-klein-007](../experiments/jetson-edgedit-flux-klein-007.md) | Jetson Orin Nano, 25W | jetson-edgedit-flux-klein-q4-embq8-512-vaetile-noflash | – | – | – (tegrastats RAM 6996/7620 MB) | failed: memory guard (MemAvailable 399 MB < 400 MB) at denoise start of generation index 2, exit 137; 2 passes completed (first 20393 ms, warm-up 15460 ms); status.json stale (`running`) |
| `jetson-edgedit-flux-klein-008__attempt__20261015-132605` | [jetson-edgedit-flux-klein-008](../experiments/jetson-edgedit-flux-klein-008.md) | Jetson Orin Nano, 25W | jetson-edgedit-flux-klein-q4-embq8-512-vaetile-easycache-smoke | – | – | – (tegrastats RAM 6.598 GiB) | complete; screen, first + one observation (12042.0 ms), not a baseline; EasyCache skipped 0/4 steps in both passes; output hash = 006 |
| `jetson-edgedit-flux-klein-009__attempt__20261015-132923` | [jetson-edgedit-flux-klein-009](../experiments/jetson-edgedit-flux-klein-009.md) | Jetson Orin Nano, 25W | jetson-edgedit-flux-klein-q4-embq8-512-vaetile-ucache-smoke | – | – | – (tegrastats RAM 6.588 GiB) | complete; screen, first + one observation (11998.0 ms), not a baseline; UCache skipped 0/4 steps in both passes; output hash = 006 |
| `jetson-edgedit-flux-klein-010__attempt__20261015-133240` | [jetson-edgedit-flux-klein-010](../experiments/jetson-edgedit-flux-klein-010.md) | Jetson Orin Nano, 25W | jetson-edgedit-flux-klein-q4-embq8-512-vaetile-dbcache-smoke | – | – | – (tegrastats RAM 6.583 GiB) | complete; screen, first + one observation (12057.0 ms), not a baseline; DBCache skipped 0/4 steps in both passes; output hash = 006 |
| `jetson-edgedit-flux-klein-011__attempt__20261015-133558` | [jetson-edgedit-flux-klein-011](../experiments/jetson-edgedit-flux-klein-011.md) | Jetson Orin Nano, 25W | jetson-edgedit-flux-klein-q4-embq8-512-vaetile-taylorseer-smoke | – | – | – (tegrastats RAM 6.683 GiB) | complete; first + one observation (12069.0 ms), not a baseline; reused 0/4 (engine log) |
| `jetson-edgedit-flux-klein-011__baseline__20261015-133915` | [jetson-edgedit-flux-klein-011](../experiments/jetson-edgedit-flux-klein-011.md) | Jetson Orin Nano, 25W | jetson-edgedit-flux-klein-q4-embq8-512-vaetile-taylorseer | 11973.0 | 920.0 / 9200.0 / 1730.0 | – (tegrastats RAM 6.768 GiB) | complete; 1+3+10, one load (65.2 s); first 17583.0 ms; reused 0/4 steps in all passes (engine log); output hash = 006; MemAvailable min 540 MB |
| `jetson-edgedit-flux-klein-012__attempt__20261015-134457` | [jetson-edgedit-flux-klein-012](../experiments/jetson-edgedit-flux-klein-012.md) | Jetson Orin Nano, 25W | jetson-edgedit-flux-klein-q4-embq8-512-vaetile-cache-dit-smoke | – | – | – (tegrastats RAM 6.574 GiB) | complete; screen, first + one observation (12057.0 ms), not a baseline; CacheDiT skipped 0/4 steps in both passes; output hash = 006 |
| `jetson-edgedit-flux-klein-013__attempt__20261015-134813` | [jetson-edgedit-flux-klein-013](../experiments/jetson-edgedit-flux-klein-013.md) | Jetson Orin Nano, 25W | jetson-edgedit-flux-klein-q4-embq8-512-vaetile-magcache-smoke | – | – | – (tegrastats RAM 6.654 GiB) | complete; first + one observation (5222.0 ms), not a baseline; reused 3/4 (engine log) |
| `jetson-edgedit-flux-klein-013__baseline__20261015-135118` | [jetson-edgedit-flux-klein-013](../experiments/jetson-edgedit-flux-klein-013.md) | Jetson Orin Nano, 25W | jetson-edgedit-flux-klein-q4-embq8-512-vaetile-magcache | 5099.5 | 910.0 / 2340.0 / 1735.0 | – (tegrastats RAM 6.771 GiB) | complete; 1+3+10, one load (64.9 s); first 10728.0 ms; reused 3/4 steps in all passes (engine log); output visibly degraded (garbled sign, blur; visual check); MemAvailable min 493 MB |
| `jetson-edgedit-flux-klein-014__attempt__20261015-135524` | [jetson-edgedit-flux-klein-014](../experiments/jetson-edgedit-flux-klein-014.md) | Jetson Orin Nano, 25W | jetson-edgedit-flux-klein-q4-embq8-512-vaetile-dicache-smoke | – | – | – (tegrastats RAM 6.805 GiB) | complete; first + one observation (12233.0 ms), not a baseline; reused 0/4 (engine log) |
| `jetson-edgedit-flux-klein-014__baseline__20261015-135841` | [jetson-edgedit-flux-klein-014](../experiments/jetson-edgedit-flux-klein-014.md) | Jetson Orin Nano, 25W | jetson-edgedit-flux-klein-q4-embq8-512-vaetile-dicache | 12174.5 | 920.0 / 9395.0 / 1730.0 | – (tegrastats RAM 6.874 GiB) | complete; 1+3+10, one load (66.0 s); first 18037.0 ms; reused 0/4 steps in all passes (engine log); denoise +2.3% vs 006; output hash = 006; MemAvailable min 430 MB |
| `jetson-edgedit-flux-klein-015__attempt__20261015-140427` | [jetson-edgedit-flux-klein-015](../experiments/jetson-edgedit-flux-klein-015.md) | Jetson Orin Nano, 25W | jetson-edgedit-flux-klein-q4-embq8-512-vaetile-sencache-smoke | – | – | – | failed (exit 1): "--cache sencache requires a calibrated profile" (--cache-profile / --cache-calibrate); no load, no generation; full protocol not run |

## Jetson sd.cpp option follow-ups (2026-10-14/15)

Distilled klein Q4 512²; same-session reference is the 2026-10-14 jetson-flux-klein-003 repeat.

| Run directory | Experiment note | Device | Config | End-to-end (ms) | Text / Denoise / VAE (ms) | Peak alloc (GiB) | Status |
|---|---|---|---|---|---|---|---|
| `jetson-flux-klein-003__repeat__20261014-163411` | [jetson-flux-klein-003](../experiments/jetson-flux-klein-003.md#results-2026-10-14-repeat-with-the-tiling-capable-harness) | Jetson Orin Nano, 25 W | jetson-flux-klein-q4-512-disk-baseline (repeat, tiling-capable harness, options inactive) | 60640.928 | 22563.202 / 36009.000 / 4638.280 | – (system 6.000) | complete; 1+3+10; same-session reference for 010/011 |
| `jetson-flux-klein-010__attempt__20261014-165043` | [jetson-flux-klein-010](../experiments/jetson-flux-klein-010.md) | Jetson Orin Nano, 25 W | jetson-flux-klein-q4-512-disk-noflash-smoke (record_kind missing) | – | – | – | failed: KeyError('record_kind') writing summary after both generations completed (config error); partial runs.csv only |
| `jetson-flux-klein-010__attempt__20261014-183050` | [jetson-flux-klein-010](../experiments/jetson-flux-klein-010.md) | Jetson Orin Nano, 25 W | jetson-flux-klein-q4-512-disk-noflash-smoke | – | – | – | complete; two-generation functional attempt, not a baseline |
| `jetson-flux-klein-010__baseline__20261014-183501` | [jetson-flux-klein-010](../experiments/jetson-flux-klein-010.md) | Jetson Orin Nano, 25 W | jetson-flux-klein-q4-512-disk-noflash (diffusion flash attention off) | 69739.370 | 24480.497 / 40811.902 / 4797.034 | – (system 5.990) | complete; 1+3+10; steady per-step 3.535 s |
| `jetson-flux-klein-011__attempt__20261014-165450` | [jetson-flux-klein-011](../experiments/jetson-flux-klein-011.md) | Jetson Orin Nano, 25 W | jetson-flux-klein-q4-512-disk-vaetile-smoke (record_kind missing) | – | – | – | failed: KeyError('record_kind') writing summary after both generations completed (config error); partial runs.csv only |
| `jetson-flux-klein-011__attempt__20261014-185236` | [jetson-flux-klein-011](../experiments/jetson-flux-klein-011.md) | Jetson Orin Nano, 25 W | jetson-flux-klein-q4-512-disk-vaetile-smoke | – | – | – | complete; two-generation functional attempt, not a baseline |
| `jetson-flux-klein-011__baseline__20261014-185639` | [jetson-flux-klein-011](../experiments/jetson-flux-klein-011.md) | Jetson Orin Nano, 25 W | jetson-flux-klein-q4-512-disk-vaetile (VAE tiling 128 px, 0.25 overlap; 25 tiles/decode) | 71633.709 | 24324.393 / 40981.023 / 5857.549 | – (system 5.967) | complete; 1+3+10; steady per-step 2.713 s |
| `jetson-flux-klein-012__attempt__20261015-115414` | [jetson-flux-klein-012](../experiments/jetson-flux-klein-012.md) | Jetson Orin Nano, 25 W | jetson-flux-klein-q4-512-resident-vaetile-smoke | – | – | – (system 6.409) | complete; two-generation functional attempt, not a baseline |
| `jetson-flux-klein-012__baseline__20261015-115718` | [jetson-flux-klein-012](../experiments/jetson-flux-klein-012.md) | Jetson Orin Nano, 25 W | jetson-flux-klein-q4-512-resident-vaetile (all weights resident on CUDA0, VAE tiling 128 px / 0.25) | – | – | – (system 6.438) | failed: generation 4 "flux segment 7/27 (flux.single_blocks.0) failed during weight preparation" (need 873.58 MB, available 872.71 MB); first + 3 warm-ups only (warm-up wall 16.28–16.38 s), no measured generation; guard not triggered (MemAvailable min 862 MB) |
| `jetson-flux-klein-013__attempt__20261015-120112` | [jetson-flux-klein-013](../experiments/jetson-flux-klein-013.md) | Jetson Orin Nano, 25 W | jetson-flux-klein-q4-512-autofit-vaetile-smoke | – | – | – (system 6.726) | failed: "qwen3 segment 12/29 (llm.text.layers.10) failed during weight preparation" (need 517.00 MB, available 515.91 MB), failed to encode prompt; no completed image; guard not triggered (MemAvailable min 496 MB) |

## A100 PyTorch under torch 2.8.0

Job 11959806 on gilbreth-g008; torch28-001 is the same-environment reference.

| Run directory | Experiment note | Device | Config | End-to-end (ms) | Text / Denoise / VAE (ms) | Peak alloc (GiB) | Status |
|---|---|---|---|---|---|---|---|
| `a100-flux-klein-torch28-001__attempt__20261015-151216` | [a100-flux-klein-torch28-001](../experiments/a100-flux-klein-torch28-001.md) | gilbreth-a100-40gb (PCIe, g008) | a100-flux-klein-torch28-bf16-smoke | – | – | 17.32 | complete (job 11959806); first + one measured (1337.4 ms), not a baseline; cold load 39.17 s, first generation 29970.1 ms |
| `a100-flux-klein-torch28-001__baseline__20261015-151548` | [a100-flux-klein-torch28-001](../experiments/a100-flux-klein-torch28-001.md) | gilbreth-a100-40gb (PCIe, g008) | a100-flux-klein-torch28-bf16 | 1290.5 | 46.2 / 1037.8 / 168.2 | 17.32 | complete (job 11959806); torch 2.8.0+cu126; first 1830.9 ms; reserved 19.61, device 20.62 GiB; output deterministic |
| `a100-flux-klein-compile-001__attempt__20261015-151623` | [a100-flux-klein-compile-001](../experiments/a100-flux-klein-compile-001.md) | gilbreth-a100-40gb (PCIe, g008) | a100-flux-klein-torch28-bf16-compile-smoke | – | – | 17.32 | complete (job 11959806); first + one measured (1024.2 ms), not a baseline; first generation 32155.9 ms (step 0 30826.5 ms, includes compilation) |
| `a100-flux-klein-compile-001__baseline__20261015-151740` | [a100-flux-klein-compile-001](../experiments/a100-flux-klein-compile-001.md) | gilbreth-a100-40gb (PCIe, g008) | a100-flux-klein-torch28-bf16-compile | 1021.9 | 46.3 / 758.1 / 171.7 | 17.32 | complete (job 11959806); −20.8% wall, steady step 190.8 vs 260.3 ms; first 6790.3 ms incl. compilation; output differs from eager (2.3 mean RGB levels), no visible change |
| `a100-flux-klein-torch28-002__attempt__20261015-151820` | [a100-flux-klein-torch28-002](../experiments/a100-flux-klein-torch28-002.md) | gilbreth-a100-40gb (PCIe, g008) | a100-flux-klein-torch28-bf16-condreuse-smoke | – | – | 17.32 | complete (job 11959806); first + one measured (1243.3 ms), not a baseline; hits 1, misses 1 |
| `a100-flux-klein-torch28-002__baseline__20261015-151840` | [a100-flux-klein-torch28-002](../experiments/a100-flux-klein-torch28-002.md) | gilbreth-a100-40gb (PCIe, g008) | a100-flux-klein-torch28-bf16-condreuse | 1261.5 | 0.19 (retrieval) / 1041.7 / 169.3 | 17.32 | complete (job 11959806); −2.2% wall vs torch28-001; hits 13, misses 1; output bit-identical to torch28-001 |
| `a100-flux-klein-torch28-003__attempt__20261015-151915` | [a100-flux-klein-torch28-003](../experiments/a100-flux-klein-torch28-003.md) | gilbreth-a100-40gb (PCIe, g008) | a100-flux-klein-torch28-bf16-fbcache-smoke | – | – | – | failed (job 11959806): `Flux2TransformerBlock` not registered for First-Block Cache in diffusers 0.40.0; no generation; full protocol not run |
| `a100-flux-klein-torch28-004__attempt__20261015-151930` | [a100-flux-klein-torch28-004](../experiments/a100-flux-klein-torch28-004.md) | gilbreth-a100-40gb (PCIe, g008) | a100-flux-klein-torch28-bf16-taylorseer-smoke | – | – | 18.64 | complete (job 11959806); first + one measured (850.0 ms), not a baseline |
| `a100-flux-klein-torch28-004__baseline__20261015-151949` | [a100-flux-klein-torch28-004](../experiments/a100-flux-klein-torch28-004.md) | gilbreth-a100-40gb (PCIe, g008) | a100-flux-klein-torch28-bf16-taylorseer | 854.2 | 46.4 / 592.8 / 168.1 | 18.64 | complete (job 11959806); steps 1 and 3 predicted (36.7 / 39.6 ms); wall −33.8% vs torch28-001; peak alloc +1.32 GiB; output visibly changed (sign "molon", tone) |

## A100 edge-dit.cpp step caches

Job 11959806 on gilbreth-g008; reference is a100-edgedit-flux-klein-001.

| Run directory | Experiment note | Device | Config | End-to-end (ms) | Text / Denoise / VAE (ms) | Peak alloc (GiB) | Status |
|---|---|---|---|---|---|---|---|
| `a100-edgedit-flux-klein-002__attempt__20261015-152019` | [a100-edgedit-flux-klein-002](../experiments/a100-edgedit-flux-klein-002.md) | A100 PCIe 40GB, g008 | a100-edgedit-flux-klein-bf16-easycache-smoke | – | – | unavailable | complete (job 11959806); screen, first + one measured (1677.0 ms), not a baseline; skipped 0/4; output hash = 001 reference; full protocol not run |
| `a100-edgedit-flux-klein-003__attempt__20261015-152035` | [a100-edgedit-flux-klein-003](../experiments/a100-edgedit-flux-klein-003.md) | A100 PCIe 40GB, g008 | a100-edgedit-flux-klein-bf16-ucache-smoke | – | – | unavailable | complete (job 11959806); screen, first + one measured (1658.0 ms), not a baseline; skipped 0/4; output hash = 001 reference; full protocol not run |
| `a100-edgedit-flux-klein-004__attempt__20261015-152044` | [a100-edgedit-flux-klein-004](../experiments/a100-edgedit-flux-klein-004.md) | A100 PCIe 40GB, g008 | a100-edgedit-flux-klein-bf16-dbcache-smoke | – | – | unavailable | complete (job 11959806); screen, first + one measured (1658.0 ms), not a baseline; skipped 0/4; output hash = 001 reference; full protocol not run |
| `a100-edgedit-flux-klein-005__attempt__20261015-152052` | [a100-edgedit-flux-klein-005](../experiments/a100-edgedit-flux-klein-005.md) | A100 PCIe 40GB, g008 | a100-edgedit-flux-klein-bf16-taylorseer-smoke | – | – | unavailable | complete (job 11959806); screen, first + one measured (1659.0 ms), not a baseline; reused 0/4; output hash = 001 reference; device peak 21.894 GiB; full protocol not run |
| `a100-edgedit-flux-klein-006__attempt__20261015-152101` | [a100-edgedit-flux-klein-006](../experiments/a100-edgedit-flux-klein-006.md) | A100 PCIe 40GB, g008 | a100-edgedit-flux-klein-bf16-cache-dit-smoke | – | – | unavailable | complete (job 11959806); screen, first + one measured (1660.0 ms), not a baseline; skipped 0/4; output hash = 001 reference; full protocol not run |
| `a100-edgedit-flux-klein-007__attempt__20261015-152110` | [a100-edgedit-flux-klein-007](../experiments/a100-edgedit-flux-klein-007.md) | A100 PCIe 40GB, g008 | a100-edgedit-flux-klein-bf16-magcache-smoke | – | – | unavailable | complete (job 11959806); first + one measured (830.0 ms), not a baseline; reused 3/4 |
| `a100-edgedit-flux-klein-007__baseline__20261015-152117` | [a100-edgedit-flux-klein-007](../experiments/a100-edgedit-flux-klein-007.md) | A100 PCIe 40GB, g008 | a100-edgedit-flux-klein-bf16-magcache | 825.5 | unavailable / 290.1 / 272.5 | unavailable | complete (job 11959806); full 1+3+10; reused 3/4 in all passes; denoise −73.5% vs 001; whole-child device peak 21.847 GiB; output visibly degraded (garbled sign, blur) |
| `a100-edgedit-flux-klein-008__attempt__20261015-152136` | [a100-edgedit-flux-klein-008](../experiments/a100-edgedit-flux-klein-008.md) | A100 PCIe 40GB, g008 | a100-edgedit-flux-klein-bf16-dicache-smoke | – | – | unavailable | complete (job 11959806); screen, first + one measured (1699.0 ms), not a baseline; reused 0/4; output hash = 001 reference; device peak 22.320 GiB; full protocol not run |
| `a100-edgedit-flux-klein-009__attempt__20261015-152145` | [a100-edgedit-flux-klein-009](../experiments/a100-edgedit-flux-klein-009.md) | A100 PCIe 40GB, g008 | a100-edgedit-flux-klein-bf16-sencache-smoke | – | – | – | failed (job 11959806, exit 1): "--cache sencache requires a calibrated profile" (--cache-profile / --cache-calibrate); no load, no generation; full protocol not run |

## A100 klein Base (2026-10-15)

klein Base, 1024², 50 steps, CFG 4. PyTorch rows: job 11959986 on gilbreth-g009, torch 2.8.0+cu126, BF16, reference a100-flux-klein-base-torch28-001. edge-dit.cpp rows: GGUF Q4_0 transformer / Q4_K_M encoder / BF16 VAE (compressed workload), job 11960173 on gilbreth-g002 (failed first attempt in job 11959986 on g009), reference a100-edgedit-flux-klein-base-001; Text / Denoise / VAE from the engine breakdown (10 ms rounding); no allocator metrics, device peak is one whole-process NVML value. Cache counts are transformer passes out of 100.

| Run directory | Experiment note | Device | Config | End-to-end (ms) | Text / Denoise / VAE (ms) | Peak alloc (GiB) | Status |
|---|---|---|---|---|---|---|---|
| `a100-flux-klein-base-torch28-001__attempt__20261015-154758` | [a100-flux-klein-base-torch28-001](../experiments/a100-flux-klein-base-torch28-001.md) | gilbreth-a100-40gb (PCIe, g009) | a100-flux-klein-base-torch28-bf16-smoke | – | – | 17.33 | complete (job 11959986); first + one measured (26077.5 ms), not a baseline; cold load 45.57 s, first generation 54393.1 ms |
| `a100-flux-klein-base-torch28-001__baseline__20261015-155338` | [a100-flux-klein-base-torch28-001](../experiments/a100-flux-klein-base-torch28-001.md) | gilbreth-a100-40gb (PCIe, g009) | a100-flux-klein-base-torch28-bf16 | 26476.6 | 93.0 / 26171.4 / 170.0 | 17.33 | complete (job 11959986); first 26518.4 ms; steady step 523.6 ms (2 CFG calls); reserved 19.61, device 20.62 GiB; output deterministic (sign "HELLLO WORLD") |
| `a100-flux-klein-base-torch28-002__attempt__20261015-160007` | [a100-flux-klein-base-torch28-002](../experiments/a100-flux-klein-base-torch28-002.md) | gilbreth-a100-40gb (PCIe, g009) | a100-flux-klein-base-torch28-bf16-condreuse-smoke | – | – | 17.33 | complete (job 11959986); first + one measured (26148.5 ms), not a baseline; hits 2, misses 2 |
| `a100-flux-klein-base-torch28-002__baseline__20261015-160117` | [a100-flux-klein-base-torch28-002](../experiments/a100-flux-klein-base-torch28-002.md) | gilbreth-a100-40gb (PCIe, g009) | a100-flux-klein-base-torch28-bf16-condreuse | 26290.4 | 0.38 (retrieval) / 26069.8 / 169.7 | 17.33 | complete (job 11959986); −0.70% wall vs base-torch28-001; hits 26, misses 2; output bit-identical to base-torch28-001 |
| `a100-flux-klein-base-torch28-003__attempt__20261015-160743` | [a100-flux-klein-base-torch28-003](../experiments/a100-flux-klein-base-torch28-003.md) | gilbreth-a100-40gb (PCIe, g009) | a100-flux-klein-base-torch28-bf16-taylorseer-smoke | – | – | 19.97 | complete (job 11959986); first + one measured (9926.5 ms), not a baseline; 13 computed / 37 predicted steps |
| `a100-flux-klein-base-torch28-003__baseline__20261015-160821` | [a100-flux-klein-base-torch28-003](../experiments/a100-flux-klein-base-torch28-003.md) | gilbreth-a100-40gb (PCIe, g009) | a100-flux-klein-base-torch28-bf16-taylorseer | 9980.7 | 92.8 / 9672.4 / 169.5 | 19.97 | complete (job 11959986); TaylorSeer interval 5 from step 3; 37 of 50 steps predicted (77.6 vs 523.0 ms); wall −62.3%; peak alloc +2.64 GiB; output changed, no visible artefact (sign "hello world") |
| `a100-flux-klein-base-torch28-004__attempt__20261015-161059` | [a100-flux-klein-base-torch28-004](../experiments/a100-flux-klein-base-torch28-004.md) | gilbreth-a100-40gb (PCIe, g009) | a100-flux-klein-base-torch28-bf16-compile-smoke | – | – | 17.33 | complete (job 11959986); first + one measured (19269.0 ms), not a baseline; first generation 50287.7 ms (step 0 31039.8 ms, includes compilation) |
| `a100-flux-klein-base-torch28-004__baseline__20261015-161253` | [a100-flux-klein-base-torch28-004](../experiments/a100-flux-klein-base-torch28-004.md) | gilbreth-a100-40gb (PCIe, g009) | a100-flux-klein-base-torch28-bf16-compile | 19525.4 | 94.5 / 19203.5 / 172.3 | 17.33 | complete (job 11959986); −26.3% wall, steady step 384.2 vs 523.6 ms; first 25021.7 ms incl. compilation; output differs from eager and from the smoke process in pixels, visually near-identical |
| `a100-edgedit-flux-klein-base-001__attempt__20261015-161752` | [a100-edgedit-flux-klein-base-001](../experiments/a100-edgedit-flux-klein-base-001.md) | A100 PCIe 40GB, g009 | a100-edgedit-flux-klein-base-q4-1024-smoke | – | – | unavailable | failed (job 11959986): runner `KeyError('denoise_step_4_ms')` in summary (step-count variable shadowed in `parse_log`, fixed 2026-10-15); both generations completed (final image hash = later runs); no summary.json |
| `a100-edgedit-flux-klein-base-001__attempt__20261015-164901` | [a100-edgedit-flux-klein-base-001](../experiments/a100-edgedit-flux-klein-base-001.md) | A100 PCIe 40GB, g002 | a100-edgedit-flux-klein-base-q4-1024-smoke | – | – | unavailable | complete (job 11960173); first + one measured (49129.0 ms), not a baseline; device peak 11.732 GiB |
| `a100-edgedit-flux-klein-base-001__baseline__20261015-165052` | [a100-edgedit-flux-klein-base-001](../experiments/a100-edgedit-flux-klein-base-001.md) | A100 PCIe 40GB, g002 | a100-edgedit-flux-klein-base-q4-1024 | 49428.5 | 120 / 48740 / 280 | unavailable | complete (job 11960173); full 1+3+10; no cache; first 49476.0 ms; whole-child device peak 11.732 GiB; sign "hello world" |
| `a100-edgedit-flux-klein-base-002__attempt__20261015-170234` | [a100-edgedit-flux-klein-base-002](../experiments/a100-edgedit-flux-klein-base-002.md) | A100 PCIe 40GB, g002 | a100-edgedit-flux-klein-base-q4-1024-easycache-smoke | – | – | unavailable | complete (job 11960173); first + one measured (35376.0 ms), not a baseline; 29/100 passes skipped/reused |
| `a100-edgedit-flux-klein-base-002__baseline__20261015-170356` | [a100-edgedit-flux-klein-base-002](../experiments/a100-edgedit-flux-klein-base-002.md) | A100 PCIe 40GB, g002 | a100-edgedit-flux-klein-base-q4-1024-easycache | 35435.5 | 120 / 34740 / 280 | unavailable | complete (job 11960173); full 1+3+10; 29/100 passes skipped/reused; denoise −28.7%, wall −28.3% vs base-001; device peak 11.732 GiB; sign legible, background slightly paler |
| `a100-edgedit-flux-klein-base-003__attempt__20261015-171222` | [a100-edgedit-flux-klein-base-003](../experiments/a100-edgedit-flux-klein-base-003.md) | A100 PCIe 40GB, g002 | a100-edgedit-flux-klein-base-q4-1024-ucache-smoke | – | – | unavailable | complete (job 11960173); first + one measured (36838.0 ms), not a baseline; 26/100 passes skipped/reused |
| `a100-edgedit-flux-klein-base-003__baseline__20261015-171344` | [a100-edgedit-flux-klein-base-003](../experiments/a100-edgedit-flux-klein-base-003.md) | A100 PCIe 40GB, g002 | a100-edgedit-flux-klein-base-q4-1024-ucache | 36879.0 | 120 / 36190 / 280 | unavailable | complete (job 11960173); full 1+3+10; 26/100 passes skipped/reused; denoise −25.7%, wall −25.4% vs base-001; device peak 11.732 GiB; sign legible |
| `a100-edgedit-flux-klein-base-004__attempt__20261015-172231` | [a100-edgedit-flux-klein-base-004](../experiments/a100-edgedit-flux-klein-base-004.md) | A100 PCIe 40GB, g002 | a100-edgedit-flux-klein-base-q4-1024-dbcache-smoke | – | – | unavailable | complete (job 11960173); first + one measured (37798.0 ms), not a baseline; 24/100 passes skipped/reused |
| `a100-edgedit-flux-klein-base-004__baseline__20261015-172354` | [a100-edgedit-flux-klein-base-004](../experiments/a100-edgedit-flux-klein-base-004.md) | A100 PCIe 40GB, g002 | a100-edgedit-flux-klein-base-q4-1024-dbcache | 37910.0 | 120 / 37150 / 280 | unavailable | complete (job 11960173); full 1+3+10; 24/100 passes skipped/reused; denoise −23.8%, wall −23.3% vs base-001; device peak 11.732 GiB; sign legible; output hash = 006 |
| `a100-edgedit-flux-klein-base-005__attempt__20261015-173255` | [a100-edgedit-flux-klein-base-005](../experiments/a100-edgedit-flux-klein-base-005.md) | A100 PCIe 40GB, g002 | a100-edgedit-flux-klein-base-q4-1024-taylorseer-smoke | – | – | unavailable | complete (job 11960173); first + one measured (31296.0 ms), not a baseline; 38/100 passes skipped/reused |
| `a100-edgedit-flux-klein-base-005__baseline__20261015-173406` | [a100-edgedit-flux-klein-base-005](../experiments/a100-edgedit-flux-klein-base-005.md) | A100 PCIe 40GB, g002 | a100-edgedit-flux-klein-base-q4-1024-taylorseer | 31128.0 | 120 / 30370 / 280 | unavailable | complete (job 11960173); full 1+3+10; 38/100 passes skipped/reused; denoise −37.7%, wall −37.0% vs base-001; device peak 12.206 GiB; sign legible, 3.0 mean RGB levels from 001 |
| `a100-edgedit-flux-klein-base-006__attempt__20261015-174134` | [a100-edgedit-flux-klein-base-006](../experiments/a100-edgedit-flux-klein-base-006.md) | A100 PCIe 40GB, g002 | a100-edgedit-flux-klein-base-q4-1024-cache-dit-smoke | – | – | unavailable | complete (job 11960173); first + one measured (37970.0 ms), not a baseline; 24/100 passes skipped/reused |
| `a100-edgedit-flux-klein-base-006__baseline__20261015-174257` | [a100-edgedit-flux-klein-base-006](../experiments/a100-edgedit-flux-klein-base-006.md) | A100 PCIe 40GB, g002 | a100-edgedit-flux-klein-base-q4-1024-cache-dit | 37925.0 | 120 / 37140 / 280 | unavailable | complete (job 11960173); full 1+3+10; 24/100 passes skipped/reused; denoise −23.8%, wall −23.3% vs base-001; device peak 11.732 GiB; output hash = 004 (DBCache) |
| `a100-edgedit-flux-klein-base-007__attempt__20261015-175159` | [a100-edgedit-flux-klein-base-007](../experiments/a100-edgedit-flux-klein-base-007.md) | A100 PCIe 40GB, g002 | a100-edgedit-flux-klein-base-q4-1024-magcache-smoke | – | – | unavailable | complete (job 11960173); first + one measured (15619.0 ms), not a baseline; 70/100 passes skipped/reused |
| `a100-edgedit-flux-klein-base-007__baseline__20261015-175238` | [a100-edgedit-flux-klein-base-007](../experiments/a100-edgedit-flux-klein-base-007.md) | A100 PCIe 40GB, g002 | a100-edgedit-flux-klein-base-q4-1024-magcache | 15707.5 | 120 / 14890 / 280 | unavailable | complete (job 11960173); full 1+3+10; 70/100 passes skipped/reused; denoise −69.5%, wall −68.2% vs base-001; device peak 12.019 GiB; sign misspelled ("hdlloo world") |
| `a100-edgedit-flux-klein-base-008__attempt__20261015-175629` | [a100-edgedit-flux-klein-base-008](../experiments/a100-edgedit-flux-klein-base-008.md) | A100 PCIe 40GB, g002 | a100-edgedit-flux-klein-base-q4-1024-dicache-smoke | – | – | unavailable | complete (job 11960173); first + one measured (24993.0 ms), not a baseline; 54/100 passes skipped/reused |
| `a100-edgedit-flux-klein-base-008__baseline__20261015-175727` | [a100-edgedit-flux-klein-base-008](../experiments/a100-edgedit-flux-klein-base-008.md) | A100 PCIe 40GB, g002 | a100-edgedit-flux-klein-base-q4-1024-dicache | 25154.5 | 120 / 24400 / 280 | unavailable | complete (job 11960173); full 1+3+10; 54/100 passes skipped/reused; denoise −49.9%, wall −49.1% vs base-001; device peak 12.630 GiB; sign legible, 4.1 mean RGB levels from 001 |

## Jetson klein Base (2026-10-15/16)

klein Base, 512², 50 steps, CFG 4; Jetson Orin Nano 25W, queue `scripts/jetson_gap_queue_20261016f.sh`. edge-dit.cpp rows: GGUF Q4_0 transformer / Q4_K_M encoder with Q8_0 embedding / BF16 VAE (compressed workload), all resident, VAE tiling, reference jetson-edgedit-flux-klein-base-001; Text / Denoise / VAE from the engine breakdown (10 ms rounding); memory is whole-system tegrastats. Cache counts are transformer passes out of 100.

| Run directory | Experiment note | Device | Config | End-to-end (ms) | Text / Denoise / VAE (ms) | Peak alloc (GiB) | Status |
|---|---|---|---|---|---|---|---|
| `jetson-edgedit-flux-klein-base-001__attempt__20261015-154751` | [jetson-edgedit-flux-klein-base-001](../experiments/jetson-edgedit-flux-klein-base-001.md) | Jetson Orin Nano, 25W | jetson-edgedit-flux-klein-base-q4-embq8-512-vaetile-smoke | – | – | – | failed; `KeyError('denoise_step_4_ms')` in the runner's log parser (fixed), not a measurement |
| `jetson-edgedit-flux-klein-base-001__attempt__20261015-192415` | [jetson-edgedit-flux-klein-base-001](../experiments/jetson-edgedit-flux-klein-base-001.md) | Jetson Orin Nano, 25W | jetson-edgedit-flux-klein-base-q4-embq8-512-vaetile-smoke | – | – | – | complete; first + one observation (232355.0 ms), not a baseline |
| `jetson-edgedit-flux-klein-base-001__baseline__20261015-193452` | [jetson-edgedit-flux-klein-base-001](../experiments/jetson-edgedit-flux-klein-base-001.md) | Jetson Orin Nano, 25W | jetson-edgedit-flux-klein-base-q4-embq8-512-vaetile | 232499.0 | 930.0 / 228770.0 / 1750.0 | – (tegrastats RAM 6.731 GiB) | complete; 1+3+10, one load (63.1 s); first 238765.0 ms; no cache; MemAvailable min 568 MB |
| `jetson-edgedit-flux-klein-base-002__attempt__20261015-155801` | [jetson-edgedit-flux-klein-base-002](../experiments/jetson-edgedit-flux-klein-base-002.md) | Jetson Orin Nano, 25W | jetson-edgedit-flux-klein-base-q4-embq8-512-vaetile-easycache-smoke | – | – | – | failed; `KeyError('denoise_step_4_ms')` in the runner's log parser (fixed), not a measurement |
| `jetson-edgedit-flux-klein-base-002__attempt__20261015-203159` | [jetson-edgedit-flux-klein-base-002](../experiments/jetson-edgedit-flux-klein-base-002.md) | Jetson Orin Nano, 25W | jetson-edgedit-flux-klein-base-q4-embq8-512-vaetile-easycache-smoke | – | – | – | complete; first + one observation (166399.0 ms), not a baseline; 29/100 passes |
| `jetson-edgedit-flux-klein-base-002__baseline__20261015-204021` | [jetson-edgedit-flux-klein-base-002](../experiments/jetson-edgedit-flux-klein-base-002.md) | Jetson Orin Nano, 25W | jetson-edgedit-flux-klein-base-q4-embq8-512-vaetile-easycache | 166730.5 | 920.0 / 163025.0 / 1750.0 | – (tegrastats RAM 6.708 GiB) | complete; 1+3+10, one load (63.0 s); first 171297.0 ms; 29/100 passes skipped/reused in all generations; wall −28.3% vs base-001; MemAvailable min 581 MB |
| `jetson-edgedit-flux-klein-base-003__attempt__20261015-160624` | [jetson-edgedit-flux-klein-base-003](../experiments/jetson-edgedit-flux-klein-base-003.md) | Jetson Orin Nano, 25W | jetson-edgedit-flux-klein-base-q4-embq8-512-vaetile-ucache-smoke | – | – | – | failed; `KeyError('denoise_step_4_ms')` in the runner's log parser (fixed), not a measurement |
| `jetson-edgedit-flux-klein-base-003__attempt__20261015-212203` | [jetson-edgedit-flux-klein-base-003](../experiments/jetson-edgedit-flux-klein-base-003.md) | Jetson Orin Nano, 25W | jetson-edgedit-flux-klein-base-q4-embq8-512-vaetile-ucache-smoke | – | – | – | complete; first + one observation (173173.0 ms), not a baseline; 26/100 passes |
| `jetson-edgedit-flux-klein-base-003__baseline__20261015-213038` | [jetson-edgedit-flux-klein-base-003](../experiments/jetson-edgedit-flux-klein-base-003.md) | Jetson Orin Nano, 25W | jetson-edgedit-flux-klein-base-q4-embq8-512-vaetile-ucache | 173627.0 | 925.0 / 169910.0 / 1740.0 | – (tegrastats RAM 6.725 GiB) | complete; 1+3+10, one load (63.5 s); first 179553.0 ms; 26/100 passes skipped/reused in all generations; wall −25.3% vs base-001; MemAvailable min 568 MB |
| `jetson-edgedit-flux-klein-base-004__attempt__20261015-161500` | [jetson-edgedit-flux-klein-base-004](../experiments/jetson-edgedit-flux-klein-base-004.md) | Jetson Orin Nano, 25W | jetson-edgedit-flux-klein-base-q4-embq8-512-vaetile-dbcache-smoke | – | – | – | failed; `KeyError('denoise_step_4_ms')` in the runner's log parser (fixed), not a measurement |
| `jetson-edgedit-flux-klein-base-004__attempt__20261015-221400` | [jetson-edgedit-flux-klein-base-004](../experiments/jetson-edgedit-flux-klein-base-004.md) | Jetson Orin Nano, 25W | jetson-edgedit-flux-klein-base-q4-embq8-512-vaetile-dbcache-smoke | – | – | – | complete; first + one observation (179941.0 ms), not a baseline; 23/100 passes |
| `jetson-edgedit-flux-klein-base-004__baseline__20261015-222249` | [jetson-edgedit-flux-klein-base-004](../experiments/jetson-edgedit-flux-klein-base-004.md) | Jetson Orin Nano, 25W | jetson-edgedit-flux-klein-base-q4-embq8-512-vaetile-dbcache | 180076.0 | 930.0 / 176355.0 / 1740.0 | – (tegrastats RAM 6.757 GiB) | complete; 1+3+10, one load (63.1 s); first 185935.0 ms; 23/100 passes skipped/reused in all generations; wall −22.5% vs base-001; MemAvailable min 479 MB |
| `jetson-edgedit-flux-klein-base-005__attempt__20261015-162349` | [jetson-edgedit-flux-klein-base-005](../experiments/jetson-edgedit-flux-klein-base-005.md) | Jetson Orin Nano, 25W | jetson-edgedit-flux-klein-base-q4-embq8-512-vaetile-taylorseer-smoke | – | – | – | failed; `KeyError('denoise_step_4_ms')` in the runner's log parser (fixed), not a measurement |
| `jetson-edgedit-flux-klein-base-005__attempt__20261015-230740` | [jetson-edgedit-flux-klein-base-005](../experiments/jetson-edgedit-flux-klein-base-005.md) | Jetson Orin Nano, 25W | jetson-edgedit-flux-klein-base-q4-embq8-512-vaetile-taylorseer-smoke | – | – | – | complete; first + one observation (146188.0 ms), not a baseline; 38/100 passes |
| `jetson-edgedit-flux-klein-base-005__baseline__20261015-231521` | [jetson-edgedit-flux-klein-base-005](../experiments/jetson-edgedit-flux-klein-base-005.md) | Jetson Orin Nano, 25W | jetson-edgedit-flux-klein-base-q4-embq8-512-vaetile-taylorseer | 146273.5 | 930.0 / 142555.0 / 1740.0 | – (tegrastats RAM 6.851 GiB) | complete; 1+3+10, one load (63.4 s); first 151041.0 ms; 38/100 passes skipped/reused in all generations; wall −37.1% vs base-001; MemAvailable min 442 MB |
| `jetson-edgedit-flux-klein-base-006__attempt__20261015-163130` | [jetson-edgedit-flux-klein-base-006](../experiments/jetson-edgedit-flux-klein-base-006.md) | Jetson Orin Nano, 25W | jetson-edgedit-flux-klein-base-q4-embq8-512-vaetile-cache-dit-smoke | – | – | – | failed; `KeyError('denoise_step_4_ms')` in the runner's log parser (fixed), not a measurement |
| `jetson-edgedit-flux-klein-base-006__attempt__20261015-235220` | [jetson-edgedit-flux-klein-base-006](../experiments/jetson-edgedit-flux-klein-base-006.md) | Jetson Orin Nano, 25W | jetson-edgedit-flux-klein-base-q4-embq8-512-vaetile-cache-dit-smoke | – | – | – | complete; first + one observation (179873.0 ms), not a baseline; 23/100 passes |
| `jetson-edgedit-flux-klein-base-006__baseline__20261016-000110` | [jetson-edgedit-flux-klein-base-006](../experiments/jetson-edgedit-flux-klein-base-006.md) | Jetson Orin Nano, 25W | jetson-edgedit-flux-klein-base-q4-embq8-512-vaetile-cache-dit | 180237.5 | 920.0 / 176535.0 / 1750.0 | – (tegrastats RAM 6.729 GiB) | complete; 1+3+10, one load (63.2 s); first 185363.0 ms; 23/100 passes skipped/reused in all generations; wall −22.5% vs base-001; MemAvailable min 552 MB |
| `jetson-edgedit-flux-klein-base-007__attempt__20261015-164019` | [jetson-edgedit-flux-klein-base-007](../experiments/jetson-edgedit-flux-klein-base-007.md) | Jetson Orin Nano, 25W | jetson-edgedit-flux-klein-base-q4-embq8-512-vaetile-magcache-smoke | – | – | – | failed; `KeyError('denoise_step_4_ms')` in the runner's log parser (fixed), not a measurement |
| `jetson-edgedit-flux-klein-base-007__attempt__20261016-004603` | [jetson-edgedit-flux-klein-base-007](../experiments/jetson-edgedit-flux-klein-base-007.md) | Jetson Orin Nano, 25W | jetson-edgedit-flux-klein-base-q4-embq8-512-vaetile-magcache-smoke | – | – | – | complete; first + one observation (73101.0 ms), not a baseline; 70/100 passes |
| `jetson-edgedit-flux-klein-base-007__baseline__20261016-005119` | [jetson-edgedit-flux-klein-base-007](../experiments/jetson-edgedit-flux-klein-base-007.md) | Jetson Orin Nano, 25W | jetson-edgedit-flux-klein-base-q4-embq8-512-vaetile-magcache | 73108.0 | 920.0 / 69410.0 / 1745.0 | – (tegrastats RAM 6.772 GiB) | complete; 1+3+10, one load (63.0 s); first 77934.0 ms; 70/100 passes skipped/reused in all generations; wall −68.6% vs base-001; MemAvailable min 531 MB |
| `jetson-edgedit-flux-klein-base-008__attempt__20261015-164533` | [jetson-edgedit-flux-klein-base-008](../experiments/jetson-edgedit-flux-klein-base-008.md) | Jetson Orin Nano, 25W | jetson-edgedit-flux-klein-base-q4-embq8-512-vaetile-dicache-smoke | – | – | – | failed; killed by memory guard (MemAvailable 390 MB < 400 MB) at the start of pass 2; first pass 121.4 s, reused 54/100 (engine log) |
| `jetson-edgedit-flux-klein-base-008__attempt__20261016-011113` | [jetson-edgedit-flux-klein-base-008](../experiments/jetson-edgedit-flux-klein-base-008.md) | Jetson Orin Nano, 25W | jetson-edgedit-flux-klein-base-q4-embq8-512-vaetile-dicache-smoke | – | – | – | failed; killed by memory guard (MemAvailable 371 MB < 400 MB) at the start of pass 2; first pass 121.6 s, reused 54/100 (engine log) |

PyTorch/diffusers rows (same section): Base Q4_0 GGUF transformer + NF4 encoder + BF16 VAE (compressed workload), sequential stage residency, reference jetson-torch-flux-klein-base-001; stage times from the portable runner; Peak alloc is the torch allocator.

| Run directory | Experiment note | Device | Config | End-to-end (ms) | Text / Denoise / VAE (ms) | Peak alloc (GiB) | Status |
|---|---|---|---|---|---|---|---|
| `jetson-torch-flux-klein-base-001__attempt__20261015-165022` | [jetson-torch-flux-klein-base-001](../experiments/jetson-torch-flux-klein-base-001.md) | Jetson Orin Nano, 25W | jetson-torch-flux-klein-base-q4-nf4-512-smoke | – | – | – | failed; `IncompleteSnapshotError` (allow_patterns requested unneeded weight files; fixed), not a measurement |
| `jetson-torch-flux-klein-base-001__attempt__20261016-011604` | [jetson-torch-flux-klein-base-001](../experiments/jetson-torch-flux-klein-base-001.md) | Jetson Orin Nano, 25W | jetson-torch-flux-klein-base-q4-nf4-512-smoke | – | – | – | failed; `AssertionError('encoder reloads 4 != 2')` after both generations completed (runner expected one swap per generation under CFG; fixed), not a baseline |
| `jetson-torch-flux-klein-base-001__attempt__20261016-044214` | [jetson-torch-flux-klein-base-001](../experiments/jetson-torch-flux-klein-base-001.md) | Jetson Orin Nano, 25W | jetson-torch-flux-klein-base-q4-nf4-512-smoke | – | – | – | complete; first + one observation (499232.0 ms), not a baseline |
| `jetson-torch-flux-klein-base-001__baseline__20261016-050136` | [jetson-torch-flux-klein-base-001](../experiments/jetson-torch-flux-klein-base-001.md) | Jetson Orin Nano, 25W | jetson-torch-flux-klein-base-q4-nf4-512 | 494634.4 | 2742.1 / 360239.1 / 1174.0 | 3.09 | complete; 1+3+10, one load (76.0 s); first 504094.0 ms; two residency swaps per generation (~66 s each, in other); steady step 7193.5 ms; output deterministic |
| `jetson-torch-flux-klein-base-002__attempt__20261015-165125` | [jetson-torch-flux-klein-base-002](../experiments/jetson-torch-flux-klein-base-002.md) | Jetson Orin Nano, 25W | jetson-torch-flux-klein-base-q4-nf4-512-condreuse-smoke | – | – | – | failed; `IncompleteSnapshotError` (allow_patterns requested unneeded weight files; fixed), not a measurement |
| `jetson-torch-flux-klein-base-002__attempt__20261016-013540` | [jetson-torch-flux-klein-base-002](../experiments/jetson-torch-flux-klein-base-002.md) | Jetson Orin Nano, 25W | jetson-torch-flux-klein-base-q4-nf4-512-condreuse-smoke | – | – | – | complete; first + one observation (359142.1 ms), not a baseline; hits 2, misses 2 |
| `jetson-torch-flux-klein-base-002__baseline__20261016-015207` | [jetson-torch-flux-klein-base-002](../experiments/jetson-torch-flux-klein-base-002.md) | Jetson Orin Nano, 25W | jetson-torch-flux-klein-base-q4-nf4-512-condreuse | 359440.7 | 1.34 (retrieval) / 358891.9 / 484.2 | 3.09 | complete; 1+3+10; first 510586.6 ms (both swaps); hits 26, misses 2; wall −27.3% vs base-001; output bit-identical to base-001 |
| `jetson-torch-flux-klein-base-003__attempt__20261015-165212` | [jetson-torch-flux-klein-base-003](../experiments/jetson-torch-flux-klein-base-003.md) | Jetson Orin Nano, 25W | jetson-torch-flux-klein-base-q4-nf4-512-condreuse-taylorseer-smoke | – | – | – | failed; `IncompleteSnapshotError` (allow_patterns requested unneeded weight files; fixed), not a measurement |
| `jetson-torch-flux-klein-base-003__attempt__20261016-032053` | [jetson-torch-flux-klein-base-003](../experiments/jetson-torch-flux-klein-base-003.md) | Jetson Orin Nano, 25W | jetson-torch-flux-klein-base-q4-nf4-512-condreuse-taylorseer-smoke | – | – | – | failed; killed by memory guard (MemAvailable 391 MB < 400 MB) during the first text-encoder load, before any denoise step |
| `jetson-torch-flux-klein-base-004__attempt__20261015-165259` | [jetson-torch-flux-klein-base-004](../experiments/jetson-torch-flux-klein-base-004.md) | Jetson Orin Nano, 25W | jetson-torch-flux-klein-base-q4-nf4-512-condreuse-compile-smoke | – | – | – | failed; `IncompleteSnapshotError` (allow_patterns requested unneeded weight files; fixed), not a measurement |
| `jetson-torch-flux-klein-base-004__attempt__20261016-032541` | [jetson-torch-flux-klein-base-004](../experiments/jetson-torch-flux-klein-base-004.md) | Jetson Orin Nano, 25W | jetson-torch-flux-klein-base-q4-nf4-512-condreuse-compile-smoke | – | – | – | complete; first + one observation (184075.1 ms), not a baseline; first 450759.7 ms incl. compilation |
| `jetson-torch-flux-klein-base-004__baseline__20261016-033850` | [jetson-torch-flux-klein-base-004](../experiments/jetson-torch-flux-klein-base-004.md) | Jetson Orin Nano, 25W | jetson-torch-flux-klein-base-q4-nf4-512-condreuse-compile | 183926.8 | 1.41 (retrieval) / 183379.9 / 485.9 | 3.09 | complete; 1+3+10; first 449911.8 ms (step 0 120805.7 ms incl. compilation); steady step 3667.6 vs 7177.4 ms; wall −48.8% vs base-002; output deterministic, not bit-identical to eager |

## Jetson PyTorch kernel profiles (2026-10-17/18)

Distilled klein 512², configs of jetson-torch-flux-klein-004 / -007 run with `run_flux_portable.py --profile`: full protocol repeat (End-to-end = its measured median) plus one torch.profiler generation excluded from it.

| Run directory | Experiment note | Device | Config | End-to-end (ms) | Text / Denoise / VAE (ms) | Peak alloc (GiB) | Status |
|---|---|---|---|---|---|---|---|
| `jetson-torch-flux-klein-004__profile__20261017-215429` | [jetson-torch-flux-klein-004](../experiments/jetson-torch-flux-klein-004.md) | Jetson Orin Nano, 25W | jetson-torch-flux-klein-q4-nf4-512-condreuse (`--profile`) | 15051.5 | – | – | complete; protocol repeat + torch.profiler generation; CUPTI insufficient privileges (non-root), trace has CPU operators only |
| `jetson-torch-flux-klein-004__profile__20261017-222734` | [jetson-torch-flux-klein-004](../experiments/jetson-torch-flux-klein-004.md) | Jetson Orin Nano, 25W | jetson-torch-flux-klein-q4-nf4-512-condreuse (`--profile`, root) | 14917.7 | – | – | complete; protocol repeat + torch.profiler generation with CUDA kernels; step 3691.8 ms, 51% weight unpack/cast |
| `jetson-torch-flux-klein-007__profile__20261017-220257` | [jetson-torch-flux-klein-007](../experiments/jetson-torch-flux-klein-007.md) | Jetson Orin Nano, 25W | jetson-torch-flux-klein-q4-nf4-512-condreuse-compile (`--profile`) | – | – | – | failed; memory guard (391 MB < 400 MB) during first-generation compilation, desktop session active |
| `jetson-torch-flux-klein-007__profile__20261017-223538` | [jetson-torch-flux-klein-007](../experiments/jetson-torch-flux-klein-007.md) | Jetson Orin Nano, 25W | jetson-torch-flux-klein-q4-nf4-512-condreuse-compile (`--profile`, root) | 7846.6 | – | – | complete; protocol repeat + torch.profiler generation with CUDA kernels; step 1839.2 ms, fused unpack 313.9 ms |
