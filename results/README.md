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
