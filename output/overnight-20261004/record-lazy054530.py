import json
from pathlib import Path
rid='jetson-flux-klein-base-003__attempt__20261005-054530';p=Path('results/runs')/rid;s=json.loads((p/'summary.json').read_text());m=s['monitor_window']
t=f'''---
type: experiment-record
id: jetson-flux-klein-base-003
status: partial
device: jetson-orin-nano
engine: stable-diffusion-cpp
runs: [{rid}]
updated: 2026-10-05
---

# Jetson klein Base: lazy disk-backed loading

Status: **partial**; correctness attempt completed and validated. Full repeated protocol started separately.

## Question

Does disabling eager loading preserve execution and output for the pinned Base Q4 disk-backed workload? This single residency-policy variant serves Week 2 image baselines and RQ1. Its latency and memory distribution require the full repeated protocol.

## Setup

[Attempt config](../configs/jetson-flux-klein-base-q4-512-disk-lazy-smoke.json), [full config](../configs/jetson-flux-klein-base-q4-512-disk-lazy.json). Reference: configs/jetson-flux-klein-base-q4-512-disk.json on the same cache-capable binary. Only eager loading changes to false. Base Q4_0, Qwen3 Q4_K_M and original VAE pins remain unchanged; fifty Euler steps, flux2 scheduler, guidance4,512×512,batch1,seed0 and fixed cat/sign prompt. Disk-backed parameters, segmented CUDA computation, prefetch and diffusion flash attention remain enabled. No step cache, conditioning reuse, automatic fitting or VAE tiling.

Jetson Orin Nano,25W, four CPU threads. sd.cpp19bbbca1c736bbb9538679fc0ae690cb2b46b492, binary SHA2562d66fa04053c02b441d61a84d24412537a16f9099d6cb31b80dcca3c8cc1f896. [Environment](../results/runs/{rid}/environment.json); [metric definitions](../wiki/methods/baseline-metrics.md). Protocol:one first plus one additional observation, no warm-ups. These are correctness observations, not warm latency estimates.

## Results

[Run](../results/runs/{rid}/), [summary](../results/runs/{rid}/summary.json), [status](../results/runs/{rid}/status.json).

| Metric | First, ms | Second observation, ms |
|---|---:|---:|
'''
for k in ['wall_ms','text_encode_ms','denoise_ms','vae_decode_ms','other_ms']:t+=f"| {k} | {s['first_run'][k]:.3f} | {s['measured'][k]['median']:.3f} |\n"
t+=f'''
Context creation:{s['load']['load_total_s']:.3f}s, with parameter loading deferred into generation. File hashing occurs before measurement and warms filesystem cache. Whole-capture system RAM peak:{m['ram_peak_gib']:.3f}GiB; swap range:{m['swap_min_gib']:.3f}–{m['swap_max_gib']:.3f}GiB from{m['samples']}one-second samples. These include loading and both generations; no stage-specific allocator peak is available.

## Validation and quality diagnostics

[Validation receipt](../results/runs/{rid}/lazy-validation.json). Each generation completed fifty callback steps and one hundred single-segment transformer passes. Requested eager_load=false was confirmed; cache audit records zero enabled generations. Settings, phase order, summary aggregates, dimensions and saved PNG hashes passed independent checks. Parameter-placement logs show preparation on CUDA0 during execution; continuous placement and physical disk traffic remain unverified.

Both saved512×512images are pixel-identical to the same-binary no-cache reference. [Paired diagnostic receipt](../results/runs/{rid}/quality-diagnostics.json): RGB MSE0, PSNR infinite because pixels are identical, calibrated AlexNet LPIPSv0.1=0 with same-image control0. Full RGB images were evaluated without crop/resize on CPU; versions, input hashes and weight hashes are retained. The reference image was visually inspected as a coherent cat holding a readable sign. This is one development prompt/seed, not formal quality eligibility.

## Interpretation

The lazy-loading configuration passes the functional gate. A short context-creation duration excludes deferred parameter work and must not be interpreted as an end-to-end speedup. The full1+3+10protocol is needed to assess repeated latency and memory behavior before any combination selection.

## Next experiment

- Complete configs/jetson-flux-klein-base-q4-512-disk-lazy.json with1+3+10.

[W&B attempt](https://wandb.ai/mfaruqi-purdue-university/on-device-diffusion/runs/{rid}).
''';Path('experiments/jetson-flux-klein-base-003.md').write_text(t)
p=Path('results/README.md');t=p.read_text().replace('## A100 Base reference',f'| `{rid}` | [jetson-flux-klein-base-003](../experiments/jetson-flux-klein-base-003.md) | Jetson Orin Nano,25W | jetson-flux-klein-base-q4-512-disk-lazy-smoke | – | – | – | complete; two-observation lazy-loading correctness attempt |\n\n## A100 Base reference');p.write_text(t)
p=Path('wiki/experiments.md');t=p.read_text().replace('## Comparisons',f'| jetson-flux-klein-base-003 | Jetson Orin Nano | stable-diffusion.cpp | Base lazy disk-backed loading | partial; correctness passed, full protocol running | [record](../experiments/jetson-flux-klein-base-003.md) | [attempt](../results/runs/{rid}/) | [W&B](https://wandb.ai/mfaruqi-purdue-university/on-device-diffusion/runs/{rid}) |\n\n## Comparisons');p.write_text(t)
p=Path('wiki/log.md');t=p.read_text();pos=t.index('\n## ');t=t[:pos]+'''\n## [2026-10-05] experiment | Jetson Base lazy-loading correctness
[Base003](../experiments/jetson-flux-klein-base-003.md) passed callback/CFG/settings/image checks; paired pixels equal reference and LPIPS0. Full repeated protocol started; no speed-selected combination yet.
'''+t[pos:];p.write_text(t)
p=Path('wiki/hot.md');t=p.read_text().replace('lazy-loading correctness test running in tmux','[lazy-loading correctness passed](../experiments/jetson-flux-klein-base-003.md) and full protocol running in tmux');p.write_text(t)
