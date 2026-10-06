import json
from pathlib import Path
rid='jetson-flux-klein-base-007__attempt__20261005-120925';root=Path('results/runs')/rid;s=json.loads((root/'summary.json').read_text());q=json.loads((root/'quality-diagnostics.json').read_text());g=json.loads((root/'combination-gate.json').read_text());assert g['status']=='passed';b=f'../results/runs/{rid}';eid='jetson-flux-klein-base-007'
t=f'''---
type: experiment-record
id: {eid}
status: partial
device: jetson-orin-nano
engine: stable-diffusion-cpp
runs: [{rid}]
updated: 2026-10-05
---

# {eid}: EasyCache with exact conditioning reuse

Status: **partial**; two-generation correctness gate passed, full protocol started separately.

## Question

Can the two individually qualified reuse policies execute together? Week 2 image baseline work, RQ1 and preparation for RQ3 joint-policy evaluation ([proposal overview](../wiki/project/overview.md)). The added change relative to the EasyCache-only configuration is conditioning cache capacity 2; the combination is explicitly labelled.

## Setup

[Config]({b}/config.json), [environment]({b}/environment.json), [status]({b}/status.json). Jetson Orin Nano, 25W, four CPU threads; pinned Base Q4_0 transformer, Qwen3 Q4_K_M and BF16 VAE; 512², 50 Euler/flux2 steps, CFG 4, seed 0, batch 1 and the fixed development prompt. Eager disk-backed segmented CUDA execution, prefetch enabled, mmap off. EasyCache threshold 0.2, start 0.15, end 0.95, plus exact positive/negative conditioning capacity 2. No precision, checkpoint or residency fallback.

Pinned sd.cpp19bbbca and unchanged cache-capable harness. Isolated campaign snapshot e614bb0cf8041b65cf4e327d70d5cb1079f3c33e. First generation plus one measured observation, no discarded warm-ups; one model context. Completed 16:09:25–16:16:40 UTC. [Metric definitions](../wiki/methods/baseline-metrics.md).

## Results

[Summary]({b}/summary.json).

| Metric | First, ms | Second observation, ms |
|---|---:|---:|
'''
for k in ['wall_ms','text_encode_ms','denoise_ms','vae_decode_ms','other_ms']:t+=f"| {k} | {s['first_run'][k]:.3f} | {s['measured'][k]['median']:.3f} |\n"
m=s['monitor_window'];t+=f"\nContext creation {s['load']['load_total_s']:.6f} s after file-cache-warming hash checks. Whole-capture RAM peak {m['ram_peak_gib']:.6f} GiB, swap {m['swap_min_gib']:.6f}–{m['swap_max_gib']:.6f} GiB, {m['samples']} samples.\n\n## Validation and quality\n\n[Independent execution validation]({b}/validation.json) and [combination gate]({b}/combination-gate.json) passed. Conditioning hits [0,2]; EasyCache enabled in both generations and skipped [25,25] steps. Each image executed 50 actual transformer passes (two CFG passes per non-skipped step), with 50 scheduler progress steps. Both saved PNGs are 512×512; hashes match the generated rows. Informal inspection shows a coherent cat with a legible hello world sign.\n\n[Paired diagnostic]({b}/quality-diagnostics.json) against the no-cache Base reference: PSNR {q['psnr_db']:.9f} dB, LPIPS {q['lpips_alex_v0_1']:.9f}. The measured RGB pixels are identical to the EasyCache-only output in the [gate receipt]({b}/combination-gate.json). One development prompt/seed; approximate reuse remains approximate, with no formal quality eligibility.\n\n## Interpretation\n\nExact conditioning reuse and approximate denoising reuse both activate in this configuration. The cached text stage measures retrieval/setup after the first generation. One measured observation cannot establish a sustained speed improvement or combination-selection result; the full protocol is required.\n\n## Next experiment\n\nComplete `configs/jetson-flux-klein-base-q4-512-easycache-conditioning.json` with the unchanged 1+3+10 protocol.\n"
p=Path('experiments')/(eid+'.md');assert not p.exists();p.write_text(t)
p=Path('results/README.md');p.write_text(p.read_text()+f'| `{rid}` | [{eid}](../experiments/{eid}.md) | Jetson Orin Nano,25W | Base EasyCache + exact conditioning | – | – | – | complete; two-generation correctness gate, full pending |\n')
p=Path('wiki/experiments.md');t=p.read_text().replace('## Comparisons',f'| {eid} | Jetson Orin Nano | stable-diffusion.cpp | Base EasyCache + exact conditioning | partial; correctness passed, full running | [record](../experiments/{eid}.md) | [attempt]({b}/) | – |\n\n## Comparisons',1);p.write_text(t)
p=Path('wiki/log.md');p.write_text(p.read_text().replace('# Log\n',f'# Log\n\n## [2026-10-05] experiment | Base combined reuse correctness passed\n[{eid}](../experiments/{eid}.md): exact conditioning and EasyCache both activate; functional/paired image diagnostics passed, full protocol running.\n',1))
