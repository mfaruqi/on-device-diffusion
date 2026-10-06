import json,re
from pathlib import Path
root=Path('results/runs');rid='jetson-flux-klein-base-005__baseline__20261005-093239';refid='jetson-flux-klein-base-001__repeat__20261005-042014';s=json.loads((root/rid/'summary.json').read_text());r=json.loads((root/refid/'summary.json').read_text());m=s['measured'];mem=s['monitor_window']
a=json.loads((root/rid/'config.json').read_text());b=json.loads((root/refid/'config.json').read_text())
for k in ['model','workload','protocol']:assert a[k]==b[k]
for k in b['harness_arguments']:
 if k!='conditioning-cache-size':assert a['harness_arguments'][k]==b['harness_arguments'][k]
assert m['wall_ms']['max']<r['measured']['wall_ms']['min']
receipt={'reference':refid,'variant':rid,'matched_model_workload_protocol':True,'only_harness_change':'conditioning-cache-size:0->2','speed_range_rule_pass':True,'reference_over_variant_median':r['measured']['wall_ms']['median']/m['wall_ms']['median'],'formal_quality_eligible':False}
(root/rid/'conditioning-comparison.json').write_text(json.dumps(receipt,indent=2)+'\n')
p=Path('experiments/jetson-flux-klein-base-005.md');t=p.read_text().replace('status: partial','status: complete').replace('runs: [','runs: ['+rid+', ',1).replace('Status: **partial**; two-generation correctness test passed; full repeated protocol running separately.','Status: **complete**; correctness test and full repeated protocol independently validated.').replace('A single warm observation cannot establish sustained latency, memory benefit, or speed-selection eligibility; cache capacity and behavior under multiple prompts are outside this run.','The complete repeated protocol is reported below; cache capacity and behavior under multiple prompts remain outside this run.').replace('- Complete configs/jetson-flux-klein-base-q4-512-conditioning-cache.json with the unchanged1+3+10protocol before combination selection.','- Test an explicitly labelled combination with EasyCache after the queued distilled tests.')
t+=f'\n## Full repeated protocol\n\n[Summary](../results/runs/{rid}/summary.json), [status](../results/runs/{rid}/status.json), [environment](../results/runs/{rid}/environment.json). Completed13:32:39–14:40:33UTC; first + three discarded warm-ups + ten measured generations.\n\n| Metric | Median, ms | Min–max, ms |\n|---|---:|---:|\n'
for k in ['wall_ms','text_encode_ms','denoise_ms','vae_decode_ms','other_ms']:
 v=m[k];t+=f"| {k} | {v['median']:.3f} | {v['min']:.3f}–{v['max']:.3f} |\n"
t+=f"\nFirst generation {s['first_run']['wall_ms']:.3f}ms; context creation {s['load']['load_total_s']:.6f}s with warmed file cache. Whole-capture RAM peak {mem['ram_peak_gib']:.6f}GiB, swap {mem['swap_min_gib']:.6f}–{mem['swap_max_gib']:.6f}GiB,{mem['samples']}samples.\n\n[Validation](../results/runs/{rid}/conditioning-validation.json) passed: all14generations retain50steps/100CFGpasses, cache hits0then2for every subsequent generation, all14RGBhashes agree. [Quality diagnostics](../results/runs/{rid}/quality-diagnostics.json): reference-identical pixels,MSE0,infinite PSNR,LPIPS0; no formal quality eligibility. The text-stage values measure cached conditioning retrieval, not encoder execution. [Matched comparison and selection](compare-jetson-base-conditioning-reuse.md).\n\n[W&B full run](https://wandb.ai/mfaruqi-purdue-university/on-device-diffusion/runs/{rid}).\n";p.write_text(t)
id='compare-jetson-base-conditioning-reuse';t=f'''---
type: experiment-record
id: {id}
status: complete
device: jetson-orin-nano
engine: stable-diffusion-cpp
runs: [{refid}, {rid}]
updated: 2026-10-05
---

# Compare: Jetson Base exact conditioning reuse

## Question

Does retaining fixed prompt conditioning reduce repeated generation latency? Week2 image baselines,RQ1; candidate selection for combined reuse policies.

## Runs and matching

| Role | Config and summary | Record |
|---|---|---|
| No-cache reference | [config](../results/runs/{refid}/config.json),[summary](../results/runs/{refid}/summary.json) | [Base001](jetson-flux-klein-base-001.md) |
| Exact conditioning reuse | [config](../results/runs/{rid}/config.json),[summary](../results/runs/{rid}/summary.json) | [Base005](jetson-flux-klein-base-005.md) |

| Held fixed | Value |
|---|---|
| Device/engine | Jetson Orin Nano25W,four CPU threads,same pinned sd.cpp and harness SHA256 |
| Model/workload | Identical Base Q4_0,Qwen3 Q4_K_M,BF16VAE pins;512²,50Euler/flux2steps,CFG4,seed0,batch1,same prompt |
| Protocol | One first,three warm-ups,ten measured; one context,one-second tegrastats,no profiler |
| Residency | Fixed eager disk-backed segmented CUDA,prefetch enabled,flash attention,no approximate step cache |
| Changed option | Conditioning cache capacity0→2 |

[Matching receipt](../results/runs/{rid}/conditioning-comparison.json). Sequential captures leave filesystem-cache,background,swap and thermal history uncontrolled; observed ratios are descriptive, not isolated causal effects.

## Results

| Metric | No-cache median (range), ms | Reuse median (range), ms |
|---|---:|---:|
'''
for k in ['wall_ms','text_encode_ms','denoise_ms','vae_decode_ms','other_ms']:
 x,y=r['measured'][k],m[k];t+=f"| {k} | {x['median']:.3f} ({x['min']:.3f}–{x['max']:.3f}) | {y['median']:.3f} ({y['min']:.3f}–{y['max']:.3f}) |\n"
x,y=r['measured']['wall_ms'],m['wall_ms'];t+=f"\nObserved median reduction {100*(1-y['median']/x['median']):.3f}%; reference/variant ratio {x['median']/y['median']:.6f}. Whole-capture sampled RAM {r['monitor_window']['ram_peak_gib']:.6f}→{mem['ram_peak_gib']:.6f}GiB; no isolated cache-allocation estimate. Cached text stage includes retrieval/setup. Denoising also changed despite unchanged transformer call count; this comparison cannot attribute that difference to a specific residency or I/O mechanism.\n\n## Findings and selection\n\nAll14images match reference pixels; [diagnostics](../results/runs/{rid}/quality-diagnostics.json) give MSE0,LPIPS0. The exact-option screen passes. Variant maximum {y['max']:.3f}ms is below reference minimum {x['min']:.3f}ms, so the predeclared speed rule also passes. This option qualifies for a separately labelled combination with the independently qualified EasyCache policy. No combination result exists yet.\n\n## Limits\n\nOne fixed development prompt/seed; prompt diversity,eviction behavior and formal/held-out quality are untested. No physical-I/O or continuous-residency measurement here. Saved exact pixels establish equality for this workload only.\n";Path('experiments/'+id+'.md').write_text(t)
p=Path('results/README.md');p.write_text(p.read_text()+f"\n| `{rid}` | [jetson-flux-klein-base-005](../experiments/jetson-flux-klein-base-005.md) | Jetson Orin Nano,25W | Base exact conditioning reuse | {m['wall_ms']['median']:.3f} | {m['text_encode_ms']['median']:.3f} / {m['denoise_ms']['median']:.3f} / {m['vae_decode_ms']['median']:.3f} | – (system {mem['ram_peak_gib']:.3f}) | complete; validated1+3+10 |\n")
p=Path('wiki/experiments.md');t=p.read_text().replace('Base exact conditioning reuse | partial; correctness passed, full running','Base exact conditioning reuse | complete; full protocol validated').replace('[attempt](../results/runs/jetson-flux-klein-base-005__attempt__20261005-091031/)',f'[attempt](../results/runs/jetson-flux-klein-base-005__attempt__20261005-091031/),[baseline](../results/runs/{rid}/)');t+=f'| Jetson Base exact conditioning reuse | jetson-flux-klein-base-001,jetson-flux-klein-base-005 | [record](../experiments/{id}.md) |\n';p.write_text(t)
p=Path('wiki/log.md');p.write_text(p.read_text().replace('# Log\n','# Log\n\n## [2026-10-05] experiment | Jetson Base conditioning full protocol validated\n[Base005](../experiments/jetson-flux-klein-base-005.md) retains identical pixels; [matched comparison](../experiments/compare-jetson-base-conditioning-reuse.md) passes the declared speed/fidelity screen.\n',1))
