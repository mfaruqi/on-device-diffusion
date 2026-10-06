import json
from pathlib import Path
rid='jetson-flux-klein-008__baseline__20261005-113721';ref='jetson-flux-klein-003__baseline__20261005-105911';root=Path('results/runs');s=json.loads((root/rid/'summary.json').read_text());r=json.loads((root/ref/'summary.json').read_text());c=json.loads((root/rid/'config.json').read_text());rc=json.loads((root/ref/'config.json').read_text());q=json.loads((root/rid/'quality-diagnostics.json').read_text());v=json.loads((root/rid/'validation.json').read_text());assert v['status']=='passed'
for k in ['model','workload','protocol']:assert c[k]==rc[k]
assert c['engine_settings']=={**rc['engine_settings'],'disable_prefetch':True}
for k,value in rc['harness_arguments'].items():
 if k!='disable-prefetch':assert c['harness_arguments'][k]==value
assert c['harness_arguments']['disable-prefetch']==1 and rc['harness_arguments']['disable-prefetch']==0
assert s['conditioning_cache_hits_per_generation']==[0]*14 and s['cache_audit']=={'requested':None,'enabled_generations':0}
assert q['mse_rgb_8bit']==0 and q['lpips_alex_v0_1']==0
x,y=r['measured']['wall_ms'],s['measured']['wall_ms'];ratio=y['median']/x['median'];passed=y['max']<x['min'];assert not passed
(root/rid/'comparison.json').write_text(json.dumps({'reference':ref,'variant':rid,'matched_model_workload_protocol':True,'engine_settings_change':'disable_prefetch false(default)->true','only_harness_change':'disable-prefetch:0->1','variant_over_reference_median':ratio,'speed_range_rule_pass':passed,'formal_quality_eligible':False},indent=2)+'\n')
b=f'../results/runs/{rid}';p=Path('experiments/jetson-flux-klein-008.md');t=p.read_text();assert rid not in t;t=t.replace('status: partial','status: complete',1).replace('runs: [','runs: ['+rid+', ',1).replace('Status: **partial**; functional attempt passed, full repeated protocol pending.','Status: **complete**; functional attempt and full repeated protocol validated.')
t+=f'''\n## Full repeated protocol\n\n[Summary]({b}/summary.json), [config]({b}/config.json), [environment]({b}/environment.json), [status]({b}/status.json). Completed 15:37:21–15:52:26 UTC. Same pinned workload, engine and snapshot as the attempt above. One context; first generation separate, three warm-ups discarded, ten measured generations.\n\n| Metric | Median, ms | Min–max, ms |\n|---|---:|---:|\n'''
for k in ['wall_ms','text_encode_ms','denoise_ms','vae_decode_ms','other_ms']:
 m=s['measured'][k];t+=f"| {k} | {m['median']:.3f} | {m['min']:.3f}–{m['max']:.3f} |\n"
m=s['monitor_window'];t+=f"\nFirst generation {s['first_run']['wall_ms']:.3f} ms; context creation {s['load']['load_total_s']:.6f} s after file-cache-warming hash checks. Whole-capture system RAM peak {m['ram_peak_gib']:.6f} GiB, swap {m['swap_min_gib']:.6f}–{m['swap_max_gib']:.6f} GiB, {m['samples']} samples.\n\n[Validation]({b}/validation.json) passed: all 14 generations executed four transformer passes, with zero conditioning hits and approximate step caching off. Prefetch-disabled settings were audited against the engine log. All reported RGB hashes agree; retained first and measured PNG hashes were checked independently. [Image diagnostic]({b}/quality-diagnostics.json): reference-identical pixels, MSE 0, infinite PSNR, LPIPS 0 for one development prompt/seed; no formal quality eligibility.\n\nInterpretation: the explicit prefetch-disabled policy completes the repeated protocol without changing this workload's output. The [matched comparison](compare-jetson-distilled-prefetch.md) reports whether its timing distribution improves on the reference. Physical I/O and continuous memory residency are not measured by this run.\n"
p.write_text(t)
cid='compare-jetson-distilled-prefetch';t=f'''---
type: experiment-record
id: {cid}
status: complete
device: jetson-orin-nano
engine: stable-diffusion-cpp
runs: [{ref}, {rid}]
updated: 2026-10-05
---

# Compare: Jetson four-step prefetch enabled versus disabled

## Question

Does disabling engine prefetch improve repeated four-step generation? Week 2 image baselines, RQ1 ([proposal overview](../wiki/project/overview.md)).

## Runs

| Role | Config and summary | Record |
|---|---|---|
| Prefetch-enabled control | [config](../results/runs/{ref}/config.json), [summary](../results/runs/{ref}/summary.json) | [003](jetson-flux-klein-003.md) |
| Prefetch disabled | [config]({b}/config.json), [summary]({b}/summary.json) | [008](jetson-flux-klein-008.md) |

## Matching

| Held fixed | Value |
|---|---|
| Device | Jetson Orin Nano, 25W, four CPU threads |
| Model | Same pinned distilled klein Q4_0 transformer, Qwen3 Q4_K_M and BF16 VAE |
| Workload | Same prompt, seed 0, 512², four Euler/flux2 steps, CFG 1, batch 1 |
| Engine | Same pinned sd.cpp and harness binary; snapshot 31f3be71d07adda2fb122de0ef75a056c27ec7b1 |
| Execution | Eager disk-backed segmented CUDA, mmap off, conditioning and approximate step caching off |
| Protocol | One context; 1 first + 3 discarded warm-ups + 10 measured; no profiler |
| Changed policy | Engine prefetch enabled→disabled |

[Matching receipt]({b}/comparison.json). Sequential captures leave filesystem-cache, swap, background and thermal history uncontrolled; the observed comparison is descriptive, not an isolated causal estimate.

## Results

| Metric | Enabled median (range), ms | Disabled median (range), ms |
|---|---:|---:|
'''
for k in ['wall_ms','text_encode_ms','denoise_ms','vae_decode_ms','other_ms']:
 a,d=r['measured'][k],s['measured'][k];t+=f"| {k} | {a['median']:.3f} ({a['min']:.3f}–{a['max']:.3f}) | {d['median']:.3f} ({d['min']:.3f}–{d['max']:.3f}) |\n"
t+=f"\nObserved median difference {y['median']-x['median']:.3f} ms ({100*(ratio-1):.3f}%); variant/reference ratio {ratio:.6f}. Whole-capture sampled system RAM peak {r['monitor_window']['ram_peak_gib']:.6f}→{m['ram_peak_gib']:.6f} GiB. These peaks include loading and are not per-stage memory estimates.\n\n## Findings and selection\n\nThe saved images are identical, with MSE 0 and LPIPS 0, and all four transformer passes remain. The timing ranges overlap. Variant maximum {y['max']:.3f} ms is not below reference minimum {x['min']:.3f} ms, so the declared speed-selection rule fails. No speed-selected combination is justified by this result.\n\nInterpretation: the small observed median difference does not establish a sustained latency improvement under this protocol. These measurements do not identify physical I/O overlap or prefetch overhead.\n\n## Limits\n\nOne fixed development prompt/seed and sequential captures. No formal quality or held-out evaluation. System memory samples are not stage-aligned. First/measured PNGs and all generation hashes are retained; other raw images were deleted after hashing.\n"
Path('experiments/'+cid+'.md').write_text(t)
p=Path('results/README.md');p.write_text(p.read_text()+f"| `{rid}` | [jetson-flux-klein-008](../experiments/jetson-flux-klein-008.md) | Jetson Orin Nano,25W | four-step prefetch disabled | {y['median']:.3f} | {s['measured']['text_encode_ms']['median']:.3f} / {s['measured']['denoise_ms']['median']:.3f} / {s['measured']['vae_decode_ms']['median']:.3f} | – (system {m['ram_peak_gib']:.3f}) | complete; validated1+3+10 |\n")
p=Path('wiki/experiments.md');lines=p.read_text().splitlines()
for i,line in enumerate(lines):
 if line.startswith('| jetson-flux-klein-008 |'):lines[i]=line.replace('partial; correctness passed, full pending','complete; full protocol validated').replace('[attempt](../results/runs/jetson-flux-klein-008',f'[baseline]({b}/), [attempt](../results/runs/jetson-flux-klein-008')
p.write_text('\n'.join(lines)+'\n'+f'| Jetson distilled prefetch enabled vs disabled | jetson-flux-klein-003, jetson-flux-klein-008 | [record](../experiments/{cid}.md) |\n')
p=Path('wiki/log.md');p.write_text(p.read_text().replace('# Log\n','# Log\n\n## [2026-10-05] experiment | Jetson four-step prefetch-disabled full protocol\n[008](../experiments/jetson-flux-klein-008.md) validated; [matched comparison](../experiments/compare-jetson-distilled-prefetch.md) retains identical pixels and overlapping ranges, failing speed selection.\n',1))
print('Documented; variant/reference ratio',ratio)
