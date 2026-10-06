import json
from pathlib import Path
rid='jetson-flux-klein-007__baseline__20261005-113106';ref='jetson-flux-klein-003__baseline__20261005-105911';root=Path('results/runs');s=json.loads((root/rid/'summary.json').read_text());r=json.loads((root/ref/'summary.json').read_text());c=json.loads((root/rid/'config.json').read_text());rc=json.loads((root/ref/'config.json').read_text());q=json.loads((root/rid/'quality-diagnostics.json').read_text());v=json.loads((root/rid/'validation.json').read_text());assert v['status']=='passed'
for k in ['model','workload','protocol']:assert c[k]==rc[k]
for k,value in rc['harness_arguments'].items():
 if k!='conditioning-cache-size':assert c['harness_arguments'][k]==value
for k,value in rc['engine_settings'].items():
 if k!='conditioning_cache_size':assert c['engine_settings'][k]==value
assert s['conditioning_cache_hits_per_generation']==[0]+[1]*13
assert s['cache_audit']=={'requested':None,'enabled_generations':0}
assert q['mse_rgb_8bit']==0 and q['lpips_alex_v0_1']==0
x,y=r['measured']['wall_ms'],s['measured']['wall_ms'];ratio=x['median']/y['median'];assert y['max']<x['min']
(root/rid/'comparison.json').write_text(json.dumps({'reference':ref,'variant':rid,'matched_model_workload_protocol':True,'only_harness_change':'conditioning-cache-size:0->1','reference_over_variant_median':ratio,'speed_range_rule_pass':True,'formal_quality_eligible':False,'campaign_scope':'No additional four-step combination authorized under the Jetson stopping boundary.'},indent=2)+'\n')
b=f'../results/runs/{rid}';p=Path('experiments/jetson-flux-klein-007.md');t=p.read_text();assert rid not in t;t=t.replace('status: partial','status: complete',1).replace('runs: [','runs: ['+rid+', ',1).replace('Status: **partial**; functional attempt passed, full repeated protocol pending.','Status: **complete**; functional attempt and full repeated protocol validated.')
t+=f'''\n## Full repeated protocol\n\n[Summary]({b}/summary.json), [config]({b}/config.json), [environment]({b}/environment.json), [status]({b}/status.json). Completed 15:31:06–15:37:21 UTC. Same pinned workload, engine and snapshot as the attempt above; one first generation, three discarded warm-ups and ten measured generations in one context.\n\n| Metric | Median, ms | Min–max, ms |\n|---|---:|---:|\n'''
for k in ['wall_ms','text_encode_ms','denoise_ms','vae_decode_ms','other_ms']:
 m=s['measured'][k];t+=f"| {k} | {m['median']:.3f} | {m['min']:.3f}–{m['max']:.3f} |\n"
m=s['monitor_window'];t+=f"\nFirst generation {s['first_run']['wall_ms']:.3f} ms; context creation {s['load']['load_total_s']:.6f} s after file-cache-warming hash checks. Whole-capture system RAM peak {m['ram_peak_gib']:.6f} GiB, swap {m['swap_min_gib']:.6f}–{m['swap_max_gib']:.6f} GiB, {m['samples']} samples.\n\n[Independent validation]({b}/validation.json) passed: all 14 generations executed four transformer passes; conditioning hits were zero initially and one for every later generation. Approximate step caching stayed off. All 14 reported RGB hashes agree; retained first and measured PNG hashes were checked. [Paired diagnostic]({b}/quality-diagnostics.json): reference-identical pixels, MSE 0, infinite PSNR, LPIPS 0 for the fixed development prompt/seed. No formal quality eligibility.\n\nThe measured text stage covers cached conditioning retrieval/setup, not encoder execution. This policy reuses the identical prompt between images; it does not skip denoising steps. The [matched comparison](compare-jetson-distilled-conditioning-reuse.md) reports stage differences and selection. No claim about memory residency or physical storage traffic follows from these timing records alone.\n"
p.write_text(t)
cid='compare-jetson-distilled-conditioning-reuse';t=f'''---
type: experiment-record
id: {cid}
status: complete
device: jetson-orin-nano
engine: stable-diffusion-cpp
runs: [{ref}, {rid}]
updated: 2026-10-05
---

# Compare: Jetson four-step exact conditioning reuse

## Question

Does retaining identical prompt conditioning improve repeated four-step generation? Week 2 image baselines, RQ1 ([proposal overview](../wiki/project/overview.md)).

## Runs

| Role | Config and summary | Record |
|---|---|---|
| No-cache control | [config](../results/runs/{ref}/config.json), [summary](../results/runs/{ref}/summary.json) | [003](jetson-flux-klein-003.md) |
| Conditioning reuse | [config]({b}/config.json), [summary]({b}/summary.json) | [007](jetson-flux-klein-007.md) |

## Matching

| Held fixed | Value |
|---|---|
| Device | Jetson Orin Nano, 25W, four CPU threads |
| Model | Same pinned distilled klein Q4_0 transformer, Qwen3 Q4_K_M and BF16 VAE |
| Workload | Same prompt, seed 0, 512², four Euler/flux2 steps, CFG 1, batch 1 |
| Engine | Same pinned sd.cpp and harness binary, snapshot 31f3be71d07adda2fb122de0ef75a056c27ec7b1 |
| Execution | Eager disk-backed segmented CUDA, prefetch on, mmap off, approximate step cache off |
| Protocol | One context; 1 first + 3 discarded warm-ups + 10 measured; no profiler |
| Changed policy | Conditioning cache capacity 0→1 |

[Matching receipt]({b}/comparison.json). Sequential captures leave filesystem-cache, swap, background and thermal history uncontrolled; observed differences are descriptive, not isolated causal estimates.

## Results

| Metric | No-cache median (range), ms | Conditioning reuse median (range), ms |
|---|---:|---:|
'''
for k in ['wall_ms','text_encode_ms','denoise_ms','vae_decode_ms','other_ms']:
 a,d=r['measured'][k],s['measured'][k];t+=f"| {k} | {a['median']:.3f} ({a['min']:.3f}–{a['max']:.3f}) | {d['median']:.3f} ({d['min']:.3f}–{d['max']:.3f}) |\n"
t+=f"\nObserved median reduction {x['median']-y['median']:.3f} ms ({100*(1-y['median']/x['median']):.3f}%); reference/variant ratio {ratio:.6f}. Whole-capture sampled system RAM peak {r['monitor_window']['ram_peak_gib']:.6f}→{m['ram_peak_gib']:.6f} GiB; these are not per-cache allocation estimates.\n\n## Findings and selection\n\nThe first generation computes conditioning, then all 13 later generations report one hit. All four transformer passes remain; there is no approximate denoising reuse. Measured text-stage time becomes retrieval/setup time. Denoising time also decreases, so the end-to-end difference cannot be explained solely by subtracting encoder time. A change in loading/residency behavior is a hypothesis, not directly measured evidence here.\n\nSaved pixels match the no-cache reference, with MSE 0 and LPIPS 0. Variant maximum {y['max']:.3f} ms is below reference minimum {x['min']:.3f} ms, so the predeclared speed and exact-fidelity diagnostic screens pass. No additional four-step combination is scheduled under the campaign stopping boundary.\n\n## Limits\n\nOne repeated development prompt/seed, not diverse prompts or cache eviction. The result measures reuse across repeated images with identical conditioning, not general per-image latency for new prompts. No formal quality or held-out evaluation. First/measured PNGs and all generation hashes are retained; physical I/O and continuous parameter residency were not measured.\n"
Path('experiments/'+cid+'.md').write_text(t)
p=Path('results/README.md');p.write_text(p.read_text()+f"| `{rid}` | [jetson-flux-klein-007](../experiments/jetson-flux-klein-007.md) | Jetson Orin Nano,25W | four-step exact conditioning reuse | {y['median']:.3f} | {s['measured']['text_encode_ms']['median']:.3f} / {s['measured']['denoise_ms']['median']:.3f} / {s['measured']['vae_decode_ms']['median']:.3f} | – (system {m['ram_peak_gib']:.3f}) | complete; validated1+3+10 |\n")
p=Path('wiki/experiments.md');lines=p.read_text().splitlines()
for i,line in enumerate(lines):
 if line.startswith('| jetson-flux-klein-007 |'):lines[i]=line.replace('partial; correctness passed, full pending','complete; full protocol validated').replace('[attempt](../results/runs/jetson-flux-klein-007',f'[baseline]({b}/), [attempt](../results/runs/jetson-flux-klein-007')
p.write_text('\n'.join(lines)+'\n'+f'| Jetson distilled exact conditioning reuse | jetson-flux-klein-003, jetson-flux-klein-007 | [record](../experiments/{cid}.md) |\n')
p=Path('wiki/log.md');p.write_text(p.read_text().replace('# Log\n','# Log\n\n## [2026-10-05] experiment | Jetson four-step exact conditioning full protocol\n[007](../experiments/jetson-flux-klein-007.md) validates exact prompt reuse; [comparison](../experiments/compare-jetson-distilled-conditioning-reuse.md) passes diagnostic speed/fidelity selection; no new combination scheduled.\n',1))
print('Documented; ratio',ratio,'median reduction',100*(1-y['median']/x['median']))
