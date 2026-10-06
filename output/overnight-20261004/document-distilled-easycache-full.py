import json
from pathlib import Path
rid='jetson-flux-klein-006__baseline__20261005-111428';ref='jetson-flux-klein-003__baseline__20261005-105911';root=Path('results/runs');s=json.loads((root/rid/'summary.json').read_text());r=json.loads((root/ref/'summary.json').read_text());c=json.loads((root/rid/'config.json').read_text());rc=json.loads((root/ref/'config.json').read_text());q=json.loads((root/rid/'quality-diagnostics.json').read_text());assert json.loads((root/rid/'validation.json').read_text())['status']=='passed'
for k in ['model','workload','protocol','engine_settings']:assert c[k]==rc[k]
for k,v in rc['harness_arguments'].items():assert c['harness_arguments'][k]==v
assert set(c['harness_arguments'])-set(rc['harness_arguments'])=={'cache-mode','cache-threshold','cache-start','cache-end'}
assert s['cache_audit']['enabled_generations']==14 and s['cache_audit']['steps_skipped']==[0]*14
assert q['mse_rgb_8bit']==0 and q['lpips_alex_v0_1']==0
x,y=r['measured']['wall_ms'],s['measured']['wall_ms'];ratio=y['median']/x['median'];passed=y['max']<x['min'];assert not passed
receipt={'reference':ref,'variant':rid,'matched_model_workload_protocol_engine_settings':True,'only_changed_policy':'EasyCache .2/.15/.95 enabled','variant_over_reference_median':ratio,'speed_range_rule_pass':passed,'steps_skipped':[0]*14,'formal_quality_eligible':False}
(root/rid/'comparison.json').write_text(json.dumps(receipt,indent=2)+'\n')
b=f'../results/runs/{rid}';p=Path('experiments/jetson-flux-klein-006.md');t=p.read_text();assert rid not in t;t=t.replace('status: partial','status: complete',1).replace('runs: [','runs: ['+rid+', ',1).replace('Status: **partial**; functional attempt passed, full repeated protocol pending.','Status: **complete**; functional attempt and full repeated protocol validated.')
t+=f'''\n## Full repeated protocol\n\n[Summary]({b}/summary.json), [config]({b}/config.json), [environment]({b}/environment.json), [status]({b}/status.json). Completed 15:14:28–15:31:05 UTC. Same pinned workload, engine and snapshot as the attempt above; one first generation, three discarded warm-ups and ten measured generations in one context.\n\n| Metric | Median, ms | Min–max, ms |\n|---|---:|---:|\n'''
for k in ['wall_ms','text_encode_ms','denoise_ms','vae_decode_ms','other_ms']:
 m=s['measured'][k];t+=f"| {k} | {m['median']:.3f} | {m['min']:.3f}–{m['max']:.3f} |\n"
m=s['monitor_window'];t+=f"\nFirst generation {s['first_run']['wall_ms']:.3f} ms; context creation {s['load']['load_total_s']:.6f} s after file-cache-warming hash checks. Whole-capture system RAM peak {m['ram_peak_gib']:.6f} GiB, swap {m['swap_min_gib']:.6f}–{m['swap_max_gib']:.6f} GiB, {m['samples']} samples.\n\n[Validation]({b}/validation.json) passed: all 14 generations completed four transformer passes, zero conditioning hits, and **zero skipped steps despite EasyCache being enabled**. All 14 reported RGB hashes agree; saved first and measured PNG hashes were checked independently. [Image diagnostic]({b}/quality-diagnostics.json): identical pixels, MSE 0, infinite PSNR and LPIPS 0 against the full control, for one development prompt/seed only. No formal quality eligibility.\n\nInterpretation: at these settings, the four-step run did not reuse any denoising steps. Enabled caching alone therefore does not demonstrate avoided transformer work. The [matched comparison](compare-jetson-distilled-easycache.md) reports performance and selection separately.\n"
p.write_text(t)
cid='compare-jetson-distilled-easycache';t=f'''---
type: experiment-record
id: {cid}
status: complete
device: jetson-orin-nano
engine: stable-diffusion-cpp
runs: [{ref}, {rid}]
updated: 2026-10-05
---

# Compare: Jetson four-step klein EasyCache

## Question

Does EasyCache help the original four-step workload? Week 2 image baselines; RQ1 asks “Which diffusion optimization choices transfer across devices and workloads?” ([proposal overview](../wiki/project/overview.md)).

## Runs

| Role | Config and summary | Record |
|---|---|---|
| No-cache control | [config](../results/runs/{ref}/config.json), [summary](../results/runs/{ref}/summary.json) | [003](jetson-flux-klein-003.md) |
| EasyCache | [config]({b}/config.json), [summary]({b}/summary.json) | [006](jetson-flux-klein-006.md) |

## Matching

| Held fixed | Value |
|---|---|
| Device | Jetson Orin Nano, 25W, four CPU threads |
| Model | Same pinned distilled klein Q4_0 transformer, Qwen3 Q4_K_M and BF16 VAE |
| Workload | Same prompt, seed 0, 512², four Euler/flux2 steps, CFG 1, batch 1 |
| Engine | Same sd.cpp commit and harness binary; snapshot 31f3be71d07adda2fb122de0ef75a056c27ec7b1 |
| Execution | Eager disk-backed segmented CUDA, prefetch on, mmap off, conditioning cache 0 |
| Protocol | One context; 1 first + 3 discarded warm-ups + 10 measured; no profiler |
| Changed policy | EasyCache enabled with threshold 0.2, start 0.15, end 0.95 |

[Matching and selection receipt]({b}/comparison.json). Sequential runs do not control filesystem-cache, swap, background or thermal history. Both hash checks warmed files. Initial noise uses the same engine and seed, not a cross-engine RNG assumption.

## Results

| Metric | No-cache median (range), ms | EasyCache median (range), ms |
|---|---:|---:|
'''
for k in ['wall_ms','text_encode_ms','denoise_ms','vae_decode_ms','other_ms']:
 a,d=r['measured'][k],s['measured'][k];t+=f"| {k} | {a['median']:.3f} ({a['min']:.3f}–{a['max']:.3f}) | {d['median']:.3f} ({d['min']:.3f}–{d['max']:.3f}) |\n"
t+=f"\nObserved median difference: +{y['median']-x['median']:.3f} ms, or +{100*(ratio-1):.3f}%; variant/reference ratio {ratio:.6f}. Whole-capture sampled system RAM peaks: {r['monitor_window']['ram_peak_gib']:.6f} GiB control and {m['ram_peak_gib']:.6f} GiB EasyCache; these are not cache-allocation estimates.\n\n## Findings and selection\n\nEasyCache was enabled in all 14 generations but skipped zero steps throughout; every generation still executed all four transformer passes. Saved output pixels match the full no-cache control, with MSE 0 and LPIPS 0. Observed median latency was higher, and the measured ranges overlap. The declared speed rule fails: variant maximum {y['max']:.3f} ms is not below reference minimum {x['min']:.3f} ms. This four-step option does not qualify for a speed-selected combination.\n\nInterpretation: the configuration provided no observed transformer-call reduction. This comparison does not isolate cache-management overhead from the sequential-run confounds. It does not establish that every cache threshold, prompt or schedule would behave identically.\n\n## Limits\n\nOne fixed development prompt/seed, no held-out set or formal quality approval. Retained first/measured images and per-generation hashes provide repeatability evidence; all raw generation images are not retained. No physical-I/O or SM-utilization attribution is made.\n"
Path('experiments/'+cid+'.md').write_text(t)
p=Path('results/README.md');p.write_text(p.read_text()+f"| `{rid}` | [jetson-flux-klein-006](../experiments/jetson-flux-klein-006.md) | Jetson Orin Nano,25W | four-step EasyCache | {y['median']:.3f} | {s['measured']['text_encode_ms']['median']:.3f} / {s['measured']['denoise_ms']['median']:.3f} / {s['measured']['vae_decode_ms']['median']:.3f} | – (system {m['ram_peak_gib']:.3f}) | complete; validated1+3+10, zero steps skipped |\n")
p=Path('wiki/experiments.md');t=p.read_text();lines=t.splitlines()
for i,line in enumerate(lines):
 if line.startswith('| jetson-flux-klein-006 |'):lines[i]=line.replace('partial; correctness passed, full pending','complete; full protocol validated').replace('[attempt](../results/runs/jetson-flux-klein-006',f'[baseline]({b}/), [attempt](../results/runs/jetson-flux-klein-006')
t='\n'.join(lines)+'\n'+f'| Jetson distilled no-cache vs EasyCache | jetson-flux-klein-003, jetson-flux-klein-006 | [record](../experiments/{cid}.md) |\n';p.write_text(t)
p=Path('wiki/log.md');p.write_text(p.read_text().replace('# Log\n','# Log\n\n## [2026-10-05] experiment | Jetson four-step EasyCache full protocol and comparison\n[006](../experiments/jetson-flux-klein-006.md) validates enabled caching with zero skipped steps; [matched comparison](../experiments/compare-jetson-distilled-easycache.md) fails the declared speed-selection rule.\n',1))
print('Documented full EasyCache and comparison; observed median change',100*(ratio-1),'percent')
