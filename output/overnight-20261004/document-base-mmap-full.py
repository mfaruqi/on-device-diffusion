import json,tarfile
from pathlib import Path
root=Path('results/runs');rid='jetson-flux-klein-base-006__baseline__20261005-131024';ref='jetson-flux-klein-base-001__repeat__20261005-042014';p=root/rid;s=json.loads((p/'summary.json').read_text());r=json.loads((root/ref/'summary.json').read_text());c=json.loads((p/'config.json').read_text());rc=json.loads((root/ref/'config.json').read_text());v=json.loads((p/'validation.json').read_text());q=json.loads((p/'quality-diagnostics.json').read_text())
for k in ['model','workload','protocol']:assert c[k]==rc[k]
a=c['harness_arguments'].copy();b=rc['harness_arguments'].copy();assert a.pop('mmap')==1 and b.pop('mmap')==0 and a==b
assert v['same_binary'] and v['reference_rgb_equal'] and v['transformer_passes']==[100]*14 and v['conditioning_hits']==[0]*14
assert s['cache_audit']['enabled_generations']==0 and len(s['engine_audit']['mmap_io']['confirmed_files'])==3
w=s['measured']['wall_ms'];rw=r['measured']['wall_ms'];speed=w['max']<rw['min'];assert not speed
(p/'comparison.json').write_text(json.dumps({'reference':ref,'variant':rid,'matched_model_workload_protocol_binary':True,'only_harness_argument_difference':'mmap:0 to1','speed_range_rule_passed':speed,'median_delta_ms':w['median']-rw['median'],'median_delta_percent':100*(w['median']/rw['median']-1),'reference_over_variant_ratio':rw['median']/w['median'],'exact_retained_rgb':True,'formal_quality_eligible':False},indent=2)+'\n')
b='../results/runs/'+rid;m=s['monitor_window'];eid='jetson-flux-klein-base-006';f=Path('experiments/'+eid+'.md');t=f.read_text();assert rid not in t;t=t.replace('status: partial','status: complete',1).replace('runs: [','runs: ['+rid+', ',1).replace('Status: **partial**; two-generation correctness gate complete, full repeated protocol running separately.','Status: **complete**; correctness gate and full repeated protocol independently validated.').replace('Complete the unchanged full protocol in `configs/jetson-flux-klein-base-q4-512-mmap.json`.','None scheduled; the authorized Jetson batch is complete.')
t+=f'''\n## Full repeated protocol\n\n[Summary]({b}/summary.json), [config]({b}/config.json), [environment]({b}/environment.json), [status]({b}/status.json). Completed 17:10:24–18:27:35 UTC in the same isolated snapshot as the smoke test. One context; one first generation, three discarded warm-ups, ten measured images; no profiler.\n\n| Metric | Median, ms | Min–max, ms |\n|---|---:|---:|\n'''
for k in ['wall_ms','text_encode_ms','denoise_ms','vae_decode_ms','other_ms']:
 x=s['measured'][k];t+=f"| {k} | {x['median']:.3f} | {x['min']:.3f}–{x['max']:.3f} |\n"
t+=f'''\nFirst generation {s['first_run']['wall_ms']:.3f} ms; context creation {s['load']['load_total_s']:.6f} s after file-cache-warming hash checks. Whole-capture RAM peak {m['ram_peak_gib']:.6f} GiB, swap {m['swap_min_gib']:.6f}–{m['swap_max_gib']:.6f} GiB, {m['samples']} samples. Swap grew during this capture; no per-stage memory attribution is available.\n\n[Validation]({b}/validation.json) passed all 14 generations: 50 scheduler steps and 100 actual CFG transformer passes each, no denoising cache and zero conditioning hits. Three component file mappings confirmed without fallback. All 14 reported RGB hashes agree; the two retained PNGs match their recorded hashes. [Paired diagnostic]({b}/quality-diagnostics.json): retained measured pixels match no-cache, PSNR infinity and LPIPS 0. One development prompt/seed, no formal quality eligibility.\n\n[Matched mmap comparison](compare-jetson-base-mmap.md) records the overlapping latency ranges and failed diagnostic speed-selection rule. This configuration is not evidence for a speed-selected mmap combination.\n''';f.write_text(t)
cid='compare-jetson-base-mmap';t=f'''---
type: experiment-record
id: {cid}
status: complete
device: jetson-orin-nano
engine: stable-diffusion-cpp
runs: [{ref}, {rid}]
updated: 2026-10-05
---

# Compare: Jetson Base ordinary weight-file reads and mmap

## Question

Does enabling memory-mapped weight-file I/O improve the fixed Base workload? Week 2 image-engine baseline evidence for RQ1 ([proposal overview](../wiki/project/overview.md)).

## Runs

| Policy | Config and summary | Record |
|---|---|---|
| Ordinary reads | [config](../results/runs/{ref}/config.json), [summary](../results/runs/{ref}/summary.json) | [Base001](jetson-flux-klein-base-001.md) |
| Mmap | [config]({b}/config.json), [summary]({b}/summary.json) | [Base006](jetson-flux-klein-base-006.md) |

## Matching

| Held fixed | Value |
|---|---|
| Device | Jetson Orin Nano, 25W, four CPU threads |
| Model | Identical pinned Base Q4_0 transformer, Qwen3 Q4_K_M encoder, BF16 VAE |
| Workload | Same prompt, seed 0, 512², 50 Euler/flux2 steps, CFG 4, batch 1 |
| Engine | Same pinned sd.cpp and exact harness binary hash |
| Execution | Eager disk-backed segmented CUDA, prefetch enabled, no denoising or conditioning reuse |
| Protocol | One context; 1 first + 3 discarded warm-ups + 10 measured; no profiler |
| Variable | mmap disabled versus enabled; sole differing harness argument |

Sequential captures leave filesystem-cache, background, swap and thermal history uncontrolled. Ratios describe these captures, not an isolated causal estimate. Hash checks warm file cache before context creation.

## Results

| Metric | Ordinary reads | Mmap |
|---|---:|---:|
'''
for k in ['wall_ms','text_encode_ms','denoise_ms','vae_decode_ms']:
 a=r['measured'][k];z=s['measured'][k];t+=f"| {k}, median (min–max) | {a['median']:.3f} ({a['min']:.3f}–{a['max']:.3f}) | {z['median']:.3f} ({z['min']:.3f}–{z['max']:.3f}) |\n"
t+=f"| Whole-capture RAM peak, GiB | {r['monitor_window']['ram_peak_gib']:.6f} | {m['ram_peak_gib']:.6f} |\n| Whole-capture swap min–max, GiB | {r['monitor_window']['swap_min_gib']:.6f}–{r['monitor_window']['swap_max_gib']:.6f} | {m['swap_min_gib']:.6f}–{m['swap_max_gib']:.6f} |\n"
t+=f'''\nMmap's observed median differs by {w['median']-rw['median']:+.3f} ms ({100*(w['median']/rw['median']-1):+.3f}%). Reference/mmap median ratio {rw['median']/w['median']:.6f}.\n\n## Findings and selection\n\nLatency ranges overlap: mmap's maximum {w['max']:.3f} ms is not below reference's minimum {rw['min']:.3f} ms. The predeclared speed-range screen fails ([receipt]({b}/comparison.json)). No speed-selected mmap combination follows.\n\nBoth execute 100 CFG transformer passes per image. Mmap confirms mappings for three component files and retains reference-identical measured RGB pixels; [paired diagnostic]({b}/quality-diagnostics.json) reports PSNR infinity and LPIPS 0. Formal quality eligibility remains false.\n\nInterpretation: this capture does not establish a latency benefit from mmap for the Base workload. The sampled memory values describe the whole system and do not identify which mappings or stages caused swap growth.\n\n## Limits\n\nOne fixed development prompt/seed, no held-out quality test. File mappings do not prove physical storage traffic, GPU zero-copy or continuous weight residency. Callback timings include loading and synchronization rather than isolated GPU kernel time. Memory samples include context loading and all generations; no allocator/stage alignment. Sequential-run confounds prevent a causal claim from the small median difference.\n''';Path('experiments/'+cid+'.md').write_text(t)
f=Path('results/README.md');f.write_text(f.read_text()+f"| `{rid}` | [{eid}](../experiments/{eid}.md) | Jetson Orin Nano,25W | Base mmap | {w['median']:.3f} | {s['measured']['text_encode_ms']['median']:.3f} / {s['measured']['denoise_ms']['median']:.3f} / {s['measured']['vae_decode_ms']['median']:.3f} | – (system {m['ram_peak_gib']:.3f}) | complete; validated1+3+10 |\n")
f=Path('wiki/experiments.md');lines=f.read_text().splitlines()
for i,line in enumerate(lines):
 if line.startswith('| '+eid+' |'):lines[i]=line.replace('partial; smoke validated, full running','complete; full protocol validated').replace('[attempt](',f'[baseline]({b}/), [attempt](',1)
f.write_text('\n'.join(lines)+f'\n| Jetson Base ordinary weight-file reads vs mmap | Base001, Base006 | [record](../experiments/{cid}.md) |\n')
f=Path('wiki/log.md');t=f.read_text();i=t.index('\n## [');t=t[:i]+f'\n## [2026-10-05] experiment | Jetson Base mmap full protocol validated\n\n[Base006](../experiments/{eid}.md) completed; [comparison](../experiments/{cid}.md) retains identical pixels, overlapping timing ranges and failed speed-selection rule.\n'+t[i:];f.write_text(t)
with tarfile.open('/tmp/base-mmap-full-upload.tar.gz','w:gz') as t:t.add(p,arcname=str(p))
f=Path('output/overnight-20261004/verify-completed-1708.py');t=f.read_text();i=t.index('names=');j=t.index('\napi=',i);Path('output/overnight-20261004/verify-mmap-full.py').write_text(t[:i]+'names='+repr([rid])+t[j:])
print('record, comparison, index and registry updated; upload archive prepared')
