import json
from pathlib import Path
root=Path('results/runs');rid='jetson-flux-klein-base-007__baseline__20261005-121955';refs={'No cache':'jetson-flux-klein-base-001__repeat__20261005-042014','EasyCache':'jetson-flux-klein-base-002__baseline__20261005-024556','Conditioning reuse':'jetson-flux-klein-base-005__baseline__20261005-093239','Combined':rid};summ={k:json.loads((root/n/'summary.json').read_text()) for k,n in refs.items()};s=summ['Combined'];c=json.loads((root/rid/'config.json').read_text());q=json.loads((root/rid/'quality-diagnostics.json').read_text());assert json.loads((root/rid/'validation.json').read_text())['status']=='passed'
for n in refs.values():
 rc=json.loads((root/n/'config.json').read_text())
 for k in ['model','workload','protocol']:assert c[k]==rc[k]
 assert json.loads((root/n/'environment.json').read_text())['binary_sha256']==json.loads((root/rid/'environment.json').read_text())['binary_sha256']
assert s['cache_audit']['steps_skipped']==[25]*14 and s['conditioning_cache_hits_per_generation']==[0]+[2]*13
selection={k:s['measured']['wall_ms']['max']<v['measured']['wall_ms']['min'] for k,v in summ.items() if k!='Combined'};assert all(selection.values())
(root/rid/'comparison.json').write_text(json.dumps({'runs':refs,'matched_model_workload_protocol_binary':True,'speed_range_rule_vs':selection,'quality':'Same saved RGB pixels as EasyCache-only; paired no-cache PSNR/LPIPS in quality-diagnostics.json.','formal_quality_eligible':False},indent=2)+'\n')
b=f'../results/runs/{rid}';p=Path('experiments/jetson-flux-klein-base-007.md');t=p.read_text();assert rid not in t;t=t.replace('status: partial','status: complete',1).replace('runs: [','runs: ['+rid+', ',1).replace('Status: **partial**; two-generation correctness gate passed, full protocol started separately.','Status: **complete**; correctness gate and full repeated protocol independently validated.')
t+=f'''\n## Full repeated protocol\n\n[Summary]({b}/summary.json), [config]({b}/config.json), [environment]({b}/environment.json), [status]({b}/status.json). Completed 16:19:55–16:55:35 UTC in the same isolated snapshot and with the same policies as the smoke test. One context; one first generation, three discarded warm-ups, ten measured generations.\n\n| Metric | Median, ms | Min–max, ms |\n|---|---:|---:|\n'''
for k in ['wall_ms','text_encode_ms','denoise_ms','vae_decode_ms','other_ms']:
 m=s['measured'][k];t+=f"| {k} | {m['median']:.3f} | {m['min']:.3f}–{m['max']:.3f} |\n"
m=s['monitor_window'];t+=f"\nFirst generation {s['first_run']['wall_ms']:.3f} ms; context creation {s['load']['load_total_s']:.6f} s after file-cache-warming hash checks. Whole-capture RAM peak {m['ram_peak_gib']:.6f} GiB, swap {m['swap_min_gib']:.6f}–{m['swap_max_gib']:.6f} GiB, {m['samples']} samples.\n\n[Validation]({b}/validation.json) passed: 50 scheduler steps each, 25 skipped denoising steps and 50 actual CFG transformer passes per image; conditioning hits 0 then 2 for all later generations. All 14 reported RGB hashes agree. [Paired diagnostic]({b}/quality-diagnostics.json): PSNR {q['psnr_db']:.9f} dB and LPIPS {q['lpips_alex_v0_1']:.9f} against no-cache. The retained measured image matches the EasyCache-only image; approximate execution has not become exact relative to no-cache. No formal quality eligibility.\n\n[Matched comparison](compare-jetson-base-combined-reuse.md) evaluates the combination against the reference and both individual policies. This fixed-prompt protocol measures exact conditioning reuse across repeated images, not fresh prompts.\n"
p.write_text(t)
cid='compare-jetson-base-combined-reuse';t=f'''---
type: experiment-record
id: {cid}
status: complete
device: jetson-orin-nano
engine: stable-diffusion-cpp
runs: [{', '.join(refs.values())}]
updated: 2026-10-05
---

# Compare: Jetson Base individual and combined reuse

## Question

Does combining exact conditioning reuse with EasyCache improve on each separately? Week 2 baseline evidence for RQ1 and preparation for RQ3's joint-policy evaluation ([proposal overview](../wiki/project/overview.md)); this is a fixed engine configuration comparison, not a planner evaluation.

## Runs

| Policy | Config and summary | Record |
|---|---|---|
'''
for k,n in refs.items():
 eid=n.split('__')[0];t+=f'| {k} | [config](../results/runs/{n}/config.json), [summary](../results/runs/{n}/summary.json) | [{eid}](../experiments/{eid}.md) |\n'
t+='''
## Matching

| Held fixed | Value |
|---|---|
| Device | Jetson Orin Nano, 25W, four CPU threads |
| Model | Same pinned Base Q4_0 transformer, Qwen3 Q4_K_M encoder and BF16 VAE |
| Workload | Identical prompt and seed 0, 512², 50 Euler/flux2 steps, CFG 4, batch 1 |
| Engine | Same pinned sd.cpp and harness binary |
| Residency | Eager disk-backed segmented CUDA, prefetch enabled, mmap disabled |
| Protocol | One context; 1 first + 3 discarded warm-ups + 10 measured; no profiler |
| Policies | Conditioning capacity 0 or 2; EasyCache off or threshold0.2/start0.15/end0.95 |

Sequential captures leave filesystem-cache, background, swap and thermal history uncontrolled. New validator snapshots explicitly support the labelled options; the inference binary is identical. Ratios are descriptive rather than isolated causal estimates.

## Results

| Policy | Wall median (min–max), ms | Text median, ms | Denoise median, ms | VAE median, ms | Whole-capture RAM peak, GiB |
|---|---:|---:|---:|---:|---:|
'''
for k,v in summ.items():
 w=v['measured']['wall_ms'];t+=f"| {k} | {w['median']:.3f} ({w['min']:.3f}–{w['max']:.3f}) | {v['measured']['text_encode_ms']['median']:.3f} | {v['measured']['denoise_ms']['median']:.3f} | {v['measured']['vae_decode_ms']['median']:.3f} | {v['monitor_window']['ram_peak_gib']:.6f} |\n"
y=s['measured']['wall_ms'];t+='\n| Compared with | Observed median reduction | Reference/combined ratio |\n|---|---:|---:|\n'
for k,v in summ.items():
 if k=='Combined':continue
 x=v['measured']['wall_ms'];t+=f"| {k} | {100*(1-y['median']/x['median']):.3f}% | {x['median']/y['median']:.6f} |\n"
t+=f'''\n## Findings and selection\n\nThe combined maximum {y['max']:.3f} ms is below each comparator's minimum, passing the predeclared speed-range screen against both individual policies and no-cache. [Receipt]({b}/comparison.json). All generations retain both policies' expected behavior: positive/negative conditioning reused after the first image; 25 of 50 denoising steps skipped, 50 actual CFG transformer passes instead of 100.\n\nThe saved combined output matches EasyCache-only pixels. Against no-cache, [diagnostics]({b}/quality-diagnostics.json) remain PSNR {q['psnr_db']:.9f} dB and LPIPS {q['lpips_alex_v0_1']:.9f}; exact conditioning adds no observed pixel change to this approximate output. Formal quality eligibility remains false.\n\nInterpretation: the combination improves observed latency on this repeated-prompt workload. This does not prove a particular I/O/residency mechanism or a planner advantage, and separate median savings must not be assumed additive.\n\n## Limits\n\nOne fixed development prompt/seed, no held-out quality test. Cache-hit behavior depends on repeated identical conditioning. Memory is whole-capture sampled RAM, not per-cache allocation or stage-aligned residency. The campaign stopping boundary permits no further combinations.\n'''
Path('experiments/'+cid+'.md').write_text(t)
p=Path('results/README.md');p.write_text(p.read_text()+f"| `{rid}` | [jetson-flux-klein-base-007](../experiments/jetson-flux-klein-base-007.md) | Jetson Orin Nano,25W | Base EasyCache + conditioning reuse | {y['median']:.3f} | {s['measured']['text_encode_ms']['median']:.3f} / {s['measured']['denoise_ms']['median']:.3f} / {s['measured']['vae_decode_ms']['median']:.3f} | – (system {m['ram_peak_gib']:.3f}) | complete; validated1+3+10 |\n")
p=Path('wiki/experiments.md');lines=p.read_text().splitlines()
for i,line in enumerate(lines):
 if line.startswith('| jetson-flux-klein-base-007 |'):lines[i]=line.replace('partial; correctness passed, full running','complete; full protocol validated').replace('[attempt](../results/runs/jetson-flux-klein-base-007',f'[baseline]({b}/), [attempt](../results/runs/jetson-flux-klein-base-007')
p.write_text('\n'.join(lines)+'\n'+f'| Jetson Base combined versus individual reuse | Base001, Base002, Base005, Base007 | [record](../experiments/{cid}.md) |\n')
p=Path('wiki/log.md');p.write_text(p.read_text().replace('# Log\n','# Log\n\n## [2026-10-05] experiment | Jetson Base combined reuse full protocol validated\n[Base007](../experiments/jetson-flux-klein-base-007.md) complete; [matched comparison](../experiments/compare-jetson-base-combined-reuse.md) passes diagnostic speed selection against each individual policy, with EasyCache-only pixels.\n',1))
print('documented combined full and four-policy comparison')
