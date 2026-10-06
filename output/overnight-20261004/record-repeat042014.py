import json
from pathlib import Path
from datetime import datetime,timezone
rid='jetson-flux-klein-base-001__repeat__20261005-042014';cid='jetson-flux-klein-base-002__baseline__20261005-024556';base=Path('results/runs');s=json.loads((base/rid/'summary.json').read_text());c=json.loads((base/cid/'summary.json').read_text())
a=json.loads((base/rid/'config.json').read_text());b=json.loads((base/cid/'config.json').read_text())
for k in ['workload','model','protocol','engine','engine_settings']:assert a[k]==b[k]
assert {k:v for k,v in a['optimizations'].items() if k!='step_cache'}=={k:v for k,v in b['optimizations'].items() if k!='step_cache'}
assert {k:v for k,v in a['harness_arguments'].items() if not k.startswith('cache-')}=={k:v for k,v in b['harness_arguments'].items() if not k.startswith('cache-')}
ratio=s['measured']['wall_ms']['median']/c['measured']['wall_ms']['median'];qual=c['measured']['wall_ms']['max']<s['measured']['wall_ms']['min'];assert qual
receipt={'status':'passed','reference':rid,'variant':cid,'held_fixed':['model component hashes/revisions','engine commit and binary SHA256','workload','protocol','engine settings','non-cache harness arguments'],'variable':'EasyCache threshold0.2,start0.15,end0.95','reference_to_variant_median_ratio':ratio,'range_rule_passed':qual,'diagnostic_screen':'PSNR/LPIPS recorded for approximate cache; identical reference and variant pixels to previously evaluated pair verified in run validation receipts.','selection':'EasyCache qualifies as a candidate for later combinations under the campaign diagnostic rule; no second qualified Base option yet.','quality_eligible':False,'limitations':['one development prompt/seed','sequential captures; uncontrolled page cache, swap and background state','observed ratio is not an isolated causal estimate']}
(base/rid/'cache-comparison.json').write_text(json.dumps(receipt,indent=2)+'\n')
p=Path('experiments/jetson-flux-klein-base-001.md');t=p.read_text().replace('runs: [','runs: ['+rid+', ',1);section=f'''## Repeated reference on the cache-capable harness

[Run](../results/runs/{rid}/), [summary](../results/runs/{rid}/summary.json), [validation](../results/runs/{rid}/repeat-validation.json). The unchanged no-cache config completed one first generation, three discarded warm-ups and ten measured generations in one context. Binary SHA256 is2d66fa04053c02b441d61a84d24412537a16f9099d6cb31b80dcca3c8cc1f896; earlier reference outputs remain preserved.

| Metric | Median, ms | Min–max, ms |
|---|---:|---:|
'''
for k in ['wall_ms','text_encode_ms','denoise_ms','denoise_step_0_ms','denoise_step_1_ms','vae_decode_ms','other_ms']:
 v=s['measured'][k];section+=f"| {k} | {v['median']:.3f} | {v['min']:.3f}–{v['max']:.3f} |\n"
m=s['monitor_window'];section+=f'''
First generation:{s['first_run']['wall_ms']:.3f}ms. Context creation:{s['load']['load_total_s']:.3f}s after file hashing. Whole-capture system RAM peak:{m['ram_peak_gib']:.3f}GiB; swap range:{m['swap_min_gib']:.3f}–{m['swap_max_gib']:.3f}GiB from{m['samples']}one-second samples. These include loading, not per-stage memory peaks.

All fourteen generations completed fifty callback steps and one hundred transformer executions, with no cache enabled. Phase order, settings, summary aggregates and PNG hashes passed independent validation. Saved RGB pixels equal the previously inspected no-cache reference. This establishes a repeated no-cache measurement on the cache-capable binary; policy comparison is in the [comparison record](compare-jetson-base-easycache.md).

[W&B repeat](https://wandb.ai/mfaruqi-purdue-university/on-device-diffusion/runs/{rid}).

''';t=t.replace('## Interpretation\n',section+'## Interpretation\n',1).replace('- Complete the same-harness no-cache repeat of configs/jetson-flux-klein-base-q4-512-disk.json before selecting combinations.','- Test configs/jetson-flux-klein-base-q4-512-disk-lazy-smoke.json as a separate loading-policy variant.');p.write_text(t)
p=Path('experiments/compare-jetson-base-easycache.md');t=p.read_text().replace('runs: [','runs: ['+rid+', ',1).replace('The repeated runs provide descriptive performance evidence, with a harness-build and sequential-capture confound. The same-binary attempts provide a separate functional control, not a repeated latency control.','A completed same-binary full no-cache repeat removes the earlier harness-build confound. Sequential-capture conditions remain uncontrolled, so the observed ratio is descriptive rather than an isolated causal estimate.').replace('## Repeated-run results','## Earlier repeated runs, different harness builds')
t=t.replace('## Limits and next evidence',f'''## Same-binary repeated comparison

[No-cache repeat](../results/runs/{rid}/config.json), [summary](../results/runs/{rid}/summary.json), [matching receipt](../results/runs/{rid}/cache-comparison.json). This repeat and the full EasyCache run use the same binary SHA2562d66fa04053c02b441d61a84d24412537a16f9099d6cb31b80dcca3c8cc1f896. Component pins, workload, protocol1+3+10, placement, segmentation, prefetch and all non-cache arguments match exactly. Only the step-cache policy differs.

| Metric | No-cache median [min–max], ms | Cache median [min–max], ms | Cache minus reference, ms |
|---|---:|---:|---:|
'''+''.join(f"| {k} | {s['measured'][k]['median']:.3f} [{s['measured'][k]['min']:.3f}–{s['measured'][k]['max']:.3f}] | {c['measured'][k]['median']:.3f} [{c['measured'][k]['min']:.3f}–{c['measured'][k]['max']:.3f}] | {c['measured'][k]['median']-s['measured'][k]['median']:.3f} |\n" for k in ['wall_ms','text_encode_ms','denoise_ms','vae_decode_ms'])+f'''
Observed median ratio:{ratio:.3f}×. EasyCache's measured maximum remains below the no-cache minimum. Both run images match their corresponding images in the existing PSNR/LPIPS evaluation, so the diagnostic pair is unchanged. Under the campaign rule for approximate caches (record diagnostic metrics, do not impose the exact-option PSNR threshold), EasyCache now qualifies as a candidate for later combinations. This is not formal quality approval; no second independently qualified Base option exists yet.

Whole-capture one-second RAM peaks:no-cache{m['ram_peak_gib']:.3f}GiB, cache{c['monitor_window']['ram_peak_gib']:.3f}GiB. This small difference does not isolate cache-allocation cost. Sequential filesystem-cache, swap and background-process conditions remain confounds; the comparison is qualitative about attribution despite the large observed separation. The reduction from100to50transformer executions per generation is direct evidence of work skipped.

## Limits and next evidence''')
t=t.replace('One prompt/seed, no held-out evaluation, different full-run harness builds and uncontrolled sequential capture conditions. No formal quality eligibility or combined-policy selection. A same-harness no-cache full repeat is the next matching check; separate profiling will inspect where the remaining latency occurs.','One prompt/seed, no held-out evaluation and uncontrolled sequential capture conditions. The new full comparison matches harness builds, while earlier builds remain explicitly separated. No formal quality eligibility or combined-policy result. Test lazy loading independently before considering a combined cache/residency configuration.');p.write_text(t)
p=Path('results/README.md');t=p.read_text().replace('## A100 Base reference',f"| `{rid}` | [jetson-flux-klein-base-001](../experiments/jetson-flux-klein-base-001.md) | Jetson Orin Nano,25W | jetson-flux-klein-base-q4-512-disk | {s['measured']['wall_ms']['median']:.1f} | {s['measured']['text_encode_ms']['median']:.1f} / {s['measured']['denoise_ms']['median']:.1f} / {s['measured']['vae_decode_ms']['median']:.1f} | – | complete; unprofiled1+3+10 on cache-capable binary |\n\n## A100 Base reference");p.write_text(t)
p=Path('wiki/experiments.md');t=p.read_text();t='\n'.join(line.replace(' | [W&B]',f', [same-binary repeat](../results/runs/{rid}/) | [W&B]',1) if line.startswith('| jetson-flux-klein-base-001 |') else line for line in t.splitlines())+'\n';p.write_text(t)
p=Path('wiki/log.md');t=p.read_text();pos=t.index('\n## ');t=t[:pos]+'''\n## [2026-10-05] experiment | Jetson same-binary reference repeat
[Base reference](../experiments/jetson-flux-klein-base-001.md) repeated and validated on the cache-capable harness; [EasyCache comparison](../experiments/compare-jetson-base-easycache.md) now includes matched binary/protocol evidence. Diagnostic selection remains separate from formal quality eligibility.
'''+t[pos:];p.write_text(t)
p=Path('wiki/hot.md');t=p.read_text().replace('targeted profile validated and no-cache repeat on the cache-capable harness running in tmux','targeted profile and same-binary no-cache repeat validated; lazy-loading correctness test running in tmux').replace('EasyCache full protocol and paired PSNR/LPIPS are recorded','EasyCache qualifies under the campaign diagnostic rule after the same-binary repeat; results and paired PSNR/LPIPS are recorded');p.write_text(t)
p=Path('output/overnight-20261004/state.json');d=json.loads(p.read_text());d['status']='jetson-base-lazy-smoke-running-matched-cache-comparison-complete'
for x in d['jobs']:
 if x.get('session')=='diffusion-base-repeat':x['status']='complete09:37:16UTC; independently validated1+3+10, same binary as EasyCache and pixels equal original; documented; W&B upload pending'
d['jobs'].append(dict(host='jetson',root=d['jetson_options_root'],session='diffusion-base-lazy-smoke',name='Base lazy-loading single-option correctness attempt',log='logs/base-lazy-smoke.log',config='configs/jetson-flux-klein-base-q4-512-disk-lazy-smoke.json',status='started09:45:30UTC; correctness gate before full protocol',run_dir='jetson-flux-klein-base-003__attempt__20261005-054530'))
d['pending'][1]='Jetson Base lazy-loading gate/full protocol; further individual memory options and qualifying combinations';d['cache_gate']['jetson_local']='Matched no-cache repeat042014 passed; EasyCache observed median ratio1.78464894 and non-overlapping measured ranges. Diagnostic screen recorded, qualifies for later combinations; formal quality eligibilityfalse and no second qualified Base option. Sequential-capture confounds retained.';d['cache_gate']['jetson_next']='Monitor lazy-loading attempt054530; validate settings/callbacks/CFG/pixels against matched no-cache reference before full protocol. One option changes: eager-load0; cacheoff.'
d['next_actions'][0]='Monitor Jetson Base lazy-loading attempt054530; validate and export, then full protocol if correctness passes. Matched repeat042014 upload pending.'
d['last_checked_utc']=datetime.now(timezone.utc).isoformat();d['last_check']='Repeat complete and validated, comparison updated, lazy-loading attempt launched serially in tmux. Six Slurm jobs remain pending. W&B repeat upload pending.';p.write_text(json.dumps(d,indent=2)+'\n')
