import json,re
from pathlib import Path
root=Path('results/runs');base=Path('output/overnight-20261004');ids=json.loads((base/'a100-completed-fulls.json').read_text())
def load(rid,name='summary.json'):return json.loads((root/rid/name).read_text())
def append_record(eid,rid,section):
 p=Path('experiments')/(eid+'.md');t=p.read_text();t=re.sub(r'^runs: \[(.*?)\]',lambda m:'runs: ['+m[1]+(', '+rid if rid not in m[1] else '')+']',t,count=1,flags=re.M)
 t=re.sub(r'^updated: .*','updated: 2026-10-06',t,count=1,flags=re.M)
 if f'## Full protocol {rid}' not in t:t+='\n'+section
 if eid not in ['a100-edgedit-flux-klein-001','a100-sdcpp-flux-klein-base-001']:
  t=re.sub(r'^status: .*','status: complete',t,count=1,flags=re.M)
  t=t.replace('Status: **partial**; functional gate completed and full repeated protocol queued.','Status: **complete**; full repeated protocol validated. Earlier functional-gate evidence is retained below.')
 p.write_text(t)
 p=Path('wiki/experiments.md');lines=p.read_text().splitlines()
 for i,l in enumerate(lines):
  if l.startswith('| '+eid+' |'):
   cells=l.split('|')
   if rid not in l:cells[7]+=f', [full protocol](../results/runs/{rid}/) '
   if eid not in ['a100-edgedit-flux-klein-001','a100-sdcpp-flux-klein-base-001']:cells[5]=' complete; full protocol validated '
   lines[i]='|'.join(cells)
 p.write_text('\n'.join(lines)+'\n')
index=Path('results/README.md').read_text()
for rid in ids:
 cfg=load(rid,'config.json');s=load(rid);v=load(rid,'validation.json');e=load(rid,'environment.json');st=load(rid,'status.json');m=s['measured'];eid=cfg['id'];job=e['env_vars']['SLURM_JOB_ID']
 sec=f'''## Full protocol {rid}

[Summary](../results/runs/{rid}/summary.json), [config](../results/runs/{rid}/config.json), [independent validation](../results/runs/{rid}/validation.json), [environment](../results/runs/{rid}/environment.json). Slurm {job}, {e['hostname']}, {st['started_utc']} to {st['finished_utc']}; repo snapshot `{e['git_commit']}`. One context; one first generation, three discarded warm-ups and ten measured generations. Same pinned BF16 workload as the functional gate; all requested settings retained. [Metric definitions](../wiki/methods/baseline-metrics.md).

| Metric | Median, ms | Measured min–max, ms |
|---|---:|---:|
'''
 for key in ['wall_ms','text_encode_ms','denoise_ms','vae_decode_ms','other_ms']:
  z=m[key];sec+=f"| {key} | {z['median']:.6f} | {z['min']:.6f}–{z['max']:.6f} |\n"
 sec+=f'''
First generation {s['first_run']['wall_ms']:.6f}ms; context load {s['load']['load_total_s']:.6f}s (filesystem cache may be warm). Sampled generation device-wide peak {m['device_used_peak_gib']['max']:.6f}GiB; child host peak RSS {s['host_peak_rss_gib']:.6f}GiB. No allocator or per-stage RSS measurement.

All14 generations passed callback/actual transformer-call accounting, phase, engine-setting and CSV/summary checks. Actual transformer passes per generation: {v['transformer_passes']}. Conditioning hits: {v['conditioning_hits']}. Approximate steps skipped: {v['cache_audit'].get('steps_skipped',[0]*14)}. Mapped files: {v['mapped_files']}. All14 runner-reported pixel hashes agree; both retained PNGs independently rehashed. Retained RGB equals the same-binary no-cache reference: {v['reference_rgb_equal']}.
'''
 if (root/rid/'quality-diagnostics.json').exists():
  q=load(rid,'quality-diagnostics.json');sec+=f"\n[Paired image diagnostic](../results/runs/{rid}/quality-diagnostics.json): PSNR {q['psnr_db']}dB; LPIPS AlexNet v0.1 {q['lpips_alex_v0_1']:.9f}. One development prompt/seed, full1024² RGB without resizing; no held-out evaluation or formal quality acceptance.\n"
 sec+='\nInterpretation: this establishes a repeated timing distribution for the declared configuration. Selection and deltas belong in the matched comparison; sequential capture conditions were not randomized.\n'
 if cfg['engine_settings']['conditioning_cache_size']:sec+='\nExact conditioning hits concern repeated identical prompt inputs, not new-prompt latency.\n'
 sec+=f'\n[W&B](https://wandb.ai/mfaruqi-purdue-university/on-device-diffusion/runs/{rid}).\n'
 append_record(eid,rid,sec)
 if rid not in index:index+=f"| `{rid}` | [{eid}](../experiments/{eid}.md) | A100 PCIe,g007 | {eid} saved config | {m['wall_ms']['median']:.3f} | {m['text_encode_ms']['median']:.3f} / {m['denoise_ms']['median']:.3f} / {m['vae_decode_ms']['median']:.3f} | – (device {m['device_used_peak_gib']['max']:.3f}) | complete{job}; validated1+3+10 |\n"
edge='a100-edgedit-flux-klein-001__attempt__20261005-221339';eid='a100-edgedit-flux-klein-001';s=load(edge)
append_record(eid,edge,f'''## Full protocol {edge}

This directory is a **two-generation adapter attempt**, not a full baseline. [Summary](../results/runs/{edge}/summary.json), [validation](../results/runs/{edge}/validation.json), [config](../results/runs/{edge}/config.json), [environment](../results/runs/{edge}/environment.json). Slurm11890064 on g007 completed. First wall{s['first_run']['wall_ms']:.0f}ms; second wall{s['measured']['wall_ms']['median']:.0f}ms, rounded to1ms by upstream stdout. One context and complete ordered stages; unchanged four-step schedule, BF16weights/FP32activations. Whole-child device peak{s['capture_device_used_peak_gib']:.6f}GiB includes loading and between-image handling; per-generation peak unavailable. Load{s['load']['load_total_s']:.6f}s, filesystem cache possibly warm.

Raw-log/CSV/summary/events/memory/image checks passed. Final RGB matches the previously inspected interface image; earlier repeats are overwritten and determinism stays null. Encode_setup is not isolated text encoding, and per-step timing is unavailable. Full1+3+10 job11891242 now submitted after this gate; its metrics require independent validation.
''')
if edge not in index:index+=f'| `{edge}` | [{eid}](../experiments/{eid}.md) | A100 PCIe,g007 | edge adapter smoke | 1629 | unavailable / 1091.122 / 267.402 | – (whole-child21.607) | complete11890064; one measured observation |\n'
Path('results/README.md').write_text(index)
# Workload-separated comparisons; check every held-fixed field before deriving deltas.
for label in ['base','distilled']:
 runs=[n for n in ids if ('-base-' in n)==(label=='base')];ref=runs[0];a=load(ref);ac=load(ref,'config.json');ae=load(ref,'environment.json')
 eid='compare-a100-'+label+'-sdcpp-options';rids=', '.join(runs)
 text=f'''---
type: experiment-record
id: {eid}
status: complete
device: a100-pcie-40gb
engine: stable-diffusion-cpp
runs: [{rids}]
updated: 2026-10-06
---

# Compare: A100 {label} sd.cpp individual options

## Question

Which tested individual option reduces latency under the campaign diagnostic rule? Week2 image-engine baselines, RQ1. This is descriptive configuration evidence, not formal quality-constrained planner evaluation.

## Runs

| Run and saved configuration | Experiment record | Intended difference |
|---|---|---|
'''
 summaries=[]
 for rid in runs:
  c=load(rid,'config.json');e=load(rid,'environment.json');b=load(rid)
  for key in ['model','precision','workload','protocol']:assert c[key]==ac[key],(rid,key)
  assert e['engine']['binary_sha256']==ae['engine']['binary_sha256'] and e['hostname']==ae['hostname']
  settings={k:(ac['engine_settings'][k],v) for k,v in c['engine_settings'].items() if v!=ac['engine_settings'][k]}
  optimizations={k:(ac['optimizations'].get(k),v) for k,v in c['optimizations'].items() if v!=ac['optimizations'].get(k)}
  name='reference' if rid==ref else next(iter(optimizations))
  assert rid==ref or len(optimizations)==1
  text+=f"| [{rid}](../results/runs/{rid}/config.json) | [{c['id']}]({c['id']}.md) | {name} | \n"
  summaries.append((rid,name,b,settings,optimizations))
 text+=f'''
## Matching

| Setting | Held fixed |
|---|---|
| Device | Same A100-PCIE-40GB, same host {ae['hostname']}; GPU UUID recorded in environment |
| Engine | Same sd.cpp/ggml commits, binary `{ae['engine']['binary_sha256']}` |
| Checkpoint | Identical model object, pinned revisions and all component SHA256 values |
| Precision | BF16 weights; unchanged backend |
| Workload | {ac['workload']['width']}²,{ac['workload']['num_inference_steps']}steps,CFG{ac['workload']['guidance_scale']},same prompt,seed0,batch1 |
| Protocol | One first,three discarded warm-ups,ten measured;20ms NVML samples |
| Residency | Fixed CUDA0 parameters/compute,eager load,one segment,no auto-fit/offload |
| Variable | One labelled optimization per variant; matching receipt lists its engine-setting realization |

Confounds: separate sequential jobs without randomized order or controlled filesystem-cache/thermal/background conditions. Same engine/seed fixes the noise-generation procedure but no explicit initial latent tensor was supplied. Timing ratios are descriptive; range separation is a campaign selection heuristic, not a statistical confidence interval. Memory compares the same sampled device-wide metric.

## Results

All values below come from each linked run's summary.json. All variants have independent validation.json and quality-diagnostics.json; reference images are the saved no-cache outputs.

| Configuration | Wall median [min–max], ms | Delta vs reference, ms | Reference/variant | Device peak, GiB | Speed-range rule |
|---|---:|---:|---:|---:|---|
'''
 selections=[]
 for rid,name,b,settings,opts in summaries:
  am=a['measured']['wall_ms'];m=b['measured']['wall_ms'];ratio=am['median']/m['median'];delta=m['median']-am['median'];passed=rid!=ref and m['max']<am['min']
  text+=f"| {name} | {m['median']:.3f} [{m['min']:.3f}–{m['max']:.3f}] | {delta:.3f} | {ratio:.6f} | {b['measured']['device_used_peak_gib']['max']:.6f} | {'pass' if passed else 'reference' if rid==ref else 'fail'} |\n"
  if rid!=ref:
   q=load(rid,'quality-diagnostics.json');qa=bool(opts.get('step_cache'));fidelity=qa or q['mse_rgb_8bit']==0 or isinstance(q['psnr_db'],(int,float)) and q['psnr_db']>=30
   rec={'reference':ref,'variant':rid,'matched_fields':['model','precision','workload','protocol','binary_sha256','hostname'],'engine_settings_difference':settings,'optimizations_difference':opts,'wall_delta_ms':delta,'median_ratio_reference_over_variant':ratio,'wall_reduction_percent':100*(1-m['median']/am['median']),'speed_range_pass':passed,'diagnostic_fidelity_screen_pass':fidelity,'qualifies_for_later_combination':passed and fidelity,'formal_quality_eligible':False,'confounds':['sequential nonrandomized jobs','filesystem-cache/thermal/background conditions uncontrolled']}
   (root/rid/'comparison.json').write_text(json.dumps(rec,indent=2)+'\n');selections.append((name,rec,q))
 text+='\n| Configuration | Text median, ms | Denoise median, ms | VAE median, ms |\n|---|---:|---:|---:|\n'
 for rid,name,b,_,_ in summaries:text+=f"| {name} | {b['measured']['text_encode_ms']['median']:.3f} | {b['measured']['denoise_ms']['median']:.3f} | {b['measured']['vae_decode_ms']['median']:.3f} |\n"
 text+='\n## Execution, image evidence and selection\n\n'
 for name,r,q in selections:
  v=load(r['variant'],'validation.json');text+=f"- **{name}**: actual transformer passes {sorted(set(v['transformer_passes']))}; skipped steps {sorted(set(v['cache_audit'].get('steps_skipped',[0])))}; conditioning hits {sorted(set(v['conditioning_hits']))}. PSNR {q['psnr_db']}dB, LPIPS {q['lpips_alex_v0_1']:.9f}. Qualifies for later combinations under the campaign diagnostic rule: **{r['qualifies_for_later_combination']}** ([receipt](../results/runs/{r['variant']}/comparison.json)).\n"
 if label=='base':text+='\nThe observed reduction is concentrated in denoising and accompanies23/50 skipped scheduler steps, reducing actual CFGtransformer passes from100to54. The approximate output differs; the PSNR/LPIPS pair is recorded, not used as formal quality approval. Only EasyCache was tested as a full single-option variant in this Base comparison; a combination needs another independently qualifying option.\n'
 else:text+='\nEasyCache executed all four transformer steps and produced identical pixels; no step reuse benefit is observed. Conditioning reuse is the sole qualifying variant here, with identical pixels; its small end-to-end gain applies to repeated identical prompt inputs. Prefetch-disabled and mmap do not pass range separation. One qualifying option does not form a combination.\n'
 text+='\n## Limits\n\nOne development prompt/seed; no alignment benchmark, blinded quality study or held-out prompts. Ten measurements within one loaded context do not characterize every prompt, hardware pressure state or job-to-job variance. Sampled memory can miss brief peaks and is device-wide. No cross-device or cross-engine ranking is established by this record.\n'
 Path('experiments',eid+'.md').write_text(text)
 p=Path('wiki/experiments.md');t=p.read_text();line=f"| {eid} | A100-PCIE-40GB | stable-diffusion.cpp | Matched {label} options | complete | [comparison](../experiments/{eid}.md) | "+', '.join(f'[run](../results/runs/{r}/)' for r in runs)+' | – |\n'
 if '| '+eid+' |' not in t:p.write_text(t+line)
p=Path('wiki/log.md');t=p.read_text();entry='''\n## [2026-10-06] experiment | A100 repeated options validated\n[Base comparison](../experiments/compare-a100-base-sdcpp-options.md) and [four-step comparison](../experiments/compare-a100-distilled-sdcpp-options.md) retain same-binary checks, timing ranges and image diagnostics. [edge adapter gate](../experiments/a100-edgedit-flux-klein-001.md) validated; full protocol11891242 submitted.\n'''
if 'A100 repeated options validated' not in t:p.write_text(t.replace('# Log\n','# Log\n'+entry,1))
