import json,sys
from pathlib import Path
for name in sys.argv[1:]:
 p=Path('results/runs')/name;s=json.loads((p/'summary.json').read_text());c=json.loads((p/'config.json').read_text());v=json.loads((p/'validation.json').read_text());assert v['status']=='passed'
 eid=c['id'];num=eid.rsplit('-',1)[1];labels={'003':'no-cache control on the cache-capable harness','006':'EasyCache','007':'exact conditioning reuse','008':'prefetch disabled','009':'memory-mapped weight-file I/O'};label=labels[num]
 measured=s['measured'];mem=s['monitor_window'];base=f'../results/runs/{name}'
 body=f'''\n## Two-generation correctness attempt: {name}\n\n[Config]({base}/config.json), [summary]({base}/summary.json), [status]({base}/status.json), [environment]({base}/environment.json), [validation]({base}/validation.json).\n\nPinned sd.cpp19bbbca; Jetson Orin Nano25W; distilled klein transformer Q4_0, Qwen3 Q4_K_M and BF16 VAE;512×512,4Euler/flux2steps,CFG1,seed0,batch1. Fixed prompt: “A cat holding a sign that says hello world”. One context, first generation then one measured observation; no discarded warm-up. The single option is **{label}**. Engine snapshot31f3be71d07adda2fb122de0ef75a056c27ec7b1; four CPU threads, eager disk-backed segmented CUDA execution. Requested settings and engine log audit are preserved in the summary; they do not establish continuous residency or physical I/O.\n\n[Metric definitions](../wiki/methods/baseline-metrics.md): host callback times include on-demand loading; no CUDA-event span or allocator peak is available.\n\n| Metric | First, ms | Second observation, ms |\n|---|---:|---:|\n'''
 for key in ['wall_ms','text_encode_ms','denoise_ms','vae_decode_ms','other_ms']:body+=f"| {key} | {s['first_run'][key]:.3f} | {measured[key]['median']:.3f} |\n"
 body+=f"\nContext creation {s['load']['load_total_s']:.6f}s after hash checks warmed the file cache. Whole-capture sampled RAM peak {mem['ram_peak_gib']:.6f}GiB, swap {mem['swap_min_gib']:.6f}–{mem['swap_max_gib']:.6f}GiB, {mem['samples']}samples; includes model loading and both generations.\n\nCallback, settings, phase, aggregate and image-hash validation passed. Conditioning hits per generation: {v['conditioning_hits']}; actual transformer passes: {v['transformer_passes']}. Both saved images are512×512RGB. Reference image equality is recorded in validation; no formal quality eligibility or held-out evaluation.\n"
 if num=='006':body+=f"\nEasyCache parameters are threshold0.2,start0.15,end0.95. Enabled for both generations, skipped steps{s['cache_audit']['steps_skipped']}. Thus this attempt demonstrates enabled execution without step reuse. It does not establish sustained performance.\n"
 if num=='007':body+='\nConditioning cache capacity1 retains the fixed prompt between generations; the text stage after a hit measures retrieval/setup, not a fresh encoder pass. Approximate step caching remains off.\n'
 if num=='008':body+='\nOnly engine prefetch is disabled; eager loading, parameter storage, segmented computation and caching settings remain explicit in the saved config.\n'
 if num=='009':body+=f"\nEngine-reported mapped component files: {s['engine_audit']['mmap_io']['confirmed_files']}. These are weight-file mappings, not evidence of GPU zero-copy or fully resident weights.\n"
 if (p/'quality-diagnostics.json').exists():body+=f"\n[Paired image diagnostic]({base}/quality-diagnostics.json): MSE0,infinite PSNR,LPIPS0 against the saved control image, one development prompt/seed only.\n"
 body+=f'\nInterpretation: functional validation passed; one measured observation cannot qualify an option under the full-protocol speed-range rule.\n\nNext experiment: unchanged full1+3+10protocol using `{c.get("reference_config") if num=="003" else "configs/jetson-flux-klein-q4-512-"+{"006":"easycache","007":"conditioning-cache","008":"no-prefetch","009":"mmap"}[num]+".json"}`.\n'
 record=Path('experiments')/(eid+'.md')
 if record.exists():
  txt=record.read_text()
  if name in txt:continue
  txt=txt.replace('runs: [','runs: ['+name+', ',1)+body
 else:
  txt=f'''---
type: experiment-record
id: {eid}
status: partial
device: jetson-orin-nano
engine: stable-diffusion-cpp
runs: [{name}]
updated: 2026-10-05
---

# {eid}: distilled klein {label}

Status: **partial**; functional attempt passed, full repeated protocol pending.

## Question

Does {label} execute correctly for the original four-step workload? Week2 image baselines, RQ1: “Which diffusion optimization choices transfer across devices and workloads?” ([proposal overview](../wiki/project/overview.md)). The reference is the separately recorded no-cache control; this record does not establish a comparative performance result.
'''+body
 record.write_text(txt)
 idx=Path('results/README.md');idx.write_text(idx.read_text()+f'| `{name}` | [{eid}](../experiments/{eid}.md) | Jetson Orin Nano,25W | distilled {label} | – | – | – | complete; two-generation functional attempt, full timing pending |\n')
 reg=Path('wiki/experiments.md');txt=reg.read_text();link=f'[attempt](../results/runs/{name}/)'
 if num=='003':txt=txt.replace('[full-protocol profile](../results/runs/jetson-flux-klein-003',link+', [full-protocol profile](../results/runs/jetson-flux-klein-003',1)
 else:
  row=f'| {eid} | Jetson Orin Nano | stable-diffusion.cpp | Distilled {label} | partial; correctness passed, full pending | [record](../experiments/{eid}.md) | {link} | – |\n'
  txt=txt.replace('## Comparisons',row+'\n## Comparisons',1)
 reg.write_text(txt)
 log=Path('wiki/log.md');txt=log.read_text().replace('# Log\n',f'# Log\n\n## [2026-10-05] experiment | {eid}: distilled {label} correctness\n[Record](../experiments/{eid}.md); two-generation callback/settings/image validation passed; full protocol pending.\n',1);log.write_text(txt)
 print('documented',name)
