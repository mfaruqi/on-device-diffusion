"""Independent saved-output audit for the existing A100 sd.cpp protocols."""
import csv, hashlib, json, re, statistics, sys
from pathlib import Path
from PIL import Image
from jetson_profile import validate_callbacks
from sdcpp_engine import audit_log, audit_cache, conditioning_hits
p=Path(sys.argv[1]); ref=Path(sys.argv[2])
cfg=json.loads((p/'config.json').read_text()); s=json.loads((p/'summary.json').read_text())
assert json.loads((p/'status.json').read_text())['status']=='complete'
w=cfg['workload']; steps=w['num_inference_steps']; passes=2 if w['guidance_scale']>1 else 1
phases=sum(([phase]*cfg['protocol'][field] for phase,field in [('first','first_runs'),('warmup','warmup_runs'),('measured','measured_runs')]),[])
records=[json.loads(x) for x in (p/'engine/results.jsonl').read_text().splitlines()]
loads=[x for x in records if x['event']=='load']; gens=[x for x in records if x['event']=='generate']
assert len(loads)==1 and len(gens)==len(phases)
log=(p/'engine/sdcpp.log').read_text(); hits=conditioning_hits(cfg,len(gens))
audit=audit_log(log,cfg['engine_settings'],gens[-1]['t_end'],expected_conditioning_hits=sum(hits)); assert not audit['problems']
ca=audit_cache(log,cfg['optimizations'].get('step_cache'),len(gens)); assert s['cache_audit']==ca
assert f'txt_cfg: {w["guidance_scale"]:.2f}' in log and f'sample_steps: {steps}' in log and 'get_sigmas with Flux2 scheduler' in log
for field in ['conditioning_cache_size','disable_prefetch']:
 assert re.search(r'^'+field+': '+str(cfg['engine_settings'][field]).lower()+r'\s*$',log,re.M)
mapped=sorted({Path(f).name for f in re.findall(r"using mmap for '([^']+)'",log)})
assert mapped==(sorted(Path(f).name for f in cfg['model']['sha256']) if cfg['engine_settings']['mmap'] else [])
assert not re.search(r'failed to memory-map|mmap: (?:failed|.*cannot map)',log)
stamps=[float(x) for x in re.findall(r'^([0-9.]+).*flux executing segment 1/1: graph',log,re.M)]
counts=[]
for i,g in enumerate(gens):
 validate_callbacks([loads[0],g],steps,hits[i]); assert g['run_index']==i
 assert (g['width'],g['height'],g['channels'])==(w['width'],w['height'],3)
 count=sum(g['t_start']<=t<=g['t_end'] for t in stamps); assert count==passes*(steps-ca.get('steps_skipped',[0]*len(gens))[i]); counts.append(count)
rows=list(csv.DictReader((p/'runs.csv').open())); assert [r['phase'] for r in rows]==phases
for row,g in zip(rows,gens):
 assert abs(float(row['wall_ms'])-(g['t_end']-g['t_start'])*1000)<1e-6
 assert abs(float(row['denoise_ms'])-sum(float(row[f'denoise_step_{i}_ms']) for i in range(steps)))<1e-6
 assert abs(float(row['wall_ms'])-sum(float(row[k]) for k in ['text_encode_ms','denoise_ms','vae_decode_ms','other_ms']))<1e-6
for key,stats in s['measured'].items():
 if stats is None: continue
 vals=[float(r[key]) for r in rows if r['phase']=='measured']
 for field,val in [('median',statistics.median(vals)),('min',min(vals)),('max',max(vals))]: assert abs(stats[field]-val)<1e-6
hashes=sorted({r['image_sha256'] for r in rows}); assert len(hashes)==1
for name,row in [('first.png',rows[0]),('measured-0.png',next(r for r in rows if r['phase']=='measured'))]:
 im=Image.open(p/'images'/name).convert('RGB'); assert im.size==(w['width'],w['height']); assert hashlib.sha256(im.tobytes()).hexdigest()==row['image_sha256']
equal=Image.open(ref/'images/measured-0.png').convert('RGB').tobytes()==Image.open(p/'images/measured-0.png').convert('RGB').tobytes()
if not any(ca.get('steps_skipped',[])): assert equal, 'Exact/no-skipped configuration changed saved pixels'
env=json.loads((p/'environment.json').read_text()); refenv=json.loads((ref/'environment.json').read_text())
assert env['engine']['binary_sha256']==refenv['engine']['binary_sha256']
assert 'A100-PCIE-40GB' in env['nvidia_smi_gpu']
out={'status':'passed','generations':len(gens),'steps':steps,'transformer_passes':counts,'conditioning_hits':hits,'mapped_files':mapped,'cache_audit':ca,'rgb_hashes':hashes,'reference':ref.name,'reference_rgb_equal':equal,'formal_quality_eligible':False,'scope':'Saved config, callback/actual CFG execution counts, one context, image, phase and CSV/summary arithmetic checked. Only two retained PNGs rehashed. One measured observation in attempts is not a repeated benchmark.'}
(p/'validation.json').write_text(json.dumps(out,indent=2)+'\n');print(p.name,counts,hits,equal)
