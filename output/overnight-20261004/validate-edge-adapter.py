"""Saved-output checks independent of the runner's parsing functions."""
import csv,hashlib,json,re,statistics,sys
from pathlib import Path
from PIL import Image
p=Path(sys.argv[1]);cfg=json.loads((p/'config.json').read_text());s=json.loads((p/'summary.json').read_text());env=json.loads((p/'environment.json').read_text());log=(p/'engine.log').read_text()
assert json.loads((p/'status.json').read_text())['status']=='complete'
phases=['first']+['warmup']*cfg['protocol']['warmup_runs']+['measured']*cfg['protocol']['measured_runs'];rows=list(csv.DictReader((p/'runs.csv').open()));assert [r['phase'] for r in rows]==phases
passes=re.findall(r'\[ed-sample\] pass (\d+)/(\d+)\s+1/1\s+seed=0\s+([\d.]+)s',log);assert len(passes)==len(rows)
markers=re.findall(r'\[\[phase\]\] stage=(\w+) event=(\w+) t=([\d.]+)',log);assert len(markers)==6*len(rows)
stages=list(csv.DictReader((p/'stages.csv').open()));assert len(stages)==3*len(rows)
for i,row in enumerate(rows):
 assert int(row['run_index'])==i and tuple(map(int,passes[i][:2]))==(i+1,len(rows))
 assert float(row['wall_ms'])==float(passes[i][2])*1000
 m=markers[i*6:i*6+6];assert [(a,b) for a,b,t in m]==[(a,b) for a in ['encode','denoise','decode'] for b in ['begin','end']]
 t=[float(x[2]) for x in m];assert t==sorted(t);assert (t[-1]-t[0])*1000<=float(row['wall_ms'])+.501
 for j,name in enumerate(['encode_setup','denoise','vae_decode']):
  sr=stages[i*3+j];assert sr['stage']==name and sr['phase']==row['phase'] and not sr['gpu_ms'];d=(t[j*2+1]-t[j*2])*1000;assert abs(float(sr['host_ms'])-d)<1e-6
  if name!='encode_setup':assert abs(float(row[name+'_ms'])-d)<1e-6
 for k in ['text_encode_ms','gpu_span_ms','other_ms','postprocess_ms','device_used_peak_gib','host_rss_gib','peak_alloc_gib','peak_reserved_gib']+[f'denoise_step_{i}_ms' for i in range(4)]:assert not row[k]
 if i<len(rows)-1:assert not row['image_sha256']
for k,v in s['measured'].items():
 if v is None:continue
 vals=[float(r[k]) for r in rows if r['phase']=='measured']
 assert v['n']==len(vals)
 for key,val in [('median',statistics.median(vals)),('min',min(vals)),('max',max(vals))]:assert abs(v[key]-val)<1e-6
im=Image.open(p/'output.png').convert('RGB');assert im.size==(1024,1024);sha=hashlib.sha256(im.tobytes()).hexdigest();assert sha==rows[-1]['image_sha256']=='c476fc325f80bc3386922415ed1e8c410a6f44ebca2a1d7cb02ca5899accad55'
assert json.loads((p/'image-retention.json').read_text())['retained_run_index']==len(rows)-1
assert s['deterministic_output'] is None
mem=json.loads((p/'device-memory.json').read_text());assert len(mem['samples'])>10;assert s['capture_device_used_peak_gib']==max(v for _,v in mem['samples'])/1024**3
assert env['engine']['binary_sha256']==cfg['engine']['binary_sha256'] and 'A100-PCIE-40GB' in env['nvidia_smi_gpu']
ev=[json.loads(x) for x in (p/'events.jsonl').read_text().splitlines()];assert len(ev)==4*len(rows)
for e in ev:
 assert (e['clock']=='edgedit_system_clock' if e['event']=='stage' else e['start_s'] is None)
(p/'validation.json').write_text(json.dumps({'status':'passed','generation_count':len(rows),'phases':phases,'final_rgb_sha256':sha,'matches_interface_gate_final_image':True,'determinism_available':False,'memory_samples':len(mem['samples']),'stage_clock':'system_clock, host phase intervals','wall_clock':'steady_clock duration printed to1ms','scope':'Independent raw-log/CSV/summary/image/memory/event checks; no per-step timing, earlier image hashes or aligned per-generation memory. Formal quality eligibility false.'},indent=2)+'\n');print(p.name,'validated',s['measured']['wall_ms'])
