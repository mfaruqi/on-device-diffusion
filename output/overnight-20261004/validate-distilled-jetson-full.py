"""Independent audit of completed campaign four-step full protocols; no inference."""
import sys,json,csv,hashlib,statistics,re
from pathlib import Path
from PIL import Image
from run_jetson_sdcpp import check_config,audit_execution
from jetson_profile import validate_callbacks,expected_conditioning_hits
from sdcpp_engine import audit_cache
p=Path(sys.argv[1]);ref=Path(sys.argv[2]);cfg=json.loads((p/'config.json').read_text())
assert json.loads((p/'status.json').read_text())['status']=='complete'
phases=['first']+['warmup']*3+['measured']*10
assert check_config(cfg)==phases
assert cfg['workload']['steps']==4 and cfg['workload']['cfg_scale']==1
records=[json.loads(x) for x in (p/'engine/results.jsonl').read_text().splitlines()]
loads=[x for x in records if x['event']=='load'];gens=[x for x in records if x['event']=='generate']
assert len(loads)==1 and len(gens)==14
log=(p/'engine/sdcpp.log').read_text();audit_execution(log,cfg);ca=audit_cache(log,cfg['optimizations'].get('step_cache'),14)
stamps=[float(x) for x in re.findall(r'^([0-9.]+).*flux executing segment 1/1: graph',log,re.MULTILINE)]
skips=ca.get('steps_skipped',[0]*14);end=loads[0]['t_end']
for i,g in enumerate(gens):
 validate_callbacks([loads[0],g],4,expected_conditioning_hits(cfg,i))
 assert g['run_index']==i and g['t_start']>=end;end=g['t_end']
 assert sum(g['t_start']<=t<=g['t_end'] for t in stamps)==4-skips[i]
 assert (g['width'],g['height'],g['channels'])==(512,512,3)
rows=list(csv.DictReader((p/'runs.csv').open()));s=json.loads((p/'summary.json').read_text())
assert [r['phase'] for r in rows]==phases
assert s['measured_runs']==10 and s['cache_audit']==ca
for key,stats in s['measured'].items():
 if stats is None:continue
 vals=[float(r[key]) for r in rows if r['phase']=='measured']
 for k,v in [('median',statistics.median(vals)),('min',min(vals)),('max',max(vals)),('mean',statistics.mean(vals)),('stdev',statistics.stdev(vals))]:assert abs(stats[k]-v)<1e-6,(key,k,stats[k],v)
 assert stats['n']==10
for row,g in zip(rows,gens):
 assert abs(float(row['wall_ms'])-(g['t_end']-g['t_start'])*1000)<1e-6
 assert abs(float(row['wall_ms'])-sum(float(row[k]) for k in ['text_encode_ms','denoise_ms','vae_decode_ms','other_ms']))<1e-6
images=list((p/'images').glob('*.png'));assert {f.name for f in images}=={'first.png','measured-0.png'}
for f in images:
 row=rows[0] if f.name=='first.png' else rows[4]
 im=Image.open(f).convert('RGB');assert im.size==(512,512)
 assert hashlib.sha256(im.tobytes()).hexdigest()==row['image_sha256'],f
assert s['deterministic_output']==(len(set(r['image_sha256'] for r in rows if r['phase']=='measured'))==1)
equal=Image.open(ref/'images/measured-0.png').convert('RGB').tobytes()==Image.open(p/'images/measured-0.png').convert('RGB').tobytes()
assert json.loads((p/'environment.json').read_text())['binary_sha256']==json.loads((ref/'environment.json').read_text())['binary_sha256']
assert s['conditioning_cache_hits_per_generation']==[expected_conditioning_hits(cfg,i) for i in range(14)]
(p/'validation.json').write_text(json.dumps({'status':'passed','scope':'Full1+3+10: independent callback/settings/CFG execution count/phase/aggregate/image-hash checks; reference image equality recorded separately from execution validity.','steps':4,'transformer_passes':[4-v for v in skips],'conditioning_hits':[g['cond_cache_hits'] for g in gens],'reference':ref.name,'reference_rgb_equal':equal,'all_generation_rgb_hashes_equal':len(set(r['image_sha256'] for r in rows))==1,'same_binary':True,'formal_quality_eligible':False},indent=2)+'\n')
print(p.name,'validated; reference RGB equality',equal)
