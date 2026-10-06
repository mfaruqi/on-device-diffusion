import csv,json,math,hashlib,statistics,collections,sys
from pathlib import Path
from PIL import Image
p=Path(sys.argv[1] if len(sys.argv)>1 else 'results/runs/a100-flux-klein-base-001__baseline__20261005-074425')
c=json.loads((p/'config.json').read_text());s=json.loads((p/'summary.json').read_text());e=json.loads((p/'environment.json').read_text());l=json.loads((p/'load.json').read_text())
assert json.loads((p/'status.json').read_text())['status']=='complete'
assert c['model']['revision']=='a3b4f4849157f664bdbc776fd7453c2783562f4d'
assert 'A100-PCIE-40GB' in e['gpu_name']
assert c['precision']=='bfloat16' and not l['is_distilled']
assert s['guidance_scale']==4 and s['transformer_calls_per_step']==2
rows=list(csv.DictReader((p/'runs.csv').open()));stages=list(csv.DictReader((p/'stages.csv').open()))
assert [r['phase'] for r in rows]==['first']+['warmup']*3+['measured']*10
for i,r in enumerate(rows):
 ss=[x for x in stages if int(x['run_index'])==i];counts=collections.Counter(x['stage'] for x in ss)
 assert counts['text_encode']==2 and counts['vae_decode']==1 and counts['postprocess']==1
 assert {k:v for k,v in counts.items() if k.startswith('denoise_step_')}=={f'denoise_step_{k}':2 for k in range(50)}
 for name in ['text_encode','vae_decode','postprocess']:
  assert abs(sum(float(x['gpu_ms']) for x in ss if x['stage']==name)-float(r[name+'_ms']))<1e-5
 denoise=sum(float(x['gpu_ms']) for x in ss if x['stage'].startswith('denoise_step_'))
 assert abs(denoise-float(r['denoise_ms']))<1e-5
 assert abs(sum(float(r[k]) for k in ['text_encode_ms','denoise_ms','vae_decode_ms','postprocess_ms','other_ms'])-float(r['gpu_span_ms']))<1e-5
 for k in ['wall_ms','gpu_span_ms','peak_alloc_gib','peak_reserved_gib']: assert math.isfinite(float(r[k])) and float(r[k])>0
for k,stats in s['measured'].items():
 if stats is None:continue
 vals=[float(r[k]) for r in rows if r['phase']=='measured']
 assert stats['n']==10
 for field,value in [('median',statistics.median(vals)),('min',min(vals)),('max',max(vals))]:
  assert abs(stats[field]-value)<1e-5
hashes={}
assert len({r['image_sha256'] for r in rows})==1
for name,row in [('first',rows[0]),('measured-0',rows[4])]:
 im=Image.open(p/'images'/f'{name}.png').convert('RGB');assert im.size==(1024,1024)
 hashes[name]=hashlib.sha256(im.tobytes()).hexdigest();assert hashes[name]==row['image_sha256']
receipt={'status':'passed','gpu':e['gpu_name'],'stage_calls_per_generation':dict(counts),'image_sha256':hashes,'scope':'Full1+3+10 unprofiled protocol; all aggregate medians/min/max, calls and saved pixels checked. Successful runner also asserts50completed scheduler steps per image.'}
(p/'validation.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
