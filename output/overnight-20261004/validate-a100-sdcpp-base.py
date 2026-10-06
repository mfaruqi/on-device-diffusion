import json,csv,re,hashlib,statistics,sys
from pathlib import Path
from PIL import Image
from jetson_profile import validate_callbacks
from sdcpp_engine import audit_log,audit_cache
p=Path(sys.argv[1] if len(sys.argv)>1 else 'results/runs/a100-sdcpp-flux-klein-base-001__attempt__20261005-040120')
cfg=json.loads((p/'config.json').read_text());s=json.loads((p/'summary.json').read_text())
recs=[json.loads(x) for x in (p/'engine/results.jsonl').read_text().splitlines()]; loads=[x for x in recs if x['event']=='load'];gens=[x for x in recs if x['event']=='generate']
phases=['first']*cfg['protocol']['first_runs']+['warmup']*cfg['protocol']['warmup_runs']+['measured']*cfg['protocol']['measured_runs']
assert json.loads((p/'status.json').read_text())['status']=='complete'
assert len(loads)==1 and len(gens)==len(phases)
log=(p/'engine/sdcpp.log').read_text()
audit=audit_log(log,cfg['engine_settings'],gens[-1]['t_end']);assert not audit['problems'];audit_cache(log,None,len(phases))
assert 'txt_cfg: 4.00' in log and 'sample_steps: 50' in log and 'get_sigmas with Flux2 scheduler' in log
stamps=[float(x) for x in re.findall(r'^([0-9.]+).*flux executing segment 1/1: graph',log,re.MULTILINE)]
for i,g in enumerate(gens):
    validate_callbacks([loads[0],g],50)
    assert g['run_index']==i and (g['width'],g['height'],g['channels'])==(1024,1024,3)
    assert sum(g['t_start']<=t<=g['t_end'] for t in stamps)==100
rows=list(csv.DictReader((p/'runs.csv').open())); assert [r['phase'] for r in rows]==phases
for row,g in zip(rows,gens):
    assert abs(float(row['wall_ms'])-(g['t_end']-g['t_start'])*1000)<1e-6
    assert abs(float(row['denoise_ms'])-sum(float(row[f'denoise_step_{i}_ms']) for i in range(50)))<1e-6
    assert abs(float(row['wall_ms'])-sum(float(row[k]) for k in ['text_encode_ms','denoise_ms','vae_decode_ms','other_ms']))<1e-6
for key,stats in s['measured'].items():
    if stats is None: continue
    vals=[float(r[key]) for r in rows if r['phase']=='measured']
    for field,val in [('median',statistics.median(vals)),('min',min(vals)),('max',max(vals))]: assert abs(stats[field]-val)<1e-6
hashes=set(r['image_sha256'] for r in rows); assert len(hashes)==1
for f in (p/'images').glob('*.png'):
    im=Image.open(f).convert('RGB');assert im.size==(1024,1024);assert hashlib.sha256(im.tobytes()).hexdigest() in hashes
(p/'validation.json').write_text(json.dumps({'status':'passed','generations':len(phases),'steps_each':50,'cfg_transformer_passes_each':100,'image_dimensions':[1024,1024],'rgb_hashes':sorted(hashes),'engine_audit':audit,'scope':'Protocol from saved config: callback, CFG, image, phase and CSV/summary arithmetic independently checked. Only retained PNGs can be rehashed; other generation hashes are recorded by the runner.'},indent=2)+'\n')
print('A100 sd.cpp Base correctness passed',hashes)
