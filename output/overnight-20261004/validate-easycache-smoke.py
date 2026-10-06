import json,csv,statistics,hashlib
from pathlib import Path
from PIL import Image
from run_jetson_sdcpp import check_config,audit_execution
from jetson_profile import validate_callbacks
p=Path('results/runs/jetson-flux-klein-base-002__attempt__20261005-023028')
cfg=json.loads((p/'config.json').read_text()); phases=check_config(cfg)
recs=[json.loads(x) for x in (p/'engine/results.jsonl').read_text().splitlines()]
loads=[x for x in recs if x['event']=='load']; gens=[x for x in recs if x['event']=='generate']
assert len(loads)==1 and len(gens)==2
end=loads[0]['t_end']
for i,g in enumerate(gens):
    validate_callbacks([loads[0],g],50)
    assert g['run_index']==i and g['t_start']>=end
    assert (g['width'],g['height'],g['channels'])==(512,512,3)
    end=g['t_end']
audit_execution((p/'engine/sdcpp.log').read_text(),cfg)
import re
log=(p/'engine/sdcpp.log').read_text()
assert 'txt_cfg: 4.00' in log and 'sample_steps: 50' in log
call_counts=[]
for g in gens:
    stamps=[float(x) for x in re.findall(r'^([0-9.]+).*flux executing segment 1/1: graph',log,re.MULTILINE)]
    call_counts.append(sum(g['t_start'] <= t <= g['t_end'] for t in stamps))
    assert call_counts[-1]==50
rows=list(csv.DictReader((p/'runs.csv').open()))
s=json.loads((p/'summary.json').read_text())
assert [r['phase'] for r in rows]==['first','measured']
assert len(set(r['image_sha256'] for r in rows))==1
for key,stats in s['measured'].items():
    if stats is None: continue
    vals=[float(r[key]) for r in rows if r['phase']=='measured']
    for field,val in [('median',statistics.median(vals)),('min',min(vals)),('max',max(vals))]:
        assert abs(stats[field]-val)<1e-6
for f in (p/'images').glob('*.png'):
    im=Image.open(f).convert('RGB')
    assert im.size==(512,512)
    assert hashlib.sha256(im.tobytes()).hexdigest()==rows[0]['image_sha256']
assert s['engine_audit']['requested']['eager_load'] is True
ref=Path('/home/mfaruqi/on-device-diffusion-campaigns/20261004-overnight/results/runs/jetson-flux-klein-base-001__baseline__20261005-005357/images/measured-0.png')
from sdcpp_engine import audit_cache
import math
cache=audit_cache(log,cfg['optimizations']['step_cache'],len(gens))
a=Image.open(ref).convert('RGB').tobytes();b=Image.open(p/'images/measured-0.png').convert('RGB').tobytes()
mse=sum((x-y)**2 for x,y in zip(a,b))/len(a)
receipt={'status':'passed','reference':ref.parent.parent.name,'rgb_equal':a==b,'psnr_db':10*math.log10(255**2/mse) if mse else None,'mse_rgb_8bit':mse,'lpips':None,'lpips_unavailable_reason':'No validated LPIPS evaluator in this capture environment; retain as pending diagnostic before selecting combinations.','cache_audit':cache,'transformer_executions':call_counts,'scope':'One prompt/seed, two observations; functional and paired-pixel diagnostic only. No formal quality eligibility or repeated timing claim.'}
(p/'cache-validation.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
