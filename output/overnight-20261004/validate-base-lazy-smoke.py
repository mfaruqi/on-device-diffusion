import json,csv,statistics,hashlib
from pathlib import Path
from PIL import Image
from run_jetson_sdcpp import check_config,audit_execution
from jetson_profile import validate_callbacks
p=Path('results/runs/jetson-flux-klein-base-003__attempt__20261005-054530')
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
for g in gens:
    stamps=[float(x) for x in re.findall(r'^([0-9.]+).*flux executing segment 1/1: graph',log,re.MULTILINE)]
    assert sum(g['t_start'] <= t <= g['t_end'] for t in stamps)==100
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
assert s['engine_audit']['requested']['eager_load'] is False
ref=Path('results/runs/jetson-flux-klein-base-001__repeat__20261005-042014/images/measured-0.png')
assert Image.open(ref).convert('RGB').tobytes()==Image.open(p/'images/measured-0.png').convert('RGB').tobytes()
(p/'lazy-validation.json').write_text(json.dumps({'status':'passed','reference':ref.parent.parent.name,'rgb_equal':True,'scope':'Two-generation lazy-loading functional attempt; not repeated timing evidence. Callback, CFG passes, settings, summary aggregates, phase counts and saved RGB hashes independently checked.'},indent=2)+'\n')
print('Lazy-loading correctness passed; saved RGB pixels identical to reference.')
assert s['cache_audit']=={'requested':None,'enabled_generations':0}
