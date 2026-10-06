import json,csv,hashlib,statistics,re,sqlite3
from pathlib import Path
from PIL import Image
from run_jetson_sdcpp import check_config,audit_execution
from jetson_profile import validate_callbacks
from analyze_profile import analyze,render_md
p=Path('results/runs/jetson-flux-klein-base-001__profile__20261005-033141')
cfg=json.loads((p/'config.json').read_text()); phases=check_config(cfg)
recs=[json.loads(x) for x in (p/'engine/results.jsonl').read_text().splitlines()]
loads=[x for x in recs if x['event']=='load']; gens=[x for x in recs if x['event']=='generate']
assert len(loads)==1 and len(gens)==5
log=(p/'engine/sdcpp.log').read_text(); audit_execution(log,cfg)
assert 'txt_cfg: 4.00' in log and 'sample_steps: 50' in log
stamps=[float(x) for x in re.findall(r'^([0-9.]+).*flux executing segment 1/1: graph',log,re.MULTILINE)]
for i,g in enumerate(gens):
    validate_callbacks([loads[0],g],50)
    assert g['run_index']==i and g['profile_capture'] is (i==4)
    assert (g['width'],g['height'],g['channels'])==(512,512,3)
    assert sum(g['t_start'] <= t <= g['t_end'] for t in stamps)==100
rows=list(csv.DictReader((p/'runs.csv').open())); s=json.loads((p/'summary.json').read_text())
assert [r['phase'] for r in rows]==phases
for key,stats in s['measured'].items():
    if stats is None: continue
    vals=[float(r[key]) for r in rows if r['phase']=='measured']
    for field,val in [('median',statistics.median(vals)),('min',min(vals)),('max',max(vals))]: assert abs(stats[field]-val)<1e-6
ref=Path('results/runs/jetson-flux-klein-base-001__baseline__20261005-005357/images/measured-0.png')
raw=Image.open(ref).convert('RGB').tobytes(); digest=hashlib.sha256(raw).hexdigest()
assert all(r['image_sha256']==digest for r in rows)
for f in (p/'images').glob('*.png'): assert Image.open(f).convert('RGB').tobytes()==raw
with sqlite3.connect(p/'profile/trace.sqlite') as con:
    assert con.execute('pragma integrity_check').fetchone()[0]=='ok'
    names=[r[0] for r in con.execute('SELECT COALESCE(n.text,s.value) FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId=s.id WHERE n.end IS NOT NULL')]
    for name in ['generate','text_encode','vae_decode']+[f'denoise_step_{i}' for i in range(50)]: assert names.count(name)==1
    assert 'load' not in names
out=p/'profile/osrt-analysis'; out.mkdir(exist_ok=True)
report=analyze(p/'profile/trace.sqlite',r'text_encode|denoise_step_\d+|vae_decode',0)
(out/'stage_kernels.json').write_text(json.dumps(report,indent=2)+'\n'); (out/'stage_kernels.md').write_text(render_md(report))
(p/'profile-validation.json').write_text(json.dumps({'status':'passed','generations':5,'captured_generation':4,'trace_generations':1,'steps_each':50,'transformer_executions_each':100,'rgb_equal_reference':True,'reference':ref.parent.parent.name,'sqlite_integrity':'ok','scope':'One captured generation; no baseline timing claim. Stage/callback/settings/phase/hash/summary checks passed.','analysis':'profile/osrt-analysis/stage_kernels.json'},indent=2)+'\n')
print('Profile validation passed.'); print(json.dumps(report['stages']['denoise_step_1'],indent=2)[:1800])
