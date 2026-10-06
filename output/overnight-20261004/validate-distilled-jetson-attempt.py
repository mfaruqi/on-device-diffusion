import sys,json,csv,hashlib,statistics,re
from pathlib import Path
from PIL import Image
from run_jetson_sdcpp import check_config,audit_execution
from jetson_profile import validate_callbacks,expected_conditioning_hits
from sdcpp_engine import audit_cache
p=Path(sys.argv[1]);cfg=json.loads((p/'config.json').read_text());assert json.loads((p/'status.json').read_text())['status']=='complete';assert check_config(cfg)==['first','measured']
records=[json.loads(x) for x in (p/'engine/results.jsonl').read_text().splitlines()];loads=[x for x in records if x['event']=='load'];gens=[x for x in records if x['event']=='generate'];assert len(loads)==1 and len(gens)==2
assert cfg['workload']['steps']==4 and cfg['workload']['cfg_scale']==1
log=(p/'engine/sdcpp.log').read_text();audit_execution(log,cfg);ca=audit_cache(log,cfg['optimizations'].get('step_cache'),2)
stamps=[float(x) for x in re.findall(r'^([0-9.]+).*flux executing segment 1/1: graph',log,re.MULTILINE)]
for i,g in enumerate(gens):
 validate_callbacks([loads[0],g],4,expected_conditioning_hits(cfg,i));assert g['run_index']==i
 skips=ca.get('steps_skipped',[0,0])[i];assert sum(g['t_start']<=t<=g['t_end'] for t in stamps)==4-skips
 assert (g['width'],g['height'],g['channels'])==(512,512,3)
rows=list(csv.DictReader((p/'runs.csv').open()));s=json.loads((p/'summary.json').read_text());assert [r['phase'] for r in rows]==['first','measured']
for key,stats in s['measured'].items():
 if stats is None:continue
 v=float(rows[1][key]);assert all(abs(stats[k]-v)<1e-6 for k in ['median','min','max'])
for f,row in [('first.png',rows[0]),('measured-0.png',rows[1])]:
 im=Image.open(p/'images'/f).convert('RGB');assert im.size==(512,512);assert hashlib.sha256(im.tobytes()).hexdigest()==row['image_sha256']
ref=Path(sys.argv[2]);equal=Image.open(ref/'images/measured-0.png').convert('RGB').tobytes()==Image.open(p/'images/measured-0.png').convert('RGB').tobytes()
assert equal, 'Saved pixels differ: evaluate quality separately before promotion'
assert s['cache_audit']==ca
(p/'validation.json').write_text(json.dumps({'status':'passed','scope':'Two-generation functional attempt; callback/CFG/settings/phase/summary/image checks, no full performance conclusion.','steps':4,'transformer_passes':[4-v for v in ca.get('steps_skipped',[0,0])],'conditioning_hits':[g['cond_cache_hits'] for g in gens],'reference':ref.name,'reference_rgb_equal':equal,'formal_quality_eligible':False},indent=2)+'\n')
print(p.name,'validated',ca)
