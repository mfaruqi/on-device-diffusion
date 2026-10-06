import json,wandb
from datetime import datetime,timezone
names=['jetson-flux-klein-base-007__baseline__20261005-121955', 'jetson-flux-klein-base-006__attempt__20261005-125535']
api=wandb.Api();out=[]
for n in names:
 r=api.run('mfaruqi-purdue-university/on-device-diffusion/'+n);s=dict(r.summary);base='__baseline__' in n
 assert r.state=='finished',(n,r.state)
 assert s['timing/baseline_comparison_eligible']==base
 assert s['timing/generate_samples']==(10 if base else 1)
 entries=[]
 for a in r.logged_artifacts():
  if a.type=='benchmark-run':entries+=list(a.manifest.entries)
 assert any(x.endswith('validation.json') for x in entries),(n,entries)
 if '003__attempt' not in n:assert any(x.endswith('quality-diagnostics.json') for x in entries),(n,entries)
 out.append({'run_id':n,'state':r.state,'baseline_comparison_eligible':base,'generate_samples':s['timing/generate_samples'],'artifact_files':entries})
print(json.dumps({'verified_utc':datetime.now(timezone.utc).isoformat(),'runs':out},indent=2))
