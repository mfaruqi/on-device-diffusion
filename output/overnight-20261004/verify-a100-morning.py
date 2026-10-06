import json,wandb
from datetime import datetime,timezone
names=['a100-edgedit-flux-klein-001__baseline__20261005-233237', 'a100-sdcpp-flux-klein-base-003__attempt__20261005-234246']
api=wandb.Api();out=[]
for n in names:
 r=api.run('mfaruqi-purdue-university/on-device-diffusion/'+n);s=dict(r.summary)
 failed=False;edge='edgedit' in n;base_run='__baseline__' in n
 assert r.state==('failed' if failed else 'finished'),(n,r.state)
 assert s['timing/baseline_comparison_eligible']==base_run
 assert s['timing/generate_samples']==(10 if base_run else 1)
 entries=[]
 for a in r.logged_artifacts():
  if a.type=='benchmark-run':entries+=list(a.manifest.entries)
 for required in (['status.json','environment.json','config.json'] if failed else ['validation.json','status.json']):
  assert any(x.endswith(required) for x in entries),(n,required)
 if not edge:
  assert any(x.endswith('quality-diagnostics.json') for x in entries),n
 out.append({'run_id':n,'state':r.state,'baseline_comparison_eligible':base_run,'generate_samples':s.get('timing/generate_samples'),'artifact_files':entries})
print(json.dumps({'verified_utc':datetime.now(timezone.utc).isoformat(),'runs':out},indent=2))
