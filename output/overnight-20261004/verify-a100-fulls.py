import json,wandb
from datetime import datetime,timezone
names=['a100-sdcpp-flux-klein-001__baseline__20261005-220435', 'a100-sdcpp-flux-klein-002__baseline__20261005-220636', 'a100-sdcpp-flux-klein-003__baseline__20261005-220737', 'a100-sdcpp-flux-klein-004__baseline__20261005-220938', 'a100-sdcpp-flux-klein-005__baseline__20261005-221039', 'a100-sdcpp-flux-klein-base-001__baseline__20261005-214624', 'a100-sdcpp-flux-klein-base-002__baseline__20261005-215731', 'a100-edgedit-flux-klein-001__attempt__20261005-221339']
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
 if '-002__' in n or any(k in n for k in ['klein-003__','klein-004__','klein-005__']):
  assert any(x.endswith('quality-diagnostics.json') for x in entries),n
 out.append({'run_id':n,'state':r.state,'baseline_comparison_eligible':base_run,'generate_samples':s.get('timing/generate_samples'),'artifact_files':entries})
print(json.dumps({'verified_utc':datetime.now(timezone.utc).isoformat(),'runs':out},indent=2))
