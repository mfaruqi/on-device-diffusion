import json,wandb
from datetime import datetime,timezone
names=['a100-sdcpp-flux-klein-base-002__attempt__20261005-191624', 'a100-sdcpp-flux-klein-001__attempt__20261005-193229', 'a100-sdcpp-flux-klein-002__attempt__20261005-193329', 'a100-sdcpp-flux-klein-003__attempt__20261005-193430', 'a100-sdcpp-flux-klein-004__attempt__20261005-193530', 'a100-sdcpp-flux-klein-005__attempt__20261005-193630', 'a100-sdcpp-flux-klein-base-001__profile__20261005-191522', 'a100-edgedit-flux-klein-001__attempt__20261005-191822']
api=wandb.Api();out=[]
for n in names:
 r=api.run('mfaruqi-purdue-university/on-device-diffusion/'+n);s=dict(r.summary)
 failed='__profile__' in n;edge='edgedit' in n
 assert r.state==('failed' if failed else 'finished'),(n,r.state)
 assert s['timing/baseline_comparison_eligible'] is False
 if not failed and not edge:assert s['timing/generate_samples']==1
 entries=[]
 for a in r.logged_artifacts():
  if a.type=='benchmark-run':entries+=list(a.manifest.entries)
 for required in (['status.json','environment.json','config.json'] if failed else ['validation.json','status.json']):
  assert any(x.endswith(required) for x in entries),(n,required)
 if '-002__' in n or any(k in n for k in ['klein-003__','klein-004__','klein-005__']):
  assert any(x.endswith('quality-diagnostics.json') for x in entries),n
 out.append({'run_id':n,'state':r.state,'baseline_comparison_eligible':False,'generate_samples':s.get('timing/generate_samples'),'artifact_files':entries})
print(json.dumps({'verified_utc':datetime.now(timezone.utc).isoformat(),'runs':out},indent=2))
