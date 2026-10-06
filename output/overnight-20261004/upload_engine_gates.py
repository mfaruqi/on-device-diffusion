"""Export the two retained engine feasibility captures via the existing exporter."""
import json
import sys
from pathlib import Path
sys.path.insert(0, 'scripts')
from export_wandb import export
import wandb
entity='mfaruqi-purdue-university';project='on-device-diffusion'
existing={r.id:r for r in wandb.Api().runs(f'{entity}/{project}')}
for rid in sys.argv[1:] or ['a100-edgedit-flux-klein-001__attempt__20261005-073914','a100-flux-klein-torchtrt-001__attempt__20261005-074014']:
    if rid in existing:
        print('Already exported:',rid,existing[rid].state);continue
    p=(Path('results/runs')/rid).resolve()
    c=json.loads((p/'config.json').read_text());s=json.loads((p/'status.json').read_text());e=json.loads((p/'environment.json').read_text())
    failed=s['status']=='failed'
    payload={'id':rid,'name':c['id']+' feasibility '+s['status'],'group':c['id'],'job_type':'attempt','tags':['feasibility','not-baseline','failed' if failed else 'complete'],
        'config':{'source_config':c,'environment':e,'run_dir':rid,'baseline_comparison_eligible':False,'formal_quality_eligible':False},
        'notes':'One functional engine/compiler gate; no repeated baseline metrics. See original logs and validation receipts.',
        'history':[],'tables':{},'images':list(p.glob('*.png')),'files':[f for f in p.iterdir() if f.suffix in {'.json','.log','.txt'}],
        'run_dir':rid,'failed':failed,'summary':{'timing/baseline_comparison_eligible':False,'quality/formal_eligibility':False,'gate/status':s['status'],'gate/phase':s.get('phase','image'),'gate/error':s.get('error','')}}
    print(export(payload,project,entity))
