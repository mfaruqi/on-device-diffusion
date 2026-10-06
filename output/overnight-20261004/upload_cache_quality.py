import json
import sys
from pathlib import Path
sys.path.insert(0, 'scripts')
from export_wandb import export
import wandb
source='jetson-flux-klein-base-002__attempt__20261005-023028'
p=(Path('results/runs')/source).resolve()
q=json.loads((p/'quality-diagnostics.json').read_text())
run_id='jetson-base-easycache-quality-20261005'
entity='mfaruqi-purdue-university';project='on-device-diffusion'
existing={r.id for r in wandb.Api().runs(f'{entity}/{project}')}
payload={'id':run_id,'name':'Jetson Base EasyCache paired-image diagnostic','group':'jetson-flux-klein-base-002','job_type':'analysis','tags':['quality-diagnostic','easycache','jetson-orin-nano'],
 'config':{'source_run':source,'reference_image':q['reference'],'variant_image':q['variant'],'evaluation_device':'Mac CPU','versions':q['versions'],'method':q['method'],'weights_sha256':q['weights_sha256'],'formal_quality_eligible':False},
 'notes':q['scope'],'history':[],'tables':{},'images':[Path(q['reference']),Path(q['variant'])],
 'files':[p/'quality-diagnostics.json'],'run_dir':source,'failed':False,
 'summary':{'timing/baseline_comparison_eligible':False,'quality/formal_eligibility':False,'quality/psnr_db':q['psnr_db'],'quality/lpips_alex_v0_1':q['lpips_alex_v0_1'],'quality/lpips_self_distance':q['lpips_self_distance'],'quality/image_pairs':1}}
if run_id in existing:
    old=wandb.Api().run(f'{entity}/{project}/{run_id}')
    if old.state=='finished' and old.summary.get('quality/lpips_alex_v0_1')==q['lpips_alex_v0_1']:
        print('Quality diagnostic already complete; no duplicate.')
    else:
        settings=wandb.Settings(x_disable_meta=True,x_disable_stats=True,x_disable_machine_info=True,disable_git=True,disable_code=True,disable_job_creation=True)
        resumed=wandb.init(entity=entity,project=project,id=run_id,resume='must',settings=settings)
        artifact=wandb.Artifact(name=f'run-{run_id}',type='benchmark-run')
        artifact.add_file(str(p/'quality-diagnostics.json'),name='quality-diagnostics.json')
        resumed.log_artifact(artifact)
        resumed.summary.update(payload['summary'])
        resumed.summary['export/recovery']='Initial export artifact failed on relative-vs-absolute path; resumed same run with absolute path, no benchmark rerun.'
        resumed.finish()
        print(resumed.url)
else:
    print(export(payload,project,entity))
