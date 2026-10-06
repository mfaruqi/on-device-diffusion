import json
from pathlib import Path
import wandb
root=Path(__file__).resolve().parent
entity='mfaruqi-purdue-university'; project='on-device-diffusion'; run_id='jetson-osrt-analysis-20261005'
api=wandb.Api(timeout=30)
existing=next((r for r in api.runs(entity+'/'+project) if r.id==run_id),None)
if existing:
 print(existing.url)
 raise SystemExit(0)
settings=wandb.Settings(x_disable_meta=True,x_disable_stats=True,x_disable_machine_info=True,disable_git=True,disable_code=True,disable_job_creation=True)
run=wandb.init(entity=entity,project=project,id=run_id,name='Jetson captured read/GPU coverage',job_type='analysis',group='jetson-flux-klein-003',tags=['analysis','nsight-systems','jetson-orin-nano'],settings=settings,config={'measurement_scope':'profiled diagnostics','baseline_comparison_eligible':False,'method':'Process-restricted GPU/read/pread64 interval unions clipped to NVTX stages. Read-only is not physical storage wait; neither is not necessarily idle.','generation_selection':'last captured generation of each report','source_runs':[p.name for p in (root/'jetson/analysis').iterdir()]})
rows=[]
artifact=wandb.Artifact('jetson-osrt-analysis-20261005',type='profile-analysis')
for directory in sorted((root/'jetson/analysis').iterdir()):
 for p in directory.iterdir():artifact.add_file(str(p),name=directory.name+'/'+p.name)
 p=directory/'stage_kernels_osrt.json'
 if not p.exists():continue
 d=json.loads(p.read_text())
 for name,s in d['stages'].items():
  c=s['osrt_coverage']
  rows.append([directory.name,name,s['span_ms'],c['gpu_covered_ms'],c['read_only_covered_ms'],c['neither_covered_ms']])
run.log({'stage_coverage':wandb.Table(columns=['capture','stage','stage_span_ms','gpu_covered_ms','read_only_covered_ms','neither_covered_ms'],data=rows)})
run.summary.update({'analyzed_captures':3,'captures_without_stage_markers':1,'baseline_comparison_eligible':False,'timing_scope':'single selected generation per profiled capture'})
run.log_artifact(artifact)
url=run.url;run.finish()
(root/'osrt-wandb-url.txt').write_text(url)
print(url)
