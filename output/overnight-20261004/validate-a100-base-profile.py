import collections, hashlib, json, math
from pathlib import Path
p=Path('results/runs/a100-flux-klein-base-001__profile__20261005-075123')
r=json.loads((p/'profile/profiled_run.json').read_text())
a=json.loads((p/'profile/stage_kernels.json').read_text())['stages']
e=json.loads((p/'profile/trace.json').read_text())['traceEvents']
expected={'text_encode':2,'vae_decode':1,'postprocess':1,**{f'denoise_step_{i}':2 for i in range(50)}}
assert r['scheduler_steps']==50 and r['calls_per_step']==2
assert dict(collections.Counter(s['stage'] for s in r['stages']))==expected
for cat in ['user_annotation','gpu_user_annotation']:
 counts=collections.Counter(x['name'] for x in e if x.get('cat')==cat)
 assert dict(counts)=={'generate':1,**expected},(cat,counts)
assert set(a)==set(expected)
for name,v in a.items():
 span=sum(x['dur']/1000 for x in e if x.get('cat')=='gpu_user_annotation' and x['name']==name)
 assert abs(span-v['span_ms'])<1e-4
 assert math.isclose(v['busy_ms']+v['idle_ms'],span,abs_tol=1e-4)
 assert v['n_kernels']>0 and 0<=v['busy_ms']<=span+1e-4
steps=[v for k,v in a.items() if k.startswith('denoise_step_')]
total=sum(v['kernel_sum_ms'] for v in steps)
groups=collections.defaultdict(float)
for v in steps:
 for k,g in v['by_group'].items(): groups[k]+=g['ms']
out={'status':'passed','scope':'Separate fifteenth generation under torch.profiler; preceding 1+3+10 CSV protocol checked independently. The profiled image is not retained; image validation covers the preceding first/measured-0 PNGs only.', 'cpu_and_gpu_window_counts':expected,'trace_sha256':hashlib.sha256((p/'profile/trace.json').read_bytes()).hexdigest(),'profiled_wall_ms':r['wall_ms'],'denoise_kernel_sum_ms':total,'denoise_kernel_group_share_percent':{k:100*v/total for k,v in groups.items()},'note':'Group shares use summed captured kernel durations. Window coverage is not SM utilization, and these measurements do not alone distinguish compute from memory bandwidth limits.'}
(p/'profile-validation.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
