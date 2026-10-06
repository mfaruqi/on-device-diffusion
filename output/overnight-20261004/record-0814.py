import json
from pathlib import Path
from datetime import datetime,timezone
root=Path('results/runs');j='jetson-flux-klein-base-001__profile__20261005-033141';a='a100-sdcpp-flux-klein-base-001__attempt__20261005-040120'
s=json.loads((root/j/'summary.json').read_text());r=json.loads((root/j/'profile/osrt-analysis/stage_kernels.json').read_text())
for x in r['stages'].values():
 c=x['osrt_coverage'];assert abs(sum(c[k] for k in ['gpu_covered_ms','read_only_covered_ms','neither_covered_ms'])-x['span_ms'])<1e-6
p=Path('experiments/jetson-flux-klein-base-001.md');t=p.read_text().replace('runs: [','runs: ['+j+', ',1)
section=f'''## Targeted Base profile

[Run](../results/runs/{j}/), [capture scope](../results/runs/{j}/profile/capture.json), [validation](../results/runs/{j}/profile-validation.json), [derived process-restricted analysis](../results/runs/{j}/profile/osrt-analysis/stage_kernels.json). Nsight Systems traces generation4 after one first and three warm-up generations. The profiler was attached throughout; initial model loading is outside the captured range. The profile is separate from unprofiled baseline timing.

The captured generation took {s['measured']['wall_ms']['median']/1000:.3f}s by host callbacks. Whole-capture sampled system RAM peaked at {s['monitor_window']['ram_peak_gib']:.3f}GiB, including loading and untraced generations. All five generations completed fifty steps and one hundred transformer passes; saved RGB pixels match the unprofiled no-cache reference. SQLite integrity, single captured generation, stage label counts and summary accounting passed.

| Captured stage | NVTX span, ms | GPU coverage, ms | Read calls without GPU coverage, ms | Neither covered, ms |
|---|---:|---:|---:|---:|
'''
for n in ['text_encode','denoise_step_0','denoise_step_1','vae_decode']:
 x=r['stages'][n];c=x['osrt_coverage'];section+=f"| {n} | {x['span_ms']:.3f} | {c['gpu_covered_ms']:.3f} | {c['read_only_covered_ms']:.3f} | {c['neither_covered_ms']:.3f} |\n"
section+='\nIntervals are restricted to the traced process and clipped to stage windows. The three coverage categories sum to each stage span. Read-call coverage means time inside captured read/pread64 calls without overlapping captured GPU activity; it is not measured physical disk wait. GPU activity coverage is not SM utilization, and the uncovered remainder is not proof of idle hardware.\n\nInterpretation: this capture separates read-call-heavy text encoding and the first denoising step from GPU-covered later denoising. It supports investigating residency transitions independently of later-step compute; it does not establish a single whole-pipeline bandwidth or compute bottleneck.\n\n'
t=t.replace('## Interpretation\n',section+'## Interpretation\n',1).replace('- Validate a no-cache control on the cache-capable harness, then test configs/jetson-flux-klein-base-q4-512-easycache-smoke.json.','- Complete the same-harness no-cache repeat of configs/jetson-flux-klein-base-q4-512-disk.json before selecting combinations.')
t+=f'\n[W&B profile](https://wandb.ai/mfaruqi-purdue-university/on-device-diffusion/runs/{j}).\n';p.write_text(t)
s=json.loads((root/a/'summary.json').read_text());e=json.loads((root/a/'environment.json').read_text());cfg=json.loads((root/a/'config.json').read_text())
t=f'''---
type: experiment-record
id: a100-sdcpp-flux-klein-base-001
status: partial
device: a100-pcie-40gb
engine: stable-diffusion-cpp
runs: [{a}]
updated: 2026-10-05
---

# A100 sd.cpp klein Base: fifty-step BF16 reference

Status: **partial**; correctness attempt completed and validated, job11880759. Full baseline11882493 and separate profile11882497 queued.

## Question

Can pinned klein Base4B execute at BF16,1024²,50steps and guidance4 with fixed GPU parameter placement and correct CFG callback accounting? This is Week1–2 image baseline evidence for RQ1, a separate checkpoint/workload reference.

## Setup

[Saved resolved config](../results/runs/{a}/config.json). Checkpoint black-forest-labs/FLUX.2-klein-base-4B at {cfg['model']['revision']}; BF16 transformer/text encoder/VAE, no quantization or reuse. Batch1,1024×1024,50steps, guidance4, seed0, fixed cat/sign prompt. Effective Flux2 scheduler is confirmed in engine logs. Eager CUDA0 parameters and execution, automatic fitting and segmented computation disabled, conditioning cache zero, diffusion flash attention enabled, four CPU threads. Protocol:one first plus one additional observation, no warm-ups; not a repeated baseline.

Node {e['hostname']}, NVIDIA A100-PCIE-40GB. sd.cpp {cfg['engine']['commit']}, ggml {cfg['engine']['ggml_commit']}; isolated runner snapshot {e['git_commit']}. [Environment](../results/runs/{a}/environment.json), [metric definitions](../wiki/methods/baseline-metrics.md). Stage timing uses host callbacks around synchronous ggml execution, not CUDA-event stage measurements.

## Results

[Run](../results/runs/{a}/), [summary](../results/runs/{a}/summary.json), [status](../results/runs/{a}/status.json).

| Metric | First, ms | Second observation, ms |
|---|---:|---:|
'''
for k in ['wall_ms','text_encode_ms','denoise_ms','vae_decode_ms','other_ms']:t+=f"| {k} | {s['first_run'][k]:.3f} | {s['measured'][k]['median']:.3f} |\n"
t+=f'''
Context creation: {s['load']['load_total_s']:.3f}s; filesystem page cache may be warm. Device-wide sampled peak during the second generation: {s['measured']['device_used_peak_gib']['median']:.3f}GiB. Harness peak host RSS: {s['host_peak_rss_gib']:.3f}GiB. Device-wide usage is not a PyTorch allocated/reserved metric; those are unavailable here.

## Validation and observations

[Validation receipt](../results/runs/{a}/validation.json). Both generations completed50steps with51progress callbacks and100single-segment transformer passes, consistent with CFG. Engine audit found BF16 weights, CUDA0 parameter placement and no forbidden release/offload behavior. Callback boundaries, stage-plus-other accounting, phase order, summary aggregates and PNG hashes were independently checked.

Both1024×1024saved images have identical RGB hashes. Visual inspection found a coherent cat holding a readable hello world sign with no obvious gross corruption. This one-prompt/seed check is functional evidence, not formal quality eligibility.

## Interpretation

The specified Base configuration fits and passes the correctness gate. Denoising dominates the observed duration. Two observations cannot establish a repeated warm timing distribution; the full protocol and separate profile remain pending.

## Next experiment

- Complete configs/a100-sdcpp-flux-klein-base-bf16.resolved.json with1+3+10, then its separate Nsight profile.

[W&B attempt](https://wandb.ai/mfaruqi-purdue-university/on-device-diffusion/runs/{a}).
''';Path('experiments/a100-sdcpp-flux-klein-base-001.md').write_text(t)
p=Path('results/README.md');t=p.read_text();t+=f'\n| `{a}` | [a100-sdcpp-flux-klein-base-001](../experiments/a100-sdcpp-flux-klein-base-001.md) | A100-PCIE-40GB | a100-sdcpp-flux-klein-base-bf16-smoke | – | – | – | complete; job11880759, two-observation correctness attempt |\n';t=t.replace('## A100 Base reference',f'| `{j}` | [jetson-flux-klein-base-001](../experiments/jetson-flux-klein-base-001.md) | Jetson Orin Nano,25W | jetson-flux-klein-base-q4-512-disk-profile | – | – | – | complete; targeted generation4 profile, separate from baseline |\n\n## A100 Base reference');p.write_text(t)
p=Path('wiki/experiments.md');t=p.read_text();lines=t.splitlines();lines=[line.replace(' | [W&B]',f', [profile](../results/runs/{j}/) | [W&B]',1) if line.startswith('| jetson-flux-klein-base-001 |') else line for line in lines];t='\n'.join(lines)+'\n';t=t.replace('## Comparisons',f'| a100-sdcpp-flux-klein-base-001 | A100-PCIE-40GB | stable-diffusion.cpp | Base fifty-step BF16 reference | partial; correctness passed, baseline/profile queued | [record](../experiments/a100-sdcpp-flux-klein-base-001.md) | [attempt](../results/runs/{a}/) | [W&B](https://wandb.ai/mfaruqi-purdue-university/on-device-diffusion/runs/{a}) |\n\n## Comparisons');p.write_text(t)
p=Path('wiki/log.md');t=p.read_text();pos=t.index('\n## ');t=t[:pos]+'''\n## [2026-10-05] experiment | Base profile and sd.cpp correctness validated
[Jetson Base](../experiments/jetson-flux-klein-base-001.md): targeted profile validated and read/GPU coverage derived. [A100 sd.cpp Base](../experiments/a100-sdcpp-flux-klein-base-001.md): two-image correctness gate passed; full baseline and separate profile queued.
'''+t[pos:];p.write_text(t)
p=Path('output/overnight-20261004/state.json');d=json.loads(p.read_text())
for x in d['jobs']:
 if x.get('job_id')=='11880759':x.update(status='complete; independent callback/CFG/image/settings validation passed; recorded; W&B upload pending',run_dir=a)
 if x.get('session')=='diffusion-base-profile':x['status']='complete08:00:20UTC; independent image/callback/CFG/SQLite/NVTX checks passed; process-restricted analysis saved; W&B upload pending'
 if x.get('job_id')=='11881347':x['status']='pending resources; preceding sd.cpp gate complete'
d['jobs'] += [dict(host='jetson',root=d['jetson_options_root'],session='diffusion-base-repeat',name='Base full no-cache repeat on cache-capable harness',log='logs/base-repeat.log',config='configs/jetson-flux-klein-base-q4-512-disk.json',status='running; started08:20:14UTC, process11821',run_dir='jetson-flux-klein-base-001__repeat__20261005-042014'),dict(host='gilbreth',job_id='11882493',name='base-sdcpp-baseline',status='pending afterany:11882223',config='configs/a100-sdcpp-flux-klein-base-bf16.resolved.json',root=d['gilbreth_root']),dict(host='gilbreth',job_id='11882497',name='base-sdcpp-profile',status='pending afterok:11882493',config='configs/a100-sdcpp-flux-klein-base-bf16.resolved.json',root=d['gilbreth_root'],profile=True)]
d['status']='jetson-matched-reference-repeat-running-a100-baselines-and-engine-gates-queued';d['last_checked_utc']=datetime.now(timezone.utc).isoformat();d['last_check']='Profile and sd.cpp correctness validated. Same-harness Jetson repeat launched; sd.cpp full baseline/profile appended to serial Slurm chain. W&B exports pending.'
d['next_actions']=[x for x in d['next_actions'] if not x.startswith(('Monitor Base profile','Monitor A100 sd.cpp correctness','After current Base profile'))]
d['next_actions'][:0]=['Monitor Jetson no-cache repeat042014; validate and compare against full EasyCache024556 on identical unprofiled binary before combination selection.','Monitor full A100 PyTorch baseline/profile11882219/11882223 and sd.cpp11882493/11882497; validate, document and export completions.','Finish W&B uploads for profile033141 and sd.cpp attempt040120; verify API state/eligibility.']
d['pending'][0]='A100 PyTorch and sd.cpp Base full baselines/profiles queued';d['pending'][1]='Jetson same-harness Base no-cache repeat, individual memory options and qualifying combinations'
p.write_text(json.dumps(d,indent=2)+'\n')
