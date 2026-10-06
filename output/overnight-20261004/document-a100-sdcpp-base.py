import json
from pathlib import Path
rid='a100-sdcpp-flux-klein-base-001__baseline__20261005-080118'
fid='a100-sdcpp-flux-klein-base-001__profile__20261005-081219'
s=json.loads((Path('results/runs')/rid/'summary.json').read_text());m=s['measured']
p=Path('experiments/a100-sdcpp-flux-klein-base-001.md');t=p.read_text().replace('20261005-040120]','20261005-040120, '+rid+', '+fid+']').replace('Full baseline11882493 and separate profile11882497 queued.','Full baseline11882493 passed; profile11882497 failed before generation with a Nsight process-probe timeout. A separately recorded retry11883400 is queued.')
sec=f'''## Full unprofiled baseline

[Run](../results/runs/{rid}/), [summary](../results/runs/{rid}/summary.json), [validation](../results/runs/{rid}/validation.json), [environment](../results/runs/{rid}/environment.json). Job11882493 on gilbreth-g007 ran12:01:18–12:11:49UTC. One first generation,three discarded warm-ups,ten measured generations; same pinned workload and no cache/offload/tiling.

| Metric | Median, ms | Measured min–max, ms |
|---|---:|---:|
'''
for k in ['wall_ms','text_encode_ms','denoise_ms','vae_decode_ms','other_ms']:
 v=m[k];sec+=f"| {k} | {v['median']:.3f} | {v['min']:.3f}–{v['max']:.3f} |\n"
sec+=f'''
First generation{s['first_run']['wall_ms']:.3f}ms; eager context loading{s['load']['load_total_s']:.6f}s (filesystem cache may be warm). Device-wide sampled generation peak{m['device_used_peak_gib']['max']:.6f}GiB; harness host RSS peak{s['host_peak_rss_gib']:.6f}GiB. Allocator counters are unavailable. All14recorded image hashes agree; both retained1024×1024PNGs independently rehashed. Every image has51progress callbacks and100transformer passes across50scheduler steps. Settings, stage accounting, phase order and all reported measured medians/min/max passed independent checks. This single development prompt/seed is not formal quality evidence.

[W&B baseline](https://wandb.ai/mfaruqi-purdue-university/on-device-diffusion/runs/{rid}).

## Failed separate profiler attempt

[Run](../results/runs/{fid}/), [status and traceback](../results/runs/{fid}/status.json), [profiler output](../results/runs/{fid}/engine/stdout.txt). Job11882497 on gilbreth-g007 failed12:12:24UTC, before model loading/generation produced any measurements. Nsight Systems2024.4.2 reported `Failed to probe the process (sync). Timeout: 2 sec`. No latency, memory or image result is inferred from this failed attempt. Its config, command and environment remain intact.

Hypothesis: the failure concerns profiler startup/environment; the available message does not establish its cause. The harness shared-library check resolved all dependencies. No model, precision or resolution fallback was attempted.

[W&B failed profile](https://wandb.ai/mfaruqi-purdue-university/on-device-diffusion/runs/{fid}).

'''
t=t.replace('## Interpretation',sec+'## Interpretation').replace('Two observations cannot establish a repeated warm timing distribution; the full protocol and separate profile remain pending.','The full protocol establishes the repeated warm timing distribution for this configuration; kernel profiling remains incomplete because the profiler failed before generation.').replace('- Complete configs/a100-sdcpp-flux-klein-base-bf16.resolved.json with1+3+10, then its separate Nsight profile.','- Retry the unchanged profile config in job11883400 on a PCIe A100 node excluding g007/g006; preserve the failed attempt.')
p.write_text(t)
p=Path('results/README.md');p.write_text(p.read_text()+f"| `{rid}` | [a100-sdcpp-flux-klein-base-001](../experiments/a100-sdcpp-flux-klein-base-001.md) | A100 PCIe,g007 | a100-sdcpp-flux-klein-base-bf16 | {m['wall_ms']['median']:.3f} | {m['text_encode_ms']['median']:.3f} / {m['denoise_ms']['median']:.3f} / {m['vae_decode_ms']['median']:.3f} | – (device {m['device_used_peak_gib']['max']:.3f}) | complete11882493; validated1+3+10 |\n"+f"| `{fid}` | [a100-sdcpp-flux-klein-base-001](../experiments/a100-sdcpp-flux-klein-base-001.md) | A100 PCIe,g007 | a100-sdcpp-flux-klein-base-bf16 + Nsight | – | – | – | failed11882497; Nsight process-probe timeout before generation |\n")
p=Path('wiki/experiments.md');t=p.read_text();lines=t.splitlines()
for i,l in enumerate(lines):
 if l.startswith('| a100-sdcpp-flux-klein-base-001 |'):
  cells=l.split('|');cells[5]=' partial; baseline validated, profile startup failed ';cells[7]+=f', [baseline](../results/runs/{rid}/), [failed profile](../results/runs/{fid}/) ';lines[i]='|'.join(cells)
p.write_text('\n'.join(lines)+'\n')
p=Path('wiki/log.md');t=p.read_text().replace('# Log\n','# Log\n\n## [2026-10-05] experiment | A100 sd.cpp Base baseline validated; profile startup failed\n[Base001](../experiments/a100-sdcpp-flux-klein-base-001.md) completed1+3+10 with CFG/settings/image checks. Nsight failure preserved; one unchanged-workload retry queued on a different PCIe node.\n',1);p.write_text(t)
