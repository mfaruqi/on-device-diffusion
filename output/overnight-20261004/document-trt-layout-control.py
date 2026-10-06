import json
from pathlib import Path
rid='a100-flux-klein-torchtrt-002__attempt__20261005-081316';root=Path('results/runs')/rid
load=json.loads((root/'load.json').read_text());cfg=json.loads((root/'config.json').read_text())
p=Path('experiments/a100-flux-klein-torchtrt-002.md')
p.write_text(f'''---
type: experiment-record
id: a100-flux-klein-torchtrt-002
status: failed
device: a100-pcie-40gb
engine: pytorch-diffusers
runs: [{rid}]
updated: 2026-10-05
---

# A100 Torch-TensorRT: BF16 conversion failure with contiguous inputs

Status: **failed**, job11883257 on gilbreth-g007,12:13:16–12:20:45UTC.

## Question

Can the real distilled klein transformer input compile at unchanged BF16 precision after explicitly making input tensors contiguous? Week2 engine feasibility, RQ1. The only integration change from the [original gate](a100-flux-klein-torchtrt-001.md) is a value-preserving input-layout copy; this is not a baseline or speed experiment.

## Setup

[Config](../results/runs/{rid}/config.json), [environment](../results/runs/{rid}/environment.json), [input shapes](../results/runs/{rid}/inputs.json). Pinned distilled klein revision e7b7dc27f91deacad38e78976d1f2b499d76a294,1024×1024,four steps,guidance1,seed0,BF16. Isolated torch2.5.1+cu124,Torch-TensorRT2.5.0,TensorRT10.3.0. Same-environment eager image precedes export/compilation. Five real transformer inputs are copied to contiguous layout; every copied tensor must equal its original. Original-layout eager forward is the intended numerical reference. Compile settings and precision are unchanged;1200second process bound.

## Results

[Status/traceback](../results/runs/{rid}/status.json), [compiler log](../results/runs/{rid}/compiler.log), [layout receipt](../results/runs/{rid}/input-layout.json), [validation](../results/runs/{rid}/validation.json), [provenance](../results/runs/{rid}/provenance.json).

All five compile inputs are contiguous and value-equal to their originals. Eager image and graph export completed. Pipeline loading took{load['load_total_s']:.6f}s and export{load['export_s']:.6f}s; these are one-off preparation observations, not timing medians.

Compilation reached operator conversion and failed at `timestep.to(hidden_states.dtype) * 1000`. The BF16 scalar multiply converter attempted to map the tensor dtype to NumPy and raised `TypeError: Unsupported numpy dtype`. No compiled transformer forward, module coverage inventory or full compiled image was produced. The eager image remains functional reference evidence only. No precision fallback, operator exclusion or library-source patch was applied.

## Interpretation

This pinned compiler stack does not pass the BF16 klein feasibility gate with these settings. The failure concerns conversion of the scalar multiply, not inference memory capacity. The bounded screen ends here; these results do not establish whether a different compiler release or explicitly labelled mixed partition could work. No compiler speedup or numerical equivalence is claimed.

## Next experiment

- A separately scoped compiler-version or converter-support investigation would precede further inference benchmarking.

[W&B failed gate](https://wandb.ai/mfaruqi-purdue-university/on-device-diffusion/runs/{rid}).
''')
control='a100-sdcpp-flux-klein-base-001__attempt__20261005-082117';cr=Path('results/runs')/control;s=json.loads((cr/'summary.json').read_text());v=json.loads((cr/'validation.json').read_text())
p=Path('experiments/a100-sdcpp-flux-klein-base-001.md');t=p.read_text().replace('20261005-081219]','20261005-081219, '+control+']')
section=f'''## Cache-capable harness no-cache control

[Control](../results/runs/{control}/), [validation](../results/runs/{control}/validation.json), [environment](../results/runs/{control}/environment.json). Job11883351 on gilbreth-g007 completed12:22:59UTC. Same pinned Base workload, no cache and unchanged engine settings, using the separately built cache-capable harness. Binary SHA256 `{v['cache_capable_binary_sha256']}`. Protocol:one first plus one additional observation; this is a correctness control, not a repeated warm baseline.

Both saved images match the existing no-cache reference pixels. All callback/CFG/stage/settings/summary checks passed. The second observation took{s['measured']['wall_ms']['median']:.3f}ms with sampled device-wide peak{s['measured']['device_used_peak_gib']['max']:.6f}GiB. A full matched-binary reference remains necessary before attributing a cache timing difference solely to reuse.

[W&B control](https://wandb.ai/mfaruqi-purdue-university/on-device-diffusion/runs/{control}).

'''
t=t.replace('## Interpretation',section+'## Interpretation');p.write_text(t)
p=Path('results/README.md');p.write_text(p.read_text()+f'| `{rid}` | [a100-flux-klein-torchtrt-002](../experiments/a100-flux-klein-torchtrt-002.md) | A100 PCIe,g007 | torchtrt contiguous BF16 gate | – | – | – | failed11883257; BF16 scalar-multiply conversion |\n'+f'| `{control}` | [a100-sdcpp-flux-klein-base-001](../experiments/a100-sdcpp-flux-klein-base-001.md) | A100 PCIe,g007 | cache-capable no-cache correctness control | – | – | – | complete11883351; two images, same reference pixels |\n')
p=Path('wiki/experiments.md');lines=p.read_text().splitlines()
for i,line in enumerate(lines):
 if line.startswith('| a100-sdcpp-flux-klein-base-001 |'):
  cells=line.split('|');cells[7]+=f', [cache-harness control](../results/runs/{control}/) ';lines[i]='|'.join(cells)
pos=lines.index('## Comparisons');lines.insert(pos,f'| a100-flux-klein-torchtrt-002 | A100 PCIe | PyTorch / Torch-TensorRT | Contiguous-input BF16 feasibility | failed; scalar-multiply dtype conversion | [record](../experiments/a100-flux-klein-torchtrt-002.md) | [attempt](../results/runs/{rid}/) | [W&B](https://wandb.ai/mfaruqi-purdue-university/on-device-diffusion/runs/{rid}) |');lines.insert(pos+1,'');p.write_text('\n'.join(lines)+'\n')
p=Path('wiki/log.md');p.write_text(p.read_text().replace('# Log\n','# Log\n\n## [2026-10-05] experiment | BF16 compiler screen ends; A100 cache control passes\n[Torch-TensorRT002](../experiments/a100-flux-klein-torchtrt-002.md) failed scalar-multiply conversion after value-preserving layout copies. [sd.cpp Base001](../experiments/a100-sdcpp-flux-klein-base-001.md) no-cache control passed on the cache-capable harness.\n',1))
