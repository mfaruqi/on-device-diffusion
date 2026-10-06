import json
from pathlib import Path
ids=['a100-flux-klein-base-001__baseline__20261005-074425','a100-sdcpp-flux-klein-base-001__baseline__20261005-080118']
roots=[Path('results/runs')/i for i in ids]
c=[json.loads((p/'config.json').read_text()) for p in roots];s=[json.loads((p/'summary.json').read_text()) for p in roots]
for key in ['revision','repo_id']:assert c[0]['model'][key]==c[1]['model'][key]
for key in ['prompt','seed','batch_size','height','width','num_inference_steps','guidance_scale','max_sequence_length','negative_prompt']:assert c[0]['workload'][key]==c[1]['workload'][key],key
assert c[0]['protocol']==c[1]['protocol']
for p in roots:assert json.loads((p/'status.json').read_text())['status']=='complete'
ratio=s[1]['measured']['wall_ms']['median']/s[0]['measured']['wall_ms']['median']
t=f'''---
type: experiment-record
id: compare-a100-base-pytorch-vs-sdcpp
status: complete
device: a100-pcie-40gb
engine: [pytorch-diffusers, stable-diffusion-cpp]
runs: [{', '.join(ids)}]
updated: 2026-10-05
---

# Compare: A100 klein Base with PyTorch and sd.cpp

## Question

How do the existing unoptimized engines behave on the50-step Base workload? This serves Week1–2 image baselines and RQ1. It is a descriptive engine comparison, not a matched-initial-noise or matched-quality performance claim.

## Runs

| Engine | Run and saved config | Record |
|---|---|---|
| PyTorch/diffusers | [{ids[0]}](../results/runs/{ids[0]}/config.json) | [PyTorch Base001](a100-flux-klein-base-001.md) |
| stable-diffusion.cpp | [{ids[1]}](../results/runs/{ids[1]}/config.json) | [sd.cpp Base001](a100-sdcpp-flux-klein-base-001.md) |

## Matching

| Property | Both runs / limitation |
|---|---|
| Checkpoint | black-forest-labs/FLUX.2-klein-base-4B, a3b4f4849157f664bdbc776fd7453c2783562f4d |
| Weights and residency | BF16, fixed CUDA placement, no quantization/offloading/reuse/tiling; internal arithmetic and kernels differ |
| Workload |1024×1024,batch1,50steps,guidance4,empty negative prompt,same cat/sign prompt,max sequence length512 |
| Seed |0in both configs; initial noise differs across RNG implementations ([D-006](../wiki/project/decisions.md#d-006-shared-initial-noise-from-sdcpps-philox-rng-proposed)) |
| Schedule | Each native Flux2 path; numerical equality of timestep/sigma trajectories across engines is not verified |
| Protocol | One first,three discarded warm-ups,ten measured; no profiler attached |
| Hardware | A100-PCIE-40GB on gilbreth-g007; separate sequential jobs, not interleaved |
| Timing | Both synchronized end-to-end wall time; PyTorch stages use CUDA events, sd.cpp stages use host callbacks |
| Memory | Compare NVML device-wide sampled peaks only; PyTorch allocated/reserved counters have no equivalent in sd.cpp |

The engine is the intended variable. Initial noise, implementation arithmetic, unverified schedule equivalence and sequential-job conditions remain confounds, so this comparison is qualitative. Saved-image repeatability within each engine does not establish cross-engine fidelity.

## Results

Sources: [PyTorch summary](../results/runs/{ids[0]}/summary.json), [sd.cpp summary](../results/runs/{ids[1]}/summary.json). Stage medians are independently aggregated and need not sum to the end-to-end median. Stage timers have different boundaries/semantics.

| Metric | PyTorch median (min–max), ms | sd.cpp median (min–max), ms |
|---|---:|---:|
'''
for key in ['wall_ms','text_encode_ms','denoise_ms','vae_decode_ms','other_ms']:
 vals=[x['measured'][key] for x in s]
 t+='| '+key+' | '+' | '.join(f"{v['median']:.3f} ({v['min']:.3f}–{v['max']:.3f})" for v in vals)+' |\n'
t+=f'''
PyTorch's separately recorded postprocess median is{s[0]['measured']['postprocess_ms']['median']:.3f}ms; sd.cpp has no separately timed postprocess. Measured device-wide peak is{s[0]['measured']['device_used_peak_gib']['max']:.6f}GiB for PyTorch and{s[1]['measured']['device_used_peak_gib']['max']:.6f}GiB for sd.cpp. This is total sampled device usage, not allocator memory.

The observed sd.cpp/PyTorch wall-time ratio is **{ratio:.4f}**. Denoising dominates both runs. Each engine produced14matching recorded hashes within its run; both retained PNGs were independently checked. No cross-engine quality metric or alignment benchmark was applied.

## Interpretation and limits

For these recorded configurations, PyTorch has the lower observed end-to-end latency and device-wide peak. This does not isolate the source of the engine difference or establish equal-quality superiority. The per-stage numbers locate where time is spent, but different timer boundaries prevent attributing every stage delta to kernel speed. The separate sd.cpp trace failed at profiler startup, so a paired kernel-level explanation remains pending. These results do not alone determine compute versus memory-bandwidth limits.
'''
Path('experiments/compare-a100-base-pytorch-vs-sdcpp.md').write_text(t)
p=Path('wiki/experiments.md');p.write_text(p.read_text()+'| PyTorch vs sd.cpp, klein Base, A100 PCIe | a100-flux-klein-base-001, a100-sdcpp-flux-klein-base-001 | [record](../experiments/compare-a100-base-pytorch-vs-sdcpp.md) |\n')
p=Path('wiki/log.md');p.write_text(p.read_text().replace('# Log\n','# Log\n\n## [2026-10-05] experiment | A100 Base engine comparison\n[Comparison](../experiments/compare-a100-base-pytorch-vs-sdcpp.md) records repeated unprofiled timings and device-wide memory; unmatched initial noise, native schedules and stage timing semantics limit causal/quality claims.\n',1))
print('Observed sd.cpp/PyTorch wall ratio',ratio)
