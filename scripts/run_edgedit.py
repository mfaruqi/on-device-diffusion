#!/usr/bin/env python3
"""Pinned ed-sample baseline adapter; one child context and no upstream changes.

Only the validated four-step no-cache A100 workload is supported. Per-generation
wall time comes from rounded steady-clock stdout; raw system-clock stage markers
have different semantics and are preserved explicitly. See baseline-metrics.md.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import shutil
import subprocess

from benchlib import GIB, DeviceMemorySampler, host_environment, visible_gpu_uuid, write_json
from measurement import (check_optimizations, create_run_directory, phase_sequence,
                         run_fields, run_status, sampling, summarize_runs, write_csv)
from measurement_events import event, fingerprint

COMMIT = '97cc585d9dbd9ac5e30b78ef2e2c4a0d8bfbe9f2'
REVISION = 'e7b7dc27f91deacad38e78976d1f2b499d76a294'
CLI = ['--backend', 'cuda', '--type', 'bf16', '--cache_method', 'off',
       '--vae-tiling', 'off', '--flash-attention', '--threads', '4']


def check_config(cfg):
    check_optimizations(cfg, {'quantization', 'cpu_offload', 'vae_tiling',
                              'auto_fit', 'auto_allocate', 'step_cache'})
    assert cfg['engine']['commit'] == COMMIT and cfg['model']['revision'] == REVISION
    assert cfg['model']['repo_id'] == 'black-forest-labs/FLUX.2-klein-4B'
    assert cfg['precision'] == 'bfloat16' and cfg['cli_arguments'] == CLI
    w = cfg['workload']
    expected = {'width': 1024, 'height': 1024, 'steps': 4, 'guidance': 1.0,
                'cfg_scale': 1.0, 'seed': 0, 'batch_size': 1, 'sampler': 'euler', 'scheduler': 'auto'}
    assert w == {**expected, 'prompt': w['prompt']} and '\n' not in w['prompt']
    assert w['prompt'].strip(), 'Exactly one nonempty prompt is required'
    assert cfg['protocol']['device_memory_sample_ms'] > 0
    return phase_sequence(cfg['protocol'])


def parse_log(log, cfg, phases, timing):
    """Refuse missing/reordered stages, changed schedules or incomplete repeats."""
    passes = re.findall(r'\[ed-sample\] pass (\d+)/(\d+)\s+1/1\s+seed=0\s+([0-9.]+)s', log)
    n = len(phases)
    assert [(int(a), int(b)) for a, b, _ in passes] == [(i+1, n) for i in range(n)]
    assert timing['num_images'] == n
    wall = [float(p[2])*1000 for p in passes]
    assert all(math.isfinite(t) and t > 0 for t in wall)
    # stdout rounds each sample to 1 ms; timing.json retains six decimal seconds.
    assert abs(sum(wall)/1000 - timing['e2e_time']['total']) <= n*0.000501
    markers = re.findall(r'\[\[phase\]\] stage=(\w+) event=(\w+) t=([0-9.]+)', log)
    names = [('encode','begin'),('encode','end'),('denoise','begin'),
             ('denoise','end'),('decode','begin'),('decode','end')]
    assert [(a,b) for a,b,_ in markers] == names*n
    schedule = re.findall(r'flux step (\d+)/4 sigma=([\d.]+) next=([\d.]+)', log)
    expected = [('1','1.000000','0.967384'),('2','0.967384','0.908144'),
                ('3','0.908144','0.767200'),('4','0.767200','0.000000')]
    assert schedule == expected*n, 'Changed or incomplete four-step schedule'
    assert log.count('1024x1024 latent=64x64 image_seq_len=4096 steps=4 flux2_mu=2.291 guidance=1.00 cfg=1.00 seed=0') == n
    for line in ['default backend: CUDA0', 'flux activation dtype: f32 (bf16_tensors=149/149)',
                 'Conditioner weight type stat: bf16:398', 'Diffusion model weight type stat: bf16:149',
                 'VAE weight type stat: bf16:248', 'cache mode   : original']:
        assert line in log, 'Missing engine confirmation: '+line
    rows, stages, events = [], [], []
    for i, phase in enumerate(phases):
        ts = [float(m[2]) for m in markers[i*6:i*6+6]]
        assert ts == sorted(ts) and all(math.isfinite(t) for t in ts)
        assert (ts[-1]-ts[0])*1000 <= wall[i]+0.501, 'Stage clock/span inconsistent with generation'
        row = dict.fromkeys(run_fields(4))
        row.update(run_index=i, phase=phase, wall_ms=wall[i],
                   denoise_ms=(ts[3]-ts[2])*1000, vae_decode_ms=(ts[5]-ts[4])*1000)
        rows.append(row)
        source = {'file': 'engine.log', 'pass': i+1}
        events.append(event('generation', 'generate', source, duration=wall[i],
                            timing='rounded_host_steady_clock_duration', run_index=i, phase=phase,
                            details={'rounding_ms': 1.0}))
        for name,k in [('encode_setup',0),('denoise',2),('vae_decode',4)]:
            duration=(ts[k+1]-ts[k])*1000
            stages.append({'run_index':i,'phase':phase,'stage':name,'host_ms':duration,'gpu_ms':None})
            events.append(event('stage',name,source,start=ts[k],end=ts[k+1],duration=duration,
                                clock='edgedit_system_clock',timing='host_phase_marker_interval',
                                run_index=i,phase=phase))
    return rows, stages, events


def benchmark(cfg, out):
    phases = check_config(cfg)
    assert os.environ.get('SLURM_JOB_ID'), 'A100 inference requires a Slurm allocation'
    binary = Path(cfg['engine']['binary']); source = binary.parents[2]
    commit = subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    assert commit == COMMIT
    assert fingerprint(binary) == cfg['engine']['binary_sha256'], 'Changed engine binary'
    assert not subprocess.check_output(['git','-C',str(source),'diff','HEAD','--'],text=True), 'Modified engine source'
    model = Path(cfg['model']['path']); assert model.is_dir() and model.name == REVISION
    sampler = DeviceMemorySampler(cfg['protocol']['device_memory_sample_ms'],visible_gpu_uuid())
    env=host_environment(extra_env_vars=['LD_LIBRARY_PATH','HF_HUB_OFFLINE'])
    env.update(engine={**cfg['engine'],'binary_sha256':fingerprint(binary)},
               device_memory_sampler=sampler.note,
               submodules=subprocess.check_output(['git','-C',str(source),'submodule','status'],text=True))
    write_json(out/'environment.json',env)
    assert 'A100-PCIE-40GB' in env['nvidia_smi_gpu']
    prompt=out/'prompt.txt';prompt.write_text(cfg['workload']['prompt']+'\n')
    cmd=[str(binary),'--model',str(model),*cfg['cli_arguments'],'--prompt_file',str(prompt),
         '--output_dir',str(out/'engine'),'--warmup','0','--repeat',str(len(phases))]
    for key,flag in {'width':'width','height':'height','steps':'num_steps','guidance':'guidance_scale',
                     'cfg_scale':'cfg_scale','seed':'seed','sampler':'sampler','scheduler':'scheduler'}.items():
        cmd.extend(['--'+flag,str(cfg['workload'][key])])
    write_json(out/'command.json',cmd)
    with sampling(sampler), (out/'engine.log').open('w') as stream:
        subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=1200)
    write_json(out/'device-memory.json',{'clock':'python_perf_counter','interval_ms':cfg['protocol']['device_memory_sample_ms'],
                                       'samples':sampler.samples,'scope':'Whole child process; not aligned to system-clock phases'})
    timing=json.loads((out/'engine/timing.json').read_text())
    rows,stages,events=parse_log((out/'engine.log').read_text(),cfg,phases,timing)
    from PIL import Image
    image=out/'engine/imgs/img_000000.png'
    with Image.open(image) as im:
        im=im.convert('RGB');assert im.size==(1024,1024)
        rows[-1]['image_sha256']=hashlib.sha256(im.tobytes()).hexdigest()
    # Only the last PNG survives upstream repeats. Keep its actual index explicit.
    shutil.copy2(image,out/'output.png')
    write_json(out/'image-retention.json',{'retained_run_index':len(rows)-1,'phase':phases[-1],
                                         'image':'output.png','overwritten_earlier_images':True})
    write_csv(out/'runs.csv',rows);write_csv(out/'stages.csv',stages)
    load={'load_total_s':timing['model_load_seconds'],'note':'Upstream steady_clock around one context creation; filesystem cache may be warm.'}
    write_json(out/'load.json',load)
    summary=summarize_runs(rows)
    summary.update(id=cfg['id'],device_label=cfg['device_label'],engine='edge-dit.cpp',
                   deterministic_output=None,load=load,
                   capture_device_used_peak_gib=max((v for _,v in sampler.samples),default=0)/GIB if sampler.samples else None,
                   host_peak_rss_gib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss*1024/GIB,
                   timing_method='ed-sample steady_clock per-pass stdout, rounded to 1ms; host system_clock phase intervals',
                   precision_note='BF16 weights; FP32 transformer activations',formal_quality_eligible=False,
                   unavailable_metrics={
                       'text_encode_ms':'Encode phase includes conditioning and latent/schedule setup; retained as encode_setup in stages.csv.',
                       'denoise_step_i_ms':'Engine prints completed steps without individual timestamps.',
                       'gpu_span_ms/gpu_ms':'No GPU event timing.',
                       'postprocess_ms/other_ms':'Final tensor-to-image conversion not separately timed; no comparable text boundary.',
                       'device_used_peak_gib':'Only whole-child NVML peak; sampler and phase clocks not aligned.',
                       'peak_alloc_gib/peak_reserved_gib/host_rss_gib':'No per-generation allocator or RSS observations.',
                       'deterministic_output':'Only final repeat image retained; preceding image hashes unavailable.'})
    write_json(out/'summary.json',summary)
    (out/'events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
    write_json(out/'events-metadata.json',{'schema_version':1,'engine':'edge-dit.cpp','event_count':len(events),
        'source_sha256':{p.name:fingerprint(p) for p in [out/'engine.log',out/'config.json',out/'runs.csv']},
        'limitations':['E2E has durations only; stages use system_clock. Do not align the clocks.',
                       'encode_setup includes conditioning and latent/schedule preparation.',
                       'No per-step timings, earlier image hashes or per-generation memory peaks.']})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--out-root',default=Path('results/runs'),type=Path)
    parser.add_argument('--kind',choices=['attempt','baseline'],default='baseline')
    args=parser.parse_args();cfg=json.loads(args.config.read_text())
    out=create_run_directory(cfg,args.out_root,args.kind)
    with run_status(out):benchmark(cfg,out)


if __name__=='__main__':
    main()
