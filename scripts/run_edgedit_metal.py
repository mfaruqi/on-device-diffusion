#!/usr/bin/env python3
"""Pinned edge-dit Metal feasibility and repeated protocols using upstream ed-sample."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

from benchlib import GIB, host_environment, sh, write_json
from measurement import check_optimizations, create_run_directory, phase_sequence, run_status, sampling, summarize_runs, write_csv
from measurement_events import fingerprint
from run_edgedit import COMMIT, parse_log
from sdcpp_device import MacMemorySampler, child_peak_rss_gib


def benchmark(cfg, out):
    import psutil
    check_optimizations(cfg, {'quantization','cpu_offload','vae_tiling','auto_fit','auto_allocate','step_cache'})
    assert cfg['backend']=='metal' and cfg['precision']=='bfloat16'
    assert os.environ.get('HF_HUB_OFFLINE')=='1'
    assert cfg['cli_arguments']==['--backend','metal','--type','bf16','--cache_method','off',
                                  '--vae-tiling','off','--flash-attention','--threads','4']
    assert sh(['sysctl','-n','machdep.cpu.brand_string'])==cfg['device_requirements']['chip']
    assert psutil.virtual_memory().total==cfg['device_requirements']['memory_gib']*GIB
    phases=phase_sequence(cfg['protocol']);w=cfg['workload']
    assert w['width']==w['height']==512 and w['seed']==0 and w['batch_size']==1
    assert (w['steps'],w['cfg_scale']) in [(4,1.0),(50,4.0)]
    assert w.get('negative_prompt','')==''
    assert w['guidance']==1 and w['sampler']=='euler' and w['scheduler']=='auto'
    assert '\n' not in w['prompt'] and w['prompt'].strip()
    model=Path(cfg['model']['path']);assert model.is_dir() and model.name==cfg['model']['revision']
    native = cfg['model'].get('native_transformer')
    if native:
        native_path = model / native['file']
        assert native_path.is_file(), 'Prepare pinned native transformer before inference'
        assert fingerprint(native_path) == native['sha256'], 'Native transformer hash mismatch'
    binary=Path(cfg['engine']['binary']);source=binary.parents[2]
    assert sh(['git','-C',str(source),'rev-parse','HEAD'])==cfg['engine']['commit']==COMMIT
    assert not sh(['git','-C',str(source),'diff','HEAD','--'])
    assert fingerprint(binary)==cfg['engine']['binary_sha256']
    cache=(binary.parents[1]/'CMakeCache.txt').read_text()
    assert 'ED_GGML_METAL:BOOL=ON' in cache and 'ED_GGML_CUDA:BOOL=OFF' in cache
    sampler=MacMemorySampler(cfg['protocol']['device_memory_sample_ms'],out)
    env=host_environment(extra_env_vars=['HF_HOME','HF_HUB_OFFLINE'])
    env.pop('nvidia_smi_gpu',None);env.pop('nvidia_driver',None)
    env.update(engine=cfg['engine'],backend='metal',power_source=sh(['pmset','-g','batt']),
               submodules=sh(['git','-C',str(source),'submodule','status']),memory_scope=sampler.note)
    write_json(out/'environment.json',env)
    prompt=out/'prompt.txt';prompt.write_text(w['prompt']+'\n')
    assert cfg['protocol']['prevent_idle_sleep']
    cmd=['/usr/bin/caffeinate','-is',str(binary),'--model',str(model),*cfg['cli_arguments'],
         '--prompt_file',str(prompt),'--output_dir',str(out/'engine'),'--warmup','0','--repeat',str(len(phases)),'--negative_prompt',w.get('negative_prompt','')]
    for key,flag in {'width':'width','height':'height','steps':'num_steps','guidance':'guidance_scale',
                     'cfg_scale':'cfg_scale','seed':'seed','sampler':'sampler','scheduler':'scheduler'}.items():
        cmd.extend(['--'+flag,str(w[key])])
    write_json(out/'command.json',cmd)
    with sampling(sampler),(out/'engine.log').open('w') as stream:
        result=subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT)
    if result.returncode:
        detail=(out/'engine.log').read_text(errors='replace')[-6000:]
        raise RuntimeError(f'edge-dit exited {result.returncode}: {detail}')
    log=(out/'engine.log').read_text()
    if native:
        assert 'from top-level Flux2 weights' in log and native['file'] in log, 'Native transformer was not selected'
    assert not re.search(r'fallback to CPU|falling back to CPU|retrying with.*tiling',log,re.I)
    timing=json.loads((out/'engine/timing.json').read_text())
    rows,stages,events=parse_log(log,cfg,phases,timing)
    from PIL import Image
    image=out/'engine/imgs/img_000000.png'
    with Image.open(image) as im:
        im=im.convert('RGB');assert im.size==(512,512)
        rows[-1]['image_sha256']=hashlib.sha256(im.tobytes()).hexdigest()
    shutil.copy2(image,out/'output.png')
    write_json(out/'image-retention.json',dict(retained_run_index=len(rows)-1,phase=phases[-1],
                                             image='output.png',overwritten_earlier_images=True))
    write_csv(out/'runs.csv',rows);write_csv(out/'stages.csv',stages)
    load=dict(load_total_s=timing['model_load_seconds'],note='Single context; cached load, not cold storage')
    write_json(out/'load.json',load)
    memory=sampler.summary()
    summary=dict(**summarize_runs(rows),id=cfg['id'],engine='edge-dit.cpp Metal',device_label=cfg['device_label'],
                 load=load,system_memory=memory,host_peak_rss_gib=child_peak_rss_gib(),formal_quality_eligible=False,
                 clock_continuity_pass=abs(memory['wall_minus_monotonic_elapsed_s'])<5,
                 timing_method='ed-sample steady_clock durations rounded to 1ms; system-clock stage markers',
                 unavailable_metrics={'text_encode_ms':'encode_setup includes latent/schedule preparation; see stages.csv.',
                                      'gpu_span_ms/peak_alloc_gib/peak_reserved_gib/device_used_peak_gib':'No Metal event or peak allocation counters.',
                                      'denoise_step_i_ms/postprocess_ms/other_ms':'Not separately timed.',
                                      'deterministic_output':'Upstream overwrites earlier images.'})
    summary['deterministic_output']=None
    write_json(out/'summary.json',summary)
    (out/'events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
    assert summary['clock_continuity_pass'],'Host clocks diverged'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True,type=Path)
    p.add_argument('--kind',choices=['attempt','baseline'],default='baseline')
    args=p.parse_args();cfg=json.loads(args.config.read_text())
    out=create_run_directory(cfg,Path('results/runs'),args.kind)
    with run_status(out):benchmark(cfg,out)


if __name__=='__main__':main()
