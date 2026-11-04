#!/usr/bin/env python3
"""MFLUX Klein on M5: pinned BF16/8-bit/4-bit; smoke-gated full protocol."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import subprocess
import time

from benchlib import GIB, REPO_ROOT, host_environment, sh, write_json
from measurement import create_run_directory, phase_sequence, run_fields, run_status, sampling, summarize_runs
from mflux_engine import HostStages, TIMING, check_config, prepare, sha256, tensor_audit, validate_run
from sdcpp_device import MacMemorySampler


def benchmark(cfg, out):
    import mlx.core as mx
    import mflux
    import psutil
    from mflux.models.common.config import ModelConfig
    from mflux.models.flux2.variants import Flux2Klein
    check_config(cfg)
    assert os.environ.get('HF_HUB_OFFLINE') == '1'
    assert mx.metal.is_available() and mx.device_info()['device_name'] == 'Apple M5'
    assert psutil.virtual_memory().total == 24*GIB
    assert 'AC Power' in sh(['pmset','-g','batt']), 'AC power required'
    mx.set_default_device(mx.gpu)
    root = Path(cfg['model']['local_path'])
    assert root.name == cfg['model']['revision']
    for name, meta in cfg['model']['files'].items():
        assert (root/name).stat().st_size == meta['bytes'], 'Changed checkpoint file: '+name
    source = Path(cfg['engine']['source_dir'])
    assert sh(['git','-C',str(source),'rev-parse','HEAD']) == cfg['engine']['commit']
    # Check installed Python source against the pinned checkout, not just its version string.
    package = Path(mflux.__file__).parent
    for p in (source/'src/mflux').rglob('*.py'):
        assert sha256(p) == sha256(package/p.relative_to(source/'src/mflux')), str(p)
    env = host_environment(extra_env_vars=['HF_HOME','HF_HUB_OFFLINE'])
    env.pop('nvidia_smi_gpu',None); env.pop('nvidia_driver',None)
    env.update(backend='mlx-metal',gpu_name=mx.device_info()['device_name'],device_info=mx.device_info(),
        packages={k:importlib.metadata.version(k) for k in ('mflux','mlx','mlx-metal','numpy','transformers','psutil','pillow')},
        memory_limit_bytes=mx.get_memory_limit(),memory_scope='MLX active allocator and separate whole-system unified memory',
        macos=sh(['sw_vers']),power_source=sh(['pmset','-g','batt']))
    write_json(out/'environment.json',env)
    write_json(out/'source-provenance.json',{p:sha256(REPO_ROOT/p) for p in
        ('scripts/run_mflux.py','scripts/mflux_engine.py','scripts/measurement.py','scripts/sdcpp_device.py')})
    sampler = MacMemorySampler(cfg['protocol']['device_memory_sample_ms'],out)
    (out/'images').mkdir()
    rows, stages = [], []
    with sampling(sampler):
        start = time.perf_counter()
        base = '-base-' in cfg['model']['repo_id']
        mc = ModelConfig.flux2_klein_base_4b() if base else ModelConfig.flux2_klein_4b()
        assert mc.model_name == cfg['model']['repo_id'] and ModelConfig.precision == mx.bfloat16
        model = Flux2Klein(quantize=cfg['optimizations']['quantization'],model_path=str(root),model_config=mc)
        mx.eval(model.parameters()); mx.synchronize()
        load_s = time.perf_counter()-start
        audit = tensor_audit(model,cfg['optimizations']['quantization'],out)
        write_json(out/'quantization-audit.json',audit)
        load = dict(load_total_s=load_s,allocated_after_load_gib=mx.get_active_memory()/GIB,
                    mlx_cache_after_load_gib=mx.get_cache_memory()/GIB,
                    components={k:{'weights_gib':v['weights_gib'],'quantized_layer_count':v['quantized_layer_count']}
                                for k,v in audit.items()},
                    note='Cached checkpoint load, on-load quantization and materialization; not cold-storage time.')
        write_json(out/'load.json',load)
        recorder = HostStages(mx,model,cfg,out)
        for i,phase in enumerate(phase_sequence(cfg['protocol'])):
            image, wall, times, peak = recorder.generate()
            steps = cfg['workload']['num_inference_steps']
            row = {k:None for k in run_fields(steps)}
            row.update(run_index=i,phase=phase,wall_ms=wall,text_encode_ms=times['text_encode'],
                denoise_ms=times['denoise'],vae_decode_ms=times['vae_decode'],
                other_ms=wall-times['text_encode']-times['denoise']-times['vae_decode'],
                peak_alloc_gib=peak,host_rss_gib=psutil.Process().memory_info().rss/GIB,
                image_sha256=hashlib.sha256(image.convert('RGB').tobytes()).hexdigest())
            row.update({f'denoise_step_{k}_ms':times[f'denoise_step_{k}'] for k in range(steps)})
            rows.append(row); stages.extend(dict(run_index=i,phase=phase,**s) for s in recorder.rows)
            image.save(out/'images'/f'{i:02d}-{phase}.png')
            for file,data in [('runs.csv',rows),('stages.csv',stages)]:
                with (out/file).open('w',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
            write_json(out/'execution-audit.json',dict(completed_generations=len(rows),
                denoising_steps_per_generation=steps,guidance_passes_per_step=2 if base else 1,
                prompt_shapes=recorder.prompt_shapes,negative_prompt='',
                prompt_reuse=cfg['optimizations'].get('conditioning_reuse',False),
                prompt_cache_hits=recorder.prompt_hits,prompt_cache_misses=recorder.prompt_misses,
                tiling=recorder.tiling,tile_decode_shapes_by_generation=recorder.generation_decode_calls,
                default_device=str(mx.default_device()),native_predict_compile=True))
            if cfg['optimizations'].get('conditioning_reuse'):
                assert recorder.prompt_misses == 1 and recorder.prompt_hits == i
            write_json(out/'scheduler.json',dict(class_name=type(recorder.scheduler).__name__,
                timesteps=recorder.scheduler.timesteps.tolist(),sigmas=recorder.scheduler.sigmas.tolist()))
            print(f'[{i} {phase}] wall {wall/1000:.3f}s text {times["text_encode"]/1000:.3f}s '
                  f'denoise {times["denoise"]/1000:.3f}s vae {times["vae_decode"]/1000:.3f}s',flush=True)
    memory=sampler.summary()
    summary=dict(id=cfg['id'],device_label=cfg['device_label'],engine='MFLUX / MLX',
        **summarize_runs(rows),load=load,system_memory=memory,timing_method=TIMING,
        memory_scope=env['memory_scope'],clock_continuity_pass=abs(memory['wall_minus_monotonic_elapsed_s'])<5,
        host_peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/GIB,
        unavailable_metrics={'gpu_span_ms':'No MLX GPU-event timestamps recorded',
            'peak_reserved_gib':'MLX cache is not PyTorch reserved memory',
            'device_used_peak_gib':'Unified system memory recorded separately',
            'postprocess_ms':'Image conversion and remaining setup are included in other_ms'},
        rng='Native MLX; same seed does not match initial noise across engines',formal_quality_eligible=False,
        units={'time':'ms (load in s)','memory':'GiB'})
    write_json(out/'summary.json',summary)
    assert summary['clock_continuity_pass'], 'Sleep/clock discontinuity invalidates timing'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True,type=Path)
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--smoke-run',type=Path)
    args=p.parse_args();cfg=json.loads(args.config.read_text());check_config(cfg)
    if args.prepare_only:
        prepare(cfg,args.config);return
    full=cfg['protocol']['measured_runs']==10
    if full:
        assert args.smoke_run, 'Full protocol requires a matching smoke'
        smoke_cfg=json.loads((args.smoke_run/'config.json').read_text())
        assert smoke_cfg['protocol']['measured_runs']==1
        validate_run(args.smoke_run,cfg)
    out=create_run_directory(cfg,REPO_ROOT/'results/runs','baseline' if full else 'attempt')
    with run_status(out):
        guard=subprocess.Popen(['/usr/bin/caffeinate','-is','-w',str(os.getpid())])
        try:
            benchmark(cfg,out)
            validate_run(out,cfg,allow_running=True)
        finally: guard.terminate();guard.wait()


if __name__=='__main__': main()
