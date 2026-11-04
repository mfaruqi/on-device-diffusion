#!/usr/bin/env python3
"""BF16 Klein on MPS: one context, first/warm-up/measured protocol, no CPU fallback."""
import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import time

from benchlib import GIB, REPO_ROOT, write_json
from flux_engine import check_config as check_reference_config, prepare
from mps_conditioning import ConditioningCache
from mps_options import check_options, install_compile, compile_receipt, TaylorAudit
from measurement import create_run_directory, phase_sequence, run_fields, run_status, sampling, summarize_runs
from mps_device import MPSDevice, MPSStageRecorder
from run_flux import result_row


def check_config(cfg):
    reference = copy.deepcopy(cfg)
    check_options(reference['optimizations'])
    reference['optimizations'].pop('step_cache', None)
    reference['optimizations']['torch_compile'] = False
    enabled = reference['optimizations'].pop('conditioning_reuse', False)
    assert type(enabled) is bool, 'conditioning_reuse must be boolean'
    tiling = reference['optimizations'].get('vae_tiling', False)
    if tiling:
        assert tiling == {'tile_sample_size': 256, 'overlap_factor': 0.25}, 'Explicit tested tiling parameters required'
        reference['optimizations']['vae_tiling'] = False
    check_reference_config(reference)


def generate(pipe, recorder, w, cache=None):
    import torch
    # MPS has no independent Generator; CPU Philox/random initialization is explicit.
    generator = torch.Generator(device='cpu').manual_seed(w['seed'])
    torch.mps.synchronize()
    recorder.reset()
    a, b = torch.mps.Event(enable_timing=True), torch.mps.Event(enable_timing=True)
    t0 = time.perf_counter()
    a.record()
    if cache:
        cache.begin()
    image = pipe(prompt=w['prompt'], width=w['width'], height=w['height'],
                 num_inference_steps=w['num_inference_steps'], guidance_scale=w['guidance_scale'],
                 max_sequence_length=w['max_sequence_length'], generator=generator, output_type='pil').images[0]
    b.record()
    torch.mps.synchronize()
    t1 = time.perf_counter()
    stages, alloc, reserved, segments = recorder.finish()
    return image, dict(calls_per_step=2 if pipe.do_classifier_free_guidance else 1,
                      scheduler_steps=recorder.scheduler_steps, wall_ms=(t1-t0)*1000,
                      gpu_span_ms=a.elapsed_time(b), t0=t0, t1=t1, stages=stages,
                      peak_alloc_gib=alloc, peak_reserved_gib=reserved, segments=segments)


def benchmark(cfg, out):
    import torch
    import diffusers
    import psutil
    check_config(cfg)
    assert cfg['backend'] == 'mps'
    assert os.environ.get('HF_HUB_OFFLINE') == '1', 'Inference must run offline'
    device = MPSDevice(cfg, out)
    write_json(out/'environment.json', device.environment())
    sources=['scripts/run_flux_mps.py','scripts/mps_device.py','scripts/mps_conditioning.py','scripts/mps_options.py','scripts/run_flux.py','scripts/torch_stages.py']
    write_json(out/'source-provenance.json',{name:hashlib.sha256((REPO_ROOT/name).read_bytes()).hexdigest() for name in sources})
    phases = phase_sequence(cfg['protocol'])
    rows, stages = [], []
    (out/'images').mkdir()
    with sampling(device.sampler):
        t0 = time.perf_counter()
        pipe = diffusers.Flux2KleinPipeline.from_pretrained(
            cfg['model']['repo_id'], revision=cfg['model']['revision'], torch_dtype=torch.bfloat16,
            local_files_only=True)
        assert pipe.config.is_distilled == (cfg['workload']['guidance_scale'] == 1.0)
        t1 = time.perf_counter()
        pipe.to('mps')
        torch.mps.synchronize()
        t2 = time.perf_counter()
        components = {}
        for name, component in pipe.components.items():
            if isinstance(component, torch.nn.Module):
                params = list(component.parameters())
                assert params and all(p.device.type == 'mps' for p in params), name+' not fully on MPS'
                assert all(p.dtype == torch.bfloat16 for p in params if p.is_floating_point()), name+' changed precision'
                components[name] = dict(weights_gib=sum(p.numel()*p.element_size() for p in params)/GIB,
                                        dtypes=sorted({str(p.dtype) for p in params}), device='mps')
        load = dict(load_total_s=t2-t0, from_pretrained_s=t1-t0, to_mps_s=t2-t1,
                    allocated_after_load_gib=torch.mps.current_allocated_memory()/GIB,
                    driver_allocated_after_load_gib=torch.mps.driver_allocated_memory()/GIB,
                    is_distilled=pipe.config.is_distilled, components=components,
                    note='Cached load and placement; not a cold-storage measurement.')
        write_json(out/'load.json', load)
        write_json(out/'scheduler.json', {'class':type(pipe.scheduler).__name__, 'config':dict(pipe.scheduler.config)})
        tiling = cfg['optimizations'].get('vae_tiling', False)
        tile_calls = []
        if tiling:
            pipe.vae.enable_tiling()
            pipe.vae.tile_sample_min_size = tiling['tile_sample_size']
            pipe.vae.tile_latent_min_size = tiling['tile_sample_size'] // (2 ** (len(pipe.vae.config.block_out_channels) - 1))
            pipe.vae.tile_overlap_factor = tiling['overlap_factor']
            original_tiled_decode = pipe.vae.tiled_decode
            def tiled_decode(*args, **kwargs):
                tile_calls.append(True)
                return original_tiled_decode(*args, **kwargs)
            pipe.vae.tiled_decode = tiled_decode
        capacity = 2 if cfg['workload']['guidance_scale'] > 1 else 1
        conditioning = ConditioningCache(pipe, capacity) if cfg['optimizations'].get('conditioning_reuse', False) else None
        compile_cfg = cfg['optimizations'].get('torch_compile', False)
        if compile_cfg:
            verify = cfg['protocol'].get('verify_compile_first_call', False)
            install_compile(pipe.transformer, compile_cfg, out/'compile-numerical-check.json' if verify else None)
            write_json(out/'compiler.json', dict(configuration=compile_cfg, status='installed_not_executed'))
        step_cache = cfg['optimizations'].get('step_cache')
        cache = TaylorAudit(pipe.transformer, step_cache) if step_cache else None
        if cache:
            import diffusers.hooks.taylorseer_cache as implementation
            write_json(out/'cache-source.json', dict(path=implementation.__file__,
                       sha256=hashlib.sha256(Path(implementation.__file__).read_bytes()).hexdigest(), parameters=step_cache))
        # Install timing outside the cache: first misses and subsequent lookups stay timed.
        recorder = MPSStageRecorder(torch)
        recorder.install(pipe)
        pipe.set_progress_bar_config(disable=True)
        process = psutil.Process()
        for i, phase in enumerate(phases):
            image, result = generate(pipe, recorder, cfg['workload'], cache)
            if cache:
                cache.end(cfg['workload']['num_inference_steps'], result['calls_per_step'])
                write_json(out/'step-cache.json', cache.receipt())
            if compile_cfg:
                receipt = compile_receipt()
                write_json(out/'compiler.json', dict(configuration=compile_cfg, **receipt))
                assert receipt['counters'].get('stats', {}).get('unique_graphs', 0) > 0, 'No graph compiled'
                assert not receipt['counters'].get('graph_break'), 'Fullgraph compilation broke'
            if tiling:
                assert len(tile_calls) == i + 1, 'VAE tiling was requested but did not execute'
                write_json(out/'vae-tiling.json', dict(**tiling, tiled_decodes=len(tile_calls)))
            if conditioning:
                assert conditioning.misses == capacity and conditioning.hits == i * capacity
                write_json(out/'conditioning-cache.json', conditioning.receipt())
            row = result_row(i, phase, image, result, cfg['workload']['num_inference_steps'], device.sampler, process)
            assert image.size == (cfg['workload']['width'], cfg['workload']['height'])
            rows.append(row)
            stages.extend({'run_index':i, 'phase':phase, **s} for s in result['stages'])
            for path, data, fields in [(out/'runs.csv',rows,run_fields(cfg['workload']['num_inference_steps'])),
                                       (out/'stages.csv',stages,list(stages[0]))]:
                with path.open('w',newline='') as f:
                    writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader(); writer.writerows(data)
            image.save(out/'images'/f'{i:02d}-{phase}.png')
            print(f'[{i} {phase}] {row["wall_ms"]/1000:.3f} s',flush=True)
    memory = device.sampler.summary()
    summary = dict(id=cfg['id'], device_label=cfg['device_label'], engine='PyTorch/Diffusers MPS',
                   **summarize_runs(rows), load=load, system_memory=memory,
                   host_peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/GIB,
                   clock_continuity_pass=abs(memory['wall_minus_monotonic_elapsed_s'])<5,
                   timing_method='MPS events at stage boundaries; host wall includes final synchronization; no per-stage sync',
                   rng='CPU generator; initial noise not matched to sd.cpp', formal_quality_eligible=False,
                   unavailable_metrics={'peak_alloc_gib/peak_reserved_gib':'No equivalent MPS peak counters recorded.',
                                        'device_used_peak_gib':'Whole-system memory is separate, not device-wide GPU usage.'},
                   conditioning_cache=conditioning.receipt() if conditioning else None,
                   compiler=compile_receipt() if compile_cfg else None,
                   first_image_includes_compile_verification=cfg['protocol'].get('verify_compile_first_call',False),
                   step_cache=cache.receipt() if cache else None,
                   vae_tiling=dict(parameters=tiling, tiled_decodes=len(tile_calls)),
                   units={'time':'ms (load in s)','memory':'GiB'})
    write_json(out/'summary.json', summary)
    assert summary['clock_continuity_pass'], 'Timing rejected: wall/monotonic divergence'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True,type=Path)
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--kind',choices=['attempt','baseline'],default='baseline')
    args=p.parse_args(); cfg=json.loads(args.config.read_text())
    if args.prepare_only:
        prepare(cfg,args.config); return
    out=create_run_directory(cfg,REPO_ROOT/'results/runs',args.kind)
    with run_status(out):
        assert cfg['protocol']['prevent_idle_sleep']
        guard=subprocess.Popen(['/usr/bin/caffeinate','-is','-w',str(os.getpid())])
        try: benchmark(cfg,out)
        finally: guard.terminate(); guard.wait()


if __name__=='__main__': main()
