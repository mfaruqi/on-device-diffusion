#!/usr/bin/env python3
"""Pinned ed-sample 1+3+10 adapter for the Jetson Orin Nano; one child context, weights resident.

Workload: klein (distilled 4 steps / Base 50 steps CFG 4), GGUF component files, 512x512 or 1024x1024, one
prompt and seed. Devices: the Jetson Orin Nano (device_label jetson-orin-nano; tegrastats system memory, power
mode checked) and the A100 (device_label a100-pcie-40gb; Slurm allocation required, NVML device memory).
All generations run in one ed-sample process (--warmup 0 --repeat N): load once, then N passes.
Per-generation wall time is ed-sample's rounded steady-clock stdout; stage durations come from the
engine's own "flux generate breakdown" line (text_encode / sampling / vae_decode / other) and the
raw system-clock [[phase]] markers are preserved in stages.csv. Memory is whole-system tegrastats
(no per-process GPU allocator on Jetson). Run it under scripts/jetson_memguard.sh.
See wiki/methods/baseline-metrics.md.
"""
import argparse
import hashlib
import json
import os
import math
from pathlib import Path
import re
import shutil
import subprocess

from benchlib import host_environment, write_json
from jetson_device import TegrastatsMonitor, memory_snapshot, summarize_monitor
from measurement import (check_optimizations, create_run_directory, phase_sequence,
                         run_fields, run_status, summarize_runs, write_csv)

COMMIT = '97cc585d9dbd9ac5e30b78ef2e2c4a0d8bfbe9f2'
WORKLOADS = {  # distilled klein (4 steps, no CFG) and klein Base (50 steps, true CFG 4)
    'distilled': {'width': 512, 'height': 512, 'steps': 4, 'guidance': 1.0, 'cfg_scale': 1.0,
                  'seed': 0, 'batch_size': 1, 'sampler': 'euler', 'scheduler': 'auto'},
    'base': {'width': 512, 'height': 512, 'steps': 50, 'guidance': 4.0, 'cfg_scale': 4.0,
             'seed': 0, 'batch_size': 1, 'sampler': 'euler', 'scheduler': 'auto'}}
SIZES = (512, 1024)


def is_jetson(cfg):
    return cfg['device_label'].startswith('jetson')
CACHE_METHODS = ('easycache', 'ucache', 'dbcache', 'taylorseer', 'cache-dit', 'magcache', 'dicache', 'sencache')
KNOWN = {'quantization', 'cpu_offload', 'vae_tiling', 'flash_attention', 'auto_fit',
         'auto_allocate', 'step_cache'}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 24), b''):
            h.update(block)
    return h.hexdigest()


def cli_arguments(cfg):
    """The only engine flags; each follows from one labelled optimization."""
    opt = cfg['optimizations']
    cache = opt.get('step_cache')
    args = ['--backend', 'cuda', '--type', 'preserve', '--cache_method', cache['method'] if cache else 'off',
            '--vae-tiling', 'on' if opt['vae_tiling'] else 'off', '--threads', '4']
    if not opt['flash_attention']:
        args.append('--no-flash-attention')
    return args


def check_config(cfg):
    opt = cfg['optimizations']
    check_optimizations(cfg, KNOWN, enabled_keys={'quantization', 'vae_tiling', 'flash_attention', 'step_cache'})
    cache = opt.get('step_cache')
    assert cache is None or (set(cache) == {'method'} and cache['method'] in CACHE_METHODS), (
        f'step_cache must be null or {{"method": one of {CACHE_METHODS}}} (engine default parameters)')
    assert type(opt['vae_tiling']) is bool and type(opt['flash_attention']) is bool
    assert cfg['engine']['commit'] == COMMIT
    w = cfg['workload']
    assert w['width'] == w['height'] and w['width'] in SIZES, 'square 512 or 1024 only'
    assert any({**w, 'width': 512, 'height': 512} == {**ref, 'prompt': w['prompt']} for ref in WORKLOADS.values()), (
        f'unsupported workload {w}')
    assert cfg['device_label'] in ('jetson-orin-nano', 'a100-pcie-40gb')
    assert w['prompt'].strip() and '\n' not in w['prompt']
    assert set(cfg['model']) == {'diffusion_model', 'llm', 'vae'}
    for part in cfg['model'].values():
        assert set(part) >= {'path', 'sha256'}, part
    assert cfg['protocol']['tegrastats_interval_ms' if is_jetson(cfg) else 'device_memory_sample_ms'] > 0
    return phase_sequence(cfg['protocol'])


def parse_log(log, cfg, phases, timing):
    """Refuse missing/reordered stages, changed schedules, changed weights or incomplete repeats."""
    n = len(phases)
    passes = re.findall(r'\[ed-sample\] pass (\d+)/(\d+)\s+1/1\s+seed=0\s+([0-9.]+)s', log)
    assert [(int(a), int(b)) for a, b, _ in passes] == [(i + 1, n) for i in range(n)], 'Incomplete repeats'
    assert timing['num_images'] == n
    wall = [float(p[2]) * 1000 for p in passes]
    assert all(math.isfinite(t) and t > 0 for t in wall)
    assert abs(sum(wall) / 1000 - timing['e2e_time']['total']) <= n * 0.000501
    markers = re.findall(r'\[\[phase\]\] stage=(\w+) event=(\w+) t=([0-9.]+)', log)
    names = [('encode', 'begin'), ('encode', 'end'), ('denoise', 'begin'),
             ('denoise', 'end'), ('decode', 'begin'), ('decode', 'end')]
    assert [(a, b) for a, b, _ in markers] == names * n, 'Missing or reordered phase markers'
    w = cfg['workload']; k = w['steps']
    schedule = re.findall(r'flux step (\d+)/' + str(k) + r' sigma=([\d.]+) next=([\d.]+)', log)
    if not cfg['optimizations'].get('step_cache'):  # caches may skip the per-step lines
        assert len(schedule) == k * n and schedule == schedule[:k] * n, 'Changed or incomplete schedule'
        assert [int(s[0]) for s in schedule[:k]] == list(range(1, k + 1))
    side = w['width'] // 16
    assert log.count(f"{w['width']}x{w['height']} latent={side}x{side} image_seq_len={side * side} steps={k}") == n
    assert log.count(f"guidance={w['guidance']:.2f} cfg={w['cfg_scale']:.2f} seed=0") == n
    breakdown = re.findall(r'flux generate breakdown: total=([\d.]+)s \| text_encode=([\d.]+)s '
                           r'sampling=([\d.]+)s vae_decode=([\d.]+)s other=([\d.]+)s', log)
    assert len(breakdown) == n, 'Missing engine stage breakdown'
    tiles = len(re.findall(r'processing \d+ tiles', log))
    assert tiles == (n if cfg['optimizations']['vae_tiling'] else 0), f'VAE tiling not as configured ({tiles})'
    cache = cfg['optimizations'].get('step_cache')
    mode = cache['method'] if cache else 'original'
    dit_stat = cfg['model']['diffusion_model'].get('expected_weight_type_stat', 'q4_0:80|bf16:69')
    for line in ['default backend: CUDA0', 'Diffusion model weight type stat: ' + dit_stat,
                 'VAE weight type stat: bf16:248', 'cache mode   : ' + mode]:
        assert line in log, 'Missing engine confirmation: ' + line
    skipped = None
    if cache:
        assert log.count('cache enabled: mode=') == n, 'Cache not enabled for every generation'
        found = re.findall(r'(\S+) (?:skipped|reused) (\d+)/(\d+) steps', log)
        skipped = [int(k) for _, k, total in found] if len(found) == n else None
    conditioner = re.search(r'Conditioner weight type stat: (\S+)', log)[1]
    assert conditioner == cfg['model']['llm']['expected_weight_type_stat'], conditioner
    rows, stages = [], []
    for i, phase in enumerate(phases):
        ts = [float(m[2]) for m in markers[i * 6:i * 6 + 6]]
        assert ts == sorted(ts)
        total, te, samp, vae, other = (float(x) * 1000 for x in breakdown[i])
        row = dict.fromkeys(run_fields(k))
        row.update(run_index=i, phase=phase, wall_ms=wall[i], text_encode_ms=te,
                   denoise_ms=samp, vae_decode_ms=vae, other_ms=other)
        rows.append(row)
        for name, m in [('encode_setup', 0), ('denoise', 2), ('vae_decode', 4)]:
            stages.append({'run_index': i, 'phase': phase, 'stage': name,
                           'host_marker_ms': (ts[m + 1] - ts[m]) * 1000, 'gpu_ms': None})
        stages.append({'run_index': i, 'phase': phase, 'stage': 'engine_breakdown_total',
                       'host_marker_ms': total, 'gpu_ms': None})
    return rows, stages, schedule[:k], skipped


def benchmark(cfg, out):
    phases = check_config(cfg)
    binary = Path(cfg['engine']['binary']).expanduser()
    source = binary.parents[2]
    commit = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    assert commit == COMMIT, commit
    assert not subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain', '--untracked-files=no'],
                                       text=True).strip(), 'Modified engine source'
    assert sha256(binary) == cfg['engine']['binary_sha256'], 'Changed engine binary'
    paths = {}
    for name, part in cfg['model'].items():
        path = Path(part['path']).expanduser()
        assert sha256(path) == part['sha256'], f'{name} hash mismatch'
        paths[name] = path
    env = host_environment(extra_env_vars=['LD_LIBRARY_PATH'])
    env.update(engine={**cfg['engine'], 'commit_verified': commit})
    if is_jetson(cfg):
        env.update(power_mode=subprocess.run(['nvpmodel', '-q'], capture_output=True, text=True).stdout,
                   l4t_release=Path('/etc/nv_tegra_release').read_text().splitlines()[0])
        assert 'NV Power Mode: ' + cfg['required_power_mode'] in env['power_mode'], env['power_mode']
    else:
        assert os.environ.get('SLURM_JOB_ID'), 'A100 inference requires a Slurm allocation'
        assert 'A100-PCIE-40GB' in env['nvidia_smi_gpu'], env['nvidia_smi_gpu']
    write_json(out / 'environment.json', env)
    root = Path(__file__).resolve().parents[1]
    write_json(out / 'source-provenance.json',
               {name: sha256(root / name) for name in ['scripts/run_edgedit_jetson.py', 'scripts/jetson_device.py',
                                                        'scripts/measurement.py', 'scripts/benchlib.py']})
    prompt = out / 'prompt.txt'
    prompt.write_text(cfg['workload']['prompt'] + '\n')
    cmd = [str(binary), '--diffusion-model', str(paths['diffusion_model']), '--llm', str(paths['llm']),
           '--vae', str(paths['vae']), *cli_arguments(cfg), '--prompt_file', str(prompt),
           '--output_dir', str(out / 'engine'), '--warmup', '0', '--repeat', str(len(phases))]
    w = cfg['workload']
    for key, flag in {'width': 'width', 'height': 'height', 'steps': 'num_steps', 'guidance': 'guidance_scale',
                      'cfg_scale': 'cfg_scale', 'seed': 'seed', 'sampler': 'sampler', 'scheduler': 'scheduler'}.items():
        cmd.extend(['--' + flag, str(w[key])])
    write_json(out / 'command.json', cmd)
    memory_snapshot(out / 'free-before.txt')
    if is_jetson(cfg):
        monitor = TegrastatsMonitor(out / 'tegrastats.log', cfg['protocol']['tegrastats_interval_ms'])
    else:
        from benchlib import DeviceMemorySampler, visible_gpu_uuid
        monitor = DeviceMemorySampler(cfg['protocol']['device_memory_sample_ms'], visible_gpu_uuid())
    monitor.start()
    try:
        with (out / 'engine.log').open('w') as stream:
            subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=3600)
    finally:
        monitor.stop()
        memory_snapshot(out / 'free-after.txt')
    timing = json.loads((out / 'engine/timing.json').read_text())
    rows, stages, schedule, skipped = parse_log((out / 'engine.log').read_text(), cfg, phases, timing)
    if cfg['optimizations'].get('step_cache'):
        write_json(out / 'step-cache.json', {'config': cfg['optimizations']['step_cache'], 'steps': cfg['workload']['steps'],
                                             'skipped_steps_per_generation': skipped,
                                             'note': 'engine-reported "<method> skipped|reused k/4 steps" lines (steps not fully computed); null if not printed'})
    from PIL import Image
    image = out / 'engine/imgs/img_000000.png'
    with Image.open(image) as im:
        im = im.convert('RGB')
        assert im.size == (cfg['workload']['width'], cfg['workload']['height'])
        rows[-1]['image_sha256'] = hashlib.sha256(im.tobytes()).hexdigest()
    shutil.copy2(image, out / 'output.png')
    write_json(out / 'image-retention.json', {'retained_run_index': len(rows) - 1, 'phase': phases[-1],
                                              'image': 'output.png', 'overwritten_earlier_images': True})
    write_csv(out / 'runs.csv', rows)
    write_csv(out / 'stages.csv', stages)
    load = {'load_total_s': timing['model_load_seconds'],
            'note': 'Upstream steady_clock around one context creation; files were hashed just before launch, '
                    'so the page cache is warm for this load.'}
    write_json(out / 'load.json', load)
    summary = summarize_runs(rows)
    summary.update(id=cfg['id'], device_label=cfg['device_label'], engine='edge-dit.cpp',
                   deterministic_output=None, load=load, schedule_sigmas=schedule,
                   system_memory=summarize_monitor(out) if is_jetson(cfg) else None,
                   device_used_peak_gib=(max((v for _, v in monitor.samples), default=0) / 2**30
                                         if not is_jetson(cfg) and monitor.samples else None),
                   residency=f'all weights resident; one load, {len(phases)} passes',
                   timing_method='ed-sample steady_clock per-pass stdout (1 ms rounding); stages from the '
                                 'engine "flux generate breakdown" line (10 ms rounding)',
                   precision_note='GGUF Q4_0 transformer, Q4_K_M encoder (see model), BF16 VAE; FP32 activations',
                   flash_attention_evidence='command-line flag only; the engine log does not report it',
                   formal_quality_eligible=False,
                   unavailable_metrics={
                       'denoise_step_i_ms': 'Engine prints completed steps without individual timestamps.',
                       'gpu_span_ms': 'No GPU event timing.',
                       'peak_alloc_gib/peak_reserved_gib/device_used_peak_gib/host_rss_gib':
                           ('No per-generation allocator observations; whole-system tegrastats only.' if is_jetson(cfg) else
                            'No per-generation allocator observations; whole-process NVML device peak only.'),
                       'deterministic_output': 'Only final repeat image retained; earlier image hashes unavailable.'})
    write_json(out / 'summary.json', summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--out-root', default=Path('results/runs'), type=Path)
    parser.add_argument('--kind', choices=['attempt', 'baseline'], default='baseline')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    check_config(cfg)
    if args.check_only:
        print('config ok')
        return
    out = create_run_directory(cfg, args.out_root, args.kind)
    with run_status(out):
        benchmark(cfg, out)


if __name__ == '__main__':
    main()
