#!/usr/bin/env python3
"""Single NVTX harness capture; not a repeated baseline runner. Standard library only."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--engine', type=Path, default=Path.home() / 'tools/stable-diffusion.cpp')
    p.add_argument('--binary', type=Path, default=Path.home() / 'tools/sd-bench-build/sd-bench-nvtx')
    p.add_argument('--models', type=Path, default=Path.home() / 'models/flux2-klein')
    a = p.parse_args()
    c = json.loads(a.config.read_text())
    r = Path.home() / 'results/runs' / (c['id'] + '__profile__' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    r.mkdir(parents=True)
    print(r, flush=True)
    write(r / 'config.json', c)
    write(r / 'status.json', {'status': 'running'})
    monitor = None
    try:
        commit = subprocess.check_output(['git', '-C', str(a.engine), 'rev-parse', 'HEAD'], text=True).strip()
        if commit != c['engine']['commit']:
            raise RuntimeError('Engine commit differs from config')
        version = subprocess.check_output(['nsys', '--version'], text=True)
        power = subprocess.check_output(['sudo', '-n', 'nvpmodel', '-q'], text=True)
        if 'NV Power Mode: ' + c['required_power_mode'] not in power:
            raise RuntimeError('Power mode differs from config')
        write(r / 'environment.json', {'engine_commit': commit, 'nsys': version, 'power_mode': power,
              'platform': list(os.uname()), 'binary_sha256': hashlib.sha256(a.binary.read_bytes()).hexdigest()})
        paths = []
        for part in c['model']['components']:
            path = a.models / part['local_file']
            h = hashlib.sha256()
            with path.open('rb') as f:
                for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
                    h.update(chunk)
            if h.hexdigest() != part['sha256']:
                raise RuntimeError('Weight hash mismatch: ' + str(path))
            paths.append(str(path))
        # Hash verification warms the filesystem cache; record this starting condition.
        cmd = [str(a.binary), '--out-dir', str(r), '--diffusion-model', paths[0], '--llm', paths[1], '--vae', paths[2]]
        for k, v in c['harness_arguments'].items():
            cmd.extend(['--' + k, str(v)])
        cmd = ['nsys', 'profile', '--trace=cuda,nvtx,osrt', '--sample=none', '--cpuctxsw=none', '--output=' + str(r / 'trace')] + cmd
        write(r / 'command.json', {'argv': cmd, 'cache_condition': 'All weight files SHA256-read immediately before capture; no cache flush.'})
        (r / 'memory-before.txt').write_text(subprocess.check_output(['free', '-h'], text=True))
        with (r / 'tegrastats.log').open('w') as mon, (r / 'profile.log').open('w') as log:
            monitor = subprocess.Popen(['tegrastats', '--interval', '1000'], stdout=mon, stderr=subprocess.STDOUT)
            code = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, env={**os.environ, 'HF_HUB_OFFLINE': '1'})
        if code:
            raise RuntimeError('Profiler/harness exit code ' + str(code))
        rows = [json.loads(line) for line in (r / 'results.jsonl').read_text().splitlines()]
        generations = [x for x in rows if x['event'] == 'generate']
        if len(generations) != 1 or not generations[0]['ok']:
            raise RuntimeError('Expected one successful generation')
        g = generations[0]
        if len(g['t_step_end']) != 4 or g['progress_calls'] != 5 or g['cond_cache_hits']:
            raise RuntimeError('Unexpected step callbacks or conditioning reuse')
        times = [g['t_start'], g['t_cond'], g['t_sampling_start'], *g['t_step_end'], g['t_sampling_end'], g['t_decode_end'], g['t_end']]
        if any(x <= 0 for x in times) or times != sorted(times):
            raise RuntimeError('Missing or unordered stage boundaries')
        with (r / 'nsys-stats.txt').open('w') as f:
            subprocess.run(['nsys', 'stats', '--report', 'nvtx_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_api_sum', str(r / 'trace.nsys-rep')], stdout=f, stderr=subprocess.STDOUT, check=True)
        stats = (r / 'nsys-stats.txt').read_text()
        for name in ['text_encode', 'vae_decode'] + ['denoise_step_' + str(i) for i in range(4)]:
            if name not in stats:
                raise RuntimeError('Missing NVTX label: ' + name)
        write(r / 'load.json', next(x for x in rows if x['event'] == 'load'))
        write(r / 'summary.json', {'record_kind': 'single stage-labelled profile', 'measured': None,
              'stage_callbacks_validated': True, 'nvtx_labels_present': True,
              'unavailable_metrics': {'baseline': 'No repeated unprofiled generations', 'memory': 'Raw whole-system tegrastats only; no NVML or per-stage peak derived', 'gpu_stage_time': 'Timeline analysis pending'}})
        # These tables intentionally contain no baseline measurement rows.
        (r / 'runs.csv').write_text('run_index,wall_ms\n')
        (r / 'stages.csv').write_text('run_index,stage,wall_ms\n')
        write(r / 'status.json', {'status': 'complete', 'trace_timeline_review_pending': True})
    except BaseException as e:
        write(r / 'status.json', {'status': 'failed', 'error': str(e)})
        raise
    finally:
        if monitor is not None:
            monitor.terminate()
            monitor.wait()
        (r / 'memory-after.txt').write_text(subprocess.check_output(['free', '-h'], text=True))
        print('Artifacts: ' + str(r), flush=True)


if __name__ == '__main__':
    main()
