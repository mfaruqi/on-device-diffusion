#!/usr/bin/env python3
"""Gated Nsight Compute capture of an unchanged portable-runner configuration.

No downloads, automatic metric substitutions, or inferred total FLOP counts.
The baseline protocol runs with capture disabled, followed by one captured image.
"""
import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import os
import signal
from pathlib import Path
import subprocess
import sys

from benchlib import REPO_ROOT, write_json
from measurement import create_run_directory, run_status

DURATION = 'gpu__time_duration.sum'
READ = 'dram__bytes_read.sum'
WRITE = 'dram__bytes_write.sum'
TENSOR = [f'sm__ops_path_tensor_src_{dtype}_dst_fp32_sparsity_off.sum'
          for dtype in ('bf16', 'fp16', 'tf32')]
SCALAR = [f'smsp__sass_thread_inst_executed_op_{op}_pred_on.sum'
          for op in ('fadd', 'fmul', 'ffma')]
METRICS = [DURATION, READ, WRITE, *TENSOR, *SCALAR]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def command(argv, log, timeout):
    print('Executing:', ' '.join(map(str, argv)), flush=True)
    with Path(log).open('w') as stream:
        proc = subprocess.Popen(list(map(str, argv)), stdout=stream, stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            returncode = proc.wait(timeout=timeout)
        except BaseException:
            # NCU launches an inference child: terminate the whole owned process group.
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            raise
        if returncode:
            raise subprocess.CalledProcessError(returncode, argv)


def read_counters(path):
    """Parse raw NCU CSV; fail on missing, unavailable, or nonfinite counters."""
    lines = Path(path).read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith('"ID",'))
    reader = csv.DictReader(io.StringIO('\n'.join(lines[start:])))
    if 'Metric Name' in reader.fieldnames:
        records = list(reader)
    else:
        # NCU 2026 raw CSV is wide: one units row, then one row per launch.
        missing = set(METRICS) - set(reader.fieldnames)
        if missing:
            raise ValueError(f'Missing counter columns: {missing}')
        units = next(reader, None)
        if units is None or units.get('ID'):
            raise ValueError('Missing wide CSV units row')
        records = []
        for launch in reader:
            if not launch.get('ID'):
                raise ValueError('Unexpected empty launch ID in wide CSV')
            for metric in METRICS:
                records.append({**launch, 'Metric Name': metric,
                                'Metric Unit': units[metric], 'Metric Value': launch[metric]})
    groups = {}
    for row in records:
        metric = row.get('Metric Name')
        if metric not in METRICS:
            continue
        key = (row['ID'], row.get('Process ID', ''), row.get('Context', ''), row.get('Stream', ''))
        group = groups.setdefault(key, {'id': row['ID'], 'kernel': row['Kernel Name'], 'metrics': {}})
        value = float(row['Metric Value'].replace(',', ''))
        if not math.isfinite(value) or value < 0:
            raise ValueError(f'Invalid counter {metric}: {value}')
        if metric in group['metrics']:
            raise ValueError(f'Duplicate metric for kernel {key}: {metric}')
        unit = row['Metric Unit']
        expected = {'ns', 'nsecond'} if metric == DURATION else {'byte'} if metric in (READ, WRITE) else None
        if expected and unit not in expected:
            raise ValueError(f'Unexpected unit for {metric}: {unit}')
        group['metrics'][metric] = value
    if not groups:
        raise ValueError('No captured kernels in NCU export')
    for group in groups.values():
        values = group['metrics']
        if set(values) != set(METRICS):
            raise ValueError(f'Missing counters: {set(METRICS) - set(values)}')
        seconds = values[DURATION] * 1e-9
        if seconds <= 0:
            raise ValueError('Nonpositive kernel duration')
        traffic = values[READ] + values[WRITE]
        tensor = sum(values[name] for name in TENSOR)
        fp32 = values[SCALAR[0]] + values[SCALAR[1]] + 2 * values[SCALAR[2]]
        group.update(duration_s=seconds, hbm_bytes=traffic, tensor_ops=tensor,
                     fp32_add_mul_fma_ops=fp32, counted_ops=tensor + fp32,
                     counted_ops_per_byte=(tensor + fp32) / traffic if traffic else None,
                     counted_tops=(tensor + fp32) / seconds / 1e12)
    return list(groups.values())


def ncu_command(plan, report, child):
    return [plan['ncu'], '--target-processes', 'all', '--profile-from-start', 'off',
            '--replay-mode', 'kernel', '--cache-control', plan['cache_control'],
            '--clock-control', 'none', '--nvtx', '--metrics', ','.join(METRICS),
            '--export', str(report), sys.executable, str(Path(__file__).resolve()), *child]


def export_report(plan, report, output, rename=False):
    args = [plan['ncu'], '--import', str(report) + '.ncu-rep', '--page', 'raw',
            '--csv', '--print-units', 'base', '--print-fp']
    if rename:
        args += ['--print-nvtx-rename', 'kernel']
    command(args, output, 300)


def check_environment(plan, output):
    import torch
    import diffusers
    import transformers
    actual = {'torch': torch.__version__, 'diffusers': diffusers.__version__,
              'transformers': transformers.__version__, 'gpu_name': torch.cuda.get_device_name(0),
              'torch_cuda_runtime': torch.version.cuda, 'cudnn': torch.backends.cudnn.version()}
    write_json(output / 'environment-gate.json', {'expected': plan['expected_environment'], 'actual': actual})
    for key, value in plan['expected_environment'].items():
        if actual[key] != value:
            raise ValueError(f'Environment mismatch {key}: expected {value}, got {actual[key]}')
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability() != (8, 0):
        raise ValueError('Requires exactly one visible SM80 A100')


def probe(plan, output):
    import torch
    check_environment(plan, output)
    a = torch.ones((2048, 2048), device='cuda', dtype=torch.bfloat16)
    b = torch.ones_like(a)
    for _ in range(3):
        a @ b
    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    try:
        result = a @ b
        torch.cuda.synchronize()
    finally:
        torch.cuda.profiler.stop()
    assert torch.isfinite(result).all().item()


def capture(plan, output):
    import torch
    import run_flux
    import run_flux_portable as portable
    from torch_stages import StageRecorder

    check_environment(plan, output)
    cfg = read_json(output / 'config.json')
    original_measure = run_flux.measure_generations
    original_start, original_end = StageRecorder.start, StageRecorder.end
    active = [False]
    stage_names = []

    def stage_start(self, name):
        original_start(self, name)
        if active[0]:
            stage_names.append(name)
            torch.cuda.nvtx.range_push(name)

    def stage_end(self):
        if active[0]:
            torch.cuda.nvtx.range_pop()
        original_end(self)

    def measure(generate, phases, steps, run_dir, sampler, process):
        rows = original_measure(generate, phases, steps, run_dir, sampler, process)
        expected = plan['expected_image_sha256']
        if any(row['image_sha256'] != expected for row in rows):
            raise ValueError('Unprofiled image differs from the saved reference; capture blocked')
        write_json(output / 'profile' / 'image-gate.json', {'status': 'passed', 'sha256': expected})
        torch.cuda.synchronize()
        active[0] = True
        torch.cuda.profiler.start()
        torch.cuda.nvtx.range_push('profile_capture')
        try:
            image, result = generate()
            torch.cuda.synchronize()
        finally:
            torch.cuda.nvtx.range_pop()
            torch.cuda.profiler.stop()
            active[0] = False
        image.save(output / 'images' / 'counter-capture.png')
        row = run_flux.result_row(len(phases), 'counter_capture', image, result, steps, sampler, process)
        write_json(output / 'profile' / 'captured-generation.json',
                   {'row': row, 'stages': result['stages'], 'nvtx_stages': stage_names,
                    'timing_scope': 'NCU replay instrumented; excluded from benchmark statistics'})
        if row['image_sha256'] != expected:
            raise ValueError('Counter-captured image differs from the saved reference')
        return rows

    StageRecorder.start, StageRecorder.end = stage_start, stage_end
    run_flux.measure_generations = measure
    try:
        portable.benchmark(cfg, output)
    finally:
        StageRecorder.start, StageRecorder.end = original_start, original_end
        run_flux.measure_generations = original_measure
    summary = read_json(output / 'summary.json')
    summary['profiler'] = 'nsight-compute'
    summary['timing_scope'] = 'profiler-attached, capture-disabled prefix; not a new unprofiled baseline'
    summary['counter_capture'] = 'profile/captured-generation.json'
    write_json(output / 'summary.json', summary)


def stage_of(label):
    """NVTX-renamed kernel names are '<innermost range>/<kernel>'; the stage is the range part."""
    return label.split('/', 1)[0] if '/' in label else label


def finalize(cfg, output, offline=False):
    """Validate and summarise the two exports already on disk (no GPU); re-runnable on a saved run."""
    profile = output / 'profile'
    rows = read_counters(profile / 'counters.csv')
    named = read_counters(profile / 'counter-stages.csv')
    if [r['id'] for r in rows] != [r['id'] for r in named]:
        raise ValueError('NVTX-labelled export does not match kernel IDs')
    for row, label in zip(rows, named):
        row['nvtx_label'] = label['kernel']
        row['stage'] = stage_of(label['kernel'])
    names = {row['stage'] for row in rows}
    required = {'text_encode', 'vae_decode', *[f'denoise_step_{i}' for i in range(cfg['workload']['num_inference_steps'])]}
    if not required <= names:
        raise ValueError(f'Missing captured stage labels: {required - names}')
    with gzip.open(profile / 'counter-kernels.json.gz', 'wt') as out:  # about 20x smaller; one row per launch
        json.dump(rows, out, indent=2)
    write_json(profile / 'counter-summary.json', {
        'status': 'validated', 'kernel_count': len(rows), 'metrics': METRICS,
        'hbm_bytes_sum': sum(r['hbm_bytes'] for r in rows),
        'counted_tensor_ops_sum': sum(r['tensor_ops'] for r in rows),
        'counted_fp32_add_mul_fma_ops_sum': sum(r['fp32_add_mul_fma_ops'] for r in rows),
        'by_stage': {st: {'kernels': sum(1 for r in rows if r['stage'] == st),
                          'hbm_bytes': sum(r['hbm_bytes'] for r in rows if r['stage'] == st),
                          'tensor_ops': sum(r['tensor_ops'] for r in rows if r['stage'] == st),
                          'fp32_add_mul_fma_ops': sum(r['fp32_add_mul_fma_ops'] for r in rows if r['stage'] == st)}
                     for st in sorted(names)},
        'scope': 'One warmed full image; kernel replay; memcpy and CPU work excluded',
        'limitations': 'Counted BF16/FP16/TF32 dense Tensor Core ops and FP32 add/mul/FMA only; not all arithmetic. '
                      'No FP64, scalar FP16, SFU or integer accounting. Replay can change cache state and serialize kernels. '
                      'Summed kernel duration is not stage elapsed time; do not use this as normal generation latency.'})
    if not offline:
        command(['nvidia-smi', '-q'], profile / 'nvidia-smi-after.txt', 30)
    write_json(output / 'validation.json', {'status': 'passed', 'reference_image_match': True,
                                           'counter_access': True, 'required_stages_present': True,
                                           **({'finalized_offline': True} if offline else {})})


def orchestrate(plan, plan_path, out_root):
    reference = REPO_ROOT / plan['reference_config']
    if digest(reference) != plan['reference_config_sha256']:
        raise ValueError('Reference config hash changed')
    cfg = read_json(reference)
    if cfg['backend'] != 'cuda' or any(cfg['optimizations'].values()):
        raise ValueError('Initial capture supports CUDA eager reference with no enabled optimizations only')
    output = create_run_directory(cfg, out_root, 'profile')
    write_json(output / 'counter-plan.json', plan)
    profile = output / 'profile'
    profile.mkdir()
    with run_status(output):
        if not os.environ.get('SLURM_JOB_ID') or os.environ.get('HF_HUB_OFFLINE') != '1':
            raise ValueError('Run inside an offline Slurm GPU allocation')
        write_json(output / 'capture-source-provenance.json',
                   {str(p.relative_to(REPO_ROOT)): digest(p) for p in sorted((REPO_ROOT / 'scripts').glob('*.py'))})
        command(['nvidia-smi', '-q'], profile / 'nvidia-smi-before.txt', 30)
        command([plan['ncu'], '--version'], profile / 'ncu-version.txt', 30)
        command([plan['ncu'], '--query-metrics', '--query-metrics-mode', 'suffix',
                 '--metrics', ','.join(METRICS)], profile / 'available-metrics.txt', 60)
        child = ['--plan', str(plan_path.resolve()), '--run-dir', str(output.resolve())]
        command(ncu_command(plan, profile / 'probe', [*child, '--child', 'probe']),
                profile / 'probe.log', plan['probe_timeout_s'])
        export_report(plan, profile / 'probe', profile / 'probe.csv')
        probe_rows = read_counters(profile / 'probe.csv')
        if not any(r['metrics'][TENSOR[0]] > 0 and r['hbm_bytes'] > 0 for r in probe_rows):
            raise ValueError('Probe did not confirm BF16 Tensor Core operations and HBM traffic')
        write_json(profile / 'counter-access.json', {'status': 'passed', 'kernels': len(probe_rows)})
        command(ncu_command(plan, profile / 'image', [*child, '--child', 'capture']),
                profile / 'capture.log', plan['capture_timeout_s'])
        export_report(plan, profile / 'image', profile / 'counters.csv')
        export_report(plan, profile / 'image', profile / 'counter-stages.csv', rename=True)
        finalize(cfg, output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--out-root', type=Path, default=REPO_ROOT / 'results' / 'runs')
    parser.add_argument('--child', choices=['probe', 'capture'])
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--finalize', type=Path, help='re-run validation/summary on a saved run directory (no GPU)')
    args = parser.parse_args()
    plan = read_json(args.plan)
    if args.finalize:
        finalize(read_json(args.finalize / 'config.json'), args.finalize, offline=True)
        return
    if args.child:
        if args.child == 'probe':
            probe(plan, args.run_dir)
        else:
            capture(plan, args.run_dir)
    else:
        orchestrate(plan, args.plan, args.out_root)


if __name__ == '__main__':
    main()
