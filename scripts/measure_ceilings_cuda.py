#!/usr/bin/env python3
"""Measured roofline ceilings on one CUDA GPU: memory bandwidth and dense GEMM throughput per precision.

  python scripts/measure_ceilings_cuda.py --config configs/a100-ceilings.json [--out-root results/runs]

No model is loaded. Bandwidth kernels are PyTorch's own copy / reduction / fill / add (STREAM-style;
bytes counted as the minimum each kernel must move, decimal GB/s like the datasheets). GEMM throughput
is 2*M*N*K per call through cuBLAS for each precision in the config, at square sizes and at the
FLUX.2 klein shapes listed in the config. Each test: warm-up calls, then `trials` timed blocks of
`iters` calls between CUDA events. A sustained block runs the BF16 GEMM for `sustained_seconds` to
expose power/thermal clock reduction. Clocks are not locked (no root on shared nodes); nvidia-smi is
sampled throughout so the observed SM/memory clocks, power and throttle reasons are recorded.

Writes config.json, environment.json, status.json, measurements.csv (one row per timed block),
gpu-samples.csv, summary.json (median and best per test, ratio to the datasheet ceiling) and
hardware-override.json (the measured ceilings in plot_roofline.py's --hardware-specs format).
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchlib import host_environment, write_json  # noqa: E402
from measurement import create_run_directory, run_status, write_csv  # noqa: E402

GIB = 2**30
SMI_FIELDS = 'timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,clocks_throttle_reasons.active'


def timed(torch, fn, iters):
    """Milliseconds per call over one block of `iters` back-to-back calls."""
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iters


def run_test(torch, rows, test, fn, work, unit, protocol, **labels):
    for _ in range(protocol['warmup_iters']):
        fn()
    torch.cuda.synchronize()
    for trial in range(protocol['trials']):
        ms = timed(torch, fn, protocol['iters'])
        rows.append({'test': test, **labels, 'trial': trial, 'ms_per_call': ms,
                     'throughput': work / (ms * 1e-3) / 1e12 if unit != 'GB/s' else work / (ms * 1e-3) / 1e9,
                     'unit': unit})
    print(f"{test:10s} {labels} {statistics.median(r['throughput'] for r in rows[-protocol['trials']:]):10.1f} {unit}", flush=True)


def bandwidth(torch, rows, cfg):
    n = int(cfg['bandwidth']['buffer_gib'] * GIB) // 4
    a, b, c = (torch.empty(n, device='cuda', dtype=torch.float32).uniform_() for _ in range(3))
    out = torch.empty((), device='cuda', dtype=torch.float32)
    size = n * 4
    tests = {'copy': (lambda: b.copy_(a), 2 * size),             # read a, write b
             'read': (lambda: torch.sum(a, dim=0, out=out), size),      # read a (reduction)
             'write': (lambda: c.fill_(1.0), size),              # write c
             'triad': (lambda: torch.add(b, c, alpha=0.5, out=a), 3 * size)}  # read b, c; write a
    for name in cfg['bandwidth']['kernels']:
        fn, moved = tests[name]
        run_test(torch, rows, 'bandwidth', fn, moved, 'GB/s', cfg['protocol'], kernel=name, dtype='fp32',
                 shape=f'{n} elements x4 B')
    del a, b, c


def gemm_operands(torch, dtype, m, n, k):
    # nn.Linear layout: weight stored (N, K) and used transposed (column-major K x N). cuBLAS picks its
    # INT8 tensor-core kernels only for this layout; a row-major (K, N) operand runs about 9x slower.
    if dtype == 'int8':
        x = torch.randint(-64, 64, (m, k), device='cuda', dtype=torch.int8)
        w = torch.randint(-64, 64, (n, k), device='cuda', dtype=torch.int8)
        return lambda: torch._int_mm(x, w.t()), 'TOP/s'
    torch_dtype = {'bf16': torch.bfloat16, 'fp16': torch.float16, 'tf32': torch.float32, 'fp32': torch.float32}[dtype]
    x = torch.randn(m, k, device='cuda', dtype=torch_dtype)
    w = torch.randn(n, k, device='cuda', dtype=torch_dtype)
    return lambda: torch.matmul(x, w.t()), 'TFLOP/s'


def gemm(torch, rows, cfg):
    shapes = [{'label': f'square {s}', 'mnk': [s, s, s]} for s in cfg['gemm']['square_sizes']] + cfg['gemm']['model_shapes']
    for dtype in cfg['gemm']['dtypes']:
        # TF32 and FP32 differ only in this switch; set it explicitly for every precision.
        torch.backends.cuda.matmul.allow_tf32 = dtype == 'tf32'
        for shape in shapes:
            m, n, k = shape['mnk']
            fn, unit = gemm_operands(torch, dtype, m, n, k)
            run_test(torch, rows, 'gemm', fn, 2 * m * n * k, unit, cfg['protocol'], kernel='matmul', dtype=dtype,
                     shape=shape['label'] + f' M{m} N{n} K{k}')
            torch.cuda.empty_cache()
    torch.backends.cuda.matmul.allow_tf32 = False


def sustained(torch, rows, cfg):
    spec = cfg['sustained']
    m, n, k = spec['mnk']
    torch.backends.cuda.matmul.allow_tf32 = False
    fn, unit = gemm_operands(torch, spec['dtype'], m, n, k)
    t0 = time.monotonic()
    block = 0
    while time.monotonic() - t0 < spec['seconds']:
        ms = timed(torch, fn, cfg['protocol']['iters'])
        rows.append({'test': 'sustained', 'kernel': 'matmul', 'dtype': spec['dtype'], 'shape': f'M{m} N{n} K{k}',
                     'trial': block, 'ms_per_call': ms, 'throughput': 2 * m * n * k / (ms * 1e-3) / 1e12,
                     'unit': unit, 'elapsed_s': round(time.monotonic() - t0, 2)})
        block += 1


def summarize(rows, cfg):
    groups = {}
    for r in rows:
        groups.setdefault((r['test'], r['kernel'], r['dtype'], r['shape'], r['unit']), []).append(r['throughput'])
    tests = [{'test': t, 'kernel': kn, 'dtype': d, 'shape': s, 'unit': u, 'median': statistics.median(v),
              'best': max(v), 'min': min(v), 'n': len(v)} for (t, kn, d, s, u), v in groups.items()]
    datasheet = cfg['datasheet']
    ceilings = {}
    bw = [x for x in tests if x['test'] == 'bandwidth']
    best_bw = max(bw, key=lambda x: x['median'])
    ceilings['bw_gbs'] = {'measured': best_bw['median'], 'kernel': best_bw['kernel'],
                          'datasheet': datasheet['bw_gbs'], 'ratio': best_bw['median'] / datasheet['bw_gbs']}
    for dtype in cfg['gemm']['dtypes']:
        g = max((x for x in tests if x['test'] == 'gemm' and x['dtype'] == dtype), key=lambda x: x['median'])
        key = datasheet['ceiling_keys'][dtype]
        ceilings[key] = {'measured': g['median'], 'shape': g['shape'], 'unit': g['unit'],
                         'datasheet': datasheet['ceilings'][key], 'ratio': g['median'] / datasheet['ceilings'][key]}
    sus = [r['throughput'] for r in rows if r['test'] == 'sustained']
    sustained_summary = None
    if sus:
        q = max(1, len(sus) // 4)
        sustained_summary = {'blocks': len(sus), 'first_quarter_median': statistics.median(sus[:q]),
                             'last_quarter_median': statistics.median(sus[-q:]),
                             'drop_pct': 100 * (1 - statistics.median(sus[-q:]) / statistics.median(sus[:q]))}
    return {'tests': tests, 'ceilings': ceilings, 'sustained': sustained_summary,
            'ceiling_rule': 'per test: median over timed blocks; per ceiling: the shape/kernel with the highest median',
            'bytes_rule': 'minimum bytes each kernel must move; decimal GB (1e9) as in the datasheet'}


def gpu_samples(path):
    """Background nvidia-smi sampler; returns the process (stopped by the caller)."""
    stream = path.open('w')
    return subprocess.Popen(['nvidia-smi', f'--query-gpu={SMI_FIELDS}', '--format=csv,nounits', '-lms', '500'],
                            stdout=stream, stderr=subprocess.STDOUT), stream


def sample_summary(path):
    lines = [l.split(', ') for l in path.read_text().splitlines()[1:] if l and l[0].isdigit()]
    if not lines:
        return None
    sm, mem, power, temp = ([float(x[i]) for x in lines if x[i].replace('.', '').isdigit()] for i in (1, 2, 3, 4))
    return {'samples': len(lines), 'sm_clock_mhz': {'min': min(sm), 'median': statistics.median(sm), 'max': max(sm)},
            'mem_clock_mhz': {'min': min(mem), 'max': max(mem)}, 'power_w_max': max(power), 'temp_c_max': max(temp),
            'throttle_reasons_seen': sorted({x[5].strip() for x in lines})}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--out-root', type=Path, default=Path('results/runs'))
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())
    import torch
    assert torch.cuda.is_available(), 'no CUDA device'
    name = torch.cuda.get_device_name(0)
    assert name == cfg['expected_gpu_name'], f'GPU is {name}, config expects {cfg["expected_gpu_name"]}'
    run_dir = create_run_directory(cfg, args.out_root, 'ceilings')
    with run_status(run_dir) as status:
        write_json(run_dir/'environment.json', {**host_environment(), 'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
                   'torch': torch.__version__,
                   'torch_cuda_runtime': torch.version.cuda, 'gpu_name': name,
                   'cublas_flags': {'allow_fp16_reduced_precision_reduction': torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
                                    'allow_bf16_reduced_precision_reduction': torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction}})
        sampler, stream = gpu_samples(run_dir/'gpu-samples.csv')
        rows = []
        try:
            bandwidth(torch, rows, cfg)
            gemm(torch, rows, cfg)
            sustained(torch, rows, cfg)
        finally:
            sampler.terminate()
            sampler.wait()
            stream.close()
            if rows:
                fields = ['test', 'kernel', 'dtype', 'shape', 'trial', 'ms_per_call', 'throughput', 'unit', 'elapsed_s']
                write_csv(run_dir/'measurements.csv', [{f: r.get(f) for f in fields} for r in rows])
        summary = summarize(rows, cfg)
        summary['gpu_samples'] = sample_summary(run_dir/'gpu-samples.csv')
        write_json(run_dir/'summary.json', summary)
        c = summary['ceilings']
        write_json(run_dir/'hardware-override.json', {cfg['device_label']: {
            'bw_gbs': c['bw_gbs']['measured'],
            'ceilings': {k: v['measured'] for k, v in c.items() if k != 'bw_gbs'},
            'source': f'measured: {run_dir.name}', 'note': 'Measured sustained ceilings (median of timed blocks); clocks not locked.'}})
        status['ceilings'] = {k: round(v['measured'], 1) for k, v in c.items()}
    print('done', run_dir)


if __name__ == '__main__':
    main()
