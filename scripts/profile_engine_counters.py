#!/usr/bin/env python3
"""Nsight Compute memory bytes for one steady denoising pass of any CUDA engine (no engine or library changes).

  python scripts/profile_engine_counters.py --plan configs/ncu/<plan>.json

The plan names a completed benchmark run (template) and the launcher of its runner. The template's config
is reused unchanged except the protocol (1 first + 0 warm-up + 1 measured generation) and run under ncu
in two passes of the same deterministic command:
  1. sequence: every kernel launch with gpu__time_duration only (one hardware pass, no replay restore);
     the kernel-name sequence is searched for its denoising period L: among all L whose name sequence
     repeats at least `steps - 2` consecutive times, the one whose last full repeat carries the most kernel
     time (a text encoder's layers also repeat, and fused compiled steps can be shorter than a block of them).
  2. counters: the same metrics as profile_a100_counters.py on a window of 1.5 L launches around that pass
     (--launch-skip/--launch-count). The captured names must equal a slice of pass 1 exactly, at a single
     offset within the margin (one-time setup kernels before the window can differ in number between
     processes; the offset is recorded); otherwise the capture fails with a record rather than guessing. Names are compared without
     namespace qualifiers: ncu demangles the same template kernel differently between passes
     (`arch::Sm80` vs `cutlass::Sm80`).
An optional warm run without ncu first fills compiler caches (torch.compile), so both passes launch the
same kernels. Launcher `edgedit-direct` runs the template's engine binary itself (its command.json, with
2 repeats and new output paths) instead of the runner, whose NVML/nvidia-smi setup hangs under ncu process
injection. --resume <run dir> re-analyses a capture whose CSVs exist without running ncu again. Bytes per pass are measured; FLOPs per pass stay analytic (plan) and the counted Tensor Core
+ FP32 work is reported next to them as a check. Kernel durations are replay times, not benchmark latency.

Devices: the plan's `device` selects the byte counters (profile_a100_counters.BYTE_BASES). `a100` (default) runs in
a Slurm allocation with DRAM bytes and kernel replay. `jetson` runs as root on the board (Nsight Compute needs it
there), counts L2 <-> system-memory sectors, and also uses kernel replay: application replay with any metric besides
gpu__time_duration fails on this board (scripts/jetson_ncu_probe3.sh). Read and write sectors together need 3 hardware
passes, and between passes kernel replay saves and restores device memory in the same 8 GB the model occupies (an
sd.cpp window left 290 MB). Each counter alone fits in one pass with no save (scripts/jetson_ncu_probe4.sh), so on the
Jetson the window runs once per counter (duration + that counter; profile/window-<i>.csv) and the runs are merged per
launch into window.csv, after checking that every run captured the same kernel sequence. Kernel durations come from
the first (read) run. The plan's `tensor_dtypes` limits the Tensor Core counters (default bf16, fp16, tf32), since
each costs one more run.

Writes results/runs/<template id>__counters__<stamp>/: config.json (derived), counter-plan.json,
status.json, profile/sequence.csv.gz, profile/window.csv, profile/pass-kernels.json, profile/counter-summary.json.
"""
import argparse
import csv
import re
import datetime as dt
import gzip
import io
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchlib import REPO_ROOT, write_json  # noqa: E402
from measurement import create_run_directory, run_status  # noqa: E402
import profile_a100_counters as counters  # noqa: E402

DEVICES = {  # device -> byte counters, replay mode, one ncu run per counter (single hardware pass each)
    'a100': ('dram', 'kernel', False),
    'jetson': ('jetson-l2-sysmem', 'kernel', True),
}
BASIS_TEXT = {
    'dram': 'Nsight Compute dram read + write',
    'jetson-l2-sysmem': 'Nsight Compute L2 read misses + L2 write sectors to system memory (32 B sectors; writes are an upper bound)',
}
RUNNERS = {  # launcher -> runner script and its arguments (the template's own runner)
    'portable': ['scripts/run_flux_portable.py', '--kind', 'attempt'],
    'jetson-sdcpp': ['scripts/run_jetson_sdcpp.py', '--kind', 'attempt'],
    'nunchaku': ['scripts/run_nunchaku.py', '--device', 'a100'],
    'sdcpp': ['scripts/run_sdcpp.py', '--kind', 'attempt'],
    'edgedit': ['scripts/run_edgedit.py', '--kind', 'attempt'],
}


def run(cmd, log, timeout):
    # Own process group: on a timeout the profiled application is killed with ncu, not left holding memory.
    try:
        counters.command(cmd, log, timeout)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f'{cmd[0]} exited {error.returncode}; see {log}') from None


def runner_command(plan, cfg_path, out_root, template):
    if plan['launcher'] == 'edgedit-direct':
        cmd = json.loads((template / 'command.json').read_text())
        out_root.mkdir(parents=True, exist_ok=True)
        prompt = out_root / 'prompt.txt'
        prompt.write_text((template / 'prompt.txt').read_text())
        paths = {'--prompt_file': prompt, '--output_dir': out_root / 'engine', '--repeat': 2}  # 1 first + 1 measured
        return [str(paths.get(cmd[i - 1], c)) if i else c for i, c in enumerate(cmd)]
    script, *args = RUNNERS[plan['launcher']]
    return [plan['python'], str(REPO_ROOT / script), '--config', str(cfg_path), *args, *plan.get('runner_args', []),
            '--out-root', str(out_root)]


def ncu(plan, metrics, report, inner, window=None):
    replay = DEVICES[plan.get('device', 'a100')][1]
    cmd = [plan['ncu'], '--target-processes', 'all', '--replay-mode', replay, '--cache-control', 'none',
           '--clock-control', 'none', '--metrics', ','.join(metrics), '--export', str(report)]
    if window:
        cmd += ['--launch-skip', str(window[0]), '--launch-count', str(window[1])]
    return cmd + inner


def sequence(path):
    """(process id, kernel name, duration ns) per launch, in launch order, from a raw ncu CSV (or .csv.gz)."""
    path = Path(path)
    lines = (gzip.open(path, 'rt').read() if path.suffix == '.gz' else path.read_text()).splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith('"ID",'))
    rows = list(csv.DictReader(io.StringIO('\n'.join(lines[start:]))))
    if rows and not rows[0]['ID']:
        rows = rows[1:]  # units row of the wide format
    out = []
    for r in rows:
        value = r.get(counters.DURATION) or r.get('Metric Value')
        out.append((r['Process ID'], r['Kernel Name'], float(str(value).replace(',', ''))))
    return out


def merge_windows(parts, output):
    """One wide CSV from window CSVs of the same launches, each with duration + other counters (duration from the first)."""
    tables = []
    for path in parts:
        lines = path.read_text().splitlines()
        start = next(i for i, line in enumerate(lines) if line.startswith('"ID",'))
        tables.append(list(csv.reader(lines[start:])))
    head, units, *rows = tables[0]
    names = [plain(r[head.index('Kernel Name')]) for r in rows]
    for path, (h, u, *r) in zip(parts[1:], tables[1:]):
        if [plain(x[h.index('Kernel Name')]) for x in r] != names:
            raise ValueError(f'{path.name} captured a different kernel sequence than {parts[0].name}')
        new = [i for i, c in enumerate(h) if c not in head]
        head, units = head + [h[i] for i in new], units + [u[i] for i in new]
        rows = [a + [b[i] for i in new] for a, b in zip(rows, r)]
    with open(output, 'w', newline='') as f:
        csv.writer(f, quoting=csv.QUOTE_ALL).writerows([head, units, *rows])


def plain(name):
    """Kernel name without namespace qualifiers (demangling of template arguments differs between ncu passes), and
    with PyTorch's vectorized/unrolled variants of the same element-wise op made equal: the variant is picked at launch
    from pointer alignment, which differs between processes (seen in Jetson torch.compile captures, 2026-11-01)."""
    name = re.sub(r'\b\w+::', '', name)
    return re.sub(r'\b(?:vectorized_elementwise_kernel<\d+, |unrolled_elementwise_kernel<)(.*?, array<char \*, \d+>).*',
                  r'elementwise_variant<\1>', name)


def find_pass(names, durations, steps):
    """Candidate periods L: for about (steps - 2) * L consecutive starts i, the L launches at i match the L
    launches at i + L up to max(3, L // 200) names (schedulers launch a step-dependent kernel between passes).
    A whole generation cannot qualify (it would need two more identical generations). For each L the last
    qualifying window (whose successor also matched) is a candidate; returns (start, L) of the candidate with
    the largest summed kernel duration, the denoising transformer pass."""
    import numpy as np
    assert steps >= 3, 'needs at least three passes per generation'
    ids = {n: i for i, n in enumerate(dict.fromkeys(names))}
    a = np.array([ids[n] for n in names])
    t = np.concatenate(([0], np.cumsum(durations)))
    best = None
    for period in range(32, (len(a) - 1) // 2 + 1):
        miss = np.concatenate(([0], np.cumsum(a[:-period] != a[period:])))
        window = miss[period:] - miss[:-period]  # mismatches of window i vs window i + L
        ok = np.concatenate(([0], (window <= max(3, period // 200)).astype(np.int8), [0]))
        edges = np.flatnonzero(np.diff(ok))
        runs = [(s, e) for s, e in zip(edges[::2], edges[1::2]) if e - s >= (steps - 2) * period - period // 10]
        if runs:
            start = int(runs[-1][1] - 1)
            if start + period <= len(a):
                time = t[start + period] - t[start]
                if best is None or time > best[0]:
                    best = (time, start, period)
    if best is None:
        raise ValueError('No repeated denoising period found in the kernel sequence')
    return best[1], best[2]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--plan', type=Path, required=True)
    ap.add_argument('--out-root', type=Path, default=REPO_ROOT / 'results' / 'runs')
    ap.add_argument('--resume', type=Path, help='existing counters run directory; reuse its ncu CSVs')
    args = ap.parse_args()
    plan = json.loads(args.plan.read_text())
    device = plan.get('device', 'a100')
    basis = DEVICES[device][0]
    tensor = [t for t in counters.TENSOR if t.split('_src_')[1].split('_')[0] in plan.get('tensor_dtypes', ['bf16', 'fp16', 'tf32'])]
    metrics = counters.metric_list(basis, tensor)
    template = REPO_ROOT / 'results' / 'runs' / plan['template_run']
    assert json.loads((template / 'status.json').read_text())['status'] == 'complete', 'template run must be complete'
    cfg = json.loads((template / 'config.json').read_text())
    cfg['protocol'] = {**cfg['protocol'], 'first_runs': 1, 'warmup_runs': 0, 'measured_runs': 1}
    cfg['counter_capture'] = {'template_run': plan['template_run'], 'changed': ['protocol -> 1 + 0 + 1']}
    if 'runs' in cfg.get('harness_arguments', {}):  # the Jetson sd.cpp harness takes the generation count separately
        cfg['harness_arguments']['runs'] = 2
        cfg['counter_capture']['changed'].append('harness_arguments.runs -> 2')
    previous = None
    if args.resume:
        out = args.resume
        assert json.loads((out / 'counter-plan.json').read_text())['template_run'] == plan['template_run']
        previous = json.loads((out / 'status.json').read_text())
    else:
        if device == 'a100' and not os.environ.get('SLURM_JOB_ID'):
            raise SystemExit('Run inside a Slurm GPU allocation')
        if device == 'jetson' and os.geteuid() != 0:
            raise SystemExit('Run as root (Nsight Compute needs it on the Jetson)')
        out = create_run_directory(cfg, args.out_root, 'counters')
        (out / 'profile').mkdir()
        write_json(out / 'counter-plan.json', plan)
    profile = out / 'profile'
    steps = int(cfg['workload'].get('num_inference_steps') or cfg['workload']['steps'])
    with run_status(out) as status:
        if previous:  # keep the capture's start, jobs and earlier outcomes; this invocation only re-analyses or adds a pass
            status['started_utc'] = previous['started_utc']
            status['analysis_resumed_utc'] = dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
            status['slurm_jobs'] = previous.get('slurm_jobs', [])
            status['history'] = previous.get('history', []) + [
                {k: previous[k] for k in ('status', 'finished_utc', 'error', 'sequence') if k in previous}]
        if os.environ.get('SLURM_JOB_ID'):
            status.setdefault('slurm_jobs', []).append({'job': os.environ['SLURM_JOB_ID'], 'node': os.uname().nodename})
        else:
            status.setdefault('hosts', []).append({'host': os.uname().nodename, 'utc': dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')})
        cfg_path = out / 'config.json'
        if plan.get('warm_run') and not (profile / 'window.csv').exists():
            run(runner_command(plan, cfg_path, out / 'warm', template), profile / 'warm.log', plan['timeout_s'])
        seq_csv = profile / 'sequence.csv.gz'
        if not seq_csv.exists():
            run(ncu(plan, [counters.DURATION], profile / 'sequence', runner_command(plan, cfg_path, out / 'seq', template)),
                profile / 'sequence.log', plan['timeout_s'])
            counters.export_report(plan, profile / 'sequence', profile / 'sequence.csv')
            with open(profile / 'sequence.csv', 'rb') as raw, gzip.open(seq_csv, 'wb') as packed:
                packed.write(raw.read())  # every launch's name and duration; about 50x smaller compressed
            (profile / 'sequence.csv').unlink()
        seq = sequence(seq_csv)
        pids = {p for p, _, _ in seq}
        assert len(pids) == 1, f'kernels launched by several processes {pids}; launch indices are ambiguous'
        names = [n for _, n, _ in seq]
        start, period = find_pass([plain(n) for n in names], [d for _, _, d in seq], steps)
        margin = period // 4
        skip, count = max(0, start - margin), period + 2 * margin
        status['sequence'] = {'launches': len(names), 'pass_start': start, 'period': period, 'window': [skip, count]}
        if not (profile / 'window.csv').exists():
            groups = ([[counters.DURATION, m] for m in metrics if m != counters.DURATION] if DEVICES[device][2]
                      else [metrics])
            parts = []
            for i, group in enumerate(groups):
                name = f'window-{i}' if len(groups) > 1 else 'window'
                if not (profile / f'{name}.csv').exists():  # a resumed capture keeps finished counter runs
                    run(ncu(plan, group, profile / name, runner_command(plan, cfg_path, out / f'win-{i}', template),
                            (skip, count)), profile / f'{name}.log', plan['timeout_s'])
                    counters.export_report(plan, profile / name, profile / f'{name}.csv')
                parts.append(profile / f'{name}.csv')
            if len(parts) > 1:
                merge_windows(parts, profile / 'window.csv')
        rows = counters.read_counters(profile / 'window.csv', basis, tensor)
        got = [plain(r['kernel']) for r in rows]
        ref = [plain(n) for n in names]
        shifts = [d for d in range(-margin, margin + 1)
                  if 0 <= skip + d and ref[skip + d:skip + d + len(got)] == got]
        if len(shifts) != 1:
            raise ValueError(f'Counter-pass launches match the sequence pass at {len(shifts)} offsets '
                             f'within +-{margin}; launch order is not reproducible')
        first = start - skip - shifts[0]  # index of the pass inside the counter window
        if not 0 <= first <= len(rows) - period:
            raise ValueError(f'Offset {shifts[0]} puts the pass outside the counter window')
        status['window_offset'] = shifts[0]
        kernels = rows[first:first + period]
        flops = plan['analytic_flops_per_pass']
        hbm = sum(k['hbm_bytes'] for k in kernels)
        counted = sum(k['counted_ops'] for k in kernels)
        replay = sum(k['duration_s'] for k in kernels)
        write_json(profile / 'pass-kernels.json', [
            {'kernel': k['kernel'], 'duration_s': k['duration_s'], 'hbm_bytes': k['hbm_bytes'],
             'tensor_ops': k['tensor_ops'], 'fp32_add_mul_fma_ops': k['fp32_add_mul_fma_ops']} for k in kernels])
        kinds = '/'.join(t.split('_src_')[1].split('_')[0].upper() for t in tensor)
        write_json(profile / 'counter-summary.json', {
            'status': 'validated', 'template_run': plan['template_run'], 'metrics': metrics,
            'bytes_per_pass': hbm, 'bytes_basis': f'one steady denoising pass ({period} launches, last generation), ' + BASIS_TEXT[basis],
            'device': device, 'replay_mode': DEVICES[device][1],
            'pass_kernels': period, 'pass_replay_kernel_s': replay,
            'counted_ops_per_pass': counted, 'analytic_flops_per_pass': flops, 'counted_over_analytic': counted / flops,
            'analytic_ai': flops / hbm, 'counted_ai': counted / hbm if hbm else None,
            'sequence_launches': len(names), 'steps': steps, 'window_offset': shifts[0],
            'scope': f'{DEVICES[device][1].capitalize()} replay, no cache flush, clocks not locked; memcpy and CPU work excluded.'
                     + (' One ncu run per counter, merged per launch; durations from the first run.' if DEVICES[device][2] else ''),
            'limitations': (f'Counted {kinds} Tensor Core ops and FP32 add/mul/FMA only; ' if counters.BYTE_BASES[basis][4]
                            else f'Counted {kinds} Tensor Core ops only (FP32 instruction counters not collected); ')
                           + 'INT4/INT8 Tensor Core, scalar FP16/BF16 and SFU work are not counted.'})
        for report in profile.glob('*.ncu-rep'):
            if not plan.get('keep_reports'):
                report.unlink()  # large; the CSV exports keep every value used
        status['bytes_per_pass_gb'] = round(hbm / 1e9, 2)
    print('done', out)


if __name__ == '__main__':
    main()
