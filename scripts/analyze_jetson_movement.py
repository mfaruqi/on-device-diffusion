#!/usr/bin/env python3
"""Data movement per generation and stage for a Jetson capture (Nsight Systems trace + movement sampler).

  python3 scripts/analyze_jetson_movement.py --run-dir results/runs/<id>__profile__<stamp> [--peak-mhz-bytes 32]

Inputs in the run directory: profile/trace.sqlite (nsys export) and movement/ (jetson_movement_sampler.py).
Stage windows come from NVTX ranges (sd.cpp harness: generate/text_encode/denoise_step_i/vae_decode; PyTorch via
nvtx_torch_runner.py: generation/... plus text_encoder_load/transformer_load) or, for edge-dit, from the engine's
[[phase]] wall-clock markers in engine.log, placed on the trace with the session's UTC start time.
Per window it reports:
  cuda_copy_gb     CUPTI memcpy bytes by direction (HtoD, DtoH, DtoD, ...) and memset bytes: copies the program asked
                   the CUDA driver to make. On the Orin both ends are the same LPDDR5, so each byte is read and written.
  storage_read_gb  sectors read from whole block devices (diskstats, 512 B sectors), zram excluded.
  zram_in/out_gb   pages swapped in/out of zram (4 KiB): compressed-RAM traffic.
  dram_est_gb      EMC busy fraction x EMC clock x 32 B/clock (128-bit LPDDR5, two transfers per clock),
                   integrated over the window: an ESTIMATE from memory-controller load, not a byte counter.
  kernel_busy_s    union of CUDA kernel intervals.
Sampler rows are interpolated linearly at window edges; windows shorter than ~3 sampler intervals are imprecise.
Writes profile/movement-summary.json and prints a table.
"""
import argparse
import bisect
import csv
import json
from pathlib import Path
import re
import sqlite3

COPY_KIND = {1: 'HtoD', 2: 'DtoH', 8: 'DtoD', 9: 'HtoH', 10: 'PtoP'}
STAGES = ('text_encode', 'denoise', 'vae_decode')


def nvtx_windows(db, t0_utc):
    rows = db.execute("select coalesce(n.text, s.value), n.start, n.end from NVTX_EVENTS n "
                      "left join StringIds s on s.id = n.textId where n.end is not null order by n.start").fetchall()
    gens = [r for r in rows if r[0] in ('generate', 'generation')]
    out = []
    for i, (_, a, b) in enumerate(gens):
        w = {'index': i, 'generation': (a, b), 'stages': {}}
        for name, s, e in rows:
            if a <= s and e <= b and name not in ('generate', 'generation') and not name.endswith('_load'):
                key = 'denoise' if name.startswith('denoise_step_') else name
                lo, hi = w['stages'].get(key, (s, e))
                w['stages'][key] = (min(lo, s), max(hi, e))
        out.append(w)
    loads = [(n, s, e) for n, s, e in rows if n.endswith('_load') or n == 'load']
    return out, loads


def edgedit_windows(log, t0_utc):
    marks = re.findall(r'\[\[phase\]\] stage=(\w+) event=(\w+) t=([0-9.]+)', log)
    rel = lambda t: int(float(t) * 1e9) - t0_utc
    out = []
    for i in range(0, len(marks), 6):
        m = marks[i:i + 6]
        names = {'encode': 'text_encode', 'denoise': 'denoise', 'decode': 'vae_decode'}
        stages = {names[m[j][0]]: (rel(m[j][2]), rel(m[j + 1][2])) for j in (0, 2, 4)}
        out.append({'index': i // 6, 'generation': (stages['text_encode'][0], stages['vae_decode'][1]),
                    'stages': stages})
    return out, []


class Sampler:
    def __init__(self, movement, t0_utc):
        rows = list(csv.DictReader((movement / 'samples.csv').open()))
        self.t = [int(r['t_wall_ns']) - t0_utc for r in rows]
        disks = [k[:-len('_sectors_read')] for k in rows[0] if k.endswith('_sectors_read') and not k.startswith('zram')]
        self.series = {
            'storage_read_gb': [sum(int(r[f'{d}_sectors_read']) for d in disks) * 512 / 1e9 for r in rows],
            'zram_in_gb': [int(r['pswpin']) * 4096 / 1e9 for r in rows],
            'zram_out_gb': [int(r['pswpout']) * 4096 / 1e9 for r in rows],
            'major_faults': [int(r['pgmajfault']) for r in rows]}
        self.emc = []  # (t_rel_ns, bytes_per_s)
        for line in (movement / 'tegrastats.log').read_text().splitlines():
            m = re.match(r'(\d+) .*EMC_FREQ (\d+)%@(\d+)', line)
            if m:
                self.emc.append((int(m[1]) - t0_utc, int(m[2]) / 100 * int(m[3]) * 1e6 * 32))

    def at(self, name, t):
        ts, ys = self.t, self.series[name]
        i = min(max(bisect.bisect_left(ts, t), 1), len(ts) - 1)
        a, b = ts[i - 1], ts[i]
        return ys[i - 1] + (ys[i] - ys[i - 1]) * min(max((t - a) / (b - a), 0), 1)

    def delta(self, a, b):
        out = {k: self.at(k, b) - self.at(k, a) for k in self.series}
        if self.emc:  # each tegrastats line reports the interval that ends at its timestamp
            total = 0.0
            for (t_prev, _), (t, bw) in zip(self.emc, self.emc[1:]):
                lo, hi = max(t_prev, a), min(t, b)
                if hi > lo:
                    total += bw * (hi - lo) / 1e9
            out['dram_est_gb'] = total / 1e9
        return out


def cuda_window(db, a, b):
    copies = {}
    for kind, nbytes in db.execute('select copyKind, sum(bytes) from CUPTI_ACTIVITY_KIND_MEMCPY '
                                   'where start >= ? and start < ? group by 1', (a, b)):
        copies[COPY_KIND.get(kind, str(kind))] = nbytes / 1e9
    memset = db.execute('select coalesce(sum(bytes), 0) from CUPTI_ACTIVITY_KIND_MEMSET where start >= ? and start < ?',
                        (a, b)).fetchone()[0]
    busy, end = 0, a
    for s, e in db.execute('select start, end from CUPTI_ACTIVITY_KIND_KERNEL where end > ? and start < ? order by start',
                           (a, b)):
        s, e = max(s, end, a), min(e, b)
        if e > s:
            busy += e - s
            end = e
    return {'cuda_copy_gb': copies, 'memset_gb': memset / 1e9, 'kernel_busy_s': busy / 1e9}


def describe(db, sampler, a, b):
    return {'seconds': (b - a) / 1e9, **cuda_window(db, a, b), **(sampler.delta(a, b) if sampler else {})}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run-dir', type=Path, required=True)
    args = ap.parse_args()
    run = args.run_dir
    db = sqlite3.connect(run / 'profile' / 'trace.sqlite')
    t0_utc = db.execute('select utcEpochNs from TARGET_INFO_SESSION_START_TIME').fetchone()[0]
    log = run / 'engine.log'
    source = 'engine.log [[phase]]' if log.exists() and '[[phase]]' in log.read_text() else 'NVTX'
    gens, loads = edgedit_windows(log.read_text(), t0_utc) if source != 'NVTX' else nvtx_windows(db, t0_utc)
    assert gens, 'no generation windows found'
    sampler = Sampler(run / 'movement', t0_utc) if (run / 'movement' / 'samples.csv').exists() else None
    out = {'run': run.name, 'source': source,
           'generations': [], 'loads': []}
    for g in gens:
        a, b = g['generation']
        row = {'index': g['index'], 'total': describe(db, sampler, a, b),
               'stages': {k: describe(db, sampler, *v) for k, v in g['stages'].items()}}
        out['generations'].append(row)
    for name, a, b in loads:
        out['loads'].append({'name': name, **describe(db, sampler, a, b)})
    end = db.execute('select max(end) from CUPTI_ACTIVITY_KIND_KERNEL').fetchone()[0]
    out['whole_capture'] = describe(db, sampler, 0, end)
    (run / 'profile' / 'movement-summary.json').write_text(json.dumps(out, indent=2))
    fmt = lambda d: (f"{d['seconds']:7.2f}s  HtoD {d['cuda_copy_gb'].get('HtoD', 0):6.2f}  DtoD {d['cuda_copy_gb'].get('DtoD', 0):6.2f}  "
                     f"disk {d.get('storage_read_gb', float('nan')):6.2f}  zram-in {d.get('zram_in_gb', float('nan')):5.2f}  "
                     f"DRAM~ {d.get('dram_est_gb', float('nan')):7.1f}  busy {d['kernel_busy_s']:6.2f}s")
    for g in out['generations'][-1:]:
        print(f"generation {g['index']}: {fmt(g['total'])}")
        for k, v in g['stages'].items():
            print(f"  {k:18s} {fmt(v)}")
    print(f"whole capture: {fmt(out['whole_capture'])}")


if __name__ == '__main__':
    main()
