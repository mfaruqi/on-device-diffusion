#!/usr/bin/env python3
"""Plot existing no-reuse engine results relative to workload-specific eager PyTorch."""
import argparse
import csv
import hashlib
import io
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENGINES = ['PyTorch eager', 'PyTorch compiled', 'sd.cpp', 'edge-dit', 'Nunchaku']
COLORS = ['#5798d3', '#7655aa', '#ed8737', '#70a844', '#ad8174']
# Columns follow ENGINES; absent combinations are deliberately not inferred.
RUNS = {
    'a100': [
        ['a100-flux-klein-torch28-001__baseline__20261015-151548',
         'a100-flux-klein-compile-001__baseline__20261015-151740',
         'a100-sdcpp-flux-klein-001__baseline__20261005-220435',
         'a100-edgedit-flux-klein-001__baseline__20261005-233237',
         'a100-nunchaku-flux-klein-001__baseline__20261011-124123'],
        ['a100-flux-klein-base-torch28-001__baseline__20261015-155338',
         'a100-flux-klein-base-torch28-004__baseline__20261015-161253',
         'a100-sdcpp-flux-klein-base-001__baseline__20261005-214624',
         'a100-edgedit-flux-klein-base-001__baseline__20261015-165052', None]],
    'jetson': [
        ['jetson-torch-flux-klein-001__baseline__20261014-171410', None,
         'jetson-flux-klein-012__baseline__20261102-165643',
         'jetson-edgedit-flux-klein-006__baseline__20261014-170149',
         'jetson-nunchaku-flux-klein-001__baseline__20261102-120036'],
        ['jetson-torch-flux-klein-base-001__baseline__20261016-050136', None,
         'jetson-flux-klein-base-001__baseline__20261005-005357',
         'jetson-edgedit-flux-klein-base-001__baseline__20261015-193452', None]],
}


def collect():
    records, hashes = [], {}
    def read(path):
        hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path.read_text()
    for device, groups in RUNS.items():
        for group, names in enumerate(groups):
            baseline = None
            for engine, name in zip(ENGINES, names):
                record = dict(device=device, workload=['Klein Distilled', 'Klein Base'][group],
                              engine=engine, source_run=name)
                records.append(record)
                if name is None:
                    record['missing_reason'] = ('No completed no-conditioning-reuse compiled run'
                                                if engine == 'PyTorch compiled' else 'No completed Base run')
                    continue
                run = ROOT/'results/runs'/name
                config = json.loads(read(run/'config.json'))
                summary = json.loads(read(run/'summary.json'))
                assert json.loads(read(run/'status.json'))['status'] == 'complete', name
                w, opt = config['workload'], config['optimizations']
                assert w['width'] == w['height'] == (1024 if device == 'a100' else 512), name
                assert w.get('steps', w.get('num_inference_steps')) == (4 if group == 0 else 50), name
                assert w.get('guidance_scale', w.get('cfg_scale', w.get('guidance'))) == (1 if group == 0 else 4), name
                assert not opt.get('conditioning_reuse') and not opt.get('conditioning_cache_size'), name
                assert not config.get('engine_settings', {}).get('conditioning_cache_size'), name
                assert not opt.get('step_cache'), name
                rows = [r for r in csv.DictReader(io.StringIO(read(run/'runs.csv'))) if r['phase'] == 'measured']
                assert len(rows) == 10, name
                record.update(optimizations=opt, precision=config.get('precision'), measured_runs=len(rows))
                for field in ['denoise_ms', 'wall_ms']:
                    values = [float(r[field]) for r in rows]
                    for stat, value in [('median', statistics.median(values)), ('min', min(values)), ('max', max(values))]:
                        assert math.isclose(value, summary['measured'][field][stat], rel_tol=1e-8), (name, field, stat)
                    record[field] = {k: summary['measured'][field][k] for k in ['median', 'min', 'max']}
                if engine == 'PyTorch eager':
                    baseline = record
                assert baseline is not None
                record['speedups'] = {field: baseline[field]['median']/record[field]['median']
                                      for field in ['denoise_ms', 'wall_ms']}
    return records, hashes


def render(records, device, field, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter
    title = 'A100 PCIe 40 GB · 1024²' if device == 'a100' else 'Jetson Orin Nano 8 GB · 512²'
    metric = 'Denoising-stage' if field == 'denoise_ms' else 'End-to-end generation'
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 12, 'axes.labelweight': 'bold',
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'}):
        fig, ax = plt.subplots(figsize=(10.6, 6.5))
        fig.subplots_adjust(left=.10, right=.98, bottom=.32, top=.75)
        handles = [Patch(facecolor=c, label=e) for e, c in zip(ENGINES, COLORS)]
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.54, .88),
                   ncol=5, frameon=False, handlelength=1, handletextpad=.4, columnspacing=1.2, fontsize=11)
        fig.suptitle(title, y=.98, fontsize=19, fontweight='bold')
        fig.text(.54, .91, metric+' speedup relative to PyTorch eager', ha='center', fontsize=13)
        width = .14
        for group, workload in enumerate(['Klein Distilled', 'Klein Base']):
            for i, (engine, color) in enumerate(zip(ENGINES, COLORS)):
                r = next(r for r in records if r['device'] == device and r['workload'] == workload and r['engine'] == engine)
                x = group + (i-2)*width
                if r['source_run'] is None:
                    ax.text(x, .125, 'N/M', rotation=90, ha='center', va='bottom', color='#777777', fontsize=10)
                    continue
                speedup = r['speedups'][field]
                hatch = '///' if device == 'a100' and group == 1 and engine == 'edge-dit' else None
                ax.bar(x, speedup-.1, bottom=.1, width=width*.88, color=color,
                       edgecolor='white' if hatch is None else '#3e6a28', linewidth=.6, hatch=hatch, zorder=3)
                ax.text(x, speedup*1.08, f'{speedup:.2f}×', ha='center', va='bottom', fontsize=11, fontweight='bold')
        ax.set_yscale('log'); ax.set_ylim(.1, 10); ax.set_xlim(-.5, 1.5)
        ticks = [.1, .2, .5, 1, 2, 5, 10]
        ax.yaxis.set_major_locator(FixedLocator(ticks))
        ax.yaxis.set_major_formatter(FixedFormatter(['0.1', '0.2', '0.5', '1', '2', '5', '10']))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.grid(axis='y', which='major', color='#dddddd', linewidth=1, zorder=0)
        ax.axhline(1, color='#555555', ls='--', lw=1.1, zorder=4)
        ax.text(1.47, 1.06, 'Eager = 1×', ha='right', va='bottom', fontsize=10, color='#444444')
        ax.set_ylabel('Speedup (×)', fontsize=17)
        ax.set_xticks([0, 1]); ax.set_xticklabels(['Klein Distilled\n4 steps · CFG 1', 'Klein Base\n50 steps · CFG 4'], fontsize=15, fontweight='bold')
        ax.tick_params(axis='x', length=0, pad=13)
        ax.tick_params(axis='y', which='both', length=0, pad=8)
        for spine in ax.spines.values():
            spine.set_visible(False)
        notes = ['Median of 10 measured images; no conditioning reuse or step cache. Above 1× is faster.',
                 'N/M: not measured. Distilled and Base are normalized separately to their own eager run.']
        if device == 'a100':
            notes.append('Eager/compiled: BF16. Nunchaku: W4A4/r32. Hatched edge-dit Base: Q4 checkpoint.')
        else:
            notes.append('Eager: GGUF Q4_0 + NF4 encoder. C++: mixed Q4. Nunchaku: W4A4/r32.')
        notes.append('Different precision/checkpoints and residency policies; quality equivalence is not established.')
        if field == 'denoise_ms':
            notes.append('Stage boundaries differ'+('; disk-backed sd.cpp Base includes reload' if device == 'jetson' else '')
                         +'. This is not a pure kernel comparison.')
        else:
            notes.append('Includes prompt encoding, reloads and decoding; excludes initial model loading and warm-ups.')
        fig.text(.10, .025, '\n'.join(notes), fontsize=9.3, linespacing=1.5, va='bottom', color='#444444')
        stem = f'{device}-'+('denoising' if field == 'denoise_ms' else 'end-to-end')+'-speedup'
        for ext in ['png', 'pdf', 'svg']:
            fig.savefig(out/f'{stem}.{ext}', dpi=190, facecolor='white')
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'output/engine-speedups')
    args = parser.parse_args()
    records, hashes = collect()
    args.out.mkdir(parents=True, exist_ok=True)
    for device in RUNS:
        for field in ['denoise_ms', 'wall_ms']:
            render(records, device, field, args.out)
    data = {'normalization': 'eager median / engine median, independently per device and workload',
            'conditioning_reuse': False, 'step_cache': False, 'records': records, 'source_sha256': hashes}
    (args.out/'figure-data.json').write_text(json.dumps(data, indent=2)+'\n')
    with (args.out/'speedups.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['device', 'workload', 'engine', 'denoising_s', 'end_to_end_s', 'denoising_speedup', 'end_to_end_speedup', 'source_run'])
        for r in records:
            writer.writerow([r['device'], r['workload'], r['engine'],
                             r.get('denoise_ms', {}).get('median', '')/1000 if r['source_run'] else '',
                             r.get('wall_ms', {}).get('median', '')/1000 if r['source_run'] else '',
                             r.get('speedups', {}).get('denoise_ms', ''), r.get('speedups', {}).get('wall_ms', ''), r['source_run']])
    print(args.out)


if __name__ == '__main__':
    main()
