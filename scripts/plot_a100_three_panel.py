#!/usr/bin/env python3
"""Render an A100 engine comparison from existing results only; no upload or model execution."""
import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELS = [
    ('PyTorch eager', '#1f77b4', 'o'),
    ('PyTorch compiled', '#9467bd', 'D'),
    ('sd.cpp', '#ff7f0e', 's'),
    ('edge-dit', '#2ca02c', '^'),
    ('Nunchaku W4A4/r32', '#b58b80', '*'),
]


def read_inputs(path):
    roof = json.loads(path.read_text())
    assert len(roof['engine_points']) == len(LABELS), 'Expected the five selected engines'
    expected_ids = ['a100-flux-klein-torch28-001', 'a100-flux-klein-compile-001',
                    'a100-sdcpp-flux-klein-001', 'a100-edgedit-flux-klein-001',
                    'a100-nunchaku-flux-klein-001']
    assert [p['source_run'].split('__')[0] for p in roof['engine_points']] == expected_ids
    records, hashes = [], {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()}
    for point, (label, color, marker) in zip(roof['engine_points'], LABELS):
        run = ROOT/'results'/'runs'/point['source_run']
        for name in ['summary.json', 'runs.csv', 'config.json', 'status.json']:
            p = run/name
            hashes[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
        summary = json.loads((run/'summary.json').read_text())
        config = json.loads((run/'config.json').read_text())
        assert json.loads((run/'status.json').read_text())['status'] == 'complete'
        w = config['workload']
        assert w['height'] == w['width'] == 1024 and w.get('steps', w.get('num_inference_steps')) == 4
        assert w.get('guidance_scale', w.get('cfg_scale', w.get('guidance'))) == 1
        rows = [r for r in csv.DictReader((run/'runs.csv').open()) if r['phase'] == 'measured']
        assert len(rows) == 10
        for field in ['wall_ms', 'denoise_ms', 'vae_decode_ms']:
            values = [float(r[field]) for r in rows]
            saved = summary['measured'][field]
            for key, actual in [('median', statistics.median(values)), ('min', min(values)), ('max', max(values))]:
                assert math.isclose(saved[key], actual, rel_tol=1e-8), f'{label}: {field}.{key}'
        rows.sort(key=lambda r: float(r['wall_ms']))
        middle = rows[4:6]
        wall = statistics.mean(float(r['wall_ms']) for r in middle)/1000
        denoise = statistics.mean(float(r['denoise_ms']) for r in middle)/1000
        vae = statistics.mean(float(r['vae_decode_ms']) for r in middle)/1000
        remaining = wall - denoise - vae
        assert remaining >= 0
        measured_memory = summary['measured'].get('device_used_peak_gib')
        capture_scope = measured_memory is None
        if capture_scope:
            memory = summary['capture_device_used_peak_gib']
            memory_min = memory_max = memory
            memory_scope = 'whole child capture including loading, first run and warmups; no generation alignment'
        else:
            values = [float(r['device_used_peak_gib']) for r in rows]
            memory = statistics.median(values)
            assert math.isclose(memory, measured_memory['median'], rel_tol=1e-9)
            memory_min, memory_max = min(values), max(values)
            memory_scope = 'median sampled device-wide peak of ten measured generations'
        assert memory > 0
        assert math.isclose(summary['measured']['denoise_ms']['median']/4, point['ms_per_pass'], rel_tol=1e-8)
        records.append({**point, 'display_label': label, 'color': color, 'marker': marker,
                        'wall_s': wall, 'wall_min_s': min(float(r['wall_ms']) for r in rows)/1000,
                        'wall_max_s': max(float(r['wall_ms']) for r in rows)/1000,
                        'denoise_stack_s': denoise, 'vae_stack_s': vae, 'remaining_stack_s': remaining,
                        'device_peak_gib': memory, 'device_peak_min_gib': memory_min,
                        'device_peak_max_gib': memory_max, 'memory_scope': memory_scope,
                        'capture_memory': capture_scope, 'first_wall_s': summary['first_run']['wall_ms']/1000,
                        'quantized_checkpoint': 'Nunchaku' in label})
    return roof, records, hashes


def render(roof, records, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    rc = {'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'], 'font.size': 10,
          'axes.labelsize': 12, 'axes.titlesize': 12, 'axes.linewidth': .8,
          'pdf.fonttype': 42, 'svg.fonttype': 'none'}
    with plt.rc_context(rc):
        fig = plt.figure(figsize=(16.8, 6.8))
        gs = fig.add_gridspec(1, 3, width_ratios=[1.12, 1, 1], left=.06, right=.985,
                              bottom=.30, top=.84, wspace=.38)
        a, b, c = [fig.add_subplot(gs[0, i]) for i in range(3)]
        peak, bw = roof['datasheet']['bf16_tensor'], roof['datasheet']['bw_gbs']
        ridge = peak/(bw/1000)
        x = np.logspace(1, 4, 300)
        a.plot(x, np.minimum(peak, x*bw/1000), color='#1f77b4', lw=2, zorder=2)
        a.axhline(peak, color='#ff3333', ls='-.', lw=1)
        a.axvline(ridge, color='#2ca02c', ls='--', lw=1)
        for r in records:
            a.scatter(r['measured_ai'], r['achieved_tflops'], s=135 if r['marker']=='*' else 90,
                      marker=r['marker'], facecolors='white' if r['quantized_checkpoint'] else r['color'],
                      edgecolors=r['color'] if r['quantized_checkpoint'] else '#404040', linewidths=1.2, zorder=5)
        for r, xytext in [(records[0], (330, 150)), (records[3], (330, 102))]:
            a.annotate(r['display_label'].replace('PyTorch ', ''),
                       xy=(r['measured_ai'], r['achieved_tflops']), xytext=xytext,
                       fontsize=8.5, color=r['color'], ha='left', va='center',
                       arrowprops={'arrowstyle': '-', 'color': r['color'], 'lw': .65})
        a.set_xscale('log'); a.set_yscale('log'); a.set_xlim(10, 1e4); a.set_ylim(10, 650)
        a.set_xlabel('Arithmetic intensity (reference FLOP / HBM byte)')
        a.set_ylabel('Effective performance (reference TFLOP/s)')
        a.grid(which='major', color='#cccccc', lw=.55, alpha=.7)
        a.grid(which='minor', color='#dddddd', lw=.4, ls=':', alpha=.8)
        a.set_axisbelow(True)
        a.legend(handles=[Line2D([], [], color='#1f77b4', lw=2, label='A100 BF16 roofline'),
                          Line2D([], [], color='#ff3333', ls='-.', lw=1, label=f'Peak performance ({peak:g} TFLOP/s)'),
                          Line2D([], [], color='#2ca02c', ls='--', lw=1, label=f'Ridge point ({ridge:.1f} FLOP/B)')],
                 loc='lower right', fontsize=8.5, framealpha=.95, edgecolor='#999999')
        a.text(.03, .95, f'HBM bandwidth: {bw:,g} GB/s', transform=a.transAxes, fontsize=9, va='top')
        positions = np.arange(len(records))
        for i, r in enumerate(records):
            left = 0
            for key, alpha, hatch in [('denoise_stack_s', 1, None), ('vae_stack_s', .38, None),
                                      ('remaining_stack_s', .13, '///')]:
                b.barh(i, r[key], left=left, height=.55, color=r['color'], alpha=alpha,
                       edgecolor='#404040', linewidth=.55, hatch=hatch)
                left += r[key]
            b.errorbar(r['wall_s'], i, xerr=[[r['wall_s']-r['wall_min_s']], [r['wall_max_s']-r['wall_s']]],
                       fmt='none', color='#222222', capsize=3, lw=1, zorder=5)
            b.text(r['wall_s']+.065, i, f"{r['wall_s']:.3f}", va='center', fontsize=10)
            c.barh(i, r['device_peak_gib'], height=.55, color=r['color'], alpha=.85,
                   edgecolor='#404040', linewidth=.65, hatch='///' if r['capture_memory'] else None)
            c.text(r['device_peak_gib']+.35, i, f"{r['device_peak_gib']:.2f}"+(' †' if r['capture_memory'] else ''),
                   va='center', fontsize=10)
        names = [r['display_label'].replace('PyTorch ', 'Torch ').replace('Nunchaku W4A4/r32', 'Nunchaku*') for r in records]
        for ax in [b, c]:
            ax.set_yticks(positions, names); ax.invert_yaxis(); ax.set_ylim(4.7, -.7)
            ax.grid(axis='x', color='#dddddd', lw=.6); ax.set_axisbelow(True)
            ax.spines[['top', 'right']].set_visible(False)
        b.set_xlim(0, 2.85); b.set_xlabel('Warm generation latency (s)')
        b.legend(handles=[Patch(facecolor='#777777', label='Denoise'),
                          Patch(facecolor='#cccccc', label='VAE'),
                          Patch(facecolor='#eeeeee', hatch='///', label='Remaining')],
                 loc='lower right', bbox_to_anchor=(1, -.32), ncol=3, frameon=False, fontsize=8.5,
                 columnspacing=.8, handlelength=1.3)
        c.set_xlim(0, 27); c.set_xlabel('Sampled device-wide GPU peak (GiB)')
        for ax, title in [(a, '(a) Denoising roofline'), (b, '(b) End-to-end latency'), (c, '(c) Peak GPU memory')]:
            ax.set_title(title, pad=13, fontweight='bold')
        fig.suptitle('FLUX.2 [klein] 4B · A100 PCIe 40GB', fontsize=20, y=.98)
        fig.text(.5, .905, 'Distilled · 1024 × 1024 · 4 steps · batch 1 · guidance 1 · no reuse or offload',
                 ha='center', fontsize=11, color='#444444')
        handles = [Line2D([], [], linestyle='none', marker=r['marker'], markersize=9,
                          markerfacecolor='white' if r['quantized_checkpoint'] else r['color'],
                          markeredgecolor=r['color'] if r['quantized_checkpoint'] else '#404040',
                          label=r['display_label']+(' *' if r['quantized_checkpoint'] else '')) for r in records]
        fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .155), ncol=5,
                   frameon=False, fontsize=10, columnspacing=1.7, handletextpad=.5)
        notes = [
            'Roofline: reference-model work / measured HBM traffic and unprofiled denoising time; dense BF16 specification roof is contextual for mixed arithmetic.',
            'Latency: median wall time, min–max whiskers (10 measured images); stacks use the two middle wall-time observations. Loading, compilation and first image excluded.',
            'Memory: NVML at 20 ms; median per-generation peak, except † edge-dit: whole-capture peak including loading. Context and other GPU processes are included.',
            '* Nunchaku: W4A4/r32 replacement transformer; BF16 encoder/VAE. Checkpoint ancestry and matched quality unverified. sd.cpp/edge-dit use FP32 transformer activations.'
        ]
        for i, note in enumerate(notes):
            fig.text(.06, .119-i*.027, note, fontsize=8.4, color='#555555', va='top')
        for extension in ['png', 'pdf', 'svg']:
            fig.savefig(out/f'a100-three-panel.{extension}', dpi=220, facecolor='white')
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--roofline-data', type=Path, default=ROOT/'output/roofline-a100-measured/plot-data.json')
    parser.add_argument('--output', type=Path, default=ROOT/'output/a100-three-panel')
    args = parser.parse_args()
    roof, records, hashes = read_inputs(args.roofline_data)
    args.output.mkdir(parents=True, exist_ok=True)
    render(roof, records, args.output)
    data = {'workload': 'distilled Klein 4B, 1024², 4 steps, guidance 1, batch 1',
            'roof': roof['datasheet'], 'records': records, 'source_sha256': hashes,
            'latency_stack_rule': 'average corresponding stage times of the two middle wall-time observations; total is median wall time',
            'memory_caveat': 'edge-dit is whole-capture peak; the other four are median measured-generation peaks',
            'publication_status': 'local preview; not uploaded to W&B'}
    (args.output/'figure-data.json').write_text(json.dumps(data, indent=2)+'\n')
    print('Wrote', args.output/'a100-three-panel.png')


if __name__ == '__main__':
    main()
