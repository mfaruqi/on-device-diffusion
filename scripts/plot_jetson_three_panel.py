#!/usr/bin/env python3
"""Jetson roofline, no-conditioning-reuse latency and whole-capture RAM; existing results only, no upload.

Roofline x uses Nsight Compute system-memory traffic of one steady denoising pass (counter captures below). Each
capture ran the same transformer checkpoint and engine build as the plotted benchmark run; the differences are
outside the steady pass (parameter storage, VAE tiling, conditioning reuse) and are asserted here. y is reference
FLOPs / mean of the step 1-3 medians of the plotted run (denoise median / 4 when no per-step times exist).
edge-dit has no capture (Nsight Compute does not fit beside its resident weights): x is a lower bound from the
datasheet bandwidth. Nunchaku has no Jetson capture and is not placed on the roofline.
"""
import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

from plot_a100_three_panel import LABELS
from plot_roofline import transformer_cost, HARDWARE_SPECS

ROOT = Path(__file__).resolve().parents[1]
RUNS = [
    'jetson-torch-flux-klein-001__baseline__20261014-171410',
    'jetson-flux-klein-012__baseline__20261102-165643',
    'jetson-edgedit-flux-klein-006__baseline__20261014-170149',
    'jetson-nunchaku-flux-klein-001__baseline__20261102-120036',
]
# label -> (counter capture, benchmark run whose config the capture repeated)
COUNTERS = {
    'PyTorch eager': ('jetson-torch-flux-klein-004__counters__20261031-002432', 'jetson-torch-flux-klein-004__baseline__20261015-141028'),
    'sd.cpp': ('jetson-flux-klein-003__counters__20261030-211513', 'jetson-flux-klein-003__repeat__20261014-163411'),
    'PyTorch compiled': ('jetson-torch-flux-klein-007__counters__20261101-154131', 'jetson-torch-flux-klein-007__baseline__20261015-145234'),
}
# settings that may differ between a capture's template run and the plotted run: none of them is in the steady pass
OUTSIDE_PASS = {'conditioning_reuse', 'parameter_storage', 'vae_tiling'}


def step_seconds(stage_text, denoise_median_s):
    """Mean of the medians of steps 1-3 (step 0 can include weight preparation); denoise/4 without per-step rows."""
    rows = [r for r in csv.DictReader(stage_text.splitlines()) if r['phase'] == 'measured' and r['stage'].startswith('denoise_step_')]
    if not rows:
        return denoise_median_s/4, 'denoising median / 4 (no per-step times recorded)'
    def ms(r):
        return float(r['gpu_ms'] or r['host_ms'])
    medians = [statistics.median(ms(r) for r in rows if r['stage'] == f'denoise_step_{i}') for i in (1, 2, 3)]
    return statistics.mean(medians)/1000, 'mean of the step 1-3 medians'


def flat(d, prefix=''):
    out = {}
    for k, v in d.items():
        out.update(flat(v, f'{prefix}{k}.') if isinstance(v, dict) else {prefix + k: v})
    return out


def counter_point(read, label, config, flops, step_s, roof):
    """Roofline coordinates from the label's counter capture, or the edge-dit bound, or None (not placed)."""
    if label == 'edge-dit':
        max_bytes = roof['bw_gbs']*1e9*step_s
        return {'roofline': 'lower bound', 'ai': flops/max_bytes, 'achieved_tflops': flops/step_s/1e12,
                'bytes_per_pass': None, 'counter_capture': None,
                'intensity_basis': f'lower bound: bytes <= datasheet {roof["bw_gbs"]:g} GB/s x step time (no capture)'}
    if label not in COUNTERS:
        return {'roofline': 'not placed', 'ai': None, 'achieved_tflops': flops/step_s/1e12, 'bytes_per_pass': None,
                'counter_capture': None, 'intensity_basis': 'no Jetson counter capture'}
    capture, template = COUNTERS[label]
    summary = json.loads(read(ROOT/'results'/'runs'/capture/'profile'/'counter-summary.json'))
    assert json.loads(read(ROOT/'results'/'runs'/capture/'status.json'))['status'] == 'complete'
    assert summary['status'] == 'validated' and summary['device'] == 'jetson' and summary['template_run'] == template
    assert math.isclose(summary['analytic_flops_per_pass'], flops, rel_tol=1e-9)
    plotted, captured = flat(config), flat(json.loads(read(ROOT/'results'/'runs'/template/'config.json')))
    differ = {k.split('.')[-1] for k in set(plotted) | set(captured)
              if k.startswith(('optimizations.', 'model.', 'quantized_transformer.', 'workload.'))
              and plotted.get(k) != captured.get(k)}
    assert differ <= OUTSIDE_PASS, f'{label}: capture config differs in the denoising pass: {differ - OUTSIDE_PASS}'
    return {'roofline': 'measured', 'ai': flops/summary['bytes_per_pass'], 'achieved_tflops': flops/step_s/1e12,
            'bytes_per_pass': summary['bytes_per_pass'], 'counter_capture': capture, 'capture_template_run': template,
            'capture_config_differences': sorted(differ), 'intensity_basis': summary['bytes_basis']}


def read_inputs():
    records, hashes = [], {}
    roof = HARDWARE_SPECS['jetson-orin-nano']
    def read(path):
        hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path.read_text()
    for index, (name, (label, color, marker)) in enumerate(zip(RUNS, [LABELS[i] for i in [0, 2, 3, 4]])):
        run = ROOT/'results'/'runs'/name
        summary = json.loads(read(run/'summary.json'))
        config = json.loads(read(run/'config.json'))
        assert json.loads(read(run/'status.json'))['status'] == 'complete'
        workload = config['workload']
        assert workload['height'] == workload['width'] == 512
        assert workload.get('steps', workload.get('num_inference_steps')) == 4
        assert config['optimizations'].get('step_cache') is None
        rows = [r for r in csv.DictReader(read(run/'runs.csv').splitlines()) if r['phase'] == 'measured']
        assert len(rows) == 10
        for field in ['wall_ms', 'denoise_ms', 'vae_decode_ms']:
            values = [float(r[field]) for r in rows]
            for key, actual in [('median', statistics.median(values)), ('min', min(values)), ('max', max(values))]:
                assert math.isclose(summary['measured'][field][key], actual, rel_tol=1e-8)
        rows.sort(key=lambda r: float(r['wall_ms']))
        wall, denoise, vae = [statistics.mean(float(r[field]) for r in rows[4:6])/1000
                              for field in ['wall_ms', 'denoise_ms', 'vae_decode_ms']]
        assert wall >= denoise+vae
        memory = summary.get('system_memory')
        if memory:
            peak, total = memory['ram_peak_gib'], memory['ram_total_gib']
            swap_min, swap_max = memory['swap_min_gib'], memory['swap_max_gib']
        else:
            samples = list(csv.DictReader(read(run/'tegrastats-samples.csv').splitlines()))
            peak = max(float(r['ram_gib']) for r in samples)
            total = max(float(r['ram_total_reported_mb']) for r in samples)/1024
            swap_min = min(float(r['swap_gib']) for r in samples)
            swap_max = max(float(r['swap_gib']) for r in samples)
        assert 0 < peak <= total
        load = json.loads(read(run/'load.json'))
        component = load.get('components', {}).get('transformer')
        weight = component['weights_gib'] if component else 2.15
        activation_bytes = 4 if label in ('sd.cpp', 'edge-dit') else 2
        flops, byte_count = transformer_cost(512, 512, weight, activation_bytes=activation_bytes)
        denoise_median = summary['measured']['denoise_ms']['median']/1000
        step_s, timing_basis = step_seconds(read(run/'stages.csv'), denoise_median)
        point = counter_point(read, label, config, flops, step_s, roof)
        reuse = False
        assert not config['optimizations'].get('conditioning_reuse', False)
        assert config['optimizations'].get('conditioning_cache_size', 0) == 0
        assert config.get('engine_settings', {}).get('conditioning_cache_size', 0) == 0
        records.append({'source_run': name, 'display_label': label, 'color': color, 'marker': marker,
                        **point, 'step_s': step_s, 'timing_basis': timing_basis, 'reference_flops_per_pass': flops,
                        'analytic_bytes_per_pass': byte_count, 'analytic_ai': flops/byte_count,
                        'activation_reference_bytes': activation_bytes,
                        'weight_gib': weight, 'weight_basis': 'recorded component' if component else 'explicit mixed-Q4 footprint estimate',
                        'wall_s': wall, 'wall_min_s': min(float(r['wall_ms']) for r in rows)/1000,
                        'wall_max_s': max(float(r['wall_ms']) for r in rows)/1000,
                        'denoise_stack_s': denoise, 'vae_stack_s': vae, 'remaining_stack_s': wall-denoise-vae,
                        'denoise_median_s': denoise_median, 'device_peak_gib': peak,
                        'memory_metric': 'whole-system RAM; whole capture including load/first/warmups',
                        'physical_ram_gib': total, 'swap_min_gib': swap_min, 'swap_max_gib': swap_max,
                        'capture_memory': False, 'first_wall_s': summary['first_run']['wall_ms']/1000,
                        'conditioning_reuse': reuse, 'vae_tiling': bool(config['optimizations'].get('vae_tiling', False)),
                        'quantized_checkpoint': label.startswith('Nunchaku'), 'optimizations': config['optimizations']})
    # Compiled PyTorch has no complete no-reuse run, so it appears only on the roofline: its steady pass is the
    # same with or without conditioning reuse (reuse skips the encoder and reloads, not denoising steps).
    label, color, marker = LABELS[1]
    capture, template = COUNTERS[label]
    run = ROOT/'results'/'runs'/template
    config, summary = json.loads(read(run/'config.json')), json.loads(read(run/'summary.json'))
    assert json.loads(read(run/'status.json'))['status'] == 'complete'
    assert config['optimizations'].get('compiler', {}).get('name') == 'torch_compile' and config['optimizations'].get('step_cache') is None
    flops = records[0]['reference_flops_per_pass']
    step_s, timing_basis = step_seconds(read(run/'stages.csv'), summary['measured']['denoise_ms']['median']/1000)
    roofline_only = [{'source_run': template, 'display_label': label, 'color': color, 'marker': marker,
                      **counter_point(read, label, config, flops, step_s, roof), 'step_s': step_s,
                      'timing_basis': timing_basis, 'reference_flops_per_pass': flops,
                      'conditioning_reuse': bool(config['optimizations'].get('conditioning_reuse'))}]
    for file in ['scripts/plot_roofline.py', 'scripts/plot_jetson_three_panel.py']:
        read(ROOT/file)
    return roof, records, roofline_only, hashes


def render(roof, records, roofline_only, out):
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
        peak, bw = roof['ceilings']['fp16_dense'], roof['bw_gbs']
        ridge = peak/(bw/1000)
        x = np.logspace(1, 4, 300)
        a.plot(x, np.minimum(peak, x*bw/1000), color='#1f77b4', lw=2, zorder=2)
        a.axhline(peak, color='#ff3333', ls='-.', lw=1)
        a.axvline(ridge, color='#2ca02c', ls='--', lw=1)
        for r in records + roofline_only:
            if r['roofline'] == 'not placed':
                continue
            measured = r['roofline'] == 'measured'
            a.scatter(r['ai'], r['achieved_tflops'], s=135 if r['marker']=='*' else 90, marker=r['marker'],
                      facecolors=r['color'] if measured else 'white', edgecolors='#333333' if measured else r['color'],
                      linewidths=1.0 if measured else 1.4, zorder=5)
            if not measured:
                a.annotate('', xy=(r['ai']*2.6, r['achieved_tflops']), xytext=(r['ai']*1.15, r['achieved_tflops']),
                           arrowprops=dict(arrowstyle='->', color=r['color'], lw=1.3), zorder=4)
                a.text(r['ai']*1.12, r['achieved_tflops']*1.07, f"≥ {r['ai']:.0f}", color=r['color'], fontsize=8.5, va='bottom')
        a.set_xscale('log'); a.set_yscale('log'); a.set_xlim(10, 1e4); a.set_ylim(.3, 35)
        a.set_xlabel('Arithmetic intensity (reference FLOP / measured byte)')
        a.set_ylabel('Effective performance (reference TFLOP/s)')
        a.grid(which='major', color='#cccccc', lw=.55, alpha=.7)
        a.grid(which='minor', color='#dddddd', lw=.4, ls=':', alpha=.8)
        a.set_axisbelow(True)
        a.legend(handles=[Line2D([], [], color='#1f77b4', lw=2, label='Orin Nano reference roofline'),
                          Line2D([], [], color='#ff3333', ls='-.', lw=1, label=f'Peak performance ({peak:g} TFLOP/s)'),
                          Line2D([], [], color='#2ca02c', ls='--', lw=1, label=f'Ridge point ({ridge:.1f} FLOP/B)')],
                 loc='lower left', fontsize=8.5, framealpha=.95, edgecolor='#999999')
        a.text(.03, .95, f'Memory bandwidth: {bw:,g} GB/s', transform=a.transAxes, fontsize=9, va='top')
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
            b.text(r['wall_max_s']+max(x['wall_s'] for x in records)*.015, i, f"{r['wall_s']:.2f}", va='center', fontsize=10)
            c.barh(i, r['device_peak_gib'], height=.55, color=r['color'], alpha=.85,
                   edgecolor='#404040', linewidth=.65, hatch='///' if r['capture_memory'] else None)
            c.text(r['device_peak_gib']+.10, i, f"{r['device_peak_gib']:.2f}"+(' †' if r['capture_memory'] else ''),
                   va='center', fontsize=10)
        names = [r['display_label'].replace('PyTorch ', 'Torch ').replace('Nunchaku W4A4/r32', 'Nunchaku*') for r in records]
        for ax in [b, c]:
            ax.set_yticks(positions, names); ax.invert_yaxis(); ax.set_ylim(len(records)-.3, -.7)
            ax.grid(axis='x', color='#dddddd', lw=.6); ax.set_axisbelow(True)
            ax.spines[['top', 'right']].set_visible(False)
        b.set_xlim(0, max(r['wall_max_s'] for r in records)*1.15); b.set_xlabel('Generation latency incl. reloads (s)')
        b.legend(handles=[Patch(facecolor='#777777', label='Denoise'),
                          Patch(facecolor='#cccccc', label='VAE'),
                          Patch(facecolor='#eeeeee', hatch='///', label='Remaining')],
                 loc='lower right', bbox_to_anchor=(1, -.32), ncol=3, frameon=False, fontsize=8.5,
                 columnspacing=.8, handlelength=1.3)
        c.set_xlim(0, 8.1); c.set_xlabel('Whole-system RAM peak (GiB)')
        c.axvline(7.44140625, color='#777777', ls='--', lw=1)
        c.text(7.44140625, -.78, '7.44 GiB physical', ha='right', fontsize=8.5, color='#555555')
        for ax, title in [(a, '(a) Denoising roofline'), (b, '(b) End-to-end latency'), (c, '(c) Peak shared memory')]:
            ax.set_title(title, pad=13, fontweight='bold')
        fig.suptitle('FLUX.2 [klein] 4B · Jetson Orin Nano 8GB · 25 W', fontsize=20, y=.98)
        fig.text(.5, .905, 'Distilled · 512 × 512 · 4 steps · batch 1 · guidance 1 · no conditioning reuse · prompt encoding performed every image',
                 ha='center', fontsize=11, color='#444444')
        def handle(r, suffix=''):
            measured = r['roofline'] == 'measured'
            return Line2D([], [], linestyle='none', marker=r['marker'], markersize=9,
                          markerfacecolor=r['color'] if measured else 'white',
                          markeredgecolor='#333333' if measured else r['color'],
                          label=r['display_label'] + suffix + (' *' if r.get('quantized_checkpoint') else ''))
        handles = [handle(r) for r in records] + [handle(r, ' (roofline only)') for r in roofline_only]
        fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .155), ncol=5,
                   frameon=False, fontsize=10, columnspacing=1.7, handletextpad=.5)
        notes = [
            'Roofline: x = reference FLOPs / Nsight Compute L2<->system-memory bytes of one steady pass (writes an upper bound); y = reference FLOPs / steady step time of the plotted run.',
            'Filled = measured traffic. edge-dit: no capture fits beside its resident weights; hollow lower bound. Nunchaku: no Jetson capture, not on the roofline. Roof = datasheet.',
            'Latency: median wall and min–max (n=10); stacks use corresponding middle observations. Per-image reloads INCLUDED; one-time startup/first image excluded.',
            'No conditioning reuse or step skipping. Edge-dit and sd.cpp keep components resident with tiled VAE; PyTorch/Nunchaku reload components.',
            'Memory: whole-capture system RAM, INCLUDING loading/first/warm-ups; OS/sampling differ. Fixed development prompt is recomputed each image; prompts were not varied.',
            '* Nunchaku W4A4/r32 replacement, quality/ancestry unverified. Residency/encoders differ. Compiled PyTorch only on the roofline (its run reuses conditioning; the steady pass is unaffected).'
        ]
        for i, note in enumerate(notes):
            fig.text(.06, .132-i*.022, note, fontsize=8.2, color='#555555', va='top')
        for extension in ['png', 'pdf', 'svg']:
            fig.savefig(out/f'jetson-three-panel.{extension}', dpi=220, facecolor='white')
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'output/jetson-three-panel')
    args = parser.parse_args()
    roof, records, roofline_only, hashes = read_inputs()
    args.output.mkdir(parents=True, exist_ok=True)
    render(roof, records, roofline_only, args.output)
    data = {'workload': 'distilled Klein 4B, 512², 4 steps, guidance 1; no conditioning reuse',
            'roof': roof, 'records': records, 'roofline_only_records': roofline_only, 'source_sha256': hashes,
            'intensity_method': 'reference FLOPs / Nsight Compute system-memory bytes of one steady denoising pass '
                                '(counter capture per record); edge-dit lower bound from datasheet bandwidth; '
                                'analytic_bytes_per_pass kept for comparison only',
            'latency_stack_rule': 'corresponding two middle wall-time observations; total is median wall time',
            'selection_caveat': 'no conditioning reuse; fixed development prompt recomputed; differing residency, VAE tiling and encoders; no matched-quality ranking',
            'publication_status': 'see wandb-upload.json'}
    (args.output/'figure-data.json').write_text(json.dumps(data, indent=2)+'\n')
    print('Wrote', args.output/'jetson-three-panel.png')


if __name__ == '__main__':
    main()
