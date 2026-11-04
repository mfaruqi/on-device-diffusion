#!/usr/bin/env python3
"""Jetson Orin Nano rooflines from measured Nsight Compute bytes and measured ceilings.

  python scripts/plot_roofline_jetson.py [--output DIR] [--upload]

Workload: FLUX.2 klein distilled 512², one denoising pass (one transformer call). Each engine is shown in its
fastest per-pass configuration with unchanged output, plus the PyTorch eager reference; step caches (EasyCache,
MagCache, TaylorSeer) skip whole passes and do not move a per-pass point, so they are not plotted.
Three figures (PNG + PDF) and a JSON of every plotted coordinate:
- jetson-engine-roofline-measured: y = analytic FLOPs per pass / benchmark step time (median of the steady
  steps 1-3, or denoising median / 4 when the engine records no per-step times). x = analytic FLOPs / measured
  bytes per pass (filled marker). An engine without a capture (edge-dit: Nsight Compute does not fit in memory
  alongside it) is drawn hollow at the lowest intensity its step time allows: bytes <= datasheet bandwidth x
  step time, so x >= FLOPs / (datasheet bandwidth x step time); the arrow points to where the true x lies.
- jetson-kernel-bandwidth-measured: one panel per captured engine, one dot per kernel of the pass:
  bytes moved vs achieved bandwidth (bytes / kernel time). Marker area is proportional to kernel time.
- jetson-traffic-breakdown: bytes per pass and kernel time per pass by kernel class, per captured engine.
Ceilings: datasheet dashed, measured (jetson-ceilings-001, measure_ceilings_cuda.py) solid.
Bytes: L2 read misses + L2 writes to system memory, 32 B sectors (writes are an upper bound); see
wiki/methods/baseline-metrics.md.
"""
import argparse
import base64
import hashlib
import json
import math
import statistics
import uuid
from pathlib import Path

from plot_roofline_measured import kernel_class, roof, save, style

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT/'results'/'runs'
CEILINGS = RUNS/'jetson-ceilings-001__ceilings__20261029-105711'
FLOPS_PER_PASS = 10172630040576.0  # plot_roofline.py transformer_cost(512, 512), as in the counter plans
# (label, engine, benchmark run, counter capture or None); fastest per-pass configuration per engine first
ENGINES = [
    ('torch.compile', 'PyTorch / Diffusers', 'jetson-torch-flux-klein-007__baseline__20261015-145234',
     'jetson-torch-flux-klein-007__counters__20261101-154131'),
    ('edge-dit', 'edge-dit.cpp', 'jetson-edgedit-flux-klein-006__baseline__20261014-170149', None),
    ('sd.cpp', 'stable-diffusion.cpp', 'jetson-flux-klein-003__repeat__20261014-163411',
     'jetson-flux-klein-003__counters__20261030-211513'),
    ('PyTorch eager', 'PyTorch / Diffusers', 'jetson-torch-flux-klein-004__baseline__20261015-141028',
     'jetson-torch-flux-klein-004__counters__20261031-002432'),
]
LEGEND_SIZE = 7  # uniform legend markers; plotted marker area encodes kernel time
STYLE = {'torch.compile': ('#009E73', 'P'), 'PyTorch eager': ('#009E73', 'o'), 'sd.cpp': ('#0072B2', 's'),
         'edge-dit': ('#E69F00', '^')}  # A100 figure colours: engine = hue, compiled = plus marker
CLASSES = {'GEMM': '#0072B2', 'attention': '#D55E00', 'norm': '#009E73', 'elementwise / reduction': '#999999'}
INK, MUTED = '#222222', '#666666'


def jetson_class(name):
    k = name.lower()
    if 'mul_mat' in k:  # ggml quantized (mul_mat_q) and vector GEMMs
        return 'GEMM'
    return kernel_class(name)


def ceilings():
    c = json.loads((CEILINGS/'summary.json').read_text())['ceilings']
    pick = lambda key: {'measured': c[key]['measured'], 'datasheet': c[key]['datasheet']}
    return {'bw_gbs': pick('bw_gbs'), 'bf16_tensor': pick('bf16_tensor'), 'fp16_tensor': pick('fp16_tensor'),
            'int8_tensor_tops': pick('int8_tensor_tops'), 'source': CEILINGS.name}


def step_seconds(run):
    """Median steady step time: steps 1-3 when recorded per step, else denoising median / 4."""
    m = json.loads((RUNS/run/'summary.json').read_text())['measured']
    steps = [m[f'denoise_step_{i}_ms']['median'] for i in (1, 2, 3) if m.get(f'denoise_step_{i}_ms')]
    if steps:
        return statistics.mean(steps)/1e3, 'mean of the step 1-3 medians'
    return m['denoise_ms']['median']/4/1e3, 'denoising median / 4 (no per-step times recorded)'


def engine_points(ceil):
    pts = []
    for label, engine, run, capture in ENGINES:
        s, timing = step_seconds(run)
        p = {'label': label, 'engine': engine, 'benchmark_run': run, 'step_s': s, 'timing_basis': timing,
             'flops_per_pass': FLOPS_PER_PASS, 'achieved_tflops': FLOPS_PER_PASS/s/1e12}
        if capture:
            c = json.loads((RUNS/capture/'profile'/'counter-summary.json').read_text())
            assert c['status'] == 'validated' and c['template_run'] == run, capture
            b = c['bytes_per_pass']
            p.update(capture=capture, bytes_per_pass=b, ai=FLOPS_PER_PASS/b, effective_gbs=b/s/1e9,
                     replay_kernel_s=c['pass_replay_kernel_s'], pass_kernels=c['pass_kernels'],
                     counted_over_analytic=c['counted_over_analytic'], bytes_basis=c['bytes_basis'])
        else:
            bound = ceil['bw_gbs']['datasheet']*1e9*s
            p.update(capture=None, max_bytes_per_pass=bound, ai_lower_bound=FLOPS_PER_PASS/bound,
                     bytes_basis='not measured (Nsight Compute exceeds memory); x is a lower bound')
        pts.append(p)
    return pts


def draw_roofs(ax, xlim, ceil):
    roofs = [('bf16_tensor', 'BF16 / FP16 Tensor Core', 'TFLOP/s'), ('int8_tensor_tops', 'INT8 Tensor Core', 'TOP/s')]
    bw = ceil['bw_gbs']
    for key, label, unit in roofs:
        d, m = ceil[key]['datasheet'], ceil[key]['measured']
        roof(ax, xlim, bw['datasheet']/1000, d, color='#aaaaaa', ls='--', lw=1)
        roof(ax, xlim, bw['measured']/1000, m, color=INK, lw=1.8)
        ax.text(xlim[1]/1.12, d*1.06, f'{label} {d:g} {unit} (datasheet)', ha='right', va='bottom', fontsize=9,
                color=MUTED)
        ax.text(xlim[1]/1.12, m/1.06, f'{label} {m:.1f} {unit} (measured)', ha='right', va='top', fontsize=9,
                color=INK)


def engine_figure(plt, ceil, pts, out):
    fig, ax = plt.subplots(figsize=(10, 7.2))
    xlim, ylim = (8, 2000), (1, 50)
    draw_roofs(ax, xlim, ceil)
    ridge = ceil['bf16_tensor']['measured']/ceil['bw_gbs']['measured']*1000
    ax.axvline(ridge, color=INK, lw=.6, ls=':', alpha=.6)
    ax.text(ridge*1.04, ylim[0]*1.15, f'ridge {ridge:.0f}', fontsize=9, color=MUTED, rotation=90, va='bottom')
    for p in pts:
        color, marker = STYLE[p['label']]
        y = p['achieved_tflops']
        if p['capture']:
            ax.scatter([p['ai']], [y], marker=marker, s=150, color=color, edgecolors='white', linewidths=1.2, zorder=5,
                       label=f"{p['label']}: {p['bytes_per_pass']/1e9:.1f} GB/pass measured, {p['step_s']:.2f} s/pass")
            right = p['label'] != 'torch.compile'  # left of the marker would cross the ridge or the bandwidth roof
            ax.annotate(p['label'], (p['ai'], y), xytext=(11 if right else -11, 0), textcoords='offset points',
                        ha='left' if right else 'right', va='center',
                        fontsize=10, color=INK)
        else:
            x = p['ai_lower_bound']
            ax.annotate('', xy=(x*2.2, y), xytext=(x*1.08, y),
                        arrowprops=dict(arrowstyle='-|>', color=color, lw=1.6, shrinkA=0, shrinkB=0), zorder=4)
            ax.scatter([x], [y], marker=marker, s=150, facecolors='white', edgecolors=color, linewidths=1.8, zorder=5,
                       label=f"{p['label']}: bytes not measurable, {p['step_s']:.2f} s/pass (x ≥ {x:.0f}, arrow)")
            ax.annotate(f"{p['label']} (x ≥ {x:.0f})", (x, y), xytext=(-10, 0), textcoords='offset points', ha='right',
                        va='center', fontsize=10, color=INK)
    style(ax, 'Arithmetic intensity (FLOP per LPDDR5 byte)', 'Effective performance (TFLOP/s)')
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_title('Jetson Orin Nano 8GB (25 W) · FLUX.2 klein distilled 512² · one denoising pass per engine',
                 fontweight='bold', fontsize=12.5)
    ax.legend(loc='upper left', bbox_to_anchor=(0, -.11), frameon=False, fontsize=10,
              title='Fastest per-pass configuration per engine (+ PyTorch eager reference)', title_fontsize=10)
    fig.text(.08, .02, f'y = analytic FLOPs per pass ({FLOPS_PER_PASS/1e12:.2f} TFLOP) / benchmark step time. '
             'Filled: x from measured bytes per pass (Nsight Compute, L2 ↔ system-memory sectors;\nwrites are an upper '
             'bound). Hollow: x is a lower bound from datasheet bandwidth × step time. Ceilings: dashed = datasheet, '
             'solid = measured (jetson-ceilings-001).\nsd.cpp runs Q4_0 weights through INT8 (mul_mat_q) GEMMs and has '
             'no single exact roof; PyTorch dequantizes Q4_0 to BF16 before Tensor Core GEMMs.', fontsize=9, color=MUTED)
    fig.subplots_adjust(left=.09, right=.97, top=.93, bottom=.33)
    bw = ceil['bw_gbs']  # the bandwidth label follows the slope of the roof (computed once the layout is final)
    p0, p1 = ax.transData.transform([(10, 1), (100, 10)])
    x = 11.0
    ax.text(x, bw['measured']/1000*x*1.15, f'{bw["measured"]:.1f} GB/s measured · {bw["datasheet"]:g} datasheet',
            rotation=math.degrees(math.atan2(p1[1]-p0[1], p1[0]-p0[0])), rotation_mode='anchor', fontsize=9,
            color=INK, ha='left', va='bottom')
    save(fig, out/'jetson-engine-roofline-measured')


def captured(pts):
    out = []
    for p in pts:
        if p['capture']:
            ks = json.loads((RUNS/p['capture']/'profile'/'pass-kernels.json').read_text())
            out.append((p, [dict(k, cls=jetson_class(k['kernel'])) for k in ks]))
    return out


def kernel_figure(plt, ceil, caps, out):
    fig, axes = plt.subplots(1, len(caps), figsize=(15, 5.8), sharex=True, sharey=True)
    bw = ceil['bw_gbs']
    for ax, (p, ks) in zip(axes, caps):
        total = sum(k['duration_s'] for k in ks)
        for cls, color in CLASSES.items():
            sel = [k for k in ks if k['cls'] == cls and k['hbm_bytes'] > 0 and k['duration_s'] > 0]
            ax.scatter([k['hbm_bytes'] for k in sel], [k['hbm_bytes']/k['duration_s']/1e9 for k in sel],
                       s=[max(5, 1800*k['duration_s']/total) for k in sel], color=color, alpha=.55, edgecolors='none',
                       label=cls if ax is axes[0] else None)
        ax.axhline(bw['datasheet'], color='#aaaaaa', ls='--', lw=1)
        ax.axhline(bw['measured'], color=INK, lw=1.6)
        tb = sum(k['hbm_bytes'] for k in ks)
        ax.set_title(f"{p['label']} · {len(ks)} kernels · {tb/1e9:.1f} GB · {total:.2f} s", fontweight='bold', fontsize=11.5)
        ax.text(.03, .04, f'time-weighted: {tb/total/1e9:.0f} GB/s', transform=ax.transAxes, fontsize=9.5, color=INK)
        style(ax, 'Bytes moved by the kernel', 'Achieved bandwidth (GB/s)' if ax is axes[0] else '')
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylim(.1, 250)
    axes[0].text(500, bw['measured']/1.12, f'{bw["measured"]:.1f} GB/s measured', ha='left',
                  va='top', fontsize=9, color=INK)
    axes[0].text(500, bw['datasheet']*1.1, f'{bw["datasheet"]:g} GB/s datasheet', ha='left',
                  va='bottom', fontsize=9, color=MUTED)
    from matplotlib.lines import Line2D
    fig.legend([Line2D([], [], ls='', marker='o', ms=LEGEND_SIZE, color=c, alpha=.8) for c in CLASSES.values()],
               list(CLASSES), loc='lower center', ncol=4, frameon=False, fontsize=10, bbox_to_anchor=(.5, .1))
    fig.suptitle('Jetson Orin Nano · klein distilled 512² · one denoising pass · achieved LPDDR5 bandwidth per kernel',
                 fontsize=13.5, fontweight='bold')
    fig.text(.05, .02, 'Nsight Compute kernel replay (one run per counter, merged per launch), no cache flush, clocks '
             'not locked. Bandwidth = (L2 read misses + L2 writes to system memory) × 32 B / kernel time.\n'
             'Marker area ∝ kernel time. Kernels with zero counted bytes are omitted.', fontsize=9, color=MUTED)
    fig.subplots_adjust(left=.06, right=.98, top=.86, bottom=.27, wspace=.08)
    save(fig, out/'jetson-kernel-bandwidth-measured')


def breakdown(caps):
    rows = []
    for p, ks in caps:
        by = {c: {'gb': sum(k['hbm_bytes'] for k in ks if k['cls'] == c)/1e9,
                  'kernel_s': sum(k['duration_s'] for k in ks if k['cls'] == c),
                  'kernels': sum(1 for k in ks if k['cls'] == c)} for c in CLASSES}
        rows.append({'label': p['label'], 'capture': p['capture'], 'by_class': by})
    return rows


def breakdown_figure(plt, rows, out):
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    ys = range(len(rows))
    for ax, key, xlabel in ((a, 'gb', 'LPDDR5 traffic per denoising pass (GB)'),
                            (b, 'kernel_s', 'Kernel time per denoising pass (s, replay)')):
        left = [0.0]*len(rows)
        for cls, color in CLASSES.items():
            vals = [r['by_class'][cls][key] for r in rows]
            ax.barh(list(ys), vals, left=left, height=.62, color=color, edgecolor='white', linewidth=2,
                    label=cls if ax is a else None)
            total = [sum(r['by_class'][c][key] for c in CLASSES) for r in rows]
            for y, v, l, t in zip(ys, vals, left, total):
                if v/t > .09:
                    ax.text(l+v/2, y, f'{v:.0f}' if key == 'gb' else f'{v:.2f}', ha='center', va='center', fontsize=9,
                            color='white')
            left = [l+v for l, v in zip(left, vals)]
        for y, t in zip(ys, left):
            ax.text(t*1.01 + .01*max(left), y, f'{t:.1f}' if key == 'gb' else f'{t:.2f}', va='center', fontsize=10,
                    color=INK, fontweight='bold')
        ax.set_xlim(0, max(left)*1.13)
        ax.set_xlabel(xlabel)
        ax.grid(axis='x', lw=.45, alpha=.3)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(axis='y', length=0)
    a.set_yticks(list(ys), [r['label'] for r in rows]); a.invert_yaxis()
    fig.legend(loc='lower center', ncol=4, frameon=False, fontsize=10, bbox_to_anchor=(.5, .0))
    fig.suptitle('Jetson Orin Nano · klein distilled 512² · where the bytes and the time go in one denoising pass',
                 fontsize=13, fontweight='bold')
    fig.subplots_adjust(left=.12, right=.97, top=.87, bottom=.24, wspace=.06)
    save(fig, out/'jetson-traffic-breakdown')


def upload(output, name, entity, project):
    """One analysis run (resumed by id on re-upload): figures, coordinate table and an evidence artifact."""
    import wandb
    receipt = output/'wandb-upload.json'
    identity = json.loads(receipt.read_text()) if receipt.exists() else {'id': uuid.uuid4().hex[:16]}
    data = json.loads((output/'plot-data.json').read_text())
    cols = ['label', 'engine', 'benchmark_run', 'capture', 'step_s', 'achieved_tflops', 'bytes_per_pass', 'ai',
            'ai_lower_bound', 'effective_gbs', 'bytes_basis']
    images = {f'roofline_images/{s}': output/f'{s}.png' for s in
              ('jetson-engine-roofline-measured', 'jetson-kernel-bandwidth-measured', 'jetson-traffic-breakdown')}
    with wandb.init(entity=entity, project=project, id=identity['id'], resume='allow', name=name, job_type='analysis',
                    group='Analysis', tags=['analysis', 'roofline', 'measured-counters', 'jetson'],
                    config={'ceilings': data['ceilings'], 'flops_per_pass': FLOPS_PER_PASS,
                            'method': 'scripts/plot_roofline_jetson.py docstring'},
                    settings=wandb.Settings(x_disable_stats=True, disable_git=True, disable_code=True)) as run:
        table = wandb.Table(columns=cols)
        for p in data['engine_points']:
            table.add_data(*[p.get(c) for c in cols])
        bt = wandb.Table(columns=['engine', 'class', 'kernels', 'gb_per_pass', 'kernel_s'])
        for r in data['breakdown']:
            for c, v in r['by_class'].items():
                bt.add_data(r['label'], c, v['kernels'], v['gb'], v['kernel_s'])
        run.log({**{k: wandb.Image(str(v)) for k, v in images.items()},
                 'roofline_engine_points': table, 'roofline_traffic_breakdown': bt})
        artifact = wandb.Artifact(name, type='analysis')
        for f in sorted(output.glob('*')):
            if f.suffix in ('.json', '.png', '.pdf') and not f.name.startswith('wandb-'):
                artifact.add_file(str(f))
        artifact.add_file(str(Path(__file__).resolve()), name='plot_roofline_jetson.py')
        run.log_artifact(artifact)
        run.summary.update({'engine_points': len(data['engine_points']),
                            'engines_with_measured_bytes': sum(bool(p['capture']) for p in data['engine_points'])})
        identity['url'] = run.url
    remote = wandb.Api().run(f"{entity}/{project}/{identity['id']}")
    keys = [*images, 'roofline_engine_points', 'roofline_traffic_breakdown']
    files = {f.name for f in remote.files()}
    missing = [k for k in keys if k not in remote.summary or remote.summary[k].get('path') not in files]
    evidence = [a for a in remote.logged_artifacts() if a.type == 'analysis' and a.name.split(':')[0] == name]
    newest = max(evidence, key=lambda a: int(a.version.lstrip('v'))) if evidence else None
    digest = base64.b64encode(hashlib.md5((output/'plot-data.json').read_bytes()).digest()).decode()
    data_ok = bool(newest) and newest.manifest.entries['plot-data.json'].digest == digest
    status = 'verified' if remote.state == 'finished' and not missing and data_ok else 'NOT verified'
    identity.update(status=status, missing_panels=missing, artifact=newest.name if newest else None)
    receipt.write_text(json.dumps(identity, indent=2) + '\n')
    print(status, identity['url'])
    if status != 'verified':
        raise RuntimeError('W&B upload not verified')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--output', type=Path, default=ROOT/'output'/'roofline-jetson-measured')
    ap.add_argument('--upload', action='store_true', help='upload to W&B as one analysis run (resumes its id)')
    ap.add_argument('--name', default='jetson-roofline-measured')
    ap.add_argument('--entity', default='mfaruqi-purdue-university')
    ap.add_argument('--project', default='on-device-diffusion')
    args = ap.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    args.output.mkdir(parents=True, exist_ok=True)
    ceil = ceilings()
    pts = engine_points(ceil)
    caps = captured(pts)
    rows = breakdown(caps)
    with plt.rc_context({'font.family': 'serif', 'font.serif': ['STIXGeneral', 'DejaVu Serif'], 'mathtext.fontset': 'stix',
                         'font.size': 11, 'xtick.direction': 'in', 'ytick.direction': 'in', 'pdf.fonttype': 42}):
        engine_figure(plt, ceil, pts, args.output)
        kernel_figure(plt, ceil, caps, args.output)
        breakdown_figure(plt, rows, args.output)
    (args.output/'plot-data.json').write_text(json.dumps(
        {'ceilings': ceil, 'flops_per_pass': FLOPS_PER_PASS, 'engine_points': pts, 'breakdown': rows,
         'selection': 'fastest per-pass configuration per engine with unchanged output, plus the PyTorch eager reference'},
        indent=2) + '\n')
    print('wrote', args.output)
    if args.upload:
        upload(args.output, args.name, args.entity, args.project)


if __name__ == '__main__':
    main()
