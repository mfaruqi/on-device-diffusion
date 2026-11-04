#!/usr/bin/env python3
"""A100 rooflines that use measured Nsight Compute bytes wherever a capture exists.

  python scripts/plot_roofline_measured.py [--ceilings results/runs/<a100-ceilings run>] [--output DIR]

Two figures (PNG + PDF) and a JSON of every plotted coordinate:
- a100-kernel-roofline-measured: one denoising step of the PyTorch BF16 eager reference, one dot per kernel.
  (a) counted operations / measured HBM bytes vs counted operations / kernel time; kernels with no counted
  operations (BF16 elementwise arithmetic is not among the counted instruction classes) cannot be placed and
  are listed with their time share. (b) the same kernels as achieved HBM bandwidth (bytes / kernel time).
- a100-engine-roofline-measured: distilled denoising stage per engine. y = analytic FLOPs per pass / measured
  denoising time per pass (the analytic count matches the counted Tensor Core work within 0.4 %). x uses
  measured HBM bytes per pass when a counter capture exists for that run (filled marker), otherwise the
  analytic estimate from output/roofline-a100-20261021 (hollow marker, labelled estimated; only drawn while a
  configuration has no capture).
Ceilings: datasheet dashed; measured (from measure_ceilings_cuda.py) solid when --ceilings is given.
"""
import argparse
import base64
import csv
import gzip
import hashlib
import json
import statistics
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POINTS = ROOT/'output'/'roofline-a100-20261021'/'roofline-points.csv'
EAGER_COUNTERS = ROOT/'results'/'runs'/'a100-flux-klein-torch28-001__profile__20261024-161538'
STEP = 'denoise_step_1'  # a steady-state step; steps 0-3 differ by < 0.2 % in bytes
DATASHEET = {'bw_gbs': 1555.0, 'bf16_tensor': 312.0, 'fp32': 19.5, 'int4_tensor_tops': 1248.0,
             'source': 'NVIDIA A100 datasheet (PCIe 40GB: same peaks, 250 W)'}
# benchmark run -> counter capture of the same configuration (filled marker when present); engine captures
# from profile_engine_counters.py name their template run in counter-summary.json
CAPTURES = {'a100-flux-klein-torch28-001__baseline__20261015-151548': EAGER_COUNTERS}
for _summary in sorted((ROOT/'results'/'runs').glob('*__counters__*/profile/counter-summary.json')):
    _s = json.loads(_summary.read_text())
    if _s.get('status') == 'validated':
        CAPTURES[_s['template_run']] = _summary.parents[1]
COLORS = {'GEMM': '#0072B2', 'attention': '#D55E00', 'norm': '#009E73', 'elementwise / reduction': '#777777'}
ENGINE = {'PyTorch / Diffusers': ('#009E73', 'o'), 'stable-diffusion.cpp': ('#0072B2', 's'),
          'edge-dit.cpp': ('#E69F00', '^'), 'Nunchaku / SVDQuant': ('#CC79A7', 'D')}
COMPILED_MARKER = 'P'  # torch.compile shares the PyTorch colour; the marker tells it from eager


def engine_style(p):
    color, marker = ENGINE.get(p['engine'], ('#777777', 'o'))
    return color, COMPILED_MARKER if 'compiled' in p['label'] else marker


LABEL_OFFSETS = {  # (x factor, y factor, ha, va) for direct labels that would collide; default: left of the marker
    'PyTorch eager': (1.0, 1.09, 'center', 'bottom'), 'edge-dit': (1.0, 1/1.09, 'center', 'top'),
    'Nunchaku W4A4': (1.0, 1.12, 'center', 'bottom')}


def short_label(p):
    return {'PyTorch / Diffusers': 'PyTorch ' + ('compiled' if 'compiled' in p['label'] else 'eager'),
            'stable-diffusion.cpp': 'sd.cpp', 'edge-dit.cpp': 'edge-dit',
            'Nunchaku / SVDQuant': 'Nunchaku W4A4'}.get(p['engine'], p['label'])


def kernel_class(name):
    k = name.lower()
    if 'flash' in k or 'fmha' in k or 'attention' in k:
        return 'attention'
    if any(s in k for s in ('gemm', 'sm80_xmma', 'cutlass', 'ampere_bf16', 'gemv', 'splitkreduce')):
        return 'GEMM'
    if 'norm' in k:
        return 'norm'
    return 'elementwise / reduction'


def step_kernels():
    with gzip.open(EAGER_COUNTERS/'profile'/'counter-kernels.json.gz', 'rt') as f:
        rows = json.load(f)
    return [dict(r, cls=kernel_class(r['kernel'])) for r in rows if r['stage'] == STEP]


def bytes_per_pass(capture):
    """Measured HBM bytes per denoising pass from a counter capture."""
    summary = json.loads((capture/'profile'/'counter-summary.json').read_text())
    if 'bytes_per_pass' in summary:  # generic engine capture
        return summary['bytes_per_pass'], summary.get('bytes_basis', 'counter capture')
    steps = [v['hbm_bytes'] for k, v in summary['by_stage'].items() if k.startswith('denoise_step_')]
    return statistics.mean(steps), f'mean of {len(steps)} NVTX-labelled denoising steps'


def ceilings(run):
    if not run:
        return None
    s = json.loads((Path(run)/'summary.json').read_text())['ceilings']
    return {'bw_gbs': s['bw_gbs']['measured'], 'bf16_tensor': s['bf16_tensor']['measured'],
            'fp32': s['fp32']['measured'], 'source': Path(run).name}


def roof(ax, xs, bw, peak, **kw):
    ridge = peak/bw
    pts = sorted({xs[0], ridge, xs[1]})
    ax.plot(pts, [min(peak, bw*x) for x in pts], **kw)


def draw_roofs(ax, xlim, measured, roofs):
    for key, label in roofs:
        ax.plot([], [])  # keep colour cycle stable
        roof(ax, xlim, DATASHEET['bw_gbs']/1000, DATASHEET[key], color='#999999', ls='--', lw=1)
        ax.text(xlim[1]/1.15, DATASHEET[key]*1.08, f'{label} {DATASHEET[key]:g} (datasheet)', ha='right',
                fontsize=9, color='#666666')
        if measured and key in measured:
            roof(ax, xlim, measured['bw_gbs']/1000, measured[key], color='#222222', lw=1.8)
            ax.text(xlim[1]/1.15, measured[key]/1.5, f'{label} {measured[key]:.0f} (measured)', ha='right',
                    fontsize=9, color='#222222')


def style(ax, xlabel, ylabel):
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.grid(which='major', lw=.45, alpha=.3); ax.grid(which='minor', lw=.3, alpha=.1)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)


def kernel_figure(plt, measured, out):
    ks = step_kernels()
    total = sum(k['duration_s'] for k in ks)
    fig, (a, b) = plt.subplots(1, 2, figsize=(14, 6.6))
    xlim = (0.1, 1e4)
    draw_roofs(a, xlim, measured, [('bf16_tensor', 'BF16 Tensor Core'), ('fp32', 'FP32 scalar')])
    placed, unplaced = [], []
    for cls, color in COLORS.items():
        pts = [k for k in ks if k['cls'] == cls and k['counted_ops'] > 0 and k['hbm_bytes'] > 0]
        unplaced += [k for k in ks if k['cls'] == cls and not (k['counted_ops'] > 0 and k['hbm_bytes'] > 0)]
        placed += pts
        if pts:
            share = 100*sum(k['duration_s'] for k in pts)/total
            a.scatter([k['counted_ops_per_byte'] for k in pts], [k['counted_tops'] for k in pts],
                      s=[max(6, 2500*k['duration_s']/total) for k in pts], color=color, alpha=.55,
                      edgecolors='none', label=f'{cls} ({len(pts)} kernels, {share:.0f}% of step time)')
    style(a, 'Arithmetic intensity (counted ops / measured HBM byte)', 'Achieved (counted TOP/s)')
    a.set_xlim(*xlim); a.set_ylim(1e-3, 600)
    a.set_title('(a) Kernels with counted operations', fontweight='bold')
    un_share = 100*sum(k['duration_s'] for k in unplaced)/total
    a.legend(loc='upper left', bbox_to_anchor=(0, -.14), frameon=False, fontsize=9.5,
             title=f'Not placeable (no counted ops): {len(unplaced)} kernels, {un_share:.0f}% of step time',
             title_fontsize=9.5)
    # (b) achieved bandwidth for every kernel that moved bytes
    for cls, color in COLORS.items():
        pts = [k for k in ks if k['cls'] == cls and k['hbm_bytes'] > 0]
        b.scatter([k['hbm_bytes'] for k in pts], [k['hbm_bytes']/k['duration_s']/1e9 for k in pts],
                  s=[max(6, 2500*k['duration_s']/total) for k in pts], color=color, alpha=.55, edgecolors='none')
    b.axhline(DATASHEET['bw_gbs'], color='#999999', ls='--', lw=1)
    bw_label = f'{DATASHEET["bw_gbs"]:g} GB/s (datasheet, dashed)'
    if measured:
        b.axhline(measured['bw_gbs'], color='#222222', lw=1.8)
        bw_label = f'{measured["bw_gbs"]:.0f} GB/s (measured, solid) · ' + bw_label  # the two lines nearly coincide
    b.text(1.2e3, DATASHEET['bw_gbs']*1.12, bw_label, fontsize=9)
    style(b, 'HBM bytes moved by the kernel', 'Achieved HBM bandwidth (GB/s)')
    b.set_ylim(1, 3000)
    b.set_title('(b) Memory traffic: achieved bandwidth per kernel', fontweight='bold')
    elem = [k for k in ks if k['cls'] == 'elementwise / reduction']
    eb, et = sum(k['hbm_bytes'] for k in elem), sum(k['duration_s'] for k in elem)
    b.text(.02, .03, f'elementwise / reduction: {eb/1e9:.1f} GB in {1e3*et:.1f} ms = {eb/et/1e9:.0f} GB/s time-weighted',
           transform=b.transAxes, fontsize=9.5)
    fig.suptitle('A100 PCIe 40GB · PyTorch 2.8 BF16 eager · klein distilled 1024² · one denoising step (measured counters)',
                 fontsize=14, fontweight='bold')
    fig.text(.06, .015, 'Nsight Compute kernel replay, no cache flush, clocks not locked. Counted: BF16/FP16/TF32 Tensor Core ops + FP32 add/mul/FMA; '
             'scalar BF16/FP16 and SFU ops are not counted,\nso elementwise kernels sit too low or cannot be placed in (a). '
             'Marker area ∝ kernel time. Ceilings: dashed = datasheet' + (
                 '; solid = measured sustained (60 s GEMM; nvidia-smi reports software power-cap throttling).\nKernel replay runs single kernels at boost '
                 'clock, so large GEMMs can sit above the measured BF16 line.' if measured else '.'),
             fontsize=9, color='#555555')
    fig.subplots_adjust(left=.06, right=.98, top=.88, bottom=.35, wspace=.22)
    save(fig, out/'a100-kernel-roofline-measured')
    return {'step': STEP, 'kernels': len(ks), 'placed': len(placed), 'unplaced_time_share_pct': un_share,
            'by_class': {c: {'kernels': sum(1 for k in ks if k['cls'] == c),
                             'time_share_pct': 100*sum(k['duration_s'] for k in ks if k['cls'] == c)/total,
                             'hbm_gb': sum(k['hbm_bytes'] for k in ks if k['cls'] == c)/1e9,
                             'counted_tops': sum(k['counted_ops'] for k in ks if k['cls'] == c)/1e12} for c in COLORS}}


def engine_points():
    pts = []
    for r in csv.DictReader(POINTS.open()):
        if r['device'] != 'gilbreth-a100-40gb' or r['tier'] != 'stage' or 'klein-base-' in r['source_run']:
            continue
        flops, passes, ms = float(r['flops']), float(r['passes']), float(r['duration_ms'])
        p = {'label': r['label'], 'engine': r['engine'], 'source_run': r['source_run'], 'precision': r['precision'],
             'flops_per_pass': flops/passes, 'ms_per_pass': ms/passes, 'achieved_tflops': flops/(ms*1e-3)/1e12,
             'analytic_ai': float(r['arithmetic_intensity'])}
        cap = CAPTURES.get(r['source_run'])
        if cap:
            b, basis = bytes_per_pass(cap)
            p.update(measured_bytes_per_pass=b, measured_ai=p['flops_per_pass']/b, bytes_basis=basis, capture=cap.name)
        pts.append(p)
    return pts


def engine_figure(plt, measured, out):
    pts = engine_points()
    fig, ax = plt.subplots(figsize=(10, 7))
    xlim = (10, 1e4)
    draw_roofs(ax, xlim, measured, [('bf16_tensor', 'BF16 Tensor Core'), ('fp32', 'FP32 scalar')])
    ax.axhline(DATASHEET['int4_tensor_tops'], color='#CC79A7', ls=':', lw=1)
    ax.text(xlim[0]*1.2, DATASHEET['int4_tensor_tops']*1.07, 'INT4 Tensor Core 1248 TOP/s (Nunchaku W4A4 GEMMs; integer ops)',
            fontsize=9, color='#CC79A7')
    for p in pts:
        color, marker = engine_style(p)
        if 'measured_ai' in p:
            ax.scatter([p['measured_ai']], [p['achieved_tflops']], marker=marker, s=110, color=color, zorder=5,
                       label=p['label'] if all('measured_ai' in q for q in pts) else f"{p['label']} — x measured")
            name = short_label(p)
            dx, dy, ha, va = LABEL_OFFSETS.get(name, (1/1.12, 1, 'right', 'center'))
            ax.text(p['measured_ai']*dx, p['achieved_tflops']*dy, name, ha=ha, va=va, fontsize=9)
        else:
            ax.scatter([p['analytic_ai']], [p['achieved_tflops']], marker=marker, s=110, facecolors='none',
                       edgecolors=color, linewidths=1.6, zorder=5, label=f"{p['label']} — x estimated")
    style(ax, 'Arithmetic intensity (FLOP per HBM byte)', 'Effective performance (TFLOP/s)')
    ax.set_xlim(*xlim); ax.set_ylim(10, 2000)
    ax.set_title('A100 PCIe 40GB · klein distilled 1024², 4 steps · denoising stage per engine', fontweight='bold')
    ax.legend(loc='upper left', bbox_to_anchor=(0, -.11), frameon=False, fontsize=10)
    fig.text(.08, .02, 'y = analytic FLOPs per pass / measured denoising time (analytic count = counted Tensor Core work within 0.4 %).\n'
             'Filled: x from measured HBM bytes per denoising pass (Nsight Compute).'
             + ('' if all('measured_ai' in p for p in pts) else ' Hollow: x still estimated (analytic bytes).') + '\nCeilings: dashed = datasheet' + ('; solid = measured.' if measured else '.') +
             ' Quantized and FP32-activation engines have no single exact roof.', fontsize=9, color='#555555')
    fig.subplots_adjust(left=.1, right=.97, top=.92, bottom=.36)
    save(fig, out/'a100-engine-roofline-measured')
    return pts


def save(fig, stem):
    fig.savefig(stem.with_suffix('.png'), dpi=200)
    fig.savefig(stem.with_suffix('.pdf'))


def upload(output, name, entity, project):
    """One analysis run (resumed by id on re-upload): both figures, the coordinate table and an evidence artifact."""
    import wandb
    receipt = output/'wandb-upload.json'
    identity = json.loads(receipt.read_text()) if receipt.exists() else {'id': uuid.uuid4().hex[:16]}
    data = json.loads((output/'plot-data.json').read_text())
    cols = ['label', 'engine', 'source_run', 'precision', 'achieved_tflops', 'analytic_ai', 'measured_ai',
            'measured_bytes_per_pass', 'bytes_basis', 'capture']
    with wandb.init(entity=entity, project=project, id=identity['id'], resume='allow', name=name, job_type='analysis',
                    group='Analysis', tags=['analysis', 'roofline', 'measured-counters', 'a100'],
                    config={'datasheet': data['datasheet'], 'measured_ceilings': data['measured_ceilings'],
                            'inputs': data['inputs'], 'method': 'scripts/plot_roofline_measured.py docstring'},
                    settings=wandb.Settings(x_disable_stats=True, disable_git=True, disable_code=True)) as run:
        table = wandb.Table(columns=cols)
        for p in data['engine_points']:
            table.add_data(*[p.get(c) for c in cols])
        kt = wandb.Table(columns=['class', 'kernels', 'time_share_pct', 'hbm_gb', 'counted_tops'])
        for c, v in data['kernel_step']['by_class'].items():
            kt.add_data(c, v['kernels'], v['time_share_pct'], v['hbm_gb'], v['counted_tops'])
        run.log({'roofline_images/a100-kernel-measured': wandb.Image(str(output/'a100-kernel-roofline-measured.png')),
                 'roofline_images/a100-engine-measured': wandb.Image(str(output/'a100-engine-roofline-measured.png')),
                 'roofline_engine_points': table, 'roofline_kernel_classes': kt})
        artifact = wandb.Artifact(name, type='analysis')
        for f in sorted(output.glob('*')):
            if f.suffix in ('.json', '.png', '.pdf') and not f.name.startswith('wandb-'):
                artifact.add_file(str(f))
        artifact.add_file(str(Path(__file__).resolve()), name='plot_roofline_measured.py')
        run.log_artifact(artifact)
        run.summary.update({'engine_points': len(data['engine_points']),
                            'engines_with_measured_bytes': sum('measured_ai' in p for p in data['engine_points']),
                            'measured_ceilings': bool(data['measured_ceilings'])})
        identity['url'] = run.url
    remote = wandb.Api().run(f"{entity}/{project}/{identity['id']}")
    keys = ['roofline_images/a100-kernel-measured', 'roofline_images/a100-engine-measured',
            'roofline_engine_points', 'roofline_kernel_classes']
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
    ap.add_argument('--ceilings', help='results/runs/<a100-ceilings run> (measured ceilings)')
    ap.add_argument('--output', type=Path, default=ROOT/'output'/'roofline-a100-measured')
    ap.add_argument('--upload', action='store_true', help='upload to W&B as one analysis run (resumes its id)')
    ap.add_argument('--name', default='a100-roofline-measured')
    ap.add_argument('--entity', default='mfaruqi-purdue-university')
    ap.add_argument('--project', default='on-device-diffusion')
    args = ap.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    args.output.mkdir(parents=True, exist_ok=True)
    measured = ceilings(args.ceilings)
    with plt.rc_context({'font.family': 'serif', 'font.serif': ['STIXGeneral', 'DejaVu Serif'], 'mathtext.fontset': 'stix',
                         'font.size': 11, 'xtick.direction': 'in', 'ytick.direction': 'in', 'pdf.fonttype': 42}):
        kernels = kernel_figure(plt, measured, args.output)
        engines = engine_figure(plt, measured, args.output)
    (args.output/'plot-data.json').write_text(json.dumps(
        {'datasheet': DATASHEET, 'measured_ceilings': measured, 'kernel_step': kernels, 'engine_points': engines,
         'inputs': {'points': str(POINTS.relative_to(ROOT)), 'counters': str(EAGER_COUNTERS.relative_to(ROOT))}},
        indent=2, default=str) + '\n')
    print('wrote', args.output)
    if args.upload:
        upload(args.output, args.name, args.entity, args.project)


if __name__ == '__main__':
    main()
