#!/usr/bin/env python3
"""Jetson Orin Nano engine comparison, 2 x 3: rows distilled / Base, columns speed and memory.

Modelled on edge-dit.cpp's RTX 4090 presentation (docs/performance-4090.md: DiT sampling time,
end-to-end time excluding load, peak memory per runtime). Reads completed run directories only:
  python scripts/plot_jetson_engines.py --output output/jetson-engines-20261021 [--upload]

Columns: (1) DiT sampling = measured denoise median, min-max bars; (2) end-to-end per image
excluding model load, stacked text encode / denoise / VAE decode / other (stage medians; their
sum can differ slightly from the wall median, shown as a diamond); (3) whole-system RAM peak
from tegrastats over the whole capture (includes load and the OS), with the capture's first
sample as the idle level. Timers differ per engine: PyTorch CUDA events, sd.cpp host callbacks,
edge-dit engine breakdown (10 ms rounding). One prompt, seed 0, 512x512, Q4_0 transformer.
"""
import argparse
import csv
import hashlib
import json
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT/'results'/'runs'
# (run directory, engine, configuration label, approximate output?)
SELECTION = {
    'distilled': [
        ('jetson-flux-klein-003__repeat__20261014-163411', 'sd.cpp', 'disk-backed reference', False),
        ('jetson-flux-klein-007__baseline__20261005-113106', 'sd.cpp', '+ conditioning reuse', False),
        ('jetson-edgedit-flux-klein-006__baseline__20261014-170149', 'edge-dit', 'resident + VAE tiling', False),
        ('jetson-torch-flux-klein-001__baseline__20261014-171410', 'PyTorch', 'sequential reference', False),
        ('jetson-torch-flux-klein-004__baseline__20261015-141028', 'PyTorch', '+ conditioning reuse', False),
        ('jetson-torch-flux-klein-007__baseline__20261015-145234', 'PyTorch', '+ reuse + torch.compile', False),
    ],
    'base': [
        ('jetson-flux-klein-base-001__baseline__20261005-005357', 'sd.cpp', 'disk-backed reference', False),
        ('jetson-flux-klein-base-005__baseline__20261005-093239', 'sd.cpp', '+ conditioning reuse', False),
        ('jetson-edgedit-flux-klein-base-001__baseline__20261015-193452', 'edge-dit', 'resident + VAE tiling', False),
        ('jetson-edgedit-flux-klein-base-005__baseline__20261015-231521', 'edge-dit', '+ TaylorSeer cache', True),
        ('jetson-torch-flux-klein-base-001__baseline__20261016-050136', 'PyTorch', 'sequential reference', False),
        ('jetson-torch-flux-klein-base-002__baseline__20261016-015207', 'PyTorch', '+ conditioning reuse', False),
        ('jetson-torch-flux-klein-base-004__baseline__20261016-033850', 'PyTorch', '+ reuse + torch.compile', False),
    ],
}
COLOR = {'sd.cpp': '#0072B2', 'edge-dit': '#E69F00', 'PyTorch': '#009E73'}   # Okabe-Ito, as the roofline figures
STAGES = [('text_encode_ms', 'text encode', '#9ecae1'), ('denoise_ms', 'denoise', None),
          ('vae_decode_ms', 'VAE decode', '#bdbdbd'), ('other_ms', 'other (incl. reloads)', '#e0e0e0')]
BOARD_GIB = 7620/1024


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(run):
    d = RUNS/run
    status = json.loads((d/'status.json').read_text())['status']
    assert status == 'complete', f'{run}: {status}'
    s = json.loads((d/'summary.json').read_text())
    assert s['measured_runs'] == 10, run
    m = s['measured']
    tegra = list(csv.DictReader((d/'tegrastats-samples.csv').read_text().splitlines()))
    ram = [float(r['ram_gib']) for r in tegra]
    row = {'run': run, 'wall_ms': m['wall_ms']['median'], 'wall_min': m['wall_ms']['min'], 'wall_max': m['wall_ms']['max'],
           'denoise_ms': m['denoise_ms']['median'], 'denoise_min': m['denoise_ms']['min'], 'denoise_max': m['denoise_ms']['max'],
           'ram_peak_gib': max(ram), 'ram_start_gib': ram[0], 'tegrastats_samples': len(ram),
           'sha256': {f: sha(d/f) for f in ('summary.json', 'status.json', 'tegrastats-samples.csv')}}
    for key, _, _ in STAGES:
        row[key] = (m.get(key) or {}).get('median') or 0.0
    return row


def build(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    data = {w: [{**load(r), 'engine': e, 'config': c, 'approximate': a} for r, e, c, a in rows] for w, rows in SELECTION.items()}
    rc = {'font.family': 'serif', 'font.serif': ['STIXGeneral', 'DejaVu Serif'], 'mathtext.fontset': 'stix',
          'font.size': 11, 'axes.titlesize': 13, 'axes.titleweight': 'bold', 'axes.labelsize': 12,
          'xtick.direction': 'in', 'ytick.direction': 'in', 'pdf.fonttype': 42}
    with plt.rc_context(rc):
        fig, axes = plt.subplots(2, 3, figsize=(17, 10.5), gridspec_kw={'wspace': .08, 'hspace': .32})
        for r, (workload, title) in enumerate([('distilled', 'Distilled · 4 steps, CFG 1'), ('base', 'Base · 50 steps, CFG 4')]):
            rows = data[workload][::-1]  # top-to-bottom in selection order
            y = range(len(rows))
            names = [f"{p['engine']}  {p['config']}" + ('  (approx.)' if p['approximate'] else '') for p in rows]
            colors = [COLOR[p['engine']] for p in rows]
            hatch = ['//' if p['approximate'] else '' for p in rows]
            # (1) DiT sampling
            ax = axes[r][0]
            vals = [p['denoise_ms']/1000 for p in rows]
            err = [[(p['denoise_ms']-p['denoise_min'])/1000 for p in rows], [(p['denoise_max']-p['denoise_ms'])/1000 for p in rows]]
            ax.barh(y, vals, color=colors, hatch=hatch, edgecolor='#333333', lw=.6, xerr=err, capsize=2)
            for i, v, p in zip(y, vals, rows):
                ax.text(p['denoise_max']/1000 + max(vals)*.015, i, f'{v:.1f} s', va='center', fontsize=9.5)
            ax.set_yticks(list(y)); ax.set_yticklabels(names, fontsize=10)
            ax.set_xlim(0, max(p['denoise_max'] for p in rows)/1000*1.2); ax.set_xlabel('DiT sampling (denoise) per image, s')
            ax.set_title(f'({"ad"[r]}) {title}: DiT sampling', loc='left')
            # (2) end-to-end stacked
            ax = axes[r][1]
            left = [0.0]*len(rows)
            for key, label, shade in STAGES:
                part = [p[key]/1000 for p in rows]
                ax.barh(y, part, left=left, color=[shade or COLOR[p['engine']] for p in rows], hatch=hatch if key == 'denoise_ms' else None,
                        edgecolor='#333333', lw=.5, label=label)
                left = [a+b for a, b in zip(left, part)]
            walls = [p['wall_ms']/1000 for p in rows]
            ax.scatter(walls, list(y), marker='D', s=22, color='#333333', zorder=5)
            for i, w in zip(y, walls):
                ax.text(w + max(walls)*.015, i, f'{w:.1f} s', va='center', fontsize=9.5)
            ax.set_yticks(list(y)); ax.set_yticklabels([])
            ax.set_xlim(0, max(walls)*1.2); ax.set_xlabel('End-to-end per image excl. load, s (◆ = wall median)')
            ax.set_title(f'({"be"[r]}) End-to-end by stage', loc='left')
            # (3) memory
            ax = axes[r][2]
            peaks = [p['ram_peak_gib'] for p in rows]
            ax.barh(y, peaks, color=colors, hatch=hatch, edgecolor='#333333', lw=.6, alpha=.85)
            ax.scatter([p['ram_start_gib'] for p in rows], list(y), marker='|', s=200, color='#333333', zorder=5)
            ax.axvline(BOARD_GIB, color='#D55E00', ls='--', lw=1)
            ax.text(BOARD_GIB, len(rows)-.45, f' board {BOARD_GIB:.2f} GiB', color='#D55E00', fontsize=9, ha='right', va='bottom')
            for i, v in zip(y, peaks):
                ax.text(v - .08, i, f'{v:.2f}', va='center', ha='right', fontsize=9.5, color='white', fontweight='bold')
            ax.set_yticks(list(y)); ax.set_yticklabels([])
            ax.set_xlim(0, 8); ax.set_xlabel('Whole-system RAM peak, GiB (| = idle at capture start)')
            ax.set_title(f'({"cf"[r]}) Peak memory', loc='left')
            for ax in axes[r]:
                ax.grid(axis='x', lw=.4, alpha=.35)
        handles = [Patch(facecolor=c, edgecolor='#333333', label=e) for e, c in COLOR.items()] + \
                  [Patch(facecolor=s, edgecolor='#333333', label=l) for _, l, s in STAGES if s] + \
                  [Patch(facecolor='white', edgecolor='#333333', hatch='//', label='approximate output (step cache)')]
        fig.legend(handles=handles, loc='lower center', ncol=7, frameon=False, fontsize=10.5, bbox_to_anchor=(.55, .045))
        fig.suptitle('Jetson Orin Nano 8GB · 25 W · FLUX.2 klein 4B · 512² · Q4_0 transformer — engine comparison',
                     fontsize=17, fontweight='bold', y=.985)
        fig.text(.01, .008, 'Medians of 10 measured generations after 1 first + 3 warm-ups; bars in (a)/(d) show min–max. One prompt, seed 0. '
                 'Timers differ: PyTorch CUDA events, sd.cpp host callbacks, edge-dit engine breakdown (10 ms).\n'
                 'Stage segments are stage medians (their sum can exceed the ◆ wall median). The sd.cpp Oct 5 runs (distilled + reuse, both Base) are from an earlier session. '
                 'Memory: tegrastats whole-system RAM over the whole capture, including load and the OS. Quality not yet measured.',
                 fontsize=9.5, color='#555555')
        fig.subplots_adjust(left=.165, right=.99, top=.93, bottom=.13)
        output.mkdir(parents=True, exist_ok=True)
        fig.savefig(output/'jetson-engines-2x3.png', dpi=200)
        fig.savefig(output/'jetson-engines-2x3.pdf')
    (output/'jetson-engines-data.json').write_text(json.dumps({'selection': SELECTION, 'rows': data,
                                                              'script_sha256': sha(Path(__file__))}, indent=2) + '\n')
    return data


def upload(data, output, name, entity, project):
    import base64
    import wandb
    receipt = output/'wandb-upload.json'
    ident = json.loads(receipt.read_text()) if receipt.exists() else {'id': uuid.uuid4().hex[:16]}
    with wandb.init(entity=entity, project=project, id=ident['id'], resume='allow', name=name, job_type='analysis',
                    group='Analysis', tags=['analysis', 'jetson', 'engine-comparison'],
                    config={'selection': SELECTION, 'reference': 'https://github.com/THU-MIG/edge-dit.cpp/blob/main/docs/performance-4090.md'},
                    settings=wandb.Settings(x_disable_stats=True, disable_git=True, disable_code=True)) as run:
        cols = ['workload', 'engine', 'config', 'approximate', 'run', 'wall_ms', 'denoise_ms', 'text_encode_ms', 'vae_decode_ms',
                'other_ms', 'ram_peak_gib', 'ram_start_gib']
        table = wandb.Table(columns=cols)
        for w, rows in data.items():
            for p in rows:
                table.add_data(w, *[p[c] for c in cols[1:]])
        run.log({'jetson_engines/2x3': wandb.Image(str(output/'jetson-engines-2x3.png')), 'jetson_engines/table': table})
        art = wandb.Artifact(name, type='analysis')
        for f in ('jetson-engines-2x3.png', 'jetson-engines-2x3.pdf', 'jetson-engines-data.json'):
            art.add_file(str(output/f))
        art.add_file(str(Path(__file__).resolve()), name='plot_jetson_engines.py')
        run.log_artifact(art)
        ident['url'] = run.url
    remote = wandb.Api().run(f"{entity}/{project}/{ident['id']}")
    files = {f.name for f in remote.files()}
    ok_panels = all(k in remote.summary and remote.summary[k].get('path') in files for k in ('jetson_engines/2x3', 'jetson_engines/table'))
    ev = [a for a in remote.logged_artifacts() if a.type == 'analysis' and a.name.split(':')[0] == name]
    digest = base64.b64encode(hashlib.md5((output/'jetson-engines-data.json').read_bytes()).digest()).decode()
    ok_art = bool(ev) and ev[-1].manifest.entries['jetson-engines-data.json'].digest == digest
    ident['status'] = 'verified' if remote.state == 'finished' and ok_panels and ok_art else 'NOT verified'
    receipt.write_text(json.dumps(ident, indent=2) + '\n')
    print(ident['status'], ident['url'])
    if ident['status'] != 'verified':
        raise RuntimeError('W&B upload not verified')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--output', type=Path, default=ROOT/'output'/'jetson-engines-20261021')
    ap.add_argument('--upload', action='store_true')
    ap.add_argument('--name', default='klein-jetson-engines-2x3')
    ap.add_argument('--entity', default='mfaruqi-purdue-university')
    ap.add_argument('--project', default='on-device-diffusion')
    args = ap.parse_args()
    data = build(args.output)
    print('wrote', args.output/'jetson-engines-2x3.png')
    if args.upload:
        upload(data, args.output, args.name, args.entity, args.project)


if __name__ == '__main__':
    main()
