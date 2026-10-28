#!/usr/bin/env python3
"""Best end-to-end time per engine with acceptable image quality, per device and workload (log scale).

  python scripts/plot_best_configs.py [--output output/best-configs-20261028] [--upload]

Rows and quality flags come from scripts/make_performance_doc.py (the same run catalogue and ⚠ rule): for each
device, workload and engine, the solid bar is the fastest completed configuration without ⚠ (no paired GenEval
interval entirely below zero and no visibly damaged benchmark image); the pale bar behind it is that engine's
reference on the same device and workload. Precision differs between engines (see the performance doc); a dotted
outline marks a configuration whose quality has not been assessed against its own reference (replacement
checkpoint). Writes best-configs.{png,pdf,json} and copies the PNG to docs/assets/.
"""
import argparse
import json
import shutil
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import make_performance_doc as perf  # noqa: E402

ENGINES = ['PyTorch / Diffusers', 'stable-diffusion.cpp', 'edge-dit.cpp', 'Nunchaku']
COLORS = {'PyTorch / Diffusers': '#2a78d6', 'stable-diffusion.cpp': '#eb6834', 'edge-dit.cpp': '#1baf7a',
          'Nunchaku': '#eda100'}
PALE = {'PyTorch / Diffusers': '#cde2fb', 'stable-diffusion.cpp': '#f9d5c6', 'edge-dit.cpp': '#c4ecdc',
        'Nunchaku': '#fbe6b3'}
GROUPS = [('A100', 'klein-distilled', 'A100\ndistilled 1024²'), ('A100', 'klein-base', 'A100\nBase 1024²'),
          ('Jetson', 'klein-distilled', 'Jetson\ndistilled 512²'), ('Jetson', 'klein-base', 'Jetson\nBase 512²')]
UNASSESSED = ('Nunchaku',)  # replacement checkpoint; fixed-seed images differ and quality is not yet measured


def short(label):
    return label.split(' (')[0].replace('Conditioning reuse', 'reuse').replace('PyTorch ', '').replace('sd.cpp ', '')


def select(geneval):
    speed_refs = perf.reference_runs()
    out = []
    for device, model, label in GROUPS:
        rows = [r for r in perf.rows_for(device, geneval) if r['model'] == model]
        for engine in ENGINES:
            mine = [r for r in rows if r['system'] == engine]
            ok = [r for r in mine if not r['degraded']]
            if not ok:
                note = ('blocked: sm_87' if device == 'Jetson' and engine == 'Nunchaku'
                        else 'no Base checkpoint' if engine == 'Nunchaku' else 'not run')
                out.append({'group': label.replace('\n', ' '), 'device': device, 'model': model, 'engine': engine,
                            'note': note})
                continue
            best = min(ok, key=lambda r: r['e2e'])
            ref_pool = [r for r in mine if r['run'] in speed_refs] or [r for r in mine if r['reference']] or [best]
            ref = max(ref_pool, key=lambda r: r['e2e'])
            out.append({'group': label.replace('\n', ' '), 'device': device, 'model': model, 'engine': engine,
                        'config': best['label'], 'run': best['run'], 'precision': best['precision'],
                        'e2e_s': best['e2e']/1e3, 'reference_run': ref['run'], 'reference_label': ref['label'],
                        'reference_e2e_s': ref['e2e']/1e3, 'speedup_vs_reference': ref['e2e']/best['e2e'],
                        'quality_unassessed': engine in UNASSESSED})
    return out


def figure(rows, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    with plt.rc_context({'font.family': 'sans-serif', 'font.size': 11, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42}):
        fig, ax = plt.subplots(figsize=(15, 7.6))
        width = .82/len(ENGINES)
        for g, (_, _, label) in enumerate(GROUPS):
            for e, engine in enumerate(ENGINES):
                x = g - .41 + width*(e + .5)
                r = next(r for r in rows if r['group'] == label.replace('\n', ' ') and r['engine'] == engine)
                if 'note' in r:
                    ax.text(x, 0.55, r['note'], rotation=90, ha='center', va='bottom', fontsize=9, color='#6b6b67')
                    continue
                ax.bar(x, r['reference_e2e_s'] - 0.4, width*.92, bottom=0.4, color=PALE[engine], linewidth=0, zorder=1)
                ax.bar(x, r['e2e_s'] - 0.4, width*.62, bottom=0.4, color=COLORS[engine], zorder=2,
                       linestyle=':' if r['quality_unassessed'] else '-',
                       edgecolor='#2b2b29' if r['quality_unassessed'] else COLORS[engine],
                       linewidth=1.3 if r['quality_unassessed'] else 0)
                t = r['e2e_s']
                value = f'{t:.2f} s' if t < 10 else f'{t:.1f} s' if t < 100 else f'{t:.0f} s'
                ax.text(x, r['reference_e2e_s']*1.08, f"{value} · {short(r['config'])}", rotation=90, ha='center',
                        va='bottom', fontsize=9, color='#2b2b29')
        ax.set_yscale('log')
        ax.set_ylim(0.4, 6000)
        ax.yaxis.set_major_locator(ticker.FixedLocator([0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500]))
        ax.yaxis.set_major_formatter(ticker.FixedFormatter(['0.5', '1', '2', '5', '10', '20', '50', '100', '200', '500']))
        ax.yaxis.set_minor_locator(ticker.NullLocator())
        ax.grid(axis='y', lw=.4, alpha=.35)
        ax.set_xticks(range(len(GROUPS)))
        ax.set_xticklabels([g for _, _, g in GROUPS], fontsize=11.5)
        ax.set_xlim(-.55, len(GROUPS) - .45)
        ax.set_ylabel('End-to-end time per image (s, lower is better)', fontsize=12.5, fontweight='bold')
        ax.set_title('Best configuration per engine with acceptable image quality', fontsize=15, fontweight='bold',
                     loc='left', pad=34)
        handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[e]) for e in ENGINES] + \
                  [plt.Rectangle((0, 0), 1, 1, color='#d9d9d6')]
        ax.legend(handles, [*ENGINES, 'engine reference (pale)'], ncol=5, loc='lower left', bbox_to_anchor=(0, 1.0),
                  frameon=False, fontsize=11, handlelength=1.2, columnspacing=1.4)
        fig.text(.07, .02,
                 'Solid bar: fastest completed configuration of that engine without ⚠ (no paired GenEval interval entirely below '
                 'zero, no visibly damaged image); pale bar: the engine\'s reference on the same device and workload.\n'
                 'Median wall time per generation, model load excluded. Precision differs by engine (A100: BF16 except '
                 'edge-dit Base Q4_0 and Nunchaku W4A4; Jetson: Q4_0 transformers with different text encoders).\n'
                 'Dotted outline: replacement checkpoint whose image quality is not yet assessed. Same-precision rules and '
                 'per-run numbers: docs/performance-a100-jetson.md.', fontsize=9.5, color='#55554f')
        fig.subplots_adjust(left=.07, right=.99, top=.86, bottom=.2)
        fig.savefig(out/'best-configs.png', dpi=200)
        fig.savefig(out/'best-configs.pdf')
    shutil.copy(out/'best-configs.png', ROOT/'docs'/'assets'/f'{out.name}.png')


def upload(out, name='klein-best-configs', entity='mfaruqi-purdue-university', project='on-device-diffusion'):
    import base64
    import hashlib
    import wandb
    receipt = out/'wandb-upload.json'
    identity = json.loads(receipt.read_text()) if receipt.exists() else {'id': uuid.uuid4().hex[:16]}
    rows = json.loads((out/'best-configs.json').read_text())['rows']
    cols = ['group', 'engine', 'config', 'precision', 'e2e_s', 'reference_label', 'reference_e2e_s',
            'speedup_vs_reference', 'quality_unassessed', 'run', 'reference_run', 'note']
    with wandb.init(entity=entity, project=project, id=identity['id'], resume='allow', name=name, job_type='analysis',
                    group='Analysis', tags=['analysis', 'best-config'],
                    settings=wandb.Settings(x_disable_stats=True, disable_git=True, disable_code=True)) as run:
        table = wandb.Table(columns=cols)
        for r in rows:
            table.add_data(*[r.get(c) for c in cols])
        run.log({'best_config_images/best-configs': wandb.Image(str(out/'best-configs.png')), 'best_configs': table})
        art = wandb.Artifact(name, type='analysis')
        for f in ('best-configs.png', 'best-configs.pdf', 'best-configs.json'):
            art.add_file(str(out/f))
        art.add_file(str(Path(__file__).resolve()), name='plot_best_configs.py')
        run.log_artifact(art)
        identity['url'] = run.url
    remote = wandb.Api().run(f"{entity}/{project}/{identity['id']}")
    files = {f.name for f in remote.files()}
    missing = [k for k in ('best_config_images/best-configs', 'best_configs') if remote.summary.get(k, {}).get('path') not in files]
    arts = [a for a in remote.logged_artifacts() if a.name.split(':')[0] == name]
    newest = max(arts, key=lambda a: int(a.version.lstrip('v'))) if arts else None
    digest = base64.b64encode(hashlib.md5((out/'best-configs.json').read_bytes()).digest()).decode()
    ok = remote.state == 'finished' and not missing and newest and newest.manifest.entries['best-configs.json'].digest == digest
    identity['status'] = 'verified' if ok else 'NOT verified'
    receipt.write_text(json.dumps(identity, indent=2) + '\n')
    print(identity['status'], identity['url'])
    if not ok:
        raise RuntimeError('W&B upload not verified')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--output', type=Path, default=ROOT/'output'/'best-configs-20261028')
    ap.add_argument('--upload', action='store_true')
    args = ap.parse_args()
    import export_quality_wandb as quality
    rows = select(quality.attachments())
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'best-configs.json').write_text(json.dumps({'rule': __doc__.split('\n\n')[1], 'rows': rows}, indent=2) + '\n')
    figure(rows, args.output)
    print('wrote', args.output)
    if args.upload:
        upload(args.output)


if __name__ == '__main__':
    main()
