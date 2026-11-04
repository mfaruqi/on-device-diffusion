#!/usr/bin/env python3
"""Grouped speedup bars (log scale) per device and workload, one panel per engine.

  python scripts/plot_speedups.py [--output output/speedups-20261028] [--upload]

Speedup = median wall time of the variant's labelled reference / median wall time of the variant (measured
generations only, from output/stage-breakdown-20261007/chart-data.json, which reads each run's summary.json).
Each bar is one labelled configuration against its own reference on the same device and workload, so groups
compare how an option transfers, not how fast devices are. A missing bar is drawn as a short note: not run,
stopped by the memory guard, or excluded. Hatched bars change the image in a way the project measured as
worse (GenEval loss or visibly damaged text); the other bars are bit-identical or have no measured quality cost.
"""
import argparse
import json
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'output'/'stage-breakdown-20261007'/'chart-data.json'
GRAY = '#8a8a86'
# colour follows the option across panels; torch.compile (panel a only) and cache-dit (panel c only) share a hue
COLORS = {'Conditioning reuse': '#2a78d6', 'torch.compile': '#eb6834', 'TaylorSeer': '#1baf7a', 'EasyCache': '#eda100',
          'UCache': '#e87ba4', 'DiCache': '#008300', 'MagCache': '#6250d6', 'DBCache': '#e34948', 'cache-dit': '#eb6834'}
NOTES = {'oom': 'mem. guard', 'na': 'not run', 'excl': 'excluded', 'gpu': 'diff. GPU'}

# (group label, reference run, {bar: variant run | (variant run, its own reference run) | note key})
PANELS = [
    ('(a) PyTorch / Diffusers', ['Conditioning reuse', 'torch.compile', 'TaylorSeer'], {'TaylorSeer'}, [
        ('A100\ndistilled 1024²', 'a100-flux-klein-torch28-001__baseline__20261015-151548', {
            'Conditioning reuse': 'a100-flux-klein-torch28-002__baseline__20261015-151840',
            'torch.compile': 'a100-flux-klein-compile-001__baseline__20261015-151740',
            'TaylorSeer': 'a100-flux-klein-torch28-004__baseline__20261015-151949'}),
        ('A100\nBase 1024²', 'a100-flux-klein-base-torch28-001__baseline__20261015-155338', {
            'Conditioning reuse': 'a100-flux-klein-base-torch28-002__baseline__20261015-160117',
            'torch.compile': 'a100-flux-klein-base-torch28-004__baseline__20261015-161253',
            'TaylorSeer': 'a100-flux-klein-base-torch28-003__baseline__20261015-160821'}),
        ('Jetson\ndistilled 512²', 'jetson-torch-flux-klein-001__baseline__20261014-171410', {
            'Conditioning reuse': 'jetson-torch-flux-klein-004__baseline__20261015-141028',
            'torch.compile': ('jetson-torch-flux-klein-007__baseline__20261015-145234', 'jetson-torch-flux-klein-004__baseline__20261015-141028'),
            'TaylorSeer': 'jetson-torch-flux-klein-006__baseline__20261015-142456'}),
        ('Jetson\nBase 512²', 'jetson-torch-flux-klein-base-001__baseline__20261016-050136', {
            'Conditioning reuse': 'jetson-torch-flux-klein-base-002__baseline__20261016-015207',
            'torch.compile': ('jetson-torch-flux-klein-base-004__baseline__20261016-033850', 'jetson-torch-flux-klein-base-002__baseline__20261016-015207'),
            'TaylorSeer': 'oom'}),
        ('M5\ndistilled 512²', 'm5-flux-klein-001__baseline__20261010-183446', {
            'Conditioning reuse': 'm5-flux-klein-002__baseline__20261010-203805', 'torch.compile': 'excl', 'TaylorSeer': 'excl'}),
        ('M5\nBase 512²', 'm5-flux-klein-base-001__baseline__20261010-184312', {
            'Conditioning reuse': 'm5-flux-klein-base-002__baseline__20261010-210907', 'torch.compile': 'excl', 'TaylorSeer': 'excl'}),
    ]),
    ('(b) stable-diffusion.cpp', ['Conditioning reuse', 'EasyCache'], set(), [
        ('A100\ndistilled 1024²', 'a100-sdcpp-flux-klein-001__baseline__20261005-220435', {
            'Conditioning reuse': 'a100-sdcpp-flux-klein-003__baseline__20261005-220737',
            'EasyCache': 'a100-sdcpp-flux-klein-002__baseline__20261005-220636'}),
        ('A100\nBase 1024²', 'a100-sdcpp-flux-klein-base-001__baseline__20261005-214624', {
            'Conditioning reuse': 'gpu', 'EasyCache': 'a100-sdcpp-flux-klein-base-002__baseline__20261005-215731'}),
        ('Jetson\ndistilled 512²', 'jetson-flux-klein-003__baseline__20261005-105911', {
            'Conditioning reuse': 'jetson-flux-klein-007__baseline__20261005-113106',
            'EasyCache': 'jetson-flux-klein-006__baseline__20261005-111428'}),
        ('Jetson\nBase 512²', 'jetson-flux-klein-base-001__baseline__20261005-005357', {
            'Conditioning reuse': 'jetson-flux-klein-base-005__baseline__20261005-093239',
            'EasyCache': 'jetson-flux-klein-base-002__baseline__20261005-024556'}),
        ('M5\ndistilled 512²', 'm5-sdcpp-flux-klein-001__baseline__20261010-165421', {
            'Conditioning reuse': 'm5-sdcpp-flux-klein-003__baseline__20261010-205038',
            'EasyCache': 'm5-sdcpp-flux-klein-002__baseline__20261010-204326'}),
        ('M5\nBase 512²', 'm5-sdcpp-flux-klein-base-001__baseline__20261010-165813', {
            'Conditioning reuse': 'm5-sdcpp-flux-klein-base-003__baseline__20261010-233405',
            'EasyCache': 'm5-sdcpp-flux-klein-base-002__baseline__20261010-223355'}),
    ]),
    ('(c) edge-dit.cpp step caches', ['EasyCache', 'UCache', 'DBCache', 'cache-dit', 'TaylorSeer', 'DiCache', 'MagCache'], {'MagCache'}, [
        ('A100\ndistilled 1024²', 'a100-edgedit-flux-klein-001__baseline__20261005-233237', {
            'MagCache': 'a100-edgedit-flux-klein-007__baseline__20261015-152117'}),
        ('A100\nBase 1024²', 'a100-edgedit-flux-klein-base-001__baseline__20261015-165052', {
            'EasyCache': 'a100-edgedit-flux-klein-base-002__baseline__20261015-170356',
            'UCache': 'a100-edgedit-flux-klein-base-003__baseline__20261015-171344',
            'DBCache': 'a100-edgedit-flux-klein-base-004__baseline__20261015-172354',
            'TaylorSeer': 'a100-edgedit-flux-klein-base-005__baseline__20261015-173406',
            'cache-dit': 'a100-edgedit-flux-klein-base-006__baseline__20261015-174257',
            'MagCache': 'a100-edgedit-flux-klein-base-007__baseline__20261015-175238',
            'DiCache': 'a100-edgedit-flux-klein-base-008__baseline__20261015-175727'}),
        ('Jetson\ndistilled 512²', 'jetson-edgedit-flux-klein-006__baseline__20261014-170149', {
            'TaylorSeer': 'jetson-edgedit-flux-klein-011__baseline__20261015-133915',
            'MagCache': 'jetson-edgedit-flux-klein-013__baseline__20261015-135118',
            'DiCache': 'jetson-edgedit-flux-klein-014__baseline__20261015-135841'}),
        ('Jetson\nBase 512²', 'jetson-edgedit-flux-klein-base-001__baseline__20261015-193452', {
            'EasyCache': 'jetson-edgedit-flux-klein-base-002__baseline__20261015-204021',
            'UCache': 'jetson-edgedit-flux-klein-base-003__baseline__20261015-213038',
            'DBCache': 'jetson-edgedit-flux-klein-base-004__baseline__20261015-222249',
            'TaylorSeer': 'jetson-edgedit-flux-klein-base-005__baseline__20261015-231521',
            'cache-dit': 'jetson-edgedit-flux-klein-base-006__baseline__20261016-000110',
            'MagCache': 'jetson-edgedit-flux-klein-base-007__baseline__20261016-005119',
            'DiCache': 'oom'}),
    ]),
]


def speedups(runs):
    rows = []
    for panel, bars, degraded, groups in PANELS:
        for group, ref, variants in groups:
            for bar in bars:
                v = variants.get(bar, 'na')
                row = {'panel': panel, 'group': group.replace('\n', ' '), 'bar': bar, 'degraded_quality': bar in degraded}
                if v in NOTES:
                    row['note'] = NOTES[v]
                else:
                    run, own_ref = v if isinstance(v, tuple) else (v, ref)
                    row.update(run=run, reference_run=own_ref, reference_median_s=runs[own_ref]['median_s'],
                               variant_median_s=runs[run]['median_s'],
                               speedup=runs[own_ref]['median_s']/runs[run]['median_s'],
                               incremental=own_ref != ref)
                rows.append(row)
    return rows


def figure(rows, out):
    import matplotlib
    import matplotlib.ticker
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with plt.rc_context({'font.family': 'sans-serif', 'font.size': 11, 'axes.spines.top': False,
                         'axes.spines.right': False, 'hatch.linewidth': 0.8, 'pdf.fonttype': 42}):
        fig, axes = plt.subplots(3, 1, figsize=(15, 13.5), gridspec_kw={'hspace': .66})
        for ax, (panel, bars, degraded, groups) in zip(axes, PANELS):
            series = ['Reference', *bars]
            colors = [GRAY, *[COLORS[b] for b in bars]]
            width = .82/len(series)
            for g, (group, _, _) in enumerate(groups):
                label = group.replace('\n', ' ')
                for b, (bar, color) in enumerate(zip(series, colors)):
                    x = g - .41 + width*(b + .5)
                    if bar == 'Reference':
                        value, note, hatch = 1.0, None, None
                    else:
                        r = next(r for r in rows if r['panel'] == panel and r['group'] == label and r['bar'] == bar)
                        value, note = r.get('speedup'), r.get('note')
                        hatch = '////' if r['degraded_quality'] else None
                    if value is None:
                        ax.text(x, 0.62, note, rotation=90, ha='center', va='bottom', fontsize=8.5, color='#6b6b67')
                        continue
                    ax.bar(x, value - 0.5, width*.88, bottom=0.5, color=color, hatch=hatch,
                           edgecolor='white', linewidth=0)
                    star = '†' if bar != 'Reference' and r.get('incremental') else ''
                    ax.text(x, value*1.07, f'{value:.2f}{star}' if value < 10 else f'{value:.1f}{star}', rotation=90,
                            ha='center', va='bottom', fontsize=9, color='#2b2b29')
            ax.set_yscale('log')
            ax.set_ylim(0.5, 9)
            ax.set_yticks([0.5, 1, 2, 4, 8])
            ax.set_yticklabels(['0.5', '1', '2', '4', '8'])
            ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
            ax.axhline(1, color='#2b2b29', lw=.8, zorder=0)
            ax.grid(axis='y', which='major', lw=.4, alpha=.35)
            ax.set_xticks(range(len(groups)))
            ax.set_xticklabels([g for g, _, _ in groups], fontsize=11)
            ax.set_xlim(-.55, len(groups) - .45)
            ax.set_ylabel('Speedup (×)', fontsize=13, fontweight='bold')
            ax.set_title(panel, fontsize=15, fontweight='bold', loc='left', pad=34)
            handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors]
            ax.legend(handles, series, ncol=len(series), loc='lower left', bbox_to_anchor=(0, 1.0), frameon=False,
                      fontsize=11, handlelength=1.2, columnspacing=1.4)
        fig.text(.07, .015,
                 'Speedup = median wall time of the reference / median wall time of the configuration (measured generations; '
                 'same device, workload and engine).\n'
                 '† torch.compile on Jetson is measured on top of conditioning reuse (its labelled reference). '
                 'Hatched: output measurably worse (PyTorch TaylorSeer: GenEval −2 to −5 pts; MagCache: deformed text).\n'
                 'mem. guard = stopped by the Jetson memory guard; excluded = M5 timings not comparable with the M5 reference; '
                 'diff. GPU = reference and variant ran on different A100s; not run = configuration not measured.',
                 fontsize=9.5, color='#55554f')
        fig.subplots_adjust(left=.07, right=.99, top=.95, bottom=.09)
        fig.savefig(out/'speedups.png', dpi=200)
        fig.savefig(out/'speedups.pdf')
    # the committed copy that docs/performance-a100-jetson.md embeds (output/ images stay local)
    __import__('shutil').copy(out/'speedups.png', ROOT/'docs'/'assets'/f'{out.name}.png')


def upload(out, name='klein-speedups', entity='mfaruqi-purdue-university', project='on-device-diffusion'):
    import base64
    import hashlib
    import wandb
    receipt = out/'wandb-upload.json'
    identity = json.loads(receipt.read_text()) if receipt.exists() else {'id': uuid.uuid4().hex[:16]}
    rows = json.loads((out/'speedups.json').read_text())['rows']
    cols = ['panel', 'group', 'bar', 'speedup', 'note', 'run', 'reference_run', 'reference_median_s',
            'variant_median_s', 'incremental', 'degraded_quality']
    with wandb.init(entity=entity, project=project, id=identity['id'], resume='allow', name=name, job_type='analysis',
                    group='Analysis', tags=['analysis', 'speedup'], config={'source': str(DATA.relative_to(ROOT))},
                    settings=wandb.Settings(x_disable_stats=True, disable_git=True, disable_code=True)) as run:
        table = wandb.Table(columns=cols)
        for r in rows:
            table.add_data(*[r.get(c) for c in cols])
        run.log({'speedup_images/speedups': wandb.Image(str(out/'speedups.png')), 'speedups': table})
        art = wandb.Artifact(name, type='analysis')
        for f in ('speedups.png', 'speedups.pdf', 'speedups.json'):
            art.add_file(str(out/f))
        art.add_file(str(Path(__file__).resolve()), name='plot_speedups.py')
        run.log_artifact(art)
        identity['url'] = run.url
    remote = wandb.Api().run(f"{entity}/{project}/{identity['id']}")
    files = {f.name for f in remote.files()}
    missing = [k for k in ('speedup_images/speedups', 'speedups') if remote.summary.get(k, {}).get('path') not in files]
    arts = [a for a in remote.logged_artifacts() if a.name.split(':')[0] == name]
    newest = max(arts, key=lambda a: int(a.version.lstrip('v'))) if arts else None
    digest = base64.b64encode(hashlib.md5((out/'speedups.json').read_bytes()).digest()).decode()
    ok = remote.state == 'finished' and not missing and newest and newest.manifest.entries['speedups.json'].digest == digest
    identity['status'] = 'verified' if ok else 'NOT verified'
    receipt.write_text(json.dumps(identity, indent=2) + '\n')
    print(identity['status'], identity['url'])
    if not ok:
        raise RuntimeError('W&B upload not verified')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--output', type=Path, default=ROOT/'output'/'speedups-20261028')
    ap.add_argument('--upload', action='store_true')
    args = ap.parse_args()
    runs = {r['run_dir']: r for r in json.loads(DATA.read_text())['runs']}
    rows = speedups(runs)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'speedups.json').write_text(json.dumps({'source': str(DATA.relative_to(ROOT)), 'rows': rows}, indent=2) + '\n')
    figure(rows, args.output)
    print('wrote', args.output)
    if args.upload:
        upload(args.output)


if __name__ == '__main__':
    main()
