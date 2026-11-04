#!/usr/bin/env python3
"""Paper-style (booktabs) quality tables from output/quality-table-20261102/quality-table.json, one per device.

  python scripts/quality_table_latex.py [--input output/quality-table-20261102/quality-table.json] [--upload]

Writes quality-booktabs-<device>.tex (LaTeX source, needs \\usepackage{booktabs}) and the same table rendered with
matplotlib as .png/.pdf. Columns: E2E latency, speedup, CLIP, PickScore, HPS v2.1, ImageReward, GenEval (×100), DPG.
Values are the means in quality-table.json; the best value of each metric within an engine/workload group is bold.
PickScore is the raw logit-scaled score (≈ 20–23); some papers divide it by 100. DPG is not measured yet ('–').
--upload logs the tables and images to one W&B analysis run (receipt: wandb-booktabs-upload.json next to the input).
"""
import argparse
import json
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COLS = [('clip', 'CLIP', 4, 1), ('pickscore', 'PickScore', 2, 1), ('hps', 'HPS', 4, 1),
        ('imagereward', 'ImageReward', 4, 1), ('geneval', 'GenEval', 2, 100), ('dpg', 'DPG', 2, 1)]
DEVICE_NAMES = {'a100-pcie-40gb': 'NVIDIA A100-PCIe-40GB', 'jetson-orin-nano': 'Jetson Orin Nano 8 GB'}


def groups(rows, device):
    out = {}
    for r in rows:
        if r['device'] == device:
            out.setdefault((r['engine'], r['workload']), []).append(r)
    return out


def cells(grp):
    """Per row: list of (text, bold) for latency, speedup and each metric column."""
    best = {k: max((r.get(k) for r in grp if r.get(k) is not None), default=None) for k, *_ in COLS}
    out = []
    for r in grp:
        c = [(f'{r["latency_s"]:.2f}' + ('$^\\dagger$' if r['latency_flag'] else ''), False),
             (f'{r["speedup"]:.2f}$\\times$', False)]
        for k, _, d, scale in COLS:
            v = r.get(k)
            c.append(('–', False) if v is None else (f'{v*scale:.{d}f}', len(grp) > 1 and v == best[k]))
        out.append(c)
    return out


def latex(rows, device):
    head = ['Configuration', 'Latency (s)', 'Speedup'] + [f'{n} $\\uparrow$' for _, n, *_ in COLS]
    lines = ['\\begin{table}[t]', '\\centering', '\\small', '\\begin{tabular}{l' + 'r'*(len(head) - 1) + '}',
             '\\toprule', ' & '.join(f'\\textbf{{{h}}}' for h in head) + ' \\\\', '\\midrule']
    for i, ((engine, workload), grp) in enumerate(groups(rows, device).items()):
        if i:
            lines.append('\\midrule')
        mark = '$^\\ddagger$' if grp[0]['generated_on'] != device else ''
        lines.append(f'\\multicolumn{{{len(head)}}}{{l}}{{\\textit{{{engine}, {workload}}}{mark}}} \\\\'.replace('²', '$^2$'))
        for r, c in zip(grp, cells(grp)):
            vals = [f'\\textbf{{{t}}}' if b else t for t, b in c]
            lines.append(f'\\quad {r["config"]} & ' + ' & '.join(vals) + ' \\\\')
    lines += ['\\bottomrule', '\\end{tabular}',
              f'\\caption{{{DEVICE_NAMES.get(device, device)}: image quality on the GenEval calibration subset '
              '(96 prompts $\\times$ 4 seeds) and end-to-end latency (median). GenEval $\\times 100$; PickScore is '
              'the raw logit-scaled score; best value per group in bold; DPG not measured.'
              + (' $^\\dagger$Latency measured with conditioning reuse.' if any(r['latency_flag'] for r in rows if r['device'] == device) else '')
              + (' $^\\ddagger$Images generated on the A100 from the identical weights and options; latency measured on this device. Nunchaku: community-wheel kernels instead of the Jetson sm\\_87 build.'
                 if any(r['generated_on'] != device for r in rows if r['device'] == device) else '')
              + '}', '\\end{table}']
    return '\n'.join(lines) + '\n'


def render(rows, device, stem):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    gs = groups(rows, device)
    head = ['Configuration', 'Latency (s)', 'Speedup'] + [f'{n} ↑' for _, n, *_ in COLS]
    # right edge of each numeric column; the configuration column is left-aligned at 0
    edges = [4.3, 5.7] + [7.3 + 1.6*i for i in range(len(COLS))]
    right = edges[-1]
    nlines = 1 + sum(1 + len(g) for g in gs.values())
    h = 0.27
    flagged = any(r['latency_flag'] for g in gs.values() for r in g)
    with plt.rc_context({'font.family': 'serif', 'font.serif': ['STIXGeneral', 'DejaVu Serif'],
                         'mathtext.fontset': 'stix', 'pdf.fonttype': 42}):
        fig = plt.figure(figsize=(right*0.98, (nlines + 3.4)*h))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(-0.15, right + 0.15)
        ax.set_ylim(-(nlines + 2.7), 1.0)
        ax.axis('off')

        def put(y, j, text, bold=False, italic=False):
            ha = 'left' if j == 0 else 'right'
            x = 0.0 if j == 0 else edges[j - 1]
            text = text.replace('$\\times$', '×').replace('$^\\dagger$', '†').replace('$^\\ddagger$', '‡')
            ax.text(x, y, text if j < 3 else text.replace('-', '−'), ha=ha, va='center',
                    fontsize=11, fontweight='bold' if bold else 'normal', fontstyle='italic' if italic else 'normal')

        rule = lambda y, lw: ax.plot([-0.15, right + 0.15], [y, y], color='black', lw=lw, solid_capstyle='butt')
        y = 0.0
        rule(0.55, 1.4)
        for j, t in enumerate(head):
            put(y, j, t, bold=True)
        rule(-0.5, 0.7)
        y = -1.0
        for i, ((engine, workload), grp) in enumerate(gs.items()):
            if i:
                rule(y + 0.5, 0.5)
            ax.text(0.0, y, f'{engine}, {workload}' + (' ‡' if grp[0]['generated_on'] != device else ''), ha='left', va='center', fontsize=11, fontstyle='italic')
            y -= 1
            for r, c in zip(grp, cells(grp)):
                put(y, 0, '    ' + r['config'])
                for j, (t, b) in enumerate(c, start=1):
                    put(y, j, t, bold=b)
                y -= 1
        rule(y + 0.5, 1.4)
        note = (f'{DEVICE_NAMES.get(device, device)}. GenEval calibration subset (96 prompts × 4 seeds); latency = '
                'end-to-end median. GenEval × 100; PickScore raw (logit-scaled); bold = best in group; DPG not measured.')
        ax.text(0.0, y - 0.35, note, ha='left', va='center', fontsize=8.5, color='#333333')
        foot = y - 1.15
        if any(g[0]['generated_on'] != device for g in gs.values()):
            ax.text(0.0, foot, '‡ Images for quality scoring generated on the A100 from the identical weights and options; '
                    'latency measured on this device. Nunchaku: community-wheel kernels instead of the Jetson sm_87 build.', ha='left', va='center', fontsize=8.5, color='#333333')
            foot -= 0.8
        if flagged:
            ax.text(0.0, foot, '† Latency measured with conditioning reuse (no run without it); excludes the '
                    'text encoder and its weight swap.', ha='left', va='center', fontsize=8.5, color='#333333')
        fig.savefig(stem.with_suffix('.png'), dpi=220, bbox_inches='tight', pad_inches=0.08, facecolor='white')
        fig.savefig(stem.with_suffix('.pdf'), bbox_inches='tight', pad_inches=0.08)
        plt.close(fig)


def upload(src, rows, outs, name='klein-quality-booktabs', entity='mfaruqi-purdue-university', project='on-device-diffusion'):
    import wandb
    receipt = src.parent/'wandb-booktabs-upload.json'
    ident = json.loads(receipt.read_text()) if receipt.exists() else {'id': uuid.uuid4().hex[:16]}
    keys = []
    with wandb.init(entity=entity, project=project, id=ident['id'], resume='allow', name=name, job_type='analysis',
                    group='Analysis', tags=['analysis', 'quality', 'geneval', 'preference-metrics'],
                    config={'source': str(src.relative_to(ROOT)), 'selection': 'configs/quality/selection-20261102.json',
                            'subset': 'GenEval calibration_subset (96 prompts x 4 seeds)',
                            'method': 'wiki/methods/baseline-metrics.md#prompt-image-preference-scores'},
                    settings=wandb.Settings(x_disable_stats=True, disable_git=True, disable_code=True)) as run:
        cols = ['device', 'workload', 'engine', 'config', 'latency_s', 'speedup', 'latency_flag',
                'clip', 'pickscore', 'hps', 'imagereward', 'geneval', 'images', 'bench']
        t = wandb.Table(columns=cols + ['dpg'])
        for r in rows:
            t.add_data(*[r.get(c) for c in cols], None)
        log = {'quality/table': t}
        for device, stem in outs.items():
            log[f'quality/booktabs/{device}'] = wandb.Image(str(stem.with_suffix('.png')))
        keys = list(log)
        run.log(log)
        art = wandb.Artifact(name, type='analysis')
        art.add_file(str(src), name=src.name)
        for stem in outs.values():
            for ext in ('.tex', '.pdf', '.png'):
                art.add_file(str(stem.with_suffix(ext)), name=stem.with_suffix(ext).name)
        art.add_file(str(Path(__file__).resolve()), name=Path(__file__).name)
        run.log_artifact(art)
        run.summary.update({'row_count': len(rows)})
        ident['url'] = run.url
    remote = wandb.Api().run(f'{entity}/{project}/{ident["id"]}')
    files = {f.name for f in remote.files()}
    ok = remote.state == 'finished' and remote.summary.get('row_count') == len(rows) and \
        all(k in remote.summary and remote.summary[k].get('path') in files for k in keys)
    ident['status'] = 'verified' if ok else 'NOT verified'
    receipt.write_text(json.dumps(ident, indent=2) + '\n')
    print(ident['status'], ident['url'])
    if not ok:
        raise RuntimeError('W&B upload not verified')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--input', type=Path, default=ROOT/'output'/'quality-table-20261102'/'quality-table.json')
    ap.add_argument('--upload', action='store_true')
    args = ap.parse_args()
    rows = json.loads(args.input.read_text())['rows']
    for r in rows:
        r['images'] = r['images_n']
    outs = {}
    for device in dict.fromkeys(r['device'] for r in rows):
        stem = args.input.parent/f'quality-booktabs-{device}'
        stem.with_suffix('.tex').write_text(latex(rows, device))
        render(rows, device, stem)
        outs[device] = stem
        print(stem.with_suffix('.png'))
    if args.upload:
        upload(args.input, rows, outs)


if __name__ == '__main__':
    main()
