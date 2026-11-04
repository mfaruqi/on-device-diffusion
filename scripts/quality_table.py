#!/usr/bin/env python3
"""Join end-to-end latency with image quality for the configurations in a selection file.

  python scripts/quality_table.py [--selection configs/quality/selection-20261102.json] [--output output/quality-table-20261102]

Per configuration: end-to-end latency (benchmark wall_ms median and range over the measured generations), peak
memory (the device's own kind, labelled), GenEval score and the four prompt-image preference means on the GenEval
calibration subset. Each variant is paired with its engine's reference by (prompt, seed): difference and 95%
bootstrap interval over prompts (10,000 draws, seed 0), the same resampling as scripts/geneval_compare.py.
Writes quality-table.{csv,md,json} and one speed-vs-quality figure per device.
"""
import argparse
import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GENEVAL = ROOT/'results'/'geneval'
RUNS = ROOT/'results'/'runs'
METRICS = [('geneval', 'GenEval'), ('clip', 'CLIP'), ('pickscore', 'PickScore'), ('hps', 'HPS v2.1'),
           ('imagereward', 'ImageReward')]
DRAWS = 10000


def geneval_rows(run):
    out = {}
    for line in (GENEVAL/run/'geneval-results.jsonl').read_text().splitlines():
        r = json.loads(line)
        m = re.search(r'/images/(\d+)/samples/(\d+)\.png$', r['filename'])
        out[(int(m.group(1)), int(m.group(2)))] = (r['tag'], float(bool(r['correct'])))
    return out


def reward_rows(run):
    path = GENEVAL/run/'reward-scores.jsonl'
    if not path.exists():
        return None
    return {(r['prompt_index'], r['seed_index']): r for r in map(json.loads, path.read_text().splitlines())}


def per_prompt(run):
    """{prompt: {'tag', 'geneval': [correct...], metric: [score...]}} for one image run."""
    ge, rw = geneval_rows(run), reward_rows(run)
    out = defaultdict(lambda: defaultdict(list))
    for (p, s), (tag, ok) in sorted(ge.items()):
        out[p]['tag'] = tag
        out[p]['geneval'].append(ok)
        if rw is not None:
            for k, _ in METRICS[1:]:
                out[p][k].append(rw[(p, s)][k])
    return out, rw is not None


def metric_value(prompts, keys, metric):
    if metric == 'geneval':  # mean over tasks of the fraction of correct images
        by = defaultdict(list)
        for p in keys:
            by[prompts[p]['tag']] += prompts[p]['geneval']
        return sum(sum(v)/len(v) for v in by.values())/len(by)
    vals = [x for p in keys for x in prompts[p][metric]]
    return sum(vals)/len(vals)


def paired_delta(ref, var, metric):
    keys = sorted(ref)
    assert keys == sorted(var) and all(len(ref[p]['geneval']) == len(var[p]['geneval']) for p in keys), 'prompt/seed sets differ'
    delta = metric_value(var, keys, metric) - metric_value(ref, keys, metric)
    rng = random.Random(0)
    draws = []
    for _ in range(DRAWS):
        sample = [rng.choice(keys) for _ in keys]
        draws.append(metric_value(var, sample, metric) - metric_value(ref, sample, metric))
    draws.sort()
    return delta, (draws[int(.025*DRAWS) - 1], draws[int(.975*DRAWS) - 1])


def bench(run):
    s = json.loads((RUNS/run/'summary.json').read_text())
    w = s['measured']['wall_ms']
    mem = (s['measured'].get('device_used_peak_gib') or {}) if isinstance(s['measured'].get('device_used_peak_gib'), dict) else {}
    if mem.get('median') is not None:
        memory = (mem['median'], 'GPU device-wide peak (NVML), median per generation')
    elif s.get('device_used_peak_gib') is not None:
        memory = (s['device_used_peak_gib'], 'GPU device-wide peak (NVML), whole capture')
    else:
        sysmem = s.get('system_memory') or s.get('monitor_window') or {}
        memory = (sysmem.get('ram_peak_gib'), 'whole-system RAM peak (tegrastats), whole run incl. load')
    return {'latency_s': w['median']/1000, 'latency_min_s': w['min']/1000, 'latency_max_s': w['max']/1000,
            'n': w['n'], 'memory_gib': memory[0], 'memory_kind': memory[1]}


def build(selection):
    rows = []
    for g in selection['groups']:
        ref_prompts, ref_has_rw = per_prompt(g['reference']['images'])
        ref_bench = bench(g['reference']['bench'])
        for entry in [g['reference']] + g['variants']:
            prompts, has_rw = per_prompt(entry['images'])
            b = bench(entry['bench'])
            row = {'device': g['device'], 'workload': g['workload'], 'engine': g['engine'], 'config': entry['label'],
                   'generated_on': g['generated_on'], 'images': entry['images'], 'bench': entry['bench'],
                   'latency_flag': entry.get('latency_flag', ''), **b,
                   'speedup': ref_bench['latency_s']/b['latency_s'], 'images_n': sum(len(p['geneval']) for p in prompts.values())}
            for k, _ in METRICS:
                if k != 'geneval' and not has_rw:
                    row[k] = None
                    continue
                row[k] = metric_value(prompts, sorted(prompts), k)
                if entry is not g['reference'] and (k == 'geneval' or ref_has_rw):
                    row[f'{k}_delta'], row[f'{k}_ci95'] = paired_delta(ref_prompts, prompts, k)
            rows.append(row)
    return rows


def fmt(v, digits=4):
    return '–' if v is None else f'{v:.{digits}f}'


def markdown(rows):
    head = ['Device', 'Workload', 'Engine', 'Configuration', 'E2E latency s (min–max)', 'Speedup', 'Peak memory GiB']
    head += [name for _, name in METRICS]
    lines = ['| ' + ' | '.join(head) + ' |', '|' + '---|'*len(head)]
    for r in rows:
        lat = f'{r["latency_s"]:.2f} ({r["latency_min_s"]:.2f}–{r["latency_max_s"]:.2f})' + (' †' if r['latency_flag'] else '')
        cells = [r['device'], r['workload'], r['engine'], r['config'], lat, f'{r["speedup"]:.2f}×',
                 fmt(r['memory_gib'], 2)]
        for k, _ in METRICS:
            d = 2 if k == 'pickscore' else 4
            v = fmt(r[k], d)
            if r.get(f'{k}_delta') is not None:
                lo, hi = r[f'{k}_ci95']
                v += f' ({r[f"{k}_delta"]:+.{d}f}; {lo:+.{d}f}…{hi:+.{d}f})'
            cells.append(v)
        lines.append('| ' + ' | '.join(cells) + ' |')
    notes = sorted({r['latency_flag'] for r in rows if r['latency_flag']})
    kinds = sorted({(r['device'], r['memory_kind']) for r in rows})
    lines += ['', 'Quality cells: value (difference from the engine reference; 95% bootstrap interval over prompts).']
    lines += [f'† {n}.' for n in notes]
    lines += [f'Memory on {d}: {k}.' for d, k in kinds]
    return '\n'.join(lines) + '\n'


def figure(rows, device, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sel = [r for r in rows if r['device'] == device]
    groups = list(dict.fromkeys((r['engine'], r['workload']) for r in sel))
    colors = ['#0072B2', '#E69F00', '#009E73', '#D55E00', '#CC79A7']
    markers = ['o', 's', '^', 'D', 'P']
    metrics = [(k, n) for k, n in METRICS if any(r.get(k) is not None for r in sel)]
    with plt.rc_context({'font.family': 'serif', 'font.serif': ['STIXGeneral', 'DejaVu Serif'], 'mathtext.fontset': 'stix',
                         'font.size': 10, 'xtick.direction': 'in', 'ytick.direction': 'in', 'pdf.fonttype': 42}):
        from matplotlib.ticker import FixedLocator, NullLocator
        xmax = max(r['speedup'] for r in sel)
        ticks = [t for t in ((1, 1.5, 2, 3) if xmax < 4 else (1, 2, 3, 5, 10, 20)) if t <= xmax*1.05]
        fig, axes = plt.subplots(2, 3, figsize=(12, 7.2))
        for ax, (k, name) in zip(axes.flat, metrics):
            ax.axhline(0, color='#555555', lw=.8)
            for gi, grp in enumerate(groups):
                pts = sorted((r for r in sel if (r['engine'], r['workload']) == grp and r['config'] != 'reference'
                              and r.get(f'{k}_delta') is not None), key=lambda r: r['speedup'])
                ax.plot([1], [0], markers[gi], ms=6, color=colors[gi])
                for j, r in enumerate(pts):
                    d, ci = r[f'{k}_delta'], r[f'{k}_ci95']
                    ax.errorbar(r['speedup'], d, yerr=[[d - ci[0]], [ci[1] - d]], fmt=markers[gi], ms=6, capsize=2,
                                lw=.9, color=colors[gi], mfc='white' if r['latency_flag'] else colors[gi], mec=colors[gi])
                    # alternate label side within a group so neighbours do not collide
                    ax.annotate(r['config'], (r['speedup'], d), xytext=(5, 6 if j % 2 == 0 else -11),
                                textcoords='offset points', fontsize=7.5, color='#333333')
            ax.set_xscale('log')
            ax.xaxis.set_major_locator(FixedLocator(ticks))
            ax.xaxis.set_minor_locator(NullLocator())
            ax.set_xticklabels([f'{t:g}×' for t in ticks])
            ax.set_xlim(.9, xmax*1.35)
            ax.set_title(name, fontweight='bold')
            ax.grid(lw=.4, alpha=.3)
        for ax in axes[1]:
            ax.set_xlabel('Speedup vs engine reference')
        for ax in axes[:, 0]:
            ax.set_ylabel('Difference from reference')
        handles = [plt.Line2D([], [], marker=markers[i], color=colors[i], ls='', ms=6, label=f'{e}\n{w}')
                   for i, (e, w) in enumerate(groups)]
        if any(r['latency_flag'] for r in sel):
            handles.append(plt.Line2D([], [], marker='o', color='#555555', mfc='white', ls='', ms=6,
                                      label='hollow: latency measured\nwith conditioning reuse'))
        leg = axes.flat[-1]
        leg.axis('off')
        leg.legend(handles=handles, loc='center left', frameon=False, fontsize=9, labelspacing=1.1,
                   title='Error bars: 95% bootstrap interval\nover prompts; references at (1×, 0)', title_fontsize=8.5)
        fig.suptitle(f'{device}: speedup vs image-quality change (GenEval calibration subset, 96 prompts × 4 seeds)',
                     fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, .96))
        stem = out/f'speed-quality-{device}'
        fig.savefig(stem.with_suffix('.png'), dpi=200)
        fig.savefig(stem.with_suffix('.pdf'))
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--selection', type=Path, default=ROOT/'configs'/'quality'/'selection-20261102.json')
    ap.add_argument('--output', type=Path, default=ROOT/'output'/'quality-table-20261102')
    ap.add_argument('--no-plots', action='store_true')
    args = ap.parse_args()
    selection = json.loads(args.selection.read_text())
    rows = build(selection)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'quality-table.json').write_text(json.dumps({'selection': str(args.selection.relative_to(ROOT)),
                                                              'subset': selection['subset'], 'rows': rows}, indent=2) + '\n')
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(args.output/'quality-table.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows({k: (json.dumps(v) if isinstance(v, (list, tuple)) else v) for k, v in r.items()} for r in rows)
    (args.output/'quality-table.md').write_text(markdown(rows))
    print(markdown(rows))
    if not args.no_plots:
        for device in dict.fromkeys(r['device'] for r in rows):
            figure(rows, device, args.output)


if __name__ == '__main__':
    main()
