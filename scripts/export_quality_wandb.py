#!/usr/bin/env python3
"""Upload validated image-quality results to W&B as one analysis run (no benchmark run is modified).

  python scripts/export_quality_wandb.py [--name klein-quality] [--dry-run]

Tables:
- geneval_runs: one row per scored GenEval image run (results/geneval/<run>/geneval-results.jsonl), skipping runs
  marked INVALID.json; GenEval score and per-task accuracy recomputed from the per-image results.
- geneval_comparisons: one row per variant in results/geneval/compare-*.json (paired against its reference).
- paired_image_diagnostics: PSNR / LPIPS per variant (output/quality-diagnostics-20261024/quality-diagnostics.json).
The evidence artifact carries the JSON inputs and this script. Re-running resumes the same run id.
"""
import argparse
import hashlib
import json
import uuid
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GENEVAL = ROOT/'results'/'geneval'
DIAG = ROOT/'output'/'quality-diagnostics-20261024'/'quality-diagnostics.json'
TASKS = ['single_object', 'two_object', 'counting', 'colors', 'position', 'color_attr']


def run_rows():
    rows = []
    for d in sorted(p for p in GENEVAL.iterdir() if p.is_dir() and (p/'geneval-results.jsonl').exists()):
        if (d/'INVALID.json').exists():
            continue
        by = defaultdict(list)
        for line in (d/'geneval-results.jsonl').read_text().splitlines():
            r = json.loads(line)
            by[r['tag']].append(bool(r['correct']))
        tasks = {t: sum(v)/len(v) for t, v in by.items()}
        config_id, subset, _ = d.name.split('__')
        rows.append({'run': d.name, 'config_id': config_id, 'subset': subset.removeprefix('geneval-'),
                     'images': sum(len(v) for v in by.values()), 'geneval': sum(tasks.values())/len(tasks),
                     **{f'task/{t}': tasks.get(t) for t in TASKS}})
    return rows


def comparison_rows():
    rows = []
    for f in sorted(GENEVAL.glob('compare-*.json')):
        rep = json.loads(f.read_text())
        for v in rep['variants']:
            rows.append({'comparison': f.stem, 'reference': rep['reference'], 'variant': v['variant'],
                         'reference_geneval': rep['reference_geneval'], 'variant_geneval': v['geneval'],
                         'delta': v['delta'], 'ci95_low': v['delta_ci95'][0], 'ci95_high': v['delta_ci95'][1],
                         'ref_only_correct': v['ref_only_correct'], 'variant_only_correct': v['variant_only_correct'],
                         'mcnemar_p': v['mcnemar_p'], 'pairs': v['pairs'],
                         **{f'task_delta/{t}': v['task_delta'].get(t) for t in TASKS}})
    return rows


def diag_rows():
    if not DIAG.exists():
        return []
    rows = []
    for p in json.loads(DIAG.read_text())['pairs']:
        rows.append({'group': p['group'], 'reference_run': p['reference_run'], 'variant_run': p['variant_run'],
                     'psnr_db': None if isinstance(p['psnr_db'], str) else p['psnr_db'],
                     'pixels_identical': p['pixels_identical'], 'lpips_alex_v0_1': p['lpips_alex_v0_1'],
                     'mean_abs_rgb_8bit': p['mean_abs_rgb_8bit']})
    return rows


def table(wandb, rows):
    cols = list(rows[0]) if rows else ['empty']
    t = wandb.Table(columns=cols)
    for r in rows:
        t.add_data(*[r.get(c) for c in cols])
    return t


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--name', default='klein-quality')
    ap.add_argument('--entity', default='mfaruqi-purdue-university')
    ap.add_argument('--project', default='on-device-diffusion')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    runs, comps, diags = run_rows(), comparison_rows(), diag_rows()
    print(f'{len(runs)} GenEval runs, {len(comps)} comparisons, {len(diags)} paired diagnostics')
    if args.dry_run:
        for r in comps:
            print(f"  {r['variant'].split('__')[0]:62s} Δ {r['delta']:+.4f} [{r['ci95_low']:+.4f}, {r['ci95_high']:+.4f}] p={r['mcnemar_p']:.3f}")
        return
    import wandb
    receipt = GENEVAL/'wandb-quality-upload.json'
    ident = json.loads(receipt.read_text()) if receipt.exists() else {'id': uuid.uuid4().hex[:16]}
    with wandb.init(entity=args.entity, project=args.project, id=ident['id'], resume='allow', name=args.name,
                    job_type='analysis', group='Analysis', tags=['analysis', 'quality', 'geneval', 'lpips'],
                    config={'geneval_split': 'configs/geneval/split-20261024.json',
                            'method': 'wiki/methods/baseline-metrics.md#geneval-formal-image-quality-metric'},
                    settings=wandb.Settings(x_disable_stats=True, disable_git=True, disable_code=True)) as run:
        run.log({'quality/geneval_runs': table(wandb, runs), 'quality/geneval_comparisons': table(wandb, comps),
                 'quality/paired_image_diagnostics': table(wandb, diags)})
        art = wandb.Artifact(args.name, type='analysis')
        for f in sorted(GENEVAL.glob('compare-*.json')):
            art.add_file(str(f), name=f'comparisons/{f.name}')
        if DIAG.exists():
            art.add_file(str(DIAG), name='quality-diagnostics.json')
        art.add_file(str(Path(__file__).resolve()), name='export_quality_wandb.py')
        run.log_artifact(art)
        run.summary.update({'geneval_run_count': len(runs), 'comparison_count': len(comps), 'diagnostic_count': len(diags)})
        ident['url'] = run.url
    remote = wandb.Api().run(f"{args.entity}/{args.project}/{ident['id']}")
    files = {f.name for f in remote.files()}
    keys = ['quality/geneval_runs', 'quality/geneval_comparisons', 'quality/paired_image_diagnostics']
    ok = remote.state == 'finished' and all(k in remote.summary and remote.summary[k].get('path') in files for k in keys) \
        and remote.summary.get('comparison_count') == len(comps)
    ident['status'] = 'verified' if ok else 'NOT verified'
    receipt.write_text(json.dumps(ident, indent=2) + '\n')
    print(ident['status'], ident['url'])
    if not ok:
        raise RuntimeError('W&B quality upload not verified')


if __name__ == '__main__':
    main()
