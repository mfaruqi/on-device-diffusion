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


RECORDS = {'a100-flux-klein-torch28': 'experiments/compare-geneval-a100-distilled-20261024.md',
           'a100-flux-klein-compile': 'experiments/compare-geneval-a100-distilled-20261024.md',
           'a100-flux-klein-base-torch28': 'experiments/compare-geneval-a100-base-20261025.md',
           'a100-geneval-torch': 'experiments/compare-geneval-jetson-matched-20261026.md'}


def benchmark_id(config_id):
    """GenEval config id -> benchmark config id (Jetson-matched copies point to their Jetson config)."""
    path = ROOT/'configs'/'geneval'/f'{config_id}.json'
    if path.exists():
        src = json.loads(path.read_text())['derived_from']['config']
        return json.loads((ROOT/src).read_text())['id'], 'A100 GenEval run of the identical Jetson weights and options (residency/reuse off)'
    return config_id, 'GenEval run of this configuration'


def attachments():
    """Per benchmark config id: GenEval score on the largest prompt set, and its paired comparison if any."""
    runs = [r for r in run_rows()]
    by_cfg = defaultdict(list)
    for r in runs:
        by_cfg[r['config_id']].append(r)
    comps = comparison_rows()
    out = {}
    for cid, rs in by_cfg.items():
        # combine disjoint subsets of the same configuration (calibration_subset + calibration_rest = 277 prompts)
        tags = defaultdict(list)
        for r in rs:
            for line in (GENEVAL/r['run']/'geneval-results.jsonl').read_text().splitlines():
                x = json.loads(line)
                tags[x['tag']].append(bool(x['correct']))
        tasks = {t: sum(v)/len(v) for t, v in tags.items()}
        images = sum(len(v) for v in tags.values())
        names = sorted(r['run'] for r in rs)
        comp = [c for c in comps if set(c['variant'].split(',')) == set(names)] or \
               [c for c in comps if set(c['variant'].split(',')) & set(names)]
        comp = max(comp, key=lambda c: c['pairs']) if comp else None
        bid, basis = benchmark_id(cid)
        rec = next((v for k, v in RECORDS.items() if cid.startswith(k)), None)
        out[bid] = {'quality/geneval': sum(tasks.values())/len(tasks), 'quality/geneval_images': images,
                    'quality/geneval_prompts': images // 4, 'quality/geneval_basis': basis,
                    'quality/geneval_runs': ','.join(names), 'quality/geneval_record': rec,
                    **{f'quality/geneval_task/{t}': tasks.get(t) for t in TASKS},
                    **({'quality/geneval_delta': comp['delta'], 'quality/geneval_delta_ci95_low': comp['ci95_low'],
                        'quality/geneval_delta_ci95_high': comp['ci95_high'], 'quality/geneval_mcnemar_p': comp['mcnemar_p'],
                        'quality/geneval_delta_pairs': comp['pairs'],
                        'quality/geneval_reference_runs': comp['reference']} if comp else {})}
    return out


def attach(api, entity, project, dry):
    att = attachments()
    diag = {r['variant_run']: r for r in diag_rows()}
    changed = []
    for run in api.runs(f'{entity}/{project}'):
        bid = run.id.split('__')[0]
        fields = dict(att.get(bid, {}))
        if run.id in diag:
            d = diag[run.id]
            fields.update({'quality/lpips_alex_v0_1': d['lpips_alex_v0_1'], 'quality/psnr_db': d['psnr_db'],
                           'quality/pixels_identical_to_reference': d['pixels_identical'],
                           'quality/paired_reference_run': d['reference_run']})
        if not fields:
            continue
        changed.append((run.id, fields.get('quality/geneval'), fields.get('quality/lpips_alex_v0_1')))
        if not dry:
            run.summary.update(fields)
    for c in changed:
        print('  ', c)
    print(f'{len(changed)} benchmark runs {"would be" if dry else ""} updated')
    if not dry:  # verify
        for rid, g, l in changed:
            r = api.run(f'{entity}/{project}/{rid}')
            assert (g is None or abs(r.summary.get('quality/geneval') - g) < 1e-12) and \
                   (l is None or abs(r.summary.get('quality/lpips_alex_v0_1') - l) < 1e-12), rid
        print('verified')


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
    ap.add_argument('--attach', action='store_true', help='also add quality/* summary fields to the matching benchmark runs')
    args = ap.parse_args()
    runs, comps, diags = run_rows(), comparison_rows(), diag_rows()
    print(f'{len(runs)} GenEval runs, {len(comps)} comparisons, {len(diags)} paired diagnostics')
    if args.attach:
        import wandb
        attach(wandb.Api(timeout=60), args.entity, args.project, args.dry_run)
        return
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
