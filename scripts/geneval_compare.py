#!/usr/bin/env python3
"""Paired GenEval comparison of variants against a reference run (same prompts and seeds).

  python scripts/geneval_compare.py --reference <run> --variant <run> [--variant <run> ...]

GenEval score = mean over tasks of the fraction of correct images (official summary_scores.py definition).
Paired by (prompt index, seed). Reports per-task and overall differences, discordant pairs (reference-only
correct vs variant-only correct) with an exact two-sided McNemar (binomial) p-value, and a 95% bootstrap
interval for the overall difference (resampling prompts, 10,000 draws, seed 0).
"""
import argparse
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(runs):
    """One run, or several comma-separated runs covering disjoint prompts of the same configuration."""
    out = {}
    lines = [l for run in runs.split(',') for l in (ROOT/'results'/'geneval'/run/'geneval-results.jsonl').read_text().splitlines()]
    for line in lines:
        r = json.loads(line)
        m = re.search(r'/images/(\d+)/samples/(\d+)\.png$', r['filename'])
        key = (int(m.group(1)), int(m.group(2)))
        assert key not in out, f'duplicate prompt/seed {key} across {runs}'
        out[key] = (r['tag'], bool(r['correct']))
    return out


def score(rows, keys):
    by = defaultdict(list)
    for k in keys:
        by[rows[k][0]].append(rows[k][1])
    tasks = {t: sum(v)/len(v) for t, v in by.items()}
    return sum(tasks.values())/len(tasks), tasks


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(b, c) + 1))/2**n
    return min(1.0, 2*tail)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--reference', required=True)
    ap.add_argument('--variant', action='append', required=True)
    ap.add_argument('--json', type=Path)
    args = ap.parse_args()
    ref = load(args.reference)
    report = {'reference': args.reference, 'variants': []}
    ref_score, ref_tasks = score(ref, ref.keys())
    print(f'reference {args.reference}: GenEval {ref_score:.4f}  ' + '  '.join(f'{t} {v:.3f}' for t, v in sorted(ref_tasks.items())))
    for run in args.variant:
        var = load(run)
        keys = sorted(set(ref) & set(var))
        assert len(keys) == len(ref) == len(var), f'{run}: prompt/seed sets differ'
        vs, vt = score(var, keys)
        b = sum(1 for k in keys if ref[k][1] and not var[k][1])
        c = sum(1 for k in keys if var[k][1] and not ref[k][1])
        prompts = sorted({k[0] for k in keys})
        rng = random.Random(0)
        diffs = []
        for _ in range(10000):
            sample = [k for p in (rng.choice(prompts) for _ in prompts) for k in keys if k[0] == p]
            diffs.append(score(var, sample)[0] - score(ref, sample)[0])
        diffs.sort()
        ci = (diffs[249], diffs[9749])
        row = {'variant': run, 'geneval': vs, 'delta': vs - ref_score, 'delta_ci95': ci,
               'tasks': vt, 'task_delta': {t: vt[t] - ref_tasks[t] for t in vt},
               'ref_only_correct': b, 'variant_only_correct': c, 'mcnemar_p': mcnemar(b, c), 'pairs': len(keys)}
        report['variants'].append(row)
        print(f'{run}: GenEval {vs:.4f} (Δ {vs-ref_score:+.4f}, 95% CI {ci[0]:+.4f}..{ci[1]:+.4f}); '
              f'discordant ref-only {b} / variant-only {c}, McNemar p={row["mcnemar_p"]:.3f}')
        print('   ' + '  '.join(f'{t} {d:+.3f}' for t, d in sorted(row['task_delta'].items())))
    report['reference_geneval'], report['reference_tasks'] = ref_score, ref_tasks
    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
