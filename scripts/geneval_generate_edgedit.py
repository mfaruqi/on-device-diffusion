#!/usr/bin/env python3
"""Generate GenEval images with edge-dit.cpp, reusing a completed benchmark run's exact engine command.

  python scripts/geneval_generate_edgedit.py --template-run results/runs/<edge-dit run> --out-root <dir>

The template run's command.json is copied unchanged except --prompt_file, --output_dir, --warmup 0,
--repeat 1 and --seed 0: every engine flag (weights, precision, cache method, tiling, steps, guidance,
resolution) stays as benchmarked. ed-sample gives image n the seed (base + n); each prompt is written four
times, so prompt i's samples 0..3 use seeds 4i..4i+3 (edge-dit's own RNG; not comparable with PyTorch seeds,
identical across edge-dit configurations, so reference/variant pairs stay matched). Each image's engine
metadata (prompt, seed) is checked before it is placed in GenEval's layout (<run>/images/<NNNNN>/samples/).
"""
import argparse
import datetime as dt
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
SPLIT = ROOT/'configs'/'geneval'/'split-20261024.json'
META = ROOT/'configs'/'geneval'/'evaluation_metadata.jsonl'
UNSEAL = dt.date(2026, 11, 30)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replace(cmd, flag, value):
    i = cmd.index(flag)
    cmd[i + 1] = value


def main():
    from benchlib import write_json
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--template-run', type=Path, required=True)
    ap.add_argument('--subset', choices=['calibration_subset', 'calibration_rest', 'calibration', 'heldout'], default='calibration_subset')
    ap.add_argument('--out-root', type=Path, required=True)
    ap.add_argument('--limit', type=int, default=None)
    args = ap.parse_args()
    if args.subset == 'heldout' and dt.date.today() < UNSEAL:
        sys.exit(f'held-out prompts are sealed until {UNSEAL}')
    template = args.template_run
    assert json.loads((template/'status.json').read_text())['status'] == 'complete', 'template run must be complete'
    cfg = json.loads((template/'config.json').read_text())
    cmd = json.loads((template/'command.json').read_text())
    split = json.loads(SPLIT.read_text())
    assert split['metadata_sha256'] == sha(META)
    meta = META.read_text().splitlines()
    chosen = ([i for i in split['calibration'] if i not in set(split['calibration_subset'])]
              if args.subset == 'calibration_rest' else split[args.subset])
    indices = chosen[:args.limit] if args.limit else chosen
    per = len(split['image_seeds'])

    stamp = dt.datetime.now().strftime('%Y%m%d-%H%M%S')
    run = args.out_root/f"{cfg['id']}__geneval-{args.subset}{'-smoke' if args.limit else ''}__{stamp}"
    (run/'images').mkdir(parents=True)
    write_json(run/'config.json', cfg)
    status = {'status': 'running', 'started_utc': dt.datetime.utcnow().isoformat(timespec='seconds') + 'Z'}
    write_json(run/'status.json', status)
    try:
        prompts = [json.loads(meta[i])['prompt'] for i in indices for _ in range(per)]
        assert all('\n' not in p for p in prompts)
        (run/'prompts.txt').write_text('\n'.join(prompts) + '\n')
        engine_out = run/'engine'
        replace(cmd, '--prompt_file', str(run/'prompts.txt'))
        replace(cmd, '--output_dir', str(engine_out))
        replace(cmd, '--warmup', '0')
        replace(cmd, '--repeat', '1')
        replace(cmd, '--seed', '0')
        write_json(run/'command.json', cmd)
        with (run/'engine.log').open('w') as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=48*3600)
        rows = []
        for n, index in enumerate(indices):
            folder = run/'images'/f'{index:05d}'
            (folder/'samples').mkdir(parents=True)
            (folder/'metadata.jsonl').write_text(meta[index] + '\n')
            for s in range(per):
                k = n*per + s
                src = engine_out/'imgs'/f'img_{k:06d}.png'
                info = json.loads((engine_out/'imgs_metadata'/f'img_{k:06d}.json').read_text())
                assert info['prompt'] == json.loads(meta[index])['prompt'], f'image {k}: prompt mismatch'
                assert info['seed'] == k and info['image_index'] == k and not info['warmup'], f'image {k}: {info}'
                dst = folder/'samples'/f'{s:04d}.png'
                shutil.copy2(src, dst)
                rows.append({'index': index, 'sample': s, 'engine_seed': k, 'latency_s': info['latency_seconds'], 'png_sha256': sha(dst)})
        log_text = (run/'engine.log').read_text()
        write_json(run/'generations.json', {
            'engine': 'edge-dit.cpp', 'template_run': str(template), 'subset': args.subset, 'indices': indices,
            'seed_rule': 'engine seed = 4 * position + sample (ed-sample base 0 + image index)', 'rows': rows,
            'cache_lines': [l for l in log_text.splitlines() if ' skipped ' in l or ' reused ' in l][:5],
            'split_sha256': sha(SPLIT), 'template_command_sha256': sha(template/'command.json'),
            'binary_sha256': sha(cmd[0]), 'script_sha256': sha(__file__)})
        status.update(status='complete', images=len(rows))
    except BaseException as error:
        status.update(status='failed', error=repr(error))
        raise
    finally:
        status['finished_utc'] = dt.datetime.utcnow().isoformat(timespec='seconds') + 'Z'
        write_json(run/'status.json', status)
    print('done', run)


if __name__ == '__main__':
    main()
