#!/usr/bin/env python3
"""Generate GenEval images for one PyTorch configuration, using the portable runner's own loading path.

  HF_HUB_OFFLINE=1 python scripts/geneval_generate.py --config configs/<id>.json --subset calibration_subset \
      --out-root results/geneval

The configuration's pipeline, quantization, VAE tiling, compiler and step cache are applied exactly as in
scripts/run_flux_portable.py; only the prompt and seed change per image. Conditioning reuse and stage
residency are not used (each prompt is new; residency only moves weights). Prompts come from the frozen
split configs/geneval/split-20261024.json; held-out prompts are refused before 2026-11-30.
Output follows GenEval's layout: <run>/images/<NNNNN>/metadata.jsonl and samples/<SSSS>.png.
Timing here is per image wall time for bookkeeping only; it is not a benchmark measurement.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
SPLIT = ROOT/'configs'/'geneval'/'split-20261024.json'
META = ROOT/'configs'/'geneval'/'evaluation_metadata.jsonl'
UNSEAL = dt.date(2026, 11, 30)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--subset', choices=['calibration_subset', 'calibration_rest', 'calibration', 'heldout'], default='calibration_subset',
                    help='calibration_rest = calibration half minus the 96-prompt subset')
    ap.add_argument('--out-root', type=Path, default=ROOT/'results'/'geneval')
    ap.add_argument('--limit', type=int, default=None, help='first N prompts only (smoke)')
    args = ap.parse_args()
    if args.subset == 'heldout' and dt.date.today() < UNSEAL:
        sys.exit(f'held-out prompts are sealed until {UNSEAL}')
    assert os.environ.get('HF_HUB_OFFLINE') == '1', 'run offline (HF_HUB_OFFLINE=1)'

    import torch
    from benchlib import write_json
    from run_flux_portable import (apply_compiler, apply_step_cache, apply_vae_tiling, check_config, load_pipeline,
                                   restrict_attention)
    from torch_backends import make_device

    cfg = json.loads(args.config.read_text())
    check_config(cfg)
    assert cfg['backend'] == 'cuda', 'GenEval generation runs on the A100 (cuda backend)'
    split = json.loads(SPLIT.read_text())
    assert split['metadata_sha256'] == sha(META), 'GenEval metadata changed since the split was frozen'
    meta = META.read_text().splitlines()
    if args.subset == 'calibration_rest':
        chosen = [i for i in split['calibration'] if i not in set(split['calibration_subset'])]
    else:
        chosen = split[args.subset]
    indices = chosen[:args.limit] if args.limit else chosen
    seeds = split['image_seeds']

    stamp = dt.datetime.now().strftime('%Y%m%d-%H%M%S')
    run_dir = args.out_root/f"{cfg['id']}__geneval-{args.subset}{'-smoke' if args.limit else ''}__{stamp}"
    images = run_dir/'images'
    images.mkdir(parents=True)
    write_json(run_dir/'config.json', cfg)
    status = {'status': 'running', 'started_utc': dt.datetime.utcnow().isoformat(timespec='seconds') + 'Z'}
    write_json(run_dir/'status.json', status)
    try:
        device = make_device(cfg, run_dir)
        pipe, load, _ = load_pipeline(cfg, device, run_dir)
        opt = cfg['optimizations']
        # Every option must be applied here or refused: an option silently ignored would mislabel the images.
        applied = {'quantization', 'vae_tiling', 'compiler', 'step_cache', 'attention_backend'}
        unknown = {k for k, v in opt.items() if v not in (None, False, {}) and k not in applied} - {'stage_residency', 'conditioning_reuse'}
        assert not unknown, f'options not applied by the GenEval generator: {sorted(unknown)}'
        assert not opt.get('stage_residency') and not opt.get('conditioning_reuse'), 'residency/reuse are not used for GenEval'
        tiled = apply_vae_tiling(pipe, opt.get('vae_tiling', False))
        compiler = apply_compiler(pipe, opt.get('compiler'))
        apply_step_cache(pipe.transformer, opt.get('step_cache'))
        w = cfg['workload']
        rows = []
        for n, index in enumerate(indices):
            item = json.loads(meta[index])
            folder = images/f'{index:05d}'
            (folder/'samples').mkdir(parents=True)
            (folder/'metadata.jsonl').write_text(meta[index] + '\n')
            for seed in seeds:
                generator = torch.Generator(device='cuda').manual_seed(seed)

                def generate_once():
                    return pipe(prompt=item['prompt'], height=w['height'], width=w['width'],
                                num_inference_steps=w['num_inference_steps'], guidance_scale=w['guidance_scale'],
                                max_sequence_length=w['max_sequence_length'], generator=generator, output_type='pil').images[0]

                generate, sdpa_calls = restrict_attention(generate_once, opt.get('attention_backend'))
                torch.cuda.synchronize()
                t = time.perf_counter()
                image = generate()
                torch.cuda.synchronize()
                if opt.get('attention_backend') is not None:
                    assert sdpa_calls and sdpa_calls[-1] > 0, 'attention backend restriction not exercised'
                ms = (time.perf_counter() - t)*1000
                path = folder/'samples'/f'{seed:04d}.png'
                image.save(path)
                rows.append({'index': index, 'tag': item['tag'], 'seed': seed, 'wall_ms': ms,
                             'png_sha256': sha(path), 'rgb_sha256': hashlib.sha256(image.convert('RGB').tobytes()).hexdigest()})
            print(f'{n+1}/{len(indices)} [{item["tag"]}] {item["prompt"]}: {ms:.0f} ms (last seed)', flush=True)
        if opt.get('vae_tiling'):
            assert len(tiled) == len(rows), f'VAE tiling requested; tiled decode ran {len(tiled)}/{len(rows)} times'
        write_json(run_dir/'generations.json', {'subset': args.subset, 'applied_options': {k: opt.get(k) for k in sorted(applied)},
                                                'tiled_decodes': len(tiled), 'indices': indices, 'seeds': seeds, 'rows': rows,
                                                'compiler': compiler, 'load': {k: load[k] for k in ('load_total_s',)},
                                                'split_sha256': sha(SPLIT), 'config_sha256': sha(args.config),
                                                'source_sha256': {p: sha(ROOT/'scripts'/p) for p in
                                                                  ('geneval_generate.py', 'run_flux_portable.py', 'torch_backends.py')},
                                                'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0)})
        status.update(status='complete', images=len(rows))
    except BaseException as error:
        status.update(status='failed', error=repr(error))
        raise
    finally:
        status['finished_utc'] = dt.datetime.utcnow().isoformat(timespec='seconds') + 'Z'
        write_json(run_dir/'status.json', status)
    print('done', run_dir)


if __name__ == '__main__':
    main()
