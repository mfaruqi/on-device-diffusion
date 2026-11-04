#!/usr/bin/env python3
"""Paired saved-image diagnostics (PSNR, LPIPS) for configurations that change pixels.

Definitions: wiki/methods/baseline-metrics.md, "Paired saved-image diagnostics". Each variant is
compared with the reference configuration of the same engine on the same device, same prompt and
seed. Decode full-resolution RGB, no crop or resize. PSNR = 10 log10(255^2 / MSE) over all 8-bit RGB
values; exact pixel equality is reported separately (no infinity in JSON). LPIPS: calibrated AlexNet
v0.1, RGB scaled to [-1, 1], float32 CPU evaluation; a same-image distance check must be ~0.
These are diagnostics of how far an image moved, not prompt alignment or formal quality (GenEval).

  python scripts/quality_diagnostics.py [--write-run-files] [--output output/quality-diagnostics-20261024]

Writes quality-diagnostics.json into each variant run directory (only with --write-run-files, and
never over an existing file) and a summary table to --output.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT/'results'/'runs'

# (group, reference run, [variant runs]); runs are full protocols with saved images.
PAIRS = [
    ('Jetson sd.cpp distilled', 'jetson-flux-klein-003__repeat__20261014-163411',
     ['jetson-flux-klein-010__baseline__20261014-183501', 'jetson-flux-klein-011__baseline__20261014-185639']),
    ('Jetson edge-dit distilled', 'jetson-edgedit-flux-klein-006__baseline__20261014-170149',
     ['jetson-edgedit-flux-klein-011__baseline__20261015-133915', 'jetson-edgedit-flux-klein-013__baseline__20261015-135118',
      'jetson-edgedit-flux-klein-014__baseline__20261015-135841']),
    ('Jetson edge-dit Base', 'jetson-edgedit-flux-klein-base-001__baseline__20261015-193452',
     [f'jetson-edgedit-flux-klein-base-00{n}__baseline__{s}' for n, s in
      [(2, '20261015-204021'), (3, '20261015-213038'), (4, '20261015-222249'), (5, '20261015-231521'),
       (6, '20261016-000110'), (7, '20261016-005119')]]),
    ('Jetson PyTorch distilled', 'jetson-torch-flux-klein-001__baseline__20261014-171410',
     ['jetson-torch-flux-klein-002__baseline__20261014-174108', 'jetson-torch-flux-klein-003__baseline__20261014-180843',
      'jetson-torch-flux-klein-004__baseline__20261015-141028', 'jetson-torch-flux-klein-006__baseline__20261015-142456',
      'jetson-torch-flux-klein-007__baseline__20261015-145234']),
    ('Jetson PyTorch Base', 'jetson-torch-flux-klein-base-001__baseline__20261016-050136',
     ['jetson-torch-flux-klein-base-002__baseline__20261016-015207', 'jetson-torch-flux-klein-base-004__baseline__20261016-033850']),
    ('A100 PyTorch torch 2.8 distilled', 'a100-flux-klein-torch28-001__baseline__20261015-151548',
     ['a100-flux-klein-compile-001__baseline__20261015-151740', 'a100-flux-klein-torch28-002__baseline__20261015-151840',
      'a100-flux-klein-torch28-004__baseline__20261015-151949']),
    ('A100 PyTorch torch 2.8 Base', 'a100-flux-klein-base-torch28-001__baseline__20261015-155338',
     ['a100-flux-klein-base-torch28-002__baseline__20261015-160117', 'a100-flux-klein-base-torch28-003__baseline__20261015-160821',
      'a100-flux-klein-base-torch28-004__baseline__20261015-161253']),
    # The 001 reference PNG is not retained; this zero-reuse screen's image has the reference's recorded RGB hash.
    ('A100 edge-dit distilled (ref. image via 002 screen)', 'a100-edgedit-flux-klein-002__attempt__20261015-152019',
     ['a100-edgedit-flux-klein-007__baseline__20261015-152117']),
    ('A100 edge-dit Base (Q4)', 'a100-edgedit-flux-klein-base-001__baseline__20261015-165052',
     [f'a100-edgedit-flux-klein-base-00{n}__baseline__{s}' for n, s in
      [(2, '20261015-170356'), (3, '20261015-171344'), (4, '20261015-172354'), (5, '20261015-173406'),
       (6, '20261015-174257'), (7, '20261015-175238'), (8, '20261015-175727')]]),
]


def image_of(run):
    d = RUNS/run
    assert json.loads((d/'status.json').read_text())['status'] == 'complete', run
    for rel in ('images/measured-0.png', 'output.png'):
        if (d/rel).exists():
            return rel
    raise FileNotFoundError(f'{run}: no saved image')


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


class Lpips:
    def __init__(self):
        import lpips
        import torch
        torch.set_num_threads(2)
        self.torch = torch
        self.net = lpips.LPIPS(net='alex', version='0.1', verbose=False).eval()
        hub = Path(torch.hub.get_dir())/'checkpoints'
        self.weights = {p.name: sha(p.read_bytes()) for p in
                        [*hub.glob('alexnet-owt-*.pth'), Path(lpips.__file__).parent/'weights'/'v0.1'/'alex.pth'] if p.exists()}

    def __call__(self, a, b):
        t = lambda x: self.torch.from_numpy(x).permute(2, 0, 1)[None].float()/127.5 - 1
        with self.torch.inference_mode():
            return float(self.net(t(a), t(b)))


def compare(metric, reference, variant):
    import numpy as np
    from PIL import Image
    ref_rel, var_rel = image_of(reference), image_of(variant)
    blobs = {k: (RUNS/r/rel).read_bytes() for k, r, rel in (('reference', reference, ref_rel), ('variant', variant, var_rel))}
    arrays = {k: np.asarray(Image.open(RUNS/r/rel).convert('RGB')) for k, r, rel in
              (('reference', reference, ref_rel), ('variant', variant, var_rel))}
    a, b = arrays['reference'], arrays['variant']
    assert a.shape == b.shape, f'{variant}: image shapes differ {a.shape} vs {b.shape}'
    mse = float(((a.astype(np.float64) - b.astype(np.float64))**2).mean())
    identical = mse == 0.0
    self_distance = metric(a, a)
    assert abs(self_distance) < 1e-6, f'LPIPS self-distance {self_distance}'
    return {
        'reference': f'results/runs/{reference}/{ref_rel}', 'variant': f'results/runs/{variant}/{var_rel}',
        'image_file_sha256': {k: sha(v) for k, v in blobs.items()},
        'pixel_sha256': {k: sha(v.tobytes()) for k, v in arrays.items()},
        'pixels_identical': identical,
        'psnr_db': 'infinity (identical pixels)' if identical else 10*math.log10(255**2/mse),
        'mse_rgb_8bit': mse,
        'mean_abs_rgb_8bit': float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean()),
        'lpips_alex_v0_1': 0.0 if identical else metric(a, b),
        'lpips_self_distance': self_distance,
        'resolution': f'{a.shape[1]}x{a.shape[0]}',
        'method': ('Decoded full-resolution RGB, no crop/resize; LPIPS calibrated AlexNet v0.1, inputs scaled to [-1,1], '
                   'CPU float32 eval/inference mode, two threads. PSNR uses peak 255 and MSE over all RGB channels.'),
        'versions': {p: importlib.metadata.version(p) for p in ('torch', 'torchvision', 'lpips', 'numpy', 'Pillow')},
        'weights_sha256': metric.weights,
        'source': 'https://github.com/richzhang/PerceptualSimilarity',
        'scope': ('One development prompt and seed; paired diagnostic against the same engine\'s reference configuration on the '
                  'same device. Not prompt alignment, not held-out data, not formal quality eligibility.'),
        'script': 'scripts/quality_diagnostics.py',
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--output', type=Path, default=ROOT/'output'/'quality-diagnostics-20261024')
    ap.add_argument('--write-run-files', action='store_true')
    args = ap.parse_args()
    metric = Lpips()
    rows = []
    for group, reference, variants in PAIRS:
        for variant in variants:
            result = compare(metric, reference, variant)
            target = RUNS/variant/'quality-diagnostics.json'
            if args.write_run_files and not target.exists():
                target.write_text(json.dumps(result, indent=2) + '\n')
            rows.append({'group': group, 'reference_run': reference, 'variant_run': variant,
                         'psnr_db': result['psnr_db'], 'lpips_alex_v0_1': result['lpips_alex_v0_1'],
                         'mean_abs_rgb_8bit': result['mean_abs_rgb_8bit'], 'pixels_identical': result['pixels_identical'],
                         'run_file_written': args.write_run_files and target.exists()})
            psnr = result['psnr_db'] if isinstance(result['psnr_db'], str) else f"{result['psnr_db']:.2f} dB"
            print(f"{group:34s} {variant.split('__')[0]:40s} PSNR {psnr:>22s}  LPIPS {result['lpips_alex_v0_1']:.4f}")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'quality-diagnostics.json').write_text(json.dumps({'pairs': rows, 'weights_sha256': metric.weights}, indent=2) + '\n')


if __name__ == '__main__':
    main()
