#!/usr/bin/env python3
"""Derive A100 (cuda) GenEval configurations from the Jetson PyTorch configurations.

Quality depends on the checkpoint files, precision, workload and options, not on the device, so the Jetson's
Q4_0 GGUF transformer + NF4 encoder at 512x512 is evaluated on the A100. Changed fields only:
backend/device_label (cuda, A100), file paths (Gilbreth copies, same sha256 where the file is the same),
stage_residency and conditioning_reuse removed (they move weights or skip identical encodings; outputs are
unchanged by them, as the bit-identical Jetson records show), and Jetson-only protocol keys. Arithmetic is
not bit-identical across devices; images are expected to be close, not equal.

  python scripts/geneval_matched_configs.py --nf4-sha256 <hash of the Gilbreth NF4 model.safetensors>
Writes configs/geneval/<id>.json (refuses to overwrite).
"""
import argparse
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HF = '/scratch/gilbreth/mfaruqi/huggingface/hub'
GGUF = {
    'flux-2-klein-4b-Q4_0.gguf': f'{HF}/models--leejet--FLUX.2-klein-4B-GGUF/snapshots/{{rev}}/flux-2-klein-4b-Q4_0.gguf',
    'flux-2-klein-base-4b-Q4_0.gguf': f'{HF}/models--leejet--FLUX.2-klein-base-4B-GGUF/snapshots/d12671125306ca6b5f6db1b33ed4c80c8511a53f/flux-2-klein-base-4b-Q4_0.gguf',
}
NF4_DIR = '/scratch/gilbreth/mfaruqi/models/flux2-klein-torch/text_encoder-nf4'
SOURCES = ['jetson-torch-flux-klein-q4-nf4-512', 'jetson-torch-flux-klein-q4-nf4-512-taylorseer',
           'jetson-torch-flux-klein-q4-nf4-512-condreuse-compile', 'jetson-torch-flux-klein-q4-nf4-512-vaetile',
           'jetson-torch-flux-klein-q4-nf4-512-sdpa-math', 'jetson-torch-flux-klein-base-q4-nf4-512',
           'jetson-torch-flux-klein-base-q4-nf4-512-condreuse-compile']


def derive(stem, nf4_sha, distilled_rev):
    src = json.loads((ROOT/'configs'/f'{stem}.json').read_text())
    cfg = copy.deepcopy(src)
    cfg['id'] = 'a100-geneval-' + stem.removeprefix('jetson-')
    cfg['description'] = f"GenEval-only A100 copy of {src['id']} ({src['description']})"
    cfg['backend'], cfg['device_label'] = 'cuda', 'gilbreth-a100-40gb'
    cfg['protocol'] = {k: v for k, v in src['protocol'].items() if k != 'tegrastats_interval_ms'}
    opt = cfg['optimizations']
    opt['stage_residency'] = None
    opt['conditioning_reuse'] = False
    q = opt['quantization']
    name = Path(q['transformer']['file']).name
    q['transformer']['file'] = GGUF[name].format(rev=distilled_rev)
    q['text_encoder'] = {**q['text_encoder'], 'path': NF4_DIR, 'sha256': nf4_sha,
                         'jetson_sha256': src['optimizations']['quantization']['text_encoder']['sha256'],
                         'source': 'klein revision e7b7dc27 text_encoder quantized on Gilbreth by scripts/save_quantized_text_encoder.py'}
    cfg.pop('environment', None)
    cfg['reference_config'] = None
    cfg['derived_from'] = {'config': f'configs/{stem}.json',
                           'changed': ['backend', 'device_label', 'file paths', 'stage_residency -> null',
                                       'conditioning_reuse -> false', 'tegrastats_interval_ms removed', 'NF4 encoder copy']}
    return cfg


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--nf4-sha256', required=True)
    ap.add_argument('--distilled-gguf-revision', required=True)
    args = ap.parse_args()
    out = ROOT/'configs'/'geneval'
    for stem in SOURCES:
        cfg = derive(stem, args.nf4_sha256, args.distilled_gguf_revision)
        path = out/f"{cfg['id']}.json"
        assert not path.exists(), f'{path} exists'
        path.write_text(json.dumps(cfg, indent=2) + '\n')
        print(path.relative_to(ROOT))


if __name__ == '__main__':
    main()
