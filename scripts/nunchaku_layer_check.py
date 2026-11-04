#!/usr/bin/env python3
"""Single-layer check of Nunchaku's W4A4 kernels: quantized linear vs its BF16 original, and device vs device.

  # 1. CPU only (any host with both checkpoints cached): write the bundle of layers and fixed inputs
  python3 scripts/nunchaku_layer_check.py extract --out /path/layer-bundle.safetensors
  # 2. On a GPU: run every layer at every token count
  python scripts/nunchaku_layer_check.py run --bundle /path/layer-bundle.safetensors --out <run dir> \
      [--nunchaku-dir ~/tools/nunchaku-sm87-0c8379d] [--reference <A100 run dir>/outputs.safetensors]

Layers are transformer block 0 of the pinned community INT4/r32 checkpoint (attn.to_qkv, attn.to_out.0,
ff.linear_in, ff.linear_out), paired with the same layers of the pinned BF16 transformer (to_qkv = to_q, to_k, to_v
concatenated, as the adapter fuses them). Inputs are N(0, 1) BF16, seed 0, stored in the bundle so every device sees
the same bytes; they are not real activations, so the error vs BF16 measures the kernels on synthetic data, not the
checkpoint's quality.

Per layer and token count M:
  rel_l2_vs_bf16   ||y_w4a4 - y_bf16|| / ||y_bf16||, y_bf16 = F.linear(x, W_bf16) on the same device
  cos_vs_bf16      cosine similarity of the flattened outputs
  w4a4_ms, bf16_ms CUDA-event time per call (activation quantize + GEMM for W4A4; F.linear for BF16),
                   median and min-max of --iters calls after --warmup calls
With --reference, the W4A4 outputs at the smallest M are also compared with another device's (max_abs, rel_l2).
Writes layer-check.json and outputs.safetensors (W4A4 outputs at the smallest M) into --out.
Each layer is the installed Nunchaku's own SVDQW4A4Linear (int4, rank 32, signed activations, no bias), as the
pipeline builds it, so the comparison runs whatever op signatures that install has: the A100 community wheel and the
pinned source differ (the wheel's activation quantizer takes an extra act_unsigned flag). --nunchaku-dir puts an
in-place source build first on sys.path and checks that it is the one imported.
"""
import argparse
import json
from pathlib import Path
import statistics
import sys

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file, save_file

HUB = '/scratch/gilbreth/mfaruqi/huggingface/hub'
QUANT = (f'{HUB}/models--tonera--FLUX.2-klein-4B-Nunchaku/snapshots/8aa64445d8e45edc2ea431743b67ac7e2ca5f739/'
         'svdq-int4_r32-FLUX.2-klein-4B-Nunchaku.safetensors')
BF16 = f'{HUB}/models--black-forest-labs--FLUX.2-klein-4B/snapshots/e7b7dc27f91deacad38e78976d1f2b499d76a294/transformer'
LAYERS = {  # quantized layer -> BF16 weights it replaces
    'transformer_blocks.0.attn.to_qkv': ['transformer_blocks.0.attn.to_q', 'transformer_blocks.0.attn.to_k',
                                         'transformer_blocks.0.attn.to_v'],
    'transformer_blocks.0.attn.to_out.0': ['transformer_blocks.0.attn.to_out.0'],
    'transformer_blocks.0.ff.linear_in': ['transformer_blocks.0.ff.linear_in'],
    'transformer_blocks.0.ff.linear_out': ['transformer_blocks.0.ff.linear_out'],
}
QPARTS = ('qweight', 'wscales', 'smooth_factor', 'proj_down', 'proj_up')
TOKENS = (1024, 4096)  # image tokens at 512^2 and 1024^2


def extract(args):
    out = {}
    q = safe_open(args.quant, 'pt')
    bf16 = {}
    for fn in sorted(Path(args.bf16).glob('*.safetensors')):
        f = safe_open(str(fn), 'pt')
        bf16.update({k: f for k in f.keys()})
    in_features = set()
    for name, parts in LAYERS.items():
        for p in QPARTS:
            out[f'{name}.{p}'] = q.get_tensor(f'{name}.{p}')
        out[f'{name}.bf16_weight'] = torch.cat([bf16[f'{p}.weight'].get_tensor(f'{p}.weight') for p in parts])
        assert f'{name}.bias' not in q.keys() and all(f'{p}.bias' not in bf16 for p in parts), 'unexpected bias'
        in_features.add(out[f'{name}.bf16_weight'].shape[1])
    g = torch.Generator().manual_seed(0)
    for k in sorted(in_features):
        for m in TOKENS:
            out[f'x.{k}.{m}'] = torch.randn(m, k, generator=g).to(torch.bfloat16)
    save_file({k: v.contiguous() for k, v in out.items()}, args.out,
              metadata={'quant': args.quant, 'bf16': args.bf16, 'inputs': 'N(0,1) bf16, torch.Generator seed 0'})
    print('wrote', args.out, sum(v.numel() * v.element_size() for v in out.values()) / 2**20, 'MiB')


def load_linear_class(nunchaku_dir):
    if nunchaku_dir:
        sys.path.insert(0, str(Path(nunchaku_dir).expanduser()))
    import nunchaku
    from nunchaku.models.linear import SVDQW4A4Linear
    if nunchaku_dir:
        assert Path(nunchaku.__file__).resolve().is_relative_to(Path(nunchaku_dir).expanduser().resolve()), (
            f'imported {nunchaku.__file__}, not the build in {nunchaku_dir}')
    so = sorted(Path(nunchaku.__file__).parent.glob('_C*.so'))
    return SVDQW4A4Linear, {'nunchaku_file': nunchaku.__file__, 'extension': str(so[0]) if so else None,
                            'version': getattr(nunchaku, '__version__', None)}


def build_layer(cls, bundle, name, dev):
    out_f, half_in = bundle[f'{name}.qweight'].shape
    layer = cls(half_in * 2, out_f, rank=bundle[f'{name}.proj_down'].shape[1], bias=False, precision='int4',
                act_unsigned=False, torch_dtype=torch.bfloat16, device=dev)
    params = dict(layer.named_parameters())
    for p in QPARTS:
        assert params[p].shape == bundle[f'{name}.{p}'].shape, (name, p)
        params[p].data.copy_(bundle[f'{name}.{p}'])
    params['smooth_factor_orig'].data.copy_(bundle[f'{name}.smooth_factor'])  # unused by forward
    return layer.eval()


def timed(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    ms = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        ms.append(a.elapsed_time(b))
    return {'median': statistics.median(ms), 'min': min(ms), 'max': max(ms), 'n': iters}


def compare(y, ref):
    d = (y.float() - ref.float())
    return {'max_abs': d.abs().max().item(), 'rel_l2': (d.norm() / ref.float().norm()).item()}


@torch.inference_mode()
def run(args):
    cls, install = load_linear_class(args.nunchaku_dir)
    dev = torch.device('cuda')
    bundle = load_file(args.bundle)
    ref = load_file(args.reference) if args.reference else None
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {'nunchaku': install, 'device': torch.cuda.get_device_name(0),
              'capability': list(torch.cuda.get_device_capability(0)), 'torch': torch.__version__,
              'cuda': torch.version.cuda, 'bundle': args.bundle, 'reference': args.reference,
              'warmup': args.warmup, 'iters': args.iters, 'layers': []}
    saved = {}
    for name in LAYERS:
        layer = build_layer(cls, bundle, name, dev)
        W = bundle[f'{name}.bf16_weight'].to(dev)
        for m in TOKENS:
            x = bundle[f'x.{W.shape[1]}.{m}'].to(dev)
            run_q = lambda: layer(x[None])[0]
            y = run_q()
            y_bf16 = F.linear(x, W)
            torch.cuda.synchronize()
            row = {'layer': name, 'in_features': W.shape[1], 'out_features': W.shape[0], 'tokens': m,
                   'finite': bool(torch.isfinite(y).all()),
                   **{f'{k}_vs_bf16': v for k, v in compare(y, y_bf16).items()},
                   'cos_vs_bf16': F.cosine_similarity(y.float().flatten(), y_bf16.float().flatten(), dim=0).item(),
                   'w4a4_ms': timed(run_q, args.warmup, args.iters),
                   'bf16_ms': timed(lambda: F.linear(x, W), args.warmup, args.iters)}
            if m == TOKENS[0]:
                saved[name] = y.cpu()
                if ref is not None:
                    row['vs_reference'] = compare(y, ref[name].to(dev))
            result['layers'].append(row)
            print(f"{name:36s} M={m:5d} rel_l2={row['rel_l2_vs_bf16']:.4f} cos={row['cos_vs_bf16']:.5f} "
                  f"w4a4 {row['w4a4_ms']['median']:.3f} ms  bf16 {row['bf16_ms']['median']:.3f} ms"
                  + (f"  vs ref rel_l2={row['vs_reference']['rel_l2']:.2e}" if 'vs_reference' in row else ''))
            del x, y, y_bf16
        del layer, W
        torch.cuda.empty_cache()
    result['max_memory_allocated_gib'] = torch.cuda.max_memory_allocated() / 2**30
    save_file(saved, str(out_dir / 'outputs.safetensors'))
    (out_dir / 'layer-check.json').write_text(json.dumps(result, indent=2))
    print('wrote', out_dir / 'layer-check.json')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    e = sub.add_parser('extract')
    e.add_argument('--out', required=True)
    e.add_argument('--quant', default=QUANT)
    e.add_argument('--bf16', default=BF16)
    r = sub.add_parser('run')
    r.add_argument('--bundle', required=True)
    r.add_argument('--out', required=True)
    r.add_argument('--nunchaku-dir')
    r.add_argument('--reference')
    r.add_argument('--warmup', type=int, default=5)
    r.add_argument('--iters', type=int, default=20)
    args = ap.parse_args()
    extract(args) if args.cmd == 'extract' else run(args)


if __name__ == '__main__':
    main()
