#!/usr/bin/env python3
"""Generate GenEval images for a Nunchaku SVDQuant configuration (pinned community wheel, A100 40 GB).

  HF_HUB_OFFLINE=1 PYTHONPATH=<bitsandbytes overlay> $NUNCHAKU_PYTHON scripts/geneval_generate_nunchaku.py \
      --config configs/a100-nunchaku-flux-klein-int4-r32.json --subset calibration_subset

Accepts the A100 benchmark config (BF16 text encoder, 1024²) or a GenEval-only copy of a Jetson Nunchaku config
(configs/geneval/a100-geneval-nunchaku-*.json: same checkpoint file, NF4 text encoder given as
optimizations.text_encoder_quantization). The transformer is loaded exactly as scripts/run_nunchaku.py loads it;
only the prompt and seed change per image. No step cache, compiler or VAE tiling is applied (none is configured;
any other option is refused). Prompts come from the frozen split configs/geneval/split-20261024.json; held-out
prompts are refused before 2026-11-30. Output follows GenEval's layout like scripts/geneval_generate.py.
Nunchaku generations are not bitwise reproducible at a fixed seed (a100-nunchaku-flux-klein-001); per-image
hashes are recorded. Timing here is per image wall time for bookkeeping only; it is not a benchmark measurement.
"""
import argparse
import datetime as dt
import hashlib
import importlib.metadata
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
QUANT = {'method': 'svdquant', 'component': 'transformer', 'weights_bits': 4, 'activations_bits': 4, 'low_rank': 32,
         'format': 'int4'}
OFF = {'cpu_offload': False, 'vae_tiling': False, 'vae_slicing': False, 'torch_compile': False}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(cfg):
    opt = dict(cfg['optimizations'])
    assert opt.pop('quantization') == QUANT, 'only the labelled W4A4 INT4/r32 transformer is implemented'
    te = opt.pop('text_encoder_quantization', None)
    assert te is None or (te['method'] == 'bnb_nf4' and len(te['sha256']) == 64), 'text encoder: pre-quantized NF4 only'
    assert {k: opt.get(k) for k in OFF} == OFF and set(opt) <= set(OFF), f'options not applied here: {opt}'
    assert cfg['engine']['name'] == 'nunchaku' and cfg['model']['repo_id'] == 'black-forest-labs/FLUX.2-klein-4B'
    assert cfg['workload']['guidance_scale'] == 1.0 and cfg['workload']['batch_size'] == 1
    return te


def load(cfg, te):
    import torch
    import transformers
    from diffusers import Flux2KleinPipeline
    from nunchaku import NunchakuFlux2Transformer2DModel
    from nunchaku.models.linear import SVDQW4A4Linear
    from run_nunchaku import cached_file, check_runtime, file_sha256

    provenance = check_runtime(cfg, torch)
    quantized = cached_file(cfg['quantized_transformer'], offline=True)
    t0 = time.perf_counter()
    transformer = NunchakuFlux2Transformer2DModel.from_pretrained(str(quantized), device='cuda', torch_dtype=torch.bfloat16,
                                                                 offload=False)
    layers = [m for m in transformer.modules() if isinstance(m, SVDQW4A4Linear)]
    assert layers and all(m.precision == 'int4' and m.rank == 32 for m in layers), 'not an INT4/r32 SVDQuant model'
    kwargs = {'transformer': transformer}
    receipt = {'transformer': {'checkpoint_sha256': cfg['quantized_transformer']['sha256'], 'svdq_layers': len(layers)}}
    if te:
        digest = file_sha256(Path(te['path'])/'model.safetensors')
        assert digest == te['sha256'], f'pre-quantized encoder sha256 mismatch: {digest}'
        # The quantization config is stored with the weights; this reads the NF4 tensors directly.
        kwargs['text_encoder'] = transformers.Qwen3ForCausalLM.from_pretrained(te['path'], torch_dtype=torch.bfloat16,
                                                                               device_map={'': 'cuda'})
        assert any('Linear4bit' in type(m).__name__ for m in kwargs['text_encoder'].modules()), 'encoder is not NF4'
        receipt['text_encoder'] = dict(te, sha256_verified=digest, bitsandbytes=importlib.metadata.version('bitsandbytes'))
    model = cfg['model']
    pipe = Flux2KleinPipeline.from_pretrained(model['repo_id'], revision=model['revision'], torch_dtype=torch.bfloat16,
                                              local_files_only=True, **kwargs).to('cuda')
    assert pipe.config.is_distilled is True
    pipe.set_progress_bar_config(disable=True)
    torch.cuda.synchronize()
    receipt.update(load_total_s=time.perf_counter() - t0, nunchaku=importlib.metadata.version('nunchaku'),
                   nunchaku_install=provenance)
    return pipe, receipt


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--subset', choices=['calibration_subset', 'calibration_rest', 'calibration', 'heldout'], default='calibration_subset')
    ap.add_argument('--out-root', type=Path, default=ROOT/'results'/'geneval')
    ap.add_argument('--limit', type=int, default=None, help='first N prompts only (smoke)')
    args = ap.parse_args()
    if args.subset == 'heldout' and dt.date.today() < UNSEAL:
        sys.exit(f'held-out prompts are sealed until {UNSEAL}')
    assert os.environ.get('HF_HUB_OFFLINE') == '1', 'run offline (HF_HUB_OFFLINE=1)'
    cfg = json.loads(args.config.read_text())
    te = check(cfg)

    import torch
    from benchlib import write_json

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
        pipe, receipt = load(cfg, te)
        w = cfg['workload']
        rows = []
        for n, index in enumerate(indices):
            item = json.loads(meta[index])
            folder = images/f'{index:05d}'
            (folder/'samples').mkdir(parents=True)
            (folder/'metadata.jsonl').write_text(meta[index] + '\n')
            for seed in seeds:
                generator = torch.Generator(device='cuda').manual_seed(seed)
                torch.cuda.synchronize()
                t = time.perf_counter()
                image = pipe(prompt=item['prompt'], height=w['height'], width=w['width'],
                             num_inference_steps=w['num_inference_steps'], guidance_scale=w['guidance_scale'],
                             max_sequence_length=w['max_sequence_length'], generator=generator, output_type='pil').images[0]
                torch.cuda.synchronize()
                ms = (time.perf_counter() - t)*1000
                path = folder/'samples'/f'{seed:04d}.png'
                image.save(path)
                rows.append({'index': index, 'tag': item['tag'], 'seed': seed, 'wall_ms': ms,
                             'png_sha256': sha(path), 'rgb_sha256': hashlib.sha256(image.convert('RGB').tobytes()).hexdigest()})
            print(f'{n+1}/{len(indices)} [{item["tag"]}] {item["prompt"]}: {ms:.0f} ms (last seed)', flush=True)
        write_json(run_dir/'generations.json', {'subset': args.subset, 'indices': indices, 'seeds': seeds, 'rows': rows,
                                                'load': receipt, 'split_sha256': sha(SPLIT), 'config_sha256': sha(args.config),
                                                'source_sha256': {p: sha(ROOT/'scripts'/p) for p in
                                                                  ('geneval_generate_nunchaku.py', 'run_nunchaku.py')},
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
