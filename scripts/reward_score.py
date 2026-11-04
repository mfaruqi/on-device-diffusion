#!/usr/bin/env python3
"""Score GenEval image folders with four prompt-image preference metrics: CLIP score, PickScore, HPS v2.1, ImageReward.

  python scripts/reward_score.py --prepare-only                       # login node: download, pin revisions
  HF_HUB_OFFLINE=1 python scripts/reward_score.py --images <run>/images [<run>/images ...]

Images use GenEval's layout (<run>/images/<NNNNN>/metadata.jsonl, samples/<SSSS>.png); each image is scored
against its own prompt. Writes <run>/reward-scores.jsonl (one row per image) and <run>/reward-summary.json
(means). Model revisions come from configs/quality/reward-models.resolved.json, written by --prepare-only;
scoring is offline and refuses a resolved file whose checkpoint hashes do not match the files on disk.
Definitions: wiki/methods/baseline-metrics.md#prompt-image-preference-scores.
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESOLVED = ROOT/'configs'/'quality'/'reward-models.resolved.json'
MODELS = {
    'clip': {'repo': 'openai/clip-vit-large-patch14'},
    'pickscore': {'repo': 'yuvalkirstain/PickScore_v1', 'processor': 'laion/CLIP-ViT-H-14-laion2B-s32B-b79K'},
    'hps': {'repo': 'xswu/HPSv2', 'file': 'HPS_v2.1_compressed.pt', 'arch': 'ViT-H-14'},
    'imagereward': {'repo': 'THUDM/ImageReward', 'file': 'ImageReward.pt', 'med_config': 'med_config.json',
                    'tokenizer': 'bert-base-uncased'},
}
METRICS = ['clip', 'pickscore', 'hps', 'imagereward']


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 24), b''):
            h.update(block)
    return h.hexdigest()


def prepare():
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    api, out = HfApi(), {}
    repos = {m['repo'] for m in MODELS.values()} | {MODELS['pickscore']['processor'], MODELS['imagereward']['tokenizer']}
    revs = {r: api.model_info(r).sha for r in sorted(repos)}
    for r in ('openai/clip-vit-large-patch14', 'yuvalkirstain/PickScore_v1'):
        snapshot_download(r, revision=revs[r], allow_patterns=['*.json', '*.txt', '*.safetensors'])
    snapshot_download(MODELS['pickscore']['processor'], revision=revs[MODELS['pickscore']['processor']],
                      allow_patterns=['*.json', '*.txt'])
    snapshot_download('bert-base-uncased', revision=revs['bert-base-uncased'], allow_patterns=['*.json', '*.txt'])
    files = {}
    for key in ('hps', 'imagereward'):
        m = MODELS[key]
        for name in [m['file']] + ([m['med_config']] if 'med_config' in m else []):
            path = hf_hub_download(m['repo'], name, revision=revs[m['repo']])
            files[f'{m["repo"]}/{name}'] = sha256(path)
    import open_clip, transformers, torch
    try:
        import ImageReward
        ir_version = getattr(ImageReward, '__version__', 'unknown')
    except ImportError as e:
        sys.exit(f'ImageReward package missing: {e}')
    out = {'models': MODELS, 'revisions': revs, 'sha256': files,
           'packages': {'torch': torch.__version__, 'transformers': transformers.__version__,
                        'open_clip': open_clip.__version__, 'image-reward': ir_version}}
    RESOLVED.parent.mkdir(parents=True, exist_ok=True)
    RESOLVED.write_text(json.dumps(out, indent=2) + '\n')
    print(f'wrote {RESOLVED}')


class Scorers:
    """All four models on one GPU, FP32, eval mode. Each call scores a list of images against one prompt."""

    def __init__(self, spec, device):
        import torch
        import open_clip
        from huggingface_hub import hf_hub_download
        from transformers import AutoModel, AutoProcessor, CLIPModel, CLIPProcessor
        rev = spec['revisions']
        self.torch, self.device = torch, device

        def local(repo, name):
            path = hf_hub_download(repo, name, revision=rev[repo], local_files_only=True)
            want = spec['sha256'][f'{repo}/{name}']
            assert sha256(path) == want, f'{repo}/{name}: sha256 differs from {RESOLVED.name}'
            return path

        r = MODELS['clip']['repo']
        self.clip = CLIPModel.from_pretrained(r, revision=rev[r], local_files_only=True).to(device).eval()
        self.clip_proc = CLIPProcessor.from_pretrained(r, revision=rev[r], local_files_only=True)
        r, p = MODELS['pickscore']['repo'], MODELS['pickscore']['processor']
        self.pick = AutoModel.from_pretrained(r, revision=rev[r], local_files_only=True).to(device).eval()
        self.pick_proc = AutoProcessor.from_pretrained(p, revision=rev[p], local_files_only=True)
        m = MODELS['hps']
        self.hps, _, self.hps_tf = open_clip.create_model_and_transforms(m['arch'], pretrained=None, device=device)
        state = torch.load(local(m['repo'], m['file']), map_location='cpu', weights_only=True)
        self.hps.load_state_dict(state['state_dict'] if 'state_dict' in state else state)
        self.hps.eval()
        self.hps_tok = open_clip.get_tokenizer(m['arch'])
        import ImageReward as RM
        from ImageReward.models.BLIP import blip_pretrain
        from transformers import BertTokenizer
        m = MODELS['imagereward']
        tok = m['tokenizer']

        def pinned_tokenizer():
            # BLIP's init_tokenizer with the pinned revision (the package loads 'bert-base-uncased' unpinned)
            t = BertTokenizer.from_pretrained(tok, revision=rev[tok], local_files_only=True)
            t.add_special_tokens({'bos_token': '[DEC]'})
            t.add_special_tokens({'additional_special_tokens': ['[ENC]']})
            t.enc_token_id = t.additional_special_tokens_ids[0]
            return t
        blip_pretrain.init_tokenizer = pinned_tokenizer
        self.ir = RM.load(local(m['repo'], m['file']), device=device, med_config=local(m['repo'], m['med_config']))
        self.ir.eval()

    def __call__(self, prompt, images):
        torch, dev = self.torch, self.device
        out = {}
        with torch.no_grad():
            x = self.clip_proc(text=[prompt], images=images, return_tensors='pt', padding=True, truncation=True).to(dev)
            i = torch.nn.functional.normalize(self.clip.get_image_features(pixel_values=x['pixel_values']), dim=-1)
            t = torch.nn.functional.normalize(self.clip.get_text_features(input_ids=x['input_ids'], attention_mask=x['attention_mask']), dim=-1)
            out['clip'] = (i @ t.T).squeeze(1).tolist()
            xi = self.pick_proc(images=images, return_tensors='pt').to(dev)
            xt = self.pick_proc(text=[prompt], padding=True, truncation=True, max_length=77, return_tensors='pt').to(dev)
            i = torch.nn.functional.normalize(self.pick.get_image_features(**xi), dim=-1)
            t = torch.nn.functional.normalize(self.pick.get_text_features(**xt), dim=-1)
            out['pickscore'] = (self.pick.logit_scale.exp() * (i @ t.T)).squeeze(1).tolist()
            pix = torch.stack([self.hps_tf(im) for im in images]).to(dev)
            i = torch.nn.functional.normalize(self.hps.encode_image(pix), dim=-1)
            t = torch.nn.functional.normalize(self.hps.encode_text(self.hps_tok([prompt]).to(dev)), dim=-1)
            out['hps'] = (i @ t.T).squeeze(1).tolist()
            ir = self.ir.score(prompt, images)
            out['imagereward'] = ir if isinstance(ir, list) else [ir]
        return out


def score_run(images_dir, scorers, spec_hash):
    from PIL import Image
    run = images_dir.parent
    rows = []
    for prompt_dir in sorted(p for p in images_dir.iterdir() if p.is_dir()):
        meta = json.loads((prompt_dir/'metadata.jsonl').read_text().splitlines()[0])
        paths = sorted((prompt_dir/'samples').glob('*.png'))
        assert paths, f'{prompt_dir}: no samples'
        images = [Image.open(p).convert('RGB') for p in paths]
        scores = scorers(meta['prompt'], images)
        assert all(len(scores[k]) == len(paths) for k in METRICS), f'{prompt_dir}: score count mismatch'
        for j, p in enumerate(paths):
            m = re.search(r'(\d+)/samples/(\d+)\.png$', str(p))
            rows.append({'prompt_index': int(m.group(1)), 'seed_index': int(m.group(2)), 'tag': meta['tag'],
                         'prompt': meta['prompt'], **{k: float(scores[k][j]) for k in METRICS}})
    (run/'reward-scores.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    summary = {'run': run.name, 'images': len(rows), 'prompts': len({r['prompt_index'] for r in rows}),
               'mean': {k: sum(r[k] for r in rows)/len(rows) for k in METRICS},
               'models_resolved_sha256': spec_hash}
    (run/'reward-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(f'{run.name}: ' + '  '.join(f'{k} {v:.4f}' for k, v in summary['mean'].items()), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--prepare-only', action='store_true')
    ap.add_argument('--images', type=Path, nargs='*', default=[])
    ap.add_argument('--force', action='store_true', help='rescore runs that already have reward-summary.json')
    args = ap.parse_args()
    if args.prepare_only:
        return prepare()
    assert args.images, '--images required'
    import torch
    assert torch.cuda.is_available(), 'scoring needs a GPU'
    spec = json.loads(RESOLVED.read_text())
    spec_hash = sha256(RESOLVED)
    scorers = Scorers(spec, 'cuda')
    for images_dir in args.images:
        assert images_dir.name == 'images' and images_dir.is_dir(), f'{images_dir}: expected <run>/images'
        if (images_dir.parent/'reward-summary.json').exists() and not args.force:
            print(f'skip (scored) {images_dir.parent.name}')
            continue
        score_run(images_dir, scorers, spec_hash)


if __name__ == '__main__':
    main()
