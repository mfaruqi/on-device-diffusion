"""One saved-image diagnostic, CPU only; no latency or formal quality evaluation."""
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CACHE = Path(__file__).resolve().parent / '.venv' / 'torch-cache'
os.environ['TORCH_HOME'] = str(CACHE)
import lpips
import numpy as np
import torch
from PIL import Image

torch.set_num_threads(2)
reference = ROOT / 'results/runs/jetson-flux-klein-base-001__repeat__20261005-042014/images/measured-0.png'
variant = ROOT / 'results/runs/jetson-flux-klein-base-003__attempt__20261005-054530/images/measured-0.png'
out = variant.parent.parent / 'quality-diagnostics.json'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def pixels(path):
    image = Image.open(path).convert('RGB')
    assert image.size == (512, 512)
    return np.array(image, dtype=np.float32)

def tensor(array):
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0) / 127.5 - 1

a, b = pixels(reference), pixels(variant)
metric = lpips.LPIPS(net='alex', version='0.1', lpips=True).cpu().eval()
with torch.inference_mode():
    self_distance = metric(tensor(a), tensor(a)).item()
    distance = metric(tensor(a), tensor(b)).item()
assert math.isfinite(distance) and abs(self_distance) < 1e-8
mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
weight_files = list((CACHE / 'hub/checkpoints').glob('alexnet-*.pth'))
weight_files += [Path(lpips.__file__).parent / 'weights/v0.1/alex.pth']
assert len(weight_files) == 2
result = {
    'reference': str(reference.relative_to(ROOT)),
    'variant': str(variant.relative_to(ROOT)),
    'image_file_sha256': {'reference': sha(reference), 'variant': sha(variant)},
    'pixel_sha256': {k: hashlib.sha256(arr.astype(np.uint8).tobytes()).hexdigest() for k, arr in [('reference', a), ('variant', b)]},
    'psnr_db': 'infinity (identical pixels)' if mse == 0 else 10 * math.log10(255**2 / mse),
    'mse_rgb_8bit': mse,
    'lpips_alex_v0_1': distance,
    'lpips_self_distance': self_distance,
    'method': 'Decoded full-resolution 512x512 RGB, no crop/resize; LPIPS calibrated AlexNet v0.1, inputs scaled to [-1,1], CPU float32 eval/inference mode, two threads. PSNR uses peak255 and mean squared error over all RGB channels.',
    'versions': {m: importlib.metadata.version(m) for m in ['torch','torchvision','lpips','numpy','Pillow']},
    'weights_sha256': {p.name: sha(p) for p in weight_files},
    'source': 'https://github.com/richzhang/PerceptualSimilarity',
    'scope': 'One existing calibration/development prompt and seed; paired diagnostic only, no held-out data or formal quality eligibility.'
}
out.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
