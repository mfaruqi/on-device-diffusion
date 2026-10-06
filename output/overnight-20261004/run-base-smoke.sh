#!/bin/bash
set -euo pipefail
cd /home/mfaruqi/on-device-diffusion-campaigns/20261004-overnight
export PYTHONUNBUFFERED=1
if pgrep -x sd-bench || pgrep -x sd-bench-nvtx; then
  echo "Another benchmark is running; refusing overlap"; exit 1
fi
PYTHONPATH=scripts python3 -c 'import json; from run_jetson_sdcpp import check_config; check_config(json.load(open("configs/jetson-flux-klein-base-q4-512-disk-smoke.json")))'
python3 - <<'DOWNLOAD'
from pathlib import Path
import hashlib, json, subprocess
dest=Path.home()/'models/flux2-klein/flux-2-klein-base-4b-Q4_0.gguf'
sha='c3a2854510677b7aa37dd7547d908c54889a76c6d6aa3ffe902fcaa092d1328b'
url='https://huggingface.co/leejet/FLUX.2-klein-base-4B-GGUF/resolve/d12671125306ca6b5f6db1b33ed4c80c8511a53f/flux-2-klein-base-4b-Q4_0.gguf'
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()
if dest.exists():
    assert digest(dest)==sha, 'Existing checkpoint hash differs; preserving it'
else:
    tmp=dest.with_suffix(dest.suffix+'.download')
    subprocess.run(['curl','--fail','--location','--retry','5','--retry-all-errors','--connect-timeout','30','--max-time','1800','--continue-at','-','--output',str(tmp),url],check=True)
    assert tmp.stat().st_size==2460378560 and digest(tmp)==sha, 'Downloaded checkpoint failed verification'
    tmp.rename(dest)
Path('logs/jetson-base-q4-prepared.json').write_text(json.dumps({'path':str(dest),'sha256':sha,'revision':'d12671125306ca6b5f6db1b33ed4c80c8511a53f','verified':True},indent=2))
print('Base Q4 checkpoint verified',flush=True)
DOWNLOAD
if pgrep -x sd-bench || pgrep -x sd-bench-nvtx; then exit 1; fi
python3 scripts/run_jetson_sdcpp.py --config configs/jetson-flux-klein-base-q4-512-disk-smoke.json --binary "$HOME/tools/sd-bench-build-20261004/sd-bench" --out-root results/runs --kind attempt
