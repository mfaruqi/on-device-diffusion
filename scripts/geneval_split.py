#!/usr/bin/env python3
"""Freeze the GenEval calibration / held-out split (proposal: separate calibration vs held-out prompts).

  python scripts/geneval_split.py   # writes configs/geneval/split-20261024.json (refuses to overwrite)

Stratified by GenEval task tag, 50/50 with a fixed seed; the calibration subset takes 16 prompts per tag
from the calibration half. Held-out prompts must not be generated or scored before Week 10 (Nov 30).
"""
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
META = ROOT/'configs'/'geneval'/'evaluation_metadata.jsonl'
OUT = ROOT/'configs'/'geneval'/'split-20261024.json'
SEED, PER_TAG = 20261024, 16

lines = META.read_text().splitlines()
by_tag = defaultdict(list)
for i, line in enumerate(lines):
    by_tag[json.loads(line)['tag']].append(i)
rng = random.Random(SEED)
calibration, heldout, subset = [], [], []
for tag in sorted(by_tag):
    idx = by_tag[tag][:]
    rng.shuffle(idx)
    half = (len(idx) + 1)//2
    cal, held = sorted(idx[:half]), sorted(idx[half:])
    calibration += cal
    heldout += held
    subset += sorted(rng.sample(cal, PER_TAG))
assert not set(calibration) & set(heldout) and len(calibration) + len(heldout) == len(lines)
assert OUT.exists() is False, f'{OUT} exists; the split is frozen'
OUT.write_text(json.dumps({
    'source': 'https://github.com/djghosh13/geneval/blob/af4902f24d3ca90ebbb446dd9891a59e0f82725f/prompts/evaluation_metadata.jsonl',
    'metadata_sha256': hashlib.sha256(META.read_bytes()).hexdigest(),
    'seed': SEED, 'rule': 'stratified by tag, 50/50; calibration subset = 16 per tag from the calibration half',
    'heldout_policy': 'sealed: do not generate or score before Week 10 (2026-11-30)',
    'images_per_prompt': 4, 'image_seeds': [0, 1, 2, 3],
    'counts': {'calibration': len(calibration), 'heldout': len(heldout), 'calibration_subset': len(subset)},
    'calibration': calibration, 'heldout': heldout, 'calibration_subset': sorted(subset),
}, indent=1) + '\n')
print(OUT, len(calibration), len(heldout), len(subset))
