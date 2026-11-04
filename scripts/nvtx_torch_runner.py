#!/usr/bin/env python3
"""run_flux_portable.py with NVTX ranges, for Nsight Systems captures of the PyTorch pipeline.

  nsys profile --trace=cuda,nvtx,osrt ... python scripts/nvtx_torch_runner.py --config ... --kind profile

Adds ranges without changing execution: `generation` around each pipe call, the recorder's own stage
names (`text_encode`, `denoise_step_<i>`, `vae_decode`, `postprocess`), and `text_encoder_load` /
`transformer_load` around each stage-residency reload. Arguments go to run_flux_portable unchanged;
torch.profiler (--profile) is not combined with Nsight. NVTX calls are no-ops without an attached tool.
"""
import sys

import flux_engine
import run_flux_portable
import torch
from torch_stages import StageRecorder

_start, _end, _generate = StageRecorder.start, StageRecorder.end, flux_engine.generate_once
_residency = run_flux_portable.apply_stage_residency


def start(self, name):
    _start(self, name)
    torch.cuda.nvtx.range_push(name)


def end(self):
    torch.cuda.nvtx.range_pop()
    _end(self)


def generate_once(pipe, recorder, w):
    torch.cuda.nvtx.range_push('generation')
    try:
        return _generate(pipe, recorder, w)
    finally:
        torch.cuda.nvtx.range_pop()


def labelled(name, load):
    def wrapper(*args, **kwargs):
        torch.cuda.nvtx.range_push(name)
        try:
            return load(*args, **kwargs)
        finally:
            torch.cuda.nvtx.range_pop()
    return wrapper


def apply_stage_residency(pipe, residency, loaders, device, on_reload=None):
    loaders = {k: labelled(f'{k}_load', v) for k, v in loaders.items()}
    return _residency(pipe, residency, loaders, device, on_reload)


if __name__ == '__main__':
    assert '--profile' not in sys.argv, 'torch.profiler and Nsight both use CUPTI; capture with Nsight only'
    StageRecorder.start, StageRecorder.end = start, end
    flux_engine.generate_once = generate_once
    run_flux_portable.apply_stage_residency = apply_stage_residency
    run_flux_portable.main()
