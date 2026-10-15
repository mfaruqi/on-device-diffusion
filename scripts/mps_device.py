"""MPS event timing without per-stage synchronization; shared-memory diagnostics."""
import os
import time

from benchlib import GIB, host_environment, sh
from sdcpp_device import MacMemorySampler
from torch_stages import StageRecorder


class MPSStageRecorder(StageRecorder):
    """Reuse pipeline hooks, replacing CUDA events and allocator peak accounting."""
    def reset(self):
        self.stages = []
        self.scheduler_steps = 0
        self._open = None

    def start(self, name):
        assert self._open is None, 'Nested stages are unsupported'
        ev = self.torch.mps.Event(enable_timing=True)
        ev.record()
        self._open = {'stage': name, 'ev0': ev, 'host0': time.perf_counter(),
                      'alloc_start_gib': self.torch.mps.current_allocated_memory() / GIB}

    def end(self):
        s = self._open
        ev = self.torch.mps.Event(enable_timing=True)
        ev.record()
        s.update(ev1=ev, host1=time.perf_counter(),
                 alloc_end_gib=self.torch.mps.current_allocated_memory() / GIB)
        self.stages.append(s)
        self._open = None

    def finish(self):
        assert self._open is None
        rows = [{'stage': s['stage'], 'gpu_ms': s['ev0'].elapsed_time(s['ev1']),
                 'host_ms': (s['host1'] - s['host0']) * 1000,
                 'alloc_start_gib': s['alloc_start_gib'], 'alloc_end_gib': s['alloc_end_gib'],
                 'peak_alloc_gib': None, 'peak_reserved_gib': None} for s in self.stages]
        return rows, None, None, []


class MPSDevice:
    def __init__(self, cfg, out):
        import torch
        import psutil
        assert torch.backends.mps.is_available(), 'Metal MPS device unavailable'
        assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', '0') == '0', 'CPU fallback is forbidden'
        assert sh(['sysctl', '-n', 'machdep.cpu.brand_string']) == cfg['device_requirements']['chip']
        assert psutil.virtual_memory().total == cfg['device_requirements']['memory_gib'] * GIB
        self.sampler = MacMemorySampler(cfg['protocol']['device_memory_sample_ms'], out)

    def environment(self):
        import torch
        import diffusers
        import transformers
        env = host_environment(extra_env_vars=['HF_HOME', 'HF_HUB_OFFLINE', 'PYTORCH_ENABLE_MPS_FALLBACK'])
        env.pop('nvidia_smi_gpu', None)
        env.pop('nvidia_driver', None)
        env.update(torch=torch.__version__, diffusers=diffusers.__version__, transformers=transformers.__version__,
                   backend='mps', macos=sh(['sw_vers']), power_source=sh(['pmset', '-g', 'batt']),
                   recommended_max_memory_gib=torch.mps.recommended_max_memory()/GIB,
                   memory_scope=self.sampler.note, host_load_average=os.getloadavg())
        return env
