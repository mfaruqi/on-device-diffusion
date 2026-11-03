"""Bounded calibration activation capture; instrumentation invalidates timing."""

import numpy as np
import torch


class ActivationCapture:
    def __init__(self, transformer, names, tokens_per_call=32, max_calls=16):
        if tokens_per_call < 1 or max_calls < 1 or not names or len(set(names)) != len(names):
            raise ValueError("Require positive bounds and unique named modules")
        modules = dict(transformer.named_modules())
        for name in names:
            if name not in modules or not isinstance(modules[name], torch.nn.Linear):
                raise ValueError(f"Not a linear: {name}")
        self.modules = {name: modules[name] for name in names}
        self.tokens_per_call = tokens_per_call
        self.max_calls = max_calls
        self.samples = {name: [] for name in names}
        self.handles = []

    def hook(self, name):
        def collect(module, args):
            if len(self.samples[name]) >= self.max_calls:
                raise RuntimeError(f"Capture call budget exceeded: {name}")
            value = args[0].detach().reshape(-1, args[0].shape[-1])
            count = min(len(value), self.tokens_per_call)
            indices = torch.arange(count, device=value.device) * len(value) // count
            # CPU transfer is intentionally diagnostic, outside any benchmark claim.
            sample = value[indices].float().cpu().numpy().copy()
            if not np.isfinite(sample).all():
                raise ValueError(f"Nonfinite activations: {name}")
            self.samples[name].append(sample)

        return collect

    def __enter__(self):
        self.handles = [
            module.register_forward_pre_hook(self.hook(name))
            for name, module in self.modules.items()
        ]
        return self

    def __exit__(self, *args):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def save(self, path):
        arrays = {
            name + f".call_{i:03d}": sample
            for name, samples in self.samples.items()
            for i, sample in enumerate(samples)
        }
        if not arrays:
            raise ValueError("No activations captured")
        np.savez(path, **arrays)
        return {
            name: {
                "calls": len(samples),
                "tokens_per_call": [len(x) for x in samples],
                "channels": samples[0].shape[-1] if samples else None,
            }
            for name, samples in self.samples.items()
        }
