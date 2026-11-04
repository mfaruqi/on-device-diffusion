"""Explicit fake activation quantization for numerical pipeline experiments."""

import torch


class ActivationQuantization:
    """Scoped hooks on named linears; floating GEMMs remain unchanged."""

    def __init__(self, transformer, names, bits=8, group_size=None):
        if bits not in (4, 8):
            raise ValueError("Expected 4 or 8 activation bits")
        if group_size is not None and (type(group_size) is not int or group_size <= 0):
            raise ValueError("Activation group size must be a positive integer")
        modules = dict(transformer.named_modules())
        if not names or len(set(names)) != len(names):
            raise ValueError("Require unique module names")
        for name in names:
            if name not in modules or not isinstance(modules[name], torch.nn.Linear):
                raise ValueError(f"Not a supported linear: {name}")
            if group_size is not None and modules[name].in_features % group_size:
                raise ValueError(f"Activation group size does not divide input channels: {name}")
        self.modules = [modules[n] for n in names]
        self.module_names = {modules[n]: n for n in names}
        self.calls_by_module = {n: 0 for n in names}
        self.bits = bits
        self.group_size = group_size
        self.handles = []
        self.calls = 0

    def hook(self, module, args):
        x = args[0]
        bound = 2 ** (self.bits - 1) - 1
        # Scales are computed in FP32; quantized values return to the original dtype.
        value = x.float()
        if self.group_size is not None:
            value = value.reshape(*x.shape[:-1], x.shape[-1] // self.group_size, self.group_size)
        maximum = value.abs().amax(dim=-1, keepdim=True)
        scale = torch.where(maximum == 0, torch.ones_like(maximum), maximum / bound)
        quantized = (value / scale).round().clamp(-bound, bound)
        self.calls += 1
        self.calls_by_module[self.module_names[module]] += 1
        return ((quantized * scale).reshape(x.shape).to(x.dtype),) + args[1:]

    def __enter__(self):
        self.handles = [m.register_forward_pre_hook(self.hook) for m in self.modules]
        return self

    def __exit__(self, *args):
        for handle in self.handles:
            handle.remove()
        self.handles = []
