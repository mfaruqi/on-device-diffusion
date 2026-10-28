"""Explicit MPS compiler and upstream TaylorSeer options with execution receipts."""
from collections import Counter


COMPILE = {'backend': 'inductor', 'mode': 'default', 'fullgraph': True,
           'dynamic': False, 'target': 'transformer.forward'}
TAYLOR = {'method': 'taylorseer', 'cache_interval': 5, 'disable_cache_before_step': 3,
          'disable_cache_after_step': None, 'max_order': 1,
          'taylor_factors_dtype': 'bfloat16', 'use_lite_mode': False,
          'cache_identifiers': ['^transformer_blocks.*attn', '^single_transformer_blocks.*attn']}


def check_options(opts):
    compile_cfg = opts.get('torch_compile', False)
    cache_cfg = opts.get('step_cache')
    assert compile_cfg is False or compile_cfg == COMPILE, 'Unsupported explicit compile configuration'
    assert cache_cfg is None or cache_cfg == TAYLOR, 'Unsupported step cache configuration'
    assert not (compile_cfg and cache_cfg), 'Compile/cache combinations are not part of this batch'


def install_compile(transformer, cfg, verification_path=None):
    import torch
    import torch._dynamo
    from torch._inductor import metrics
    torch._dynamo.config.suppress_errors = False
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    metrics.reset()
    original = transformer.forward
    compiled = torch.compile(original, **{k: v for k, v in cfg.items() if k != 'target'})
    if verification_path is None:
        transformer.forward = compiled
        return
    checked = False
    def checked_forward(*args, **kwargs):
        nonlocal checked
        if checked:
            return compiled(*args, **kwargs)
        from benchlib import write_json
        eager = original(*args, **kwargs)
        actual = compiled(*args, **kwargs)
        torch.mps.synchronize()
        # Pipeline requests return_dict=False; compare identical inputs and weights.
        assert isinstance(eager, tuple) and isinstance(actual, tuple) and len(eager) == len(actual)
        errors = []
        for a, b in zip(actual, eager):
            assert a.shape == b.shape and a.dtype == b.dtype
            delta = a.float()-b.float()
            relative_rmse = (delta.square().mean().sqrt() / b.float().square().mean().sqrt().clamp_min(1e-8)).item()
            errors.append(dict(relative_rmse=relative_rmse, max_abs_error=delta.abs().max().item(),
                               finite=bool(torch.isfinite(a).all().item())))
        passed = all(e['finite'] and e['relative_rmse'] <= 0.01 for e in errors)
        write_json(verification_path, dict(passed=passed, outputs=errors, relative_rmse_limit=0.01,
            scope='First transformer call, identical inputs; engineering numerical gate, not final-image quality approval.',
            first_image_includes_extra_eager_forward=True))
        assert passed, 'Compiled transformer numerical gate failed'
        checked = True
        return actual
    transformer.forward = checked_forward


def compile_receipt():
    import torch._dynamo
    from torch._inductor import metrics
    return {'counters': {k: dict(v) for k, v in torch._dynamo.utils.counters.items()},
            'generated_kernel_count': metrics.generated_kernel_count,
            'fallback_allowed': False, 'first_image_includes_lazy_compilation': True}


class TaylorAudit:
    def __init__(self, transformer, cfg):
        import torch
        from diffusers import TaylorSeerCacheConfig
        self.transformer = transformer
        self.events = []
        self.generations = []
        self.names = []
        params = {k: v for k, v in cfg.items() if k != 'method'}
        params['taylor_factors_dtype'] = torch.bfloat16
        transformer.enable_cache(TaylorSeerCacheConfig(**params))
        for name, module in transformer.named_modules():
            registry = getattr(module, '_diffusers_hook', None)
            hook = registry.hooks.get('taylorseer_cache') if registry else None
            if hook is None:
                continue
            self.names.append(name)
            original = hook._measure_should_compute
            def measure(original=original, hook=hook, name=name):
                compute, state = original()
                self.events.append((name, hook.state_manager._current_context, state.current_step, bool(compute)))
                return compute, state
            hook._measure_should_compute = measure
        assert len(self.names) == 25, f'Expected 25 Klein attention hooks, found {self.names}'

    def begin(self):
        # Per-image state must never leak across prompt/seed generations.
        self.transformer._reset_stateful_cache()
        self.events = []

    def end(self, steps, cfg_passes):
        expected_contexts = {'cond', 'uncond'} if cfg_passes == 2 else {'cond'}
        assert {e[1] for e in self.events} == expected_contexts
        counts = Counter()
        for name in self.names:
            for context in expected_contexts:
                events = [e for e in self.events if e[0] == name and e[1] == context]
                assert [e[2] for e in events] == list(range(steps)), 'Cache reset or CFG context mismatch'
                assert events[0][3], 'First step must compute'
                counts.update('compute' if e[3] else 'predict' for e in events)
        assert counts['predict'] > 0, 'Cache requested but no predictions executed'
        self.generations.append(dict(counts, contexts=sorted(expected_contexts), steps=steps))

    def receipt(self):
        return {'implementation': 'diffusers.hooks.taylorseer_cache', 'hooked_modules': self.names,
                'generations': self.generations, 'formal_quality_eligible': False,
                'scope': 'Attention-output prediction, not whole-transformer step skipping'}
