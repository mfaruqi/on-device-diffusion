"""Pinned Klein MLX adapter: explicit precision audit and synchronized host stages."""
import hashlib
import json
import math
import re
import time
from pathlib import Path

from benchlib import GIB, REPO_ROOT, write_json
from measurement import check_optimizations, phase_sequence

COMMIT = 'f0bf9cf0506a2ae15115af1c72e3dc0093e41597'
REVISIONS = {
    'black-forest-labs/FLUX.2-klein-4B': 'e7b7dc27f91deacad38e78976d1f2b499d76a294',
    'black-forest-labs/FLUX.2-klein-base-4B': 'a3b4f4849157f664bdbc776fd7453c2783562f4d',
}
TIMING = ('Synchronized host stage timing (not GPU events): prompt pair, each denoise step '
          'including CFG and scheduler, VAE decode. MLX lazy outputs are evaluated at boundaries. '
          'Native predict compilation retained; synchronization overhead is included.')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def check_config(cfg):
    assert cfg['engine']['name'] == 'mflux' and cfg['engine']['commit'] == COMMIT
    assert cfg['backend'] == 'mlx-metal' and cfg['precision'] == 'bfloat16'
    m, w = cfg['model'], cfg['workload']
    assert m['revision'] == REVISIONS[m['repo_id']]
    base = '-base-' in m['repo_id']
    assert w['num_inference_steps'] == (50 if base else 4)
    assert w['guidance_scale'] == (4.0 if base else 1.0)
    assert w['negative_prompt'] == '' and w['max_sequence_length'] == 512
    assert (w['width'], w['height'], w['batch_size']) == (512, 512, 1)
    check_optimizations(cfg, {'quantization', 'conditioning_reuse', 'vae_tiling', 'offload'},
                       {'quantization', 'conditioning_reuse', 'vae_tiling'})
    assert cfg['optimizations']['quantization'] in (None, 4, 8)
    assert type(cfg['optimizations'].get('conditioning_reuse', False)) is bool
    tiling = cfg['optimizations'].get('vae_tiling', False)
    assert tiling is False or tiling == {'tile_size_pixels': 256, 'overlap_pixels': 64}
    assert not (tiling and cfg['optimizations'].get('conditioning_reuse')), 'Individual options only'
    assert cfg['engine_settings'] == {
        'quantization_scope': 'all_eligible_layers', 'group_size': 64,
        'scheduler': 'flow_match_euler_discrete', 'native_predict_compile': True,
        'negative_prompt_override': '', 'timing': 'synchronized_host',
    }
    p = cfg['protocol']
    phase_sequence(p)
    assert (p['first_runs'], p['warmup_runs'], p['measured_runs']) in ((1, 0, 1), (1, 3, 10))
    assert p['prevent_idle_sleep'] and p['device_memory_sample_ms'] == 100


def prepare(cfg, path):
    from huggingface_hub import snapshot_download
    check_config(cfg)
    root = Path(snapshot_download(cfg['model']['repo_id'], revision=cfg['model']['revision'],
        cache_dir=str(REPO_ROOT/'output/m5-prep/huggingface/hub'),
        allow_patterns=['vae/*', 'transformer/*', 'text_encoder/*', 'tokenizer/*',
                        'model_index.json', 'scheduler/*']))
    files = {str(p.relative_to(root)): {'sha256': sha256(p), 'bytes': p.stat().st_size}
             for p in sorted(root.rglob('*')) if p.is_file()}
    assert all(any(n.startswith(c+'/') and n.endswith('.safetensors') for n in files)
               for c in ('transformer', 'text_encoder', 'vae'))
    cfg['model']['local_path'] = str(root)
    cfg['model']['files'] = files
    write_json(path.with_suffix('.resolved.json'), cfg)


def tensor_audit(model, bits, out):
    import mlx.core as mx
    from mlx.utils import tree_flatten
    audit, errors = {}, []
    for name in ('text_encoder', 'transformer', 'vae'):
        component = getattr(model, name)
        arrays = tree_flatten(component.parameters())
        quantized = []
        for path, module in component.named_modules():
            if hasattr(module, 'bits') and hasattr(module, 'scales'):
                quantized.append({'path': path, 'class': type(module).__name__,
                                  'bits': module.bits, 'group_size': module.group_size,
                                  'mode': getattr(module, 'mode', 'affine')})
        floats = [(p, a) for p, a in arrays if mx.issubdtype(a.dtype, mx.floating)]
        unexpected = [p for p, a in floats if a.dtype != mx.bfloat16 and not (
            name == 'text_encoder' and p == 'rotary_emb.inv_freq'
            and a.dtype == mx.float32 and a.shape == (64,))]
        errors.extend(f'{name}.{p}: unexpected floating dtype' for p in unexpected)
        if not all(x['bits'] == bits and x['group_size'] == 64 for x in quantized):
            errors.append(name+': wrong quantization parameters')
        if bits is None:
            if quantized: errors.append(name+': unexpected quantization')
        elif name in ('transformer', 'text_encoder'):
            if not quantized: errors.append(name+' was not quantized')
        audit[name] = {'weights_gib': sum(a.nbytes for _, a in arrays)/GIB,
                       'quantized_layers': quantized, 'quantized_layer_count': len(quantized),
                       'tensors': [{'name': p, 'shape': list(a.shape), 'dtype': str(a.dtype)} for p, a in arrays]}
    write_json(out/'quantization-audit.json',audit)
    assert not errors, '; '.join(errors)
    assert model.bits == bits and model.tiling_config is None
    return audit


class HostStages:
    """Keep compiled predict intact; evaluate at the existing iteration callback boundary."""
    def __init__(self, mx, model, cfg, out):
        self.mx, self.model, self.cfg, self.out = mx, model, cfg, out
        self.rows = []
        self.step = 0
        self.prompt_result = None
        self.prompt_key = None
        self.prompt_hits = self.prompt_misses = 0
        self.decode_calls = []
        self.generation_decode_calls = []
        self.tiling = cfg['optimizations'].get('vae_tiling', False)
        if self.tiling:
            from mflux.models.common.vae.tiling_config import TilingConfig
            model.tiling_config = TilingConfig(vae_decode_tiles_per_dim=2,
                vae_decode_tile_size=256, vae_decode_overlap=8, vae_encode_tiled=False)
            native_decode = model.vae.decode
            def tile_decode(latent, *args, **kwargs):
                self.decode_calls.append(list(latent.shape))
                return native_decode(latent, *args, **kwargs)
            model.vae.decode = tile_decode
        encode = model._encode_prompt_pair
        decode = model.vae.decode_packed_latents

        def encode_pair(**kwargs):
            kwargs['negative_prompt'] = cfg['workload']['negative_prompt']
            def lookup():
                if cfg['optimizations'].get('conditioning_reuse'):
                    key = tuple(sorted(kwargs.items()))
                    if self.prompt_result is not None:
                        assert key == self.prompt_key, 'Single-conditioning cache key changed'
                        self.prompt_hits += 1
                        return self.prompt_result
                    self.prompt_result = encode(**kwargs)
                    self.prompt_key = key
                    self.prompt_misses += 1
                    return self.prompt_result
                return encode(**kwargs)
            result = self.timed('text_encode', lookup)
            expected = 2 if cfg['workload']['guidance_scale'] > 1 else 1
            assert (2 if result[2] is not None else 1) == expected
            self.prompt_shapes = [list(a.shape) if a is not None else None for a in result]
            return result

        def decode_packed(*args, **kwargs):
            assert kwargs.get('tiling_config') == model.tiling_config
            result = self.timed('vae_decode', lambda: decode(*args, **kwargs))
            if self.tiling:
                assert len(self.decode_calls) > 1, 'Tiling enabled but multiple tiles did not execute'
                self.generation_decode_calls.append(list(self.decode_calls))
            return result

        model._encode_prompt_pair = encode_pair
        model.vae.decode_packed_latents = decode_packed
        model.callbacks.register(self)

    def evaluate(self, value):
        if isinstance(value, self.mx.array):
            self.mx.eval(value)
        elif isinstance(value, (list, tuple)):
            self.mx.eval(*[x for x in value if isinstance(x, self.mx.array)])
        self.mx.synchronize()

    def append(self, name, start):
        self.rows.append({'stage': name, 'gpu_ms': None, 'host_ms': (time.perf_counter()-start)*1000,
                          'alloc_start_gib': None, 'alloc_end_gib': self.mx.get_active_memory()/GIB,
                          'peak_alloc_gib': None, 'peak_reserved_gib': None})

    def timed(self, name, fn):
        self.mx.synchronize()
        start = time.perf_counter()
        value = fn()
        self.evaluate(value)
        self.append(name, start)
        return value

    def call_before_loop(self, latents, config, **kwargs):
        self.evaluate(latents)
        self.step = 0
        self.step_start = time.perf_counter()
        self.scheduler = config.scheduler

    def call_in_loop(self, t, latents, **kwargs):
        assert int(t) == self.step
        self.evaluate(latents)
        self.append(f'denoise_step_{self.step}', self.step_start)
        self.step += 1
        self.step_start = time.perf_counter()

    def call_after_loop(self, **kwargs):
        assert self.step == self.cfg['workload']['num_inference_steps']

    def generate(self):
        self.rows = []
        self.decode_calls = []
        self.mx.synchronize()
        self.mx.reset_peak_memory()
        start = time.perf_counter()
        w = self.cfg['workload']
        output = self.model.generate_image(seed=w['seed'], prompt=w['prompt'],
            num_inference_steps=w['num_inference_steps'], width=w['width'], height=w['height'],
            guidance=w['guidance_scale'], scheduler=self.cfg['engine_settings']['scheduler'])
        self.mx.synchronize()
        wall = (time.perf_counter()-start)*1000
        assert [r['stage'] for r in self.rows] == ['text_encode',
            *[f'denoise_step_{i}' for i in range(w['num_inference_steps'])], 'vae_decode']
        times = {r['stage']: r['host_ms'] for r in self.rows}
        times['denoise'] = sum(v for k, v in times.items() if k.startswith('denoise_step_'))
        assert wall >= times['text_encode'] + times['denoise'] + times['vae_decode']
        return output.image, wall, times, self.mx.get_peak_memory()/GIB


def validate_run(directory, cfg, allow_running=False):
    """Functional/protocol gate only. Quantization quality eligibility is never implied."""
    import csv
    from PIL import Image, ImageStat
    directory = Path(directory)
    recorded = json.loads((directory/'config.json').read_text())
    for key in ('id', 'model', 'precision', 'workload', 'optimizations', 'engine', 'engine_settings'):
        assert recorded[key] == cfg[key], 'Gate configuration mismatch: '+key
    assert json.loads((directory/'status.json').read_text())['status'] in (
        ('running', 'complete') if allow_running else ('complete',))
    assert json.loads((directory/'summary.json').read_text())['clock_continuity_pass']
    rows = list(csv.DictReader((directory/'runs.csv').open()))
    assert [r['phase'] for r in rows] == phase_sequence(recorded['protocol'])
    diagnostics = []
    for r in rows:
        for k in ('wall_ms', 'text_encode_ms', 'denoise_ms', 'vae_decode_ms'):
            assert math.isfinite(float(r[k])) and float(r[k]) > 0
        path = directory/'images'/f"{int(r['run_index']):02d}-{r['phase']}.png"
        with Image.open(path) as image:
            assert image.size == (cfg['workload']['width'], cfg['workload']['height'])
            assert hashlib.sha256(image.convert('RGB').tobytes()).hexdigest() == r['image_sha256']
            std = ImageStat.Stat(image.convert('RGB')).stddev
            assert max(std) > 1, 'Flat/invalid image'
            diagnostics.append({'file': str(path.relative_to(directory)), 'stddev': std})
    result = {'passed': True, 'phase_counts': {p: sum(r['phase']==p for r in rows)
              for p in ('first','warmup','measured')}, 'image_diagnostics': diagnostics,
              'formal_quality_eligible': False}
    write_json(directory/'validation.json', result)
    return result
