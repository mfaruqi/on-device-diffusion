#!/usr/bin/env python3
"""Rooflines from immutable completed FLUX.2 klein 4B artifacts; no model execution.

Requires plotly; --upload requires wandb; --static requires kaleido + Chrome.
python scripts/plot_roofline.py --output output/roofline --upload
Manifest: {"runs": [{"run_dir": "...", "label": "..."}], "hardware_specs": {...}}.
Optional run overrides: weight_gib, activation_bytes, ceiling. Hardware overrides
are explicit JSON dictionaries via --hardware-specs or the manifest.
"""
import argparse
import copy
import csv
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import re
import statistics
import textwrap
import uuid

ROOT = Path(__file__).resolve().parents[1]
A100_SOURCE = 'https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet.pdf'
JETSON_SOURCE = 'https://developer.nvidia.com/blog/nvidia-jetpack-6-2-brings-super-mode-to-nvidia-jetson-orin-nano-and-jetson-orin-nx-modules/'
MODEL_SOURCE = 'https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/blob/e7b7dc27f91deacad38e78976d1f2b499d76a294/transformer/config.json'
HARDWARE_SPECS = {
    'gilbreth-a100-40gb': {
        'label': 'A100 PCIe 40GB', 'bw_gbs': 1555.0,
        'ceilings': {'bf16_tensor': 312.0, 'fp16_tensor': 312.0,
                     'fp32': 19.5, 'int4_tensor_tops': 1248.0},
        'source': A100_SOURCE, 'note': 'Dense theoretical ceilings; INT4 TOP/s is not floating-point throughput.'},
    'jetson-orin-nano': {
        'label': 'Jetson Orin Nano 8GB · 25W', 'bw_gbs': 102.0,
        'ceilings': {'fp16_dense': 17.0, 'int8_dense_tops': 33.0},
        'source': JETSON_SOURCE,
        'note': 'Super-mode theoretical ceilings, not measured sustained bandwidth/clocks. Q4 storage uses floating-point compute.'},
    'macbook-air-m5': {
        'label': 'MacBook Air M5', 'bw_gbs': 153.0,
        'ceilings': {'bf16_fp16_gpu': 18.0},
        'source': 'https://www.apple.com/macbook-air/specs/',
        'note': '153 GB/s Apple theoretical spec; 18 TFLOP/s user-supplied reference assumption, not an Apple published or measured ceiling.'},
}
METHOD = (
    'Analytic tensor traffic, not PMU DRAM counters. GEMM: 2MNK FLOPs, '
    'MK*b_a + KN*b_w + MN*b_a bytes. Flash attention: 4BHSqSkD FLOPs, '
    '2BH(Sq+Sk)D*b_a bytes, no score materialization. Stage numerator is dominant '
    'linear + attention FLOPs of a reference klein 4B pass times executed CFG passes; '
    'excludes elementwise, normalization, positional encoding, modulation, dequantization '
    'and low-rank correction overhead. Stage bytes = one transformer weight read per '
    'pass + ideal operator activation reads/writes; assumes no inter-operator cache reuse. '
    'Operator throughput uses summed associated kernel durations, not stage host time '
    'or GPU busy-union time. sd.cpp inferred attention assumes one flash call per block. '
    'Stage effective throughput uses measured-generation median denoising time including '
    'host callback reloads for sd.cpp, CUDA/MPS events for PyTorch. Profile samples are '
    'separate instrumented generations. First/warmups excluded from baseline medians. '
    'Q4 storage does not imply INT4 compute. Mixed Nunchaku stage normalization uses '
    'BF16 as a reference ceiling, not an assertion of its mixed-kernel utilization. '
    'No-cache and EasyCache use executed work (not skipped reference FLOPs); approximate '
    'cache hits still incur uncounted overhead. Cross-device workloads/precision/residency '
    'differ; quality equivalence unassessed. Percent of theoretical roof is descriptive, '
    'not measured hardware utilization; above 100% is retained as a model discrepancy.'
)
FOOTER = (
    'Analytic bytes ≠ measured DRAM traffic; stage rates count reference linear/attention work only.'
    '<br>Kernel time ≠ host stage time; sd.cpp stages include reloads. Profile samples are instrumented.'
    '<br>Workloads, precision and residency differ; quality equivalence unassessed. Q4 ≠ INT4 compute.'
)
PROFILE_FILES = ['profile/denoise_kernels.json', 'profile/stage_kernels.json',
                 'profile/osrt-analysis/stage_kernels_osrt.json',
                 'profile/osrt-analysis/stage_kernels.json']
REQUIRED = ['summary.json', 'runs.csv', 'stages.csv', 'config.json', 'status.json']
TABLE_COLUMNS = ['point_id', 'source_run', 'label', 'device', 'engine', 'precision',
                 'resolution', 'tier', 'stage', 'shape', 'flops', 'analytic_bytes',
                 'arithmetic_intensity', 'achieved_tflops', 'roof_tops', 'roof_pct',
                 'ceiling', 'duration_ms', 'passes', 'weight_basis', 'timing_basis',
                 'source_file', 'source_sha256']


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def positive(value, name):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be finite and positive')
    return value


def gemm_cost(m, n, k, activation_bytes=2, weight_bytes=2):
    m, n, k = [positive(v, 'GEMM dimension') for v in (m, n, k)]
    a, w = positive(activation_bytes, 'activation bytes'), positive(weight_bytes, 'weight bytes')
    return 2*m*n*k, (m*k+m*n)*a+k*n*w


def attention_cost(b, h, sq, sk, d, activation_bytes=2):
    b, h, sq, sk, d = [positive(v, 'attention dimension') for v in (b, h, sq, sk, d)]
    return 4*b*h*sq*sk*d, 2*b*h*(sq+sk)*d*positive(activation_bytes, 'activation bytes')


def transformer_cost(width, height, weight_gib, activation_bytes=2, text_tokens=512):
    """Dominant reference ops: D=3072, 5 dual + 20 single blocks, SwiGLU r=3.

    Dual: QKV + output + (D -> 6D -> 3D -> D) for both streams.
    Single: fused D -> 9D, fused 4D -> D. Add input/output projections.
    Weight traffic uses recorded footprint once per pass, not per-operator sum.
    """
    if width % 16 or height % 16 or min(width, height, text_tokens) <= 0:
        raise ValueError('Positive dimensions divisible by 16 and positive text length required')
    image = (width//16)*(height//16)
    seq, d = image+text_tokens, 3072
    ops = [(image, d, 128, 1), (text_tokens, d, 7680, 1), (image, 128, d, 1)]
    for s in (image, text_tokens):
        ops.extend([(s, 3*d, d, 5), (s, d, d, 5),
                    (s, 6*d, d, 5), (s, d, 3*d, 5)])
    ops.extend([(seq, 9*d, d, 20), (seq, d, 4*d, 20)])
    flops, activation = 0, 0
    for m, n, k, calls in ops:
        f, _ = gemm_cost(m, n, k, activation_bytes)
        flops += calls*f
        activation += calls*(m*k+m*n)*activation_bytes
    f, a = attention_cost(1, 24, seq, seq, 128, activation_bytes)
    return flops+25*f, positive(weight_gib, 'weight GiB')*2**30+activation+25*a


def hardware_specs(overrides=None):
    specs = copy.deepcopy(HARDWARE_SPECS)
    for device, override in (overrides or {}).items():
        if device not in specs:
            raise ValueError(f'Unknown hardware: {device}')
        specs[device].update({k: v for k, v in override.items() if k != 'ceilings'})
        specs[device]['ceilings'].update(override.get('ceilings', {}))
    for spec in specs.values():
        positive(spec['bw_gbs'], 'bandwidth')
        for value in spec['ceilings'].values():
            positive(value, 'ceiling')
    return specs


def executed_passes(config, summary, rows):
    workload = config['workload']
    steps = int(workload.get('num_inference_steps', workload.get('steps', 0)))
    guidance = workload.get('guidance_scale', workload.get('cfg_scale', 1))
    cfg = 2 if guidance > 1 else 1
    if steps <= 0:
        raise ValueError('Invalid step count')
    if not config.get('optimizations', {}).get('step_cache'):
        return [steps*cfg]*len(rows)
    skips = summary.get('cache_audit', {}).get('steps_skipped')
    if not isinstance(skips, list):
        raise ValueError('EasyCache requires saved per-generation skip audit')
    result = []
    for row in rows:
        index = int(row['run_index'])
        if index >= len(skips) or not isinstance(skips[index], int) or not 0 <= skips[index] < steps:
            raise ValueError('Invalid or incomplete cache skip audit')
        result.append((steps-skips[index])*cfg)
    return result


def device_for(config, summary, environment):
    label = config.get('device_label', '')
    if 'a100' in label:
        gpu = summary.get('gpu_name') or environment.get('gpu_name') or environment.get('nvidia_smi_gpu', '')
        if 'PCIE' not in gpu.upper():
            raise ValueError(f'A100 PCIe chart requires verified PCIe hardware, got {gpu!r}')
        return 'gilbreth-a100-40gb'
    if label == 'jetson-orin-nano':
        return label
    if label.startswith('macbook-air-m5'):
        return 'macbook-air-m5'
    raise ValueError(f'Unsupported device {label}')


def precision_for(config, name):
    if 'nunchaku' in name:
        return 'W4A4/r32 (mixed BF16)', 2, .5
    quant = config.get('optimizations', {}).get('quantization')
    if quant and 'Q4_0' in json.dumps(quant).upper():
        return 'Mixed Q4_0/BF16', 2, .5625
    dtype = config.get('precision', '')
    if dtype not in ('bfloat16', 'bf16', 'float16', 'fp16', 'float32', 'fp32'):
        raise ValueError(f'Unsupported precision {dtype!r}')
    return dtype, 4 if '32' in dtype else 2, 4 if '32' in dtype else 2


def point(record, specs, tier, stage, shape, flops, byte_count, duration, ceiling, **extra):
    flops, byte_count, duration = [positive(v, 'point coordinate') for v in (flops, byte_count, duration)]
    ai = flops/byte_count
    achieved = extra.pop('achieved_tflops', flops/duration/1e9)
    positive(achieved, 'throughput')
    spec = specs[record['device']]
    roof = min(spec['ceilings'][ceiling], spec['bw_gbs']/1000*ai)
    return {**{k: record[k] for k in ['source_run', 'label', 'device', 'engine', 'precision', 'resolution', 'source_sha256']},
            'point_id': f"{record['source_run']}:{tier}:{stage}:{shape}",
            'tier': tier, 'stage': stage, 'shape': shape,
            'flops': flops, 'analytic_bytes': byte_count, 'arithmetic_intensity': ai,
            'duration_ms': duration, 'achieved_tflops': achieved, 'roof_tops': roof,
            'roof_pct': 100*achieved/roof, 'ceiling': ceiling, **extra}


def collect(root, manifest=None, overrides=None):
    root = Path(root).resolve()
    base = root/'results'/'runs'
    specs_override = copy.deepcopy((manifest or {}).get('hardware_specs', {}))
    for key, value in (overrides or {}).items():
        specs_override.setdefault(key, {}).update(value)
    specs = hardware_specs(specs_override)
    auto = manifest is None
    selections = (manifest or {}).get('runs', [dict(run_dir=p.name) for p in sorted(base.iterdir()) if p.is_dir()])
    if len({s['run_dir'] for s in selections}) != len(selections):
        raise ValueError('Duplicate source run in manifest')
    records, points, exclusions = [], [], list((manifest or {}).get('exclusions', []))
    for selection in selections:
        name = selection['run_dir']
        directory = (base/name).resolve()
        if directory.parent != base.resolve():
            raise ValueError('Source run must be a direct child of results/runs')
        if auto and not any(f'__{kind}__' in name for kind in ('baseline', 'repeat', 'profile')):
            exclusions.append({'run_dir': name, 'reason': 'not baseline/repeat/profile'})
            continue
        hashes = {}
        def read_file(relative):
            path = directory/relative
            blob = path.read_bytes()
            hashes[relative] = hashlib.sha256(blob).hexdigest()
            return blob.decode()
        try:
            status = json.loads(read_file('status.json'))
            if status['status'] != 'complete':
                raise ValueError('status is not complete')
            config = json.loads(read_file('config.json'))
            model = json.dumps(config.get('model', {})).lower()
            if 'klein' not in model or '4b' not in model:
                raise ValueError('Only the FLUX.2 klein 4B architecture is supported')
            if 'mflux' in name:
                raise ValueError('MLX/mflux is outside the supported PyTorch/sd.cpp/edge-dit/Nunchaku timing scopes')
            summary = json.loads(read_file('summary.json'))
            env = json.loads(read_file('environment.json')) if (directory/'environment.json').exists() else {}
            device = device_for(config, summary, env)
        except (ValueError, KeyError, FileNotFoundError) as error:
            if not auto:
                raise ValueError(f'{name}: {error}') from error
            exclusions.append({'run_dir': name, 'reason': str(error), 'source_sha256': hashes})
            continue
        # All selected completed inputs are hashed, including CSVs in profile runs.
        rows_all = list(csv.DictReader(read_file('runs.csv').splitlines()))
        stages = list(csv.DictReader(read_file('stages.csv').splitlines()))
        profile_data = {f: json.loads(read_file(f)) for f in PROFILE_FILES if (directory/f).exists()}
        load = json.loads(read_file('load.json')) if (directory/'load.json').exists() else {}
        precision, a_bytes, w_bytes = precision_for(config, name)
        a_bytes = positive(selection.get('activation_bytes', a_bytes), 'activation bytes')
        engine = config.get('engine', {}).get('name', 'PyTorch / Diffusers')
        if 'nunchaku' in name:
            engine = 'Nunchaku / SVDQuant'
        workload = config['workload']
        width, height = int(workload['width']), int(workload['height'])
        if workload.get('batch_size', 1) != 1 or workload.get('batch_count', 1) != 1:
            raise ValueError(f'Only batch one is supported: {name}')
        text = int(workload.get('max_sequence_length', 512))
        record = {'source_run': name, 'label': selection.get('label', config.get('id', name)),
                  'device': device, 'engine': engine, 'precision': precision,
                  'resolution': f'{width}x{height}', 'source_sha256': hashes}
        default_ceiling = {'gilbreth-a100-40gb': 'bf16_tensor', 'jetson-orin-nano': 'fp16_dense',
                           'macbook-air-m5': 'bf16_fp16_gpu'}[device]
        if a_bytes == 4 and device == 'gilbreth-a100-40gb':
            default_ceiling = 'fp32'
        ceiling = selection.get('ceiling', default_ceiling)
        if ceiling not in specs[device]['ceilings']:
            raise ValueError(f'Unknown ceiling {ceiling}')
        if '__profile__' not in name:
            measured = [r for r in rows_all if r['phase'] == 'measured']
            expected = config['protocol']['measured_runs']
            if len(measured) != expected or summary['measured_runs'] != expected or expected != 10:
                raise ValueError(f'Expected ten measured generations: {name}')
            times = [positive(r['denoise_ms'], 'denoising time') for r in measured]
            median = statistics.median(times)
            if not math.isclose(median, summary['measured']['denoise_ms']['median'], rel_tol=1e-7):
                raise ValueError(f'CSV/summary denoising median mismatch: {name}')
            weight = summary.get('weights_gib', {}).get('transformer')
            if weight is None:
                weight = load.get('components', {}).get('transformer', {}).get('weights_gib')
            weight_basis = 'recorded transformer footprint'
            if selection.get('weight_gib') is not None:
                weight, weight_basis = selection['weight_gib'], 'explicit manifest estimate'
            elif weight is None:
                weight, weight_basis = (2.15 if w_bytes < 1 else 7.218764305114746), 'reference estimate; not recorded in this run'
            flops, traffic = transformer_cost(width, height, weight, a_bytes, text)
            passes = executed_passes(config, summary, measured)
            # Fixed work uses the benchmark median duration; variable skips need paired rates.
            rate = (flops*passes[0]/median/1e9 if len(set(passes)) == 1 else
                    statistics.median(flops*p/t/1e9 for p, t in zip(passes, times)))
            timing = summary.get('timing_method', 'CUDA/MPS device events' if engine.startswith(('PyTorch', 'Nunchaku')) else 'host stage callbacks')
            points.append(point(record, specs, 'stage', 'denoise', '', flops*statistics.median(passes),
                                traffic*statistics.median(passes), median, ceiling,
                                achieved_tflops=rate, passes=statistics.median(passes),
                                passes_per_generation=passes, weight_gib=weight, weight_basis=weight_basis,
                                timing_basis=timing, source_file='runs.csv + summary.json',
                                min_ms=min(times), max_ms=max(times)))
            if device == 'jetson-orin-nano' and config['id'] in ('jetson-flux-klein-003', 'jetson-flux-klein-base-001'):
                cfg = 2 if workload.get('guidance_scale', workload.get('cfg_scale', 1)) > 1 else 1
                # Pair first and steady timings within the same retained baseline; no borrowed profile timing.
                for stage, keys in [('denoise_step_0', ['denoise_step_0']),
                                    ('denoise_step_1..3', [f'denoise_step_{i}' for i in range(1, 4)])]:
                    durations = []
                    for row in measured:
                        matching = [s for s in stages if s['phase'] == 'measured' and s['run_index'] == row['run_index'] and s['stage'] in keys]
                        if len(matching) != len(keys):
                            raise ValueError(f'Incomplete Jetson stage split: {name}')
                        durations.append(statistics.mean(positive(s['host_ms'], 'host stage time') for s in matching))
                    duration = statistics.median(durations)
                    points.append(point(record, specs, 'step', stage, '', flops*cfg, traffic*cfg, duration,
                                        ceiling, passes=cfg, weight_gib=weight, weight_basis=weight_basis,
                                        timing_basis='median host callback step; steady = per-generation mean steps 1..3',
                                        source_file='stages.csv', min_ms=min(durations), max_ms=max(durations)))
        # Prefer one OSRT-enriched file per profile; hash all available files but avoid duplicate points.
        selected_profile = next((f for f in reversed(PROFILE_FILES) if f in profile_data), None)
        if selected_profile:
            for stage, info in profile_data[selected_profile]['stages'].items():
                if not stage.startswith('denoise_step_'):
                    continue
                for kind, formula, pattern in [
                    ('gemm_shapes', gemm_cost, r'M(\d+) N(\d+) K(\d+)'),
                    ('attention_shapes', attention_cost, r'B(\d+) H(\d+) Sq(\d+) Sk(\d+) D(\d+)')]:
                    for shape, entry in info.get(kind, {}).items():
                        match = re.fullmatch(pattern, shape)
                        if not match:
                            raise ValueError(f'Unknown shape: {shape}')
                        dims = [int(v) for v in match.groups()]
                        f, b = formula(*dims, a_bytes, w_bytes) if kind == 'gemm_shapes' else formula(*dims, 2)
                        calls, ms = positive(entry['calls'], 'calls'), positive(entry['ms'], 'kernel ms')
                        rate = f*calls/ms/1e9
                        if entry.get('tflops') is not None and not math.isclose(rate, entry['tflops'], rel_tol=1e-6):
                            raise ValueError(f'Profile FLOP/time mismatch: {shape}')
                        points.append(point(record, specs, 'operator', stage, shape, f, b, ms/calls, ceiling,
                                            achieved_tflops=rate, timing_basis='sum of associated GPU kernel durations / calls',
                                            source_file=selected_profile, calls=calls, weight_basis='analytic shape traffic'))
                attention = info.get('by_group', {}).get('attention', {})
                leaf = info.get('by_leaf_op', {}).get('flash_attn_ext_f16', {})
                if not info.get('attention_shapes') and attention and leaf:
                    if not math.isclose(attention['ms'], leaf['ms'], rel_tol=1e-6):
                        raise ValueError('Cannot assign entire attention group to flash_attn_ext_f16')
                    seq = (width//16)*(height//16)+text
                    f, b = attention_cost(1, 24, seq, seq, 128)
                    points.append(point(record, specs, 'operator-inferred', stage,
                                        f'B1 H24 Sq{seq} Sk{seq} D128', f, b,
                                        attention['ms']/positive(leaf['kernels'], 'flash calls'), ceiling,
                                        timing_basis='flash_attn_ext_f16 kernel sum; shape inferred from reference architecture',
                                        source_file=selected_profile, calls=leaf['kernels'], weight_basis='analytic F16 attention traffic'))
        records.append(record)
    return {'schema_version': 1, 'method': METHOD, 'model_source': MODEL_SOURCE,
            'hardware_specs': specs, 'runs': records, 'points': points, 'exclusions': exclusions}


def validate_data(data):
    if data.get('schema_version') != 1 or not data.get('points'):
        raise ValueError('Expected nonempty roofline schema version 1')
    for p in data['points']:
        for key in ['flops', 'analytic_bytes', 'duration_ms', 'achieved_tflops', 'arithmetic_intensity', 'roof_tops', 'roof_pct']:
            positive(p[key], key)
        if not math.isclose(p['arithmetic_intensity'], p['flops']/p['analytic_bytes'], rel_tol=1e-9):
            raise ValueError('Inconsistent arithmetic intensity')
        spec = data['hardware_specs'][p['device']]
        roof = min(spec['ceilings'][p['ceiling']], spec['bw_gbs']/1000*p['arithmetic_intensity'])
        if not math.isclose(roof, p['roof_tops'], rel_tol=1e-9) or not math.isclose(p['roof_pct'], 100*p['achieved_tflops']/roof, rel_tol=1e-9):
            raise ValueError('Inconsistent roof normalization')
        if not p['source_sha256'] or any(not re.fullmatch('[a-f0-9]{64}', v) for v in p['source_sha256'].values()):
            raise ValueError('Missing/invalid source provenance')
    if len({p['point_id'] for p in data['points']}) != len(data['points']):
        raise ValueError('Duplicate point identity')


ENGINE_STYLE = {
    'PyTorch / Diffusers': ('PyTorch', '#4C91C2', 'circle'),
    'stable-diffusion.cpp': ('sd.cpp', '#F29A38', 'square'),
    'edge-dit.cpp': ('edge-dit', '#78AE62', 'triangle-up'),
    'Nunchaku / SVDQuant': ('Nunchaku W4A4/r32', '#A57FC1', 'diamond'),
}
PRIMARY_CEILING = {'gilbreth-a100-40gb': 'bf16_tensor',
                   'jetson-orin-nano': 'fp16_dense', 'macbook-air-m5': 'bf16_fp16_gpu'}


def chart_bounds(points, spec=None):
    """Frame measured coordinates and the primary ridge, rather than seven empty decades."""
    xs = [p['arithmetic_intensity'] for p in points] or [100]
    ys = [p['achieved_tflops'] for p in points] or [1]
    if spec:
        peak = spec['ceilings'][next(k for k in PRIMARY_CEILING.values() if k in spec['ceilings'])]
        xs.append(peak/(spec['bw_gbs']/1000))
        ys.append(peak)
    lo = math.floor(math.log10(min(xs)))-1
    hi = math.ceil(math.log10(max(xs)))
    return [lo, hi], [math.floor(math.log10(min(ys)*.65)), math.log10(max(ys)*1.6)]


def plot_axis(title, limits, logarithmic=True):
    return {'type': 'log' if logarithmic else 'linear', 'title': title, 'range': limits,
            'showline': True, 'mirror': True, 'linecolor': '#333333', 'linewidth': 1,
            'ticks': 'outside', 'tickcolor': '#333333', 'gridcolor': '#d6d6d6',
            'griddash': 'dot', 'zeroline': False,
            **({'dtick': 1, 'exponentformat': 'power', 'showexponent': 'all',
                'minor': {'showgrid': True, 'gridcolor': '#ececec', 'griddash': 'dot', 'ticks': 'outside'}}
               if logarithmic else {})}


def add_roof(fig, spec, ceiling, limits, row=None, col=None, legend='legend'):
    import plotly.graph_objects as go
    target = {'row': row, 'col': col} if row else {}
    peak, bw = spec['ceilings'][ceiling], spec['bw_gbs']/1000
    ridge = peak/bw
    xs = [10**limits[0], ridge, 10**limits[1]]
    for trace in [
        go.Scatter(x=xs, y=[min(peak, bw*x) for x in xs], mode='lines', name='Reference roofline',
                   line={'color': '#347DBE', 'width': 2.8}),
        go.Scatter(x=[xs[0], xs[-1]], y=[peak, peak], mode='lines',
                   name=f'Peak ({peak:g} TFLOP/s)', line={'color': '#E95151', 'width': 1.3, 'dash': 'dash'}),
    ]:
        trace.legend = legend
        fig.add_trace(trace, **target)
    fig.add_trace(go.Scatter(x=[ridge, ridge], y=[peak/1e4, peak], mode='lines',
                             name=f'Ridge ({ridge:.1f} FLOP/B)', legend=legend,
                             line={'color': '#6E9957', 'width': 1.3, 'dash': 'dash'}), **target)


def point_hover(p):
    return (f"{p['label']}<br>{p['source_run']}<br>{p['stage']} {p['shape']}<br>"
            f"{p['precision']} · {p['resolution']}<br>{p['achieved_tflops']:.4g} effective TFLOP/s"
            f" · {p['roof_pct']:.3g}% of {p['ceiling']}<br>{p['timing_basis']}<br>{p.get('weight_basis', '')}")


def figures(data):
    import plotly.graph_objects as go
    specs, points = data['hardware_specs'], data['points']
    charts = {}
    definitions = [('a100-roofline', 'gilbreth-a100-40gb', 'A100 PCIe 40GB · 1024 × 1024'),
                   ('jetson-roofline', 'jetson-orin-nano', 'Jetson Orin Nano 25W · 512 × 512'),
                   ('m5-roofline', 'macbook-air-m5', 'Apple M5 · 512 × 512'),
                   ('cross-device-roofline-pct', None, 'Fraction of reference roof · differing workloads')]
    for key, device, title in definitions:
        subset = [p for p in points if device is None or p['device'] == device]
        spec = specs[device] if device else None
        xlim, ylim = chart_bounds(subset, spec)
        fig = go.Figure()
        if device:
            add_roof(fig, spec, PRIMARY_CEILING[device], xlim)
            if device == 'gilbreth-a100-40gb':
                for ceiling, label in [('fp32', 'FP32 reference'), ('int4_tensor_tops', 'INT4 context (TOP/s)')]:
                    peak = spec['ceilings'][ceiling]
                    ridge = peak/(spec['bw_gbs']/1000)
                    xs = sorted({10**xlim[0], ridge, 10**xlim[1]})
                    fig.add_trace(go.Scatter(x=xs, y=[min(peak, spec['bw_gbs']/1000*x) for x in xs],
                                             name=label, mode='lines', line={'color': '#999999', 'width': 1, 'dash': 'dot'}))
                    ylim[1] = max(ylim[1], math.log10(peak*1.4))
        else:
            fig.add_trace(go.Scatter(x=[10**xlim[0], 10**xlim[1]], y=[100, 100],
                                     mode='lines', name='100% reference', line={'color': '#347DBE', 'width': 2.5}))
        groups = {}
        for p in subset:
            workload = 'Base' if 'klein-base-' in p['source_run'] else 'distilled'
            group = (p['device'] if not device else p['engine'], p['tier'], workload)
            groups.setdefault(group, []).append(p)
        for (group, tier, workload), rows in groups.items():
            short, color, symbol = ENGINE_STYLE.get(group, (specs[group]['label'] if group in specs else group, '#8270B4', 'circle'))
            if not device:
                color = {'gilbreth-a100-40gb': '#4C91C2', 'jetson-orin-nano': '#78AE62', 'macbook-air-m5': '#F29A38'}[group]
            scope = {'stage': 'stage', 'step': 'step', 'operator': 'kernels', 'operator-inferred': 'inferred attention'}[tier]
            fig.add_trace(go.Scatter(
                x=[p['arithmetic_intensity'] for p in rows],
                y=[p['achieved_tflops'] if device else p['roof_pct'] for p in rows],
                mode='markers', name=f'{short} · {workload} · {scope}', text=[point_hover(p) for p in rows],
                marker={'size': 12 if tier in ('stage', 'step') else 6,
                        'symbol': ('triangle-up' if workload == 'Base' else symbol) + ('-open' if tier == 'step' else ''),
                        'color': color, 'opacity': .75 if tier.startswith('operator') else .95,
                        'line': {'color': '#333333', 'width': .8}},
                hovertemplate='%{text}<br>AI %{x:.3g} FLOP/B<extra></extra>'))
        if device == 'jetson-orin-nano':
            for name in sorted({p['source_run'] for p in subset if p['tier'] == 'step'}):
                pair = {p['stage']: p for p in subset if p['source_run'] == name and p['tier'] == 'step'}
                slow, fast = pair['denoise_step_0'], pair['denoise_step_1..3']
                ratio = fast['achieved_tflops']/slow['achieved_tflops']
                fig.add_trace(go.Scatter(x=[fast['arithmetic_intensity'], slow['arithmetic_intensity']],
                                         y=[fast['achieved_tflops'], slow['achieved_tflops']], mode='lines+markers',
                                         showlegend=False, line={'dash': 'dash', 'color': '#E95151'},
                                         marker={'symbol': ['circle', 'triangle-down'], 'size': [0, 10]},
                                         text=[f'{ratio:.1f}× reload-stage rate difference']*2,
                                         hovertemplate='%{text}<extra></extra>'))
        fig.update_layout(
            template='plotly_white', title={'text': title, 'font': {'size': 19}},
            width=960, height=720, paper_bgcolor='white', plot_bgcolor='white',
            font={'family': 'Arial', 'size': 13, 'color': '#202020'},
            xaxis=plot_axis('Arithmetic intensity (FLOP/byte, analytic)', xlim),
            yaxis=plot_axis('Effective performance (TFLOP/s)', ylim) if device else
                  plot_axis('Reference roof achieved (%)', [0, max(105, max(p['roof_pct'] for p in subset)*1.05)], False),
            legend={'x': .015, 'y': .02, 'xanchor': 'left', 'yanchor': 'bottom',
                    'bgcolor': 'rgba(255,255,255,.90)', 'bordercolor': '#777777', 'borderwidth': .7,
                    'font': {'size': 10}, 'tracegroupgap': 1},
            margin={'t': 65, 'b': 125, 'l': 95, 'r': 30})
        note = FOOTER + '<br>' + (spec['note'] if device else 'Percentages are model estimates. M5 uses an assumed 18 TFLOP/s ceiling; workloads and timer scopes differ.')
        footer = '<br>'.join(part for line in note.split('<br>') for part in textwrap.wrap(line, 120))
        fig.add_annotation(x=0, y=-.20, xref='paper', yref='paper', text=footer,
                           showarrow=False, align='left', xanchor='left', yanchor='top', font={'size': 9})
        charts[key] = fig
    return charts


def wandb_figure(fig):
    """Responsive panel geometry; the full methods remain in the table/artifact."""
    import plotly.graph_objects as go
    compact = go.Figure(fig)
    compact.update_layout(width=None, height=None, autosize=True, title=None,
                          margin={'t': 15, 'b': 75, 'l': 65, 'r': 15},
                          font={'size': 12}, legend={'font': {'size': 10}})
    compact.layout.annotations = ()
    compact.add_annotation(x=0, y=-.16, xref='paper', yref='paper', showarrow=False,
                           xanchor='left', align='left', font={'size': 9},
                           text='Analytic traffic; effective reference FLOPs. Differing timers/quality.<br>M5: assumed 18 TFLOP/s ceiling. Full scope notes in evidence artifact.')
    return compact


def paper_m5(data):
    """Two workload panels like a paper figure, retaining every M5 stage point."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    spec = data['hardware_specs']['macbook-air-m5']
    subset = [p for p in data['points'] if p['device'] == 'macbook-air-m5' and p['tier'] == 'stage']
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=.12,
                        subplot_titles=['(a) Klein 4B distilled · 4 steps', '(b) Klein 4B Base · 50 steps, CFG 4'])
    xlim, ylim = chart_bounds(subset, spec)
    # Both panels share bounds; points use their actual data coordinates, with no jitter.
    styles = [('Reference', '#4C91C2', 'circle'), ('Conditioning reuse', '#78AE62', 'triangle-up'),
              ('VAE tiling', '#A57FC1', 'diamond'), ('EasyCache', '#D88A7A', 'star'),
              ('Prefetch off', '#888888', 'hexagon'), ('mmap', '#D0BF43', 'cross')]
    for col, base in [(1, False), (2, True)]:
        legend = 'legend' if col == 1 else 'legend2'
        add_roof(fig, spec, 'bf16_fp16_gpu', xlim, 1, col, legend)
        groups = {}
        for p in subset:
            if ('klein-base-' in p['source_run']) != base:
                continue
            engine = 'sd.cpp Q4' if p['engine'] == 'stable-diffusion.cpp' else 'MPS BF16'
            number = int(p['source_run'].split('__')[0].split('-')[-1])
            option = ({1: 0, 2: 3, 3: 1, 4: 4, 5: 5} if engine.startswith('sd.cpp') else {1: 0, 2: 1, 3: 2})[number]
            groups.setdefault((engine, option), []).append(p)
        for (engine, option), rows in groups.items():
            label, color, symbol = styles[option]
            fig.add_trace(go.Scatter(x=[p['arithmetic_intensity'] for p in rows], y=[p['achieved_tflops'] for p in rows],
                                     mode='markers', name=f'{engine}: {label}', legend=legend,
                                     marker={'size': 13, 'color': color, 'symbol': symbol + ('-open' if engine.startswith('sd.cpp') else ''),
                                             'line': {'width': 1, 'color': '#333333'}},
                                     text=[point_hover(p) for p in rows], hovertemplate='%{text}<extra></extra>'), row=1, col=col)
        fig.update_xaxes(**plot_axis('Arithmetic intensity (FLOP/byte)', xlim), row=1, col=col)
        fig.update_yaxes(**plot_axis('Effective performance (TFLOP/s)', ylim), row=1, col=col)
    legend_style = {'y': .025, 'yanchor': 'bottom', 'xanchor': 'left', 'bordercolor': '#777777', 'borderwidth': .7,
                    'bgcolor': 'rgba(255,255,255,.93)', 'font': {'size': 10}}
    fig.update_layout(width=1260, height=650, template='plotly_white',
                      font={'family': 'Arial', 'size': 14, 'color': '#202020'},
                      margin={'t': 65, 'b': 105, 'l': 85, 'r': 30},
                      legend={**legend_style, 'x': .01}, legend2={**legend_style, 'x': .57})
    fig.add_annotation(x=0, y=-.21, xref='paper', yref='paper', showarrow=False, xanchor='left', align='left',
                       font={'size': 11}, text='Apple M5 · 512² · 153 GB/s theoretical bandwidth; 18 TFLOP/s assumed compute ceiling.'
                       '<br>Reference linear/attention FLOPs and analytic bytes; mixed precision and quality equivalence unassessed. Exact points, no jitter.')
    return fig


def guard_output(root, output):
    output = Path(output).resolve()
    for protected in [ROOT/'results'/'runs', ROOT/'originals', Path(root)/'results'/'runs', Path(root)/'originals']:
        protected = protected.resolve()
        if output == protected or protected in output.parents or output in protected.parents:
            raise ValueError('Output must not overlap results/runs or originals')
    return output


def upload(data, charts, output, args):
    import wandb
    receipt = output/'wandb-upload.json'
    identity = json.loads(receipt.read_text()) if receipt.exists() else {
        'id': uuid.uuid4().hex[:16], 'entity': args.entity, 'project': args.project}
    if (identity['entity'], identity['project']) != (args.entity, args.project):
        raise ValueError('Output directory belongs to a different W&B destination')
    identity['status'] = 'uploading'
    write_json(receipt, identity)
    write_json(output/'wandb-verification.json', {'status': 'verification pending', 'id': identity['id']})
    settings = wandb.Settings(x_disable_meta=True, x_disable_stats=True,
                              x_disable_machine_info=True, disable_git=True,
                              disable_code=True, disable_job_creation=True)
    panel_keys = {'a100-roofline': 'roofline/a100', 'jetson-roofline': 'roofline/jetson',
                  'm5-roofline': 'roofline/m5', 'cross-device-roofline-pct': 'roofline/normalized_pct'}
    with wandb.init(entity=args.entity, project=args.project, id=identity['id'], resume='allow',
                    name=args.name, job_type='analysis', allow_val_change=True,
                    tags=['analysis', 'roofline', 'analytic-traffic'], settings=settings, dir=str(output),
                    config={'method': data['method'], 'hardware_specs': data['hardware_specs'],
                            'source_runs': [r['source_run'] for r in data['runs']],
                            'packages': {p: importlib.metadata.version(p) for p in
                                         ['plotly', 'wandb'] + (['pillow'] if any(output.glob('*roofline*.png')) else [])}}) as run:
        table = wandb.Table(columns=TABLE_COLUMNS)
        for p in data['points']:
            row = {**p, 'source_sha256': json.dumps(p['source_sha256'], sort_keys=True)}
            table.add_data(*[row.get(k) for k in TABLE_COLUMNS])
        payload = {panel_keys[key]: wandb.Plotly(wandb_figure(fig)) for key, fig in charts.items()}
        image_keys = []
        for key in [*charts, 'm5-paper-roofline']:
            path = output/f'{key}.png'
            if path.exists():
                image_key = 'roofline_images/' + key.replace('-roofline', '').replace('cross-device-', '')
                payload[image_key] = wandb.Image(str(path), caption='Analytic roofline; scope and ceiling assumptions shown in the figure.')
                image_keys.append(image_key)
        run.log({**payload, 'roofline_points': table})
        artifact = wandb.Artifact(args.name, type='analysis')
        for filename in ['roofline-data.json', 'roofline-points.csv', 'README.md']:
            artifact.add_file(str(output/filename), name=filename)
        for key in [*charts, 'm5-paper-roofline']:
            for ext in ['html', 'png', 'pdf']:
                path = output/f'{key}.{ext}'
                if path.exists():
                    artifact.add_file(str(path), name=path.name)
        if args.manifest:
            artifact.add_file(str(args.manifest), name='manifest.json')
        artifact.add_file(str(Path(__file__).resolve()), name='plot_roofline.py')
        run.log_artifact(artifact)
        run.summary.update({'point_count': len(data['points']), 'source_run_count': len(data['runs']),
                            'chart_count': len(charts), 'exclusions': data['exclusions'],
                            'data_sha256': hashlib.sha256((output/'roofline-data.json').read_bytes()).hexdigest()})
        identity['url'] = run.url
    identity['status'] = 'uploaded; remote verification pending'
    write_json(receipt, identity)
    # Verify remote run state, all panels, exact table contents and evidence artifact.
    remote = wandb.Api().run(f"{args.entity}/{args.project}/{identity['id']}")
    files = {f.name: f for f in remote.files()}
    table_ref = remote.summary.get('roofline_points', {})
    table_path = table_ref.get('path', '')
    required_keys = [*panel_keys.values(), *image_keys, 'roofline_points']
    if remote.state != 'finished' or any(k not in remote.summary for k in required_keys):
        raise RuntimeError('W&B run/panels not remotely verified')
    if table_path not in files:
        raise RuntimeError('Remote table file missing')
    downloaded = files[table_path].download(root=str(output/'verification'), replace=True)
    remote_table = json.loads(Path(downloaded.name).read_text())
    expected_rows = [[json.dumps(p['source_sha256'], sort_keys=True) if k == 'source_sha256' else p.get(k)
                      for k in TABLE_COLUMNS] for p in data['points']]
    if remote_table.get('columns') != TABLE_COLUMNS or remote_table.get('data') != expected_rows:
        raise RuntimeError('Remote table contents differ from local coordinates/provenance')
    for key in panel_keys.values():
        if remote.summary[key].get('path') not in files:
            raise RuntimeError(f'Remote Plotly panel missing: {key}')
    for key in image_keys:
        if remote.summary[key].get('path') not in files:
            raise RuntimeError(f'Remote image panel missing: {key}')
    artifacts = list(remote.logged_artifacts())
    candidates = [a for a in artifacts if a.type == 'analysis' and a.name.split(':')[0] == args.name]
    if not candidates:
        raise RuntimeError('Remote evidence artifact missing')
    evidence = max(candidates, key=lambda a: int(a.version.removeprefix('v')))
    entry = evidence.manifest.entries['roofline-data.json']
    artifact_file = evidence.get_entry('roofline-data.json').download(root=str(output/'verification'/'artifact'))
    if Path(artifact_file).read_bytes() != (output/'roofline-data.json').read_bytes():
        raise RuntimeError('Remote evidence JSON differs from local dataset')
    verification = {'status': 'verified', 'url': identity['url'], 'state': remote.state,
                    'point_count': len(expected_rows), 'panels': required_keys,
                    'artifact': evidence.name, 'artifact_data_digest': entry.digest,
                    'table_contents_match': True, 'artifact_data_match': True}
    write_json(output/'wandb-verification.json', verification)
    identity['status'] = 'verified'
    write_json(receipt, identity)
    print(identity['url'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, default=ROOT/'output'/'roofline')
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--data-json', type=Path)
    parser.add_argument('--hardware-specs', type=Path, help='Explicit JSON ceiling overrides')
    parser.add_argument('--name', default='klein-roofline-analysis')
    parser.add_argument('--static', action='store_true')
    parser.add_argument('--upload', action='store_true')
    parser.add_argument('--entity', default='mfaruqi-purdue-university')
    parser.add_argument('--project', default='on-device-diffusion')
    args = parser.parse_args()
    if args.data_json and (args.manifest or args.hardware_specs):
        parser.error('--data-json cannot be combined with manifest/hardware overrides')
    args.output = guard_output(args.root, args.output)
    manifest = json.loads(args.manifest.read_text()) if args.manifest else None
    overrides = json.loads(args.hardware_specs.read_text()) if args.hardware_specs else None
    data = json.loads(args.data_json.read_text()) if args.data_json else collect(args.root, manifest, overrides)
    validate_data(data)
    # Hash control inputs and script separately from immutable run artifacts.
    data.setdefault('analysis_sha256', {})
    for path in [Path(__file__).resolve(), args.manifest, args.hardware_specs, args.data_json]:
        if path:
            data['analysis_sha256'][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output/'roofline-data.json', data)
    with (args.output/'roofline-points.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=TABLE_COLUMNS, extrasaction='ignore')
        writer.writeheader()
        for p in data['points']:
            writer.writerow({**p, 'source_sha256': json.dumps(p['source_sha256'], sort_keys=True)})
    charts = figures(data)
    for key, fig in charts.items():
        fig.write_html(args.output/f'{key}.html', include_plotlyjs=True)
        if args.static:
            fig.write_image(args.output/f'{key}.png', scale=2)
            fig.write_image(args.output/f'{key}.pdf')
    if args.static:
        paper = paper_m5(data)
        paper.write_html(args.output/'m5-paper-roofline.html', include_plotlyjs=True)
        paper.write_image(args.output/'m5-paper-roofline.png', scale=2)
        paper.write_image(args.output/'m5-paper-roofline.pdf')
    links = '\n'.join(f'- [{key}]({key}.html)' for key in charts)
    if (args.output/'m5-paper-roofline.html').exists():
        links += '\n- [M5 paper-style panels](m5-paper-roofline.html) · [PDF](m5-paper-roofline.pdf) · [PNG](m5-paper-roofline.png)'
    (args.output/'README.md').write_text(f'# Klein roofline analysis\n\n{len(data["runs"])} source runs; {len(data["points"])} points.\n\n{links}\n\n{METHOD}\n\nSee [building instructions](../../docs/roofline-plots.md).\n')
    print(f'Validated {len(data["runs"])} source runs; wrote {len(data["points"])} points and four charts to {args.output}')
    if args.upload:
        upload(data, charts, args.output, args)


if __name__ == '__main__':
    main()
