#!/usr/bin/env python3
"""Repeated Jetson 003 measurement: validate → load once/run → parse → summarize.

Uses the successful quantized disk-backed workload, with optional full or targeted tracing. Read benchmark()
first. Timing conversion and image hashes are shared with the A100 sd.cpp adapter;
tegrastats is a whole-capture diagnostic, never a per-generation NVML substitute.
"""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess

from benchlib import sh, write_json
from jetson_device import TegrastatsMonitor, memory_snapshot, summarize_monitor, UnavailableDeviceMemory
from jetson_profile import check_execution_config, collect_environment, harness_command, validate_callbacks, verified_model_paths
from jetson_profile import targeted_profile_command, validate_targeted_trace
from jetson_profile import full_profile_command, validate_profile_trace
from jetson_profile import expected_conditioning_hits
from measurement import create_run_directory, phase_sequence, run_fields, run_status, sampling, summarize_runs, write_csv
from measurement_events import export_events
from sdcpp_engine import audit_cache, cache_arguments, rows_from_results, save_images


def check_config(config):
    check_execution_config(config)
    phases = phase_sequence(config['protocol'])
    profile = config.get('profiling')
    if profile is not None:
        if profile == {'profiler': 'nsight-systems', 'scope': 'full-process'}:
            if phases != ['first'] + ['warmup'] * 3 + ['measured'] * 10:
                raise ValueError('Full profile requires first + three warm-ups + ten measured generations')
        elif profile != {'profiler': 'nsight-systems', 'generation': 4} or phases != ['first'] + ['warmup'] * 3 + ['measured']:
            raise ValueError('Targeted profile requires first + three warm-ups + one captured generation (index 4)')
    arguments = config['harness_arguments']
    automatic = config['optimizations'].get('auto_fit', False)
    if type(automatic) is not bool:
        raise ValueError('auto_fit must be a boolean')
    expected = {'backend': 'CUDA0', 'params-backend': 'unset' if automatic else 'disk', 'auto-fit': int(automatic),
                'disable-segmented-compute': 0, 'eager-load': int(config['optimizations'].get('eager_loading', True)),
                'conditioning-cache-size': config['optimizations'].get('conditioning_cache_size', 0),
                'mmap': int(config['optimizations'].get('mmap', False)), 'fa': 0,
                'diffusion-fa': 1, 'disable-prefetch': int(not config['optimizations'].get('prefetch', True))}
    for key, value in expected.items():
        if arguments[key] != value:
            raise ValueError(f'Unsupported Jetson baseline setting: {key}={arguments[key]}')
    allowed = set(expected) | set(cache_arguments(config['optimizations'].get('step_cache'))) | {'runs', 'threads', 'prompt', 'width', 'height', 'seed',
                               'steps', 'cfg-scale', 'sampling-method', 'scheduler'}
    if set(arguments) != allowed:
        raise ValueError('Missing or unknown harness arguments')
    if arguments['runs'] != len(phases) or config['workload']['batch_count'] != 1:
        raise ValueError('Harness generation count must match the protocol; batch_count must be one')
    if config['optimizations']['quantization'] != {'transformer': 'Q4_0', 'text_encoder': 'Q4_K_M'}:
        raise ValueError('Only the existing Jetson 003 quantization is supported')
    if config['protocol']['tegrastats_interval_ms'] != 1000:
        raise ValueError('Keep the recorded one-second system-memory sampling interval')
    for component in config['model']['components']:
        if not re.fullmatch(r'[0-9a-f]{40}', component['revision']) or not re.fullmatch(r'[0-9a-f]{64}', component['sha256']):
            raise ValueError('Model revisions and file hashes must be pinned')
    if os.environ.get('GGML_CUDA_CUBLAS_COMPUTE_TYPE'):
        raise ValueError('GGML_CUDA_CUBLAS_COMPUTE_TYPE would override the configured execution')
    return phases


def read_generations(run_dir, config, phases):
    path = run_dir / 'engine/results.jsonl'
    if not path.is_file():
        raise RuntimeError('Harness returned without engine/results.jsonl; see engine/stdout.txt')
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    loads = [row for row in records if row['event'] == 'load']
    generations = [row for row in records if row['event'] == 'generate']
    if len(loads) != 1 or len(generations) != len(phases):
        raise RuntimeError('Expected one load and exactly the configured number of generations')
    load = loads[0]
    previous_end = load.get('t_end', 0)
    if not 0 < load.get('t_start', 0) <= previous_end:
        raise RuntimeError('Invalid model-load boundaries')
    for index, generation in enumerate(generations):
        validate_callbacks([load, generation], config['workload']['steps'], expected_conditioning_hits(config, index))
        if generation['run_index'] != index or generation['t_start'] < previous_end:
            raise RuntimeError('Missing, duplicate or overlapping generations')
        previous_end = generation['t_end']
        target = (config.get('profiling') or {}).get('generation')
        if target is not None and generation.get('profile_capture') is not (index == target):
            raise RuntimeError('Harness did not confirm the configured profile capture generation')
        expected_shape = (config['workload']['width'], config['workload']['height'], 3)
        if (generation['width'], generation['height'], generation['channels']) != expected_shape:
            raise RuntimeError('Output dimensions differ from the configured RGB image')
        raw = run_dir / 'engine/raw' / f'run-{index}.rgb'
        if raw.stat().st_size != expected_shape[0] * expected_shape[1] * expected_shape[2]:
            raise RuntimeError('Incomplete output image; raw evidence retained')
    return load, generations


def audit_execution(log, config):
    """Keep requested context settings separate from engine-reported placement."""
    arguments = config['harness_arguments']
    names = {'backend': 'backend', 'params-backend': 'params_backend', 'threads': 'n_threads',
             'auto-fit': 'auto_fit', 'eager-load': 'eager_load',
             'disable-segmented-compute': 'disable_segmented_compute',
             'disable-prefetch': 'disable_prefetch', 'conditioning-cache-size': 'conditioning_cache_size',
             'fa': 'flash_attn', 'diffusion-fa': 'diffusion_flash_attn'}
    booleans = {'auto-fit', 'eager-load', 'disable-segmented-compute', 'disable-prefetch', 'fa', 'diffusion-fa'}
    for argument, field in names.items():
        value = str(bool(arguments[argument])).lower() if argument in booleans else str(arguments[argument])
        if argument == 'params-backend' and value == 'unset':
            value = ''  # sd_ctx_params_to_str renders a null backend as an empty string.
        if not re.search(r'^' + re.escape(field + ': ' + value) + r'\s*$', log, re.MULTILINE):
            raise RuntimeError(f'Harness context does not contain {field}={value}')
    automatic = bool(arguments['auto-fit'])
    if ('auto-fit plan:' in log) != automatic:
        raise RuntimeError('Engine-reported auto-fitting differs from request')
    hits = len(re.findall(r'condition(?:ing)? cache hit', log))
    expected_hits = sum(expected_conditioning_hits(config, i) for i in range(arguments['runs']))
    if hits != expected_hits:
        raise RuntimeError(f'Unexpected conditioning reuse: expected {expected_hits} hits, saw {hits}')
    if 'Using flash attention in the diffusion model' not in log:
        raise RuntimeError('Diffusion flash attention was not confirmed')
    mapped_files = sorted({Path(path).name for path in re.findall(r"using mmap for '([^']+)'", log)})
    expected_files = sorted(c['local_file'] for c in config['model']['components']) if arguments['mmap'] else []
    if mapped_files != expected_files or re.search(r'failed to memory-map|mmap: (?:failed|.*cannot map)', log):
        raise RuntimeError('mmap file confirmations differ from request or report fallback')
    lines = log.splitlines()
    placement = [line for line in lines if 'prepared params backend buffers' in line]
    plan_index = next((i for i, line in enumerate(lines) if 'auto-fit plan:' in line), None)
    if automatic and not placement:
        raise RuntimeError('No effective parameter placement reported for auto-fit')
    return {'settings_confirmed': True, 'settings_confirmation_scope': 'requested context fields only',
            'automatic_fitting': automatic, 'conditioning_reuse': bool(expected_hits),
            'conditioning_cache_hits': hits,
            'mmap_io': {'requested': bool(arguments['mmap']), 'confirmed_files': mapped_files,
                        'scope': 'Engine-reported file mappings, not continuous parameter residency or physical I/O'},
            'parameter_storage': arguments['params-backend'], 'segmented_compute_allowed': True,
            'requested': dict(config['engine_settings']),
            'reported_parameter_placement': placement,
            'auto_fit_plan_excerpt': lines[plan_index:plan_index + 40] if plan_index is not None else [],
            'unverified': ['physical storage traffic', 'placement between logged transitions']}


def benchmark(config, run_dir, engine, binary, models):
    phases = check_config(config)
    profile = config.get('profiling')
    if config['optimizations'].get('step_cache') is not None:
        capabilities = json.loads(subprocess.check_output(
            [str(binary), '--capabilities', '1'], text=True, timeout=10))
        if capabilities.get('easycache') is not True:
            raise RuntimeError('Rebuild a separate EasyCache-capable harness before this run')
    # Check image-output dependency before spending time loading or generating.
    from PIL import Image  # noqa: F401
    collect_environment(config, engine, binary, run_dir, profile=bool(profile))
    environment = json.loads((run_dir / 'environment.json').read_text())
    environment.update({'hostname': os.uname().nodename,
                        'display_processes': sh(['pgrep', '-a', '-f', 'Xorg|Xwayland|gnome-shell']),
                        'engine_status': sh(['git', '-C', str(engine), 'status', '--porcelain']),
                        'engine_submodules': sh(['git', '-C', str(engine), 'submodule', 'status']),
                        'nvcc': sh(['/usr/local/cuda/bin/nvcc', '--version']),
                        'ldd': sh(['ldd', str(binary)]),
                        'memory_source': 'whole-system tegrastats, 1000 ms; no stage alignment'})
    write_json(run_dir / 'environment.json', environment)
    paths = verified_model_paths(config, models)
    engine_dir = run_dir / 'engine'
    engine_dir.mkdir()
    command = harness_command(config, binary, paths, engine_dir)
    if profile:
        if profile.get('scope') == 'full-process':
            command = full_profile_command(command, binary, run_dir / 'profile')
        else:
            command = targeted_profile_command(command, binary, run_dir / 'profile', profile['generation'])
    write_json(engine_dir / 'command.json', {'argv': command,
               'cache_condition': 'All weight files SHA256-read immediately before run; no cache flush.',
               'profiler_attached': bool(profile), 'profiling': profile})
    memory_snapshot(run_dir / 'memory-before.txt')
    monitor = TegrastatsMonitor(run_dir / 'tegrastats.log', config['protocol']['tegrastats_interval_ms'])
    print(f'Running {len(phases)} generations with one model context; progress: {engine_dir}/results.jsonl', flush=True)
    try:
        with sampling(monitor):
            with (engine_dir / 'stdout.txt').open('w') as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                        env={**os.environ, 'HF_HUB_OFFLINE': '1'})
    finally:
        memory_snapshot(run_dir / 'memory-after.txt')
    if result.returncode:
        raise RuntimeError(f'Harness exited with {result.returncode}; see engine/stdout.txt; partial JSON/RGB retained')
    load, generations = read_generations(run_dir, config, phases)
    engine_log = (engine_dir / 'sdcpp.log').read_text()
    audit = audit_execution(engine_log, config)
    cache_audit = audit_cache(engine_log, config['optimizations'].get('step_cache'), len(generations))
    if profile:
        if profile.get('scope') == 'full-process':
            validate_profile_trace(run_dir / 'profile', config['workload']['steps'],
                                   generations=len(phases), include_load=True)
            receipt = {**profile, 'captured_generations': len(phases), 'initial_load_captured': True,
                       'trace_generation_indices': list(range(len(phases))),
                       'measured_generation_indices': [i for i, phase in enumerate(phases) if phase == 'measured']}
        else:
            validate_targeted_trace(run_dir / 'profile', config['workload']['steps'])
            receipt = {**profile, 'trace_generation_index': 0,
                       'scope': 'Only generation 4 is traced; Nsight is attached throughout the process.'}
        write_json(run_dir / 'profile/capture.json', {**receipt, 'nvtx_and_cuda_validated': True})
    monitor_summary = summarize_monitor(run_dir)
    save_images(run_dir, engine_dir, generations, phases)
    condition_hits = [expected_conditioning_hits(config, i) for i in range(len(generations))]
    rows, stages = rows_from_results(generations, phases, config['workload']['steps'], UnavailableDeviceMemory(),
                                    expected_conditioning_hits=condition_hits)
    write_csv(run_dir / 'runs.csv', rows)
    write_csv(run_dir / 'stages.csv', stages)
    load_record = {'load_total_s': load['t_end'] - load['t_start'], 'model_version': load.get('model_version'),
                   'note': 'new_sd_ctx; parameter policy: ' + config['optimizations']['parameter_storage']
                           + '. Requested policy is not proof of continuous residency. Hash checks warmed file cache.'}
    write_json(run_dir / 'load.json', load_record)
    summary = {'id': config['id'], 'record_kind': config['record_kind'],
               'profiler': 'nsight-systems' if profile else 'none', 'profiling': profile,
               **summarize_runs(rows), 'load': load_record, 'monitor_window': monitor_summary,
               'engine_audit': audit, 'cache_audit': cache_audit, 'stage_callbacks_validated': True,
               'conditioning_cache_hits_per_generation': condition_hits,
               'timing_method': 'host steady_clock at sd.cpp callbacks; includes on-demand loading',
               'unavailable_metrics': {
                   'gpu_span_ms': 'No CUDA event span; use host wall_ms',
                   'postprocess_ms': 'Included in sd.cpp decode/output boundary; no separate stage',
                   'peak_alloc_gib/peak_reserved_gib': 'No PyTorch allocator',
                   'device_used_peak_gib': 'No NVML; system RAM capture is separate and not stage-aligned',
                   'host_rss_gib': 'Per-generation process RSS not sampled'},
               'units': {'time': 'ms (load in s)', 'memory': 'GiB (2^30 bytes)'}}
    write_json(run_dir / 'summary.json', summary)
    export_events(run_dir, 'sdcpp')
    print(json.dumps(summary['measured']['wall_ms'], indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--engine', type=Path, default=Path.home() / 'tools/stable-diffusion.cpp')
    parser.add_argument('--binary', type=Path, help='Default: sd-bench-nvtx for profile configs, otherwise sd-bench')
    parser.add_argument('--models', type=Path, default=Path.home() / 'models/flux2-klein')
    parser.add_argument('--out-root', type=Path, default=Path.home() / 'results/runs')
    parser.add_argument('--kind', choices=['baseline', 'repeat', 'attempt', 'profile'])
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    profiling = bool(config.get('profiling'))
    args.kind = args.kind or ('profile' if profiling else 'baseline')
    if (args.kind == 'profile') != profiling:
        parser.error('Profile config and --kind profile must be used together')
    args.binary = args.binary or Path.home() / 'tools/sd-bench-build' / ('sd-bench-nvtx' if profiling else 'sd-bench')
    directory = create_run_directory(config, args.out_root, args.kind)
    with run_status(directory):
        # Header-only files remain useful if the engine fails before producing rows.
        (directory / 'runs.csv').write_text(','.join(run_fields(config['workload']['steps'])) + '\n')
        (directory / 'stages.csv').write_text('run_index,phase,stage,gpu_ms,host_ms,alloc_start_gib,alloc_end_gib,device_used_peak_gib\n')
        benchmark(config, directory, args.engine, args.binary, args.models)


if __name__ == '__main__':
    main()
