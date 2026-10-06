"""CPU integration: saved callback evidence + a subprocess fake engine, no GPU."""
import contextlib
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_jetson_sdcpp as runner
from benchlib import write_json
from jetson_device import summarize_monitor
from measurement import run_status
from export_wandb import describe

SAVED = ROOT / 'results/runs/jetson-flux-klein-003__profile__20260928-195625/originals'
CONTEXT = ROOT / 'tests/fixtures/jetson_disk_context.txt'


def config(smoke=False):
    suffix = 'smoke' if smoke else 'baseline'
    return json.loads((ROOT / f'configs/jetson-flux-klein-q4-512-disk-{suffix}.json').read_text())


class FakeMonitor:
    def __init__(self, path, interval):
        self.path = path
        self.stopped = False

    def start(self):
        self.path.write_text('09-30-2026 12:00:00 RAM 5000/7620MB SWAP 900/8192MB gpu@45C\n')

    def stop(self):
        self.stopped = True


class JetsonBaselineTests(unittest.TestCase):
    def test_preserves_original_execution_and_models(self):
        original = json.loads((ROOT / 'configs/jetson-flux-klein-stage-profile.json').read_text())
        for smoke, count in [(False, 14), (True, 2)]:
            c = config(smoke)
            self.assertEqual(len(runner.check_config(c)), count)
            for key in ['model', 'workload', 'engine', 'engine_settings', 'optimizations']:
                self.assertEqual(c[key], original[key])
            self.assertEqual({k: v for k, v in c['harness_arguments'].items() if k != 'runs'},
                             {k: v for k, v in original['harness_arguments'].items() if k != 'runs'})

    def test_rejects_count_drift_and_unsupported_arguments(self):
        for key, value in [('runs', 13), ('conditioning-cache-size', 1), ('typo', 0), ('params-backend', 'CUDA0')]:
            c = config()
            c['harness_arguments'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                runner.check_config(c)

    def test_audit_accepts_disk_context_and_rejects_fallback(self):
        log = CONTEXT.read_text()
        self.assertTrue(runner.audit_execution(log, config())['settings_confirmed'])
        for bad in [log.replace('params_backend: disk', 'params_backend: CUDA0'),
                    log + '\nauto-fit plan: CPU\n', log + '\nconditioning cache hit\n']:
            with self.assertRaises(RuntimeError):
                runner.audit_execution(bad, config())

    def test_prefetch_variant_requires_explicit_policy_and_context_audit(self):
        c = config()
        c['harness_arguments']['disable-prefetch'] = 1
        with self.assertRaisesRegex(ValueError, 'prefetch'):
            runner.check_config(c)
        c['optimizations']['prefetch'] = False
        c['engine_settings']['disable_prefetch'] = True
        self.assertEqual(len(runner.check_config(c)), 14)
        log = CONTEXT.read_text()
        with self.assertRaisesRegex(RuntimeError, 'disable_prefetch'):
            runner.audit_execution(log, c)
        self.assertTrue(runner.audit_execution(
            log.replace('disable_prefetch: false', 'disable_prefetch: true'), c)['settings_confirmed'])
        for bad in [None, 0, 'false']:
            c['optimizations']['prefetch'] = bad
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, 'prefetch'):
                runner.check_config(c)

    def test_mmap_requires_all_file_confirmations_and_no_fallback(self):
        c = config()
        c['harness_arguments']['mmap'] = 1
        with self.assertRaisesRegex(ValueError, 'mmap'):
            runner.check_config(c)
        c['optimizations']['mmap'] = True
        c['engine_settings']['mmap'] = True
        self.assertEqual(len(runner.check_config(c)), 14)
        log = CONTEXT.read_text()
        receipts = ["\nusing mmap for '/models/" + v['local_file'] + "'" for v in c['model']['components']]
        with self.assertRaisesRegex(RuntimeError, 'mmap'):
            runner.audit_execution(log + ''.join(receipts[:2]), c)
        good = log + ''.join(receipts)
        self.assertEqual(len(runner.audit_execution(good, c)['mmap_io']['confirmed_files']), 3)
        for warning in ['failed to memory-map', 'mmap: failed to create backend buffer',
                        "mmap: CUDA0 cannot map '/models/file', loading it instead"]:
            with self.subTest(warning=warning), self.assertRaisesRegex(RuntimeError, 'mmap'):
                runner.audit_execution(good + '\n' + warning, c)
        with self.assertRaisesRegex(RuntimeError, 'mmap'):
            runner.audit_execution(good, config())
        for bad in [None, 1, 'true']:
            c['optimizations']['mmap'] = bad
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, 'mmap'):
                runner.check_config(c)

    def test_auto_fit_requires_unset_backend_and_reported_decision(self):
        c = json.loads((ROOT / 'configs/jetson-flux-klein-q4-512-autofit.json').read_text())
        self.assertEqual(len(runner.check_config(c)), 14)
        log = CONTEXT.read_text().replace('params_backend: disk', 'params_backend: ')
        log = log.replace('auto_fit: false', 'auto_fit: true')
        with self.assertRaisesRegex(RuntimeError, 'auto-fitting differs'):
            runner.audit_execution(log, c)
        audit = runner.audit_execution(log + '\nauto-fit plan:\n', c)
        self.assertTrue(audit['automatic_fitting'])
        self.assertTrue(audit['reported_parameter_placement'])
        c['harness_arguments']['params-backend'] = 'disk'
        c['engine_settings']['params_backend'] = 'disk'
        c['optimizations']['parameter_storage'] = 'disk'
        with self.assertRaisesRegex(ValueError, 'explicitly unset'):
            runner.check_config(c)

    def run_fake(self, directory, c, mode='success'):
        # Replay real callback structure. Distinct generated durations ensure first/warmups
        # cannot accidentally enter the median; image bytes test output hashing.
        saved = [json.loads(line) for line in (SAVED / 'results.jsonl').read_text().splitlines()]
        fixture = directory / 'fixture.json'
        fixture.write_text(json.dumps(saved))
        fake = directory / 'fake.py'
        fake.write_text('''import json,sys
from pathlib import Path
args=dict(zip(sys.argv[1::2],sys.argv[2::2]))
out=Path(args['--out-dir']); (out/'raw').mkdir()
records=json.loads(Path(args['--fixture']).read_text())
load=records[0]; origin=load['t_start']; load['t_start']=100.; load['t_end']=101.
with (out/'results.jsonl').open('w') as stream:
 stream.write(json.dumps(load)+'\\n'); stream.flush()
 count=int(args['--runs'])
 if args['--mode']=='exit': sys.exit(7)
 for i in range(count):
  g=json.loads(json.dumps(records[1])); t=g['t_start']
  duration=100. if i<4 else 10.+i
  scale=duration/(g['t_end']-t)
  for key in ['t_start','t_cond','t_sampling_start','t_sampling_end','t_decode_end','t_end']:
   g[key]=200.+i*200.+(g[key]-t)*scale
  g['t_step_end']=[200.+i*200.+(v-t)*scale for v in g['t_step_end']]
  g.update(run_index=i,width=2,height=2,channels=3)
  g['cond_cache_hits']=int(args['--conditioning-cache-size']) if i>0 else 0
  if args['--mode']=='missing_condition_hit' and i==1: g['cond_cache_hits']=0
  if '--profile-generation' in args:
   g['profile_capture']=(i==int(args['--profile-generation']))
   if args['--mode']=='missing_capture': del g['profile_capture']
  if args['--mode']=='callbacks' and i==1: g['progress_calls']=42
  stream.write(json.dumps(g)+'\\n'); stream.flush()
  (out/'raw'/('run-'+str(i)+'.rgb')).write_bytes(bytes([i if i<4 else 4])*12)
entries=int(args['--conditioning-cache-size'])
log=Path(args['--log']).read_text().replace('conditioning_cache_size: 0','conditioning_cache_size: '+str(entries))
(out/'sdcpp.log').write_text(log+'\\nconditioning cache hit'*entries*(count-1))
''')
        c['workload'].update(width=2, height=2)
        c['harness_arguments'].update(width=2, height=2)
        write_json(directory / 'config.json', c)
        monitor = FakeMonitor(directory / 'tegrastats.log', 1000)
        real_command = runner.harness_command
        def command(cfg, binary, models, engine_dir):
            return [sys.executable, str(fake), *real_command(cfg, binary, models, engine_dir)[1:],
                    '--fixture', str(fixture), '--log', str(CONTEXT), '--mode', mode]
        def environment(cfg, engine, binary, run_dir, **kwargs):
            self.assertEqual(kwargs['profile'], bool(c.get('profiling')))
            write_json(run_dir / 'environment.json', {})
        def profile_command(argv, binary, profile_dir, generation):
            # Stand-in for the external profiler, still execute the fake engine as a subprocess.
            profile_dir.mkdir()
            return [*argv, '--profile-generation', str(generation)]
        def full_command(argv, binary, profile_dir):
            profile_dir.mkdir()
            return argv
        try:
            import PIL
            image_dependency = contextlib.nullcontext()
        except ImportError:
            # Minimal image sink only when Pillow is absent; actual PNG encoding is
            # exercised by the same suite in the provisioned Python environment.
            image = SimpleNamespace(frombytes=lambda mode, size, data:
                                    SimpleNamespace(save=lambda path: path.write_bytes(data)))
            image_dependency = patch.dict(sys.modules, {'PIL': SimpleNamespace(Image=image)})
        with image_dependency, patch.object(runner, 'collect_environment', side_effect=environment), \
             patch.object(runner, 'verified_model_paths', return_value=['a', 'b', 'c']), \
             patch.object(runner, 'harness_command', side_effect=command), \
             patch.object(runner, 'targeted_profile_command', side_effect=profile_command), \
             patch.object(runner, 'validate_targeted_trace') as validate_trace, \
             patch.object(runner, 'full_profile_command', side_effect=full_command), \
             patch.object(runner, 'validate_profile_trace') as validate_full, \
             patch.object(runner, 'TegrastatsMonitor', return_value=monitor), \
             patch.object(runner, 'memory_snapshot'), patch.object(runner, 'sh', return_value='fake environment'):
            try:
                with run_status(directory):
                    runner.benchmark(c, directory, Path('engine'), Path('binary'), Path('models'))
                if (c.get('profiling') or {}).get('scope') == 'full-process':
                    validate_full.assert_called_once_with(directory / 'profile', c['workload']['steps'],
                                                         generations=14, include_load=True)
                    validate_trace.assert_not_called()
                elif c.get('profiling'):
                    validate_trace.assert_called_once_with(directory / 'profile', c['workload']['steps'])
                    validate_full.assert_not_called()
                else:
                    validate_trace.assert_not_called()
                    validate_full.assert_not_called()
            finally:
                self.assertTrue(monitor.stopped)

    def test_full_protocol_subprocess_outputs_and_export(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / 'jetson-flux-klein-003__baseline__20260930-120000'
            directory.mkdir()
            self.run_fake(directory, config())
            summary = json.loads((directory / 'summary.json').read_text())
            self.assertAlmostEqual(summary['measured']['wall_ms']['median'], 18500)
            self.assertEqual(summary['measured']['wall_ms']['n'], 10)
            self.assertTrue(summary['deterministic_output'])
            self.assertIsNone(summary['measured']['device_used_peak_gib'])
            with (directory / 'runs.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([r['phase'] for r in rows], ['first'] + ['warmup']*3 + ['measured']*10)
            with (directory / 'stages.csv').open() as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 14*6)
            self.assertTrue((directory / 'images/measured-0.png').exists())
            self.assertTrue((directory / 'events.jsonl').exists())
            payload = describe(directory)
            self.assertTrue(payload['config']['baseline_comparison_eligible'])
            self.assertEqual(payload['config']['measurement_scope'], 'unprofiled')
            self.assertEqual(len(payload['history']), 14)
            self.assertAlmostEqual(payload['summary']['timing/generate_s'], 18.5)

    def test_smoke_is_not_baseline_eligible(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / 'jetson-flux-klein-003__attempt__20260930-120000'
            directory.mkdir()
            self.run_fake(directory, config(True))
            self.assertFalse(describe(directory)['config']['baseline_comparison_eligible'])

    def test_conditioning_reuse_requires_every_expected_hit(self):
        for mode in ['success', 'missing_condition_hit']:
            with tempfile.TemporaryDirectory() as temp:
                directory = Path(temp) / 'jetson-flux-klein-003__attempt__20260930-120000'
                directory.mkdir()
                c = config(True)
                c['optimizations']['conditioning_cache_size'] = 1
                c['harness_arguments']['conditioning-cache-size'] = 1
                c['engine_settings']['conditioning_cache_size'] = 1
                if mode == 'missing_condition_hit':
                    with self.assertRaisesRegex(RuntimeError, 'conditioning reuse'):
                        self.run_fake(directory, c, mode)
                    self.assertEqual(json.loads((directory / 'status.json').read_text())['status'], 'failed')
                else:
                    self.run_fake(directory, c, mode)
                    summary = json.loads((directory / 'summary.json').read_text())
                    self.assertEqual(summary['conditioning_cache_hits_per_generation'], [0, 1])
                    self.assertEqual(summary['engine_audit']['conditioning_cache_hits'], 1)

    def test_base_conditioning_cache_needs_two_entries(self):
        c = json.loads((ROOT / 'configs/jetson-flux-klein-base-q4-512-disk-smoke.json').read_text())
        for value in [1, True, 3, '2', None]:
            c['optimizations']['conditioning_cache_size'] = value
            c['harness_arguments']['conditioning-cache-size'] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'conditioning_cache_size'):
                runner.check_config(c)
        c['optimizations']['conditioning_cache_size'] = 2
        c['harness_arguments']['conditioning-cache-size'] = 2
        c['engine_settings']['conditioning_cache_size'] = 2
        self.assertEqual(runner.check_config(c), ['first', 'measured'])

    def test_full_profile_matches_baseline_protocol_but_remains_profiled(self):
        c = json.loads((ROOT / 'configs/jetson-flux-klein-q4-512-disk-profile-full.json').read_text())
        for key in ['model', 'workload', 'engine', 'engine_settings', 'optimizations', 'protocol', 'harness_arguments']:
            self.assertEqual(c[key], config()[key])
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / 'jetson-flux-klein-003__profile__20260930-120000'
            directory.mkdir()
            self.run_fake(directory, c)
            payload = describe(directory)
            self.assertEqual(len(payload['history']), 14)
            self.assertEqual(payload['summary']['timing/generate_samples'], 10)
            self.assertEqual(payload['summary']['timing/generate_s'], 18.5)
            self.assertEqual(payload['config']['measurement_basis'], 'measured_median')
            self.assertEqual(payload['config']['measurement_scope'], 'profiled')
            self.assertEqual(payload['summary']['timing/load_scope'], 'profiled')
            self.assertFalse(payload['config']['baseline_comparison_eligible'])
            receipt = json.loads((directory / 'profile/capture.json').read_text())
            self.assertEqual(receipt['captured_generations'], 14)
            self.assertEqual(receipt['measured_generation_indices'], list(range(4, 14)))
            self.assertTrue(receipt['initial_load_captured'])

    def test_full_profile_rejects_reduced_or_unknown_scope(self):
        for profiling in [{'profiler': 'nsight-systems', 'scope': 'full-process'},
                          {'profiler': 'nsight-systems', 'scope': 'typo'}]:
            c = config(True)
            c['profiling'] = profiling
            with self.assertRaises(ValueError):
                runner.check_config(c)

    def test_later_profile_preserves_execution_and_excludes_prefix_from_summary(self):
        c = json.loads((ROOT / 'configs/jetson-flux-klein-q4-512-disk-profile.json').read_text())
        baseline = config()
        for key in ['model', 'workload', 'engine', 'engine_settings', 'optimizations']:
            self.assertEqual(c[key], baseline[key])
        for mode in ['success', 'missing_capture']:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temp:
                directory = Path(temp) / 'jetson-flux-klein-003__profile__20260930-120000'
                directory.mkdir()
                if mode == 'missing_capture':
                    with self.assertRaisesRegex(RuntimeError, 'did not confirm'):
                        self.run_fake(directory, c, mode)
                    self.assertTrue((directory / 'engine/raw/run-4.rgb').exists())
                    self.assertFalse((directory / 'summary.json').exists())
                    continue
                self.run_fake(directory, c)
                payload = describe(directory)
                self.assertEqual(payload['summary']['timing/generate_s'], 14)
                self.assertEqual(payload['summary']['timing/generate_samples'], 1)
                self.assertEqual(payload['summary']['timing/load_scope'], 'profiler_attached_untraced')
                self.assertEqual(payload['config']['profiler'], 'nsight-systems')
                self.assertFalse(payload['config']['baseline_comparison_eligible'])
                self.assertEqual(payload['config']['measurement_scope'], 'profiled')
                self.assertEqual(len(payload['history']), 5)
                receipt = json.loads((directory / 'profile/capture.json').read_text())
                self.assertEqual(receipt['generation'], 4)

    def test_targeted_profile_rejects_protocol_drift(self):
        c = config()
        c['profiling'] = {'profiler': 'nsight-systems', 'generation': 4}
        with self.assertRaisesRegex(ValueError, 'Targeted profile requires'):
            runner.check_config(c)

    def test_failed_engine_or_bad_callbacks_preserve_partial_evidence(self):
        for mode in ['exit', 'callbacks']:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temp:
                directory = Path(temp)
                with self.assertRaises(RuntimeError):
                    self.run_fake(directory, config(True), mode)
                self.assertEqual(json.loads((directory / 'status.json').read_text())['status'], 'failed')
                self.assertTrue((directory / 'engine/results.jsonl').exists())
                self.assertFalse((directory / 'summary.json').exists())
                if mode == 'callbacks':
                    self.assertTrue((directory / 'engine/raw/run-0.rgb').exists())

    def test_monitor_uses_system_ram_and_rejects_empty_capture(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            FakeMonitor(directory / 'tegrastats.log', 1000).start()
            result = summarize_monitor(directory)
            self.assertEqual(result['ram_peak_gib'], 5000/1024)
            self.assertEqual(result['gpu_temperature_max_c'], 45)
            (directory / 'tegrastats.log').write_text('')
            with self.assertRaisesRegex(RuntimeError, 'No tegrastats'):
                summarize_monitor(directory)


if __name__ == '__main__':
    unittest.main()
