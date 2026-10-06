"""CPU checks for the portable Jetson profile capture. No Jetson commands are run."""
import contextlib
import hashlib
import json
import sqlite3
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from build_jetson_bundle import FILES, build_bundle
from jetson_device import TegrastatsMonitor, memory_snapshot
from jetson_profile import check_config, profile_command, save_profile_results, validate_callbacks, verified_model_paths
from jetson_profile import targeted_profile_command, validate_targeted_trace
from jetson_profile import full_profile_command, validate_profile_trace
import profile_jetson_stages as capture

EVIDENCE = ROOT / 'results/runs/jetson-flux-klein-003__profile__20260928-195625/originals'


def configuration():
    return json.loads((ROOT / 'configs/jetson-flux-klein-stage-profile.json').read_text())


def observations():
    return [json.loads(line) for line in (EVIDENCE / 'results.jsonl').read_text().splitlines()]


class JetsonCaptureTests(unittest.TestCase):
    def test_full_command_traces_entire_harness_without_capture_selector(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch('jetson_profile.subprocess.check_output', return_value='{"profile_generation":true}'):
                command = full_profile_command(['binary', '--runs', '14'], Path('binary'), Path(temp) / 'profile')
            self.assertEqual(command[-3:], ['binary', '--runs', '14'])
            self.assertIn('--trace=cuda,nvtx,osrt', command)
            self.assertFalse(any('capture-range' in word or 'profile-generation' in word for word in command))

    def test_full_trace_requires_all_fourteen_generations_and_initial_load(self):
        for generations, load, valid in [(14, True, True), (1, True, False), (14, False, False), (13, True, False)]:
            with self.subTest(generations=generations, load=load), tempfile.TemporaryDirectory() as temp:
                directory = Path(temp)
                db = sqlite3.connect(directory / 'trace.sqlite')
                db.executescript('CREATE TABLE StringIds (id INTEGER, value TEXT); '
                                 'CREATE TABLE NVTX_EVENTS (text TEXT, textId INTEGER, end INTEGER); '
                                 'CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER);')
                names = (['generate', 'text_encode', 'vae_decode'] + [f'denoise_step_{i}' for i in range(4)]) * generations
                if load: names.append('load')
                db.executemany('INSERT INTO NVTX_EVENTS VALUES (?, NULL, 100)', [(n,) for n in names])
                db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (10)')
                db.commit(); db.close()
                with patch('jetson_profile.validate_trace_labels'):
                    if valid:
                        validate_profile_trace(directory, 4, generations=14, include_load=True)
                    else:
                        with self.assertRaises(RuntimeError):
                            validate_profile_trace(directory, 4, generations=14, include_load=True)

    def test_targeted_command_and_stale_binary_preflight(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / 'profile'
            with patch('jetson_profile.subprocess.check_output', return_value='{"profile_generation":false}'):
                with self.assertRaisesRegex(RuntimeError, 'Rebuild'):
                    targeted_profile_command(['binary'], Path('binary'), directory, 4)
            self.assertFalse(directory.exists())
            with patch('jetson_profile.subprocess.check_output', return_value='{"profile_generation":true}'):
                command = targeted_profile_command(['binary', '--runs', '5'], Path('binary'), directory, 4)
            for flag in ['--capture-range=nvtx', '--nvtx-capture=profile_capture', '--capture-range-end=stop']:
                self.assertIn(flag, command)
            self.assertEqual(command[-2:], ['--profile-generation', '4'])

    def test_targeted_trace_rejects_extra_generations_and_absent_cuda(self):
        for case in ['valid', 'extra_generation', 'load', 'no_cuda', 'missing_stage']:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temp:
                directory = Path(temp)
                db = sqlite3.connect(directory / 'trace.sqlite')
                db.executescript('CREATE TABLE StringIds (id INTEGER, value TEXT); '
                                 'CREATE TABLE NVTX_EVENTS (text TEXT, textId INTEGER, end INTEGER); '
                                 'CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER);')
                names = ['generate', 'text_encode', 'vae_decode'] + [f'denoise_step_{i}' for i in range(4)]
                if case == 'extra_generation': names.append('generate')
                if case == 'load': names.append('load')
                if case == 'missing_stage': names.remove('text_encode')
                # Include both inline and registered strings, as real Nsight exports do.
                db.execute('INSERT INTO StringIds VALUES (1, ?)', (names[0],))
                db.execute('INSERT INTO NVTX_EVENTS VALUES (NULL, 1, 100)')
                db.executemany('INSERT INTO NVTX_EVENTS VALUES (?, NULL, 100)', [(n,) for n in names[1:]])
                if case != 'no_cuda': db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (10)')
                db.commit(); db.close()
                with patch('jetson_profile.validate_trace_labels'):
                    if case == 'valid':
                        validate_targeted_trace(directory, 4)
                    else:
                        with self.assertRaises(RuntimeError):
                            validate_targeted_trace(directory, 4)

    def test_current_config_and_observed_callbacks_are_valid(self):
        check_config(configuration())
        rows = observations()
        self.assertEqual(validate_callbacks(rows, 4), rows[0])

    def test_command_matches_previous_capture(self):
        old = json.loads((EVIDENCE / 'command.json').read_text())['argv']
        run_dir = Path(old[old.index('--out-dir') + 1])
        model_paths = [old[old.index(flag) + 1] for flag in ['--diffusion-model', '--llm', '--vae']]
        binary = old[old.index('--out-dir') - 1]
        self.assertEqual(profile_command(configuration(), binary, model_paths, run_dir), old)

    def test_rejects_inconsistent_settings_and_unknown_options(self):
        changes = [('workload', 'steps', 5), ('engine_settings', 'threads', 8),
                   ('protocol', 'warmups', 1), ('optimizations', 'typo', False),
                   ('optimizations', 'vae_tiling', True), ('optimizations', 'parameter_storage', 'cuda0')]
        for section, key, value in changes:
            config = configuration()
            config[section][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                check_config(config)

    def test_rejects_bad_callbacks_and_failed_load(self):
        cases = []
        rows = observations(); rows[1]['progress_calls'] = 42; cases.append(rows)
        rows = observations(); rows[1]['t_cond'] = 0; cases.append(rows)
        rows = observations(); rows[1]['t_step_end'].reverse(); cases.append(rows)
        rows = observations(); rows[1]['cond_cache_hits'] = 1; cases.append(rows)
        rows = observations(); rows[0]['ok'] = False; cases.append(rows)
        for rows in cases:
            with self.assertRaises(RuntimeError):
                validate_callbacks(rows, 4)

    def test_hash_validation_rejects_changed_weight_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            weight = directory / 'test.gguf'
            weight.write_bytes(b'fixture weights')
            config = {'model': {'components': [{'local_file': weight.name,
                       'sha256': hashlib.sha256(weight.read_bytes()).hexdigest()}]}}
            self.assertEqual(verified_model_paths(config, directory), [str(weight)])
            weight.write_bytes(b'changed weights')
            with self.assertRaisesRegex(RuntimeError, 'Weight hash mismatch'):
                verified_model_paths(config, directory)

    def test_saved_profile_summary_and_headers_are_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            (directory / 'results.jsonl').write_text((EVIDENCE / 'results.jsonl').read_text())

            def nsys_stats(command, **kwargs):
                kwargs['stdout'].write((EVIDENCE / 'nsys-stats.txt').read_text())

            with patch('jetson_profile.subprocess.run', side_effect=nsys_stats):
                save_profile_results(directory, 4)
            self.assertEqual(json.loads((directory / 'summary.json').read_text()),
                             json.loads((EVIDENCE / 'summary.json').read_text()))
            for name in ['runs.csv', 'stages.csv']:
                self.assertEqual((directory / name).read_bytes(), (EVIDENCE / name).read_bytes())

    def test_capture_failure_stops_monitor_and_keeps_log(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.ExitStack() as stack:
            directory = Path(temp)
            monitor = Mock()
            stack.enter_context(patch.object(capture, 'collect_environment'))
            stack.enter_context(patch.object(capture, 'verified_model_paths', return_value=['a', 'b', 'c']))
            stack.enter_context(patch.object(capture, 'memory_snapshot'))
            stack.enter_context(patch.object(capture, 'TegrastatsMonitor', return_value=monitor))
            child = stack.enter_context(patch.object(capture.subprocess, 'call', return_value=7))
            with self.assertRaisesRegex(RuntimeError, 'exit code 7'):
                capture.capture_profile(configuration(), directory, Path('engine'), Path('binary'), Path('models'))
            monitor.start.assert_called_once()
            monitor.stop.assert_called_once()
            self.assertEqual(child.call_args.kwargs['env']['HF_HUB_OFFLINE'], '1')
            self.assertTrue((directory / 'profile.log').exists())
            self.assertFalse((directory / 'summary.json').exists())

    def test_monitor_is_reaped_on_timeout_and_stream_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            monitor = TegrastatsMonitor(Path(temp) / 'tegrastats.log', 1000)
            process = Mock()
            process.wait.side_effect = [subprocess.TimeoutExpired('tegrastats', 5), 0]
            with patch('jetson_device.subprocess.Popen', return_value=process) as start:
                monitor.start()
                monitor.stop()
            self.assertEqual(start.call_args.args[0], ['tegrastats', '--interval', '1000'])
            process.terminate.assert_called_once()
            process.kill.assert_called_once()
            self.assertTrue(monitor.stream.closed)

    def test_unavailable_snapshot_is_recorded(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'memory-after.txt'
            with patch('jetson_device.subprocess.check_output', side_effect=FileNotFoundError('free')):
                memory_snapshot(path)
            self.assertIn('Memory snapshot unavailable', path.read_text())

    def test_bundle_is_reproducible_and_runs_without_repository(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            first, second = directory / 'one.tar.gz', directory / 'two.tar.gz'
            build_bundle(first)
            build_bundle(second)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(first.read_bytes(), (ROOT / 'bundles/jetson-stage-profile.tar.gz').read_bytes())
            with tarfile.open(first) as archive:
                self.assertEqual(archive.getnames(), list(FILES))
                for member in archive.getmembers():
                    target = directory / member.name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    data = archive.extractfile(member).read()
                    self.assertEqual(data, (ROOT / member.name).read_bytes())
                    target.write_bytes(data)
            run = subprocess.run([sys.executable, '-S', str(directory / 'scripts/profile_jetson_stages.py'), '--help'],
                                 cwd=directory, capture_output=True, text=True, env={})
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertIn('--models', run.stdout)
            baseline = subprocess.run([sys.executable, "-S", str(directory / "scripts/run_jetson_sdcpp.py"), "--help"],
                                      cwd=directory, capture_output=True, text=True, env={})
            self.assertEqual(baseline.returncode, 0, baseline.stderr)
            self.assertIn("--kind", baseline.stdout)


if __name__ == '__main__':
    unittest.main()
