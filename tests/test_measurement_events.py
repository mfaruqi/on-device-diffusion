"""Normalize existing evidence without changing timings or inventing residency moves."""

import contextlib
import csv
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from measurement_events import export_events, fingerprint, harness_events, residency_events
import run_flux
import run_sdcpp

TORCH_RUN = ROOT / 'results/runs/a100-flux-klein-001__repeat__20260925-123908'
SDCPP_RUN = ROOT / 'results/runs/a100-sdcpp-flux-klein-001__baseline__20260925-114328'
JETSON_RUN = ROOT / 'results/runs/jetson-flux-klein-003__profile__20260928-195625'


class MeasurementEventTests(unittest.TestCase):
    def test_pytorch_replay_preserves_durations_and_unknown_timestamps(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            original = fingerprint(TORCH_RUN / 'stages.csv')
            events = export_events(TORCH_RUN, 'pytorch', output)
            stages = [row for row in events if row['event'] == 'stage']
            with (TORCH_RUN / 'stages.csv').open() as stream:
                expected = list(csv.DictReader(stream))
            self.assertEqual(len(stages), len(expected))
            for event, stage in zip(stages, expected):
                self.assertEqual(event['duration_ms'], float(stage['gpu_ms']))
                self.assertEqual(event['name'], stage['stage'])
                self.assertEqual(event['phase'], stage['phase'])
                self.assertIsNone(event['start_s'])
                self.assertIsNone(event['end_s'])
                self.assertIsNone(event['clock'])
                self.assertEqual(event['timing_method'], 'cuda_event_duration')
            metadata = json.loads((output / 'events-metadata.json').read_text())
            self.assertEqual(metadata['parameter_release_observations'], 0)
            self.assertIn('PyTorch CSVs retain durations', metadata['limitations'][-1])
            self.assertEqual(fingerprint(TORCH_RUN / 'stages.csv'), original)

    def test_sdcpp_replay_matches_existing_callback_stages(self):
        with tempfile.TemporaryDirectory() as temp:
            records = export_events(SDCPP_RUN, 'sdcpp', Path(temp))
            stages = [record for record in records if record['event'] == 'stage']
            with (SDCPP_RUN / 'stages.csv').open() as stream:
                expected = list(csv.DictReader(stream))
            self.assertEqual(len(stages), len(expected))
            for event, row in zip(stages, expected):
                self.assertEqual(event['duration_ms'], float(row['host_ms']))
                self.assertEqual(event['clock'], 'sdcpp_steady_clock')
                self.assertEqual(event['phase'], row['phase'])
                self.assertEqual(event['run_index'], int(row['run_index']))

    def test_jetson_header_only_csv_still_labels_profile(self):
        with tempfile.TemporaryDirectory() as temp:
            events = export_events(JETSON_RUN, 'sdcpp', Path(temp))
            stages = [event for event in events if event['event'] == 'stage']
            self.assertEqual(len(stages), 6)
            self.assertEqual({event['phase'] for event in stages}, {'profile'})

    def test_buffer_messages_preserve_timestamps_and_unknown_destination(self):
        log = ('12.500000 [1] model manager prepared params backend buffers (10.50 MB, 5 tensors, 1 blocks, VRAM) on CUDA0\n'
               '13.500000 [1] model manager released params backend buffers (10.50 MB, 5 tensors, 1 blocks, VRAM) from CUDA0\n')
        events = residency_events(log, 'engine/sdcpp.log')
        self.assertEqual([event['event'] for event in events], ['parameters_prepared', 'parameters_released'])
        released = events[1]
        self.assertEqual(released['start_s'], 13.5)
        self.assertEqual(released['end_s'], 13.5)
        self.assertIsNone(released['duration_ms'])
        self.assertEqual(released['source'], {'file': 'engine/sdcpp.log', 'line': 2})
        self.assertEqual(released['details']['reported_mb'], 10.5)
        self.assertEqual(released['details']['backend'], 'CUDA0')
        self.assertIsNone(released['details']['destination'])
        self.assertIsNone(released['details']['component'])

    def test_saved_jetson_cli_evidence_has_no_invented_clock(self):
        path = ROOT / 'results/runs/jetson-flux-klein-003__attempt__20260928-185009/engine-audit.json'
        evidence = json.loads(path.read_text())['selected_evidence']
        events = residency_events('\n'.join(evidence), 'selected_evidence_fixture')
        self.assertEqual(sum(row['event'] == 'parameters_released' for row in events), 5)
        for event in events:
            self.assertIsNone(event['start_s'])
            self.assertIsNone(event['end_s'])
            self.assertIsNone(event['clock'])
        tensor_load = next(row for row in events if row['event'] == 'tensor_load_complete')
        self.assertEqual(tensor_load['duration_ms'], 64110)
        self.assertEqual(tensor_load['timing_method'], 'rounded_engine_diagnostic')

    def test_failed_load_preserves_unknown_times(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'results.jsonl'
            path.write_text('{"event":"load","ok":false}\n')
            event = harness_events(path, 'engine/results.jsonl', {})[0]
            self.assertFalse(event['details']['ok'])
            self.assertIsNone(event['duration_ms'])
            self.assertIsNone(event['start_s'])

    def test_all_source_fingerprints_match_and_originals_are_protected(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            export_events(SDCPP_RUN, 'sdcpp', output)
            metadata = json.loads((output / 'events-metadata.json').read_text())
            for name, digest in metadata['source_sha256'].items():
                self.assertEqual(fingerprint(SDCPP_RUN / name), digest)
            with self.assertRaisesRegex(ValueError, 'inside originals'):
                export_events(SDCPP_RUN, 'sdcpp', SDCPP_RUN / 'originals/events')

    def test_runner_entrypoints_export_only_after_benchmark(self):
        for runner, source in [(run_flux, TORCH_RUN), (run_sdcpp, SDCPP_RUN)]:
            with self.subTest(runner=runner.__name__), tempfile.TemporaryDirectory() as temp:
                output_root = Path(temp)

                def benchmark(config, run_dir, profile):
                    for name in ['load.json', 'runs.csv', 'stages.csv']:
                        shutil.copyfile(source / name, run_dir / name)
                    if runner is run_sdcpp:
                        (run_dir / 'engine').mkdir()
                        shutil.copyfile(source / 'engine/results.jsonl', run_dir / 'engine/results.jsonl')
                    self.assertFalse((run_dir / 'events.jsonl').exists())

                with patch.object(runner, 'benchmark', side_effect=benchmark):
                    with patch.object(sys, 'argv', ['runner', '--config', str(source / 'config.json'), '--out-root', temp]):
                        with contextlib.redirect_stdout(io.StringIO()):
                            runner.main()
                run_dir = next(output_root.iterdir())
                self.assertEqual(json.loads((run_dir / 'status.json').read_text())['status'], 'complete')
                self.assertTrue((run_dir / 'events.jsonl').is_file())


if __name__ == '__main__':
    unittest.main()
