"""Replay saved measurement scopes; exercise updates without the W&B SDK or network."""

import contextlib
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import export_wandb as exporter


class TimingLabelTests(unittest.TestCase):
    def payload(self, name):
        return exporter.describe(exporter.RUNS / name)

    def test_edge_engine_identity_and_unknown_determinism_survive_export(self):
        payload = self.payload('a100-edgedit-flux-klein-001__baseline__20261005-233237')
        config = payload['config']
        self.assertEqual(config['engine'], 'edge-dit.cpp')
        self.assertEqual(config['engine_commit'], '97cc585d9dbd9ac5e30b78ef2e2c4a0d8bfbe9f2')
        self.assertEqual(config['engine_version'], config['engine_commit'][:12])
        self.assertIn('edge-dit.cpp', payload['tags'])
        self.assertIsNone(payload['summary']['deterministic_output'])

    def test_fractional_iso_timestamps_preserve_elapsed_duration(self):
        original = exporter.read_json
        def read(path):
            result = original(path)
            if path.name == 'status.json':
                result = {**result, 'started_utc': '2026-10-05T23:18:22Z',
                          'finished_utc': '2026-10-05T23:18:23.500000+00:00'}
            return result
        with patch.object(exporter, 'read_json', side_effect=read):
            payload = self.payload('a100-flux-klein-001__baseline__20260923-221530')
        self.assertEqual(payload['summary']['timing/job_wall_s'], 1.5)

    def test_profiler_identifies_tool_without_claiming_capture_success(self):
        for name, profiler in [
            ('a100-flux-klein-001__baseline__20260923-221530', 'none'),
            ('a100-flux-klein-001__profile__20260923-221717', 'torch.profiler'),
            ('a100-sdcpp-flux-klein-001__profile__20260925-124703', 'nsight-systems'),
            ('a100-sdcpp-flux-klein-001__profile__20260925-124021', 'nsight-systems'),
            ('jetson-flux-klein-003__profile__20260928-191418', 'nsight-systems'),
            ('jetson-flux-klein-003__profile__20260928-195625', 'nsight-systems')]:
            payload = self.payload(name)
            self.assertEqual(payload['config']['profiler'], profiler)
            self.assertEqual(payload['summary']['profiler'], profiler)

    def test_repeated_protocol_scopes_and_original_values(self):
        cases = [
            ("a100-flux-klein-001__baseline__20260923-221530", "unprofiled", True),
            ("a100-flux-klein-001__profile__20260923-221717", "unprofiled", True),
            ("a100-sdcpp-flux-klein-001__profile__20260925-124703", "profiled", False),
        ]
        for name, scope, eligible in cases:
            with self.subTest(run=name):
                payload = self.payload(name)
                config, summary = payload["config"], payload["summary"]
                source = exporter.read_json(exporter.RUNS / name / "summary.json")
                self.assertEqual(config["measurement_scope"], scope)
                self.assertEqual(config["measurement_basis"], "measured_median")
                self.assertEqual(config["baseline_comparison_eligible"], eligible)
                self.assertEqual(summary["timing/generate_s"], source["measured"]["wall_ms"]["median"] / 1000)
                self.assertEqual(summary["median/wall_ms"], source["measured"]["wall_ms"]["median"])
                self.assertEqual(summary["timing/load_s"], summary["load_total_s"])
                self.assertEqual(summary["timing/load_scope"], scope)
                self.assertEqual(summary["timing/generate_samples"], 10)
        torch_profile = self.payload(cases[1][0])["summary"]
        self.assertIn("separate torch.profiler generation", torch_profile["timing/scope_note"])
        self.assertTrue(any(key.startswith("profile/") for key in torch_profile))

    def test_jetson_single_attempt_and_profile_have_distinct_bases(self):
        attempt = self.payload("jetson-flux-klein-003__attempt__20260928-185009")
        profile = self.payload("jetson-flux-klein-003__profile__20260928-195625")
        self.assertEqual(attempt["config"]["measurement_scope"], "unprofiled")
        self.assertEqual(attempt["config"]["measurement_basis"], "single_engine_log")
        self.assertEqual(attempt["summary"]["timing/generate_s"], 39.42)
        self.assertEqual(profile["config"]["measurement_scope"], "profiled")
        self.assertEqual(profile["config"]["measurement_basis"], "single_profile_callback")
        source = exporter.read_json(exporter.RUNS / profile["run_dir"] / "timing-diagnostics.json")
        self.assertEqual(profile["summary"]["timing/generate_s"], source["generation_seconds"])
        self.assertEqual(profile["summary"]["diag/profile_generation_seconds"], source["generation_seconds"])
        for payload in (attempt, profile):
            self.assertFalse(payload["config"]["baseline_comparison_eligible"])
            self.assertEqual(payload["summary"]["timing/generate_samples"], 1)

    def test_failed_capture_does_not_invent_generation_timing(self):
        for name in ("a100-sdcpp-flux-klein-001__profile__20260925-124021",
                     "jetson-flux-klein-001__attempt__20260928-183618",
                     "jetson-flux-klein-003__profile__20260928-194910"):
            with self.subTest(run=name):
                payload = self.payload(name)
                self.assertEqual(payload["config"]["measurement_scope"], "unavailable")
                self.assertFalse(payload["config"]["baseline_comparison_eligible"])
                self.assertNotIn("timing/generate_s", payload["summary"])

    def test_incomplete_protocol_and_smoke_are_not_baseline_eligible(self):
        name = "a100-flux-klein-001__baseline__20260923-221530"
        path = exporter.RUNS / name
        cfg = exporter.read_json(path / "config.json")
        summ = exporter.read_json(path / "summary.json")
        rows = exporter.read_csv(path / "runs.csv")
        for status, observed_rows in (("failed", rows), ("complete", rows[:-1])):
            self.assertFalse(exporter.timing_labels("baseline", False, summ,
                             {"status": status}, cfg, observed_rows)[2])
        smoke = self.payload("a100-flux-klein-001__attempt__20260929-001645")
        self.assertFalse(smoke["config"]["baseline_comparison_eligible"])


class ExistingRunUpdateTests(unittest.TestCase):
    def test_update_keeps_unrelated_fields_and_skips_missing_runs(self):
        old = SimpleNamespace(id="existing", config={"user_note": "keep"},
                              summary={"user_metric": 42, "timing/generate_s": 2.5},
                              update=Mock(), delete=Mock(), url="test-url")
        api = SimpleNamespace(runs=Mock(return_value=[old]), default_entity="test")
        self.run_main(api)
        self.assertEqual(old.config["user_note"], "keep")
        self.assertEqual(old.config["measurement_scope"], "profiled")
        self.assertEqual(old.summary["user_metric"], 42)
        self.assertEqual(old.summary["timing/generate_s"], 2.5)
        old.update.assert_called_once_with()
        old.delete.assert_not_called()

    def test_api_failure_aborts_update(self):
        api = SimpleNamespace(runs=Mock(side_effect=RuntimeError("offline")), default_entity="test")
        with self.assertRaisesRegex(RuntimeError, "offline"):
            self.run_main(api)

    def run_main(self, api):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("existing", "not_uploaded"):
                (root / name).mkdir()
            def payload(path):
                return {"id": path.name, "run_dir": path.name,
                        "config": {"measurement_scope": "profiled"},
                        "summary": {"timing/generate_s": 2.5}}
            with patch.object(exporter, "RUNS", root), \
                 patch.object(exporter, "describe", side_effect=payload), \
                 patch.object(exporter, "export") as upload, \
                 patch.dict(sys.modules, {"wandb": SimpleNamespace(Api=lambda: api)}), \
                 patch.dict("os.environ", {"WANDB_DIR": tmp}), \
                 patch.object(sys, "argv", ["export_wandb.py", "--update"]), \
                 contextlib.redirect_stdout(io.StringIO()):
                try:
                    exporter.main()
                finally:
                    upload.assert_not_called()


class SummaryOnlyMedianTests(unittest.TestCase):
    def export_saved(self, name):
        payload = exporter.describe(exporter.RUNS / name)
        run = Mock(summary={}, url='test-url')
        sdk = SimpleNamespace(Settings=Mock(), init=Mock(return_value=run),
                              Table=Mock(), Image=Mock(), Artifact=Mock())
        with patch.dict(sys.modules, {'wandb': sdk}):
            exporter.export(payload, 'test-project', 'test-entity')
        return payload, run

    def test_medians_remain_in_summary_without_history_rows_or_changed_generations(self):
        for name in ['jetson-flux-klein-003__baseline__20260930-193208',
                     'a100-flux-klein-001__baseline__20260923-221530',
                     'a100-sdcpp-flux-klein-001__profile__20260925-124703']:
            with self.subTest(run=name):
                payload, run = self.export_saved(name)
                logged = [call.args[0] for call in run.log.call_args_list]
                generation_rows = [row for row in logged if 'generation' in row]
                self.assertEqual(generation_rows, [{k: v for k, v in row.items() if k != 'phase'}
                                                  for row in payload['history']])
                self.assertFalse(any(key.startswith('median/') for row in logged for key in row))
                self.assertIn('median/wall_ms', run.summary)
                self.assertEqual(run.summary['median/wall_ms'], payload['summary']['timing/generate_s'] * 1000)
                self.assertEqual(run.summary, payload['summary'])
                run.finish.assert_called_once_with(exit_code=0)

    def test_run_without_measured_medians_does_not_invent_chart_values(self):
        payload, run = self.export_saved('jetson-flux-klein-003__attempt__20260928-185009')
        self.assertFalse(any(key.startswith('median/') for call in run.log.call_args_list
                             for key in call.args[0]))
        self.assertEqual(run.summary, payload['summary'])


if __name__ == "__main__":
    unittest.main()
