"""CPU-only regression checks. Saved run artifacts are read, never rewritten.

Run from the repository root: python3 -m unittest discover -s tests -v
"""

import contextlib
import csv
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from benchlib import GIB
from flux_engine import check_config, generate_once
from measurement import phase_sequence, run_fields, run_status, sampling, summarize_runs
from run_flux import measure_generations, result_row
import run_flux
import run_sdcpp
from sdcpp_engine import conditioning_hits, rows_from_results
from torch_stages import StageRecorder


def read_rows(path):
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key, value in row.items():
            if key.endswith("_ms") or key.endswith("_gib"):
                row[key] = float(value) if value else None
            elif key == "run_index":
                row[key] = int(value)
    return rows


class SavedRunTests(unittest.TestCase):
    def test_summaries_match_every_completed_a100_run(self):
        checked = 0
        for directory in sorted((ROOT / "results/runs").glob("a100-*")):
            summary_path = directory / "summary.json"
            if not summary_path.exists():
                continue
            expected = json.loads(summary_path.read_text())
            if not expected.get("measured"):
                continue
            with self.subTest(run=directory.name):
                rows = read_rows(directory / "runs.csv")
                actual = summarize_runs(rows)
                for key, value in actual.items():
                    if key == "deterministic_output" and expected.get("engine") == "edge-dit.cpp":
                        # The adapter overrides this field: upstream keeps only the final image.
                        self.assertIsNone(expected[key])
                        self.assertEqual(sum(bool(row['image_sha256']) for row in rows), 1)
                        self.assertIn(key, expected['unavailable_metrics'])
                        continue
                    if key != "measured":
                        self.assertEqual(value, expected[key], key)
                        continue
                    self.assertEqual(value.keys(), expected[key].keys())
                    for metric, stats in value.items():
                        if stats is None:
                            self.assertIsNone(expected[key][metric])
                            continue
                        for statistic, number in stats.items():
                            saved = expected[key][metric][statistic]
                            if statistic == "stdev":
                                # Python versions differ in the last few bits of sqrt.
                                self.assertTrue(math.isclose(number, saved, rel_tol=1e-14, abs_tol=1e-12))
                            else:
                                self.assertEqual(number, saved, (metric, statistic))
                checked += 1
        self.assertGreaterEqual(checked, 4, "Saved regression evidence is missing")

    def test_pytorch_stage_arithmetic_matches_saved_rows(self):
        checked = 0
        for directory in sorted((ROOT / "results/runs").glob("a100-flux-*")):
            if not (directory / "stages.csv").exists():
                continue
            stages = read_rows(directory / "stages.csv")
            cfg = json.loads((directory / "config.json").read_text())
            for expected in read_rows(directory / "runs.csv"):
                with self.subTest(run=directory.name, generation=expected["run_index"]):
                    result = {key: expected[key] for key in
                              ["wall_ms", "gpu_span_ms", "peak_alloc_gib", "peak_reserved_gib"]}
                    result.update(t0=0, t1=1, stages=[s for s in stages if s["run_index"] == expected["run_index"]])
                    # Historical distilled runs predate these execution counters.
                    summary = json.loads((directory / "summary.json").read_text())
                    result.update(calls_per_step=summary.get("transformer_calls_per_step", 1),
                                  scheduler_steps=cfg["workload"]["num_inference_steps"])
                    sampler = Mock()
                    sampler.peak_between.return_value = expected["device_used_peak_gib"]
                    process = Mock()
                    process.memory_info.return_value.rss = expected["host_rss_gib"] * GIB
                    image = Mock()
                    image.tobytes.return_value = b"fixture pixels"
                    actual = result_row(expected["run_index"], expected["phase"], image, result,
                                        cfg["workload"]["num_inference_steps"], sampler, process)
                    self.assertEqual(list(actual), run_fields(cfg["workload"]["num_inference_steps"]))
                    for key in expected:
                        if key != "image_sha256":  # Original image pixels are intentionally not in Git.
                            self.assertEqual(actual[key], expected[key], key)
                    checked += 1
        self.assertGreaterEqual(checked, 28)

    def test_sdcpp_callbacks_match_saved_rows_and_stages(self):
        checked = 0
        for source in sorted((ROOT / "results/runs").glob("a100-sdcpp-*/engine/results.jsonl")):
            directory = source.parent.parent
            if not (directory / "runs.csv").exists():
                continue
            events = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
            generations = [event for event in events if event["event"] == "generate"]
            expected = read_rows(directory / "runs.csv")
            cfg = json.loads((directory / "config.json").read_text())
            sampler = Mock()
            sampler.peak_between.return_value = None  # NVML samples are not stored in these artifacts.
            actual, stages = rows_from_results(generations, phase_sequence(cfg["protocol"]),
                                               cfg["workload"]["num_inference_steps"], sampler,
                                               expected_conditioning_hits=conditioning_hits(cfg, len(generations)))
            with self.subTest(run=directory.name):
                self.assertEqual(len(actual), len(expected))
                for row, original in zip(actual, expected):
                    for key in original:
                        if key not in ("device_used_peak_gib", "image_sha256"):
                            self.assertEqual(row[key], original[key], key)
                saved_stages = read_rows(directory / "stages.csv")
                self.assertEqual(len(stages), len(saved_stages))
                for row, original in zip(stages, saved_stages):
                    for key in original:
                        if key != "device_used_peak_gib":
                            self.assertEqual(row[key], original[key], key)
                checked += 1
        self.assertGreaterEqual(checked, 2)


class ProtocolTests(unittest.TestCase):
    def test_phase_order_and_exclusion(self):
        phases = phase_sequence(dict(first_runs=1, warmup_runs=3, measured_runs=2))
        self.assertEqual(phases, ["first", "warmup", "warmup", "warmup", "measured", "measured"])
        rows = [dict(phase=phase, wall_ms=value, gpu_span_ms=None, image_sha256="same")
                for phase, value in zip(phases, [900, 800, 700, 600, 10, 20])]
        summary = summarize_runs(rows)
        self.assertEqual(summary["first_run"]["wall_ms"], 900)
        self.assertEqual(summary["measured"]["wall_ms"]["median"], 15)
        self.assertEqual(summary["measured"]["wall_ms"]["n"], 2)
        self.assertIsNone(summary["measured"]["gpu_span_ms"])

    def test_invalid_protocol_is_rejected(self):
        for changes in [dict(first_runs=0), dict(first_runs=2), dict(warmup_runs=-1),
                        dict(measured_runs=0), dict(measured_runs=True)]:
            with self.subTest(changes=changes), self.assertRaises(AssertionError):
                phase_sequence(dict(dict(first_runs=1, warmup_runs=3, measured_runs=10), **changes))

    def test_unknown_disabled_optimization_is_rejected(self):
        cfg = json.loads((ROOT / "configs/a100-flux-klein-bf16.json").read_text())
        cfg["model"]["revision"] = "a" * 40
        check_config(cfg)
        cfg["optimizations"]["typo_offload"] = False
        with self.assertRaisesRegex(AssertionError, "Unknown optimizations"):
            check_config(cfg)

    def test_cli_help_requires_no_gpu_packages(self):
        for script in ["run_flux.py", "run_sdcpp.py"]:
            result = subprocess.run([sys.executable, "-S", str(ROOT / "scripts" / script), "--help"],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("--prepare-only", result.stdout)
        code = "import sys; sys.path.insert(0, 'scripts'); import run_sdcpp; assert 'torch' not in sys.modules"
        subprocess.run([sys.executable, "-S", "-c", code], cwd=ROOT, check=True)


class FakeImage:
    """Only the two image operations used by the runner; requires neither PIL nor torch."""
    def tobytes(self):
        return b"fake image; not an inference result"

    def save(self, path):
        path.write_bytes(self.tobytes())


def fake_generation():
    stages = [dict(stage=name, gpu_ms=10, host_ms=11, alloc_start_gib=1, alloc_end_gib=1,
                   peak_alloc_gib=2, peak_reserved_gib=3)
              for name in ["text_encode", "denoise_step_0", "vae_decode", "postprocess"]]
    return FakeImage(), dict(wall_ms=50, gpu_span_ms=45, t0=0, t1=1, stages=stages,
                             segments=[], peak_alloc_gib=2, peak_reserved_gib=3,
                             calls_per_step=1, scheduler_steps=1)


class KleinGuidanceTests(unittest.TestCase):
    def test_generation_uses_native_empty_negative_prompt_and_reports_counts(self):
        cfg = json.loads((ROOT / "configs/a100-flux-klein-base-bf16.json").read_text())
        image = FakeImage()
        pipe = Mock(return_value=SimpleNamespace(images=[image]), do_classifier_free_guidance=True)
        recorder = Mock(scheduler_steps=50)
        recorder.finish.return_value = ([], 2, 3, [])
        torch = Mock()
        torch.cuda.Event.return_value.elapsed_time.return_value = 100
        profiler = SimpleNamespace(record_function=lambda _: contextlib.nullcontext())
        with patch.dict(sys.modules, torch=torch, **{"torch.profiler": profiler}):
            returned_image, result = generate_once(pipe, recorder, cfg["workload"])
        self.assertIs(returned_image, image)
        self.assertNotIn("negative_prompt", pipe.call_args.kwargs)
        self.assertNotIn("negative_prompt_embeds", pipe.call_args.kwargs)
        self.assertEqual(pipe.call_args.kwargs["guidance_scale"], 4.0)
        self.assertEqual(result["calls_per_step"], 2)
        self.assertEqual(result["scheduler_steps"], 50)
        self.assertEqual(torch.cuda.synchronize.call_count, 2)

    def test_base_config_requires_explicit_empty_negative_prompt(self):
        cfg = json.loads((ROOT / "configs/a100-flux-klein-base-bf16.json").read_text())
        cfg["model"]["revision"] = "a" * 40
        check_config(cfg)
        for value in ["bad anatomy", None]:
            cfg["workload"]["negative_prompt"] = value
            with self.assertRaisesRegex(AssertionError, "empty negative prompt"):
                check_config(cfg)
        del cfg["workload"]["negative_prompt"]
        with self.assertRaisesRegex(AssertionError, "explicit negative_prompt"):
            check_config(cfg)

    def test_cfg_sums_both_forwards_and_both_encodes(self):
        image, result = fake_generation()
        result.update(calls_per_step=2, scheduler_steps=2, gpu_span_ms=100, wall_ms=105)
        result["stages"] = [dict(stage=name, gpu_ms=value) for name, value in [
            ("text_encode", 3), ("text_encode", 4),
            ("denoise_step_0", 10), ("denoise_step_0", 20),
            ("denoise_step_1", 11), ("denoise_step_1", 21),
            ("vae_decode", 8), ("postprocess", 2),
        ]]
        sampler, process = Mock(), Mock()
        sampler.peak_between.return_value = None
        process.memory_info.return_value.rss = GIB

        def row():
            return result_row(0, "first", image, result, 2, sampler, process)

        actual = row()
        self.assertEqual(actual["text_encode_ms"], 7)
        self.assertEqual(actual["denoise_step_0_ms"], 30)
        self.assertEqual(actual["denoise_step_1_ms"], 32)
        self.assertEqual(actual["denoise_ms"], 62)
        self.assertEqual(actual["other_ms"], 21)
        result["scheduler_steps"] = 1
        with self.assertRaisesRegex(AssertionError, "scheduler steps"):
            row()
        result["scheduler_steps"] = 2
        # Total forward count still matches, but one belongs to the wrong step.
        result["stages"][3]["stage"] = "denoise_step_1"
        with self.assertRaisesRegex(AssertionError, "logical step"):
            row()
        result["stages"][3]["stage"] = "denoise_step_0"
        del result["stages"][1]
        with self.assertRaisesRegex(AssertionError, "text encodes"):
            row()

    def test_recorder_counts_scheduler_updates_not_transformer_calls(self):
        pipe = SimpleNamespace(encode_prompt=Mock(), vae=SimpleNamespace(decode=Mock()),
                               image_processor=SimpleNamespace(postprocess=Mock()),
                               scheduler=SimpleNamespace(step=Mock(return_value="latents")),
                               transformer=Mock())
        recorder = StageRecorder(Mock())
        recorder.start, recorder.end = Mock(), Mock()
        scheduler_step = pipe.scheduler.step
        recorder.install(pipe)
        before = pipe.transformer.register_forward_pre_hook.call_args.args[0]
        after = pipe.transformer.register_forward_hook.call_args.args[0]
        for calls_per_step in [1, 2]:
            recorder.reset()
            recorder.start.reset_mock()
            for _ in range(2):
                for _ in range(calls_per_step):
                    before(None, ())
                    after(None, (), None)
                self.assertEqual(pipe.scheduler.step("noise", return_dict=False), "latents")
            self.assertEqual(recorder.scheduler_steps, 2)
            self.assertEqual([call.args[0] for call in recorder.start.call_args_list],
                             [f"denoise_step_{i}" for i in range(2) for _ in range(calls_per_step)])
        # A failed scheduler update is not counted as a completed step.
        scheduler_step.side_effect = RuntimeError("scheduler failed")
        with self.assertRaisesRegex(RuntimeError, "scheduler failed"):
            pipe.scheduler.step("noise")
        self.assertEqual(recorder.scheduler_steps, 2)


class LifecycleTests(unittest.TestCase):
    def test_sdcpp_missing_results_preserves_real_process_failure(self):
        cfg = json.loads((ROOT / "configs/a100-sdcpp-flux-klein-bf16.json").read_text())
        for exit_code, message in [(127, "libcudart.so.12"), (0, "did not write")]:
            with self.subTest(exit_code=exit_code), tempfile.TemporaryDirectory() as temp:
                directory = Path(temp)
                monitor = Mock()

                def child_process(command, **kwargs):
                    kwargs["stdout"].write("error while loading shared libraries: libcudart.so.12\n")
                    return SimpleNamespace(returncode=exit_code)

                with patch.object(run_sdcpp.subprocess, "run", side_effect=child_process):
                    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        with self.assertRaisesRegex(RuntimeError, message), run_status(directory):
                            run_sdcpp.run_harness(cfg, directory, Path("/fake/sd-bench"), Path("/fake/cache"),
                                                 ["first", "measured"], monitor, False)
                status = json.loads((directory / "status.json").read_text())
                self.assertEqual(status["status"], "failed")
                self.assertIn(message, status["error"])
                self.assertTrue((directory / "engine/stdout.txt").exists())
                monitor.stop.assert_called_once()

    def test_sdcpp_runs_one_process_for_the_entire_protocol(self):
        cfg = json.loads((ROOT / "configs/a100-sdcpp-flux-klein-bf16.json").read_text())
        phases = ["first", "warmup", "measured"]
        events = [dict(event="load", ok=True)] + [
            dict(event="generate", ok=True, run_index=i) for i in range(len(phases))
        ]
        for profile in [False, True]:
            with self.subTest(profile=profile), tempfile.TemporaryDirectory() as temp:
                directory = Path(temp)
                sampler = Mock()

                def child_process(command, **kwargs):
                    (directory / "engine/results.jsonl").write_text(
                        "\n".join(json.dumps(event) for event in events) + "\n"
                    )
                    return SimpleNamespace(returncode=0)

                with patch.object(run_sdcpp.subprocess, "run", side_effect=child_process) as child:
                    with contextlib.redirect_stdout(io.StringIO()):
                        _, load, generations, _ = run_sdcpp.run_harness(
                            cfg, directory, Path("/fake/sd-bench"), Path("/fake/cache"), phases, sampler, profile
                        )
                child.assert_called_once()
                command = child.call_args.args[0]
                self.assertEqual(command[command.index("--runs") + 1], "3")
                self.assertEqual(command[:2], ["nsys", "profile"] if profile else ["/fake/sd-bench", "--diffusion-model"])
                self.assertEqual(len(generations), 3)
                self.assertTrue(load["ok"])
                sampler.stop.assert_called_once()

    def test_coordinator_keeps_extra_profile_out_of_summary(self):
        torch = SimpleNamespace(cuda=SimpleNamespace(
            is_available=lambda: True,
            get_device_properties=lambda _: SimpleNamespace(uuid="fake"),
            get_device_name=lambda _: "fake GPU",
        ))
        process = Mock()
        process.memory_info.return_value.rss = GIB
        sampler = Mock()
        sampler.peak_between.return_value = None
        cfg = json.loads((ROOT / "configs/a100-flux-klein-bf16.json").read_text())
        cfg["model"]["revision"] = "a" * 40
        cfg["protocol"].update(first_runs=1, warmup_runs=1, measured_runs=1)
        cfg["workload"]["num_inference_steps"] = 1
        load = dict(load_total_s=2, allocated_after_load_gib=1, peak_alloc_during_load_gib=2, components={}, is_distilled=True)
        with tempfile.TemporaryDirectory() as temp, contextlib.ExitStack() as stack:
            directory = Path(temp)
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            stack.enter_context(patch.dict(sys.modules, torch=torch, psutil=SimpleNamespace(Process=lambda: process)))
            for name, value in [("DeviceMemorySampler", sampler), ("collect_environment", None),
                                ("load_pipeline", (Mock(), load)), ("StageRecorder", Mock()),
                                ("peak_rss_gib", 1), ("sh", "fake device")]:
                stack.enter_context(patch.object(run_flux, name, return_value=value))
            generate = stack.enter_context(patch.object(run_flux, "generate_once", side_effect=lambda *args: fake_generation()))
            profile = stack.enter_context(patch.object(run_flux, "run_profile", side_effect=lambda generate, _: generate()))
            with run_status(directory):
                run_flux.benchmark(cfg, directory, do_profile=True)
            self.assertEqual(generate.call_count, 4)  # first + warmup + measured + profile
            profile.assert_called_once()
            summary = json.loads((directory / "summary.json").read_text())
            self.assertEqual(summary["measured_runs"], 1)
            self.assertEqual(len(read_rows(directory / "runs.csv")), 3)
            sampler.stop.assert_called_once()

    def run_fake_protocol(self, directory, generate):
        sampler = Mock()
        sampler.peak_between.return_value = None
        process = Mock()
        process.memory_info.return_value = SimpleNamespace(rss=GIB)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            with run_status(directory), sampling(sampler):
                rows = measure_generations(generate, ["first", "warmup", "measured"], 1,
                                           directory, sampler, process)
        return rows, sampler

    def test_fake_engine_writes_protocol_and_stops_monitor(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            rows, sampler = self.run_fake_protocol(directory, fake_generation)
            self.assertEqual(len(rows), 3)
            self.assertEqual(len(read_rows(directory / "stages.csv")), 12)
            self.assertEqual(json.loads((directory / "status.json").read_text())["status"], "complete")
            self.assertTrue((directory / "images/first.png").exists())
            self.assertTrue((directory / "images/measured-0.png").exists())
            sampler.stop.assert_called_once()

    def test_interrupt_preserves_completed_csv_rows_and_traceback(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            generate = Mock(side_effect=[fake_generation(), KeyboardInterrupt("test interruption")])
            with self.assertRaises(KeyboardInterrupt):
                self.run_fake_protocol(directory, generate)
            self.assertEqual(len(read_rows(directory / "runs.csv")), 1)
            self.assertEqual(len(read_rows(directory / "stages.csv")), 4)
            status = json.loads((directory / "status.json").read_text())
            self.assertEqual(status["status"], "failed")
            self.assertIn("KeyboardInterrupt", status["traceback"])
            self.assertIn("started_utc", status)

    def test_monitor_stops_on_error(self):
        monitor = Mock()
        with self.assertRaisesRegex(RuntimeError, "load failed"):
            with sampling(monitor):
                raise RuntimeError("load failed")
        monitor.start.assert_called_once()
        monitor.stop.assert_called_once()

    def test_missing_denoise_callback_fails(self):
        image, result = fake_generation()
        result["stages"] = [stage for stage in result["stages"] if stage["stage"] != "denoise_step_0"]
        with self.assertRaisesRegex(AssertionError, "transformer calls per logical step"):
            result_row(0, "first", image, result, 1, Mock(), Mock())


if __name__ == "__main__":
    unittest.main()
