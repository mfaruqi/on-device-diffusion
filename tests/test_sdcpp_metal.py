"""Apple bring-up must preserve units, explicit placement and absence of GPU counters."""
import json
import sys
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from benchlib import GIB
from sdcpp_device import child_peak_rss_gib, MacMemorySampler, SdcppDevice
from sdcpp_engine import audit_log, check_config

ROOT = Path(__file__).resolve().parents[1]


class MetalTests(unittest.TestCase):
    def test_rss_platform_units(self):
        self.assertEqual(child_peak_rss_gib(GIB, 'Darwin'), 1)
        self.assertEqual(child_peak_rss_gib(GIB / 1024, 'Linux'), 1)

    def test_system_memory_is_not_a_gpu_counter(self):
        with tempfile.TemporaryDirectory() as d, patch('sdcpp_device.sh', return_value='thermal unavailable'), patch(
                'psutil.virtual_memory', return_value=SimpleNamespace(total=24*GIB, available=12*GIB)), patch(
                'psutil.swap_memory', return_value=SimpleNamespace(used=0)):
            sampler = MacMemorySampler(100, Path(d))
            sampler.start()
            sampler.stop()
            self.assertIsNone(sampler.peak_between(0, 1e20))
            self.assertGreater(sampler.summary()['system_total_gib'], 0)
            self.assertTrue((Path(d) / 'system-memory.csv').exists())

    def test_metal_keeps_explicit_quantization_and_placement(self):
        c = json.loads((ROOT / 'configs/m5-sdcpp-flux-klein-q4-512-smoke.json').read_text())
        c['model']['resolved_at_utc'] = 'test'
        c['model']['sha256'] = {c['model']['files'][r]: v['sha256'] for r, v in c['model']['components'].items()}
        def git_result(args):
            return c['engine']['ggml_commit'] if 'ggml' in args[2] else c['engine']['commit']
        with patch('sdcpp_engine.sh', side_effect=git_result):
            check_config(c)
            c['engine_settings']['params_backend'] = 'cpu'
            with self.assertRaises(AssertionError):
                check_config(c)
            c['engine_settings']['params_backend'] = 'MTL0'
            c['precision'] = 'bfloat16'
            with self.assertRaises(AssertionError):
                check_config(c)

    def test_automatic_vae_tiling_and_cpu_placement_fail_audit(self):
        s = {'params_backend': 'metal', 'diffusion_fa': False}
        log = '1.0 prepared params backend buffers ON CPU\n2.0 VAE decode ran out of memory; retrying with spatial tiling'
        audit = audit_log(log, s, 3)
        self.assertIn('automatic_vae_tiling', audit['problems'])
        self.assertTrue(any('off metal' in p for p in audit['problems']))

    def test_sleep_guard_and_clock_divergence(self):
        device = SdcppDevice.__new__(SdcppDevice)
        device.metal = True
        device.cfg = {'protocol': {'prevent_idle_sleep': True}}
        self.assertEqual(device.wrap_command(['sd-bench']), ['/usr/bin/caffeinate', '-is', 'sd-bench'])
        device.sampler = SimpleNamespace(summary=lambda: {'wall_minus_monotonic_elapsed_s': 659}, note='test')
        self.assertFalse(device.summary_fields()['clock_continuity_pass'])
        device.sampler.summary = lambda: {'wall_minus_monotonic_elapsed_s': .001}
        self.assertTrue(device.summary_fields()['clock_continuity_pass'])
