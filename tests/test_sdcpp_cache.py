"""Cache requests must fail rather than silently running without the policy."""
import sys
import json
import tempfile
from unittest.mock import patch
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from sdcpp_engine import audit_cache, audit_log, cache_arguments, check_execution_options, conditioning_hits
from measurement import check_optimizations


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.cache = dict(mode='easycache', reuse_threshold=.2, start_percent=.15, end_percent=.95)
        self.enabled = 'EasyCache enabled - threshold: 0.200, start: 0.15, end: 0.95\n'

    def test_only_explicit_verified_policy_and_finite_parameters(self):
        self.assertEqual(cache_arguments(None), {})
        for key, value in [('mode', 'teacache'), ('mode', 'ucache'), ('start_percent', 1),
                           ('reuse_threshold', float('nan')), ('end_percent', -.1)]:
            with self.subTest(key=key, value=value), self.assertRaises(AssertionError):
                cache_arguments({**self.cache, key: value})
        with self.assertRaises(AssertionError):
            cache_arguments({'mode': 'easycache'})
        with self.assertRaises(AssertionError):
            check_optimizations({'optimizations': {'step_cache': self.cache, 'vae_tiling': True}},
                                {'step_cache', 'vae_tiling'}, {'step_cache'})

    def test_audit_requires_each_generation_and_exact_logged_parameters(self):
        log = self.enabled + 'EasyCache skipped 12/50 steps\n'
        log += self.enabled + 'EasyCache completed without skipping steps\n'
        self.assertEqual(audit_cache(log, self.cache, 2)['steps_skipped'], [12, 0])
        for bad in [log.replace('0.200', '0.100'), log.replace(self.enabled, ''),
                    self.enabled + 'EasyCache requested but not supported for this model type']:
            with self.assertRaises(AssertionError):
                audit_cache(bad, self.cache, 2)
        with self.assertRaises(AssertionError):
            audit_cache(log, None, 2)

    def test_jetson_rejects_cache_argument_drift_and_stale_binary(self):
        import run_jetson_sdcpp as runner
        root = Path(__file__).resolve().parents[1]
        cfg = json.loads((root / 'configs/jetson-flux-klein-base-q4-512-disk-smoke.json').read_text())
        cfg['optimizations']['step_cache'] = self.cache
        with self.assertRaisesRegex(ValueError, 'Cache harness arguments differ'):
            runner.check_config(cfg)

        cfg['harness_arguments'].update(cache_arguments(self.cache))
        self.assertEqual(len(runner.check_config(cfg)), 2)
        with tempfile.TemporaryDirectory() as directory, patch.object(
                runner.subprocess, 'check_output', return_value='{"profile_generation":false}'):
            with self.assertRaisesRegex(RuntimeError, 'EasyCache-capable'):
                runner.benchmark(cfg, Path(directory), Path('engine'), Path('old-binary'), Path('models'))
        cfg['harness_arguments']['cache-threshold'] = .5
        with self.assertRaisesRegex(ValueError, 'Cache harness arguments differ'):
            runner.check_config(cfg)

    def test_a100_exact_reuse_and_io_policies_are_explicit(self):
        root = Path(__file__).resolve().parents[1]
        cfg = json.loads((root / 'configs/a100-sdcpp-flux-klein-bf16.resolved.json').read_text())
        check_execution_options(cfg)
        cfg['engine_settings']['conditioning_cache_size'] = 1
        with self.assertRaises(AssertionError):
            check_execution_options(cfg)
        cfg['optimizations']['conditioning_cache_size'] = 1
        check_execution_options(cfg)
        self.assertEqual(conditioning_hits(cfg, 3), [0, 1, 1])
        cfg['workload']['guidance_scale'] = 4
        with self.assertRaises(AssertionError):
            check_execution_options(cfg)
        cfg['optimizations']['conditioning_cache_size'] = 2
        cfg['engine_settings']['conditioning_cache_size'] = 2
        check_execution_options(cfg)
        for option, field, value in [('prefetch', 'disable_prefetch', False), ('mmap', 'mmap', True)]:
            cfg['optimizations'][option] = value
            with self.assertRaises(AssertionError):
                check_execution_options(cfg)
            cfg['engine_settings'][field] = not value if option == 'prefetch' else value
            check_execution_options(cfg)

    def test_a100_conditioning_log_count_must_match_observations(self):
        settings = {'params_backend': 'cuda0', 'diffusion_fa': False}
        log = '1.0 conditioning cache hit\n2.0 conditioning cache hit\n'
        self.assertTrue(audit_log(log, settings, 3)['problems'])
        self.assertFalse(audit_log(log, settings, 3, expected_conditioning_hits=2)['problems'])
        self.assertTrue(audit_log('', settings, 3, expected_conditioning_hits=2)['problems'])



if __name__ == '__main__':
    unittest.main()
