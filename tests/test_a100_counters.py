import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import profile_a100_counters as counters


class CounterTests(unittest.TestCase):
    def fixture(self, path, missing=None, unit='nsecond', invalid=None):
        with path.open('w') as stream:
            stream.write('==PROF== fixture header\n')
            writer = csv.writer(stream, quoting=csv.QUOTE_ALL)
            writer.writerow(['ID', 'Process ID', 'Context', 'Stream', 'Kernel Name',
                             'Metric Name', 'Metric Unit', 'Metric Value'])
            for metric in counters.METRICS:
                if metric == missing:
                    continue
                value = 1000 if metric == counters.DURATION else 100
                value = invalid if invalid is not None and metric == counters.READ else value
                units = unit if metric == counters.DURATION else 'byte' if metric in (counters.READ, counters.WRITE) else 'inst'
                writer.writerow([0, 123, 1, 7, 'gemm', metric, units, value])

    def test_stage_from_nvtx_renamed_kernel(self):
        from profile_a100_counters import stage_of
        self.assertEqual(stage_of('denoise_step_0/void at::native::elementwise_kernel<(int)128, (int)4>'), 'denoise_step_0')
        self.assertEqual(stage_of('text_encode/ampere_bf16_s16816gemm'), 'text_encode')
        self.assertEqual(stage_of('vae_decode'), 'vae_decode')

    def test_units_and_fma_accounting(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'raw.csv'
            self.fixture(path)
            point = counters.read_counters(path)[0]
            self.assertAlmostEqual(point['duration_s'], 1e-6)
            self.assertEqual(point['hbm_bytes'], 200)
            self.assertEqual(point['tensor_ops'], 300)
            self.assertEqual(point['fp32_add_mul_fma_ops'], 400)
            self.assertEqual(point['counted_ops_per_byte'], 3.5)

    def test_missing_unavailable_and_wrong_units_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'raw.csv'
            for options in ({'missing': counters.READ}, {'unit': 'usecond'}, {'invalid': 'nan'}, {'invalid': 'n/a'}):
                with self.subTest(options=options):
                    self.fixture(path, **options)
                    with self.assertRaises(ValueError):
                        counters.read_counters(path)

    def test_real_ncu_2026_wide_export(self):
        # Fixture is the selected columns of the preserved probe CSV, not synthetic counts.
        path = Path(__file__).parent / 'fixtures' / 'ncu-probe-wide.csv'
        point = counters.read_counters(path)[0]
        self.assertAlmostEqual(point['duration_s'], 110336e-9)
        self.assertEqual(point['metrics'][counters.TENSOR[0]], 2 * 2048**3)
        self.assertEqual(point['hbm_bytes'], 13052288 + 1848192)
        self.assertEqual(point['fp32_add_mul_fma_ops'], 4194304)

    def test_capture_disabled_until_explicit_start(self):
        argv = counters.ncu_command({'ncu': 'ncu', 'cache_control': 'none'}, Path('out'), ['--child', 'probe'])
        self.assertEqual(argv[argv.index('--profile-from-start') + 1], 'off')
        self.assertEqual(argv[argv.index('--clock-control') + 1], 'none')
        self.assertNotIn('--launch-count', argv)

    def test_reference_is_unchanged_and_pinned(self):
        plan = counters.read_json(counters.REPO_ROOT / 'configs/a100-flux-klein-torch28-ncu.json')
        reference = counters.REPO_ROOT / plan['reference_config']
        self.assertEqual(counters.digest(reference), plan['reference_config_sha256'])
        cfg = json.loads(reference.read_text())
        self.assertEqual(cfg['protocol']['measured_runs'], 10)
        self.assertEqual(cfg['workload']['num_inference_steps'], 4)
        self.assertEqual(len(cfg['model']['revision']), 40)


if __name__ == '__main__':
    unittest.main()
