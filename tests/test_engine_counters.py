import gzip
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import profile_engine_counters as pec  # noqa: E402


class FindPassTest(unittest.TestCase):
    def test_prefers_time_heavy_step_over_longer_text_encoder_period(self):
        # 36 cheap text-encoder layers (8 launches each) repeat more often and over longer periods than the
        # 4 denoising steps (40 launches each, one step-dependent scheduler kernel), but carry far less time.
        names, durations = [], []
        for _ in range(36):
            names += [f'te{i}' for i in range(8)]
            durations += [1.0] * 8
        for step in range(4):
            names += [f'dit{i}' for i in range(39)] + [f'sched{step}']
            durations += [100.0] * 39 + [1.0]
        start, period = pec.find_pass(names, durations, 4)
        self.assertEqual(period, 40)
        self.assertEqual(names[start], 'dit0')
        self.assertGreaterEqual(start, 36 * 8 + 2 * 40)  # the last full step whose successor also matched

    def test_plain_drops_namespace_qualifiers(self):
        a = 'fmha(PyTorchMemEffAttention::AttentionKernel<cutlass::bfloat16_t, arch::Sm80, 1>::Params)'
        b = 'fmha(PyTorchMemEffAttention::AttentionKernel<cutlass::bfloat16_t, cutlass::Sm80, 1>::Params)'
        self.assertNotEqual(a, b)
        self.assertEqual(pec.plain(a), pec.plain(b))

    def test_sequence_reads_gzip_export(self):
        csv_text = ('"ID","Process ID","Kernel Name","gpu__time_duration.sum"\n'
                    '"","","","ns"\n"0","7","k0","1,000"\n"1","7","k1","2000"\n')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'sequence.csv.gz'
            with gzip.open(path, 'wt') as f:
                f.write(csv_text)
            self.assertEqual(pec.sequence(path), [('7', 'k0', 1000.0), ('7', 'k1', 2000.0)])


class JetsonBasisTest(unittest.TestCase):
    def test_sectors_are_scaled_to_bytes(self):
        import profile_a100_counters as counters
        metrics = counters.metric_list('jetson-l2-sysmem')
        units = {m: 'ns' if m == counters.DURATION else 'sector' if 'lts__' in m else 'inst' for m in metrics}
        values = {m: '0' for m in metrics}
        values.update({counters.DURATION: '1000', counters.JETSON_READ: '16,777,216', counters.JETSON_WRITE: '8388608'})
        head = ['ID', 'Process ID', 'Context', 'Stream', 'Kernel Name', *metrics]
        rows = [head, ['', '', '', '', '', *(units[m] for m in metrics)], ['0', '7', '1', '7', 'add', *(values[m] for m in metrics)]]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'window.csv'
            path.write_text('\n'.join(','.join(f'"{c}"' for c in r) for r in rows) + '\n')
            (row,) = counters.read_counters(path, 'jetson-l2-sysmem')
            self.assertEqual(row['hbm_bytes'], 768 * 2**20)  # 512 MiB read misses + 256 MiB writes
            with self.assertRaises(ValueError):
                counters.read_counters(path)  # the A100 basis needs dram__ columns


class MergeWindowsTest(unittest.TestCase):
    def write(self, path, metric, values, names=('add', 'gemm')):
        import profile_a100_counters as counters
        head = ['ID', 'Process ID', 'Kernel Name', counters.DURATION, metric]
        rows = [head, ['', '', '', 'ns', 'sector' if 'lts__' in metric else 'inst']]
        rows += [[str(i), '7', n, '1000', v] for i, (n, v) in enumerate(zip(names, values))]
        path.write_text('\n'.join(','.join(f'"{c}"' for c in r) for r in rows) + '\n')

    def test_single_counter_runs_merge_per_launch(self):
        import profile_a100_counters as counters
        bf16 = counters.TENSOR[0]
        with tempfile.TemporaryDirectory() as tmp:
            parts = [Path(tmp) / f'window-{i}.csv' for i in range(3)]
            self.write(parts[0], counters.JETSON_READ, ['1048576', '0'])
            self.write(parts[1], counters.JETSON_WRITE, ['524288', '32'])
            self.write(parts[2], bf16, ['0', '2000'])
            pec.merge_windows(parts, Path(tmp) / 'window.csv')
            add, gemm = counters.read_counters(Path(tmp) / 'window.csv', 'jetson-l2-sysmem', [bf16])
            self.assertEqual(add['hbm_bytes'], 48 * 2**20)
            self.assertEqual((gemm['hbm_bytes'], gemm['tensor_ops']), (1024, 2000))
            self.write(parts[2], bf16, ['0', '2000'], names=('add', 'other'))
            with self.assertRaises(ValueError):
                pec.merge_windows(parts, Path(tmp) / 'window.csv')


if __name__ == '__main__':
    unittest.main()
