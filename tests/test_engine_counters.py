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


if __name__ == '__main__':
    unittest.main()
