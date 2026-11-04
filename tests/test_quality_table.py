import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import quality_table as qt  # noqa: E402


def prompts(correct, scores, tags=('a', 'b')):
    """Two prompts (one per tag), two images each."""
    return {p: {'tag': tags[p], 'geneval': correct[p], 'clip': scores[p]} for p in range(2)}


class QualityTableTest(unittest.TestCase):
    def test_geneval_is_mean_over_tasks(self):
        # task a: 2/2 correct, task b: 0/2 -> 0.5, not the image mean weighted by task size
        p = {0: {'tag': 'a', 'geneval': [1.0, 1.0]}, 1: {'tag': 'b', 'geneval': [0.0, 0.0]},
             2: {'tag': 'b', 'geneval': [0.0, 0.0]}}
        self.assertAlmostEqual(qt.metric_value(p, [0, 1, 2], 'geneval'), 0.5)

    def test_identical_runs_have_zero_delta_and_interval(self):
        ref = prompts([[1.0, 0.0], [1.0, 1.0]], [[.3, .2], [.25, .35]])
        delta, ci = qt.paired_delta(ref, ref, 'clip')
        self.assertEqual((delta, ci), (0.0, (0.0, 0.0)))

    def test_constant_shift_is_recovered(self):
        ref = prompts([[1.0, 0.0], [1.0, 1.0]], [[.3, .2], [.25, .35]])
        var = prompts([[1.0, 0.0], [1.0, 1.0]], [[.31, .21], [.26, .36]])
        delta, ci = qt.paired_delta(ref, var, 'clip')
        self.assertAlmostEqual(delta, .01)
        self.assertAlmostEqual(ci[0], .01)
        self.assertAlmostEqual(ci[1], .01)

    def test_mismatched_seed_sets_are_refused(self):
        ref = prompts([[1.0, 0.0], [1.0, 1.0]], [[.3, .2], [.25, .35]])
        var = prompts([[1.0], [1.0, 1.0]], [[.3], [.25, .35]])
        with self.assertRaises(AssertionError):
            qt.paired_delta(ref, var, 'clip')


if __name__ == '__main__':
    unittest.main()
