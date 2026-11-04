import unittest

import numpy as np

from shiftadd_flux.format import decode
from shiftadd_flux.radix import encode_radix


class WeightedFitTest(unittest.TestCase):
    def test_diagonal_objective_does_not_increase(self):
        rng = np.random.default_rng(51)
        w = rng.normal(size=(8, 256)).astype(np.float32)
        importance = np.exp(rng.normal(size=256) * 3).astype(np.float32)
        p, e = encode_radix(w)
        unweighted = decode(p, e)
        p, e = encode_radix(w, importance=importance)
        weighted = decode(p, e)
        before = np.sum((w - unweighted) ** 2 * importance)
        after = np.sum((w - weighted) ** 2 * importance)
        self.assertLessEqual(after, before + 1e-4)
        with self.assertRaises(ValueError):
            encode_radix(w, importance=np.ones(3))


if __name__ == "__main__":
    unittest.main()
