import unittest

import numpy as np

from shiftadd_flux.format import decode
from shiftadd_flux.radix import encode_radix


class RadixTest(unittest.TestCase):
    def test_exact_representable_levels(self):
        w = np.tile(np.arange(-15, 16, 2, dtype=np.float32) * 0.125, (4, 8))
        p, e = encode_radix(w, 4)
        np.testing.assert_array_equal(decode(p, e), w)

    def test_zero_and_nonfinite(self):
        p, e = encode_radix(np.zeros((3, 128), np.float32))
        self.assertFalse(decode(p, e).any())
        with self.assertRaises(ValueError):
            encode_radix(np.full((1, 128), np.nan))

    def test_best_exponent_search(self):
        w = np.random.default_rng(12).normal(size=(4, 128)).astype(np.float32)
        p, e = encode_radix(w, 4)
        actual = np.square(w - decode(p, e)).sum(axis=-1)
        center = np.floor(np.log2(np.max(np.abs(w), axis=-1) / 15))
        candidates = []
        for offset in [-2, -1, 0, 1, 2]:
            unit = np.exp2(center + offset)[:, None]
            q = np.rint((w / unit + 15) / 2).clip(0, 15)
            candidates.append(np.square(w - (2 * q - 15) * unit).sum(axis=-1))
        np.testing.assert_allclose(actual, np.min(candidates, axis=0))


if __name__ == "__main__":
    unittest.main()
