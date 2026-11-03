import unittest

import numpy as np

from shiftadd_flux.format import decode, encode


class FormatTest(unittest.TestCase):
    def test_exact_power_two(self):
        w = np.tile(np.array([-0.25, 0.25], np.float32), (3, 64))
        p, e = encode(w, planes=1)
        np.testing.assert_array_equal(decode(p, e), w)
        self.assertEqual(p.nbytes + e.nbytes, 51)

    def test_zero(self):
        p, e = encode(np.zeros((2, 128), np.float32))
        np.testing.assert_array_equal(decode(p, e), np.zeros((2, 128)))

    def test_independent_packed_sum(self):
        w = np.random.default_rng(9).normal(size=(4, 256)).astype(np.float32)
        p, e = encode(w, planes=3)
        expected = np.zeros_like(w)
        for row in range(4):
            for col in range(256):
                for plane in range(3):
                    group, within = divmod(col, 128)
                    bit = (int(p[plane, row, group, within // 8]) >> (within % 8)) & 1
                    exponent = int(e[plane, row, group])
                    expected[row, col] += (2 * bit - 1) * (0 if exponent == -128 else 2.0**exponent)
        np.testing.assert_array_equal(decode(p, e), expected)


if __name__ == "__main__":
    unittest.main()
