import unittest

import numpy as np

from shiftadd_flux.compact_radix import compact_exponents, expand_exponents


class CompactRadixTest(unittest.TestCase):
    def test_zero_and_signed_exponents_roundtrip(self):
        base = np.array([[-128, -126, 0, 119]], np.int8)
        for planes in (1, 4, 8):
            expanded = expand_exponents(base, planes)
            np.testing.assert_array_equal(compact_exponents(expanded), base)
            np.testing.assert_array_equal(expanded[:, 0, 0], -128)
        invalid = expand_exponents(base, 4)
        invalid[2, 0, 2] += 1
        with self.assertRaises(ValueError):
            compact_exponents(invalid)
        with self.assertRaises(ValueError):
            expand_exponents(np.array([[127]], np.int8), 2)
