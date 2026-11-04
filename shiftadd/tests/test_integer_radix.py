import unittest

import numpy as np

from shiftadd_flux.activation import quantize_grouped
from shiftadd_flux.format import decode
from shiftadd_flux.integer_radix import grouped_integer_matmul, to_integer_codes
from shiftadd_flux.radix import encode_radix


class IntegerRadixTest(unittest.TestCase):
    def test_exact_integer_bridge_and_group_sums(self):
        rng = np.random.default_rng(68)
        w = rng.normal(size=(3, 256)).astype(np.float32)
        w[0] = 0
        for planes, target in ((4, 4), (4, 8), (5, 8)):
            p, e = encode_radix(w, planes)
            codes, base, multiplier, offset = to_integer_codes(p, e, target)
            scale = np.where(base == -128, 0.0, np.exp2(base.astype(np.float32)))
            restored = (
                (codes.astype(np.float32) * multiplier + offset) * scale[..., None]
            ).reshape(w.shape)
            np.testing.assert_array_equal(restored, decode(p, e))
            for group in (32, 128, 256):
                x = rng.normal(size=(4, 256)).astype(np.float32)
                q, s = quantize_grouped(x, 4, group)
                actual = grouped_integer_matmul(q, s, codes, base, multiplier, offset)
                ref = (q.astype(np.float32) * s).reshape(x.shape) @ decode(p, e).T
                np.testing.assert_allclose(actual, ref, rtol=3e-5, atol=3e-5)
        with self.assertRaises(ValueError):
            to_integer_codes(p, e, 4)
