import unittest

import numpy as np

from shiftadd_flux.activation import quantize_grouped
from shiftadd_flux.scaled_plane_arithmetic import grouped_scaled_plane_matmul


class ScaledPlaneArithmeticTest(unittest.TestCase):
    def test_arbitrary_scales_and_group_intersections(self):
        rng = np.random.default_rng(719)
        signs = rng.integers(0, 2, (4, 7, 3, 128), dtype=np.uint8)
        packed = np.packbits(signs, axis=-1, bitorder="little")
        scales = rng.uniform(0, 0.5, (4, 7, 3)).astype(np.float32)
        scales[:, 0] = 0
        weights = (
            ((signs.astype(np.float32) * 2 - 1) * scales[..., None]).sum(axis=0).reshape(7, 384)
        )
        x = rng.normal(size=(9, 384)).astype(np.float32)
        for group in (32, 96, 128, 384):
            for bits in (4, 8):
                q, a = quantize_grouped(x, bits, group)
                actual = grouped_scaled_plane_matmul(q, a, packed, scales)
                expected = (q.astype(np.float32) * a).reshape(x.shape) @ weights.T
                np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=2e-5)
                np.testing.assert_array_equal(actual[:, 0], 0)
        with self.assertRaises(ValueError):
            grouped_scaled_plane_matmul(q, a, packed, -scales)
