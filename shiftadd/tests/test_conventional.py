import unittest

import numpy as np

from shiftadd_flux.conventional import decode_uniform, encode_uniform


class UniformTest(unittest.TestCase):
    def test_exact_levels_zero_and_payload(self):
        w = np.tile(np.array([-7.0, -3.0, 0.0, 7.0], np.float32), (2, 32))
        w[0] = 0
        p, s = encode_uniform(w, 4)
        np.testing.assert_array_equal(decode_uniform(p, s, w.shape, 4), w)
        self.assertEqual(p.nbytes + s.nbytes, 136)
        self.assertTrue(np.isfinite(s).all())

    def test_search_no_worse_than_absmax(self):
        w = np.random.default_rng(8).normal(size=(3, 256)).astype(np.float32)
        for bits in (4, 8):
            p, s = encode_uniform(w, bits)
            reconstructed = decode_uniform(p, s, w.shape, bits)
            g = w.reshape(3, 2, 128)
            b = 2 ** (bits - 1) - 1
            scale = np.abs(g).max(axis=-1, keepdims=True) / b
            reference = (np.rint(g / scale).clip(-b, b) * scale).reshape(w.shape)
            self.assertLessEqual(
                float(np.square(w - reconstructed).sum()),
                float(np.square(w - reference).sum()) + 1e-5,
            )
