import unittest

import numpy as np

from shiftadd_flux.activation import quantize_grouped, quantize_per_token


class GroupedActivationTest(unittest.TestCase):
    def test_outlier_is_local_and_zero_group_is_finite(self):
        x = np.array([[0.0, 0.0, 1.0, 2.0, 100.0, 0.0, -1.0, -2.0]], np.float32)
        q, scale = quantize_grouped(x, 4, 2)
        actual = (q.astype(np.float32) * scale).reshape(x.shape)
        self.assertTrue(np.isfinite(actual).all())
        np.testing.assert_array_equal(actual[:, :2], 0)
        self.assertLess(np.max(np.abs(actual[:, 2:4] - x[:, 2:4])), 0.15)
        q0, s0 = quantize_per_token(x, 4)
        q1, s1 = quantize_grouped(x, 4, 8)
        np.testing.assert_array_equal((q1 * s1).reshape(x.shape), q0 * s0)
        with self.assertRaises(ValueError):
            quantize_grouped(x, 4, 3)

    def test_group_intersections_match_dense_with_zero_exponents(self):
        from shiftadd_flux.activation import grouped_signed_plane_matmul
        from shiftadd_flux.format import decode, encode

        rng = np.random.default_rng(172)
        w = rng.normal(size=(5, 384)).astype(np.float32)
        w[0] = 0
        p, e = encode(w, planes=3, group_size=128)
        x = rng.normal(size=(7, 384)).astype(np.float32)
        for group in (32, 96, 128, 384):
            for bits in (4, 8):
                q, s = quantize_grouped(x, bits, group)
                got = grouped_signed_plane_matmul(q, s, p, e)
                expected = (q.astype(np.float32) * s).reshape(x.shape) @ decode(p, e).T
                np.testing.assert_allclose(got, expected, rtol=3e-5, atol=3e-5)
                np.testing.assert_array_equal(got[:, 0], 0)
        with self.assertRaises(ValueError):
            grouped_signed_plane_matmul(q, -s, p, e)

    def test_pipeline_hook_matches_numpy_for_batched_tokens(self):
        import torch

        from shiftadd_flux.activation_hooks import ActivationQuantization

        torch.manual_seed(23)
        model = torch.nn.Sequential(torch.nn.Linear(128, 9, bias=False)).to(torch.bfloat16)
        x = torch.randn(2, 3, 128).to(torch.bfloat16)
        baseline = model(x).detach()
        for bits in (4, 8):
            for group in (32, 128):
                q, s = quantize_grouped(x.float().numpy().reshape(-1, 128), bits, group)
                reconstructed = torch.from_numpy((q * s).reshape(x.shape)).to(x.dtype)
                expected = model(reconstructed).detach()
                with ActivationQuantization(model, ["0"], bits, group_size=group) as scope:
                    actual = model(x).detach()
                    self.assertEqual(scope.calls, 1)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(model(x), baseline, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            ActivationQuantization(model, ["0"], 4, group_size=96)
        self.assertEqual(len(model[0]._forward_pre_hooks), 0)
