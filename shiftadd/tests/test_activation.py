import unittest

import torch

from shiftadd_flux.activation_hooks import ActivationQuantization


class ActivationTest(unittest.TestCase):
    def test_scoped_quantization(self):
        model = torch.nn.Sequential(torch.nn.Linear(128, 128, bias=False))
        x = torch.linspace(-1, 1, 128).reshape(1, 128)
        baseline = model(x).detach()
        with ActivationQuantization(model, ["0"], 4) as context:
            actual = model(x).detach()
            q = (x * 7).round().clamp(-7, 7) / 7
            torch.testing.assert_close(actual, torch.nn.functional.linear(q, model[0].weight))
        self.assertEqual(context.calls, 1)
        self.assertEqual(context.calls_by_module, {"0": 1})
        torch.testing.assert_close(model(x), baseline)
        with self.assertRaises(ValueError):
            ActivationQuantization(model, ["absent"], 4)

    def test_zero(self):
        model = torch.nn.Sequential(torch.nn.Linear(128, 2, bias=False))
        with ActivationQuantization(model, ["0"], 8):
            y = model(torch.zeros((1, 128)))
            self.assertTrue(torch.isfinite(y).all())
            self.assertFalse(y.any())


if __name__ == "__main__":
    unittest.main()
