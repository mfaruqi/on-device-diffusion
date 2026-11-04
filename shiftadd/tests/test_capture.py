import unittest

import numpy as np
import torch

from shiftadd_flux.capture import ActivationCapture


class CaptureTest(unittest.TestCase):
    def test_capture_is_bounded_and_preserves_output(self):
        model = torch.nn.Sequential(torch.nn.Linear(4, 2, bias=False))
        x = torch.arange(40, dtype=torch.float32).reshape(10, 4)
        expected = model(x)
        with ActivationCapture(model, ["0"], tokens_per_call=3, max_calls=1) as capture:
            actual = model(x)
            torch.testing.assert_close(expected, actual)
            np.testing.assert_array_equal(capture.samples["0"][0], x[[0, 3, 6]].numpy())
            with self.assertRaises(RuntimeError):
                model(x)
        torch.testing.assert_close(model(x), expected)


if __name__ == "__main__":
    unittest.main()
