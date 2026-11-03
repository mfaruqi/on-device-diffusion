import unittest

from shiftadd_flux.reporting.energy import integrate


class EnergyTest(unittest.TestCase):
    def setUp(self):
        self.samples = [
            dict(
                timestamp_s=t,
                power_w=10 + 2 * t,
                clock_id="host-boot-monotonic",
                power_scope="device",
            )
            for t in range(5)
        ]
        self.window = dict(
            start_s=0.5,
            end_s=3.5,
            images=2,
            clock_id="host-boot-monotonic",
            power_scope="device",
            max_gap_s=1.0,
        )

    def test_linear_power_exact_with_fractional_boundaries(self):
        # Integral of 10+2t from .5 to 3.5 is 42 joules.
        result = integrate(self.samples, self.window)
        self.assertEqual(result["gross_energy_j"], 42.0)
        self.assertEqual(result["gross_j_per_image"], 21.0)
        self.assertEqual(result["samples_within_window"], 3)

    def test_rejects_inadequate_or_incompatible_measurement(self):
        for change in (
            {"start_s": -1},
            {"images": 0},
            {"power_scope": "board"},
            {"clock_id": "other"},
            {"max_gap_s": 0.5},
            {"start_s": 1.5, "end_s": 2.5},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                integrate(self.samples, dict(self.window, **change))
        with self.assertRaises(ValueError):
            integrate(self.samples[::-1], self.window)
        bad = [dict(r) for r in self.samples]
        bad[2]["power_w"] = float("nan")
        with self.assertRaises(ValueError):
            integrate(bad, self.window)
