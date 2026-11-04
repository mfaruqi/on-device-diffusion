import unittest

import numpy as np

from shiftadd_flux.activation_storage import pack_codes, unpack_codes


class ActivationStorageTest(unittest.TestCase):
    def test_known_layout_all_codes_and_odd_padding(self):
        codes = np.arange(-8, 8, dtype=np.int8)
        payload = pack_codes(codes)
        np.testing.assert_array_equal(
            payload,
            np.array([0x98, 0xBA, 0xDC, 0xFE, 0x10, 0x32, 0x54, 0x76], np.uint8),
        )
        np.testing.assert_array_equal(unpack_codes(payload, (2, 8)), codes.reshape(2, 8))
        odd = pack_codes(codes[:3])
        self.assertEqual(int(odd[-1]), 10)
        np.testing.assert_array_equal(unpack_codes(odd, (3,)), codes[:3])
        with self.assertRaises(ValueError):
            unpack_codes(np.array([0xF1], np.uint8), (1,))
        with self.assertRaises(ValueError):
            pack_codes(np.array([8], np.int8))
        with self.assertRaises(ValueError):
            unpack_codes(payload, (20,))

    def test_int8_and_empty_roundtrip(self):
        codes = np.arange(-128, 128, dtype=np.int16).astype(np.int8).reshape(16, 16)
        np.testing.assert_array_equal(unpack_codes(pack_codes(codes, 8), codes.shape, 8), codes)
        np.testing.assert_array_equal(unpack_codes(pack_codes(codes[:0]), (0, 16)), codes[:0])
