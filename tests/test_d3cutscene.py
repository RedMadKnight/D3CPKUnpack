import os
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import d3cutscene


def ogg_page(flags, serial, sequence, payload=b"x"):
    if len(payload) > 255:
        raise ValueError("test helper only supports one Ogg segment")
    return (
        b"OggS"
        + bytes([0, flags])
        + struct.pack("<qIII", 0, serial, sequence, 0)
        + bytes([1, len(payload)])
        + payload
    )


class D3CutsceneTests(unittest.TestCase):
    def test_read_bits_msb_first(self):
        self.assertEqual(d3cutscene.read_bits(bytes([0b10110000]), 0, 4), 0b1011)
        self.assertEqual(d3cutscene.read_bits(bytes([0b10110000]), 2, 3), 0b110)

    def test_validate_complete_ogg_range(self):
        data = b"".join(
            [
                ogg_page(0x02, 1, 0),
                ogg_page(0x02, 2, 0),
                ogg_page(0x04, 1, 1),
                ogg_page(0x04, 2, 1),
            ]
        )
        with tempfile.TemporaryFile() as f:
            f.write(data)
            f.flush()
            result = d3cutscene.validate_ogg_range(f, 0, len(data))
        self.assertTrue(result.ok)
        self.assertEqual(result.pages, 4)
        self.assertEqual(result.streams, frozenset({1, 2}))

    def test_validate_missing_eos(self):
        data = ogg_page(0x02, 1, 0) + ogg_page(0x00, 1, 1)
        with tempfile.TemporaryFile() as f:
            f.write(data)
            f.flush()
            result = d3cutscene.validate_ogg_range(f, 0, len(data))
        self.assertFalse(result.ok)
        self.assertIn("missing EOS", result.status)

    def test_scan_ogg_groups(self):
        data = (
            ogg_page(0x02, 1, 0)
            + ogg_page(0x04, 1, 1)
            + ogg_page(0x02, 2, 0)
            + ogg_page(0x04, 2, 1)
        )
        with tempfile.NamedTemporaryFile(delete=False) as f:
            path = f.name
            f.write(data)
        try:
            groups = d3cutscene.scan_ogg_groups(path, 0)
        finally:
            os.unlink(path)
        self.assertEqual(len(groups), 2)
        self.assertEqual(groups[0].serials, frozenset({1}))
        self.assertEqual(groups[1].serials, frozenset({2}))

    def test_output_path_rejects_unsafe_components(self):
        entry = d3cutscene.CutsceneEntry(0, r"..\bad.ogv", 0, 1, 1, 0, 0)
        with self.assertRaises(ValueError):
            d3cutscene.output_path_for_entry("out", entry, flat=False)


if __name__ == "__main__":
    unittest.main()
