"""Corrections 19, V19-2: the tolerated last seek entry (synthetic data only)."""

import unittest

import d3sno
from d3sno import sound_write as W
from test_stage15_1_content import fsb_with_table, packets


class Marker(unittest.TestCase):
    def test_version(self):
        self.assertGreaterEqual(tuple(int(x) for x in d3sno.__version__.split(".")), (0, 15, 2))


class ToleratedLastEntry(unittest.TestCase):
    def setUp(self):
        self.offsets = W.vorbis_packet_offsets(packets())

    def test_unmatched_last_entry_with_a_lower_sample(self):
        data, _ = fsb_with_table([(44000, self.offsets[44]), (10, self.offsets[88] + 1)])
        self.assertEqual(W.vorbis_seek_problems(data), [])
        strict = W.vorbis_seek_problems(data, allow_unmatched_last=False)
        self.assertEqual(len(strict), 2)  # packet start and sample order
        self.assertTrue(any("before the previous" in p for p in strict))

    def test_matched_last_entry_out_of_order_is_tolerated(self):
        # ``33`` V20-2 (replaces the V19-2 expectation): exempt even at a packet start
        data, _ = fsb_with_table([(44000, self.offsets[44]), (10, self.offsets[88])])
        self.assertEqual(W.vorbis_seek_problems(data), [])
        problems = W.vorbis_seek_problems(data, allow_unmatched_last=False)
        self.assertEqual(len(problems), 1)
        self.assertIn("before the previous", problems[0])

    def test_unmatched_entry_elsewhere_still_tested(self):
        data, _ = fsb_with_table([(44000, self.offsets[44] + 1), (10, self.offsets[88]), (90000, self.offsets[90] + 1)])
        problems = W.vorbis_seek_problems(data)
        self.assertEqual(len(problems), 2)  # entry 0 offset, entry 1 order; the last is exempt

    def test_docstring(self):
        self.assertIn("V19-2", W.vorbis_seek_problems.__doc__)


if __name__ == "__main__":
    unittest.main()
