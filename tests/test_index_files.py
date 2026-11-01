import os
import struct
import sys
import tempfile
import unittest
import zlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import coretoc_parser
import cpklib
import lua51
import prefetch_parser
import relua
from test_cpk import FILES, TempDirTest, build_legacy
from test_lua import abc, abx, chunk, function


def prefetch(endian, parents, deps):
    out = struct.pack(endian + "I", len(parents))
    out += b"".join(struct.pack(endian + "IIII", *p) for p in parents)
    out += struct.pack(endian + "I", len(deps))
    return out + b"".join(struct.pack(endian + "II", *d) for d in deps)


def segmented_toc(groups):
    """CoreTOC with the 844-byte header, from {group: [(handle, name), ...]}."""
    counts, offsets = [0] * 70, [0] * 70
    body = b""
    for group, entries in groups.items():
        counts[group], offsets[group] = len(entries), len(body)
        names = b""
        for handle, name in entries:
            body += struct.pack("<iii", group, handle, len(names))
            names += name.encode() + b"\0"
        body += names
    return struct.pack("<70i70i70iI", *counts, *offsets, *counts, 0) + body


class SectorZeroTests(TempDirTest):
    def real_first_chunk(self, version, endian):
        p = self.path("a.cpk")
        build_legacy(p, version=version, endian=endian)
        ar = cpklib.CPKArchive.open(p)
        order = ">" if endian == "big" else "<"
        plain = ar._decompress_sector(0)
        packed = zlib.compress(plain)
        packet = struct.pack(order + "HHH", len(plain) & 0xFFFF, len(plain) >> 16, len(packed))
        data = bytearray(ar._data)
        ar.close()
        first = ar._first_sec
        data[first:first + 0x4000] = (packet + packed).ljust(0x4000, b"\0")
        return cpklib.CPKArchive(bytes(data))

    def test_first_chunk_of_sector_zero_holds_data(self):
        for version, endian in ((6, "big"), (7, "little")):
            ar = self.real_first_chunk(version, endian)
            for name, data in FILES.items():
                self.assertEqual(ar.read(name), data)

    def test_empty_first_chunk_is_read_through(self):
        p = self.path("a.cpk")
        build_legacy(p)
        ar = cpklib.CPKArchive.open(p)
        plain = ar._decompress_sector(0)
        packed = zlib.compress(plain)
        empty = zlib.compress(b"")
        data = bytearray(ar._data)
        ar.close()
        first = ar._first_sec
        slot = struct.pack(">HHH", 0, 0, len(empty)) + empty
        slot += struct.pack(">HHH", len(plain) & 0xFFFF, len(plain) >> 16, len(packed)) + packed
        data[first:first + 0x4000] = slot.ljust(0x4000, b"\0")
        ar = cpklib.CPKArchive(bytes(data))
        self.assertEqual(ar.read("Dir\\A.txt"), FILES["Dir\\A.txt"])

    def test_verify_reports_a_short_sector(self):
        p = self.path("a.cpk")
        build_legacy(p)
        ar = cpklib.CPKArchive.open(p)
        self.assertEqual(ar.verify().problems, [])
        ar.header.decomp_fs += 10
        self.assertTrue(any("less than" in x for x in ar.verify().problems))
        ar.close()


class PrefetchTests(unittest.TestCase):
    def parse(self, data):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "Prefetch.dat")
            with open(path, "wb") as f:
                f.write(data)
            return prefetch_parser.parse_prefetch(path)

    def test_ranges_use_first_dep_not_a_running_sum(self):
        for endian in "><":
            graph, meta = self.parse(prefetch(
                endian, [(1, 10, 1, 2), (2, 20, 2, 0)], [(44, 1), (44, 2), (6, 3)]))
            self.assertEqual(meta, {"parent_count": 2, "total_deps": 3})
            self.assertEqual(graph[0]["deps"], [(6, 3)])
            self.assertEqual(graph[1]["deps"], [(44, 1), (44, 2)])

    def test_overlapping_ranges_are_refused(self):
        with self.assertRaises(ValueError):
            self.parse(prefetch(">", [(1, 10, 1, 0), (2, 20, 1, 0)], [(44, 1), (44, 2)]))

    def test_size_must_fit(self):
        with self.assertRaises(ValueError):
            self.parse(prefetch(">", [(1, 10, 1, 0)], [(44, 1)]) + b"\0")


class CoreTocTests(unittest.TestCase):
    def test_fixed_records(self):
        data = struct.pack(">III", 1, 29, 7) + b"Fire".ljust(128, b"\0")
        self.assertEqual(coretoc_parser.parse_coretoc(data), [(29, 7, "Fire")])

    def test_segmented_form(self):
        data = segmented_toc({29: [(7, "Fire"), (8, "Ice")], 69: [(9, "Engine")]})
        self.assertEqual(coretoc_parser.parse_coretoc(data), [
            (29, 7, "Fire"), (29, 8, "Ice"), (69, 9, "Engine")])
        self.assertEqual(coretoc_parser.get_group_info(69), ("DungeonFinder", "dfn"))

    def test_bad_segment_is_refused(self):
        data = bytearray(segmented_toc({29: [(7, "Fire")]}))
        struct.pack_into("<i", data, 29 * 4, 1000)
        with self.assertRaises(ValueError):
            coretoc_parser.parse_coretoc(bytes(data))


class ReluaLocalsTests(unittest.TestCase):
    def test_closure_captures_a_local_with_no_write(self):
        inner = function(">", [abc("RETURN", 0, 1)], upvalues=1)
        code = [
            abx("CLOSURE", 1, 0),
            abc("MOVE", 0, 0),
            abx("SETGLOBAL", 1, 0),
            abc("RETURN", 0, 1),
        ]
        parsed = lua51.parse_chunk(chunk(function(">", code, ("handler",), (inner,))))
        text = relua.render(relua.decompile(parsed))
        self.assertTrue(text.startswith("local v"), text)
        self.assertIn("function handler()", text)


if __name__ == "__main__":
    unittest.main()
