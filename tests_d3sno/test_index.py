import struct
import unittest

from d3sno import FormatError
from d3sno import index as I
from d3sno.sno import hash33


def toc_ps3(entries):
    out = struct.pack(">I", len(entries))
    for group, handle, name in entries:
        out += struct.pack(">II", group, handle) + name.encode("latin-1").ljust(128, b"\0")
    return out


def toc_switch(groups, unknown=0x1234, count2_delta=0):
    """``groups`` maps a group id to a list of (handle, name)."""
    counts = [0] * 70
    offsets = [0] * 70
    body = bytearray()
    for g, items in groups.items():
        counts[g] = len(items)
        offsets[g] = len(body)
        names = bytearray()
        records = bytearray()
        for handle, name in items:
            records += struct.pack("<iii", g, handle, len(names))
            names += name.encode("latin-1") + b"\0"
        body += records + names
    counts2 = list(counts)
    counts2[3] += count2_delta
    return struct.pack("<70i70i70iI", *counts, *offsets, *counts2, unknown) + bytes(body)


class CoreToc(unittest.TestCase):
    def test_ps3(self):
        toc = I.parse_core_toc(toc_ps3([(1, 2, "Foo"), (29, -1 & 0xFFFFFFFF, "Bar")]))
        self.assertEqual(toc.build, "ps3")
        self.assertEqual(toc.entries[0], I.TocEntry(1, 2, "Foo"))
        self.assertEqual(toc.entries[1].handle, 0xFFFFFFFF)

    def test_ps3_size_exact(self):
        with self.assertRaises(FormatError):
            I.parse_core_toc(toc_ps3([(1, 2, "Foo")]) + b"\0", "ps3")

    def test_switch(self):
        toc = I.parse_core_toc(toc_switch({1: [(5, "One"), (6, "Two")], 29: [(7, "Pow")]}))
        self.assertEqual(toc.build, "switch")
        self.assertEqual(toc.unknown, 0x1234)
        self.assertEqual(
            [(e.group, e.handle, e.name) for e in toc.entries],
            [(1, 5, "One"), (1, 6, "Two"), (29, 7, "Pow")],
        )

    def test_switch_header_size(self):
        self.assertEqual(I.SWITCH_TOC_HEADER, 844)

    def test_switch_count2(self):
        with self.assertRaises(FormatError):
            I.parse_core_toc(toc_switch({1: [(5, "One")]}, count2_delta=1))

    def test_switch_negative_count(self):
        data = bytearray(toc_switch({}))
        struct.pack_into("<i", data, 0, -1)
        struct.pack_into("<i", data, 560, -1)
        with self.assertRaises(FormatError):
            I.parse_core_toc(bytes(data), "switch")

    def test_switch_records_past_end(self):
        data = toc_switch({1: [(5, "One")]})
        with self.assertRaises(FormatError):
            I.parse_core_toc(data[:850], "switch")

    def test_switch_name_past_end(self):
        data = bytearray(toc_switch({1: [(5, "One")]}))
        struct.pack_into("<i", data, 844 + 8, 9999)
        with self.assertRaises(FormatError):
            I.parse_core_toc(bytes(data), "switch")

    def test_switch_unterminated_name(self):
        data = toc_switch({1: [(5, "One")]})[:-1]
        with self.assertRaises(FormatError):
            I.parse_core_toc(data, "switch")

    def test_too_short(self):
        with self.assertRaises(FormatError):
            I.parse_core_toc(b"\0\0", "switch")


def prefetch(order, parents, deps):
    out = struct.pack(order + "I", len(parents))
    for parent in parents:
        out += struct.pack(order + "4I", *parent)
    out += struct.pack(order + "I", len(deps))
    for dep in deps:
        out += struct.pack(order + "II", *dep)
    return out


class Prefetch(unittest.TestCase):
    PARENTS = [(1, 0x10001, 1, 2), (2, 0x20002, 2, 0)]  # not a running sum
    DEPS = [(5, 50), (6, 60), (7, 70)]

    def test_both_builds(self):
        for build, order in (("ps3", ">"), ("switch", "<")):
            parsed = I.parse_prefetch(prefetch(order, self.PARENTS, self.DEPS), build)
            self.assertEqual(parsed.dependencies(parsed.parents[0]), [(7, 70)])
            self.assertEqual(parsed.dependencies(parsed.parents[1]), [(5, 50), (6, 60)])

    def test_detection(self):
        self.assertEqual(I.parse_prefetch(prefetch(">", self.PARENTS, self.DEPS)).build, "ps3")
        self.assertEqual(I.parse_prefetch(prefetch("<", self.PARENTS, self.DEPS)).build, "switch")

    def test_sum_mismatch(self):
        with self.assertRaises(FormatError):
            I.parse_prefetch(prefetch("<", [(1, 2, 2, 0)], self.DEPS), "switch")

    def test_gap(self):
        parents = [(1, 1, 1, 0), (2, 2, 2, 2)]
        with self.assertRaises(FormatError):
            I.parse_prefetch(prefetch("<", parents, self.DEPS + [(8, 80)])[:-8], "switch")

    def test_overlap(self):
        parents = [(1, 1, 2, 0), (2, 2, 1, 1)]
        with self.assertRaises(FormatError):
            I.parse_prefetch(prefetch("<", parents, self.DEPS), "switch")

    def test_truncated(self):
        with self.assertRaises(FormatError):
            I.parse_prefetch(prefetch("<", self.PARENTS, self.DEPS)[:-1], "switch")

    def test_sorted_by_low_handle_bits(self):
        good = I.parse_prefetch(prefetch("<", [(1, 0xFFFF0001, 0, 0), (2, 2, 0, 0)], []), "switch")
        self.assertTrue(good.parents_sorted)
        bad = I.parse_prefetch(prefetch("<", [(1, 3, 0, 0), (2, 0xFFFF0002, 0, 0)], []), "switch")
        self.assertFalse(bad.parents_sorted)

    def test_unresolved_count(self):
        parsed = I.parse_prefetch(prefetch("<", self.PARENTS, self.DEPS), "switch")
        toc = I.CoreToc("switch", [I.TocEntry(5, 50, "A")])
        self.assertEqual(parsed.unresolved(toc), 2)


class Descriptors(unittest.TestCase):
    def build(self, count=2):
        out = struct.pack("<I", 0xAABBCCDD)
        for i in range(count):
            out += f"Type{i}".encode().ljust(64, b"\0")
            out += struct.pack("<I", 100 + i) + struct.pack("<256I", *range(256))
        return out

    def test_read(self):
        entries = I.parse_descriptor_sizes(self.build())
        self.assertEqual(len(self.build(1)), 4 + 1092)
        self.assertEqual((entries[1].name, entries[1].size), ("Type1", 101))
        self.assertEqual(entries[0].words[255], 255)

    def test_smallest_file_rule(self):
        entry = I.parse_descriptor_sizes(self.build())[0]
        self.assertTrue(entry.fits_smallest_file(116))
        self.assertFalse(entry.fits_smallest_file(115))

    def test_bad_magic(self):
        with self.assertRaises(FormatError):
            I.parse_descriptor_sizes(b"\0" * 8)

    def test_bad_size(self):
        with self.assertRaises(FormatError):
            I.parse_descriptor_sizes(self.build()[:-1])


class Manifest(unittest.TestCase):
    def test_both_orders(self):
        for order in (">", "<"):
            data = struct.pack(order + "I4I", 0xAABB0004, 1, 2, 3, 4)
            self.assertEqual(I.parse_manifest(data).entries, [(1, 2), (3, 4)])
            self.assertEqual(I.dat_kind(data), "manifest")

    def test_bad_magic(self):
        with self.assertRaises(FormatError):
            I.parse_manifest(b"\1\2\3\4")

    def test_bad_size(self):
        with self.assertRaises(FormatError):
            I.parse_manifest(struct.pack("<II", 0xAABB0004, 1))


class Sprites(unittest.TestCase):
    def build(self, rows):
        return struct.pack("<I", 0xBBCC0005) + b"".join(struct.pack("<III", *r) for r in rows)

    def test_read_and_find_by_hash(self):
        rows = [(hash33("Foo"), 1, 0), (hash33("Bar"), 1, 1), (hash33("Baz"), 2, 0)]
        table = I.parse_sprite_table(self.build(rows))
        self.assertEqual(table.problems(), [])
        self.assertEqual(table.find("Bar").frame, 1)
        self.assertIsNone(table.find("bar"))
        self.assertEqual(I.dat_kind(self.build(rows)), "sprite")

    def test_handles_unsorted(self):
        table = I.parse_sprite_table(self.build([(0, 2, 0), (0, 1, 0)]))
        self.assertIn("texture handles are not sorted", table.problems())

    def test_frames_not_from_zero(self):
        table = I.parse_sprite_table(self.build([(0, 1, 1)]))
        self.assertTrue(table.problems())

    def test_bad_magic(self):
        with self.assertRaises(FormatError):
            I.parse_sprite_table(b"\0" * 16)

    def test_bad_size(self):
        with self.assertRaises(FormatError):
            I.parse_sprite_table(self.build([(0, 1, 0)])[:-2])


class Subtitles(unittest.TestCase):
    def test_blocks(self):
        text = (
            "﻿00:00:01:00 - 00:00:02:50\r\nHello\r\nWorld\r\n\r\n"
            "No time code\r\n\r\n00:00:03:00 - 00:00:04:00\r\nZażółć\r\n"
        )
        blocks = I.parse_subtitles(text.encode("utf-8"))
        self.assertEqual(blocks[0].lines, ["Hello", "World"])
        self.assertEqual(blocks[0].start, "00:00:01:00")
        self.assertEqual((blocks[1].start, blocks[1].lines), (None, ["No time code"]))
        self.assertEqual(blocks[2].lines, ["Zażółć"])

    def test_not_utf8(self):
        with self.assertRaises(FormatError):
            I.parse_subtitles(b"\xFF\xFE\xFD")
