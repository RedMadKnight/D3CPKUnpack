import unittest

import builders as B
from d3sno import FormatError
from d3sno import sno as S
from d3sno.binary import BitReader, BitWriter, ByteView, Pair


class Header(unittest.TestCase):
    def test_both_orders(self):
        for order, build in ((">", "ps3"), ("<", "switch")):
            h = S.parse_header(B.sno_head(1442, order, handle=99))
            self.assertEqual((h.build, h.version, h.handle), (build, 1442, 99))
            self.assertEqual((h.revision, h.source_hash), (1, 0x1234))
            self.assertEqual((h.runtime_1, h.runtime_2), (0, 0))

    def test_zero_revision_and_hash_are_accepted(self):
        h = S.parse_header(B.sno_head(1, ">", revision=0, source=0))
        self.assertEqual((h.revision, h.source_hash), (0, 0))

    def test_stub_handle_zero(self):
        self.assertEqual(S.parse_header(B.sno_head(1, "<", handle=0)).handle, 0)

    def test_bad_signature(self):
        with self.assertRaises(FormatError):
            S.parse_header(b"\0" * 32)

    def test_wrong_build(self):
        with self.assertRaises(FormatError):
            S.parse_header(B.sno_head(1, ">"), "switch")

    def test_too_short(self):
        with self.assertRaises(FormatError):
            S.parse_header(b"\xDE\xAD\xBE\xEF")

    def test_unknown_build(self):
        with self.assertRaises(FormatError):
            S.parse_header(B.sno_head(1, ">"), "xbox")


class Raw(unittest.TestCase):
    def test_header_plus_bytes(self):
        data = B.sno_head(1, ">") + b"body"
        raw = S.parse_raw(data)
        self.assertEqual((raw.header.build, raw.raw), ("ps3", data))

    def test_bad_file(self):
        with self.assertRaises(FormatError):
            S.parse_raw(b"nope")


class Versions(unittest.TestCase):
    def test_lenient_then_strict(self):
        self.assertTrue(S.check_version("pow", 1442, "ps3"))
        self.assertFalse(S.check_version("pow", 1722, "ps3"))
        with self.assertRaises(FormatError):
            S.check_version("pow", 1722, "ps3", strict=True)

    def test_unknown_type_passes(self):
        self.assertTrue(S.check_version("zzz", 1, "ps3", strict=True))

    def test_build_from_version(self):
        self.assertEqual(S.resolve_build("pow", S.parse_header(B.sno_head(1442, ">"))), "ps3")
        self.assertEqual(S.resolve_build("app", S.parse_header(B.sno_head(306, "<"))), "switch")

    def test_version_contradicts_order(self):
        with self.assertRaises(FormatError):
            S.resolve_build("pow", S.parse_header(B.sno_head(1722, ">")))

    def test_same_version_on_both_builds_uses_order(self):
        self.assertEqual(S.resolve_build("sbk", S.parse_header(B.sno_head(46, "<"))), "switch")

    def test_table(self):
        self.assertEqual(S.VERSIONS["scn"], (163, 179))
        self.assertEqual(len(S.VERSIONS), 9)


class Groups(unittest.TestCase):
    def test_table(self):
        self.assertEqual(len(S.GROUPS), 66)
        self.assertEqual(S.group_by_id(29).extension, "pow")
        self.assertEqual(S.group_by_extension(".APP").id, 9)
        self.assertEqual(S.group_by_id(69).name, "DungeonFinder")

    def test_unused_ids(self):
        for unused in (0, 16, 30, 50):
            self.assertIsNone(S.group_by_id(unused))

    def test_compact(self):
        self.assertEqual(S.compact_handle(29, 0x123), 0x1231D)
        self.assertEqual(S.split_compact(0x1231D), (29, 0x123))
        self.assertEqual(S.compact_handle(-1, -1), -1)


class Pairs(unittest.TestCase):
    def test_data_at_offset_plus_0x10(self):
        data = bytes(0x10) + b"abcdef"
        self.assertEqual(S.slice_pair(data, Pair(2, 3)), b"cde")

    def test_empty(self):
        self.assertTrue(Pair(0, 0).is_empty)
        self.assertEqual(S.slice_pair(b"", Pair(0, 0)), b"")

    def test_out_of_bounds(self):
        with self.assertRaises(FormatError):
            S.slice_pair(bytes(0x20), Pair(0, 0x11))

    def test_error_carries_pair_position(self):
        with self.assertRaises(FormatError) as caught:
            S.slice_pair(bytes(0x20), Pair(0x100, 1), pos=0x18)
        self.assertEqual(caught.exception.offset, 0x18)

    def test_read_pair_order(self):
        self.assertEqual(S.read_pair(b"\0\0\0\1\0\0\0\2", 0, "ps3"), Pair(1, 2))
        self.assertEqual(S.read_pair(b"\1\0\0\0\2\0\0\0", 0, "switch"), Pair(1, 2))

    def test_read_pair_past_end(self):
        with self.assertRaises(FormatError):
            S.read_pair(b"\0" * 7, 0, "ps3")


class Hashes(unittest.TestCase):
    def test_values(self):
        self.assertEqual(S.hash33(""), 0)
        self.assertEqual(S.hash33("a"), 97)
        self.assertEqual(S.hash33("ab"), 97 * 33 + 98)

    def test_case(self):
        self.assertNotEqual(S.hash33("Ab"), S.hash33("ab"))
        self.assertEqual(S.hash33_ci("AB"), S.hash33("ab"))

    def test_signed_bytes(self):
        self.assertEqual(S.hash33(b"\xFF"), 0xFFFFFFFF)
        self.assertEqual(S.hash33(b"\x80"), 0xFFFFFF80)

    def test_wraparound(self):
        self.assertLess(S.hash33("a very long string for wrapping around"), 1 << 32)


class Bits(unittest.TestCase):
    def test_msb_first(self):
        reader = BitReader(b"\xA5\xC0", 0)
        self.assertEqual(reader.read(4), 0xA)
        self.assertEqual(reader.read(6), 0b010111)
        self.assertEqual(reader.align(), 2)

    def test_past_end(self):
        with self.assertRaises(FormatError):
            BitReader(b"\xFF", 0).read(9)

    def test_writer_inverse(self):
        writer = BitWriter()
        for value, width in ((5, 3), (0xABCDEF0123, 40), (1, 1)):
            writer.write(value, width)
        reader = BitReader(writer.finish(), 0)
        self.assertEqual([reader.read(w) for w in (3, 40, 1)], [5, 0xABCDEF0123, 1])

    def test_writer_overflow(self):
        with self.assertRaises(FormatError):
            BitWriter().write(8, 3)


class View(unittest.TestCase):
    def test_every_accessor_is_bounded(self):
        view = ByteView(b"\0\0", ">")
        for call in (lambda: view.u32(0), lambda: view.u16(1), lambda: view.u8(2),
                     lambda: view.f32(0), lambda: view.take(1, 2), lambda: view.u8(-1)):
            with self.assertRaises(FormatError):
                call()
