import struct
import unittest

import builders as B
from d3sno import FormatError
from d3sno import power as P
from d3sno.binary import ByteView, Pair


class Fixed(unittest.TestCase):
    def test_sizes(self):
        self.assertEqual(len(B.build_power("ps3")), 0x1A0)
        self.assertEqual(len(B.build_power("switch")), 0x320)

    def test_build_from_version(self):
        self.assertEqual(P.parse_power(B.build_power("ps3")).build, "ps3")
        self.assertEqual(P.parse_power(B.build_power("switch")).build, "switch")

    def test_fields(self):
        for build in ("ps3", "switch"):
            with self.subTest(build):
                p = P.parse_power(B.build_power(build, power_bits=0x401, formula_count=4), build)
                self.assertEqual(p.power_bits, 0x401)
                self.assertEqual(p.collision[0], P.CollisionRecord(1, 2, 3, 4))
                self.assertEqual(p.collision[1], P.CollisionRecord(5, 6, 7, 8))
                self.assertEqual(p.word_0f4, 0x99)
                self.assertEqual(p.formula_count, 4)
                self.assertTrue(p.uses_secondary_resource)
                self.assertEqual((p.quest_handle, p.padding_end), (-1, 0))
                self.assertEqual(len(p.buff_filters), 8 if build == "ps3" else 32)

    def test_power_bit_names(self):
        p = P.parse_power(B.build_power("ps3", power_bits=(1 << 10) | (1 << 16)))
        self.assertEqual(p.power_bit_names(), ["radial_targeting", "target_range_grows"])

    def test_truncated_fixed_part(self):
        with self.assertRaises(FormatError):
            P.parse_power(B.build_power("switch")[:0x300])

    def test_version_strict(self):
        data = B.build_power("ps3", version=1000)
        self.assertEqual(P.parse_power(data).build, "ps3")
        with self.assertRaises(FormatError):
            P.parse_power(data, strict=True)

    def test_buff_filters(self):
        ps3 = P.parse_power(B.build_power("ps3", buff=[[10, 11]]))
        self.assertEqual(ps3.buff_filters[0].handles, [10, 11])
        switch = P.parse_power(B.build_power("switch", buff=[[10, 11]]))
        self.assertEqual(switch.buff_filters[0].handles, [10, 11])
        self.assertEqual(switch.buff_filters[0].extra, [0, 0])

    def test_buff_filter_size(self):
        data = bytearray(B.build_power("switch", buff=[[10]]))
        struct.pack_into("<I", data, 0xFC, 12)
        data += bytes(4)
        with self.assertRaises(FormatError):
            P.parse_power(bytes(data))

    def test_buff_filter_out_of_bounds(self):
        data = bytearray(B.build_power("ps3"))
        struct.pack_into(">II", data, 0xF8, 0x1000, 4)
        with self.assertRaises(FormatError):
            P.parse_power(bytes(data))

    def test_script_slice(self):
        chunk = B.lua_chunk("switch")
        p = P.parse_power(B.build_power("switch", script=chunk))
        self.assertEqual(p.script, chunk)
        self.assertTrue(p.scripted)

    def test_scripted_with_empty_pair(self):
        p = P.parse_power(B.build_power("ps3", scripted=True))
        self.assertTrue(p.scripted)
        self.assertEqual(p.script, b"")

    def test_script_must_end_file(self):
        data = B.build_power("ps3", script=B.lua_chunk("ps3")) + b"\0"
        with self.assertRaises(FormatError):
            P.parse_power(data)

    def test_script_signature(self):
        with self.assertRaises(FormatError):
            P.parse_power(B.build_power("ps3", script=b"not lua at all"))


class TagMaps(unittest.TestCase):
    def map_of(self, records, build="switch"):
        return P.parse_power(B.build_power(build, general=records)).general

    def test_every_value_type(self):
        for build in ("ps3", "switch"):
            with self.subTest(build):
                m = self.map_of([
                    (0, 1, -5), (1, 2, 1.5), (2, 3, 77), (3, 4, -1),
                    (4, 5, ("a+b", b"\0\0\0\0")), (5, 6, 9), (6, 7, 8),
                ], build)
                self.assertEqual(m.get(1).value, -5)
                self.assertEqual(m.get(2).value, 1.5)
                self.assertEqual(m.get(4).value, -1)
                self.assertEqual(m.get(5).value, P.Formula("a+b", b"\0\0\0\0"))
                self.assertEqual(m.get(7).type, 6)

    def test_empty_formula(self):
        m = self.map_of([(4, 5, ("", b""))])
        self.assertEqual(m.get(5).value, P.Formula("", b""))

    def test_text_padding(self):
        for text in ("a", "ab", "abc", "abcd"):
            m = self.map_of([(4, 5, (text, b"\1\0\0\0")), (0, 6, 3)])
            self.assertEqual(m.get(5).value.text, text)
            self.assertEqual(m.get(6).value, 3)

    def test_records_are_unsorted(self):
        m = self.map_of([(0, 9, 1), (0, 2, 2)])
        self.assertEqual([r.tag for r in m.records], [9, 2])

    def raw_map(self, build, raw):
        order = ">" if build == "ps3" else "<"
        img = B.Image(0, order)
        img.b = bytearray(B.build_power(build))
        img.pair(0x30, raw)
        return img.done()

    def test_type_seven_rejected(self):
        raw = struct.pack("<III", 1, 7, 1) + bytes(4)
        with self.assertRaises(FormatError):
            P.parse_power(self.raw_map("switch", raw))

    def test_unknown_type_rejected(self):
        raw = struct.pack("<III", 1, 8, 1) + bytes(4)
        with self.assertRaises(FormatError):
            P.parse_power(self.raw_map("switch", raw))

    def test_record_overrun(self):
        raw = B.tag_map([(0, 1, 5)], "<")[:-2]
        with self.assertRaises(FormatError):
            P.parse_power(self.raw_map("switch", raw))

    def test_trailing_bytes_in_map(self):
        raw = B.tag_map([(0, 1, 5)], "<") + bytes(4)
        with self.assertRaises(FormatError):
            P.parse_power(self.raw_map("switch", raw))

    def test_count_mismatch(self):
        raw = bytearray(B.tag_map([(0, 1, 5)], "<"))
        struct.pack_into("<I", raw, 0, 2)
        with self.assertRaises(FormatError):
            P.parse_power(self.raw_map("switch", bytes(raw)))

    def test_duplicate_tag(self):
        raw = B.tag_map([(0, 1, 5), (1, 1, 2.0)], "<")
        with self.assertRaises(FormatError):
            P.parse_power(self.raw_map("switch", raw))

    def test_map_outside_file(self):
        data = bytearray(B.build_power("switch"))
        struct.pack_into("<II", data, 0x30, 0x10000, 8)
        with self.assertRaises(FormatError):
            P.parse_power(bytes(data))

    def test_map_too_short_for_count(self):
        with self.assertRaises(FormatError):
            P.parse_power(self.raw_map("switch", b"\0\0"))

    def formula_raw(self, change):
        raw = bytearray(B.tag_map([(4, 1, ("ab", b"\0\0\0\0"))], "<"))
        change(raw)
        return bytes(raw)

    def test_formula_zero_words(self):
        for word in (0, 1, 2, 3, 4, 6):
            raw = self.formula_raw(lambda r: struct.pack_into("<I", r, 12 + word * 4, 1))
            with self.assertRaises(FormatError):
                P.parse_power(self.raw_map("switch", raw))

    def test_formula_code_length_multiple_of_4(self):
        raw = self.formula_raw(lambda r: struct.pack_into("<I", r, 12 + 28, 3))
        with self.assertRaises(FormatError):
            P.parse_power(self.raw_map("switch", raw))

    def test_formula_text_nul(self):
        raw = self.formula_raw(lambda r: r.__setitem__(12 + 32 + 2, 0x41))
        with self.assertRaises(FormatError):
            P.parse_power(self.raw_map("switch", raw))

    def test_formula_padding_zero(self):
        raw = self.formula_raw(lambda r: r.__setitem__(12 + 32 + 3, 1))
        with self.assertRaises(FormatError):
            P.parse_power(self.raw_map("switch", raw))

    def test_formula_text_length_zero(self):
        raw = self.formula_raw(lambda r: struct.pack_into("<I", r, 12 + 20, 0))
        with self.assertRaises(FormatError):
            P.parse_power(self.raw_map("switch", raw))

    def test_formula_overrun(self):
        raw = self.formula_raw(lambda r: struct.pack_into("<I", r, 12 + 28, 400))
        with self.assertRaises(FormatError):
            P.parse_power(self.raw_map("switch", raw))


class Lenient(unittest.TestCase):
    def view(self, raw):
        return ByteView(bytes(0x10) + raw, "<"), Pair(0, len(raw))

    def test_type_7_is_4_bytes_in_a_marker_map(self):
        view, pair = self.view(B.tag_map([(7, 1, 5), (0, 2, 6)], "<"))
        read = P.read_tag_map(view, pair, lenient=True)
        self.assertEqual((read.get(1).value, read.get(2).value), (5, 6))
        self.assertIsNone(read.error)

    def test_type_7_is_still_refused_elsewhere(self):
        view, pair = self.view(B.tag_map([(7, 1, 5)], "<"))
        with self.assertRaises(FormatError):
            P.read_tag_map(view, pair)

    def test_a_bad_record_does_not_hide_the_ones_before_it(self):
        raw = B.tag_map([(0, 2, 6), (0, 3, 7)], "<")
        raw = struct.pack("<I", 3) + raw[4:] + struct.pack("<III", 99, 4, 0)
        view, pair = self.view(raw)
        read = P.read_tag_map(view, pair, lenient=True)
        self.assertEqual(len(read.records), 2)
        self.assertIn("99", read.error)


class Slots(unittest.TestCase):
    def test_tag_formula(self):
        self.assertEqual(P.formula_tag(0, "ps3"), 0x41100)
        self.assertEqual(P.formula_tag(23, "switch"), 0x41100 + 0x200 + 0x30)
        self.assertEqual(P.formula_tag(63, "switch"), 0x41100 + 0x600 + 0x30)

    def test_limits(self):
        self.assertEqual(P.formula_tag(39, "ps3"), 0x41100 + 0x300 + 0x90)
        with self.assertRaises(FormatError):
            P.formula_tag(40, "ps3")
        with self.assertRaises(FormatError):
            P.formula_tag(64, "switch")
        with self.assertRaises(FormatError):
            P.formula_tag(-1, "switch")

    def test_live_and_records_are_held_apart(self):
        tag = P.formula_tag(5, "switch")
        p = P.parse_power(B.build_power(
            "switch", general=[(4, tag, ("x", b""))], formula_count=2,
        ))
        self.assertEqual(p.formula_count, 2)
        self.assertEqual(len(p.formula_records), 1)
        self.assertFalse(p.is_live_slot(5))
        self.assertTrue(p.is_live_slot(1))
        self.assertIs(p.slot_record(5), p.general.records[0])

    def test_contact_map(self):
        p = P.parse_power(B.build_power("ps3", contact0=[(0, 1, 2)]))
        self.assertEqual(p.maps["Contact0"].get(1).value, 2)
        self.assertEqual(set(p.maps), {
            "Effect", "General", "PvpGeneral", "Contact0", "Contact1", "Contact2", "Contact3",
            "PvpContact0", "PvpContact1", "PvpContact2", "PvpContact3",
        })
