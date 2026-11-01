import struct
import unittest

import builders as B
from d3sno import FormatError
from d3sno import text as T
from d3sno.sno import hash33

BUILDS = ("ps3", "switch")


def strings(rows, build="switch"):
    return T.parse_string_list(B.build_stl(build, [(hash33(k), v.encode()) for k, v in rows]))


class StringLists(unittest.TestCase):
    def test_both_builds(self):
        for build in BUILDS:
            data = B.build_stl(build, [(hash33("Foo"), "Zażółć".encode())])
            read = T.parse_string_list(data)
            self.assertEqual(read.build, build)
            self.assertEqual(T.lookup(read, "Foo"), "Zażółć")

    def test_order_is_kept_and_not_sorted(self):
        read = strings([("B", "2"), ("A", "1")])
        self.assertEqual([e.text for e in read.entries], ["2", "1"])

    def test_empty_text(self):
        self.assertEqual(T.lookup(strings([("Foo", "")]), "Foo"), "")

    def test_shared_bytes(self):
        data = B.build_stl("switch", [(1, b"same"), (2, b"same")])
        self.assertEqual(data.count(b"same"), 1)
        read = T.parse_string_list(data)
        self.assertEqual((read.by_hash[1], read.by_hash[2]), ("same", "same"))

    def test_tokens_stay_as_text(self):
        text = "{c_green}x{/c} {s1} {VALUE} {icon:bullet}"
        self.assertEqual(T.lookup(strings([("Foo", text)]), "Foo"), text)

    def test_label_is_case_sensitive(self):
        self.assertIsNone(T.lookup(strings([("Foo", "x")]), "foo"))

    def test_duplicate_label(self):
        data = B.build_stl("switch", [(1, b"a"), (1, b"b")])
        read = T.parse_string_list(data)
        self.assertTrue(read.warnings)
        self.assertEqual(read.by_hash[1], "a")
        with self.assertRaises(FormatError):
            T.parse_string_list(data, strict=True)

    def test_array_size(self):
        with self.assertRaises(FormatError):
            T.parse_string_list(B.build_stl("switch", [(1, b"a")], array_extra=b"\0"))

    def test_array_out_of_bounds(self):
        data = bytearray(B.build_stl("switch", []))
        struct.pack_into("<II", data, 0x28, 0xFFFF, 24)
        with self.assertRaises(FormatError):
            T.parse_string_list(bytes(data))

    def corrupt_text(self, change):
        data = bytearray(B.build_stl("switch", [(1, b"abc")]))
        table = struct.unpack_from("<I", data, 0x28)[0] + 0x10
        change(data, table)
        return bytes(data)

    def test_text_pair_out_of_bounds(self):
        data = self.corrupt_text(lambda d, t: struct.pack_into("<I", d, t + 8, 0xFFFF))
        with self.assertRaises(FormatError):
            T.parse_string_list(data)

    def test_text_needs_its_nul(self):
        data = self.corrupt_text(lambda d, t: struct.pack_into("<I", d, t + 12, 3))
        with self.assertRaises(FormatError):
            T.parse_string_list(data)

    def test_text_size_zero(self):
        data = self.corrupt_text(lambda d, t: struct.pack_into("<I", d, t + 12, 0))
        with self.assertRaises(FormatError):
            T.parse_string_list(data)

    def test_text_not_utf8(self):
        data = B.build_stl("switch", [(1, b"\xFF\xFE")])
        with self.assertRaises(FormatError):
            T.parse_string_list(data)


class Lookups(unittest.TestCase):
    def test_power(self):
        read = strings([("Fire_name", "Fire"), ("Fire_desc", "Hot"), ("Fire_simple_desc", "H")])
        self.assertEqual(T.lookup_power(read, "Fire"), "Fire")
        self.assertEqual(T.lookup_power(read, "Fire", "simple_desc"), "H")

    def test_labels_when_the_player_speaks(self):
        self.assertEqual(
            T.line_labels(1, "", "", "Barb", "F", True),
            ["Line1BarbF", "Line1", "Line1Barb", "Line1F", "Line1"],
        )
        self.assertEqual(T.line_labels(1, "Wiz", "M", "Barb", "F", True)[:2], ["Line1WizM"] * 2)

    def test_labels_when_someone_else_speaks(self):
        self.assertEqual(T.line_labels(1, "Wiz", "", "Barb", "F")[0], "Line1WizF")
        self.assertEqual(T.line_labels(1, "Wiz", "", "Barb", "M")[0], "Line1Wiz")
        self.assertEqual(T.line_labels(1, "Wiz", "M", "Barb", "F")[0], "Line1WizM")

    def test_gender_numbers(self):
        self.assertEqual(T.line_labels(1, "", 0)[1], "Line1M")
        self.assertEqual(T.line_labels(1, "", 1)[1], "Line1F")
        self.assertEqual(T.line_labels(1, "", 2)[1], "Line1")

    def test_plain_wins_over_a_class_only_form(self):
        read = strings([("Line1Barb", "class"), ("Line1", "plain")])
        self.assertEqual(T.lookup_line(read, 1, "", "", "Barb", "F", True), "plain")

    def test_later_labels_are_tried(self):
        read = strings([("Line1Barb", "class")])
        self.assertEqual(T.lookup_line(read, 1, "", "", "Barb", "F", True), "class")
        read = strings([("Line1F", "gender")])
        self.assertEqual(T.lookup_line(read, 1, "", "", "Barb", "F", True), "gender")

    def test_override_list_comes_first(self):
        read, override = strings([("Foo", "stored")]), strings([("Foo", "override")])
        self.assertEqual(T.lookup(read, "Foo", override), "override")
        self.assertEqual(T.lookup(read, "Foo"), "stored")
        self.assertEqual(T.lookup(strings([("Bar", "x")]), "Bar", override), "x")

    def test_empty_text_is_not_missing(self):
        self.assertEqual(T.lookup(strings([("Foo", "")]), "Foo", strings([])), "")

    def test_line_miss_is_not_an_error(self):
        self.assertIsNone(T.lookup_line(strings([]), 1))

    def test_locale(self):
        self.assertEqual(T.locale_of("StringList\\Foo.stl|frFR"), "frFR")
        self.assertIsNone(T.locale_of("StringList\\Foo.stl"))
        self.assertEqual(T.locale_of_archive("esMX_Common.cpk"), "esMX")
        self.assertIsNone(T.locale_of_archive("Common.cpk"))

    def test_join_by_hash(self):
        joined = T.join_locales({
            "enUS": strings([("Foo", "Hello")]), "frFR": strings([("Foo", "Salut")]),
        })
        self.assertEqual(joined[hash33("Foo")], {"enUS": "Hello", "frFR": "Salut"})


class Declension(unittest.TestCase):
    def test_read(self):
        data = "﻿miecz\t1\tmiecza\r\nmiecz\t2\tmieczowi\r\n".encode()
        forms = T.parse_declension(data)
        self.assertEqual(forms[("miecz", 2)], "mieczowi")

    def test_missing_column(self):
        with self.assertRaises(FormatError):
            T.parse_declension(b"a\t1\r\n")

    def test_bad_case(self):
        with self.assertRaises(FormatError):
            T.parse_declension(b"a\tx\tb\r\n")

    def test_not_utf8(self):
        with self.assertRaises(FormatError):
            T.parse_declension(b"\xFF\tb\tc")
