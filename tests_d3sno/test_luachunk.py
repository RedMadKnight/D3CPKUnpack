import struct
import unittest

import builders as B
from d3sno import FormatError
from d3sno import luachunk as L


class Header(unittest.TestCase):
    def test_detect(self):
        self.assertEqual(L.check_header(B.lua_chunk("ps3")), "ps3")
        self.assertEqual(L.check_header(B.lua_chunk("switch")), "switch")

    def test_values_must_match_exactly(self):
        for position in range(6, 12):
            data = bytearray(B.lua_chunk("switch"))
            data[position] ^= 0x10
            with self.assertRaises(FormatError) as caught:
                L.check_header(bytes(data), "switch")
            self.assertEqual(caught.exception.offset, position)

    def test_wrong_build(self):
        with self.assertRaises(FormatError):
            L.check_header(B.lua_chunk("ps3"), "switch")

    def test_signature(self):
        data = bytearray(B.lua_chunk("ps3"))
        data[4] = 0x52
        with self.assertRaises(FormatError):
            L.check_header(bytes(data))

    def test_format(self):
        data = bytearray(B.lua_chunk("ps3"))
        data[5] = 1
        with self.assertRaises(FormatError):
            L.check_header(bytes(data))

    def test_short(self):
        with self.assertRaises(FormatError):
            L.check_header(b"\x1BLua")


class Parse(unittest.TestCase):
    def full(self, build):
        inner = {"code": (L.OPCODES.index("MOVE"), B.RETURN_WORD), "lines": (3, 9)}
        return B.lua_chunk(
            build, code=(L.OPCODES.index("CLOSURE") | (1 << 6), B.RETURN_WORD),
            constants=(None, True, False, 2.5, b"hello"), protos=(inner, inner), lines=(0, 12),
        )

    def test_function_tree(self):
        for build in ("ps3", "switch"):
            with self.subTest(build):
                chunk = L.parse_chunk(self.full(build))
                main = chunk.main
                self.assertEqual(chunk.build, build)
                self.assertEqual(main.constants, [None, True, False, 2.5, b"hello"])
                self.assertEqual(len(main.protos), 2)
                self.assertEqual(main.protos[1].line_span, (3, 9))
                self.assertEqual(main.line_span, (0, 12))
                self.assertEqual(L.opcode_name(main.code[0]), "CLOSURE")
                self.assertEqual(chunk.trailing_bytes, 0)
                self.assertTrue(chunk.stripped)
                self.assertEqual(len(list(main.walk())), 3)

    def test_empty_script(self):
        self.assertTrue(L.parse_chunk(B.lua_chunk("ps3")).is_empty_script)
        self.assertFalse(L.parse_chunk(self.full("ps3")).is_empty_script)

    def test_not_stripped_is_kept(self):
        writer = B.LuaWriter("switch")
        writer.header()
        writer.string(b"@source.lua")
        for value in (1, 2):
            writer.int(value)
        writer.out += bytes((0, 0, 2, 2))
        writer.int(1)
        writer.out += struct.pack("<I", B.RETURN_WORD)
        writer.int(0)
        writer.int(0)
        writer.int(1)
        writer.int(1)
        writer.int(1)
        writer.string(b"x")
        writer.int(0)
        writer.int(1)
        writer.int(1)
        writer.string(b"up")
        chunk = L.parse_chunk(bytes(writer.out))
        self.assertFalse(chunk.stripped)
        self.assertEqual(chunk.main.source, b"@source.lua")
        self.assertEqual(chunk.main.locals[0].name, b"x")
        self.assertEqual(chunk.main.upvalue_names, [b"up"])

    def test_every_truncation_raises(self):
        data = self.full("switch")
        for cut in range(len(data)):
            with self.assertRaises(FormatError):
                L.parse_chunk(data[:cut])

    def test_trailing_bytes_are_counted(self):
        self.assertEqual(L.parse_chunk(B.lua_chunk("ps3") + b"\0\0").trailing_bytes, 2)

    def test_bad_opcode(self):
        with self.assertRaises(FormatError):
            L.parse_chunk(B.lua_chunk("ps3", code=(38,)))

    def test_bad_constant_type(self):
        data = bytearray(B.lua_chunk("ps3", constants=(None,)))
        position = data.rindex(b"\0\0\0\1\0") + 4
        data[position] = 9
        with self.assertRaises(FormatError):
            L.parse_chunk(bytes(data))

    def test_negative_count(self):
        data = bytearray(B.lua_chunk("ps3"))
        struct.pack_into(">i", data, 12 + 4 + 8 + 4, -1)
        with self.assertRaises(FormatError):
            L.parse_chunk(bytes(data))

    def test_opcode_count(self):
        self.assertEqual(len(L.OPCODES), 38)


class Slices(unittest.TestCase):
    def file_with_pair(self, build, position, chunk):
        order = ">" if build == "ps3" else "<"
        img = B.Image(0x400, order)
        img.raw(0, B.sno_head(1, order))
        img.pair(position, chunk)
        return img.done()

    def test_positions(self):
        cases = [
            ("scr", "ps3", 0x20), ("scr", "switch", 0x20), ("con", "ps3", 0x48),
            ("con", "switch", 0x48), ("cnv", "ps3", 0x1E8), ("cnv", "switch", 0xE8),
            ("pow", "ps3", 0x190), ("pow", "switch", 0x310),
        ]
        for extension, build, position in cases:
            with self.subTest((extension, build)):
                chunk = B.lua_chunk(build)
                data = self.file_with_pair(build, position, chunk)
                self.assertEqual(L.script_slice(data, extension, build), chunk)
                self.assertEqual(L.script_slice(data, "." + extension), chunk)

    def test_empty_pair(self):
        order = "<"
        img = B.Image(0x400, order)
        img.raw(0, B.sno_head(1, order))
        self.assertEqual(L.script_slice(img.done(), "scr", "switch"), b"")

    def test_signature_checked(self):
        data = self.file_with_pair("switch", 0x20, b"garbage")
        with self.assertRaises(FormatError):
            L.script_slice(data, "scr", "switch")

    def test_out_of_bounds(self):
        data = bytearray(self.file_with_pair("switch", 0x20, b"garbage"))
        struct.pack_into("<I", data, 0x24, 0x10000)
        with self.assertRaises(FormatError):
            L.script_slice(bytes(data), "scr", "switch")

    def test_unknown_type(self):
        with self.assertRaises(FormatError):
            L.script_slice(b"", "txt")
