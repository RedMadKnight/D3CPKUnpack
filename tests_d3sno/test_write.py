import random
import struct
import unittest

import builders as B
from d3sno import FormatError
from d3sno import archive as A
from d3sno import power as P
from d3sno import scripts as S
from d3sno import text as X
from d3sno import texture as T
from d3sno import write as W
from d3sno.sno import hash33

BUILDS = ("ps3", "switch")


class StringLists(unittest.TestCase):
    ROWS = [(hash33("B"), b"second"), (hash33("A"), b"x"), (3, b"second"), (4, b""), (5, b"12345678")]

    def test_round_trip(self):
        for build in BUILDS:
            data = B.build_stl(build, self.ROWS, revision=9, source=0xBEEF)
            self.assertEqual(W.write(X.parse_string_list(data)), data)

    def test_empty_list(self):
        data = B.build_stl("switch", [])
        self.assertEqual(W.write(X.parse_string_list(data)), data)

    def test_layout(self):
        out = W.write(X.parse_string_list(B.build_stl("ps3", self.ROWS)))
        entry = struct.unpack_from(">I", out, 0x38 + 8)[0] + 0x10
        self.assertEqual(entry, 0x38 + 24 * 5)
        self.assertEqual(entry % 8, 0)
        self.assertEqual(out.count(b"second"), 1)

    def test_revision_and_hash_are_kept(self):
        data = B.build_stl("ps3", self.ROWS, revision=9, source=0xBEEF)
        strings = X.parse_string_list(data)
        W.set_text(strings, "A", "new")
        read = X.parse_string_list(W.write(strings))
        self.assertEqual((read.header.revision, read.header.source_hash), (9, 0xBEEF))

    def test_set_text(self):
        strings = X.parse_string_list(B.build_stl("switch", self.ROWS))
        W.set_text(strings, "A", "Zażółć")
        read = X.parse_string_list(W.write(strings))
        self.assertEqual(X.lookup(read, "A"), "Zażółć")
        self.assertEqual(X.lookup(read, "B"), "second")
        self.assertEqual([e.label_hash for e in read.entries], [e[0] for e in self.ROWS])

    def test_set_text_unknown_label(self):
        strings = X.parse_string_list(B.build_stl("switch", self.ROWS))
        with self.assertRaises(FormatError):
            W.set_text(strings, "Nope", "x")

    def test_add_goes_last(self):
        strings = X.parse_string_list(B.build_stl("switch", self.ROWS))
        W.add_entry(strings, "New", "added")
        read = X.parse_string_list(W.write(strings))
        self.assertEqual(read.entries[-1].label_hash, hash33("New"))
        self.assertEqual(X.lookup(read, "New"), "added")

    def test_add_existing_label(self):
        strings = X.parse_string_list(B.build_stl("switch", self.ROWS))
        with self.assertRaises(FormatError):
            W.add_entry(strings, "A", "x")

    def test_remove(self):
        strings = X.parse_string_list(B.build_stl("switch", self.ROWS))
        W.remove_entry(strings, "A")
        read = X.parse_string_list(W.write(strings))
        self.assertIsNone(X.lookup(read, "A"))
        self.assertEqual(len(read.entries), 4)
        with self.assertRaises(FormatError):
            W.remove_entry(strings, "A")

    def test_remove_all(self):
        strings = X.parse_string_list(B.build_stl("switch", [(1, b"a")]))
        W.remove_entry(strings, 1)
        self.assertEqual(X.parse_string_list(W.write(strings)).entries, [])


def sample_power(build, **kw):
    order = ">" if build == "ps3" else "<"
    return B.build_power(
        build,
        general=[(0, 1, 5), (1, 2, 1.5), (4, 3, ("ab", B.Code(order).f(2).end()))],
        contact0=[(0, 9, 8)],
        buff=[[1, 2], [3]],
        script=B.lua_chunk(build),
        **kw,
    )


class Powers(unittest.TestCase):
    def test_round_trip_every_map_with_its_own_bytes(self):
        for build in BUILDS:
            data = sample_power(build, all_maps=True, revision=4, source=0xCAFE)
            self.assertEqual(W.write(P.parse_power(data)), data)

    def test_empty_map_keeps_its_own_4_bytes(self):
        data = sample_power("ps3", all_maps=True)
        parsed = P.parse_power(data)
        pairs = {parsed.maps[name].pair.offset for name in ("Effect", "PvpGeneral", "Contact1")}
        self.assertEqual(len(pairs), 3)
        self.assertEqual(parsed.maps["Effect"].pair.size, 4)

    def test_round_trip_shared_contact_block(self):
        data = sample_power("switch", all_maps=True, shared_contacts=True)
        parsed = P.parse_power(data)
        self.assertEqual(parsed.maps["Contact2"].shares_with, "Contact0")
        self.assertIsNone(parsed.maps["Contact0"].shares_with)
        self.assertEqual(W.write(parsed), data)

    def test_round_trip_maps_with_no_bytes(self):
        data = sample_power("switch")
        self.assertEqual(W.write(P.parse_power(data)), data)

    def test_no_script(self):
        data = B.build_power("ps3", general=[(0, 1, 5)], scripted=True)
        self.assertEqual(W.write(P.parse_power(data)), data)

    def test_alignment(self):
        out = W.write(P.parse_power(sample_power("ps3", all_maps=True)))
        parsed = P.parse_power(out)
        self.assertEqual(parsed.buff_filters[1].pair.file_offset % 8, 0)
        self.assertEqual(parsed.script_pair.file_offset % 8, 0)
        self.assertEqual(parsed.maps["General"].pair.file_offset % 4, 0)

    def test_set_tag(self):
        for build in BUILDS:
            parsed = P.parse_power(sample_power(build))
            W.set_tag(parsed, "General", 1, 77)
            W.set_tag(parsed, "General", 2, 2.5)
            read = P.parse_power(W.write(parsed))
            self.assertEqual(read.general.get(1).value, 77)
            self.assertEqual(read.general.get(2).value, 2.5)
            self.assertEqual(read.general.get(3).value.text, "ab")

    def test_set_tag_refuses_a_formula(self):
        parsed = P.parse_power(sample_power("switch"))
        with self.assertRaises(FormatError):
            W.set_tag(parsed, "General", 3, 1)

    def test_set_tag_unknown(self):
        parsed = P.parse_power(sample_power("switch"))
        with self.assertRaises(FormatError):
            W.set_tag(parsed, "General", 99, 1)
        with self.assertRaises(FormatError):
            W.set_tag(parsed, "Nope", 1, 1)

    def test_set_tag_bad_value(self):
        parsed = P.parse_power(sample_power("switch"))
        with self.assertRaises(FormatError):
            W.set_tag(parsed, "General", 1, "text")

    def test_set_tag_in_a_shared_block(self):
        data = sample_power("switch", all_maps=True, shared_contacts=True)
        parsed = P.parse_power(data)
        W.set_tag(parsed, "Contact2", 9, 11)
        read = P.parse_power(W.write(parsed))
        self.assertEqual(read.maps["Contact0"].get(9).value, 11)
        self.assertEqual(read.maps["Contact3"].get(9).value, 11)
        self.assertEqual(read.maps["Contact3"].pair, read.maps["Contact0"].pair)

    def test_set_formula(self):
        parsed = P.parse_power(sample_power("switch"))
        code = B.Code("<").f(7).f(1).op(11).end()
        W.set_formula(parsed, "General", 3, "7+1", code)
        read = P.parse_power(W.write(parsed)).general.get(3).value
        self.assertEqual((read.text, read.code), ("7+1", code))

    def test_set_formula_checks(self):
        parsed = P.parse_power(sample_power("switch"))
        with self.assertRaises(FormatError):
            W.set_formula(parsed, "General", 1, "x", b"")
        with self.assertRaises(FormatError):
            W.set_formula(parsed, "General", 3, "x", b"\0")

    def test_replace_script(self):
        for build in BUILDS:
            parsed = P.parse_power(sample_power(build))
            chunk = B.lua_chunk(build, lines=(1, 2))
            W.replace_script(parsed, chunk)
            self.assertEqual(P.parse_power(W.write(parsed)).script, chunk)

    def test_remove_script(self):
        parsed = P.parse_power(sample_power("switch"))
        W.replace_script(parsed, b"")
        self.assertEqual(P.parse_power(W.write(parsed)).script, b"")

    def test_replace_script_checks_the_signature(self):
        parsed = P.parse_power(sample_power("switch"))
        with self.assertRaises(FormatError):
            W.replace_script(parsed, b"not lua")


def sample_script(build):
    entries = [
        {"event": 1, "handler": "OnStart", "filters": [(7, 1, 0, "Name"), (1, 2, 0, None)]},
        {"event": 2, "handler": "OnStart", "filters": [(15, 1, 0, "Name"), (21, 1, 0, "Other")]},
        {"event": 3},
    ]
    return B.build_scr(build, entries, B.lua_chunk(build))


class Scripts(unittest.TestCase):
    def test_round_trip(self):
        for build in BUILDS:
            data = sample_script(build)
            self.assertEqual(W.write(S.parse_script(data)), data)

    def test_equal_strings_are_shared_across_the_file(self):
        self.assertEqual(W.write(S.parse_script(sample_script("ps3"))).count(b"OnStart\0"), 1)

    def test_no_entries(self):
        for build in BUILDS:
            data = B.build_scr(build, [], B.lua_chunk(build))
            self.assertEqual(W.write(S.parse_script(data)), data)

    def test_no_chunk(self):
        data = B.build_scr("switch", [{"event": 1}])
        self.assertEqual(W.write(S.parse_script(data)), data)

    def test_replace_script(self):
        parsed = S.parse_script(sample_script("switch"))
        chunk = B.lua_chunk("switch", lines=(4, 5))
        W.replace_script(parsed, chunk)
        read = S.parse_script(W.write(parsed))
        self.assertEqual(read.chunk, chunk)
        self.assertEqual(read.messages[1].handler, "OnStart")

    def test_set_handler(self):
        for build in BUILDS:
            parsed = S.parse_script(sample_script(build))
            W.set_handler(parsed, 0, "OnChanged")
            read = S.parse_script(W.write(parsed))
            self.assertEqual(read.messages[0].handler, "OnChanged")
            self.assertEqual(read.messages[1].handler, "OnStart")

    def test_set_handler_on_entry_with_none(self):
        parsed = S.parse_script(sample_script("switch"))
        W.set_handler(parsed, 2, "Added")
        self.assertEqual(S.parse_script(W.write(parsed)).messages[2].handler, "Added")

    def test_set_handler_out_of_range(self):
        with self.assertRaises(FormatError):
            W.set_handler(S.parse_script(sample_script("ps3")), 9, "x")


class Textures(unittest.TestCase):
    FRAMES = [(5, 0.0, 0.0, 0.5, 1.0), (6, 0.5, 0.0, 1.0, 1.0)]

    def test_round_trip(self):
        for build in BUILDS:
            for frames, ids in ((self.FRAMES, None), ((), None), (self.FRAMES, [7]), ((), [1, 2, 3])):
                with self.subTest((build, len(frames))):
                    data = B.build_tex(build, 0, 2, 2, bytes(16), frames=frames, image_ids=ids)
                    self.assertEqual(W.write(T.parse_texture(data)), data)

    def test_layout_on_ps3_is_tight_from_0x270(self):
        out = W.write(T.parse_texture(B.build_tex("ps3", 0, 2, 2, bytes(16), frames=self.FRAMES)))
        tex = T.parse_texture(out)
        self.assertEqual(struct.unpack_from(">I", out, 0x24C)[0] + 0x10, 0x270)
        self.assertEqual(tex.payload_pair.file_offset, 0x270 + 8 + 40)

    def test_layout_on_switch_aligns_ids_and_frames(self):
        data = B.build_tex("switch", 0, 2, 2, bytes(16), frames=self.FRAMES, image_ids=[1])
        out = W.write(T.parse_texture(data))
        self.assertEqual((struct.unpack_from("<I", out, 0x228 - 4)[0] + 0x10) % 8, 0)

    def test_replace_pixels_ps3(self):
        tex = T.parse_texture(B.build_tex("ps3", 0, 2, 2, bytes(16)))
        W.replace_pixels(tex, bytes(range(16)))
        read = T.parse_texture(W.write(tex))
        self.assertEqual(read.payload, bytes(range(16)))

    def test_replace_pixels_ps3_dxt_block(self):
        block = struct.pack("<HHI", 0xF800, 0x001F, 0)
        tex = T.parse_texture(B.build_tex("ps3", 9, 4, 4, bytes(8)))
        W.replace_pixels(tex, block)
        image = T.decode_image(T.parse_texture(W.write(tex)))
        self.assertEqual(image.rgba[:4], bytes((255, 0, 0, 255)))

    def test_replace_pixels_ps3_second_level_and_face(self):
        face = bytes(4 * 4 + 4) + bytes(128 - 20)
        tex = T.parse_texture(B.build_tex("ps3", 7, 4, 1, face * 6, faces=6, mip_max=1))
        W.replace_pixels(tex, b"\x09\x08", 1, 5)
        self.assertEqual(T.decode_image(T.parse_texture(W.write(tex)), 1, 5).rgba[:1], b"\x09")

    def test_replace_pixels_ps3_volume(self):
        tex = T.parse_texture(B.build_tex("ps3", 0, 2, 2, bytes(2 * 2 * 64 * 4), depth=64))
        W.replace_pixels(tex, bytes(range(16)), z=37)
        image = T.decode_image(T.parse_texture(W.write(tex)), z=37)
        self.assertEqual(image.rgba[:4], bytes((1, 2, 3, 0)))
        self.assertEqual(T.decode_image(tex, z=36).rgba, bytes(16))

    def test_replace_pixels_switch_goes_through_the_swizzle(self):
        linear = bytes(32 * 32 * 4)
        new = random.Random(1).randbytes(32 * 32 * 4)
        tex = T.parse_texture(B.build_tex("switch", 0, 32, 32, linear, layout=2))
        W.replace_pixels(tex, new)
        self.assertEqual(tex.nv.image, B.swizzle(new, 32, 32, 4, 4))
        read = T.parse_texture(W.write(tex))
        self.assertEqual(T.decode_image(read).rgba, new)
        self.assertEqual(read.nv.image, B.swizzle(new, 32, 32, 4, 4))

    def test_replace_pixels_wrong_size(self):
        tex = T.parse_texture(B.build_tex("ps3", 0, 2, 2, bytes(16)))
        with self.assertRaises(FormatError):
            W.replace_pixels(tex, bytes(15))

    def test_replace_pixels_no_such_level(self):
        tex = T.parse_texture(B.build_tex("ps3", 0, 2, 2, bytes(16)))
        with self.assertRaises(FormatError):
            W.replace_pixels(tex, bytes(4), 1)


class Dispatch(unittest.TestCase):
    def test_unknown_type(self):
        with self.assertRaises(FormatError):
            W.write(object())


class ArchiveReplace(unittest.TestCase):
    def test_parsed_file_goes_into_an_archive(self):
        stl = B.build_stl("switch", [(hash33("A"), b"old")])
        source = A.write_archive({"StringList\\Foo.stl": stl, "other.bin": b"keep"})
        strings = X.parse_string_list(stl)
        W.set_text(strings, "A", "new")
        result = W.replace_in_archive(source, {"StringList\\Foo.stl": strings})
        arc = A.read_archive(result.data)
        read = X.parse_string_list(arc.read_file("StringList\\Foo.stl"))
        self.assertEqual(X.lookup(read, "A"), "new")
        self.assertEqual(arc.read_file("other.bin"), b"keep")
        self.assertTrue(A.verify_archive(result.data).ok)

    def test_every_copy_is_updated(self):
        stl = B.build_stl("ps3", [(1, b"old")])
        source = A.write_archive([A.InputFile("a.stl", [stl, stl])], version=6)
        strings = X.parse_string_list(stl)
        W.set_text(strings, 1, "new")
        arc = A.read_archive(W.replace_in_archive(source, {"a.stl": strings}).data)
        self.assertEqual(arc.read_file("a.stl", 0), arc.read_file("a.stl", 1))
        self.assertEqual(X.parse_string_list(arc.read_file("a.stl", 1)).by_hash[1], "new")
