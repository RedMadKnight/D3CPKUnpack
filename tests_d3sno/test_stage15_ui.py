"""Stage 15, W15-B8 to W15-B11: switch-form UI dialogs and fonts, UI keys, font images and font
derived values. Synthetic data only."""

import _libpath  # noqa: F401  (python -I: the library folder on sys.path)
import struct
import unittest

import builders as B
import test_ui_font as UF
from d3sno import FormatError
from d3sno import sno_layout as L
from d3sno import texture as T
from d3sno import ui_font as U
from d3sno import ui_write as W
from d3sno import x360_texture as XT

BODY = 0x10
ORDER = {"switch": "<", "x360": ">"}
BUILDS = ("switch", "x360")


def sample_dialog() -> dict:
    """Two controls (root and child) with every list kind, a style with a child spec, constants."""
    d = {"machine_time": 77, "controls": [], "styles": []}
    W.append_control(d, "Main", "", 1, {"name": "Inline", "int32s": [{"key": 5, "value": -7}]})
    child = W.append_control(d, "Button", "Main", 6)
    child["inline_style"]["child_specs"] = [{
        "path_to_child": "a.b", "strings": [{"key": 9, "text": "x"}],
        "float32s": [{"key": 8, "value": 0.5}],
    }]
    child["anim_int32s"] = [{"time": 0.0, "value": 1}, {"time": 1.0, "value": 2}]
    child["anim_floats"] = [{"time": 0.0, "value": 0.25}]
    child["anim_vector2s"] = [{"time": 0.5, "x": 1.0, "y": 2.0}]
    child["anim_descriptions"] = [
        {"name": "fade", "type": 0, "node_range": (0, 2)},
        {"name": "grow", "type": 1, "node_range": (0, 1)},
        {"name": "move", "type": 2, "node_range": (0, 1)},
    ]
    child["anim_bindings"] = [{"animation_name": "^Main.Button.fade", "property": 6, "scale": 0.6}]
    d["styles"] = [{"name": "Style", "strings": [{"key": 1, "text": "hello"}],
                    "child_specs": [{"path_to_child": "c", "int32s": [{"key": 2, "value": 3}]}]}]
    d["constant_strings"] = [{"key": 1, "text": "abc"}, {"key": 2, "text": None}]
    d["constant_int32s"] = [{"key": 3, "value": 42}]
    d["constant_float32s"] = [{"key": 4, "value": 2.5}]
    W.derive_ui_keys(d)
    return d


def hand_dialog(build: str) -> bytes:
    """A switch-form dialog built byte by byte (independent of ``write_ui``): one root control
    named "Root" and one constant string "hi"."""
    o = ORDER[build]
    img = B.Image(BODY + 0xB8, o)
    img.raw(0, B.sno_head(480 if build == "switch" else 471, o))
    img.put(0x1C, "I", 1234)                       # machine time, body 0x0C
    img.put(BODY + 0x10, "iII", 1, 0xB8, 0x170)     # controls list
    img.b += bytes(0x170)
    control = BODY + 0xB8
    img.put(control, "i", 0)
    img.put(control + 0x08, "Q", W.ui_key("Root"))
    img.put(control + 0x10, "II", 0x228, 5)
    img.put(control + 0x38, "i", 3)
    img.b += b"Root\0" + bytes(3)                  # name at body 0x228, pad to 0x230
    img.put(BODY + 0x40, "iII", 1, 0x230, 24)       # constant strings list
    img.b += bytes(24)
    img.put(BODY + 0x230, "Q", 7)
    img.put(BODY + 0x230 + 8, "II", 0x248, 3)
    img.b += b"hi\0"
    return img.done()


def sample_font(build: str) -> dict:
    tex = {"width": 32, "height": 32}
    rect_a, rect_b = (0.0, 0.0, 0.25, 0.5), (0.25, 0.0, 0.5, 0.25)
    pixels = bytes((x * 7 + y * 3) & 0xFF for y in range(32) for x in range(32))
    f = {
        "font_face": "0000.ttf", "original_font_file": "face.ttf", "line_gap_adjustment": 1.0,
        "space_scalar": 0.5,
        "textures": [{"resolution_flags": 1, "format": 23, "width": 32, "height": 32,
                      "image": W.font_image_bytes(pixels, 32, 32, build), "texture_handle": -1,
                      "frames": [{"u0_v0_u1_v1": rect_a}, {"u0_v0_u1_v1": rect_b}]}],
        "point_sizes": [{
            "height": 640, "ascender": 512, "descender": -128, "resolution_flags": 1, "icon_height": 640,
            "point_size": 12.0,
            "glyphs": [W.derive_glyph(rect_b, tex, 66, 1, 0, 1, -2, 5),
                       W.derive_glyph(rect_a, tex, 65, 0, 0, 0, 7, 9),
                       W.line_break_glyph(10)],
            "kernings": [{"previous_char": 66, "char": 65, "x": -1, "y": 0},
                         {"previous_char": 65, "char": 86, "x": -2, "y": 0}],
        }],
    }
    return W.derive_font_values(f)


# --- W15-B8 -------------------------------------------------------------------------------------

class SwitchFormUi(unittest.TestCase):
    def test_hand_built_file_reads_and_round_trips(self):
        for build in BUILDS:
            data = hand_dialog(build)
            read = W.parse_ui(data, build)
            self.assertEqual(read.problems, [], build)
            root = read.root
            self.assertEqual(root["machine_time"], 1234)
            control = root["controls"][0]
            self.assertEqual((control["name"], control["parent_name"], control["control_type"]), ("Root", None, 3))
            self.assertEqual(control["key"], W.ui_key("Root"))
            self.assertEqual(root["constant_strings"], [{"key": 7, "text": "hi"}])
            self.assertEqual(W.write_ui(read, build), data, build)

    def test_layout_offsets(self):
        data = W.write_ui(sample_dialog(), "switch")
        count, offset, size = struct.unpack_from("<iII", data, BODY + 0x10)
        self.assertEqual((count, offset, size), (2, 0xB8, 2 * 368))
        control = BODY + 0xB8 + 368  # second control
        self.assertEqual(struct.unpack_from("<i", data, control)[0], 1)
        self.assertEqual(struct.unpack_from("<i", data, control + 0x38)[0], 6)
        name_at, name_size = struct.unpack_from("<II", data, control + 0x10)
        self.assertEqual(data[BODY + name_at:BODY + name_at + name_size], b"Button\0")
        bindings = struct.unpack_from("<iII", data, control + 0x140)
        binding = BODY + bindings[1]
        self.assertEqual(bindings[0], 1)
        self.assertEqual(struct.unpack_from("<if", data, binding + 0x10), (6, struct.unpack("<f", struct.pack("<f", 0.6))[0]))
        desc = BODY + struct.unpack_from("<iI", data, control + 0x128)[1]
        self.assertEqual(struct.unpack_from("<i2i", data, desc + 0x18), (0, 0, 2))
        styles = struct.unpack_from("<iII", data, BODY + 0x28)
        self.assertEqual((styles[0], styles[2]), (1, 160))
        self.assertEqual(struct.unpack_from("<iII", data, BODY + 0x58)[0], 1)  # constant int32s
        self.assertEqual(struct.unpack_from("<iII", data, BODY + 0x70)[0], 1)  # constant float32s

    def test_blocks_aligned_and_zero_filled(self):
        for build in BUILDS:
            data = W.write_ui(sample_dialog(), build)
            read = W.parse_ui(data, build)
            self.assertEqual(read.problems, [])
            for offset, _, name in read.blocks:
                self.assertEqual((offset - BODY) % 8, 0, name)

    def test_round_trip_and_generic_writer(self):
        for build in BUILDS:
            data = W.write_ui(sample_dialog(), build)
            self.assertEqual(W.write_ui(W.parse_ui(data, build), build), data)
            self.assertEqual(L.rebuild_sno(data, "ui", build), data)  # W11 walk agrees

    def test_x360_is_the_switch_form_big_endian(self):
        a = W.parse_ui(W.write_ui(sample_dialog(), "switch"), "switch").root
        b = W.parse_ui(W.write_ui(sample_dialog(), "x360"), "x360").root
        a.pop("header"), b.pop("header")
        self.assertEqual(a, b)
        self.assertEqual(len(W.write_ui(sample_dialog(), "switch")), len(W.write_ui(sample_dialog(), "x360")))

    def test_values_read_back(self):
        root = W.parse_ui(W.write_ui(sample_dialog(), "x360"), "x360").root
        child = root["controls"][1]
        self.assertEqual(child["parent_name"], "Main")
        self.assertEqual(child["anim_vector2s"], [{"time": 0.5, "x": 1.0, "y": 2.0}])
        self.assertEqual(child["inline_style"]["child_specs"][0]["path_to_child"], "a.b")
        self.assertEqual(root["styles"][0]["child_specs"][0]["int32s"], [{"key": 2, "value": 3}])
        self.assertEqual(root["constant_strings"][1]["text"], None)
        self.assertEqual(root["constant_float32s"][0]["value"], 2.5)

    def test_zero_fields_and_slots(self):
        data = W.write_ui(sample_dialog(), "switch")
        control = BODY + 0xB8
        for at in (BODY + 0x88, BODY + 0x10 + 0x0C, control + 0x04, control + 0x18, control + 0x3C,
                   control + 0x158, control + 0x40 + 0x78):
            bad = bytearray(data)
            bad[at] = 1
            self.assertTrue(W.verify_ui(bytes(bad), "switch"), hex(at))

    def test_tiling_problems(self):
        data = W.write_ui(sample_dialog(), "switch")
        self.assertTrue(any("tiling" in p for p in W.verify_ui(data + bytes(8), "switch")))
        # move the first control's name one byte (misaligned)
        bad = bytearray(data)
        at = BODY + 0xB8 + 0x10
        offset = struct.unpack_from("<I", bad, at)[0]
        struct.pack_into("<I", bad, at, offset + 1)
        self.assertTrue(any("tiling" in p for p in W.verify_ui(bytes(bad), "switch")))

    def test_gap_must_be_zero(self):
        data = bytearray(W.write_ui(sample_dialog(), "switch"))
        name_at, size = struct.unpack_from("<II", data, BODY + 0xB8 + 0x10)  # "Main\0", 3 pad bytes
        data[BODY + name_at + size] = 0x55
        self.assertTrue(any("gap" in p for p in W.verify_ui(bytes(data), "switch")))

    def test_list_size_and_text(self):
        data = bytearray(W.write_ui(sample_dialog(), "switch"))
        struct.pack_into("<i", data, BODY + 0x58, 2)
        self.assertTrue(any("elements of 16" in p for p in W.verify_ui(bytes(data), "switch")))
        data = bytearray(hand_dialog("switch"))
        data[BODY + 0x248 + 2] = ord("!")  # "hi!" without its NUL
        self.assertTrue(any("NUL" in p for p in W.verify_ui(bytes(data), "switch")))

    def test_animation_rules_checked(self):
        d = sample_dialog()
        d["controls"][1]["anim_descriptions"][0]["node_range"] = (0, 1)
        self.assertTrue(any("exactly once" in p for p in W.verify_ui(W.write_ui(d, "switch"), "switch")))

    def test_ps3_dispatch_and_refusals(self):
        read = W.parse_ui(UF.dialog().done(), "ps3")
        self.assertEqual(read.problems, [])
        with self.assertRaises(FormatError):
            W.write_ui(sample_dialog(), "ps3")
        with self.assertRaises(FormatError):
            W.parse_ui(hand_dialog("switch")[:0x40], "switch")

    def test_reexported_from_ui_font(self):
        self.assertIs(U.parse_ui, W.parse_ui)
        self.assertIs(U.ui_key, W.ui_key)


# --- W15-B9 -------------------------------------------------------------------------------------

class UiKeys(unittest.TestCase):
    def test_fnv1a_64(self):
        self.assertEqual(W.ui_key(""), 0xCBF29CE484222325)
        self.assertEqual(W.ui_key("a"), 0xAF63DC4C8601EC8C)
        self.assertEqual(W.ui_key("foobar"), 0x85944171F73967E8)
        self.assertEqual(W.ui_key("FooBar"), W.ui_key("foobar"))
        self.assertEqual(W.ui_key("AZ"), W.ui_key("az"))
        self.assertNotEqual(W.ui_key("["), W.ui_key("{"))  # "[" is not a letter
        self.assertNotEqual(W.ui_key("\xc9"), W.ui_key("\xe9"))  # only A..Z are lower-cased
        self.assertEqual(W.ui_key(b"AB"), W.ui_key("ab"))

    def test_derive_keys(self):
        d = sample_dialog()
        root, child = d["controls"]
        self.assertEqual((root["ordinal"], child["ordinal"]), (0, 1))
        self.assertEqual(root["key"], W.ui_key("Main"))
        self.assertEqual(root["parent_key"], 0)
        self.assertEqual(child["key"], W.ui_key("Main.Button"))
        self.assertEqual(child["parent_key"], W.ui_key("Main"))
        self.assertEqual(root["inline_style"]["key"], W.ui_key("inline"))
        self.assertEqual(child["anim_descriptions"][1]["key"], W.ui_key("grow"))
        self.assertEqual(d["styles"][0]["key"], W.ui_key("style"))
        self.assertEqual(W.control_paths(d), ["Main", "Main.Button"])

    def test_derive_fixes_stale_keys(self):
        d = sample_dialog()
        d["controls"][1]["key"] = 1
        d["controls"][1]["ordinal"] = 5
        d["styles"][0]["key"] = 0
        self.assertEqual(len(W.ui_link_problems(d)), 3)
        W.derive_ui_keys(d)
        self.assertEqual(W.ui_link_problems(d, bindings=True), [])

    def test_link_problems(self):
        d = sample_dialog()
        d["controls"][1]["parent_name"] = None
        W.derive_ui_keys(d)
        self.assertTrue(any("root controls" in p for p in W.ui_link_problems(d)))
        d = sample_dialog()
        d["controls"][1]["parent_name"] = "Other"
        W.derive_ui_keys(d)
        self.assertTrue(any("names no control" in p for p in W.ui_link_problems(d)))
        self.assertEqual(W.ui_link_problems(d, parents=False), [])
        d = sample_dialog()
        d["controls"][1]["anim_descriptions"][0]["key"] = 3
        self.assertEqual(len(W.ui_link_problems(d)), 1)

    def test_bindings(self):
        d = sample_dialog()
        self.assertEqual(W.binding_target("^Main.Button.fade", d), (1, 0))
        self.assertEqual(W.binding_target("^Main.Button.move", d), (1, 2))
        self.assertIsNone(W.binding_target("^Main.fade", d))
        self.assertIsNone(W.binding_target("Main.Button.fade", d))
        d["controls"][1]["anim_bindings"][0]["animation_name"] = "^Other.x"
        self.assertEqual(W.ui_link_problems(d), [])
        self.assertEqual(len(W.ui_link_problems(d, bindings=True)), 1)

    def test_appended_control_keeps_order(self):
        d = sample_dialog()
        data = W.write_ui(d, "switch")
        new = W.append_control(d, "Extra", "Main.Button", 2)
        self.assertEqual(new["ordinal"], 2)
        self.assertEqual(new["key"], W.ui_key("Main.Button.Extra"))
        read = W.parse_ui(W.write_ui(d, "switch"), "switch")
        self.assertEqual(read.problems, [])
        self.assertEqual([c["name"] for c in read.root["controls"]], ["Main", "Button", "Extra"])
        self.assertEqual(W.ui_link_problems(read), [])
        self.assertNotEqual(len(data), len(W.write_ui(d, "switch")))


# --- W15-B10 ------------------------------------------------------------------------------------

class SwitchFormFont(unittest.TestCase):
    def test_round_trip_and_generic_writer(self):
        for build in BUILDS:
            data = W.write_font(sample_font(build), build)
            read = W.parse_font(data, build)
            self.assertEqual(read.problems, [], build)
            self.assertEqual(W.write_font(read, build), data)
            self.assertEqual(W.write_font(read, build, derive=False), data)
            self.assertEqual(L.rebuild_sno(data, "fnt", build), data)

    def test_layout(self):
        for build in BUILDS:
            o = ORDER[build]
            glyph_size = 24 if build == "switch" else 16
            data = W.write_font(sample_font(build), build)
            tex_at, tex_size = struct.unpack_from(o + "II", data, BODY + 0xA8)
            self.assertEqual((tex_at, tex_size), (0xD8, 40))
            self.assertEqual(struct.unpack_from(o + "II", data, BODY + 0xB8), (1, 1))
            size_at, size_size = struct.unpack_from(o + "II", data, BODY + 0xC0)
            self.assertEqual((size_at, size_size), (0x100, 88))
            size = BODY + size_at
            self.assertEqual(struct.unpack_from(o + "I", data, size + 0x24)[0], 3)  # glyph count
            self.assertEqual(struct.unpack_from(o + "I", data, size + 0x38)[0], 2)  # kerning count
            self.assertEqual(struct.unpack_from(o + "f", data, size + 0x54)[0], 12.0)
            kern_at, kern_size = struct.unpack_from(o + "II", data, size + 0x40)
            self.assertEqual((kern_at, kern_size), (0x158, 32))  # in the walk, aligned to 8
            glyph_at, glyph_size_all = struct.unpack_from(o + "II", data, size + 0x28)
            self.assertEqual((glyph_at, glyph_size_all), (0x178, 3 * glyph_size))  # first deferred block
            self.assertEqual(struct.unpack_from(o + "I", data, BODY + 0xD0)[0], glyph_at)
            image_at, image_size = struct.unpack_from(o + "II", data, BODY + tex_at + 0x10)
            self.assertEqual(image_at, glyph_at + glyph_size_all)  # group 1, no alignment
            frames_at, frames_size = struct.unpack_from(o + "II", data, BODY + tex_at + 0x20)
            self.assertEqual((frames_at, frames_size), (image_at + image_size, 32))
            self.assertEqual(len(data), BODY + frames_at + frames_size)

    def test_glyph_forms(self):
        sw = W.write_font(sample_font("switch"), "switch")
        at = BODY + struct.unpack_from("<I", sw, BODY + 0x100 + 0x28)[0]
        self.assertEqual(struct.unpack_from("<iihhhhhH4x", sw, at), (-2, -1, 0, 0, 0, 0, -1, 10))
        self.assertEqual(struct.unpack_from("<iihhhhhH4x", sw, at + 24), (-1, 0, 0, 7, 15, 9, 0, 65))
        xb = W.write_font(sample_font("x360"), "x360")
        at = BODY + struct.unpack_from(">I", xb, BODY + 0x100 + 0x28)[0]
        self.assertEqual(struct.unpack_from(">iibbBBhH", xb, at + 16), (-1, 0, 0, 7, 15, 9, 0, 65))
        self.assertEqual(struct.unpack_from(">iibbBBhH", xb, at + 32), (-1, 1, 1, -2, 7, 5, 0, 66))
        f = sample_font("x360")
        f["point_sizes"][0]["glyphs"][0]["bearing_x"] = 200
        with self.assertRaises(FormatError):
            W.write_font(f, "x360")

    def test_problems(self):
        data = W.write_font(sample_font("switch"), "switch")
        for at in (BODY + 0xB0, BODY + 0xD4, BODY + 0x100 + 0x1C, BODY + 0x100 + 0x30, BODY + 0x100 + 0x3C):
            bad = bytearray(data)
            bad[at] = 1
            self.assertTrue(W.verify_font(bytes(bad), "switch"), hex(at))
        bad = bytearray(data)
        struct.pack_into("<I", bad, BODY + 0xD0, 4)
        self.assertTrue(any("custom-data" in p for p in W.verify_font(bytes(bad), "switch")))
        bad = bytearray(data)
        struct.pack_into("<I", bad, BODY + 0x100 + 0x24, 2)
        self.assertTrue(any("elements of 24" in p for p in W.verify_font(bytes(bad), "switch")))
        self.assertTrue(any("tiling" in p for p in W.verify_font(data + b"\0", "switch")))

    def test_custom_offset_without_glyphs(self):
        f = sample_font("switch")
        f["point_sizes"][0]["glyphs"] = []
        data = W.write_font(W.derive_font_values(f), "switch")
        read = W.parse_font(data, "switch")
        image_at = struct.unpack_from("<I", data, BODY + 0xD8 + 0x10)[0]
        self.assertEqual(read.root["custom_data_offset"], image_at)
        self.assertEqual(read.problems, [])

    def test_ps3_dispatch(self):
        self.assertEqual(W.parse_font(UF.font().done(), "ps3").problems, [])


class FontImages(unittest.TestCase):
    PIXELS = bytes((x * 5 + y * 11) & 0xFF for y in range(64) for x in range(32))

    def test_ps3_rows(self):
        self.assertEqual(W.font_image_bytes(self.PIXELS, 32, 64, "ps3"), self.PIXELS)
        with self.assertRaises(FormatError):
            W.font_image_pixels(self.PIXELS[:-1], 32, 64, "ps3")

    def test_switch_container(self):
        blob = W.font_image_bytes(self.PIXELS, 32, 64, "switch")
        nv = T._parse_nv(blob, 0)
        self.assertEqual((nv.pixel_format, nv.mip_count, nv.width, nv.height, nv.target), (1, 1, 32, 64, T.NV_TARGET_2D))
        self.assertEqual(W.font_image_pixels(blob, 32, 64, "switch"), self.PIXELS)
        self.assertNotEqual(nv.image[:len(self.PIXELS)], self.PIXELS)  # block-linear, not rows
        self.assertEqual(W.font_image_problems(blob, 32, 64, "switch"), [])
        self.assertTrue(W.font_image_problems(blob[:-1] + b"\1", 32, 64, "switch"))
        self.assertEqual(len(blob), W.font_image_size(32, 64, "switch"))

    def test_x360_tiled(self):
        blob = W.font_image_bytes(self.PIXELS, 32, 64, "x360")
        for x, y in ((0, 0), (5, 3), (31, 63), (17, 40)):
            self.assertEqual(blob[XT.tiled_element(x, y, 32, 1)], self.PIXELS[y * 32 + x])
        self.assertEqual(W.font_image_pixels(blob, 32, 64, "x360"), self.PIXELS)
        self.assertEqual(W.x360_font_pitch(16), 32)
        small = bytes(range(16 * 8))
        self.assertEqual(W.font_image_pixels(W.font_image_bytes(small, 16, 8, "x360"), 16, 8, "x360"), small)

    def test_font_problems_check_images(self):
        f = sample_font("x360")
        f["textures"][0]["image"] = bytes(10)
        self.assertTrue(any("texture 0" in p for p in U.font_problems(f, build="x360")))


# --- W15-B11 ------------------------------------------------------------------------------------

class FontDerived(unittest.TestCase):
    def test_glyph_height(self):
        tex = {"width": 64, "height": 32}
        self.assertEqual(W.derive_glyph((0.0, 0.25, 0.5, 0.75), tex, 65, 0, 0)["height"], 15)
        self.assertEqual(W.glyph_height((0.0, 0.25, 0.5, 0.75), tex, "ps3"), 15 * 64)
        self.assertEqual(W.frame_pixel_size((0.0, 0.25, 0.5, 0.75), tex), (32, 16))
        with self.assertRaises(FormatError):
            W.derive_glyph((0.0, 0.0, 0.5, 0.01), tex, 65, 0, 0)

    def test_line_break(self):
        g = W.line_break_glyph(10)
        self.assertEqual((g["texture_handle"], g["texture_index"], g["frame"]), (-2, -1, -1))
        self.assertTrue(W.is_line_break(g))

    def test_sample_is_clean_and_sorted(self):
        f = sample_font("switch")
        size = f["point_sizes"][0]
        self.assertEqual([g["char"] for g in size["glyphs"]], [10, 65, 66])
        self.assertEqual([(k["previous_char"], k["char"]) for k in size["kernings"]], [(65, 86), (66, 65)])
        self.assertEqual((size["glyph_count"], f["texture_count"], f["textures"][0]["frame_count"]), (3, 1, 2))
        self.assertEqual(W.font_derived_problems(f, "switch"), [])

    def test_problems(self):
        def broken(change, build="switch"):
            f = sample_font("switch")
            change(f, f["point_sizes"][0])
            return W.font_derived_problems(f, build)
        self.assertTrue(broken(lambda f, s: s["glyphs"].reverse()))
        self.assertTrue(broken(lambda f, s: s["kernings"].reverse()))
        self.assertTrue(broken(lambda f, s: s["glyphs"].pop(0)))  # no line break
        self.assertTrue(broken(lambda f, s: s["glyphs"][1].update(height=14)))
        self.assertTrue(broken(lambda f, s: s.update(resolution_flags=2)))
        self.assertTrue(broken(lambda f, s: s["glyphs"][1].update(frame=-1)))
        self.assertTrue(broken(lambda f, s: s.update(glyph_count=4)))
        self.assertTrue(broken(lambda f, s: f.update(texture_count=2)))
        def duplicate(f, s):
            s["glyphs"].insert(2, dict(s["glyphs"][1]))
            s["glyph_count"] = 4
        self.assertTrue(any("is not after" in p for p in broken(duplicate, "x360")))  # V18-5: strict on x360

    def test_parse_reports_derived(self):
        f = sample_font("switch")
        f["point_sizes"][0]["glyphs"][1]["height"] = 3
        data = W.write_font(f, "switch")
        self.assertTrue(any("height" in p for p in W.verify_font(data, "switch")))

    def test_ps3_compatible(self):
        data = UF.font().done()
        self.assertEqual(U.verify_ps3_font(data), [])
        root = U.parse_ps3_font(data).root
        self.assertEqual(U.font_problems(root, data), [])
        self.assertTrue(U.font_problems(root, data, derived=True))  # its glyph height is not the frame's


if __name__ == "__main__":
    unittest.main()
