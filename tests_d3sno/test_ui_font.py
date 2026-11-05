"""Stage 7, T5 and T6: PS3 ``.ui`` and font files read from the layout table."""

import struct
import unittest

import builders as B
from d3sno import FormatError
from d3sno import ui_font as U

BODY = 0x10


def text(img, at, value: bytes):
    img.put(at, "II", *img.append(value))


def elements(img, at, record, count, fill=None):
    """A 16-byte list at ``at`` with ``count`` zeroed elements; returns their file position."""
    size = U.record_size(record)
    img.put(at, "i", count)
    if not count:
        return None
    offset, _ = img.pair(at + 4, bytes(size * count))
    return offset + BODY


def dialog(stale=True):
    """A dialog with one control, one style with a child spec and one of each constant."""
    img = B.ps3_file(BODY + 0x78)
    img.put(BODY + 0x0C, "I", 1234)
    control = elements(img, BODY + 0x10, "ui_control", 1)
    style = elements(img, BODY + 0x20, "ui_style", 1)
    strings = elements(img, BODY + 0x30, "ui_string_element", 1)
    ints = elements(img, BODY + 0x40, "ui_int32_element", 1)
    floats = elements(img, BODY + 0x50, "ui_float32_element", 1)

    img.put(control, "iIQ", 3, 0xDEAD if stale else 0, 77)  # +4 is an alignment hole
    text(img, control + 0x10, b"Button\0")
    img.put(control + 0x20, "Q", 0)
    text(img, control + 0x28, b"Root\0")
    img.put(control + 0x38, "i", 6)
    if stale:
        img.put(control + 0x1C, "I", 0xCAFEBABE)
        img.put(control + 0x34, "I", 0x12345678)
        img.put(control + 0x40 + 0x14, "I", 0xFFFFFFFF)  # inline style hole
    inline_ints = elements(img, control + 0x40 + 0x28, "ui_int32_element", 1)
    img.put(inline_ints, "Qi", 5, -7)
    anim = elements(img, control + 0xB0, "ui_anim_int32", 2)
    img.put(anim, "fifi", 0.0, 1, 1.0, 2)
    desc = elements(img, control + 0xE0, "ui_anim_description", 1)
    img.put(desc, "Q", 99)
    text(img, desc + 8, b"fade\0")
    img.put(desc + 0x18, "iii", 0, 0, 2)  # V7-T5a: type 0 selects anim_int32s, range [0, 2)
    binding = elements(img, control + 0xF0, "ui_anim_binding", 1)
    text(img, binding, b"^dlg_main.Button.fade\0")
    img.put(binding + 0x10, "i", 6)

    img.put(style, "Q", 0x55)
    text(img, style + 8, b"Style\0")
    spec = elements(img, style + 0x48, "ui_child_spec", 1)
    text(img, spec, b"a/b\0")
    spec_floats = elements(img, spec + 0x30, "ui_float32_element", 1)
    img.put(spec_floats, "Qf", 8, 0.5)

    img.put(strings, "Q", 1)
    text(img, strings + 8, b"hello\0")
    img.put(ints, "Qi", 2, 42)
    img.put(floats, "Qf", 3, 2.5)
    return img


def font(glyph_texture=0, glyph_frame=1, image=b"\1" * 8, custom=None):
    img = B.ps3_file(BODY + 0xCC)
    img.raw(BODY + 0x10, b"0000.ttf")
    img.raw(BODY + 0x50, b"face.ttf")
    img.put(BODY + 0xA0, "ff", 1.0, 0.5)
    img.put(BODY + 0xA8, "I", 1)
    tex_offset, _ = img.pair(BODY + 0xAC, bytes(40))
    tex = tex_offset + BODY
    img.put(BODY + 0xB8, "I", 1)
    size_offset, _ = img.pair(BODY + 0xBC, bytes(72))
    size = size_offset + BODY
    img.put(tex, "Iiii", 1, 23, 4, 2)
    image_offset, _ = img.pair(tex + 0x10, image)
    img.put(tex + 0x18, "iI", -1, 2)
    img.pair(tex + 0x20, struct.pack(">8f", 0, 0, 0.5, 0.5, 0.5, 0.5, 1, 1))
    img.put(BODY + 0xC8, "I", image_offset if custom is None else custom)
    img.put(size, "7i", 640, 640, 512, 520, -128, 0, 768)
    img.put(size + 0x1C, "II", 1, 2)
    img.pair(size + 0x24, struct.pack(">8i", -1, glyph_frame, 0, 0, 640, 576, glyph_texture, 65)
             + struct.pack(">8i", -2, -1, 0, 0, 0, 0, -1, 10))
    img.put(size + 0x30, "I", 1)
    img.pair(size + 0x34, struct.pack(">IIii", 65, 86, -64, 0))
    img.put(size + 0x40, "if", 640, 12.0)
    return img


class Table(unittest.TestCase):
    def test_sizes_from_the_description(self):
        sizes = {
            "ui_dialog": 0x78, "ui_list": 16, "ui_string_element": 24, "ui_int32_element": 16,
            "ui_float32_element": 16, "ui_style": 112, "ui_child_spec": 76, "ui_control": 272,
            "ui_anim_int32": 8, "ui_anim_float": 8, "ui_anim_vector2": 12,
            "ui_anim_description": 40, "ui_anim_binding": 20, "font": 0xCC, "font_texture": 40,
            "font_frame": 16, "font_point_size": 72, "font_glyph": 32, "font_kerning": 16,
            "path": 40,
        }
        for record, size in sizes.items():
            self.assertEqual(U.record_size(record), size, record)

    def test_dialog_lists_in_order(self):
        lists = [f for f in U.ui_font_layouts()["ui_dialog"] if f.kind == "list"]
        self.assertEqual([(f.offset, f.type) for f in lists], [
            (0x10, "ui_control"), (0x20, "ui_style"), (0x30, "ui_string_element"),
            (0x40, "ui_int32_element"), (0x50, "ui_float32_element"),
        ])

    def test_path_header_in_the_table(self):
        path = {f.name: f.offset for f in U.ui_font_layouts()["path"]}
        self.assertEqual((path["distribution"], path["scalar_function"], path["scalar_lerp_one"]),
                         (0x0C, 0x10, 0x18))


class Dialogs(unittest.TestCase):
    def test_read(self):
        read = U.parse_ps3_ui(dialog().done())
        self.assertEqual(read.problems, [])
        root = read.root
        self.assertEqual(root["machine_time"], 1234)
        control = root["controls"][0]
        self.assertEqual((control["ordinal"], control["key"], control["name"]), (3, 77, "Button"))
        self.assertEqual((control["parent_key"], control["parent_name"], control["control_type"]),
                         (0, "Root", 6))
        self.assertEqual(control["inline_style"]["int32s"][0], {"key": 5, "value": -7})
        self.assertEqual([a["value"] for a in control["anim_int32s"]], [1, 2])
        self.assertEqual(control["anim_descriptions"][0]["node_range"], (0, 2))
        self.assertEqual(control["anim_bindings"][0],
                         {"animation_name": "^dlg_main.Button.fade", "property": 6})
        style = root["styles"][0]
        self.assertEqual(style["name"], "Style")
        self.assertEqual(style["child_specs"][0]["path_to_child"], "a/b")
        self.assertEqual(style["child_specs"][0]["float32s"][0]["value"], 0.5)
        self.assertEqual(root["constant_strings"][0], {"key": 1, "text": "hello"})
        self.assertEqual(root["constant_int32s"][0]["value"], 42)
        self.assertEqual(root["constant_float32s"][0]["value"], 2.5)

    def test_alignment_holes_are_ignored(self):
        self.assertEqual(U.verify_ps3_ui(dialog(stale=True).done()), [])
        self.assertEqual(U.verify_ps3_ui(dialog(stale=False).done()), [])

    def test_slot_and_reserved_must_be_zero(self):
        img = dialog()
        img.put(BODY + 0x1C, "I", 1)  # controls list slot
        self.assertEqual(len(U.verify_ps3_ui(img.done())), 1)
        img = dialog()
        img.put(BODY + 0x60, "I", 1)  # reserved
        self.assertEqual(len(U.verify_ps3_ui(img.done())), 1)
        img = dialog()
        control = struct.unpack_from(">I", img.b, BODY + 0x14)[0] + BODY
        img.put(control + 0x18, "I", 9)  # name slot
        self.assertEqual(len(U.verify_ps3_ui(img.done())), 1)

    def test_text_must_have_one_nul(self):
        img = dialog()
        strings = struct.unpack_from(">I", img.b, BODY + 0x34)[0] + BODY
        text(img, strings + 8, b"he\0lo\0")
        problems = U.verify_ps3_ui(img.done())
        self.assertTrue(any("NUL" in p for p in problems))
        img = dialog()
        text(img, strings + 8, b"hello")
        self.assertTrue(any("NUL" in p for p in U.verify_ps3_ui(img.done())))

    def test_list_size_is_count_times_element(self):
        img = dialog()
        img.put(BODY + 0x40, "i", 2)
        self.assertTrue(any("elements of 16" in p for p in U.verify_ps3_ui(img.done())))

    def test_tiling(self):
        img = dialog()
        img.b += bytes(16)
        self.assertTrue(any("R4" in p for p in U.verify_ps3_ui(img.done())))
        img = dialog()
        img.b += bytes(8)
        self.assertEqual(U.verify_ps3_ui(img.done()), [])

    def test_truncated(self):
        data = dialog().done()
        with self.assertRaises(FormatError):
            U.parse_ps3_ui(data[:0x60])
        self.assertTrue(U.verify_ps3_ui(data[:-20]))


class Fonts(unittest.TestCase):
    def test_read(self):
        read = U.parse_ps3_font(font().done())
        self.assertEqual(read.problems, [])
        root = read.root
        self.assertEqual((root["font_face"], root["original_font_file"]), ("0000.ttf", "face.ttf"))
        self.assertEqual((root["line_gap_adjustment"], root["space_scalar"]), (1.0, 0.5))
        texture = root["textures"][0]
        self.assertEqual((texture["format"], texture["width"], texture["height"]), (23, 4, 2))
        self.assertEqual(texture["texture_handle"], -1)
        self.assertEqual(texture["frames"][1]["u0_v0_u1_v1"], (0.5, 0.5, 1.0, 1.0))
        size = root["point_sizes"][0]
        self.assertEqual(U.fixed_26_6(size["height"]), 10.0)
        self.assertEqual(size["point_size"], 12.0)
        self.assertEqual([g["char"] for g in size["glyphs"]], [65, 10])
        self.assertEqual(size["glyphs"][1]["texture_handle"], -2)
        self.assertEqual(size["kernings"][0], {"previous_char": 65, "char": 86, "x": -64, "y": 0})

    def test_glyph_indices(self):
        self.assertEqual(len(U.verify_ps3_font(font(glyph_texture=1).done())), 1)
        self.assertEqual(len(U.verify_ps3_font(font(glyph_frame=2).done())), 1)
        self.assertEqual(U.verify_ps3_font(font(glyph_frame=-1).done()), [])

    def test_image_size(self):
        self.assertEqual(len(U.verify_ps3_font(font(image=b"\1" * 6).done())), 1)

    def test_custom_data_offset(self):
        self.assertEqual(len(U.verify_ps3_font(font(custom=4).done())), 1)

    def test_face_name_zero_after_nul(self):
        img = font()
        img.raw(BODY + 0x10 + 20, b"x")
        self.assertEqual(len(U.verify_ps3_font(img.done())), 1)

    def test_counts_must_match(self):
        img = font()
        img.put(BODY + 0xB8, "I", 2)
        self.assertTrue(U.verify_ps3_font(img.done()))


if __name__ == "__main__":
    unittest.main()
