"""Stage 15.1 (corrections 18), UI area: V18-2 shared texts in switch-form dialogs, V18-4 control
ordinals, V18-5 glyph order. Synthetic data only."""

import _libpath  # noqa: F401  (python -I: the library folder on sys.path)
import struct
import unittest

import test_stage15_ui as S
from d3sno import FormatError
from d3sno import sno_layout as L
from d3sno import ui_font as U
from d3sno import ui_write as W

BODY = 0x10
BUILDS = ("switch", "x360")
CONTROL = 368


def shared_dialog() -> dict:
    """Repeated texts across fields: the root's name "Main" is the child's parent path and the
    name of the child's inline style; "Main" is also a string of the root's inline style; the
    child's name "Button" is the grandchild's parent-path tail; a constant string and a dialog
    style name repeat "Main" too (HYPOTHESIS fields)."""
    d = {"machine_time": 5, "controls": [], "styles": []}
    W.append_control(d, "Main", "", 1, {"name": "Inline", "strings": [{"key": 1, "text": "Main"}]})
    W.append_control(d, "Button", "Main", 6, {"name": "Main"})
    W.append_control(d, "Label", "Main.Button", 2, {"name": "Inline"})
    d["styles"] = [{"name": "Main"}]
    d["constant_strings"] = [{"key": 1, "text": "Main"}, {"key": 2, "text": "Button"}]
    W.derive_ui_keys(d)
    return d


def pair(data: bytes, at: int, build: str) -> tuple[int, int]:
    return struct.unpack_from(S.ORDER[build] + "II", data, at)


def control_at(i: int) -> int:
    return BODY + 0xB8 + i * CONTROL


# --- V18-2 --------------------------------------------------------------------------------------

class SharedTexts(unittest.TestCase):
    def test_proven_fields_share_one_block(self):
        for build in BUILDS:
            data = W.write_ui(shared_dialog(), build)
            main = pair(data, control_at(0) + 0x10, build)                 # root name
            self.assertEqual(pair(data, control_at(1) + 0x28, build), main)  # child parent path
            self.assertEqual(pair(data, control_at(1) + 0x40 + 0x08, build), main)  # inline style name
            inline = pair(data, control_at(0) + 0x40 + 0x08, build)
            self.assertEqual(pair(data, control_at(2) + 0x40 + 0x08, build), inline)
            self.assertNotEqual(pair(data, control_at(1) + 0x10, build), main)

    def test_round_trip_and_tiling(self):
        for build in BUILDS:
            for share in W.TEXT_SHARING:
                data = W.write_ui(shared_dialog(), build, share)
                read = W.parse_ui(data, build)
                self.assertEqual(read.problems, [], (build, share))
                self.assertEqual(W.write_ui(read, build), data, (build, share))
                self.assertEqual(L.rebuild_sno(data, "ui", build), data, (build, share))
                self.assertEqual(W.ui_link_problems(read), [])

    def test_texts_read_back(self):
        root = W.parse_ui(W.write_ui(shared_dialog(), "x360", "all"), "x360").root
        self.assertEqual(root["controls"][1]["parent_name"], "Main")
        self.assertEqual(root["controls"][1]["inline_style"]["name"], "Main")
        self.assertEqual(root["controls"][0]["inline_style"]["strings"][0]["text"], "Main")
        self.assertEqual(root["styles"][0]["name"], "Main")
        self.assertEqual([s["text"] for s in root["constant_strings"]], ["Main", "Button"])

    def test_scopes(self):
        counts = {s: W.write_ui(shared_dialog(), "switch", s).count(b"Main\0") for s in W.TEXT_SHARING}
        # proven: the controls' texts share; the style name and the constant string do not
        self.assertEqual(counts, {"proven": 3, "all": 1, "none": 6})
        all_ = W.write_ui(shared_dialog(), "switch", "all")
        self.assertEqual(all_.count(b"Button\0") - all_.count(b"Main.Button\0"), 1)
        self.assertEqual(all_.count(b"Inline\0"), 1)
        self.assertLess(len(all_), len(W.write_ui(shared_dialog(), "switch", "proven")))
        with self.assertRaises(FormatError):
            W.write_ui(shared_dialog(), "switch", "some")

    def test_first_occurrence_in_walk_order(self):
        layout = W.text_layout(shared_dialog(), "switch", "all")
        # walk: Main, Inline, Main(string), Button, Main(parent), Main(inline), Label,
        # Main.Button, Inline, Main(style), Main(const), Button(const)
        self.assertEqual(layout, [-1, -1, 0, -1, 0, 0, -1, -1, 1, 0, 0, 3])
        self.assertEqual(W.text_layout(shared_dialog(), "switch", "none"), [-1] * 12)

    def test_parsed_layout_is_kept(self):
        for build in BUILDS:
            for share in W.TEXT_SHARING:
                data = W.write_ui(shared_dialog(), build, share)
                read = W.parse_ui(data, build)
                for other in W.TEXT_SHARING:  # the stored layout wins over the scope
                    self.assertEqual(W.write_ui(read, build, other), data, (build, share, other))

    def test_keep_layout_off_and_invalid_layout(self):
        stored = W.write_ui(shared_dialog(), "switch", "none")
        read = W.parse_ui(stored, "switch")
        self.assertEqual(W.write_ui(read, "switch", keep_layout=False), W.write_ui(shared_dialog(), "switch"))
        # a shared text changed: the layout is no longer valid, the scope decides
        read = W.parse_ui(W.write_ui(shared_dialog(), "switch", "all"), "switch")
        read.root["controls"][1]["parent_name"] = "Other"
        data = W.write_ui(read, "switch")
        expected = dict(read.root)
        expected.pop("text_layout")
        self.assertEqual(data, W.write_ui(expected, "switch"))
        again = W.parse_ui(data, "switch")
        self.assertEqual(again.problems, [])
        self.assertEqual(again.root["controls"][1]["parent_name"], "Other")
        self.assertEqual(again.root["controls"][1]["inline_style"]["name"], "Main")
        self.assertEqual(again.root["styles"][0]["name"], "Main")
        # a layout of the wrong length is ignored
        d = shared_dialog()
        d["text_layout"] = [-1]
        self.assertEqual(W.write_ui(d, "switch"), W.write_ui(shared_dialog(), "switch"))

    def test_hand_built_shared_constant(self):
        """Independent bytes: the constant string's pair points at the root's name block."""
        for build in BUILDS:
            data = bytearray(S.hand_dialog(build)[:BODY + 0x248])
            struct.pack_into(S.ORDER[build] + "II", data, BODY + 0x230 + 8, 0x228, 5)
            data = bytes(data)
            read = W.parse_ui(data, build)
            self.assertEqual(read.problems, [], build)
            self.assertEqual(read.root["constant_strings"][0]["text"], "Root")
            self.assertEqual(read.root["text_layout"], [-1, 0])
            self.assertEqual(W.write_ui(read, build), data)
            self.assertEqual(len(read.blocks), 3)  # controls, name, constants list
            root = dict(read.root)
            root.pop("text_layout")
            self.assertNotEqual(W.write_ui(root, build), data)  # proven scope: a constant is stored
            self.assertEqual(W.write_ui(root, build, "all"), data)

    def test_tiling_rejects_other_repeats(self):
        data = W.write_ui(shared_dialog(), "switch")
        offset, size = pair(data, control_at(0) + 0x10, "switch")
        bad = bytearray(data)  # same offset, other size: not a repeat of a text pair
        struct.pack_into("<II", bad, control_at(1) + 0x28, offset, size - 1)
        self.assertTrue(any("tiling" in p for p in W.verify_ui(bytes(bad), "switch")))
        bad = bytearray(data)  # a text pair repeating the controls list block
        struct.pack_into("<II", bad, control_at(1) + 0x28, 0xB8, 3 * CONTROL)
        self.assertTrue(any("tiling" in p for p in W.verify_ui(bytes(bad), "switch")))

    def test_no_repeat_dialog_unchanged(self):
        d = S.sample_dialog()
        d["controls"][1]["parent_name"] = None  # no repeated text left
        none = W.write_ui(d, "switch", "none")
        self.assertEqual(W.write_ui(d, "switch"), none)
        self.assertEqual(W.write_ui(d, "switch", "all"), none)


# --- V18-4 --------------------------------------------------------------------------------------

def three_controls() -> dict:
    d = {"controls": []}
    W.append_control(d, "Main")
    W.append_control(d, "A", "Main")
    W.append_control(d, "B", "Main")
    return d


class Ordinals(unittest.TestCase):
    def test_derive_keeps_stored_permutation(self):
        d = three_controls()
        for control, ordinal in zip(d["controls"], (2, 0, 1)):
            control["ordinal"] = ordinal
        W.derive_ui_keys(d)
        self.assertEqual([c["ordinal"] for c in d["controls"]], [2, 0, 1])
        self.assertEqual(W.ui_link_problems(d), [])
        for build in BUILDS:
            read = W.parse_ui(W.write_ui(d, build), build)
            self.assertEqual([c["ordinal"] for c in read.root["controls"]], [2, 0, 1])
            self.assertEqual(W.ui_link_problems(read), [])

    def test_derive_fills_only_bad_ordinals(self):
        d = three_controls()
        for control, ordinal in zip(d["controls"], (2, 2, 7)):
            control["ordinal"] = ordinal
        W.derive_ui_keys(d)
        self.assertEqual([c["ordinal"] for c in d["controls"]], [2, 0, 1])
        d = three_controls()
        for control in d["controls"]:
            del control["ordinal"]
        W.derive_ui_keys(d)
        self.assertEqual([c["ordinal"] for c in d["controls"]], [0, 1, 2])

    def test_check_is_the_permutation_only(self):
        d = three_controls()
        for control, ordinal in zip(d["controls"], (1, 2, 0)):
            control["ordinal"] = ordinal
        self.assertEqual(W.ordinal_problems(d), [])
        d["controls"][2]["ordinal"] = 1
        self.assertEqual(len(W.ordinal_problems(d)), 1)
        self.assertEqual(len(W.ui_link_problems(d)), 1)
        d["controls"][2]["ordinal"] = 3
        self.assertTrue(W.ordinal_problems(d))
        self.assertEqual(W.ordinal_problems({"controls": []}), [])

    def test_append_gives_n(self):
        d = three_controls()
        for control, ordinal in zip(d["controls"], (2, 0, 1)):
            control["ordinal"] = ordinal
        self.assertEqual(W.append_control(d, "C", "Main")["ordinal"], 3)
        self.assertEqual([c["ordinal"] for c in d["controls"]], [2, 0, 1, 3])
        self.assertEqual(W.ui_link_problems(d), [])

    def test_remove_lowers_ordinals_above(self):
        d = three_controls()
        W.append_control(d, "C", "Main")
        for control, ordinal in zip(d["controls"], (3, 0, 1, 2)):
            control["ordinal"] = ordinal
        removed = W.remove_control(d, "Main.A")  # ordinal 0
        self.assertEqual(removed["name"], "A")
        self.assertEqual([(c["name"], c["ordinal"]) for c in d["controls"]], [("Main", 2), ("B", 0), ("C", 1)])
        self.assertEqual(W.ui_link_problems(d), [])
        W.remove_control(d, 2)  # "C", ordinal 1
        self.assertEqual([(c["name"], c["ordinal"]) for c in d["controls"]], [("Main", 1), ("B", 0)])
        self.assertEqual(W.ordinal_problems(d), [])
        with self.assertRaises(FormatError):
            W.remove_control(d, "Main.Z")
        with self.assertRaises(FormatError):
            W.remove_control(d, 5)

    def test_remove_drops_text_layout_and_writes(self):
        read = W.parse_ui(W.write_ui(shared_dialog(), "switch"), "switch")
        W.remove_control(read, "Main.Button.Label")
        self.assertNotIn("text_layout", read.root)
        again = W.parse_ui(W.write_ui(read, "switch"), "switch")
        self.assertEqual(again.problems, [])
        self.assertEqual(W.ui_link_problems(again), [])

    def test_reexported(self):
        self.assertIs(U.remove_control, W.remove_control)
        self.assertIs(U.glyph_order_problems, W.glyph_order_problems)


# --- V18-5 --------------------------------------------------------------------------------------

def repeated_font(build: str) -> dict:
    """Point size with char 65 twice: two glyphs with different frames on the same texture."""
    f = S.sample_font(build)
    tex = f["textures"][0]
    a = W.derive_glyph(tex["frames"][1]["u0_v0_u1_v1"], tex, 65, 1, 0, 3, 4, 5)
    f["point_sizes"][0]["glyphs"].insert(2, a)  # [10, 65 (frame 0), 65 (frame 1), 66]
    return W.derive_font_values(f)


class GlyphOrder(unittest.TestCase):
    def test_switch_allows_repeat(self):
        f = repeated_font("switch")
        self.assertEqual([(g["char"], g["frame"]) for g in f["point_sizes"][0]["glyphs"]],
                         [(10, -1), (65, 0), (65, 1), (66, 1)])
        self.assertEqual(W.font_derived_problems(f, "switch"), [])
        self.assertTrue(W.font_derived_problems(f, "switch", strict_order=True))

    def test_ps3_and_x360_strict(self):
        for build in ("x360", "ps3"):
            glyphs = repeated_font("switch")["point_sizes"][0]["glyphs"]
            found = W.glyph_order_problems(glyphs, build)
            self.assertEqual(len(found), 1, build)
            self.assertIn("repeated", found[0])
            self.assertEqual(W.glyph_order_problems(glyphs, build, strict=False), [])
        self.assertTrue(any("repeated" in p for p in W.font_derived_problems(repeated_font("x360"), "x360")))

    def test_lower_always_reported(self):
        glyphs = [{"char": 65}, {"char": 64}, {"char": 66}]
        for build in ("switch", "x360", "ps3"):
            found = W.glyph_order_problems(glyphs, build)
            self.assertEqual(len(found), 1, build)
            self.assertIn("lower", found[0])
            self.assertIn("glyph 1", found[0])

    def test_round_trip_keeps_repeated_order(self):
        for build in BUILDS:
            f = repeated_font(build)
            data = W.write_font(f, build)
            read = W.parse_font(data, build)
            frames = [(g["char"], g["frame"]) for g in read.root["point_sizes"][0]["glyphs"]]
            self.assertEqual(frames, [(10, -1), (65, 0), (65, 1), (66, 1)])
            if build == "switch":
                self.assertEqual(read.problems, [])
            else:
                self.assertTrue(any("repeated" in p for p in read.problems))
            self.assertEqual(W.write_font(read, build), data)
            W.derive_font_values(read)
            self.assertEqual(W.write_font(read, build), data)

    def test_stable_sort(self):
        f = repeated_font("switch")
        size = f["point_sizes"][0]
        g10, g65a, g65b, g66 = size["glyphs"]
        size["glyphs"] = [g66, g65b, g10, g65a]  # later 65 first: its order is kept
        W.derive_font_values(f)
        self.assertEqual([id(g) for g in size["glyphs"]], [id(g10), id(g65b), id(g65a), id(g66)])
        size["glyphs"] = [g10, g65b, g65a, g66]  # non-decreasing: untouched
        W.derive_font_values(f)
        self.assertEqual([id(g) for g in size["glyphs"]], [id(g10), id(g65b), id(g65a), id(g66)])


if __name__ == "__main__":
    unittest.main()
