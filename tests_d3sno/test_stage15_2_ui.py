"""Stage 15.2 (corrections 19), UI area: V19-1 shared lists in switch-form dialogs. Synthetic data
only."""

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
INLINE = 0x40
# list pair positions (count at +0, pair at +4)
STYLE_LISTS = {"strings": 0x18, "int32s": 0x30, "float32s": 0x48, "child_specs": 0x60}
ANIM = {"anim_int32s": 0xE0, "anim_floats": 0xF8, "anim_vector2s": 0x110,
        "anim_descriptions": 0x128, "anim_bindings": 0x140}


def inline(name: str, value: int = 5) -> dict:
    return {"name": name, "strings": [{"key": 1, "text": "t"}], "int32s": [{"key": 2, "value": value}],
            "float32s": [{"key": 3, "value": 1.5}],
            "child_specs": [{"path_to_child": "p", "int32s": [{"key": 2, "value": 5}]}]}


def animate(control: dict) -> None:
    control["anim_int32s"] = [{"time": 0.0, "value": 1}, {"time": 1.0, "value": 2}]
    control["anim_floats"] = [{"time": 1.0, "value": 0.0}]
    control["anim_vector2s"] = [{"time": 0.0, "x": 1.0, "y": 2.0}]
    control["anim_descriptions"] = [{"name": "i", "type": 0, "node_range": (0, 2)},
                                    {"name": "f", "type": 1, "node_range": (0, 1)},
                                    {"name": "v", "type": 2, "node_range": (0, 1)}]
    control["anim_bindings"] = [{"animation_name": "^Main.i", "property": 1, "scale": 1.0}]


def repeat_dialog() -> dict:
    """Controls 0 and 1 have equal lists in every covered field; control 2 repeats only the
    animation float list. Cross-field equal bytes: control 0's child-spec int list equals its
    inline int list, and the animation float list [(1.0, 0.0)] has the bytes of an animation
    int list [(1.0, 0)] (control 2). Dialog styles 0 and 1 have equal int lists."""
    d = {"machine_time": 3, "controls": [], "styles": []}
    animate(W.append_control(d, "Main", "", 1, inline("S")))
    animate(W.append_control(d, "A", "Main", 2, inline("S2")))
    c = W.append_control(d, "B", "Main", 3, inline("S3", 9))
    c["anim_int32s"] = [{"time": 1.0, "value": 0}]
    c["anim_floats"] = [{"time": 1.0, "value": 0.0}]
    c["anim_descriptions"] = [{"name": "i", "type": 0, "node_range": (0, 1)},
                              {"name": "f", "type": 1, "node_range": (0, 1)}]
    d["styles"] = [{"name": "X", "int32s": [{"key": 4, "value": 1}]},
                   {"name": "Y", "int32s": [{"key": 4, "value": 1}]}]
    W.derive_ui_keys(d)
    return d


def pair(data: bytes, at: int, build: str) -> tuple[int, int]:
    return struct.unpack_from(S.ORDER[build] + "II", data, at)


def control_at(i: int) -> int:
    return BODY + 0xB8 + i * CONTROL


def style_list(data, i, name, build):
    return pair(data, control_at(i) + INLINE + STYLE_LISTS[name] + 4, build)


def anim_list(data, i, name, build):
    return pair(data, control_at(i) + ANIM[name] + 4, build)


def child_spec_ints(data, i, build):
    spec = BODY + style_list(data, i, "child_specs", build)[0]
    return pair(data, spec + 0x28 + 4, build)


def dialog_style_ints(data, s, build):
    styles = BODY + pair(data, BODY + 0x28 + 4, build)[0]
    return pair(data, styles + s * 0xA0 + 0x30 + 4, build)


def tiling(problems):
    return [p for p in problems if "tiling" in p]


# --- writer: new dialogs ------------------------------------------------------------------------

class NewDialogSharing(unittest.TestCase):
    def test_same_field_shares(self):
        for build in BUILDS:
            data = W.write_ui(repeat_dialog(), build)
            for name in STYLE_LISTS:
                self.assertEqual(style_list(data, 1, name, build), style_list(data, 0, name, build), (build, name))
            for name in ("anim_int32s", "anim_floats", "anim_vector2s"):
                self.assertEqual(anim_list(data, 1, name, build), anim_list(data, 0, name, build), (build, name))
            self.assertEqual(anim_list(data, 2, "anim_floats", build), anim_list(data, 0, "anim_floats", build))
            # control 2's inline int list (value 9) differs: stored
            self.assertNotEqual(style_list(data, 2, "int32s", build), style_list(data, 0, "int32s", build))

    def test_never_across_fields(self):
        for build in BUILDS:
            for texts in W.TEXT_SHARING:
                data = W.write_ui(repeat_dialog(), build, texts, share_lists="all")
                # equal bytes, other field: the child spec's int list and the inline int list
                inline_ints = style_list(data, 0, "int32s", build)
                spec_ints = child_spec_ints(data, 0, build)
                self.assertEqual(data[BODY + spec_ints[0]:BODY + spec_ints[0] + spec_ints[1]],
                                 data[BODY + inline_ints[0]:BODY + inline_ints[0] + inline_ints[1]])
                self.assertNotEqual(spec_ints, inline_ints)
                # equal bytes: control 2's int list [(1.0, 0)] and the float list [(1.0, 0.0)]
                ints, floats = anim_list(data, 2, "anim_int32s", build), anim_list(data, 2, "anim_floats", build)
                self.assertEqual(data[BODY + ints[0]:BODY + ints[0] + 8], data[BODY + floats[0]:BODY + floats[0] + 8])
                self.assertNotEqual(ints, floats)
                # a dialog style never shares with an inline style
                self.assertNotEqual(dialog_style_ints(data, 0, build)[0], style_list(data, 0, "int32s", build)[0])

    def test_texts_decide_byte_equality(self):
        """Description and binding lists hold texts that the proven text scope does not share, so
        their blocks differ unless share_texts is "all"."""
        for build in BUILDS:
            proven = W.write_ui(repeat_dialog(), build)
            for name in ("anim_descriptions", "anim_bindings"):
                self.assertNotEqual(anim_list(proven, 1, name, build), anim_list(proven, 0, name, build))
            all_ = W.write_ui(repeat_dialog(), build, "all")
            for name in ("anim_descriptions", "anim_bindings"):
                self.assertEqual(anim_list(all_, 1, name, build), anim_list(all_, 0, name, build), name)
            self.assertNotEqual(anim_list(all_, 2, "anim_descriptions", build),
                                anim_list(all_, 0, "anim_descriptions", build))

    def test_dialog_styles_behind_option(self):
        for build in BUILDS:
            proven = W.write_ui(repeat_dialog(), build)
            self.assertNotEqual(dialog_style_ints(proven, 1, build), dialog_style_ints(proven, 0, build))
            all_ = W.write_ui(repeat_dialog(), build, share_lists="all")
            self.assertEqual(dialog_style_ints(all_, 1, build), dialog_style_ints(all_, 0, build))
            self.assertEqual(len(all_), len(proven) - 22)  # the last block: 6 pad + 16

    def test_none_and_scopes(self):
        for build in BUILDS:
            none = W.write_ui(repeat_dialog(), build, share_lists="none")
            self.assertEqual(W.list_layout(repeat_dialog(), build, "none").count(-1),
                             len(W.list_layout(repeat_dialog(), build, "none")))
            self.assertNotEqual(style_list(none, 1, "int32s", build), style_list(none, 0, "int32s", build))
            sizes = [len(W.write_ui(repeat_dialog(), build, share_lists=s)) for s in ("none", "proven", "all")]
            self.assertTrue(sizes[0] > sizes[1] > sizes[2], sizes)
        with self.assertRaises(FormatError):
            W.write_ui(repeat_dialog(), "switch", share_lists="some")

    def test_first_occurrence_in_walk_order(self):
        layout = W.list_layout(repeat_dialog(), "switch")
        # walk: controls; control 0: strings, ints, floats, child specs, spec ints, 5 anim lists;
        # control 1: 4 inline lists shared, anim ints, floats, vectors shared, descriptions and
        # bindings stored; control 2: strings shared, ints (9) stored, floats and child specs
        # shared, anim ints stored, floats shared, descriptions stored; styles, style 0 ints,
        # style 1 ints (dialog level: not shared by default)
        expected = [-1,
                    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                    1, 2, 3, 4, 6, 7, 8, -1, -1,
                    1, -1, 3, 4, -1, 7, -1,
                    -1, -1, -1]
        self.assertEqual(layout, expected)
        self.assertEqual(U.list_layout, W.list_layout)
        self.assertEqual(U.LIST_SHARING, ("proven", "all", "none"))


# --- reader and round trips ---------------------------------------------------------------------

class RoundTrip(unittest.TestCase):
    def test_round_trip_every_scope(self):
        for build in BUILDS:
            for texts in W.TEXT_SHARING:
                for lists in W.LIST_SHARING:
                    data = W.write_ui(repeat_dialog(), build, texts, share_lists=lists)
                    read = W.parse_ui(data, build)
                    self.assertEqual(read.problems, [], (build, texts, lists))
                    self.assertEqual(W.ui_link_problems(read), [])
                    for t in W.TEXT_SHARING:  # the stored layouts win over the scopes
                        for li in W.LIST_SHARING:
                            self.assertEqual(W.write_ui(read, build, t, share_lists=li), data, (build, texts, lists, t, li))
                    self.assertEqual(L.rebuild_sno(data, "ui", build), data, (build, texts, lists))

    def test_values_read_back_and_independent(self):
        for build in BUILDS:
            read = W.parse_ui(W.write_ui(repeat_dialog(), build, "all", share_lists="all"), build)
            expected = repeat_dialog()
            for got, want in zip(read.root["controls"], expected["controls"]):
                for name in ("anim_int32s", "anim_floats"):
                    self.assertEqual(got[name], want[name])
                self.assertEqual(got["inline_style"]["child_specs"][0]["path_to_child"], "p")
                self.assertEqual(got["inline_style"]["int32s"], want["inline_style"]["int32s"])
            self.assertEqual([d["name"] for d in read.root["controls"][1]["anim_descriptions"]], ["i", "f", "v"])
            self.assertEqual(read.root["styles"][1]["int32s"], [{"key": 4, "value": 1}])
            # a shared list reads as its own copy
            c0, c1 = read.root["controls"][0], read.root["controls"][1]
            self.assertEqual(c0["anim_floats"], c1["anim_floats"])
            self.assertIsNot(c0["anim_floats"], c1["anim_floats"])
            c1["anim_floats"][0]["value"] = 4.0
            self.assertEqual(c0["anim_floats"][0]["value"], 0.0)

    def test_layout_recorded(self):
        read = W.parse_ui(W.write_ui(repeat_dialog(), "x360"), "x360")
        self.assertEqual(read.root["list_layout"], W.list_layout(repeat_dialog(), "switch"))
        # shared lists are not blocks of their own
        shared = sum(1 for e in read.root["list_layout"] if e >= 0)
        none = W.parse_ui(W.write_ui(repeat_dialog(), "x360", share_lists="none"), "x360")
        self.assertGreater(len(none.blocks) - len(read.blocks), shared - 1)

    def test_unchanged_sample_dialogs(self):
        for build in BUILDS:
            for d in (S.sample_dialog(), {"controls": []}):
                data = W.write_ui(d, build)
                self.assertEqual(W.write_ui(W.parse_ui(data, build), build), data)


def twice_file(build: str) -> tuple[bytes, bytes]:
    """By hand: a dialog in which control 1's description list shares control 0's; a copy of
    the description block is appended (the last walked block) and control 1's pair is moved to
    it, so an equal description list is stored twice. Returns (shared file, twice file)."""
    d = {"controls": []}
    for name, parent in (("Main", ""), ("A", "Main")):
        c = W.append_control(d, name, parent, 1)
        c["anim_floats"] = [{"time": 0.0, "value": 0.5}]
        c["anim_descriptions"] = [{"name": "f", "type": 1, "node_range": (0, 1)}]
    W.derive_ui_keys(d)
    shared = W.write_ui(d, build, "all")
    o = S.ORDER[build]
    desc0 = pair(shared, control_at(0) + ANIM["anim_descriptions"] + 4, build)
    assert pair(shared, control_at(1) + ANIM["anim_descriptions"] + 4, build) == desc0
    data = bytearray(shared)
    data += bytes((-(len(data) - BODY)) % 8)
    at = len(data) - BODY
    data += shared[BODY + desc0[0]:BODY + desc0[0] + desc0[1]]
    struct.pack_into(o + "II", data, control_at(1) + ANIM["anim_descriptions"] + 4, at, desc0[1])
    return shared, bytes(data)


class StoredTwice(unittest.TestCase):
    def test_reproduced(self):
        for build in BUILDS:
            shared, data = twice_file(build)
            read = W.parse_ui(data, build)
            self.assertEqual(read.problems, [], build)
            # controls, floats 0, descriptions 0, floats 1 shared, descriptions 1 stored again
            self.assertEqual(read.root["list_layout"], [-1, -1, -1, 1, -1])
            for texts in W.TEXT_SHARING:
                for lists in W.LIST_SHARING:
                    self.assertEqual(W.write_ui(read, build, texts, share_lists=lists), data, (build, texts, lists))
            root = dict(read.root)
            root.pop("list_layout")
            self.assertEqual(W.write_ui(root, build, "all"), shared)  # a new dialog shares it
            self.assertEqual(W.write_ui(read, build, "all", keep_layout=False), shared)

    def test_edited(self):
        for build in BUILDS:
            _, data = twice_file(build)
            # a text edit: the text layout breaks, the list layout (stored twice) is kept
            read = W.parse_ui(data, build)
            read.root["controls"][1]["name"] = "Other"
            out = W.write_ui(read, build, "all")
            again = W.parse_ui(out, build)
            self.assertEqual(again.problems, [])
            self.assertEqual(again.root["list_layout"], [-1, -1, -1, 1, -1])
            self.assertEqual(again.root["controls"][1]["name"], "Other")
            # a shared list edited: it is stored, every other entry is kept
            read = W.parse_ui(data, build)
            read.root["controls"][1]["anim_floats"][0]["value"] = 0.75
            out = W.write_ui(read, build)
            again = W.parse_ui(out, build)
            self.assertEqual(again.problems, [])
            self.assertEqual(again.root["list_layout"], [-1, -1, -1, -1, -1])
            self.assertEqual(again.root["controls"][1]["anim_floats"][0]["value"], 0.75)
            self.assertEqual(again.root["controls"][0]["anim_floats"][0]["value"], 0.5)
            self.assertEqual(W.write_ui(again, build), out)
            # the first of the stored-twice lists edited: both stay stored, layout kept
            read = W.parse_ui(data, build)
            read.root["controls"][0]["anim_descriptions"][0]["key"] = 99
            out = W.write_ui(read, build)
            again = W.parse_ui(out, build)
            self.assertEqual(again.problems, [])
            self.assertEqual(again.root["list_layout"], [-1, -1, -1, 1, -1])
            self.assertEqual(len(out), len(data))
            self.assertEqual([c["anim_descriptions"][0]["key"] for c in again.root["controls"]], [99, W.ui_key("f")])
            # an unchanged parse is the file
            self.assertEqual(W.write_ui(W.parse_ui(data, build), build), data)
            # a layout of another length is dropped: the scopes decide
            d = repeat_dialog()
            d["list_layout"] = [-1]
            self.assertEqual(W.write_ui(d, build), W.write_ui(repeat_dialog(), build))
            # a share pointing at another field with equal values is not followed: control 2's
            # float list [(1.0, 0.0)] pointed at its int list [(1.0, 0)] is stored instead
            d = repeat_dialog()
            layout = W.list_layout(d, build)
            self.assertEqual(layout[24:26], [-1, 7])
            d["list_layout"] = list(layout)
            d["list_layout"][25] = 24
            out = W.write_ui(d, build)
            again = W.parse_ui(out, build)
            self.assertEqual(again.problems, [])
            self.assertEqual(again.root["list_layout"][24:26], [-1, -1])
            self.assertNotEqual(anim_list(out, 2, "anim_floats", build), anim_list(out, 2, "anim_int32s", build))

    def test_append_and_remove_drop_layout(self):
        for build in BUILDS:
            _, data = twice_file(build)
            read = W.parse_ui(data, build)
            added = W.append_control(read, "C", "Main", 1)
            self.assertNotIn("list_layout", read.root)
            added["anim_floats"] = [{"time": 0.0, "value": 0.5}]
            added["anim_descriptions"] = [{"name": "f", "type": 1, "node_range": (0, 1)}]
            W.derive_ui_keys(read)
            out = W.write_ui(read, build, "all")
            again = W.parse_ui(out, build)
            self.assertEqual(again.problems, [])
            self.assertEqual(again.root["list_layout"], [-1, -1, -1, 1, 2, 1, 2])
            self.assertEqual(W.ui_link_problems(again), [])
            read = W.parse_ui(data, build)
            W.remove_control(read, 1)
            self.assertNotIn("list_layout", read.root)
            again = W.parse_ui(W.write_ui(read, build), build)
            self.assertEqual(again.problems, [])
            self.assertEqual(again.root["list_layout"], [-1, -1, -1])


# --- tiling check -------------------------------------------------------------------------------

class Tiling(unittest.TestCase):
    def test_same_field_repeat_accepted(self):
        for build in BUILDS:
            data = W.write_ui(repeat_dialog(), build, "all", share_lists="all")
            self.assertEqual(W.verify_ui(data, build), [])
            read = W.parse_ui(data, build)
            self.assertGreater(sum(1 for e in read.root["list_layout"] if e >= 0), 10)

    def test_cross_field_repeat_reported(self):
        for build in BUILDS:
            data = W.write_ui(repeat_dialog(), build)
            o = S.ORDER[build]
            # control 2's float list pointed at its int list: equal bytes, another field
            bad = bytearray(data)
            struct.pack_into(o + "II", bad, control_at(2) + ANIM["anim_floats"] + 4,
                             *anim_list(data, 2, "anim_int32s", build))
            self.assertTrue(tiling(W.verify_ui(bytes(bad), build)), build)
            # control 1's inline int list pointed at control 0's child-spec int list
            bad = bytearray(data)
            struct.pack_into(o + "II", bad, control_at(1) + INLINE + STYLE_LISTS["int32s"] + 4,
                             *child_spec_ints(data, 0, build))
            self.assertTrue(tiling(W.verify_ui(bytes(bad), build)), build)

    def test_same_field_other_bytes_reported(self):
        for build in BUILDS:
            data = W.write_ui(repeat_dialog(), build)
            o = S.ORDER[build]
            ints = anim_list(data, 0, "anim_int32s", build)  # two elements
            self.assertEqual(anim_list(data, 1, "anim_int32s", build), ints)
            # control 1 keeps the offset but counts one element: a prefix, other bytes
            bad = bytearray(data)
            struct.pack_into(o + "iII", bad, control_at(1) + ANIM["anim_int32s"], 1, ints[0], 8)
            bad_problems = W.verify_ui(bytes(bad), build)
            self.assertTrue(tiling(bad_problems), build)
            # same pair, count disagreeing with the size: reported as a list size problem
            bad = bytearray(data)
            struct.pack_into(o + "i", bad, control_at(1) + ANIM["anim_int32s"], 3)
            self.assertTrue(any("elements of 8" in p for p in W.verify_ui(bytes(bad), build)), build)

    def test_repeat_of_shared_pair_points_at_stored(self):
        """Three equal lists: the third repeats the pair of the second (itself a repeat); the
        layout points both at the stored first."""
        d = {"controls": []}
        for name, parent in (("Main", ""), ("A", "Main"), ("B", "Main")):
            W.append_control(d, name, parent, 1, {"int32s": [{"key": 1, "value": 2}]})
        for build in BUILDS:
            read = W.parse_ui(W.write_ui(d, build), build)
            self.assertEqual(read.root["list_layout"], [-1, -1, 1, 1])
            self.assertEqual(read.problems, [])


if __name__ == "__main__":
    unittest.main()
