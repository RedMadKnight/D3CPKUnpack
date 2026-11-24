"""Corrections 20 (`33`): V20-1 string lists shared across fields, V20-2 tolerated last seek
entry. Synthetic data only."""

import _libpath  # noqa: F401  (python -I: the library folder on sys.path)
import unittest

import d3sno
from d3sno import sound_write as SW
from d3sno import ui_write as W
from test_stage15_1_content import fsb_with_table, packets
from test_stage15_2_ui import BODY, BUILDS, INLINE, STYLE_LISTS, control_at, pair, style_list, tiling

STRINGS = [{"key": 1, "text": "t"}, {"key": 2, "text": "u"}]


def child_spec_strings(data, i, build):
    spec = BODY + style_list(data, i, "child_specs", build)[0]
    return pair(data, spec + 0x10 + 4, build)


def dialog_style_strings(data, s, build):
    styles = BODY + pair(data, BODY + 0x28 + 4, build)[0]
    return pair(data, styles + s * 0xA0 + 0x18 + 4, build)


def cross_dialog() -> dict:
    """Control 0's inline string list equals control 1's child-spec string list and dialog style
    0's string list; control 1's inline int list equals control 0's child-spec int list."""
    d = {"controls": []}
    W.append_control(d, "Main", "", 1, {"name": "S", "strings": [dict(s) for s in STRINGS],
                                         "child_specs": [{"path_to_child": "p",
                                                          "int32s": [{"key": 7, "value": 1}]}]})
    W.append_control(d, "A", "Main", 2, {"name": "S2", "int32s": [{"key": 7, "value": 1}],
                                          "child_specs": [{"path_to_child": "q",
                                                           "strings": [dict(s) for s in STRINGS]}]})
    d["styles"] = [{"name": "X", "strings": [dict(s) for s in STRINGS]}]
    W.derive_ui_keys(d)
    return d


class Marker(unittest.TestCase):
    def test_version(self):
        self.assertGreaterEqual(tuple(int(x) for x in d3sno.__version__.split(".")), (0, 15, 3))
        self.assertGreaterEqual(d3sno.STAGE, 15)


class StringListsAcrossFields(unittest.TestCase):
    def test_child_spec_string_list_shares_inline_block(self):
        for build in BUILDS:
            data = W.write_ui(cross_dialog(), build)
            self.assertEqual(child_spec_strings(data, 1, build), style_list(data, 0, "strings", build), build)
            self.assertEqual(W.verify_ui(data, build), [])

    def test_dialog_style_string_list_shares_by_default(self):
        """With the texts shared too, the dialog style's string list has the inline list's bytes
        and points at its block, without share_lists="all"."""
        for build in BUILDS:
            data = W.write_ui(cross_dialog(), build, "all")
            self.assertEqual(dialog_style_strings(data, 0, build), style_list(data, 0, "strings", build), build)
            self.assertEqual(W.verify_ui(data, build), [])
            none = W.write_ui(cross_dialog(), build, "all", share_lists="none")
            self.assertNotEqual(dialog_style_strings(none, 0, build), style_list(none, 0, "strings", build))

    def test_other_lists_stay_within_their_field(self):
        for build in BUILDS:
            data = W.write_ui(cross_dialog(), build, "all", share_lists="all")
            spec_ints = pair(data, BODY + style_list(data, 0, "child_specs", build)[0] + 0x28 + 4, build)
            inline_ints = style_list(data, 1, "int32s", build)
            self.assertEqual(data[BODY + spec_ints[0]:BODY + spec_ints[0] + spec_ints[1]],
                             data[BODY + inline_ints[0]:BODY + inline_ints[0] + inline_ints[1]])
            self.assertNotEqual(spec_ints, inline_ints)

    def test_parser_layout_and_round_trip(self):
        for build in BUILDS:
            for texts in W.TEXT_SHARING:
                data = W.write_ui(cross_dialog(), build, texts)
                read = W.parse_ui(data, build)
                self.assertEqual(tiling(read.problems), [], (build, texts))
                self.assertEqual(W.write_ui(read.root, build), data, (build, texts))
                self.assertEqual(read.root["controls"][1]["inline_style"]["child_specs"][0]["strings"],
                                 read.root["controls"][0]["inline_style"]["strings"])

    def test_unchanged_layout_kept_when_stored_apart(self):
        """A file storing equal string lists in two fields apart (written without sharing) is
        reproduced, not merged."""
        for build in BUILDS:
            data = W.write_ui(cross_dialog(), build, share_lists="none")
            self.assertNotEqual(child_spec_strings(data, 1, build), style_list(data, 0, "strings", build))
            self.assertEqual(W.write_ui(W.parse_ui(data, build).root, build), data, build)

    def test_cross_field_string_repeat_accepted_by_tiling(self):
        """A stored file whose later child-spec string pair repeats the inline string pair."""
        for build in BUILDS:
            shared = W.write_ui(cross_dialog(), build)
            read = W.parse_ui(shared, build)
            # list 0 is the controls list, list 1 control 0's inline strings: the only share
            self.assertEqual([e for e in read.root["list_layout"] if e >= 0], [1], build)
            self.assertEqual(read.problems, [])


class ToleratedLastSeekEntry(unittest.TestCase):
    def test_last_entry_at_a_packet_start_out_of_order(self):
        offsets = SW.vorbis_packet_offsets(packets())
        data, _ = fsb_with_table([(44000, offsets[44]), (88000, offsets[88]), (1, offsets[3])])
        self.assertEqual(SW.vorbis_seek_problems(data), [])
        strict = SW.vorbis_seek_problems(data, allow_unmatched_last=False)
        self.assertEqual(len(strict), 1)
        self.assertIn("seek entry 2", strict[0])

    def test_only_the_last_entry_is_exempt(self):
        offsets = SW.vorbis_packet_offsets(packets())
        data, _ = fsb_with_table([(44000, offsets[44]), (1, offsets[3]), (88000, offsets[88])])
        problems = SW.vorbis_seek_problems(data)
        self.assertEqual(len(problems), 1)
        self.assertIn("seek entry 1", problems[0])
        self.assertIn("V20-2", SW.vorbis_seek_problems.__doc__)


if __name__ == "__main__":
    unittest.main()
