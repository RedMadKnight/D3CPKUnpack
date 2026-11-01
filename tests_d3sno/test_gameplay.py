import struct
import unittest

import builders as B
from d3sno import FormatError
from d3sno import anims, scripts, skills
from d3sno.sno import hash33_ci

BUILDS = ("ps3", "switch")


class AnimSets(unittest.TestCase):
    def test_maps(self):
        for build, slots in (("ps3", 19), ("switch", 28)):
            data = B.build_ans(build, [(2, 5, 700)], {slots - 1: [(2, 6, 800)]}, parent=33)
            read = anims.parse_anim_set(data)
            self.assertEqual(read.parent, 33)
            self.assertEqual(len(read.weapons), slots)
            self.assertEqual(read.animation(5), 700)
            self.assertEqual(read.animation(6, slots - 1), 800)
            self.assertIsNone(read.animation(9))
            self.assertEqual(read.warnings, [])

    def test_other_record_type_warns(self):
        read = anims.parse_anim_set(B.build_ans("switch", [(0, 5, 700)]))
        self.assertTrue(read.warnings)

    def test_bad_map(self):
        data = bytearray(B.build_ans("switch", [(2, 5, 700)]))
        struct.pack_into("<I", data, 0x24, 7)
        with self.assertRaises(FormatError):
            anims.parse_anim_set(bytes(data))


class SkillKits(unittest.TestCase):
    def test_entries(self):
        for build in BUILDS:
            read = skills.parse_skill_kit(B.build_skl(
                build, [(100, 1, 10)], [(200, 1, 2, 5, 1, 11, 12, 13, 14, 15)],
            ))
            self.assertEqual(read.traits[0], skills.Trait(100, 1, 10))
            active = read.actives[0]
            self.assertEqual((active.power, active.new_category, active.granted), (200, 2, 1))
            self.assertEqual(active.rune_unlock_levels, (11, 12, 13, 14, 15))

    def test_entry_sizes(self):
        for build, at in (("ps3", 0x2C), ("switch", 0x3C)):
            data = bytearray(B.build_skl(build, [(1, 1, 1)], [tuple(range(10))]))
            order = ">" if build == "ps3" else "<"
            size = struct.unpack_from(order + "I", data, at)[0]
            struct.pack_into(order + "I", data, at, size - 1)
            with self.assertRaises(FormatError):
                skills.parse_skill_kit(bytes(data))


class Small(unittest.TestCase):
    def test_timed_event(self):
        self.assertEqual(skills.parse_timed_event(B.build_words("switch", [600])).duration_ticks, 600)

    def test_timed_event_short(self):
        with self.assertRaises(FormatError):
            skills.parse_timed_event(B.sno_head(1, "<"))

    def test_lore(self):
        read = skills.parse_lore(B.build_words("ps3", [1, 2, 3, 4, 5, 6]))
        self.assertEqual((read.flags, read.category, read.xp_level), (1, 2, 3))
        self.assertEqual((read.xp_tier, read.conversation, read.signature_label_hash), (4, 5, 6))

    def test_lore_short(self):
        with self.assertRaises(FormatError):
            skills.parse_lore(B.build_words("ps3", [1, 2]))


def word(value: float, order: str) -> int:
    return struct.unpack(order + "I", struct.pack(order + "f", value))[0]


class Scripts(unittest.TestCase):
    def read(self, build, entries, **kw):
        return scripts.parse_script(B.build_scr(build, entries, B.lua_chunk(build), **kw))

    def test_fixed_part(self):
        for build in BUILDS:
            read = self.read(build, [], uid=77, quest=-1)
            self.assertEqual(read.chunk, B.lua_chunk(build))
            self.assertEqual((read.uid, read.quest_handle), (77, -1))

    def test_entry(self):
        for build in BUILDS:
            with self.subTest(build):
                entry = {"event": 12, "unique": 3, "handler": "OnThing", "flags": 9}
                message = self.read(build, [entry]).messages[0]
                self.assertEqual((message.event_id, message.unique_id), (12, 3))
                self.assertEqual(message.handler, "OnThing")
                self.assertEqual((message.listener_flags, message.event_count), (9, 7))
                self.assertEqual(len(message.filters), 4)

    def test_filter_fields_on_both_builds(self):
        for build in BUILDS:
            filters = [(1, 42, 0, None), (27, 900, 0, None)]
            read = self.read(build, [{"event": 1, "filters": filters}]).messages[0].filters
            self.assertEqual((read[0].kind, read[0].value), (1, 42))
            self.assertEqual((read[1].value, read[1].group), (900, 29))

    def test_float_kinds(self):
        for build in BUILDS:
            order = ">" if build == "ps3" else "<"
            filters = [
                (2, word(1.5, order), 0, None),
                (3, word(1.0, order), word(2.0, order), None),
                (22, word(3.0, order), word(4.0, order), None),
            ]
            read = self.read(build, [{"event": 1, "filters": filters}]).messages[0].filters
            self.assertEqual(read[0].value, 1.5)
            self.assertEqual(read[1].value, (1.0, 2.0))
            self.assertEqual(read[2].value, (3.0, 4.0))

    def test_no_value_kinds(self):
        for kind in (0, 12, 20, 25):
            filters = [(kind, 5, 0, None)]
            read = self.read("switch", [{"event": 1, "filters": filters}]).messages[0].filters
            self.assertIsNone(read[0].value)

    def test_string_hash(self):
        filters = [(7, hash33_ci("Name"), 0, "Name")]
        read = self.read("switch", [{"event": 1, "filters": filters}])
        self.assertEqual(read.messages[0].filters[0].text, "Name")
        self.assertEqual(read.warnings, [])

    def test_string_hash_mismatch(self):
        filters = [(15, 1, 0, "Name")]
        data = B.build_scr("ps3", [{"event": 1, "filters": filters}])
        self.assertTrue(scripts.parse_script(data).warnings)
        self.assertTrue(scripts.parse_script(data, strict=True).warnings)

    def test_handle_groups(self):
        self.assertEqual(scripts.HANDLE_GROUPS[30], 64)
        self.assertEqual(len(scripts.HANDLE_GROUPS), 10)

    def test_count_must_match_the_pair(self):
        with self.assertRaises(FormatError):
            scripts.parse_script(B.build_scr("switch", [{"event": 1}], count=2))

    def test_map_out_of_bounds(self):
        data = bytearray(B.build_scr("switch", [{"event": 1}]))
        struct.pack_into("<I", data, 0x30, 0xFFFFFF)
        with self.assertRaises(FormatError):
            scripts.parse_script(bytes(data))

    def test_fixed_part_too_short(self):
        with self.assertRaises(FormatError):
            scripts.parse_script(B.build_scr("switch")[:0x40])

    def test_version_picks_nothing_but_can_be_strict(self):
        data = B.build_scr("switch", version=1)
        scripts.parse_script(data)
        with self.assertRaises(FormatError):
            scripts.parse_script(data, strict=True)

    def test_event_map(self):
        self.assertEqual(scripts.switch_event_id(7, "ps3"), 7)
        self.assertEqual(scripts.switch_event_id(8, "ps3"), 9)
        self.assertEqual(scripts.switch_event_id(51, "ps3"), 55)
        self.assertIsNone(scripts.switch_event_id(15, "ps3"))
        self.assertEqual(scripts.switch_event_id(15, "switch"), 15)
