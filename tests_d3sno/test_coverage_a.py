"""Stage 6, part A: tag and enum names, versions, header words, index and script checks."""

import struct
import unittest

import builders as B
from d3sno import archive as A
from d3sno import enums as E
from d3sno import index as I
from d3sno import power as P
from d3sno import scripts
from d3sno import sno as S
from d3sno import tags as T
from d3sno.sno import hash33, hash33_ci


class TagNames(unittest.TestCase):
    def test_table_sizes(self):
        self.assertEqual(len(T.tag_table()), 2749)
        self.assertEqual(len(T.tag_types()), 143)

    def test_lookup(self):
        info = T.tag_info("power_general", 0x40436)
        self.assertEqual((info.name, info.record_type), ("Scaled Animation Timing Min", 1))
        self.assertEqual(T.tag_name("power_tuning", 0x40100), "Projectile Actor")
        self.assertEqual(T.tag_info("power_tuning", 0x40100).record_kind, "sno")
        self.assertIsNone(T.tag_name(None, 0x40436))  # the effect map has no table
        self.assertIsNone(T.tag_name("power_general", 0x50281))  # ps3 only, unnamed

    def test_table_follows_the_field(self):
        self.assertIsNone(T.POWER_MAP_TABLES["Effect"])
        self.assertEqual(T.POWER_MAP_TABLES["PvpGeneral"], "power_general")
        self.assertEqual(T.POWER_MAP_TABLES["PvpContact3"], "power_tuning")
        self.assertEqual(len(T.POWER_MAP_TABLES), 11)

    def test_player_class_both_forms(self):
        for index, name in enumerate(
            ("DemonHunter", "Barbarian", "Wizard", "Witchdoctor", "Monk", "Crusader", "Necromancer"),
        ):
            self.assertEqual(T.value_name("actor", 0x1009E, index), name)
            hashed = E.signed32(hash33_ci(name))
            self.assertEqual(T.value_name("actor", 0x1009E, hashed), f"{name} (name hash)")
        self.assertEqual(T.value_name("actor", 0x1009E, -1), "- None -")

    def test_component_look(self):
        self.assertEqual(E.signed32(hash33("Unique01")), -2104614664)
        self.assertEqual(T.value_name("actor", 0x10401, -2104614664), "Unique01")
        self.assertEqual(hash33("Q"), 81)

    def test_check_map(self):
        good = P.TagMap("General", None, [
            P.TagRecord(1, 0x40436, 1.0, 0), P.TagRecord(0, 0x40200, 3, 8),
        ])
        self.assertEqual(T.check_tag_map(good, "power_general", "switch"), [])
        wrong = P.TagMap("General", None, [P.TagRecord(0, 0x40436, 1, 0)])
        self.assertEqual(len(T.check_tag_map(wrong, "power_general", "switch")), 1)
        unknown = P.TagMap("General", None, [P.TagRecord(4, 0x50281, P.Formula("1", b""), 0)])
        self.assertEqual(len(T.check_tag_map(unknown, "power_general", "switch")), 1)
        self.assertEqual(T.check_tag_map(unknown, "power_general", "ps3"), [])

    def test_ps3_retyped_tags_trust_the_stored_type(self):
        record = P.TagMap("General", None, [P.TagRecord(0, 0x50210, 1, 0)])
        self.assertEqual(T.check_tag_map(record, "power_general", "ps3"), [])
        self.assertEqual(len(T.check_tag_map(record, "power_general", "switch")), 1)

    def test_effect_map_must_be_empty(self):
        tag_map = P.TagMap("Effect", None, [P.TagRecord(0, 0x40200, 1, 0)])
        self.assertEqual(len(T.check_tag_map(tag_map, None, "switch")), 1)
        self.assertEqual(T.check_tag_map(P.TagMap("Effect", None, []), None, "switch"), [])

    def test_power_file(self):
        data = B.build_power("switch", general=[(1, 0x40436, 0.5), (0, 0x40200, 2)])
        power = P.parse_power(data)
        self.assertEqual(T.power_map_problems(power), [])
        named = T.name_records(power.general, "power_general")
        self.assertEqual(named[0].name, "Scaled Animation Timing Min")
        self.assertEqual(named[0].type_name, "TAGVARIANTTYPE_FLOAT")
        self.assertEqual(named[1].xml_name, "TAG_POWER_ANIMATION_TAG")
        self.assertEqual(power.script_reference, 0)

    def test_power_bits(self):
        power = P.parse_power(B.build_power("switch", power_bits=0b11000))
        self.assertEqual(power.power_bit_constants(), [
            "POWERBIT_CONTACTFRAME_ENABLED_SECOND", "POWERBIT_CONTACTFRAME_ENABLED_THIRD",
        ])
        names, unknown = E.bit_names("power_bit", 1 << 31)
        self.assertEqual((names, unknown), ([], [31]))


class CoreEnums(unittest.TestCase):
    def test_named(self):
        self.assertEqual(E.enum_name("tag_record_type", 7), "TAGVARIANTTYPE_STARTLOC_NAME")
        self.assertEqual(E.enum_name("game_balance_type", 23), "Item Enhancements")
        self.assertEqual(E.enum_name("formula_opcode", 15), E.core_enums()["formula_opcode"][15])
        self.assertIsNone(E.enum_name("no_such_enum", 1))
        self.assertIsNone(E.enum_name("lore_category", 99))

    def test_unsigned_value_finds_signed_name(self):
        self.assertEqual(E.enum_name("skill_category", 0xFFFFFFFF), "SKILLCAT_INVALID")

    def test_groups(self):
        self.assertEqual(I.group_name(29), "POWER_SNO")
        self.assertEqual(I.TocEntry(35, 1, "x").group_name, "SCRIPT_SNO")


class Versions(unittest.TestCase):
    def test_every_extension_has_one_version_per_build(self):
        for extension, (ps3, switch) in S.VERSIONS.items():
            for build, version in (("ps3", ps3), ("switch", switch)):
                if version is not None:
                    self.assertTrue(S.check_version(extension, version, build))
                    self.assertFalse(S.check_version(extension, version + 1, build))

    def test_missing_build(self):
        missing = [(e, v) for e, v in S.VERSIONS.items() if None in v]
        for extension, (ps3, switch) in missing:
            build = "ps3" if ps3 is None else "switch"
            self.assertFalse(S.check_version(extension, 1, build))
            with self.assertRaises(S.FormatError):
                S.check_version(extension, 1, build, strict=True)

    def test_header_words(self):
        header = S.parse_header(B.sno_head(1, "<"))
        self.assertEqual(S.header_problems(header), [])
        data = bytearray(B.sno_head(1, ">"))
        struct.pack_into(">I", data, 0x18, 5)
        self.assertEqual(len(S.header_problems(S.parse_header(bytes(data)))), 1)


def toc_switch(entries, last_word):
    counts, offsets = [0] * 70, [0] * 70
    body = bytearray()
    for group, items in entries.items():
        counts[group], offsets[group] = len(items), len(body)
        records, names = bytearray(), bytearray()
        for handle, name in items:
            records += struct.pack("<iii", group, handle, len(names))
            names += name.encode() + b"\0"
        body += records + names
    return struct.pack("<70i70i70iI", *counts, *offsets, *counts, last_word) + bytes(body)


class Index(unittest.TestCase):
    def test_switch_last_word_is_the_highest_handle(self):
        toc = I.parse_core_toc(toc_switch({29: [(5, "a"), (900, "b")], 35: [(17, "c")]}, 900))
        self.assertEqual((toc.max_handle, toc.problems()), (900, []))
        for word in (4, 6):
            toc = I.parse_core_toc(toc_switch({29: [(5, "a")]}, word))
            self.assertEqual(len(toc.problems()), 1)
        self.assertEqual(I.SWITCH_TOC_HEADER, 844)

    def test_ps3_name_padding(self):
        clean = struct.pack(">III", 1, 29, 4) + b"name".ljust(128, b"\0")
        self.assertEqual(I.parse_core_toc(clean, "ps3").problems(), [])
        dirty = bytearray(clean)
        dirty[12 + 100] = 1
        toc = I.parse_core_toc(bytes(dirty), "ps3")
        self.assertEqual(toc.entries[0].name, "name")
        self.assertEqual(toc.name_padding_problems, 1)
        self.assertEqual(len(toc.problems()), 1)


class ScriptCoverage(unittest.TestCase):
    def read(self, build, entries):
        return scripts.parse_script(B.build_scr(build, entries, B.lua_chunk(build)))

    def test_unused_filter_slot(self):
        for build in ("ps3", "switch"):
            filters = [(0xFFFFFFFF, 0, 0, None)]
            read = self.read(build, [{"event": 9, "filters": filters}])
            first = read.messages[0].filters[0]
            self.assertEqual((first.kind, first.value, first.text), (-1, None, None))
            self.assertEqual(first.kind_name, "(unused slot)")

    def test_event_names(self):
        read = self.read("switch", [{"event": 9}])
        self.assertEqual(read.messages[0].event_name("switch"), "Conversation Started")
        ps3 = self.read("ps3", [{"event": 8}, {"event": 15}])
        self.assertEqual(ps3.messages[0].event_name("ps3"), "Conversation Started")
        self.assertIsNone(ps3.messages[1].event_name("ps3"))  # no switch id known


class ArchiveV6(unittest.TestCase):
    def test_word_3c_defaults_to_zero(self):
        data = A.write_archive({"a.bin": b"x" * 100}, version=6)
        self.assertEqual(A.read_header(data).word_3c, 0)
        data = A.write_archive({"a.bin": b"x" * 100}, version=6, word_3c=0x10000)
        self.assertEqual(A.read_header(data).word_3c, 0x10000)


if __name__ == "__main__":
    unittest.main()
