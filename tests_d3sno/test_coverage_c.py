"""Stage 6, part C: record enums and flags, display times, objectives, presentation, achu, FSB5."""

import _libpath  # noqa: F401  (python -I: the library folder on sys.path)
import struct
import unittest

import builders as B
from d3sno import FormatError
from d3sno import enums as E
from d3sno import presentation as V
from d3sno import records as R


def record(kind, build="switch", **values):
    return R.Record(kind, 0, b"", values, build=build)


class RecordEnums(unittest.TestCase):
    def test_parsed_record_knows_its_build(self):
        for build in ("ps3", "switch"):
            data = B.build_record(build, "monster", {"flags": 0b100101})
            monster = R.parse_record_file(data, "mon").record
            self.assertEqual(monster.build, build)
            self.assertEqual(monster.flags("flags").names, [
                "MONSTERDEF_IS_UNIQUE_BIT", "MONSTERDEF_IS_ELITE_BIT", "MONSTERDEF_IS_BOSS_BIT",
            ])

    def test_enum_value(self):
        row = record("treasure_class", choose_kind=-1).enum("choose_kind")
        self.assertEqual((row.constant, row.label, row.status), ("TCC_NONE", "none", "PROVEN"))
        self.assertIsNone(record("treasure_class", choose_kind=9).enum("choose_kind"))
        self.assertIsNone(record("treasure_class").enum("choose_kind"))

    def test_class_mask_ignores_bits_above_the_class_count(self):
        value = 0b1100001 | 0x18F780
        switch = record("list_entry", "switch", class_flags=value).flags("class_flags")
        self.assertEqual(switch.names, [
            "PLAYER_CLASS_DEMONHUNTER", "PLAYER_CLASS_CRUSADER", "PLAYER_CLASS_NECROMANCER",
        ])
        self.assertEqual(switch.ignored_bits, [7, 8, 9, 10, 12, 13, 14, 15, 19, 20])
        ps3 = record("list_entry", "ps3", class_flags=value).flags("class_flags")
        self.assertEqual(ps3.names, ["PLAYER_CLASS_DEMONHUNTER"])
        self.assertEqual(ps3.ignored_bits[:2], [5, 6])

    def test_monster_flags_holding_a_number(self):
        for number in (10000, 9988):
            flags = record("monster", flags=number).flags("flags")
            self.assertEqual((flags.plain_number, flags.names), (number, []))
        self.assertIsNone(record("monster", flags=3).flags("flags").plain_number)

    def test_unknown_and_open_bits(self):
        flags = record("objective", flags=(1 << 1) | (1 << 30)).flags("flags")
        self.assertEqual((flags.open_bits, flags.unknown_bits), ([1], [30]))

    def test_named(self):
        named = record("treasure_class", choose_kind=0).named()
        self.assertEqual(named["choose_kind"], "TCC_ONE_OF")
        named = record("monster", flags=1, race=99999).named()
        self.assertEqual(named["flags"].names, ["MONSTERDEF_IS_UNIQUE_BIT"])
        self.assertIsNone(named["race"])


class DisplayTimes(unittest.TestCase):
    def test_locales(self):
        locales = R.display_time_locales()
        self.assertEqual(len(locales), 20)
        self.assertEqual(locales[2].code, "enUS")
        self.assertEqual(locales[16].constant, "XLOCALE_POLISH")

    def test_word(self):
        time = R.decode_display_time(B.display_time(2, 7, 2, 150))
        self.assertEqual((time.seconds, time.locale_code), (2.5, "enUS"))
        self.assertTrue(time.any_class("switch"))
        self.assertFalse(time.any_class("ps3"))
        self.assertTrue(time.any_gender)
        self.assertTrue(time.carries_a_time)
        none = R.decode_display_time(B.display_time(19, 5, 0, 60))
        self.assertIsNone(none.locale_code)
        self.assertTrue(none.any_class("ps3"))
        for slot in (0, 1, 17):
            self.assertFalse(R.decode_display_time(B.display_time(slot, 0, 0, 1)).carries_a_time)
        self.assertEqual(R.TICKS_PER_SECOND, 60)


class Objectives(unittest.TestCase):
    def test_table(self):
        params = R.objective_params()
        self.assertEqual(len(params), 20)
        kill = params[("objective", 2)]
        self.assertEqual((kill.kind_name, kill.target_1_group), ("Kill Monster", 1))
        self.assertEqual(params[("objective", 7)].text_1, "trigger name")
        self.assertIsNone(params[("objective", 7)].target_1_group)
        self.assertEqual(params[("objective", 1)].row_param_1, "item row id")

    def test_record(self):
        objective = record("objective", objective_kind=10, param_1=0, param_2=3)
        meaning = R.objective_meaning(objective)
        self.assertEqual((meaning.kind_name, meaning.target_1_group), ("Timed Event Expired", 55))
        self.assertEqual(R.objective_count(objective), 3)
        failure = record("failure_condition", failure_kind=7)
        self.assertEqual(R.objective_meaning(failure).text_1, "event name")
        self.assertIsNone(R.objective_meaning(record("objective", objective_kind=99)))


def path_at(img, at, count, nodes, node_size=12, padding=0):
    raw = b"".join(n.ljust(node_size, b"\0") for n in nodes)
    img.put(at, "iffiiffI", count, 0.25, 0.75, 2, 10, 0.0, 1.0, padding)
    img.put(at + 32, "II", *img.append(raw))


class PresentationPaths(unittest.TestCase):
    def test_table(self):
        fields = V.presentation_fields()
        self.assertEqual(sum(1 for f in fields["lit"] if f.note == "path"), 26)
        self.assertEqual(sum(1 for f in fields["trl"] if f.note == "path"), 4)
        self.assertEqual(sum(1 for f in fields["rop"] if f.note == "path"), 19)
        self.assertEqual(len(fields["mus"]), 15)

    def test_path_header_and_nodes(self):
        img = B.ps3_file(0x10 + 0x118)
        node = struct.pack(">3f", 1.0, 2.0, 0.5)
        colour = bytes((1, 2, 3, 4, 5, 6, 7, 8)) + struct.pack(">f", 0.25)
        path_at(img, 0x10 + 0x28, 1, [colour])
        for i in range(1, 4):
            path_at(img, 0x10 + 0x28 + i * 40, 2, [node, node])
        img.raw(0x10 + 0xD0, B.material_block(img, "ps3"))
        trail = V.parse_trail(img.done())
        first = trail.named_paths["color_over_time"]
        self.assertEqual((first.node_count, first.loop_start, first.loop_end), (1, 0.25, 0.75))
        self.assertEqual((first.distribution, first.interpolation, first.padding), (2, (10, 0.0, 1.0), 0))
        self.assertEqual((first.distribution_name, first.scalar_function_name), ("mid", "random"))
        self.assertEqual(first.decoded()[0], V.PathNode((1, 2, 3, 4), (5, 6, 7, 8), 0.25))
        self.assertEqual(trail.named_paths["alpha"].decoded()[1], V.PathNode(1.0, 2.0, 0.5))
        self.assertEqual(V.path_problems(trail.named_paths), [])

    def test_path_problems(self):
        img = B.ps3_file(0x10 + 0x118)
        path_at(img, 0x10 + 0x28, 3, [bytes(12)])
        path_at(img, 0x10 + 0x50, 0, [], padding=1)
        path_at(img, 0x10 + 0x78, 0, [bytes(12)])
        img.raw(0x10 + 0xD0, B.material_block(img, "ps3"))
        trail = V.parse_trail(img.done())
        self.assertEqual(len(V.path_problems(trail.named_paths)), 3)

    def test_rope_vector_path(self):
        img = B.ps3_file(0x3A4)
        img.raw(0x40, B.material_block(img, "ps3"))
        for i in range(14):
            size = 28 if i == 12 else 12
            path_at(img, 0x9C + i * 40, 1, [struct.pack(f">{size // 4}f", *range(size // 4))], size)
        img.put(0x9C + 14 * 40, "2f", 1.5, 2.5)
        for i in range(5):
            path_at(img, 0x9C + 14 * 40 + 8 + i * 40, 0, [])
        rope = V.parse_rope(img.done())
        jitter = rope.named_paths["jitter_amplitude"]
        self.assertEqual(jitter.node_kind, "vector")
        self.assertEqual(jitter.decoded()[0], V.PathNode((0.0, 1.0, 2.0), (3.0, 4.0, 5.0), 6.0))
        self.assertEqual((rope.jitter_approach_speed, rope.expected_length), (1.5, 2.5))
        self.assertIn("phase_speed_over_time", rope.named_paths)
        self.assertEqual(len(rope.named_paths), 19)

    def test_light_states(self):
        img = B.ps3_file(0x10 + 0x240 + 13 * 40)
        path_at(img, 0x10 + 0x240 + 12 * 40, 1, [bytes(12)])
        light = V.parse_light(img.done())
        self.assertEqual(len(light.paths), 26)
        self.assertEqual(light.paths["state_b.specular_intensity"].node_count, 1)
        self.assertEqual(light.paths["state_a.hotspot"].node_kind, "angle")


def ui_list(img, at, records_raw, count):
    img.put(at, "i", count)
    if records_raw:
        img.put(at + 4, "II", *img.append(records_raw))


class UiRecords(unittest.TestCase):
    def build(self, animation_counts=(2, 0, 0, 0, 0), animation_sizes=(8, 8, 12, 40, 20)):
        img = B.ps3_file(0x10)
        control_at = len(img.b)
        img.b += bytes(272)
        img.put(control_at + 0x38, "i", 6)
        style_at = control_at + 0x40
        child = bytearray(76)
        img.put(style_at + 0x48, "i", 1)
        img.put(style_at + 0x4C, "II", *img.append(bytes(child)))
        ui_list(img, style_at + 0x18, bytes(24 * 2), 2)
        for i, (count, size) in enumerate(zip(animation_counts, animation_sizes)):
            ui_list(img, control_at + 0xB0 + i * 16, bytes(count * size), count)
        return img.done(), control_at

    def test_control(self):
        data, at = self.build()
        control = V.read_ui_control(data, at)
        self.assertEqual(control.type, 6)
        self.assertEqual(control.problems, [])
        self.assertEqual(control.animation_lists[0].element_size, 8)
        self.assertEqual(len(control.style.child_specs), 1)
        self.assertEqual(control.style.element_lists[0].element_size, 24)
        self.assertEqual(len(control.style.child_specs[0].element_lists), 3)

    def test_animation_list_size_checked(self):
        data, at = self.build((1, 0, 0, 0, 0), (12, 8, 12, 40, 20))
        self.assertEqual(len(V.read_ui_control(data, at).problems), 1)

    def test_child_specs_must_be_whole(self):
        img = B.ps3_file(0x10)
        style_at = len(img.b)
        img.b += bytes(112)
        img.put(style_at + 0x48, "i", 2)
        img.put(style_at + 0x4C, "II", *img.append(bytes(76)))
        with self.assertRaises(FormatError):
            V.read_ui_style(img.done(), style_at)


class Achievements(unittest.TestCase):
    def test_walk(self):
        modifier = B.proto(1, 5) + B.proto(3, 2)
        event = B.proto(1, 1) + B.proto(3, modifier)
        criteria = B.proto(1, 77) + B.proto(3, event) + B.proto(9, 1 << 8)
        achievement = B.proto(1, 10) + B.proto(6, (1 << 8) | 1) + B.proto(9, B.proto(1, b"k") + B.proto(2, b"v"))
        data = B.proto(2, achievement) + B.proto(3, criteria)
        decoded, problems = V.decode_achu(data)
        self.assertEqual(problems, [])
        self.assertEqual(decoded["achievements"][0]["id"], [10])
        self.assertEqual(decoded["achievements"][0]["attributes"][0], {"key": [b"k"], "value": [b"v"]})
        start = decoded["criteria"][0]["start_event"][0]
        self.assertEqual(start["modifiers"][0]["condition_id"], [5])
        flags = decoded["achievements"][0]["flags"][0]
        self.assertEqual(V.achu_flag_names("achievement", flags),
                         ["ACHIEVEMENT_FLAG_STATISTIC", "ACHIEVEMENT_FLAG_HIDDEN"])
        self.assertEqual(V.achu_event_name(1), "ACH_EVENT_EQUIP_ITEM_PLAYER")

    def test_unknown_field_and_wire(self):
        decoded, problems = V.decode_achu(B.proto(2, B.proto(42, 1) + B.proto(1, b"x")))
        self.assertEqual(len(problems), 2)
        self.assertEqual(decoded["achievements"][0]["#42"], [1])

    def test_schema_size(self):
        schema = V.achu_schema()
        self.assertEqual(len(schema["fields"]), 38)
        self.assertEqual(len(schema["names"]["event_id"]), 136)
        self.assertEqual(len(schema["names"]["modifier_condition_id"]), 119)


class FsbMarkers(unittest.TestCase):
    def test_markers(self):
        marker = struct.pack("<I", 4410) + b"intro".ljust(256, b"\0")
        data = B.fsb5(15, b"", chunks=[(4, marker + struct.pack("<I", 9) + b"x".ljust(256, b"\0"))])
        markers = V.parse_fsb5(data).sample.markers()
        self.assertEqual(markers, [V.FsbMarker(4410, "intro"), V.FsbMarker(9, "x")])

    def test_marker_size(self):
        data = B.fsb5(15, b"", chunks=[(4, bytes(261))])
        with self.assertRaises(FormatError):
            V.parse_fsb5(data).sample.markers()


class CoreEnumHelpers(unittest.TestCase):
    def test_record_tables(self):
        self.assertEqual(len(E.record_enums()), 69)


class RecordLayouts(unittest.TestCase):
    def test_no_overlap(self):
        for (build, name), lay in R.layouts().items():
            fields = sorted(lay.fields, key=lambda f: f.offset)
            for a, b in zip(fields, fields[1:]):
                self.assertLessEqual(a.offset + a.size, b.offset, (build, name, a.name, b.name))

    def test_enum_fields_are_typed(self):
        by_name = {}
        for (build, name), lay in R.layouts().items():
            for f in lay.fields:
                by_name.setdefault((name, f.name), set()).add(f.type)
                by_name.setdefault((name, R._table_field(name, f.name)), set()).add(f.type)
        for record_name, field_name in E.record_enums():
            types = by_name.get((record_name, field_name))
            self.assertIsNotNone(types, (record_name, field_name))
            self.assertTrue(any("enum" in t or "flags" in t for t in types), (record_name, field_name, types))

    def test_numbered_fields_find_their_table(self):
        event = record("actor_event", hardpoint_flags_1=2, hardpoint_source_2=1)
        self.assertEqual(event.flags("hardpoint_flags_1").names, [
            E.record_enums()[("actor_event", "hardpoint_flags")][1].constant,
        ])
        self.assertEqual(event.enum("hardpoint_source_2").value, 1)
        self.assertEqual(set(event.named()), {"hardpoint_flags_1", "hardpoint_source_2"})

    def test_ps3_monster_block(self):
        lay = R.layout("ps3", "monster")
        block = next(f for f in lay.fields if f.name == "attribute_modifiers")
        self.assertEqual((block.offset, block.count), (0x54, 142))


if __name__ == "__main__":
    unittest.main()
