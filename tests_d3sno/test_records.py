import struct
import unittest

import builders as B
from d3sno import FormatError
from d3sno import formula as F
from d3sno import power as P
from d3sno import records as R
from d3sno import text as X
from d3sno.sno import hash33

BUILDS = ("ps3", "switch")


class Layouts(unittest.TestCase):
    def test_every_record_on_both_builds(self):
        self.assertEqual(len([k for k in R.layouts() if k[0] != "x360"]), 44)
        self.assertEqual(len([k for k in R.layouts() if k[0] == "x360"]), 7)  # stage 8, X3
        self.assertEqual(R.layout("ps3", "actor").size, 432)
        self.assertEqual(R.layout("switch", "conversation_node").size, 136)
        self.assertEqual(R.layout("ps3", "conversation_node").size, 104)

    def test_repeated_names_are_numbered(self):
        names = [f.name for f in R.layout("switch", "condition").fields]
        self.assertIn("lore_1", names)
        self.assertIn("lore_3", names)
        self.assertNotIn("lore", names)
        self.assertIn("male", names)

    def test_unknown_layout(self):
        with self.assertRaises(FormatError):
            R.layout("ps3", "champion_group")


class Files(unittest.TestCase):
    def test_every_file_type_reads_on_both_builds(self):
        for extension, record in R.RECORDS_BY_EXTENSION.items():
            for build in BUILDS:
                with self.subTest((extension, build)):
                    parsed = R.parse_record_file(B.build_record(build, record, {}), extension)
                    self.assertEqual((parsed.build, parsed.record.type), (build, record))
                    self.assertEqual(parsed.warnings, [])

    def test_header_words(self):
        parsed = R.parse_record_file(B.build_record("switch", "actor", {
            "self_handle": 77, "actor_kind": 1, "bound_sphere_radius": 2.5,
            "bound_box_min": [1.0, 2.0, 3.0],
        }), "acr")
        record = parsed.record
        self.assertEqual((record["self_handle"], record["actor_kind"]), (77, 1))
        self.assertEqual(record["bound_sphere_radius"], 2.5)
        self.assertEqual(record["bound_box_min"], (1.0, 2.0, 3.0))
        self.assertNotIn("tag_map_ptr", record.values)

    def test_actor_events_and_texts(self):
        for build in BUILDS:
            data = B.build_record(build, "actor", {
                "events": [{"color_add": 0x11223344}, {}], "event_count": 2,
                "internal_description": "Desc", "voice_role": "",
            })
            parsed = R.parse_record_file(data, "acr")
            events = parsed.record["events"]
            self.assertEqual(len(events), 2)
            self.assertEqual(events[0]["color_add"], 0x11223344)
            self.assertEqual(parsed.record["internal_description"], "Desc")
            self.assertEqual(parsed.warnings, [])

    def test_actor_tag_map_accepts_type_7(self):
        for build in BUILDS:
            order = ">" if build == "ps3" else "<"
            tags = B.tag_map([(7, 0x10050, 3), (0, 0x10014, 5)], order)
            parsed = R.parse_record_file(B.build_record(build, "actor", {"tag_map": tags}), "acr")
            self.assertEqual(parsed.record["tag_map"].get(0x10050).value, 3)
            self.assertEqual(parsed.record["tag_map"].get(0x10014).value, 5)

    def test_actor_tag_map(self):
        tags = B.tag_map([(0, 0x10014, 5)], "<")
        parsed = R.parse_record_file(B.build_record("switch", "actor", {"tag_map": tags}), "acr")
        self.assertEqual(parsed.record["tag_map"].get(0x10014).value, 5)

    def test_count_must_match_the_array(self):
        data = B.build_record("switch", "actor", {"events": [{}], "event_count": 3})
        parsed = R.parse_record_file(data, "acr")
        self.assertIn("event_count", parsed.warnings[0])
        with self.assertRaises(FormatError):
            R.parse_record_file(data, "acr", strict=True)

    def test_array_size_must_fit_records(self):
        data = bytearray(B.build_record("switch", "actor", {"events": [{}]}))
        struct.pack_into("<I", data, 0x10 + 0x78 + 4, 100)
        with self.assertRaises(FormatError):
            R.parse_record_file(bytes(data), "acr")

    def test_pair_out_of_bounds(self):
        data = bytearray(B.build_record("ps3", "actor", {"internal_description": "x"}))
        struct.pack_into(">I", data, 0x10 + 0x198, 0xFFFFFF)
        with self.assertRaises(FormatError):
            R.parse_record_file(bytes(data), "acr")

    def test_blocks_must_fill_the_rest_exactly(self):
        data = B.build_record("switch", "actor", {"internal_description": "x"}) + bytes(16)
        parsed = R.parse_record_file(data, "acr")
        self.assertTrue(any("end" in w for w in parsed.warnings))

    def test_gap_of_non_zero_bytes(self):
        data = bytearray(B.build_record("switch", "actor", {"internal_description": "abc"}))
        start = struct.unpack_from("<I", data, 0x10 + 0x1A8)[0] + 0x10
        struct.pack_into("<I", data, 0x10 + 0x1A8, start - 0x10 + 8)
        data += b"abc\0\0\0\0\0"
        parsed = R.parse_record_file(bytes(data), "acr")
        self.assertTrue(parsed.warnings)

    def test_areas_are_listed(self):
        data = B.build_record("switch", "actor", {"events": [{}], "voice_role": "v"})
        parsed = R.parse_record_file(data, "acr")
        self.assertEqual(len(parsed.areas), 2)
        self.assertTrue(all(offset % 8 == 0 for offset, _ in parsed.areas))

    def test_too_short(self):
        with self.assertRaises(FormatError):
            R.parse_record_file(B.build_record("switch", "actor", {})[:100], "acr")

    def test_kind_by_record_name(self):
        data = B.build_record("switch", "recipe", {})
        self.assertEqual(R.parse_record_file(data, "recipe").record.type, "recipe")


class Monsters(unittest.TestCase):
    def test_ps3_levels_then_modifiers(self):
        floats = [0.0] * 146
        floats[8] = 1.5
        floats = floats[:142]
        floats[58] = 0.0
        floats[75] = 1.0
        floats[85] = 15.7
        data = B.build_record("ps3", "monster", {
            "level_by_difficulty": [10, 20, 30, 40], "attribute_modifiers": floats,
        })
        record = R.parse_record_file(data, "mon").record
        self.assertEqual(record["level_by_difficulty"], [10, 20, 30, 40])
        self.assertEqual(len(record["attribute_modifiers"]), 142)
        values = R.monster_attributes(record, ">")
        self.assertEqual(values["Hitpoints_Regen_Per_Second"], 1.5)
        self.assertEqual(values["Immunity#0"], 1)
        self.assertIsInstance(values["Immunity#0"], int)
        self.assertEqual(values["Attack_Cooldown_Min"], 15)  # truncated, not rounded
        self.assertFalse(R.monster_grants_experience(record))

    def test_slot_names_and_shared_slot(self):
        floats = [0.0] * 146
        floats[58] = 1.0
        floats[71] = 0.25
        record = R.parse_record_file(B.build_record("switch", "monster", {
            "attribute_modifiers": floats,
        }), "mon").record
        values = R.monster_attributes(record, "<")
        self.assertEqual((values["Resistance#5"], values["Resistance#6"]), (0.25, 0.25))
        self.assertTrue(R.monster_grants_experience(record))
        slots = R.monster_slots(record)
        self.assertEqual(slots["GBA_EXPERIENCE"], 1.0)
        self.assertEqual(len(slots), 146)

    def test_switch_index_order_is_the_same(self):
        floats = [float(i) for i in range(146)]
        record = R.parse_record_file(B.build_record("switch", "monster", {
            "attribute_modifiers": floats,
        }), "mon").record
        self.assertEqual(R.monster_attributes(record, "<")["Damage_Min#0"], 22.0)

    def test_table_offsets_agree(self):
        for attribute in R.monster_attribute_indices():
            self.assertGreaterEqual(attribute.index, 0)
        rows = R.monster_attribute_indices()
        self.assertEqual(len(rows), 146)
        self.assertEqual([r.index for r in rows], list(range(146)))
        for row in rows:
            self.assertEqual(row.switch_offset, 0x44 + 4 * row.index)
            if row.index < 142:
                self.assertEqual(row.ps3_offset, 0x54 + 4 * row.index)
            else:
                self.assertIsNone(row.ps3_offset)
        self.assertEqual(rows[58].slot_name, "GBA_EXPERIENCE")
        self.assertEqual(sum(1 for r in rows if r.attribute_kind == "int"), 21)


class Conversations(unittest.TestCase):
    def tree(self, build):
        word = B.display_time(19, 7 if build == "switch" else 5, 2, 120)
        line = {"node_kind": 5, "player_class": 2, "gender": 1, "display_times": [word]}
        group = {"node_kind": 3, "line_uid": 42, "children": [line, {"node_kind": 1}]}
        return {"root_nodes": [{"node_kind": 4, "children": [group]}], "next_line_uid": 43}

    def test_tree(self):
        for build in BUILDS:
            with self.subTest(build):
                data = B.build_record(build, "conversation", self.tree(build))
                parsed = R.parse_record_file(data, "cnv")
                self.assertEqual(parsed.warnings, [])
                groups = R.line_groups(parsed.record)
                self.assertEqual(len(groups), 1)
                uid, lines = groups[0]
                self.assertEqual((uid, len(lines)), (42, 1))
                time = R.decode_display_time(lines[0]["display_times"][0])
                self.assertEqual((time.locale, time.gender, time.time), (19, 2, 120))
                self.assertEqual(len(list(R.walk_nodes(parsed.record["root_nodes"]))), 4)

    def test_ps3_script_name(self):
        data = B.build_record("ps3", "conversation", {"script_file_name": "0000.lua"})
        self.assertEqual(R.parse_record_file(data, "cnv").record["script_file_name"], "0000.lua")

    def test_script_chunk(self):
        chunk = B.lua_chunk("switch")
        data = B.build_record("switch", "conversation", {"script": chunk})
        self.assertEqual(R.parse_record_file(data, "cnv").record["script"], chunk)

    def test_display_time_bits(self):
        time = R.decode_display_time(B.display_time(3, 4, 1, 999))
        self.assertEqual(time, R.DisplayTime(3, 4, 1, 999))
        self.assertEqual(R.decode_display_time(-1).time, (1 << 22) - 1)


class Conditions(unittest.TestCase):
    def condition(self, build="switch", **values):
        return R.parse_record_file(B.build_record(build, "condition", values), "cnd").record

    def test_nothing_set_passes(self):
        result = R.evaluate_condition(self.condition(), R.Player(0, 3))
        self.assertTrue(result.passed)
        self.assertEqual(result.unchecked, [])

    def test_gender(self):
        male = self.condition(male=1)
        self.assertTrue(R.evaluate_condition(male, R.Player(0, 0)).passed)
        self.assertFalse(R.evaluate_condition(male, R.Player(1, 0)).passed)
        female = self.condition(female=1)
        self.assertFalse(R.evaluate_condition(female, R.Player(0, 0)).passed)

    def test_classes(self):
        record = self.condition(classes=[0, 1, 0, 1, 0, 0, 0])
        self.assertTrue(R.evaluate_condition(record, R.Player(0, 3)).passed)
        self.assertFalse(R.evaluate_condition(record, R.Player(0, 2)).passed)

    def test_ps3_has_5_classes_and_4_difficulties(self):
        record = self.condition("ps3", classes=[1, 0, 0, 0, 0], difficulties=[0, 0, 1, 0])
        self.assertEqual(len(record["classes"]), 5)
        self.assertTrue(R.evaluate_condition(record, R.Player(0, 0, difficulty=2)).passed)
        self.assertFalse(R.evaluate_condition(record, R.Player(0, 0, difficulty=1)).passed)

    def test_level_limits(self):
        record = self.condition(player_level_min=10, player_level_max=20)
        self.assertTrue(R.evaluate_condition(record, R.Player(0, 0, 15)).passed)
        self.assertEqual(R.evaluate_condition(record, R.Player(0, 0, 5)).failed, ["player_level_min"])
        self.assertEqual(R.evaluate_condition(record, R.Player(0, 0, 25)).failed, ["player_level_max"])

    def test_tests_needing_game_state_are_listed(self):
        record = self.condition(quest_1=1234, game_flag_2="flag", act_min=-1)
        result = R.evaluate_condition(record, R.Player(0, 0))
        self.assertTrue(result.passed)
        self.assertEqual(sorted(result.unchecked), ["game_flag_2", "quest_1"])


class Treasure(unittest.TestCase):
    def test_modifiers_and_live_affixes(self):
        for build in BUILDS:
            data = B.build_record(build, "treasure_class", {
                "choose_kind": 1, "modifier_count": 1,
                "modifiers": [{"drop_kind": 3, "affix_count": 2, "affixes": [5, 6, 7, 8, 9, 10]}],
            })
            parsed = R.parse_record_file(data, "trs")
            modifier = parsed.record["modifiers"][0]
            self.assertEqual(R.live_affixes(modifier), [5, 6])
            self.assertEqual(R.DROP_KINDS[modifier["drop_kind"]], "targeted_legendary")
            self.assertTrue(R.rolls_every_modifier(parsed.record["choose_kind"]))
            self.assertEqual(parsed.record.enum("choose_kind").constant, "TCC_ALL_OF")

    def test_choose_kinds(self):
        self.assertFalse(R.rolls_every_modifier(0))  # one of, by weight
        self.assertFalse(R.rolls_every_modifier(3))
        self.assertTrue(R.rolls_every_modifier(1))  # all of, each by its chance
        self.assertTrue(R.rolls_every_modifier(2))
        self.assertEqual(R.choose_rule(3), R.ChooseRule(3, "one_of", True))
        self.assertEqual(R.choose_rule(-1).mode, "none")
        self.assertFalse(R.rolls_every_modifier(-1))
        with self.assertRaises(FormatError):
            R.rolls_every_modifier(4)

    def test_negative_affix_count(self):
        record = R.parse_record_file(B.build_record("switch", "recipe", {
            "affix_count": -1, "affixes": [1] * 6,
        }), "rcp").record
        self.assertEqual(R.live_affixes(record), [])


class Quests(unittest.TestCase):
    def test_ps3_step_fields(self):
        data = B.build_record("ps3", "quest", {
            "steps": [{"name": "Step one", "level_by_difficulty": [1, 2, 3, 4],
                       "reward_recipes": [1, 2, 3, 4, 5]}],
            "step_count": 1,
        })
        parsed = R.parse_record_file(data, "qst")
        step = parsed.record["steps"][0]
        self.assertEqual(step["name"], "Step one")
        self.assertEqual(len(step["reward_recipes"]), 5)
        self.assertEqual(parsed.warnings, [])

    def test_switch_step_name_is_a_text_pair(self):
        data = B.build_record("switch", "quest", {"steps": [{"name": "Step"}], "step_count": 1})
        self.assertEqual(R.parse_record_file(data, "qst").record["steps"][0]["name"], "Step")


class Defaults(unittest.TestCase):
    def test_forms(self):
        self.assertEqual(R.tag_default("actor", 0x10007).value, 1.0)
        self.assertEqual(R.tag_default("actor", 0x10030).value, 1200)
        self.assertEqual(R.tag_default("actor", 0x10077).form, "none")
        self.assertEqual(R.tag_default("monster", 0x20100).value, "1")
        self.assertTrue(any(d.form == "name" for d in R.tag_defaults().values()))

    def test_numbers(self):
        self.assertEqual(R.default_number(R.tag_default("actor", 0x10077)), -1)
        self.assertEqual(R.default_number(R.tag_default("monster", 0x20100)), 1.0)
        self.assertIsNone(R.default_number(None))

    def test_formula_text_that_is_not_a_number(self):
        entry = next(d for d in R.tag_defaults().values() if d.value == "Attacks_Per_Second_Total")
        self.assertIsNone(R.default_number(entry))

    def test_missing(self):
        self.assertIsNone(R.tag_default("actor", 0x1))

    def test_context_uses_the_power_tables(self):
        tag = next(
            t for (table, t), d in R.tag_defaults().items()
            if table == "power_general" and d.form == "formula" and d.value == "1"
        )

        class Ctx(R.DefaultsMixin, F.EvalContext):
            pass

        power = P.parse_power(B.build_power("switch"))
        self.assertEqual(F.evaluate_tag(power, tag, Ctx()), 1.0)
        ctx = Ctx()
        self.assertEqual(F.evaluate_tag(power, 0x7FFFFFF, ctx), 0)
        self.assertTrue(ctx.warnings)


class Attributes(unittest.TestCase):
    def test_safe_rows_only(self):
        self.assertEqual(R.ps3_attribute(65).name, "Level")
        self.assertEqual(R.ps3_attribute(65).switch_id, 57)
        self.assertEqual(R.ps3_attribute(66).status, "SUPPORTED")  # stage 7 table
        # V9-G3-A: 12 = Intelligence is SUPPORTED since stage 10.1 (was HYPOTHESIS).
        self.assertEqual(R.ps3_attribute(12).name, "Intelligence")
        self.assertEqual(R.ps3_attribute(0).name, "Axe_Bad_Data")
        self.assertEqual(R.ps3_attribute(0).switch_id, 0)
        self.assertIsNone(R.ps3_attribute(99999))

    def test_order_is_kept(self):
        known = [a for a in R.ps3_attributes().values() if a.switch_id is not None]
        known.sort(key=lambda a: a.ps3_id)
        ids = [a.switch_id for a in known]
        self.assertEqual(ids, sorted(ids))


class Labels(unittest.TestCase):
    def test_forms(self):
        self.assertEqual(R.buff_label("Fire", 2), "Fire_2")
        self.assertEqual(R.buff_label("Fire", 2, True), "Fire_2_desc")
        self.assertEqual(R.attribute_label("Strength"), "Strength")
        self.assertEqual(R.attribute_label("Resistance", "Fire"), "Resistance#Fire")
        runes = R.rune_labels("Bash", "C")
        self.assertEqual(runes, {
            "name": "NameRune_C#Bash", "short": "Simple_Rune_C#Bash", "full": "Rune_C#Bash",
        })
        self.assertEqual(R.quest_labels(3, "Barb")["objective"], "Objective3Desc")
        self.assertEqual(R.quest_labels(3, "Barb")["description"], "Step3Desc_Barb")

    def test_bad_rune(self):
        with self.assertRaises(FormatError):
            R.rune_labels("Bash", "F")

    def test_hash(self):
        self.assertEqual(R.label_hash("Name"), hash33("Name"))
        self.assertEqual(R.label_hash("\xff"), 0xFFFFFFFF)


class Marks(unittest.TestCase):
    def test_plural_by_the_last_number(self):
        self.assertEqual(R.render_marks("1 |4apple:apples;"), "1 apple")
        self.assertEqual(R.render_marks("2 or 5 |4apple:apples;"), "2 or 5 apples")

    def test_plural_rule_is_pluggable(self):
        rule = lambda n, forms: 2 if n >= 5 else 0
        self.assertEqual(R.render_marks("5 |4a:b:c;", plural=rule), "5 c")

    def test_gender(self):
        self.assertEqual(R.render_marks("|5he:she; came", gender=1), "she came")
        self.assertEqual(R.render_marks("|5he:she; came", gender=0), "he came")

    def test_declension(self):
        forms = X.parse_declension("miecz\t2x\tmiecza\r\n".encode())
        self.assertEqual(forms[("miecz", 2)], "miecza")
        self.assertEqual(R.render_marks("bez |3-2(miecz)", declension=forms), "bez miecza")
        self.assertEqual(R.render_marks("|3-5(miecz)", declension=forms), "miecz")

    def test_values(self):
        value = {"dmg": 12.345, "neg": -2.0}.__getitem__
        self.assertEqual(R.render_marks("[dmg|1|]", value=value), "12.3")
        self.assertEqual(R.render_marks("[dmg|+0|]", value=value), "+12")
        self.assertEqual(R.render_marks("[neg|+1|]", value=value), "-2.0")

    def test_value_needs_a_source(self):
        with self.assertRaises(FormatError):
            R.render_marks("[dmg|1|]")

    def test_plain_text(self):
        self.assertEqual(R.render_marks("{c_green}x{/c}"), "{c_green}x{/c}")


class Listeners(unittest.TestCase):
    def test_flags(self):
        self.assertEqual(R.listener_rules(0), R.ListenerRules(True, False, False))
        self.assertEqual(R.listener_rules(7), R.ListenerRules(False, True, True))
        self.assertEqual(R.listener_rules(2), R.ListenerRules(True, True, False))
        self.assertEqual(R.listener_rules(4), R.ListenerRules(True, False, True))

    def test_arguments(self):
        self.assertEqual(R.handler_arguments(0, 5, "Quest", 1, 2), (5, "Quest", 1, 2))
        self.assertEqual(R.handler_arguments(1, 5, "Quest", 1, 2), (1, 2))
        with self.assertRaises(FormatError):
            R.handler_arguments(0, 5, "Item")

    def test_ticks(self):
        self.assertEqual(R.ticks_to_seconds(90), 1.5)
