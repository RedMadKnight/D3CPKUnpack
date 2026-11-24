"""Stage 10: ``20_corrections9.md`` (G1 geometry, G2 x360 textures, archives and names, G3 names).

There are no game files here: every file is synthetic, built from the rules.
"""

import _libpath  # noqa: F401  (python -I: the library folder on sys.path)
import csv
import io
import struct
import unittest
import zlib
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import builders as B
import d3sno
from d3sno import FormatError
from d3sno import appearance as P
from d3sno import archive as C
from d3sno import balance as G
from d3sno import cli
from d3sno import cloth as CL
from d3sno import datafiles as D
from d3sno import enums as E
from d3sno import power as PW
from d3sno import records as R
from d3sno import scripts as S
from d3sno import tags as T
from d3sno import texture as TX
from d3sno import write as WR
from d3sno import x360_archive as XA
from d3sno import x360_fields as XF
from d3sno import x360_texture as XT

TRIANGLE = [(0, 0, 0), (10, 0, 0), (0, 10, 0)]
DATA = Path(D.DATA)


def tsv(name):
    with open(DATA / name, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


class Marker(unittest.TestCase):
    def test_stage(self):
        self.assertGreaterEqual(d3sno.STAGE, 10)
        self.assertFalse(d3sno.__version__.startswith("0.9."))

    def test_tables_shipped(self):
        for name in ("s10_cloth_rules", "s10_enums_core", "s10_enums_geometry", "s10_enums_records",
                     "s10_not_knowable", "s10_ps3_attributes", "s10_ps3_draw_rules",
                     "s10_ps3_vertex_formats", "s10_tags", "x360_tex_formats", "x360_tex_tail_slots",
                     "x360_tex_volume_bits", "x360_struct_names", "x360_xp_bounty", "x360_gam_type_names"):
            self.assertTrue((DATA / f"{name}.tsv").is_file(), name)


# --- Row updates (s10_*.tsv) -----------------------------------------------------------------

class RowUpdates(unittest.TestCase):
    def test_replace_in_place_and_add(self):
        base = tsv("ps3_attributes.tsv")
        rows = D.table_rows("ps3_attributes.tsv")
        self.assertEqual(len(rows), len(base))  # every s10 id already had a row
        position = [r["ps3_id"] for r in base].index("33")
        self.assertEqual((rows[position]["name"], rows[position]["status"]), ("Defense", "PROVEN"))
        core = D.table_rows("cov_enums_core.tsv")
        self.assertEqual(len(core), len(tsv("cov_enums_core.tsv")) + 1)  # TEAM_INVALID is new
        self.assertEqual(core[-1]["name"], "TEAM_INVALID")

    def test_shipped_tables_unchanged(self):
        self.assertEqual([r["status"] for r in tsv("ps3_attributes.tsv") if r["ps3_id"] == "33"],
                         ["UNRESOLVED"])

    def test_tag_key_ignores_hex_case(self):
        self.assertEqual(D._key({"table": "marker", "tag_hex": "0x8801A"}, ("table", "tag_hex")),
                         ("marker", "0x8801a"))

    def test_other_tables_as_shipped(self):
        self.assertEqual(len(D.table_rows("x360_struct_names.tsv")), len(tsv("x360_struct_names.tsv")))


# --- G3 --------------------------------------------------------------------------------------

class Ps3Attributes(unittest.TestCase):
    def test_new_names(self):
        expected = {13: ("Vitality", 13), 31: ("Attack", None), 32: ("Precision", None),
                    33: ("Defense", None), 78: ("Resource_Set_Point_Bonus", 71),
                    87: ("Health_Globe_Bonus_Chance", 82), 128: ("Hitpoints_Regen_Per_Second", 126),
                    150: ("Movement_Scalar", 171)}
        for ps3_id, (name, switch_id) in expected.items():
            row = R.ps3_attribute(ps3_id)  # safe_only
            self.assertEqual((row.name, row.switch_id), (name, switch_id), ps3_id)
        self.assertEqual(R.ps3_attribute(33).status, "PROVEN")
        self.assertEqual(R.attribute_name(31, "ps3"), "Attack")
        self.assertEqual(R.PS3_ONLY_ATTRIBUTES, (31, 32, 33))

    def test_199_stays_unresolved(self):
        self.assertIsNone(R.ps3_attribute(199))
        row = R.ps3_attribute(199, safe_only=False)
        self.assertEqual((row.status, row.name), ("UNRESOLVED", None))
        self.assertEqual(row.candidates, ("Damage_Weapon_Bonus_Delta", "Damage_Weapon_Bonus_Delta_X1"))
        self.assertEqual(R.PS3_ATTRIBUTE_DESCRIPTIONS[199], "weapon bonus maximum-damage delta")
        self.assertEqual(E.value_status("ps3_attributes", 199), "NOT_KNOWABLE_FROM_DATA")

    def test_hypothesis_still_excluded(self):
        # V9-G3-A made the last HYPOTHESIS (12) SUPPORTED; the safe filter still drops UNRESOLVED.
        hypotheses = [a for a in R.ps3_attributes().values() if a.status == "HYPOTHESIS"]
        self.assertEqual(hypotheses, [])
        self.assertIsNone(R.ps3_attribute(199))


class ScriptEvents(unittest.TestCase):
    def test_31_and_32_stay_raw(self):
        for event in (31, 32):
            self.assertIsNone(S.switch_event_id(event, "ps3"))
        self.assertEqual(S.EVENT_PS3_GUESSES, {31: 35, 32: 36})


class Ps3Tags(unittest.TestCase):
    def test_rows_without_names(self):
        for table, tag, kind in (("power_general", 0x50281, 4), ("marker", 0x88010, 0),
                                 ("marker", 0x88020, 0)):
            info = T.tag_info(table, tag)
            self.assertEqual(info.record_type, kind)
            self.assertIsNone(T.tag_name(table, tag))

    def test_read_by_record_type_on_ps3_only(self):
        good = PW.TagMap("General", None, [PW.TagRecord(4, 0x50281, PW.Formula("1", b""), 0)])
        self.assertEqual(T.check_tag_map(good, "power_general", "ps3"), [])
        wrong = PW.TagMap("General", None, [PW.TagRecord(0, 0x50281, 1, 0)])
        self.assertEqual(len(T.check_tag_map(wrong, "power_general", "ps3")), 1)
        for build in ("switch", "x360"):
            problems = T.check_tag_map(good, "power_general", build)
            self.assertEqual(len(problems), 1)
            self.assertIn("only in ps3", problems[0])
        marker = PW.TagMap("Marker", None, [PW.TagRecord(0, 0x88020, 3, 0)])
        self.assertEqual(T.check_tag_map(marker, "marker", "ps3"), [])

    def test_named_record_has_no_name(self):
        named = T.name_records(PW.TagMap("Marker", None, [PW.TagRecord(0, 0x88010, 1, 0)]), "marker")
        self.assertEqual((named[0].name, named[0].xml_name), (None, None))


class EnumValues(unittest.TestCase):
    def test_team_invalid(self):
        self.assertEqual(E.enum_name("tag_value:gTeamNames", -1), "TEAM_INVALID")
        self.assertEqual(E.enum_name("tag_value:gTeamNames", 0xFFFFFFFF), "TEAM_INVALID")

    def test_gizmo_locations(self):
        for value in (76, 80, 88, 101):
            row = E.geometry_value("Marker", "eType", value)
            self.assertEqual((row.status, row.constant), ("SUPPORTED", ""))
            self.assertTrue(row.name.startswith(f"gizmo location {value - 49}"))
            self.assertIn(f"slot {value - 50}", row.name)
        self.assertIsNone(E.geometry_value("Marker", "eType", 85))  # no row: never seen

    def test_not_knowable_record_values(self):
        for record, field, bit in (("objective", "flags", 1), ("objective", "flags", 4),
                                   ("actor_event", "pin_flags", 0),
                                   ("actor", "coll_dynamic_no_overlap", 17)):
            row = E.record_value(record, field, bit)
            self.assertEqual((row.status, row.constant), ("NOT_KNOWABLE", ""))
            self.assertEqual(row.name, row.label)  # G3: named() gives the label
            flags = E.record_flags(record, field, 1 << bit)
            self.assertEqual((flags.names, flags.open_bits), ([], [bit]))  # treated like OPEN

    def test_not_knowable_list(self):
        rows = E.not_knowable()
        self.assertEqual(len(rows), 14)  # 13 NOT_KNOWABLE values and TEAM_INVALID (NAMED)
        self.assertEqual(E.value_status("tag_value:gActorGizmoNames", 24), "NOT_KNOWABLE")
        self.assertEqual(E.value_status("tag_value:gActorBrainNames", 39), "NOT_KNOWABLE")
        self.assertEqual(E.value_status("marker", 0x88010), "NOT_KNOWABLE")
        self.assertEqual(E.value_status("tag_value:gTeamNames", -1), "NAMED")
        self.assertIsNone(E.value_status("tag_value:gActorGizmoNames", 23))

    def test_labels_never_constants(self):
        for enum, value in (("tag_value:gActorGizmoNames", 22), ("tag_value:gActorBrainNames", 23)):
            self.assertIsNone(E.enum_name(enum, value))
            self.assertEqual(E.value_label(enum, value), f"unnamed value {value}")

    def test_proximity_minimum_is_not_none(self):
        self.assertEqual(E.enum_name(E.PROXIMITY_ENUM, -1), "AI_PROXIMITY_SHAPE_NONE")
        self.assertIsNone(E.enum_name(E.PROXIMITY_ENUM, -2147483648))
        self.assertEqual(E.value_label(E.PROXIMITY_ENUM, 0x80000000),
                         "minimum 32-bit integer, no shape constant")


# --- G2-S --------------------------------------------------------------------------------------

class BalanceNames(unittest.TestCase):
    def test_type_names(self):
        self.assertEqual(G.balance_type_name(23, "x360"), "Item Enchantments")
        self.assertEqual(G.balance_type_name(44, "x360"), "Devil's Hand Redeemable Sets")
        self.assertEqual(G.balance_type_name(38, "x360"), "Item Durability Cost Scalars")
        self.assertEqual(G.balance_type_name(17, "x360"), "Scenery")
        self.assertEqual(len(G.x360_type_names()), 37)
        self.assertIsNone(G.balance_type_name(38, "switch"))
        self.assertNotIn(17, G.x360_gam_tables())

    def test_help_codes_on_x360(self):
        row = b"HC1".ljust(256, b"\0") + b"First text".ljust(256, b"\0") + b"Two".ljust(128, b"\0")
        data = B.build_gam_layout("x360", 24, 0x68, 640, [{"fields": {0: ("640s", row)}}] * 2, {})
        parsed = G.parse_balance(data, "x360")
        self.assertEqual(parsed.table, "help_codes")
        self.assertEqual(G.help_codes(parsed), [{"code": "HC1", "text_1": "First text", "text_2": "Two"}] * 2)
        self.assertEqual(WR.write(parsed), data)
        with self.assertRaises(FormatError):
            G.help_codes(G.parse_balance(B.build_gam_layout("x360", 38, 0x1E8, 8, [{}], {}), "x360"))


class StructFields(unittest.TestCase):
    def test_bounty_block(self):
        for offset, name in XF.BOUNTY_BLOCK.items():
            field = XF.x360_field("ExperienceLevel", offset)
            self.assertEqual((field.name, field.status), (name, "SUPPORTED"), offset)
        self.assertEqual(XF.x360_field("ExperienceLevel", 124).switch_offset, 128)
        absent = [f.name for f in XF.x360_struct_fields()["ExperienceLevel"] if f.status == "ABSENT"]
        self.assertIn("BountyCampsXP", absent)
        self.assertIn("BountyCampsGold", absent)
        self.assertNotIn("GoblinHuntBonusGold", absent)
        self.assertNotIn("BountyKillUniqueXP", absent)

    def test_other_structs_untouched(self):
        self.assertEqual(len(XF.x360_struct_fields()["ExperienceLevelAlt"]),
                         sum(1 for r in tsv("x360_struct_fields.tsv") if r["struct"] == "ExperienceLevelAlt"))

    def test_proposed_names_are_display_only(self):
        names = XF.x360_struct_names()
        self.assertEqual(len(names), len(tsv("x360_struct_names.tsv")))
        self.assertEqual({n.status for n in names}, {"SUPPORTED", "OPEN"})
        self.assertEqual(XF.display_name("ItemDurabilityCostScalarsTable", 0), "serItemDurabilityCostScalars")
        self.assertEqual(XF.display_name("ExperienceLevel", 0), "DeltaXP")  # no proposal: table name


# --- G2-T: x360 textures --------------------------------------------------------------------------

def doc_tiled(x, y, pitch, bpe):
    """G2-T2 transcribed from the description, independent of the package."""
    b = {1: 0, 2: 1, 4: 2, 8: 3, 16: 4}[bpe]
    macro = ((x >> 5) + (y >> 5) * (pitch >> 5)) << (b + 7)
    micro = ((x & 7) + ((y & 6) << 2)) << b
    t = macro + ((micro & ~15) << 1) + (micro & 15) + ((y & 8) << (3 + b)) + ((y & 1) << 4)
    u = ((t & ~511) << 3) + ((t & 448) << 2) + (t & 63) + ((y & 16) << 7) \
        + ((((((y & 8) >> 2) + (x >> 3)) & 3)) << 6)
    return u >> b


def dxt1_block(c0, c1, bits=0):
    return struct.pack("<HHI", c0, c1, bits)


def place(payload_size, surfaces, bpe, swap):
    """Write elements by hand: surfaces are (offset, pitch, origin, rows of element bytes)."""
    out = bytearray(payload_size)
    for offset, pitch, (ox, oy), rows in surfaces:
        for y, row in enumerate(rows):
            for x, element in enumerate(row):
                if swap:
                    element = bytes(b for i in range(0, len(element), 2) for b in (element[i + 1], element[i]))
                at = offset + doc_tiled(x + ox, y + oy, pitch, bpe) * bpe
                out[at:at + bpe] = element
    return bytes(out)


def x360_tex(fmt, w, h, payload, mip_max=0, faces=1, depth=1):
    return TX.parse_texture(B.build_tex("x360", fmt, w, h, b"", mip_max=mip_max, faces=faces,
                                        depth=depth, payload=payload))


class Tiling(unittest.TestCase):
    def test_formula_matches_description(self):
        for bpe in (1, 2, 4, 8, 16):
            for x, y in ((0, 0), (1, 0), (7, 1), (8, 8), (31, 31), (33, 2), (5, 40)):
                self.assertEqual(XT.tiled_element(x, y, 64, bpe), doc_tiled(x, y, 64, bpe))

    def test_first_elements(self):
        self.assertEqual([XT.tiled_element(x, 0, 32, 4) for x in range(4)], [0, 1, 2, 3])

    def test_tile_is_a_permutation(self):
        for bpe in (4, 8, 16):
            seen = {XT.tiled_element(x, y, 32, bpe) for x in range(32) for y in range(32)}
            self.assertEqual(seen, set(range(1024)))

    def test_volume_bits_follow_the_table(self):
        rows = tsv("x360_tex_volume_bits.tsv")
        self.assertEqual([tuple(r["value"].replace(" xor ", " ").split()) for r in rows],
                         list(XT.VOLUME_BITS))
        self.assertEqual(XT.volume_element(1, 0, 0), 1)
        self.assertEqual(XT.volume_element(0, 1, 0), 4)
        self.assertEqual(XT.volume_element(0, 0, 1), 1 << 8)
        self.assertEqual(XT.volume_element(0, 0, 4), (1 << 14) | (1 << 9) | (1 << 5))
        seen = {XT.volume_element(x, y, z) for x in range(64) for y in range(64) for z in range(0, 64, 7)}
        self.assertEqual(len(seen), 64 * 64 * 10)

    def test_tail_slots_follow_the_table(self):
        for row in tsv("x360_tex_tail_slots.tsv"):
            got = XT.tail_offset(int(row["packed_index"]), row["first_packed_level_wider_than_tall"] == "yes")
            self.assertEqual(got, (int(row["x_texels"]), int(row["y_texels"])))


class Layout(unittest.TestCase):
    def test_dxt1_sizes(self):
        surfaces, size = XT.layout(9, 256, 128, 9)
        self.assertEqual(size, 16384 + 8192 + 8192 + 8192)
        self.assertEqual([s.offset for s in surfaces[:4]], [0, 16384, 24576, 32768])
        self.assertEqual(XT.first_packed_level(256, 128, 9), 3)
        self.assertTrue(all(s.packed for s in surfaces[3:]))
        self.assertEqual(surfaces[3].origin, (0, 4))  # 32 x 16 is wide: (0, 16) texels

    def test_one_level_never_packed(self):
        surfaces, size = XT.layout(0, 8, 8, 1)
        self.assertFalse(surfaces[0].packed)
        self.assertEqual(size, 32 * 32 * 4)

    def test_cube_faces_follow_each_other(self):
        # 64 x 64 DXT1, 3 levels: levels 0 and 1 have surfaces of their own, level 2 is packed.
        surfaces, size = XT.layout(9, 64, 64, 3, faces=6)
        own = [(s.level, s.face, s.offset) for s in surfaces if not s.packed]
        self.assertEqual(own, [(m, f, (m * 6 + f) * 8192) for m in range(2) for f in range(6)])
        tails = sorted({s.offset for s in surfaces if s.packed})
        self.assertEqual(tails, [(12 + f) * 8192 for f in range(6)])
        self.assertEqual(size, 18 * 8192)

    def test_volume(self):
        self.assertEqual(XT.layout(0, 64, 64, 1, 1, 64)[1], 1048576)
        with self.assertRaises(FormatError):
            XT.layout(9, 64, 64, 1, 1, 64)

    def test_payload_size_check(self):
        tex = x360_tex(0, 8, 8, bytes(4096))
        self.assertEqual(XT.payload_problems(tex), [])
        tex = x360_tex(0, 8, 8, bytes(4000))
        self.assertTrue(XT.payload_problems(tex))


class Decode(unittest.TestCase):
    def test_dxt1_two_levels(self):
        # 64 x 64, 2 levels: level 1 (32 x 32, smaller side > 16) is a surface of its own.
        red, blue = dxt1_block(0xF800, 0xF800), dxt1_block(0x001F, 0x001F)
        level0 = [[red if (x + y) % 2 else blue for x in range(16)] for y in range(16)]
        level1 = [[blue] * 8 for _ in range(8)]
        payload = place(8192 * 2, [(0, 32, (0, 0), level0), (8192, 32, (0, 0), level1)], 8, True)
        tex = x360_tex(9, 64, 64, payload, mip_max=1)
        image = XT.decode_image(tex, 0)
        self.assertEqual(image.rgba[0:4], bytes((0, 0, 255, 255)))
        self.assertEqual(image.rgba[16:20], bytes((255, 0, 0, 255)))  # block (1, 0)
        self.assertEqual(XT.decode_image(tex, 1).rgba[:4], bytes((0, 0, 255, 255)))
        standard = XT.standard_elements(tex, 0)
        self.assertEqual(standard[:8], blue)
        self.assertEqual(XT.build_payload(9, 64, 64, 2, {(0, 0): b"".join(sum(level0, [])),
                                                         (1, 0): b"".join(sum(level1, []))}), payload)

    def test_packed_tail(self):
        # 32 x 32 DXT1, 3 levels: level 0 is already packed (smaller side 32? no: 32 > 16).
        # Level 1 (16 x 16) and level 2 (8 x 8) share the tail; not wide: (16, 0) and (8, 0) texels.
        level0 = [[dxt1_block(0xF800, 0xF800)] * 8 for _ in range(8)]
        level1 = [[dxt1_block(0x07E0, 0x07E0)] * 4 for _ in range(4)]
        level2 = [[dxt1_block(0x001F, 0x001F)] * 2 for _ in range(2)]
        payload = place(8192 * 2, [(0, 32, (0, 0), level0), (8192, 32, (4, 0), level1),
                                   (8192, 32, (2, 0), level2)], 8, True)
        tex = x360_tex(9, 32, 32, payload, mip_max=2)
        self.assertEqual(XT.payload_problems(tex), [])
        self.assertEqual([XT.decode_image(tex, m).rgba[:4] for m in range(3)],
                         [bytes((255, 0, 0, 255)), bytes((0, 255, 0, 255)), bytes((0, 0, 255, 255))])

    def test_argb8888_is_argb_without_swap(self):
        texels = [[bytes((0x80, 1, 2, 3))] * 8 for _ in range(8)]
        tex = x360_tex(0, 8, 8, place(4096, [(0, 32, (0, 0), texels)], 4, False))
        self.assertEqual(XT.decode_image(tex).rgba[:4], bytes((1, 2, 3, 0x80)))

    def test_argb1555_is_swapped(self):
        word = 1 << 15 | 31 << 10  # opaque red, as a little-endian word after the swap
        texels = [[struct.pack("<H", word)] * 8 for _ in range(8)]
        tex = x360_tex(4, 8, 8, place(4096, [(0, 32, (0, 0), texels)], 2, True))
        self.assertEqual(XT.decode_image(tex).rgba[:4], bytes((255, 0, 0, 255)))

    def test_argb1555_payload_is_swapped(self):
        words = [[struct.pack("<H", 0x8000 | (x << 5) | y) for x in range(8)] for y in range(8)]
        payload = XT.build_payload(4, 8, 8, 1, {(0, 0): b"".join(sum(words, []))})
        self.assertEqual(payload, place(4096, [(0, 32, (0, 0), words)], 2, True))

    def test_l8_and_a8(self):
        texels = [[bytes((x * 16 + y,)) for x in range(8)] for y in range(8)]
        tex = x360_tex(7, 8, 8, place(4096, [(0, 32, (0, 0), texels)], 1, False))
        image = XT.decode_image(tex)
        self.assertEqual(image.rgba[4 * 9:4 * 9 + 4], bytes((17, 17, 17, 255)))
        tex = x360_tex(23, 8, 8, place(4096, [(0, 32, (0, 0), texels)], 1, False))
        self.assertEqual(XT.decode_image(tex).rgba[4 * 9 + 3], 17)

    def test_cube_face(self):
        faces = [[[dxt1_block(0xF800 if f == 3 else 0x001F, 0)] * 2 for _ in range(2)] for f in range(6)]
        surfaces = [(f * 8192, 32, (0, 0), faces[f]) for f in range(6)]
        tex = x360_tex(9, 8, 8, place(6 * 8192, surfaces, 8, True), faces=6)
        self.assertEqual(XT.decode_image(tex, 0, 3).rgba[:3], bytes((255, 0, 0)))
        self.assertEqual(XT.decode_image(tex, 0, 2).rgba[:3], bytes((0, 0, 255)))

    def test_volume_slice(self):
        payload = bytearray(1048576)
        for x, y, z in ((0, 0, 0), (5, 9, 3), (63, 63, 63)):
            at = XT.volume_element(x, y, z) * 4
            payload[at:at + 4] = bytes((0xFF, x, y, z))
        tex = x360_tex(0, 64, 64, bytes(payload), depth=64)
        image = XT.decode_image(tex, 0, 0, 3)
        at = (9 * 64 + 5) * 4
        self.assertEqual(image.rgba[at:at + 4], bytes((5, 9, 3, 0xFF)))

    def test_round_trip_of_build_payload(self):
        rng = bytes((i * 37 + 11) % 256 for i in range(16 * 16 * 16))
        blocks = {(0, 0): rng[:8 * 8 * 16], (1, 0): rng[:4 * 4 * 16], (2, 0): rng[:2 * 2 * 16]}
        payload = XT.build_payload(12, 32, 32, 3, blocks)
        tex = x360_tex(12, 32, 32, payload, mip_max=2)
        for level in range(3):
            self.assertEqual(XT.standard_elements(tex, level), blocks[(level, 0)])

    def test_default_reader_and_writer_unchanged(self):
        data = B.build_tex("x360", 0, 8, 8, b"", payload=bytes(4096))
        tex = TX.parse_texture(data)
        with self.assertRaises(FormatError):
            TX.decode_image(tex)  # decoding is the optional x360_texture module
        self.assertEqual(WR.write_tex(tex), data)

    def test_block_difference(self):
        a = TX.Image(4, 4, bytes(64))
        b = TX.Image(4, 4, bytes([4] * 64))
        self.assertEqual(XT.block_difference(a, b), 4.0)

    def test_cli_option(self):
        texels = [[bytes((0xFF, 9, 8, 7))] * 8 for _ in range(8)]
        data = B.build_tex("x360", 0, 8, 8, b"", payload=place(4096, [(0, 32, (0, 0), texels)], 4, False))
        folder = Path(__file__).parent / "_stage10_tmp"  # stays inside the package folder
        folder.mkdir(exist_ok=True)
        source, target = folder / "t.tex", folder / "t.png"
        try:
            source.write_bytes(data)
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertNotEqual(cli.main(["texture", str(source), "-o", str(target)]), 0)
                self.assertEqual(cli.main(["texture", str(source), "-o", str(target), "--x360-pixels"]), 0)
            self.assertTrue(target.read_bytes().startswith(b"\x89PNG"))
        finally:
            for path in (source, target):
                if path.exists():
                    path.unlink()
            folder.rmdir()


# --- G2-A: archives -------------------------------------------------------------------------------

def pieces():
    random = bytes((i * 7919 + i // 3) % 251 for i in range(40000))
    return [
        XA.Piece("b.bin", 0, b"hello " * 3000),
        XA.Piece("a.fsb", 0, random),
        XA.Piece("z.bin", 0, b""),
        XA.Piece("b.bin", 1, b"hello " * 3000),
        XA.Piece("c.sbt", 0, bytes(range(256)) * 200),
    ]


class ArchiveRules(unittest.TestCase):
    def write(self, flags, **kw):
        return XA.write_x360_archive(pieces(), flags=flags, read_sector=0x800, **kw)

    def test_valid_and_complete(self):
        for flags in (0, 2, 10):
            data = self.write(flags).data
            report = C.verify_archive(data)
            self.assertTrue(report.ok, report.format())
            archive = C.read_archive(data)
            self.assertEqual(archive.header.build, "x360")
            for piece in pieces():
                self.assertEqual(archive.read_file(piece.name, piece.copy), piece.data)

    def test_tables(self):
        archive = C.read_archive(self.write(2).data)
        h = archive.header
        self.assertEqual([f.hash for f in archive.files], sorted(f.hash for f in archive.files))
        self.assertEqual(h.first_location_bits, h.location_count.bit_length())  # A2
        self.assertEqual(h.location_bits, h.stream_size.bit_length())
        self.assertEqual(h.size_bits, max(f.size for f in archive.files).bit_length())
        self.assertEqual(h.copies_bits, 2)
        firsts = 0
        for entry in archive.files:
            self.assertEqual(entry.first, firsts)
            locations = archive.locations[entry.first:entry.first + entry.count]
            self.assertEqual(locations, sorted(locations))
            firsts += entry.count
        self.assertEqual(archive.names, sorted(archive.names, key=C.name_hash))
        copies = archive.locations[archive.find("b.bin").first:][:2]
        self.assertEqual(copies[1] - copies[0], 18000 + 40000)  # not adjacent (A1)

    def test_first_location_width_is_the_location_count(self):
        four = [XA.Piece(f"f{i}.bin", 0, bytes([i]) * 100) for i in range(4)]
        h = C.read_header(XA.write_x360_archive(four).data)
        self.assertEqual((h.location_count, h.first_location_bits), (4, 3))

    def test_packets_never_mix_pieces(self):
        result = self.write(2)
        archive = C.read_archive(result.data)
        starts = {0, 18000, 58000, 76000}
        for slot, offset in enumerate(archive.sector_offsets):
            at = offset
            for packet in archive.sector(slot).packets:
                ends = [s for s in starts if at < s < at + packet.declared]
                self.assertEqual(ends, [], (slot, at))
                at += packet.declared

    def test_level_0_for_sound_and_subtitles(self):
        result = self.write(2)
        for plan in result.packets:
            name = pieces()[plan.piece].name
            self.assertEqual(plan.stored[:2] == bytes((0x78, 0x01)), name.endswith((".fsb", ".sbt")), name)
        result = self.write(10)
        self.assertFalse(any(p.stored[:2] == bytes((0x78, 0x01)) for p in result.packets))

    def test_end_of_file(self):
        flag2 = self.write(2).data
        h = C.read_header(flag2)
        self.assertEqual((len(flag2) - h.start) % h.comp_sector, 0)
        result = self.write(10, encrypted={(4, 0)})
        data = result.data
        self.assertEqual(data[-6:], struct.pack(">HHH", 0, 0, 0x8000 if result.packets[-1].encrypted else 0))
        self.assertNotEqual((len(data) - C.read_header(data).start) % 0x4000, 0)
        raw = self.write(0).data
        h = C.read_header(raw)
        self.assertEqual(len(raw) - h.start, h.stream_size)
        self.assertEqual(h.sector_index_bits, 0)

    def test_header_sector_word(self):
        compact = C.read_header(self.write(10).data)
        self.assertEqual(compact.header_sector % 4, 0)
        self.assertEqual(compact.start, compact.header_sector)
        sectors = C.read_header(self.write(2).data)
        self.assertEqual(sectors.start, sectors.header_sector * 0x800)

    def test_encryption_per_packet(self):
        result = self.write(2, encrypted={(0, 0), (3, 0)})
        self.assertEqual({(p.piece, p.position) for p in result.packets if p.encrypted}, {(0, 0), (3, 0)})
        archive = C.read_archive(result.data)
        self.assertEqual(archive.read_file("b.bin", 1), b"hello " * 3000)

    def test_stored_block_rule(self):
        self.assertEqual(XA.stored_block_length(100000, 0), (32768, False))
        self.assertEqual(XA.stored_block_length(100000, 10000), (32768 + 6384, False))
        self.assertEqual(XA.stored_block_length(20000, 3000), (17000, True))

    def test_cut_level_9_declares_what_inflates(self):
        stream = XA.deflate9(bytes(300000))
        stored = stream[:40]
        declared = XA.declared_for_cut(stored)
        self.assertEqual(declared % 0x10000 == 0 or declared < len(zlib.decompressobj().decompress(stored)) + 1, True)
        self.assertLessEqual(declared, len(zlib.decompressobj().decompress(stored)))

    def test_rebuild_is_identical(self):
        for flags in (0, 2, 10):
            data = self.write(flags, encrypted={(1, 0)}).data
            result = XA.rebuild_x360(data)
            self.assertEqual(result.data, data, flags)
            self.assertEqual(result.kept_from_source, 0)

    def test_rebuild_keeps_packets_it_cannot_reproduce(self):
        other = lambda data: zlib.compress(data, 1)  # noqa: E731  a different deflater
        source = XA.write_x360_archive(pieces(), read_sector=0x800, compress=other).data
        result = XA.rebuild_x360(source)
        self.assertEqual(result.data, source)
        self.assertGreater(result.kept_from_source, 0)
        self.assertIn("kept from the source", result.notes[0])
        plain = XA.rebuild_x360(source, keep_source_packets=False)
        self.assertNotEqual(plain.data, source)
        self.assertIn("not byte-identical", plain.notes[-1])
        self.assertTrue(C.verify_archive(plain.data).ok)

    def test_zlib_is_reported(self):
        is_ng = hasattr(zlib, "ZLIBNG_VERSION") or "ng" in zlib.ZLIB_RUNTIME_VERSION
        self.assertEqual(XA.ZLIB_IS_CANONICAL, not is_ng)
        self.assertIn("zlib", XA.zlib_implementation())

    def test_only_big_endian_v7(self):
        with self.assertRaises(FormatError):
            XA.rebuild_x360(C.write_archive({"a.bin": b"x" * 10}))


# --- G1: geometry -------------------------------------------------------------------------------

class GeoSets(unittest.TestCase):
    def test_names(self):
        sub = P.SubObject(0, None, None, (0,) * 3, (1,) * 3, None, None, [], geo_set=P.GEOSET_LOWPOLY)
        self.assertEqual(sub.label, "low-poly set, sub-object 0")

    def test_zero_bounds_skip_in_both_sets(self):
        data = B.build_app_switch(sub_objects=[{"rigid": TRIANGLE, "index_count": 3, "zero_bounds": True}],
                                  second=[{"rigid": TRIANGLE, "index_count": 3, "zero_bounds": True}])
        app = P.parse_appearance(data, strict=True)
        self.assertEqual(len(app.notes), 2)
        self.assertTrue(app.notes[0].startswith("high-poly set"))

    def test_zero_bounds_vertex_at_origin_no_note(self):
        sub = P.SubObject(0, None, None, (0.0,) * 3, (0.5,) * 3, (0.0,) * 3, (0.0,) * 3,
                          [P.Draw(False, [(0.0, 0.0, 0.0)], [], [], [], [], [], [])], bounds_raw=bytes(24))
        warnings, notes = [], []
        P._check_bounds(sub, warnings, notes)
        self.assertEqual((warnings, notes), ([], []))

    def test_low_poly_set_bounds_checked_when_stored(self):
        data = B.build_app_switch(second=[{"rigid": TRIANGLE, "index_count": 3, "half": (1.0, 1.0, 1.0)}])
        self.assertIn("low-poly set, sub-object 0: vertex outside bounds", P.parse_appearance(data).warnings)

    def test_ps3_low_poly(self):
        data = B.build_app_ps3(second=[{"zero_bounds": True}])
        app = P.parse_appearance(data, strict=True)
        self.assertEqual(app.extra_sub_objects[0].geo_set, P.GEOSET_LOWPOLY)
        self.assertEqual(app.notes, ["low-poly set, sub-object 0: bounds not stored (all zero), "
                                     "check skipped (G1-1)"])

    def test_export_keeps_sets_apart(self):
        app = P.parse_appearance(B.build_app_switch(
            second=[{"rigid": TRIANGLE, "index_count": 3}]))
        text = P.to_obj(app)
        self.assertEqual(text.count("\no "), 2)
        self.assertIn("o lowpoly_subobject_0", text)
        self.assertEqual(P.to_obj(app, low_poly=False).count("\no "), 1)
        nodes = [n["name"] for n in P.to_gltf(app)["nodes"]]
        self.assertEqual(nodes, ["subobject_0", "lowpoly_subobject_0"])
        self.assertEqual([n["name"] for n in P.to_gltf(app, low_poly=False)["nodes"]], ["subobject_0"])


class FatVertices(unittest.TestCase):
    def cloth_only(self, order="<", **spec):
        version = 302 if order == ">" else 306
        return P.parse_appearance(B.build_app_switch(version=version, order=order, sub_objects=[
            {"index_count": 3, "cloth": True, **spec}]))

    def test_read(self):
        for order in ("<", ">"):
            app = self.cloth_only(order)
            self.assertEqual(app.warnings, [], order)
            draw = app.sub_objects[0].draws[0]
            self.assertEqual(draw.kind, "fat")
            self.assertEqual(draw.positions, B.TRIANGLE_FAT)  # used as is
            self.assertEqual(draw.indices, [0, 1, 2])
            self.assertEqual(draw.normals[0], (0.5, 0.0, -0.5))
            self.assertEqual(draw.colours_1[0], bytes((1, 2, 3, 4)))  # same bytes on x360
            self.assertEqual(draw.uv1[0], (1.0, 0.0))
            self.assertIs(app.sub_objects[0].fat_draw, draw)

    def test_pool_size(self):
        data = bytearray(B.build_app_switch(sub_objects=[{"index_count": 3, "cloth": True}]))
        struct.pack_into("<I", data, 0x22C, 64)  # vertex pool of 2 fat vertices for 3
        warnings = P.parse_appearance(bytes(data)).warnings
        self.assertTrue(any("vertex pool" in w for w in warnings))

    def test_running_sums_over_both_sets(self):
        tri2 = [(1.0, 1.0, 1.0), (2.0, 1.0, 1.0), (1.0, 2.0, 1.0)]
        data = B.build_app_switch(
            sub_objects=[{"index_count": 3, "cloth": True}],
            second=[{"index_count": 3, "cloth": True, "fat": tri2}])
        app = P.parse_appearance(data, strict=True)
        self.assertEqual(app.extra_sub_objects[0].draws[0].positions, tri2)

    def test_index_sum_counts_every_sub_object(self):
        data = B.build_app_switch(sub_objects=[
            {"rigid": TRIANGLE, "index_count": 3},
            {"index_count": 3, "cloth": True, "indices": (2, 1, 0)}])
        app = P.parse_appearance(data, strict=True)
        self.assertEqual(app.sub_objects[1].draws[0].indices, [2, 1, 0])

    def test_every_fat_vertex_used(self):
        quad = B.TRIANGLE_FAT + [(5.0, 5.0, 0.0)]
        app = P.parse_appearance(B.build_app_switch(sub_objects=[
            {"index_count": 3, "fat": quad, "flags": 1 << 8}]))  # fat vertices without cloth
        self.assertTrue(any("not used" in w for w in app.warnings))

    def test_skinned_cloth_keeps_both(self):
        skin = {"palette": [0], "vertices": TRIANGLE, "indices": [0, 1, 2], "weights": (255, 0, 0, 0)}
        bones = [(0x10, -1, (0.0, 0.0, 0.0))]
        app = P.parse_appearance(B.build_app_switch(bones=bones, sub_objects=[
            {"skin": skin, "cloth": True, "index_count": 3}]), strict=True)
        sub = app.sub_objects[0]
        self.assertEqual([d.kind for d in sub.draws], ["packed"])  # export the skinned copy
        self.assertEqual(sub.fat_draw.kind, "fat")
        bad = P.parse_appearance(B.build_app_switch(bones=bones, sub_objects=[
            {"skin": skin, "cloth": True, "index_count": 3, "fat": B.TRIANGLE_FAT * 2,
             "indices": (0, 1, 2)}]))
        self.assertTrue(any("skinning record counts" in w for w in bad.warnings))

    def test_ps3_fat_and_pools(self):
        fat = [(0.0, 0.0, 0.0), (4.0, 0.0, 0.0), (0.0, 4.0, 0.0)]
        app = P.parse_appearance(B.build_app_ps3(fat=fat, cloth=True), strict=True)
        sub = app.sub_objects[0]
        self.assertEqual((sub.cloth_vertex_count, sub.index_count), (3, 3))
        self.assertEqual(sub.draws[0].positions, fat)
        self.assertEqual(len(sub.cloth.vertices), 3)
        data = bytearray(B.build_app_ps3(fat=fat, cloth=True))
        struct.pack_into(">I", data, 0x234, 8)  # index pool of 4 indices for 3
        self.assertTrue(any("index pool" in w for w in P.parse_appearance(bytes(data)).warnings))


class Cloth(unittest.TestCase):
    def build(self, mutate=None, pins=(0,), extra_faces=0, face_count=None):
        fat = [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (0.0, 10.0, 2.0),
               (10.0, 10.0, 0.0), (20.0, 0.0, 5.0), (20.0, 10.0, 0.0)]
        tris = [(0, 1, 2), (1, 3, 2), (1, 4, 3), (4, 5, 3)]
        arrays = B.cloth_arrays(fat, tris, pins)
        if mutate:
            mutate(arrays)
        flat = [i for t in tris for i in t]
        return P.parse_appearance(_switch_with_cloth(
            fat, flat, lambda img: B.cloth_record(img, "switch", arrays, extra_faces, face_count)))

    def test_consistent_record_passes(self):
        app = self.build()
        self.assertEqual(app.warnings, [])
        cloth = app.sub_objects[0].cloth
        self.assertEqual((cloth.vertex_count, len(cloth.faces), cloth.staple_count), (6, 4, 1))
        self.assertEqual(CL.model_position(cloth.vertices[2].wp), (0.0, 10.0, 2.0))

    def test_fewer_faces_counted_than_stored(self):
        app = self.build(extra_faces=2)
        self.assertEqual(app.warnings, [])
        self.assertEqual(app.sub_objects[0].cloth.face_count, 4)
        self.assertEqual(len(app.sub_objects[0].cloth.faces), 6)

    def test_rules(self):
        def set_vertex(index, **changes):
            def mutate(arrays):
                v = list(arrays["vertices"][index])
                names = ["wp", "wp0", "inv_mass", "parent", "attach", "fat", "island"]
                for name, value in changes.items():
                    v[names.index(name)] = value
                arrays["vertices"][index] = tuple(v)
            return mutate

        cases = {
            "wp0": set_vertex(1, wp0=(9.0, 9.0, 9.0)),
            "inverse mass": set_vertex(0, inv_mass=1.0),
            "parent": set_vertex(1, parent=-1),
            "attachment distance": set_vertex(2, attach=0),
            "nFatVertexIndex": set_vertex(3, fat=0),
            "island": set_vertex(4, island=3),
            "vertex map": lambda a: a.__setitem__("mapping", a["mapping"][:-1]),
            "not cloth vertex": lambda a: a.__setitem__("mapping", [0, 1, 2, 3, 5, 4]),
            "rest area": lambda a: a["faces"].__setitem__(0, (99.0, a["faces"][0][1])),
            "flArea": lambda a: a.__setitem__("area", 1.0),
            "rest length": lambda a: a["distance"].__setitem__(0, (a["distance"][0][0], 7.0, a["distance"][0][2])),
            "flHorizontal": lambda a: a["distance"].__setitem__(
                0, (a["distance"][0][0], a["distance"][0][1], 1.0 - a["distance"][0][2])),
            "face edges": lambda a: a["distance"].pop(),
            "bending": lambda a: a["bending"].append(((0, 3), 1.0, 1.0)),
            "staple weights": lambda a: a["staples"].__setitem__(0, (0, (0, 0, 0), (0.0, 1.0, 0.0))),
        }
        for name, mutate in cases.items():
            warnings = self.build(mutate).warnings
            self.assertTrue(warnings, name)

    def test_unchecked_rules(self):
        # The attachment step and the parent being a neighbour have exceptions: not checked.
        def mutate(arrays):
            v = list(arrays["vertices"][5])
            v[3], v[4] = 0, 7
            arrays["vertices"][5] = tuple(v)
        self.assertEqual(self.build(mutate).warnings, [])

    def test_record_size(self):
        app = P.parse_appearance(B.build_app_switch(sub_objects=[
            {"index_count": 3, "cloth": bytes(8)}]))
        self.assertTrue(any("cloth record not read" in w for w in app.warnings))

    def test_components(self):
        self.assertEqual(CL.components(5, [(0, 1), (3, 4)]), [0, 0, 1, 2, 2])


def _switch_with_cloth(fat, flat, make_cloth):
    """A Switch appearance with one cloth-only sub-object whose cloth record is built in place."""
    img = B.Image(0x400, "<")
    img.raw(0, B.sno_head(306, "<"))
    record = bytearray(240)
    rec = B.Image(0, "<")
    rec.b = record
    rec.put(0x00, "I", 1 << 8)
    rec.put(0x04, "I", len(fat))
    rec.put(0x28, "i", len(flat))
    rec.put(0x40, "II", *img.append(make_cloth(img)))
    rec.put(0x6C, "6f", 0.0, 0.0, 0.0, 1000.0, 1000.0, 1000.0)
    rec.put(0xD8, "3f", 1.0, 2.0, 3.0)
    rec.put(0xE4, "3f", 0.5, 0.5, 0.5)
    img.pair(0x220, struct.pack(f"<{len(flat)}H", *flat))
    img.pair(0x228, b"".join(B.fat_vertex(v) for v in fat))
    img.put(0x70, "i", 1)
    img.pair(0x74, bytes(record))
    return img.done()


class Ps3Draws(unittest.TestCase):
    def test_built_draws_pass(self):
        for skinned in (False, True):
            bones = [(0x10, -1, (0.0, 0.0, 0.0)), (0x11, 0, (1.0, 0.0, 0.0)),
                     (0x12, 1, (2.0, 0.0, 0.0))] if skinned else ()
            app = P.parse_appearance(B.build_app_ps3(skinned=skinned, bones=bones), strict=True)
            self.assertEqual(app.warnings, [])

    def test_each_rule(self):
        cases = {
            0x20: ("HH", 0, 0),
            0x34: ("H", 2),
            0x36: ("H", 1),
            0x5C: ("HH", 4, 0),
            0x98: ("I", 100),
            0x9C: ("I", 7),
            0x00: ("I", 1),
            0x04: ("I", 1),
            0x14: ("I", 1),
            0x28: ("I", 1),
            0x68: ("I", 1),
            0x74: ("I", 1),
            0x8C: ("I", 1),
        }
        for at, fix in cases.items():
            warnings = P.parse_appearance(B.build_app_ps3(draw_fix={at: fix})).warnings
            self.assertTrue(any("(G1-5)" in w for w in warnings), hex(at))
        data = bytearray(B.build_app_ps3())
        descriptor = data.find(bytes((0x81, 8)) + bytes.fromhex("FF0000FF1000"))
        data[descriptor + 13] = 0
        self.assertTrue(any("descriptor" in w for w in P.parse_appearance(bytes(data)).warnings))

    def test_format_blocks(self):
        self.assertEqual(P.PS3_FORMAT_BLOCKS[P.FORMAT_RIGID][0][1], 6)
        self.assertEqual(P.PS3_FORMAT_BLOCKS[P.FORMAT_SKINNED][0][1], 10)
        self.assertEqual([len(b) for b in P.PS3_FORMAT_BLOCKS[P.FORMAT_RIGID]], [16, 16])
        self.assertEqual([len(b) for b in P.PS3_FORMAT_BLOCKS[P.FORMAT_SKINNED]], [32, 32])
        data = bytearray(B.build_app_ps3())
        at = data.find(P.PS3_FORMAT_BLOCKS[P.FORMAT_RIGID][1])
        data[at + 1] = 0x10
        self.assertTrue(any("block B" in w for w in P.parse_appearance(bytes(data)).warnings))

    def test_writer_keeps_the_stored_values(self):
        data = B.build_app_ps3()
        self.assertEqual(WR.write_app(P.parse_appearance(data)), data)


if __name__ == "__main__":
    unittest.main()
