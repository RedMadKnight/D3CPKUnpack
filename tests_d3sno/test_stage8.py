"""Stage 8: the x360 build (``17_x360.md``) and the corrections after stage 7 (``18_corrections7.md``).

There are no x360 files here: every x360 file is synthetic, built from the layouts.
"""

import struct
import unittest

import builders as B
import test_ui_font as UF
from d3sno import FormatError
from d3sno import animation as A
from d3sno import appearance as P
from d3sno import archive as C
from d3sno import balance as G
from d3sno import binary as Y
from d3sno import enums as E
from d3sno import formula as F
from d3sno import index as I
from d3sno import luachunk as L
from d3sno import navigation as N
from d3sno import power as PW
from d3sno import presentation as PR
from d3sno import records as R
from d3sno import scripts as S
from d3sno import sno
from d3sno import structure as ST
from d3sno import tags as T
from d3sno import texture as TX
from d3sno import ui_font as U
from d3sno import world as W
from d3sno import write as WR
from d3sno.datafiles import DATA

TRIANGLE = [(0, 0, 0), (10, 0, 0), (0, 10, 0)]


def header(version, order=">", size=0x40):
    return B.sno_head(version, order) + bytes(size - 0x1C)


# --- X1, X2: the build and how it is found ----------------------------------------------------

class Build(unittest.TestCase):
    def test_third_build_is_big_endian(self):
        self.assertEqual(Y.BUILDS, ("ps3", "switch", "x360"))
        self.assertEqual(Y.build_order("x360"), ">")
        self.assertEqual(Y.layout_build("x360"), "switch")
        self.assertEqual(Y.layout_build("ps3"), "ps3")

    def test_per_build_tables_take_switch_for_x360(self):
        table = Y.ByBuild({"ps3": 1, "switch": 2})
        self.assertEqual(table["x360"], 2)
        with self.assertRaises(KeyError):
            table["pc"]

    def test_versions_from_the_table(self):
        self.assertEqual(len(sno.X360_VERSIONS), 64)
        self.assertEqual(sno.expected_version("app", "x360"), 302)
        self.assertEqual(sno.expected_version("gam", "x360"), 2288)
        self.assertEqual(sno.expected_version("app", "ps3"), 295)
        self.assertEqual(sno.expected_version("dfn", "x360"), 26)
        self.assertTrue(sno.check_version("app", 302, "x360"))
        with self.assertRaises(FormatError):
            sno.check_version("app", 306, "x360", strict=True)

    def test_detection_rules(self):
        # 1. little-endian signature: switch
        self.assertEqual(sno.resolve_build("app", sno.parse_header(header(306, "<"))), "switch")
        # 2. big-endian: the version decides
        self.assertEqual(sno.resolve_build("app", sno.parse_header(header(302))), "x360")
        self.assertEqual(sno.resolve_build("app", sno.parse_header(header(295))), "ps3")
        # 3. same version on ps3 and x360: the archive's or the caller's build
        shared = sno.parse_header(header(25))
        self.assertIsNone(sno.detect_build("adv", shared))
        self.assertEqual(sno.detect_build("adv", shared, "x360"), "x360")
        self.assertEqual(sno.resolve_build("adv", shared, fallback="x360"), "x360")
        self.assertEqual(sno.resolve_build("adv", shared), "ps3")
        self.assertEqual(sno.resolve_build("adv", shared, "x360"), "x360")

    def test_every_shared_version_is_listed(self):
        shared = {e for e, v in sno.X360_VERSIONS.items() if sno.VERSIONS.get(e, (None,))[0] == v}
        self.assertEqual(shared, {"aco", "adv", "ams", "flg", "lor", "scg", "stl"})

    def test_archive_gives_the_build(self):
        big = C.read_header(C.write_archive({"a.bin": b"x" * 40}, version=7, order=">"))
        little = C.read_header(C.write_archive({"a.bin": b"x" * 40}))
        old = C.read_header(C.write_archive({"a.bin": b"x" * 40}, version=6))
        self.assertEqual((big.build, little.build, old.build), ("x360", "switch", "ps3"))
        adv = header(25)
        self.assertEqual(C.file_build(big, adv, "adv"), "x360")
        self.assertEqual(C.file_build(old, adv, "adv"), "ps3")
        self.assertEqual(C.file_build(old, header(302), "app"), "x360")  # the version wins

    def test_caller_build_is_kept_on_the_header(self):
        self.assertEqual(sno.parse_header(header(1), "x360").build, "x360")
        self.assertEqual(sno.parse_header(header(1)).build, "ps3")

    def test_tables_ship_as_package_data(self):
        B.need_layout_tables()  # ``42`` V27-3: the layout tables hidden, as the check of V27-3 runs
        for name in ("gam_tables", "gam_pairs", "struct_fields", "struct_diff_list", "typed_all",
                     "layouts", "cov_layouts", "byte_order", "sno_versions", "names", "attributes"):
            self.assertTrue((DATA / f"x360_{name}.tsv").is_file(), name)


# --- X3: records -------------------------------------------------------------------------

class Records(unittest.TestCase):
    SIZES = {"actor": 440, "monster": 1360, "quest": 272, "quest_step": 168,
             "treasure_class": 40, "drop_modifier": 120, "condition": 1188}

    def test_seven_records_of_their_own(self):
        for record, size in self.SIZES.items():
            self.assertEqual(R.layout("x360", record).size, size, record)
            self.assertEqual(R.layout("x360", record).build, "x360")
        self.assertEqual(R.layout("x360", "recipe"), R.layout("switch", "recipe"))

    def test_every_type_reads_strictly(self):
        for extension, record in R.RECORDS_BY_EXTENSION.items():
            data = B.build_record("x360", record, {}, version=sno.X360_VERSIONS[extension])
            parsed = R.parse_record_file(data, extension, strict=True)
            self.assertEqual(parsed.build, "x360", extension)

    def test_moved_field(self):
        # The socketed image is at 0x190 on x360, 0x198 on Switch.
        data = B.build_record("x360", "actor", {"self_handle": 77, "socketed_image": 9}, version=282)
        self.assertEqual(struct.unpack_from(">i", data, 0x10 + 0x190)[0], 9)
        parsed = R.parse_record_file(data, "acr")
        self.assertEqual((parsed.record["self_handle"], parsed.record["socketed_image"]), (77, 9))

    def test_switch_fields_absent_on_x360(self):
        names = {f.name for f in R.layout("x360", "treasure_class").fields}
        self.assertNotIn("ancient_override", names)
        self.assertIn("ancient_override", {f.name for f in R.layout("switch", "treasure_class").fields})

    def test_struct_tables(self):
        import csv
        with open(DATA / "x360_struct_diff_list.tsv", encoding="utf-8", newline="") as handle:
            self.assertEqual(len(list(csv.DictReader(handle, delimiter="\t"))), 61)
        with open(DATA / "x360_typed_all.tsv", encoding="utf-8", newline="") as handle:
            self.assertEqual(len({row["struct"] for row in csv.DictReader(handle, delimiter="\t")}), 407)


# --- X4: scene ---------------------------------------------------------------------------

class Scene(unittest.TestCase):
    def test_handles_at_the_ps3_positions(self):
        info = W.parse_scene_info(B.build_scene_info("x360", appearance=11, collision=12), strict=True)
        self.assertEqual((info.build, info.appearance, info.collision_mesh, info.xp_density),
                         ("x360", 11, 12, None))

    def test_bytes_0x208_must_be_zero(self):
        data = bytearray(B.build_scene_info("x360"))
        data[0x10 + 0x20C] = 1
        self.assertTrue(W.parse_scene_info(bytes(data)).warnings)
        with self.assertRaises(FormatError):
            W.parse_scene_info(bytes(data), strict=True)

    def test_sections_from_the_nav(self):
        scn, nav = B.build_scene("x360")
        x360 = N.parse_scene(scn, nav=nav, strict=True)
        switch = N.parse_scene(B.build_scene("switch")[0], strict=True)
        self.assertEqual(x360.build, "x360")
        self.assertEqual(x360.special_flags, switch.special_flags)  # read as on Switch
        self.assertEqual(len(x360.cells), len(switch.cells))
        self.assertFalse(N.parse_scene(scn).has_sections)

    def test_section_pairs_must_be_empty(self):
        scn, nav = B.build_scene("x360")
        data = bytearray(scn)
        struct.pack_into(">II", data, 0x10 + 0x80, 0, 4)
        with self.assertRaises(FormatError):
            N.parse_scene(bytes(data), nav=nav, strict=True)

    def test_table_layout(self):
        scene = {f.name: f for f in ST.layout("x360", "SceneDefinition").fields}
        self.assertEqual((scene["u208"].kind, scene["snoAppearance"].offset, scene["snoPhysMesh"].offset),
                         ("zero", 0x210, 0x214))
        self.assertEqual(ST.layout("x360", "SceneDefinition").size, 536)
        self.assertIs(ST.layout("x360", "NavZoneDefinition"), ST.layout("switch", "NavZoneDefinition"))


# --- X5: animation, texture, sound ---------------------------------------------------------

class Banks(unittest.TestCase):
    def test_ani_is_the_ps3_form(self):
        perm = B.simple_permutation("ps3")
        ani = A.parse_ani(B.build_ani("x360", [perm]), can=B.build_can([perm]))
        self.assertEqual((ani.build, len(ani.permutations)), ("x360", 1))
        self.assertTrue(ani.permutations[0].has_data)
        self.assertEqual(ST.layout("x360", "AnimationDefinition").size, 64)

    def test_texture_uses_the_switch_record(self):
        data = B.build_tex("x360", 0, 4, 4, b"", payload=b"\x11" * 32, v_scale=0.25)
        tex = TX.parse_texture(data, strict=True)
        self.assertEqual((tex.build, tex.lightmap_v_scale, tex.nv), ("x360", 0.25, None))
        self.assertEqual(tex.payload, b"\x11" * 32)
        with self.assertRaises(FormatError):
            TX.decode_image(tex)

    def test_texture_writer_is_the_switch_rule(self):
        data = B.build_tex("x360", 0, 4, 4, b"", payload=b"\x11" * 32, frames=[(0, 0.0, 0.0, 1.0, 1.0)])
        self.assertEqual(WR.write_tex(TX.parse_texture(data)), data)

    def test_sound_bank_version(self):
        self.assertEqual(sno.resolve_build("sbk", sno.parse_header(header(57))), "x360")
        self.assertEqual(sno.resolve_build("sbk", sno.parse_header(header(46))), "ps3")

    def test_fsb5_stays_little_endian(self):
        bank = PR.parse_fsb5(B.fsb5(15, bytes(16)))
        self.assertEqual(bank.mode, 15)


# --- X6: appearance vertices ---------------------------------------------------------------

def app(order, **kw):
    version = 302 if order == ">" else 306
    return P.parse_appearance(B.build_app_switch(version=version, order=order, **kw))


class Vertices(unittest.TestCase):
    def test_rigid_vertices_equal_switch(self):
        x360, switch = app(">"), app("<")
        self.assertEqual(x360.build, "x360")
        a, b = x360.sub_objects[0].draws[0], switch.sub_objects[0].draws[0]
        self.assertEqual(a.positions, b.positions)
        self.assertEqual(a.normals, b.normals)
        self.assertEqual(a.uv1, b.uv1)
        self.assertEqual(a.uv2_raw, b.uv2_raw)
        self.assertEqual(a.colours_1, b.colours_1)  # reversed on disk, Switch order once read
        self.assertEqual(a.colours_2, b.colours_2)
        self.assertEqual(a.indices, b.indices)

    def test_colour_bytes_are_reversed_on_disk(self):
        raw = B.switch_vertex((1, 2, 3), order=">")
        self.assertEqual(raw[:6], struct.pack(">3h", 1, 2, 3))
        self.assertEqual(raw[12:20], bytes((4, 3, 2, 1, 8, 7, 6, 5)))
        self.assertEqual(raw[20:28], B.switch_vertex((1, 2, 3))[20:28])  # UV little-endian

    def test_skinned_vertices(self):
        bones = [(0x10, -1, (0.0, 0.0, 0.0)), (0x11, 0, (1.0, 0.0, 0.0))]
        skin = {"palette": [0, 1], "vertices": TRIANGLE, "indices": [0, 1, 2],
                "slots": (0, 1, 0, 0), "weights": (200, 55, 0, 0)}
        x360 = app(">", bones=bones, sub_objects=[{"skin": skin}])
        switch = app("<", bones=bones, sub_objects=[{"skin": skin}])
        a, b = x360.sub_objects[0].draws[0], switch.sub_objects[0].draws[0]
        self.assertEqual(x360.warnings, [])
        # V8-X6: weights may differ from Switch by one rounding unit; never compared across builds.
        self.assertEqual((a.palette, a.joints), (b.palette, b.joints))
        self.assertEqual(a.weights[0], (200, 55, 0, 0))
        self.assertEqual(a.positions, b.positions)
        self.assertEqual(x360.bones[1].rest.t, switch.bones[1].rest.t)

    def test_writer_copies(self):
        data = B.build_app_switch(version=302, order=">")
        self.assertEqual(WR.write_app(P.parse_appearance(data)), data)

    def test_main_list_bounds_is_a_warning(self):
        parsed = app(">", sub_objects=[{"rigid": TRIANGLE, "index_count": 3, "half": (1.0, 1.0, 1.0)}])
        self.assertIn("high-poly set, sub-object 0: vertex outside bounds", parsed.warnings)


# --- V7-T13, V7-T11: second list and the cloth vertex count ------------------------------------

class SecondList(unittest.TestCase):
    def test_second_list_is_not_bounds_checked(self):
        # Stage 10 (G1-1): the skip follows all-zero bounds, not the list.
        for order in ("<", ">"):
            version = 302 if order == ">" else 306
            data = B.build_app_switch(version=version, order=order, second=[
                {"rigid": TRIANGLE, "index_count": 3, "zero_bounds": True},
            ])
            parsed = P.parse_appearance(data, strict=True)
            self.assertEqual(len(parsed.extra_sub_objects), 1)
            self.assertEqual(parsed.extra_sub_objects[0].list_name, "low-poly set")
            self.assertEqual(parsed.notes, ["low-poly set, sub-object 0: bounds not stored (all zero), "
                                            "check skipped (G1-1)"])

    def test_main_list_rule_unchanged(self):
        data = B.build_app_switch(sub_objects=[{"rigid": TRIANGLE, "index_count": 3, "half": (1.0, 1.0, 1.0)}])
        with self.assertRaises(FormatError):
            P.parse_appearance(data, strict=True)

    def test_messages_name_the_list(self):
        data = B.build_app_switch(second=[{"index_count": 3}])
        parsed = P.parse_appearance(data)
        self.assertTrue(parsed.warnings[0].startswith("low-poly set, sub-object 0:"))
        self.assertEqual(P.SubObject(2, None, None, (0,) * 3, (1,) * 3, None, None, []).label,
                         "high-poly set, sub-object 2")

    def test_ps3_second_list(self):
        sub = P.SubObject(0, None, None, (0.0,) * 3, (1.0,) * 3, (0.0,) * 3, (0.0,) * 3,
                          [P.Draw(False, [(5.0, 0.0, 0.0)], [], [], [], [], [], [])],
                          geo_set=P.GEOSET_LOWPOLY, bounds_raw=bytes(24))
        warnings, notes = [], []
        P._check_bounds(sub, warnings, notes)
        self.assertEqual((warnings, len(notes)), ([], 1))

    def test_cloth_vertex_count(self):
        parsed = app("<", sub_objects=[{"index_count": 3, "cloth": True}])
        self.assertEqual(parsed.sub_objects[0].cloth_vertex_count, 3)
        self.assertEqual(parsed.warnings, [])
        for spec in (
            {"index_count": 3, "cloth": True, "cloth_vertices": 0},  # cloth-only, word 0
            {"rigid": TRIANGLE, "index_count": 3, "cloth_vertices": 5},  # no cloth, word set
            {"rigid": TRIANGLE, "index_count": 3, "cloth": True, "cloth_vertices": 4},  # packed rigid
        ):
            parsed = app("<", sub_objects=[spec])
            self.assertTrue(any("+0x04" in w for w in parsed.warnings), spec)

    def test_skinned_cloth_counts(self):
        skin = {"palette": [0], "vertices": TRIANGLE, "indices": [0, 1, 2], "weights": (255, 0, 0, 0)}
        parsed = app("<", bones=[(0x10, -1, (0.0, 0.0, 0.0))],
                     sub_objects=[{"skin": skin, "cloth": True, "index_count": 3}])
        self.assertEqual(parsed.sub_objects[0].cloth_vertex_count, 3)
        self.assertEqual(parsed.warnings, [])


# --- X7: balance files ---------------------------------------------------------------------

def xcode(value=2):
    return B.Code(">").f(value).end()


class Balance(unittest.TestCase):
    def test_tables(self):
        tables = G.x360_gam_tables()
        self.assertEqual(len(tables), 37)
        self.assertEqual((tables[38].slot, tables[38].row_size), (0x1E8, 8))
        self.assertEqual((tables[25].slot, tables[25].name), (0x128, "item_drops"))
        self.assertEqual(max(t.slot for t in tables.values()) + 16, G.FIXED_END["x360"])
        self.assertEqual(G.FIXED_END["x360"], 0x278)

    def test_x360_only_type(self):
        rows = [{"fields": {0: ("i", 5), 4: ("f", 1.5)}}, {"fields": {0: ("i", 6)}}]
        data = B.build_gam_layout("x360", 38, 0x1E8, 8, rows, {})
        parsed = G.parse_balance(data, "x360")
        self.assertEqual((parsed.table, parsed.row_size, len(parsed.rows)), ("item_durability_cost_scalars", 8, 2))
        self.assertEqual(parsed.type_name, "Item Durability Cost Scalars")
        self.assertEqual(G.balance_type_name(43, "x360"), "Made To Order Affix Folders")
        self.assertIsNone(G.balance_type_name(43, "switch"))

    def test_version_finds_x360(self):
        data = bytearray(B.build_gam_layout("x360", 38, 0x1E8, 8, [{}], {}))
        struct.pack_into(">I", data, 4, 2288)
        self.assertEqual(G.parse_balance(bytes(data)).build, "x360")

    def test_item_rows_rebuild(self):
        element = struct.pack(">4i", 1, 2, 3, 4)
        rows = [
            {"name": "Itm", "arrays": {0xF0: (24, [(element, {0x10: xcode(5)}), (element, {0x10: None})])}},
            {"name": "Plain"},
            {"name": "Abcdefghi"},
        ]
        data = B.build_gam_layout("x360", 2, 0x38, 656, rows, {0: "name", 0xF0: "array"})
        parsed = G.parse_balance(data, "x360")
        self.assertEqual([r.name for r in parsed.rows], ["Itm", "Plain", "Abcdefghi"])
        self.assertEqual(parsed.rows[0].blocks[0xF0].elements[0][0x10].data, xcode(5))
        self.assertEqual(WR.write(parsed), data)
        self.assertEqual(struct.unpack_from(">I", data, 0x38)[0] + 0x10, 0x278)  # rows follow the fixed part

    def test_names_align_to_8(self):
        data = B.build_gam_layout("x360", 2, 0x38, 656, [{"name": "Ab"}, {"name": "Cdefghijk"}], {0: "name"})
        parsed = G.parse_balance(data, "x360")
        for row in parsed.rows:
            self.assertEqual(row.blocks[0].source % 8, 0)
        self.assertEqual(WR.write(parsed), data)

    def test_set_bonus_code_pairs(self):
        offsets = [0x30, 0x48, 0x60, 0x78, 0x90, 0xA8, 0xC0, 0xD8]
        self.assertEqual([s.offset for s in G.gam_pair_slots("x360", 33) if s.kind == "code"], offsets)
        rows = [{"name": "Set", "blocks": {0x30: xcode(1)}}]
        data = B.build_gam_layout("x360", 33, 0x188, 224, rows, {0: "name", **{o: "code" for o in offsets}})
        self.assertEqual(WR.write(G.parse_balance(data, "x360")), data)


# --- X8: scripts, formulas, tags -----------------------------------------------------------

class Scripts(unittest.TestCase):
    def test_lua_header_is_the_ps3_one(self):
        chunk = B.lua_chunk("x360")
        self.assertEqual(tuple(chunk[6:12]), (0, 4, 4, 4, 8, 0))
        self.assertEqual(L.parse_chunk(chunk, "x360").trailing_bytes, 0)
        with self.assertRaises(FormatError):
            L.parse_chunk(chunk, "switch")

    def test_script_at_the_switch_position(self):
        chunk = B.lua_chunk("x360")
        power = PW.parse_power(B.build_power("x360", script=chunk), strict=True)
        self.assertEqual((power.build, power.script), ("x360", chunk))
        self.assertEqual(L.script_slice(B.build_power("x360", script=chunk), "pow"), chunk)
        script = S.parse_script(B.build_scr("x360", chunk=chunk))
        self.assertEqual(script.build, "x360")

    def test_formula_words_big_endian(self):
        code = B.Code(">").f(2.5).end()
        self.assertEqual([i.name for i in F.decode(code, "x360")][-1], "END")
        self.assertEqual(F.decode(code, "x360")[0].operands, F.decode(code, "ps3")[0].operands)

    def test_four_power_tags_trust_the_stored_type(self):
        self.assertEqual(T.RETYPED_TAGS["x360"], frozenset({0x50210, 0x50230, 0x502F0, 0x50335}))
        power = PW.parse_power(B.build_power("x360", general=[(0, 0x502F0, 3), (0, 0x50335, 1)]))
        self.assertEqual(T.power_map_problems(power), [])
        power = PW.parse_power(B.build_power("switch", general=[(0, 0x502F0, 3)]))
        self.assertTrue(T.power_map_problems(power))

    def test_switch_numbering(self):
        self.assertEqual(S.switch_event_id(17, "x360"), 17)
        self.assertEqual(F.first_table_class("x360"), F.first_table_class("switch"))
        self.assertEqual(E.CLASS_COUNTS["x360"], 7)


# --- X9: index files -----------------------------------------------------------------------

def x360_toc(groups, totals=None, highest=0):
    """``groups``: {group: [(handle, name)]}. Blocks start on 4-byte boundaries."""
    counts, offsets = [0] * 70, [0] * 70
    blocks = bytearray()
    for group in sorted(groups):
        blocks += bytes(-len(blocks) % 4)
        offsets[group] = len(blocks)
        entries, names = bytearray(), bytearray()
        for handle, name in groups[group]:
            entries += struct.pack(">iii", group, handle, len(names))
            names += name.encode() + b"\0"
        counts[group] = len(groups[group])
        blocks += entries + names
    totals = totals or counts
    return struct.pack(">70i70i70iI", *counts, *offsets, *totals, highest) + bytes(blocks)


class Index(unittest.TestCase):
    def test_toc(self):
        data = x360_toc({9: [(5, "Barbarian"), (6, "Witch")], 33: [(100, "a_scene")]},
                        totals=[0] * 9 + [3] + [0] * 23 + [2] + [0] * 36, highest=777)
        toc = I.parse_core_toc(data, "x360")
        self.assertEqual(toc.build, "x360")
        self.assertEqual([(e.group, e.handle, e.name) for e in toc.entries],
                         [(9, 5, "Barbarian"), (9, 6, "Witch"), (33, 100, "a_scene")])
        self.assertEqual(toc.totals[9], 3)
        self.assertEqual(toc.problems(), [])  # counts and the header word are over all six files
        self.assertEqual(I.parse_core_toc(data).build, "x360")

    def test_toc_totals_over_six_files(self):
        one = I.parse_core_toc(x360_toc({9: [(5, "A")]}, totals=[0] * 9 + [2] + [0] * 60, highest=6), "x360")
        two = I.parse_core_toc(x360_toc({9: [(5, "A"), (6, "B")]}, totals=[0] * 9 + [2] + [0] * 60, highest=6), "x360")
        self.assertEqual(I.toc_totals_problems([one, two]), [])
        bad = I.parse_core_toc(x360_toc({9: [(5, "A")]}, totals=[0] * 9 + [3] + [0] * 60, highest=6), "x360")
        self.assertTrue(I.toc_totals_problems([bad, two]))

    def test_descriptor_sizes_by_magic(self):
        entry = b"Actor".ljust(64, b"\0") + struct.pack(">I", 440) + bytes(1024)
        found = I.parse_descriptor_sizes(struct.pack(">I", I.DESCRIPTOR_MAGIC) + entry)
        self.assertEqual((found[0].name, found[0].size), ("Actor", 440))
        little = b"Actor".ljust(64, b"\0") + struct.pack("<I", 448) + bytes(1024)
        self.assertEqual(I.parse_descriptor_sizes(struct.pack("<I", I.DESCRIPTOR_MAGIC) + little)[0].size, 448)

    def test_prefetch_with_the_build(self):
        data = struct.pack(">I4II2I", 1, 9, 5, 1, 0, 1, 33, 100)
        self.assertEqual(I.parse_prefetch(data, "x360").deps, [(33, 100)])


# --- X10: names ----------------------------------------------------------------------------

class Names(unittest.TestCase):
    def test_x360_attributes(self):
        table = R.x360_attributes()
        self.assertEqual(len(table), 1346)
        self.assertEqual(sum(1 for a in table.values() if a.switch_id == a.id), 39)
        self.assertEqual(R.attribute_name(1, "x360"), "Attribute_Timer")
        self.assertEqual(R.attribute_name(0, "x360"), "Axe_Bad_Data")
        self.assertIsNone(R.attribute_name(99999, "x360"))

    def test_record_enums_use_the_switch_numbering(self):
        data = B.build_record("x360", "actor", {"actor_kind": 1}, version=282)
        x360 = R.parse_record_file(data, "acr").record
        switch = R.parse_record_file(B.build_record("switch", "actor", {"actor_kind": 1}, 288), "acr").record
        self.assertEqual(x360.named()["actor_kind"], switch.named()["actor_kind"])
        self.assertIsNotNone(x360.named()["actor_kind"])


# --- X11: byte-order sites -----------------------------------------------------------------

class ByteOrder(unittest.TestCase):
    def test_table(self):
        import csv
        with open(DATA / "x360_byte_order.tsv", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        self.assertEqual(len(rows), 17)
        self.assertEqual(sum(1 for r in rows if r["status"] == "OPEN"), 1)

    def test_presentation_readers_follow_the_build(self):
        blob = struct.pack(">Iff", 1, 0.5, 0.25)
        self.assertEqual(PR.parse_texture_animation(blob, "x360").scale, (0.5, 0.25))
        with self.assertRaises(FormatError):
            PR.parse_texture_animation(blob, "switch")


# --- X12: archives -------------------------------------------------------------------------

class Archives(unittest.TestCase):
    def test_big_endian_version_7(self):
        data = C.write_archive({"a.bin": b"x" * 300, "b.bin": b"y" * 50}, order=">")
        report = C.verify_archive(data)
        self.assertTrue(report.ok, report.format())
        self.assertEqual(C.read_archive(data).read_file("a.bin"), b"x" * 300)

    def test_copies_on_big_endian_version_7(self):
        files = [C.InputFile("two.bin", [b"a" * 100, b"a" * 100]), C.InputFile("one.bin", [b"b" * 50])]
        self.assertTrue(C.verify_archive(C.write_archive(files, order=">")).ok)
        self.assertFalse(C.verify_archive(C.write_archive(files)).ok)

    def test_flag_10(self):
        data = C.write_archive({"a.bin": bytes(range(256)) * 40}, order=">", flags=10)
        self.assertTrue(C.verify_archive(data).ok)

    def test_header_order_still_checked_for_v6(self):
        h = C.read_header(C.write_archive({"a.bin": b"x"}, version=6))
        h.order = "<"
        self.assertTrue(C.header_problems(h))


# --- Corrections after stage 7 -------------------------------------------------------------

class Corrections(unittest.TestCase):
    def test_dsp_no_effect_is_named(self):
        self.assertEqual(E.geometry_value("DspEffect", "eDspType", -1).name, "no effect")

    def test_geometry_table_has_no_open_row(self):
        rows = [r for table in E.geometry_enums().values() for r in table.values()]
        self.assertNotIn("OPEN", {r.status for r in rows})
        # Stage 10 (G3-F): the 16 marker types 76..101 are SUPPORTED labels without a constant.
        markers = [r for r in rows if r.label.startswith("gizmo location") and not r.constant]
        self.assertEqual(len(markers), 16)
        self.assertEqual({r.status for r in markers}, {"SUPPORTED"})

    def test_font_offset_is_a_body_offset(self):
        img = UF.font()
        stored = struct.unpack_from(">I", img.b, 0x10 + 0xC8)[0]
        texture = struct.unpack_from(">I", img.b, 0x10 + 0xAC)[0] + 0x10
        self.assertEqual(stored, struct.unpack_from(">I", img.b, texture + 0x10)[0])
        self.assertEqual(U.verify_ps3_font(img.done()), [])


class UiAnimations(unittest.TestCase):
    def control(self, descriptions, int32s=2, floats=0, bindings=("^d_main.c.fade",)):
        return {
            "anim_int32s": [{"time": float(i), "value": i} for i in range(int32s)],
            "anim_floats": [{"time": 0.0, "value": 1.0} for _ in range(floats)],
            "anim_vector2s": [],
            "anim_descriptions": [{"type": t, "node_range": r} for t, r in descriptions],
            "anim_bindings": [{"animation_name": b, "property": 6} for b in bindings],
        }

    def test_ranges_cover_each_list_once(self):
        self.assertEqual(U.animation_problems(self.control([(0, (0, 1)), (0, (1, 2))])), [])
        self.assertEqual(U.animation_problems(self.control([(0, (0, 2)), (1, (0, 1))], floats=1)), [])

    def test_gap_or_overlap(self):
        self.assertTrue(U.animation_problems(self.control([(0, (0, 1))])))  # element 1 uncovered
        self.assertTrue(U.animation_problems(self.control([(0, (0, 2)), (0, (1, 2))])))  # overlap

    def test_range_inside_its_list(self):
        found = U.animation_problems(self.control([(0, (0, 3))]))
        self.assertTrue(any("outside" in p for p in found))
        found = U.animation_problems(self.control([(3, (0, 2))]))
        self.assertTrue(any("selects no value list" in p for p in found))

    def test_times_do_not_decrease(self):
        control = self.control([(0, (0, 2))])
        control["anim_int32s"][1]["time"] = -1.0
        self.assertTrue(any("decrease" in p for p in U.animation_problems(control)))

    def test_range_need_not_start_at_zero(self):
        control = self.control([(0, (0, 2))])
        control["anim_int32s"][0]["time"] = 0.5
        control["anim_int32s"][1]["time"] = 0.75
        self.assertEqual(U.animation_problems(control), [])

    def test_binding_prefix(self):
        found = U.animation_problems(self.control([(0, (0, 2))], bindings=("fade",)))
        self.assertTrue(any("^" in p for p in found))

    def test_dialog_reports_them(self):
        img = UF.dialog()
        self.assertEqual(U.verify_ps3_ui(img.done()), [])
        control = struct.unpack_from(">I", img.b, 0x10 + 0x14)[0] + 0x10
        desc = struct.unpack_from(">I", img.b, control + 0xE0 + 4)[0] + 0x10
        img.put(desc + 0x18, "iii", 0, 0, 1)
        self.assertTrue(any("exactly once" in p for p in U.verify_ps3_ui(img.done())))

    def test_empty_text_pair_still_accepted(self):
        img = UF.dialog()
        strings = struct.unpack_from(">I", img.b, 0x10 + 0x34)[0] + 0x10
        img.put(strings + 8, "II", 0, 0)
        read = U.parse_ps3_ui(img.done())
        self.assertIsNone(read.root["constant_strings"][0]["text"])
        self.assertFalse(any("NUL" in p for p in read.problems))


if __name__ == "__main__":
    unittest.main()
