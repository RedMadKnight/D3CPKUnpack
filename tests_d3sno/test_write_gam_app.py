import _libpath  # noqa: F401  (python -I: the library folder on sys.path)
import struct
import unittest

import builders as B
from d3sno import FormatError
from d3sno import appearance as P
from d3sno import balance as G
from d3sno import formula as F
from d3sno import write as W
from d3sno.transform import Transform

# Paragon bonuses (type 40): a name and four code pairs per row.
ROW = {"switch": (0x1E8, 400), "ps3": (0x3D8, 392)}
CODES = {"switch": (0x30, 0x48, 0x60, 0x78), "ps3": (0x28, 0x40, 0x58, 0x70)}


def code(build, value=2):
    return B.Code(">" if build == "ps3" else "<").f(value).end()


def slots(build):
    return {0: "name", **{offset: "code" for offset in CODES[build]}}


def rows(build):
    first = CODES[build][0]
    return [
        {"name": "Axe", "blocks": {first: code(build)}, "fields": {20: ("i", 7)}},
        {"name": "Sword"},
        {"name": "Mace", "blocks": {first: code(build, 3), CODES[build][3]: code(build, 4)}},
    ]


def gam(build, **kw):
    slot, size = ROW[build]
    return B.build_gam_layout(build, 40, slot, size, rows(build), slots(build), **kw)


class GameBalance(unittest.TestCase):
    def test_round_trip(self):
        for build in ("ps3", "switch"):
            with self.subTest(build):
                data = gam(build)
                parsed = G.parse_balance(data)
                self.assertEqual(W.write(parsed), data)
                block = parsed.rows[0].blocks[CODES[build][0]]
                self.assertEqual((block.kind, block.data), ("code", code(build)))
                self.assertEqual(parsed.rows[1].blocks[CODES[build][0]].data, b"")

    def test_code_blocks_are_bytecode_only(self):
        for build in ("ps3", "switch"):
            parsed = G.parse_balance(gam(build))
            found = list(G.code_blocks(parsed))
            self.assertEqual(len(found), 12)
            self.assertTrue(all(F.stream_ends_with_end(b.data, build) for *_, b in found))

    def test_shared_name_block(self):
        data = gam("ps3", shared_name=True)
        parsed = G.parse_balance(data)
        self.assertEqual([r.name for r in parsed.rows], ["Axe"] * 3)
        self.assertEqual(W.write(parsed), data)

    def test_empty_code_can_end_the_file(self):
        slot, size = ROW["switch"]
        data = B.build_gam_layout("switch", 40, slot, size, [{"name": "Ab"}], slots("switch"))
        self.assertEqual(len(data) % 4, 0)
        self.assertEqual(W.write(G.parse_balance(data)), data)

    def test_empty_pairs_point_at_the_cursor(self):
        slot, size = ROW["ps3"]
        data = B.build_gam_layout("ps3", 40, slot, size, [{}], slots("ps3"))
        parsed = G.parse_balance(data)
        self.assertIsNone(parsed.rows[0].name)
        self.assertEqual(W.write(parsed), data)
        self.assertEqual(len(data), 0x3E8 + size)  # five empty pairs add nothing

    def test_rename(self):
        for build in ("ps3", "switch"):
            parsed = G.parse_balance(gam(build))
            W.set_row_name(parsed, 1, "Greatsword")
            read = G.parse_balance(W.write(parsed))
            self.assertEqual([r.name for r in read.rows], ["Axe", "Greatsword", "Mace"])
            self.assertEqual(read.rows[1].id, G.gbid("Greatsword"))

    def test_rename_one_of_shared(self):
        parsed = G.parse_balance(gam("ps3", shared_name=True))
        W.set_row_name(parsed, 2, "Other")
        read = G.parse_balance(W.write(parsed))
        self.assertEqual([r.name for r in read.rows], ["Axe", "Axe", "Other"])

    def test_code(self):
        for build in ("ps3", "switch"):
            parsed = G.parse_balance(gam(build))
            W.set_row_code(parsed, 1, CODES[build][1], code(build, 9))
            out = W.write(parsed)
            read = G.parse_balance(out)
            self.assertEqual(read.rows[1].blocks[CODES[build][1]].data, code(build, 9))
            self.assertEqual(W.write(read), out)

    def test_old_formula_helper_stores_code_only(self):
        parsed = G.parse_balance(gam("switch"))
        W.set_row_formula(parsed, 1, 0x30, "9", code("switch", 9))
        read = G.parse_balance(W.write(parsed))
        self.assertEqual(read.rows[1].blocks[0x30].data, code("switch", 9))

    def test_field(self):
        parsed = G.parse_balance(gam("switch"))
        W.set_row_field(parsed, 0, 20, "i", 99)
        out = W.write(parsed)
        read = G.parse_balance(out)
        self.assertEqual(struct.unpack_from("<i", read.rows[0].raw, 20)[0], 99)

    def test_field_checks(self):
        parsed = G.parse_balance(gam("switch"))
        for args in ((0, 4, "i", 1), (0, 0x34, "i", 1), (0, 0x2E, "i", 1), (0, 398, "i", 1), (9, 20, "i", 1),
                     (0, 20, "i", "x")):
            with self.assertRaises(FormatError):
                W.set_row_field(parsed, *args)

    def test_edit_checks(self):
        parsed = G.parse_balance(gam("switch"))
        with self.assertRaises(FormatError):
            W.set_row_code(parsed, 0, 0, code("switch"))  # the name pair
        with self.assertRaises(FormatError):
            W.set_row_code(parsed, 0, 0x30, b"\0")
        with self.assertRaises(FormatError):
            W.set_row_code(parsed, 0, 0x30, bytes(B.Code("<").f(1).out))  # no END
        with self.assertRaises(FormatError):
            W.set_row_code(parsed, 0, 0x34, code("switch"))
        with self.assertRaises(FormatError):
            W.set_row_name(parsed, 5, "x")
        with self.assertRaises(FormatError):
            W.set_row_text(parsed, 0, 0x30, "x")
        W.set_row_code(parsed, 0, 0x30, b"")  # empty: no formula
        self.assertEqual(G.parse_balance(W.write(parsed)).rows[0].blocks[0x30].data, b"")

    def test_trailing_bytes_are_not_kept(self):
        data = gam("switch")
        self.assertEqual(W.write(G.parse_balance(data + bytes(8))), data)

    def test_no_table_is_copied(self):
        for kind in (9, -1):
            data = B.build_gam("switch", kind)
            parsed = G.parse_balance(data)
            self.assertEqual(W.write(parsed), data)
        self.assertEqual(G.parse_balance(B.build_gam("ps3", 9)).type_name, "Inventory")

    def test_empty_slot_is_copied(self):
        data = B.build_gam("ps3", 11)
        self.assertEqual(W.write(G.parse_balance(data)), data)

    def test_formula_table_has_no_blocks(self):
        data = B.build_formula_gam("switch", [("Row", [1.0])])
        parsed = G.parse_balance(data)
        self.assertEqual(W.write(parsed), data)

    def test_ps3_event_text(self):
        rows_ = [{"name": "E1", "blocks": {0x14: b"KilledDiablo\0"}}, {"name": "E2"}]
        data = B.build_gam_layout("ps3", 37, 0x3C8, 40, rows_, {0: "name", 0x14: "text"})
        parsed = G.parse_balance(data)
        self.assertEqual(parsed.rows[0].blocks[0x14].kind, "text")
        self.assertEqual(W.write(parsed), data)
        W.set_row_text(parsed, 1, 0x14, "KilledBelial")
        read = G.parse_balance(W.write(parsed))
        self.assertEqual(read.rows[1].blocks[0x14].data, b"KilledBelial\0")

    def test_switch_item_attribute_array(self):
        element = struct.pack("<4i", 1, 2, 3, 4)
        rows_ = [
            {"name": "Itm", "arrays": {0x108: (24, [
                (element, {0x10: code("switch", 5)}), (element, {0x10: None}),
            ])}, "blocks": {0x288: code("switch", 6)}},
            {"name": "Plain"},
        ]
        item_slots = {0: "name", 0x108: "array", 0x270: "code", 0x288: "code"}
        data = B.build_gam_layout("switch", 2, 0x38, 696, rows_, item_slots)
        parsed = G.parse_balance(data)
        array = parsed.rows[0].blocks[0x108]
        self.assertEqual(len(array.elements), 2)
        self.assertEqual(array.elements[0][0x10].data, code("switch", 5))
        self.assertEqual(parsed.rows[1].blocks[0x108].data, b"")
        self.assertEqual(W.write(parsed), data)
        W.set_row_code(parsed, 0, 0x10, code("switch", 7), element=1)
        read = G.parse_balance(W.write(parsed))
        self.assertEqual(read.rows[0].blocks[0x108].elements[1][0x10].data, code("switch", 7))
        with self.assertRaises(FormatError):
            W.set_row_code(parsed, 0, 0x10, code("switch"), element=2)

    def test_fixed_part_fields(self):
        data = bytearray(gam("ps3"))
        data[0x20:0x28] = b"0000.gbi"
        path = b".\\Resources\\Excel\\Paragon.xls"
        data[0x120:0x120 + len(path)] = path
        struct.pack_into(">I", data, 0x220, 0x5EED)
        parsed = G.parse_balance(bytes(data))
        self.assertEqual((parsed.gbi_name, parsed.machine_time), ("0000.gbi", 0x5EED))
        self.assertEqual(parsed.source_path, path.decode())
        self.assertEqual(parsed.fixed_problems, [])
        self.assertEqual(parsed.type_name, "Paragon Bonuses")
        data[0x30] = 1  # after the NUL of the gbi text
        self.assertTrue(G.parse_balance(bytes(data)).fixed_problems)
        switch = bytearray(gam("switch"))
        struct.pack_into("<II", switch, 0x20, 77, 1)
        parsed = G.parse_balance(bytes(switch))
        self.assertEqual((parsed.machine_time, parsed.ignore_on_load), (77, 1))
        self.assertTrue(parsed.fixed_problems)

    def test_name_alignment(self):
        for build, slot, size, step in (("switch", 0xB8, 32, 8), ("ps3", 0x2A8, 24, 4)):
            rows_ = [{"name": "Ab"}, {"name": "Cd"}]
            data = B.build_gam_layout(build, 11, slot, size, rows_, {0: "name"})
            parsed = G.parse_balance(data)
            self.assertEqual(W.write(parsed), data)
            order = ">" if build == "ps3" else "<"
            table = struct.unpack_from(order + "I", data, slot)[0] + 0x10
            first = struct.unpack_from(order + "I", data, table)[0]
            second = struct.unpack_from(order + "I", data, table + size)[0]
            self.assertEqual(second - first, step)

    def test_array_alignment(self):
        element = struct.pack("<4i", 1, 2, 3, 4)
        rows_ = [{"name": "Abc", "arrays": {0x108: (24, [(element, {0x10: None})])}}]
        item_slots = {0: "name", 0x108: "array", 0x270: "code", 0x288: "code"}
        data = B.build_gam_layout("switch", 2, 0x38, 696, rows_, item_slots)
        self.assertEqual(struct.unpack_from("<I", data, 0x38)[0] % 8, 0)
        table = struct.unpack_from("<I", data, 0x38)[0] + 0x10
        array_at = struct.unpack_from("<I", data, table + 0x108)[0]
        self.assertEqual((array_at + 0x10) % 8, 0)
        self.assertEqual(W.write(G.parse_balance(data)), data)

    def test_ps3_row_sizes(self):
        for kind, size in ((23, 456), (28, 60), (29, 28)):
            self.assertEqual(G.TABLES[kind][4], size)


BONES = [(0xAA, -1, (1.0, 2.0, 3.0)), (0xBB, 0, (2.0, 2.0, 3.0))]


class Appearance(unittest.TestCase):
    def app(self, build):
        data = B.build_app_ps3(bones=BONES) if build == "ps3" else B.build_app_switch(bones=BONES)
        data = B.add_app_materials(data, build, [1], [(0x10, [[(1, 44), (2, 45)]])])
        return data

    def test_unedited_is_identical(self):
        for build in ("ps3", "switch"):
            data = self.app(build)
            self.assertEqual(W.write(P.parse_appearance(data)), data)

    def test_bone_transform(self):
        for build in ("ps3", "switch"):
            app = P.parse_appearance(self.app(build))
            moved = Transform((0.0, 0.0, 0.0, 1.0), (5.0, 6.0, 7.0), 1.0)
            W.set_bone_transform(app, 1, "local", moved)
            read = P.parse_appearance(W.write(app))
            self.assertEqual(read.bones[1].local.t, (5.0, 6.0, 7.0))
            self.assertEqual(app.bones[1].local.t, (5.0, 6.0, 7.0))

    def test_bind_only_on_switch(self):
        app = P.parse_appearance(self.app("ps3"))
        with self.assertRaises(FormatError):
            W.set_bone_transform(app, 0, "bind", Transform())
        with self.assertRaises(FormatError):
            W.set_bone_transform(app, 5, "local", Transform())

    def test_hardpoint(self):
        app = P.parse_appearance(self.app("ps3"))
        W.set_hardpoint(app, 0, (0.0, 1.0, 0.0, 0.0), (4.0, 5.0, 6.0))
        point = P.parse_appearance(W.write(app)).hardpoints[0]
        self.assertEqual((point.rotation, point.position), ((0.0, 1.0, 0.0, 0.0), (4.0, 5.0, 6.0)))
        with self.assertRaises(FormatError):
            W.set_hardpoint(app, 3, (0, 0, 0, 1), (0, 0, 0))

    def test_material_texture(self):
        for build in ("ps3", "switch"):
            app = P.parse_appearance(self.app(build))
            W.set_material_texture(app, 0x10, 0, 2, 99)
            block = P.parse_appearance(W.write(app)).material_data.find(0x10).per_look[0][0]
            self.assertEqual((block.texture(1), block.texture(2)), (44, 99))

    def test_material_texture_checks(self):
        app = P.parse_appearance(self.app("switch"))
        for args in ((0x11, 0, 1, 5), (0x10, 1, 1, 5), (0x10, 0, 7, 5)):
            with self.assertRaises(FormatError):
                W.set_material_texture(app, *args)
