import struct
import unittest

import builders as B
from d3sno import FormatError
from d3sno import appearance as P
from d3sno import balance as G
from d3sno import write as W
from d3sno.transform import Transform

ROW = {"switch": (0xB8, 32), "ps3": (0x2A8, 24)}


def code(build):
    return B.Code(">" if build == "ps3" else "<").f(2).end()


def rows(build):
    return [
        {"name": "Axe", "formulas": {8: ("1+1", code(build))}, "fields": {20: ("i", 7)}},
        {"name": "Sword", "formulas": {8: None}},
        {"name": "Mace", "formulas": {8: ("x", code(build))}},
    ]


def gam(build, **kw):
    slot, size = ROW[build]
    return B.build_gam_layout(build, 11, slot, size, rows(build), **kw)


class GameBalance(unittest.TestCase):
    def test_round_trip(self):
        for build in ("ps3", "switch"):
            with self.subTest(build):
                data = gam(build)
                self.assertEqual(W.write(G.parse_balance(data)), data)

    def test_shared_name_block(self):
        data = gam("ps3", shared_name=True)
        parsed = G.parse_balance(data)
        self.assertEqual([r.name for r in parsed.rows], ["Axe"] * 3)
        self.assertEqual(W.write(parsed), data)

    def test_empty_formula_can_end_the_file(self):
        slot, size = ROW["switch"]
        data = B.build_gam_layout("switch", 11, slot, size, [{"name": "Ab", "formulas": {8: None}}])
        self.assertEqual(len(data) % 4, 0)
        self.assertEqual(W.write(G.parse_balance(data)), data)

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

    def test_formula(self):
        for build in ("ps3", "switch"):
            parsed = G.parse_balance(gam(build))
            W.set_row_formula(parsed, 1, 8, "2", code(build))
            out = W.write(parsed)
            order = ">" if build == "ps3" else "<"
            table = struct.unpack_from(order + "I", out, ROW[build][0])[0] + 0x10
            offset, size = struct.unpack_from(order + "II", out, table + ROW[build][1] + 8)
            self.assertEqual(out[offset + 0x10 + 32:offset + 0x10 + 34], b"2\0")
            self.assertEqual(W.write(G.parse_balance(out)), out)

    def test_field(self):
        parsed = G.parse_balance(gam("switch"))
        W.set_row_field(parsed, 0, 20, "i", 99)
        out = W.write(parsed)
        read = G.parse_balance(out)
        self.assertEqual(struct.unpack_from("<i", read.rows[0].raw, 20)[0], 99)

    def test_field_checks(self):
        parsed = G.parse_balance(gam("switch"))
        for args in ((0, 4, "i", 1), (0, 30, "i", 1), (9, 20, "i", 1), (0, 20, "i", "x")):
            with self.assertRaises(FormatError):
                W.set_row_field(parsed, *args)

    def test_edit_checks(self):
        parsed = G.parse_balance(gam("switch"))
        with self.assertRaises(FormatError):
            W.set_row_formula(parsed, 0, 0, "x", b"")
        with self.assertRaises(FormatError):
            W.set_row_formula(parsed, 0, 8, "x", b"\0")
        with self.assertRaises(FormatError):
            W.set_row_name(parsed, 0 + 5, "x")

    def test_layout_not_understood_is_refused(self):
        data = gam("switch") + bytes(8)
        with self.assertRaises(FormatError):
            W.write(G.parse_balance(data))

    def test_no_table(self):
        with self.assertRaises(FormatError):
            W.write(G.parse_balance(B.build_gam("switch", 9)))

    def test_formula_table_has_no_blocks(self):
        data = B.build_formula_gam("switch", [("Row", [1.0])])
        parsed = G.parse_balance(data)
        self.assertEqual(W.write(parsed), data)


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
