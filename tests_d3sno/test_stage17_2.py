"""Corrections 24 (``39_corrections24.md``): V24-1 the fat part of a cloth or fat sub-object
counts as a part, V24-2 the palette refusal names its draw, V24-4 the limit checks of a changed
ps3 draw in the order of V23-2. Synthetic data only."""

import copy
import unittest

import d3sno
from d3sno import FormatError
from d3sno import app_mesh as M
from d3sno import appearance as P
from test_stage12 import BONES, ps3
from test_stage17_1 import dense, set_geometry
from test_stage17_writer import FAT, canon, ps3_file, switch_cloth_two, switch_skin


class Marker(unittest.TestCase):
    def test_version(self):
        self.assertGreaterEqual(tuple(int(x) for x in d3sno.__version__.split(".")), (0, 17, 2))
        self.assertGreaterEqual(d3sno.STAGE, 17)


def fat_cases():
    """(data, build, sub-object index, has a cloth record) on every build. switch and x360 store fat
    vertices only with a cloth record (V7-T11), so the case without one is ps3 only."""
    yield ps3_file(fat=FAT, cloth=True) + (0, True)
    yield ps3_file(fat=FAT, cloth=False) + (0, False)
    yield ps3_file(fat=FAT, cloth=True, cloth_draw=True, bones=BONES, skinned=True) + (0, True)
    for order in ("<", ">"):
        yield switch_cloth_two(order) + (1, True)


EDITS = {
    "positions": lambda fat: fat.positions.__setitem__(1, (fat.positions[1][0] + 1.0, *fat.positions[1][1:])),
    "normals": lambda fat: fat.normals.__setitem__(0, (0.0, 1.0, 0.0)),
    "colours": lambda fat: fat.colours_1.__setitem__(0, bytes((9, 9, 9, 9))),
    "UV1": lambda fat: fat.uv1.__setitem__(0, (fat.uv1[0][0] + 0.5, fat.uv1[0][1])),
    "UV2": lambda fat: fat.uv2_raw.__setitem__(0, (fat.uv2_raw[0][0] + 1, fat.uv2_raw[0][1])),
    "indices": lambda fat: fat.indices.reverse(),
}


class FatPartIsAPart(unittest.TestCase):
    """V24-1: any change to the fat vertices of a V21-1 sub-object is refused with the part label."""

    def test_every_value_refused(self):
        seen = set()
        for data, build, k, cloth in fat_cases():
            seen.add((build or "switch", cloth))
            for what, edit in EDITS.items():
                app = P.parse_appearance(data, build)
                subs = app.sub_objects
                edit(subs[k].fat_draw)
                with self.assertRaisesRegex(FormatError, rf"^high-poly set, sub-object {k}, fat vertices: a part of a "
                                                         r"cloth sub-object .*V23-5", msg=(build, what)):
                    M.write_app_mesh(app, derive=True)
                with self.assertRaisesRegex(FormatError, "fat vertices", msg=(build, what)):
                    M.write_app_mesh(app)  # without derive too
        self.assertEqual(seen, {("ps3", True), ("ps3", False), ("switch", True), ("x360", True)})

    def test_dropping_the_fat_part_refused(self):
        for data, build, k, _ in fat_cases():
            app = P.parse_appearance(data, build)
            app.sub_objects[k].fat_draw = None
            with self.assertRaisesRegex(FormatError, rf"sub-object {k}, fat vertices: "):
                M.write_app_mesh(app, derive=True)

    def test_unchanged_and_removed_whole(self):
        for data, build, k, _ in fat_cases():
            self.assertEqual(M.write_app_mesh(P.parse_appearance(data, build), derive=True), data)
            app = P.parse_appearance(data, build)
            mesh = copy.copy(app)
            mesh.sub_objects = [s for i, s in enumerate(app.sub_objects) if i != k]
            out, report = M.write_app_mesh_report(app, mesh, derive=True)
            self.assertIn(f"high-poly set, sub-object {k}", report.removed)
            P.parse_appearance(out, build, strict=True)

    def test_plain_sub_object_of_a_cloth_file_still_edited(self):
        for order in ("<", ">"):
            data, build = switch_cloth_two(order)
            app = P.parse_appearance(data, build)
            draw = app.sub_objects[0].draws[0]
            draw.positions[1] = (11.0, 0.0, 0.0)
            again = P.parse_appearance(M.write_app_mesh(app, derive=True), build, strict=True)
            self.assertEqual(again.sub_objects[0].draws[0].positions[1], (11.0, 0.0, 0.0))


class PaletteRefusalLabel(unittest.TestCase):
    """V24-2: "joint j is not in the palette (W2-6)" begins with the draw or record label."""

    def test_switch_and_x360_record(self):
        for order in ("<", ">"):
            data, build = switch_skin(order)
            app = P.parse_appearance(data, build)
            draw = app.sub_objects[0].draws[0]
            draw.joints[0] = (0, 0, 0, 0)  # bone 0 is not in the stored palette [2, 1]
            draw.weights[0] = (255, 0, 0, 0)
            with self.assertRaisesRegex(FormatError, r"^high-poly set, sub-object 0, skinning record 0: vertex 0: "
                                                     r"joint 0 is not in the palette \(W2-6\)"):
                M.write_app_mesh(app)  # stored palette (no W13-5)

    def test_ps3_draw(self):
        app = P.parse_appearance(canon(ps3(skinned=True, bones=BONES), "ps3"), "ps3")
        draw = app.sub_objects[0].draws[0]
        outside = next(b for b in range(len(app.bones)) if b not in draw.palette)
        draw.joints[0] = (outside, 0, 0, 0)
        draw.weights[0] = (255, 0, 0, 0)
        with self.assertRaisesRegex(FormatError, rf"^high-poly set, sub-object 0, draw 0: vertex 0: joint {outside} "
                                                 r"is not in the palette \(W2-6\)"):
            M.write_app_mesh(app)  # stored ranges (no W13-4)

    def test_function_without_label(self):
        app = P.parse_appearance(switch_skin()[0])
        draw = app.sub_objects[0].draws[0]
        draw.joints[0], draw.weights[0] = (0, 0, 0, 0), (255, 0, 0, 0)
        with self.assertRaisesRegex(FormatError, r"^vertex 0: joint 0"):
            M.skin_slots(draw, 0, None)


class LimitOrder(unittest.TestCase):
    """V24-4: with ``derive=True`` a changed 0x83 draw is checked against the budget, stream A and
    then the work buffer, as V23-2 orders them."""

    def changed(self, nv, ni):
        app = P.parse_appearance(canon(ps3(skinned=True, bones=BONES), "ps3"), "ps3")
        set_geometry(app.sub_objects[0].draws[0], *dense(nv, ni))
        return app

    def test_budget_before_work_buffer(self):
        app = self.changed(1500, 4500)  # budget 73,500 and work buffer both over
        with self.assertRaisesRegex(FormatError, r"draw 0: budget value 73500 is above 58168 \(W17-5a\)"):
            M.write_app_mesh(app, derive=True)

    def test_work_buffer_alone(self):
        app = self.changed(1000, 12000)  # budget 58,000 fits
        with self.assertRaisesRegex(FormatError, r"draw 0: work buffer \d+ is above 0xC000 \(W2-10, W13-1\); a changed "
                                                 r"draw is not cut \(V23-3\)"):
            M.write_app_mesh(app, derive=True)


if __name__ == "__main__":
    unittest.main()
