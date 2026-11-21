"""Corrections 23 (``38_corrections23.md``): V23-3 a changed ps3 draw over a limit is refused,
V23-5 cloth and fat sub-objects are kept unchanged or removed whole (and drops are reported by the
writer), V23-6 an editor copy of a cloth or fat mesh is refused, V23-7 the warning for a material
hash not in the table, V23-9 labels. Synthetic data only."""

import copy
import tempfile
import unittest
from pathlib import Path

import builders as B
import d3sno
from d3sno import FormatError
from d3sno import app_mesh as M
from d3sno import appearance as P
from d3sno import mesh_import as I
from test_stage12 import BONES, laid_out, ps3
from test_stage16_cli import run
from test_stage16_mesh import read
from test_stage17_import import editor_copy
from test_stage17_writer import (
    FAT, SKELETON_60, SKIN2, Checked, canon, edited, multi_record, new_sub, ps3_cloth_draw, ps3_file, skinned_draw,
    switch_cloth_two, switch_skin, switch_two, with_materials,
)


def set_geometry(draw, positions, indices):
    """Replace a draw's geometry (every per-vertex list follows the first vertex's values)."""
    n = len(positions)
    for name in ("normals", "colours_1", "colours_2", "uv1", "uv2_raw"):
        setattr(draw, name, [getattr(draw, name)[0]] * n)
    if draw.skinned:
        draw.joints, draw.weights = [draw.joints[0]] * n, [draw.weights[0]] * n
    draw.positions, draw.indices = list(positions), list(indices)


def dense(nv, ni):
    """``nv`` distinct positions and ``ni`` indices (whole triangles cycling over the vertices)."""
    positions = [(float(i % 50), float(i // 50), 0.0) for i in range(nv)]
    indices = [(t + k) % nv for t in range(ni // 3) for k in range(3)]
    return positions, indices


def switch_skinned_cloth(order="<"):
    """A switch / x360 file whose only sub-object has a skinning record, a cloth record and fat
    vertices (a V21-1 sub-object)."""
    version, build = (302, "x360") if order == ">" else (306, None)
    return canon(laid_out(B.build_app_switch(version=version, order=order, bones=BONES, pool_spare=0, sub_objects=[
        {"skin": SKIN2, "index_count": 3, "cloth": True}]), build), build), build


class Marker(unittest.TestCase):
    def test_version(self):
        self.assertGreaterEqual(tuple(int(x) for x in d3sno.__version__.split(".")), (0, 17, 1))
        self.assertGreaterEqual(d3sno.STAGE, 17)


class ChangedPs3DrawLimits(unittest.TestCase):
    """V23-3: a changed draw stays one draw; over any V23-2 limit it is refused with its label."""

    HINT = r"a changed draw is not cut \(V23-3\); import the geometry as a new mesh"

    def changed(self, nv, ni, skinned=False):
        data = canon(ps3(skinned=True, bones=BONES), "ps3") if skinned else ps3_file()[0]
        app = P.parse_appearance(data, "ps3")
        set_geometry(app.sub_objects[0].draws[0], *dense(nv, ni))
        return app

    def test_rigid_budget_only(self):
        # 21 x 2600 + 1500 = 56,100 > 55,998; stream A 15,600 bytes and the work buffer fit
        app = self.changed(2600, 1500)
        with self.assertRaisesRegex(FormatError, r"^high-poly set, sub-object 0, draw 0: budget value 56100 is above "
                                                 r"55998 \(W17-5a\); " + self.HINT):
            M.write_app_mesh(app, derive=True)

    def test_rigid_work_buffer_only(self):
        """The library bug of V23-3: a changed 0x81 draw over the work-buffer limit was written whole
        (work above 0xC000) and failed the strict reader."""
        nv, ni = 1500, 15273  # budget 46,773, stream A 9,008 bytes, work 49,280: just above 0xC000
        self.assertLessEqual(21 * nv + ni, 55998)
        self.assertEqual(M.ps3_work_size(0x81, nv, ni), 49280)
        for derive in (True, False):
            app = self.changed(nv, ni)
            with self.assertRaisesRegex(FormatError, r"^high-poly set, sub-object 0, draw 0: work buffer 49280 is above "
                                                     r"0xC000 \(W2-10, W13-1\); " + self.HINT):
                M.write_app_mesh(app, derive=derive)

    def test_rigid_stream_a(self):
        """Stream A over 0x4000 bytes: on real counts the budget is passed first, so the stream
        check is shown on the check itself."""
        blocks = {"descriptor": bytes([0x81, 0]), "a": bytes(0x4010)}
        with self.assertRaisesRegex(FormatError, r"^lbl: stream A of 16400 bytes is above 0x4000 \(G1-5\); " + self.HINT):
            M._changed_draw_limits(None, blocks,  # 0x81 reads no stored word
                                     10, 30, None, "lbl")
        app = self.changed(2800, 3)  # 16,800 bytes of stream A
        with self.assertRaisesRegex(FormatError, self.HINT):
            M.write_app_mesh(app, derive=True)

    def test_skinned_budget(self):
        app = self.changed(1300, 300, skinned=True)  # 46 x 1300 + 300 = 60,100 > 58,168
        with self.assertRaisesRegex(FormatError, r"draw 0: budget value 60100 .*" + self.HINT):
            M.write_app_mesh(app, derive=True)

    def test_within_limits_written(self):
        app = self.changed(600, 1800)
        again = P.parse_appearance(M.write_app_mesh(app, derive=True), "ps3", strict=True)
        self.assertEqual(len(again.sub_objects[0].draws), 1)  # one draw, one record: not cut

    def test_import_names_the_draw_once(self):
        """V23-3 through the import, and V23-9: the label is printed once."""
        data, build = ps3_file()
        app = P.parse_appearance(data, build)
        doc = P.to_gltf(app)
        big = copy.deepcopy(app)
        set_geometry(big.sub_objects[0].draws[0], *dense(1900, 15198))
        big_doc = P.to_gltf(big)
        doc["meshes"], doc["accessors"], doc["bufferViews"], doc["buffers"] = (
            big_doc["meshes"], big_doc["accessors"], big_doc["bufferViews"], big_doc["buffers"])
        with self.assertRaises(FormatError) as caught:
            I.import_mesh(app, read(doc))
        message = caught.exception.message
        self.assertEqual(message.count("high-poly set, sub-object 0, draw 0"), 1, message)
        self.assertIn("V23-3", message)


class ClothKeptOrRemovedWhole(Checked):
    """V23-5: a V21-1 sub-object is kept unchanged or removed whole, on every build."""

    def cases(self):
        yield ps3_file(fat=FAT, cloth=True, cloth_draw=True, bones=BONES, skinned=True)
        yield ps3_file(fat=FAT, cloth=False)
        for order in ("<", ">"):
            yield switch_skinned_cloth(order)

    def test_in_place_emptying_refused(self):
        for data, build in self.cases():
            for what in ("indices", "positions"):
                app = P.parse_appearance(data, build)
                sub = app.sub_objects[0]
                draw = next(d for d in sub.draws if d.kind != "fat")
                for name in (what,) if what == "indices" else ("positions", "normals", "colours_1", "colours_2",
                                                                  "uv1", "uv2_raw", "indices"):
                    getattr(draw, name).clear()
                if what == "positions" and draw.skinned:
                    draw.joints.clear()
                    draw.weights.clear()
                part = "draw 0" if build == "ps3" else "skinning record 0"
                with self.assertRaisesRegex(FormatError, rf"^high-poly set, sub-object 0, {part}: a part of a cloth "
                                                         r"sub-object .*V23-5"):
                    M.write_app_mesh(app, derive=True)

    def test_any_change_refused(self):
        for data, build in self.cases():
            app = P.parse_appearance(data, build)
            draw = next(d for d in app.sub_objects[0].draws if d.kind != "fat")
            draw.positions[0] = (draw.positions[0][0] + 1.0, draw.positions[0][1], draw.positions[0][2])
            with self.assertRaisesRegex(FormatError, "V23-5"):
                M.write_app_mesh(app, derive=True)
            self.assertEqual(M.write_app_mesh(P.parse_appearance(data, build), derive=True), data)  # unchanged: kept

    def test_removed_whole_and_reported(self):
        for data, build in self.cases():
            app = P.parse_appearance(data, build)
            mesh = copy.copy(app)
            mesh.sub_objects = []
            out, report = M.write_app_mesh_report(app, mesh, derive=True)
            self.assertEqual(report.removed[0], "high-poly set, sub-object 0")
            self.assertEqual(P.parse_appearance(out, build, strict=True).sub_objects, [])

    def test_drops_reported_by_the_writer(self):
        """Parts dropped by W13-7 or W17-14 are in the writer's report (not only the importer's)."""
        data = canon(ps3(extra_draw_format=0x81), "ps3")
        app = P.parse_appearance(data, "ps3")
        app.sub_objects[0].draws[1].indices.clear()  # W13-7
        out, report = M.write_app_mesh_report(app, derive=True)
        self.assertEqual(report.removed, ["high-poly set, sub-object 0, draw 1"])
        self.check(out, "ps3")
        for order in ("<", ">"):
            data, build = multi_record(order)
            app = P.parse_appearance(data, build)
            skins = [d for d in app.sub_objects[1].draws if d.skinned]  # the 51-bone part: two records
            self.assertEqual(len(skins), 2)
            skins[1].indices.clear()
            out, report = M.write_app_mesh_report(app, derive=True)
            self.assertEqual(report.removed, ["high-poly set, sub-object 1, skinning record 1"])
            data, build = switch_skin(order)
            app = P.parse_appearance(data, build)
            app.sub_objects[0].draws[0].indices.clear()  # W13-7 then W17-14
            out, report = M.write_app_mesh_report(app, derive=True)
            self.assertEqual(report.removed, ["high-poly set, sub-object 0, skinning record 0",
                                              "high-poly set, sub-object 0"])


class ShortNonLastRecord(Checked):
    """V23-4: a written non-last skinning record may have fewer than 45 palette entries; the reader
    and every check accept it."""

    def test_record_of_44(self):
        for order, version, build in (("<", 306, None), (">", 302, "x360")):
            data = canon(laid_out(B.build_app_switch(version=version, order=order, bones=SKELETON_60, pool_spare=0,
                                                     sub_objects=[{"skin": SKIN2, "index_count": 0}]), build), build)
            app = P.parse_appearance(data, build)
            # one bone per vertex: 14 triangles of 3 new bones (42), one of 2 new bones (44), then one
            # of 2 new bones, which would make 46: W17-7 starts a new record before it
            bones = [b for t in range(14) for b in (3 * t + 1, 3 * t + 2, 3 * t + 3)] + [1, 43, 44] + [1, 45, 46]
            positions = [(float(i), float(i % 7), 0.0) for i in range(len(bones))]
            joints = [(b, 0, 0, 0) for b in bones]
            draw = skinned_draw(positions, list(range(len(bones))), joints, [(255, 0, 0, 0)] * len(bones))
            out = M.write_app_mesh(app, edited(app, app.sub_objects + [new_sub("Short", [draw])]), derive=True)
            again = self.check(out, build)
            palettes = [len(d.palette) for d in again.sub_objects[1].draws if d.skinned]
            self.assertEqual(palettes, [44, 3])


class ClothCopyRefused(unittest.TestCase):
    """V23-6: an editor copy whose template part is a V21-1 sub-object is refused, naming the cloth
    rule and the part's label."""

    def test_switch_and_x360(self):
        for order in ("<", ">"):
            data, build = switch_cloth_two(order)
            app = P.parse_appearance(data, build)
            doc = P.to_gltf(app)
            editor_copy(doc, 1, "subobject_1.001")
            with self.assertRaisesRegex(FormatError, r"an editor copy of high-poly set, sub-object 1, .*cloth data "
                                                     r"cannot be copied by import \(V21-1, V23-6\)"):
                I.import_mesh(app, read(doc))

    def test_ps3_with_bones(self):
        data, build = ps3_cloth_draw()
        app = P.parse_appearance(data, build)
        doc = P.to_gltf(app)
        editor_copy(doc, 0, "Body.001")
        with self.assertRaisesRegex(FormatError, r"editor copy of high-poly set, sub-object 0, .*V23-6"):
            I.import_mesh(app, read(doc))

    def test_plain_copy_still_added(self):
        data, build = switch_cloth_two()
        app = P.parse_appearance(data, build)
        doc = P.to_gltf(app)
        editor_copy(doc, 0, "subobject_0.001")
        _, report = I.import_mesh_report(app, read(doc))
        self.assertEqual(report.added, ["high-poly set, sub-object 3"])


class UnknownMaterialWarning(Checked):
    """V23-7: the import creates no material entry; an unknown ``extras.d3sno.material`` gives a
    warning naming the part, the hash and the rule used instead."""

    def test_warning(self):
        data, build = switch_two()
        data = with_materials(data, build)
        app = P.parse_appearance(data, build)
        doc = P.to_gltf(app)
        copied = editor_copy(doc, 1, "subobject_1.001")
        doc["meshes"][copied]["extras"]["d3sno"]["material"] = 0xC0FFEE
        out, report = I.import_mesh_report(app, read(doc))
        again = self.check(out, build)
        self.assertEqual(again.sub_objects[3].material_hash, 0xBBB)  # rule 2: the copied extras' sub-object
        self.assertEqual(sorted(m.hash for m in again.material_data.materials),
                         sorted(m.hash for m in app.material_data.materials))  # no entry created
        warnings = [w for w in report.warnings if "V23-7" in w]
        self.assertEqual(len(warnings), 1, report.warnings)
        self.assertIn("0x00C0FFEE", warnings[0])
        self.assertIn("subobject_1.001", warnings[0])
        self.assertIn("rule 2 gave 0x00000BBB", warnings[0])

    def test_known_material_no_warning(self):
        data, build = switch_two()
        app = P.parse_appearance(with_materials(data, build), build)
        doc = P.to_gltf(app)
        editor_copy(doc, 1, "subobject_1.001")
        _, report = I.import_mesh_report(app, read(doc))
        self.assertFalse([w for w in report.warnings if "V23-7" in w])

    def test_rule_4_named(self):
        data, build = switch_two()
        app = P.parse_appearance(with_materials(data, build), build)
        warnings = []
        material, part = I.new_part_material(app, {"material": 0x1234}, [], 0, None, warnings=warnings, what="Prop")
        self.assertEqual((material, part), (app.sub_objects[0].material_hash, app.sub_objects[0]))
        self.assertEqual(len(warnings), 1)
        self.assertTrue(warnings[0].startswith("Prop: extras.d3sno.material 0x00001234 "), warnings[0])
        self.assertIn("rule 4", warnings[0])


class Labels(unittest.TestCase):
    """V23-9: the transplant skeleton refusal names the source sub-object as ``G:I``; the CLI
    prints the same message."""

    def test_transplant_skeleton(self):
        target, build = switch_skin()
        source, _ = multi_record()
        with self.assertRaisesRegex(FormatError, r"^source sub-object 0:0: the files have different skeletons"):
            M.transplant(P.parse_appearance(target, build), P.parse_appearance(source, build), 0, 0)
        with tempfile.TemporaryDirectory() as tmp:
            paths = [Path(tmp) / "t.app", Path(tmp) / "s.app"]
            paths[0].write_bytes(target)
            paths[1].write_bytes(source)
            code, _, err = run("mesh-transplant", *paths, "--sub-object", "0:0", "-o", Path(tmp) / "o.app")
            self.assertEqual(code, 2)
            self.assertIn("source sub-object 0:0: the files have different skeletons", err)
            self.assertFalse((Path(tmp) / "o.app").exists())


if __name__ == "__main__":
    unittest.main()
