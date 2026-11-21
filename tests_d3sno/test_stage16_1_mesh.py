"""Stage 16.1 mesh half: Corrections 21 (``35_corrections21.md``) V21-1 to V21-5 for the mesh
exporter, the mesh import and the mesh writer. Synthetic files only."""

import copy
import struct
import unittest

import builders as B
from d3sno import FormatError
from d3sno import app_mesh as M
from d3sno import appearance as P
from d3sno import mesh_import as I
from d3sno import sno_layout as L
from test_stage12 import BONES, laid_out, ps3, switch_cloth, switch_rigid
from test_stage13 import set_mesh
from test_stage16_mesh import SKIN2, derived, grid_template, read, switch_skinned3

FAT = [(0.0, 0.0, 0.0), (4.0, 0.0, 0.0), (0.0, 4.0, 0.0)]
LABEL = "high-poly set, sub-object 0, draw 0"


def gltf(app, **options):
    return read(P.to_gltf(app, **options))


def sub_record(data, build=None, geo=0, index=0):
    return L.parse_tree(data, "app", build).top.pairs[M.GEO_SUB_OBJECTS[geo]].block.elements[index]


def block(record, offset):
    return M._block_bytes(record.pairs.get(offset))


def entries(raw):
    return [raw[k:k + 6] for k in range(0, len(raw), 6)]


# --- V21-1 -----------------------------------------------------------------------------------------

def cloth_templates():
    return {
        "switch fat cloth": (switch_cloth(), None),
        "switch packed cloth": (laid_out(B.build_app_switch(pool_spare=0, sub_objects=[
            {"rigid": [(0, 0, 0), (10, 0, 0), (0, 10, 0)], "index_count": 3, "cloth": True}])), None),
        "switch fat, no cloth": (laid_out(B.build_app_switch(pool_spare=0, sub_objects=[
            {"fat": FAT, "index_count": 3}])), None),
        "ps3 fat cloth": (ps3(fat=FAT, cloth=True), "ps3"),
        "ps3 skinned cloth": (ps3(skinned=True, bones=BONES, fat=FAT, cloth=True, cloth_draw=True), "ps3"),
    }


EDITS = {
    "positions": lambda d: d.positions.__setitem__(1, tuple(c + 1.0 for c in d.positions[1])),
    "normals": lambda d: d.normals.__setitem__(0, (0.0, 1.0, 0.0)),
    "UV1": lambda d: d.uv1.__setitem__(2, (0.5, 0.25)),
    "UV2": lambda d: d.uv2_raw.__setitem__(0, (1, 2)),
    r"colours \(_D3_COLOR1\)": lambda d: d.colours_1.__setitem__(0, bytes((9, 9, 9, 9))),
    "indices": lambda d: d.indices.reverse(),
}


def grown(draw):
    for name in ("positions", "normals", "colours_1", "colours_2", "uv1", "uv2_raw", "joints", "weights"):
        values = getattr(draw, name)
        if values is not None:
            values.append(values[0])
    draw.indices += [0, 1, 3]


class ClothAndFat(unittest.TestCase):
    def test_which_sub_objects(self):
        for name, (data, build) in cloth_templates().items():
            self.assertTrue(I.keeps_every_value(P.parse_appearance(data, build).sub_objects[0]), name)
        for data, build in ((switch_rigid(), None), (ps3(skinned=True, bones=BONES), "ps3")):
            self.assertFalse(I.keeps_every_value(P.parse_appearance(data, build).sub_objects[0]))

    def test_unchanged_passes(self):
        for name, (data, build) in cloth_templates().items():
            app = P.parse_appearance(data, build)
            self.assertEqual(I.import_mesh(app, gltf(app)), data, name)
            self.assertEqual(I.import_mesh(app, P.to_obj(app)), data, name)

    def test_every_change_refused(self):
        for name, (data, build) in cloth_templates().items():
            app = P.parse_appearance(data, build)
            for what, edit in EDITS.items():
                edited = copy.deepcopy(app)
                edit(edited.sub_objects[0].draws[0])
                with self.assertRaisesRegex(FormatError, f"^{LABEL}: the import changes the {what} of a sub-object "
                                                         "with cloth data or fat vertices.*V21-1", msg=(name, what)):
                    I.import_mesh(app, gltf(edited))
            edited = copy.deepcopy(app)
            grown(edited.sub_objects[0].draws[0])
            with self.assertRaisesRegex(FormatError, f"^{LABEL}: the import changes the vertex count \\(3 -> 4\\)",
                                        msg=name):
                I.import_mesh(app, gltf(edited))
            with self.assertRaisesRegex(FormatError, f"^{LABEL}: the import changes the vertex count "
                                                     r"\(emptying the draw\)", msg=name):
                I.import_mesh(app, read({"asset": {"version": "2.0"}}), missing="empty")

    def test_obj_change_refused(self):
        app = P.parse_appearance(switch_cloth())
        text = P.to_obj(app)
        moved = text.replace("v 10.000000 0.000000 0.000000", "v 11.000000 0.000000 0.000000")
        self.assertNotEqual(moved, text)
        with self.assertRaisesRegex(FormatError, f"^{LABEL}: .*positions.*V21-1"):
            I.import_mesh(app, moved)

    def test_joints_and_weights_refused(self):
        data, build = cloth_templates()["ps3 skinned cloth"]
        app = P.parse_appearance(data, build)
        for name, (joints, weights) in (("joints", ((1, 0, 0, 0), (255, 0, 0, 0))),
                                        ("weights", (None, (100, 155, 0, 0)))):
            edited = copy.deepcopy(app)
            draw = edited.sub_objects[0].draws[0]
            draw.joints[0] = joints or draw.joints[0]
            draw.weights[0] = weights
            with self.assertRaisesRegex(FormatError, f"^{LABEL}: the import changes the (joints|weights)"):
                I.import_mesh(app, gltf(edited))


# --- V21-2 -----------------------------------------------------------------------------------------

def distinct_skins(data, build=None, same=()):
    """The 3 x 3 grid template with a different skin per vertex: bones (2, 1), weights
    (200 - 10 v, 55 + 10 v); vertices in ``same`` get the skin of vertex 0. Written with every W13
    rule, so it follows them."""
    app = P.parse_appearance(grid_template(data, build), build)
    draw = app.sub_objects[0].draws[0]
    draw.joints[:] = [(2, 1, 0, 0)] * len(draw.positions)
    draw.weights[:] = [(200 - 10 * (0 if v in same else v), 55 + 10 * (0 if v in same else v), 0, 0)
                       for v in range(len(draw.positions))]
    return derived(M.write_app_mesh(app, derive=True), build)


def with_influences(data, raw, build=None):
    tree = L.parse_tree(data, "app", build)
    tree.top.pairs[0x64].block.elements[0].pairs[M.SWITCH_SUB["influence"]] = L.PairValue(L.Block(bytes(raw)))
    return L.write_tree(tree)


def skin_record(data, build=None):
    return sub_record(data, build).pairs[M.SWITCH_SUB["skins"]].block.elements[0]


def vertex_slots(data, build=None):
    raw = block(skin_record(data, build), M.SKIN["vertices"])
    out = [raw[i + 28:i + 36] for i in range(0, len(raw), 36)]
    return [s[:4][::-1] + s[4:][::-1] if build == "x360" else s for s in out]


def one_weight(app, v, joints=(1, 2, 0, 0), weights=(155, 100, 0, 0)):
    edited = copy.deepcopy(app)
    draw = edited.sub_objects[0].draws[0]
    draw.joints[v], draw.weights[v] = joints, weights
    return edited


class SkinEditSwitch(unittest.TestCase):
    BUILDS = ((None, "<"), ("x360", ">"))

    def test_one_weight_changes_one_vertex_and_one_entry(self):
        for build, order in self.BUILDS:
            template = distinct_skins(switch_skinned3(order), build)
            stored = entries(block(sub_record(template, build), M.SWITCH_SUB["influence"]))
            self.assertEqual(len(set(stored)), 9)
            permuted = with_influences(template, b"".join(reversed(stored)), build)  # not W13-6 order
            app = P.parse_appearance(permuted, build)
            out, report = I.import_mesh_report(app, gltf(one_weight(app, 2)))
            self.assertEqual(report.changed, [LABEL])
            before, after = skin_record(permuted, build), skin_record(out, build)
            self.assertEqual(block(after, M.SKIN["palette"]), block(before, M.SKIN["palette"]))
            raw_before, raw_after = block(before, M.SKIN["vertices"]), block(after, M.SKIN["vertices"])
            changed = [i for i in range(len(raw_before)) if raw_before[i] != raw_after[i]]
            self.assertTrue(changed and all(2 * 36 + 28 <= i < 3 * 36 for i in changed), (build, changed))
            got = entries(block(sub_record(out, build), M.SWITCH_SUB["influence"]))
            want = list(reversed(stored))
            want[6] = bytes((1, 155, 2, 100, 0, 0))
            self.assertEqual(got, want, build)
            again = P.parse_appearance(out, build)
            self.assertEqual(again.warnings, [])
            self.assertEqual(again.sub_objects[0].draws[0].weights[2], (155, 100, 0, 0))
            self.assertEqual(again.sub_objects[0].draws[0].joints[2], (1, 2, 0, 0))

    def test_ambiguous_entry_rebuilds_the_list(self):
        for build, order in self.BUILDS:
            template = distinct_skins(switch_skinned3(order), build, same=(3,))  # vertices 0 and 3 alike
            stored = entries(block(sub_record(template, build), M.SWITCH_SUB["influence"]))
            permuted = with_influences(template, b"".join(reversed(stored)), build)
            app = P.parse_appearance(permuted, build)
            out = I.import_mesh(app, gltf(one_weight(app, 3)))
            draw = P.parse_appearance(out, build).sub_objects[0].draws[0]
            rebuilt = b"".join(M.influence_entry(draw, i) for i in range(len(draw.positions)))  # W13-6
            self.assertEqual(block(sub_record(out, build), M.SWITCH_SUB["influence"]), rebuilt, build)

    def test_no_matching_entry_rebuilds_the_list(self):
        template = distinct_skins(switch_skinned3())
        app = P.parse_appearance(with_influences(template, bytes(6 * 9)))
        out = I.import_mesh(app, gltf(one_weight(app, 2)))
        draw = P.parse_appearance(out).sub_objects[0].draws[0]
        rebuilt = b"".join(M.influence_entry(draw, i) for i in range(len(draw.positions)))
        self.assertEqual(block(sub_record(out), M.SWITCH_SUB["influence"]), rebuilt)

    def test_bone_outside_the_palette(self):
        for build, order in self.BUILDS:
            template = distinct_skins(switch_skinned3(order), build)
            app = P.parse_appearance(template, build)
            out = I.import_mesh(app, gltf(one_weight(app, 0, (0, 0, 0, 0), (255, 0, 0, 0))))
            again = P.parse_appearance(out, build).sub_objects[0].draws[0]
            self.assertEqual(again.palette, [0, 2, 1], build)  # W13-5: first use
            before = app.sub_objects[0].draws[0]
            self.assertEqual(again.joints[1:], before.joints[1:])
            self.assertEqual(again.weights[1:], before.weights[1:])
            slots = vertex_slots(out, build)
            # stored bone 2 (slot 0) -> 1, bone 1 (slot 1) -> 2; zero-weight slots named stored slot 0 (bone 2) -> 1
            self.assertEqual([s[:4] for s in slots[1:]], [bytes((1, 2, 1, 1))] * 8, build)
            self.assertEqual(slots[0][:4], bytes((0, 0, 0, 0)))

    def test_zero_weight_slots_of_unchanged_vertices_kept(self):
        data = laid_out(B.build_app_switch(bones=BONES, pool_spare=0, sub_objects=[
            {"skin": dict(SKIN2, slots=(0, 1, 1, 0), weights=(255, 0, 0, 0)), "index_count": 3}]))
        app = P.parse_appearance(data)
        out = I.import_mesh(app, gltf(one_weight(app, 0)))
        slots = vertex_slots(out)
        self.assertEqual([s[:4] for s in slots[1:]], [bytes((0, 1, 1, 0))] * 2)  # as stored
        self.assertEqual(slots[0], bytes((1, 0, 0, 0, 155, 100, 0, 0)))  # palette [2, 1]: bone 1 -> 1, bone 2 -> 0
        self.assertEqual(block(skin_record(out), M.SKIN["palette"]), block(skin_record(data), M.SKIN["palette"]))

    def test_position_change_derives_as_before(self):
        data = laid_out(B.build_app_switch(bones=BONES, pool_spare=0, sub_objects=[
            {"skin": dict(SKIN2, slots=(0, 1, 1, 0), weights=(255, 0, 0, 0)), "index_count": 3}]))
        app = P.parse_appearance(data)
        edited = one_weight(app, 0)
        edited.sub_objects[0].draws[0].positions[1] = (12.0, 2.0, 3.0)
        slots = vertex_slots(I.import_mesh(app, gltf(edited)))
        # W13-5 as before: first-use palette [1, 2] (bone 2 -> slot 1), zero-weight slots 0
        self.assertEqual([s[:4] for s in slots[1:]], [bytes((1, 0, 0, 0))] * 2)

    def test_writer_option_off_by_default(self):
        template = distinct_skins(switch_skinned3())
        stored = entries(block(sub_record(template), M.SWITCH_SUB["influence"]))
        app = P.parse_appearance(with_influences(template, b"".join(reversed(stored))))
        app.sub_objects[0].draws[0].weights[2] = (100, 155, 0, 0)
        plain = M.write_app_mesh(app, derive=True)
        kept = M.write_app_mesh(app, derive=True, keep_skin_bytes=True)
        self.assertNotEqual(plain, kept)
        self.assertEqual(entries(block(sub_record(kept), M.SWITCH_SUB["influence"]))[0], stored[-1])
        self.assertEqual(entries(block(sub_record(plain), M.SWITCH_SUB["influence"]))[0], stored[0])

    def test_patched_influences(self):
        a, b, c, n = (bytes((k, 255, 0, 0, 0, 0)) for k in (1, 2, 3, 9))
        self.assertEqual(M.patched_influences(a + b + c, [(b, n)]), a + n + c)
        self.assertEqual(M.patched_influences(a + b + c, [(b, b)]), a + b + c)  # nothing to replace
        self.assertIsNone(M.patched_influences(a + b + c, [(n, a)]))  # no match
        self.assertIsNone(M.patched_influences(a + b + b, [(b, n)]))  # several
        self.assertIsNone(M.patched_influences(a + b + c, [(b, n), (b, a)]))  # two vertices, one entry
        self.assertIsNone(M.patched_influences(a + b[:3], [(b, n)]))  # not whole entries


class SkinEditPs3(unittest.TestCase):
    def template(self):
        return distinct_skins(ps3(skinned=True, bones=BONES), "ps3")

    def draw_record(self, data):
        return sub_record(data, "ps3").pairs[M.PS3_SUB["draws"]].block.elements[0]

    def test_ranges_kept_and_one_vertex_changed(self):
        template = self.template()
        app = P.parse_appearance(template, "ps3")
        out = I.import_mesh(app, gltf(one_weight(app, 4)))
        before, after = self.draw_record(template), self.draw_record(out)
        self.assertEqual(after.raw[M.PS3_RANGES:M.PS3_RANGES + 8], before.raw[M.PS3_RANGES:M.PS3_RANGES + 8])
        b0, b1 = block(before, M.DRAW["b"]), block(after, M.DRAW["b"])
        changed = [i for i in range(len(b0)) if b0[i] != b1[i]]
        self.assertTrue(changed and all(4 * 8 <= i < 5 * 8 for i in changed), changed)
        got = P.parse_appearance(out, "ps3").sub_objects[0].draws[0]
        self.assertEqual((got.joints[4], got.weights[4]), ((1, 2, 0, 0), (155, 100, 0, 0)))

    def test_wider_stored_ranges_kept(self):
        """Stored ranges over bones 0..2 while only 1 and 2 are weighted: W13-4 would narrow them;
        a skin edit within them keeps them."""
        tree = L.parse_tree(self.template(), "app", "ps3")
        record = tree.top.pairs[0x64].block.elements[0].pairs[M.PS3_SUB["draws"]].block.elements[0]
        struct.pack_into(">4H", record.raw, M.PS3_RANGES, *M.ps3_bone_ranges({0, 1, 2}))
        stream = bytearray(block(record, M.DRAW["b"]))
        for at in range(0, len(stream), 2):
            if stream[at]:
                stream[at + 1] += 1  # palette [1, 2] -> [0, 1, 2]
        record.pairs[M.DRAW["b"]] = L.PairValue(L.Block(bytes(stream)))
        wide = L.write_tree(tree)
        app = P.parse_appearance(wide, "ps3")
        self.assertEqual(app.sub_objects[0].draws[0].palette, [0, 1, 2])
        out = I.import_mesh(app, gltf(one_weight(app, 4)))
        self.assertEqual(struct.unpack_from(">4H", self.draw_record(out).raw, M.PS3_RANGES), M.ps3_bone_ranges({0, 1, 2}))
        b0, b1 = block(self.draw_record(wide), M.DRAW["b"]), block(self.draw_record(out), M.DRAW["b"])
        self.assertEqual(b0[:32] + b0[40:], b1[:32] + b1[40:])
        self.assertEqual(P.parse_appearance(out, "ps3").sub_objects[0].draws[0].weights[4], (155, 100, 0, 0))

    def test_ranges_derived_for_a_new_bone(self):
        template = self.template()
        app = P.parse_appearance(template, "ps3")
        self.assertEqual(app.sub_objects[0].draws[0].palette, [1, 2])
        out = I.import_mesh(app, gltf(one_weight(app, 4, (0, 2, 0, 0), (155, 100, 0, 0))))
        after = self.draw_record(out)
        self.assertEqual(struct.unpack_from(">4H", after.raw, M.PS3_RANGES), M.ps3_bone_ranges({0, 1, 2}))
        got = P.parse_appearance(out, "ps3").sub_objects[0].draws[0]
        before = app.sub_objects[0].draws[0]
        self.assertEqual(got.palette, [0, 1, 2])
        self.assertEqual(got.joints[:4] + got.joints[5:], before.joints[:4] + before.joints[5:])
        self.assertEqual(got.weights[:4] + got.weights[5:], before.weights[:4] + before.weights[5:])
        self.assertEqual(got.local_joints[0], (2, 1, 1, 1))  # renumbered: bone 2 -> 2, bone 1 -> 1, zero slots bone 1

    def test_zero_weight_slots_kept(self):
        data = ps3(skinned=True, bones=BONES)
        tree = L.parse_tree(data, "app", "ps3")
        draw_record = tree.top.pairs[0x64].block.elements[0].pairs[M.PS3_SUB["draws"]].block.elements[0]
        stream = bytearray(block(draw_record, M.DRAW["b"]))
        palette = P.parse_appearance(data, "ps3").sub_objects[0].draws[0].palette
        self.assertGreater(len(palette), 1)
        for v in range(3):  # weights (255, 0, 0, 0), zero-weight slots 1
            stream[v * 8:v * 8 + 8] = bytes((255, 0, 0, 1, 0, 1, 0, 1))
        draw_record.pairs[M.DRAW["b"]] = L.PairValue(L.Block(bytes(stream)))
        data = L.write_tree(tree)
        app = P.parse_appearance(data, "ps3")
        bone = palette[0]
        other = palette[1]
        out = I.import_mesh(app, gltf(one_weight(app, 0, (bone, other, 0, 0), (155, 100, 0, 0))))
        got = block(self.draw_record(out), M.DRAW["b"])
        self.assertEqual(got[8:24], bytes(stream[8:24]))  # unchanged vertices: stored bytes
        self.assertEqual(got[0:8], bytes((155, 0, 100, 1, 0, 0, 0, 0)))


# --- V21-3 -----------------------------------------------------------------------------------------

def two_records(order="<"):
    build = "x360" if order == ">" else None
    tree = L.parse_tree(derived(switch_skinned3(order), build), "app", build)
    value = tree.top.pairs[0x64].block.elements[0].pairs[M.SWITCH_SUB["skins"]]
    first = value.block.elements[0]
    value.block.elements.append(L.Record(first.name, bytearray(first.raw), dict(first.pairs)))
    return L.write_tree(tree), build


class EmptyEveryRecord(unittest.TestCase):
    # stage 17 (W17-13, W17-14) replaces the V21-3 refusal: emptying every skinning record of a
    # sub-object removes the sub-object
    def test_only_record_refused(self):
        for order, build in (("<", None), (">", "x360")):
            app = P.parse_appearance(derived(switch_skinned3(order), build), build)
            out, report = I.import_mesh_report(app, read({"asset": {"version": "2.0"}}), missing="empty")
            self.assertEqual(P.parse_appearance(out, build).sub_objects, [])
            self.assertEqual(report.removed[-1], "high-poly set, sub-object 0")

    def test_no_indices_mark_on_the_only_record_refused(self):
        app = P.parse_appearance(derived(switch_skinned3()))
        doc = P.to_gltf(app)
        primitive = doc["meshes"][0]["primitives"][0]
        del primitive["indices"]
        primitive["extras"]["d3sno"]["no_indices"] = True
        out, report = I.import_mesh_report(app, read(doc))
        self.assertEqual(P.parse_appearance(out).sub_objects, [])
        self.assertEqual(report.removed, ["high-poly set, sub-object 0, skinning record 0", "high-poly set, sub-object 0"])

    def test_some_of_several_records(self):
        for order in ("<", ">"):
            data, build = two_records(order)
            app = P.parse_appearance(data, build)
            self.assertEqual(len(app.sub_objects[0].draws), 2)
            doc = P.to_gltf(app)
            doc["meshes"][0]["primitives"].pop(1)
            out, report = I.import_mesh_report(app, read(doc), missing="empty")
            self.assertEqual(report.changed, ["high-poly set, sub-object 0, draw 1"])
            again = P.parse_appearance(out, build)
            self.assertEqual(len(again.sub_objects[0].draws), 1)
            self.assertEqual(again.sub_objects[0].draws[0].positions, app.sub_objects[0].draws[0].positions)
            out = I.import_mesh(app, read({"asset": {"version": "2.0"}}), missing="empty")  # stage 17: removed
            self.assertEqual(P.parse_appearance(out, build).sub_objects, [])

    def test_ps3_not_affected(self):
        app = P.parse_appearance(ps3(skinned=True, bones=BONES), "ps3")
        out = I.import_mesh(app, read({"asset": {"version": "2.0"}}), missing="empty")  # W13-7: dropped
        # stage 17 (W17-13, W17-14): a sub-object left with no geometry is removed
        self.assertEqual(P.parse_appearance(out, "ps3").sub_objects, [])


# --- V21-4 -----------------------------------------------------------------------------------------

def no_triangles(order="<"):
    build = "x360" if order == ">" else None
    version = 302 if order == ">" else 306
    return laid_out(B.build_app_switch(version=version, order=order, pool_spare=0, sub_objects=[
        {"rigid": [(0, 0, 0), (10, 0, 0), (0, 10, 0)], "index_count": 0}]), build), build


class NoIndices(unittest.TestCase):
    def test_exporter_mark(self):
        data, build = no_triangles()
        primitive = P.to_gltf(P.parse_appearance(data, build))["meshes"][0]["primitives"][0]
        self.assertNotIn("indices", primitive)
        self.assertEqual(primitive["extras"], {"d3sno": {"draw": 0, "no_indices": True}})
        normal = P.to_gltf(P.parse_appearance(switch_rigid()))["meshes"][0]["primitives"][0]
        self.assertNotIn("no_indices", normal["extras"]["d3sno"])
        self.assertIn("indices", normal)

    def test_round_trip_byte_exact(self):
        for order in ("<", ">"):
            data, build = no_triangles(order)
            app = P.parse_appearance(data, build)
            self.assertEqual(app.sub_objects[0].draws[0].indices, [])
            doc = P.to_gltf(app)
            out, report = I.import_mesh_report(app, read(doc))
            self.assertEqual(out, data)
            self.assertEqual(report.changed, [])
            self.assertEqual(I.import_mesh(app, I.read_gltf(I.gltf_to_glb(doc))), data)
            self.assertEqual(I.import_mesh(app, P.to_obj(app)), data)
            skinned = laid_out(B.build_app_switch(version=302 if order == ">" else 306, order=order, bones=BONES,
                                                  pool_spare=0, sub_objects=[
                                                      {"skin": dict(SKIN2, indices=[]), "index_count": 0}]), build)
            app = P.parse_appearance(skinned, build)
            self.assertEqual(I.import_mesh(app, gltf(app)), skinned)

    def test_importer_honours_the_mark(self):
        app = P.parse_appearance(switch_rigid())
        doc = P.to_gltf(app)
        primitive = doc["meshes"][0]["primitives"][0]
        del primitive["indices"]
        self.assertEqual(I.import_mesh(app, read(doc)), bytes(app.source))  # no mark: 0, 1, 2
        primitive["extras"]["d3sno"]["no_indices"] = True
        out = P.parse_appearance(I.import_mesh(app, read(doc)))
        self.assertEqual(out.sub_objects[0].draws[0].indices, [])
        self.assertEqual(out.sub_objects[0].draws[0].positions, app.sub_objects[0].draws[0].positions)

    def test_mark_values(self):
        doc = P.to_gltf(P.parse_appearance(no_triangles()[0]))
        source = doc["meshes"][0]["primitives"][0]
        self.assertEqual(I._gltf_source(read(doc), source, None, None, "x").indices, [])
        for value in (False, 1, "true"):  # only JSON true is the mark: 0..count-1
            source["extras"]["d3sno"]["no_indices"] = value
            self.assertEqual(I._gltf_source(read(doc), source, None, None, "x").indices, [0, 1, 2], value)

    def test_marked_count_need_not_be_triangles(self):
        app = P.parse_appearance(no_triangles()[0])
        draw = app.sub_objects[0].draws[0]
        for name in ("positions", "normals", "colours_1", "colours_2", "uv1", "uv2_raw"):
            getattr(draw, name).append(getattr(draw, name)[0])
        doc = P.to_gltf(app)
        read(doc)  # 4 vertices without indices: accepted with the mark
        del doc["meshes"][0]["primitives"][0]["extras"]["d3sno"]["no_indices"]
        with self.assertRaisesRegex(FormatError, "4 vertices without indices are not whole triangles"):
            read(doc)

    def test_mark_with_indices_refused(self):
        doc = P.to_gltf(P.parse_appearance(switch_rigid()))
        doc["meshes"][0]["primitives"][0]["extras"]["d3sno"]["no_indices"] = True
        with self.assertRaisesRegex(FormatError, "no_indices with an indices accessor"):
            read(doc)


# --- V21-5 -----------------------------------------------------------------------------------------

def named(names, second=()):
    data = laid_out(B.build_app_switch(pool_spare=0, sub_objects=[
        {"rigid": [(0, 0, 0), (10 + k, 0, 0), (0, 10, 0)], "index_count": 3} for k in range(len(names))],
        second=[{"rigid": [(0, 0, 0), (5 + k, 0, 0), (0, 5, 0)], "index_count": 3} for k in range(len(second))]
        or None))
    app = P.parse_appearance(data)
    for sub, name in zip(app.sub_objects + app.extra_sub_objects, list(names) + list(second)):
        sub.name = name
    return app


def objects(text):
    return [line[2:] for line in text.splitlines() if line.startswith("o ")]


class RepeatedNames(unittest.TestCase):
    def test_exporter_names(self):
        app = named(["Body", "Head", "Body"], ["Body", "Arm", "Arm"])
        want = ["Body__sub0", "Head", "Body__sub2", "lowpoly_Body", "lowpoly_Arm__sub1", "lowpoly_Arm__sub2"]
        self.assertEqual(objects(P.to_obj(app)), want)
        doc = P.to_gltf(app)
        self.assertEqual([m["name"] for m in doc["meshes"]], want)
        self.assertEqual([n["name"] for n in doc["nodes"] if "mesh" in n], want)
        self.assertEqual(objects(P.to_obj(named(["A", "B"]))), ["A", "B"])  # unique: unchanged
        self.assertEqual(objects(P.to_obj(app, low_poly=False)), want[:3])

    def test_obj_round_trip_byte_exact(self):
        app = named(["Body", "Body"], ["Arm", "Arm"])
        for flip in (False, True):
            self.assertEqual(I.import_mesh(app, P.to_obj(app, flip_v=flip), flip_v=flip), bytes(app.source))
        doc = P.to_gltf(app)
        for mesh in doc["meshes"]:
            del mesh["extras"]
        self.assertEqual(I.import_mesh(app, read(doc)), bytes(app.source))

    def test_edit_reaches_the_named_sub_object(self):
        app = named(["Body", "Body"])
        edited = copy.deepcopy(app)
        edited.sub_objects[1].draws[0].normals[0] = (0.0, 1.0, 0.0)
        text = P.to_obj(edited)
        for name in ("Body__sub1", "Body__sub1.001"):  # step 3 after step 2
            out, report = I.import_mesh_report(app, text.replace("o Body__sub1", f"o {name}"))
            self.assertEqual(report.changed, ["high-poly set, sub-object 1, draw 0"], name)
        self.assertEqual(report.kept, ["high-poly set, sub-object 0, draw 0"])

    def test_unmatched_names(self):
        app = named(["Body", "Head"])
        text = P.to_obj(app)
        for wrong in ("Body__sub1", "Body__sub7", "Head__sub0"):  # sub-object i is not named <name>
            # stage 17 (W17-12): a name that matches no sub-object makes a new one (was refused)
            out, report = I.import_mesh_report(app, text.replace("o Head", f"o {wrong}"))
            self.assertEqual(report.added, ["high-poly set, sub-object 2"], wrong)
            self.assertEqual(report.kept, ["high-poly set, sub-object 0, draw 0", "high-poly set, sub-object 1, draw 0"])
        self.assertEqual(I.import_mesh(app, text.replace("o Head", "o Head__sub1")), bytes(app.source))

    def test_literal_name_and_suffix_ambiguous(self):
        app = named(["Body", "Body", "Body__sub1"])
        names = objects(P.to_obj(app))
        self.assertEqual(names, ["Body__sub0", "Body__sub1", "Body__sub1"])
        with self.assertRaisesRegex(FormatError, "fits several sub-objects"):
            I.import_mesh(app, P.to_obj(app))


if __name__ == "__main__":
    unittest.main()
