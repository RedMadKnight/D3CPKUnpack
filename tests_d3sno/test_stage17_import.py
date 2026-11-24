"""Stage 17 import half (``37_mesh_parts.md`` W17-12 to W17-14): exporter additions, new meshes and
primitives from glTF and OBJ (editor copies, naming, material choice, template part, skin rules),
``missing`` = keep | empty | remove and the report of added and removed parts. Synthetic files only
(``builders``, made to follow the W17 rules by ``test_stage17_writer.canon``).

Every written result goes through ``Checked.check``: strict parse, structure checks, W17-1 flags,
rebuild from zero and an unchanged ``write_app_mesh(result, derive=True)``."""

import _libpath  # noqa: F401  (python -I: the library folder on sys.path)
import copy
import struct
import unittest

import builders as B
from d3sno import FormatError
from d3sno import app_mesh as M
from d3sno import appearance as P
from d3sno import mesh_import as I
from test_stage12 import BONES, laid_out, ps3
from test_stage13 import grid
from test_stage16_mesh import add_accessor, read
from test_stage17_writer import (
    Checked, canon, djb, edited, multi_record, new_sub, ps3_cloth_draw, rigid_draw, switch_cloth_two, switch_skin,
    switch_two, triangles, with_materials,
)

FLOAT, USHORT, UBYTE = 5126, 5123, 5121


# --- helpers --------------------------------------------------------------------------------------

def tris(positions, indices, offset=(0.0, 0.0, 0.0)):
    """Triangles as position triples, rounded like ``test_stage17_writer.triangles``."""
    return [tuple(tuple(round(positions[i][c] + offset[c], 3) for c in range(3)) for i in indices[t:t + 3])
            for t in range(0, len(indices), 3)]


def primitive(doc, positions, indices, joints=None, weights=None, colours=True, material=None, extras=None):
    """A glTF primitive with POSITION, NORMAL, TEXCOORD_0, indices and optionally colours, joints and
    weights (as an editor writes them)."""
    n = len(positions)
    attributes = {
        "POSITION": add_accessor(doc, struct.pack(f"<{3 * n}f", *[c for p in positions for c in p]), FLOAT, "VEC3", n,
                                 min=[min(p[c] for p in positions) for c in range(3)],
                                 max=[max(p[c] for p in positions) for c in range(3)]),
        "NORMAL": add_accessor(doc, struct.pack(f"<{3 * n}f", *([0.0, 0.0, 1.0] * n)), FLOAT, "VEC3", n),
        "TEXCOORD_0": add_accessor(doc, struct.pack(f"<{2 * n}f", *([0.5, 0.25] * n)), FLOAT, "VEC2", n),
    }
    if colours:
        attributes["_D3_COLOR1"] = add_accessor(doc, bytes((1, 2, 3, 4)) * n, UBYTE, "VEC4", n)
        attributes["_D3_COLOR2"] = add_accessor(doc, bytes((5, 6, 7, 8)) * n, UBYTE, "VEC4", n)
    if joints is not None:
        attributes["JOINTS_0"] = add_accessor(doc, struct.pack(f"<{4 * n}H", *[c for j in joints for c in j]),
                                              USHORT, "VEC4", n)
        attributes["WEIGHTS_0"] = add_accessor(doc, struct.pack(f"<{4 * n}f", *[c for w in weights for c in w]),
                                               FLOAT, "VEC4", n)
    out = {"attributes": attributes, "mode": 4,
           "indices": add_accessor(doc, struct.pack(f"<{len(indices)}H", *indices), USHORT, "SCALAR", len(indices))}
    if material is not None:
        doc.setdefault("materials", []).append({"name": material})
        out["material"] = len(doc["materials"]) - 1
    if extras is not None:
        out["extras"] = extras
    return out


def add_mesh(doc, name, primitives, extras=None, skin=None, node_name=None):
    """A new mesh and a node using it (an editor adds an object)."""
    doc.setdefault("meshes", []).append({"name": name, "primitives": primitives})
    if extras is not None:
        doc["meshes"][-1]["extras"] = extras
    node = {"name": node_name if node_name is not None else name, "mesh": len(doc["meshes"]) - 1}
    if skin is not None:
        node["skin"] = skin
    doc.setdefault("nodes", []).append(node)
    doc["scenes"][0]["nodes"].append(len(doc["nodes"]) - 1)
    return len(doc["meshes"]) - 1


def mesh_node(doc, mesh):
    return next(n for n in doc["nodes"] if n.get("mesh") == mesh)


def editor_copy(doc, mesh, name, translation=None):
    """Duplicate a mesh as an editor does: custom properties (extras) copied, a ``.001`` name."""
    copied = copy.deepcopy(doc["meshes"][mesh])
    copied["name"] = name
    node = copy.deepcopy(mesh_node(doc, mesh))
    doc["meshes"].append(copied)
    node["mesh"], node["name"] = len(doc["meshes"]) - 1, name
    if translation is not None:
        node["translation"] = list(translation)
    doc["nodes"].append(node)
    doc["scenes"][0]["nodes"].append(len(doc["nodes"]) - 1)
    return len(doc["meshes"]) - 1


def drop_mesh(doc, mesh):
    """Delete a mesh's node (the mesh stays unreferenced)."""
    node = doc["nodes"].index(mesh_node(doc, mesh))
    doc["nodes"][node] = {"name": "empty"}


TRIANGLE = [(20.0, 0.0, 0.0), (26.0, 0.0, 0.0), (20.0, 6.0, 0.0)]
QUAD = ([(30.0, 0.0, 1.0), (34.0, 0.0, 1.0), (30.0, 4.0, 1.0), (34.0, 4.0, 1.0)], [0, 1, 2, 1, 3, 2])


def materials_file():
    """switch_two with a material table: sub-objects 0xAAA, 0xBBB, 0xAAA; diffuse handles 5, 7, 5 for
    look 0 and 6, 8, 6 for look 1."""
    data, build = switch_two()
    return with_materials(data, build), build


def ps3_two_draws():
    """A ps3 file whose sub-object "Body" has two rigid draws."""
    data = canon(ps3(extra_draw_format=0x81), "ps3")
    app = P.parse_appearance(data, "ps3")
    assert len(app.sub_objects[0].draws) == 2
    return data, "ps3"


def ps3_named(names):
    """A ps3 rigid file with one sub-object per name (the first is the builder's "Body")."""
    data, build = ps3_file_rigid()
    app = P.parse_appearance(data, build)
    extra = [new_sub(name, [rigid_draw([(4 * k, 0, 0), (4 * k + 3, 0, 0), (4 * k, 3, 0)], kind="ps3")])
             for k, name in enumerate(names[1:], 1)]
    return canon(M.write_app_mesh(app, edited(app, app.sub_objects + extra), derive=True), build), build


def ps3_file_rigid():
    return canon(ps3(), "ps3"), "ps3"


def ps3_skinned():
    return canon(ps3(skinned=True, bones=BONES), "ps3"), "ps3"


# --- W17-12 exporter additions --------------------------------------------------------------------

class ExporterExtras(unittest.TestCase):
    def test_ps3_name_and_material(self):
        data, build = ps3_named(["Body", "Left Arm"])
        app = P.parse_appearance(data, build)
        doc = P.to_gltf(app)
        self.assertEqual([m["name"] for m in doc["meshes"]], ["Body", "Left_Arm"])  # object names
        self.assertEqual([m["extras"]["d3sno"]["name"] for m in doc["meshes"]], ["Body", "Left Arm"])  # stored
        self.assertEqual([m["extras"]["d3sno"]["material"] for m in doc["meshes"]],
                         [s.material_hash for s in app.sub_objects])

    def test_switch_object_names_and_materials(self):
        data, build = materials_file()
        app = P.parse_appearance(data, build)
        doc = P.to_gltf(app)
        self.assertEqual([m["extras"]["d3sno"] for m in doc["meshes"]],
                         [{"geo_set": 0, "sub_object": i, "name": f"subobject_{i}", "material": h}
                          for i, h in enumerate((0xAAA, 0xBBB, 0xAAA))])
        # V21-5 names and the low-poly prefix are the exporter's object names
        low = P.parse_appearance(canon(laid_out(B.build_app_switch(pool_spare=0, second=[
            {"rigid": [(0, 0, 0), (2, 0, 0), (0, 2, 0)], "index_count": 3}]))))
        self.assertEqual([m["extras"]["d3sno"]["name"] for m in P.to_gltf(low)["meshes"]],
                         ["subobject_0", "lowpoly_subobject_0"])

    def test_round_trips_still_byte_exact(self):
        """``37`` Check: the stage-16 glTF and OBJ round trips with the new mesh extras."""
        cases = [switch_two(), switch_two(">"), switch_skin(), switch_skin(">"), ps3_file_rigid(), ps3_skinned(),
                 multi_record(), materials_file(), ps3_named(["Body", "Arm", "Leg"]), ps3_two_draws()]
        for data, build in cases:
            app = P.parse_appearance(data, build)
            self.assertEqual(I.import_mesh(app, read(P.to_gltf(app))), data, build)
            self.assertEqual(I.import_mesh(app, read(P.to_gltf(app)), missing="remove"), data, build)
            self.assertEqual(I.import_mesh(app, P.to_obj(app)), data, build)
            out, report = I.import_mesh_report(app, P.to_obj(app), missing="remove")
            self.assertEqual((out, report.added, report.removed), (data, [], []))


# --- W17-12 editor copies ------------------------------------------------------------------------

class EditorCopies(Checked):
    def test_duplicate_mesh_as_an_editor_does(self):
        """``37`` Check: duplicate a mesh in glTF, as an editor does (copied extras, ``.001`` name)."""
        for data, build in (materials_file(), (with_materials(*switch_two(">")), "x360")):
            app = P.parse_appearance(data, build)
            doc = P.to_gltf(app)
            editor_copy(doc, 1, "subobject_1.001", translation=(0.0, 0.0, 5.0))
            out, report = I.import_mesh_report(app, read(doc))
            self.assertEqual(report.added, ["high-poly set, sub-object 3"])
            self.assertEqual(report.removed, [])
            again = self.check(out, build)
            self.assertEqual(len(again.sub_objects), 4)
            new = again.sub_objects[3]
            self.assertEqual(new.sub_hash, djb("subobject_1_001"))  # "." -> "_" (W17-12)
            self.assertEqual(new.material_hash, 0xBBB)  # rule 1: the copied extras' material
            self.assertEqual(triangles(new),  # the copy's node transform applied
                             [tuple((x, y, z + 5.0) for x, y, z in t) for t in triangles(app.sub_objects[1])])
            # the template sub-objects are untouched
            for a, b in zip(again.sub_objects[:3], app.sub_objects):
                self.assertEqual(triangles(a), triangles(b))

    def test_exact_name_is_the_match(self):
        data, build = materials_file()
        app = P.parse_appearance(data, build)
        doc = P.to_gltf(app)
        copy_mesh = editor_copy(doc, 1, "subobject_1.001")
        # the copy comes first in node order; the original (exact exporter name) is still the match
        nodes = doc["nodes"]
        original = nodes.index(mesh_node(doc, 1))
        nodes[original], nodes[-1] = nodes[-1], nodes[original]
        doc["meshes"][copy_mesh]["primitives"][0]["attributes"]["POSITION"] = add_accessor(
            doc, struct.pack("<9f", *[c for p in TRIANGLE for c in p]), FLOAT, "VEC3", 3)
        out, report = I.import_mesh_report(app, read(doc))
        again = self.check(out, build)
        self.assertEqual(triangles(again.sub_objects[1]), triangles(app.sub_objects[1]))  # the original
        self.assertEqual(triangles(again.sub_objects[3]), tris(TRIANGLE, [0, 1, 2]))  # the copy
        self.assertEqual(report.kept[1], "high-poly set, sub-object 1, draw 0")
        self.assertEqual(report.added, ["high-poly set, sub-object 3"])

    def test_no_exact_name_first_in_node_order(self):
        data, build = materials_file()
        app = P.parse_appearance(data, build)
        doc = P.to_gltf(app)
        editor_copy(doc, 1, "subobject_1.001", translation=(0.0, 0.0, 2.0))
        doc["meshes"][1]["name"] = mesh_node(doc, 1)["name"] = "Renamed"
        out = I.import_mesh(app, read(doc))
        again = self.check(out, build)
        self.assertEqual(again.sub_objects[3].sub_hash, djb("subobject_1_001"))  # the later one is the copy
        self.assertEqual(triangles(again.sub_objects[1]), triangles(app.sub_objects[1]))

    def test_instanced_mesh_and_ps3_names(self):
        """The same mesh on two nodes: the second is a copy; ps3 names take ``_2``, ``_3``."""
        data, build = ps3_file_rigid()
        app = P.parse_appearance(data, build)
        doc = P.to_gltf(app)
        node = copy.deepcopy(mesh_node(doc, 0))
        node["translation"] = [0.0, 0.0, 1.0]
        doc["nodes"] += [node, dict(node, translation=[0.0, 0.0, 2.0])]
        editor_copy(doc, 0, "Body.001")
        out, report = I.import_mesh_report(app, read(doc))
        again = self.check(out, build)
        self.assertEqual([s.name for s in again.sub_objects], ["Body", "Body_2", "Body_3", "Body_001"])
        self.assertEqual([s.sub_hash for s in again.sub_objects], [djb(s.name) for s in again.sub_objects])
        self.assertEqual(report.added, [f"high-poly set, sub-object {i}" for i in (1, 2, 3)])
        self.assertEqual(triangles(again.sub_objects[2]),
                         [tuple((x, y, z + 2.0) for x, y, z in t) for t in triangles(app.sub_objects[0])])

    def test_obj_copy(self):
        data, build = materials_file()
        app = P.parse_appearance(data, build)
        text = P.to_obj(app)
        start = text.index("o subobject_2")
        block = text[start:].replace("o subobject_2", "o subobject_2.001")
        block = "\n".join(line for line in block.splitlines() if not line.startswith("f "))
        base = text.count("\nv ")
        faces = [line for line in text[start:].splitlines() if line.startswith("f ")]
        shifted = []
        for line in faces:
            corners = [c.split("/") for c in line.split()[1:]]
            shifted.append("f " + " ".join("/".join(str(int(v) + 3) for v in c) for c in corners))
        out, report = I.import_mesh_report(app, text + block + "\n" + "\n".join(shifted) + "\n")
        again = self.check(out, build)
        self.assertEqual(report.added, ["high-poly set, sub-object 3"])
        self.assertEqual(again.sub_objects[3].sub_hash, djb("subobject_2_001"))
        self.assertEqual(again.sub_objects[3].material_hash, 0xAAA)  # its template part's (OBJ: no extras)
        self.assertEqual(triangles(again.sub_objects[3]), triangles(app.sub_objects[2]))
        self.assertEqual(base, 9)


# --- W17-12 new meshes: geo set, name, material, template part ------------------------------------

class NewMeshes(Checked):
    def doc_with(self, app, name, extras=None, material=None, skin=None, positions=TRIANGLE, indices=(0, 1, 2),
                 **options):
        doc = P.to_gltf(app)
        add_mesh(doc, name, [primitive(doc, positions, list(indices), material=material, **options)],
                 extras=extras, skin=skin)
        return doc

    def test_new_mesh_new_material(self):
        """``37`` Check: add a new mesh with a material other than its template part's (rule 1)."""
        data, build = materials_file()
        app = P.parse_appearance(data, build)
        doc = self.doc_with(app, "Lamp", extras={"d3sno": {"material": 0xBBB}})
        out, report = I.import_mesh_report(app, read(doc))
        again = self.check(out, build)
        self.assertEqual(report.added, ["high-poly set, sub-object 3"])
        lamp = again.sub_objects[3]
        self.assertEqual((lamp.sub_hash, lamp.material_hash), (djb("Lamp"), 0xBBB))
        self.assertEqual(triangles(lamp), tris(TRIANGLE, [0, 1, 2]))
        self.assertEqual(again.sub_objects[3].draws[0].colours_1, [bytes((1, 2, 3, 4))] * 3)
        self.assertEqual(len(again.material_data.materials), 2)  # no entry added

    def test_material_rules_in_order(self):
        data, build = materials_file()
        app = P.parse_appearance(data, build)
        cases = [
            ({"d3sno": {"material": 0xBBB}}, None, 0, 0xBBB),  # 1: in the table
            ({"d3sno": {"material": 0x123}}, None, 0, 0xAAA),  # 1 not in the table -> 4: the first sub-object
            ({"d3sno": {"material": 0x123, "sub_object": 1}}, None, 0, 0xBBB),  # 2: the named sub-object
            (None, "texture_7_draw_0", 0, 0xBBB),  # 3: handle 7 is sub-object 1's (look 0)
            (None, "texture_8_draw_0", 1, 0xBBB),  # 3 with look 1
            (None, "texture_8_draw_0", 0, 0xAAA),  # no sub-object uses handle 8 in look 0 -> 4
            (None, "my_material", 0, 0xAAA),  # 4
        ]
        for extras, material, look, want in cases:
            doc = self.doc_with(app, "Lamp", extras=extras, material=material)
            out = I.import_mesh(app, read(doc), look=look)
            self.assertEqual(P.parse_appearance(out, build).sub_objects[3].material_hash, want, (extras, material))
        self.assertEqual(I.new_part_material(app, None, [7], 0, None), (0xBBB, app.sub_objects[1]))
        self.assertEqual(I.new_part_material(app, None, [], 0, None), (0xAAA, app.sub_objects[0]))
        # template part: the copy's matched sub-object wins over the material's first user
        self.assertEqual(I.new_part_material(app, {"material": 0xAAA}, [], 0, app.sub_objects[2]),
                         (0xAAA, app.sub_objects[2]))
        with self.assertRaises(ValueError):
            I.import_mesh(app, read(P.to_gltf(app)), look=-1)

    def test_template_part_values(self):
        """The new record copies its W17-3 values from the template part (W17-10 via W17-12)."""
        from d3sno import sno_layout as L
        data, build = materials_file()
        tree = L.parse_tree(data, "app", build)
        records = tree.top.pairs[0x64].block.elements
        struct.pack_into("<iif", records[1].raw, 0x50, 77, 3, 1.0)  # physics, fade group, fade alpha
        data = L.write_tree(tree)
        app = P.parse_appearance(data, build)
        out = I.import_mesh(app, read(self.doc_with(app, "Lamp", extras={"d3sno": {"material": 0xBBB}})))
        record = L.parse_tree(out, "app", build).top.pairs[0x64].block.elements[3]
        self.assertEqual(struct.unpack_from("<iif", record.raw, 0x50), (77, 3, 1.0))
        out = I.import_mesh(app, read(self.doc_with(app, "Lamp")))  # template: the first sub-object
        record = L.parse_tree(out, "app", build).top.pairs[0x64].block.elements[3]
        self.assertEqual(struct.unpack_from("<iif", record.raw, 0x50), (0, 0, 0.0))

    def test_geo_set_and_low_poly(self):
        data, build = switch_two()
        app = P.parse_appearance(data, build)
        out, report = I.import_mesh_report(app, read(self.doc_with(app, "lowpoly_Extra.1")))
        again = self.check(out, build)
        self.assertEqual(report.added, ["low-poly set, sub-object 0"])
        self.assertEqual(len(again.sub_objects), 3)
        self.assertEqual(again.extra_sub_objects[0].sub_hash, djb("Extra_1"))  # prefix gone, "." -> "_"
        doc = self.doc_with(app, "Prop", extras={"d3sno": {"geo_set": 1}})
        self.assertEqual(I.import_mesh_report(app, read(doc))[1].added, ["low-poly set, sub-object 0"])
        doc = self.doc_with(app, "lowpoly_Prop", extras={"d3sno": {"geo_set": 0}})
        out = I.import_mesh(app, read(doc))
        self.assertEqual(P.parse_appearance(out).sub_objects[3].sub_hash, djb("Prop"))
        # low_poly=False: a new geo-set-1 mesh is skipped
        out, report = I.import_mesh_report(app, read(self.doc_with(app, "lowpoly_Extra")), low_poly=False)
        self.assertEqual(out, data)
        self.assertIn("new low-poly sub-object and is not imported", report.warnings[0])
        out = I.import_mesh(app, read(self.doc_with(app, "Extra")), low_poly=False)
        self.assertEqual(len(P.parse_appearance(out).sub_objects), 4)

    def test_switch_name_collisions(self):
        data, build = switch_two()
        app = P.parse_appearance(data, build)
        doc = self.doc_with(app, "Extra")
        add_mesh(doc, "Extra", [primitive(doc, QUAD[0], QUAD[1])])
        add_mesh(doc, "extra", [primitive(doc, QUAD[0], QUAD[1])])  # W17-2 lowers A..Z: the same hash
        again = self.check(I.import_mesh(app, read(doc)), build)
        self.assertEqual([s.sub_hash for s in again.sub_objects[3:]], [djb("Extra"), djb("Extra_2"), djb("extra_3")])
        self.assertEqual(I.new_part_name("lowpoly_A.b.c", "switch", {djb("A_b_c")}), "A_b_c_2")
        # a stored switch sub-object hash takes its name too
        from d3sno import sno_layout as L
        tree = L.parse_tree(data, "app", build)
        struct.pack_into("<I", tree.top.pairs[0x64].block.elements[1].raw, 0x68, djb("Extra"))
        app = P.parse_appearance(L.write_tree(tree), build)
        again = P.parse_appearance(I.import_mesh(app, read(self.doc_with(app, "Extra"))), build)
        self.assertEqual(again.sub_objects[3].sub_hash, djb("Extra_2"))
        with self.assertRaisesRegex(FormatError, "leaves no sub-object name"):
            I.new_part_name("lowpoly_", "switch", set())

    def test_ps3_names(self):
        data, build = ps3_named(["Body", "Body_2"])
        app = P.parse_appearance(data, build)
        doc = self.doc_with(app, "Body")
        out = I.import_mesh(app, read(doc))
        self.assertEqual(self.check(out, build).sub_objects[2].name, "Body_3")  # smallest free number
        for name in ("n" * 127, "Arm 2"):
            again = P.parse_appearance(I.import_mesh(app, read(self.doc_with(app, name))), build)
            self.assertEqual(again.sub_objects[2].name, name)
        for name, message in (("n" * 128, "longer than 127 bytes"), ("Ärm", "not ASCII")):
            with self.assertRaisesRegex(FormatError, rf"^glTF mesh 2 \('{name}'\): .*{message}"):
                I.import_mesh(app, read(self.doc_with(app, name)))
        doc = self.doc_with(app, "n" * 126)
        add_mesh(doc, "n" * 126, [primitive(doc, QUAD[0], QUAD[1])])  # the second gets "_2": 128 bytes
        with self.assertRaisesRegex(FormatError, "glTF mesh 3 .*longer than 127 bytes"):
            I.import_mesh(app, read(doc))

    def test_unnamed_mesh_refused(self):
        data, build = switch_two()
        app = P.parse_appearance(data, build)
        doc = self.doc_with(app, "Extra")
        del doc["meshes"][-1]["name"]
        doc["nodes"][-1].pop("name")
        with self.assertRaisesRegex(FormatError, "matches no sub-object and has no name for a new one"):
            I.import_mesh(app, read(doc))
        doc["nodes"][-1]["name"] = "FromNode"  # the node name stands in
        self.assertEqual(P.parse_appearance(I.import_mesh(app, read(doc))).sub_objects[3].sub_hash, djb("FromNode"))

    def test_needs_normals_and_uv(self):
        data, build = switch_two()
        app = P.parse_appearance(data, build)
        for attribute, message in (("NORMAL", "normals missing for a new part"), ("TEXCOORD_0", "UV1")):
            doc = self.doc_with(app, "Extra")
            del doc["meshes"][-1]["primitives"][0]["attributes"][attribute]
            with self.assertRaisesRegex(FormatError, message):
                I.import_mesh(app, read(doc))
        doc = self.doc_with(app, "Extra", colours=False)
        out, report = I.import_mesh_report(app, read(doc))
        self.assertIn("no colours given for a new part", report.warnings[-1])
        self.assertEqual(P.parse_appearance(out).sub_objects[3].draws[0].colours_1, [bytes(4)] * 3)

    def test_several_primitives_one_rigid_list(self):
        data, build = switch_two()
        app = P.parse_appearance(data, build)
        doc = self.doc_with(app, "Extra")
        doc["meshes"][-1]["primitives"].append(primitive(doc, QUAD[0], QUAD[1]))
        again = self.check(I.import_mesh(app, read(doc)), build)
        self.assertEqual(len(again.sub_objects[3].draws), 1)
        self.assertEqual(triangles(again.sub_objects[3]), tris(TRIANGLE, [0, 1, 2]) + tris(*QUAD))

    def test_ps3_new_mesh_rigid_and_skinned(self):
        data, build = ps3_file_rigid()
        app = P.parse_appearance(data, build)
        again = self.check(I.import_mesh(app, read(self.doc_with(app, "Prop"))), build)
        self.assertEqual([s.name for s in again.sub_objects], ["Body", "Prop"])
        self.assertEqual(again.sub_objects[1].draws[0].skinned, False)
        data, build = ps3_skinned()
        app = P.parse_appearance(data, build)
        doc = self.doc_with(app, "Arm", skin=0, joints=[(0, 2, 0, 0)] * 3, weights=[(0.75, 0.25, 0, 0)] * 3)
        again = self.check(I.import_mesh(app, read(doc)), build)
        arm = again.sub_objects[1].draws[0]
        self.assertEqual(arm.joints, [(0, 2, 0, 0)] * 3)
        self.assertEqual(arm.weights, [(191, 64, 0, 0)] * 3)  # W16-9
        self.assertEqual(triangles(again.sub_objects[1]), tris(TRIANGLE, [0, 1, 2]))


# --- W17-12 skin rules ---------------------------------------------------------------------------

class SkinRules(Checked):
    def test_new_part_in_a_file_with_bones_needs_joints(self):
        for data, build in (switch_skin(), ps3_skinned()):
            app = P.parse_appearance(data, build)
            doc = P.to_gltf(app)
            add_mesh(doc, "Arm", [primitive(doc, TRIANGLE, [0, 1, 2])])
            with self.assertRaisesRegex(FormatError, "needs joints and weights"):
                I.import_mesh(app, read(doc))
            doc = P.to_gltf(app)
            doc["meshes"][0]["primitives"].append(primitive(doc, TRIANGLE, [0, 1, 2]))
            with self.assertRaisesRegex(FormatError, "needs joints and weights"):
                I.import_mesh(app, read(doc))

    def test_joints_in_a_file_without_bones(self):
        data, build = switch_two()
        app = P.parse_appearance(data, build)
        doc = P.to_gltf(app)
        add_mesh(doc, "Arm", [primitive(doc, TRIANGLE, [0, 1, 2], joints=[(0, 0, 0, 0)] * 3,
                                        weights=[(1, 0, 0, 0)] * 3)])
        with self.assertRaisesRegex(FormatError, r"glTF mesh 3 \('Arm'\) primitive 0: a primitive with joints "
                                                 r"in a file without bones \(W17-12\)"):
            I.import_mesh(app, read(doc))
        doc = P.to_gltf(app)
        doc["meshes"][0]["primitives"].append(primitive(doc, TRIANGLE, [0, 1, 2], joints=[(0, 0, 0, 0)] * 3,
                                                        weights=[(1, 0, 0, 0)] * 3))
        with self.assertRaisesRegex(FormatError, "primitive 1: a primitive with joints in a file without bones"):
            I.import_mesh(app, read(doc))

    def test_obj_new_skinned_part_needs_nearest(self):
        data, build = switch_skin()
        app = P.parse_appearance(data, build)
        text = P.to_obj(app) + "o Arm\nv 0 0 1\nv 9 0 1\nv 0 1 1\nvt 0 0\nvn 0 0 1\nf 4/2/2 5/2/2 6/2/2\n"
        with self.assertRaisesRegex(FormatError, "OBJ object 'Arm': a new skinned part from OBJ .* skin='nearest'"):
            I.import_mesh(app, text)
        out, report = I.import_mesh_report(app, text, skin="nearest")
        again = self.check(out, build)
        self.assertEqual(report.added, ["high-poly set, sub-object 1"])
        stored = app.sub_objects[0].draws[0]
        arm = again.sub_objects[1].draws[0]
        # nearest template vertex: (0,0,0) -> vertex 0, (9,0,0) -> vertex 1 (10,0,0), (0,1,0) -> vertex 0
        self.assertEqual(arm.joints, [stored.joints[0], stored.joints[1], stored.joints[0]])
        self.assertEqual(arm.weights, [stored.weights[0], stored.weights[1], stored.weights[0]])

    def test_obj_new_group_in_skinned_sub_object(self):
        data, build = ps3_skinned()
        app = P.parse_appearance(data, build)
        text = P.to_obj(app) + "g extra\nv 0 9 0\nv 1 9 0\nv 0 10 0\nvt 0 0\nvn 0 0 1\nf 4/2/2 5/2/2 6/2/2\n"
        with self.assertRaisesRegex(FormatError, "new skinned part from OBJ"):
            I.import_mesh(app, text)
        out, report = I.import_mesh_report(app, text, skin="nearest")
        self.assertEqual(report.added, ["high-poly set, sub-object 0, draw 1"])
        again = self.check(out, build)
        stored = app.sub_objects[0].draws[0]
        self.assertEqual(again.sub_objects[0].draws[1].joints, [stored.joints[2]] * 3)  # nearest: (0, 10, 0)

    def test_nearest_takes_the_nearest_vertex(self):
        """Template vertices with different skins: each new vertex takes its nearest one's slots."""
        data, build = switch_skin()
        app = P.parse_appearance(data, build)
        draw = app.sub_objects[0].draws[0]
        draw.joints[:] = [(0, 0, 0, 0), (1, 0, 0, 0), (2, 0, 0, 0)]
        draw.weights[:] = [(255, 0, 0, 0)] * 3
        data = canon(M.write_app_mesh(app, derive=True), build)
        app = P.parse_appearance(data, build)
        stored = app.sub_objects[0].draws[0]
        self.assertEqual(len({j for j in stored.joints}), 3)
        text = P.to_obj(app) + ("o Arm\nv 1 1 0\nv 9 1 0\nv 1 8 0\nv 20 20 0\nvt 0 0\nvn 0 0 1\n"
                                "f 4/2/2 5/2/2 6/2/2\nf 5/2/2 7/2/2 6/2/2\n")
        again = self.check(I.import_mesh(app, text, skin="nearest"), build)
        arm = again.sub_objects[1].draws[0]
        nearest = [0, 1, 2, 1]  # (20, 20): (10, 0) and (0, 10) are equally near: the first in draw order
        self.assertEqual(arm.joints, [stored.joints[q] for q in nearest])
        self.assertEqual(arm.weights, [stored.weights[q] for q in nearest])

    def test_nearest_needs_a_skinned_template(self):
        sub = new_sub("X", [rigid_draw(TRIANGLE)])
        with self.assertRaisesRegex(FormatError, "no skinned vertex to take the skin from"):
            I._nearest_slots(sub, TRIANGLE, "label")


# --- W17-12 new primitives ----------------------------------------------------------------------------

class NewPrimitives(Checked):
    def test_ps3_new_primitive_cut_by_budget(self):
        data, build = ps3_file_rigid()
        app = P.parse_appearance(data, build)
        positions, indices = grid(50)
        positions = [(x * 0.5, y * 0.5, 1.0) for x, y, _ in positions]
        doc = P.to_gltf(app)
        doc["meshes"][0]["primitives"].append(primitive(doc, positions, indices))
        out, report = I.import_mesh_report(app, read(doc))
        again = self.check(out, build)
        draws = again.sub_objects[0].draws
        self.assertGreater(len(draws), 2)  # 21 * 2601 + 15000 > 55998: cut (W17-5a)
        self.assertEqual(report.added, [f"high-poly set, sub-object 0, draw {k}" for k in range(1, len(draws))])
        for draw in draws[1:]:
            self.assertLessEqual(21 * len(draw.positions) + len(draw.indices), 55998)
        self.assertEqual(triangles(again.sub_objects[0]), triangles(app.sub_objects[0]) + tris(positions, indices))

    def test_switch_skinned_records_rebuilt(self):
        for order in ("<", ">"):
            data, build = switch_skin(order)
            app = P.parse_appearance(data, build)
            doc = P.to_gltf(app)
            doc["meshes"][0]["primitives"].append(primitive(doc, TRIANGLE, [0, 1, 2], joints=[(1, 0, 0, 0)] * 3,
                                                            weights=[(1, 0, 0, 0)] * 3))
            out, report = I.import_mesh_report(app, read(doc))
            self.assertEqual(report.added, ["high-poly set, sub-object 0, skinning record 1"])
            again = self.check(out, build)
            self.assertEqual(len(again.sub_objects[0].draws), 1)  # merged and re-split by W17-7: 2 bones
            self.assertEqual(triangles(again.sub_objects[0]), triangles(app.sub_objects[0]) + tris(TRIANGLE, [0, 1, 2]))

    def test_switch_rigid_appended_to_the_rigid_list(self):
        for order in ("<", ">"):
            data, build = switch_two(order)
            app = P.parse_appearance(data, build)
            doc = P.to_gltf(app)
            doc["meshes"][2]["primitives"].append(primitive(doc, QUAD[0], QUAD[1]))
            out, report = I.import_mesh_report(app, read(doc))
            self.assertEqual((report.added, report.removed), ([], []))
            self.assertEqual(report.changed, ["high-poly set, sub-object 2, draw 0"])
            again = self.check(out, build)
            self.assertEqual(len(again.sub_objects[2].draws), 1)
            self.assertEqual(triangles(again.sub_objects[2]), triangles(app.sub_objects[2]) + tris(*QUAD))

    def test_primitive_with_material_suffix_is_matched(self):
        """A primitive whose material names a draw is that draw, not new geometry."""
        data, build = ps3_two_draws()
        app = P.parse_appearance(data, build)
        doc = P.to_gltf(app)
        for p in doc["meshes"][0]["primitives"]:
            del p["extras"]
        self.assertEqual(I.import_mesh(app, read(doc)), data)
        doc["meshes"][0]["primitives"].append(primitive(doc, TRIANGLE, [0, 1, 2], material="Glass"))
        out, report = I.import_mesh_report(app, read(doc))
        self.assertEqual(report.added, ["high-poly set, sub-object 0, draw 2"])
        doc["meshes"][0]["primitives"][2]["material"] = 0  # "d3sno_draw_0": draw 0 twice
        with self.assertRaisesRegex(FormatError, "two primitives for"):
            I.import_mesh(app, read(doc))

    def test_primitives_naming_no_draw_need_matching_counts(self):
        """When no primitive names a draw, the counts must agree (W16-7 kept): not new geometry."""
        data, build = ps3_two_draws()
        app = P.parse_appearance(data, build)
        doc = P.to_gltf(app)
        for p in doc["meshes"][0]["primitives"]:
            del p["extras"], p["material"]
        self.assertEqual(I.import_mesh(app, read(doc)), data)  # by order
        doc["meshes"][0]["primitives"].append(primitive(doc, TRIANGLE, [0, 1, 2]))
        with self.assertRaisesRegex(FormatError, "names no draw and 3 primitives do not match the 2 draws"):
            I.import_mesh(app, read(doc))

    def test_cloth_sub_object_gets_no_new_geometry(self):
        data, build = switch_cloth_two()
        app = P.parse_appearance(data, build)
        doc = P.to_gltf(app)
        doc["meshes"][1]["primitives"].append(primitive(doc, TRIANGLE, [0, 1, 2]))
        with self.assertRaisesRegex(FormatError, "high-poly set, sub-object 1: new geometry .*V21-1"):
            I.import_mesh(app, read(doc))

    def test_obj_new_group(self):
        """An OBJ ``g`` that matches no draw: ps3 new draw, switch appended to the rigid list."""
        data, build = ps3_file_rigid()
        app = P.parse_appearance(data, build)
        text = P.to_obj(app) + "g extra\nv 20 0 0\nv 26 0 0\nv 20 6 0\nvt 0 0\nvn 0 0 1\nf 4/2/2 5/2/2 6/2/2\n"
        out, report = I.import_mesh_report(app, text)
        self.assertEqual(report.added, ["high-poly set, sub-object 0, draw 1"])
        again = self.check(out, build)
        self.assertEqual(triangles(again.sub_objects[0]), triangles(app.sub_objects[0]) + tris(TRIANGLE, [0, 1, 2]))
        data, build = switch_two()
        app = P.parse_appearance(data, build)
        text = P.to_obj(app) + "g extra\nv 20 0 0\nv 26 0 0\nv 20 6 0\nvt 0 0\nvn 0 0 1\nf 10/4/4 11/4/4 12/4/4\n"
        out, report = I.import_mesh_report(app, text)
        self.assertEqual(report.changed, ["high-poly set, sub-object 2, draw 0"])
        again = self.check(out, build)
        self.assertEqual(triangles(again.sub_objects[2]), triangles(app.sub_objects[2]) + tris(TRIANGLE, [0, 1, 2]))

    def test_obj_new_object(self):
        data, build = materials_file()
        app = P.parse_appearance(data, build)
        text = P.to_obj(app) + "o Crate.2\nv 20 0 0\nv 26 0 0\nv 20 6 0\nvt 0 0\nvn 0 0 1\nf 10/4/4 11/4/4 12/4/4\n"
        out, report = I.import_mesh_report(app, text)
        self.assertEqual(report.added, ["high-poly set, sub-object 3"])
        again = self.check(out, build)
        self.assertEqual(again.sub_objects[3].sub_hash, djb("Crate_2"))
        self.assertEqual(again.sub_objects[3].material_hash, 0xAAA)  # OBJ: the template part's material
        self.assertEqual(triangles(again.sub_objects[3]), tris(TRIANGLE, [0, 1, 2]))
        with self.assertRaisesRegex(FormatError, "matches no sub-object and has no name"):
            I.import_mesh(app, P.to_obj(app) + "o\nv 20 0 0\nv 26 0 0\nv 20 6 0\nf 10 11 12\n")


# --- W17-13 missing, W17-14 removal and report --------------------------------------------------------

class Missing(Checked):
    def test_remove_sub_object(self):
        """``37`` Check: remove the first and the last sub-object."""
        for order in ("<", ">"):
            data, build = switch_two(order)
            app = P.parse_appearance(data, build)
            for gone, left in ((0, (1, 2)), (2, (0, 1))):
                doc = P.to_gltf(app)
                drop_mesh(doc, gone)
                self.assertEqual(I.import_mesh(app, read(doc)), data)  # keep
                out, report = I.import_mesh_report(app, read(doc), missing="remove")
                self.assertEqual(report.removed, [f"high-poly set, sub-object {gone}"])
                again = self.check(out, build)
                self.assertEqual([triangles(s) for s in again.sub_objects],
                                 [triangles(app.sub_objects[i]) for i in left])

    def test_remove_material_entry_goes(self):
        data, build = materials_file()
        app = P.parse_appearance(data, build)
        doc = P.to_gltf(app)
        drop_mesh(doc, 1)  # the only user of 0xBBB
        out, report = I.import_mesh_report(app, read(doc), missing="remove")
        self.assertEqual(report.removed, ["high-poly set, sub-object 1", "material 0x00000BBB"])
        again = self.check(out, build)
        self.assertEqual([m.hash for m in again.material_data.materials], [0xAAA])

    def test_remove_ps3_draw(self):
        """``37`` Check: remove a draw."""
        data, build = ps3_two_draws()
        app = P.parse_appearance(data, build)
        doc = P.to_gltf(app)
        doc["meshes"][0]["primitives"].pop(0)
        self.assertEqual(I.import_mesh(app, read(doc)), data)
        out, report = I.import_mesh_report(app, read(doc), missing="remove")
        self.assertEqual(report.removed, ["high-poly set, sub-object 0, draw 0"])
        self.assertEqual(report.kept, ["high-poly set, sub-object 0, draw 1"])
        again = self.check(out, build)
        self.assertEqual(triangles(again.sub_objects[0]), triangles(edited_draws(app, [1])))

    def test_remove_skinning_record(self):
        """``37`` Check: remove a skinning record; the rest is re-split by W17-7."""
        data, build = multi_record()
        app = P.parse_appearance(data, build)
        doc = P.to_gltf(app)
        doc["meshes"][1]["primitives"].pop(1)
        out, report = I.import_mesh_report(app, read(doc), missing="remove")
        self.assertEqual(report.removed, ["high-poly set, sub-object 1, skinning record 1"])
        again = self.check(out, build)
        self.assertEqual(triangles(again.sub_objects[1]), triangles(edited_draws(app, [0], 1)))

    def test_remove_every_draw_of_a_sub_object(self):
        """``37`` Check: remove every draw of a sub-object (the sub-object goes, W17-14)."""
        data, build = ps3_two_draws()
        app = P.parse_appearance(data, build)
        for missing in ("empty", "remove"):
            out, report = I.import_mesh_report(app, read({"asset": {"version": "2.0"}}), missing=missing)
            self.assertEqual(P.parse_appearance(out, build).sub_objects, [])
            self.assertEqual(report.removed[-1], "high-poly set, sub-object 0")
        self.assertEqual(I.import_mesh_report(app, read({"asset": {"version": "2.0"}}), missing="empty")[1].removed,
                         ["high-poly set, sub-object 0, draw 0", "high-poly set, sub-object 0, draw 1",
                          "high-poly set, sub-object 0"])
        # draw 0 replaced, draw 1 removed, two primitives naming no draw added (3 primitives, 2 draws)
        doc = P.to_gltf(app)
        doc["meshes"][0]["primitives"] = [primitive(doc, TRIANGLE, [0, 1, 2], extras={"d3sno": {}}),
                                          primitive(doc, QUAD[0], QUAD[1], extras={"d3sno": {"draw": 0}}),
                                          primitive(doc, TRIANGLE, [0, 2, 1])]
        out, report = I.import_mesh_report(app, read(doc), missing="remove")
        self.assertEqual(report.removed, ["high-poly set, sub-object 0, draw 1"])
        self.assertEqual(report.added, ["high-poly set, sub-object 0, draw 1", "high-poly set, sub-object 0, draw 2"])
        self.assertEqual(report.changed, ["high-poly set, sub-object 0, draw 0"])
        again = self.check(out, build)
        self.assertEqual(triangles(again.sub_objects[0]),
                         tris(*QUAD) + tris(TRIANGLE, [0, 1, 2]) + tris(TRIANGLE, [0, 2, 1]))
        # two primitives for two draws: one naming no draw is matched by order (W16-7)
        primitives = doc["meshes"][0]["primitives"]
        doc["meshes"][0]["primitives"] = [primitives[1], primitives[0]]
        out, report = I.import_mesh_report(app, read(doc), missing="remove")
        self.assertEqual((report.added, report.removed), ([], []))
        self.assertEqual(report.changed, ["high-poly set, sub-object 0, draw 0", "high-poly set, sub-object 0, draw 1"])

    def test_empty_rigid_and_record(self):
        """W17-13 "empty": an emptied part goes (W17-14), and a sub-object left without geometry."""
        data, build = switch_two()
        app = P.parse_appearance(data, build)
        doc = P.to_gltf(app)
        drop_mesh(doc, 1)
        out, report = I.import_mesh_report(app, read(doc), missing="empty")
        self.assertEqual(report.changed, ["high-poly set, sub-object 1, draw 0"])
        self.assertEqual(report.removed, ["high-poly set, sub-object 1, rigid list", "high-poly set, sub-object 1"])
        self.check(out, build)
        data, build = multi_record()
        app = P.parse_appearance(data, build)
        doc = P.to_gltf(app)
        doc["meshes"][1]["primitives"].pop(0)
        out, report = I.import_mesh_report(app, read(doc), missing="empty")
        self.assertEqual(report.removed, ["high-poly set, sub-object 1, skinning record 0"])
        again = self.check(out, build)
        self.assertEqual(triangles(again.sub_objects[1]), triangles(edited_draws(app, [1], 1)))

    def test_no_indices_mark_empties_one_ps3_draw(self):
        data, build = ps3_two_draws()
        app = P.parse_appearance(data, build)
        doc = P.to_gltf(app)
        p = doc["meshes"][0]["primitives"][1]
        del p["indices"]
        p["extras"]["d3sno"]["no_indices"] = True
        out, report = I.import_mesh_report(app, read(doc))
        self.assertEqual(report.removed, ["high-poly set, sub-object 0, draw 1"])
        again = self.check(out, build)
        self.assertEqual(len(again.sub_objects[0].draws), 1)

    def test_cloth(self):
        data, build = switch_cloth_two()
        app = P.parse_appearance(data, build)
        doc = P.to_gltf(app)
        drop_mesh(doc, 1)  # the cloth sub-object: removed whole
        out, report = I.import_mesh_report(app, read(doc), missing="remove")
        self.assertEqual(report.removed, ["high-poly set, sub-object 1"])
        self.assertEqual(len(self.check(out, build).sub_objects), 2)
        with self.assertRaisesRegex(FormatError, "cloth"):
            I.import_mesh(app, read(doc), missing="empty")
        data, build = ps3_cloth_draw()
        app = P.parse_appearance(data, build)
        app.sub_objects[0].draws.append(copy.deepcopy(app.sub_objects[0].draws[0]))  # refused before writing
        doc = P.to_gltf(app)
        self.assertEqual(len(doc["meshes"][0]["primitives"]), 2)
        doc["meshes"][0]["primitives"].pop(0)
        with self.assertRaisesRegex(FormatError, r"sub-object 0, draw 0: .*draw list \(removing the draw\).*V21-1"):
            I.import_mesh(app, read(doc), missing="remove")
        self.assertEqual(I.import_mesh(app, read(doc)), data)  # keep: nothing changed

    def test_low_poly_false_keeps_geo_set_1(self):
        data = canon(ps3(extra_draw_format=0x81, second=[{"vertices": ((0, 0, 0), (2, 0, 0), (0, 2, 0))}]), "ps3")
        app = P.parse_appearance(data, "ps3")
        doc = P.to_gltf(app, low_poly=False)
        self.assertEqual(I.import_mesh(app, read(doc), missing="remove", low_poly=False), data)
        out, report = I.import_mesh_report(app, read(doc), missing="remove")
        self.assertEqual(report.removed, ["low-poly set, sub-object 0"])

    def test_bad_option(self):
        app = P.parse_appearance(switch_two()[0])
        with self.assertRaisesRegex(ValueError, "'keep', 'empty' or 'remove'"):
            I.import_mesh(app, read(P.to_gltf(app)), missing="drop")
        self.assertEqual(I.MISSING, ("keep", "empty", "remove"))


def edited_draws(app, keep, index=0):
    """Sub-object ``index`` of ``app`` with only the draws ``keep``."""
    sub = (app.sub_objects)[index]
    return new_sub("x", [sub.draws[k] for k in keep])


class ReportAndRules(Checked):
    def test_unchanged_document_gives_template_bytes(self):
        data, build = materials_file()
        app = P.parse_appearance(data, build)
        out, report = I.import_mesh_report(app, read(P.to_gltf(app)))
        self.assertIs(type(out), bytes)
        self.assertEqual((out, report.added, report.removed), (data, [], []))

    def test_added_and_removed_together(self):
        data, build = switch_two()
        app = P.parse_appearance(data, build)
        doc = P.to_gltf(app)
        drop_mesh(doc, 0)
        add_mesh(doc, "Extra", [primitive(doc, TRIANGLE, [0, 1, 2])])
        out, report = I.import_mesh_report(app, read(doc), missing="remove")
        self.assertEqual(report.removed, ["high-poly set, sub-object 0"])
        self.assertEqual(report.added, ["high-poly set, sub-object 2"])
        again = self.check(out, build)
        self.assertEqual([s.sub_hash for s in again.sub_objects], [0, 0, djb("Extra")])


if __name__ == "__main__":
    unittest.main()
