"""Stage 16 mesh half: exporter changes (W16-5), glTF and OBJ reading and mesh import onto a
template ``.app`` (W16-6 to W16-11). Synthetic files only: the test builders' files, made to follow
the W13 rules by writing them once with ``derive=True``."""

import base64
import copy
import json
import os
import struct
import tempfile
import unittest

import builders as B
from d3sno import FormatError
from d3sno import app_mesh as M
from d3sno import appearance as P
from d3sno import mesh_import as I
from test_stage12 import BONES, laid_out, ps3, switch_cloth, switch_rigid, switch_skinned, x360_skinned
from test_stage13 import grid, set_mesh

HERE = os.path.dirname(os.path.abspath(__file__))
SKIN2 = {"palette": [2, 1], "vertices": [(0, 0, 0), (10, 0, 0), (0, 10, 0)], "indices": [0, 1, 2]}


def switch_skinned3(order="<"):
    """Switch (or x360) with the three-bone skeleton; joints (2, 1), weights (200, 55)."""
    version, build = (302, "x360") if order == ">" else (306, "switch")
    return laid_out(B.build_app_switch(version=version, order=order, bones=BONES, pool_spare=0,
                                       sub_objects=[{"skin": SKIN2, "index_count": 3}]), build)


def derived(data, build=None):
    return M.write_app_mesh(P.parse_appearance(data, build), derive=True)


def grid_template(data, build=None, cells=2):
    """The file with its first draw replaced by a grid of 2 x cells^2 triangles, written once with
    every W13 rule so that it follows them."""
    app = P.parse_appearance(data, build)
    positions, indices = grid(cells)
    set_mesh(app.sub_objects[0].draws[0], [(x + 1.0, y + 2.0, 3.0) for x, y, _ in positions], indices)
    return derived(M.write_app_mesh(app, derive=True), build)


ROUND_TRIP = {
    "switch rigid": (switch_rigid(), None),
    "x360 rigid": (switch_rigid(">"), "x360"),
    "switch skinned": (switch_skinned3(), None),
    "x360 skinned": (switch_skinned3(">"), "x360"),
    "switch skinned, one bone": (switch_skinned(), None),
    "x360 skinned, one bone": (x360_skinned(), "x360"),
    "switch cloth": (switch_cloth(), None),
    "ps3 rigid": (ps3(), "ps3"),
    "ps3 skinned": (ps3(skinned=True, bones=BONES), "ps3"),
    "ps3 fat cloth": (ps3(fat=[(0.0, 0.0, 0.0), (4.0, 0.0, 0.0), (0.0, 4.0, 0.0)], cloth=True), "ps3"),
    "ps3 two draws, low poly": (ps3(extra_draw_format=0x81, second=[{"vertices": ((0, 0, 0), (2, 0, 0), (0, 2, 0))}]),
                                "ps3"),
    "switch two sub-objects": (laid_out(B.build_app_switch(pool_spare=0, sub_objects=[
        {"rigid": [(0, 0, 0), (10, 0, 0), (0, 10, 0)], "index_count": 3},
        {"rigid": [(0, 0, 0), (5, 0, 0), (0, 5, 0)], "index_count": 3}])), None),
}


def gltf_text(app, **options) -> bytes:
    return P.gltf_dumps(P.to_gltf(app, **options)).encode()


def blob_of(doc: dict) -> bytearray:
    return bytearray(base64.b64decode(doc["buffers"][0]["uri"].split(",", 1)[1]))


def set_blob(doc: dict, blob: bytes) -> None:
    buffer = {"byteLength": len(blob),
              "uri": "data:application/octet-stream;base64," + base64.b64encode(bytes(blob)).decode()}
    if doc["buffers"]:
        doc["buffers"][0] = buffer
    else:
        doc["buffers"].append(buffer)


def add_accessor(doc: dict, raw: bytes, component: int, kind: str, count: int, **extra) -> int:
    blob = blob_of(doc)
    blob += bytes(-len(blob) % 4)
    doc["bufferViews"].append({"buffer": 0, "byteOffset": len(blob), "byteLength": len(raw)})
    blob += raw
    set_blob(doc, blob)
    accessor = {"bufferView": len(doc["bufferViews"]) - 1, "componentType": component, "count": count, "type": kind}
    accessor.update(extra)
    doc["accessors"].append(accessor)
    return len(doc["accessors"]) - 1


def accessor_rows(doc: dict, index: int, fmt: str) -> list:
    accessor = doc["accessors"][index]
    view = doc["bufferViews"][accessor["bufferView"]]
    raw = blob_of(doc)[view["byteOffset"]:view["byteOffset"] + view["byteLength"]]
    return list(struct.iter_unpack("<" + fmt, raw))


def read(doc: dict) -> I.GltfDocument:
    return I.read_gltf(json.dumps(doc).encode())


def triangle_doc(positions=((0, 0, 0), (1, 0, 0), (0, 1, 0)), extensions=None) -> dict:
    """A minimal glTF with one triangle mesh named ``subobject_0``."""
    doc = {"asset": {"version": "2.0"}, "buffers": [], "bufferViews": [], "accessors": [],
           "meshes": [{"name": "subobject_0", "primitives": [{"attributes": {}}]}],
           "nodes": [{"mesh": 0}]}
    set_blob(doc, b"")
    doc["meshes"][0]["primitives"][0]["attributes"]["POSITION"] = add_accessor(
        doc, struct.pack(f"<{3 * len(positions)}f", *[c for p in positions for c in p]), 5126, "VEC3", len(positions))
    if extensions:
        doc["extensionsRequired"] = extensions
    return doc


def moved(app, delta=(0.25, -0.5, 0.125)):
    out = copy.deepcopy(app)
    for sub in out.sub_objects + out.extra_sub_objects:
        for draw in sub.draws:
            draw.positions[:] = [tuple(p[k] + delta[k] for k in range(3)) for p in draw.positions]
    return out


# --- W16-5 ---------------------------------------------------------------------------------------

class Exporter(unittest.TestCase):
    def test_mesh_and_primitive_extras(self):
        app = P.parse_appearance(ROUND_TRIP["ps3 two draws, low poly"][0], "ps3")
        doc = P.to_gltf(app)
        # stage 17 (W17-12): the mesh extras also carry the sub-object name and material hash
        self.assertEqual([m["extras"] for m in doc["meshes"]],
                         [{"d3sno": {"geo_set": 0, "sub_object": 0, "name": "Body", "material": 0}},
                          {"d3sno": {"geo_set": 1, "sub_object": 0, "name": "Low", "material": 0}}])
        self.assertEqual([p["extras"] for p in doc["meshes"][0]["primitives"]],
                         [{"d3sno": {"draw": 0}}, {"d3sno": {"draw": 1}}])

    def test_draw_index_counts_skipped_draws(self):
        app = P.parse_appearance(ps3(extra_draw_format=0x81), "ps3")
        draws = app.sub_objects[0].draws
        draws[0] = P.Draw(False, [], [], [], [], [], [], [], kind="ps3")  # no positions: skipped
        doc = P.to_gltf(app)
        primitives = doc["meshes"][0]["primitives"]
        self.assertEqual([p["extras"]["d3sno"]["draw"] for p in primitives], [1])
        self.assertEqual(doc["materials"][primitives[0]["material"]]["name"], "d3sno_draw_1")
        lines = P.to_obj(app).splitlines()
        self.assertNotIn("g draw_0", lines)
        self.assertIn("g draw_1", lines)

    def test_materials_per_draw_shared_by_name(self):
        data = laid_out(B.build_app_switch(pool_spare=0, sub_objects=[
            {"rigid": [(0, 0, 0), (1, 0, 0), (0, 1, 0)], "index_count": 3},
            {"rigid": [(0, 0, 0), (1, 0, 0), (0, 1, 0)], "index_count": 3}]))
        doc = P.to_gltf(P.parse_appearance(data))
        self.assertEqual(doc["materials"], [{"name": "d3sno_draw_0"}])
        self.assertEqual([m["primitives"][0]["material"] for m in doc["meshes"]], [0, 0])
        two = P.to_gltf(P.parse_appearance(ps3(extra_draw_format=0x81), "ps3"))
        self.assertEqual([m["name"] for m in two["materials"]], ["d3sno_draw_0", "d3sno_draw_1"])
        self.assertEqual([p["material"] for p in two["meshes"][0]["primitives"]], [0, 1])

    def test_texture_material_names(self):
        data = B.add_app_materials(B.build_app_switch(), "switch", [1], [(0x10, [[(1, 44)]])])
        data = B.set_sub_object_material(data, "switch", 0, 0x10)
        doc = P.to_gltf(P.parse_appearance(data), texture_resolver=lambda handle: "t.png")
        self.assertEqual(doc["materials"][0]["name"], "texture_44_draw_0")
        self.assertEqual(doc["materials"][0]["pbrMetallicRoughness"]["baseColorTexture"], {"index": 0})

    def test_colours_and_uv2_attributes(self):
        app = P.parse_appearance(switch_rigid())
        doc = P.to_gltf(app)
        attributes = doc["meshes"][0]["primitives"][0]["attributes"]
        draw = app.sub_objects[0].draws[0]
        for name, colours in (("_D3_COLOR1", draw.colours_1), ("_D3_COLOR2", draw.colours_2)):
            accessor = doc["accessors"][attributes[name]]
            self.assertEqual((accessor["componentType"], accessor["type"]), (5121, "VEC4"))
            self.assertNotIn("normalized", accessor)
            self.assertEqual([bytes(r) for r in accessor_rows(doc, attributes[name], "4B")], colours)
        self.assertEqual(draw.colours_1[0], bytes((1, 2, 3, 4)))
        accessor = doc["accessors"][attributes["_D3_UV2"]]
        self.assertEqual((accessor["componentType"], accessor["type"]), (5123, "VEC2"))
        self.assertEqual(accessor_rows(doc, attributes["_D3_UV2"], "2H"), [(9, 10)] * 3)
        self.assertNotIn("TEXCOORD_1", attributes)
        self.assertIn("TEXCOORD_1", P.to_gltf(app, lightmap_v_scale=0.5)["meshes"][0]["primitives"][0]["attributes"])

    def test_x360_colours_in_switch_order(self):
        doc = P.to_gltf(P.parse_appearance(switch_rigid(">"), "x360"))
        attributes = doc["meshes"][0]["primitives"][0]["attributes"]
        self.assertEqual(accessor_rows(doc, attributes["_D3_COLOR1"], "4B")[0], (1, 2, 3, 4))

    def test_empty_colour_list_not_written(self):
        app = P.parse_appearance(switch_rigid())
        app.sub_objects[0].draws[0].colours_2 = []
        attributes = P.to_gltf(app)["meshes"][0]["primitives"][0]["attributes"]
        self.assertIn("_D3_COLOR1", attributes)
        self.assertNotIn("_D3_COLOR2", attributes)

    def test_skin_extras(self):
        doc = P.to_gltf(P.parse_appearance(switch_skinned3()))
        self.assertEqual(doc["skins"][0]["extras"], {"d3sno": {"bones": 3}})
        doc = P.to_gltf(P.parse_appearance(ps3(skinned=True, bones=BONES), "ps3"))
        self.assertEqual(doc["skins"][0]["extras"], {"bind_from_rest_inverse": True, "d3sno": {"bones": 3}})

    def test_obj_groups(self):
        text = P.to_obj(P.parse_appearance(ps3(extra_draw_format=0x81), "ps3"))
        lines = text.splitlines()
        self.assertEqual([l for l in lines if l[:2] in ("o ", "g ")], ["o Body", "g draw_0", "g draw_1"])
        self.assertLess(lines.index("g draw_1"), lines.index("v 1.000000 2.000000 3.000000", lines.index("g draw_1")))


# --- W16-6 ---------------------------------------------------------------------------------------

class ReadGltf(unittest.TestCase):
    def test_json_and_glb(self):
        app = P.parse_appearance(switch_rigid())
        doc = P.to_gltf(app)
        a, b = I.read_gltf(P.gltf_dumps(doc).encode()), I.read_gltf(I.gltf_to_glb(doc))
        self.assertEqual(a.json["meshes"], b.json["meshes"])
        self.assertEqual(a.buffers, b.buffers)
        self.assertEqual(I.read_gltf(P.gltf_dumps(doc)).buffers, a.buffers)  # str input

    def test_component_types_and_normalised(self):
        doc = triangle_doc()
        cases = [(5120, "b", (-128, 0, 127, 5), True, (-1.0, 0.0, 1.0, 5 / 127)),
                 (5121, "B", (0, 255, 51, 1), True, (0.0, 1.0, 0.2, 1 / 255)),
                 (5122, "h", (-32768, 32767, 0, 1), True, (-1.0, 1.0, 0.0, 1 / 32767)),
                 (5123, "H", (0, 65535, 1, 2), True, (0.0, 1.0, 1 / 65535, 2 / 65535)),
                 (5125, "I", (0, 7, 70000, 1), False, (0, 7, 70000, 1)),
                 (5123, "H", (0, 65535, 1, 2), False, (0, 65535, 1, 2)),
                 (5126, "f", (0.5, -2.0, 3.0, 0.0), False, (0.5, -2.0, 3.0, 0.0))]
        indices = []
        for comp, code, values, normalised, _ in cases:
            extra = {"normalized": True} if normalised else {}
            indices.append(add_accessor(doc, struct.pack("<4" + code, *values) + bytes(-4 * struct.calcsize(code) % 4),
                                        comp, "VEC4", 1, **extra))
        gltf = read(doc)
        for index, case in zip(indices, cases):
            self.assertEqual(gltf.accessor(index), [case[4]], case)

    def test_byte_stride(self):
        doc = triangle_doc()
        raw = b"".join(struct.pack("<3fI", x, y, 0.0, 0xDEADBEEF) for x, y in ((5, 6), (7, 8)))
        index = add_accessor(doc, raw, 5126, "VEC3", 2)
        doc["bufferViews"][-1]["byteStride"] = 16
        self.assertEqual(read(doc).accessor(index), [(5.0, 6.0, 0.0), (7.0, 8.0, 0.0)])
        doc["bufferViews"][-1]["byteStride"] = 8  # below the element size
        with self.assertRaisesRegex(FormatError, "byteStride 8 below"):
            read(doc)

    def test_matrix_column_padding(self):
        doc = triangle_doc()
        index = add_accessor(doc, bytes((1, 2, 0, 0, 3, 4, 0, 0)), 5121, "MAT2", 1)
        self.assertEqual(read(doc).accessor(index), [(1, 2, 3, 4)])

    def test_sparse(self):
        doc = triangle_doc()
        base = add_accessor(doc, struct.pack("<6f", 1, 1, 1, 2, 2, 2), 5126, "VEC3", 2)
        ind = add_accessor(doc, struct.pack("<H", 1) + bytes(2), 5123, "SCALAR", 1)
        val = add_accessor(doc, struct.pack("<3f", 9, 8, 7), 5126, "VEC3", 1)
        sparse = {"count": 1, "indices": {"bufferView": doc["accessors"][ind]["bufferView"], "componentType": 5123},
                  "values": {"bufferView": doc["accessors"][val]["bufferView"]}}
        doc["accessors"][base]["sparse"] = sparse
        self.assertEqual(read(doc).accessor(base), [(1.0, 1.0, 1.0), (9.0, 8.0, 7.0)])
        doc["accessors"].append({"componentType": 5126, "type": "VEC3", "count": 2, "sparse": sparse})
        self.assertEqual(read(doc).accessor(len(doc["accessors"]) - 1), [(0.0, 0.0, 0.0), (9.0, 8.0, 7.0)])
        bad = copy.deepcopy(doc)
        bad["accessors"][base]["count"] = 1  # sparse index 1 past count 1
        bad["accessors"][base]["sparse"]["count"] = 1
        with self.assertRaisesRegex(FormatError, "sparse index 1 past count 1"):
            read(bad)
        bad = copy.deepcopy(doc)
        two = add_accessor(bad, struct.pack("<2H", 1, 1), 5123, "SCALAR", 2)
        bad["accessors"][base]["sparse"].update(count=2, indices={"bufferView": bad["accessors"][two]["bufferView"],
                                                                  "componentType": 5123})
        with self.assertRaisesRegex(FormatError, "strictly increase|past buffer view"):
            read(bad)

    def test_bounds(self):
        doc = triangle_doc()
        doc["accessors"][0]["count"] = 4
        doc["meshes"] = []
        with self.assertRaisesRegex(FormatError, "past buffer view"):
            read(doc)
        doc = triangle_doc()
        doc["bufferViews"][0]["byteLength"] += 4
        with self.assertRaisesRegex(FormatError, "past buffer 0"):
            read(doc)
        doc = triangle_doc()
        doc["accessors"][0]["byteOffset"] = 2
        with self.assertRaisesRegex(FormatError, "not a multiple of 4"):
            read(doc)
        doc = triangle_doc()
        doc["buffers"][0]["byteLength"] += 1
        with self.assertRaisesRegex(FormatError, "for byteLength"):
            read(doc)

    def test_extensions(self):
        doc = triangle_doc(extensions=["KHR_mesh_quantization"])
        doc["extensionsUsed"] = ["KHR_mesh_quantization"]
        quantised = add_accessor(doc, struct.pack("<3h3h3h", 0, 0, 0, 2, 0, 0, 0, 2, 0) + bytes(2), 5122, "VEC3", 3)
        doc["meshes"][0]["primitives"][0]["attributes"]["POSITION"] = quantised
        self.assertEqual(read(doc).accessor(quantised)[1], (2, 0, 0))
        with self.assertRaisesRegex(FormatError, "KHR_draco_mesh_compression"):
            read(triangle_doc(extensions=["KHR_draco_mesh_compression"]))

    def test_mode_and_indices(self):
        doc = triangle_doc()
        doc["meshes"][0]["primitives"][0]["mode"] = 1
        with self.assertRaisesRegex(FormatError, "mode 1"):
            read(doc)
        doc = triangle_doc()
        doc["meshes"][0]["primitives"][0]["indices"] = add_accessor(doc, struct.pack("<3H", 0, 1, 3) + bytes(2),
                                                                    5123, "SCALAR", 3)
        with self.assertRaisesRegex(FormatError, "index 3 past 3 vertices"):
            read(doc)
        doc = triangle_doc()
        doc["meshes"][0]["primitives"][0]["indices"] = add_accessor(doc, struct.pack("<2H", 0, 1), 5123, "SCALAR", 2)
        with self.assertRaisesRegex(FormatError, "whole triangles"):
            read(doc)

    def test_no_indices_uses_the_vertex_order(self):
        doc = read(triangle_doc())
        source = I._gltf_source(doc, doc.json["meshes"][0]["primitives"][0], None, None, "x")
        self.assertEqual(source.indices, [0, 1, 2])

    def test_containers_refused(self):
        glb = I.gltf_to_glb(P.to_gltf(P.parse_appearance(switch_rigid())))
        for bad, message in ((b"glTF" + glb[4:8], "shorter"), (glb[:4] + struct.pack("<I", 1) + glb[8:], "version 1"),
                             (glb + b"\0\0\0\0", "header length"),
                             (glb[:16] + struct.pack("<I", 0x004E4942) + glb[20:], "first chunk is not JSON"),
                             (b'{"asset": {"version": "1.0"}}', "asset.version"),
                             (b"{not json", "does not parse"),
                             (b'{"asset": {"version": "2.0"}, "x": NaN}', "does not parse")):
            with self.assertRaisesRegex(FormatError, message):
                I.read_gltf(bad)
        doc = triangle_doc()
        doc["buffers"][0]["uri"] = "data:application/octet-stream;base64,@@@"
        with self.assertRaisesRegex(FormatError, "base64"):
            read(doc)

    def test_side_files(self):
        doc = triangle_doc()
        blob = blob_of(doc)
        with tempfile.TemporaryDirectory(dir=HERE) as folder:
            with open(os.path.join(folder, "model data.bin"), "wb") as handle:
                handle.write(bytes(blob))
            doc["buffers"][0] = {"byteLength": len(blob), "uri": "model%20data.bin"}
            self.assertEqual(I.read_gltf(json.dumps(doc).encode(), base_path=folder).buffers, [bytes(blob)])
            with self.assertRaisesRegex(FormatError, "needs a base path"):
                read(doc)
            for uri in ("../outside.bin", "sub/../../outside.bin", "/abs.bin", "C:/abs.bin", "http://x/y.bin"):
                doc["buffers"][0]["uri"] = uri
                with self.assertRaisesRegex(FormatError, "leaves the folder|is not relative", msg=uri):
                    I.read_gltf(json.dumps(doc).encode(), base_path=folder)
            doc["buffers"][0]["uri"] = "missing.bin"
            with self.assertRaisesRegex(FormatError, "cannot read"):
                I.read_gltf(json.dumps(doc).encode(), base_path=folder)


# --- exact round trips (W16-5 .. W16-11, Check) ------------------------------------------------------

class RoundTrip(unittest.TestCase):
    def test_gltf_json_glb_and_obj(self):
        for name, (data, build) in ROUND_TRIP.items():
            app = P.parse_appearance(data, build)
            for flip in (False, True):
                doc = P.to_gltf(app, flip_v=flip)
                out, report = I.import_mesh_report(app, I.read_gltf(P.gltf_dumps(doc).encode()), flip_v=flip)
                self.assertEqual(out, data, name)
                self.assertEqual(report.changed, [], name)
                self.assertEqual(len(report.kept), sum(len(s.draws) for s in P.export_sub_objects(app)), name)
                self.assertEqual(I.import_mesh(app, I.read_gltf(I.gltf_to_glb(doc)), flip_v=flip), data, name)
                self.assertEqual(I.import_mesh(app, P.to_obj(app, flip_v=flip), flip_v=flip), data, name)
                self.assertEqual(I.import_mesh(data, I.read_obj(P.to_obj(app, flip_v=flip)), flip_v=flip), data, name)

    def test_rule_following_grids(self):
        for data, build in ((switch_rigid(), None), (switch_skinned3(">"), "x360"), (ps3(skinned=True, bones=BONES), "ps3")):
            template = grid_template(data, build)
            self.assertEqual(derived(template, build), template)
            app = P.parse_appearance(template, build)
            self.assertEqual(I.import_mesh(app, read(P.to_gltf(app))), template)
            self.assertEqual(I.import_mesh(app, P.to_obj(app)), template)

    def test_lightmap_texcoord_and_options(self):
        app = P.parse_appearance(switch_rigid())
        for scale in (0.5, 0):
            doc = P.to_gltf(app, lightmap_v_scale=scale)
            for primitive in doc["meshes"][0]["primitives"]:
                del primitive["attributes"]["_D3_UV2"]
            self.assertEqual(I.import_mesh(app, read(doc), lightmap_v_scale=scale), bytes(app.source))
        self.assertEqual(I.import_mesh(app, P.to_obj(app, low_poly=False), low_poly=False), bytes(app.source))

    def test_editor_names_and_order(self):
        """No extras: matched by name with a '.001' suffix, draws by order."""
        app = P.parse_appearance(ROUND_TRIP["ps3 two draws, low poly"][0], "ps3")
        doc = P.to_gltf(app)
        for mesh in doc["meshes"]:
            del mesh["extras"]
            mesh["name"] += ".001"
            for primitive in mesh["primitives"]:
                del primitive["extras"]
                del primitive["material"]
        del doc["materials"]
        self.assertEqual(I.import_mesh(app, read(doc)), bytes(app.source))


# --- W16-7 ---------------------------------------------------------------------------------------

def two_subs():
    return P.parse_appearance(ROUND_TRIP["switch two sub-objects"][0])


class Matching(unittest.TestCase):
    def strip(self, doc):
        for mesh in doc["meshes"]:
            mesh.pop("extras", None)
        return doc

    def test_extras_win_over_names(self):
        app = two_subs()
        doc = P.to_gltf(moved(app))
        doc["meshes"][0]["name"], doc["meshes"][1]["name"] = "subobject_1", "subobject_0"
        out = P.parse_appearance(I.import_mesh(app, read(doc)))
        self.assertEqual(out.sub_objects[0].draws[0].positions[1][0], app.sub_objects[0].draws[0].positions[1][0] + 0.25)

    def test_ambiguous_name(self):
        app = two_subs()
        for sub in app.sub_objects:
            sub.name = "Body"
        doc = self.strip(P.to_gltf(app))
        for mesh, node in zip(doc["meshes"], (n for n in doc["nodes"] if "mesh" in n)):
            mesh["name"] = node["name"] = "Body"  # V21-5 exports Body__sub<i>; an editor's plain name
        with self.assertRaisesRegex(FormatError, "fits several sub-objects: high-poly set, sub-object 0; "
                                                 "high-poly set, sub-object 1"):
            I.import_mesh(app, read(doc))
        with self.assertRaisesRegex(FormatError, "fits several"):
            I.import_mesh(app, P.to_obj(app).replace("__sub0", "").replace("__sub1", ""))

    def test_unmatched_mesh(self):
        # stage 17 (W17-12): a mesh that matches no sub-object is a new sub-object (was refused, W16-7)
        app = two_subs()
        doc = self.strip(P.to_gltf(app))
        doc["meshes"][1]["name"] = doc["nodes"][1]["name"] = "Extra"
        for document in (read(doc), P.to_obj(app).replace("o subobject_1", "o Extra")):
            out, report = I.import_mesh_report(app, document)
            self.assertEqual(report.added, ["high-poly set, sub-object 2"])
            self.assertEqual(len(P.parse_appearance(out).sub_objects), 3)
        doc["meshes"][1].pop("name")
        doc["nodes"][1].pop("name")
        with self.assertRaisesRegex(FormatError, "matches no sub-object and has no name"):
            I.import_mesh(app, read(doc))

    def test_two_meshes_for_one_sub_object(self):
        # stage 17 (W17-12): the second mesh for one sub-object is an editor copy and becomes a new
        # sub-object (was refused "two meshes for ...", W16-7)
        app = two_subs()
        doc = P.to_gltf(app)
        doc["nodes"].append({"name": "copy", "mesh": 0})
        doc2 = self.strip(P.to_gltf(app))
        doc2["meshes"][1]["name"] = doc2["nodes"][1]["name"] = "subobject_0.002"
        text = P.to_obj(app).replace("o subobject_1", "o subobject_0.001")
        for document, copied in ((read(doc), 0), (read(doc2), 1), (text, 1)):
            out, report = I.import_mesh_report(app, document)
            self.assertEqual(report.added, ["high-poly set, sub-object 2"])
            again = P.parse_appearance(out)
            self.assertEqual(again.sub_objects[2].draws[0].positions, app.sub_objects[copied].draws[0].positions)

    def test_draw_by_material_name(self):
        app = P.parse_appearance(ps3(extra_draw_format=0x81), "ps3")
        doc = P.to_gltf(moved(app))
        primitives = doc["meshes"][0]["primitives"]
        for primitive in primitives:
            del primitive["extras"]
        primitives.reverse()  # order no longer helps; the material names do
        out = P.parse_appearance(I.import_mesh(app, read(doc)), "ps3")
        self.assertEqual(out.sub_objects[0].draws[1].positions[0], (1.25, 1.5, 3.125))

    def test_draw_refusals(self):
        app = P.parse_appearance(ps3(extra_draw_format=0x81), "ps3")
        doc = P.to_gltf(app)
        primitives = doc["meshes"][0]["primitives"]
        for primitive in primitives:
            del primitive["extras"]
            del primitive["material"]
        primitives.append(copy.deepcopy(primitives[0]))
        with self.assertRaisesRegex(FormatError, "names no draw and 3 primitives do not match the 2 draws"):
            I.import_mesh(app, read(doc))
        doc = P.to_gltf(app)
        doc["meshes"][0]["primitives"][1]["extras"]["d3sno"]["draw"] = 0
        with self.assertRaisesRegex(FormatError, "two primitives for high-poly set, sub-object 0, draw 0"):
            I.import_mesh(app, read(doc))
        doc = P.to_gltf(app)
        doc["meshes"][0]["primitives"][1]["extras"]["d3sno"]["draw"] = 5
        with self.assertRaisesRegex(FormatError, "draw 5 of high-poly set, sub-object 0, which has 2 draws"):
            I.import_mesh(app, read(doc))
        text = P.to_obj(app).replace("g draw_0\n", "").replace("g draw_1", "g other")
        with self.assertRaisesRegex(FormatError, "no 'g draw_<k>' groups and 2 draws"):
            I.import_mesh(app, text)
        text = P.to_obj(app).replace("g draw_1", "g other")
        # stage 17 (W17-12): a group that matches no draw is new geometry (was refused, W16-10)
        self.assertEqual(I.import_mesh_report(app, text)[1].added, ["high-poly set, sub-object 0, draw 2"])
        with self.assertRaisesRegex(FormatError, "draw 7 of"):
            I.import_mesh(app, P.to_obj(app).replace("g draw_1", "g draw_7"))

    def test_missing_keep_and_empty(self):
        app = P.parse_appearance(ROUND_TRIP["ps3 two draws, low poly"][0], "ps3")
        doc = P.to_gltf(app)
        doc["nodes"] = [n for n in doc["nodes"] if n.get("name") != "lowpoly_Low"]
        doc["scenes"][0]["nodes"] = list(range(len(doc["nodes"])))
        doc["meshes"][0]["primitives"].pop(1)
        out, report = I.import_mesh_report(app, read(doc))
        self.assertEqual(out, bytes(app.source))
        self.assertEqual(report.kept, ["high-poly set, sub-object 0, draw 0", "high-poly set, sub-object 0, draw 1",
                                       "low-poly set, sub-object 0, draw 0"])
        out, report = I.import_mesh_report(app, read(doc), missing="empty")  # ps3 rigid draws: dropped
        self.assertEqual(report.changed, ["high-poly set, sub-object 0, draw 1", "low-poly set, sub-object 0, draw 0"])
        again = P.parse_appearance(out, "ps3")
        # stage 17 (W17-13, W17-14): the low-poly sub-object left without draws is removed
        self.assertEqual([len(s.draws) for s in again.sub_objects + again.extra_sub_objects], [1])
        self.assertEqual(again.sub_objects[0].draws[0].positions, app.sub_objects[0].draws[0].positions)
        empty = {"asset": {"version": "2.0"}}
        sapp = P.parse_appearance(switch_rigid())
        # stage 17 (W17-13, W17-14): the emptied rigid list and only skinning record are removed with
        # their sub-object (were refused: Q376 / V21-3)
        out, report = I.import_mesh_report(sapp, read(empty), missing="empty")
        self.assertEqual(P.parse_appearance(out).sub_objects, [])
        self.assertEqual(report.removed, ["high-poly set, sub-object 0, rigid list", "high-poly set, sub-object 0"])
        skinned = P.parse_appearance(derived(switch_skinned3()))
        out, report = I.import_mesh_report(skinned, read(empty), missing="empty")
        self.assertEqual(P.parse_appearance(out).sub_objects, [])
        self.assertEqual(report.removed, ["high-poly set, sub-object 0, skinning record 0", "high-poly set, sub-object 0"])
        with self.assertRaisesRegex(FormatError, "cloth"):
            I.import_mesh(P.parse_appearance(switch_cloth()), read(empty), missing="empty")
        with self.assertRaises(ValueError):
            I.import_mesh(sapp, read(empty), missing="drop")

    def test_low_poly_skipped(self):
        app = P.parse_appearance(ROUND_TRIP["ps3 two draws, low poly"][0], "ps3")
        out, report = I.import_mesh_report(app, read(P.to_gltf(moved(app))), low_poly=False)
        self.assertIn("low-poly sub-object and is not imported", report.warnings[0])
        again = P.parse_appearance(out, "ps3")
        self.assertEqual(again.extra_sub_objects[0].draws[0].positions, app.extra_sub_objects[0].draws[0].positions)
        self.assertEqual(report.kept, ["low-poly set, sub-object 0, draw 0"])

    def test_rigid_node_transform(self):
        app = P.parse_appearance(derived(switch_rigid()))
        doc = P.to_gltf(app)
        node = next(n for n in doc["nodes"] if "mesh" in n)
        node["translation"] = [1.0, 0.0, 0.0]
        node["rotation"] = [0.0, 0.0, 0.7071067811865476, 0.7071067811865476]  # 90 degrees about z
        out = P.parse_appearance(I.import_mesh(app, read(doc)))
        draw, before = out.sub_objects[0].draws[0], app.sub_objects[0].draws[0]
        step = out.sub_objects[0].scale
        for p, q in zip(draw.positions, before.positions):
            want = (1.0 - q[1], q[0], q[2])
            self.assertTrue(all(abs(p[k] - want[k]) <= step[k] for k in range(3)), (p, want))
        n = P._unit(before.normals[0])
        want = M.normal_bytes((-n[1], n[0], n[2]))
        self.assertEqual(M.normal_bytes(draw.normals[0]), want)
        # a parent with a matrix
        doc = P.to_gltf(app)
        index = next(i for i, n in enumerate(doc["nodes"]) if "mesh" in n)
        doc["nodes"].append({"matrix": [2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1], "children": [index]})
        out = P.parse_appearance(I.import_mesh(app, read(doc)))
        p, q = out.sub_objects[0].draws[0].positions[1], before.positions[1]
        self.assertTrue(all(abs(p[k] - 2 * q[k]) <= out.sub_objects[0].scale[k] for k in range(3)))

    def test_skinned_node_transform_ignored(self):
        app = P.parse_appearance(switch_skinned3())
        doc = P.to_gltf(app)
        next(n for n in doc["nodes"] if "mesh" in n)["translation"] = [5.0, 5.0, 5.0]
        self.assertEqual(I.import_mesh(app, read(doc)), bytes(app.source))


# --- W16-8 ---------------------------------------------------------------------------------------

class Values(unittest.TestCase):
    def template(self):
        return P.parse_appearance(derived(switch_rigid()))

    def remove(self, doc, name):
        for mesh in doc["meshes"]:
            for primitive in mesh["primitives"]:
                primitive["attributes"].pop(name, None)
        return doc

    def grown(self, app):
        """The app with one vertex added to its first draw (a new vertex count)."""
        out = copy.deepcopy(app)
        draw = out.sub_objects[0].draws[0]
        for name in ("positions", "normals", "colours_1", "colours_2", "uv1", "uv2_raw"):
            getattr(draw, name).append(getattr(draw, name)[-1])
        draw.positions[-1] = (2.0, 3.0, 3.0)
        draw.indices += [1, 3, 2]
        if draw.joints is not None:
            draw.joints.append(draw.joints[-1])
            draw.weights.append(draw.weights[-1])
        return out

    def test_same_index_keeps_missing_attributes(self):
        app = self.template()
        doc = P.to_gltf(app)
        for name in ("NORMAL", "TEXCOORD_0", "_D3_UV2", "_D3_COLOR1", "_D3_COLOR2"):
            self.remove(doc, name)
        self.assertEqual(I.import_mesh(app, read(doc)), bytes(app.source))

    def test_new_count(self):
        app = self.template()
        grown = self.grown(app)
        out = P.parse_appearance(I.import_mesh(app, read(P.to_gltf(grown))))
        draw = out.sub_objects[0].draws[0]
        self.assertEqual(len(draw.positions), 4)
        self.assertEqual(draw.uv2_raw[3], (9, 10))
        self.assertEqual(draw.colours_1[3], bytes((1, 2, 3, 4)))
        doc = self.remove(P.to_gltf(grown), "_D3_UV2")
        draw = P.parse_appearance(I.import_mesh(app, read(doc))).sub_objects[0].draws[0]
        self.assertEqual(draw.uv2_raw, [(0x8000, 0x8000)] * 4)
        for name, message in (("NORMAL", "normals missing"), ("TEXCOORD_0", "UV1"),
                              ("_D3_COLOR1", "_D3_COLOR1 missing"), ("_D3_COLOR2", "_D3_COLOR2 missing")):
            with self.assertRaisesRegex(FormatError, "high-poly set, sub-object 0, draw 0: " + message, msg=name):
                I.import_mesh(app, read(self.remove(P.to_gltf(grown), name)))

    def test_texcoord_1_inverted(self):
        app = self.template()
        edited = copy.deepcopy(app)
        edited.sub_objects[0].draws[0].uv2_raw[1] = (0x8000 + 8192, 0x8000 - 4096)
        doc = self.remove(P.to_gltf(edited, lightmap_v_scale=0.5), "_D3_UV2")
        draw = P.parse_appearance(I.import_mesh(app, read(doc), lightmap_v_scale=0.5)).sub_objects[0].draws[0]
        self.assertEqual(draw.uv2_raw, [(9, 10), (0x8000 + 8192, 0x8000 - 4096), (9, 10)])

    def test_changed_values(self):
        app = self.template()
        edited = copy.deepcopy(app)
        draw = edited.sub_objects[0].draws[0]
        draw.normals[0] = (0.0, 0.0, 1.0)
        draw.uv1[2] = (0.5, 0.25)
        draw.colours_2[1] = bytes((9, 9, 9, 9))
        out = P.parse_appearance(I.import_mesh(app, read(P.to_gltf(edited))))
        got = out.sub_objects[0].draws[0]
        self.assertEqual(M.normal_bytes(got.normals[0]), bytes((128, 128, 255)))  # W2-5, clamped
        self.assertEqual(got.normals[1:], app.sub_objects[0].draws[0].normals[1:])
        self.assertEqual(got.uv1[2], (0.5, 0.25))
        self.assertEqual(got.colours_2[1], bytes((9, 9, 9, 9)))
        self.assertEqual(got.positions, app.sub_objects[0].draws[0].positions)

    def test_rigid_and_skinned_mismatch(self):
        rigid = P.parse_appearance(switch_rigid())
        skinned = P.parse_appearance(switch_skinned3())
        doc = P.to_gltf(skinned)
        with self.assertRaisesRegex(FormatError, "with joints for a rigid draw"):
            I.import_mesh(rigid, read(doc))  # names subobject_0 in both
        doc = self.remove(self.remove(P.to_gltf(skinned), "JOINTS_0"), "WEIGHTS_0")
        with self.assertRaisesRegex(FormatError, "without joints for a skinned draw"):
            I.import_mesh(skinned, read(doc))

    def test_uv_out_of_range_names_the_draw(self):
        app = self.template()
        edited = copy.deepcopy(app)
        edited.sub_objects[0].draws[0].uv1[0] = (100.0, 0.0)
        with self.assertRaisesRegex(FormatError, r"^high-poly set, sub-object 0, draw 0: UV"):
            I.import_mesh(app, read(P.to_gltf(edited)))


# --- W16-9 ---------------------------------------------------------------------------------------

class Weights(unittest.TestCase):
    def test_weight_bytes(self):
        self.assertEqual(I.weight_bytes([(3, 0.5), (1, 0.5)]), ((3, 1, 0, 0), (128, 127, 0, 0)))
        self.assertEqual(I.weight_bytes([(1, 0.25), (2, 0.75)]), ((2, 1, 0, 0), (191, 64, 0, 0)))
        # equal bones added, zeros dropped, five pairs -> four largest
        # 5: 0.2, 7: 0.3, 8: 0.1, 9: 0.2 (4 dropped) -> 63.75, 95.625, 31.875, 63.75; 252 floored,
        # three units to 8 (.875), then 5 and 9 (.75, by position)
        self.assertEqual(I.weight_bytes([(5, 0.1), (6, 0.0), (5, 0.1), (7, 0.3), (8, 0.1), (9, 0.2), (4, 0.05)]),
                         ((7, 5, 9, 8), (95, 64, 64, 32)))
        joints, weights = I.weight_bytes([(1, 1), (2, 1), (3, 1)])
        self.assertEqual((joints, weights), ((1, 2, 3, 0), (85, 85, 85, 0)))
        joints, weights = I.weight_bytes([(1, 1.0), (2, 1.0), (3, 1.0), (4, 1.0)])  # 63.75 each: units to 1, 2, 3
        self.assertEqual((joints, weights), ((1, 2, 3, 4), (64, 64, 64, 63)))
        self.assertEqual(I.weight_bytes([(9, 1e-9), (2, 1.0)]), ((2, 0, 0, 0), (255, 0, 0, 0)))
        with self.assertRaisesRegex(FormatError, "no weight left"):
            I.weight_bytes([(1, 0.0), (2, 0.0)])
        with self.assertRaisesRegex(FormatError, "finite"):
            I.weight_bytes([(1, -0.5)])

    def test_keep_four_largest_ties_by_first(self):
        joints, weights = I.weight_bytes([(1, 1), (2, 1), (3, 1), (4, 1), (5, 1)])
        self.assertEqual(joints, (1, 2, 3, 4))
        self.assertEqual(sum(weights), 255)

    def template(self, order="<"):
        return P.parse_appearance(grid_template(switch_skinned3(order), "x360" if order == ">" else None))

    def test_change_one_weight(self):
        for app in (self.template(), self.template(">"),
                    P.parse_appearance(grid_template(ps3(skinned=True, bones=BONES), "ps3"), "ps3")):
            edited = copy.deepcopy(app)
            draw = edited.sub_objects[0].draws[0]
            draw.joints[4], draw.weights[4] = (1, 2, 0, 0), (155, 100, 0, 0)
            out, report = I.import_mesh_report(app, read(P.to_gltf(edited)))
            self.assertEqual(report.changed, ["high-poly set, sub-object 0, draw 0"])
            again = P.parse_appearance(out, app.build)
            self.assertEqual(again.warnings, [])
            got, before = again.sub_objects[0].draws[0], app.sub_objects[0].draws[0]
            self.assertEqual(got.weights[4], (155, 100, 0, 0))
            self.assertEqual(got.joints[4], (1, 2, 0, 0))
            for name in ("positions", "normals", "uv1", "uv2_raw", "colours_1", "colours_2", "indices"):
                self.assertEqual(getattr(got, name), getattr(before, name), name)
            self.assertEqual(got.weights[:4] + got.weights[5:], before.weights[:4] + before.weights[5:])
            self.assertEqual(got.joints[:4] + got.joints[5:], before.joints[:4] + before.joints[5:])
            # through OBJ the skin is kept at the same index
            self.assertEqual(I.import_mesh(app, P.to_obj(app)), bytes(app.source))

    def test_joint_mapping(self):
        app = P.parse_appearance(switch_skinned3())
        doc = P.to_gltf(app)
        for node in doc["nodes"][:3]:
            node["name"] = "renamed"
        self.assertEqual(I.import_mesh(app, read(doc)), bytes(app.source))  # by node index (extras bones)
        doc["skins"][0]["extras"]["d3sno"]["bones"] = 4
        with self.assertRaisesRegex(FormatError, r"vertex 0: skin 0 joint 2 \(node 2 'renamed'\) names no skeleton bone"):
            I.import_mesh(app, read(doc))
        doc = P.to_gltf(app)
        doc["skins"][0].pop("extras")
        doc["nodes"][2]["name"] = "bone_2_000000CD"  # hash checked: 0xCC is stored
        with self.assertRaisesRegex(FormatError, "joint 2 .* names no skeleton bone"):
            I.import_mesh(app, read(doc))
        doc["nodes"][2]["name"] = "bone_2_000000CC"
        self.assertEqual(I.import_mesh(app, read(doc)), bytes(app.source))  # by name, no extras

    def test_joints_reordered_by_name(self):
        """Joint order differs from the skeleton: the names map them back."""
        app = P.parse_appearance(switch_skinned3())
        doc = P.to_gltf(app)
        doc["skins"][0]["joints"] = [2, 1, 0]
        doc["skins"][0].pop("extras")
        attributes = doc["meshes"][0]["primitives"][0]["attributes"]
        joints = accessor_rows(doc, attributes["JOINTS_0"], "4H")
        blob = blob_of(doc)
        view = doc["bufferViews"][doc["accessors"][attributes["JOINTS_0"]]["bufferView"]]
        remap = {0: 2, 1: 1, 2: 0}
        struct.pack_into(f"<{4 * len(joints)}H", blob, view["byteOffset"], *[remap[j] for row in joints for j in row])
        set_blob(doc, blob)
        self.assertEqual(I.import_mesh(app, read(doc)), bytes(app.source))

    def test_joint_refusals(self):
        app = P.parse_appearance(switch_skinned3())
        doc = P.to_gltf(app)
        next(n for n in doc["nodes"] if "mesh" in n).pop("skin")
        with self.assertRaisesRegex(FormatError, "joints without a skin on the node"):
            I.import_mesh(app, read(doc))
        doc = P.to_gltf(app)
        attributes = doc["meshes"][0]["primitives"][0]["attributes"]
        attributes["JOINTS_0"] = add_accessor(doc, struct.pack("<12H", *([7, 0, 0, 0] * 3)), 5123, "VEC4", 3)
        with self.assertRaisesRegex(FormatError, "joint 7 past the skin's 3 joints"):
            I.import_mesh(app, read(doc))
        attributes["JOINTS_0"] = add_accessor(doc, struct.pack("<12f", *([0.0] * 12)), 5126, "VEC4", 3)
        with self.assertRaisesRegex(FormatError, "JOINTS_0 is not unsigned byte or short"):
            I.import_mesh(app, read(doc))
        doc = P.to_gltf(app)
        doc["meshes"][0]["primitives"][0]["attributes"]["NORMAL"] = doc["meshes"][0]["primitives"][0]["attributes"]["TEXCOORD_0"]
        with self.assertRaisesRegex(FormatError, "NORMAL has type VEC2"):
            I.import_mesh(app, read(doc))

    def test_bone_transform_warning(self):
        app = P.parse_appearance(switch_skinned3())
        doc = P.to_gltf(app)
        doc["nodes"][1]["translation"] = [9.0, 0.0, 0.0]
        out, report = I.import_mesh_report(app, read(doc))
        self.assertEqual(out, bytes(app.source))
        self.assertEqual(len(report.warnings), 1)
        self.assertIn("bone 1: the glTF node transform differs", report.warnings[0])

    def test_two_sets_and_normalised_weights(self):
        app = P.parse_appearance(switch_skinned3())
        doc = P.to_gltf(app)
        attributes = doc["meshes"][0]["primitives"][0]["attributes"]
        # weights as normalised bytes, split over two sets: (2: 200/255) and (1: 55/255)
        attributes["JOINTS_0"] = add_accessor(doc, bytes((2, 0, 0, 0)) * 3, 5121, "VEC4", 3)
        attributes["WEIGHTS_0"] = add_accessor(doc, bytes((200, 0, 0, 0)) * 3, 5121, "VEC4", 3, normalized=True)
        attributes["JOINTS_1"] = add_accessor(doc, bytes((1, 0, 0, 0)) * 3, 5121, "VEC4", 3)
        attributes["WEIGHTS_1"] = add_accessor(doc, bytes((55, 0, 0, 0)) * 3, 5121, "VEC4", 3, normalized=True)
        self.assertEqual(I.import_mesh(app, read(doc)), bytes(app.source))
        del attributes["WEIGHTS_1"]
        with self.assertRaisesRegex(FormatError, "do not come in a pair"):
            I.import_mesh(app, read(doc))

    def test_vertex_without_weight_refused(self):
        app = P.parse_appearance(switch_skinned3())
        doc = P.to_gltf(app)
        attributes = doc["meshes"][0]["primitives"][0]["attributes"]
        attributes["WEIGHTS_0"] = add_accessor(doc, struct.pack("<12f", *([0.0] * 12)), 5126, "VEC4", 3)
        with self.assertRaisesRegex(FormatError, "draw 0: vertex 0: a vertex of a skinned draw has no weight left"):
            I.import_mesh(app, read(doc))

    def test_export_form(self):
        draw = P.parse_appearance(switch_skinned3()).sub_objects[0].draws[0]
        form = I.export_weights(draw, 0)
        self.assertEqual(form[:2], [(2, struct.unpack("<f", struct.pack("<f", 200 / 255))[0]),
                                    (1, struct.unpack("<f", struct.pack("<f", 55 / 255))[0])])

    def test_export_form_keeps_stored_weights(self):
        """Stored weights that the byte rounding would not give back (sum 200, ascending) are kept
        when the imported values equal their export form."""
        for weights in ((100, 100, 0, 0), (55, 200, 0, 0)):
            data = laid_out(B.build_app_switch(bones=BONES, pool_spare=0, sub_objects=[
                {"skin": dict(SKIN2, weights=weights), "index_count": 3}]))
            app = P.parse_appearance(data)
            self.assertEqual(I.import_mesh(app, read(P.to_gltf(app))), data, weights)
            self.assertNotEqual(I.weight_bytes(zip((2, 1), (w / sum(weights) for w in weights[:2])))[1], weights)

    def test_zero_weight_slots_kept(self):
        """Q379: a stored non-zero slot of weight 0 survives an import that keeps the vertex."""
        data = laid_out(B.build_app_switch(bones=BONES, pool_spare=0, sub_objects=[
            {"skin": dict(SKIN2, slots=(0, 1, 1, 0), weights=(255, 0, 0, 0)), "index_count": 3}]))
        app = P.parse_appearance(data)
        self.assertEqual(I.import_mesh(app, read(P.to_gltf(app))), data)
        edited = copy.deepcopy(app)
        edited.sub_objects[0].draws[0].positions[0] = (1.5, 2.0, 3.0)
        out = I.import_mesh(app, read(P.to_gltf(edited)))
        self.assertEqual(P.parse_appearance(out).sub_objects[0].draws[0].weights, app.sub_objects[0].draws[0].weights)


# --- edits (W16-11, Check) ---------------------------------------------------------------------------

EDIT_TEMPLATES = [
    (switch_rigid(), None), (switch_rigid(">"), "x360"), (ps3(), "ps3"),
    (switch_skinned3(), None), (switch_skinned3(">"), "x360"), (ps3(skinned=True, bones=BONES), "ps3"),
]


class Edits(unittest.TestCase):
    def test_move_every_position(self):
        for data, build in EDIT_TEMPLATES:
            app = P.parse_appearance(grid_template(data, build), build)
            for form in ("gltf", "obj"):
                edited = moved(app)
                doc = read(P.to_gltf(edited)) if form == "gltf" else P.to_obj(edited)
                out, report = I.import_mesh_report(app, doc)
                self.assertEqual(report.changed, ["high-poly set, sub-object 0, draw 0"])
                again = P.parse_appearance(out, build)
                self.assertEqual(again.warnings, [], (build, form))
                sub = again.sub_objects[0]
                for p, q in zip(sub.draws[0].positions, edited.sub_objects[0].draws[0].positions):
                    self.assertTrue(all(abs(p[k] - q[k]) <= abs(sub.scale[k]) for k in range(3)), (p, q))
                before = app.sub_objects[0].draws[0]
                for name in ("normals", "uv1", "uv2_raw", "colours_1", "colours_2", "indices", "joints", "weights"):
                    self.assertEqual(getattr(sub.draws[0], name), getattr(before, name), (build, form, name))

    def test_remove_last_triangle(self):
        for data, build in EDIT_TEMPLATES:
            app = P.parse_appearance(grid_template(data, build), build)
            for form in ("gltf", "obj"):
                edited = copy.deepcopy(app)
                for sub in edited.sub_objects:
                    for draw in sub.draws:
                        del draw.indices[-3:]
                doc = read(P.to_gltf(edited)) if form == "gltf" else P.to_obj(edited)
                again = P.parse_appearance(I.import_mesh(app, doc), build)
                self.assertEqual(again.warnings, [], (build, form))
                got = again.sub_objects[0].draws[0]
                self.assertEqual(len(got.indices), len(app.sub_objects[0].draws[0].indices) - 3)
                self.assertEqual(got.indices, edited.sub_objects[0].draws[0].indices)
                self.assertEqual(got.positions, app.sub_objects[0].draws[0].positions)

    def test_obj_editor_form(self):
        """An OBJ written by an editor: shared v lines, other numbering, quads."""
        app = P.parse_appearance(grid_template(switch_rigid(), cells=1))
        draw = app.sub_objects[0].draws[0]
        lines = ["mtllib x.mtl", "o subobject_0", "s off"]
        lines += [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in draw.positions]
        lines += [f"vt {u:.6f} {v:.6f}" for u, v in draw.uv1]
        lines += ["vn 0 0 1", "usemtl m", "f 1/1/1 2/2/1 4/4/1 3/3/1"]  # one quad -> fan
        text = "\n".join(lines)
        # numbered by first use (1, 2, 4, 3): not the same index, and OBJ has no colours (W16-8)
        with self.assertRaisesRegex(FormatError, "_D3_COLOR1 missing for a new vertex list"):
            I.import_mesh(app, text)
        draw.colours_1, draw.colours_2 = [], []  # a draw without colours (in memory)
        out, report = I.import_mesh_report(app, text)
        got = P.parse_appearance(out).sub_objects[0].draws[0]
        self.assertEqual(got.indices, [0, 1, 2, 0, 2, 3])
        self.assertEqual(len(got.positions), 4)
        self.assertEqual(got.normals, [(0.0, 0.0, 127 / 128)] * 4)  # bytes (128, 128, 255)
        self.assertEqual(sorted(got.positions), sorted(draw.positions))

    def test_obj_first_use_same_positions_keeps(self):
        """Numbered by first use, same count and positions: same index keeping applies."""
        app = P.parse_appearance(switch_rigid())
        text = P.to_obj(app)
        vt = [l for l in text.splitlines() if l.startswith("vt ")]
        text = text.replace("f 1/1/1 2/2/2 3/3/3", "\n".join(vt) + "\nf 1/4/1 2/5/2 3/6/3")
        self.assertEqual(I.import_mesh(app, text), bytes(app.source))
        relative = text.replace("f 1/4/1 2/5/2 3/6/3", "f -3/-3 -2/-2 -1/-1")  # vn missing: kept, same index
        self.assertEqual(I.import_mesh(app, relative), bytes(app.source))
        rotated = text.replace("f 1/4/1 2/5/2 3/6/3", "f 2/5/2 3/6/3 1/4/1")  # first use 2, 3, 1: positions differ
        with self.assertRaisesRegex(FormatError, "_D3_COLOR1 missing for a new vertex list"):
            I.import_mesh(app, rotated)

    def skin_grid(self):
        """A one-cell skinned grid whose four vertices have different skins; its draw holds no
        colours in memory, so that OBJ (which has none) can add vertices."""
        app = P.parse_appearance(grid_template(switch_skinned3(), cells=1))
        draw = app.sub_objects[0].draws[0]
        draw.joints[:] = [(2, 1, 0, 0), (1, 0, 0, 0), (0, 0, 0, 0), (1, 2, 0, 0)]
        draw.weights[:] = [(200, 55, 0, 0), (255, 0, 0, 0), (255, 0, 0, 0), (155, 100, 0, 0)]
        app = P.parse_appearance(M.write_app_mesh(app, derive=True))
        draw = app.sub_objects[0].draws[0]
        draw.colours_1, draw.colours_2 = [], []
        return app

    def grown_text(self, app, position):
        grown = copy.deepcopy(app)
        g = grown.sub_objects[0].draws[0]
        g.positions.append(position)
        g.normals.append(g.normals[0])
        g.uv1.append(g.uv1[0])
        g.indices += [1, 4, 3]
        return P.to_obj(grown)

    def test_obj_skin_refuse(self):
        app = self.skin_grid()
        with self.assertRaisesRegex(FormatError, "draw 0: OBJ has no skin and the vertex list is new"):
            I.import_mesh(app, self.grown_text(app, (2.1, 3.1, 3.0)))
        with self.assertRaises(ValueError):
            I.import_mesh(app, self.grown_text(app, (2.1, 3.1, 3.0)), skin="closest")

    def test_obj_nearest(self):
        app = self.skin_grid()
        draw = app.sub_objects[0].draws[0]
        self.assertEqual(draw.positions, [(1.0, 2.0, 3.0), (2.0, 2.0, 3.0), (1.0, 3.0, 3.0), (2.0, 3.0, 3.0)])
        for position, nearest in (((2.1, 3.1, 3.0), 3), ((1.5, 2.0, 3.0), 0), ((2.0, 2.5, 3.0), 1)):
            out, report = I.import_mesh_report(app, self.grown_text(app, position), skin="nearest")
            got = P.parse_appearance(out).sub_objects[0].draws[0]
            self.assertEqual(got.joints[4], draw.joints[nearest], position)  # ties: lower index
            self.assertEqual(got.weights[4], draw.weights[nearest], position)
            self.assertEqual(got.joints[:4], draw.joints)
            self.assertIn("new vertices get colour 0", report.warnings[0])


# --- W16-10 --------------------------------------------------------------------------------------

class ReadObj(unittest.TestCase):
    def test_statements(self):
        doc = I.read_obj("# c\nmtllib a.mtl\no A\ng draw_2\nv 1 2 3\nv 4 5 6 1\nv 7 8 9\nvt 0.5\nvn 0 0 1\n"
                         "usemtl x\ns 1\nf 1 2 -1\nf 1/1 2/1 3/1\nf 1//1 2//1 3//1\nf 1/1/1 2/1/1 3/1/1 1/1/1\n")
        self.assertEqual(doc.positions[1], (4.0, 5.0, 6.0))
        self.assertEqual(doc.texcoords, [(0.5, 0.0)])
        group = doc.objects[0].groups[0]
        self.assertEqual((doc.objects[0].name, group.name), ("A", "draw_2"))
        self.assertEqual(group.vertex_lines, [0, 1, 2])
        self.assertEqual(group.faces[0], ((0, None, None), (1, None, None), (2, None, None)))
        self.assertEqual(group.faces[1][1], (1, 0, None))
        self.assertEqual(group.faces[2][2], (2, None, 0))
        self.assertEqual(len(group.faces), 5)  # the quad as two triangles
        self.assertEqual(group.faces[4], ((0, 0, 0), (2, 0, 0), (0, 0, 0)))

    def test_refused(self):
        for text, message in (("v 1 2\n", "needs 3"), ("v 1 2 x\n", "not a number"), ("v 1 2 nan\n", "not finite"),
                              ("v 0 0 0\nf 1 1\n", "three corners"), ("v 0 0 0\nf 0 1 1\n", "index 0"),
                              ("v 0 0 0\nf 1 1 4\n", "v index 4 past 1"), ("v 0 0 0\nf -2 1 1\n", "before the first"),
                              ("v 0 0 0\nf 1/a 1 1\n", "corner"), ("l 1 2\n", "statement 'l'"),
                              ("v 0 0 0\nf 1/2 1 1\n", "vt index 2 past 0")):
            with self.assertRaisesRegex(FormatError, message, msg=text):
                I.read_obj(text)

    def test_object_needs_a_match(self):
        app = P.parse_appearance(switch_rigid())
        with self.assertRaisesRegex(FormatError, "matches no sub-object"):
            I.import_mesh(app, "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")

    def test_missing_texcoords_for_new_vertices(self):
        app = P.parse_appearance(switch_rigid())
        text = "o subobject_0\nv 1 2 3\nv 6 2 3\nv 1 7 3\nv 6 7 3\nvn 0 0 1\nf 1//1 2//1 3//1\nf 2//1 4//1 3//1\n"
        with self.assertRaisesRegex(FormatError, "UV1 .* missing for a new vertex list"):
            I.import_mesh(app, text)


# --- W16-11 --------------------------------------------------------------------------------------

class Writing(unittest.TestCase):
    def test_unchanged_returns_template_bytes(self):
        data = ps3(skinned=True, bones=BONES)  # does not follow W13: a derive would change it
        app = P.parse_appearance(data, "ps3")
        self.assertNotEqual(derived(data, "ps3"), data)
        self.assertEqual(I.import_mesh(app, read(P.to_gltf(app))), data)

    def test_cloth_count_change_refused(self):
        app = P.parse_appearance(switch_cloth())
        grown = copy.deepcopy(app)
        d = grown.sub_objects[0].draws[0]
        for name in ("positions", "normals", "colours_1", "colours_2", "uv1", "uv2_raw"):
            getattr(d, name).append(getattr(d, name)[0])
        d.indices += [0, 1, 3]
        with self.assertRaisesRegex(FormatError, "changes the vertex count .* cloth data"):
            I.import_mesh(app, read(P.to_gltf(grown)))
        edited = copy.deepcopy(app)  # same count: refused too since V21-1 (replaces Q716)
        edited.sub_objects[0].draws[0].positions[1] = (12.0, 0.0, 0.0)
        with self.assertRaisesRegex(FormatError, "V21-1"):
            I.import_mesh(app, read(P.to_gltf(edited)))

    def test_too_many_vertices(self):
        app = P.parse_appearance(switch_rigid())
        n = 65537
        lines = ["o subobject_0", "g draw_0"] + ["v 0 0 0"] * n + ["vt 0 0"] * n + ["vn 0 0 1"] * n
        lines += [f"f {i}/{i}/{i} {i + 1}/{i + 1}/{i + 1} {i + 2}/{i + 2}/{i + 2}" for i in range(1, n - 1, 3)]
        with self.assertRaisesRegex(FormatError, "65537 vertices, indices are u16"):
            I.import_mesh(app, "\n".join(lines), missing="keep")

    def test_writer_refusal_is_labelled(self):
        app = P.parse_appearance(ps3(skinned=True, bones=BONES), "ps3")
        edited = copy.deepcopy(app)
        positions = [(float(i % 30), float(i // 30), 0.0) for i in range(1500)]
        set_mesh(edited.sub_objects[0].draws[0], positions, list(range(1500)) * 3)
        # ``39`` V24-4: the budget is checked first (V23-2 order)
        with self.assertRaisesRegex(FormatError, r"^high-poly set, sub-object 0, draw 0: budget value"):
            I.import_mesh(app, read(P.to_gltf(edited)))

    def test_report_and_bad_arguments(self):
        app = P.parse_appearance(ROUND_TRIP["switch two sub-objects"][0])
        edited = copy.deepcopy(app)
        edited.sub_objects[1].draws[0].normals[0] = (0.0, 1.0, 0.0)
        out, report = I.import_mesh_report(app, read(P.to_gltf(edited)))
        self.assertEqual(report.changed, ["high-poly set, sub-object 1, draw 0"])
        self.assertEqual(report.kept, ["high-poly set, sub-object 0, draw 0"])
        self.assertEqual(I.import_mesh(app, read(P.to_gltf(edited))), out)
        with self.assertRaises(ValueError):
            I.import_mesh(app, read(P.to_gltf(app)), lightmap_v_scale=float("nan"))
        with self.assertRaises(ValueError):
            I.import_mesh(app, 42)


if __name__ == "__main__":
    unittest.main()
