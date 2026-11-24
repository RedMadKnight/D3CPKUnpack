"""Stage 20 (``44_stage20_uv2_rotation.md``): W20-1 to W20-5. Synthetic data only; every export goes to a
temporary folder outside any git working tree. Shader names of the library tests are our own, in a test
table; the export tests use names from the shipped tables and are skipped when a table is missing."""

import _libpath  # noqa: F401  (python -I: the library folder on sys.path)
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import builders as B
import d3sno
import export_fixture as F
import test_stage17_writer as W
from d3sno import app_mesh as M
from d3sno import appearance as P
from d3sno import export as E
from d3sno import index_write as IW
from d3sno import mesh_import as I
from d3sno import uv2_use as U
from d3sno import world as WD
from d3sno.index import TocEntry
from d3sno.sno import group_by_extension
from d3sno.transform import Transform, quat_multiply, quat_normalise, rotate
from test_stage15_content import shader_map
from test_stage18_export import glb_json
from test_stage19 import LIGHT, accessor, export, lightmap_texture, primitives

TABLE_PATHS = tuple(U.DATA_DIR / name for name in (U.ROW_TABLE, U.KNOWN_TABLE, U.KNOWN_TABLE_X360))
MISSING_TABLES = tuple(str(p) for p in TABLE_PATHS if not p.is_file())


def need_uv2_tables():
    """V26-8: a test that needs a shipped table is skipped with a message that names the file
    (``45`` X8: ``shader_known_x360.txt`` included)."""
    if MISSING_TABLES:
        raise unittest.SkipTest(f"UV2 use needs {', '.join(MISSING_TABLES)}, which is not there (V26-8)")


S = 1.25
LAYER_TEX = 530  # a slot-19 texture that is not in the export
ROOT_HALF = math.sqrt(0.5)
# UV2 per sub-object (all varying); sub-object 1 holds the W20 layer UV examples
RAW_UV2 = [[(0x8000, 0x8000), (0x8000 + 4096, 0x8000 + 2048), (0x8000 - 1024, 0x8000 + 8192)],
           [(0x8000, 0x8000), (0xA000, 0x9000), (0x7000, 0x8800)],
           [(0x9000, 0x7000), (0x8100, 0x8200), (0xA000, 0x8000)],
           [(0x8001, 0x8000), (0x8000, 0x8001), (0x8002, 0x8002)],
           [(0x1000, 0x2000), (0x3000, 0x4000), (0x5000, 0x6000)],
           [(0x8800, 0x8800), (0x8000, 0x9000), (0x9000, 0x8000)]]
HASHES = (0xA00, 0xA01, 0xA02, 0xA03, 0xA04, 0xA05)


def six_mesh(textures, shader_maps, uv2=RAW_UV2) -> bytes:
    """Six rigid sub-objects, one material each (``textures[i]``: extra (slot, handle) pairs beside
    the diffuse texture; ``shader_maps[i]``: the block's shader map handle) and the UV2 of RAW_UV2."""
    data = W.laid_out(B.build_app_switch(version=306, pool_spare=0, sub_objects=[
        {"rigid": [(i, 0, 0), (i + 10, 0, 0), (i, 10, 0)], "index_count": 3} for i in range(6)]), None)
    data = W.canon(data, None)
    entries = [(h, [[(1, F.TEX), *extra]]) for h, extra in zip(HASHES, textures)]
    data = B.add_app_materials(data, "switch", [0x61], entries, dict(zip(HASHES, shader_maps)))
    for i, value in enumerate(HASHES):
        data = B.set_sub_object_material(data, "switch", i, value, 0x70 + i)
    data = W.canon(W.laid_out(data, None), None)
    app = P.parse_appearance(data)
    for sub, rows in zip(P.export_sub_objects(app, True), uv2):
        sub.draws[0].uv2_raw = list(rows)
    data = W.canon(M.write_app_mesh(app, derive=True), None)
    check = P.parse_appearance(data)
    assert [s.draws[0].uv2_raw for s in P.export_sub_objects(check, True)] == [list(r) for r in uv2]
    return F.with_handle(data, F.PLAIN_APP, "<")


# --- the library, with a test table of our own ---------------------------------------------------

TEST_ROWS = [("t_layer", 0, 1, 19, 8192, 0), ("t_layer", 1, 0, 12, 8192, 0), ("t_odd", 0, 2, 12, 512, 0),
             ("t_light", 0, 1, 2, 8192, 1), ("a_first", 2, 0, 13, 8192, 0)]
TEST_KNOWN = ["t_layer", "T_Odd", "t_light", "t_plain", "a_first"]
# shader map handle -> the shader names of its uses
TEST_MAPS = {1: ("t_layer",), 2: ("T_LAYER", "t_plain"), 3: ("t_odd",), 4: ("t_light",), 5: ("t_plain",),
             6: ("t_plain", "mystery")}
LIBRARY_TEXTURES = [[(2, LIGHT)], [(19, LAYER_TEX)], [], [], [], []]


def make_tables(case, rows=TEST_ROWS, known=TEST_KNOWN, skip=(), known_x360=None):
    """A table folder of our own in the layout of ``45`` X6: a row of six values is a ``ps3`` row,
    a row of seven starts with its build. ``known_x360`` (default: ``known``) is the x360 list."""
    holder = tempfile.TemporaryDirectory(prefix="d3sno_uv2_")
    case.addCleanup(holder.cleanup)
    folder = Path(holder.name)
    if U.ROW_TABLE not in skip:
        lines = ["build\tshader\tpass\tstage\tslot_type\tdivisor\tlightmap_v_scale"]
        lines += ["\t".join("" if v is None else str(v) for v in (row if len(row) == 7 else ("ps3", *row)))
                  for row in rows]
        (folder / U.ROW_TABLE).write_text("\n".join(lines) + "\n", encoding="utf-8")
    if U.KNOWN_TABLE not in skip:
        (folder / U.KNOWN_TABLE).write_text("\n".join(known) + "\n", encoding="utf-8")
    if U.KNOWN_TABLE_X360 not in skip:
        listed = known if known_x360 is None else known_x360
        (folder / U.KNOWN_TABLE_X360).write_text("\n".join(listed) + "\n", encoding="utf-8")
    return folder


def library_document(case, table, lightmap=True):
    source = six_mesh(LIBRARY_TEXTURES, [1, 2, 3, 4, 5, 6])
    doc = P.to_gltf(P.parse_appearance(source), texture_resolver=lambda h: f"../textures/{h}.png",
                    lightmap_resolver=lambda h: (f"../textures/{h}.png", S) if lightmap and h == LIGHT else None,
                    uv2_rows=lambda block: U.rows_for(TEST_MAPS.get(block.shader_map) if block else None, table))
    return source, doc


class Version(unittest.TestCase):
    def test_version(self):
        self.assertGreaterEqual(tuple(int(x) for x in d3sno.__version__.split(".")), (0, 20, 1))
        self.assertGreaterEqual(d3sno.STAGE, 20)


class Tables(unittest.TestCase):
    """The shipped tables load as package data and are consistent."""

    def setUp(self):
        need_uv2_tables()

    def test_rows(self):
        table = U.load_tables()
        self.assertIsNotNone(table)
        self.assertEqual(set(table.rows), set(U.TABLE_BUILDS))
        self.assertEqual(set(table.known), set(U.TABLE_BUILDS))
        rows = [r for names in table.rows.values() for rows in names.values() for r in rows]
        self.assertTrue(rows)
        for row in rows:
            self.assertIn(row.build, U.TABLE_BUILDS)
            if row.slot_type is None:  # a few x360 rows leave it empty (Q894)
                self.assertEqual(row.build, "x360", row)
            else:
                self.assertIsInstance(row.slot_type, int)
            self.assertIn(row.divisor, (8192, 512))
            self.assertIn(row.lightmap_v_scale, (0, 1))
            if row.slot_type == 2:  # X6: 1 for every slot-2 row (and for a few others)
                self.assertEqual(row.lightmap_v_scale, 1, row)
            self.assertIn(row.shader.lower(), table.known[row.build], row)  # every table shader is known
        self.assertIs(U.tables(), U.tables())

    def test_missing_table_gives_none(self):
        self.assertIsNone(U.load_tables(make_tables(self, skip=(U.ROW_TABLE,))))
        self.assertIsNone(U.load_tables(make_tables(self, skip=(U.KNOWN_TABLE, U.KNOWN_TABLE_X360))))
        for missing, build in ((U.KNOWN_TABLE, "switch"), (U.KNOWN_TABLE_X360, "x360")):  # X6: per build
            self.assertIsNone(U.load_tables(make_tables(self, skip=(missing,))).for_build(build))
        self.assertIsNotNone(U.load_tables(make_tables(self)))


class Uv2UseLibrary(unittest.TestCase):
    """W20-1 to W20-3 on ``to_gltf`` with a test table: the six cases of W20-2."""

    def setUp(self):
        self.table = U.load_tables(make_tables(self))
        self.source, self.doc = library_document(self, self.table)
        self.prims = primitives(self.doc)
        self.glb = I.gltf_to_glb(self.doc)

    def test_six_cases(self):
        uses = [p["extras"]["d3sno_export"]["uv2_use"] for p in self.prims]
        self.assertEqual(uses, ["lightmap", "layer", "open", "lightmap_without_texture", "unused", "unknown"])
        scales = [p["extras"]["d3sno_export"]["uv2_scale"] for p in self.prims]
        self.assertEqual(scales, ["lightmap", "layer", "open", "open", "unused", "open"])
        self.assertEqual([("TEXCOORD_1" in p["attributes"]) for p in self.prims], [True, True] + [False] * 4)
        for p in self.prims:
            self.assertIn("_D3_UV2", p["attributes"])

    def test_no_shaders_is_unknown(self):
        self.assertEqual(U.uv2_use(U.rows_for(None, self.table), False), "unknown")
        self.assertEqual(U.uv2_use(U.rows_for((), self.table), False), "unknown")
        self.assertEqual(U.uv2_use(U.rows_for(("t_plain",), None), False), "unknown")  # no table
        self.assertEqual(U.uv2_use(U.rows_for((None,), self.table), False), "unknown")  # unnamed shader
        self.assertEqual(U.uv2_use(U.rows_for(("t_plain",), self.table), False), "unused")
        self.assertEqual(U.uv2_use(U.rows_for(("Shaders\\T_Plain.shd",), self.table), False), "unused")

    def test_case_order(self):
        """Layer rows win over odd rows; odd rows over lightmap rows; rows over an unknown shader."""
        rows = U.rows_for(("t_layer", "t_odd", "t_light"), self.table)
        self.assertEqual(U.uv2_use(rows, False), "layer")
        self.assertEqual(U.uv2_use(rows, True), "lightmap")
        self.assertEqual(U.uv2_use(U.rows_for(("t_odd", "t_light"), self.table), False), "open")
        self.assertEqual(U.uv2_use(U.rows_for(("t_light", "mystery"), self.table), False),
                         "lightmap_without_texture")

    def test_layer_uv(self):
        """Raw (0x8000, 0x8000) gives (0, 0), (0xA000, 0x9000) gives (1, 0.5); no V scale."""
        rows = accessor(self.glb, self.prims[1]["attributes"]["TEXCOORD_1"])
        self.assertEqual(rows[0], (0.0, 0.0))
        self.assertEqual(rows[1], (1.0, 0.5))
        self.assertEqual(rows[2], (-0.5, 0.25))
        self.assertNotIn("lightmap_v_scale", self.prims[1]["extras"]["d3sno_export"])

    def test_lightmap_plus_layer(self):
        first = self.prims[0]
        rows = accessor(self.glb, first["attributes"]["TEXCOORD_1"])
        for got, (u, v) in zip(rows, RAW_UV2[0]):
            self.assertAlmostEqual(got[0], (u - 0x8000) / 8192, places=6)
            self.assertAlmostEqual(got[1], (v - 0x8000) / 8192 * S, places=6)
        extras = self.doc["materials"][first["material"]]["extras"]["d3sno_export"]
        self.assertEqual(extras["lightmap"], f"../textures/{LIGHT}.png")
        self.assertEqual([(e["pass"], e["stage"], e["v_divide"], e["texcoord"]) for e in extras["uv2_layers"]],
                         [(0, 1, S, 1), (1, 0, S, 1)])

    def test_layer_entries(self):
        extras = self.doc["materials"][self.prims[1]["material"]]["extras"]["d3sno_export"]
        self.assertEqual(extras["uv2_layers"], [
            {"shader": "t_layer", "pass": 0, "stage": 1, "slot_type": 19, "slot_name": "alpha", "divisor": 8192,
             "texture": f"../textures/{LAYER_TEX}.png", "texcoord": 1},
            {"shader": "t_layer", "pass": 1, "stage": 0, "slot_type": 12, "slot_name": "layer_12", "divisor": 8192,
             "texture": None, "texcoord": 1}])
        self.assertNotIn("lightmap", extras)
        odd = self.doc["materials"][self.prims[2]["material"]]["extras"]["d3sno_export"]["uv2_layers"]
        self.assertEqual(odd, [{"shader": "t_odd", "pass": 0, "stage": 2, "slot_type": 12, "slot_name": "layer_12",
                                "divisor": 512, "texture": None, "texcoord": None}])
        for p in self.prims[3:]:  # no layer or odd row: no ``uv2_layers`` (V27-1), no extras at all
            self.assertNotIn("extras", self.doc["materials"][p["material"]])
        for material in self.doc["materials"]:  # layer textures are not bound to glTF core inputs
            self.assertEqual(set(material) - {"name", "pbrMetallicRoughness", "extras"}, set())
            core = material.get("pbrMetallicRoughness", {})
            self.assertEqual(set(core) - {"baseColorTexture", "metallicFactor"}, set())

    def test_sorted_without_duplicates(self):
        rows = U.rows_for(("t_odd", "t_layer", "a_first", "T_LAYER"), self.table)
        self.assertEqual([(r.shader.lower(), r.pass_index, r.stage) for r in U.layer_rows(rows)],
                         [("a_first", 2, 0), ("t_layer", 0, 1), ("t_layer", 1, 0), ("t_odd", 0, 2)])

    def test_missing_lightmap_texture(self):
        """The lightmap case keeps W19-2: no ``TEXCOORD_1``; the layer entries have no texcoord."""
        _, doc = library_document(self, self.table, lightmap=False)
        first = primitives(doc)[0]
        self.assertEqual(first["extras"]["d3sno_export"], {"uv2_scale": "open", "uv2_use": "lightmap",
                                                           "lightmap": LIGHT})
        self.assertNotIn("TEXCOORD_1", first["attributes"])
        layers = doc["materials"][first["material"]]["extras"]["d3sno_export"]["uv2_layers"]
        self.assertEqual({(e["texcoord"], "v_divide" in e) for e in layers}, {(None, False)})

    def test_without_rows_everything_unknown(self):
        source = six_mesh(LIBRARY_TEXTURES, [1, 2, 3, 4, 5, 6])
        doc = P.to_gltf(P.parse_appearance(source), texture_resolver=lambda h: f"{h}.png",
                        lightmap_resolver=lambda h: None)
        self.assertEqual([p["extras"]["d3sno_export"].get("uv2_use") for p in primitives(doc)],
                         ["lightmap"] + ["unknown"] * 5)

    def test_round_trip(self):
        """The mesh import gives the source byte for byte for every ``uv2_use``, also from
        ``TEXCOORD_1`` alone where it is written."""
        app = P.parse_appearance(self.source)
        self.assertEqual(I.import_mesh(app, I.read_gltf(self.glb)), self.source)
        for p in self.prims:
            if "TEXCOORD_1" in p["attributes"]:
                del p["attributes"]["_D3_UV2"]
        self.assertEqual(I.import_mesh(app, I.read_gltf(I.gltf_to_glb(self.doc))), self.source)


# --- the export, with the shipped tables -----------------------------------------------------------

SHADERS = {9001: "actor_additive_alphaComp_uv2_skin",  # one layer row: pass 0, stage 1, slot 19, 8192
           9002: "actor2_additive_skin"}  # known, no row
MAPS = {3001: (9001,), 3002: (9002,)}  # shader map handle -> shader handles
# sub-object: lightmap + layer, layer with a missing slot-19 texture, layer without a slot-19
# texture, unused, shader map -1, shader map not in the archives
EXPORT_TEXTURES = [[(2, LIGHT)], [(19, LAYER_TEX)], [], [], [], []]
EXPORT_MAPS = [3001, 3001, 3001, 3002, -1, 3999]


def export_files(light=S):
    files = F.switch_files()
    files["Common"][F.name_of("app", "Rock")] = six_mesh(EXPORT_TEXTURES, EXPORT_MAPS)
    if light is not None:
        files["Common"][F.name_of("tex", "Light")] = lightmap_texture(light)
    for handle, links in MAPS.items():
        data = shader_map("switch", [(2, 0x100 + k, h) for k, h in enumerate(links)] + [(2, 0x1FF, -1)])
        files["Act1"][F.name_of("shm", f"Map{handle}")] = F.with_handle(data, handle, "<")
    toc = [TocEntry(group_by_extension("shd").id, h, name) for h, name in SHADERS.items()]
    toc += [TocEntry(group_by_extension("shm").id, h, f"Map{h}") for h in MAPS]
    files["Act1"]["Act1TOC.dat"] = IW.write_core_toc(toc, "switch")
    return files


class Uv2UseExport(unittest.TestCase):
    """W20-1 to W20-3 in the export, shader maps read from the archives."""

    def setUp(self):
        need_uv2_tables()
        self.result, self.out = export(self, export_files())
        self.glb = (self.out / f"meshes/{F.PLAIN_APP}.glb").read_bytes()
        self.doc = glb_json(self.glb)
        self.prims = primitives(self.doc)

    def test_uses(self):
        uses = [p["extras"]["d3sno_export"]["uv2_use"] for p in self.prims]
        self.assertEqual(uses, ["lightmap", "layer", "layer", "unused", "unknown", "unknown"])
        listed = self.doc["extras"]["d3sno_export"]["primitives"]
        self.assertEqual([p["uv2_use"] for p in listed], uses)
        self.assertEqual([p["uv2_scale"] for p in listed], ["lightmap", "layer", "layer", "unused", "open", "open"])

    def test_layer_entries(self):
        def layers(i):
            return self.doc["materials"][self.prims[i]["material"]]["extras"]["d3sno_export"]["uv2_layers"]

        entry = {"shader": SHADERS[9001], "pass": 0, "stage": 1, "slot_type": 19, "slot_name": "alpha",
                 "divisor": 8192, "texcoord": 1}
        self.assertEqual(layers(0), [{**entry, "texture": None, "v_divide": S}])
        self.assertEqual(layers(1), [{**entry, "texture": f"../textures/{LAYER_TEX}.png"}])
        self.assertEqual(layers(2), [{**entry, "texture": None}])
        rows = accessor(self.glb, self.prims[0]["attributes"]["TEXCOORD_1"])
        self.assertAlmostEqual(rows[1][1], 2048 / 8192 * S, places=6)  # the lightmap rule

    def test_missing_dependency(self):
        """A layer texture not in the export is a missing dependency; a null one is not."""
        (note,) = [f for f in self.result.coverage["groups"]["app"]["files"] if f["handle"] == F.PLAIN_APP]
        self.assertEqual(note["reason"], "missing_dependency")
        self.assertEqual([n["handle"] for n in note["needs"]], [LAYER_TEX])

    def test_round_trip(self):
        source = six_mesh(EXPORT_TEXTURES, EXPORT_MAPS)
        self.assertEqual(I.import_mesh(P.parse_appearance(source), I.read_gltf(self.glb)), source)

    def test_shader_maps_not_exported(self):
        self.assertFalse(list(self.out.rglob("*shm*")))
        self.assertNotIn("shm", self.result.coverage["groups"])


class MissingTables(unittest.TestCase):
    """Without the tables the export runs, and every varying UV2 without a lightmap is ``"unknown"``."""

    def test_export(self):
        with mock.patch.object(U, "tables", lambda: None):
            _, out = export(self, export_files())
        prims = primitives(glb_json((out / f"meshes/{F.PLAIN_APP}.glb").read_bytes()))
        self.assertEqual([p["extras"]["d3sno_export"]["uv2_use"] for p in prims], ["lightmap"] + ["unknown"] * 5)
        self.assertEqual([("TEXCOORD_1" in p["attributes"]) for p in prims], [True] + [False] * 5)

    def test_message(self):
        with mock.patch(__name__ + ".MISSING_TABLES", (str(TABLE_PATHS[0]),)):
            with self.assertRaises(unittest.SkipTest) as caught:
                need_uv2_tables()
        self.assertIn("V26-8", str(caught.exception))
        self.assertIn(U.ROW_TABLE, str(caught.exception))


# --- rotation sense (W20-4) --------------------------------------------------------------------------

QX = (ROOT_HALF, 0.0, 0.0, ROOT_HALF)  # 90 degrees about x
QZ = (0.0, 0.0, ROOT_HALF, ROOT_HALF)  # 90 degrees about z


def conjugate(q):
    return (-q[0], -q[1], -q[2], q[3])


def gltf_world(nodes, index, parents):
    """A node's world matrix by the glTF reading of its TRS (row-major 3x4 here)."""
    def local(node):
        x, y, z, w = node.get("rotation", (0.0, 0.0, 0.0, 1.0))
        t = node.get("translation", (0.0, 0.0, 0.0))
        return [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), t[0]],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w), t[1]],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y), t[2]]]

    def times(a, b):
        out = [[sum(a[r][k] * b[k][c] for k in range(3)) for c in range(4)] for r in range(3)]
        for r in range(3):
            out[r][3] += a[r][3]
        return out

    matrix = local(nodes[index])
    while parents.get(index) is not None:
        index = parents[index]
        matrix = times(local(nodes[index]), matrix)
    return matrix


class Rotation(unittest.TestCase):
    def assertClose(self, got, want, places=6):
        for a, b in zip(got, want):
            self.assertAlmostEqual(a, b, places=places)

    def test_skeleton(self):
        """A child rotated 90 degrees about z under a parent rotated 90 degrees about x: the
        grandchild's world position follows ``q_parent * q_child``; the conjugated reading differs."""
        bones = [SimpleNamespace(index=0, parent=-1, name_hash=1, local=Transform(QX, (0.0, 0.0, 0.0))),
                 SimpleNamespace(index=1, parent=0, name_hash=2, local=Transform(QZ, (0.0, 1.0, 0.0))),
                 SimpleNamespace(index=2, parent=1, name_hash=3, local=Transform((0.0, 0.0, 0.0, 1.0), (1.0, 0.0, 0.0)))]
        nodes, roots, _ = P.skeleton_nodes(SimpleNamespace(bones=bones), False)
        for bone, node in zip(bones, nodes):  # written as stored, never conjugated
            self.assertEqual(node["rotation"], list(quat_normalise(bone.local.q)))
        world = bones[0].local.compose(bones[1].local).compose(bones[2].local)
        self.assertClose(world.t, (0.0, 0.0, 2.0))
        self.assertClose(world.q, quat_multiply(QX, QZ))
        matrix = gltf_world(nodes, 2, {0: None, 1: 0, 2: 1})
        self.assertClose([row[3] for row in matrix], (0.0, 0.0, 2.0))
        child = rotate(QX, (0.0, 1.0, 0.0))
        self.assertClose(child, (0.0, 0.0, 1.0))
        wrong_child = rotate(conjugate(QX), (0.0, 1.0, 0.0))
        wrong = tuple(a + b for a, b in zip(wrong_child, rotate(conjugate(quat_multiply(QX, QZ)), (1.0, 0.0, 0.0))))
        self.assertGreater(max(abs(a - b) for a, b in zip(wrong, world.t)), 1.0)  # rejected

    def test_placed_marker(self):
        """A marker at (1, 0, 0) with identity rotation, in a scene rotated 90 degrees about z and
        placed at (10, 0, 0), lands at (10, 1, 0) with the scene's rotation."""
        marker = SimpleNamespace(position=(1.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0, 1.0), type=0)
        entry = SimpleNamespace(handle=5, position=(10.0, 0.0, 0.0), rotation=QZ)
        wrl = SimpleNamespace(scene_entries=[entry], marker_sets=[])
        layout = WD.lay_out_world(wrl, {5: SimpleNamespace(marker_sets=[7])},
                                  {7: SimpleNamespace(markers=[marker], always_active=True, label=-1)})
        (placed,) = layout.markers
        self.assertClose(placed.position, (10.0, 1.0, 0.0))
        self.assertClose(placed.rotation, QZ)
        # a rotated marker: q_scene * q_marker, not q_marker * q_scene
        turned = SimpleNamespace(position=(0.0, 0.0, 1.0), rotation=QX, type=0)
        layout = WD.lay_out_world(wrl, {5: SimpleNamespace(marker_sets=[7])},
                                  {7: SimpleNamespace(markers=[turned], always_active=True, label=-1)})
        (placed,) = layout.markers
        self.assertClose(placed.position, (10.0, 0.0, 1.0))
        self.assertClose(placed.rotation, quat_multiply(QZ, QX))
        self.assertGreater(max(abs(a - b) for a, b in zip(quat_multiply(QX, QZ), quat_multiply(QZ, QX))), 0.1)

    def world_files(self, rotation):
        files = F.switch_files()
        files["Act1"][F.name_of("mrk", "Room")] = F.with_handle(B.build_mrk("switch", markers=[
            {"type": 0, "group": 1, "handle": F.ACR, "position": (1.0, 0.0, 0.0)}]), F.MRK, "<")
        files["Act1"][F.name_of("wrl", "Town")] = F.with_handle(B.build_wrl("switch", chunks=[[
            B.scene_entry("switch", F.SCN, position=(10.0, 0.0, 0.0), rotation=rotation)]]), F.WRL, "<")
        return files

    def test_world_in_export(self):
        _, out = export(self, self.world_files(QZ))
        doc = json.loads((out / f"data/wrl/{F.WRL}.json").read_text(encoding="utf-8"))
        (placed,) = doc["layout"]["markers"]
        self.assertClose(placed["position"], (10.0, 1.0, 0.0), places=5)
        self.assertClose(placed["rotation"]["quaternion"], QZ, places=6)
        (entry,) = doc["scene_entries"]
        self.assertEqual(entry["note"], "world_rotation_untested")
        _, plain = export(self, self.world_files((0.0, 0.0, 0.0, 1.0)))
        doc = json.loads((plain / f"data/wrl/{F.WRL}.json").read_text(encoding="utf-8"))
        self.assertNotIn("note", doc["scene_entries"][0])

    def test_marker_quaternion_as_stored(self):
        _, out = export(self, F.switch_files())
        doc = json.loads((out / f"data/mrk/{F.MRK}.json").read_text(encoding="utf-8"))
        second = doc["markers"][1]["rotation"]
        self.assertClose(second["quaternion"], (0.0, 0.0, 0.6, 0.8))
        self.assertClose(second["raw"], (0.0, 0.0, 0.6, 0.8))


class Flags(unittest.TestCase):
    """W20-4, W20-5."""

    def test_flags(self):
        _, out = export(self, F.switch_files())
        mesh = glb_json((out / "meshes/600.glb").read_bytes())["extras"]["d3sno_export"]
        anim = glb_json((out / "anims/700.glb").read_bytes())["extras"]["d3sno_export"]
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(mesh["rotation"], "active_xyzw")
        self.assertEqual(anim["rotation"], "active_xyzw")
        self.assertEqual(manifest["coordinates"]["rotation"], "active_xyzw")
        self.assertEqual(manifest["format_version"], 4)  # ``46`` W21-1
        for path in sorted(out.rglob("*.glb")):  # no empty array anywhere (V27-1)
            self.assertNotIn("[]", json.dumps(glb_json(path.read_bytes())), path.name)


if __name__ == "__main__":
    unittest.main()
