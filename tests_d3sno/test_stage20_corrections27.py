"""Corrections 27 (``45_corrections27.md``): X1 to X8, version 0.20.1. Synthetic data only; every export
goes to a temporary folder outside any git working tree. Shader names of the library tests are our own,
in a test table; tests that read the shipped tables are skipped when one is missing (V26-8)."""

import _libpath  # noqa: F401  (python -I: the library folder on sys.path)
import json
import unittest
from types import SimpleNamespace
from unittest import mock

import d3sno
import export_fixture as F
from d3sno import appearance as P
from d3sno import export as E
from d3sno import mesh_import as I
from d3sno import uv2_use as U
from test_stage18_export import glb_json
from test_stage19 import LIGHT, export, files_with, primitives
from test_stage20 import S, make_tables, need_uv2_tables, six_mesh

MISSING_LIGHT = 999  # a lightmap texture that no resolver knows
CONST = [(0x8800, 0x8800)] * 3
VARY = [(0x8000, 0x8000), (0xA000, 0x9000), (0x7000, 0x8800)]

ROWS = [("t_layer", 0, 1, 19, 8192, 0), ("t_odd", 0, 2, 12, 512, 0), ("t_light", 0, 1, 2, 8192, 1),
        ("t_glow", 1, 0, 6, 8192, 1),  # X6: a layer slot read through the lightmap coordinate
        ("t_diffuse", 0, 0, 1, 8192, 0),  # X3: the diffuse slot fed by UV2
        ("ps3", "t_both", 0, 0, 12, 8192, 0), ("x360", "t_both", 0, 0, 12, 512, 0),  # X6: per build
        ("x360", "t_blank", 0, 3, None, 512, 0)]  # an x360 row without a slot type (Q894)
KNOWN = ["t_layer", "t_odd", "t_light", "t_glow", "t_diffuse", "t_plain", "t_both"]
KNOWN_X360 = KNOWN + ["t_xonly", "t_blank"]
MAPS = {1: ("t_layer", "t_glow", "t_odd"), 2: ("t_light",), 3: ("t_layer",), 4: ("t_plain",), 5: ("t_glow",),
        6: ("t_diffuse",)}
# sub-object: X1 lightmap exported with constant UV2; X1 lightmap missing; X1/X2 constant UV2 without a
# lightmap; X1 varying (unused); X6 a lightmap_v_scale-1 row on slot 6; X3 a slot-1 row
TEXTURES = [[(2, LIGHT), (6, 540)], [(2, MISSING_LIGHT)], [(19, 541)], [], [(6, 542)], []]
UV2 = [CONST, VARY, CONST, VARY, VARY, VARY]


def document(case):
    table = U.load_tables(make_tables(case, rows=ROWS, known=KNOWN, known_x360=KNOWN_X360))
    source = six_mesh(TEXTURES, [1, 2, 3, 4, 5, 6], UV2)
    doc = P.to_gltf(P.parse_appearance(source), texture_resolver=lambda h: f"../textures/{h}.png",
                    lightmap_resolver=lambda h: (f"../textures/{h}.png", S) if h == LIGHT else None,
                    uv2_rows=lambda block: U.rows_for(MAPS.get(block.shader_map) if block else None, table, "switch"))
    return table, source, doc


def layers(doc, prim):
    return doc["materials"][prim["material"]].get("extras", {}).get("d3sno_export", {}).get("uv2_layers")


class Version(unittest.TestCase):
    def test_version(self):
        self.assertGreaterEqual(tuple(int(x) for x in d3sno.__version__.split(".")), (0, 20, 1))
        self.assertGreaterEqual(d3sno.STAGE, 20)


class LightmapSlot(unittest.TestCase):
    """X1: the four cases."""

    def setUp(self):
        self.table, self.source, self.doc = document(self)
        self.prims = primitives(self.doc)

    def info(self, i):
        return self.prims[i]["extras"]["d3sno_export"]

    def test_exported_with_constant_uv2(self):
        self.assertEqual(self.info(0), {"uv2_scale": "lightmap", "uv2_use": "lightmap", "lightmap": LIGHT,
                                        "lightmap_v_scale": S})
        self.assertIn("TEXCOORD_1", self.prims[0]["attributes"])

    def test_missing(self):
        self.assertEqual(self.info(1), {"uv2_scale": "open", "uv2_use": "lightmap", "lightmap": MISSING_LIGHT})
        self.assertNotIn("TEXCOORD_1", self.prims[1]["attributes"])

    def test_constant_without_lightmap(self):
        self.assertEqual(self.info(2), {"uv2_scale": "none"})
        self.assertNotIn("TEXCOORD_1", self.prims[2]["attributes"])

    def test_varying_without_lightmap(self):
        self.assertEqual([self.info(i)["uv2_use"] for i in (3, 4, 5)], ["unused", "layer", "layer"])

    def test_missing_in_export_coverage(self):
        """X1: a lightmap slot whose texture is missing lists the mesh under ``missing_dependency``."""
        files = files_with(light=None)
        result, _ = export(self, files)
        notes = [f for f in result.coverage["groups"]["app"]["files"] if f.get("reason") == "missing_dependency"]
        self.assertIn(LIGHT, [n["handle"] for f in notes for n in f["needs"]])

    def test_round_trip(self):
        self.assertEqual(I.import_mesh(P.parse_appearance(self.source), I.read_gltf(I.gltf_to_glb(self.doc))),
                         self.source)


class Entries(unittest.TestCase):
    """X2, X3 and the entry keys of X6."""

    def setUp(self):
        self.table, _, self.doc = document(self)
        self.prims = primitives(self.doc)

    def test_constant_uv2_entries(self):
        """X2: a constant-UV2 primitive keeps its ``uv2_layers``, each with ``texcoord`` null."""
        self.assertEqual(layers(self.doc, self.prims[2]), [
            {"shader": "t_layer", "pass": 0, "stage": 1, "slot_type": 19, "slot_name": "alpha", "divisor": 8192,
             "texture": "../textures/541.png", "texcoord": None}])

    def test_lightmap_coordinate(self):
        """X3 with the lightmap ``TEXCOORD_1``: a ``lightmap_v_scale`` 0 row is v divided by S, a 1 row
        names it as it is; ``48`` Z2: a 512 row reads it as a ``lightmap_v_scale`` 0 row does (before: nothing)."""
        got = {e["shader"]: (e["texcoord"], e.get("v_divide"), e.get("v_scale")) for e in layers(self.doc, self.prims[0])}
        self.assertEqual(got, {"t_glow": (1, None, "lightmap"), "t_layer": (1, S, None), "t_odd": (1, S, None)})
        glow = [e for e in layers(self.doc, self.prims[0]) if e["shader"] == "t_glow"][0]
        self.assertEqual(glow["texture"], "../textures/540.png")

    def test_layer_coordinate(self):
        """X3 with the layer ``TEXCOORD_1``; ``47`` Y6: a ``lightmap_v_scale`` 1 row names it (``texcoord`` 1,
        ``v_scale``, no ``v_divide``; before Y6 ``texcoord`` was null)."""
        (glow,) = layers(self.doc, self.prims[4])
        self.assertEqual(glow, {"shader": "t_glow", "pass": 1, "stage": 0, "slot_type": 6, "slot_name": "glow", "divisor": 8192,
                                "texture": "../textures/542.png", "texcoord": 1, "v_scale": "lightmap"})
        self.assertIn("TEXCOORD_1", self.prims[4]["attributes"])

    def test_also_base_color(self):
        (entry,) = layers(self.doc, self.prims[5])
        self.assertEqual((entry["slot_type"], entry["texcoord"], entry["also_base_color"]), (1, 1, True))
        self.assertEqual(entry["texture"], f"../textures/{F.TEX}.png")
        material = self.doc["materials"][self.prims[5]["material"]]
        base = material["pbrMetallicRoughness"]["baseColorTexture"]
        self.assertNotIn("texCoord", base)  # still bound with TEXCOORD_0 (W18-4)
        for prim in self.prims:
            for e in layers(self.doc, prim) or ():
                self.assertEqual("also_base_color" in e, e["slot_type"] == 1)
                self.assertEqual("v_scale" in e, e["shader"] == "t_glow")

    def test_table_rows(self):
        """Each row of the X3 table on ``entry_texcoord``."""
        def row(divisor, scale):
            return U.Uv2Row("x", 0, 0, 12, divisor, scale)

        cases = [(row(8192, 0), "layer", (1, None)), (row(8192, 0), "lightmap", (1, S)),
                 (row(8192, 1), "lightmap", (1, None)), (row(8192, 1), "layer", (1, None)),  # Y6
                 (row(512, 0), "layer", (1, None)), (row(512, 0), "lightmap", (1, S)),  # ``48`` Z2
                 (row(8192, 0), None, (None, None)), (row(8192, 1), None, (None, None))]
        for item, written, want in cases:
            self.assertEqual(U.entry_texcoord(item, written, S), want, (item, written))


class Handedness(unittest.TestCase):
    """X4: the manifest's ``coordinates`` holds the convention; no JSON record repeats it."""

    def test_no_key_per_record(self):
        _, out = export(self, F.switch_files())
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        for key in ("handedness", "forward", "winding", "rotation"):
            self.assertIn(key, manifest["coordinates"])

        def keys(value):
            if isinstance(value, dict):
                for k, v in value.items():
                    yield k
                    yield from keys(v)
            elif isinstance(value, list):
                for v in value:
                    yield from keys(v)

        records = sorted((out / "data").rglob("*.json"))
        self.assertTrue(any(p.parent.name in ("scn", "wrl", "mrk") for p in records))
        for path in records:
            found = set(keys(json.loads(path.read_text(encoding="utf-8"))))
            self.assertFalse(found & {"handedness", "forward", "winding"}, path)


class Velocity(unittest.TestCase):
    """X5: ``average_velocity`` per permutation, ``ticks_per_second`` once per file."""

    def test_two_permutations(self):
        _, out = export(self, files_with(mesh=False, light=None, velocity=(0.125, 0.0, 0.0)))
        extras = glb_json((out / "anims/700.glb").read_bytes())["extras"]["d3sno_export"]
        perms = extras["permutations"]
        self.assertEqual(len(perms), 2)
        self.assertEqual([p["average_velocity"] for p in perms], [[0.125, 0.0, 0.0], [1.0, 0.0, 0.0]])
        self.assertIn("ticks_per_second", extras)
        for p in perms:
            self.assertNotIn("ticks_per_second", p)


class PerBuild(unittest.TestCase):
    """X6: each build's own rows and list of known shaders."""

    def setUp(self):
        self.table = U.load_tables(make_tables(self, rows=ROWS, known=KNOWN, known_x360=KNOWN_X360))

    def use(self, names, build):
        return U.uv2_use(U.rows_for(names, self.table, build), False)

    def test_same_name_per_build(self):
        ps3 = U.layer_entries(U.rows_for(("t_both",), self.table, "ps3"), None, str, None)
        x360 = U.layer_entries(U.rows_for(("t_both",), self.table, "x360"), None, str, None)
        self.assertNotEqual(ps3, x360)
        self.assertEqual([e["divisor"] for e in ps3], [8192])
        self.assertEqual([e["divisor"] for e in x360], [512])
        # ``48`` Z2: a divisor-512 row is ``"layer"`` (before: ``"open"``)
        self.assertEqual((self.use(("t_both",), "ps3"), self.use(("t_both",), "x360")), ("layer", "layer"))

    def test_switch_uses_own_rows(self):
        """``48`` Z1 (before: switch used the ``ps3`` rows): the test table repeats each ps3 row as a switch row."""
        switch = U.rows_for(("t_layer",), self.table, "switch")
        self.assertEqual([r.build for r in switch.rows], ["switch"])
        self.assertEqual([r.key for r in switch.rows], [r.key for r in U.rows_for(("t_layer",), self.table, "ps3").rows])
        self.assertEqual(self.use(("t_both",), "switch"), "unused")  # a ps3-only row does not serve switch
        self.assertEqual(U.table_build("switch"), "switch")

    def test_x360_only_shader(self):
        self.assertEqual(self.use(("t_xonly",), "x360"), "unused")
        self.assertEqual(self.use(("t_xonly",), "switch"), "unknown")  # not in shader_known.txt

    def test_glow_row_is_layer(self):
        rows = U.rows_for(("t_glow",), self.table, "ps3")
        self.assertEqual(len(rows.layer), 1)
        self.assertEqual(self.use(("t_glow",), "ps3"), "layer")

    def test_row_without_slot_type(self):
        """Q894: kept as an odd row with ``slot_type`` and ``slot_name`` null and no texture."""
        rows = U.rows_for(("t_blank",), self.table, "x360")
        self.assertEqual(self.use(("t_blank",), "x360"), "layer")  # ``48`` Z2 (before: ``"open"``)
        (entry,) = U.layer_entries(rows, SimpleNamespace(texture=lambda slot: 5), str, "layer")
        self.assertEqual((entry["slot_type"], entry["slot_name"], entry["texture"], entry["texcoord"]),
                         (None, None, None, 1))  # Z2: ``texcoord`` 1 (before: null)
        self.assertEqual(U.layer_texture_handles(rows, SimpleNamespace(texture=lambda slot: 5)), [])

    def test_export_uses_archive_build(self):
        block = SimpleNamespace(shader_map=1)
        with mock.patch.object(U, "tables", lambda: self.table):
            for build, want in (("x360", "unused"), ("switch", "unknown"), ("ps3", "unknown")):
                run = SimpleNamespace(build=build, shader_names=lambda handle: ("t_xonly",))
                self.assertEqual(U.uv2_use(E._Run.uv2_rows(run, block), False), want, build)

    def test_shipped_rows_per_build(self):
        need_uv2_tables()
        table = U.load_tables()
        name = "assaultBeast_deathTransition"
        # ``48`` Z1, Z2: ``"layer"`` on every build now (before: ps3 and switch ``"open"``); the rows still differ
        self.assertEqual(U.uv2_use(U.rows_for((name,), table, "ps3"), False), "layer")
        self.assertEqual(U.uv2_use(U.rows_for((name,), table, "switch"), False), "layer")
        self.assertEqual(U.uv2_use(U.rows_for((name,), table, "x360"), False), "layer")
        self.assertTrue(U.rows_for((name,), table, "ps3").odd)
        self.assertFalse(U.rows_for((name,), table, "x360").odd)
        self.assertEqual(len(U.rows_for(("actor_hydra",), table, "ps3").rows), 5)
        self.assertEqual(len(U.rows_for(("actor_hydra",), table, "x360").rows), 9)
        glow = U.rows_for(("prop_transparent_glow_ice",), table, "ps3")  # lightmap_v_scale 1 on slot 6
        self.assertEqual([(r.slot_type, r.lightmap_v_scale) for r in glow.layer], [(6, 1)])
        self.assertEqual(U.uv2_use(glow, False), "layer")
        only = U.rows_for(("actor2_alphatest_alphamask",), table, "x360")  # x360 list only, no row
        self.assertEqual(U.uv2_use(only, False), "unused")
        self.assertEqual(U.uv2_use(U.rows_for(("actor2_alphatest_alphamask",), table, "ps3"), False), "unknown")


class MissingX360List(unittest.TestCase):
    """X8: without ``shader_known_x360.txt`` x360 primitives with varying UV2 and no lightmap are unknown."""

    def setUp(self):
        self.table = U.load_tables(make_tables(self, rows=ROWS, known=KNOWN, known_x360=KNOWN_X360,
                                               skip=(U.KNOWN_TABLE_X360,)))

    def test_library(self):
        self.assertEqual(set(self.table.known), {"ps3", "switch"})
        for names in (("t_xonly",), ("t_layer",), ("t_both",), ("t_plain",)):
            self.assertEqual(U.uv2_use(U.rows_for(names, self.table, "x360"), False), "unknown", names)
        self.assertEqual(U.uv2_use(U.rows_for(("t_layer",), self.table, "switch"), False), "layer")
        self.assertEqual(U.uv2_use(U.rows_for(("t_layer",), self.table, "x360"), True), "lightmap")

    def test_export_run(self):
        block = SimpleNamespace(shader_map=1)
        with mock.patch.object(U, "tables", lambda: self.table):
            run = SimpleNamespace(build="x360", shader_names=lambda handle: ("t_layer",))
            self.assertEqual(U.uv2_use(E._Run.uv2_rows(run, block), False), "unknown")

    def test_message(self):
        import test_stage20 as T
        path = str(U.DATA_DIR / U.KNOWN_TABLE_X360)
        self.assertIn(path, [str(p) for p in T.TABLE_PATHS])
        with mock.patch.object(T, "MISSING_TABLES", (path,)):
            with self.assertRaises(unittest.SkipTest) as caught:
                need_uv2_tables()
        self.assertIn("V26-8", str(caught.exception))
        self.assertIn(U.KNOWN_TABLE_X360, str(caught.exception))


if __name__ == "__main__":
    unittest.main()
