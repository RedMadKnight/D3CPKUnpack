"""Stage 19 (``43_settled_questions.md``): W19-1 to W19-5. Synthetic data only; every export goes to a
temporary folder outside any git working tree."""

import dataclasses
import io
import json
import struct
import unittest
from contextlib import redirect_stderr, redirect_stdout

import builders as B
import d3sno
import export_fixture as F
import test_stage17_writer as W
from d3sno import app_mesh as M
from d3sno import appearance as P
from d3sno import cli
from d3sno import export as E
from d3sno import mesh_import as I
from d3sno import texture as T
from d3sno import texture_write as TW
from test_stage18_export import glb_json

LIGHT = 520
RAW_UV2 = [[(0x8000, 0x8000), (0x8000 + 4096, 0x8000 + 2048), (0x8000 - 1024, 0x8000 + 8192)],
           [(0x9000, 0x7000), (0x8100, 0x8200), (0xA000, 0x8000)],
           [(0x8000, 0x8000)] * 3]


def lightmap_texture(scale: float) -> bytes:
    """A switch ``.tex`` whose lightmap V scale is ``scale``."""
    plain = TW.encode_texture(F.image(4, 4, 5), "switch", 0, levels=1)
    template = dataclasses.replace(T.parse_texture(plain, "switch"), lightmap_v_scale=scale)
    data = TW.encode_texture(F.image(4, 4, 5), "switch", 0, levels=1, template=template)
    assert T.parse_texture(data, "switch").lightmap_v_scale == scale
    return F.with_handle(data, LIGHT, "<")


def lightmapped_mesh() -> bytes:
    """Three rigid sub-objects: a lightmap slot and a varying UV2, no lightmap slot and a varying
    UV2, no lightmap slot and a constant UV2 (all 0x8000)."""
    data, build = W.switch_two()
    data = W.with_materials(data, build, looks=(0x61,), assign=(0xAAA, 0xBBB, 0xCCC), entries=[
        (0xAAA, [[(1, F.TEX), (2, LIGHT)]]), (0xBBB, [[(1, F.TEX)]]), (0xCCC, [[(1, F.TEX)]])])
    app = P.parse_appearance(data, build)
    for sub, rows in zip(P.export_sub_objects(app, True), RAW_UV2):
        sub.draws[0].uv2_raw = list(rows)
    data = W.canon(M.write_app_mesh(app, derive=True), build)
    check = P.parse_appearance(data, build)
    assert [s.draws[0].uv2_raw for s in P.export_sub_objects(check, True)] == RAW_UV2
    return F.with_handle(data, F.PLAIN_APP, "<")


def files_with(mesh=True, light: float | None = 1.25, velocity=None):
    files = F.switch_files()
    if mesh:
        files["Common"][F.name_of("app", "Rock")] = lightmapped_mesh()
    if light is not None:
        files["Common"][F.name_of("tex", "Light")] = lightmap_texture(light)
    if velocity is not None:
        files["Act1"][F.name_of("ani", "Hero_walk")] = F.with_handle(B.build_ani("switch", [
            B.simple_permutation("switch", velocity=velocity),
            B.simple_permutation("switch", name="Run", nonlinear=[(0.0, 0.0, 0.0)] * 3)]), F.ANI, "<")
    return files


def export(case, files, **options):
    root = F.temp_root(case)
    game = F.write_game(root / "game", files)
    result = E.export_game(game, root / "out", created="X", **options)
    case.assertEqual(result.exit_code, 0, result.message)
    return result, root / "out"


def accessor(glb: bytes, index: int) -> list[tuple]:
    return I.read_gltf(glb).accessor(index)


def primitives(doc: dict) -> list[dict]:
    return [p for mesh in doc["meshes"] for p in mesh["primitives"]]


class Version(unittest.TestCase):
    def test_version(self):
        self.assertGreaterEqual(tuple(int(x) for x in d3sno.__version__.split(".")), (0, 19, 0))
        self.assertEqual(d3sno.STAGE, 19)


class Flags(unittest.TestCase):
    """W19-1: the settled questions are statements."""

    def setUp(self):
        self.result, self.out = export(self, F.switch_files())

    def test_mesh(self):
        extras = glb_json((self.out / "meshes/600.glb").read_bytes())["extras"]["d3sno_export"]
        self.assertEqual(extras["v_direction"], "top_left_origin")
        self.assertEqual(extras["tangents"], "not_stored")
        self.assertNotIn("uv2_scale", extras)  # per primitive
        self.assertEqual({p["uv2_scale"] for p in extras["primitives"]}, {"none"})
        self.assertIn("MikkTSpace", extras["tangent_note"])

    def test_animation(self):
        extras = glb_json((self.out / "anims/700.glb").read_bytes())["extras"]["d3sno_export"]
        self.assertEqual(extras["root_motion"], "in_place")
        self.assertEqual(extras["ticks_per_second"], 60)
        self.assertNotIn("travel", extras)

    def test_manifest(self):
        manifest = json.loads((self.out / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["format_version"], 2)
        self.assertEqual(manifest["coordinates"], {"source": "stored", "up": "+z", "unit": "stored",
                                                   "handedness": "right", "forward": "+x", "winding": "ccw"})
        self.assertIs(manifest["options"]["travel"], False)
        coverage = json.loads((self.out / "coverage.json").read_text(encoding="utf-8"))
        self.assertEqual(coverage["format_version"], 1)

    def test_no_open_statement(self):
        for path in sorted(self.out.rglob("*.glb")):
            extras = glb_json(path.read_bytes())["extras"]["d3sno_export"]
            for key in ("v_direction", "tangents", "root_motion"):
                self.assertNotEqual(extras.get(key), "open", path.name)

    def test_no_tangents(self):
        """W19-4: no ``TANGENT`` attribute in any exported file."""
        _, out = export(self, files_with())
        for folder in (self.out, out):
            for path in sorted(folder.rglob("*.glb")):
                doc = glb_json(path.read_bytes())
                for p in primitives(doc) if "meshes" in doc else []:
                    self.assertNotIn("TANGENT", p["attributes"], path.name)


class LightmapUv2(unittest.TestCase):
    """W19-2, W19-3."""

    def test_texcoord_1(self):
        result, out = export(self, files_with())
        glb = (out / f"meshes/{F.PLAIN_APP}.glb").read_bytes()
        doc = glb_json(glb)
        prims = primitives(doc)
        self.assertEqual([p["extras"]["d3sno_export"]["uv2_scale"] for p in prims], ["lightmap", "open", "none"])
        self.assertEqual([p["uv2_scale"] for p in doc["extras"]["d3sno_export"]["primitives"]],
                         ["lightmap", "open", "none"])
        first = prims[0]
        self.assertEqual(first["extras"]["d3sno_export"], {"uv2_scale": "lightmap", "lightmap": LIGHT,
                                                           "lightmap_v_scale": 1.25})
        rows = accessor(glb, first["attributes"]["TEXCOORD_1"])
        want = [((u - 0x8000) / 8192, (v - 0x8000) / 8192 * 1.25) for u, v in RAW_UV2[0]]
        for got, (u, v) in zip(rows, want):
            self.assertAlmostEqual(got[0], u, places=6)
            self.assertAlmostEqual(got[1], v, places=6)
        self.assertEqual(accessor(glb, first["attributes"]["_D3_UV2"]), RAW_UV2[0])
        self.assertEqual([("TEXCOORD_1" in p["attributes"]) for p in prims], [True, False, False])
        material = doc["materials"][first["material"]]
        self.assertEqual(material["extras"]["d3sno_export"],
                         {"lightmap": f"../textures/{LIGHT}.png", "lightmap_channel": "red", "lightmap_texcoord": 1})
        for p in prims[1:]:
            self.assertNotIn("extras", doc["materials"][p["material"]])
        self.assertFalse([f for f in result.coverage["groups"]["app"]["files"] if f["handle"] == F.PLAIN_APP])

    def test_import_byte_for_byte(self):
        _, out = export(self, files_with())
        source = lightmapped_mesh()
        glb = (out / f"meshes/{F.PLAIN_APP}.glb").read_bytes()
        self.assertEqual(I.import_mesh(P.parse_appearance(source), I.read_gltf(glb)), source)

    def document(self, scale=1.25):
        source = lightmapped_mesh()
        doc = P.to_gltf(P.parse_appearance(source), texture_resolver=lambda h: f"{h}.png",
                        lightmap_resolver=lambda h: (f"{h}.png", scale))
        for p in primitives(doc):
            del p["attributes"]["_D3_UV2"]
        return source, doc

    def test_import_texcoord_1_only(self):
        """W19-3: without ``_D3_UV2`` the import inverts ``TEXCOORD_1`` with the exporter's V scale."""
        source, doc = self.document()
        app = P.parse_appearance(source)
        self.assertEqual(I.import_mesh(app, I.read_gltf(I.gltf_to_glb(doc))), source)
        self.assertEqual(I.import_mesh(app, I.read_gltf(I.gltf_to_glb(doc)), lightmap_v_scale=1.25), source)
        changed = I.import_mesh(app, I.read_gltf(I.gltf_to_glb(doc)), lightmap_v_scale=1.0)  # given wins
        self.assertNotEqual(changed, source)
        del primitives(doc)[0]["extras"]["d3sno_export"]["lightmap_v_scale"]
        self.assertNotEqual(I.import_mesh(app, I.read_gltf(I.gltf_to_glb(doc))), source)  # 1 without it

    def test_missing_lightmap(self):
        result, out = export(self, files_with(light=None))
        doc = glb_json((out / f"meshes/{F.PLAIN_APP}.glb").read_bytes())
        prims = primitives(doc)
        self.assertNotIn("TEXCOORD_1", prims[0]["attributes"])
        self.assertIn("_D3_UV2", prims[0]["attributes"])
        self.assertEqual(prims[0]["extras"]["d3sno_export"], {"uv2_scale": "open", "lightmap": LIGHT})
        self.assertNotIn("extras", doc["materials"][prims[0]["material"]])
        (note,) = [f for f in result.coverage["groups"]["app"]["files"] if f["handle"] == F.PLAIN_APP]
        self.assertEqual(note["reason"], "missing_dependency")
        self.assertTrue(note["exported"])
        self.assertIn(LIGHT, [n["handle"] for n in note["needs"]])

    def test_zero_scale_uses_one(self):
        _, out = export(self, files_with(light=0.0))
        glb = (out / f"meshes/{F.PLAIN_APP}.glb").read_bytes()
        first = primitives(glb_json(glb))[0]
        self.assertEqual(first["extras"]["d3sno_export"]["lightmap_v_scale"], 1.0)
        rows = accessor(glb, first["attributes"]["TEXCOORD_1"])
        self.assertEqual([r[1] for r in rows], [(v - 0x8000) / 8192 for _, v in RAW_UV2[0]])

    def test_library_default_unchanged(self):
        """Without a lightmap resolver ``to_gltf`` is as before (``lightmap_v_scale`` for every draw)."""
        doc = P.to_gltf(P.parse_appearance(lightmapped_mesh()), lightmap_v_scale=2.0)
        for p in primitives(doc):
            self.assertIn("TEXCOORD_1", p["attributes"])
            self.assertNotIn("d3sno_export", p["extras"])


class Velocity(unittest.TestCase):
    """W19-5."""

    def test_extras(self):
        _, out = export(self, files_with(mesh=False, light=None, velocity=(0.125, 0.0, 0.0)))
        perms = glb_json((out / "anims/700.glb").read_bytes())["extras"]["d3sno_export"]["permutations"]
        self.assertEqual(perms[0]["average_velocity"], [0.125, 0.0, 0.0])
        self.assertEqual(perms[0]["nonlinear_offsets"], [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0]])
        self.assertEqual(perms[1]["average_velocity"], [1.0, 0.0, 0.0])
        self.assertNotIn("nonlinear_offsets", perms[1])  # all rows zero

    def test_travel(self):
        files = files_with(mesh=False, light=None, velocity=(0.125, 0.0, 0.0))
        result, out = export(self, files, travel=True)
        _, plain = export(self, files)
        glb = (out / "anims/700.glb").read_bytes()
        doc, base = glb_json(glb), glb_json((plain / "anims/700.glb").read_bytes())
        count = len(base["nodes"])
        self.assertEqual(doc["nodes"][:count], base["nodes"])  # joint nodes unchanged
        travel = doc["nodes"][count]
        self.assertEqual(travel, {"name": "d3sno_travel", "children": base["scenes"][0]["nodes"],
                                  "extras": {"d3sno_export": {"added": True}}})
        self.assertEqual(doc["scenes"][0]["nodes"], [count])
        self.assertIs(doc["extras"]["d3sno_export"]["travel"], True)
        self.assertIs(result.manifest["options"]["travel"], True)
        walk = doc["animations"][0]
        (channel,) = [c for c in walk["channels"] if c["target"]["node"] == count]
        self.assertEqual(channel["target"]["path"], "translation")
        sampler = walk["samplers"][channel["sampler"]]
        times = [t[0] for t in accessor(glb, sampler["input"])]
        moves = accessor(glb, sampler["output"])
        self.assertEqual(moves[0], (0.0, 0.0, 0.0))
        self.assertAlmostEqual((moves[-1][0] - moves[0][0]) / (times[-1] - times[0]), 7.5, places=5)
        self.assertEqual([m[1:] for m in moves], [(0.0, 0.0)] * len(moves))
        for a, b in zip(doc["animations"], base["animations"]):  # the bone channels are unchanged
            self.assertEqual(a["channels"][:len(b["channels"])], b["channels"])
        self.assertNotIn("d3sno_travel", [n["name"] for n in base["nodes"]])

    def test_command_line(self):
        root = F.temp_root(self)
        game = F.write_game(root / "game", files_with(mesh=False, light=None))
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            code = cli.main(["export", str(game), "--out", str(root / "out"), "--travel", "--groups", "app,ani,acr"])
        self.assertEqual(code, 0)
        doc = glb_json((root / "out/anims/700.glb").read_bytes())
        self.assertIn("d3sno_travel", [n["name"] for n in doc["nodes"]])


class SkipExisting(unittest.TestCase):
    """An earlier manifest of format version 1, or of another ``--travel``, is not reused."""

    def run_twice(self, change):
        root = F.temp_root(self)
        game = F.write_game(root / "game", F.switch_files())
        out = root / "out"
        self.assertEqual(E.export_game(game, out, created="X").exit_code, 0)
        path = out / "meshes/600.glb"
        first = path.read_bytes()
        path.write_bytes(b"stale")
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        change(manifest)
        (out / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        self.assertEqual(E.export_game(game, out, created="X", skip_existing=True).exit_code, 0)
        return path.read_bytes(), first

    def test_same_version_kept(self):
        now, _ = self.run_twice(lambda m: None)
        self.assertEqual(now, b"stale")

    def test_version_1_written_again(self):
        now, first = self.run_twice(lambda m: m.update(format_version=1))
        self.assertEqual(now, first)

    def test_other_travel_written_again(self):
        now, first = self.run_twice(lambda m: m["options"].update(travel=True))
        self.assertEqual(now, first)


if __name__ == "__main__":
    unittest.main()
