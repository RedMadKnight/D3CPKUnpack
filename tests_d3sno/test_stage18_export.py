"""Stage 18 (``40_export.md``): the ``export`` command and ``d3sno.export`` (W18-1 to W18-10), plus
V25-1. Synthetic data only; every output goes to a temporary folder outside any git working tree."""

import copy
import hashlib
import io
import json
import os
import struct
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

import d3sno
import export_fixture as F
from d3sno import FormatError
from d3sno import app_mesh as M
from d3sno import appearance as P
from d3sno import archive as A
from d3sno import cli
from d3sno import export as E
from d3sno import mesh_import as I
from d3sno import texture as T
from test_stage17_2 import fat_cases


class Marker(unittest.TestCase):
    def test_version(self):
        self.assertGreaterEqual(tuple(int(x) for x in d3sno.__version__.split(".")), (0, 18, 0))
        self.assertGreaterEqual(d3sno.STAGE, 18)


def glb_json(data: bytes) -> dict:
    length = struct.unpack_from("<I", data, 12)[0]
    return json.loads(data[20:20 + length])


class Exported(unittest.TestCase):
    """One export of the switch set, shared by the read-only tests."""

    files = None

    def setUp(self):
        self.root = F.temp_root(self)
        self.files = F.switch_files()
        self.game = F.write_game(self.root / "game", self.files)
        self.out = self.root / "out"

    def export(self, out=None, **options):
        return E.export_game(self.game, out or self.out, created="2026-01-01T00:00:00Z", **options)

    def read(self, path):
        return json.loads((self.out / path).read_text(encoding="utf-8"))


class Layout(Exported):
    """W18-2, W18-3: the paths, the manifest and the coverage report."""

    def test_paths(self):
        result = self.export()
        self.assertEqual(result.exit_code, 0, result.message)
        expected = {"manifest.json", "coverage.json", "meshes/600.glb", "meshes/601.glb", "anims/700.glb",
                    "textures/500.png", "textures/502.png", "textures/503.png", "data/acr/100.json",
                    "data/mon/110.json", "data/mat/1100.json", "data/scn/1200.json", "data/wrl/1400.json",
                    "data/lvl/1600.json", "data/mrk/1300.json", "data/phm/1000.json", "data/text/enUS/1500.json",
                    *(f"textures/501_f{k}.png" for k in range(6))}
        found = {p.relative_to(self.out).as_posix() for p in self.out.rglob("*") if p.is_file()}
        self.assertEqual(found, expected)
        self.assertEqual({e["path"] for e in result.manifest["entries"]}, expected - {"manifest.json", "coverage.json"})

    def test_manifest(self):
        result = self.export()
        manifest = self.read("manifest.json")
        self.assertEqual(manifest, json.loads(json.dumps(result.manifest)))
        self.assertEqual(manifest["format"], "d3sno-export")
        self.assertEqual(manifest["format_version"], 2)  # ``43`` W19-1
        self.assertEqual(manifest["build"], "switch")
        self.assertEqual(manifest["library_version"], d3sno.__version__)
        self.assertEqual(manifest["coordinates"]["up"], "+z")
        self.assertEqual(manifest["coordinates"]["source"], "stored")
        self.assertEqual(manifest["options"], {"archives": None, "groups": sorted(E.EXPORT_GROUPS), "all_levels": False,
                                               "travel": False})
        entries = manifest["entries"]
        self.assertEqual(entries, sorted(entries, key=lambda e: (e["group"], e["handle"], e["path"])))
        by_path = {e["path"]: e for e in entries}
        mesh = by_path["meshes/600.glb"]
        self.assertEqual((mesh["group"], mesh["handle"], mesh["name"]), ("app", 600, "Hero"))
        stored = self.files["Common"][F.name_of("app", "Hero")]
        self.assertEqual(mesh["source"], {"archive": "Common", "hash": hashlib.sha256(stored).hexdigest()})
        self.assertEqual(by_path["textures/503.png"].get("normal_map"), True)
        self.assertNotIn("normal_map", by_path["textures/500.png"])
        self.assertEqual([by_path[f"textures/501_f{k}.png"]["face"] for k in range(6)], list(range(6)))
        self.assertEqual(by_path["data/text/enUS/1500.json"]["locale"], "enUS")
        # one copy per handle: texture 500 is in Act1 and Common, written once from Act1 (sorted)
        self.assertEqual([e["source"]["archive"] for e in entries if e["handle"] == 500], ["Act1"])

    def test_privacy(self):
        self.export()
        for name in ("manifest.json", "coverage.json"):
            raw = (self.out / name).read_text(encoding="utf-8")
            for secret in (str(self.root), str(self.root).replace("\\", "/"), str(self.root).replace("\\", "\\\\")):
                self.assertNotIn(secret, raw)

    def test_coverage_counts(self):
        result = self.export()
        cov = result.coverage
        listing = {}
        for stem in ("Act1", "Common"):
            cpk = A.read_archive((self.game / f"{stem}.cpk").read_bytes())
            for entry in cpk.files:
                ext = entry.name.partition("|")[0].rpartition(".")[2]
                listing[ext] = listing.get(ext, 0) + 1
        for ext in E.EXPORT_GROUPS:
            group = cov["groups"][ext]
            self.assertEqual(group["found"], listing.get(ext, 0), ext)
            self.assertEqual(group["unique"], group["exported"] + group["skipped"], ext)
            skipped = [f for f in group["files"] if not f["exported"]]
            self.assertEqual(len(skipped), group["skipped"], ext)
            for f in group["files"]:
                self.assertIn(f["reason"], E.REASONS)
        self.assertEqual([a["archive"] for a in cov["archives"]], ["Act1", "Common"])  # no cutscene archive
        self.assertIsNone(cov["stopped"])

    def test_filtered_groups(self):
        result = self.export(groups="tex,app")
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(result.manifest["options"]["groups"], ["app", "tex"])
        self.assertEqual({e["group"] for e in result.manifest["entries"]}, {"app", "tex"})
        files = result.coverage["groups"]["acr"]["files"]
        self.assertEqual([(f["handle"], f["reason"]) for f in files], [(100, "filtered")])
        # W18-6: the animation links are read even when --groups leaves out .acr and .ans
        result = self.export(out=self.root / "anims", groups=["app", "ani"])
        self.assertIn("anims/700.glb", {e["path"] for e in result.manifest["entries"]})

    def test_archive_patterns(self):
        result = self.export(archives=["Act1*"])
        self.assertEqual(result.exit_code, 0)
        self.assertEqual({e["source"]["archive"] for e in result.manifest["entries"]}, {"Act1"})
        self.assertEqual(result.manifest["options"]["archives"], ["Act1*"])
        # the cutscene archive is never exported, even when a pattern names it
        self.assertEqual(self.export(out=self.root / "c", archives=["*Cutscene"]).exit_code, 2)

    def test_summary_lines(self):
        result = self.export()
        self.assertEqual(len(result.summary), len(E.EXPORT_GROUPS) + 1)
        self.assertTrue(result.summary[0].startswith("tex: found 6, exported 4, skipped 1"))
        self.assertTrue(result.summary[-1].startswith("total: "))


class Deterministic(Exported):
    def tree(self, out):
        return {p.relative_to(out).as_posix(): p.read_bytes() for p in out.rglob("*") if p.is_file()}

    def test_two_runs(self):
        E.export_game(self.game, self.root / "one", created="2026-01-01T00:00:00Z")
        E.export_game(self.game, self.root / "two")
        one, two = self.tree(self.root / "one"), self.tree(self.root / "two")
        self.assertEqual(one.keys(), two.keys())
        for path in one:
            if path == "manifest.json":
                a, b = json.loads(one[path]), json.loads(two[path])
                self.assertNotEqual(a.pop("created"), b.pop("created"))
                self.assertEqual(a, b)
            else:
                self.assertEqual(one[path], two[path], path)

    def test_json_form(self):
        self.export()
        for path in self.out.rglob("*.json"):
            raw = path.read_text(encoding="utf-8")
            again = json.dumps(json.loads(raw), sort_keys=True, indent=1, ensure_ascii=False) + "\n"
            self.assertEqual(raw, again, path.name)

    def test_float32_text(self):
        self.assertEqual(E.f32_text(0.1), "0.1")
        self.assertEqual(E.f32_text(1.0), "1.0")
        self.assertEqual(E.f32_text(1 / 3), "0.33333334")
        self.assertEqual(E.f32_text(struct.unpack("<f", struct.pack("<f", 0.7))[0]), "0.7")
        self.assertEqual(E.f32_text(-2.5e-8), "-2.5e-08")
        self.assertEqual(E.f32_text(0.33333333), E.f32_text(1 / 3))  # one float32, one text
        for value in (0.6, 123456.79, 3.4e38, 1e-45):
            text = E.f32_text(value)
            self.assertEqual(struct.pack("<f", float(text)), struct.pack("<f", value))
        self.assertEqual(E.dumps({"b": [1.0, float("nan")], "a": None}), b'{\n "a": null,\n "b": [\n  1.0,\n  "nan"\n ]\n}\n')


class SkipExisting(Exported):
    def test_keeps_matching_files(self):
        self.export()
        target = self.out / "textures/500.png"
        target.write_bytes(b"kept")
        self.export(skip_existing=True)
        self.assertEqual(target.read_bytes(), b"kept")  # same source and hash: kept
        self.export()
        self.assertNotEqual(target.read_bytes(), b"kept")  # without the option: written again

    def test_changed_source_rewritten(self):
        self.export()
        target = self.out / "data/phm/1000.json"
        target.write_bytes(b"old")
        files = F.switch_files()
        phm = bytearray(files["Act1"][F.name_of("phm", "Room")])
        struct.pack_into("<I", phm, 0x1C, 77)  # another flags word: another hash
        files["Act1"][F.name_of("phm", "Room")] = bytes(phm)
        F.write_game(self.game, files)
        self.export(skip_existing=True)
        self.assertEqual(self.read("data/phm/1000.json")["flags"], 77)

    def test_missing_file_rewritten(self):
        self.export()
        (self.out / "meshes/600.glb").unlink()
        self.export(skip_existing=True)
        self.assertTrue((self.out / "meshes/600.glb").is_file())


class Refusals(Exported):
    """W18-1: exit code 2, nothing written."""

    def refused(self, out, **options):
        result = E.export_game(self.game if "game" not in options else options.pop("game"), out, **options)
        self.assertEqual(result.exit_code, 2, result.message)
        self.assertIsNone(result.manifest)
        return result

    def test_git_working_tree(self):
        repo = self.root / "repo"
        (repo / ".git").mkdir(parents=True)
        out = repo / "deep" / "out"
        result = self.refused(out)
        self.assertIn("git working tree", result.message)
        self.assertFalse(out.exists())
        (repo / ".git").rmdir()
        (repo / ".git").write_text("gitdir: elsewhere")  # a worktree's .git file counts too
        self.refused(out)
        forced = E.export_game(self.game, out, force=True)
        self.assertEqual(forced.exit_code, 0)

    def test_game_dir(self):
        self.refused(self.game)
        self.refused(self.game / "inside")
        empty = self.root / "empty"
        (empty / "x").mkdir(parents=True)
        (empty / "x" / "not.cpk").write_bytes(b"nothing")
        self.refused(self.root / "o1", game=empty)
        self.refused(self.root / "o2", game=self.root / "nowhere")

    def test_options(self):
        self.refused(self.root / "o", groups="tex,pow")
        self.refused(self.root / "o", build="ps3")  # the archives say switch
        self.refused(self.root / "o", archives=["Act9*"])
        self.assertFalse((self.root / "o").exists())

    def test_build_given_that_matches(self):
        self.assertEqual(E.export_game(self.game, self.out, build="switch").exit_code, 0)


class Errors(Exported):
    def test_broken_file_reported(self):
        result = self.export()
        self.assertEqual(result.exit_code, 0)
        broken = [f for f in result.coverage["groups"]["tex"]["files"] if f["handle"] == F.BROKEN]
        self.assertEqual(len(broken), 1)
        self.assertEqual((broken[0]["reason"], broken[0]["exported"]), ("parse_error", False))
        self.assertTrue(broken[0]["message"])
        self.assertTrue((self.out / "textures/502.png").is_file())  # the export went on

    def test_failed_file_leaves_no_output(self):
        real = T.encode_png
        calls = []

        def failing(image):
            calls.append(image)
            if len(calls) == 4:  # the fourth face of the cube
                raise FormatError("synthetic failure")
            return real(image)

        with mock.patch.object(T, "encode_png", failing):
            result = self.export(groups="tex")
        self.assertEqual(result.exit_code, 0)
        self.assertFalse(any(self.out.glob("textures/501*")))
        self.assertFalse([p for p in self.out.rglob("*") if p.name.endswith(".part")])
        reasons = [(f["handle"], f["reason"]) for f in result.coverage["groups"]["tex"]["files"]]
        self.assertIn((F.CUBE, "parse_error"), reasons)

    def test_reader_fault_stops(self):
        with mock.patch.object(E.material, "parse_material", side_effect=RuntimeError("boom")):
            result = self.export()
        self.assertEqual(result.exit_code, 1)
        self.assertIn("boom", result.message)
        self.assertEqual(result.coverage["stopped"]["group"], "mat")
        self.assertEqual(self.read("coverage.json")["stopped"]["handle"], F.MAT)
        self.assertTrue((self.out / "meshes/600.glb").is_file())  # written before the stop

    def test_duplicate_differs(self):
        files = F.switch_files()
        other = bytearray(files["Act1"][F.name_of("tex", "Stone")])
        other[-1] ^= 0xFF
        files["Act1"][F.name_of("tex", "Stone")] = bytes(other)
        F.write_game(self.game, files)
        result = self.export()
        notes = [f for f in result.coverage["groups"]["tex"]["files"] if f["reason"] == "duplicate_differs"]
        self.assertEqual({(n["archive"], n["copy"]) for n in notes}, {("Act1", 0), ("Common", 0)})  # V26-5
        self.assertTrue(all(n["exported"] for n in notes))
        self.assertEqual([e["source"]["archive"] for e in result.manifest["entries"] if e["handle"] == F.TEX], ["Act1"])


class Meshes(Exported):
    """W18-4."""

    def test_round_trip(self):
        self.export()
        for handle, stem in ((F.APP, "Hero"), (F.PLAIN_APP, "Rock")):
            source = self.files["Common"][F.name_of("app", stem)]
            document = I.read_gltf((self.out / f"meshes/{handle}.glb").read_bytes())
            self.assertEqual(I.import_mesh(P.parse_appearance(source), document), source)

    def test_extras_and_textures(self):
        self.export()
        doc = glb_json((self.out / "meshes/600.glb").read_bytes())
        extras = doc["extras"]["d3sno_export"]
        self.assertEqual(extras["handle"], 600)
        self.assertEqual({k: extras[k] for k in ("v_direction", "tangents")},  # ``43`` W19-1
                         {"v_direction": "top_left_origin", "tangents": "not_stored"})
        self.assertEqual(extras["primitives"], [{"mesh": 0, "primitive": 0, "material": 0xAAA, "uv2_scale": "none"}])
        self.assertEqual(doc["images"], [{"uri": "../textures/500.png"}])
        self.assertTrue((self.out / "meshes" / doc["images"][0]["uri"]).resolve().is_file())

    def test_missing_texture(self):
        result = self.export()
        doc = glb_json((self.out / "meshes/601.glb").read_bytes())
        self.assertEqual(doc["images"], [{"uri": f"../textures/{F.MISSING_TEX}.png"}])
        notes = [f for f in result.coverage["groups"]["app"]["files"] if f["handle"] == F.PLAIN_APP]
        self.assertEqual(notes[0]["reason"], "missing_dependency")
        self.assertTrue(notes[0]["exported"])
        self.assertEqual(notes[0]["needs"], [{"group": "tex", "handle": F.MISSING_TEX}])

    def test_glb_structure(self):
        """The checks a glTF 2.0 validator makes first: header, chunk order, alignment, lengths, the
        index ranges of views and accessors."""
        self.export()
        for path in list(self.out.glob("meshes/*.glb")) + list(self.out.glob("anims/*.glb")):
            data = path.read_bytes()
            magic, version, length = struct.unpack_from("<4sII", data)
            self.assertEqual((magic, version, length), (b"glTF", 2, len(data)))
            json_length, json_kind = struct.unpack_from("<II", data, 12)
            self.assertEqual((json_kind, json_length % 4), (0x4E4F534A, 0))
            bin_at = 20 + json_length
            bin_length, bin_kind = struct.unpack_from("<II", data, bin_at)
            self.assertEqual((bin_kind, bin_length % 4, bin_at + 8 + bin_length), (0x004E4942, 0, len(data)))
            doc = glb_json(data)
            self.assertEqual(doc["asset"]["version"], "2.0")
            self.assertNotIn("uri", doc["buffers"][0])
            for view in doc["bufferViews"]:
                self.assertLessEqual(view.get("byteOffset", 0) + view["byteLength"], doc["buffers"][0]["byteLength"])
            for accessor in doc["accessors"]:
                self.assertLess(accessor["bufferView"], len(doc["bufferViews"]))
            I.read_gltf(data)


class Animations(Exported):
    """W18-5, W18-6."""

    def test_linked_animation(self):
        result = self.export()
        doc = glb_json((self.out / "anims/700.glb").read_bytes())
        extras = doc["extras"]["d3sno_export"]
        self.assertEqual((extras["appearance"], extras["root_motion"]), (600, "in_place"))  # ``43`` W19-1
        self.assertEqual([a["name"] for a in doc["animations"]], ["Walk", "Run"])
        self.assertEqual([p["animation"] for p in extras["permutations"]], [0, 1])
        lone = [f for f in result.coverage["groups"]["ani"]["files"] if f["handle"] == F.LONE_ANI]
        self.assertEqual((lone[0]["reason"], lone[0]["exported"]), ("missing_dependency", False))

    def test_without_exported_mesh(self):
        result = self.export(groups="ani")
        reasons = {f["handle"]: f["reason"] for f in result.coverage["groups"]["ani"]["files"]}
        self.assertEqual(reasons, {F.ANI: "missing_dependency", F.LONE_ANI: "missing_dependency"})


class Textures(Exported):
    """W18-7."""

    def test_all_levels(self):
        result = self.export(groups="tex", all_levels=True)
        paths = {e["path"] for e in result.manifest["entries"] if e["handle"] == F.MIP}
        self.assertEqual(paths, {"textures/502.png", "textures/502_l1.png", "textures/502_l2.png"})
        self.assertTrue(result.manifest["options"]["all_levels"])
        tex = T.parse_texture(self.files["Common"][F.name_of("tex", "Mip")])
        from d3sno import png

        for level in range(3):
            name = "textures/502.png" if not level else f"textures/502_l{level}.png"
            image = png.read_png((self.out / name).read_bytes())
            self.assertEqual(image.rgba, T.decode_image(tex, level).rgba)

    def test_cube_faces_in_stored_order(self):
        self.export(groups="tex")
        tex = T.parse_texture(self.files["Common"][F.name_of("tex", "Sky")])
        from d3sno import png

        for face in range(6):
            image = png.read_png((self.out / f"textures/501_f{face}.png").read_bytes())
            self.assertEqual(image.rgba, T.decode_image(tex, 0, face).rgba)

    def test_normal_map_as_stored(self):
        self.export(groups="tex")
        from d3sno import png

        tex = T.parse_texture(self.files["Common"][F.name_of("tex", "Bumps")])
        self.assertEqual(png.read_png((self.out / "textures/503.png").read_bytes()).rgba, T.decode_image(tex).rgba)


class Records(Exported):
    """W18-8."""

    def setUp(self):
        super().setUp()
        self.result = self.export()

    def test_actor(self):
        actor = self.read("data/acr/100.json")
        self.assertEqual(actor["links"], {"appearance": {"group": "app", "handle": 600},
                                          "anim_set": {"group": "ans", "handle": 800},
                                          "collision": {"group": "phm", "handle": 1000}, "monster": None})
        self.assertEqual(actor["actor_type"], {"value": 1, "name": "ACTORTYPE_MONSTER"})
        self.assertEqual(actor["bounds"]["bound_sphere_radius"], 0.5)
        self.assertEqual(set(actor["fields"]["flags"]), {"value", "names"})
        self.assertIsNone(actor["fields"]["anim_tree_ref"])  # -1: no link

    def test_monster(self):
        monster = self.read("data/mon/110.json")
        self.assertEqual(monster["links"]["actor"], {"group": "acr", "handle": 100})
        self.assertIn("Level", monster["attributes"])

    def test_material(self):
        mat = self.read("data/mat/1100.json")
        self.assertEqual([(t["slot"], t["texture"]) for t in mat["textures"]], [
            ({"value": 1, "name": "diffuse"}, {"group": "tex", "handle": 500}),
            ({"value": 3, "name": "bump"}, {"group": "tex", "handle": F.MISSING_TEX})])
        self.assertIn("shader", mat)
        self.assertIn("power", mat["parameters"])

    def test_scene(self):
        scene = self.read("data/scn/1200.json")
        cells = scene["navigation"]["cells"]
        self.assertEqual(len(cells), 3)
        self.assertEqual(cells[0]["flags"], {"value": 0x1001, "names": ["walk", "rounded_corner_1"]})
        self.assertEqual(len(cells[0]["neighbours"]), 2)
        self.assertEqual(scene["marker_sets"], [{"group": "mrk", "handle": 1300}])
        self.assertEqual(scene["collision"], {"group": "phm", "handle": 1000})

    def test_marker_set(self):
        markers = self.read("data/mrk/1300.json")["markers"]
        self.assertEqual([m["link"] for m in markers], [{"group": "acr", "handle": 100}] * 2)
        self.assertEqual(markers[1]["rotation"], {"quaternion": [0.0, 0.0, 0.6, 0.8], "raw": [0.0, 0.0, 0.6, 0.8]})
        self.assertEqual(markers[0]["scale"], 1.0)
        self.assertEqual(markers[1]["position"], [4.0, 5.0, 0.0])

    def test_world_layout(self):
        """The placement of W18-8: every marker at the scene position plus the rotated marker position."""
        layout = self.read("data/wrl/1400.json")["layout"]
        self.assertEqual([m["position"] for m in layout["markers"]], [[101.0, 202.0, 0.0], [104.0, 205.0, 0.0]])
        self.assertEqual([m["scene"] for m in layout["markers"]], [{"group": "scn", "handle": 1200}] * 2)
        self.assertEqual((layout["missing_scenes"], layout["missing_marker_sets"]), ([], []))
        floor = self.read("data/phm/1000.json")["meshes"][0]
        self.assertEqual(floor["vertices"][3], [1.0, 1.0, 0.0])
        self.assertTrue(all(m["position"][2] == floor["vertices"][0][2] for m in layout["markers"]))

    def test_rotation_is_unit(self):
        files = F.switch_files()
        import builders as B

        files["Act1"][F.name_of("mrk", "Room")] = F.with_handle(B.build_mrk("switch", markers=[
            {"type": 0, "group": 1, "handle": F.ACR, "rotation": (0.0, 0.0, 1.2, 1.6)}]), F.MRK, "<")
        F.write_game(self.game, files)
        self.export()
        rotation = self.read("data/mrk/1300.json")["markers"][0]["rotation"]
        self.assertEqual(rotation, {"quaternion": [0.0, 0.0, 0.6, 0.8], "raw": [0.0, 0.0, 1.2, 1.6]})

    def test_collision(self):
        phm = self.read("data/phm/1000.json")
        mesh = phm["meshes"][0]
        self.assertEqual([t["vertices"] for t in mesh["triangles"]], [[0, 1, 2], [1, 3, 2]])
        self.assertEqual(set(mesh["bounds"]), {"centre", "half_extent"})
        self.assertNotIn("nodes", mesh)

    def test_strings_and_level_area(self):
        strings = self.read("data/text/enUS/1500.json")["strings"]
        self.assertEqual(strings, {"00001234": "Hello", "00000099": "Zdanie ż"})
        lvl = self.read("data/lvl/1600.json")
        self.assertEqual((lvl["header"]["handle"], lvl["body"]), (1600, "open"))


class Ps3(unittest.TestCase):
    """The separate ``.can`` and ``.nav`` files of ps3 (W18-5, W18-8)."""

    def test_companions(self):
        root = F.temp_root(self)
        game = F.write_game(root / "game", F.ps3_files(), version=6, order=">")
        result = E.export_game(game, root / "out")
        self.assertEqual(result.exit_code, 0, result.message)
        self.assertEqual(result.manifest["build"], "ps3")
        scene = json.loads((root / "out/data/scn/1200.json").read_text(encoding="utf-8"))
        self.assertTrue(scene["navigation"]["has_sections"])
        self.assertEqual(len(scene["navigation"]["cells"]), 3)
        hall = json.loads((root / "out/data/scn/1201.json").read_text(encoding="utf-8"))
        self.assertFalse(hall["navigation"]["has_sections"])
        notes = {f["handle"]: f["reason"] for f in result.coverage["groups"]["scn"]["files"]}
        self.assertEqual(notes, {1201: "missing_dependency"})
        doc = glb_json((root / "out/anims/700.glb").read_bytes())
        self.assertEqual([a["name"] for a in doc["animations"]], ["Walk"])

    def test_animation_without_can(self):
        root = F.temp_root(self)
        files = F.ps3_files()
        del files["Common"]["Anim\\Hero_walk\\0000.can"]
        game = F.write_game(root / "game", files, version=6, order=">")
        result = E.export_game(game, root / "out")
        reasons = {f["handle"]: f["reason"] for f in result.coverage["groups"]["ani"]["files"]}
        self.assertEqual(reasons, {F.ANI: "missing_dependency"})


class X360(unittest.TestCase):
    """x360: the x360 texture decoder for every texture (W18-7) and the locale of the archive name."""

    def test_texture_and_text(self):
        import builders as B
        from d3sno import png
        from d3sno import texture_write as TW
        from d3sno import x360_texture as XT

        tex = F.with_handle(TW.encode_texture(F.image(8, 8, 2), "x360", 0, levels=1), F.TEX, ">")
        stl = F.with_handle(B.build_stl("x360", [(5, b"Hi")]), F.STL, ">")
        root = F.temp_root(self)
        game = F.write_game(root / "game", {"enUS_Common": {F.name_of("tex", "Stone"): tex,
                                                             F.name_of("stl", "Words"): stl}}, 7, ">")
        result = E.export_game(game, root / "out")
        self.assertEqual((result.exit_code, result.manifest["build"]), (0, "x360"))
        self.assertEqual([e["path"] for e in result.manifest["entries"]], ["data/text/enUS/1500.json", "textures/500.png"])
        image = png.read_png((root / "out/textures/500.png").read_bytes())
        self.assertEqual(image.rgba, XT.decode_image(T.parse_texture(tex)).rgba)


class Command(Exported):
    """W18-1 as a command line."""

    def run_cli(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = cli.main(list(argv))
        return code, out.getvalue(), err.getvalue()

    def test_export(self):
        code, out, _ = self.run_cli("export", str(self.game), "--out", str(self.out), "--groups", "tex,app",
                                    "--archives", "Common*", "--all-levels")
        self.assertEqual(code, 0)
        lines = out.splitlines()
        self.assertEqual(len(lines), len(E.EXPORT_GROUPS) + 1)
        self.assertTrue(lines[-1].startswith("total: "))
        manifest = self.read("manifest.json")
        self.assertEqual(manifest["options"], {"archives": ["Common*"], "groups": ["app", "tex"], "all_levels": True,
                                               "travel": False})
        code, _, _ = self.run_cli("export", str(self.game), "--out", str(self.out), "--skip-existing")
        self.assertEqual(code, 0)

    def test_refusal(self):
        code, out, err = self.run_cli("export", str(self.game), "--out", str(self.game / "x"))
        self.assertEqual((code, out), (2, ""))
        self.assertIn("error:", err)


class FatDrawList(unittest.TestCase):
    """V25-1: a fat draw changed only in the list of draws is refused with the V24-1 label."""

    def test_changed_in_the_list_only(self):
        for data, build, k, _ in fat_cases():
            app = P.parse_appearance(data, build)
            sub = app.sub_objects[k]
            changed = copy.deepcopy(sub.fat_draw)
            changed.positions[0] = (changed.positions[0][0] + 1.0, *changed.positions[0][1:])
            listed = [i for i, d in enumerate(sub.draws) if d.kind == "fat"]
            if listed:
                sub.draws[listed[0]] = changed
            else:
                sub.draws.append(changed)
            with self.assertRaisesRegex(FormatError, rf"^high-poly set, sub-object {k}, fat vertices: the fat draw "
                                                     r"in the list of draws differs .*V25-1", msg=build):
                M.write_app_mesh(app, derive=True)
            with self.assertRaisesRegex(FormatError, "V25-1"):
                M.write_app_mesh(app)

    def test_dropped_from_the_list_only(self):
        seen = 0
        for data, build, k, _ in fat_cases():
            app = P.parse_appearance(data, build)
            sub = app.sub_objects[k]
            if not any(d.kind == "fat" for d in sub.draws):
                continue
            seen += 1
            sub.draws = [d for d in sub.draws if d.kind != "fat"]
            with self.assertRaisesRegex(FormatError, "fat vertices: .*V25-1"):
                M.write_app_mesh(app, derive=True)
        self.assertGreater(seen, 0)

    def test_equal_copy_accepted(self):
        for data, build, k, _ in fat_cases():
            app = P.parse_appearance(data, build)
            sub = app.sub_objects[k]
            sub.draws = [copy.deepcopy(d) if d.kind == "fat" else d for d in sub.draws]
            self.assertEqual(M.write_app_mesh(app, derive=True), data)


if __name__ == "__main__":
    unittest.main()
