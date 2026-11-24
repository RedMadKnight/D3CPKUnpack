"""Corrections 26 (``42_corrections26.md``): V27-1 to V27-3. Synthetic data only; every export goes to a
temporary folder outside any git working tree."""

import _libpath  # noqa: F401  (python -I: the library folder on sys.path)
import hashlib
import unittest

import builders as B
import d3sno
import export_fixture as F
import test_stage17_writer as W
from d3sno import appearance as P
from d3sno import export as E
from d3sno import texture_write as TW
from test_stage18_export import glb_json

OTHER_ACR = 101
LANG, LANG2, LANG3 = 510, 511, 512


def boneless_mesh() -> bytes:
    return W.canon(W.laid_out(B.build_app_switch(version=306, pool_spare=0, sub_objects=[
        {"rigid": [(0, 0, 0), (10, 0, 0), (0, 10, 0)], "index_count": 3}]), None), None)


def actor(handle: int, app: int) -> bytes:
    return F.with_handle(B.build_record("switch", "actor", {
        "self_handle": handle, "appearance_ref": app, "anim_set_ref": F.ANS, "physics_mesh_ref": -1,
        "monster_ref": -1, "anim_tree_ref": -1, "actor_kind": 1, "bound_sphere_radius": 0.5}), handle, "<")


def empty_arrays(value, path="") -> list[str]:
    if isinstance(value, dict):
        return [p for k, v in value.items() for p in empty_arrays(v, f"{path}.{k}")]
    if isinstance(value, list):
        return [path] if not value else [p for i, v in enumerate(value) for p in empty_arrays(v, f"{path}[{i}]")]
    return []


def export(case, files):
    root = F.temp_root(case)
    game = F.write_game(root / "game", files)
    result = E.export_game(game, root / "out", created="X")
    case.assertEqual(result.exit_code, 0, result.message)
    return result, root / "out"


class Version(unittest.TestCase):
    def test_version(self):
        self.assertGreaterEqual(tuple(int(x) for x in d3sno.__version__.split(".")), (0, 18, 2))
        self.assertGreaterEqual(d3sno.STAGE, 18)


class MeshWithoutBones(unittest.TestCase):
    """V27-1: the skeleton comes from the first linked appearance that has a bone."""

    def test_only_boneless_linked(self):
        files = F.switch_files()
        files["Common"][F.name_of("app", "Hero")] = F.with_handle(boneless_mesh(), F.APP, "<")
        self.assertEqual(P.parse_appearance(files["Common"][F.name_of("app", "Hero")]).bones, [])
        result, out = export(self, files)
        notes = {f["handle"]: f for f in result.coverage["groups"]["ani"]["files"]}
        self.assertEqual(notes[F.ANI]["reason"], "missing_dependency")
        self.assertFalse(notes[F.ANI]["exported"])
        self.assertIn("no linked appearance has a skeleton", notes[F.ANI]["message"])
        self.assertFalse((out / "anims/700.glb").exists())
        self.assertTrue((out / "meshes/600.glb").is_file())  # the mesh itself is still exported
        self.assertEqual(empty_arrays(glb_json((out / "meshes/600.glb").read_bytes())), [])

    def test_next_appearance_with_bones(self):
        files = F.switch_files()
        files["Common"][F.name_of("app", "Hero")] = F.with_handle(boneless_mesh(), F.APP, "<")
        files["Common"][F.name_of("acr", "Other")] = actor(OTHER_ACR, F.PLAIN_APP)
        result, out = export(self, files)
        doc = glb_json((out / "anims/700.glb").read_bytes())
        self.assertEqual(doc["extras"]["d3sno_export"]["appearance"], F.PLAIN_APP)
        self.assertEqual(doc["extras"]["d3sno_export"]["mesh"], f"meshes/{F.PLAIN_APP}.glb")
        self.assertEqual([a["name"] for a in doc["animations"]], ["Walk", "Run"])

    def test_lowest_handle_with_bones_kept(self):
        files = F.switch_files()
        files["Common"][F.name_of("acr", "Other")] = actor(OTHER_ACR, F.PLAIN_APP)
        _, out = export(self, files)
        self.assertEqual(glb_json((out / "anims/700.glb").read_bytes())["extras"]["d3sno_export"]["appearance"], F.APP)

    def test_no_empty_array(self):
        _, out = export(self, F.switch_files())
        for path in sorted(out.rglob("*.glb")):
            self.assertEqual(empty_arrays(glb_json(path.read_bytes())), [], path.name)
        self.assertNotIn("dropped_tracks", glb_json((out / "anims/700.glb").read_bytes())["extras"]["d3sno_export"])

    def test_prune(self):
        self.assertEqual(E.prune_empty({"a": [], "b": [{"c": [], "d": 1}], "e": {"f": []}, "g": [[]]}),
                         {"b": [{"d": 1}], "e": {}, "g": [[]]})


class LocaleVariants(unittest.TestCase):
    """V27-2: of entries that differ only by ``|locale``: none, then ``enUS``, then sorted."""

    def files(self):
        files = F.switch_files()
        self.data = {}
        for handle, stem, locales in ((LANG, "Lang", ("deDE", "frFR", "enUS")), (LANG2, "Lang2", ("frFR", "deDE")),
                                      (LANG3, "Lang3", ("enUS", None))):
            for k, locale in enumerate(locales):
                data = F.with_handle(TW.encode_texture(F.image(4, 4, handle + k), "switch", 0, levels=1), handle, "<")
                self.data[(handle, locale)] = data
                files["Common"][F.name_of("tex", stem, locale)] = data
        return files

    def test_choice(self):
        result, out = export(self, self.files())
        entries = {e["handle"]: e for e in result.manifest["entries"] if e["group"] == "tex"}
        for handle, locale in ((LANG, "enUS"), (LANG2, "deDE"), (LANG3, None)):
            self.assertEqual(entries[handle]["locale"], locale)
            self.assertEqual(entries[handle]["source"]["hash"], hashlib.sha256(self.data[(handle, locale)]).hexdigest())
        self.assertIsNone(entries[F.TEX]["locale"])
        self.assertEqual(sorted(p.name for p in (out / "textures").glob("51*.png")), ["510.png", "511.png", "512.png"])

    def test_rows(self):
        result, _ = export(self, self.files())
        rows = [f for f in result.coverage["groups"]["tex"]["files"] if f["reason"] == "duplicate_differs"]
        by_handle = {}
        for row in rows:
            by_handle.setdefault(row["handle"], set()).add(row["locale"])
        self.assertEqual(by_handle[LANG], {"deDE", "frFR", "enUS"})
        self.assertEqual(by_handle[LANG2], {"frFR", "deDE"})
        self.assertEqual(by_handle[LANG3], {"enUS", None})
        self.assertTrue(all("locale" in r for r in rows))
        cov = result.coverage["groups"]["tex"]
        self.assertEqual(cov["unique"], cov["exported"] + cov["skipped"])

    def test_string_tables_keep_their_locale(self):
        result, _ = export(self, F.switch_files())
        (entry,) = [e for e in result.manifest["entries"] if e["group"] == "stl"]
        self.assertEqual(entry["locale"], "enUS")


    def test_archive_prefix_locale(self):
        """A string table without ``|locale`` keeps the locale of its archive name (Q853), also on
        its ``duplicate_differs`` rows; a texture there has none."""
        tex = F.with_handle(TW.encode_texture(F.image(8, 8, 2), "x360", 0, levels=1), F.TEX, ">")
        stl = F.with_handle(B.build_stl("x360", [(5, b"Hi")]), F.STL, ">")
        other = F.with_handle(B.build_stl("x360", [(5, b"Ho")]), F.STL, ">")
        root = F.temp_root(self)
        name = F.name_of("stl", "Words")
        game = F.write_game(root / "game", {"enUS_Common": {F.name_of("tex", "Stone"): tex, name: stl},
                                            "enUS_Act1": {name: other}}, 7, ">")
        result = E.export_game(game, root / "out")
        self.assertEqual(result.exit_code, 0, result.message)
        locales = {e["group"]: e["locale"] for e in result.manifest["entries"]}
        self.assertEqual(locales, {"stl": "enUS", "tex": None})
        rows = [f for f in result.coverage["groups"]["stl"]["files"] if f["reason"] == "duplicate_differs"]
        self.assertEqual(sorted((r["archive"], r["locale"]) for r in rows), [("enUS_Act1", "enUS"), ("enUS_Common", "enUS")])


class PruneMesh(unittest.TestCase):
    def test_mesh_file_pruned(self):
        from unittest import mock

        app = P.parse_appearance(boneless_mesh())
        bare = {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": []}], "meshes": []}
        with mock.patch.object(P, "to_gltf", return_value=bare):
            doc = glb_json(E._mesh_glb(app, 1))
        self.assertEqual(empty_arrays(doc), [])
        self.assertEqual(doc["scenes"], [{}])


class TablesHidden(unittest.TestCase):
    """V27-3: with a layout table missing, the library's layout loader skips the test."""

    def test_loader_skips(self):
        from d3sno import records

        missing = ("C:/nowhere/layouts.tsv",)
        saved = (B.MISSING_LAYOUT_TABLES, records.layouts)
        try:
            B.MISSING_LAYOUT_TABLES = missing
            records.layouts = B._skip_record_layouts
            with self.assertRaises(unittest.SkipTest) as caught:
                records.layout("switch", "actor")
            self.assertIn("layouts.tsv", str(caught.exception))
            self.assertIn("V26-8", str(caught.exception))
        finally:
            B.MISSING_LAYOUT_TABLES, records.layouts = saved
        if not B.MISSING_LAYOUT_TABLES:  # restored: the loader reads the tables again
            self.assertGreater(records.layout("switch", "actor").size, 0)


if __name__ == "__main__":
    unittest.main()
