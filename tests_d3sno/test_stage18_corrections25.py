"""Corrections 25 (``41_corrections25.md``): V26-1 to V26-5 and V26-8. Synthetic data only; every
export goes to a temporary folder outside any git working tree."""

import json
import os
import struct
import unittest

import builders as B
import d3sno
import export_fixture as F
import test_stage17_writer as W
from d3sno import animation as N
from d3sno import appearance as P
from d3sno import archive as A
from d3sno import export as E
from d3sno import mesh_import as I
from test_stage18_export import glb_json

FLT_MAX = struct.unpack("<f", b"\xff\xff\x7f\x7f")[0]
TWO_ROOTS = [(0xAA, -1, (1.0, 2.0, 3.0)), (0xBB, 0, (2.0, 2.0, 3.0)), (0xCC, -1, (2.0, 4.0, 3.0))]


def two_root_mesh() -> bytes:
    return W.canon(W.laid_out(B.build_app_switch(version=306, bones=TWO_ROOTS, pool_spare=0,
                                                 sub_objects=[{"skin": W.SKIN2, "index_count": 0}]), None), None)


def common_roots(doc: dict, skin: dict) -> set[int]:
    """The glTF validator's rule: walking up from every joint must reach one common root node."""
    parent = {}
    for index_, node in enumerate(doc["nodes"]):
        for child in node.get("children", []):
            parent[child] = index_
    roots = set()
    for joint in skin["joints"]:
        while joint in parent:
            joint = parent[joint]
        roots.add(joint)
    return roots


class Version(unittest.TestCase):
    def test_version(self):
        self.assertGreaterEqual(tuple(int(x) for x in d3sno.__version__.split(".")), (0, 18, 1))
        self.assertGreaterEqual(d3sno.STAGE, 18)


class OneChannelPerTarget(unittest.TestCase):
    """V26-1: a repeated bone hash gives one channel per (node, path)."""

    def test_library(self):
        app = P.parse_appearance(F.textured_mesh())
        ani = N.parse_ani(B.build_ani("switch", [B.simple_permutation("switch", hashes=[0xAA, 0xAA])]), "switch")
        doc = N.to_gltf_animated(app, ani)
        targets = [(c["target"]["node"], c["target"]["path"]) for c in doc["animations"][0]["channels"]]
        self.assertEqual(len(targets), len(set(targets)))
        self.assertEqual(len(targets), 3)
        self.assertEqual(doc["extras"]["d3sno_export"]["dropped_tracks"], [{"track": 1, "bone_hash": 0xAA}])

    def test_no_repeat_unchanged(self):
        app = P.parse_appearance(F.textured_mesh())
        ani = N.parse_ani(B.build_ani("switch", [B.simple_permutation("switch")]), "switch")
        self.assertNotIn("extras", N.to_gltf_animated(app, ani))

    def test_export(self):
        root = F.temp_root(self)
        files = F.switch_files()
        files["Act1"][F.name_of("ani", "Hero_walk")] = F.with_handle(B.build_ani("switch", [
            B.simple_permutation("switch"), B.simple_permutation("switch", name="Run", hashes=[0xBB, 0xBB])]),
            F.ANI, "<")
        game = F.write_game(root / "game", files)
        result = E.export_game(game, root / "out", created="X")
        self.assertEqual(result.exit_code, 0, result.message)
        doc = glb_json((root / "out/anims/700.glb").read_bytes())
        for anim in doc["animations"]:
            targets = [(c["target"]["node"], c["target"]["path"]) for c in anim["channels"]]
            self.assertEqual(len(targets), len(set(targets)))
        self.assertEqual(doc["extras"]["d3sno_export"]["dropped_tracks"],
                         [{"permutation": 1, "track": 1, "bone_hash": 0xBB}])


class SkinRoot(unittest.TestCase):
    """V26-2: one added node above several root bones; the round trip still holds."""

    def test_two_roots(self):
        data = two_root_mesh()
        doc = P.to_gltf(P.parse_appearance(data))
        skin = doc["skins"][0]
        roots = common_roots(doc, skin)
        self.assertEqual(len(roots), 1)
        (added,) = roots
        node = doc["nodes"][added]
        self.assertEqual(node["name"], "d3sno_skin_root")
        self.assertEqual(node["extras"], {"d3sno_export": {"added": True}})
        self.assertFalse({"translation", "rotation", "scale", "matrix", "mesh", "skin"} & set(node))
        self.assertNotIn(added, skin["joints"])
        self.assertEqual(skin["skeleton"], added)
        self.assertEqual(sorted(node["children"]), [0, 2])
        self.assertIn(added, doc["scenes"][0]["nodes"])
        self.assertNotIn(0, doc["scenes"][0]["nodes"])

    def test_round_trip(self):
        data = two_root_mesh()
        glb = I.gltf_to_glb(P.to_gltf(P.parse_appearance(data)))
        self.assertEqual(I.import_mesh(P.parse_appearance(data), I.read_gltf(glb)), data)

    def test_one_root_unchanged(self):
        doc = P.to_gltf(P.parse_appearance(F.textured_mesh()))
        self.assertNotIn("skeleton", doc["skins"][0])
        self.assertNotIn("d3sno_skin_root", [n.get("name") for n in doc["nodes"]])
        self.assertEqual(len(common_roots(doc, doc["skins"][0])), 1)

    def test_unskinned_not_added(self):
        app = P.parse_appearance(two_root_mesh())
        nodes, roots, added = P.skeleton_nodes(app, skinned=False)
        self.assertEqual((roots, added), ([0, 2], None))


class AnimationWithoutMesh(unittest.TestCase):
    """V26-3: an animation file holds the joint nodes and the animations only."""

    def export(self, files):
        root = F.temp_root(self)
        game = F.write_game(root / "game", files)
        result = E.export_game(game, root / "out", created="X")
        self.assertEqual(result.exit_code, 0, result.message)
        return root / "out"

    def check_binding(self, out):
        anim = glb_json((out / "anims/700.glb").read_bytes())
        self.assertFalse({"meshes", "skins", "images", "materials", "textures", "samplers"} & set(anim))
        extras = anim["extras"]["d3sno_export"]
        self.assertEqual(extras["mesh"], "meshes/600.glb")
        mesh = glb_json((out / extras["mesh"]).read_bytes())
        joints = [mesh["nodes"][j]["name"] for j in mesh["skins"][0]["joints"]]
        self.assertEqual([n["name"] for n in anim["nodes"]][:len(joints)], joints)
        by_name = {n["name"]: i for i, n in enumerate(mesh["nodes"])}
        for a in anim["animations"]:
            for channel in a["channels"]:
                name = anim["nodes"][channel["target"]["node"]]["name"]
                self.assertIn(by_name[name], mesh["skins"][0]["joints"])  # binds to a joint of the mesh
        return anim, mesh

    def test_bind_by_name(self):
        anim, _ = self.check_binding(self.export(F.switch_files()))
        self.assertEqual([a["name"] for a in anim["animations"]], ["Walk", "Run"])
        self.assertEqual(len(anim["nodes"]), 3)

    def test_with_skin_root(self):
        files = F.switch_files()
        mesh = W.with_materials(two_root_mesh(), None, looks=(0x61,), entries=[(0xAAA, [[(1, F.TEX)]])],
                                assign=(0xAAA,))
        files["Common"][F.name_of("app", "Hero")] = F.with_handle(mesh, F.APP, "<")
        anim, mesh_doc = self.check_binding(self.export(files))
        self.assertEqual([n["name"] for n in anim["nodes"]], [n["name"] for n in mesh_doc["nodes"][:4]])
        self.assertEqual(anim["nodes"][3]["name"], "d3sno_skin_root")
        self.assertEqual(anim["scenes"][0]["nodes"], [3])

    def test_direct_call_keeps_mesh(self):
        app = P.parse_appearance(F.textured_mesh())
        ani = N.parse_ani(B.build_ani("switch", [B.simple_permutation("switch")]), "switch")
        doc = N.to_gltf_animated(app, ani)
        self.assertIn("meshes", doc)
        self.assertIn("skins", doc)


class Float32Maximum(unittest.TestCase):
    """V26-4: ±FLT_MAX is written as its shortest text."""

    def test_text(self):
        self.assertEqual(E.f32_text(FLT_MAX), "3.4028235e+38")
        self.assertEqual(E.f32_text(-FLT_MAX), "-3.4028235e+38")
        self.assertEqual(struct.pack("<f", float(E.f32_text(FLT_MAX))), b"\xff\xff\x7f\x7f")

    def test_json(self):
        self.assertEqual(json.loads(E.dumps({"v": [FLT_MAX, -FLT_MAX]})), {"v": [3.4028235e38, -3.4028235e38]})
        self.assertIn(b"3.4028235e+38", E.dumps({"v": FLT_MAX}))


class CopiesInOneEntry(unittest.TestCase):
    """V26-5: every copy of an archive entry is read; copy 0 is exported."""

    def export(self, copies):
        root = F.temp_root(self)
        files = F.ps3_files()["Common"]
        name = F.name_of("acr", "Hero")
        inputs = [A.InputFile(n, copies(d) if n == name else [d]) for n, d in files.items()]
        game = root / "game"
        game.mkdir()
        (game / "Common.cpk").write_bytes(A.write_archive(inputs, version=6, order=">"))
        result = E.export_game(game, root / "out", created="X")
        self.assertEqual(result.exit_code, 0, result.message)
        return result, files[name], root / "out"

    def test_one_copy_differs(self):
        def copies(data):
            other = bytearray(data)
            other[8] ^= 1  # the revision word of the SNO header (``01``: copies may differ in bytes 8 to 15)
            return [data, data, bytes(other)]

        result, first, out = self.export(copies)
        cov = result.coverage["groups"]["acr"]
        self.assertEqual((cov["found"], cov["exported"]), (1, 1))
        notes = [f for f in cov["files"] if f["reason"] == "duplicate_differs"]
        self.assertEqual([(n["archive"], n["copy"]) for n in notes], [("Common", 0), ("Common", 1), ("Common", 2)])
        self.assertTrue(all(n["exported"] for n in notes))
        self.assertEqual(len({n["copy_hash"] for n in notes}), 2)
        entry = [e for e in result.manifest["entries"] if e["group"] == "acr"][0]
        import hashlib
        self.assertEqual(entry["source"]["hash"], hashlib.sha256(first).hexdigest())  # copy 0

    def test_equal_copies_no_row(self):
        result, _, _ = self.export(lambda data: [data, data, data])
        self.assertEqual(result.coverage["groups"]["acr"]["files"], [])
        self.assertEqual(result.coverage["groups"]["acr"]["found"], 1)


class PackageTables(unittest.TestCase):
    """V26-8: the tests read the record tables from the package data."""

    def test_tables_in_package(self):
        package = os.path.join(os.path.dirname(d3sno.__file__), "data")
        B.need_layout_tables()  # V27-3
        for path in B.LAYOUT_TABLES:
            self.assertEqual(os.path.dirname(path), package)
            self.assertTrue(os.path.isfile(path))

    def test_missing_table_skips(self):
        root = F.temp_root(self)
        missing = str(root / "layouts.tsv")
        layouts = B._layout_tables((missing,))
        with self.assertRaises(unittest.SkipTest) as caught:
            layouts[("switch", "actor")]
        self.assertIn(missing, str(caught.exception))


if __name__ == "__main__":
    unittest.main()
