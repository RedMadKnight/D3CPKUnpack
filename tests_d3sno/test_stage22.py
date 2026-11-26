"""Stage 22 (``49_stage22_json_import.md``): W22-1 to W22-12. Synthetic games of the test builders only;
every export and output goes to a temporary folder outside any git working tree."""

import _libpath  # noqa: F401  (python -I: the library folder on sys.path)
import copy
import json
import math
import struct
import unittest
from unittest import mock

import export_fixture as F
import test_stage21 as S
import d3sno
from d3sno import anim_import as AI
from d3sno import anim_write as AW
from d3sno import animation
from d3sno import appearance
from d3sno import apply as AP
from d3sno import archive as A
from d3sno import collision
from d3sno import collision_write as CW
from d3sno import export as E
from d3sno import json_import as J
from d3sno import mesh_import as I
from d3sno import navigation
from d3sno import records
from d3sno import sno_layout as L
from d3sno import world

BUILD = "switch"
FILES = F.switch_files()
BY_EXT = {}
for _pack in FILES.values():
    for _name, _data in _pack.items():
        _ext = _name.rpartition(".")[2].split("|")[0].lower()
        if _ext in J.GROUPS:
            BY_EXT.setdefault(_ext, _data)


def template(ext: str, canonical: bool = False) -> bytes:
    data = BY_EXT[ext]
    return L.rebuild_sno(data, ext, BUILD) if canonical else data


def exported(ext: str, data: bytes, **kw) -> dict:
    return J._json_round(J.template_json(ext, data, BUILD, **kw))


def run(ext: str, data: bytes, edit, **kw):
    """(result, edited JSON): the import of ``edit(json)`` onto ``data``."""
    document = copy.deepcopy(exported(ext, data))
    edit(document)
    return J.import_json(ext, data, BUILD, document, **kw), document


def differing(a: bytes, b: bytes) -> list:
    return [i for i in range(min(len(a), len(b))) if a[i] != b[i]] + list(range(min(len(a), len(b)), max(len(a), len(b))))


def leaves(document, path=()):
    if isinstance(document, dict):
        for key in document:
            yield from leaves(document[key], path + (key,))
    elif isinstance(document, list):
        for i, value in enumerate(document):
            yield from leaves(value, path + (i,))
    else:
        yield path


class Version(unittest.TestCase):
    def test_version(self):
        self.assertGreaterEqual(tuple(int(x) for x in d3sno.__version__.split(".")), (0, 22, 0))
        self.assertGreaterEqual(d3sno.STAGE, 22)


class Places(unittest.TestCase):
    """W22-2: every leaf of every export is stored, fixed or derived (the place builders and the exporters
    do not drift apart)."""

    def test_classified(self):
        for ext in ("acr", "mon", "mat", "lvl", "scn", "wrl", "mrk"):
            with self.subTest(ext):
                data = template(ext)
                found = J.places(ext, data, BUILD)
                unknown = [J.path_text(p) for p in leaves(exported(ext, data)) if J.classify(ext, p, found) == "unknown"]
                self.assertEqual(unknown, [])
                self.assertTrue(any(J.classify(ext, p, found) == "stored" for p in leaves(exported(ext, data)))
                                or ext == "lvl")

    def test_ps3_scene(self):
        files = F.ps3_files()["Common"]
        scn, nav = files[F.name_of("scn", "Room")], files["Scene\\Room\\0000.nav"]
        found = J.places("scn", scn, "ps3", nav)
        document = J._json_round(J.template_json("scn", scn, "ps3", nav))
        self.assertEqual([p for p in leaves(document) if J.classify("scn", p, found) == "unknown"], [])

    def test_fixed_and_count(self):
        data = template("mat")
        found = J.places("mat", data, BUILD)
        self.assertEqual(J.classify("mat", ("flags",), found), "fixed")
        self.assertEqual(J.classify("mat", ("textures", 0, "slot", "value"), found), "fixed")
        self.assertEqual(J.classify("mat", ("textures", 0, "texture", "handle"), found), "stored")
        self.assertEqual(J.classify("mat", ("header", "handle"), found), "derived")

    def test_path_text(self):
        self.assertEqual(J.path_text(("a", "b", 3, "c")), "a.b[3].c")
        self.assertEqual(J.path_text(("markers", 0, "position", 2)), "markers[0].position[2]")


class Unedited(unittest.TestCase):
    """W22-1: an unedited export imported onto its template gives the template's bytes, forced through the
    import (not short-cut by a hash)."""

    def test_each_group(self):
        for ext in J.GROUPS:
            with self.subTest(ext):
                data = template(ext)
                result = J.import_json(ext, data, BUILD, exported(ext, data))
                self.assertEqual(result.data, data)
                self.assertEqual(result.changes, [])

    def test_ps3_scene_with_nav(self):
        files = F.ps3_files()["Common"]
        scn, nav = files[F.name_of("scn", "Room")], files["Scene\\Room\\0000.nav"]
        result = J.import_json("scn", scn, "ps3", J._json_round(J.template_json("scn", scn, "ps3", nav)), nav=nav)
        self.assertEqual((result.data, result.nav, result.changes), (scn, None, []))

    def test_animation(self):
        for build, files in (("switch", F.switch_files()), ("ps3", F.ps3_files())):
            with self.subTest(build):
                ani, app, can = animation_files(build, files)
                result = AI.import_animation(ani, build, app, build, animation_glb(ani, app, build, can), F.ANI, F.APP, can)
                self.assertEqual((result.data, result.nav, result.changes), (ani, None, []))


class OneEdit(unittest.TestCase):
    """W22-3: one edit of each kind changes exactly its field; parsing gives the new value."""

    def check(self, ext, data, edit, at, size, **kw):
        result, document = run(ext, data, edit, **kw)
        self.assertTrue(differing(data, result.data), "nothing changed")
        self.assertLessEqual(set(differing(data, result.data)), set(range(at, at + size)))
        self.assertEqual(len(result.changes), 1)
        return result, exported(ext, result.data)

    def place(self, ext, data, path):
        return J.places(ext, data, BUILD)[path]

    def test_number(self):
        data = template("acr")
        at = self.place("acr", data, ("fields", "bound_sphere_radius")).at
        _, again = self.check("acr", data, lambda d: d["fields"].__setitem__("bound_sphere_radius", 2.25), at, 4)
        self.assertEqual(again["fields"]["bound_sphere_radius"], 2.25)
        self.assertEqual(again["bounds"]["bound_sphere_radius"], 2.25)  # the same stored field

    def test_float_nearest(self):
        data = template("acr")
        result, _ = run("acr", data, lambda d: d["fields"].__setitem__("bound_sphere_radius", 0.1))
        value = records.parse_record_file(result.data, "acr", BUILD).record["bound_sphere_radius"]
        self.assertEqual(value, struct.unpack("<f", struct.pack("<f", 0.1))[0])

    def test_enum_value_and_name(self):
        data = template("acr")
        place = self.place("acr", data, ("fields", "actor_kind"))
        name, value = next((n, v) for n, v in sorted(place.names.items()) if v not in (1,))
        _, again = self.check("acr", data, lambda d: d["fields"]["actor_kind"].__setitem__("value", value), place.at, 4)
        self.assertEqual(again["fields"]["actor_kind"], {"name": name, "value": value})
        _, again = self.check("acr", data, lambda d: d["fields"]["actor_kind"].__setitem__("name", name), place.at, 4)
        self.assertEqual(again["fields"]["actor_kind"]["value"], value)

    def test_flag_names(self):
        data = template("acr")
        place = self.place("acr", data, ("fields", "physics_flags"))
        name, bit = sorted(place.names.items(), key=lambda kv: kv[1])[0]

        def edit(d):
            d["fields"]["physics_flags"]["names"] = d["fields"]["physics_flags"]["names"] + [name]

        _, again = self.check("acr", data, edit, place.at, 4)
        self.assertIn(name, again["fields"]["physics_flags"]["names"])
        self.assertTrue(again["fields"]["physics_flags"]["value"] >> bit & 1)

    def test_link_and_null_link(self):
        data = template("acr")
        at = self.place("acr", data, ("fields", "appearance_ref")).at
        _, again = self.check("acr", data, lambda d: d["fields"]["appearance_ref"].__setitem__("handle", F.PLAIN_APP), at, 4)
        self.assertEqual(again["links"]["appearance"], {"group": "app", "handle": F.PLAIN_APP})
        result, again = self.check("acr", data, lambda d: d["fields"].__setitem__("appearance_ref", None), at, 4)
        self.assertIsNone(again["fields"]["appearance_ref"])
        self.assertEqual(struct.unpack_from("<i", result.data, at)[0], -1)

    def test_quaternion_only(self):
        data = template("mrk")
        at = self.place("mrk", data, ("markers", 1, "rotation")).at
        q = [0.0, 0.0, -0.8, -0.6]  # within 90 degrees of the stored (0, 0, 0.6, 0.8) after the sign: kept stored sign
        result, again = self.check("mrk", data, lambda d: d["markers"][1]["rotation"].__setitem__("quaternion", [0, 0, 0.7071068, 0.7071068]), at, 16)
        raw = struct.unpack_from("<4f", result.data, at)
        self.assertAlmostEqual(raw[2], 0.7071068, places=6)
        self.assertGreater(raw[3], 0)
        result, _ = run("mrk", data, lambda d: d["markers"][1]["rotation"].__setitem__("quaternion", q))
        raw = struct.unpack_from("<4f", result.data, at)
        self.assertGreater(raw[3], 0)  # a small edit given in the other sign keeps the stored sign
        self.assertAlmostEqual(raw[2], 0.8, places=6)

    def test_raw_only(self):
        data = template("mrk")
        at = self.place("mrk", data, ("markers", 1, "rotation")).at
        result, again = self.check("mrk", data, lambda d: d["markers"][1]["rotation"].__setitem__("raw", [0.0, 0.0, 0.0, 2.0]), at, 16)
        self.assertEqual(struct.unpack_from("<4f", result.data, at), (0.0, 0.0, 0.0, 2.0))
        self.assertEqual(again["markers"][1]["rotation"]["quaternion"], [0.0, 0.0, 0.0, 1.0])

    def test_string(self):
        data = template("mon")
        place = self.place("mon", data, ("fields", "internal_name"))
        _, again = self.check("mon", data, lambda d: d["fields"].__setitem__("internal_name", "Gargoyle"), place.at, place.size)
        self.assertEqual(again["fields"]["internal_name"], "Gargoyle")

    def test_raw_bytes(self):
        data = template("wrl")
        place = self.place("wrl", data, ("server", "clusters_raw"))
        new = bytes(range(1, place.size + 1)).hex()
        _, again = self.check("wrl", data, lambda d: d["server"]["clusters_raw"].__setitem__("bytes", new), place.at, place.size)
        self.assertEqual(again["server"]["clusters_raw"]["bytes"], new)

    def test_text_pair(self):
        data = template("acr", canonical=True)
        result, _ = run("acr", data, lambda d: d["fields"].__setitem__("internal_description", "A guard"))
        self.assertEqual(exported("acr", result.data)["fields"]["internal_description"], "A guard")

    def test_tag_map(self):
        data = template("acr", canonical=True)

        def edit(d):
            d["fields"]["tag_map"]["records"].append({"tag": 0x10007, "type": {"value": 1, "name": "float"}, "value": 1.5})

        result, _ = run("acr", data, edit)
        rows = exported("acr", result.data)["fields"]["tag_map"]["records"]
        self.assertEqual([(r["tag"], r["value"]) for r in rows][-1], (0x10007, 1.5))

    def test_dangling_link(self):
        data = template("mon")
        at = self.place("mon", data, ("fields", "actor_ref")).at
        result, _ = self.check("mon", data, lambda d: d["fields"]["actor_ref"].__setitem__("handle", 4242), at, 4,
                               known=lambda group, handle: False)
        self.assertEqual(struct.unpack_from("<i", result.data, at)[0], 4242)
        self.assertTrue(any(w.startswith("dangling_link") for w in result.warnings))

    def test_material(self):
        data = template("mat")
        place = self.place("mat", data, ("textures", 0, "texture"))
        _, again = self.check("mat", data, lambda d: d["textures"][0]["texture"].__setitem__("handle", F.MIP), place.at, 4)
        self.assertEqual(again["textures"][0]["texture"]["handle"], F.MIP)
        result, _ = run("mat", data, lambda d: d["parameters"].__setitem__("power", 7.0))
        self.assertEqual(exported("mat", result.data)["parameters"]["power"], 7.0)
        shader = self.place("mat", data, ("shader",))
        result, _ = run("mat", data, lambda d: d["shader"].__setitem__("handle", 78), known=lambda g, h: True)
        self.assertEqual(struct.unpack_from("<i", result.data, shader.at)[0], 78)


class Lists(unittest.TestCase):
    """W22-4: markers of a ``mrk`` and scene entries of a ``wrl``."""

    def test_add_and_remove_marker(self):
        data = template("mrk", canonical=True)

        def add(d):
            new = copy.deepcopy(d["markers"][0])
            new["position"] = [9.0, 8.0, 7.0]
            d["markers"].insert(1, new)

        result, _ = run("mrk", data, add)
        parsed = world.parse_marker_set(result.data, BUILD)
        self.assertEqual([m.position for m in parsed.markers], [(1.0, 2.0, 0.0), (9.0, 8.0, 7.0), (4.0, 5.0, 0.0)])
        self.assertEqual(L.rebuild_sno(result.data, "mrk", BUILD), result.data)
        result, _ = run("mrk", data, lambda d: d["markers"].pop(0))
        parsed = world.parse_marker_set(result.data, BUILD)
        self.assertEqual([m.position for m in parsed.markers], [(4.0, 5.0, 0.0)])
        self.assertEqual(parsed.markers[0].rotation, struct.unpack("<4f", struct.pack("<4f", 0.0, 0.0, 0.6, 0.8)))

    def test_add_and_remove_scene_entry(self):
        data = template("wrl", canonical=True)

        def add(d):
            new = copy.deepcopy(d["scene_entries"][0])
            new["position"] = [7.0, 8.0, 9.0]
            new["scene"]["handle"] = F.SCN + 1
            d["scene_entries"].append(new)

        result, _ = run("wrl", data, add)
        parsed = world.parse_world(result.data, BUILD)
        self.assertEqual([(e.handle, e.position) for e in parsed.scene_entries],
                         [(F.SCN, (100.0, 200.0, 0.0)), (F.SCN + 1, (7.0, 8.0, 9.0))])
        lists = parsed.server.scene_lists
        self.assertEqual([len(x) for x in lists], [2])
        result, _ = run("wrl", result.data, lambda d: d["scene_entries"].pop(0))
        self.assertEqual([e.handle for e in world.parse_world(result.data, BUILD).scene_entries], [F.SCN + 1])


class Refusals(unittest.TestCase):
    """W22-3 to W22-10: one test for each reason."""

    def refused(self, ext, data, edit, reason, **kw):
        with self.assertRaises(J.Refused) as caught:
            run(ext, data, edit, **kw)
        self.assertEqual(caught.exception.reason, reason, [r.as_row() for r in caught.exception.refusals])
        return caught.exception

    def test_value_out_of_range(self):
        data = template("acr")
        self.refused("acr", data, lambda d: d["fields"].__setitem__("bound_sphere_radius", 1e60), "value_out_of_range")
        found = J.places("acr", data, BUILD)
        name = next(p[1] for p, v in found.items() if v.kind == "num" and v.code == "i" and len(p) == 2)
        self.refused("acr", data, lambda d: d["fields"].__setitem__(name, 2 ** 40), "value_out_of_range")
        self.refused("acr", data, lambda d: d["fields"].__setitem__(name, 1.5), "value_out_of_range")
        error = self.refused("mon", template("mon"), lambda d: d["fields"].__setitem__("internal_name", "x" * 129),
                             "value_out_of_range")
        self.assertIn("129 bytes for a field of 128", error.message)
        self.refused("wrl", template("wrl"), lambda d: d["server"]["clusters_raw"].__setitem__("bytes", "00"), "value_out_of_range")

    def test_unknown_name(self):
        self.refused("acr", template("acr"), lambda d: d["fields"]["actor_kind"].__setitem__("name", "NO_SUCH"), "unknown_name")
        self.refused("acr", template("acr"), lambda d: d["fields"]["physics_flags"]["names"].append("NO_SUCH"), "unknown_name")

    def test_conflicting_values(self):
        data = template("acr")
        place = J.places("acr", data, BUILD)[("fields", "actor_kind")]
        name = next(n for n, v in place.names.items() if v != 1)
        self.refused("acr", data, lambda d: d["fields"]["actor_kind"].update(name=name, value=1234), "conflicting_values")
        self.refused("mrk", template("mrk"), lambda d: d["markers"][1]["rotation"].update(
            quaternion=[0.0, 0.0, 0.0, 1.0], raw=[0.0, 0.0, 0.6, 0.8 + 0.5]), "conflicting_values")

        def twice(d):
            d["fields"]["bound_sphere_radius"] = 3.0
            d["bounds"]["bound_sphere_radius"] = 4.0

        self.refused("acr", data, twice, "conflicting_values")

    def test_wrong_group(self):
        self.refused("acr", template("acr"), lambda d: d["fields"]["appearance_ref"].__setitem__("group", "tex"), "wrong_group")

    def test_list_length_fixed(self):
        error = self.refused("scn", template("scn"), lambda d: d["exclusions"].append(5), "list_length_fixed")
        self.assertEqual(J.path_text(error.path), "exclusions")
        self.refused("mrk", template("mrk", canonical=True), lambda d: d["circles"].append([1.0, 2.0, 3.0]), "list_length_fixed")

    def test_not_canonical(self):
        """A template the tree writer does not lay out as stored takes same-length values only."""
        data = template("mrk")
        self.assertNotEqual(L.rebuild_sno(data, "mrk", BUILD), data)
        self.refused("mrk", data, lambda d: d["markers"].append(copy.deepcopy(d["markers"][0])), "not_canonical")
        padded = template("acr") + bytes(4)  # bytes after the last block: not rebuilt as stored
        self.refused("acr", padded, lambda d: d["fields"].__setitem__("internal_description", "new"), "not_canonical")
        result, _ = run("acr", padded, lambda d: d["fields"].__setitem__("bound_sphere_radius", 2.0))  # same length
        self.assertEqual(len(result.data), len(padded))

    def test_derived_field(self):
        self.refused("mon", template("mon"), lambda d: d["attributes"].__setitem__("Level", 7), "derived_field")
        self.refused("mrk", template("mrk"), lambda d: d["markers"][0].__setitem__("scale", 2.0), "derived_field")
        self.refused("acr", template("acr"), lambda d: d["header"].__setitem__("handle", 5), "derived_field")

    def test_recomputed_derived_is_accepted(self):
        """W22-5: a derived leaf that the export computes from the new stored values (edited consistently)."""
        def edit(d):
            d["fields"]["actor_kind"]["value"] = 2
            d["actor_type"] = dict(d["fields"]["actor_kind"])

        result, _ = run("acr", template("acr"), edit)
        self.assertEqual(exported("acr", result.data)["actor_type"]["value"], 2)

    def test_navigation_geometry(self):
        self.refused("scn", template("scn"), lambda d: d["navigation"]["cells"][0]["min"].__setitem__(0, 9.0), "navigation_geometry")
        self.refused("scn", template("scn"), lambda d: d["navigation"]["cells"].pop(), "navigation_geometry")

    def test_field_fixed(self):
        self.refused("mat", template("mat"), lambda d: d["textures"][0]["slot"].__setitem__("value", 9), "field_fixed")

    def test_missing_shader(self):
        self.refused("mat", template("mat"), lambda d: d["shader"].__setitem__("handle", 4242), "missing_shader",
                     known=lambda g, h: False)

    def test_all_refusals_named(self):
        def edit(d):
            d["fields"]["actor_kind"]["name"] = "NO_SUCH"
            d["fields"]["appearance_ref"]["group"] = "tex"

        error = self.refused("acr", template("acr"), edit, "unknown_name")  # the first path in key order
        self.assertEqual(sorted(r.reason for r in error.refusals), ["unknown_name", "wrong_group"])


class Navigation(unittest.TestCase):
    """W22-8: cell flags and neighbour cells; a ps3 scene with a separate ``.nav`` writes the ``.nav``."""

    def test_switch_cell_flag(self):
        data = template("scn")
        result, _ = run("scn", data, lambda d: d["navigation"]["cells"][1]["flags"].update(value=1, names=["walk"]))
        self.assertEqual(navigation.parse_scene(result.data, BUILD).cells[1].flags, 1)
        self.assertIsNone(result.nav)
        self.assertEqual(len(differing(data, result.data)), 1)

    def test_neighbour_cell(self):
        data = template("scn")
        nav = navigation.parse_scene(data, BUILD)
        k = next(i for i, n in enumerate(nav.neighbours) if n.border_index is None)
        other = (nav.neighbours[k].cell + 1) % len(nav.cells)
        result, _ = run("scn", data, lambda d: d["navigation"]["neighbours"][k].__setitem__("cell", other))
        self.assertEqual(navigation.parse_scene(result.data, BUILD).neighbours[k].cell, other)
        with self.assertRaises(J.Refused) as caught:
            run("scn", data, lambda d: d["navigation"]["neighbours"][k].__setitem__("cell", 99))
        self.assertEqual(caught.exception.reason, "value_out_of_range")

    def test_ps3_nav_through_apply(self):
        setup = S.Setup(self, F.ps3_files())
        path = setup.edit / f"data/scn/{F.SCN}.json"
        document = json.loads(path.read_text(encoding="utf-8"))
        document["navigation"]["cells"][1]["flags"]["value"] = 1
        path.write_text(json.dumps(document), encoding="utf-8")
        report, out = setup.ok("out", only="world")
        (row,) = [r for r in report["changed"] if r["path"] == f"data/scn/{F.SCN}.json"]
        self.assertEqual(row["result"], "applied")
        self.assertEqual(row["also_written"], ["Scene\\Room\\0000.nav"])
        self.assertEqual(row["fields_changed"], 1)
        cpk = A.read_archive((out / "Common.cpk").read_bytes())
        scn = cpk.read_file(F.name_of("scn", "Room"))
        nav = cpk.read_file("Scene\\Room\\0000.nav")
        self.assertEqual(scn, F.ps3_files()["Common"][F.name_of("scn", "Room")])
        self.assertEqual(navigation.parse_scene(scn, "ps3", nav).cells[1].flags, 1)


class Collision(unittest.TestCase):
    """W22-9: a moved vertex and an added triangle rebuild one mesh; the other is byte-equal."""

    def setUp(self):
        quad = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 1.0, 0.0)]
        tri = [(5.0, 5.0, 0.0), (6.0, 5.0, 0.0), (5.0, 6.0, 1.0)]
        meshes = [CW.build_collision_mesh(quad, [(0, 1, 2), (1, 3, 2)], [4, 4], [1, 2], BUILD),
                  CW.build_collision_mesh(tri, [(0, 1, 2)], 7, 0, BUILD)]
        self.data = CW.write_collision(CW.new_collision_file(meshes, BUILD, handle=F.PHM))

    def test_rebuild_one_mesh(self):
        def edit(d):
            mesh = d["meshes"][0]
            mesh["vertices"][3] = [1.0, 1.0, 0.5]
            mesh["vertices"].append([2.0, 0.0, 0.0])
            mesh["triangles"].append({"vertices": [1, 4, 3], "neighbours": [-1, -1, -1], "material": 4, "flags": 3})

        result, _ = run("phm", self.data, edit)
        old, new = collision.parse_collision(self.data, BUILD), collision.parse_collision(result.data, BUILD)
        self.assertEqual(new.meshes[0].triangle_count, 3)
        self.assertIn((1.0, 1.0, 0.5), [tuple(p) for p in new.meshes[0].positions])
        self.assertEqual(sorted(t.flags for t in new.meshes[0].triangles), [1, 2, 3])
        for field in ("centre", "half_extent", "box_scale", "positions", "table", "nodes", "vertices", "flags", "material"):
            self.assertEqual(getattr(new.meshes[1], field), getattr(old.meshes[1], field), field)
        tree_old, tree_new = L.parse_tree(self.data, "phm", BUILD), L.parse_tree(result.data, "phm", BUILD)
        a, b = tree_old.top.pairs[0x14].block.elements[1], tree_new.top.pairs[0x14].block.elements[1]
        self.assertEqual({k: v.block.data for k, v in a.pairs.items() if v.block}, {k: v.block.data for k, v in b.pairs.items() if v.block})

    def test_bounds_derived(self):
        # ``52`` AC1: bounds that do not enclose the vertices are derived again (not refused), with a warning
        result, _ = run("phm", self.data, lambda d: d["meshes"][1]["bounds"]["centre"].__setitem__(0, 99.0))
        self.assertIn("bounds_refitted: mesh 1", result.warnings)
        self.assertEqual(collision.parse_collision(result.data, BUILD).meshes[1], collision.parse_collision(self.data, BUILD).meshes[1])

    def test_mesh_count_fixed(self):
        with self.assertRaises(J.Refused) as caught:
            run("phm", self.data, lambda d: d["meshes"].pop())
        self.assertEqual(caught.exception.reason, "list_length_fixed")


# --- animations (W22-10) ---------------------------------------------------------------------------

def animation_files(build, files):
    found = {n: d for p in files.values() for n, d in p.items()}
    can = found.get("Anim\\Hero_walk\\0000.can")
    return found[F.name_of("ani", "Hero_walk")], found[F.name_of("app", "Hero")], can


def animation_glb(ani, app, build, can=None) -> bytes:
    return E._animation_glb(appearance.parse_appearance(app, build), animation.parse_ani(ani, build, can), F.ANI, F.APP)


def glb_split(glb: bytes):
    length = struct.unpack_from("<I", glb, 12)[0]
    document = json.loads(glb[20:20 + length])
    at = 20 + length
    size = struct.unpack_from("<I", glb, at)[0]
    return document, bytearray(glb[at + 8:at + 8 + size])


def glb_join(document, binary) -> bytes:
    document["buffers"][0]["byteLength"] = len(binary)
    text = json.dumps(document).encode("utf-8")
    text += b" " * (-len(text) % 4)
    binary = bytes(binary) + bytes(-len(binary) % 4)
    body = struct.pack("<II", len(text), 0x4E4F534A) + text + struct.pack("<II", len(binary), 0x004E4942) + binary
    return struct.pack("<4sII", b"glTF", 2, 12 + len(body)) + body


def _accessor(document, binary, rows, kind, bounds=False) -> int:
    binary += bytes(-len(binary) % 4)
    flat = [c for row in rows for c in row]
    document["bufferViews"].append({"buffer": 0, "byteOffset": len(binary), "byteLength": 4 * len(flat)})
    binary += struct.pack(f"<{len(flat)}f", *flat)
    accessor = {"bufferView": len(document["bufferViews"]) - 1, "componentType": 5126, "count": len(rows), "type": kind}
    if bounds:
        accessor["min"], accessor["max"] = [min(flat)], [max(flat)]
    document["accessors"].append(accessor)
    return len(document["accessors"]) - 1


def with_channel(glb, anim, node, path, times, values) -> bytes:
    document, binary = glb_split(glb)
    animation_json = document["animations"][anim]
    (channel,) = [c for c in animation_json["channels"] if c["target"] == {"node": node, "path": path}]
    sampler = animation_json["samplers"][channel["sampler"]]
    sampler["input"] = _accessor(document, binary, [(t,) for t in times], "SCALAR", True)
    sampler["output"] = _accessor(document, binary, values, {"translation": "VEC3", "rotation": "VEC4", "scale": "VEC3"}[path])
    return glb_join(document, binary)


class Animations(unittest.TestCase):
    def setUp(self):
        ani, self.app, _ = animation_files("switch", F.switch_files())
        parsed = animation.parse_ani(ani, "switch")
        self.ani = AW.write_ani(parsed.permutations, "switch", head=ani)  # the writer's own layout
        self.glb = animation_glb(self.ani, self.app, "switch")
        self.end = 2 / 30

    def imported(self, glb, **kw):
        return AI.import_animation(self.ani, "switch", self.app, "switch", glb, F.ANI, F.APP, **kw)

    def refused(self, glb, reason):
        with self.assertRaises(J.Refused) as caught:
            self.imported(glb)
        self.assertEqual(caught.exception.reason, reason)

    def test_one_rotation_key(self):
        edited = with_channel(self.glb, 0, 0, "rotation", [0.0], [(0.0, 0.0, 0.6, 0.8)])
        result = self.imported(edited)
        old, new = animation.parse_ani(self.ani, "switch"), animation.parse_ani(result.data, "switch")
        for p, (a, b) in enumerate(zip(old.permutations, new.permutations)):
            for kind in ("translation", "rotation", "scale"):
                for track, (x, y) in enumerate(zip(getattr(a, f"{kind}_blobs"), getattr(b, f"{kind}_blobs"))):
                    self.assertEqual(x != y, (p, kind, track) == (0, "rotation", 0), (p, kind, track))
        curve = animation.decode_rotation(new.permutations[0].rotation_blobs[0], 3, "<")
        self.assertEqual(curve.frames, [0])
        self.assertTrue(all(abs(a - b) < 1e-4 for a, b in zip(curve.values[0], (0.0, 0.0, 0.6, 0.8))))
        self.assertEqual(result.changes, [{"path": "animations[0].channels[0].rotation", "old": {"keys": 1}, "new": {"keys": 1}}])

    def test_translation_keys(self):
        edited = with_channel(self.glb, 1, 0, "translation", [0.0, 1 / 30, self.end], [(1, 2, 3), (2, 2, 2), (3, 2, 1)])
        result = self.imported(edited)
        curve = animation.decode_translation(animation.parse_ani(result.data, "switch").permutations[1].translation_blobs[0], 3, "<")
        self.assertEqual(curve.frames, [0, 1, 2])
        self.assertTrue(all(math.isclose(a, b, abs_tol=1e-5) for a, b in zip(curve.values[1], (2.0, 2.0, 2.0))))

    def test_key_just_below_a_frame(self):
        """A key time a little below a frame (float32 of 1/30) is that frame."""
        near = struct.unpack("<f", struct.pack("<f", 1 / 30 - 1e-8))[0]
        self.assertLess(near * 30, 1.0)
        edited = with_channel(self.glb, 1, 0, "translation", [0.0, near, self.end], [(1, 2, 3), (2, 2, 2), (3, 2, 1)])
        curve = animation.decode_translation(
            animation.parse_ani(self.imported(edited).data, "switch").permutations[1].translation_blobs[0], 3, "<")
        self.assertEqual(curve.frames, [0, 1, 2])

    def test_reencoded_unedited_gives_stored_bytes(self):
        for build, files in (("switch", F.switch_files()), ("ps3", F.ps3_files())):
            with self.subTest(build):
                ani, app, can = animation_files(build, files)
                result = AI.import_animation(ani, build, app, build, animation_glb(ani, app, build, can), F.ANI, F.APP,
                                             can, force=True)
                self.assertEqual((result.data, result.nav), (ani, None))

    def test_ps3_writes_the_can(self):
        ani, app, can = animation_files("ps3", F.ps3_files())
        glb = with_channel(animation_glb(ani, app, "ps3", can), 0, 0, "rotation", [0.0], [(0.0, 0.0, 0.6, 0.8)])
        result = AI.import_animation(ani, "ps3", app, "ps3", glb, F.ANI, F.APP, can)
        self.assertEqual(result.data, ani)
        self.assertIsNotNone(result.nav)
        merged = animation.parse_ani(ani, "ps3", result.nav)
        self.assertEqual(animation.decode_rotation(merged.permutations[0].rotation_blobs[0], merged.permutations[0].keyframes,
                                                   ">").frames, [0])

    def test_skeleton_mismatch(self):
        document, binary = glb_split(self.glb)
        document["nodes"][0]["name"] = "another_bone"
        self.refused(glb_join(document, binary), "skeleton_mismatch")

    def test_duration_changed(self):
        self.refused(with_channel(self.glb, 0, 0, "translation", [0.0, 0.1], [(1, 2, 3), (3, 2, 1)]), "duration_changed")

    def test_keyframe_count_mismatch(self):
        self.refused(with_channel(self.glb, 0, 0, "translation", [0.0, 0.05, self.end], [(1, 2, 3), (2, 2, 2), (3, 2, 1)]),
                     "keyframe_count_mismatch")

    def test_permutation_count(self):
        document, binary = glb_split(self.glb)
        document["animations"].pop()
        self.refused(glb_join(document, binary), "permutation_count")

    def test_extras_derived(self):
        document, binary = glb_split(self.glb)
        document["extras"]["d3sno_export"]["permutations"][0]["average_velocity"] = [9.0, 0.0, 0.0]
        self.refused(glb_join(document, binary), "derived_field")
        document, binary = glb_split(self.glb)
        document["extras"]["d3sno_export"]["ticks_per_second"] = 30
        self.refused(glb_join(document, binary), "derived_field")


class Apply(unittest.TestCase):
    """W22-11 in ``apply``: rows with ``fields_changed`` and ``fields``, refusals with every path, ``--only``, the dry run."""

    def setUp(self):
        self.setup = S.Setup(self)

    def edit_json(self, rel, change):
        path = self.setup.edit / rel
        document = json.loads(path.read_text(encoding="utf-8"))
        change(document)
        path.write_text(json.dumps(document, indent=1), encoding="utf-8")

    def rows(self, report):
        return {r["path"]: r for r in report["changed"]}

    def test_report_and_archives(self):
        self.edit_json(f"data/acr/{F.ACR}.json", lambda d: d["fields"].__setitem__("bound_sphere_radius", 2.0))
        self.edit_json(f"data/mat/{F.MAT}.json", lambda d: d["textures"][0]["slot"].__setitem__("value", 9))
        self.edit_json(f"data/mrk/{F.MRK}.json", lambda d: d["markers"][0]["position"].__setitem__(2, 7.0))
        S.move_vertex(self.setup.edit / S.MESH_PATH)  # a mesh: applied before the records though its path sorts after
        order = []
        change = AP._Apply.change

        def recording(job, entry, target, kind):
            order.append(kind)
            return change(job, entry, target, kind)

        with mock.patch.object(AP._Apply, "change", recording):
            report, out = self.setup.ok()
        rows = self.rows(report)
        acr = rows[f"data/acr/{F.ACR}.json"]
        self.assertEqual((acr["result"], acr["fields_changed"], acr["fields"]),
                         ("applied", 1, [{"path": "fields.bound_sphere_radius", "old": 0.5, "new": 2.0}]))
        mat = rows[f"data/mat/{F.MAT}.json"]
        self.assertEqual((mat["result"], mat["reason"]), ("refused", "field_fixed"))
        self.assertEqual(mat["refused_fields"][0]["path"], "textures[0].slot")
        self.assertEqual(rows[f"data/mrk/{F.MRK}.json"]["result"], "applied")
        cpk = A.read_archive((out / "Common.cpk").read_bytes())
        written = cpk.read_file(F.name_of("acr", "Hero"))
        self.assertEqual(records.parse_record_file(written, "acr", BUILD).record["bound_sphere_radius"], 2.0)
        self.assertEqual(A.verify_archive((out / "Common.cpk").read_bytes()).ok, True)
        self.assertEqual(order[0], "meshes")
        self.assertEqual(order, sorted(order, key=AP.KINDS.index))  # records after meshes and textures
        self.assertEqual([r["path"] for r in report["changed"]], sorted(r["path"] for r in report["changed"]))

    def test_only_and_dry_run(self):
        self.edit_json(f"data/acr/{F.ACR}.json", lambda d: d["fields"].__setitem__("bound_sphere_radius", 2.0))
        self.edit_json(f"data/phm/{F.PHM}.json", lambda d: d["meshes"][0]["vertices"][1].__setitem__(1, 3.0))
        report, _ = self.setup.ok("only", only="collision")
        rows = self.rows(report)
        self.assertEqual(rows[f"data/acr/{F.ACR}.json"]["result"], "filtered")
        self.assertEqual(rows[f"data/phm/{F.PHM}.json"]["result"], "applied")
        dry, out = self.setup.ok("dry", dry_run=True, only="records,world")
        acr = self.rows(dry)[f"data/acr/{F.ACR}.json"]
        self.assertEqual((acr["result"], acr["fields_changed"]), ("applied", 1))
        self.assertFalse(list(out.rglob("*.cpk")))
        self.assertEqual(AP._only("records,world,collision,anims"), ("records", "world", "collision", "anims"))
        self.assertEqual(AP.KINDS, ("meshes", "textures", "text", "records", "world", "collision", "anims"))

    def test_animation_entry(self):
        glb = self.setup.edit / f"anims/{F.ANI}.glb"
        glb.write_bytes(with_channel(glb.read_bytes(), 0, 0, "rotation", [0.0], [(0.0, 0.0, 0.6, 0.8)]))
        report, out = self.setup.ok(only="anims")
        row = self.rows(report)[f"anims/{F.ANI}.glb"]
        self.assertEqual((row["result"], row["fields_changed"]), ("applied", 1))
        cpk = A.read_archive((out / "Act1.cpk").read_bytes())
        ani = animation.parse_ani(cpk.read_file(F.name_of("ani", "Hero_walk")), BUILD)
        self.assertEqual(animation.decode_rotation(ani.permutations[0].rotation_blobs[0], 3, "<").frames, [0])

    def test_fields_listed_up_to_200(self):
        def every_float(d):
            for key, value in list(d["fields"].items()):
                if isinstance(value, float):
                    d["fields"][key] = 1.5

        self.edit_json(f"data/mon/{F.MON}.json", every_float)
        report, _ = self.setup.ok("many")
        row = self.rows(report)[f"data/mon/{F.MON}.json"]
        self.assertEqual(row["fields_changed"], len(row["fields"]))
        self.assertLessEqual(len(row["fields"]), AP.FIELDS_LISTED)

    def test_unedited_record_is_unchanged(self):
        report, _ = self.setup.ok("none")
        self.assertEqual(report["changed"], [])


if __name__ == "__main__":
    unittest.main()
