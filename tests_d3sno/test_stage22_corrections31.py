"""Corrections 31 (``51_corrections31.md``): AB1 to AB7. Synthetic games and archives of the test builders
only; every export and output goes to a temporary folder outside any git working tree."""

import _libpath  # noqa: F401  (python -I: the library folder on sys.path)
import copy
import io
import json
import shutil
import struct
import unittest
from unittest import mock

import builders as B
import export_fixture as F
import test_stage21 as S
import test_stage21_corrections28 as C28
import test_stage22 as T
import y1_cases as Y
import d3sno
from d3sno import anim_import as AI
from d3sno import animation
from d3sno import apply as AP
from d3sno import archive as A
from d3sno import collision
from d3sno import collision_write as CW
from d3sno import export as E
from d3sno import index_write as IW
from d3sno import json_import as J
from d3sno import navigation
from d3sno import sno_layout as L
from d3sno import world
from d3sno import x360_archive as XA
from d3sno.binary import ByteView, FormatError
from d3sno.material_write import encode_tag_map


def grid_mesh(build: str, n: int = 4, material=3, flags=1):
    """An n x n grid of quads (2 n^2 triangles, a tree with inner nodes), written by the builder."""
    positions = [(float(x), float(y), 0.1 * ((x * 7 + y * 3) % 5)) for y in range(n + 1) for x in range(n + 1)]
    tris = []
    for y in range(n):
        for x in range(n):
            a, b, c, d = y * (n + 1) + x, y * (n + 1) + x + 1, (y + 1) * (n + 1) + x, (y + 1) * (n + 1) + x + 1
            tris += [(a, b, c), (b, d, c)]
    return positions, tris, CW.build_collision_mesh(positions, tris, material, flags, build)


def collision_file(build: str) -> bytes:
    _, _, first = grid_mesh(build)
    second = CW.build_collision_mesh([(5.0, 5.0, 0.0), (6.0, 5.0, 0.0), (5.0, 6.0, 1.0)], [(0, 1, 2)], 7, 0, build)
    return CW.write_collision(CW.new_collision_file([first, second], build, handle=F.PHM))


def imported(ext, data, build, edit, **kw):
    document = J._json_round(J.template_json(ext, data, build))
    edit(document)
    return J.import_json(ext, data, build, document, **kw), document


class Refit(unittest.TestCase):
    """AB1: an edit that keeps the triangle list refits the mesh; one that changes it rebuilds."""

    def test_moved_vertex_keeps_order_tree_and_neighbours(self):
        for build in ("switch", "ps3", "x360"):
            with self.subTest(build):
                data = collision_file(build)
                def edit(d):
                    d["meshes"][0]["vertices"][6][0] = 1.375  # vertex 6 is (1, 1, z)
                    d["meshes"][0]["vertices"][24][2] = 3.0  # a corner: new mesh bounds and box scale

                result, _ = imported("phm", data, build, edit)
                old, new = collision.parse_collision(data, build).meshes[0], collision.parse_collision(result.data, build).meshes[0]
                self.assertEqual([(t.vertices, t.neighbours) for t in new.table], [(t.vertices, t.neighbours) for t in old.table])
                self.assertEqual([n.data for n in new.nodes], [n.data for n in old.nodes])  # tree and leaf ranges
                self.assertNotEqual(new.nodes, old.nodes)  # new boxes
                self.assertNotEqual(new.box_scale, old.box_scale)
                self.assertEqual((new.positions[6][0], new.positions[24][2]), (1.375, 3.0))
                self.assertEqual(J.template_json("phm", result.data, build)["meshes"][0]["triangles"],
                                 J.template_json("phm", data, build)["meshes"][0]["triangles"])
                self.assertEqual(result.warnings, ["bounds_refitted: mesh 0"])  # ``52`` AC1: the corner leaves the bounds
                rebuilt =CW.build_collision_mesh(new.positions, [t.vertices for t in new.triangles], 3, 1, build)
                if [t.vertices for t in rebuilt.triangles] == [t.vertices for t in new.triangles]:
                    self.assertEqual((rebuilt.nodes, rebuilt.box_scale), (new.nodes, new.box_scale))
                self.assertEqual(collision.parse_collision(result.data, build).meshes[1], collision.parse_collision(data, build).meshes[1])

    def test_moved_back_gives_the_template(self):
        for build in ("switch", "ps3", "x360"):
            with self.subTest(build):
                data = collision_file(build)
                original = J._json_round(J.template_json("phm", data, build))
                moved, _ = imported("phm", data, build, lambda d: d["meshes"][0]["vertices"][7].__setitem__(0, 1.25))
                self.assertNotEqual(moved.data, data)
                back = J.import_json("phm", moved.data, build, original)
                self.assertEqual(back.data, data)

    def test_material_and_flags_refit_in_place(self):
        data = collision_file("ps3")
        def edit(d):
            for i, triangle in enumerate(d["meshes"][0]["triangles"]):
                triangle.update(material=9 + i, flags=4)

        result, _ = imported("phm", data, "ps3", edit)
        old, new = collision.parse_collision(data, "ps3").meshes[0], collision.parse_collision(result.data, "ps3").meshes[0]
        self.assertEqual([(t.material, t.flags) for t in new.triangles], [(9 + i, 4) for i in range(32)])
        self.assertNotEqual(new.table[new.triangle_count:], old.table[old.triangle_count:])
        self.assertEqual((new.nodes, new.centre, new.box_scale), (old.nodes, old.centre, old.box_scale))
        last = new.table[new.triangle_count - 1]
        self.assertEqual(new.table[new.triangle_count:], [last] * 3)  # the ps3 tail follows the last used triangle
        self.assertEqual(len(result.data), len(data))  # written in place

    def test_changed_neighbour_on_a_refit(self):
        data = collision_file("switch")
        with self.assertRaises(J.Refused) as caught:
            imported("phm", data, "switch", lambda d: d["meshes"][0]["triangles"][0]["neighbours"].__setitem__(0, 9))
        self.assertEqual((caught.exception.reason, J.path_text(caught.exception.path)),
                         ("derived_field", "meshes[0].triangles[0].neighbours[0]"))

    def test_added_triangle_rebuilds(self):
        for build in ("switch", "ps3"):
            with self.subTest(build):
                data = collision_file(build)

                def edit(d):
                    mesh = d["meshes"][0]
                    mesh["vertices"].append([-1.0, 0.0, 0.0])
                    mesh["triangles"].insert(0, {"vertices": [25, 0, 5], "neighbours": [-1, -1, -1], "material": 3, "flags": 1})

                result, document = imported("phm", data, build, edit)
                edited = [tuple(t["vertices"]) for t in document["meshes"][0]["triangles"]]
                built = [tuple(t.vertices) for t in collision.parse_collision(result.data, build).meshes[0].triangles]
                self.assertEqual(sorted(built), sorted(edited))
                self.assertEqual("triangles_reordered: mesh 0" in result.warnings, built != edited)
                # the original JSON put back: a rebuild again (another triangle count), not refused for neighbours
                original = J._json_round(J.template_json("phm", data, build))
                back = J.import_json("phm", result.data, build, original)
                self.assertEqual(collision.parse_collision(back.data, build).meshes[0].triangle_count, 32)

    def test_reordered_triangles(self):
        """The triangles given in another order (with their neighbours): a rebuild in the builder's order."""
        data = collision_file("switch")
        result, document = imported("phm", data, "switch", lambda d: d["meshes"][0]["triangles"].reverse())
        self.assertIn("triangles_reordered: mesh 0", result.warnings)
        built = collision.parse_collision(result.data, "switch").meshes[0]
        given = [tuple(t["vertices"]) for t in document["meshes"][0]["triangles"]]
        self.assertEqual(sorted(t.vertices for t in built.triangles), sorted(given))
        self.assertNotEqual([t.vertices for t in built.triangles], given)

    def test_rebuilt_neighbours(self):
        data = collision_file("switch")

        def edit(d, value):
            mesh = d["meshes"][0]
            mesh["triangles"][0]["vertices"] = [0, 1, 6]  # another vertex index: a rebuild
            mesh["triangles"][1]["neighbours"][0] = value

        tris = [tuple(t["vertices"]) for t in J._json_round(J.template_json("phm", data, "switch"))["meshes"][0]["triangles"]]
        tris[0] = (0, 1, 6)
        derived = CW.triangle_neighbours(tris)[1][0]
        imported("phm", data, "switch", lambda d: edit(d, derived))  # as W14-15 gives it: accepted
        with self.assertRaises(J.Refused) as caught:
            imported("phm", data, "switch", lambda d: edit(d, derived + 100))
        self.assertEqual(caught.exception.reason, "derived_field")


class Diagnostics(unittest.TestCase):
    """AB2: export diagnostics are never compared; a new row is an ``export_warning``."""

    def setUp(self):
        self.data = T.template("scn")
        self.original = T.exported("scn", self.data)

    def test_new_warning_row(self):
        result, _ = T.run("scn", self.data, lambda d: d["navigation"]["neighbours"][0].__setitem__("cell", 0))
        self.assertEqual(navigation.parse_scene(result.data, T.BUILD).neighbours[0].cell, 0)
        self.assertIn("export_warning: navigation.warnings: neighbour 0: flags differ from the target cell", result.warnings)
        back = J.import_json("scn", result.data, T.BUILD, self.original)  # no warning row: applied
        self.assertEqual(back.data, self.data)

    def test_any_content_in_a_diagnostic(self):
        document = copy.deepcopy(self.original)
        document["navigation"]["warnings"] = ["anything", {"at": 3}, 7]
        document["warnings"] = ["x"] * 9
        result = J.import_json("scn", self.data, T.BUILD, document)
        self.assertEqual((result.data, result.changes, result.warnings), (self.data, [], []))
        del document["warnings"]  # a diagnostic left out: as the template's too
        self.assertEqual(J.import_json("scn", self.data, T.BUILD, document).data, self.data)

    def test_classified(self):
        found = J.places("scn", self.data, T.BUILD)
        self.assertEqual(J.classify("scn", ("navigation", "warnings", 0), found), "diagnostic")
        self.assertEqual(J.classify("wrl", ("scene_entries", 2, "note"), {}), "diagnostic")
        self.assertEqual(J.classify("acr", ("warnings",), {}), "diagnostic")
        self.assertEqual(J.export_warnings("scn", {"warnings": ["a"]}, {"warnings": ["a", "b", "a"]}),
                         ["export_warning: warnings: b", "export_warning: warnings: a"])


def edit_json(path, change):
    document = json.loads(path.read_text(encoding="utf-8"))
    change(document)
    path.write_text(json.dumps(document), encoding="utf-8")


def canonical_files():
    """The switch game with its marker set laid out as the tree writer lays it out (a list may change)."""
    files = F.switch_files()
    name = F.name_of("mrk", "Room")
    files["Act1"][name] = L.rebuild_sno(files["Act1"][name], "mrk", "switch")
    return files


def added_marker(d):
    new = copy.deepcopy(d["markers"][0])
    new["position"] = [9.0, 8.0, 7.0]
    d["markers"].append(new)


class WorldLayout(unittest.TestCase):
    """AB3: worlds last; the layout is compared with the state after the apply."""

    def test_undo_in_one_apply(self):
        files = canonical_files()
        setup = S.Setup(self, files)
        edit_json(setup.edit / f"data/mrk/{F.MRK}.json", added_marker)
        first, out = setup.ok("first", copy_unchanged=True)
        self.assertEqual({r["path"]: r["result"] for r in first["changed"]}, {f"data/mrk/{F.MRK}.json": "applied"})
        export2 = setup.root / "export2"
        self.assertEqual(E.export_game(out, export2, created="X").exit_code, 0)
        world_now = json.loads((export2 / f"data/wrl/{F.WRL}.json").read_text(encoding="utf-8"))
        self.assertEqual(len(world_now["layout"]["markers"]), 3)
        edit2 = setup.root / "edit2"
        shutil.copytree(export2, edit2)
        for rel in (f"data/mrk/{F.MRK}.json", f"data/wrl/{F.WRL}.json"):  # the original JSON put back
            shutil.copyfile(setup.export / rel, edit2 / rel)
        result = AP.apply_export(out, edit2, setup.root / "second")
        self.assertEqual(result.exit_code, 0, result.message)
        rows = {r["path"]: r for r in result.report["changed"]}
        self.assertEqual(rows[f"data/mrk/{F.MRK}.json"]["result"], "applied")
        self.assertNotEqual(rows[f"data/wrl/{F.WRL}.json"]["result"], "refused", rows[f"data/wrl/{F.WRL}.json"])
        cpk = A.read_archive((setup.root / "second" / "Act1.cpk").read_bytes())
        self.assertEqual(cpk.read_file(F.name_of("mrk", "Room")), files["Act1"][F.name_of("mrk", "Room")])

    def test_world_applied_last_with_the_new_marker_sets(self):
        setup = S.Setup(self, canonical_files())
        edit_json(setup.edit / f"data/mrk/{F.MRK}.json", added_marker)
        new_layout = []

        def world_edit(d):
            d["music"]["handle"] = 555
            new_layout.append(d["layout"])

        edit_json(setup.edit / f"data/wrl/{F.WRL}.json", world_edit)
        # a collision file: its kind comes after the world's, yet the world is applied after it
        edit_json(setup.edit / f"data/phm/{F.PHM}.json", lambda d: d["meshes"][0]["vertices"][1].__setitem__(1, 3.0))
        order = []
        change = AP._Apply.change

        def recording(job, entry, target, kind):
            order.append(entry["group"])
            return change(job, entry, target, kind)

        with mock.patch.object(AP._Apply, "change", recording):
            report, _ = setup.ok()
        self.assertEqual(order[-1], "wrl")
        rows = {r["path"]: r for r in report["changed"]}
        self.assertEqual(rows[f"data/wrl/{F.WRL}.json"]["result"], "applied")  # its layout untouched: accepted

    def test_layout_edited_by_hand(self):
        setup = S.Setup(self)
        edit_json(setup.edit / f"data/wrl/{F.WRL}.json", lambda d: d["layout"]["markers"][0]["position"].__setitem__(0, 42.0))
        report, _ = setup.ok()
        (row,) = report["changed"]
        self.assertEqual((row["result"], row["reason"], row["message"]), ("refused", "derived_field", J.LAYOUT_MESSAGE))
        self.assertEqual(row["refused_fields"][0]["path"], "layout")


def shared_tag_markers(build):
    """Three markers whose tag maps are one shared block (W11-6), canonical."""
    tags = encode_tag_map([(2, 0x10, 5)], build)
    markers = [{"hash": 0x201 + i, "type": 0, "group": 1, "handle": F.ACR, "position": (1.0 * i, 2.0, 0.0),
                "links": [(i, i + 1)] if i == 1 else [], "tags": tags} for i in range(3)]
    tree = L.parse_tree(B.build_mrk(build, markers=markers, circles=[(1.0, 2.0, 3.0)], special=[1, 2, 3]), "mrk", build)
    elements = tree.top.pairs[0x0C].block.elements
    for element in elements[1:]:
        element.pairs[0x30] = elements[0].pairs[0x30]
    return L.write_tree(tree)


class ListLayout(unittest.TestCase):
    """AB4: a list change and a leaf change, then the reverse, give the template's bytes."""

    def test_reverse_gives_the_template(self):
        for build in ("switch", "x360", "ps3"):
            data = shared_tag_markers(build)
            original = J._json_round(J.template_json("mrk", data, build))
            for where, source, changed in ((0, 1, 1), (0, 0, 1), (1, 1, 0), (3, 2, 0), (0, 2, 2)):
                with self.subTest(build=build, where=where, source=source, changed=changed):
                    document = copy.deepcopy(original)
                    document["markers"][changed]["name_hash"] = 0x999
                    new = copy.deepcopy(original["markers"][source])
                    new["position"] = [5.0, 5.0, 5.0]
                    document["markers"].insert(where, new)
                    first = J.import_json("mrk", data, build, document).data
                    markers = world.parse_marker_set(first, build).markers
                    self.assertEqual(len(markers), 4)
                    self.assertIn(0x999, [m.name_hash for m in markers])
                    self.assertEqual(J.import_json("mrk", first, build, original).data, data)

    def test_copy_keeps_the_base_block_until_changed(self):
        data = shared_tag_markers("switch")

        def edit(d):
            new = copy.deepcopy(d["markers"][1])
            new["links"][0] = [70, 71]
            d["markers"].append(new)
            d["markers"][1]["links"][0][1] = 99  # the base's own block changes too

        result, _ = imported("mrk", data, "switch", edit)
        markers = world.parse_marker_set(result.data, "switch").markers
        self.assertEqual([m.links for m in markers], [[], [(1, 99)], [], [(70, 71)]])
        view = ByteView(result.data, "<")
        table = view.pair(0x10 + 0x0C)
        tags = {view.pair(table.file_offset + i * world.MARKER_SIZE + 0x30).file_offset for i in range(4)}
        self.assertEqual(len(tags), 1)  # one tag block for every marker, the copy's too

    def test_matching(self):
        """``52`` AC3 replaces the alignment of AB4: rows by position, equal rows from their stored record."""
        old = [{"name_hash": h, "type": 0, "link": None, "links": []} for h in (1, 2, 3)]
        x = dict(old[1], name_hash=99)
        copied = dict(old[1])

        def match(a, b):
            return J._match_elements(a, b, "mrk", ("markers",))

        self.assertEqual(match(old, [copied, old[0], x, old[2]]), [(1, True), (0, True), None, (2, True)])
        self.assertEqual(match(old, old[:1] + old[2:]), [(0, True), (2, True)])
        self.assertEqual(match(old[:1], [old[0], old[0]]), [(0, True), (0, False)])
        long = dict(old[1], links=[[1, 2]])  # row 1 with its links changed: refused later (``50`` AA2)
        self.assertEqual(match(old, [old[0], long, old[2], old[2]]), [(0, True), (1, True), (2, True), (2, False)])
        renamed = dict(old[1], name_hash=98)  # another name hash: still row 1 (never paired by a key)
        self.assertEqual(match(old, [old[0], renamed, old[2]]), [(0, True), (1, True), (2, True)])
        self.assertEqual(match(old, [x, old[1], old[2], old[2]]), [(0, True), (1, True), (2, True), (2, False)])


class Duration(unittest.TestCase):
    """AB5: the duration check follows the template's own keys."""

    def test_keys_ending_before_the_duration(self):
        """A long permutation: its last key, stored in the glTF as a float32, lies more than the tolerance
        before the duration (as the one switch animation of ``51``); re-encoded unedited it gives the stored bytes."""
        order = "<"
        keyframes = next(k for k in range(7700, 15000)
                         if abs(struct.unpack("<f", struct.pack("<f", (k - 1) / 30))[0] - (k - 1) / 30) > 1.2 * AI.TOLERANCE)
        last = keyframes - 1
        perm = B.simple_permutation(
            "switch", keyframes=keyframes,
            translation=[B.translation_blob([0, last], [(1.0, 2.0, 3.0), (3.0, 2.0, 1.0)], 4, keyframes, order), b""],
            rotation=[B.rotation_blob([0], [(0, 0, 0, 32767)], keyframes, order), b""],
            scale=[B.scale_blob([0, last], [127, 254], 1, keyframes, order), b""])
        ani = F.with_handle(B.build_ani("switch", [perm]), F.ANI, order)
        _, app, _ = T.animation_files("switch", F.switch_files())
        glb = T.animation_glb(ani, app, "switch")
        result = AI.import_animation(ani, "switch", app, "switch", glb, F.ANI, F.APP, force=True)
        self.assertEqual(result.data, ani)
        duration = animation.parse_ani(ani, "switch").permutations[0].duration
        end =struct.unpack("<f", struct.pack("<f", last / 30))[0]
        self.assertGreater(abs(end - duration), AI.TOLERANCE)
        edited = T.with_channel(glb, 0, 0, "translation", [0.0, end], [(1, 2, 3), (2, 2, 2)])
        changed = AI.import_animation(ani, "switch", app, "switch", edited, F.ANI, F.APP)
        new = animation.decode_translation(
            animation.parse_ani(changed.data, "switch").permutations[0].translation_blobs[0], keyframes, "<")
        self.assertEqual((new.frames, new.values[1]), ([0, last], (2.0, 2.0, 2.0)))
        with self.assertRaises(J.Refused) as caught:  # a range other than the template's channel
            AI.import_animation(ani, "switch", app, "switch",
                                T.with_channel(glb, 0, 0, "translation", [0.0, end / 2], [(1, 2, 3), (2, 2, 2)]),
                                F.ANI, F.APP)
        self.assertEqual(caught.exception.reason, "duration_changed")


class Bounded(unittest.TestCase):
    """AB6: at most 200 refused paths and fields per entry, with totals."""

    def test_rows(self):
        rows = [{"path": f"meshes[0].triangles[{i}].neighbours[0]", "reason": "derived_field"} for i in range(999, 0, -1)]
        rows.insert(0, {"path": "zz.first", "reason": "value_out_of_range", "message": "first"})
        found = AP._refused_rows(rows)
        self.assertEqual(len(found["refused_fields"]), AP.FIELDS_LISTED)
        self.assertEqual(found["refused_fields"][0]["path"], "zz.first")
        listed = [r["path"] for r in found["refused_fields"][1:]]
        self.assertEqual(listed, sorted(listed))
        self.assertEqual(found["refused_fields_total"], 1000)
        self.assertEqual(found["refused_by_reason"], {"derived_field": 999, "value_out_of_range": 1})

    def test_apply_row(self):
        setup = S.Setup(self)
        edit_json(setup.edit / f"data/acr/{F.ACR}.json", lambda d: d["fields"].__setitem__("bound_sphere_radius", 2.0))
        refusals = [J.Refused(("fields", f"f{i:04d}"), "derived_field" if i % 4 else "value_out_of_range")
                    for i in range(1000)]
        error = refusals[0]
        error.refusals = refusals
        with mock.patch.object(J, "import_json", side_effect=error):
            report, _ = setup.ok()
        (row,) = report["changed"]
        self.assertEqual((row["result"], row["reason"]), ("refused", "value_out_of_range"))
        self.assertEqual(len(row["refused_fields"]), 200)
        self.assertEqual(row["refused_fields"][0]["path"], "fields.f0000")
        self.assertEqual((row["refused_fields_total"], row["refused_by_reason"]),
                         (1000, {"derived_field": 750, "value_out_of_range": 250}))

    def test_fields_cut_in_path_order(self):
        setup = S.Setup(self)
        rows = [{"path": f"p{i:04d}", "old": 0, "new": 1} for i in range(500, 0, -1)]
        with mock.patch.object(AP._Apply, "import_json", lambda job, item, template, entry, target: (
                template + b"x", [], rows, [])), mock.patch.object(AP._Apply, "targets", lambda job, item, entry, t: (
                [("Common", F.name_of("acr", "Hero"))], [])):
            edit_json(setup.edit / f"data/acr/{F.ACR}.json", lambda d: d["fields"].__setitem__("bound_sphere_radius", 2.0))
            report, _ = setup.ok(dry_run=True)
        (row,) = report["changed"]
        self.assertEqual((row["fields_changed"], row["fields_total"], len(row["fields"])), (500, 500, 200))
        self.assertEqual(row["fields"][0]["path"], "p0001")


def segment_slots_end_early(test, data: bytes, segment: int) -> None:
    """Every slot before a slot that starts a segment ends with a stored size of 0 or < 7 bytes of room."""
    cpk = A.read_archive(data)
    offsets = cpk.sector_offsets
    starts = [k for k in range(1, len(offsets)) if offsets[k] % segment == 0 and offsets[k] != offsets[k - 1]]
    test.assertTrue(starts)
    comp = cpk.header.comp_sector
    for k in starts:
        packets = cpk._decode_sector(k - 1).packets
        last = packets[-1]
        test.assertFalse(last.cut)
        end = last.offset + 6 + last.stored
        room = cpk.slot_position(k - 1) + comp - end
        if room >= 7:
            test.assertEqual(struct.unpack_from(cpk.header.order + "H", data, end + 4)[0] & 0x7FFF, 0)


class Segments(unittest.TestCase):
    """AB7: the default layout as 0.21.1 for every ``jobs``; the segmented layout the same for any ``jobs``."""

    def setUp(self):
        self.files = {f"Base\\f{i:02d}.dat": (b"%05d " % i) * 9000 + Y.data(i, 30_000) for i in range(12)}

    def sources(self):
        yield "ps3", A.write_archive(self.files, version=6, order=">")
        yield "switch", A.write_archive(self.files, version=7)
        yield "x360", XA.write_x360_archive([XA.Piece(n, 0, d) for n, d in self.files.items()]).data

    def test_default_layout_unchanged(self):
        C28.same_zlib(self)
        for key in ("ps3/replace_small", "switch/replace_small", "x360/replace_small"):
            source = Y.SOURCES[key.split("/")[0]]()
            replace, add = Y.CHANGES["replace_small"]()
            for jobs in (1, 2, 8):
                written = IW.write_changed(source, None, replace, add, jobs=jobs)
                self.assertEqual((written.sha256, written.layout), (Y.GOLDEN_0_21_0[key], "default"), (key, jobs))

    def test_segmented(self):
        with mock.patch.object(A, "SEGMENT_SIZE", 120_000):
            for kind, source in self.sources():
                with self.subTest(kind):
                    outputs = {}
                    for jobs in (1, 2, 8):
                        buffer = io.BytesIO()
                        written = IW.write_changed(source, buffer, {"Base\\f03.dat": b"new" * 999}, jobs=jobs,
                                                   segment=True)
                        self.assertEqual(written.layout, "segmented")
                        outputs[jobs] = buffer.getvalue()
                    data = outputs[1]
                    self.assertEqual(outputs[2], data)
                    self.assertEqual(outputs[8], data)
                    self.assertNotEqual(data, IW.replace_file(source, "Base\\f03.dat", b"new" * 999))
                    self.assertTrue(A.verify_archive(data).ok)
                    cpk = A.read_archive(data)
                    self.assertGreater(cpk.header.stream_size, 5 * 120_000)
                    for name, content in self.files.items():
                        self.assertEqual(cpk.read_file(name), b"new" * 999 if name == "Base\\f03.dat" else content)
                    if kind != "x360":
                        segment_slots_end_early(self, data, 120_000)

    def test_x360_segments_start_new_pieces_and_slots(self):
        with mock.patch.object(A, "SEGMENT_SIZE", 120_000):
            source = dict(self.sources())["x360"]
            buffer = io.BytesIO()
            IW.write_changed(source, buffer, {}, {"Base\\extra.dat": b"e" * 10}, jobs=4, segment=True)
            data = buffer.getvalue()
        self.assertTrue(A.verify_archive(data).ok)
        cpk = A.read_archive(data)
        starts = sorted(set(cpk.locations))
        segment_starts, begun = [], 0
        for start in starts:
            if start - begun >= 120_000:
                segment_starts.append(start)
                begun = start
        self.assertGreater(len(segment_starts), 3)
        for start in segment_starts:  # each segment starts a new piece and a new slot
            k = cpk.sector_offsets.index(start)
            self.assertFalse(cpk._decode_sector(k - 1).packets[-1].cut)
        self.assertEqual(cpk.read_file("Base\\extra.dat"), b"e" * 10)

    def test_workers_statistics(self):
        source = A.write_archive(self.files, version=7)
        with mock.patch.object(A, "SEGMENT_SIZE", 120_000):
            written = IW.write_changed(source, None, {"Base\\f03.dat": b"x"}, jobs=3, segment=True)
        stats = written.workers
        self.assertEqual(set(stats), {"jobs", "submitted", "taken", "dropped", "no_room", "waited_seconds",
                                      "per_worker", "files_read_ahead", "files_read_by_packer"})  # ``52`` AC5, ``53`` AD3
        self.assertEqual(stats["jobs"], 3)
        self.assertGreater(stats["submitted"], 0)
        self.assertEqual(IW.write_changed(source, None, {"Base\\f03.dat": b"x"}, jobs=1).workers["submitted"], 0)

    def test_apply_segment_and_profile(self):
        setup = S.Setup(self)
        S.three_edits(setup)
        report, out = setup.ok("seg", segment=True, profile=True)
        self.assertTrue(report["options"]["segment"])
        for rel, info in report["archives"].items():
            self.assertEqual(info["layout"], "segmented")
            self.assertIn("workers", info)
            self.assertTrue(A.verify_archive((out / rel).read_bytes()).ok)
        plain, _ = setup.ok("plain")
        self.assertEqual({i["layout"] for i in plain["archives"].values()}, {"default"})
        self.assertFalse(any("workers" in i for i in plain["archives"].values()))

    def test_encrypted_packets_refused(self):
        with self.assertRaises(FormatError):
            A.write_archive_to(io.BytesIO(), {"a": b"x" * 100}, encrypt=lambda n: True, segment=True)


class Version(unittest.TestCase):
    def test_version(self):
        self.assertGreaterEqual(tuple(int(p) for p in d3sno.__version__.split(".")), (0, 22, 2))
        self.assertGreaterEqual(d3sno.STAGE, 22)


if __name__ == "__main__":
    unittest.main()
