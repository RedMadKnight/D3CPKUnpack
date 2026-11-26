"""Corrections 32 (``52_corrections32.md``): AC1 to AC6. Synthetic games and archives of the test builders
only; every export and output goes to a temporary folder outside any git working tree."""

import _libpath  # noqa: F401  (python -I: the library folder on sys.path)
import copy
import json
import math
import struct
import threading
import tracemalloc
import unittest

import builders as B
import export_fixture as F
import test_stage21 as S
import test_stage22 as T
import test_stage22_corrections31 as C31
import y1_cases as Y
import d3sno
from d3sno import archive as A
from d3sno import collision
from d3sno import collision_write as CW
from d3sno import index
from d3sno import index_write as IW
from d3sno import json_import as J
from d3sno import navigation
from d3sno import sno_layout as L
from d3sno import world
from d3sno.binary import ByteView
from d3sno.material_write import encode_tag_map
from d3sno.sno import group_by_extension

BUILDS = ("switch", "ps3", "x360")


def phm_json(data, build):
    return J._json_round(J.template_json("phm", data, build))


def collision_with(build, centre_shift=0.0, w=0.0):
    """The grid mesh of ``51`` and a one-triangle mesh; the first mesh's stored centre moved by
    ``centre_shift`` on x (its vertex rows moved back, so it is not the centre W14-12 derives) and its
    vertex rows' fourth word set to ``w``."""
    _, _, mesh = C31.grid_mesh(build)
    c = mesh.centre
    centre = (CW.f32(c[0] + centre_shift), c[1], c[2])
    rows = [(CW.f32(r[0] - centre_shift) if centre_shift else r[0], r[1], r[2], w) for r in mesh.vertices]
    positions = [(r[0] + centre[0], r[1] + centre[1], r[2] + centre[2]) for r in rows]
    first = collision.CollisionMesh(centre, mesh.half_extent, mesh.box_scale, mesh.tree_height, mesh.triangle_count,
                                    mesh.flags, mesh.material, positions, mesh.table, mesh.nodes, mesh.triangles, rows)
    second = CW.build_collision_mesh([(5.0, 5.0, 0.0), (6.0, 5.0, 0.0), (5.0, 6.0, 1.0)], [(0, 1, 2)], 7, 0, build)
    return CW.write_collision(CW.new_collision_file([first, second], build, handle=F.PHM))


def phm_import(data, build, edit):
    document = phm_json(data, build)
    edit(document)
    return J.import_json("phm", data, build, document), document


def move_inside(d):
    d["meshes"][0]["vertices"][6][0] = 1.25  # vertex 6 is (1, 1, z): inside the bounds


def move_outside(d):
    d["meshes"][0]["vertices"][24][2] = 3.0  # a corner above the bounds


def encloses(node, rows, tris, scale, build, left=None, right=None) -> bool:
    if build == "ps3":  # an inner node's split words: a above its left child, b below its right child
        if node.is_leaf:
            return True
        a = node.a - 0x10000 if node.a >= 0x8000 else node.a
        b = node.b - 0x10000 if node.b >= 0x8000 else node.b
        s = scale[node.axis]
        return (a * s >= max(rows[v][node.axis] for t in left for v in t)
                and b * s <= min(rows[v][node.axis] for t in right for v in t))
    lo = [min(rows[v][k] for t in tris for v in t) for k in range(3)]
    hi = [max(rows[v][k] for t in tris for v in t) for k in range(3)]
    return all(node.lower[k] * scale[k] <= lo[k] and node.upper[k] * scale[k] >= hi[k] for k in range(3))


class Refit(unittest.TestCase):
    """AC1: a refit changes the least it can; the undo gives the stored bytes."""

    def test_moved_inside_keeps_bounds_scale_and_boxes(self):
        for build in BUILDS:
            with self.subTest(build):
                data = C31.collision_file(build)
                result, _ = phm_import(data, build, move_inside)
                self.assertEqual(result.warnings, [])
                old = collision.parse_collision(data, build).meshes[0]
                new = collision.parse_collision(result.data, build).meshes[0]
                self.assertEqual((new.centre, new.half_extent, new.box_scale), (old.centre, old.half_extent, old.box_scale))
                runs = CW.leaf_runs(old.nodes, build)
                kept = 0
                for i, (a, b) in enumerate(zip(old.nodes, new.nodes)):
                    tris = [new.table[e].vertices for e in runs[i]]
                    left = right = None
                    if not a.is_leaf:
                        left = [new.table[e].vertices for e in runs[i + 1]]
                        right = [new.table[e].vertices for e in runs[i + (a.data >> (2 if build == "ps3" else 6))]]
                    if a == b:
                        kept += 1
                    else:  # only a node whose stored box no longer encloses its triangles is fitted again
                        self.assertFalse(encloses(a, new.vertices, tris, new.box_scale, build, left, right), (build, i))
                self.assertGreater(kept, len(old.nodes) // 2)
                self.assertEqual([r for k, r in enumerate(new.vertices) if k != 6],
                                 [r for k, r in enumerate(old.vertices) if k != 6])

    def test_moved_outside_derives_the_bounds(self):
        for build in BUILDS:
            with self.subTest(build):
                data = C31.collision_file(build)
                result, document = phm_import(data, build, move_outside)
                self.assertEqual(result.warnings, ["bounds_refitted: mesh 0"])
                new = collision.parse_collision(result.data, build).meshes[0]
                positions = [tuple(v) for v in document["meshes"][0]["vertices"]]
                centre, _, half, scale = CW.mesh_bounds(positions, build)
                self.assertEqual((new.centre, new.half_extent, new.box_scale), (tuple(centre), tuple(half), tuple(scale)))
                self.assertEqual(new.positions[24][2], 3.0)

    def test_undo_gives_the_template(self):
        """Edit, then the original JSON: the template's bytes, also where the stored centre is not the one
        W14-12 derives (1.5e-05 off). An edit that refits the bounds is undone exactly where every vertex row
        follows from its exported position (not with the shifted centre: a row of about -1.5e-05 next to a
        centre of 2.000015 is lost in the float32 position the export writes, Q976)."""
        for build in BUILDS:
            for shift in (0.0, 1.5e-05):
                data = collision_with(build, shift)
                original = phm_json(data, build)
                self.assertEqual(J.import_json("phm", data, build, original).data, data)  # not refused
                for edit in (move_inside, move_outside) if not shift else (move_inside,):
                    with self.subTest(build=build, shift=shift, edit=edit.__name__):
                        edited, _ = phm_import(data, build, edit)
                        self.assertNotEqual(edited.data, data)
                        back = J.import_json("phm", edited.data, build, original)
                        self.assertEqual(back.data, data)
                        self.assertEqual(back.warnings, [])

    def test_given_boxes(self):
        for build in BUILDS:
            with self.subTest(build):
                data = C31.collision_file(build)
                mesh = collision.parse_collision(data, build).meshes[0]
                if build == "ps3":
                    inner = next(i for i, n in enumerate(mesh.nodes) if not n.is_leaf)
                    bad, wide = [0x8000, 0x7FFF], None
                else:
                    inner = next(i for i, n in enumerate(mesh.nodes) if n.is_leaf)
                    bad = [0, 0, 0, 0, 0, 0]
                    wide = [*(v - 5 for v in mesh.nodes[inner].lower), *(v + 5 for v in mesh.nodes[inner].upper)]
                # a row that does not enclose is fitted again (here: the builder's box, so the template)
                result, _ = phm_import(data, build, lambda d: d["meshes"][0]["tree"]["boxes"].__setitem__(inner, bad))
                self.assertEqual(result.data, data)
                if wide is not None:  # a row that encloses is written as given
                    result, _ = phm_import(data, build, lambda d: d["meshes"][0]["tree"]["boxes"].__setitem__(inner, wide))
                    node = collision.parse_collision(result.data, build).meshes[0].nodes[inner]
                    self.assertEqual([*node.lower, *node.upper], wide)
                    self.assertEqual(phm_json(result.data, build)["meshes"][0]["tree"]["boxes"][inner], wide)
                with self.assertRaises(J.Refused) as caught:
                    phm_import(data, build, lambda d: d["meshes"][0]["tree"]["boxes"].pop())
                self.assertEqual(caught.exception.reason, "list_length_fixed")

    def test_boxes_exported(self):
        for build in BUILDS:
            with self.subTest(build):
                data = C31.collision_file(build)
                mesh = collision.parse_collision(data, build).meshes[0]
                boxes = phm_json(data, build)["meshes"][0]["tree"]["boxes"]
                self.assertEqual(len(boxes), len(mesh.nodes))
                for row, node in zip(boxes, mesh.nodes):
                    if build == "ps3":
                        self.assertEqual(row, None if node.is_leaf else [node.a, node.b])
                    else:
                        self.assertEqual(row, [*node.lower, *node.upper])

    def test_unedited_vertices_keep_their_stored_rows(self):
        for build in BUILDS:
            with self.subTest(build):
                data = collision_with(build, w=1.0)  # rows a new quantisation would not give (w 0)
                old = collision.parse_collision(data, build).meshes[0]
                inside, _ = phm_import(data, build, move_inside)
                new = collision.parse_collision(inside.data, build).meshes[0]
                self.assertEqual([r for k, r in enumerate(new.vertices) if k != 6],
                                 [r for k, r in enumerate(old.vertices) if k != 6])
                self.assertEqual(new.vertices[6][1:], old.vertices[6][1:])  # ``53`` AD1: per coordinate, fourth word kept
                outside, _ = phm_import(data, build, move_outside)  # new bounds: the coordinates quantised anew
                rows = collision.parse_collision(outside.data, build).meshes[0].vertices
                self.assertTrue(all(r[3] == 1.0 for r in rows))  # ``53`` AD1: the fourth word is kept

    def test_older_export_without_boxes(self):
        for build in BUILDS:
            with self.subTest(build):
                data = C31.collision_file(build)
                with_boxes, _ = phm_import(data, build, move_inside)

                def edit(d):
                    for mesh in d["meshes"]:
                        del mesh["tree"]
                    move_inside(d)

                without, _ = phm_import(data, build, edit)
                self.assertEqual(without.data, with_boxes.data)
                original = phm_json(data, build)
                for mesh in original["meshes"]:
                    del mesh["tree"]
                self.assertEqual(J.import_json("phm", data, build, original).data, data)


class RefitDetails(unittest.TestCase):
    """AC1: the nearest row, a parent fitted again around a refitted child, a rebuild in the given bounds."""

    def test_nearest_row_gives_the_exported_value(self):
        centre, value = 125.63050842285156, 114.95602  # f32(value - centre) does not export as value
        base = CW.f32(value - centre)
        self.assertNotEqual(CW.f32(base + centre), CW.f32(value))
        row = J._nearest_row(value, centre)
        self.assertEqual(CW.f32(row + centre), CW.f32(value))
        self.assertLessEqual(abs(row - base), 2 * abs(J._f32_step(base, True) - base))

    def test_parent_fitted_around_a_refitted_child(self):
        for build in ("switch", "x360"):
            with self.subTest(build):
                data = C31.collision_file(build)
                mesh = collision.parse_collision(data, build).meshes[0]
                runs = CW.leaf_runs(mesh.nodes, build)
                parent = next(i for i, n in enumerate(mesh.nodes) if not n.is_leaf and mesh.nodes[i + 1].is_leaf)
                rows, scale = mesh.vertices, mesh.box_scale
                tris = [mesh.table[e].vertices for e in runs[parent]]
                tight = [*(int(math.floor(min(rows[v][k] for t in tris for v in t) / scale[k])) for k in range(3)),
                         *(int(math.ceil(max(rows[v][k] for t in tris for v in t) / scale[k])) for k in range(3))]

                def edit(d):
                    boxes = d["meshes"][0]["tree"]["boxes"]
                    boxes[parent + 1] = [0, 0, 0, 0, 0, 0]  # the leaf: fitted again, with its padding
                    boxes[parent] = tight  # encloses the triangles, not the leaf's new box

                result, _ = phm_import(data, build, edit)
                new = collision.parse_collision(result.data, build).meshes[0].nodes
                child, top = new[parent + 1], new[parent]
                self.assertEqual((child.lower, child.upper), (mesh.nodes[parent + 1].lower, mesh.nodes[parent + 1].upper))
                self.assertNotEqual([*top.lower, *top.upper], tight)
                self.assertTrue(all(top.lower[k] <= child.lower[k] and top.upper[k] >= child.upper[k] for k in range(3)))

    def test_rebuild_keeps_enclosing_bounds(self):
        for build in BUILDS:
            with self.subTest(build):
                data = collision_with(build, 1.5e-05)  # stored bounds other than W14-12 derives
                old = collision.parse_collision(data, build).meshes[0]
                result, _ = phm_import(data, build, lambda d: d["meshes"][0]["triangles"].append(
                    {"vertices": [0, 6, 5], "neighbours": [-1, -1, -1], "material": 3, "flags": 1}))
                new = collision.parse_collision(result.data, build).meshes[0]
                self.assertEqual(new.triangle_count, old.triangle_count + 1)  # rebuilt
                self.assertEqual((new.centre, new.half_extent, new.box_scale), (old.centre, old.half_extent, old.box_scale))
                self.assertNotIn("bounds_refitted: mesh 0", result.warnings)


class DerivedLeaves(unittest.TestCase):
    """AC2: a derived leaf is compared with the value derived from the edited JSON."""

    def setUp(self):
        self.data = T.template("scn")
        self.original = T.exported("scn", self.data)

    @staticmethod
    def cell(d, value):
        d["navigation"]["cells"][0]["neighbours"][1]["cell"] = value

    def test_cell_alone_border_derived(self):
        result, _ = T.run("scn", self.data, lambda d: self.cell(d, 32769))
        self.assertEqual(navigation.parse_scene(result.data, T.BUILD).neighbours[1].cell, 32769)
        self.assertEqual(T.exported("scn", result.data)["navigation"]["cells"][0]["neighbours"][1]["border"], 1)

    def test_cell_and_border_together(self):
        """The undo of a cell edit: the original cell and border, both other than the template's."""
        edited, _ = T.run("scn", self.data, lambda d: self.cell(d, 32769))
        back = J.import_json("scn", edited.data, T.BUILD, self.original)
        self.assertEqual(back.data, self.data)

        def both(d):
            self.cell(d, 32770)
            d["navigation"]["cells"][0]["neighbours"][1]["border"] = 2

        result, _ = T.run("scn", self.data, both)
        self.assertEqual(navigation.parse_scene(result.data, T.BUILD).neighbours[1].cell, 32770)

    def test_border_alone_refused(self):
        with self.assertRaises(J.Refused) as caught:
            T.run("scn", self.data, lambda d: d["navigation"]["cells"][0]["neighbours"][1].__setitem__("border", 5))
        self.assertEqual((caught.exception.reason, J.path_text(caught.exception.path)),
                         ("derived_field", "navigation.cells[0].neighbours[1].border"))
        self.assertIn("0", caught.exception.message)  # the derived value

    def test_two_views_disagree(self):
        def edit(d):
            self.cell(d, 32769)
            d["navigation"]["neighbours"][1]["cell"] = 32770

        with self.assertRaises(J.Refused) as caught:
            T.run("scn", self.data, edit)
        self.assertEqual(caught.exception.reason, "derived_field")
        self.assertIn("32769", caught.exception.message)

    def test_geometry_still_refused(self):
        with self.assertRaises(J.Refused) as caught:
            T.run("scn", self.data, lambda d: d["navigation"]["cells"][0]["min"].__setitem__(0, 9.0))
        self.assertEqual(caught.exception.reason, "navigation_geometry")

    def test_other_group(self):
        """A derived leaf of another group (a marker set's ``always_active``) changed alone: refused with the
        derived value; left as it was while its sources change: applied."""
        data = C31.shared_tag_markers("switch")
        document = J._json_round(J.template_json("mrk", data, "switch"))
        derived = document["always_active"]
        document["always_active"] = "something else"
        with self.assertRaises(J.Refused) as caught:
            J.import_json("mrk", data, "switch", document)
        self.assertEqual(caught.exception.reason, "derived_field")
        self.assertIn(json.dumps(derived), caught.exception.message)


def formula_tags(order, tag):
    """A tag map of one formula record (``power._read_formula``): one the writer does not reproduce."""
    record = struct.pack(order + "II", 4, tag) + struct.pack(order + "8I", 0, 0, 0, 0, 0, 2, 0, 4) + b"1\0\0\0" + bytes(4)
    return struct.pack(order + "I", 1) + record


def formula_markers(build):
    order = ">" if build in ("ps3", "x360") else "<"
    markers = [{"hash": 0x301 + i, "type": 0, "group": 1, "handle": F.ACR, "position": (1.0 * i, 2.0, 0.0),
                "links": [(i, i + 1)], "tags": formula_tags(order, 0x40 + i)} for i in range(3)]
    return L.rebuild_sno(B.build_mrk(build, markers=markers, circles=[(1.0, 2.0, 3.0)], special=[1, 2, 3]), "mrk", build)


def tags_key(marker: dict) -> str:
    (key,) = [k for k, v in marker.items() if isinstance(v, dict) and "records" in v]
    return key


class ListPairing(unittest.TestCase):
    """AC3: rows paired by position; equal rows written from their stored bytes."""

    def test_hash_and_copy_with_unreproduced_tag_maps(self):
        for build in BUILDS:
            data = formula_markers(build)
            original = J._json_round(J.template_json("mrk", data, build))
            key = tags_key(original["markers"][0])
            with self.assertRaises(J.Refused):  # the writer cannot write these tag maps anew
                document = copy.deepcopy(original)
                document["markers"][0][key]["records"][0]["tag"] = 0x99
                J.import_json("mrk", data, build, document)
            for where in (3, 1, 0):
                with self.subTest(build=build, where=where):
                    document = copy.deepcopy(original)
                    document["markers"][0]["name_hash"] = 0x999
                    document["markers"].insert(where, copy.deepcopy(original["markers"][2]))
                    first = J.import_json("mrk", data, build, document)
                    markers = world.parse_marker_set(first.data, build).markers
                    self.assertEqual(len(markers), 4)
                    self.assertEqual(J._json_round(J.template_json("mrk", first.data, build))["markers"], document["markers"])
                    self.assertEqual(J.import_json("mrk", first.data, build, original).data, data)

    def test_record_count_still_refused(self):
        data = C31.shared_tag_markers("switch")
        document = J._json_round(J.template_json("mrk", data, "switch"))
        key = tags_key(document["markers"][0])
        records = document["markers"][0][key]["records"]
        records.append(dict(records[0], tag=0x11))
        with self.assertRaises(J.Refused) as caught:
            J.import_json("mrk", data, "switch", document)
        self.assertEqual(caught.exception.reason, "list_length_fixed")


def padded_markers(build):
    """A canonical marker set of four markers, each with a tag map of two records (28 bytes, 4-aligned) and
    links: the layout holds padding after the marker array (before each links array, 8-aligned)."""
    markers = [{"hash": 0x401 + i, "type": 0, "group": 1, "handle": F.ACR, "position": (1.0 * i, 2.0, 0.0),
                "links": [(i, i + 1)], "tags": encode_tag_map([(2, 0x10, i), (2, 0x11, 7)], build)} for i in range(4)]
    return L.rebuild_sno(B.build_mrk(build, markers=markers, circles=[(1.0, 2.0, 3.0)], special=[1, 2, 3]), "mrk", build)


def padding(data, build) -> int:
    """The bytes of the body that no block holds (alignment padding)."""
    tree = L.parse_tree(data, "mrk", build)
    spans = []

    def walk(record):
        for value in record.pairs.values():
            if value is not None and value.block is not None and value.block.source is not None and not value.shared:
                spans.append((value.block.source, value.block.size()))
                for element in value.block.elements or []:
                    walk(element)

    walk(tree.top)
    spans.sort()
    return sum(max(0, b[0] - (a[0] + a[1])) for a, b in zip(spans, spans[1:]))


class Layout(unittest.TestCase):
    """AC4: a list change and a leaf change, then the reverse: the stored bytes, padding included."""

    def test_padding_after_the_marker_array(self):
        for build in BUILDS:
            data = padded_markers(build)
            self.assertGreater(padding(data, build), 0)
            original = J._json_round(J.template_json("mrk", data, build))
            for where, changed, source in ((4, 0, 3), (0, 1, 1), (2, 3, 0), (1, 2, 2)):
                with self.subTest(build=build, where=where, changed=changed, source=source):
                    document = copy.deepcopy(original)
                    document["markers"][changed]["name_hash"] = 0x777
                    new = copy.deepcopy(original["markers"][source])
                    new["position"] = [3.0, 3.0, 3.0]
                    document["markers"].insert(where, new)
                    first = J.import_json("mrk", data, build, document).data
                    self.assertEqual(len(world.parse_marker_set(first, build).markers), 5)
                    back = J.import_json("mrk", first, build, original).data
                    self.assertEqual(back, data)
            document = copy.deepcopy(original)  # a marker removed and the original put back
            del document["markers"][1]
            first = J.import_json("mrk", data, build, document).data
            self.assertEqual(J.import_json("mrk", first, build, original).data, data)


def _written(source, jobs):
    import io

    buffer = io.BytesIO()
    IW.write_changed(source, buffer, {"Base\\small.txt": b"x"}, jobs=jobs)
    return buffer.getvalue()


class Jobs(unittest.TestCase):
    """AC5: without ``--segment`` the workers only read and inflate ahead; the bytes and memory are bounded."""

    def test_bytes_for_any_jobs(self):
        for name in ("ps3", "switch", "x360"):
            with self.subTest(name):
                source = Y.SOURCES[name]()
                self.assertEqual(len({_written(source, jobs) for jobs in (1, 2, 8)}), 1)

    def test_memory_bound(self):
        """Peak memory with 8 workers within 2 x the one-worker peak plus 2 x jobs source files."""
        files = {f"Base\\f{i:02d}.dat": (b"%07d some text " % i) * 50_000 for i in range(16)}
        files["Base\\small.txt"] = b"small"
        source = A.write_archive(files, version=7)
        largest = max(len(v) for v in files.values())
        peaks = {}
        for jobs in (1, 8):
            tracemalloc.start()
            try:
                written = IW.write_changed(source, None, {"Base\\small.txt": b"x"}, jobs=jobs)
                peaks[jobs] = tracemalloc.get_traced_memory()[1]
            finally:
                tracemalloc.stop()
            self.assertIsNotNone(written)
        self.assertLessEqual(peaks[8], 2 * peaks[1] + 2 * 8 * largest, peaks)

    def test_dropped_work_is_let_go(self):
        gate = threading.Event()
        workers = A.Workers(2)  # 4 in flight at most
        try:
            started = threading.Barrier(3, timeout=5)

            def slow():
                started.wait()
                gate.wait(5)
                return bytes(1000)

            running = [workers.submit(slow) for _ in range(2)]
            started.wait()  # both run
            queued = [workers.submit(lambda: 2) for _ in range(2)]
            self.assertTrue(all(running + queued))
            for future in running:
                workers.drop(future)  # running: not cancelled, still held in flight
            self.assertEqual(len(workers.dropped), 2)
            self.assertIsNone(workers.submit(lambda: 3))  # two in flight and two dropped but running
            gate.set()
            for future in running + queued:
                future.result(5)
            self.assertIsNotNone(workers.submit(lambda: 4))
            self.assertEqual(workers.dropped, [])  # ended: let go with their results
        finally:
            gate.set()
            workers.close()
        stats = workers.stats()
        self.assertEqual((stats["dropped"], stats["no_room"]), (2, 1))
        self.assertEqual(sum(row["done"] for row in stats["per_worker"]), 5)
        self.assertLessEqual(len(stats["per_worker"]), 2)

    def test_profile_per_worker(self):
        files = {f"Base\\f{i}.dat": Y.data(i, 300_000) for i in range(4)}
        files["Base\\small.txt"] = b"small"
        source = A.write_archive(files, version=7)
        stats = IW.write_changed(source, None, {"Base\\small.txt": b"x"}, jobs=3).workers
        self.assertGreater(stats["submitted"], 0)
        self.assertTrue(stats["per_worker"])
        self.assertIn("worker 0 done", A.workers_text(stats))


def extra_scene_files(count):
    """The canonical switch game of ``51`` with ``count`` more scenes that no world refers to."""
    files = C31.canonical_files()
    scn = files["Act1"][F.name_of("scn", "Room")]
    toc = index.parse_core_toc(files["Common"]["CoreTOC.dat"], "switch")
    entries = list(toc.entries)
    for i in range(count):
        handle = F.SCN + 100 + i
        files["Act1"][F.name_of("scn", f"Extra{i}")] = F.with_handle(scn, handle, "<")
        entries.append(index.TocEntry(group_by_extension("scn").id, handle, f"Extra{i}"))
    files["Common"]["CoreTOC.dat"] = IW.write_core_toc(entries, "switch")
    return files


class LayoutMemory(unittest.TestCase):
    """AC6: only the scenes and marker sets the worlds refer to are held, in parsed form."""

    def test_only_referenced_are_held(self):
        setup = S.Setup(self, extra_scene_files(5))
        C31.edit_json(setup.edit / f"data/mrk/{F.MRK}.json", C31.added_marker)
        C31.edit_json(setup.edit / f"data/wrl/{F.WRL}.json", lambda d: d["music"].__setitem__("handle", 555))
        for i in range(5):
            C31.edit_json(setup.edit / f"data/scn/{F.SCN + 100 + i}.json",
                          lambda d: d["navigation"]["cells"][1]["flags"].__setitem__("value", 7))
        report, _ = setup.ok("out", profile=True)
        results = {r["path"]: r["result"] for r in report["changed"]}
        self.assertEqual(sorted(set(results.values())), ["applied"], results)
        self.assertEqual(len(results), 7)
        profile = report["profile"]
        self.assertEqual(profile["layout_held"], 1)  # the marker set; the five scenes are not referenced
        self.assertGreaterEqual(profile["layout_referenced"], 2)
        self.assertTrue(profile["peak_memory_bytes"] is None or profile["peak_memory_bytes"] > 0)
        plain, _ = setup.ok("plain")
        self.assertNotIn("profile", plain)

    def test_held_parts_are_parsed(self):
        setup = S.Setup(self, C31.canonical_files())
        C31.edit_json(setup.edit / f"data/mrk/{F.MRK}.json", C31.added_marker)
        C31.edit_json(setup.edit / f"data/wrl/{F.WRL}.json", lambda d: d["music"].__setitem__("handle", 555))
        held = []
        original = d3sno.apply._Apply.write_archives

        def record(job):
            held.extend(job.layout_parts.values())
            return original(job)

        from unittest import mock
        with mock.patch.object(d3sno.apply._Apply, "write_archives", record):
            setup.ok("out")
        self.assertEqual([type(h) for h in held], [world.MarkerSet])


class LayoutReferences(unittest.TestCase):
    def test_world_marker_sets_and_scene_sets(self):
        """The scenes of a world's entries, its own marker sets (stored and edited) and the marker sets of those
        scenes are referenced."""
        from types import SimpleNamespace
        from unittest import mock

        target = F.temp_root(self) / "w.json"
        target.write_text(json.dumps({"scene_entries": [{"scene": {"group": "scn", "handle": 6}}],
                                      "marker_sets": [{"group": "mrk", "handle": 88}]}), encoding="utf-8")
        scene = SimpleNamespace(marker_sets=[99])
        job = SimpleNamespace(only=list(d3sno.apply.KINDS), run=SimpleNamespace(build="switch"),
                              template=lambda entry: (None, b""),
                              scenes_and_sets=lambda after=False: (lambda h: scene if h == 5 else None, None))
        stored = SimpleNamespace(scene_entries=[SimpleNamespace(handle=5)], marker_sets=[77])
        with mock.patch.object(world, "parse_world", lambda data, build: stored):
            d3sno.apply._Apply.layout_references(job, [({"group": "wrl", "handle": 1, "path": "data/wrl/1.json"}, target)])
        self.assertEqual(job.layout_refs, {("scn", 5), ("scn", 6), ("mrk", 77), ("mrk", 88), ("mrk", 99)})


class Version(unittest.TestCase):
    def test_version(self):
        major, minor, patch = (int(p) for p in d3sno.__version__.split("."))
        self.assertGreaterEqual((major, minor, patch), (0, 22, 3))
        self.assertEqual(d3sno.STAGE, 22)


if __name__ == "__main__":
    unittest.main()
