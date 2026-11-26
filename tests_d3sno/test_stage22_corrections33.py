"""Corrections 33 (``53_corrections33.md``): AD1 to AD3. Synthetic games and archives of the test builders
only; every export and output goes to a temporary folder outside any git working tree."""

import _libpath  # noqa: F401  (python -I: the library folder on sys.path)
import copy
import io
import struct
import unittest

import builders as B
import export_fixture as F
import test_stage22_corrections32 as C32
import y1_cases as Y
import d3sno
from d3sno import archive as A
from d3sno import collision
from d3sno import collision_write as CW
from d3sno import index_write as IW
from d3sno import json_import as J
from d3sno import sno_layout as L
from d3sno import world
from d3sno.material_write import encode_tag_map

BUILDS = ("switch", "ps3", "x360")
MIDDLE = 12  # vertex (2, 2, z) of the 4 x 4 grid: its x row is about -1.5e-05 next to a centre of 2.000015


def shifted(build):
    """The grid mesh with its stored centre 1.5e-05 off on x: the x rows of the middle column are not those
    its exported values give."""
    return C32.collision_with(build, 1.5e-05)


def hexes(row):
    return [struct.pack(">f", v).hex().upper() for v in row[:3]]


class StoredVertices(unittest.TestCase):
    """AD1: vertices keep their stored values per coordinate; ``stored_vertices`` makes the undo exact."""

    def test_lossy_coordinate(self):
        """The case of the finding: the export does not identify the stored x of the middle vertex."""
        data = shifted("switch")
        mesh = collision.parse_collision(data, "switch").meshes[0]
        value = C32.phm_json(data, "switch")["meshes"][0]["vertices"][MIDDLE][0]
        self.assertNotEqual(J._nearest_row(value, mesh.centre[0]), mesh.vertices[MIDDLE][0])

    def test_exported(self):
        for build in BUILDS:
            with self.subTest(build):
                data = shifted(build)
                mesh = collision.parse_collision(data, build).meshes[0]
                rows = C32.phm_json(data, build)["meshes"][0]["stored_vertices"]
                self.assertEqual(rows, [hexes(r) for r in mesh.vertices])
                self.assertEqual([tuple(struct.unpack(">f", bytes.fromhex(v))[0] for v in row) for row in rows],
                                 [tuple(r[:3]) for r in mesh.vertices])

    def test_one_coordinate_edited_then_undone(self):
        for build in BUILDS:
            for axis in (0, 1):
                with self.subTest(build=build, axis=axis):
                    data = shifted(build)
                    original = C32.phm_json(data, build)
                    edited, _ = C32.phm_import(data, build, lambda d: d["meshes"][0]["vertices"][MIDDLE].__setitem__(axis, 2.25))
                    old = collision.parse_collision(data, build).meshes[0].vertices[MIDDLE]
                    new = collision.parse_collision(edited.data, build).meshes[0].vertices[MIDDLE]
                    for a in range(3):  # the other coordinates of the edited vertex keep their stored values
                        if a != axis:
                            self.assertEqual(new[a], old[a])
                    self.assertEqual(J.import_json("phm", edited.data, build, original).data, data)

    def test_refit_then_undone(self):
        """A vertex moved out of the bounds (a refit), then the original JSON: the template's bytes (Q976 of
        ``52`` no longer)."""
        for build in BUILDS:
            with self.subTest(build):
                data = shifted(build)
                original = C32.phm_json(data, build)
                edited, _ = C32.phm_import(data, build, C32.move_outside)
                self.assertIn("bounds_refitted: mesh 0", edited.warnings)
                back = J.import_json("phm", edited.data, build, original)
                self.assertEqual(back.data, data)
                self.assertEqual(back.warnings, [])

    def test_rebuild_then_undone(self):
        """A triangle added and the x of the middle vertex moved (a rebuild, in the stored bounds), then the
        original JSON (a rebuild again): the stored values of every vertex come back from ``stored_vertices``
        (the middle x is not identified by its exported value)."""
        for build in BUILDS:
            with self.subTest(build):
                data = shifted(build)
                original = C32.phm_json(data, build)

                def edit(d):
                    d["meshes"][0]["triangles"].append(
                        {"vertices": [0, 6, 5], "neighbours": [-1, -1, -1], "material": 3, "flags": 1})
                    d["meshes"][0]["vertices"][MIDDLE][0] = 2.25

                edited, _ = C32.phm_import(data, build, edit)
                back = J.import_json("phm", edited.data, build, original)
                self.assertEqual(collision.parse_collision(back.data, build).meshes[0].vertices,
                                 collision.parse_collision(data, build).meshes[0].vertices)

    def test_without_stored_vertices(self):
        for build in BUILDS:
            with self.subTest(build):
                data = shifted(build)

                def edit(d):
                    for mesh in d["meshes"]:
                        del mesh["stored_vertices"]
                    d["meshes"][0]["vertices"][MIDDLE][1] = 2.25

                result, _ = C32.phm_import(data, build, edit)
                old = collision.parse_collision(data, build).meshes[0].vertices[MIDDLE]
                new = collision.parse_collision(result.data, build).meshes[0].vertices[MIDDLE]
                self.assertEqual((new[0], new[2], new[3]), (old[0], old[2], old[3]))  # rule 1, per coordinate
                original = C32.phm_json(data, build)
                for mesh in original["meshes"]:
                    del mesh["stored_vertices"]
                self.assertEqual(J.import_json("phm", data, build, original).data, data)

    def test_row_that_does_not_give_the_value(self):
        for build in BUILDS:
            with self.subTest(build):
                data = shifted(build)
                mesh = collision.parse_collision(data, build).meshes[0]

                def edit(d):
                    d["meshes"][0]["stored_vertices"][MIDDLE][0] = "3F800000"  # 1.0: not the exported x

                result, document = C32.phm_import(data, build, edit)
                new = collision.parse_collision(result.data, build).meshes[0]
                value = document["meshes"][0]["vertices"][MIDDLE][0]
                self.assertEqual(new.vertices[MIDDLE][0], J._nearest_row(value, mesh.centre[0]))  # quantised anew
                self.assertEqual(new.vertices[MIDDLE][1:], mesh.vertices[MIDDLE][1:])  # that coordinate only
                self.assertEqual([r for k, r in enumerate(new.vertices) if k != MIDDLE],
                                 [r for k, r in enumerate(mesh.vertices) if k != MIDDLE])

    def test_bad_row_refused(self):
        data = shifted("switch")
        for bad in (["3F80", "0", "0"], ["3F800000", "3F800000"], "x", ["ZZZZZZZZ", "00000000", "00000000"]):
            with self.subTest(bad=bad), self.assertRaises(J.Refused) as caught:
                C32.phm_import(data, "switch", lambda d: d["meshes"][0]["stored_vertices"].__setitem__(0, bad))
            self.assertEqual(caught.exception.reason, "value_out_of_range")


def own_maps(build, kind):
    """Four markers, each with a tag map of its own, all with equal records (``rec``), or each a map of no
    records (``count0``, 4 bytes), or none (``none``)."""
    order = ">" if build in ("ps3", "x360") else "<"
    tags = {"rec": encode_tag_map([(2, 0x10, 5), (2, 0x11, 6)], build), "count0": struct.pack(order + "I", 0),
            "none": b""}[kind]
    markers = [{"hash": 0x601 + i, "type": 0, "group": 1, "handle": F.ACR, "position": (1.0 * i, 2.0, 0.0),
                "links": [(i, i + 1)], "tags": tags} for i in range(4)]
    return L.rebuild_sno(B.build_mrk(build, markers=markers, circles=[(1.0, 2.0, 3.0)], special=[1, 2, 3]), "mrk", build)


def map_blocks(data, build) -> list:
    """The tag map block of each marker (by its body offset; None for an empty pair)."""
    tree = L.parse_tree(data, "mrk", build)
    out = []
    for element in tree.top.pairs[0x0C].block.elements:
        value = element.pairs.get(0x30)
        out.append(value.block.source if value is not None and value.block is not None else None)
    return out


class TagMapPlacement(unittest.TestCase):
    """AD2: a tag map keeps the placement of the template: an equal map is never pooled."""

    def test_hash_change_and_added_marker(self):
        for build in BUILDS:
            for kind in ("rec", "count0", "none"):
                data = own_maps(build, kind)
                stored = map_blocks(data, build)
                if kind != "none":
                    self.assertEqual(len(set(stored)), 4)  # four blocks of their own
                original = J._json_round(J.template_json("mrk", data, build))
                for where, source, changed in ((4, 3, 0), (0, 0, 1), (2, 1, 1), (4, 2, 2)):
                    with self.subTest(build=build, kind=kind, where=where, source=source, changed=changed):
                        document = copy.deepcopy(original)
                        document["markers"][changed]["name_hash"] = 0x999
                        document["markers"].insert(where, copy.deepcopy(original["markers"][source]))
                        first = J.import_json("mrk", data, build, document).data
                        blocks = map_blocks(first, build)
                        if kind != "none":
                            self.assertEqual(len(set(blocks)), 5, blocks)  # the copy's map is its own too
                        self.assertEqual(J.import_json("mrk", first, build, original).data, data)

    def test_moved_copy_with_a_changed_leaf(self):
        """A new marker (a copy with another position) beside a renamed one, and back."""
        for build in BUILDS:
            data = own_maps(build, "rec")
            original = J._json_round(J.template_json("mrk", data, build))
            document = copy.deepcopy(original)
            document["markers"][1]["name_hash"] = 0x998
            new = copy.deepcopy(original["markers"][1])
            new["position"] = [8.0, 8.0, 8.0]
            document["markers"].append(new)
            first = J.import_json("mrk", data, build, document).data
            self.assertEqual(len(set(map_blocks(first, build))), 5)
            self.assertEqual(J.import_json("mrk", first, build, original).data, data)

    def test_shared_maps_stay_shared(self):
        """A map the template shares between markers (W11-6) is shared by a copy too (``51`` AB4)."""
        data = C32.C31.shared_tag_markers("switch")
        original = J._json_round(J.template_json("mrk", data, "switch"))
        document = copy.deepcopy(original)
        document["markers"].append(copy.deepcopy(original["markers"][2]))
        first = J.import_json("mrk", data, "switch", document).data
        self.assertEqual(len(set(map_blocks(first, "switch"))), 1)
        self.assertEqual(J.import_json("mrk", first, "switch", original).data, data)


def small_files_archive(version, order, large=2):
    files = {f"Base\\big{i}.dat": Y.data(40 + i, 1_200_000) for i in range(large)}  # first; many sectors
    files.update({f"Base\\d{i // 100:02d}\\f{i:04d}.dat": Y.data(1000 + i, 200 + i % 1500) for i in range(1500)})
    files["Base\\small.txt"] = b"small"
    return files, A.write_archive(files, version=version, order=order)


def written(source, jobs):
    buffer = io.BytesIO()
    result = IW.write_changed(source, buffer, {"Base\\small.txt": b"x"}, jobs=jobs)
    return buffer.getvalue(), result.workers


class Jobs(unittest.TestCase):
    """AD3: small files are read by the packer itself; the workers take only the larger files."""

    def test_same_bytes_and_counts(self):
        for version, order in ((6, ">"), (7, "<")):
            with self.subTest(version=version):
                files, source = small_files_archive(version, order)
                one, _ = written(source, 1)
                many, stats = written(source, 12)
                self.assertEqual(one, many)
                read = [n for n, d in files.items() if n != "Base\\small.txt" and d]
                large = [n for n in read if len(files[n]) >= A.AHEAD_MIN_FILE]
                self.assertEqual(stats["files_read_ahead"] + stats["files_read_by_packer"], len(read))
                self.assertEqual(stats["files_read_ahead"], len(large))
                self.assertEqual(stats["submitted"], stats["taken"] + stats["dropped"])
                self.assertEqual(sum(row["done"] for row in stats["per_worker"]) <= stats["submitted"], True)

    def test_small_files_only(self):
        """Without a large file, nothing is read ahead while packing (the counting pass still is)."""
        files, source = small_files_archive(7, "<", large=0)
        submitted = []
        plan = A._SectorsAhead.plan

        def recording(self, slot):
            before = self.workers.submitted
            plan(self, slot)
            if self.current is not None:
                submitted.append(self.workers.submitted - before)

        from unittest import mock
        with mock.patch.object(A._SectorsAhead, "plan", recording):
            _, stats = written(source, 8)
        self.assertEqual(sum(submitted), 0)
        self.assertEqual(stats["files_read_ahead"], 0)
        self.assertGreater(stats["files_read_by_packer"], 1000)

    def test_work_ahead_stays_in_the_large_file(self):
        """The workers inflate only the sectors of the large file being read, never past its last one."""
        files, source = small_files_archive(7, "<", large=1)
        cpk = A.read_archive(source)
        big = cpk.find("Base\\big0.dat")
        start = cpk.locations[big.first]
        last = A.bisect.bisect_right(cpk.sector_offsets, start + big.size - 1) - 1
        planned = []
        plan = A._SectorsAhead.plan

        def recording(self, slot):
            before = set(self.futures)
            plan(self, slot)
            if self.current is not None:
                planned.extend(b for b in self.futures if b not in before)

        from unittest import mock
        with mock.patch.object(A._SectorsAhead, "plan", recording):
            written(source, 8)
        self.assertTrue(planned)
        self.assertLessEqual(max(planned) * A.SECTOR_BATCH, last)

    def test_each_file_counted_once(self):
        _, source = small_files_archive(7, "<", large=0)
        cpk = A.read_archive(source)
        ahead = A._SectorsAhead(cpk, A.Workers(1))
        for offset, size in ((0, 10), (100, 10), (0, 10), (200, A.AHEAD_MIN_FILE)):
            ahead.file(offset, size)
        self.assertEqual((ahead.files_self, ahead.files_ahead), (2, 1))

    def test_profile_text(self):
        _, source = small_files_archive(7, "<")
        _, stats = written(source, 4)
        text = A.workers_text(stats)
        self.assertIn("files_read_ahead", text)
        self.assertIn("files_read_by_packer", text)


class Version(unittest.TestCase):
    def test_version(self):
        self.assertGreaterEqual(tuple(int(p) for p in d3sno.__version__.split(".")), (0, 22, 4))
        self.assertEqual(d3sno.STAGE, 22)


if __name__ == "__main__":
    unittest.main()
