"""Corrections 34 (``54_corrections34.md``): AE1, tag maps of marker sets written from scratch and of added
rows follow the build. Synthetic marker sets of the test builders only; nothing is written to disk."""

import _libpath  # noqa: F401  (python -I: the library folder on sys.path)
import copy
import unittest

import builders as B
import export_fixture as F
import test_stage22_corrections33 as C33
import d3sno
from d3sno import json_import as J
from d3sno import sno_layout as L
from d3sno import world
from d3sno.material_write import encode_tag_map

BUILDS = ("switch", "x360", "ps3")
MAPS = ([(2, 0x10, 5)], [(2, 0x11, 6), (2, 0x12, 7)], [(2, 0x13, 8)])
KINDS = (0, 1, 0, 2, 1, 0)  # the map records of each marker: three equal to an earlier one


def marker_set(build, kinds=KINDS, pooled=None):
    """Markers with the tag maps ``MAPS[k]``. ``pooled`` True: a marker whose map equals an earlier one
    points at that marker's block (W11-6), as switch and x360 store it; False: every marker has a block of
    its own, as ps3 stores it; None: as the build stores it."""
    if pooled is None:
        pooled = build in L.MAPS_POOLED
    markers = [{"hash": 0x301 + i, "type": 0, "group": 1, "handle": F.ACR, "position": (1.0 * i, 2.0, 0.0),
                "links": [(i, i + 1)], "tags": encode_tag_map(MAPS[k], build)} for i, k in enumerate(kinds)]
    tree = L.parse_tree(B.build_mrk(build, markers=markers, circles=[(1.0, 2.0, 3.0)], special=[1, 2, 3]),
                        "mrk", build)
    if pooled:
        elements = tree.top.pairs[0x0C].block.elements
        first: dict = {}
        for i, k in enumerate(kinds):
            if k in first:
                elements[i].pairs[0x30] = elements[first[k]].pairs[0x30]
            else:
                first[k] = i
    return L.write_tree(tree)


def from_scratch(data, build, one_object=False):
    """The set's tree with every tag map a new block (no stored place), as a set written from scratch; with
    ``one_object`` markers of equal records hold one block object."""
    tree = L.parse_tree(data, "mrk", build)
    objects: dict = {}
    for element in tree.top.pairs[0x0C].block.elements:
        value = element.pairs.get(0x30)
        if value is None or value.block is None:
            continue
        raw = bytes(value.block.data)
        block = objects.setdefault(raw, L.Block(raw)) if one_object else L.Block(raw)
        element.pairs[0x30] = L.PairValue(block)
    return tree


def blocks(data, build):
    return C33.map_blocks(data, build)


class FromScratch(unittest.TestCase):
    """A set written from scratch (``write_tree(derive=True)``) is laid out as its build stores it."""

    def test_stored_layout_of_each_build(self):
        for build in BUILDS:
            for one_object in (False, True):
                with self.subTest(build=build, one_object=one_object):
                    stored = marker_set(build)
                    self.assertEqual(L.write_tree(from_scratch(stored, build, one_object), derive=True), stored)
                    found = blocks(stored, build)
                    self.assertEqual(len(set(found)), 3 if build in L.MAPS_POOLED else len(KINDS))

    def test_pooled_at_the_first_marker(self):
        for build in ("switch", "x360"):
            found = blocks(L.write_tree(from_scratch(marker_set(build, pooled=False), build), derive=True), build)
            self.assertEqual(found[2], found[0])
            self.assertEqual(found[5], found[0])
            self.assertEqual(found[4], found[1])
            self.assertEqual(found[0], min(found))  # written at the first marker holding those records

    def test_ps3_keeps_equal_maps_apart(self):
        stored = C33.own_maps("ps3", "rec")
        self.assertEqual(L.write_tree(from_scratch(stored, "ps3"), derive=True), stored)
        self.assertEqual(L.write_tree(from_scratch(stored, "ps3", True), derive=True), stored)
        self.assertEqual(len(set(blocks(stored, "ps3"))), 4)

    def test_template_layout_kept(self):
        """A parsed set keeps its layout: a switch file of equal maps kept apart rebuilds as it is."""
        for build in BUILDS:
            for pooled in (False, True):
                data = marker_set(build, pooled=pooled)
                self.assertEqual(L.rebuild_sno(data, "mrk", build), data)
                self.assertEqual(L.write_tree(L.parse_tree(data, "mrk", build), derive=True), data)

    def test_other_pairs_never_pooled(self):
        """Pooling is for marker tag maps only: a new special index list of a map's bytes stays apart."""
        for build in ("switch", "x360"):
            tree = from_scratch(marker_set(build), build)
            first_map = tree.top.pairs[0x0C].block.elements[0].pairs[0x30].block.data
            tree.top.pairs[0xB4] = L.PairValue(L.Block(bytes(first_map)))
            written = L.parse_tree(L.write_tree(tree, derive=True), "mrk", build)
            special = written.top.pairs[0xB4].block.source
            self.assertNotIn(special, blocks(L.write_tree(written), build))


class AddedRow(unittest.TestCase):
    """A new row's tag map follows the build; the original JSON gives the template back."""

    def test_switch_and_x360_share(self):
        for build in ("switch", "x360"):
            data = marker_set(build)
            original = J._json_round(J.template_json("mrk", data, build))
            document = copy.deepcopy(original)
            document["markers"].append(copy.deepcopy(original["markers"][1]))  # marker 1 owns its block
            first = J.import_json("mrk", data, build, document).data
            found = blocks(first, build)
            self.assertEqual(found[-1], found[1])
            self.assertEqual(len(set(found)), 3)
            self.assertEqual(J.import_json("mrk", first, build, original).data, data)

    def test_ps3_own_block(self):
        data = marker_set("ps3")
        original = J._json_round(J.template_json("mrk", data, "ps3"))
        document = copy.deepcopy(original)
        document["markers"].append(copy.deepcopy(original["markers"][1]))
        first = J.import_json("mrk", data, "ps3", document).data
        found = blocks(first, "ps3")
        self.assertEqual(len(set(found)), len(KINDS) + 1)
        self.assertNotIn(found[-1], found[:-1])
        self.assertEqual(J.import_json("mrk", first, "ps3", original).data, data)

    def test_copy_of_a_pointing_row(self):
        """A copy of a row that points back to an earlier block: that block on switch and x360, its own on
        ps3."""
        for build in BUILDS:
            data = marker_set(build)
            original = J._json_round(J.template_json("mrk", data, build))
            document = copy.deepcopy(original)
            document["markers"].insert(3, copy.deepcopy(original["markers"][2]))
            first = J.import_json("mrk", data, build, document).data
            found = blocks(first, build)
            if build in L.MAPS_POOLED:
                self.assertEqual(found[3], found[0])
            else:
                self.assertNotIn(found[3], found[:3] + found[4:])
            self.assertEqual(J.import_json("mrk", first, build, original).data, data)

    def test_first_equal_map(self):
        """A template keeping equal maps apart: a new row points at the first of them, not the last."""
        for build in ("switch", "x360"):
            data = marker_set(build, pooled=False)
            original = J._json_round(J.template_json("mrk", data, build))
            document = copy.deepcopy(original)
            document["markers"].append(copy.deepcopy(original["markers"][5]))  # marker 5 owns an equal map
            first = J.import_json("mrk", data, build, document).data
            found = blocks(first, build)
            self.assertEqual(len(set(found)), len(KINDS))
            self.assertEqual(found[-1], found[0])
            self.assertEqual(J.import_json("mrk", first, build, original).data, data)

    def test_undo_every_build_and_place(self):
        for build in BUILDS:
            for pooled in (False, True):
                data = marker_set(build, pooled=pooled)
                original = J._json_round(J.template_json("mrk", data, build))
                for where in range(len(KINDS) + 1):
                    for source in (0, 1, 3):
                        with self.subTest(build=build, pooled=pooled, where=where, source=source):
                            document = copy.deepcopy(original)
                            document["markers"][4]["name_hash"] = 0x777
                            document["markers"].insert(where, copy.deepcopy(original["markers"][source]))
                            first = J.import_json("mrk", data, build, document).data
                            self.assertEqual(len(world.parse_marker_set(first, build).markers), len(KINDS) + 1)
                            self.assertEqual(J.import_json("mrk", first, build, original).data, data)


class ChangedMap(unittest.TestCase):
    """A changed tag map follows the build."""

    def test_changed_to_an_earlier_map(self):
        for build in BUILDS:
            data = marker_set(build)
            original = J._json_round(J.template_json("mrk", data, build))
            document = copy.deepcopy(original)
            document["markers"][3]["tags"] = copy.deepcopy(original["markers"][0]["tags"])  # one record each
            first = J.import_json("mrk", data, build, document).data
            found = blocks(first, build)
            if build in L.MAPS_POOLED:
                self.assertEqual(found[3], found[0])
            else:
                self.assertNotIn(found[3], found[:3] + found[4:])
            self.assertEqual(J._json_round(J.template_json("mrk", first, build))["markers"][3]["tags"],
                             original["markers"][0]["tags"])
            self.assertEqual(J.import_json("mrk", first, build, original).data, data)

    def test_new_records_own_block(self):
        for build in BUILDS:
            data = marker_set(build)
            original = J._json_round(J.template_json("mrk", data, build))
            document = copy.deepcopy(original)
            tags = J._json_round(J.template_json("mrk", marker_set(build, kinds=(2,)), build))["markers"][0]["tags"]
            document["markers"].append(copy.deepcopy(original["markers"][0]))
            document["markers"][-1]["tags"] = tags
            first = J.import_json("mrk", data, build, document).data
            found = blocks(first, build)
            if build in L.MAPS_POOLED:
                self.assertEqual(found[-1], found[3])  # marker 3 holds those records
            else:
                self.assertEqual(len(set(found)), len(KINDS) + 1)


class Version(unittest.TestCase):
    def test_version(self):
        self.assertGreaterEqual(tuple(int(p) for p in d3sno.__version__.split(".")), (0, 22, 5))
        self.assertGreaterEqual(d3sno.STAGE, 22)


if __name__ == "__main__":
    unittest.main()
