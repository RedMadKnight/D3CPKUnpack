"""Corrections 38 (``61_corrections38.md``): 64-bit rows (AI1), negative zero (AI2), the PS3 curve count and one
place per leaf (AI3), PS3 blocks shared across element types (AI4). Every file is synthetic."""

import _libpath  # noqa: F401  (python -I: the library folder on sys.path)
import copy
import struct
import tempfile
import unittest
from pathlib import Path

import test_stage23 as T
import test_stage24 as S
import d3sno
from d3sno import effects as E
from d3sno import json_import as J
from d3sno import sno_layout as L
from d3sno.binary import FormatError, build_order
from d3sno.datafiles import DATA

BUILDS = ("switch", "x360", "ps3")
KEYS = ("nKey", "nParentKey", "tInlineStyle.nKey")
VALUES = (0x050D900942C32B60, -0x0123456789ABCDEF, 0x7FFFFFFF00000001)


def doc(data, ext, build):
    return J._json_round(E.decode(data, ext, build))


def ui_file(build, values=VALUES):
    """A dialog whose control list holds one record with three 64-bit keys, both halves set."""
    tree = T.blank(build, "ui", 7001)
    slot = next(s for s in L.pair_slots(build, "ui", tree.top.name) if s.field.element == "UIControlData")
    element = L.new_record(build, "ui", "UIControlData")
    for name, value in zip(KEYS, values):
        struct.pack_into(build_order(build) + "q", element.raw, S.row(build, "ui", "UIControlData", name).offset, value)
    tree.top.pairs[slot.offset] = L.PairValue(L.Block(elements=[element, L.new_record(build, "ui", "UIControlData")]))
    return L.write_tree(tree, derive=True)


def float_leaf(build):
    """A float leaf of the light's top record that is no curve part."""
    view = E._view(build, "lit", E._top_table_record(build, "lit"), L.top_record(build, "lit"))
    return next(n for n in view.plain if n.code == "f")


def lit_minus_zero(build):
    tree = T.blank(build, "lit", 7002)
    struct.pack_into(build_order(build) + "f", tree.top.raw, float_leaf(build).offset, -0.0)
    return L.write_tree(tree, derive=True)


def lit_across(build, equal=True):
    """A light whose AngleNode curve points at the block of its first FloatNode curve (same bytes, other element
    type); with ``equal`` False each has its own block."""
    tree = T.blank(build, "lit", 7003)
    slots = [s for s in L.pair_slots(build, "lit", tree.top.name) if s.field.is_record_array]
    floats = next(s for s in slots if s.field.element == "FloatNode")
    angles = next(s for s in slots if s.field.element == "AngleNode")
    value = T.nodes_block(build, "lit", "FloatNode", T.NODES)
    tree.top.pairs[floats.offset] = value
    if equal:
        tree.top.pairs[angles.offset] = L.PairValue(value.block)
    else:
        raws = [L.Record("AngleNode", bytearray(e.raw)) for e in value.block.elements]
        tree.top.pairs[angles.offset] = L.PairValue(L.Block(elements=raws))
    return L.write_tree(tree, derive=True), floats, angles


class Wide(unittest.TestCase):
    """AI1."""

    def test_ui_keys_rebuilt(self):
        for build in ("switch", "x360"):
            data = ui_file(build)
            document = doc(data, "ui", build)
            control = document["record"][next(k for k, v in document["record"].items()
                                              if isinstance(v, list) and v and "nParentKey" in v[0])][0]
            self.assertEqual([control[k] for k in KEYS], list(VALUES), build)
            self.assertEqual(E.create(document, "ui", build), data)
            self.assertEqual(E.encode(data, "ui", build, document), data)

    def test_edit_and_range(self):
        build = "x360"
        data = ui_file(build)
        document = doc(data, "ui", build)
        key = next(k for k, v in document["record"].items() if isinstance(v, list) and v and "nKey" in v[0])
        document["record"][key][0]["nKey"] = -1
        again = doc(E.encode(data, "ui", build, document), "ui", build)
        self.assertEqual(again["record"][key][0]["nKey"], -1)
        self.assertEqual(len(E.encode(data, "ui", build, document)), len(data))
        for value in (1 << 63, -(1 << 63) - 1):
            document["record"][key][0]["nKey"] = value
            with self.assertRaises(J.Refused) as caught:
                E.create(document, "ui", build)
            self.assertEqual((caught.exception.path[-1], caught.exception.reason), ("nKey", "value_out_of_range"))

    def test_wrong_length_refused(self):
        record = L.new_record("switch", "ui", "UIControlData")
        named = E._Named("nKey", 8, 8, "q")
        with self.assertRaises(J.Refused) as caught:
            E._put(record, named, b"\1\2\3\4", ("record", "nKey"))
        self.assertEqual(caught.exception.reason, "value_out_of_range")
        self.assertEqual(len(record.raw), 368)
        with self.assertRaises(J.Refused):
            E._put(record, E._Named("x", 366, 4, "i"), b"\1\2\3\4", ("record", "x"))  # past the end
        self.assertEqual(len(record.raw), 368)


class MinusZero(unittest.TestCase):
    """AI2."""

    def test_created_with_the_sign_bit(self):
        for build in BUILDS:
            data = lit_minus_zero(build)
            document = doc(data, "lit", build)
            leaf = float_leaf(build)
            self.assertEqual(struct.pack(">f", document["record"][leaf.name]), b"\x80\0\0\0")
            self.assertEqual(E.create(document, "lit", build), data, build)

    def test_edit_to_plus_zero_and_back(self):
        for build in BUILDS:
            data = lit_minus_zero(build)
            leaf = float_leaf(build)
            document = doc(data, "lit", build)
            document["record"][leaf.name] = 0.0
            plus = E.encode(data, "lit", build, document)
            raw = L.parse_tree(plus, "lit", build).top.raw
            self.assertEqual(bytes(raw[leaf.offset:leaf.offset + 4]), bytes(4))
            back = doc(plus, "lit", build)
            back["record"][leaf.name] = -0.0
            self.assertEqual(E.encode(plus, "lit", build, back), data)

    def test_same(self):
        self.assertFalse(J._same_json(0.0, -0.0))
        self.assertFalse(J._close(0.0, -0.0))
        self.assertTrue(J._close(-0.0, -0.0))
        self.assertTrue(J._same_json(0, 0.0))
        self.assertTrue(J._close(1.0, 1.0000001))


class OnePlace(unittest.TestCase):
    """AI3."""

    def test_table(self):
        self.assertEqual(E.LAYOUT_TABLE, "f61_layouts.tsv")
        self.assertTrue((DATA / "f58_layouts.tsv").is_file())
        rows = [r for r in E.leaf_table("ps3", "lit") if r.status == "curve header (PS3 rule, aligned on the files)"
                and r.name.endswith("tHeader.nNodeCount")]
        self.assertTrue(rows)

    def test_leaf_placed_twice_fails_to_load(self):
        with open(DATA / E.LAYOUT_TABLE, encoding="utf-8") as handle:
            head = handle.readline()
        line = "ps3\tvib\tVibrationDefinition\tdwFlags\t{}\t4\tINT\t\t\t\t\t\ttest\n"
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "t.tsv"
            path.write_text(head + line.format(12) + line.format(12), encoding="utf-8")
            self.assertEqual(len(E.load_layouts(path)[("ps3", "vib")]), 2)  # one place: a copy of a row
            path.write_text(head + line.format(12) + line.format(16), encoding="utf-8")
            with self.assertRaises(FormatError) as caught:
                E.load_layouts(path)
            self.assertIn("ps3 .vib VibrationDefinition: leaf 'dwFlags' is placed twice", str(caught.exception))

    def test_ps3_curve_count(self):
        for count in (3, 5):
            data = T.vib_file("ps3", count=count)
            document = doc(data, "vib", "ps3")
            curve = document["record"][T.CURVE]
            self.assertEqual(curve.get("stored_count"), None if count == 3 else 5)
            self.assertEqual(E.create(document, "vib", "ps3"), data)

    def test_no_byte_lost(self):
        """Every byte of every top record is a named leaf, a curve part, a pair or an unnamed word (each once)."""
        for build in BUILDS:
            for ext in E.types(build):
                top = L.top_record(build, ext)
                view = E._view(build, ext, E._top_table_record(build, ext), top)
                seen = bytearray(view.size)
                parts = [p for c in view.curves for p in c.parts.values()]
                spans = [(n.offset, n.size) for n in view.plain + parts] + view.unnamed + \
                        [(p.slot.offset, 8) for p in view.pairs]
                for offset, size in spans:
                    for i in range(offset, min(offset + size, view.size)):
                        seen[i] += 1
                with self.subTest(build=build, ext=ext):
                    self.assertTrue(all(seen), [i for i, s in enumerate(seen) if not s][:4])

    def test_unused_row_is_an_unnamed_word(self):
        """A row whose name a curve already has is not exported: its word stays an unnamed word, so its value is
        kept. (Rows in a layout conflict too: ``test_stage23`` ``test_overlapping_row_reported_not_used``.)"""
        table = dict(E._rows())
        top = E._top_table_record("switch", "vib")
        free = E._view("switch", "vib", top, L.top_record("switch", "vib")).unnamed[0][0]
        extra = E.Leaf("switch", "vib", top, T.CURVE, free, 4, "FLOAT", "", "", "", None, "aligned by value")
        table[("switch", "vib")] = table[("switch", "vib")] + (extra,)
        caches = (E._by_record, E._view, E._top_table_record, E._zero_text)
        for cache in caches:
            cache.cache_clear()
        self.addCleanup(lambda: [cache.cache_clear() for cache in caches])
        from unittest import mock
        with mock.patch.object(E, "_rows", lambda: table):
            tree = L.parse_tree(T.vib_file("switch"), "vib", "switch")
            struct.pack_into("<f", tree.top.raw, free, 1.0)
            data = L.write_tree(tree)
            document = doc(data, "vib", "switch")
            self.assertIn(E._word_key(free), document["record"]["unnamed"])
            self.assertEqual(E.create(document, "vib", "switch"), data)


class Across(unittest.TestCase):
    """AI4."""

    def test_listed_and_rebuilt(self):
        for build in BUILDS:
            data, floats, angles = lit_across(build)
            tree = L.parse_tree(data, "lit", build)
            value = tree.top.pairs[angles.offset]
            self.assertTrue(value.shared and value.alias_of is tree.top.pairs[floats.offset].block)
            self.assertEqual(L.write_tree(tree), data)
            document = doc(data, "lit", build)
            self.assertEqual(len(document["shared"]), 1)
            self.assertEqual(E.create(document, "lit", build), data, build)
            self.assertEqual(E.create(document, "lit", build, pool=False), data, build)

    def test_pool_keeps_element_types_apart(self):
        for build in BUILDS:
            data, _, _ = lit_across(build)
            document = doc(data, "lit", build)
            del document["shared"]
            self.assertEqual(E.create(document, "lit", build), lit_across(build, equal=False)[0])

    def test_alias_not_pooled_again(self):
        """An earlier FloatNode list with the same bytes does not take the place of a FloatNode pair that points at an
        AngleNode block (the pool joins only pairs of one element type, and a listed pair keeps its first_path)."""
        build = "switch"
        tree = T.blank(build, "lit", 7004)
        slots = [s for s in L.pair_slots(build, "lit", tree.top.name) if s.field.is_record_array]
        angle = next(i for i, s in enumerate(slots) if s.field.element == "AngleNode")
        before = next((s for s in slots[:angle] if s.field.element == "FloatNode"), None)
        after = next((s for s in slots[angle + 1:] if s.field.element == "FloatNode"), None)
        if before is None or after is None:
            self.skipTest("no FloatNode list on both sides of an AngleNode list")
        tree.top.pairs[before.offset] = T.nodes_block(build, "lit", "FloatNode", T.NODES)
        nodes = T.nodes_block(build, "lit", "FloatNode", T.NODES).block.elements
        target = L.Block(elements=[L.Record("AngleNode", bytearray(e.raw)) for e in nodes])
        tree.top.pairs[slots[angle].offset] = L.PairValue(target)
        tree.top.pairs[after.offset] = L.PairValue(target)
        data = L.write_tree(tree, derive=True)
        document = doc(data, "lit", build)
        self.assertEqual(len(document["shared"]), 1)
        self.assertEqual(E.create(document, "lit", build), data)

    def test_different_bytes_refused(self):
        for build in BUILDS:
            data, _, _ = lit_across(build)
            document = doc(data, "lit", build)
            path, first = document["shared"][0]
            key = path[len("record."):-len(".nodes")]
            edited = copy.deepcopy(document)
            edited["record"][key]["nodes"][0]["time"] = 0.25
            with self.assertRaises(J.Refused) as caught:
                E.create(edited, "lit", build)
            self.assertEqual((caught.exception.path, caught.exception.reason), (("shared", 0), "conflicting_values"))
            self.assertIn(path, caught.exception.message)
            self.assertIn(first, caught.exception.message)
            shorter = copy.deepcopy(document)
            del shorter["record"][key]["nodes"][-1]
            with self.assertRaises(J.Refused) as caught:
                E.create(shorter, "lit", build)
            self.assertEqual(caught.exception.path, ("shared", 0))


class Version(unittest.TestCase):
    def test_version(self):
        self.assertGreaterEqual(tuple(int(p) for p in d3sno.__version__.split(".")), (0, 25, 1))
        self.assertEqual(d3sno.STAGE, 25)


if __name__ == "__main__":
    unittest.main()
