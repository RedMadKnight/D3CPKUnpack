"""Corrections 35 (``56_corrections35.md``): AF1 to AF4, with the tables ``f56_*``. Synthetic files built from
the table's rows only; nothing is written to disk."""

import _libpath  # noqa: F401  (python -I: the library folder on sys.path)
import copy
import struct
import unittest
from unittest import mock

import test_stage23 as T
import d3sno
from d3sno import effects as E
from d3sno import json_import as J
from d3sno import sno_layout as L
from d3sno.binary import build_order

BUILDS = T.BUILDS
CURVE, OUTRO = T.CURVE, "outroVibrationIntensityCurve"


def slot_at(build, ext, record, offset):
    return next(s for s in L.pair_slots(build, ext, record) if s.offset == offset)


def row(build, ext, record, name):
    return next(r for r in E.leaf_table(build, ext) if r.record == record and r.name == name)


def set_outro_count(tree, build, count=3):
    if build != "ps3":
        at = row(build, "vib", "VibrationDefinition", f"{OUTRO}.tHeader.nNodeCount").offset
        struct.pack_into(build_order(build) + "i", tree.top.raw, at, count)


class Tables(unittest.TestCase):
    def test_f56_in_use(self):
        self.assertNotEqual(E.LAYOUT_TABLE, "f55_layouts.tsv")  # ``57``: f57 replaces f56
        for build in BUILDS:
            for ext in E.GROUPS:
                arrays = [r for r in E.leaf_table(build, ext) if r.is_array and r.offset is not None]
                self.assertTrue(all(r.pair_offset is not None for r in arrays), (build, ext))

    def test_switch_without_conflicts(self):
        """AF2, AF3: a record of zeros of every group has no conflict on Switch."""
        for ext in E.GROUPS:
            found = E.decode_full(L.write_tree(T.blank("switch", ext, 9)), ext, "switch")
            self.assertEqual(found.conflicts, [], ext)


class LaidOutAsNew(unittest.TestCase):
    """AF1: after a change the file is laid out as from scratch; the original JSON gives the stored bytes."""

    def back(self, data, ext, build, edit):
        original = T.doc(data, ext, build)
        edited = copy.deepcopy(original)
        edit(edited)
        first = E.encode(data, ext, build, edited)
        self.assertNotEqual(first, data)
        self.assertEqual(E.encode(first, ext, build, original), data)
        return first

    def test_node_added_to_a_shared_node_list(self):
        """Two curves pointing at one node block (W11-6): a node added through one is in both; then back."""
        for build in BUILDS:
            with self.subTest(build=build):
                tree = L.parse_tree(T.vib_file(build), "vib", build)
                top = tree.top
                primary = T.node_slot(build)
                outro = T.node_slot(build, curve=OUTRO)
                top.pairs[outro.offset] = top.pairs[primary.offset]
                set_outro_count(tree, build)
                data = L.write_tree(tree)
                original = T.doc(data, "vib", build)
                self.assertEqual(len(original["record"][OUTRO]["nodes"]), 3)
                edited = copy.deepcopy(original)
                edited["record"][CURVE]["nodes"].append({"time": 1.0, "min": 0.0, "max": 0.0})
                data_json, rows, warnings = E.encode_result(data, "vib", build, edited)
                self.assertTrue(any(w.startswith("shared_block") for w in warnings))
                again = T.doc(data_json, "vib", build)
                self.assertEqual(len(again["record"][OUTRO]["nodes"]), 4)  # one block, as stored
                self.assertEqual(E.encode(data_json, "vib", build, original), data)

    def test_both_sharers_edited_alike(self):
        build = "switch"
        tree = L.parse_tree(T.vib_file(build), "vib", build)
        tree.top.pairs[T.node_slot(build, curve=OUTRO).offset] = tree.top.pairs[T.node_slot(build).offset]
        set_outro_count(tree, build)
        data = L.write_tree(tree)
        original = T.doc(data, "vib", build)
        edited = copy.deepcopy(original)
        node = {"time": 1.0, "min": 0.0, "max": 0.0}
        edited["record"][CURVE]["nodes"].append(node)
        edited["record"][OUTRO]["nodes"].append(dict(node))
        first = E.encode(data, "vib", build, edited)
        self.assertEqual(len(T.doc(first, "vib", build)["record"][CURVE]["nodes"]), 4)
        edited["record"][OUTRO]["nodes"][-1]["min"] = 5.0
        edited["record"][OUTRO]["nodes"].append(dict(node))
        with self.assertRaises(J.Refused) as caught:
            E.encode(data, "vib", build, edited)
        self.assertEqual(caught.exception.reason, "conflicting_values")

    def test_byte_list(self):
        """A byte of a byte list changed, then the original JSON (a light's ``pTexAnimCompParams``)."""
        for build in ("switch", "x360"):
            tree = L.parse_tree(T.lit_file(build), "lit", build)
            top_name = E._top_table_record(build, "lit")
            pair = row(build, "lit", top_name, "tMatTexProjected.pTexAnimCompParams").pair_offset
            tree.top.pairs[pair] = L.PairValue(L.Block(bytes(range(144))))
            data = L.write_tree(tree)

            def edit(d):
                d["record"]["tMatTexProjected.pTexAnimCompParams"][5] = 200
            first = self.back(data, "lit", build, edit)
            self.assertEqual(len(first), len(data))  # written where it was, not as a second copy

    def test_shared_byte_list(self):
        """Two material texture entries sharing one byte list: changed through one, both change; then back."""
        build, ext = "switch", "prt"
        tree = T.blank(build, ext, 9)
        top = L.top_record(build, ext)
        list_pair = row(build, ext, "ParticleDefinition", "tUberMaterial.ptMatTexList").pair_offset
        entry = "MaterialTextureEntry"
        size = L.record_layout(build, ext, entry).size
        bytes_pair = row(build, ext, entry, "tMatTex.pTexAnimCompParams").pair_offset
        shared = L.PairValue(L.Block(bytes(range(32))))
        entries = [L.Record(entry, bytearray(size)) for _ in range(2)]
        for e in entries:
            e.pairs[bytes_pair] = shared
        tree.top.pairs[list_pair] = L.PairValue(L.Block(elements=entries))
        data = L.write_tree(tree, derive=True)
        original = T.doc(data, ext, build)
        edited = copy.deepcopy(original)
        edited["record"]["tUberMaterial.ptMatTexList"][0]["tMatTex.pTexAnimCompParams"][3] = 99
        first = E.encode(data, ext, build, edited)
        self.assertEqual(len(first), len(data))
        again = T.doc(first, ext, build)["record"]["tUberMaterial.ptMatTexList"]
        self.assertEqual(again[1]["tMatTex.pTexAnimCompParams"][3], 99)
        self.assertEqual(E.encode(first, ext, build, original), data)

    def test_element_added(self):
        """An array element added (a particle path), then the original JSON, on every build."""
        for build in BUILDS:
            with self.subTest(build=build):
                ext = "prt"
                tree = T.blank(build, ext, 9)
                top = L.top_record(build, ext)
                pair = row(build, ext, "ParticleDefinition", "pathSystemScale").pair_offset
                slot = slot_at(build, ext, top, pair)
                element = L.Record(slot.field.element, bytearray(L.record_layout(build, ext, slot.field.element).size))
                struct.pack_into(build_order(build) + "f", element.raw, 4, 0.25)
                tree.top.pairs[pair] = L.PairValue(L.Block(elements=[element]))
                data = L.write_tree(tree, derive=True)

                def edit(d):
                    d["record"]["pathSystemScale"].append(copy.deepcopy(d["record"]["pathSystemScale"][0]))
                self.back(data, ext, build, edit)

    def test_unedited_json_keeps_the_bytes(self):
        for build in BUILDS:
            data = T.vib_file(build)
            self.assertIs(type(E.encode(data, "vib", build, T.doc(data, "vib", build))), bytes)
            self.assertEqual(E.encode(data, "vib", build, T.doc(data, "vib", build)), data)


class PairOffsets(unittest.TestCase):
    """AF2: the pair is at ``pair_offset``."""

    def test_pair_before_the_pointer(self):
        build, ext = "switch", "efg"
        top_name = E._top_table_record(build, ext)
        effects = row(build, ext, top_name, "ptEffects")
        self.assertEqual((effects.offset, effects.pair_offset), (32, 16))
        tree = T.blank(build, ext, 9)
        top = L.top_record(build, ext)
        slot = slot_at(build, ext, top, 16)
        size = L.record_layout(build, ext, slot.field.element).size
        tree.top.pairs[16] = L.PairValue(L.Block(elements=[L.Record(slot.field.element, bytearray(size))]))
        data = L.write_tree(tree, derive=True)
        found = E.decode_full(data, ext, build)
        self.assertEqual(found.conflicts, [])
        self.assertEqual(len(found.json["record"]["ptEffects"]), 1)
        original = J._json_round(found.json)
        edited = copy.deepcopy(original)
        edited["record"]["ptEffects"].append(copy.deepcopy(original["record"]["ptEffects"][0]))
        first = E.encode(data, ext, build, edited)
        self.assertEqual(len(T.doc(first, ext, build)["record"]["ptEffects"]), 2)
        self.assertEqual(E.encode(first, ext, build, original), data)

    def test_sharer_with_another_count_kept(self):
        """A curve sharing the node block whose count already differed keeps its stored count."""
        build = "switch"
        tree = L.parse_tree(T.vib_file(build), "vib", build)
        tree.top.pairs[T.node_slot(build, curve=OUTRO).offset] = tree.top.pairs[T.node_slot(build).offset]
        set_outro_count(tree, build, 9)
        data = L.write_tree(tree)
        edited = T.doc(data, "vib", build)
        self.assertEqual(edited["record"][OUTRO]["stored_count"], 9)
        edited["record"][CURVE]["nodes"].append({"time": 1.0, "min": 0.0, "max": 0.0})
        again = T.doc(E.encode(data, "vib", build, edited), "vib", build)
        self.assertEqual(again["record"][OUTRO]["stored_count"], 9)
        self.assertNotIn("stored_count", again["record"][CURVE])

    def test_no_plus_eight(self):
        """A row whose ``pair_offset`` holds no pair is a conflict, even with a pair at its offset + 8."""
        table = dict(E._rows())
        top_name = E._top_table_record("switch", "efg")
        wrong = E.Leaf("switch", "efg", top_name, "ptEffects", 8, 8, "ARRAY:EffectItem", "", "", "", None,
                       "type information", 4)
        table[("switch", "efg")] = tuple(r for r in table[("switch", "efg")] if r.name != "ptEffects") + (wrong,)
        caches = (E._by_record, E._view, E._top_table_record)
        for cache in caches:
            cache.cache_clear()
        self.addCleanup(lambda: [cache.cache_clear() for cache in caches])
        with mock.patch.object(E, "_rows", lambda: table):
            found = E.decode_full(L.write_tree(T.blank("switch", "efg", 9)), "efg", "switch")
            self.assertIn((top_name, "ptEffects"), found.conflicts)


class Ps3Curves(unittest.TestCase):
    """AF3: the PS3 curve forms of ``f56``."""

    def test_inline_curve(self):
        """Header rows of the parent at the node pair - 28; no count (the table's word at - 32 is a plain leaf)."""
        data = T.vib_file("ps3", count=7)
        found = E.decode_full(data, "vib", "ps3")
        curve = found.json["record"][CURVE]
        self.assertEqual(curve["loop"], [0.25, 0.75])
        self.assertEqual(curve["scalar"]["at_one"], 1.5)
        self.assertEqual(len(curve["nodes"]), 3)
        self.assertNotIn("stored_count", curve)
        self.assertEqual(found.json["record"][f"{CURVE}.tHeader.nNodeCount"], 7)
        self.assertEqual(row("ps3", "vib", "VibrationDefinition", f"{CURVE}.tHeader.tLoopPointStart").status,
                         "curve header (PS3 rule, aligned on the files)")

    def test_emitter_curve(self):
        build, ext = "ps3", "prt"
        tree = T.blank(build, ext, 9)
        order = build_order(build)
        start = row(build, ext, "ParticleDefinition", "tEmitterParams.pathRadius.tHeader.tLoopPointStart")
        pair = row(build, ext, "ParticleDefinition", "tEmitterParams.pathRadius.ptArrayNodes").pair_offset
        self.assertEqual(start.offset, pair - 28)
        struct.pack_into(order + "f", tree.top.raw, start.offset, 0.5)
        slot = slot_at(build, ext, L.top_record(build, ext), pair)
        size = L.record_layout(build, ext, slot.field.element).size
        tree.top.pairs[pair] = L.PairValue(L.Block(elements=[L.Record(slot.field.element, bytearray(size))] * 2))
        data = L.write_tree(tree, derive=True)
        found = E.decode_full(data, ext, build)
        curve = found.json["record"]["tEmitterParams.pathRadius"]
        self.assertEqual(curve["loop"][0], 0.5)
        self.assertEqual(len(curve["nodes"]), 2)
        self.assertNotIn(("ParticleDefinition", "tEmitterParams.pathRadius.ptArrayNodes"), found.conflicts)
        edited = J._json_round(found.json)
        edited["record"]["tEmitterParams.pathRadius"]["nodes"].append(copy.deepcopy(edited["record"]["tEmitterParams.pathRadius"]["nodes"][0]))
        first = E.encode(data, ext, build, edited)
        self.assertEqual(E.encode(first, ext, build, J._json_round(found.json)), data)

    def test_path_element(self):
        """A particle path element (``FloatPath#1``): the header with its count at 0, the node pair at 32."""
        build, ext = "ps3", "prt"
        tree = T.blank(build, ext, 9)
        order = build_order(build)
        pair = row(build, ext, "ParticleDefinition", "pathSystemScale").pair_offset
        slot = slot_at(build, ext, L.top_record(build, ext), pair)
        self.assertEqual(slot.field.element, "FloatPath#1")
        element = L.Record("FloatPath#1", bytearray(L.record_layout(build, ext, "FloatPath#1").size))
        self.assertEqual(row(build, ext, "FloatPath#1", "ptArrayNodes").pair_offset, 32)
        struct.pack_into(order + "i", element.raw, 0, 4)  # a count of 4 for no node
        struct.pack_into(order + "f", element.raw, 4, 0.125)
        tree.top.pairs[pair] = L.PairValue(L.Block(elements=[element]))
        found = E.decode_full(L.write_tree(tree), ext, build)
        path = found.json["record"]["pathSystemScale"][0]
        self.assertEqual(path["loop"][0], 0.125)
        self.assertEqual(path["stored_count"], 4)
        self.assertIn(("record", "pathSystemScale", 0), found.counts_differ)

    def test_anonymous_record_named(self):
        """PS3 rows name the tree's records, ``anon:`` ones included."""
        self.assertEqual(E._table_record("ps3", "prt", "anon:ParticleDefinition@1E0"), "anon:ParticleDefinition@1E0")
        self.assertIsNone(E._table_record("ps3", "prt", "anon:Nothing@0"))

    def test_material_colours(self):
        """AF3 item 1: a material's colours are four floats each."""
        build, ext = "switch", "prt"
        tree = T.blank(build, ext, 9)
        struct.pack_into("<f", tree.top.raw, row(build, ext, "ParticleDefinition", "tUberMaterial.mat.rgbavalAmbient.r").offset, 0.75)
        record = T.doc(L.write_tree(tree), ext, build)["record"]
        self.assertEqual(record["tUberMaterial.mat.rgbavalAmbient.r"], 0.75)
        self.assertIn("tUberMaterial.dwMatTexSortHash", record)


def flag_set(build="switch", label=b"Ready", raw=None):
    ext = "flg"
    tree = T.blank(build, ext, 9)
    pair = row(build, ext, E._top_table_record(build, ext), "arFlagLookups").pair_offset
    slot = slot_at(build, ext, L.top_record(build, ext), pair)
    size = L.record_layout(build, ext, slot.field.element).size
    element = L.Record(slot.field.element, bytearray(size))
    text = row(build, ext, "FlagLookup", "szLabel")
    field = raw if raw is not None else label + bytes(text.size - len(label))
    element.raw[text.offset:text.offset + text.size] = field
    tree.top.pairs[pair] = L.PairValue(L.Block(elements=[element]))
    return L.write_tree(tree, derive=True), text.size


class Text(unittest.TestCase):
    """AF4: ``CHARARRAY`` fields."""

    def test_round_trip(self):
        for build in ("switch", "x360"):
            data, size = flag_set(build)
            original = T.doc(data, "flg", build)
            self.assertEqual(original["record"]["arFlagLookups"][0]["szLabel"], "Ready")
            edited = copy.deepcopy(original)
            edited["record"]["arFlagLookups"][0]["szLabel"] = "Set"
            first = E.encode(data, "flg", build, edited)
            self.assertEqual(T.doc(first, "flg", build)["record"]["arFlagLookups"][0]["szLabel"], "Set")
            self.assertEqual(E.encode(first, "flg", build, original), data)
            edited["record"]["arFlagLookups"][0]["szLabel"] = "x" * (size - 1)
            self.assertEqual(T.doc(E.encode(data, "flg", build, edited), "flg", build)["record"]["arFlagLookups"][0]["szLabel"],
                             "x" * (size - 1))

    def test_refused(self):
        data, size = flag_set()
        for text in ("x" * size, "x" * (size + 3), "Zażółć", "a\0b"):
            edited = T.doc(data, "flg", "switch")
            edited["record"]["arFlagLookups"][0]["szLabel"] = text
            with self.assertRaises(J.Refused) as caught:
                E.encode(data, "flg", "switch", edited)
            self.assertEqual(caught.exception.reason, "value_out_of_range", text)

    def test_bytes_after_the_end(self):
        data, size = flag_set(raw=b"Ready\0junk" + bytes(118))
        original = T.doc(data, "flg", "switch")
        field = original["record"]["arFlagLookups"][0]["szLabel"]
        self.assertEqual(field, {"text": "Ready", "raw": (b"Ready\0junk" + bytes(118)).hex().upper()})
        self.assertEqual(E.encode(data, "flg", "switch", original), data)
        edited = copy.deepcopy(original)
        edited["record"]["arFlagLookups"][0]["szLabel"]["raw"] = (b"Ready\0JUNK" + bytes(118)).hex().upper()
        first = E.encode(data, "flg", "switch", edited)
        self.assertEqual(T.doc(first, "flg", "switch")["record"]["arFlagLookups"][0]["szLabel"]["raw"],
                         edited["record"]["arFlagLookups"][0]["szLabel"]["raw"])
        self.assertEqual(E.encode(first, "flg", "switch", original), data)
        edited = copy.deepcopy(original)
        edited["record"]["arFlagLookups"][0]["szLabel"]["text"] = "Go"
        second = E.encode(data, "flg", "switch", edited)
        self.assertEqual(T.doc(second, "flg", "switch")["record"]["arFlagLookups"][0]["szLabel"], "Go")


class Version(unittest.TestCase):
    def test_version(self):
        self.assertGreaterEqual(tuple(int(p) for p in d3sno.__version__.split(".")), (0, 23, 1))
        self.assertEqual(d3sno.STAGE, 23)


if __name__ == "__main__":
    unittest.main()
