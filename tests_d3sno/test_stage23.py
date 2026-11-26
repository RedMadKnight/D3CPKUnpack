"""Stage 23 (``55_stage23_effect_types.md``): named export and import of the 31 remaining SNO types
(W23-1 to W23-9). Every file is synthetic, built from the rows of ``f55_layouts.tsv``; exports and outputs go
to a temporary folder outside any git working tree."""

import _libpath  # noqa: F401  (python -I: the library folder on sys.path)
import copy
import json
import struct
import unittest
from unittest import mock

import builders as B
import export_fixture as F
import test_stage21 as S
import d3sno
from d3sno import apply as AP
from d3sno import archive as A
from d3sno import effects as E
from d3sno import export as X
from d3sno import index_write as IW
from d3sno import json_import as J
from d3sno import sno_layout as L
from d3sno.binary import FormatError, build_order
from d3sno.index import TocEntry
from d3sno.sno import expected_version, group_by_extension, group_by_id

BUILDS = ("switch", "x360", "ps3")
CURVE = "primaryVibrationIntensityCurve"
NODES = [(0.0, 1.0, 2.0), (0.5, 3.0, 4.0), (1.0, 5.0, 6.0)]  # (time, min, max)
VIB, LIT, TEX = 4100, 4200, 4300


def rows(build, ext, record):
    return {leaf.name: leaf for leaf in E.leaf_table(build, ext) if leaf.record == record and leaf.offset is not None}


def put(raw, leaf, value, order):
    code = "f" if leaf.type in E.FLOATS or (not leaf.type and ".fl" in "." + leaf.name or "tLoop" in leaf.name) else "i"
    struct.pack_into(order + code, raw, leaf.offset, value)


def blank(build, ext, handle):
    order = build_order(build)
    top = L.top_record(build, ext)
    raw = bytearray(L.record_layout(build, ext, top).size)
    struct.pack_into(order + "I", raw, 0, handle)
    head = struct.pack(order + "4I", 0xDEADBEEF, expected_version(ext, build) or 1, 1, 0x1234)
    return L.SnoTree(build, ext, head, L.Record(top, raw))


def node_slot(build, ext="vib", curve=CURVE):
    top = L.top_record(build, ext)
    slots = L.pair_slots(build, ext, top)
    if build == "ps3":
        return next(s for s in slots if s.path == f"{curve}.ptArrayNodes")
    at = rows(build, ext, E._top_table_record(build, ext))[f"{curve}.ptArrayNodes"].offset + 8
    return next(s for s in slots if s.offset == at)


def nodes_block(build, ext, element, nodes):
    order = build_order(build)
    node_rows = rows(build, ext, "FloatNode")
    size = L.record_layout(build, ext, element).size
    out = []
    for time, low, high in nodes:
        raw = bytearray(size)
        struct.pack_into(order + "f", raw, node_rows["tTime"].offset, time)
        struct.pack_into(order + "f", raw, node_rows["tValueMin"].offset, low)
        struct.pack_into(order + "f", raw, node_rows["tValueMax"].offset, high)
        out.append(L.Record(element, raw))
    return L.PairValue(L.Block(elements=out)) if out else None


def vib_file(build, nodes=NODES, count=None, flags=0x21, handle=VIB):
    """A vibration with a curve of three nodes (W23-3), a flag set with an unnamed bit 5 (W23-5) and a time."""
    tree = blank(build, "vib", handle)
    order = build_order(build)
    top = rows(build, "vib", "VibrationDefinition")
    raw = tree.top.raw
    struct.pack_into(order + "i", raw, top["dwFlags"].offset, flags)
    struct.pack_into(order + "i", raw, top["tSystemDuration"].offset, 120)
    header = f"{CURVE}.tHeader."
    values = {"tLoopPointStart": ("f", 0.25), "tLoopPointEnd": ("f", 0.75), "nDistributionFunction": ("i", 1),
              "tInterpScalar.nScalarFunction": ("i", 2), "tInterpScalar.flScalarLerpZero": ("f", 0.5),
              "tInterpScalar.flScalarLerpOne": ("f", 1.5),
              "nNodeCount": ("i", len(nodes) if count is None else count)}
    for name, (code, value) in values.items():
        if header + name in top:
            struct.pack_into(order + code, raw, top[header + name].offset, value)
    slot = node_slot(build)
    tree.top.pairs[slot.offset] = nodes_block(build, "vib", slot.field.element, nodes)
    return L.write_tree(tree)


def lit_file(build, light_type=1, texture=-1, handle=LIT, open_word=None):
    """A light: an enum, a link (W23-5) and, on PS3, a word no row names (W23-6)."""
    tree = blank(build, "lit", handle)
    order = build_order(build)
    top = rows(build, "lit", E._top_table_record(build, "lit"))
    struct.pack_into(order + "i", tree.top.raw, top["eType"].offset, light_type)
    struct.pack_into(order + "i", tree.top.raw, top["tMatTexProjected.snoTex"].offset, texture)
    if open_word is not None:
        struct.pack_into(order + "I", tree.top.raw, 24, open_word)
    return L.write_tree(tree)


def doc(data, ext, build):
    return J._json_round(E.decode(data, ext, build))


class Tables(unittest.TestCase):
    """W23-1: the tables load; W23-9: ``leaf_table``."""

    def test_rows(self):
        for build in BUILDS:
            for ext in E.GROUPS:
                self.assertTrue(E.leaf_table(build, ext), (build, ext))
        flags = rows("switch", "vib", "VibrationDefinition")["dwFlags"]
        self.assertEqual((flags.offset, flags.size, flags.type), (12, 4, "INT"))
        self.assertTrue(flags.flags)
        opened = [leaf for leaf in E.leaf_table("ps3", "flg") if leaf.offset is None]
        self.assertTrue(opened and all(leaf.status == "open" for leaf in opened))

    def test_enums(self):
        self.assertTrue(E.enum_names("DistributionFunction"))
        self.assertTrue(E.enum_names("ScalarFunction"))
        self.assertIn(0, E.enum_names("<anon_enum:VIBRATION_DEF_CONTINUOUS>", "bit"))

    def test_every_group_every_build_round_trips(self):
        """A record of zeros of every group and build decodes and encodes to its bytes."""
        for build in BUILDS:
            for ext in E.GROUPS:
                with self.subTest(build=build, ext=ext):
                    data = L.write_tree(blank(build, ext, 9))
                    found = E.decode_full(data, ext, build)
                    self.assertEqual(found.json["group"], ext)
                    self.assertEqual(E.encode(data, ext, build, doc(data, ext, build)), data)

    def test_overlapping_row_reported_not_used(self):
        """W23-2: a row overlapping another is a ``layout_conflict``; its bytes are unnamed."""
        table = dict(E._rows())
        extra = E.Leaf("switch", "vib", "VibrationDefinition", "wide", 18, 4, "INT", "", "", "", None,
                       "type information")
        table[("switch", "vib")] = table[("switch", "vib")] + (extra,)
        caches = (E._by_record, E._view, E._top_table_record)
        for cache in caches:
            cache.cache_clear()
        self.addCleanup(lambda: [cache.cache_clear() for cache in caches])
        with mock.patch.object(E, "_rows", lambda: table):
            data = vib_file("switch")
            found = E.decode_full(data, "vib", "switch")
            body = found.json["record"]
            self.assertIn(("VibrationDefinition", "wide"), found.conflicts)
            self.assertNotIn("wide", body)
            self.assertIn("0x0010", body["unnamed"])  # the bytes the faulty rows held
            self.assertEqual(E.encode(data, "vib", "switch", J._json_round(found.json)), data)

    def test_array_row_not_on_a_pair(self):
        table = dict(E._rows())
        extra = E.Leaf("switch", "vib", "VibrationDefinition", "ptStray", 120 + 4, 8, "ARRAY:FloatNode", "", "", "",
                       None, "type information")
        table[("switch", "vib")] = table[("switch", "vib")] + (extra,)
        caches = (E._by_record, E._view, E._top_table_record)
        for cache in caches:
            cache.cache_clear()
        self.addCleanup(lambda: [cache.cache_clear() for cache in caches])
        with mock.patch.object(E, "_rows", lambda: table):
            found = E.decode_full(vib_file("switch"), "vib", "switch")
            self.assertIn(("VibrationDefinition", "ptStray"), found.conflicts)


class Curves(unittest.TestCase):
    """W23-3: a curve of three nodes on each build."""

    def test_three_nodes(self):
        for build in BUILDS:
            with self.subTest(build=build):
                curve = doc(vib_file(build), "vib", build)["record"][CURVE]
                self.assertEqual(curve["loop"], [0.25, 0.75])
                self.assertEqual(curve["distribution"]["value"], 1)
                self.assertEqual(curve["distribution"]["name"], E.enum_names("DistributionFunction").get(1))
                self.assertEqual(curve["scalar"], {"function": {"value": 2, "name": E.enum_names("ScalarFunction").get(2)},
                                                   "at_zero": 0.5, "at_one": 1.5})
                self.assertEqual(curve["nodes"], [{"time": t, "min": a, "max": b} for t, a, b in NODES])
                self.assertNotIn("stored_count", curve)

    def test_count_derived_and_reported(self):
        for build in ("switch", "x360"):
            found = E.decode_full(vib_file(build, count=5), "vib", build)
            self.assertEqual(found.json["record"][CURVE]["stored_count"], 5)
            self.assertIn(("record", CURVE), found.counts_differ)
            self.assertNotIn(f"{CURVE}.tHeader.nNodeCount", found.json["record"])

    def test_ps3_has_no_stored_count(self):
        """A PS3 table row of a node count is a plain leaf, never the curve's count."""
        found = E.decode_full(vib_file("ps3", count=5), "vib", "ps3")
        self.assertNotIn("stored_count", found.json["record"][CURVE])
        self.assertEqual(found.counts_differ, [])

    def test_colour_and_vector_nodes(self):
        """Colours are [r, g, b, a] in stored order; vectors [x, y, z]."""
        build, ext = "switch", "prt"
        view = E._view(build, ext, E._top_table_record(build, ext), L.top_record(build, ext))
        names = [p.name for p in view.pairs] + [n.name for n in view.plain] + [c.name for c in view.curves]
        self.assertEqual(len(names), len(set(names)))  # every key once
        kinds = {}
        for pair in view.pairs:
            if pair.slot.field.is_record_array and pair.element in ("ColorPath", "VectorPath") and pair.name.startswith("path"):
                kinds.setdefault(pair.element, pair)
        self.assertEqual(set(kinds), {"ColorPath", "VectorPath"})
        tree = blank(build, ext, 9)
        for element, pair in kinds.items():
            path = L.Record(pair.slot.field.element, bytearray(L.record_layout(build, ext, pair.slot.field.element).size))
            node_name = {"ColorPath": "ColorNode", "VectorPath": "VectorNode"}[element]
            node_rows = rows(build, ext, node_name)
            node = bytearray(L.record_layout(build, ext, node_name).size)
            if element == "ColorPath":
                node[node_rows["tValueMin"].offset:node_rows["tValueMin"].offset + 4] = bytes((10, 20, 30, 40))
            else:
                for k, axis in enumerate("xyz"):
                    struct.pack_into("<f", node, node_rows[f"tValueMax.{axis}"].offset, float(k + 1))
            path_view = E._view(build, ext, element, pair.slot.field.element)
            (curve,) = path_view.curves
            path.pairs[curve.pair.slot.offset] = L.PairValue(L.Block(elements=[L.Record(node_name, node)]))
            struct.pack_into("<i", path.raw, curve.parts["count"].offset, 1)
            tree.top.pairs[pair.slot.offset] = L.PairValue(L.Block(elements=[path]))
        data = L.write_tree(tree)
        record = doc(data, ext, build)["record"]
        colour = record[kinds["ColorPath"].name][0]["nodes"][0]
        vector = record[kinds["VectorPath"].name][0]["nodes"][0]
        self.assertEqual(colour["min"], [10, 20, 30, 40])
        self.assertEqual(vector["max"], [1.0, 2.0, 3.0])
        self.assertEqual(E.encode(data, ext, build, J._json_round(E.decode(data, ext, build))), data)
        edited = J._json_round(E.decode(data, ext, build))
        edited["record"][kinds["ColorPath"].name][0]["nodes"][0]["min"] = [1, 2, 3, 255]
        changed = E.encode(data, ext, build, edited)
        self.assertEqual(doc(changed, ext, build)["record"][kinds["ColorPath"].name][0]["nodes"][0]["min"],
                         [1, 2, 3, 255])
        edited["record"][kinds["VectorPath"].name][0]["nodes"][0]["max"] = [1.0, 2.5, 3.0]
        vector_edit = doc(E.encode(data, ext, build, edited), ext, build)
        self.assertEqual(vector_edit["record"][kinds["VectorPath"].name][0]["nodes"][0]["max"], [1.0, 2.5, 3.0])
        edited["record"][kinds["ColorPath"].name][0]["nodes"][0]["min"] = [1, 2, 3, 256]
        with self.assertRaises(J.Refused) as caught:
            E.encode(data, ext, build, edited)
        self.assertEqual(caught.exception.reason, "value_out_of_range")


class Lists(unittest.TestCase):
    """W23-8: list elements added and removed."""

    def test_count_written_without_a_count_link(self):
        """The curve's node count follows its nodes even where no count link of W14-20 sets it."""
        for build in ("switch", "x360"):
            data = vib_file(build)
            edited = doc(data, "vib", build)
            edited["record"][CURVE]["nodes"].append({"time": 1.0, "min": 0.0, "max": 0.0})
            with mock.patch.object(L, "derive_counts", lambda tree: 0),                     mock.patch.object(L, "update_count_words", lambda tree, record, offset: 0):
                first = E.encode(data, "vib", build, edited)
            self.assertNotIn("stored_count", doc(first, "vib", build)["record"][CURVE])

    def test_added_element_copies_the_last(self):
        """A new element is a copy of the last one: its unnamed words come along."""
        build, ext = "switch", "prt"
        tree = blank(build, ext, 9)
        slot = next(s for s in L.pair_slots(build, ext, L.top_record(build, ext)) if s.path == "pathSystemScale")
        element = L.Record(slot.field.element, bytearray(L.record_layout(build, ext, slot.field.element).size))
        struct.pack_into("<I", element.raw, 0x1C, 0x55)
        tree.top.pairs[slot.offset] = L.PairValue(L.Block(elements=[element]))
        data = L.write_tree(tree)
        original = doc(data, ext, build)
        self.assertEqual(original["record"]["pathSystemScale"][0]["unnamed"]["0x001C"], 0x55)
        edited = copy.deepcopy(original)
        edited["record"]["pathSystemScale"].append(copy.deepcopy(original["record"]["pathSystemScale"][0]))
        first = E.encode(data, ext, build, edited)
        self.assertEqual(doc(first, ext, build)["record"]["pathSystemScale"][1]["unnamed"]["0x001C"], 0x55)
        self.assertEqual(E.encode(first, ext, build, original), data)

    def test_keys_stay_unique(self):
        """A leaf named as a tree pair: the pair's key takes its offset, and nothing is lost."""
        table = dict(E._rows())
        extra = E.Leaf("ps3", "vib", "VibrationDefinition", f"{CURVE}.ptArrayNodes", 76, 4, "INT", "", "", "", None,
                       "aligned by value")
        table[("ps3", "vib")] = table[("ps3", "vib")] + (extra,)
        caches = (E._by_record, E._view, E._top_table_record)
        for cache in caches:
            cache.cache_clear()
        self.addCleanup(lambda: [cache.cache_clear() for cache in caches])
        with mock.patch.object(E, "_rows", lambda: table):
            tree = L.parse_tree(vib_file("ps3"), "vib", "ps3")
            struct.pack_into(">i", tree.top.raw, 76, 7)
            data = L.write_tree(tree)
            record = doc(data, "vib", "ps3")["record"]
            self.assertEqual(record[f"{CURVE}.ptArrayNodes"], 7)
            self.assertEqual(len(record[f"{CURVE}.ptArrayNodes@44"]), 3)
            self.assertEqual(E.encode(data, "vib", "ps3", doc(data, "vib", "ps3")), data)


class Values(unittest.TestCase):
    """W23-4 to W23-6."""

    def test_flags_with_an_unnamed_bit(self):
        names = E.enum_names("<anon_enum:VIBRATION_DEF_CONTINUOUS>", "bit")
        for build in BUILDS:
            flags = doc(vib_file(build), "vib", build)["record"]["dwFlags"]
            self.assertEqual(flags, {"value": 0x21, "names": [names[0], "bit5"]})

    def test_time_as_stored(self):
        for build in BUILDS:
            self.assertEqual(doc(vib_file(build), "vib", build)["record"]["tSystemDuration"], 120)

    def test_enum_named_and_unnamed(self):
        names = E.enum_names("LightType")
        for build in BUILDS:
            record = doc(lit_file(build, 1), "lit", build)["record"]
            self.assertEqual(record["eType"], {"value": 1, "name": names.get(1)})
            self.assertIsNotNone(names.get(1))
            record = doc(lit_file(build, 999), "lit", build)["record"]
            self.assertEqual(record["eType"], {"value": 999, "name": None})

    def test_links(self):
        for build in BUILDS:
            group = rows(build, "lit", E._top_table_record(build, "lit"))["tMatTexProjected.snoTex"].group
            record = doc(lit_file(build, texture=-1), "lit", build)["record"]
            self.assertIsNone(record["tMatTexProjected.snoTex"])
            self.assertIsNone(doc(lit_file(build, texture=0), "lit", build)["record"]["tMatTexProjected.snoTex"])
            found = E.decode_full(lit_file(build, texture=TEX), "lit", build)
            self.assertEqual(found.json["record"]["tMatTexProjected.snoTex"],
                             {"group": group_by_id(group).extension, "handle": TEX})
            self.assertEqual(found.links, [(("record", "tMatTexProjected.snoTex"), group, TEX)])

    def test_ps3_open_word(self):
        """W23-6: a PS3 word no row places is exported under ``unnamed`` and imports as stored."""
        data = lit_file("ps3", open_word=0xCAFEF00D)
        found = E.decode_full(data, "lit", "ps3")
        self.assertGreater(found.open, 0)
        self.assertEqual(found.json["record"]["unnamed"]["0x0018"], 0xCAFEF00D)
        self.assertEqual(E.encode(data, "lit", "ps3", J._json_round(found.json)), data)

    def test_schema(self):
        schema = E.schema("vib")
        self.assertEqual(set(schema), set(BUILDS))
        row = next(r for r in schema["switch"] if r["leaf"] == "tSystemDuration")
        self.assertEqual((row["type"], row["unit"]), ("TIME", "ticks (60 per second), int32"))


class RoundTrip(unittest.TestCase):
    """W23-8 on every build."""

    def test_unedited(self):
        for build in BUILDS:
            for data, ext in ((vib_file(build), "vib"), (lit_file(build, texture=TEX), "lit")):
                self.assertEqual(E.encode(data, ext, build, doc(data, ext, build)), data)

    def test_node_added_then_original(self):
        for build in BUILDS:
            with self.subTest(build=build):
                data = vib_file(build)
                original = doc(data, "vib", build)
                edited = copy.deepcopy(original)
                edited["record"][CURVE]["nodes"].append({"time": 1.0, "min": 7.0, "max": 8.0})
                first = E.encode(data, "vib", build, edited)
                self.assertNotEqual(first, data)
                again = doc(first, "vib", build)
                self.assertEqual(again["record"][CURVE]["nodes"][-1], {"time": 1.0, "min": 7.0, "max": 8.0})
                self.assertEqual(len(again["record"][CURVE]["nodes"]), 4)
                self.assertNotIn("stored_count", again["record"][CURVE])  # the count follows (Switch, x360)
                self.assertEqual(E.encode(first, "vib", build, original), data)

    def test_node_removed_and_first_node_added(self):
        for build in BUILDS:
            data = vib_file(build, nodes=[])
            original = doc(data, "vib", build)
            edited = copy.deepcopy(original)
            edited["record"][CURVE]["nodes"] = [{"time": 0.0, "min": 1.0, "max": 1.0}]
            first = E.encode(data, "vib", build, edited)
            self.assertEqual(doc(first, "vib", build)["record"][CURVE]["nodes"], edited["record"][CURVE]["nodes"])
            self.assertEqual(E.encode(first, "vib", build, original), data)

    def test_unnamed_word_changed_and_back(self):
        for build in BUILDS:
            data = vib_file(build)
            original = doc(data, "vib", build)
            key = sorted(original["record"]["unnamed"])[-1]
            edited = copy.deepcopy(original)
            edited["record"]["unnamed"][key] = 0x01020304
            first = E.encode(data, "vib", build, edited)
            self.assertEqual(doc(first, "vib", build)["record"]["unnamed"][key], 0x01020304)
            self.assertEqual(E.encode(first, "vib", build, original), data)

    def test_value_changes(self):
        for build in BUILDS:
            data = lit_file(build, texture=TEX)
            original = doc(data, "lit", build)
            names = E.enum_names("LightType")
            other = next(v for v in sorted(names) if v != 1)
            edited = copy.deepcopy(original)
            edited["record"]["eType"] = {"value": other, "name": names[other]}
            edited["record"]["tMatTexProjected.snoTex"]["handle"] = TEX + 1
            first = E.encode(data, "lit", build, edited)
            record = doc(first, "lit", build)["record"]
            self.assertEqual(record["eType"]["value"], other)
            self.assertEqual(record["tMatTexProjected.snoTex"]["handle"], TEX + 1)
            edited["record"]["tMatTexProjected.snoTex"] = None
            self.assertIsNone(doc(E.encode(data, "lit", build, edited), "lit", build)["record"]["tMatTexProjected.snoTex"])
            self.assertEqual(E.encode(first, "lit", build, original), data)

    def test_flag_names(self):
        for build in BUILDS:
            data = vib_file(build, flags=0x1)
            edited = doc(data, "vib", build)
            edited["record"]["dwFlags"]["names"].append("bit31")
            record = doc(E.encode(data, "vib", build, edited), "vib", build)["record"]
            self.assertEqual(record["dwFlags"]["value"] & 0xFFFFFFFF, 0x80000001)


class Refusals(unittest.TestCase):
    """W23-8, W22-3: a changed name must agree with a changed value."""

    def refused(self, data, ext, build, edit, reason):
        edited = doc(data, ext, build)
        edit(edited)
        with self.assertRaises(J.Refused) as caught:
            E.encode(data, ext, build, edited)
        self.assertEqual(caught.exception.reason, reason, caught.exception.message)
        return caught.exception

    def test_enum_name_and_value_disagree(self):
        names = E.enum_names("LightType")
        other = next(v for v in sorted(names) if v != 1)
        self.refused(lit_file("switch"), "lit", "switch",
                     lambda d: d["record"].__setitem__("eType", {"value": 999, "name": names[other]}),
                     "conflicting_values")
        self.refused(lit_file("switch"), "lit", "switch",
                     lambda d: d["record"]["eType"].__setitem__("name", "NO_SUCH_NAME"), "unknown_name")

    def test_flag_name_unknown(self):
        self.refused(vib_file("x360"), "vib", "x360",
                     lambda d: d["record"]["dwFlags"]["names"].append("NO_SUCH_BIT"), "unknown_name")

    def test_link_group(self):
        self.refused(lit_file("ps3", texture=TEX), "lit", "ps3",
                     lambda d: d["record"]["tMatTexProjected.snoTex"].__setitem__("group", "acr"), "wrong_group")

    def test_header_fixed(self):
        self.refused(vib_file("switch"), "vib", "switch", lambda d: d["header"].__setitem__("version", 1), "field_fixed")

    def test_keys(self):
        self.refused(vib_file("switch"), "vib", "switch", lambda d: d["record"].__setitem__("new", 1), "bad_json")

    def test_unnamed_out_of_range(self):
        def edit(d):
            key = sorted(d["record"]["unnamed"])[-1]
            d["record"]["unnamed"][key] = -1
        self.refused(vib_file("switch"), "vib", "switch", edit, "value_out_of_range")

    def test_number_out_of_range(self):
        self.refused(vib_file("ps3"), "vib", "ps3", lambda d: d["record"].__setitem__("tSystemDuration", 1 << 40),
                     "value_out_of_range")

    def test_count_with_changed_nodes(self):
        def edit(d):
            d["record"][CURVE]["nodes"].append({"time": 1.0, "min": 0.0, "max": 0.0})
        self.refused(vib_file("switch", count=5), "vib", "switch", edit, "derived_field")

    def test_stored_count_written(self):
        data = vib_file("switch", count=5)
        edited = doc(data, "vib", "switch")
        edited["record"][CURVE]["stored_count"] = 6
        self.assertEqual(doc(E.encode(data, "vib", "switch", edited), "vib", "switch")["record"][CURVE]["stored_count"], 6)

    def test_every_refusal_listed(self):
        def edit(d):
            d["record"]["tSystemDuration"] = 1 << 40
            d["record"]["dwFlags"]["names"].append("NO_SUCH_BIT")
        error = self.refused(vib_file("switch"), "vib", "switch", edit, "unknown_name")
        self.assertEqual(len(error.refusals), 2)


class ThroughImportJson(unittest.TestCase):
    """W23-8: ``json_import`` hands the effect groups to ``effects``."""

    def test_import_json(self):
        data = vib_file("switch")
        self.assertIn("vib", J.GROUPS)
        old = J._json_round(J.template_json("vib", data, "switch"))
        self.assertEqual(old, doc(data, "vib", "switch"))
        edited = copy.deepcopy(old)
        edited["record"]["tSystemDuration"] = 240
        result = J.import_json("vib", data, "switch", edited)
        self.assertEqual(result.changes, [{"path": "record.tSystemDuration", "old": 120, "new": 240}])
        self.assertEqual(J.import_json("vib", result.data, "switch", old).data, data)

    def test_dangling_link_warning(self):
        data = lit_file("switch", texture=TEX)
        edited = doc(data, "lit", "switch")
        edited["record"]["tMatTexProjected.snoTex"]["handle"] = 77
        result = J.import_json("lit", data, "switch", edited, known=lambda g, h: False)
        self.assertTrue(any(w.startswith("dangling_link") for w in result.warnings))


def effect_files():
    o = "<"
    vib = vib_file("switch")
    lit = lit_file("switch", texture=TEX)
    vib_count = vib_file("switch", count=9, handle=VIB + 1)
    toc = IW.write_core_toc([TocEntry(group_by_extension("vib").id, VIB, "Rumble"),
                             TocEntry(group_by_extension("vib").id, VIB + 1, "Rumble2"),
                             TocEntry(group_by_extension("lit").id, LIT, "Torch")], "switch")
    return {"Common": {F.name_of("vib", "Rumble"): vib, F.name_of("vib", "Rumble2"): vib_count,
                       F.name_of("lit", "Torch"): lit, "CoreTOC.dat": toc}}


class Export(unittest.TestCase):
    """W23-7: the groups are exported by default, with schemas and coverage."""

    def test_export_and_apply(self):
        files = effect_files()
        setup = S.Setup(self, files)
        for ext in E.GROUPS:
            self.assertIn(ext, X.EXPORT_GROUPS)
        written = json.loads((setup.export / f"data/vib/{VIB}.json").read_text(encoding="utf-8"))
        self.assertEqual(written, doc(files["Common"][F.name_of("vib", "Rumble")], "vib", "switch"))
        manifest = setup.manifest
        self.assertEqual(set(manifest["schemas"]), set(E.GROUPS))
        self.assertEqual(manifest["schemas"]["vib"], E.schema("vib"))
        coverage = json.loads((setup.export / "coverage.json").read_text(encoding="utf-8"))["groups"]
        self.assertEqual(coverage["vib"]["exported"], 2)
        self.assertGreater(coverage["vib"]["leaves"]["switch"]["named"], 0)
        self.assertGreater(coverage["vib"]["leaves"]["switch"]["unnamed"], 0)
        reasons = {(f["handle"], f["reason"]) for g in ("vib", "lit") for f in coverage[g]["files"]}
        self.assertIn((VIB + 1, "curve_count_differs"), reasons)
        self.assertIn((LIT, "link_unresolved"), reasons)  # the texture is not in the index
        self.assertNotIn((VIB, "curve_count_differs"), reasons)
        path = setup.edit / f"data/vib/{VIB}.json"
        edited = json.loads(path.read_text(encoding="utf-8"))
        edited["record"][CURVE]["nodes"].append({"time": 1.0, "min": 9.0, "max": 9.5})
        path.write_text(json.dumps(edited), encoding="utf-8")
        report, out = setup.ok()
        rows_ = {r["path"]: r["result"] for r in report["changed"]}
        self.assertEqual(rows_, {f"data/vib/{VIB}.json": "applied"})
        stored = A.read_archive((out / "Common.cpk").read_bytes()).read_file(F.name_of("vib", "Rumble"))
        self.assertEqual(len(doc(stored, "vib", "switch")["record"][CURVE]["nodes"]), 4)
        self.assertEqual(E.encode(stored, "vib", "switch", written), files["Common"][F.name_of("vib", "Rumble")])

    def test_groups_filter(self):
        root = F.temp_root(self)
        game = F.write_game(root / "game", effect_files())
        result = X.export_game(game, root / "out", created="X", groups=["vib"])
        self.assertEqual(result.exit_code, 0, result.message)
        self.assertEqual(set(result.manifest["schemas"]), {"vib"})
        self.assertFalse((root / "out" / "data" / "lit").exists())


class Version(unittest.TestCase):
    def test_version(self):
        self.assertGreaterEqual(tuple(int(p) for p in d3sno.__version__.split(".")), (0, 23, 0))
        self.assertEqual(d3sno.STAGE, 23)


if __name__ == "__main__":
    unittest.main()
