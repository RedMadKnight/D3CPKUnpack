"""Stage 25 (``60_stage25_create.md``): creating files of the 59 table types from their JSON alone (W25-1 to
W25-8). Every file is synthetic, built from the rows of ``f58_layouts.tsv``; archives, index files and outputs go
to a temporary folder outside any git working tree."""

import _libpath  # noqa: F401  (python -I: the library folder on sys.path)
import contextlib
import copy
import io
import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import export_fixture as F
import test_stage23 as T
import test_stage24 as S
import d3sno
from d3sno import archive as A
from d3sno import cli
from d3sno import effects as E
from d3sno import export as X
from d3sno import index as I
from d3sno import index_write as IW
from d3sno import json_import as J
from d3sno import sno_layout as L
from d3sno.binary import build_order
from d3sno.index import TocEntry
from d3sno.sno import expected_version, group_by_extension, parse_header

BUILDS = ("switch", "x360", "ps3")
GID = {ext: group_by_extension(ext).id for ext in ("acr", "app", "lit", "tex", "ans", "glo", "phm", "cnd")}
LIT, ANS, CND = 5001, 5002, 5003


def roundtrip(case, data, ext, build):
    document = J._json_round(E.decode(data, ext, build))
    case.assertEqual(E.create(document, ext, build), data, (build, ext))
    return document


def node_slots(build, ext, element):
    top = L.top_record(build, ext)
    return [s for s in L.pair_slots(build, ext, top) if s.field.element == element and s.field.is_record_array]


def lit_shared(build, shared=True):
    """A light whose first two curves of ``FloatNode`` hold equal nodes: one block (``shared``) or two."""
    tree = T.blank(build, "lit", LIT)
    first, second = node_slots(build, "lit", node_slots(build, "lit", "FloatNode")[0].field.element)[:2]
    value = T.nodes_block(build, "lit", first.field.element, T.NODES)
    tree.top.pairs[first.offset] = value
    tree.top.pairs[second.offset] = L.PairValue(value.block) if shared else \
        T.nodes_block(build, "lit", first.field.element, T.NODES)
    return L.write_tree(tree, derive=True)


def ans_maps(build, shared=True):
    """An animation set whose two tag maps hold equal bytes: one block (``shared``) or two."""
    tree = T.blank(build, "ans", ANS)
    first, second = [s for s in L.pair_slots(build, "ans", L.top_record(build, "ans")) if s.field.kind == "tagmap"][:2]
    data = S.tag_map(build, [(0x1025A, 7)])
    block = L.Block(data=data)
    tree.top.pairs[first.offset] = L.PairValue(block)
    tree.top.pairs[second.offset] = L.PairValue(block if shared else L.Block(data=bytes(data)))
    return L.write_tree(tree, derive=True)


def particles(build, sharing):
    """A particle system whose first two path lists of one element type hold one path with nodes: the lists share
    one block (``outer``), or each has its own path whose nodes are one block (``inner``); ``both``: as ``inner``,
    and a third list shares the second list's block."""
    tree = T.blank(build, "prt", 5004)
    slots = [s for s in L.pair_slots(build, "prt", tree.top.name) if s.field.is_record_array]
    first = next(s for s in slots if sum(o.field.element == s.field.element for o in slots) > 1
                 and [n for n in L.pair_slots(build, "prt", s.field.element) if n.field.element == "FloatNode"])
    second = next(s for s in slots if s is not first and s.field.element == first.field.element)
    nodes_slot = next(s for s in L.pair_slots(build, "prt", first.field.element) if s.field.is_record_array)

    def path(nodes):
        element = L.new_record(build, "prt", first.field.element)
        element.pairs[nodes_slot.offset] = nodes
        return element

    nodes = T.nodes_block(build, "prt", nodes_slot.field.element, T.NODES)
    if sharing == "outer":
        block = L.Block(elements=[path(nodes)])
        tree.top.pairs[first.offset] = L.PairValue(block)
        tree.top.pairs[second.offset] = L.PairValue(block)
    else:
        tree.top.pairs[first.offset] = L.PairValue(L.Block(elements=[path(nodes)]))
        other = path(L.PairValue(nodes.block))
        view = E._view(build, "prt", E._table_record(build, "prt", first.field.element), first.field.element)
        parts = view.curves[0].parts if view.curves else {}
        at = parts["loop_start"].offset if "loop_start" in parts else view.unnamed[0][0]
        struct.pack_into(build_order(build) + "f", other.raw, at, 2.0)  # the two paths differ
        tree.top.pairs[second.offset] = L.PairValue(L.Block(elements=[other]))
        if sharing == "both":
            third = [s for s in slots if s.field.element == first.field.element][2]
            tree.top.pairs[third.offset] = L.PairValue(tree.top.pairs[second.offset].block)
    return L.write_tree(tree, derive=True)


def actor_text(build):
    """An actor with a tag map, links of -1 and 0 (``snoAnimSet``, ``snoPhysMesh``) and two text pairs."""
    tree = L.parse_tree(S.actor(build, 4001, 4003, [(S.PROJECTILE, 4002)]), "acr", build)
    for k, slot in enumerate(s for s in L.pair_slots(build, "acr", tree.top.name) if s.field.kind == "text"):
        tree.top.pairs[slot.offset] = L.PairValue(L.Block(data=f"Text {k}\0".encode()))
    return L.write_tree(tree, derive=True)


def condition(build):
    """A condition with a text field (``CHARARRAY``)."""
    tree = T.blank(build, "cnd", CND)
    leaf = S.row(build, "cnd", E._top_table_record(build, "cnd"), "tGameFlagSubconditions[0].szLabel")
    tree.top.raw[leaf.offset:leaf.offset + 6] = b"Label\0"
    return L.write_tree(tree, derive=True)


class Rebuild(unittest.TestCase):
    """``create(decode(f))`` gives ``f`` (W25-5)."""

    def test_top_record_only_every_type(self):
        for build in BUILDS:
            for ext in E.types(build):
                with self.subTest(build=build, ext=ext):
                    roundtrip(self, L.write_tree(T.blank(build, ext, 9)), ext, build)

    def test_kinds(self):
        for build in BUILDS:
            files = [(T.vib_file(build), "vib"), (T.lit_file(build, texture=77), "lit"), (lit_shared(build), "lit"),
                     (actor_text(build), "acr"),
                     (ans_maps(build), "ans"), (ans_maps(build, shared=False), "ans")]
            if build != "ps3":  # PS3 has no such rows (or they are open)
                files += [(condition(build), "cnd"), (S.material(build, 4006), "mat"),
                          (S.global_file(build, [4001, 4002, 0]), "glo"), (particles(build, "outer"), "prt"),
                          (particles(build, "inner"), "prt"),
                          (S.recipe(build, 11, [12, 13], 2), "rcp")]
            for data, ext in files:
                with self.subTest(build=build, ext=ext):
                    roundtrip(self, data, ext, build)

    def test_curve_with_stored_count(self):
        """W23-3: a count that differs from the nodes is kept as stored."""
        for build in ("switch", "x360"):
            data = T.vib_file(build, count=5)
            document = roundtrip(self, data, "vib", build)
            self.assertEqual(document["record"][T.CURVE]["stored_count"], 5)

    def test_nodes_added_in_the_json(self):
        for build in BUILDS:
            document = J._json_round(E.decode(T.vib_file(build), "vib", build))
            nodes = document["record"][T.CURVE]["nodes"]
            nodes.append(copy.deepcopy(nodes[-1]))
            nodes[-1]["time"] = 2.0
            data = E.create(document, "vib", build)
            again = E.decode(data, "vib", build)["record"][T.CURVE]
            self.assertEqual(len(again["nodes"]), 4)
            self.assertNotIn("stored_count", again)
            self.assertEqual(data, T.vib_file(build, nodes=T.NODES + [(2.0, 5.0, 6.0)]))


class EmptyLinks(unittest.TestCase):
    """W25-2: ``null`` is -1; a stored 0 is ``{"group", "handle": 0}``."""

    def test_export_and_import(self):
        for build in BUILDS:
            data = actor_text(build)
            record = E.decode(data, "acr", build)["record"]
            self.assertIsNone(record["snoAnimSet"])
            self.assertEqual(record["snoPhysMesh"], {"group": "phm", "handle": 0})
            self.assertEqual(record["snoAppearance"], {"group": "app", "handle": 4003})
            self.assertEqual(E.decode_full(data, "acr", build).links[0][1:], (GID["app"], 4003))
            self.assertFalse([link for link in E.decode_full(data, "acr", build).links if link[2] == 0])

    def test_null_writes_minus_one(self):
        build = "switch"
        document = J._json_round(E.decode(actor_text(build), "acr", build))
        document["record"]["snoPhysMesh"] = None
        document["record"]["snoAnimSet"] = {"group": "ans", "handle": 0}
        record = E.decode(E.create(document, "acr", build), "acr", build)["record"]
        self.assertIsNone(record["snoPhysMesh"])
        self.assertEqual(record["snoAnimSet"], {"group": "ans", "handle": 0})
        top = E._top_table_record(build, "acr")
        raw = L.parse_tree(E.create(document, "acr", build), "acr", build).top.raw
        self.assertEqual(struct.unpack_from("<i", raw, S.row(build, "acr", top, "snoPhysMesh").offset)[0], -1)

    def test_unedited_json_gives_the_template(self):
        for build in BUILDS:
            for data, ext in ((actor_text(build), "acr"), (T.lit_file(build, texture=0), "lit")):
                document = J._json_round(E.decode(data, ext, build))
                self.assertEqual(E.encode(data, ext, build, document), data)
                del document["shared"]  # an older export without the list
                self.assertEqual(E.encode(data, ext, build, document), data)


class Shared(unittest.TestCase):
    """W25-3."""

    def test_listed(self):
        for build in BUILDS:
            data = lit_shared(build)
            document = J._json_round(E.decode(data, "lit", build))
            self.assertEqual(len(document["shared"]), 1, build)
            path, first = document["shared"][0]
            self.assertTrue(path.startswith("record.") and first.startswith("record.") and path != first)
            self.assertEqual(E.create(document, "lit", build, pool=False), data)
            self.assertEqual(E.decode(lit_shared(build, shared=False), "lit", build)["shared"], [])

    def test_hand_written_pools_on_a_type_that_always_shares(self):
        for build in BUILDS:
            self.assertIn("lit", E.ALWAYS_SHARED[build])
            document = J._json_round(E.decode(lit_shared(build), "lit", build))
            del document["shared"]
            self.assertEqual(E.create(document, "lit", build), lit_shared(build))
            apart = E.create(document, "lit", build, pool=False)
            self.assertEqual(apart, lit_shared(build, shared=False))
            self.assertEqual(E.decode(apart, "lit", build)["shared"], [])

    def test_unlisted_pair_kept_apart_where_sharing_varies(self):
        for build in BUILDS:
            self.assertNotIn("ans", E.ALWAYS_SHARED[build])
            apart = ans_maps(build, shared=False)
            document = J._json_round(E.decode(apart, "ans", build))
            self.assertEqual(document["shared"], [])
            self.assertEqual(E.create(document, "ans", build), apart)
            document["shared"] = J._json_round(E.decode(ans_maps(build), "ans", build))["shared"]
            self.assertEqual(E.create(document, "ans", build), ans_maps(build))

    def test_listed_pair_must_hold_the_same_json(self):
        build = "switch"
        document = J._json_round(E.decode(lit_shared(build), "lit", build))
        path = document["shared"][0][0]
        curve = path[len("record."):-len(".nodes")]
        document["record"][curve]["nodes"][0]["time"] = 0.125
        with self.assertRaises(J.Refused) as caught:
            E.create(document, "lit", build)
        self.assertEqual(caught.exception.reason, "conflicting_values")
        self.assertEqual(caught.exception.path, ("shared", 0))
        document["shared"] = [["record.nothing", path]]
        with self.assertRaises(J.Refused) as caught:
            E.create(document, "lit", build)
        self.assertEqual(caught.exception.reason, "bad_json")

    def test_pairs_inside_a_shared_block_are_not_listed(self):
        """A list of paths shared by two pairs: the nodes of its elements come with it, unlisted. Two lists of
        their own whose elements share one node block: that pair is listed. (PS3 ``prt`` has no node rows.)"""
        for build in ("switch", "x360"):
            outer = J._json_round(E.decode(particles(build, "outer"), "prt", build))
            self.assertEqual(len(outer["shared"]), 1, build)
            self.assertFalse(outer["shared"][0][0].endswith("nodes"))
            inner = J._json_round(E.decode(particles(build, "inner"), "prt", build))
            self.assertEqual(len(inner["shared"]), 1, build)
            self.assertTrue(inner["shared"][0][0].endswith("ptArrayNodes") or inner["shared"][0][0].endswith("nodes"))
            both = J._json_round(E.decode(particles(build, "both"), "prt", build))
            self.assertEqual(len(both["shared"]), 2, build)  # the nodes once, not again inside the third list
            self.assertEqual(E.create(both, "prt", build, pool=False), particles(build, "both"))
            for document, data in ((outer, particles(build, "outer")), (inner, particles(build, "inner"))):
                del document["shared"]
                if "prt" in E.ALWAYS_SHARED[build]:
                    self.assertEqual(E.create(document, "prt", build), data, build)
                else:
                    self.assertEqual(E.decode(E.create(document, "prt", build), "prt", build)["shared"], [])


class Header(unittest.TestCase):
    """W25-4."""

    def test_defaults(self):
        for build in BUILDS:
            data = E.create({"build": build, "record": {}}, "lit", build)
            header = parse_header(data, build)
            self.assertEqual((header.version, header.revision, header.source_hash, header.handle),
                             (expected_version("lit", build), 1, 0, 0))
            self.assertEqual(data, E.create({"record": {}}, "lit", build))

    def test_given(self):
        document = {"header": {"handle": 77, "revision": 3, "source_hash": 0xABCD}, "build": "ps3", "group": "lit"}
        data = E.create(document, "lit")
        header = parse_header(data, "ps3")
        self.assertEqual((header.handle, header.revision, header.source_hash), (77, 3, 0xABCD))
        self.assertEqual(E.decode(data, "lit", "ps3")["record"][E.OWN_HANDLE], 77)

    def test_version_from_the_table(self):
        data, warnings = E.create_result({"header": {"version": 1}, "record": {}}, "lit", "switch")
        self.assertEqual(parse_header(data, "switch").version, expected_version("lit", "switch"))
        self.assertTrue(warnings and warnings[0].startswith("version"))

    def test_handle_conflict(self):
        with self.assertRaises(J.Refused) as caught:
            E.create({"header": {"handle": 5}, "record": {E.OWN_HANDLE: 6}}, "lit", "switch")
        self.assertEqual(caught.exception.reason, "conflicting_values")
        self.assertEqual(parse_header(E.create({"record": {E.OWN_HANDLE: 6}}, "lit", "switch"), "switch").handle, 6)


class HandWritten(unittest.TestCase):
    """W25-5: keys left out stay zero; lists take the JSON's length."""

    def test_partial_json(self):
        build = "switch"
        top = E._top_table_record(build, "vib")
        document = {"build": build, "record": {T.CURVE: {"nodes": [{"time": 0.5}, {"time": 1.0, "max": 2.0}]},
                                               "tSystemDuration": 30}}
        data = E.create(document, "vib")
        record = E.decode(data, "vib", build)["record"]
        self.assertEqual([n["time"] for n in record[T.CURVE]["nodes"]], [0.5, 1.0])
        self.assertEqual(record[T.CURVE]["nodes"][1]["max"], 2.0)
        self.assertEqual(record["tSystemDuration"], 30)
        raw = L.parse_tree(data, "vib", build).top.raw
        count = S.row(build, "vib", top, f"{T.CURVE}.tHeader.nNodeCount")
        self.assertEqual(struct.unpack_from("<i", raw, count.offset)[0], 2)

    def test_enum_and_flags(self):
        build = "switch"
        document = J._json_round(E.decode(T.lit_file(build, light_type=1), "lit", build))
        name = E.enum_names(S.row(build, "lit", E._top_table_record(build, "lit"), "eType").enum)[2]
        document["record"]["eType"] = {"name": name}
        with self.assertRaises(J.Refused):  # an enum is {"value", "name"}
            E.create(document, "lit", build)
        document["record"]["eType"] = {"value": 2, "name": name}
        self.assertEqual(E.decode(E.create(document, "lit", build), "lit", build)["record"]["eType"]["value"], 2)
        document = J._json_round(E.decode(T.vib_file(build, flags=-0x7FFFFFDF), "vib", build))
        self.assertEqual(E.create(document, "vib", build), T.vib_file(build, flags=-0x7FFFFFDF))
        document["record"]["dwFlags"] = {"names": document["record"]["dwFlags"]["names"]}
        self.assertEqual(E.create(document, "vib", build), T.vib_file(build, flags=-0x7FFFFFDF))


    def test_count_without_a_count_link(self):
        """A curve whose count no PROVEN link sets (``lit`` ``tStateA.pathHotspot``) takes its node count."""
        for build in ("switch", "x360"):
            document = {"record": {"tStateA.pathHotspot": {"nodes": [{"time": 0.0}, {"time": 1.0}]}}}
            curve = E.decode(E.create(document, "lit", build), "lit", build)["record"]["tStateA.pathHotspot"]
            self.assertEqual(len(curve["nodes"]), 2)
            self.assertNotIn("stored_count", curve)

    def test_stored_count_without_a_count(self):
        """A curve with no count row refuses a stored count (``61``: every curve of ``f61`` has one, so the row is
        taken out of a copy of the table)."""
        table = dict(E._rows())
        name = f"{T.CURVE}.tHeader.nNodeCount"
        table[("ps3", "vib")] = tuple(r for r in table[("ps3", "vib")] if r.name != name)
        caches = (E._by_record, E._view, E._top_table_record, E._zero_text)
        for cache in caches:
            cache.cache_clear()
        self.addCleanup(lambda: [cache.cache_clear() for cache in caches])
        with mock.patch.object(E, "_rows", lambda: table):
            with self.assertRaises(J.Refused) as caught:
                E.create({"record": {T.CURVE: {"stored_count": 4, "nodes": []}}}, "vib", "ps3")
        self.assertEqual((caught.exception.path[-1], caught.exception.reason), ("stored_count", "field_fixed"))

    def test_bytes_form_of_a_value_array(self):
        for build in ("switch", "x360"):
            data = S.global_file(build, [4001, 4002, 0])
            document = J._json_round(E.decode(data, "glo", build))
            key = "tBannerParams.arActorBases"
            values = document["record"][key]
            order = "<" if build == "switch" else ">"
            document["record"][key] = {"bytes": struct.pack(order + "3i", *values).hex().upper()}
            self.assertEqual(E.create(document, "glo", build), data)

    def test_non_finite_float(self):
        """A non-finite float, as the export writes it (a string)."""
        build = "switch"
        document = {"record": {T.CURVE: {"nodes": [{"time": "inf", "min": "-inf"}]}}}
        node = E.decode(E.create(document, "vib", build), "vib", build)["record"][T.CURVE]["nodes"][0]
        self.assertEqual((node["time"], node["min"]), ("inf", "-inf"))


class Refusals(unittest.TestCase):
    """W25-5: with the path and a reason (W22-3); nothing is written."""

    def refused(self, document, ext="lit", build="switch"):
        with self.assertRaises(J.Refused) as caught:
            E.create(document, ext, build)
        return caught.exception

    def test_unknown_leaf(self):
        error = self.refused({"record": {"noSuchLeaf": 1}})
        self.assertEqual((error.path, error.reason), (("record", "noSuchLeaf"), "unknown_name"))
        error = self.refused({"record": {}, "extra": 1})
        self.assertEqual(error.reason, "unknown_name")

    def test_overlapping_unnamed_word(self):
        top = E._top_table_record("switch", "lit")
        leaf = S.row("switch", "lit", top, "eType")
        error = self.refused({"record": {"unnamed": {E._word_key(leaf.offset): 1}}})
        self.assertEqual((error.path[-1], error.reason), (E._word_key(leaf.offset), "conflicting_values"))

    def test_out_of_range(self):
        error = self.refused({"record": {"eType": {"value": 1 << 40, "name": None}}})
        self.assertEqual(error.reason, "value_out_of_range")
        error = self.refused({"header": {"revision": -1}, "record": {}})
        self.assertEqual((error.path, error.reason), (("header", "revision"), "value_out_of_range"))

    def test_every_refusal_listed(self):
        error = self.refused({"record": {"a": 1, "b": 2}})
        self.assertEqual([r.path for r in error.refusals], [("record", "a"), ("record", "b")])

    def test_type_build_and_group(self):
        self.assertEqual(self.refused({"record": {}}, "tex").reason, "unsupported_value")
        self.assertEqual(self.refused({"build": "ps3", "record": {}}).reason, "conflicting_values")
        self.assertEqual(self.refused({"group": "vib", "record": {}}).reason, "conflicting_values")
        with self.assertRaises(J.Refused):
            E.create({"record": {}}, "lit")  # no build


def light_archive(build="switch"):
    """An archive with a texture; its CoreTOC (the texture, a light) and Prefetch."""
    order = build_order(build)
    files = {F.name_of("tex", "Glow"): b"texture bytes", F.name_of("lit", "Lamp"): T.lit_file(build, texture=-1)}
    toc = IW.write_core_toc([TocEntry(GID["tex"], 6001, "Glow"), TocEntry(GID["lit"], 6002, "Lamp")], build)
    prefetch = IW.write_prefetch({(GID["lit"], 6002): []}, build)
    return A.write_archive(files), toc, prefetch


class AddNew(unittest.TestCase):
    """W25-6, W25-8."""

    def document(self):
        document = J._json_round(E.decode(T.lit_file("switch", texture=6001), "lit", "switch"))
        document["record"]["eType"]["value"] = 2
        document["record"]["eType"]["name"] = E.enum_names(S.row("switch", "lit", E._top_table_record(
            "switch", "lit"), "eType").enum).get(2)
        return document

    def test_new_handle_toc_and_prefetch(self):
        cpk, toc, prefetch = light_archive()
        name = F.name_of("lit", "Lamp_Blue")
        result = E.add_new(cpk, name, self.document(), tocs={"CoreTOC.dat": toc}, prefetch={"Prefetch.dat": prefetch})
        self.assertTrue(A.verify_archive(result.archive).ok)
        data = A.read_archive(result.archive).read_file(name)
        handle = result.handle
        self.assertNotIn(handle, (6001, 6002))
        self.assertEqual(parse_header(result.data, "switch").handle, handle)
        self.assertEqual(data, result.data)
        self.assertEqual(E.decode(data, "lit", "switch")["record"]["eType"]["value"], 2)
        entries = I.parse_core_toc(result.index_files["CoreTOC.dat"], "switch").entries
        self.assertIn((GID["lit"], handle, "Lamp_Blue"), [(e.group, e.handle, e.name) for e in entries])
        pf = I.parse_prefetch(result.index_files["Prefetch.dat"], "switch")
        parent = next(p for p in pf.parents if (p.group, p.handle) == (GID["lit"], handle))
        self.assertEqual(pf.dependencies(parent), [(GID["tex"], 6001)])
        self.assertEqual(result.depends, [(GID["tex"], 6001)])
        self.assertEqual(result.unresolved, [])

    def test_unresolved_link_reported(self):
        cpk, toc, prefetch = light_archive()
        document = self.document()
        document["record"]["tMatTexProjected.snoTex"] = {"group": "tex", "handle": 6999}
        result = E.add_new(cpk, F.name_of("lit", "Lamp_Lost"), document, tocs={"CoreTOC.dat": toc},
                           prefetch={"Prefetch.dat": prefetch})
        self.assertEqual(result.depends, [])
        self.assertEqual([link.target for link in result.unresolved], [("sno", GID["tex"], 6999)])
        self.assertNotIn("Prefetch.dat", result.index_files)  # no dependency: Prefetch unchanged (W16-12)

    def test_without_prefetch(self):
        cpk, toc, _ = light_archive()
        result = E.add_new(cpk, F.name_of("lit", "Lamp_2"), self.document(), tocs=toc)
        self.assertEqual(result.depends, [(GID["tex"], 6001)])
        self.assertEqual(set(result.index_files), {"CoreTOC.dat"})

    def test_refused(self):
        cpk, toc, _ = light_archive()
        with self.assertRaises(Exception):
            E.add_new(cpk, F.name_of("lit", "Lamp"), self.document(), tocs=toc)  # the name is taken
        with self.assertRaises(J.Refused):
            E.add_new(cpk, F.name_of("lit", "Lamp_3"), {"record": {"x": 1}}, tocs=toc)


def quiet(argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli.main(argv)
    return code, out.getvalue(), err.getvalue()


class Command(unittest.TestCase):
    """W25-7."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="d3sno_create_"))
        self.addCleanup(__import__("shutil").rmtree, self.root, True)

    def test_out(self):
        data = T.vib_file("switch")
        source = self.root / "vib.json"
        source.write_text(json.dumps(E.decode(data, "vib", "switch")), encoding="utf-8")
        code, out, _ = quiet(["create", "vib", str(source), "--out", str(self.root / "new.vib")])
        self.assertEqual(code, 0)
        self.assertEqual((self.root / "new.vib").read_bytes(), data)

    def test_refusal(self):
        source = self.root / "bad.json"
        source.write_text(json.dumps({"build": "switch", "record": {"noSuchLeaf": 1}}), encoding="utf-8")
        code, _, err = quiet(["create", "vib", str(source), "--out", str(self.root / "x.vib")])
        self.assertEqual(code, 2)
        self.assertIn("record.noSuchLeaf: unknown_name", err)
        self.assertFalse((self.root / "x.vib").exists())

    def test_archive(self):
        cpk, toc, prefetch = light_archive()
        for name, data in (("Act1.cpk", cpk), ("CoreTOC.dat", toc), ("Prefetch.dat", prefetch)):
            (self.root / name).write_bytes(data)
        source = self.root / "lamp.json"
        source.write_text(json.dumps(AddNew().document()), encoding="utf-8")
        out = self.root / "out"
        code, text, err = quiet(["create", "lit", str(source), "--archive", str(self.root / "Act1.cpk"),
                                 "--name", F.name_of("lit", "Lamp_Blue"), "--toc", str(self.root / "CoreTOC.dat"),
                                 "--prefetch", str(self.root / "Prefetch.dat"), "--out-dir", str(out), "--jobs", "1"])
        self.assertEqual(code, 0, err)
        want = E.add_new(cpk, F.name_of("lit", "Lamp_Blue"), AddNew().document(), tocs={"CoreTOC.dat": toc},
                         prefetch={"Prefetch.dat": prefetch})
        self.assertEqual((out / "Act1.cpk").read_bytes(), bytes(want.archive))
        for name, data in want.index_files.items():
            self.assertEqual((out / name).read_bytes(), data)
        self.assertIn(f"handle {want.handle}", text)
        self.assertIn("dependencies (1)", text)
        self.assertIn("tex:6001", text)

    def test_unresolved_printed(self):
        cpk, toc, _ = light_archive()
        for name, data in (("Act1.cpk", cpk), ("CoreTOC.dat", toc)):
            (self.root / name).write_bytes(data)
        document = AddNew().document()
        document["record"]["tMatTexProjected.snoTex"] = {"group": "tex", "handle": 6999}
        source = self.root / "lost.json"
        source.write_text(json.dumps(document), encoding="utf-8")
        code, text, err = quiet(["create", "lit", str(source), "--archive", str(self.root / "Act1.cpk"), "--name",
                                 F.name_of("lit", "Lamp_Lost"), "--toc", str(self.root / "CoreTOC.dat"), "--out-dir",
                                 str(self.root / "out"), "--jobs", "1"])
        self.assertEqual(code, 0, err)
        self.assertIn("unresolved: record.tMatTexProjected.snoTex -> tex:6999", text)
        self.assertIn("dependencies (0) (not written: no --prefetch): none", text)

    def test_git_tree_refused(self):
        source = self.root / "vib.json"
        source.write_text(json.dumps(E.decode(T.vib_file("switch"), "vib", "switch")), encoding="utf-8")
        with mock.patch.object(X, "in_git_tree", return_value=self.root):
            code, _, err = quiet(["create", "vib", str(source), "--out", str(self.root / "a.vib")])
            self.assertEqual(code, 2)
            self.assertIn("git working tree", err)
            self.assertEqual(quiet(["create", "vib", str(source), "--out", str(self.root / "a.vib"), "--force"])[0], 0)

    def test_options(self):
        source = self.root / "vib.json"
        source.write_text("{}", encoding="utf-8")
        self.assertEqual(quiet(["create", "vib", str(source)])[0], 2)
        self.assertEqual(quiet(["create", "vib", str(source), "--archive", "x.cpk", "--out-dir", "d"])[0], 2)


class Version(unittest.TestCase):
    def test_version(self):
        self.assertGreaterEqual(tuple(int(p) for p in d3sno.__version__.split(".")), (0, 25, 0))
        self.assertGreaterEqual(d3sno.STAGE, 25)
        self.assertEqual(len(E.types()), 59)
        for build in BUILDS:
            self.assertTrue(set(E.GROUPS) <= set(E.types(build)))
        self.assertTrue({"prt", "mrk"} <= E.ALWAYS_SHARED["switch"] == E.ALWAYS_SHARED["x360"])
        self.assertFalse({"prt", "mrk", "app", "ans", "con", "ui"} & E.ALWAYS_SHARED["ps3"])
        self.assertTrue({"acr", "glo", "lit"} <= E.ALWAYS_SHARED["ps3"])


if __name__ == "__main__":
    unittest.main()
