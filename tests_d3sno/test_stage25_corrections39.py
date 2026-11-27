"""Corrections 39 (``62_corrections39.md``): AJ1, a NaN keeps its bits (``"nan:XXXXXXXX"``); AJ2, the PS3 material
texture entry (``f62_layouts.tsv``). Every file is synthetic."""

import _libpath  # noqa: F401  (python -I: the library folder on sys.path)
import copy
import struct
import unittest

import test_stage23 as T
import test_stage24 as S
import test_stage25_corrections38 as C
import d3sno
from d3sno import effects as E
from d3sno import export as X
from d3sno import json_import as J
from d3sno import links as K
from d3sno import sno_layout as L
from d3sno.binary import build_order
from d3sno.datafiles import DATA

BUILDS = C.BUILDS
TEX = 44  # the group of ``Texture``


def lit_bits(build, bits):
    """A light whose first plain float leaf holds the float32 ``bits``."""
    tree = T.blank(build, "lit", 7101)
    struct.pack_into(build_order(build) + "I", tree.top.raw, C.float_leaf(build).offset, bits)
    return L.write_tree(tree, derive=True)


def stored_bits(data, build):
    raw = L.parse_tree(data, "lit", build).top.raw
    return struct.unpack_from(build_order(build) + "I", raw, C.float_leaf(build).offset)[0]


def material(build, entries):
    """A material with one texture entry per ``(slot type, texture handle)``."""
    tree = T.blank(build, "mat", 7102)
    slot = next(s for s in L.pair_slots(build, "mat", tree.top.name) if s.field.element == "MaterialTextureEntry")
    elements = []
    for kind, texture in entries:
        element = L.new_record(build, "mat", "MaterialTextureEntry")
        order = build_order(build)
        struct.pack_into(order + "i", element.raw, S.row(build, "mat", "MaterialTextureEntry", "eShaderTex").offset,
                         kind)
        struct.pack_into(order + "i", element.raw,
                         S.row(build, "mat", "MaterialTextureEntry", "tMatTex.snoTex").offset, texture)
        elements.append(element)
    tree.top.pairs[slot.offset] = L.PairValue(L.Block(elements=elements))
    return L.write_tree(tree, derive=True)


class NanBits(unittest.TestCase):
    """AJ1."""

    def test_export_and_create(self):
        for build in BUILDS:
            with self.subTest(build=build):
                leaf = C.float_leaf(build)
                data = lit_bits(build, 0x7FFFFFFF)
                document = C.doc(data, "lit", build)
                self.assertEqual(document["record"][leaf.name], "nan:7FFFFFFF")
                again = E.create(document, "lit", build)
                self.assertEqual(again, data)
                self.assertEqual(stored_bits(again, build), 0x7FFFFFFF)
                order = ">" if build_order(build) == ">" else "<"
                raw = L.parse_tree(again, "lit", build).top.raw
                self.assertEqual(bytes(raw[leaf.offset:leaf.offset + 4]), struct.pack(order + "I", 0x7FFFFFFF))
                quiet = lit_bits(build, 0x7FC00000)
                self.assertEqual(C.doc(quiet, "lit", build)["record"][leaf.name], "nan")
                self.assertEqual(E.create(C.doc(quiet, "lit", build), "lit", build), quiet)

    def test_both_byte_orders(self):
        self.assertEqual({build_order(b) for b in BUILDS}, {"<", ">"})

    def test_negative_nan_and_refusals(self):
        for build in BUILDS:
            leaf = C.float_leaf(build)
            document = C.doc(lit_bits(build, 0), "lit", build)
            document["record"][leaf.name] = "nan:FFC00000"
            self.assertEqual(stored_bits(E.create(document, "lit", build), build), 0xFFC00000)
            self.assertEqual(C.doc(E.create(document, "lit", build), "lit", build)["record"][leaf.name],
                             "nan:FFC00000")
            for bad in ("nan:7F800000", "nan:12", "nan:XYZ", "nan:7F80000G", "nan:", "nan:7FFFFFFF0"):
                with self.subTest(build=build, bad=bad):
                    wrong = copy.deepcopy(document)
                    wrong["record"][leaf.name] = bad
                    with self.assertRaises(J.Refused) as caught:
                        E.create(wrong, "lit", build)
                    self.assertEqual(caught.exception.path[-1], leaf.name)
                    self.assertEqual(caught.exception.reason, "value_out_of_range")

    def test_edit_onto_a_template(self):
        for build in BUILDS:
            with self.subTest(build=build):
                leaf = C.float_leaf(build)
                data = lit_bits(build, 0x7FC00000)
                document = C.doc(data, "lit", build)
                document["record"][leaf.name] = "nan:7FFFFFFF"
                edited = E.encode(data, "lit", build, document)
                self.assertEqual(stored_bits(edited, build), 0x7FFFFFFF)
                back = C.doc(edited, "lit", build)
                back["record"][leaf.name] = "nan"
                self.assertEqual(E.encode(edited, "lit", build, back), data)

    def test_texts(self):
        self.assertEqual(X.plain(struct.unpack("<f", struct.pack("<I", 0x7FFFFFFF))[0]), "nan:7FFFFFFF")
        self.assertEqual(X.plain(struct.unpack("<f", struct.pack("<I", 0xFFC00000))[0]), "nan:FFC00000")
        self.assertEqual(X.plain(struct.unpack("<f", struct.pack("<I", 0x7F800001))[0]), "nan:7F800001")
        self.assertEqual(X.plain(float("nan")), "nan")
        self.assertEqual(X.plain(float("inf")), "inf")
        self.assertEqual(X.plain(float("-inf")), "-inf")
        self.assertEqual(X.float_text_bits("nan:7fffffff"), 0x7FFFFFFF)
        self.assertEqual(X.float_text_bits("-inf"), 0xFF800000)
        for bad in ("nan:7F800000", "nan:FF800000", "nan:00000001", "nan:3FC00000"):
            with self.subTest(bad=bad):
                with self.assertRaisesRegex(ValueError, "not a NaN"):
                    X.float_text_bits(bad)
        with self.assertRaises(J.Refused) as caught:
            J._pack("f", "nan:7F800000", ("x",), "<")
        self.assertIn("not a NaN", caught.exception.message)
        self.assertFalse(J._same_json("nan", "nan:7FFFFFFF"))
        self.assertFalse(J._close("nan", "nan:7FFFFFFF"))


class TextureEntry(unittest.TestCase):
    """AJ2."""

    def test_table(self):
        self.assertEqual(E.LAYOUT_TABLE, "f62_layouts.tsv")
        self.assertTrue((DATA / "f61_layouts.tsv").is_file())
        rows = {r.name: r for r in E.leaf_table("ps3", "mat") if r.record == "MaterialTextureEntry"}
        self.assertEqual((rows["eShaderTex"].offset, rows["eShaderTex"].size, rows["eShaderTex"].enum),
                         (0, 4, "ShaderTex"))
        self.assertEqual((rows["tMatTex.snoTex"].offset, rows["tMatTex.snoTex"].size, rows["tMatTex.snoTex"].group),
                         (20, 4, TEX))
        self.assertEqual(rows["eShaderTex"].status, "type information (matches the material decoder)")

    def test_links_json_and_create(self):
        data = material("ps3", [(1, 501), (2, 502)])
        found = K.links_of(data, "mat", "ps3")
        self.assertEqual([(l.kind, l.target) for l in found], [("field", ("sno", TEX, 501)), ("field", ("sno", TEX, 502))])
        document = C.doc(data, "mat", "ps3")
        entries = document["record"]["tUberMaterial.serMatTexList"]
        self.assertEqual([e["tMatTex.snoTex"] for e in entries],
                         [{"group": "tex", "handle": 501}, {"group": "tex", "handle": 502}])
        self.assertEqual([e["eShaderTex"]["value"] for e in entries], [1, 2])
        for entry in entries:
            self.assertEqual(set(entry["unnamed"]), {"0x000C", "0x0010"})  # +12 to +19 stay unnamed
        self.assertEqual(E.create(document, "mat", "ps3"), data)

    def test_the_0_25_1_form(self):
        data = material("ps3", [(1, 501), (2, 502)])
        document = C.doc(data, "mat", "ps3")
        for entry in document["record"]["tUberMaterial.serMatTexList"]:
            entry["unnamed"]["0x0000"] = entry.pop("eShaderTex")["value"]
            entry["unnamed"]["0x0014"] = entry.pop("tMatTex.snoTex")["handle"]
        self.assertEqual(E.create(document, "mat", "ps3"), data)
        self.assertEqual(E.encode(data, "mat", "ps3", document), data)
        # only the two leaves f62 names newly are read from unnamed words
        self.assertEqual(E._newly_named(), {("ps3", "mat", "MaterialTextureEntry", "eShaderTex", 0),
                                            ("ps3", "mat", "MaterialTextureEntry", "tMatTex.snoTex", 20)})
        both = C.doc(data, "mat", "ps3")
        both["record"]["tUberMaterial.serMatTexList"][0]["unnamed"]["0x0014"] = 501  # the name is given too
        with self.assertRaises(J.Refused) as caught:
            E.create(both, "mat", "ps3")
        self.assertEqual(caught.exception.reason, "conflicting_values")

    def test_old_form_edit(self):
        data = material("ps3", [(1, 501)])
        document = C.doc(data, "mat", "ps3")
        entry = document["record"]["tUberMaterial.serMatTexList"][0]
        entry.pop("tMatTex.snoTex")
        entry["unnamed"]["0x0014"] = 777
        edited = E.encode(data, "mat", "ps3", document)
        self.assertEqual([l.target for l in K.links_of(edited, "mat", "ps3")], [("sno", TEX, 777)])

    def test_no_build_skipped(self):
        for build in BUILDS:
            with self.subTest(build=build):
                found = K.links_of(S.material(build, 4006), "mat", build)
                self.assertEqual([(l.kind, l.target) for l in found], [("field", ("sno", TEX, 4006))])


class Version(unittest.TestCase):
    def test_version(self):
        self.assertGreaterEqual(tuple(int(p) for p in d3sno.__version__.split(".")), (0, 25, 2))
        self.assertEqual(d3sno.STAGE, 25)


if __name__ == "__main__":
    unittest.main()
