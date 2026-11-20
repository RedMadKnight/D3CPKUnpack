"""Corrections 22 (``36``): V22-2 removing the last triangle of a skinning record, V22-4 the
ASCII locale of the x360 pack rule. Synthetic data only."""

import unittest

import d3sno
from d3sno import FormatError
from d3sno import appearance as P
from d3sno import index_write as W
from d3sno import mesh_import as I
from test_stage12 import BONES, ps3
from test_stage16_cli import X360_GROUPS
from test_stage16_mesh import derived, read, switch_skinned3


class Marker(unittest.TestCase):
    def test_version(self):
        self.assertGreaterEqual(tuple(int(x) for x in d3sno.__version__.split(".")), (0, 16, 2))


class AsciiLocale(unittest.TestCase):
    NAMES = list(X360_GROUPS)

    def test_non_ascii_locale_refused(self):
        for name in ("énUS_Act1.cpk", "enÜS_Common.cpk", "деDE_CacheAct2.cpk",
                     "enUİ_Act3.cpk"):
            with self.assertRaises(FormatError, msg=name):
                W.x360_pack(name, self.NAMES)

    def test_ascii_locales_still_accepted(self):
        self.assertEqual(W.x360_pack("enUS_Act1.cpk", self.NAMES), "Act1TOC.dat")
        self.assertEqual(W.x360_pack("zhTW_CacheCommon.cpk", self.NAMES), "CommonTOC.dat")
        self.assertEqual(W.x360_pack("énUS_Act1.cpk", self.NAMES, pack="Act1"), "Act1TOC.dat")


def without_triangles(app):
    """The exported glTF with every triangle of the first primitive removed (V21-4 mark)."""
    doc = P.to_gltf(app)
    primitive = doc["meshes"][0]["primitives"][0]
    del primitive["indices"]
    primitive["extras"]["d3sno"]["no_indices"] = True
    return read(doc)


class LastTriangle(unittest.TestCase):
    def test_switch_and_x360_only_record_refused(self):
        for order, build in (("<", None), (">", "x360")):
            app = P.parse_appearance(derived(switch_skinned3(order), build), build)
            with self.assertRaisesRegex(FormatError, "V21-3"):
                I.import_mesh(app, without_triangles(app))

    def test_ps3_skinned_draw_dropped(self):
        app = P.parse_appearance(ps3(skinned=True, bones=BONES), "ps3")
        out = I.import_mesh(app, without_triangles(app))
        self.assertEqual(P.parse_appearance(out, "ps3").sub_objects[0].draws, [])


if __name__ == "__main__":
    unittest.main()
