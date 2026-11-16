"""Stage 11.2: ``24_corrections13.md`` (V12-1 the ``.app`` length guard, V12-2 the sound name table).

There are no game files here: the ``.app`` files are the test builders' files laid out by W11,
the sound files are built by hand from S11, V11 and V12.
"""

import struct
import unittest

import builders as B
import d3sno
from d3sno import FormatError
from d3sno import appearance as P
from d3sno import presentation as V
from d3sno import sno_layout as L
from d3sno import write as W
from test_stage11 import SKIN, fsb5

SUB = "tStructure.tGeoSets[0].ptSubObjects[0]."
SKIN0 = SUB + "ptSkinningInfo[0]."
DRAW = "tStructure.tGeoSets[0].serSubObjects[0].serDraws[0]."


def skinned_app():
    """A switch .app with one skinned sub-object: its own packed list is empty (no rigid
    vertices), the skinning record has 1 bone id, 3 packed vertices and 3 indices."""
    data = L.write_tree(L.parse_tree(B.build_app_switch(
        bones=[(0x10, -1, (0.0, 0.0, 0.0))], sub_objects=[{"skin": SKIN, "index_count": 3}]), "app"))
    return P.parse_appearance(data)


def ps3_app():
    return P.parse_appearance(L.write_tree(L.parse_tree(B.build_app_ps3(), "app", "ps3")), "ps3")


def block(app, path):
    value = L.find_pair(W.app_tree(app), path)[2]
    return b"" if value is None or value.block is None else value.block.data


class Marker(unittest.TestCase):
    def test_version(self):
        self.assertTrue(d3sno.__version__.startswith(("0.11.", "0.12.")))
        self.assertGreaterEqual(d3sno.STAGE, 11)


class GuardV12_1(unittest.TestCase):
    def test_tied_set(self):
        for build in ("switch", "x360"):
            tied = W.APP_COUNT_TIED[build]
            self.assertIn(("SkinningInfo", 0x04), tied)
            self.assertNotIn(("SkinningInfo", 0x40), tied)
            self.assertIn(("SubObject", 0xB0), tied)
        self.assertEqual(W.APP_COUNT_TIED["ps3"],
                         frozenset({("DrawP3", 0x08), ("DrawP3", 0x2C), ("DrawP3", 0x6C)}))
        app = skinned_app()
        self.assertTrue(W.is_count_tied(app, SKIN0 + "pBoneIDs"))
        self.assertFalse(W.is_count_tied(app, SKIN0 + "serPackIndexList"))
        self.assertTrue(W.is_count_tied(ps3_app(), DRAW + "serStreamA"))

    def test_bone_id_list_refused(self):
        app = skinned_app()
        old = block(app, SKIN0 + "pBoneIDs")
        self.assertEqual(len(old), 4)
        with self.assertRaisesRegex(FormatError, "V11-9"):
            W.set_app_block(app, SKIN0 + "pBoneIDs", old + struct.pack("<i", 0))
        W.set_app_block(app, SKIN0 + "pBoneIDs", struct.pack("<i", 0))  # same length
        P.parse_appearance(W.write_app(app))

    def test_bone_id_list_lengthened_does_not_read(self):
        app = skinned_app()
        old = block(app, SKIN0 + "pBoneIDs")
        W.set_app_block(app, SKIN0 + "pBoneIDs", old + old, allow_count_change=True)
        with self.assertRaisesRegex(FormatError, "does not read.*V11-9"):
            W.write_app(app)

    def test_ps3_stream_a_refused(self):
        app = ps3_app()
        old = block(app, DRAW + "serStreamA")
        with self.assertRaisesRegex(FormatError, "V11-9"):
            W.set_app_block(app, DRAW + "serStreamA", old + bytes(16))
        W.set_app_block(app, DRAW + "serStreamA", bytes(len(old)))
        P.parse_appearance(W.write_app(app), "ps3")

    def test_skinning_index_list_free_but_mesh_checked(self):
        app = skinned_app()
        old = block(app, SKIN0 + "serPackIndexList")
        W.set_app_block(app, SKIN0 + "serPackIndexList", old + bytes(3))  # no longer guarded
        out = W.write_app(app)  # V12-1: lengthened by 3 bytes, the mesh still reads
        P.parse_appearance(out)
        self.assertEqual(len(L.find_pair(L.parse_tree(out, "app", "switch"),
                                         SKIN0 + "serPackIndexList")[2].block.data), len(old) + 3)


class EmptyPackedList(unittest.TestCase):
    def test_filled_with_whole_vertices(self):
        app = skinned_app()
        self.assertEqual(block(app, SUB + "serPackVertList"), b"")
        W.set_packed_vertices(app, 0, bytes(2 * 28))
        out = W.write_app(app)  # the sub-object has no rigid vertices: the mesh reads
        P.parse_appearance(out)
        self.assertTrue(L.is_canonical(out, "app", "switch"))
        self.assertEqual(len(L.find_pair(L.parse_tree(out, "app", "switch"),
                                         SUB + "serPackVertList")[2].block.data), 56)

    def test_not_a_whole_number_of_vertices(self):
        app = skinned_app()
        with self.assertRaisesRegex(FormatError, "V11-9"):
            W.set_packed_vertices(app, 0, bytes(30))

    def test_non_empty_list_still_refused(self):
        app = P.parse_appearance(L.write_tree(L.parse_tree(B.build_app_switch(), "app")))
        old = block(app, W.app_pair_path(app, 0))
        self.assertTrue(old)
        with self.assertRaisesRegex(FormatError, "V11-9"):
            W.set_packed_vertices(app, 0, old + bytes(28))

    def test_filled_list_whose_mesh_does_not_read(self):
        # the empty list's sub-object gets a rigid vertex count of 3 in place (record 0xA0): one
        # whole vertex passes the guard, the mesh check refuses 28 bytes for 3 vertices
        app = skinned_app()
        offset = L.parse_tree(bytes(app.source), "app", "switch").top.pairs[0x64].stored[0]
        struct.pack_into("<i", app.source, 0x10 + offset + 0xA0, 3)
        W.set_packed_vertices(app, 0, bytes(28))
        with self.assertRaisesRegex(FormatError, "does not read.*V11-9"):
            W.write_app(app)


class NameTableV12_2(unittest.TestCase):
    def test_range_is_not_a_limit(self):
        self.assertEqual(V.NAME_TABLE_SIZES, (16, 104))
        for name in (b"B-1", b"A" * 99, b"Z" * 200):
            fsb = V.parse_fsb5(fsb5(15, bytes(64), name=name))
            self.assertEqual(fsb.name_table, V.fsb5_name_table(name.decode(), len(fsb.headers_raw)))
            self.assertEqual(V.fsb5_problems(fsb), [], name)
        self.assertGreater(len(V.fsb5_name_table("Z" * 200, 16)), 104)

    def test_short_table_by_the_rule(self):
        # 60 + 16 + 4 + 15 + 1 = 96: the rule adds no zero bytes, a 20-byte table
        self.assertEqual(len(V.fsb5_name_table("x" * 15, 16)), 20)
        # 60 + 52 + 12 = 124 -> 4 more: a 16-byte table; a shorter one is also by the rule
        self.assertEqual(len(V.fsb5_name_table("abcdefg", 52)), 16)
        self.assertEqual(len(V.fsb5_name_table("abcdefghi", 54)), 14)

    def test_table_not_by_the_rule(self):
        fsb = V.parse_fsb5(fsb5(15, bytes(64), table_size=52))  # 32 bytes more than the rule
        problems = V.fsb5_problems(fsb)
        self.assertIn("name table of 52 bytes, not the padding rule's 20 (V12-2)", problems)


if __name__ == "__main__":
    unittest.main()
