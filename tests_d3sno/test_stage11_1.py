"""Stage 11.1: ``23_corrections12.md`` (V11-1 to V11-9, the checks of round 11).

There are no game files here. Sound files are built by hand from the rules of S11 and V11; the
``.app`` files are the test builders' files laid out by W11.
"""

import csv
import struct
import unittest

import builders as B
import d3sno
from d3sno import FormatError
from d3sno import appearance as P
from d3sno import presentation as V
from d3sno import sno_layout as L
from d3sno import write as W
from test_stage11 import fsb5

PATH = "SoundBank\\Bank\\0001.fsb"


class Marker(unittest.TestCase):
    def test_version(self):
        self.assertGreaterEqual(tuple(int(x) for x in d3sno.__version__.split(".")), (0, 11))
        self.assertGreaterEqual(d3sno.STAGE, 11)


# --- V11-1, V11-2, V11-5, V11-8: the layout table ---------------------------------------------

class LayoutTable(unittest.TestCase):
    def test_v11_1_types_of_the_table(self):
        # the table lists 59 types per build, ps3 .dfn included (V11-1: no ps3 file has it)
        for build in ("switch", "x360", "ps3"):
            self.assertEqual(len(L.covered_types(build)), 59)
        self.assertIn("dfn", L.covered_types("ps3"))

    def test_v11_2_four_types_not_in_the_table(self):
        for build in ("switch", "x360", "ps3"):
            for ext in ("stl", "pow", "scr", "gam"):
                self.assertNotIn(ext, L.covered_types(build))
                with self.assertRaises((FormatError, KeyError)):
                    L.top_record(build, ext)

    def test_v11_5_no_code_rows(self):
        with open(d3sno.__path__[0] + "/data/" + L.TABLE, encoding="utf-8", newline="") as handle:
            kinds = {row["kind"] for row in csv.DictReader(handle, delimiter="\t")}
        self.assertNotIn("code", kinds)

    def test_v11_8_ps3_app_walk_is_not_offset_order(self):
        slots = L.pair_slots("ps3", "app", "AppearanceDefinition")
        offsets = [s.offset for s in slots]
        self.assertEqual(offsets[0], 0x180)  # serAppearanceMaterials comes first
        self.assertNotEqual(offsets, sorted(offsets))


# --- V11-3, V11-6, V11-7: sound ------------------------------------------------------------------

class ModeSixteen(unittest.TestCase):
    def test_v11_3_padding_to_32(self):
        # 200 samples mono: one frame of 140 bytes, 20 zero bytes to 160 (padding to 16 gives 144)
        good = V.parse_fsb5(fsb5(16, b"\x01" * 140 + bytes(20), samples=200))
        self.assertEqual(good.frame_bytes(), 140)
        self.assertEqual(len(good.packets()), 1)
        self.assertEqual(good.padding(), bytes(20))
        self.assertEqual(V.fsb5_problems(good, PATH), [])
        sixteen = V.parse_fsb5(fsb5(16, b"\x01" * 140 + bytes(4), samples=200))
        self.assertTrue(any("160 expected" in p for p in V.fsb5_problems(sixteen)))

    def test_v11_3_padding_sizes(self):
        # 0 to 28 bytes, a multiple of 4, over 1 to 8 frames mono and stereo
        for frames in range(1, 9):
            for stereo in (False, True):
                size = frames * 140 * (2 if stereo else 1)
                pad = -size % 32
                self.assertIn(pad, range(0, 29, 4))
                fsb = V.parse_fsb5(fsb5(16, bytes(size + pad), samples=frames * 256, stereo=stereo))
                self.assertEqual(V.fsb5_problems(fsb, PATH), [], (frames, stereo))


class NameTable(unittest.TestCase):
    def test_v11_6_data_start_and_area(self):
        fsb = V.parse_fsb5(fsb5(15, bytes(64)))
        self.assertEqual((60 + len(fsb.headers_raw) + len(fsb.name_table)) % 32, 0)
        self.assertEqual(V.fsb5_problems(fsb, PATH), [])
        moved = V.parse_fsb5(fsb5(15, bytes(64), table_size=24))
        self.assertTrue(any("not a multiple of 32 (V11-6)" in p and "start" in p
                            for p in V.fsb5_problems(moved)))
        short_area = V.parse_fsb5(fsb5(15, bytes(40)))
        self.assertTrue(any("data area of 40" in p for p in V.fsb5_problems(short_area)))
        trailing = V.parse_fsb5(fsb5(15, bytes(64)) + bytes(32))
        self.assertEqual(len(trailing.trailing), 32)
        self.assertIn("32 bytes after the sample data (V11-6)", V.fsb5_problems(trailing))
        self.assertEqual(V.write_fsb5(trailing), fsb5(15, bytes(64)))  # the file ends at the area

    def test_v11_6_name_offset(self):
        data = bytearray(fsb5(15, bytes(64)))
        table_at = 60 + 16
        struct.pack_into("<I", data, table_at, 5)
        data[table_at + 4] = 0  # the name moves one byte on
        data[table_at + 5:table_at + 15] = b"Bank-0001\0"
        fsb = V.parse_fsb5(bytes(data))
        self.assertEqual(fsb.name, "Bank-0001")
        self.assertIn("name offset 5 is not 4 (V11-6)", V.fsb5_problems(fsb, PATH))

    def test_v11_6_table_size_range(self):
        long_name = V.parse_fsb5(fsb5(15, bytes(64), name=b"B" * 120))
        # V12-2: 16..104 is the range seen, not a limit; a table by the padding rule is good
        self.assertGreater(len(long_name.name_table), 104)
        self.assertEqual(V.fsb5_problems(long_name), [])

    def test_v11_6_renaming_pads_again(self):
        data = fsb5(15, bytes(64))
        fsb = V.parse_fsb5(data)
        for name in ("Bank-0001", "LongerBankName-0123", "B-1", "x" * 60):
            V.fsb5_set_name(fsb, name)
            out = V.parse_fsb5(V.write_fsb5(fsb))
            self.assertEqual(out.name, name)
            self.assertEqual((60 + len(out.headers_raw) + len(out.name_table)) % 32, 0)
            self.assertEqual(out.name_table[:4], struct.pack("<I", 4))
            self.assertEqual(out.data_area, fsb.data_area)
        V.fsb5_set_name(fsb, "Bank-0001")
        self.assertEqual(V.write_fsb5(fsb), data)
        self.assertEqual(V.fsb5_name_table("Bank-0001", 16), fsb.name_table)

    def test_locale_suffix(self):
        fsb = V.parse_fsb5(fsb5(15, bytes(64)))
        self.assertEqual(V.fsb5_name_for_path(PATH + "|enUS"), "Bank-0001")
        self.assertEqual(V.fsb5_name_for_path(PATH + "@frFR"), "Bank-0001")
        self.assertEqual(V.fsb5_problems(fsb, PATH + "|enUS"), [])
        self.assertTrue(V.fsb5_problems(fsb, "SoundBank\\Other\\0001.fsb|enUS"))
        suffixed = V.parse_fsb5(fsb5(15, bytes(64), name=b"Bank-0001|enUS"))
        self.assertEqual(V.fsb5_problems(suffixed, PATH + "|enUS"), [])


class ChunkTypes(unittest.TestCase):
    def test_v11_7_known_types(self):
        self.assertEqual(set(V.CHUNK_TYPES), {3, 4, 5, 11})
        marker = struct.pack("<I", 7) + b"cue".ljust(256, b"\0")
        chunks = [(3, struct.pack("<II", 0, 255)), (4, marker), (5, bytes(4)), (11, struct.pack("<III", 9, 0, 0))]
        data = fsb5(15, bytes(64), chunks=chunks)
        fsb = V.parse_fsb5(data)
        self.assertEqual(V.fsb5_problems(fsb, PATH), [])
        self.assertEqual(fsb.sample.loop, (0, 255))
        self.assertEqual([(m.position, m.name) for m in fsb.sample.markers()], [(7, "cue")])
        self.assertEqual([c.data for c in fsb.unknown_chunks], [bytes(4)])
        self.assertEqual(fsb.sample.vorbis_setup, (9, [(0, 0)]))
        self.assertEqual(V.write_fsb5(fsb), data)

    def test_v11_7_other_type_reported(self):
        fsb = V.parse_fsb5(fsb5(15, bytes(64), chunks=[(7, bytes(8))]))
        self.assertIn("chunk type 7 (V11-7)", V.fsb5_problems(fsb, PATH))


# --- V11-9: count-tied blocks of .app ------------------------------------------------------------

def switch_app():
    return P.parse_appearance(L.write_tree(L.parse_tree(B.build_app_switch(), "app")))


def ps3_app():
    return P.parse_appearance(L.write_tree(L.parse_tree(B.build_app_ps3(), "app", "ps3")), "ps3")


DRAW = "tStructure.tGeoSets[0].serSubObjects[0].serDraws[0]."


class CountTied(unittest.TestCase):
    def test_tied_blocks(self):
        app = switch_app()
        self.assertTrue(W.is_count_tied(app, W.app_pair_path(app, 0)))
        self.assertFalse(W.is_count_tied(app, "serAppearanceSubObjectsIndices"))
        ps3 = ps3_app()
        self.assertTrue(W.is_count_tied(ps3, DRAW + "serDescriptor"))
        self.assertTrue(W.is_count_tied(ps3, DRAW + "serStreamC"))
        self.assertFalse(W.is_count_tied(ps3, DRAW + "serVertexFormatA"))

    def test_length_change_refused(self):
        app = switch_app()
        old = L.find_pair(W.app_tree(app), W.app_pair_path(app, 0))[2].block.data
        with self.assertRaisesRegex(FormatError, "V11-9"):
            W.set_packed_vertices(app, 0, old + old[:28])
        with self.assertRaisesRegex(FormatError, "V11-9"):
            W.set_packed_vertices(app, 0, old[:-28])
        self.assertEqual(app.block_edits, {})
        ps3 = ps3_app()
        for field, size in (("serDescriptor", 32), ("serStreamC", 80)):
            with self.assertRaisesRegex(FormatError, "V11-9"):
                W.set_app_block(ps3, DRAW + field, bytes(size))

    def test_same_length_allowed(self):
        app = switch_app()
        path = W.app_pair_path(app, 0)
        old = L.find_pair(W.app_tree(app), path)[2].block.data
        new = bytes(len(old))
        W.set_packed_vertices(app, 0, new)
        out = W.write_app(app)
        self.assertEqual(L.find_pair(L.parse_tree(out, "app", "switch"), path)[2].block.data, new)
        self.assertTrue(L.is_canonical(out, "app", "switch"))
        P.parse_appearance(out)  # the mesh still reads

    def test_untied_block_may_change_length(self):
        app = switch_app()
        W.set_app_block(app, "serAppearanceSubObjectsIndices", bytes(16))  # no count: accepted
        self.assertIn("serAppearanceSubObjectsIndices", app.block_edits)

    def test_mesh_check_after_an_allowed_length_change(self):
        app = switch_app()
        old = L.find_pair(W.app_tree(app), W.app_pair_path(app, 0))[2].block.data
        W.set_packed_vertices(app, 0, old + old[:28], allow_count_change=True)
        with self.assertRaisesRegex(FormatError, "does not read.*V11-9"):
            W.write_app(app)
        out = W.write_app(app, check_mesh=False)  # structure only: pairs correct, canonical
        self.assertTrue(L.is_canonical(out, "app", "switch"))
        self.assertEqual(len(L.find_pair(L.parse_tree(out, "app", "switch"),
                                         W.app_pair_path(app, 0))[2].block.data), len(old) + 28)


if __name__ == "__main__":
    unittest.main()
