"""Stage 9: the corrections after stage 8 (``19_corrections8.md``).

There are no game files here: every file is synthetic, built from the layouts.
"""

import struct
import unittest
import zlib

import builders as B
import d3sno
from d3sno import appearance as P
from d3sno import archive as C
from d3sno import balance as G
from d3sno import write as WR


def name_pair(data, build, row_at):
    return struct.unpack_from(B.order_of(build) + "II", data, row_at)


# --- V8-X7: empty name pairs -------------------------------------------------------------------

class EmptyNames(unittest.TestCase):
    def table_at(self, data):
        return struct.unpack_from(">I", data, 360)[0] + 0x10  # type 29 table pair (x360)

    def test_type_29_every_row_empty(self):
        self.assertEqual([(s.offset, s.kind) for s in G.gam_pair_slots("x360", 29)], [(0, "name")])
        data = B.build_gam_layout("x360", 29, 360, 40, [{}, {}, {}], {0: "name"})
        parsed = G.parse_balance(data, "x360")
        self.assertEqual([r.name for r in parsed.rows], [None] * 3)
        written = WR.write(parsed)
        self.assertEqual(written, data)
        table = self.table_at(written)
        for row in range(3):
            self.assertEqual(name_pair(written, "x360", table + row * 40), (0, 0))
        self.assertEqual(len(written), table + 3 * 40)  # empty names add nothing

    def test_type_49_every_row_empty(self):
        self.assertEqual([(s.offset, s.kind) for s in G.gam_pair_slots("x360", 49)], [(0, "name")])
        data = B.build_gam_layout("x360", 49, 616, 48, [{}] * 21, {0: "name"})
        self.assertEqual(WR.write(G.parse_balance(data, "x360")), data)

    def test_empty_name_does_not_align_the_cursor(self):
        # "Ab\0" leaves the cursor 3 past an 8-aligned position; an empty name must not pad it.
        data = B.build_gam_layout("x360", 29, 360, 40, [{"name": "Ab"}, {}], {0: "name"})
        written = WR.write(G.parse_balance(data, "x360"))
        self.assertEqual(written, data)
        table = self.table_at(written)
        offset, size = name_pair(written, "x360", table)
        self.assertEqual(len(written), offset + 0x10 + size)
        self.assertEqual(len(written) % 4, 3)
        self.assertEqual(name_pair(written, "x360", table + 40), (0, 0))

    def test_empty_code_pair_still_uses_the_cursor(self):
        offsets = [s.offset for s in G.gam_pair_slots("x360", 33) if s.kind == "code"]
        rows = [{"name": "Ab"}]
        data = B.build_gam_layout("x360", 33, 0x188, 224, rows, {0: "name", **{o: "code" for o in offsets}})
        written = WR.write(G.parse_balance(data, "x360"))
        self.assertEqual(written, data)
        table = struct.unpack_from(">I", written, 0x188)[0] + 0x10
        position, size = name_pair(written, "x360", table + offsets[0])
        self.assertEqual(size, 0)
        self.assertEqual(position + 0x10, len(written))  # aligned cursor, not zero
        self.assertEqual(len(written) % 4, 0)

    def test_empty_name_on_every_build(self):
        # Switch and ps3 files have no empty name pair; the writer uses the same rule there.
        for build, slot, size in (("switch", 0xB8, 32), ("ps3", 0x2A8, 24)):
            self.assertEqual([(s.offset, s.kind) for s in G.gam_pair_slots(build, 11)], [(0, "name")])
            data = B.build_gam_layout(build, 11, slot, size, [{"name": "Abc"}, {}], {0: "name"})
            written = WR.write(G.parse_balance(data, build))
            self.assertEqual(written, data, build)
            table = struct.unpack_from(B.order_of(build) + "I", written, slot)[0] + 0x10
            self.assertEqual(name_pair(written, build, table + size), (0, 0))


# --- V8-X12: archives ending inside the last slot ----------------------------------------------

def cut_last_slot(data: bytes) -> bytes:
    h = C.read_header(data)
    cut = bytearray(data.rstrip(b"\0"))
    struct.pack_into(h.order + "I", cut, 0x38, zlib.crc32(cut[h.start:]) & 0xFFFFFFFF)
    return bytes(cut)


class ArchiveEnd(unittest.TestCase):
    FILES = {"a.bin": bytes(range(1, 256)) * 40}

    def test_raw_archive_ends_inside_last_slot_on_every_build(self):
        for name, data in (
            ("ps3", C.write_archive(self.FILES, flags=0, version=6)),
            ("switch", C.write_archive(self.FILES, flags=0)),
            ("x360", C.write_archive(self.FILES, flags=0, order=">")),
        ):
            cut = cut_last_slot(data)
            h = C.read_header(cut)
            self.assertEqual(h.build, name)
            self.assertNotEqual((h.file_size - h.start) % h.comp_sector, 0)
            report = C.verify_archive(cut)
            self.assertTrue(report.ok, report.format())
            self.assertTrue(any("into the last slot" in n for n in report.checks["header"].notes), name)
            self.assertEqual(C.read_archive(cut).read_file("a.bin"), self.FILES["a.bin"])

    def test_flag_10_ends_inside_last_slot(self):
        cut = cut_last_slot(C.write_archive({"a.bin": bytes(range(256)) * 40}, order=">", flags=10))
        report = C.verify_archive(cut)
        self.assertTrue(report.ok, report.format())


# --- V8-X6: weights across builds --------------------------------------------------------------

class Weights(unittest.TestCase):
    TRIANGLE = [(0, 0, 0), (10, 0, 0), (0, 10, 0)]

    def draw(self, order, weights):
        bones = [(0x10, -1, (0.0, 0.0, 0.0)), (0x11, 0, (1.0, 0.0, 0.0))]
        skin = {"palette": [0, 1], "vertices": self.TRIANGLE, "indices": [0, 1, 2],
                "slots": (0, 1, 0, 0), "weights": weights}
        parsed = P.parse_appearance(B.build_app_switch(
            version=302 if order == ">" else 306, order=order, bones=bones, sub_objects=[{"skin": skin}],
        ), strict=True)
        return parsed.sub_objects[0].draws[0]

    def test_one_unit_apart_both_read(self):
        x360, switch = self.draw(">", (179, 76, 0, 0)), self.draw("<", (178, 77, 0, 0))
        self.assertEqual(x360.joints, switch.joints)
        self.assertEqual(x360.positions, switch.positions)
        self.assertNotEqual(x360.weights, switch.weights)
        for weights in x360.weights:
            self.assertEqual(list(weights), sorted(weights, reverse=True))  # Switch order descends


class Marker(unittest.TestCase):
    def test_stage_marked(self):
        self.assertGreaterEqual(d3sno.STAGE, 9)  # later stages keep every stage-9 rule
        self.assertTrue(d3sno.__version__)


if __name__ == "__main__":
    unittest.main()
