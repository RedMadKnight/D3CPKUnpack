"""Stage 10.1: ``21_corrections10.md`` (V9-T4, V9-G1-4, V9-A5, V9-S4, V9-G3-A, V9-G1-3 and the
writer speed note).

There are no game files here: every file is synthetic, built from the rules.
"""

import _libpath  # noqa: F401  (python -I: the library folder on sys.path)
import random
import unittest
import zlib

import builders as B
import d3sno
from d3sno import appearance as P
from d3sno import archive as C
from d3sno import records as R
from d3sno import x360_archive as XA
from d3sno import x360_fields as XF
from d3sno import x360_texture as XT
from test_stage10 import _switch_with_cloth, tsv, x360_tex


class Marker(unittest.TestCase):
    def test_version(self):
        major, minor, patch = (int(x) for x in d3sno.__version__.split("."))
        self.assertGreaterEqual((major, minor, patch), (0, 10, 1))
        self.assertGreaterEqual(d3sno.STAGE, 10)


# --- V9-T4: packed levels from the fourth on ---------------------------------------------------

class TailLevels(unittest.TestCase):
    def test_longer_side(self):
        self.assertEqual(XT.tail_longer_side(16, 64), 64)
        self.assertEqual(XT.tail_longer_side(16, 16), 16)
        self.assertEqual(XT.tail_longer_side(16, 40), 64)  # rounded up to a power of two
        self.assertEqual(XT.tail_longer_side(48, 16), 64)

    def test_levels_below_3_keep_the_table(self):
        for row in tsv("x360_tex_tail_slots.tsv"):
            p, wide = int(row["packed_index"]), row["first_packed_level_wider_than_tall"] == "yes"
            if p < 3:
                for longer in (16, 32, 64, 128):
                    self.assertEqual(XT.tail_offset(p, wide, longer),
                                     (int(row["x_texels"]), int(row["y_texels"])))

    def test_from_level_3_on_based_on_l(self):
        self.assertEqual(XT.tail_offset(3, False, 64), (0, 32))
        self.assertEqual(XT.tail_offset(4, False, 64), (0, 16))
        self.assertEqual(XT.tail_offset(3, True, 64), (32, 0))
        self.assertEqual(XT.tail_offset(4, True, 32), (8, 0))
        # L = 16 is the old rule (the table)
        self.assertEqual(XT.tail_offset(3, False, 16), (0, 8))
        self.assertEqual(XT.tail_offset(4, True), (4, 0))

    def test_check_texture_32_by_128(self):
        # 32 x 128 ARGB8888, 6 levels: first packed level 1 (16 x 64), L = 64; packed levels 3 and
        # 4 (levels 4 and 5, 2 x 8 and 1 x 4) sit at (0, 32) and (0, 16).
        surfaces, _ = XT.layout(0, 32, 128, 6)
        packed = {s.packed_index: s for s in surfaces if s.packed}
        self.assertEqual(sorted(packed), [0, 1, 2, 3, 4])
        self.assertEqual(packed[3].origin, (0, 32))
        self.assertEqual(packed[4].origin, (0, 16))
        self.assertEqual((packed[3].width, packed[3].height), (2, 8))

    def test_l16_textures_unchanged(self):
        # first packed level 16 x 16: L = 16, the table's offsets
        surfaces, _ = XT.layout(0, 64, 64, 7)
        packed = {s.packed_index: s.origin for s in surfaces if s.packed}
        self.assertEqual(packed[3], (0, 8))
        self.assertEqual(packed[4], (0, 4))

    def test_round_trip_keeps_every_level(self):
        rng = random.Random(4)
        width, height, levels = 32, 128, 6
        elements = {(m, 0): bytes(rng.randrange(1, 256) for _ in range(
            4 * max(width >> m, 1) * max(height >> m, 1))) for m in range(levels)}
        payload = XT.build_payload(0, width, height, levels, elements)
        tex = x360_tex(0, width, height, payload, mip_max=levels - 1)
        for m in range(levels):
            self.assertEqual(XT.standard_elements(tex, m), elements[(m, 0)], m)
        # the 80 bytes of levels 4 and 5 are inside the tail, not at the old (0, 8) / (0, 4)
        tail = next(s for s in XT.tex_layout(tex)[0] if s.packed)
        self.assertNotEqual(XT.read_elements(tex, 4), b"")
        self.assertEqual(len(XT.read_elements(tex, 4)) + len(XT.read_elements(tex, 5)), 80)
        self.assertEqual(tail.level, 1)


# --- V9-G1-4: faces past nFaceCount -----------------------------------------------------------

FAT = [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (0.0, 10.0, 2.0),
       (10.0, 10.0, 0.0), (20.0, 0.0, 5.0), (20.0, 10.0, 0.0)]
TRIS = [(0, 1, 2), (1, 3, 2), (1, 4, 3), (4, 5, 3)]


def cloth_app(extra_entries=(), mutate=None):
    arrays = B.cloth_arrays(FAT, TRIS)
    if mutate:
        mutate(arrays)
    flat = [i for t in TRIS for i in t]
    return P.parse_appearance(_switch_with_cloth(
        FAT, flat, lambda img: B.cloth_record(img, "switch", arrays, extra_entries=extra_entries)))


class ExtraFaces(unittest.TestCase):
    def test_zero_entries_not_checked(self):
        app = cloth_app([(0.0, (0, 0, 0))] * 3)
        self.assertEqual(app.warnings, [])
        cloth = app.sub_objects[0].cloth
        self.assertEqual((cloth.face_count, len(cloth.faces)), (4, 7))
        self.assertEqual(cloth.faces[4].vertices, (0, 0, 0))  # kept

    def test_triangle_with_wrong_rest_area_not_checked(self):
        app = cloth_app([(0.00043, (0, 1, 2))])
        self.assertEqual(app.warnings, [])
        self.assertAlmostEqual(app.sub_objects[0].cloth.faces[4].rest_area, 0.00043, places=6)

    def test_extra_entry_past_vertex_range_not_checked(self):
        self.assertEqual(cloth_app([(1.0, (99, 100, 101))]).warnings, [])

    def test_extra_entries_not_added_to_area(self):
        # flArea is the sum over the counted faces only: an area counting the extra entry fails.
        def add(arrays):
            arrays["area"] = B._f32(arrays["area"] + 5.0)
        app = cloth_app([(5.0, (0, 1, 2))], add)
        self.assertTrue(any("flArea" in w for w in app.warnings))

    def test_counted_faces_still_checked(self):
        def bad(arrays):
            area, vs = arrays["faces"][1]
            arrays["faces"][1] = (area * 2, vs)
        self.assertTrue(any("rest area" in w for w in cloth_app(mutate=bad).warnings))

    def test_written_unchanged(self):
        from d3sno import write as WR
        data = _switch_with_cloth(FAT, [i for t in TRIS for i in t], lambda img: B.cloth_record(
            img, "switch", B.cloth_arrays(FAT, TRIS), extra_entries=[(0.00043, (0, 1, 2))]))
        app = P.parse_appearance(data)
        self.assertEqual(WR.write_app(app), data)


# --- V9-G1-3: PS3 cloth with a draw ------------------------------------------------------------

class Ps3ClothDraw(unittest.TestCase):
    # V10-G1-3: the draw holds the fat positions (its decoded triangle) within one step.
    FAT3 = [(1.0, 2.0, 3.0), (6.0, 2.0, 3.0), (1.0, 7.0, 3.0)]

    def test_draw_exported_not_both(self):
        app = P.parse_appearance(B.build_app_ps3(fat=self.FAT3, cloth=True, cloth_draw=True), strict=True)
        sub = app.sub_objects[0]
        self.assertEqual([d.kind for d in sub.draws], ["ps3"])
        self.assertEqual(sub.fat_draw.kind, "fat")
        self.assertEqual(P.to_obj(app).count("\nf "), 1)  # one triangle, not two copies
        self.assertEqual(len(P.to_gltf(app)["meshes"][0]["primitives"]), 1)

    def test_without_draw_the_fat_vertices_are_exported(self):
        app = P.parse_appearance(B.build_app_ps3(fat=self.FAT3, cloth=True), strict=True)
        self.assertEqual([d.kind for d in app.sub_objects[0].draws], ["fat"])

    def test_draw_counts_must_equal_the_fat_counts(self):
        six = self.FAT3 + [(1.0, 1.0, 0.0), (2.0, 1.0, 0.0), (1.0, 2.0, 0.0)]
        app = P.parse_appearance(B.build_app_ps3(fat=six, cloth=True, cloth_draw=True))
        self.assertTrue(any("V9-G1-3" in w for w in app.warnings), app.warnings)


# --- V9-A5 and the writer speed note ------------------------------------------------------------

def zero_declared_pieces(compress=XA.deflate9):
    """A first piece that leaves a few bytes in its slot, then a piece that does not fit: the cut
    packet in those bytes declares 0 (V9-A5)."""
    rng = random.Random(7)
    big = bytes(rng.randrange(256) for _ in range(40000))
    whole = bytes(rng.randrange(256) for _ in range(16384))
    for size in range(12000, 16384):
        first = whole[:size]
        left = 0x4000 - XA.PACKET_HEADER - len(compress(first))
        if not XA.MIN_PACKET_ROOM <= left <= XA.MIN_PACKET_ROOM + 4:
            continue  # the next packet would not start in this slot with a few bytes of room
        pieces = [XA.Piece("a.bin", 0, first), XA.Piece("b.bin", 0, big)]
        result = XA.write_x360_archive(pieces, read_sector=0x800, compress=compress)
        if any(p.declared == 0 for p in result.packets):
            return pieces, result
    raise AssertionError("no size gives a packet that declares 0")


class ZeroDeclared(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pieces, cls.result = zero_declared_pieces()

    def test_cut_packet_can_declare_nothing(self):
        plans = self.result.packets
        k = next(i for i, p in enumerate(plans) if p.declared == 0)
        zero, after = plans[k], plans[k + 1]
        self.assertEqual((zero.piece, zero.position), (after.piece, after.position))
        self.assertEqual(after.slot, zero.slot + 1)
        self.assertEqual(XA.declared_for_cut(zero.stored), 0)
        archive = C.read_archive(self.result.data)
        self.assertEqual(archive.read_file("b.bin"), self.pieces[1].data)

    def test_source_keyed_by_slot(self):
        archive = C.read_archive(self.result.data)
        found, _ = XA.source_packets(archive, XA.source_pieces(archive))
        self.assertEqual(len(found), len(self.result.packets))
        self.assertTrue(all(len(key) == 3 for key in found))

    def test_rebuild_keeps_both_packets_at_one_position(self):
        other = lambda data: zlib.compress(data, 1)  # noqa: E731  a deflater this one cannot match
        pieces, source = zero_declared_pieces(other)
        k = next(i for i, p in enumerate(source.packets) if p.declared == 0)
        self.assertEqual(source.packets[k].position, source.packets[k + 1].position)
        rebuilt = XA.rebuild_x360(source.data)
        self.assertEqual(rebuilt.data, source.data)
        self.assertGreater(rebuilt.kept_from_source, 1)
        self.assertNotIn("not byte-identical", " ".join(rebuilt.notes))
        # keyed by (piece, position) only, the 0-declaring packet is not found
        archive = C.read_archive(source.data)
        found, encrypted = XA.source_packets(archive, XA.source_pieces(archive))
        by_position = {key[:2]: value for key, value in found.items()}

        def old_lookup(number, position, slot, slot_offset, room, level):
            hit = by_position.get((number, position))
            return None if hit is None or hit[0] != slot_offset else (hit[1], hit[2])
        old = XA.write_x360_archive(pieces, read_sector=0x800, source=old_lookup)
        self.assertNotEqual(old.data, source.data)

    def test_encryption_by_slot(self):
        zero = next(p for p in self.result.packets if p.declared == 0)
        result = XA.write_x360_archive(self.pieces, read_sector=0x800,
                                       encrypted={(zero.piece, zero.position, zero.slot)})
        flagged = [(p.piece, p.position, p.slot) for p in result.packets if p.encrypted]
        self.assertEqual(flagged, [(zero.piece, zero.position, zero.slot)])
        self.assertEqual(XA.rebuild_x360(result.data).data, result.data)
        self.assertEqual(C.read_archive(result.data).read_file("b.bin"), self.pieces[1].data)


class CountingCompressor:
    def __init__(self):
        self.fed = 0

    def __call__(self):
        inner, outer = XA.deflate9_object(), self

        class Wrapped:
            def compress(self, data):
                outer.fed += len(data)
                return inner.compress(data)

            def flush(self):
                return inner.flush()
        return Wrapped()


def text_piece(size, seed=3):
    rng = random.Random(seed)
    words = [bytes(rng.randrange(97, 123) for _ in range(rng.randrange(2, 9))) for _ in range(800)]
    out = bytearray()
    while len(out) < size:
        out += rng.choice(words) + b" "
    return bytes(out[:size])


class WriterSpeed(unittest.TestCase):
    def test_stream_head_is_a_prefix(self):
        data = text_piece(200000)
        whole = XA.deflate9(data[1234:])
        head, complete = XA.stream_head(data, 1234, 5000)
        self.assertFalse(complete)
        self.assertGreater(len(head), 5000)
        self.assertEqual(head, whole[:len(head)])
        small, complete = XA.stream_head(data[:300], 0, 5000)
        self.assertTrue(complete)
        self.assertEqual(small, XA.deflate9(data[:300]))

    def test_same_bytes_as_compressing_the_whole_rest(self):
        pieces = [XA.Piece("t.bin", 0, text_piece(300000)), XA.Piece("u.bin", 0, text_piece(90000, 5))]
        fast = XA.write_x360_archive(pieces, read_sector=0x800)
        full = XA.write_x360_archive(pieces, read_sector=0x800, compress=lambda d: XA.deflate9(d))
        self.assertEqual(fast.data, full.data)
        self.assertGreater(sum(p.piece == 0 for p in fast.packets), 3)
        for flags in (0, 10):
            self.assertEqual(XA.write_x360_archive(pieces, flags=flags).data,
                             XA.write_x360_archive(pieces, flags=flags, compress=lambda d: XA.deflate9(d)).data)

    def test_each_piece_read_about_once(self):
        # Level 9 holds output back, so a cut packet reads about one slot of output's worth of
        # input plus what is held back: about twice the piece in all, whatever its size. The old
        # way (the whole rest for every cut packet) grows with the square of the size.
        piece = text_piece(1000000)
        counter = CountingCompressor()
        result = XA.write_x360_archive([XA.Piece("t.bin", 0, piece)], read_sector=0x800, compressor=counter)
        per_packet = sum(len(piece) - p.position for p in result.packets)  # the old way
        self.assertGreater(per_packet, 10 * len(piece))
        self.assertLess(counter.fed, 3 * len(piece))
        self.assertEqual(C.read_archive(result.data).read_file("t.bin"), piece)

    def test_rebuild_passes_the_compressor(self):
        pieces = [XA.Piece("t.bin", 0, text_piece(120000))]
        data = XA.write_x360_archive(pieces, read_sector=0x800).data
        counter = CountingCompressor()
        self.assertEqual(XA.rebuild_x360(data, compressor=counter).data, data)
        self.assertGreater(counter.fed, 0)


# --- V9-S4: proposals that share an offset --------------------------------------------------------

class SharedOffsets(unittest.TestCase):
    def test_every_proposal_reachable_by_index(self):
        rows = [r for r in XF.x360_struct_names() if r.field_index is not None]
        self.assertEqual(len(rows), 130)
        for row in rows:
            self.assertIs(XF.proposal(row.struct, row.field_index), row)
            self.assertEqual(XF.display_name(row.struct, field_index=row.field_index), row.proposed_name)

    def test_offset_alone_misses_rows(self):
        rows = [r for r in XF.x360_struct_names() if r.x360_offset is not None]
        reachable = {(r.struct, r.x360_offset) for r in rows}
        # Q338: 102 reachable here, 107 in the description.
        self.assertEqual((len(rows), len(reachable)), (130, 102))

    def test_variants_and_bit_fields(self):
        self.assertEqual(len(XF.proposals_at("ScriptEventVariantParam", 4)), 23)
        self.assertEqual(len(XF.proposals_at("ScriptEventVariantParam", 8)), 6)
        bits = XF.proposals_at("DevilsHandRedeemableSet", 28)
        self.assertEqual([r.proposed_name for r in bits], ["Flag0 (bit 0)", "Flag1 (bit 1)"])
        self.assertEqual(XF.display_name("DevilsHandRedeemableSet", 28, name="Flag1 (bit 1)"), "Flag1 (bit 1)")
        self.assertEqual(XF.display_name("DevilsHandRedeemableSet", field_index=6), "Flag1 (bit 1)")
        # offset alone with two rows: no proposal is picked
        self.assertNotIn(XF.display_name("DevilsHandRedeemableSet", 28), ("Flag0 (bit 0)", "Flag1 (bit 1)"))

    def test_single_row_offsets_unchanged(self):
        self.assertEqual(XF.display_name("ItemDurabilityCostScalarsTable", 0), "serItemDurabilityCostScalars")
        self.assertEqual(XF.display_name("ExperienceLevel", 0), "DeltaXP")

    def test_needs_a_key(self):
        with self.assertRaises(ValueError):
            XF.display_name("ExperienceLevel")


# --- V9-G3-A: PS3 ids 12, 13, 31 to 33 -----------------------------------------------------------

class Ps3Ids(unittest.TestCase):
    def test_intelligence_supported(self):
        attribute = R.ps3_attribute(12)
        self.assertEqual((attribute.name, attribute.status), ("Intelligence", "SUPPORTED"))
        self.assertEqual(R.PS3_STATUS_UPDATES, {12: "SUPPORTED"})
        base = [r for r in tsv("ps3_attributes.tsv") if r["ps3_id"] == "12"]
        self.assertEqual(base[0]["status"], "HYPOTHESIS")  # the shipped table is unchanged

    def test_names_stand(self):
        self.assertEqual([R.ps3_attribute(i).name for i in (13, 31, 32, 33)],
                         ["Vitality", "Attack", "Precision", "Defense"])

    def test_slot_counts(self):
        counts = R.PS3_ATTRIBUTE_SLOT_COUNTS
        self.assertEqual(counts[31], (36, 28))
        self.assertEqual(counts[31][0], 4 + 4 + 28)
        self.assertEqual([counts[i][0] - counts[i][1] for i in (32, 33)], [4, 4])
        self.assertEqual(counts[13][1], 35)


if __name__ == "__main__":
    unittest.main()
