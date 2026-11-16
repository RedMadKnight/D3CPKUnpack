"""Stage 11: ``22_corrections11.md`` (the generic SNO writer W11, the ``.app`` writer, sound S11 and
the round-10 corrections).

There are no game files here. The W11 files are laid out by hand in each test, from the rules,
not by the writer under test.
"""

import struct
import unittest
from unittest import mock

import builders as B
import d3sno
from d3sno import FormatError
from d3sno import appearance as P
from d3sno import cloth as CL
from d3sno import presentation as V
from d3sno import sno_layout as L
from d3sno import write as W
from d3sno import x360_fields as XF
from test_stage10_1 import FAT, TRIS, cloth_app

BODY = 0x10


class Marker(unittest.TestCase):
    def test_version(self):
        self.assertTrue(d3sno.__version__.startswith(("0.11.", "0.12.")))
        self.assertGreaterEqual(d3sno.STAGE, 11)


class HandLayout:
    """Places blocks by hand: pad to the alignment (from file 0x10), append, give the body offset."""

    def __init__(self, header: bytes, top: bytes, order: str):
        self.out = bytearray(header) + bytearray(top)
        self.order = order

    def place(self, data: bytes, align: int) -> int:
        self.out += bytes(-(len(self.out) - BODY) % align)
        at = len(self.out) - BODY
        self.out += data
        return at

    @property
    def cursor(self) -> int:
        return len(self.out) - BODY

    def pair(self, file_at: int, offset: int, size: int) -> None:
        struct.pack_into(self.order + "II", self.out, file_at, offset, size)


def header(order: str, version: int = 1) -> bytes:
    return B.sno_head(version, order, revision=3, source=0xABCD)[:16]


# --- The table -------------------------------------------------------------------------------------

class Table(unittest.TestCase):
    def test_types(self):
        for build in ("switch", "x360"):
            self.assertEqual(len(L.covered_types(build)), 59)
        # the table also has ps3 .dfn and .glo rows (Q345)
        self.assertEqual(len(L.covered_types("ps3")), 59)
        self.assertIn("glo", L.covered_types("ps3"))
        self.assertNotIn(("ps3", "glo"), L.RESIDUE)  # stage 12: W3-6 to W3-8 rows merged

    def test_records_by_build_then_type(self):
        # AABB is listed once for switch (under .acr) and found for .app
        self.assertEqual(L.record_layout("switch", "app", "AABB").size, 24)
        # ps3 lists records per type
        self.assertEqual(L.record_layout("ps3", "app", "UberMaterial").size, 72)
        with self.assertRaises(FormatError):
            L.record_layout("switch", "app", "NoSuchRecord")

    def test_slots_in_walk_order(self):
        slots = L.pair_slots("switch", "acr", "ActorDefinition")
        self.assertEqual([s.path for s in slots],
                         ["hTagMap", "ptMsgTriggeredEvents", "szInterenalDescription", "szVORole"])
        app = [s.path for s in L.pair_slots("switch", "app", "AppearanceDefinition")]
        # W11-2: declaration order, not pair-offset order
        self.assertEqual(app[0], "ptAppearanceMaterials")
        self.assertLess(app.index("ptAppearanceMaterials"), app.index("tStructure.ptBoneStructure"))
        self.assertIn("tStructure.tGeoSets[1].ptSubObjects", app)
        self.assertEqual(app[-2:], ["serAppearanceSubObjectsIndices", "serAppearanceSubObjectsVertices"])

    def test_union_members_give_each_pair_once(self):
        slots = L.pair_slots("switch", "snd", "SoundDefinition")
        self.assertEqual(len({s.offset for s in slots}), len(slots))


# --- W11-1 to W11-7 on hand-laid files -------------------------------------------------------------

def actor_file(trailing=b""):
    """switch .acr: tag map (4), MsgTriggeredEvent array (8), two texts (8)."""
    top = bytearray(range(256)) + bytearray(range(192))  # every non-pair byte set (W11-1)
    for at in (0x58, 0x78, 0x1A8, 0x1B8):
        top[at:at + 8] = bytes(8)
    event = bytearray(b"\x5A" * 192)
    h = HandLayout(header("<"), top, "<")
    tags = h.place(b"TAGMAP", 4)               # body 448
    events = h.place(bytes(event), 8)          # pad 2 -> 456
    text1 = h.place(b"desc\0", 8)              # 648
    text2 = h.place(b"VO!\0", 8)               # pad 3 -> 656
    for at, offset, size in ((0x58, tags, 6), (0x78, events, 192), (0x1A8, text1, 5), (0x1B8, text2, 4)):
        h.pair(BODY + at, offset, size)
    return bytes(h.out) + trailing, (tags, events, text1, text2)


def font_file(order="<", build="switch"):
    """.fnt with deferred groups 0 and 1 (W11-3), a shared kerning block (W11-6) and an equal but
    separate texture block."""
    if build == "ps3":
        top_size, tex_pair, ps_pair, ps_size, char_at, kern_at, align = 204, 0xAC, 0xBC, 72, 0x24, 0x34, 4
    else:
        top_size, tex_pair, ps_pair, ps_size, char_at, kern_at, align = 216, 0xA8, 0xC0, 88, 0x28, 0x40, 8
    top = bytearray(b"\x01" * top_size)
    for at in (tex_pair, ps_pair):
        top[at:at + 8] = bytes(8)
    tex = [bytearray(b"\x02" * 40) for _ in range(2)]
    for t in tex:
        t[0x10:0x18] = t[0x20:0x28] = bytes(8)
    sizes = [bytearray(b"\x03" * ps_size) for _ in range(2)]
    for s in sizes:
        s[char_at:char_at + 8] = s[kern_at:kern_at + 8] = bytes(8)
    h = HandLayout(header(order, 1), top, order)
    t0 = h.place(b"".join(tex), align)
    p0 = h.place(b"".join(sizes), align)
    pairs = [(BODY + tex_pair, t0, 80), (BODY + ps_pair, p0, 2 * ps_size)]
    tex_at = [BODY + t0 + 40 * i for i in range(2)]
    ps_at = [BODY + p0 + ps_size * i for i in range(2)]
    kerning = b"K" * 32
    if build == "ps3":
        # group 0 (deferred): each point size's char table, then its kerning table
        c0 = h.place(b"chars", 4)
        k0 = h.place(kerning, 4)
        c1 = h.place(b"abc", 4)
        pairs += [(ps_at[0] + char_at, c0, 5), (ps_at[0] + kern_at, k0, 32),
                  (ps_at[1] + char_at, c1, 3), (ps_at[1] + kern_at, k0, 16)]  # shared, own size
    else:
        # the kerning arrays are walked; the char tables are group 0
        k0 = h.place(kerning, 8)
        c0 = h.place(b"chars", 1)
        c1 = h.place(b"abc", 1)
        pairs += [(ps_at[0] + kern_at, k0, 32), (ps_at[1] + kern_at, k0, 16),
                  (ps_at[0] + char_at, c0, 5), (ps_at[1] + char_at, c1, 3)]
    blob_align = 4 if build == "ps3" else 1
    d0 = h.place(b"PIXELS!", blob_align)     # group 1: texture 0 data, frame
    f0 = h.place(b"fr", blob_align)
    d1 = h.place(b"PIXELS!", blob_align)     # equal to d0, stored again (not shared)
    pairs += [(tex_at[0] + 0x10, d0, 7), (tex_at[0] + 0x20, f0, 2), (tex_at[1] + 0x10, d1, 7)]
    for at, offset, size in pairs:
        h.pair(at, offset, size)
    return bytes(h.out)


class GenericWriter(unittest.TestCase):
    def test_actor_rebuilt_with_alignment(self):
        data, (tags, events, text1, text2) = actor_file()
        self.assertEqual((tags, events, text1, text2), (448, 456, 648, 656))
        tree = L.parse_tree(data, "acr", "switch")
        self.assertEqual(L.write_tree(tree), data)
        self.assertEqual(W.write(tree), data)
        self.assertTrue(L.is_canonical(data, "acr", "switch"))

    def test_record_bytes_copied(self):
        data, _ = actor_file()
        tree = L.parse_tree(data, "acr", "switch")
        self.assertEqual(tree.header, data[:16])
        self.assertEqual(bytes(tree.top.raw[:12]), data[16:28])  # run-time words kept (W11-1)

    def test_end_of_file(self):
        data, _ = actor_file(trailing=b"\0\0\0\0")
        tree = L.parse_tree(data, "acr", "switch")
        self.assertTrue(any("W11-7" in n for n in tree.notes))
        self.assertEqual(L.write_tree(tree), data[:-4])

    def test_deferred_groups_and_sharing_switch(self):
        data = font_file()
        tree = L.parse_tree(data, "fnt", "switch")
        self.assertEqual(L.write_tree(tree), data)
        shared = L.shared_pairs(tree)
        self.assertEqual([path for path, _ in shared], ["ptPointSizeTable[1].ptKerningTable"])
        # equal content is not sharing (W11-6)
        paths = {path: value for path, _, _, value in L.iter_pairs(tree)}
        a, b = paths["ptTextureTable[0].serTextureData"], paths["ptTextureTable[1].serTextureData"]
        self.assertEqual(a.block.data, b.block.data)
        self.assertIsNot(a.block, b.block)
        self.assertEqual(paths["ptPointSizeTable[1].ptKerningTable"].pair_size(), 16)

    def test_deferred_groups_ps3(self):
        data = font_file(">", "ps3")
        tree = L.parse_tree(data, "fnt", "ps3")
        self.assertEqual(L.write_tree(tree), data)

    def test_walk_order_breaks_offset_order(self):
        # Placing by pair offset instead would put the char tables first: the files differ.
        data = font_file()
        tree = L.parse_tree(data, "fnt", "switch")
        values = {p: v for p, _, _, v in L.iter_pairs(tree)}
        self.assertLess(values["ptPointSizeTable[0].ptKerningTable"].stored[0],
                        values["ptPointSizeTable[0].serCharTable"].stored[0])

    def test_changed_block_laid_out_again(self):
        data = font_file()
        tree = L.parse_tree(data, "fnt", "switch")
        L.set_block(tree, "ptPointSizeTable[0].serCharTable", b"a much longer char table")
        out = L.write_tree(tree)
        again = L.parse_tree(out, "fnt", "switch")
        values = {p: v for p, _, _, v in L.iter_pairs(again)}
        self.assertEqual(values["ptPointSizeTable[0].serCharTable"].block.data, b"a much longer char table")
        self.assertEqual(values["ptTextureTable[1].serTextureData"].block.data, b"PIXELS!")
        self.assertEqual(L.write_tree(again), out)
        with self.assertRaises(FormatError):
            L.set_block(tree, "ptTextureTable", b"x")
        with self.assertRaises(KeyError):
            L.set_block(tree, "no.such.path", b"x")

    def test_set_elements(self):
        data = font_file()
        tree = L.parse_tree(data, "fnt", "switch")
        kerning = [L.new_record("switch", "fnt", "FontConsoleKerning") for _ in range(3)]
        L.set_elements(tree, "ptPointSizeTable[0].ptKerningTable", kerning)
        again = L.parse_tree(L.write_tree(tree), "fnt", "switch")
        values = {p: v for p, _, _, v in L.iter_pairs(again)}
        self.assertEqual(len(values["ptPointSizeTable[0].ptKerningTable"].block.elements), 3)
        # the other point size kept the old block (it shared it)
        kept = values["ptPointSizeTable[1].ptKerningTable"].block.elements
        self.assertEqual([bytes(e.raw) for e in kept], [b"K" * 16])
        with self.assertRaises(FormatError):
            L.set_elements(tree, "ptPointSizeTable[0].ptKerningTable", [L.new_record("switch", "fnt", "FontConsoleTexture")])

    def test_empty_pair_notes(self):
        data, _ = actor_file()
        data = bytearray(data)
        struct.pack_into("<II", data, BODY + 0x1B8, 0x99, 0)  # an empty pair with an offset
        tree = L.parse_tree(bytes(data), "acr", "switch")
        self.assertTrue(any("W11-5" in n for n in tree.notes))


def ps3_draw_app():
    """ps3 .app with one sub-object and one draw: descriptor (16), formats (4), deferred index
    list and streams A, B (group 1, 2-aligned, B empty in cursor form) and C (group 2); the pools
    in the walk, empty, in cursor form."""
    top = bytearray(568)
    top[0:12] = b"\x7F" * 12  # RequiredDefinitionHeader: run-time words, copied
    struct.pack_into(">i", top, 0x60, 1)
    sub = bytearray(308)
    sub[0:4] = b"\x00\x00\x01\x00"
    draw = bytearray(160)
    draw[0x00:0x04] = b"\x00\x00\x00\x09"
    h = HandLayout(header(">", 295), top, ">")
    s_at = h.place(bytes(sub), 8)        # 568
    d_at = h.place(bytes(draw), 4)       # 876
    desc = h.place(b"D" * 16, 16)        # 1036 -> 1040
    fa = h.place(b"FA01", 4)             # 1056
    fb = h.place(b"FB0", 4)              # 1060
    pools = h.cursor                     # 1063: both pools {cursor, 0}
    idx = h.place(b"\x00\x01\x00\x02\x00\x00", 2)   # group 1: 1064
    sa = h.place(b"AAAAAA", 2)
    sb = h.cursor                        # stream B empty, cursor form
    sc = h.place(b"CCCC", 2)             # group 2
    sub_file, draw_file = BODY + s_at, BODY + d_at
    for at, offset, size in ((BODY + 0x64, s_at, 308), (sub_file + 0x100, d_at, 160),
                             (draw_file + 0x08, desc, 16), (draw_file + 0x80, fa, 4),
                             (draw_file + 0x90, fb, 3), (BODY + 0x220, pools, 0),
                             (BODY + 0x228, pools, 0), (draw_file + 0x18, idx, 6),
                             (draw_file + 0x2C, sa, 6), (draw_file + 0x54, sb, 0),
                             (draw_file + 0x6C, sc, 4)):
        h.pair(at, offset, size)
    return bytes(h.out), dict(desc=desc, fa=fa, pools=pools, idx=idx, sb=sb, sc=sc)


class Ps3Draws(unittest.TestCase):
    def test_hand_layout(self):
        data, at = ps3_draw_app()
        self.assertEqual((at["desc"], at["pools"], at["idx"], at["sc"]), (1040, 1063, 1064, 1076))
        tree = L.parse_tree(data, "app", "ps3")
        self.assertEqual(L.write_tree(tree), data)

    def test_cursor_form_follows_the_cursor(self):
        data, _ = ps3_draw_app()
        tree = L.parse_tree(data, "app", "ps3")
        L.set_block(tree, "tStructure.tGeoSets[0].serSubObjects[0].serDraws[0].serStreamA", b"AAAAAAAA")
        out = L.write_tree(tree)
        sb = struct.unpack_from(">II", out, BODY + 568 + 308 + 0x54)
        sc = struct.unpack_from(">II", out, BODY + 568 + 308 + 0x6C)
        self.assertEqual(sb, (1064 + 6 + 8, 0))
        self.assertEqual(sc, (1078, 4))


class CodeKind(unittest.TestCase):
    """W11-5: formula code aligns to 4 and stores {cursor - 0x10, 0} when empty. No row of the
    table has kind ``code`` (Q346); a one-record table is put in its place."""

    def test_empty_code_and_cursor_blob(self):
        fields = (
            L.LayoutField(0, "code", 0x0, "code", "", None, 4, "zero", None),
            L.LayoutField(1, "blob", 0x8, "blob", "", None, 1, "zero", None),
            L.LayoutField(2, "pool", 0x10, "blob", "", None, 1, "cursor", None),
        )
        fake = ({("switch", "zzz", "Top"): L.RecordLayout("switch", "zzz", "Top", 26, fields)},
                {("switch", "Top"): ("switch", "zzz", "Top")}, {("switch", "zzz"): "Top"})
        L.pair_slots.cache_clear()
        try:
            with mock.patch.object(L, "_table", lambda: fake):
                top = L.Record("Top", bytearray(26))
                top.pairs[0x8] = L.PairValue(L.Block(b"xyz"))
                tree = L.SnoTree("switch", "zzz", header("<"), top)
                out = L.write_tree(tree)
        finally:
            L.pair_slots.cache_clear()
        self.assertEqual(struct.unpack_from("<II", out, BODY), (28, 0))       # 26 aligned to 4
        self.assertEqual(struct.unpack_from("<II", out, BODY + 8), (28, 3))
        self.assertEqual(struct.unpack_from("<II", out, BODY + 0x10), (31, 0))  # not aligned
        self.assertEqual(len(out), BODY + 31)


# --- Section 3: the .app writer ---------------------------------------------------------------------

def canonical(data, build=None):
    return L.write_tree(L.parse_tree(data, "app", build))


SKIN = {"palette": [0], "vertices": [(0, 0, 0), (10, 0, 0), (0, 10, 0)], "indices": [0, 1, 2],
        "weights": (255, 0, 0, 0)}


class AppWriter(unittest.TestCase):
    def test_rebuilds_a_w11_file(self):
        for data in (canonical(B.build_app_switch()), canonical(B.build_app_ps3(), "ps3"),
                     canonical(B.build_app_switch(version=302, order=">"), "x360")):
            app = P.parse_appearance(data)
            self.assertEqual(app.warnings, [])
            self.assertEqual(W.write_app(app), data)

    def test_in_place_edits_kept(self):
        data = canonical(B.build_app_switch(bones=[(0x10, -1, (0.0, 0.0, 0.0))]))
        app = P.parse_appearance(data)
        app.source[16 + 0x100] ^= 0xFF  # an in-place byte change in a record
        self.assertEqual(W.write_app(app), bytes(app.source))

    def test_new_packed_vertex_list(self):
        data = canonical(B.build_app_switch())
        app = P.parse_appearance(data)
        old = L.find_pair(W.app_tree(app), W.app_pair_path(app, 0))[2].block.data
        new = old + old[:28]  # one more packed vertex
        W.set_packed_vertices(app, 0, new, allow_count_change=True)  # V11-9
        out = W.write_app(app, check_mesh=False)
        tree = L.parse_tree(out, "app", "switch")
        _, _, value = L.find_pair(tree, W.app_pair_path(app, 0))
        self.assertEqual(value.block.data, new)
        self.assertEqual(value.stored[1], len(new))
        self.assertEqual(len(out), len(data) + 28)
        self.assertEqual(L.write_tree(tree), out)

    def test_skinning_lists(self):
        data = canonical(B.build_app_switch(bones=[(0x10, -1, (0.0, 0.0, 0.0))],
                                            sub_objects=[{"skin": SKIN, "index_count": 3}]))
        app = P.parse_appearance(data)
        path = W.app_pair_path(app, 0, skin=0)
        self.assertTrue(path.endswith("ptSkinningInfo[0].serPackVertList"))
        self.assertTrue(W.app_pair_path(app, 0, skin=0, which="indices").endswith("serPackIndexList"))
        W.set_packed_vertices(app, 0, b"\x00" * 36 * 4, skin=0, allow_count_change=True)
        tree = L.parse_tree(W.write_app(app, check_mesh=False), "app", "switch")
        self.assertEqual(len(L.find_pair(tree, path)[2].block.data), 144)
        with self.assertRaises(FormatError):
            W.app_pair_path(app, 5)

    def test_source_not_in_w11_layout(self):
        data = B.build_app_switch()  # the builder leaves gaps
        self.assertFalse(L.is_canonical(data, "app"))
        app = P.parse_appearance(data)
        self.assertEqual(W.write_app(app), data)  # copied when nothing changes a block
        W.set_packed_vertices(app, 0, b"\x00" * 28, allow_count_change=True)
        with self.assertRaises(FormatError):
            W.write_app(app)

    def test_ps3_has_no_packed_lists(self):
        app = P.parse_appearance(canonical(B.build_app_ps3(), "ps3"))
        with self.assertRaises(FormatError):
            W.app_pair_path(app, 0)


# --- Section 4: sound ---------------------------------------------------------------------------------

def fsb5(mode, data, name=b"Bank-0001", table_size=None, chunks=(), samples=300, stereo=False):
    chunk_raw = bytearray()
    for i, (kind, body) in enumerate(chunks):
        more = 1 if i + 1 < len(chunks) else 0
        chunk_raw += struct.pack("<I", more | len(body) << 1 | kind << 25) + body
    word = (1 if chunks else 0) | 9 << 1 | int(stereo) << 5 | samples << 34
    headers = struct.pack("<Q", word) + bytes(chunk_raw)
    headers += bytes(-len(headers) % 16)
    table = b""
    if name is not None:
        table = struct.pack("<I", 4) + name + b"\0"
        if table_size is None:  # V11-6: zero bytes until the data start at a multiple of 32
            table_size = len(table) + -(60 + len(headers) + len(table)) % 32
        table = table.ljust(table_size, b"\0")
    head = b"FSB5" + struct.pack("<6I", 1, 1, len(headers), len(table), len(data), mode)
    return head.ljust(60, b"\0") + headers + table + data


class Sound(unittest.TestCase):
    def test_name_table(self):
        data = fsb5(16, bytes(280) + bytes(8))
        fsb = V.parse_fsb5(data)
        self.assertEqual(fsb.name, "Bank-0001")
        self.assertEqual(len(fsb.name_table), 20)  # 60 + 16 + 20 = 96
        self.assertEqual(V.fsb5_problems(fsb, "SoundBank\\Bank\\0001.fsb"), [])
        self.assertTrue(V.fsb5_problems(fsb, "SoundBank\\Other\\0001.fsb"))
        self.assertEqual(V.fsb5_name_for_path("SoundBank/Bank/0042.fsb"), "Bank-0042")
        self.assertEqual(V.write_fsb5(fsb), data)

    def test_no_name_table(self):
        fsb = V.parse_fsb5(fsb5(16, bytes(288), name=None))
        self.assertIsNone(fsb.name)
        self.assertIn("no name table (S11-1)", V.fsb5_problems(fsb))

    def test_name_table_padding(self):
        table = bytearray((struct.pack("<I", 4) + b"Bank-0001\0").ljust(28, b"\0"))
        table[-1] = 1
        fsb = V.parse_fsb5(fsb5(16, bytes(288)))
        fsb.name_table = bytes(table)
        self.assertIn("name table padding is not zero (S11-1)", V.fsb5_problems(fsb))

    def test_chunk_type_5_kept_raw(self):
        data = fsb5(16, bytes(288), chunks=[(5, bytes(4)), (3, struct.pack("<II", 1, 2))])
        fsb = V.parse_fsb5(data)
        self.assertEqual([c.data for c in fsb.unknown_chunks], [bytes(4)])
        self.assertEqual(fsb.sample.loop, (1, 2))
        self.assertEqual(V.write_fsb5(fsb), data)

    def test_mode_16_padding(self):
        # 300 samples, mono: 2 frames = 280 bytes, then 8 zero bytes to 288
        fsb = V.parse_fsb5(fsb5(16, b"\x01" * 280 + bytes(8)))
        self.assertEqual(fsb.frame_bytes(), 280)
        self.assertEqual(len(fsb.packets()), 2)
        self.assertEqual(fsb.padding(), bytes(8))
        self.assertEqual(V.fsb5_problems(fsb), [])
        stereo = V.parse_fsb5(fsb5(16, bytes(560), stereo=True))
        self.assertEqual((stereo.frame_bytes(), len(stereo.packets())), (560, 4))
        bad = V.parse_fsb5(fsb5(16, bytes(280) + b"\x01" + bytes(7)))
        self.assertIn("mode-16 padding is not zero (S11-3)", V.fsb5_problems(bad))
        short = V.parse_fsb5(fsb5(16, bytes(280)))
        self.assertTrue(any("288 expected" in p for p in V.fsb5_problems(short)))


# --- Section 5: round-10 corrections ---------------------------------------------------------------

class Round10(unittest.TestCase):
    def test_v10_s4_counts(self):
        rows = XF.x360_struct_names()
        with_offset = [r for r in rows if r.x360_offset is not None]
        self.assertEqual(len({(r.struct, r.x360_offset) for r in with_offset}), 102)
        alone = [r for r in with_offset if len(XF.proposals_at(r.struct, r.x360_offset)) == 1]
        self.assertEqual(len(alone), 99)
        last = [r for r in rows if r.x360_offset is None]
        self.assertEqual([(r.struct, r.field_index, r.proposed_name) for r in last],
                         [("ExperienceLevel", None, "BountyCampsXP, BountyCampsGold")])

    def test_v10_g1_4_edges_of_extra_faces(self):
        # an extra triangle whose edges (0, 4) and (5, 0) are not distance pairs
        app = cloth_app([(1.0, (0, 4, 5))])
        self.assertEqual(app.warnings, [])
        cloth = app.sub_objects[0].cloth
        sub = app.sub_objects[0]
        cloth.face_count = len(cloth.faces)  # over all stored faces the rule fails
        problems = CL.cloth_problems(cloth, sub.fat_draw.positions, (P.FAT_AXIS_TOLERANCE,) * 3, 0, "x")
        self.assertTrue(any("distance pairs" in p for p in problems))

    def test_v10_g1_3_draw_holds_the_fat_positions(self):
        drawn = [(1.0, 2.0, 3.0), (6.0, 2.0, 3.0), (1.0, 7.0, 3.0)]
        # another order, within one step (0.5)
        fat = [drawn[2], (drawn[0][0] + 0.4, drawn[0][1], drawn[0][2]), drawn[1]]
        app = P.parse_appearance(B.build_app_ps3(fat=fat, cloth=True, cloth_draw=True))
        self.assertEqual([w for w in app.warnings if "V10-G1-3" in w], [])
        far = [drawn[0], drawn[1], (1.0, 9.0, 3.0)]
        app = P.parse_appearance(B.build_app_ps3(fat=far, cloth=True, cloth_draw=True))
        self.assertTrue(any("V10-G1-3" in w for w in app.warnings))


if __name__ == "__main__":
    unittest.main()
