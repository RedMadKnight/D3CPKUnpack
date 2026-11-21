"""Stage 17 writer half (``37_mesh_parts.md`` sections 1 and 2): record rules W17-1 to W17-8, the
writer with added and removed parts (W17-10, W17-14), moving a sub-object between files (W17-11) and
the rebuild of every record from zero (the "Check"). Synthetic files only (``builders``)."""

import copy
import dataclasses
import struct
import unittest
from unittest import mock

import builders as B
from d3sno import FormatError
from d3sno import app_mesh as M
from d3sno import app_parts as PP
from d3sno import appearance as P
from d3sno import octree as O
from d3sno import sno_layout as L
from d3sno import structure as S
from test_stage12 import BONES, laid_out, ps3
from test_stage13 import SKELETON_8, grid

CONTENT = 1 | 1 << 8 | 1 << 13 | 1 << 14 | 1 << 15 | 1 << 16
TRI = [(0, 0, 0), (10, 0, 0), (0, 10, 0)]
FAT = [(0.0, 0.0, 0.0), (4.0, 0.0, 0.0), (0.0, 4.0, 0.0)]
SKIN2 = {"palette": [2, 1], "vertices": [(0, 0, 0), (10, 0, 0), (0, 10, 0)], "indices": [0, 1, 2]}
SKELETON_60 = [(0x300 + i, i - 1, (float(i), 0.0, 0.0)) for i in range(60)]


# --- Test-side rules (written independently of the library) ----------------------------------------

def djb(text) -> int:
    """W17-2 as the test computes it."""
    data = text.encode("ascii") if isinstance(text, str) else bytes(text)
    h = 0
    for c in data:
        h = (h * 33 + (c + 32 if 65 <= c <= 90 else c)) % 2 ** 32
    return h


def order_of(build):
    return "<" if build == "switch" else ">"


def align(value, step):
    return (value + step - 1) // step * step


def sub_records(tree, geo=0):
    value = tree.top.pairs.get((0x64, 0x7C)[geo])
    return list(value.block.elements) if value is not None and value.block else []


def elements(record, offset):
    value = record.pairs.get(offset)
    return list(value.block.elements) if value is not None and value.block and value.block.elements else []


def block(record, offset):
    value = record.pairs.get(offset)
    return b"" if value is None or value.block is None else value.block.data


def expected_flags(build, record):
    """W17-1 from a record's content, as the test reads it."""
    order = order_of(build)
    flags = struct.unpack_from(order + "I", record.raw, 0)[0]
    keep = flags & 0x1FFFF & ~CONTENT
    if build == "ps3":
        draws = elements(record, 0x100)
        fat = struct.unpack_from(order + "i", record.raw, 4)[0]
        skinned = any(block(d, 0x08)[0] == 0x83 for d in draws)
        return keep | 1 | bool(fat) << 8 | bool(draws) << 13 | skinned << 14
    rigid = struct.unpack_from(order + "i", record.raw, 0xA0)[0] > 0
    skins = len(elements(record, 0xB8))
    cloth = bool(block(record, 0x40)) if record.pairs.get(0x40) is not None and record.pairs[0x40].block is not None \
        and record.pairs[0x40].block.elements is None else bool(elements(record, 0x40))
    fat = struct.unpack_from(order + "I", record.raw, 4)[0]
    return keep | 1 | bool(cloth or fat or flags >> 8 & 1) << 8 | bool(rigid or skins) << 15 | bool(skins) << 16


def canon(data, build=None):
    """A synthetic file that follows the W17 rules: written once with every W13 rule, then the flag
    bits (W17-1), ps3 name hashes (W17-2), rigid ps3 bone ranges (0), rigid work sizes (W2-10) and
    shared ps3 blocks (W17-6) set by the test's own code."""
    data = M.write_app_mesh(P.parse_appearance(data, build), derive=True)
    tree = L.parse_tree(data, "app", build)
    build = tree.build
    order = order_of(build)
    first = {}
    for geo in (0, 1):
        for record in sub_records(tree, geo):
            struct.pack_into(order + "I", record.raw, 0, expected_flags(build, record))
            if build == "ps3":
                empty = not elements(record, 0x100)
                at = 0x118
            else:
                empty = not struct.unpack_from(order + "i", record.raw, 0xA0)[0] and not elements(record, 0xB8)
                at = 0xD8
            if empty:  # W17-3: no packed vertices or draws: bias and step 0
                record.raw[at:at + 24] = bytes(24)
            if build != "ps3":
                continue
            name = bytes(record.raw[0x48:0xC8]).split(b"\0")[0]
            struct.pack_into(order + "I", record.raw, 0xCC, djb(name))
            for draw in elements(record, 0x100):
                descriptor = block(draw, 0x08)
                if descriptor[0] == 0x81:
                    nv, ni = struct.unpack_from(">HH", descriptor, 8)
                    struct.pack_into(">4H", draw.raw, 0x60, 0, 0, 0, 0)
                    work = align(12 * align(nv, 8) + align(2 * ni, 16) + 16 * descriptor[1] + 192, 128)
                    struct.pack_into(">I", draw.raw, 0x98, work)
                for offset in (0x08, 0x80, 0x90):
                    key = (offset == 0x08, block(draw, offset))
                    if key in first:
                        draw.pairs[offset] = L.PairValue(first[key], shared=True)
                    else:
                        first[key] = draw.pairs[offset].block
    return L.write_tree(tree)


def triangles(app_sub):
    """The triangles of a sub-object as position triples (draws end to end)."""
    out = []
    for draw in app_sub.draws:
        for t in range(0, len(draw.indices), 3):
            out.append(tuple(tuple(round(c, 3) for c in draw.positions[i]) for i in draw.indices[t:t + 3]))
    return out


def rigid_draw(positions, indices=None, kind="packed"):
    n = len(positions)
    return P.Draw(False, [tuple(map(float, p)) for p in positions], [(0.0, 0.0, 1.0)] * n,
                  [bytes((1, 2, 3, 4))] * n, [bytes((5, 6, 7, 8))] * n, [(0.5, 0.25)] * n,
                  [(0x8000, 0x8000)] * n, list(range(n)) if indices is None else list(indices), kind=kind)


def skinned_draw(positions, indices, joints, weights, kind="packed"):
    draw = rigid_draw(positions, indices, kind)
    draw.skinned, draw.joints, draw.weights = True, [tuple(j) for j in joints], [tuple(w) for w in weights]
    return draw


def new_sub(name=None, draws=(), geo=0, material=None, **options):
    return P.SubObject(0, name, None, (0.0,) * 3, (0.0,) * 3, None, None, list(draws), material_hash=material,
                       geo_set=geo, **options)


def edited(app, geo0=None, geo1=None):
    return dataclasses.replace(app, sub_objects=list(app.sub_objects if geo0 is None else geo0),
                               extra_sub_objects=list(app.extra_sub_objects if geo1 is None else geo1))


# --- Fixtures ----------------------------------------------------------------------------------------

def switch_two(order="<"):
    version, build = (302, "x360") if order == ">" else (306, None)
    return canon(laid_out(B.build_app_switch(version=version, order=order, pool_spare=0, sub_objects=[
        {"rigid": [(0, 0, 0), (10, 0, 0), (0, 10, 0)], "index_count": 3},
        {"rigid": [(0, 0, 0), (6, 0, 0), (0, 6, 0)], "index_count": 3},
        {"rigid": [(2, 2, 0), (8, 2, 0), (2, 8, 0)], "index_count": 3}]), build), build), build


def switch_skin(order="<"):
    version, build = (302, "x360") if order == ">" else (306, None)
    return canon(laid_out(B.build_app_switch(version=version, order=order, bones=BONES, pool_spare=0,
                                             sub_objects=[{"skin": SKIN2, "index_count": 0}]), build), build), build


def switch_cloth_two(order="<"):
    version, build = (302, "x360") if order == ">" else (306, None)
    return canon(laid_out(B.build_app_switch(version=version, order=order, pool_spare=0, sub_objects=[
        {"rigid": [(0, 0, 0), (10, 0, 0), (0, 10, 0)], "index_count": 3},
        {"index_count": 3, "cloth": True},
        {"rigid": [(0, 0, 0), (4, 0, 0), (0, 4, 0)], "index_count": 3}]), build), build), build


def ps3_file(**options):
    return canon(ps3(**options), "ps3"), "ps3"


def fan_skin(bones=50):
    """Triangles (0, 2t + 1, 2t + 2): vertex 0 on bone 0, the others on bone t + 1, so triangle t uses
    bones {0, t + 1} (W17-7: the cut falls before the triangle that brings bone 45)."""
    positions = [(0, 0, 0)]
    joints = [(0, 0, 0, 0)]
    indices = []
    for t in range(bones):
        positions += [(t + 1, 1, 0), (t + 1, 2, 0)]
        joints += [(t + 1, 0, 0, 0)] * 2
        indices += [0, 2 * t + 1, 2 * t + 2]
    return skinned_draw(positions, indices, joints, [(255, 0, 0, 0)] * len(positions))


def ps3_cloth_draw():
    """ps3 skinned cloth with fat vertices and one draw holding the fat positions (V9-G1-3)."""
    app = P.parse_appearance(ps3(fat=FAT, cloth=True, cloth_draw=True, bones=BONES, skinned=True), "ps3")
    app.sub_objects[0].draws[0].positions[:] = FAT
    with allow_cloth_edits():  # fixture only: ``38`` V23-5 refuses this edit in the library
        return canon(M.write_app_mesh(app, derive=True), "ps3"), "ps3"


def allow_cloth_edits():
    """Test fixtures only: lift the V23-5 refusal to build a cloth sub-object whose draw differs
    from what the builders give."""
    return mock.patch.object(M, "_cloth_parts_unchanged", lambda *args: None)


def multi_record(order="<"):
    """A switch / x360 file whose skinned sub-object has two skinning records (51 bones), written by
    the writer and then made canonical."""
    version, build = (302, "x360") if order == ">" else (306, None)
    data = canon(laid_out(B.build_app_switch(version=version, order=order, bones=SKELETON_60, pool_spare=0,
                                             sub_objects=[{"skin": SKIN2, "index_count": 0}]), build), build)
    app = P.parse_appearance(data, build)
    out = M.write_app_mesh(app, edited(app, app.sub_objects + [new_sub("Fan", [fan_skin()])]), derive=True)
    return canon(out, build), build


def with_materials(data, build, looks=(0x61, 0x62), entries=None, assign=(0xAAA, 0xBBB, 0xAAA)):
    """The file with a material table (one per-look record per look) and sub-object materials."""
    build_name = build or "switch"
    entries = entries if entries is not None else [
        (0xAAA, [[(1, 5)], [(1, 6)]][:len(looks)]), (0xBBB, [[(1, 7)], [(1, 8)]][:len(looks)])]
    data = B.add_app_materials(data, build_name, list(looks), entries)
    for i, value in enumerate(assign):
        data = B.set_sub_object_material(data, build_name, i, value, 0x70 + i)
    return canon(laid_out(data, build), build)


class Checked(unittest.TestCase):
    def check(self, out, build=None):
        """The edit checks of ``37``: strict parse, structure checks, W17-1 flags, rebuild from zero
        and an unchanged second write."""
        app = P.parse_appearance(out, build, strict=True)
        report = S.verify_structure(out, "app", build)
        self.assertTrue(report.ok, report.problems)
        self.assertEqual(M.index_pool_problems(app), [])
        tree = L.parse_tree(out, "app", build)
        for geo in (0, 1):
            for record in sub_records(tree, geo):
                self.assertEqual(struct.unpack_from(order_of(tree.build) + "I", record.raw, 0)[0],
                                 expected_flags(tree.build, record))
        self.assertEqual(M.write_app_mesh(app, derive=True), out)
        self.assertEqual(M.rebuild_from_zero(app), out)
        return app


# --- W17-2 ---------------------------------------------------------------------------------------------

class NameHash(unittest.TestCase):
    def test_hash(self):
        self.assertEqual(PP.name_hash("a"), 0x61)
        self.assertEqual(PP.name_hash("ab"), 0x61 * 33 + 0x62)
        self.assertEqual(PP.name_hash("AB"), PP.name_hash("ab"))
        self.assertEqual(PP.name_hash(b"Body_2"), djb("Body_2"))
        self.assertEqual(PP.name_hash("[Z@"), djb("[Z@"))  # only A..Z are lowered
        self.assertEqual(PP.name_hash("x" * 40), djb("x" * 40))  # mod 2^32
        self.assertEqual(PP.name_hash(""), 0)
        with self.assertRaisesRegex(FormatError, "ASCII"):
            PP.name_hash("Bödy")

    def test_free_name(self):
        self.assertEqual(PP.free_name("Body", set()), "Body")
        self.assertEqual(PP.free_name("Body", {djb("body")}), "Body_2")
        self.assertEqual(PP.free_name("Body", {djb("Body"), djb("Body_2")}), "Body_3")
        self.assertEqual(PP.free_name("Body", {djb("Body"), djb("Body_3")}), "Body_2")

    def test_ps3_name_field(self):
        self.assertEqual(PP.ps3_name_field("Arm"), b"Arm" + bytes(125))
        self.assertEqual(len(PP.ps3_name_field("n" * 127)), 128)
        for bad in ("n" * 128, "Ärm", "", "a\0b"):
            with self.assertRaises(FormatError):
                PP.ps3_name_field(bad)


# --- W17-1 ---------------------------------------------------------------------------------------------

class Flags(unittest.TestCase):
    def test_ps3(self):
        stored = 1 << 12 | 1 << 3 | 1 << 20 | 1 << 8 | 1 << 15 | 1 << 16 | 1 << 13
        self.assertEqual(PP.sub_object_flags("ps3", stored), 1 | 1 << 12 | 1 << 3)
        self.assertEqual(PP.sub_object_flags("ps3", 0, draws=2), 1 | 1 << 13)
        self.assertEqual(PP.sub_object_flags("ps3", 0, draws=1, skinned_draws=1), 1 | 1 << 13 | 1 << 14)
        self.assertEqual(PP.sub_object_flags("ps3", 0, fat=True), 1 | 1 << 8)
        self.assertEqual(PP.sub_object_flags("ps3", 0, rigid=True, skin_records=1, cloth=True), 1)
        self.assertEqual(PP.sub_object_flags("ps3", 1 << 17 | 1 << 16 | 1 << 1), 1 | 1 << 1)  # 17..31 clear

    def test_switch(self):
        self.assertEqual(PP.sub_object_flags("switch", 0x55 | 1 << 31 | 1 << 13), 0x55)
        self.assertEqual(PP.sub_object_flags("switch", 0, rigid=True), 1 | 1 << 15)
        self.assertEqual(PP.sub_object_flags("x360", 0, skin_records=2), 1 | 1 << 15 | 1 << 16)
        self.assertEqual(PP.sub_object_flags("switch", 0, cloth=True), 1 | 1 << 8)
        self.assertEqual(PP.sub_object_flags("switch", 0, fat=True), 1 | 1 << 8)
        self.assertEqual(PP.sub_object_flags("switch", 1 << 8, rigid=True), 1 | 1 << 15)  # a new part
        self.assertEqual(PP.sub_object_flags("switch", 1 << 8, rigid=True, keep_bit8=True), 1 | 1 << 8 | 1 << 15)
        self.assertEqual(PP.sub_object_flags("switch", 0, draws=3, skinned_draws=3), 1)


# --- W17-3, W17-4 --------------------------------------------------------------------------------------

class RecordsFromZero(unittest.TestCase):
    def template(self, build):
        order = order_of(build)
        size = 308 if build == "ps3" else 240
        raw = bytearray(b"\xEE" * size)
        fields = PP.sub_fields(build)
        struct.pack_into(order + "I", raw, 0, 0x1055 | 1 << 8 | 1 << 13 | 1 << 15 | 1 << 16 | 1 << 20)
        struct.pack_into(order + "iif", raw, fields["physics"], -1, 7, 1.0)
        if build != "ps3":
            struct.pack_into(order + "I", raw, fields["name_info"], 0x05020006)
        return bytes(raw)

    def test_switch(self):
        for build in ("switch", "x360"):
            order = order_of(build)
            raw = self.template(build)
            record = PP.new_sub_record(build, raw, 0xAAA, djb("Extra"))
            self.assertEqual(record.name, "SubObject")
            out = bytes(record.raw)
            self.assertEqual(struct.unpack_from(order + "I", out, 0)[0], 0x1055)  # content bits, 17..31 clear
            self.assertEqual(struct.unpack_from(order + "iifIII", out, 0x50), (-1, 7, 1.0, 0x05020006, 0, 0xAAA)[:4]
                             + struct.unpack_from(order + "II", out, 0x60))
            self.assertEqual(struct.unpack_from(order + "II", out, 0x64), (0xAAA, djb("Extra")))
            copied = {0x00, 0x50, 0x54, 0x58, 0x5C, 0x64, 0x68}
            for at in range(0, 240, 4):
                if at not in copied:
                    self.assertEqual(out[at:at + 4], bytes(4), hex(at))
            self.assertEqual(record.pairs, {})

    def test_ps3(self):
        raw = self.template("ps3")
        record = PP.new_sub_record("ps3", raw, 0xBBB, djb("Arm"), "Arm")
        out = bytes(record.raw)
        self.assertEqual(record.name, "SubObjectP3")
        self.assertEqual(struct.unpack_from(">I", out, 0)[0], 0x1055)
        self.assertEqual(struct.unpack_from(">iif", out, 0x3C), (-1, 7, 1.0))
        self.assertEqual(out[0x48:0xC8], b"Arm" + bytes(125))
        self.assertEqual(struct.unpack_from(">II", out, 0xC8), (0xBBB, djb("Arm")))
        for at in list(range(4, 0x3C, 4)) + list(range(0xD0, 308, 4)):
            self.assertEqual(out[at:at + 4], bytes(4), hex(at))
        with self.assertRaises(FormatError):
            PP.new_sub_record("ps3", raw, 1, 2, None)


# --- W17-5a --------------------------------------------------------------------------------------------

class Budget(unittest.TestCase):
    def test_limits(self):
        self.assertEqual(PP.draw_budget(0x81, 10, 30), 240)
        self.assertEqual(PP.draw_budget(0x83, 10, 30), 490)
        self.assertTrue(M.ps3_draw_fits(0x81, 2600, 1398))  # 21 x 2600 + 1398 = 55,998
        self.assertFalse(M.ps3_draw_fits(0x81, 2600, 1399))
        self.assertTrue(M.ps3_draw_fits(0x83, 1200, 2968, {0}))  # 46 x 1200 + 2968 = 58,168
        self.assertFalse(M.ps3_draw_fits(0x83, 1200, 2969, {0}))
        self.assertFalse(M.ps3_draw_fits(0x81, 100, 50000))  # within the budget, above the work buffer
        self.assertTrue(M.ps3_draw_fits(0x81, 100, 20000))

    def cut_and_check(self, draw, fmt):
        pieces = M.cut_ps3_draw(draw, fmt)
        self.assertGreater(len(pieces), 1)
        got = []
        for n, piece in enumerate(pieces):
            nv, ni = len(piece.positions), len(piece.indices)
            bones = set(j for k in range(nv) for j, w in zip(piece.joints[k], piece.weights[k]) if w) \
                if piece.skinned else set()
            self.assertTrue(M.ps3_draw_fits(fmt, nv, ni, bones))
            self.assertLessEqual(PP.draw_budget(fmt, nv, ni), PP.BUDGET[fmt][1])
            firsts = list(dict.fromkeys(piece.indices))
            self.assertEqual(firsts, list(range(nv)))  # renumbered by first use, every vertex used
            got += [tuple(piece.positions[i] for i in piece.indices[t:t + 3]) for t in range(0, ni, 3)]
            if n + 1 < len(pieces):  # the next triangle would not have fitted
                nxt = pieces[n + 1]
                extra = len(set(map(tuple, (nxt.positions[i] for i in nxt.indices[:3]))) - set(piece.positions))
                self.assertFalse(M.ps3_draw_fits(fmt, nv + extra, ni + 3, bones))
        want = [tuple(draw.positions[i] for i in draw.indices[t:t + 3]) for t in range(0, len(draw.indices), 3)]
        self.assertEqual(got, want)  # every triangle, in order

    def test_cut_rigid(self):
        positions, indices = grid(60)
        self.cut_and_check(rigid_draw(positions, indices, "ps3"), 0x81)

    def test_cut_skinned(self):
        positions, indices = grid(40)
        draw = skinned_draw(positions, indices, [(1, 2, 0, 0)] * len(positions), [(200, 55, 0, 0)] * len(positions))
        self.cut_and_check(draw, 0x83)

    def test_fits_whole(self):
        draw = rigid_draw(TRI, None, "ps3")
        self.assertEqual(M.cut_ps3_draw(draw), [draw])
        self.assertIs(M.cut_ps3_draw(draw)[0], draw)


# --- W17-6 ---------------------------------------------------------------------------------------------

class SharedBlocks(Checked):
    def test_rebuild_shares_equal_blocks(self):
        data = ps3(extra_draw_format=0x81)  # the builder never shares
        self.assertEqual(L.shared_pairs(L.parse_tree(data, "app", "ps3")), [])
        tree = L.parse_tree(M.rebuild_from_zero(P.parse_appearance(data, "ps3")), "app", "ps3")
        shared = sorted(path.rsplit(".", 1)[1] for path, _ in L.shared_pairs(tree))
        self.assertEqual(shared, ["serDescriptor", "serVertexFormatA", "serVertexFormatB"])
        fixed = canon(data, "ps3")
        self.assertEqual(len(L.shared_pairs(L.parse_tree(fixed, "app", "ps3"))), 3)
        self.assertEqual(M.rebuild_from_zero(P.parse_appearance(fixed, "ps3")), fixed)

    def test_writer_shares_a_new_draw(self):
        data, build = ps3_file()
        app = P.parse_appearance(data, build)
        sub = app.sub_objects[0]
        sub.draws.append(rigid_draw([(1, 1, 0), (5, 1, 0), (1, 5, 0)], None, "ps3"))  # 3 vertices, 3 indices
        out, report = M.write_app_mesh_report(app, derive=True)
        self.assertEqual(report.added, ["high-poly set, sub-object 0, draw 1"])
        tree = L.parse_tree(out, "app", build)
        first, second = elements(sub_records(tree)[0], 0x100)
        for offset in (0x08, 0x80, 0x90):
            self.assertIs(first.pairs[offset].block, second.pairs[offset].block)
        self.check(out, build)

    def test_unequal_blocks_not_shared(self):
        data, build = ps3_file()
        app = P.parse_appearance(data, build)
        app.sub_objects[0].draws.append(rigid_draw([(1, 1, 0), (5, 1, 0), (1, 5, 0), (5, 5, 0)], [0, 1, 2, 1, 3, 2],
                                                   "ps3"))
        tree = L.parse_tree(M.write_app_mesh(app, derive=True), "app", build)
        first, second = elements(sub_records(tree)[0], 0x100)
        self.assertIsNot(first.pairs[0x08].block, second.pairs[0x08].block)  # other counts
        self.assertIs(first.pairs[0x80].block, second.pairs[0x80].block)

    def test_function(self):
        tree = L.parse_tree(ps3(extra_draw_format=0x81), "app", "ps3")
        self.assertEqual(PP.share_ps3_blocks(tree), 3)
        self.assertEqual(PP.share_ps3_blocks(tree), 3)  # already shared: the same


# --- W17-7 ---------------------------------------------------------------------------------------------

class SkinRecords(Checked):
    def test_split_before_the_46th_bone(self):
        for order in ("<", ">"):
            data, build = multi_record(order)
            app = self.check(data, build)
            sub = app.sub_objects[1]
            self.assertEqual(len(sub.draws), 2)
            first, second = sub.draws
            self.assertEqual(first.palette, list(range(45)))  # first use: bone 0, then 1..44
            self.assertEqual(second.palette, [0] + list(range(45, 51)))
            self.assertEqual(len(first.indices), 44 * 3)
            self.assertEqual(second.indices[:3], [0, 1, 2])  # local indices
            self.assertEqual(first.positions[0], second.positions[0])  # vertex 0 written in both
            self.assertEqual(len(sub.influences), 101)  # W13-6: each distinct vertex once
            self.assertEqual(sub.flags & (1 << 15 | 1 << 16), 1 << 15 | 1 << 16)
            tree = L.parse_tree(data, "app", build)
            for skin in elements(sub_records(tree)[1], 0xB8):
                raw = bytes(skin.raw)
                self.assertEqual(raw[0x48:0x68], bytes(0x20))
                self.assertEqual(struct.unpack_from(order + "i", raw, 0)[0], len(block(skin, 0x04)) // 4)

    def test_removing_a_record_resplits(self):
        data, build = multi_record()
        app = P.parse_appearance(data, build)
        sub = app.sub_objects[1]
        sub.draws.pop(1)
        out, report = M.write_app_mesh_report(app, derive=True)
        self.assertEqual(report.removed, ["high-poly set, sub-object 1, skinning record 1"])
        again = self.check(out, build)
        self.assertEqual(len(again.sub_objects[1].draws), 1)  # 45 bones left: one record
        self.assertEqual(triangles(again.sub_objects[1]), triangles(app.sub_objects[1]))
        self.assertEqual(len(again.sub_objects[1].influences), 89)

    def test_adding_a_record(self):
        data, build = switch_skin()
        app = P.parse_appearance(data, build)
        app.sub_objects[0].draws.append(skinned_draw([(20, 0, 0), (30, 0, 0), (20, 10, 0)], [0, 1, 2],
                                                     [(0, 0, 0, 0)] * 3, [(255, 0, 0, 0)] * 3))
        out, report = M.write_app_mesh_report(app, derive=True)
        self.assertEqual(report.added, ["high-poly set, sub-object 0, skinning record 1"])
        again = self.check(out, build)
        self.assertEqual(len(again.sub_objects[0].draws), 1)  # 3 bones: one record
        self.assertEqual(again.sub_objects[0].draws[0].palette, [2, 1, 0])
        self.assertEqual(len(triangles(again.sub_objects[0])), 2)


# --- W17-8 ---------------------------------------------------------------------------------------------

def handles(app, material_hash):
    entry = app.material_data.find(material_hash)
    return [block.texture(1) for block, _ in entry.per_look]


class Materials(Checked):
    def setUp(self):
        data, self.build = switch_two()
        self.data = with_materials(data, self.build)

    def test_removed_entry(self):
        app = P.parse_appearance(self.data)
        self.assertEqual([m.hash for m in app.material_data.materials], [0xAAA, 0xBBB])
        out, report = M.write_app_mesh_report(app, edited(app, [app.sub_objects[0], app.sub_objects[2]]), derive=True)
        again = self.check(out)
        self.assertEqual([m.hash for m in again.material_data.materials], [0xAAA])
        self.assertIn("material 0x00000BBB", report.removed)
        self.assertEqual(handles(again, 0xAAA), [5, 6])

    def test_new_entry_from_the_same_file(self):
        app = P.parse_appearance(self.data)
        value = PP.new_material_hash(app, "newmat")
        self.assertEqual(value, djb("newmat"))
        part = new_sub("Extra", [rigid_draw([(1, 1, 0), (3, 1, 0), (1, 3, 0)])], material=value, material_source=0xBBB)
        out, report = M.write_app_mesh_report(app, edited(app, app.sub_objects + [part]), derive=True)
        again = self.check(out)
        self.assertEqual([m.hash for m in again.material_data.materials], [0xAAA, 0xBBB, value])  # appended
        self.assertEqual(handles(again, value), [7, 8])
        self.assertEqual(again.sub_objects[3].material_hash, value)
        self.assertIn(f"material 0x{value:08X}", report.added)

    def test_donor_with_another_look_count(self):
        data, build = switch_two()
        donor = P.parse_appearance(with_materials(data, build, looks=(0x61,), entries=[(0xCCC, [[(1, 9)]])],
                                                  assign=(0xCCC,)))
        app = P.parse_appearance(self.data)
        part = new_sub("Extra", [rigid_draw([(1, 1, 0), (3, 1, 0), (1, 3, 0)])], material=0xCCC,
                       material_source=(donor, 0xCCC))
        again = self.check(M.write_app_mesh(app, edited(app, app.sub_objects + [part]), derive=True))
        self.assertEqual(handles(again, 0xCCC), [9, 9])  # the first per-look record for every look
        part.material_source = PP.MaterialSource(0xDDD, donor)
        with self.assertRaisesRegex(FormatError, "not in the donor file's table"):
            M.write_app_mesh(app, edited(app, app.sub_objects + [part]), derive=True)

    def test_refusals(self):
        app = P.parse_appearance(self.data)
        part = new_sub("Extra", [rigid_draw(TRI)], material=0x1234)
        with self.assertRaisesRegex(FormatError, "^high-poly set, sub-object 3: material hash 0x00001234 has no "
                                                 "table entry and no source"):
            M.write_app_mesh(app, edited(app, app.sub_objects + [part]), derive=True)
        part.material_source = 0x999
        with self.assertRaisesRegex(FormatError, "not in the file's table"):
            M.write_app_mesh(app, edited(app, app.sub_objects + [part]), derive=True)
        for name in ("", ):
            with self.assertRaisesRegex(FormatError, "hash 0"):
                PP.new_material_hash(app, name)
        app.material_data.materials[0].hash = djb("taken")
        with self.assertRaisesRegex(FormatError, "already in the table"):
            PP.new_material_hash(app, "TAKEN")

    def test_stored_unused_entry_kept(self):
        data, build = switch_two()
        data = with_materials(data, build, entries=[(0xAAA, [[(1, 5)], [(1, 6)]]), (0xDDD, [[(1, 1)], [(1, 2)]]),
                                                    (0xBBB, [[(1, 7)], [(1, 8)]])])
        app = P.parse_appearance(data)
        self.assertEqual(M.write_app_mesh(app, derive=True), data)  # unchanged: as stored
        out = M.write_app_mesh(app, edited(app, app.sub_objects[:2]), derive=True)  # 0xAAA still used
        self.assertEqual([m.hash for m in P.parse_appearance(out).material_data.materials], [0xAAA, 0xDDD, 0xBBB])
        out = M.write_app_mesh(app, edited(app, [app.sub_objects[0]]), derive=True)
        self.assertEqual([m.hash for m in P.parse_appearance(out).material_data.materials], [0xAAA, 0xDDD])

    def test_material_of_a_kept_sub_object_changed(self):
        app = P.parse_appearance(self.data)
        app.sub_objects[1].material_hash = 0xAAA
        out, report = M.write_app_mesh_report(app, derive=True)
        again = self.check(out)
        self.assertEqual([m.hash for m in again.material_data.materials], [0xAAA])
        self.assertEqual(report.removed, ["material 0x00000BBB"])

    def test_ps3_table(self):
        data, build = ps3_file()
        data = canon(laid_out(B.set_sub_object_material(B.add_app_materials(
            data, "ps3", [0x61], [(0xAAA, [[(1, 5)]])]), "ps3", 0, 0xAAA, 0), "ps3"), "ps3")
        app = P.parse_appearance(data, build)
        part = new_sub("Extra", [rigid_draw(TRI, None, "ps3")], material=djb("m2"), material_source=0xAAA)
        again = self.check(M.write_app_mesh(app, edited(app, app.sub_objects + [part]), derive=True), build)
        self.assertEqual(handles(again, djb("m2")), [5])


# --- W17-10 --------------------------------------------------------------------------------------------

def stored_words(data, build, geo, index, offsets):
    record = sub_records(L.parse_tree(data, "app", build), geo)[index]
    return [struct.unpack_from(order_of(L.parse_tree(data, "app", build).build) + "I", record.raw, at)[0]
            for at in offsets]


def with_copied_fields(data, build, values):
    """Set physics, fade group, fade alpha and (switch) name info per sub-object of geo set 0."""
    tree = L.parse_tree(data, "app", build)
    order = order_of(tree.build)
    fields = PP.sub_fields(tree.build)
    for record, (physics, group, alpha, info, flags) in zip(sub_records(tree), values):
        struct.pack_into(order + "iif", record.raw, fields["physics"], physics, group, alpha)
        if tree.build != "ps3":
            struct.pack_into(order + "I", record.raw, fields["name_info"], info)
        struct.pack_into(order + "I", record.raw, 0, struct.unpack_from(order + "I", record.raw, 0)[0] | flags)
    return L.write_tree(tree)


class AddRemove(Checked):
    def test_add_rigid_sub_object(self):
        for order in ("<", ">"):
            data, build = switch_two(order)
            app = P.parse_appearance(data, build)
            part = new_sub("Extra", [rigid_draw([(20, 20, 4), (26, 20, 4), (20, 26, 5)])])
            out, report = M.write_app_mesh_report(app, edited(app, app.sub_objects + [part]), derive=True)
            self.assertEqual(report.added, ["high-poly set, sub-object 3"])
            again = self.check(out, build)
            added = again.sub_objects[3]
            self.assertEqual(triangles(added), [((20, 20, 4), (26, 20, 4), (20, 26, 5))])
            self.assertEqual(added.sub_hash, djb("Extra"))
            self.assertEqual(tuple(added.bias), (23.0, 23.0, 4.0))  # W2-2: round(box centre)
            self.assertEqual(tuple(added.bounds_centre), (23.0, 23.0, 4.5))
            self.assertEqual(added.flags, 0x54 | 1 | 1 << 15)  # copied from the first sub-object
            for k in range(3):
                self.assertEqual(triangles(again.sub_objects[k]), triangles(app.sub_objects[k]))

    def test_template_part_choice(self):
        data, build = switch_two()
        data = with_materials(data, build)
        data = canon(with_copied_fields(data, build, [(-1, 1, 0.0, 0x8, 0), (11, 2, 1.0, 0x02050003, 1 << 8),
                                                      (12, 3, 1.0, 0x9, 1 << 12)]), build)
        app = P.parse_appearance(data, build)
        cases = [(dict(template_part=(0, 2)), 0xAAA, (12, 3, 1.0, 0x9), 1 << 12),
                 (dict(), 0xBBB, (11, 2, 1.0, 0x02050003), 0),  # the first with the same material
                 (dict(), None, (-1, 1, 0.0, 0x8), 0)]  # no material: the file's first
        for options, material, words, extra_bit in cases:
            part = new_sub("Extra", [rigid_draw(TRI)], material=material, **options)
            out = M.write_app_mesh(app, edited(app, app.sub_objects + [part]), derive=True)
            again = self.check(out)
            record = sub_records(L.parse_tree(out, "app"))[3]
            self.assertEqual(struct.unpack_from("<iifI", record.raw, 0x50), words)
            self.assertEqual(again.sub_objects[3].material_hash, material if material is not None else 0xAAA)
            # bit 8 by content only (the template part of the second case stores it)
            self.assertEqual(again.sub_objects[3].flags, 0x54 | 1 | 1 << 15 | extra_bit)
        self.assertTrue(app.sub_objects[1].flags & 1 << 8)
        self.assertTrue(P.parse_appearance(out).sub_objects[1].flags & 1 << 8)  # kept on the existing part
        with self.assertRaisesRegex(FormatError, "template part"):
            M.write_app_mesh(app, edited(app, app.sub_objects + [new_sub("X", [rigid_draw(TRI)], template_part=(0, 9))]))

    def test_remove_first_and_last(self):
        for maker in (switch_two, lambda: switch_two(">"), lambda: switch_cloth_two()):
            data, build = maker()
            app = P.parse_appearance(data, build)
            for keep in ([1, 2], [0, 1], [1]):
                out, report = M.write_app_mesh_report(app, edited(app, [app.sub_objects[k] for k in keep]),
                                                      derive=True)
                again = self.check(out, build)
                self.assertEqual([triangles(s) for s in again.sub_objects],
                                 [triangles(app.sub_objects[k]) for k in keep])
                self.assertEqual(report.removed, [f"high-poly set, sub-object {k}" for k in range(3) if k not in keep])
                if maker is not switch_two:
                    continue
                self.assertEqual([s.sub_hash for s in again.sub_objects], [app.sub_objects[k].sub_hash for k in keep])

    def test_remove_ps3(self):
        data, build = ps3_file(extra_draw_format=0x81, second=[{"vertices": ((0, 0, 0), (2, 0, 0), (0, 2, 0))}])
        app = P.parse_appearance(data, build)
        for geo0, geo1 in (([], None), (None, [])):
            again = self.check(M.write_app_mesh(app, edited(app, geo0, geo1), derive=True), build)
            self.assertEqual(len(again.sub_objects) + len(again.extra_sub_objects), 1)
        with_fat, _ = ps3_file(fat=FAT, cloth=True)
        app = P.parse_appearance(with_fat, build)
        again = self.check(M.write_app_mesh(app, edited(app, []), derive=True), build)  # cloth removed whole
        self.assertEqual(again.sub_objects, [])
        tree = L.parse_tree(M.write_app_mesh(app, edited(app, []), derive=True), "app", build)
        self.assertIsNone(tree.top.pairs[0x228].block)  # the fat vertices went with it

    def test_add_ps3_sub_objects(self):
        data, build = ps3_file()
        app = P.parse_appearance(data, build)
        part = new_sub("Second", [rigid_draw([(0, 0, 1), (8, 0, 1), (0, 8, 1)], None, "ps3")])
        low = new_sub("Low2", [rigid_draw([(0, 0, 0), (2, 0, 0), (0, 2, 0)])], geo=1)
        out, report = M.write_app_mesh_report(app, edited(app, app.sub_objects + [part], [low]), derive=True)
        self.assertEqual(report.added, ["high-poly set, sub-object 1", "low-poly set, sub-object 0"])
        again = self.check(out, build)
        self.assertEqual([s.name for s in again.sub_objects + again.extra_sub_objects], ["Body", "Second", "Low2"])
        self.assertEqual(again.extra_sub_objects[0].sub_hash, djb("Low2"))
        self.assertEqual(triangles(again.sub_objects[1]), [((0, 0, 1), (8, 0, 1), (0, 8, 1))])

    def test_ps3_rigid_geometry_in_a_file_with_bones(self):
        data, build = ps3_file(skinned=True, bones=BONES)
        app = P.parse_appearance(data, build)
        part = new_sub("Rigid", [rigid_draw([(0, 0, 1), (8, 0, 1), (0, 8, 1)])])
        again = self.check(M.write_app_mesh(app, edited(app, app.sub_objects + [part]), derive=True), build)
        draw = again.sub_objects[1].draws[0]
        self.assertTrue(draw.skinned)  # W17-5: 0x83, one bone with weight 255
        self.assertEqual(draw.weights, [(255, 0, 0, 0)] * 3)
        self.assertEqual(draw.joints, [(0, 0, 0, 0)] * 3)

    def test_new_draw_cut_in_the_writer(self):
        data, build = ps3_file()
        app = P.parse_appearance(data, build)
        positions, indices = grid(60)
        part = new_sub("Grid", [rigid_draw(positions, indices, "ps3")])
        again = self.check(M.write_app_mesh(app, edited(app, app.sub_objects + [part]), derive=True), build)
        draws = again.sub_objects[1].draws
        self.assertGreater(len(draws), 1)
        for draw in draws:
            self.assertLessEqual(PP.draw_budget(0x81, len(draw.positions), len(draw.indices)), 55998)
        self.assertEqual(sum(len(d.indices) for d in draws), len(indices))
        tree = L.parse_tree(M.write_app_mesh(again, derive=True), "app", build)
        self.assertEqual(struct.unpack_from(">i", sub_records(tree)[1].raw, 0xFC)[0], len(draws))

    def test_octree_rebuilt_on_removal(self):
        flagged = 0x55 | 1 << 15 | O.FLAG_OCTREE
        data = canon(laid_out(B.build_app_switch(pool_spare=0, sub_objects=[
            {"rigid": [(0, 0, 0), (10, 0, 0), (0, 10, 0)], "index_count": 3, "flags": flagged},
            {"rigid": [(0, 0, 5), (4, 0, 5), (0, 4, 5)], "index_count": 3, "flags": flagged}])))
        app = P.parse_appearance(data)
        out = M.write_app_mesh(app, edited(app, [app.sub_objects[1]]), derive=True)
        again = self.check(out)
        tree = O.read_octree(L.parse_tree(out, "app"))
        self.assertEqual([u for u, _, _ in O.octree_triangles(again)], [0])
        self.assertEqual(O.octree_problems(again, tree), [])
        self.assertEqual(O.missing_primitives(tree, O.octree_triangles(again)), [])

    def test_geo_set_1_change_keeps_the_octree(self):
        data, build = ps3_file(second=[{"vertices": ((0, 0, 0), (2, 0, 0), (0, 2, 0))}])
        tree = L.parse_tree(data, "app", build)
        struct.pack_into(">6I", tree.top.raw, M.STRUCT_OCTREE, 1, 1, 2, 1, 1, 1)
        for offset, name in zip(M.OCTREE_PAIRS, ("OctreeNode", "OctreeLeaf", "OctreePrimitive")):
            element = L.new_record("ps3", "app", name)
            element.raw[:] = bytes([0x5A]) * len(element.raw)
            tree.top.pairs[offset] = L.PairValue(L.Block(elements=[element]))
        data = L.write_tree(tree)
        app = P.parse_appearance(data, build)
        out = M.write_app_mesh(app, edited(app, None, []), derive=True)
        self.assertEqual(O.read_octree(L.parse_tree(out, "app", build)).counts,
                         O.read_octree(tree).counts)  # kept: only a geo-set-1 part changed
        out = M.write_app_mesh(app, edited(app, [], None), derive=True)
        self.assertNotEqual(O.read_octree(L.parse_tree(out, "app", build)), O.read_octree(tree))

    def test_identity_refusals(self):
        data, build = switch_two()
        app = P.parse_appearance(data)
        subs = app.sub_objects
        cases = [
            ([subs[0], subs[0]], "two sub-objects come from template sub-object"),
            ([subs[1], subs[0]], "keep their order"),
            ([new_sub("A", [rigid_draw(TRI)]), subs[0]], "keep their order"),
            ([dataclasses.replace(subs[0], origin=(1, 0))], "geo set 1, not 0"),
            ([dataclasses.replace(subs[0], origin=(0, 7))], "not one of the 3 template"),
        ]
        for geo0, message in cases:
            with self.assertRaisesRegex(FormatError, message):
                M.write_app_mesh(app, edited(app, geo0), derive=True)
        twice = copy.deepcopy(subs[0])
        twice.draws.append(twice.draws[0])
        with self.assertRaisesRegex(FormatError, "second rigid list"):
            M.write_app_mesh(app, edited(app, [twice]), derive=True)

    def test_refusals(self):
        switch, _ = switch_two()
        skinned, _ = switch_skin()
        ps3_rigid, _ = ps3_file()
        ps3_bones_rigid, _ = ps3_file(bones=BONES)
        one = [(255, 0, 0, 0)] * 3
        cases = [
            (switch, None, new_sub("S", [skinned_draw(TRI, [0, 1, 2], [(0, 0, 0, 0)] * 3, one)]),
             "^high-poly set, sub-object 3: skinned geometry in a file without bones"),
            (skinned, None, new_sub("R", [rigid_draw(TRI)]),
             "^high-poly set, sub-object 1: rigid geometry in a switch/x360 file with bones"),
            (ps3_rigid, "ps3", new_sub("S", [skinned_draw(TRI, [0, 1, 2], [(0, 0, 0, 0)] * 3, one, "ps3")]),
             "skinned geometry in a file without bones"),
            (ps3_bones_rigid, "ps3", new_sub("S", [skinned_draw(TRI, [0, 1, 2], [(1, 0, 0, 0)] * 3, one, "ps3")]),
             "mixed ps3 file"),
            (switch, None, new_sub("Big", [rigid_draw([(k % 100, k // 100 % 100, k // 10000) for k in range(65537)],
                                                      [0, 1, 2])]), "65,536"),
            (ps3_rigid, "ps3", new_sub("body", [rigid_draw(TRI, None, "ps3")]), "duplicate name 'body'"),
            (ps3_rigid, "ps3", new_sub(None, [rigid_draw(TRI, None, "ps3")]), "needs a name"),
            (switch, None, new_sub(None, [rigid_draw(TRI)]), "needs a name or a sub-object hash"),
            (switch, None, new_sub(None, [rigid_draw(TRI)], sub_hash=0), "duplicate sub-object hash"),
            (switch, None, new_sub("C", [rigid_draw(TRI)], has_cloth=True), "only moved in whole"),
            (switch, None, new_sub("D", [rigid_draw(TRI, [0, 1])]), "not whole triangles"),
        ]
        for data, build, part, message in cases:
            app = P.parse_appearance(data, build)
            with self.assertRaisesRegex(FormatError, message):
                M.write_app_mesh(app, edited(app, app.sub_objects + [part]), derive=True)
        app = P.parse_appearance(skinned)
        app.sub_objects[0].draws.append(rigid_draw(TRI))
        with self.assertRaisesRegex(FormatError, "rigid geometry in a switch/x360 file with bones"):
            M.write_app_mesh(app, derive=True)

    def test_cloth_whole_or_unchanged(self):
        data, build = switch_cloth_two()
        app = P.parse_appearance(data, build)
        cloth = app.sub_objects[1]
        self.assertTrue(cloth.has_cloth)
        cloth.draws.append(rigid_draw(TRI))
        with self.assertRaisesRegex(FormatError, "^high-poly set, sub-object 1: a cloth sub-object.*V21-1"):
            M.write_app_mesh(app, derive=True)
        data, build = ps3_file(fat=FAT, cloth=True, cloth_draw=True)
        app = P.parse_appearance(data, build)
        app.sub_objects[0].draws.pop(0)
        with self.assertRaisesRegex(FormatError, "V21-1"):
            M.write_app_mesh(app, derive=True)

    def test_new_ps3_skinned_sub_object(self):
        data, build = ps3_file(skinned=True, bones=BONES)
        app = P.parse_appearance(data, build)
        joints = [(2, 0, 0, 0), (0, 2, 0, 0), (2, 0, 0, 0)]
        part = new_sub("Hand", [skinned_draw([(0, 0, 1), (8, 0, 1), (0, 8, 1)], [0, 1, 2], joints,
                                             [(255, 0, 0, 0), (155, 100, 0, 0), (255, 0, 0, 0)], "ps3")])
        out = M.write_app_mesh(app, edited(app, app.sub_objects + [part]), derive=True)
        again = self.check(out, build)
        draw = again.sub_objects[1].draws[0]
        self.assertEqual(draw.joints, joints)
        self.assertEqual(draw.palette, [0, 2])  # W13-4: two ranges split at the gap
        record = elements(sub_records(L.parse_tree(out, "app", build))[1], 0x100)[0]
        self.assertEqual(struct.unpack_from(">4H", record.raw, 0x60), (0, 96, 48, 48))
        self.assertEqual(again.sub_objects[1].flags & (1 << 13 | 1 << 14), 1 << 13 | 1 << 14)

    def test_draws_without_origin_match_by_position(self):
        data, build = switch_two()
        app = P.parse_appearance(data, build)
        fresh = rigid_draw([(0, 0, 0), (10, 0, 0), (0, 12, 0)])
        self.assertIsNone(fresh.origin)
        app.sub_objects[0].draws[0] = fresh
        out, report = M.write_app_mesh_report(app, derive=True)
        self.assertFalse(report)  # the same rigid list, changed: nothing added or removed
        again = self.check(out, build)
        self.assertEqual(triangles(again.sub_objects[0]), [((0, 0, 0), (10, 0, 0), (0, 12, 0))])
        app.sub_objects[0].draws[0].origin = 3
        with self.assertRaisesRegex(FormatError, "rigid list origin 3"):
            M.write_app_mesh(app, derive=True)

    def test_duplicate_a_sub_object(self):
        """As an editor duplicates a mesh: a copy marked new, its template part the original (W17-10)."""
        for data, build in (switch_two(), switch_skin(">"), ps3_file(skinned=True, bones=BONES), multi_record()):
            app = P.parse_appearance(data, build)
            original = app.sub_objects[-1]
            copy_ = copy.deepcopy(original)
            copy_.origin, copy_.template_part, copy_.name = None, original.origin, "Copy"
            out, report = M.write_app_mesh_report(app, edited(app, app.sub_objects + [copy_]), derive=True)
            again = self.check(out, build)
            self.assertEqual(report.added, [f"high-poly set, sub-object {len(app.sub_objects)}"])
            self.assertEqual(triangles(again.sub_objects[-1]), triangles(original))
            self.assertEqual(again.sub_objects[-1].sub_hash, djb("Copy"))
            self.assertEqual(again.sub_objects[-1].material_hash, original.material_hash)

    def test_empty_new_draw_not_written(self):
        data, build = ps3_file()
        app = P.parse_appearance(data, build)
        app.sub_objects[0].draws.append(rigid_draw([], [], "ps3"))
        out, report = M.write_app_mesh_report(app, derive=True)
        self.assertEqual(out, data)
        self.assertFalse(report)

    def test_without_w17_14_an_emptied_part_stays(self):
        data, build = switch_skin()
        app = P.parse_appearance(data, build)
        app.sub_objects[0].draws[0].indices.clear()
        again = P.parse_appearance(M.write_app_mesh(app, derive={"W13-7"}), build, strict=True)
        kept = again.sub_objects[0]
        self.assertEqual(kept.draws, [])
        self.assertEqual((tuple(kept.bias), tuple(kept.scale)), ((0.0,) * 3, (0.0,) * 3))  # W17-3
        self.assertEqual(kept.flags & (1 << 15 | 1 << 16), 0)  # W17-1

    def test_empty_new_part_skipped(self):
        data, build = switch_two()
        app = P.parse_appearance(data)
        out, report = M.write_app_mesh_report(app, edited(app, app.sub_objects + [new_sub("E", [])]), derive=True)
        self.assertEqual(out, data)
        self.assertEqual(report.skipped, ["high-poly set, sub-object 3"])
        self.assertEqual(report.added, [])

    def test_unchanged_mesh_unchanged_bytes(self):
        for data, build in (switch_two(), switch_two(">"), switch_skin(), ps3_file(), multi_record(),
                            ps3_file(fat=FAT, cloth=True)):
            app = P.parse_appearance(data, build)
            out, report = M.write_app_mesh_report(app, derive=True)
            self.assertEqual(out, data)
            self.assertFalse(report)


# --- W17-11 --------------------------------------------------------------------------------------------

class Transplant(Checked):
    def test_switch_rigid(self):
        target, build = switch_two()
        donor_data = canon(laid_out(B.build_app_switch(pool_spare=0, sub_objects=[
            {"rigid": [(0, 0, 0), (30, 0, 0), (0, 30, 2)], "index_count": 3, "flags": 0x55 | 1 << 15 | 1 << 12}])))
        tree = L.parse_tree(donor_data, "app")
        struct.pack_into("<I", sub_records(tree)[0].raw, 0x68, 0x4242)
        donor = P.parse_appearance(L.write_tree(tree))
        app = P.parse_appearance(target)
        mesh = M.transplant(app, donor, 0, 0)
        out, report = M.write_app_mesh_report(app, mesh, derive=True)
        self.assertEqual(report.added, ["high-poly set, sub-object 3"])
        again = self.check(out)
        moved = again.sub_objects[3]
        self.assertEqual(triangles(moved), triangles(donor.sub_objects[0]))
        self.assertEqual(moved.sub_hash, 0x4242)  # not taken: kept
        record = sub_records(L.parse_tree(out, "app"))[3]
        source = sub_records(L.parse_tree(bytes(donor.source), "app"))[0]
        self.assertEqual(bytes(record.raw[:0xB0]), bytes(source.raw[:0xB0]))  # every stored value
        self.assertEqual(block(record, 0xB0), block(source, 0xB0))
        # a taken hash: the exporter name with the smallest free number
        mesh = M.transplant(P.parse_appearance(out), donor, 0, 0)
        self.assertEqual(mesh.sub_objects[4].name, "subobject_0")
        again = self.check(M.write_app_mesh(P.parse_appearance(out), mesh, derive=True))
        self.assertEqual(again.sub_objects[4].sub_hash, djb("subobject_0"))
        mesh = M.transplant(app, donor, 0, 0, name="Prop")
        self.assertEqual(self.check(M.write_app_mesh(app, mesh, derive=True)).sub_objects[3].sub_hash, djb("Prop"))

    def test_ps3_skinned_names(self):
        target, build = ps3_file(skinned=True, bones=BONES)
        donor = P.parse_appearance(canon(ps3(skinned=True, bones=BONES, vertices=((0, 0, 0), (20, 0, 0), (0, 20, 0))),
                                         "ps3"), build)
        app = P.parse_appearance(target, build)
        mesh = M.transplant(app, donor, 0, 0)
        self.assertEqual(mesh.sub_objects[1].name, "Body_2")  # W17-12: taken
        again = self.check(M.write_app_mesh(app, mesh, derive=True), build)
        self.assertEqual([s.name for s in again.sub_objects], ["Body", "Body_2"])
        self.assertEqual(again.sub_objects[1].sub_hash, djb("Body_2"))
        self.assertEqual(triangles(again.sub_objects[1]), triangles(donor.sub_objects[0]))
        self.assertEqual(again.sub_objects[1].draws[0].weights, donor.sub_objects[0].draws[0].weights)
        mesh = M.transplant(app, donor, 0, 0, name="Arm")
        self.assertEqual(self.check(M.write_app_mesh(app, mesh, derive=True), build).sub_objects[1].name, "Arm")
        mesh = M.transplant(app, donor, 0, 0, name="BODY")
        with self.assertRaisesRegex(FormatError, "duplicate name 'BODY'"):
            M.write_app_mesh(app, mesh, derive=True)

    def test_cloth_moves_whole(self):
        target, build = switch_two()
        donor, _ = switch_cloth_two()
        donor = P.parse_appearance(donor)
        app = P.parse_appearance(target)
        out = M.write_app_mesh(app, M.transplant(app, donor, 0, 1), derive=True)
        again = self.check(out)
        moved = again.sub_objects[3]
        self.assertEqual(moved.cloth, donor.sub_objects[1].cloth)
        self.assertEqual(moved.fat_draw.positions, donor.sub_objects[1].fat_draw.positions)
        data, build = ps3_file(fat=FAT, cloth=True)
        donor = P.parse_appearance(data, build)
        target, _ = ps3_file()
        app = P.parse_appearance(target, build)
        again = self.check(M.write_app_mesh(app, M.transplant(app, donor, 0, 0, name="Cape"), derive=True), build)
        self.assertEqual(again.sub_objects[1].fat_draw.positions, donor.sub_objects[0].fat_draw.positions)
        self.assertEqual(again.sub_objects[1].cloth, donor.sub_objects[0].cloth)

    def test_material_comes_along(self):
        data, build = switch_two()
        target = P.parse_appearance(with_materials(data, build))
        donor = P.parse_appearance(with_materials(data, build, entries=[(0xEEE, [[(1, 3)], [(1, 4)]])],
                                                  assign=(0xEEE, 0xEEE, 0xEEE)))
        mesh = M.transplant(target, donor, 0, 2)
        out, report = M.write_app_mesh_report(target, mesh, derive=True)
        again = self.check(out)
        self.assertEqual(handles(again, 0xEEE), [3, 4])
        self.assertEqual(again.sub_objects[3].material_hash, 0xEEE)
        self.assertIn("material 0x00000EEE", report.added)

    def test_low_poly(self):
        data, build = ps3_file(second=[{"vertices": ((0, 0, 0), (2, 0, 0), (0, 2, 0))}])
        app = P.parse_appearance(data, build)
        mesh = M.transplant(app, app, 1, 0)
        self.assertEqual(mesh.extra_sub_objects[1].name, "Low_2")
        again = self.check(M.write_app_mesh(app, mesh, derive=True), build)
        self.assertEqual([s.name for s in again.extra_sub_objects], ["Low", "Low_2"])

    def test_refusals(self):
        switch, _ = switch_two()
        x360, _ = switch_two(">")
        skinned, _ = switch_skin()
        other = canon(laid_out(B.build_app_switch(bones=SKELETON_8, pool_spare=0,
                                                  sub_objects=[{"skin": SKIN2, "index_count": 0}])))
        ps3_bones_rigid, _ = ps3_file(bones=BONES)
        ps3_skinned, _ = ps3_file(skinned=True, bones=BONES)
        cases = [
            (P.parse_appearance(switch), P.parse_appearance(x360, "x360"), "a x360 sub-object for a switch file"),
            (P.parse_appearance(skinned), P.parse_appearance(other), "different skeletons"),
            (P.parse_appearance(switch), P.parse_appearance(skinned), "different skeletons"),
            (P.parse_appearance(ps3_bones_rigid, "ps3"), P.parse_appearance(ps3_skinned, "ps3"), "draw formats differ"),
        ]
        for target, source, message in cases:
            with self.assertRaisesRegex(FormatError, message):
                M.transplant(target, source, 0, 0)
        app = P.parse_appearance(switch)
        for geo, index in ((0, 3), (2, 0), (1, 0)):
            with self.assertRaises(FormatError):
                M.transplant(app, app, geo, index)


# --- W17-14 --------------------------------------------------------------------------------------------

class Removing(Checked):
    def test_ps3_draws(self):
        data, build = ps3_file(extra_draw_format=0x81)
        app = P.parse_appearance(data, build)
        sub = app.sub_objects[0]
        mesh = edited(app, [dataclasses.replace(sub, draws=[sub.draws[1]])])
        out, report = M.write_app_mesh_report(app, mesh, derive=True)
        self.assertEqual(report.removed, ["high-poly set, sub-object 0, draw 0"])
        again = self.check(out, build)
        self.assertEqual(len(again.sub_objects[0].draws), 1)
        out, report = M.write_app_mesh_report(app, edited(app, [dataclasses.replace(sub, draws=[])]), derive=True)
        self.assertEqual(report.removed, ["high-poly set, sub-object 0, draw 0", "high-poly set, sub-object 0, draw 1",
                                          "high-poly set, sub-object 0"])
        self.assertEqual(self.check(out, build).sub_objects, [])

    def test_ps3_split_a_draw(self):
        data, build = ps3_file()
        app = P.parse_appearance(data, build)
        sub = app.sub_objects[0]
        positions, indices = grid(2)
        draw = sub.draws[0]
        draw.positions[:] = [(float(x), float(y), 0.0) for x, y, _ in positions]
        for name in ("normals", "colours_1", "colours_2", "uv1", "uv2_raw"):
            getattr(draw, name)[:] = [getattr(draw, name)[0]] * len(positions)
        draw.indices[:] = indices[:12]
        second = rigid_draw(positions, indices[12:], "ps3")
        again = self.check(M.write_app_mesh(app, edited(app, [dataclasses.replace(sub, draws=[draw, second])]),
                                            derive=True), build)
        self.assertEqual(len(again.sub_objects[0].draws), 2)
        want = [tuple(tuple(float(c) for c in positions[i]) for i in indices[t:t + 3]) for t in range(0, 24, 3)]
        self.assertEqual(triangles(again.sub_objects[0]), want)

    def test_ps3_fat_sub_object_keeps_without_draws(self):
        """``38`` V23-5 deletes "unless it owns fat vertices": emptying the draws of a sub-object
        with fat vertices in place is refused; removing it whole is allowed and reported."""
        data, build = ps3_file(fat=FAT, cloth=False)
        app = P.parse_appearance(data, build)
        sub = app.sub_objects[0]
        self.assertEqual(len([d for d in sub.draws if d.kind == "ps3"]), 1)
        sub.draws[0].indices.clear()  # emptied in place (W13-7)
        with self.assertRaisesRegex(FormatError, r"sub-object 0, draw 0: a part of a cloth sub-object .*V23-5"):
            M.write_app_mesh(app, derive=True)

    def test_switch_last_record_and_rigid_list(self):
        for order in ("<", ">"):
            data, build = switch_skin(order)
            app = P.parse_appearance(data, build)
            sub = app.sub_objects[0]
            out, report = M.write_app_mesh_report(app, edited(app, [dataclasses.replace(sub, draws=[])]), derive=True)
            self.assertEqual(report.removed, ["high-poly set, sub-object 0, skinning record 0", "high-poly set, sub-object 0"])
            self.assertEqual(self.check(out, build).sub_objects, [])
            app.sub_objects[0].draws[0].indices.clear()  # emptied in place: the record and its sub-object go
            self.assertEqual(self.check(M.write_app_mesh(app, derive=True), build).sub_objects, [])
            self.assertEqual(len(P.parse_appearance(M.write_app_mesh(app, derive={"W13-7"}), build).sub_objects), 1)
            data, build = switch_two(order)
            app = P.parse_appearance(data, build)
            draw = app.sub_objects[1].draws[0]
            for name in ("positions", "normals", "colours_1", "colours_2", "uv1", "uv2_raw", "indices"):
                getattr(draw, name).clear()
            out, report = M.write_app_mesh_report(app, derive=True)
            # ``38`` V23-5: the emptied rigid list is reported before its sub-object
            self.assertEqual(report.removed, ["high-poly set, sub-object 1, rigid list", "high-poly set, sub-object 1"])
            again = self.check(out, build)
            self.assertEqual([triangles(s) for s in again.sub_objects],
                             [triangles(app.sub_objects[0]), triangles(app.sub_objects[2])])


# --- Check: every record from zero ---------------------------------------------------------------------

REBUILD = {
    "ps3 rigid": lambda: ps3_file(),
    "ps3 skinned": lambda: ps3_file(skinned=True, bones=BONES),
    "ps3 two draws, low poly": lambda: ps3_file(extra_draw_format=0x81,
                                                second=[{"vertices": ((0, 0, 0), (2, 0, 0), (0, 2, 0))}]),
    "ps3 two skinned draws": lambda: ps3_file(skinned=True, bones=BONES, extra_draw_format=0x83),
    "ps3 fat cloth": lambda: ps3_file(fat=FAT, cloth=True),
    "ps3 fat cloth with a draw": ps3_cloth_draw,
    "switch rigid": lambda: switch_two(),
    "switch skinned": lambda: switch_skin(),
    "switch multi-record": lambda: multi_record(),
    "switch cloth": lambda: switch_cloth_two(),
    "x360 rigid": lambda: switch_two(">"),
    "x360 skinned": lambda: switch_skin(">"),
    "x360 multi-record": lambda: multi_record(">"),
    "x360 cloth": lambda: switch_cloth_two(">"),
}


def with_shapes(data, build=None, count=2):
    """The file with ``count`` collision shapes on its first sub-object (their bytes kept, W2-15)."""
    tree = L.parse_tree(data, "app", build)
    record = sub_records(tree)[0]
    shapes = []
    for k in range(count):
        shape = L.new_record(tree.build, "app", "CollisionShape")
        shape.raw[:4] = bytes((k + 1, 2, 3, 4))
        shapes.append(shape)
    record.pairs[PP.sub_fields(tree.build)["shapes"]] = L.PairValue(L.Block(elements=shapes))
    struct.pack_into(order_of(tree.build) + "i", record.raw, PP.sub_fields(tree.build)["shape_count"], count)
    return L.write_tree(tree)


class Shapes(Checked):
    def test_shape_count(self):
        for maker in (switch_two, ps3_file, lambda: switch_two(">")):
            data, build = maker()
            data = with_shapes(data, build)
            app = P.parse_appearance(data, build)
            self.assertEqual(M.rebuild_from_zero(app), data)
            tree = L.parse_tree(data, "app", build)
            struct.pack_into(order_of(tree.build) + "i", sub_records(tree)[0].raw,
                             PP.sub_fields(tree.build)["shape_count"], 0)
            self.assertEqual(M.rebuild_from_zero(P.parse_appearance(L.write_tree(tree), build)), data)

    def test_shapes_travel_with_a_moved_part(self):
        data, build = switch_two()
        donor = P.parse_appearance(with_shapes(data, build, 1))
        app = P.parse_appearance(data)
        out = M.write_app_mesh(app, M.transplant(app, donor, 0, 0, name="Rock"), derive=True)
        self.check(out)
        record = sub_records(L.parse_tree(out, "app"))[3]
        self.assertEqual(struct.unpack_from("<i", record.raw, 0x84)[0], 1)
        self.assertEqual(bytes(elements(record, 0x88)[0].raw[:4]), bytes((1, 2, 3, 4)))


class RebuildFromZero(unittest.TestCase):
    def test_every_record_kind(self):
        for name, maker in REBUILD.items():
            data, build = maker()
            app = P.parse_appearance(data, build, strict=True)
            self.assertEqual(M.rebuild_from_zero(app), data, name)

    def corrupt(self, data, build, change):
        tree = L.parse_tree(data, "app", build)
        change(tree, order_of(tree.build))
        return L.write_tree(tree)

    def test_derived_fields_restored(self):
        def flags(tree, order):
            record = sub_records(tree)[0]
            struct.pack_into(order + "I", record.raw, 0, struct.unpack_from(order + "I", record.raw, 0)[0] ^ 1)

        def counts(tree, order):
            record = sub_records(tree)[0]
            at = PP.sub_fields(tree.build)["shape_count"]
            struct.pack_into(order + "i", record.raw, at, 3)
            struct.pack_into(order + "I", record.raw, 0x60 if tree.build != "ps3" else 0x130, 0xDEAD)

        def skin_words(tree, order):
            skin = elements(sub_records(tree)[0], 0xB8)[0]
            skin.raw[0x50] = 7
            skin.raw[0x0C] = 1

        def ps3_draw(tree, order):
            draw = elements(sub_records(tree)[0], 0x100)[0]
            struct.pack_into(">I", draw.raw, 0x98, 0x1000)
            struct.pack_into(">H", draw.raw, 0x66, 48)
            draw.raw[0x74] = 1

        def ps3_name_hash(tree, order):
            struct.pack_into(">I", sub_records(tree)[0].raw, 0xCC, 0x1234)

        def unshared(tree, order):
            first, second = elements(sub_records(tree)[0], 0x100)
            second.pairs[0x08] = L.PairValue(L.Block(bytes(first.pairs[0x08].block.data)))
            second.pairs[0x80] = L.PairValue(L.Block(bytes(first.pairs[0x80].block.data)))

        def scale(tree, order):
            struct.pack_into(order + "f", sub_records(tree)[0].raw, PP.sub_fields(tree.build)["scale"], 4.0)

        cases = [(switch_two, flags), (switch_two, counts), (switch_skin, skin_words), (lambda: switch_skin(">"), skin_words),
                 (lambda: ps3_file(skinned=True, bones=BONES), ps3_draw), (ps3_file, ps3_name_hash),
                 (lambda: ps3_file(extra_draw_format=0x81), unshared), (ps3_file, flags), (ps3_file, counts),
                 (switch_two, scale), (ps3_file, scale)]
        for maker, change in cases:
            data, build = maker()
            bad = self.corrupt(data, build, change)
            self.assertNotEqual(bad, data, change.__name__)
            self.assertEqual(M.rebuild_from_zero(P.parse_appearance(bad, build)), data, change.__name__)

    def test_copied_fields_kept(self):
        for maker in (switch_two, ps3_file, lambda: switch_two(">")):
            data, build = maker()

            def change(tree, order):
                record = sub_records(tree)[0]
                fields = PP.sub_fields(tree.build)
                struct.pack_into(order + "iif", record.raw, fields["physics"], 5, 6, 1.0)
                struct.pack_into(order + "I", record.raw, 0, struct.unpack_from(order + "I", record.raw, 0)[0] | 1 << 12)
                struct.pack_into(order + "I", record.raw, fields["material"], 0x31)
            changed = self.corrupt(data, build, change)
            self.assertEqual(M.rebuild_from_zero(P.parse_appearance(changed, build)), changed)


if __name__ == "__main__":
    unittest.main()
