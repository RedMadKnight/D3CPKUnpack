"""Stage 14, index files and new files: ``28_derived_values.md`` W14-1 to W14-6.

Synthetic data only: every file is built here from the described layout, read with the existing
parsers, written again and compared byte for byte."""

import _libpath  # noqa: F401  (python -I: the library folder on sys.path)
import struct
import unittest

from d3sno import archive as A
from d3sno import index as I
from d3sno import index_write as W
from d3sno import x360_archive as X
from d3sno.binary import FormatError
from d3sno.sno import hash33


# --- builders from the layout ------------------------------------------------------------------

def ps3_toc(rows):
    """rows: (group, handle, name) already sorted by handle."""
    out = struct.pack(">I", len(rows))
    for g, h, n in rows:
        out += struct.pack(">II", g, h) + n.encode("latin-1").ljust(128, b"\0")
    return out


def switch_toc(groups, order="<", thirds=None, last=None):
    """groups: {group: [(handle, name), ...]} in stored order."""
    counts, offsets = [0] * 70, [0] * 70
    body = b""
    for g in range(70):
        rows = groups.get(g, [])
        offsets[g] = 0 if g == 0 and not rows else len(body)
        if not rows:
            continue
        counts[g] = len(rows)
        recs, names = b"", b""
        for h, n in rows:
            recs += struct.pack(order + "III", g, h, len(names))
            names += n.encode() + b"\0"
        block = recs + names
        body += block + b"\0" * (-len(block) % 4)
    thirds = counts if thirds is None else thirds
    if last is None:
        last = max((h for rows in groups.values() for h, _ in rows), default=0)
    return struct.pack(order + "210I", *counts, *offsets, *thirds) + struct.pack(order + "I", last) + body


def prefetch_bytes(parents, deps, order):
    """parents: (group, handle, count, first) in list order; deps in stored order."""
    out = struct.pack(order + "I", len(parents))
    for p in parents:
        out += struct.pack(order + "4I", *p)
    out += struct.pack(order + "I", len(deps))
    for d in deps:
        out += struct.pack(order + "II", *d)
    return out


def sno_file(order="<", handle=0):
    return struct.pack(order + "4I", 0xDEADBEEF, 66, 0, 0) + struct.pack(order + "I", handle) + bytes(12)


# --- W14-1 -------------------------------------------------------------------------------------

class Identity(unittest.TestCase):
    def test_new_handle_over_all_tocs(self):
        a = I.parse_core_toc(ps3_toc([(44, 5, "a"), (33, 90, "b")]))
        b = I.parse_core_toc(switch_toc({44: [(7, "c"), (300, "d")]}))
        self.assertEqual(W.new_handle([a, b]), 301)
        self.assertEqual(W.new_handle(a), 91)
        self.assertEqual(W.new_handle([]), 1)

    def test_new_handle_counts_stored_word(self):
        # x360: the header word is the highest handle over six files, may exceed this file's
        toc = I.parse_core_toc(switch_toc({44: [(7, "c")]}, ">", last=5000), "x360")
        self.assertEqual(W.new_handle(toc), 5001)

    def test_identity_path_entry_and_stamp(self):
        ident = W.sno_identity(44, "MyTex", 1234)
        self.assertEqual(ident.path, "Textures\\MyTex.tex")
        self.assertEqual(ident.entry, I.TocEntry(44, 1234, "MyTex"))
        self.assertEqual(W.sno_identity(42, "Lines", 7, locale="enUS").path, "StringList\\Lines.stl|enUS")
        for order in "<>":
            out = ident.stamp(sno_file(order, 0))
            self.assertEqual(struct.unpack_from(order + "I", out, 0x10)[0], 1234)
            self.assertEqual(out[:0x10], sno_file(order)[:0x10])
            self.assertEqual(out[0x14:], sno_file(order)[0x14:])

    def test_identity_refusals(self):
        with self.assertRaises(FormatError):
            W.sno_identity(16, "x", 1)  # no group 16
        with self.assertRaises(FormatError):
            W.sno_identity(44, "a\\b", 1)
        with self.assertRaises(FormatError):
            W.sno_identity(44, "x" * 129, 1)
        with self.assertRaises(FormatError):
            W.sno_identity(44, "x", 1).stamp(b"\0" * 32)


# --- W14-2 -------------------------------------------------------------------------------------

class CoreTocPs3(unittest.TestCase):
    ROWS = [(44, 3, "tex"), (1, 10, "actor"), (33, 11, "x" * 128), (44, 40000, "big")]

    def test_round_trip(self):
        data = ps3_toc(self.ROWS)
        toc = I.parse_core_toc(data)
        self.assertEqual(toc.build, "ps3")
        self.assertEqual(W.write_core_toc(toc), data)
        self.assertEqual(W.write_core_toc(toc.entries, "ps3"), data)

    def test_sorted_by_handle(self):
        shuffled = [I.TocEntry(*r) for r in reversed(self.ROWS)]
        self.assertEqual(W.write_core_toc(shuffled), ps3_toc(self.ROWS))

    def test_add_inserts_by_handle(self):
        toc = I.parse_core_toc(ps3_toc(self.ROWS))
        W.add_toc_entry(toc, 9, 12, "new")
        self.assertEqual([e.handle for e in toc.entries], [3, 10, 11, 12, 40000])
        rows = self.ROWS[:3] + [(9, 12, "new")] + self.ROWS[3:]
        self.assertEqual(W.write_core_toc(toc), ps3_toc(rows))

    def test_refusals(self):
        toc = I.parse_core_toc(ps3_toc(self.ROWS))
        with self.assertRaises(FormatError):
            W.add_toc_entry(toc, 1, 10, "dup")  # unique over all groups
        with self.assertRaises(FormatError):
            W.write_core_toc([I.TocEntry(1, 5, "a"), I.TocEntry(2, 5, "b")])
        with self.assertRaises(FormatError):
            W.write_core_toc([I.TocEntry(1, 5, "x" * 129)])


# --- W14-3 -------------------------------------------------------------------------------------

GROUPS = {0: [(2, "zero")], 1: [(5, "a"), (9, "bc")], 44: [(3, "tex1"), (100, "t")], 69: [(77, "last!")]}


class CoreTocSwitch(unittest.TestCase):
    def test_round_trip_switch(self):
        data = switch_toc(GROUPS)
        toc = I.parse_core_toc(data, "switch")
        self.assertEqual(W.write_core_toc(toc), data)
        self.assertEqual(W.write_core_toc(toc.entries, "switch"), data)

    def test_empty_group_offsets(self):
        data = switch_toc({3: [(1, "abc")], 5: [(2, "d")]})
        offsets = struct.unpack_from("<70I", data, 280)
        self.assertEqual(offsets[0], 0)
        self.assertEqual(offsets[1], 0)
        self.assertEqual((offsets[3], offsets[4], offsets[5], offsets[6], offsets[69]), (0, 16, 16, 32, 32))
        self.assertEqual(W.write_core_toc(I.parse_core_toc(data, "switch")), data)

    def test_group_zero_empty_stores_zero_and_trailing_offsets(self):
        out = W.write_core_toc([I.TocEntry(2, 9, "ab")], "switch")
        offsets = struct.unpack_from("<70I", out, 280)
        self.assertEqual(offsets[:3], (0, 0, 0))
        self.assertEqual(set(offsets[3:]), {16})  # 12-byte entry + "ab\0" -> 15 -> 16
        self.assertEqual(len(out), 844 + 16)

    def test_sorted_inside_group_and_header_words(self):
        entries = [I.TocEntry(1, 9, "bc"), I.TocEntry(44, 100, "t"), I.TocEntry(1, 5, "a"),
                   I.TocEntry(69, 77, "last!"), I.TocEntry(44, 3, "tex1"), I.TocEntry(0, 2, "zero")]
        out = W.write_core_toc(entries, "switch")
        self.assertEqual(out, switch_toc(GROUPS))
        self.assertEqual(struct.unpack_from("<I", out, 840)[0], 100)
        self.assertEqual(struct.unpack_from("<70I", out, 560), struct.unpack_from("<70I", out, 0))

    def test_add_switch(self):
        toc = I.parse_core_toc(switch_toc(GROUPS), "switch")
        W.add_toc_entry(toc, 44, 50, "mid")
        W.add_toc_entry(toc, 2, 500, "newgroup")
        expected = dict(GROUPS)
        expected[44] = [(3, "tex1"), (50, "mid"), (100, "t")]
        expected[2] = [(500, "newgroup")]
        self.assertEqual(W.write_core_toc(toc), switch_toc(expected))
        self.assertEqual(I.parse_core_toc(W.write_core_toc(toc), "switch").entries, toc.entries)
        with self.assertRaises(FormatError):
            W.add_toc_entry(toc, 3, 50, "dup")

    def test_x360_round_trip_with_stored_totals(self):
        thirds = [0] * 70
        thirds[1], thirds[44], thirds[0], thirds[69], thirds[5] = 3, 2, 1, 1, 4
        data = switch_toc(GROUPS, ">", thirds=thirds, last=9999)
        toc = I.parse_core_toc(data, "x360")
        self.assertEqual(W.write_core_toc(toc), data)
        self.assertEqual(W.write_core_toc(toc.entries, "x360", totals=thirds, top=9999), data)

    def test_x360_title_update_keeps_order(self):
        tu = {44: [(100, "t"), (3, "tex1")]}
        data = switch_toc(tu, ">", last=100)
        toc = I.parse_core_toc(data, "x360")
        self.assertEqual(W.write_core_toc(toc.entries, "x360", keep_order=True), data)
        W.add_toc_entry(toc, 44, 50, "app", keep_order=True)
        self.assertEqual(W.write_core_toc(toc.entries, "x360", keep_order=True),
                         switch_toc({44: tu[44] + [(50, "app")]}, ">", last=100))

    def test_write_x360_tocs(self):
        common = {1: [(5, "a")], 44: [(3, "tex1")]}
        act1 = {1: [(5, "a"), (9, "b")], 44: [(700, "x")]}
        act2 = {69: [(8, "z")]}
        tu = {44: [(900, "u"), (800, "v")]}
        thirds = [0] * 70
        thirds[1], thirds[44], thirds[69] = 2, 2, 1
        before = {
            "CommonTOC.dat": switch_toc(common, ">", thirds, 700),
            "Act1TOC.dat": switch_toc(act1, ">", thirds, 700),
            "Act2TOC.dat": switch_toc(act2, ">", thirds, 700),
            "Update\\CoreTOC.dat": switch_toc(tu, ">", None, 900),
        }
        files = {n: I.parse_core_toc(d, "x360") for n, d in before.items()}
        out = W.write_x360_tocs(files)
        self.assertEqual(out, before)
        self.assertEqual(I.toc_totals_problems([I.parse_core_toc(out[n], "x360") for n in
                                                ("CommonTOC.dat", "Act1TOC.dat", "Act2TOC.dat")]), [])
        # a new entry changes the third values and the last word of every main file
        W.add_toc_entry(files["Act2TOC.dat"], 44, 1000, "new")
        out = W.write_x360_tocs(files)
        thirds[44] = 3
        self.assertEqual(out["CommonTOC.dat"], switch_toc(common, ">", thirds, 1000))
        self.assertEqual(out["Act1TOC.dat"], switch_toc(act1, ">", thirds, 1000))
        self.assertEqual(out["Act2TOC.dat"], switch_toc({44: [(1000, "new")], 69: [(8, "z")]}, ">", thirds, 1000))
        self.assertEqual(out["Update\\CoreTOC.dat"], before["Update\\CoreTOC.dat"])

    def test_refusals(self):
        with self.assertRaises(FormatError):
            W.write_core_toc([I.TocEntry(70, 1, "a")], "switch")
        with self.assertRaises(FormatError):
            W.write_core_toc([I.TocEntry(1, 1, "a"), I.TocEntry(1, 1, "b")], "switch")
        with self.assertRaises(FormatError):
            W.write_core_toc([I.TocEntry(1, 1, "a")], "ps3", top=5)


# --- W14-4 -------------------------------------------------------------------------------------

class PrefetchWriter(unittest.TestCase):
    # handle -> (group, deps)
    DATA = {0x101: (1, [(44, 1), (44, 2)]), 0x201: (2, [(9, 9)]), 0x10002: (1, [(1, 1)] * 3),
            0x0102: (3, []), 0x20002: (44, [(5, 5)])}

    def items(self):
        return [(g, h, d) for h, (g, d) in self.DATA.items()]

    def expected(self, listed, blocks, order):
        first, deps = {}, []
        for h in blocks:
            first[h] = len(deps)
            deps += self.DATA[h][1]
        parents = [(self.DATA[h][0], h, len(self.DATA[h][1]), first[h]) for h in listed]
        return prefetch_bytes(parents, deps, order)

    def test_ps3_order(self):
        # (handle & 0xFF), then group, then descending handle; blocks by handle
        listed = sorted(self.DATA, key=lambda h: (h & 0xFF, self.DATA[h][0], -h))
        data = self.expected(listed, sorted(self.DATA), ">")
        self.assertEqual(W.write_prefetch(self.items(), "ps3"), data)
        self.assertEqual(W.write_prefetch(I.parse_prefetch(data, "ps3")), data)
        self.assertEqual([p.handle for p in I.parse_prefetch(data, "ps3").parents][:2], [0x101, 0x201])

    def test_switch_order(self):
        listed = sorted(self.DATA, key=lambda h: (h & 0xFFFF, -h))
        self.assertEqual(listed, [0x20002, 0x10002, 0x101, 0x0102, 0x201])
        data = self.expected(listed, sorted(self.DATA), "<")
        self.assertEqual(W.write_prefetch(self.items(), "switch"), data)
        parsed = I.parse_prefetch(data, "switch")
        self.assertTrue(parsed.parents_sorted)
        self.assertEqual(W.write_prefetch(parsed), data)

    def test_x360_bucket_order_and_blocks_in_list_order(self):
        listed = sorted(self.DATA, key=W.prefetch_bucket)
        data = self.expected(listed, listed, ">")
        self.assertEqual(W.write_prefetch(self.items(), "x360"), data)
        self.assertEqual(W.write_prefetch(I.parse_prefetch(data, "x360")), data)
        self.assertEqual(W.write_prefetch(I.parse_prefetch(data, "ps3"), "x360"), data)

    def test_fnv_bucket(self):
        value = 0x811C9DC5
        for byte in (0x01, 0x01, 0x00, 0x00):
            value = ((value ^ byte) * 0x01000193) & 0xFFFFFFFF
        self.assertEqual(W.prefetch_bucket(0x101), value & 0x3FFF)

    def test_x360_title_update_keeps_order(self):
        listed = [0x20002, 0x101, 0x0102, 0x201, 0x10002]
        data = self.expected(listed, listed, ">")
        parsed = I.parse_prefetch(data, "x360")
        self.assertEqual(W.write_prefetch(parsed, keep_order=True), data)
        W.add_prefetch_parent(parsed, 7, 0x55, [(1, 2)], keep_order=True)
        self.assertEqual([p.handle for p in parsed.parents], listed + [0x55])
        self.assertEqual(parsed.dependencies(parsed.parents[-1]), [(1, 2)])

    def test_add_parent_switch_and_ps3(self):
        for build in ("switch", "ps3"):
            parsed = I.parse_prefetch(W.write_prefetch(self.items(), build), build)
            W.add_prefetch_parent(parsed, 5, 0x180, [(44, 7), (44, 8)])
            items = self.items() + [(5, 0x180, [(44, 7), (44, 8)])]
            self.assertEqual(W.write_prefetch(parsed), W.write_prefetch(items, build))
            reread = I.parse_prefetch(W.write_prefetch(parsed), build)
            self.assertEqual(reread, parsed)
            new = next(p for p in reread.parents if p.handle == 0x180)
            self.assertEqual(reread.dependencies(new), [(44, 7), (44, 8)])
            with self.assertRaises(FormatError):
                W.add_prefetch_parent(parsed, 5, 0x180, [])

    def test_add_parent_x360_goes_to_end_of_bucket(self):
        parsed = I.parse_prefetch(W.write_prefetch(self.items(), "x360"), "x360")
        handle = next(h for h in range(0x1000, 0x100000)
                      if W.prefetch_bucket(h) == W.prefetch_bucket(0x101) and h != 0x101)
        W.add_prefetch_parent(parsed, 9, handle, [(1, 1)])
        handles = [p.handle for p in parsed.parents]
        self.assertEqual(handles.index(handle), handles.index(0x101) + 1)
        buckets = [W.prefetch_bucket(h) for h in handles]
        self.assertEqual(buckets, sorted(buckets))

    def test_refusals(self):
        with self.assertRaises(FormatError):
            W.write_prefetch(self.items())  # no build
        with self.assertRaises(FormatError):
            W.write_prefetch(self.items(), "switch", keep_order=True)
        with self.assertRaises(FormatError):
            W.write_prefetch([(1, 2, []), (1, 2, [])], "ps3")


# --- W14-5 -------------------------------------------------------------------------------------

class SmallTables(unittest.TestCase):
    def test_manifest_round_trip_keeps_order(self):
        for order in "<>":
            data = struct.pack(order + "I", 0xAABB0004) + b"".join(
                struct.pack(order + "II", g, h) for g, h in [(44, 9), (1, 3), (44, 2)])
            m = I.parse_manifest(data)
            self.assertEqual(W.write_manifest(m), data)
            self.assertEqual(W.write_manifest(m.entries, order), data)
        self.assertEqual(W.write_manifest([(1, 2)], "x360"), bytes.fromhex("AABB0004 00000001 00000002"))
        with self.assertRaises(FormatError):
            W.write_manifest([(1, 2)])

    def test_sprite_table_round_trip_and_sort(self):
        rows = [(hash33("a"), 5, 0), (hash33("b"), 5, 1), (hash33("c"), 9, 0), (77, 70000, 0)]
        for order in "<>":
            data = struct.pack(order + "I", 0xBBCC0005) + b"".join(struct.pack(order + "III", *r) for r in rows)
            table = I.parse_sprite_table(data)
            self.assertEqual(table.problems(), [])
            self.assertEqual(W.write_sprite_table(table), data)
            shuffled = [rows[3], rows[1], ("c", 9, 0), rows[0]]
            self.assertEqual(W.write_sprite_table(shuffled, order), data)

    def test_descriptor_sizes_round_trip(self):
        recs = [("SNOActor", 0x420, list(range(256))), ("x" * 64, 7, [0xFFFFFFFF] * 256)]
        for order, build in (("<", "switch"), (">", "x360")):
            data = struct.pack(order + "I", 0xAABBCCDD) + b"".join(
                n.encode().ljust(64, b"\0") + struct.pack(order + "I", s) + struct.pack(order + "256I", *w)
                for n, s, w in recs)
            parsed = I.parse_descriptor_sizes(data)
            self.assertEqual(W.write_descriptor_sizes(parsed, build), data)
            self.assertEqual(W.write_descriptor_sizes(parsed, order), data)
        with self.assertRaises(FormatError):
            W.write_descriptor_sizes([I.DescriptorSize("a", 1, (0,) * 255)])

    def test_slo(self):
        self.assertEqual(W.write_slo(), b"\xAC\xBF\x00\x04")


# --- W14-6 -------------------------------------------------------------------------------------

FILES = {
    "Textures\\a.tex": bytes(range(256)) * 40,
    "Actor\\b.acr": b"actor" * 1000,
    "StringList\\c.stl|enUS": b"hello" * 3,
}


def contents(data):
    cpk = A.read_archive(data)
    return {e.name: [cpk.read_file(e, k) for k in range(e.count)] for e in cpk.files}


class Archives(unittest.TestCase):
    def sources(self):
        yield "ps3", A.write_archive(FILES, version=6)
        yield "ps3", A.write_archive(
            [A.InputFile(n, [d, d] if n.startswith("Actor") else [d]) for n, d in FILES.items()], version=6)
        yield "switch", A.write_archive(FILES, version=7)  # v7 Switch archives hold one copy per file
        pieces = [X.Piece(n, 0, d) for n, d in FILES.items()] + [X.Piece("Actor\\b.acr", 1, FILES["Actor\\b.acr"])]
        yield "x360", X.write_x360_archive(pieces).data

    def test_add_file(self):
        new = b"new file" * 5000
        for build, data in self.sources():
            out = W.add_file(data, "Textures\\new.tex", new)
            self.assertEqual(A.read_archive(out).header.build, build)
            self.assertTrue(A.verify_archive(out).ok, build)
            got, before = contents(out), contents(data)
            self.assertEqual(got.pop("Textures\\new.tex"), [new])
            self.assertEqual(got, before, build)
            with self.assertRaises(FormatError):
                W.add_file(data, "Actor\\b.acr", b"x")

    def test_replace_file_grows_every_copy(self):
        grown = b"grown" * 9000
        for build, data in self.sources():
            out = W.replace_file(A.read_archive(data), "Actor\\b.acr", grown)
            self.assertTrue(A.verify_archive(out).ok, build)
            got, before = contents(out), contents(data)
            self.assertEqual(got["Actor\\b.acr"], [grown] * len(before["Actor\\b.acr"]))
            del got["Actor\\b.acr"], before["Actor\\b.acr"]
            self.assertEqual(got, before, build)
            with self.assertRaises(FormatError):
                W.replace_file(data, "Actor\\missing.acr", b"x")

    def test_identity_into_archive_and_toc(self):
        data = dict(self.sources())["switch"]
        toc = I.parse_core_toc(switch_toc({44: [(3, "a")], 1: [(4, "b")]}), "switch")
        handle = W.new_handle(toc)
        ident = W.sno_identity(44, "new", handle)
        out = W.add_file(data, ident.path, ident.stamp(sno_file("<")))
        W.add_toc_entry(toc, ident.group, ident.handle, ident.name)
        cpk = A.read_archive(out)
        self.assertEqual(struct.unpack_from("<I", cpk.read_file(ident.path), 0x10)[0], handle)
        reread = I.parse_core_toc(W.write_core_toc(toc), "switch")
        self.assertIn(ident.entry, reread.entries)
        self.assertEqual(reread.unknown, handle)


if __name__ == "__main__":
    unittest.main()
