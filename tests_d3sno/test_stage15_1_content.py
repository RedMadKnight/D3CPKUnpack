"""Stage 15.1, content: ``31_corrections18.md`` V18-1, V18-6, V18-7, V18-8, V18-9.
Synthetic data only."""

import _libpath  # noqa: F401  (python -I: the library folder on sys.path)
import struct
import unittest

from d3sno import FormatError
from d3sno import material as M
from d3sno import material_write as MW
from d3sno import presentation as P
from d3sno import shader as SH
from d3sno import sno_layout as T
from d3sno import sound_write as W
from d3sno.index import CoreToc, TocEntry

from test_stage15_content import BUILDS, colours, shader_file

BLOB_A = struct.pack("<I2f", 1, 2.0, 3.0)  # 12 bytes
BLOB_B = struct.pack("<I4f", 2, 1.0, 1.0, 0.5, 0.25)  # 20 bytes


# --- V18-1 -------------------------------------------------------------------------------------

def x360_record(h, n=3, a=8, b=16, extra=8):
    return struct.pack(">4I", h, n, a, b) + bytes(a + b + extra)


class X360ShaderHead(unittest.TestCase):
    def test_h_one_is_read(self):
        raws = [x360_record(1), x360_record(0)]
        block = SH.shader_block(raws, "x360")
        pairs = P.parse_shader_block(block, 1, "x360")
        self.assertEqual(pairs[0][0].raw, raws[0])
        found = SH.shader_records(block, 1, "x360")
        self.assertEqual([p.head[0] for p in found], [1, 0])
        self.assertEqual(SH.x360_head_problems(found), [])

    def test_head_rules(self):
        for bad in (x360_record(2), x360_record(1, a=6, b=2, extra=8), x360_record(0, a=4, extra=4),
                    struct.pack(">4I", 0, 0, 64, 0) + bytes(8)):
            with self.subTest(bad[:16].hex()):
                with self.assertRaises(FormatError):
                    P.parse_shader_block(SH.shader_block([bad, x360_record(0)], "x360"), 1, "x360")

    def test_pixel_h_one_reported_not_refused(self):
        found = SH.shader_records(SH.shader_block([x360_record(0), x360_record(1)], "x360"), 1, "x360")
        problems = SH.x360_head_problems(found)
        self.assertEqual(len(problems), 1)
        self.assertIn("pixel", problems[0])

    def _with_h(self, data, values):
        """The .shd ``data`` (x360) with head word h of each record set to ``values``."""
        shader = SH.read_shader(data, "x360")
        out = bytearray(data)
        start = len(data) - len(shader.block)
        for program, h in zip(shader.programs, values):
            struct.pack_into(">I", out, start + len(shader.passes) * 16 + program.offset, h)
        return bytes(out)

    def test_file_keeps_h(self):
        data = self._with_h(shader_file("x360", [[1], [2]], [1, 2, 3, 4]), [1, 0, 1, 0])
        shader = SH.read_shader(data, "x360")
        self.assertEqual([p.head[0] for p in shader.programs], [1, 0, 1, 0])
        self.assertEqual(shader.problems, [])
        self.assertEqual(SH.write_shader(shader), data)
        pixel = self._with_h(data, [1, 1, 1, 0])
        self.assertEqual(SH.write_shader(SH.read_shader(pixel, "x360")), pixel)
        self.assertEqual(len(SH.read_shader(pixel, "x360").problems), 1)

    def test_replacement_copies_donor_h(self):
        target = SH.read_shader(shader_file("x360", [[1], [2]], [1, 1, 1, 1]), "x360")
        donor_data = self._with_h(shader_file("x360", [[4], [6]], [3, 2, 5, 4], fill=9, key_base=500),
                                  [1, 0, 0, 0])
        donor = SH.read_shader(donor_data, "x360")
        out = SH.replace_shader_programs(target, donor)
        self.assertTrue(out.endswith(donor.block))
        self.assertEqual([p.head for p in target.programs], [p.head for p in donor.programs])
        self.assertEqual(target.programs[0].head[0], 1)


# --- V18-6 -------------------------------------------------------------------------------------

def pairs_of(data, build):
    """(offset, size) of every texture entry's animation pair, from the written file."""
    parsed = M.parse_material(data, build)
    order = "<" if build == "switch" else ">"
    anim_at = 0x08 if build != "ps3" else 0x04
    handle_at = 0x18 if build != "ps3" else 0x14
    return [struct.unpack_from(order + "II", data, t.position - handle_at + anim_at)
            for t in parsed.block.textures]


class SharedAnimationBlobs(unittest.TestCase):
    ENTRIES = [(1, 10, BLOB_A), (2, 11, BLOB_B), (3, 12, BLOB_A), (4, 13, b""), (5, 14, b""),
               (6, 15, BLOB_B)]

    def test_new_material_shares(self):
        for build in BUILDS:
            with self.subTest(build):
                data = T.write_tree(MW.new_material(build, 70, colours(build), 1.0, 0, self.ENTRIES))
                pairs = pairs_of(data, build)
                self.assertEqual(pairs[0], pairs[2])
                self.assertEqual(pairs[1], pairs[5])
                self.assertNotEqual(pairs[0][0], pairs[1][0])
                self.assertEqual(pairs[3], (0, 0))  # an empty blob: {0, 0}, never shared
                self.assertEqual(pairs[4], (0, 0))
                read = M.parse_material(data, build)
                self.assertEqual([t.animation_raw for t in read.block.textures],
                                 [e[2] for e in self.ENTRIES])
                unshared = T.write_tree(MW.new_material(build, 70, colours(build), 1.0, 0, self.ENTRIES,
                                                        share_animations=False))
                self.assertGreaterEqual(len(unshared), len(data) + len(BLOB_A) + len(BLOB_B))
                self.assertNotEqual(pairs_of(unshared, build)[0], pairs_of(unshared, build)[2])

    def test_rebuilt_from_fields_equals_file(self):
        for build in BUILDS:
            with self.subTest(build):
                data = T.write_tree(MW.new_material(build, 70, colours(build), 1.0, 0, self.ENTRIES,
                                                    handle=5))
                tree = MW.material_tree(data, build)
                self.assertEqual(T.write_tree(tree), data)
                again = MW.new_material(build, 70, colours(build), 1.0, 0, MW.material_textures(tree),
                                        handle=5)
                self.assertEqual(T.write_tree(again), data)

    def test_add_texture_reuses(self):
        for build in BUILDS:
            with self.subTest(build):
                data = T.write_tree(MW.new_material(build, 70, colours(build), 1.0, 0,
                                                    [(1, 10, BLOB_A), (2, 11, b"")]))
                tree = MW.material_tree(data, build)
                MW.add_texture(tree, 3, 12, BLOB_A)
                out = T.write_tree(tree)
                pairs = pairs_of(out, build)
                self.assertEqual(pairs[2], pairs[0])
                MW.add_texture(tree, 4, 13, BLOB_B)
                MW.add_texture(tree, 5, 14, BLOB_B)
                pairs = pairs_of(T.write_tree(tree), build)
                self.assertEqual(pairs[3], pairs[4])
                self.assertNotEqual(pairs[3][0], pairs[0][0])
                copy = MW.material_tree(data, build)
                MW.add_texture(copy, 3, 12, BLOB_A, share_animations=False)
                self.assertNotEqual(pairs_of(T.write_tree(copy), build)[2], pairs_of(out, build)[0])


# --- V18-7 -------------------------------------------------------------------------------------

class UnresolvedTextText(unittest.TestCase):
    def test_docstring(self):
        doc = MW.check_material_links.__doc__
        self.assertIn("88 texture handles in 22 material", doc)
        self.assertIn("12 x360 appearance files", doc)


# --- V18-8 -------------------------------------------------------------------------------------

def packets(n=100):
    return [bytes([(i * 3 + k) & 0xFF for k in range(30 + i % 4)]) for i in range(n)]


def fsb_with_table(entries, n=100, duration=1000):
    pk = packets(n)
    chunk = W.vorbis_chunk(0xC0FFEE, entries)
    return W.build_fsb5(15, 8, 1, n * duration, W.vorbis_data_area(pk), [chunk], "Bank-0001"), pk


class SeekTableCheck(unittest.TestCase):
    def test_writer_table_is_clean(self):
        pk = packets()
        data = W.build_vorbis_fsb5(pk, W.packet_positions([1000] * 100), 100000, 8, 1, 1, "Bank-0001")
        self.assertEqual(W.vorbis_seek_problems(data, check_count=True), [])
        self.assertEqual(W.vorbis_seek_problems(data, allow_unmatched_last=False), [])

    def test_unmatched_last_entry_tolerated(self):
        offsets = W.vorbis_packet_offsets(packets())
        data, _ = fsb_with_table([(44000, offsets[44]), (88000, offsets[88] + 1)])
        self.assertEqual(W.vorbis_seek_problems(data), [])
        self.assertEqual(len(W.vorbis_seek_problems(data, allow_unmatched_last=False)), 1)
        self.assertEqual(W.vorbis_seek_problems(P.parse_fsb5(data)), [])

    def test_unmatched_other_entry_reported(self):
        offsets = W.vorbis_packet_offsets(packets())
        data, _ = fsb_with_table([(44000, offsets[44] + 1), (88000, offsets[88])])
        self.assertEqual(len(W.vorbis_seek_problems(data)), 1)

    def test_count_only_on_request(self):
        offsets = W.vorbis_packet_offsets(packets())
        data, _ = fsb_with_table([(44000, offsets[44])])
        self.assertEqual(W.vorbis_seek_problems(data), [])
        self.assertEqual(len(W.vorbis_seek_problems(data, check_count=True)), 1)

    def test_docstring_counts(self):
        self.assertIn("99.1 %", W.vorbis_seek_table.__doc__)
        self.assertNotIn("99.5", W.vorbis_seek_table.__doc__)


# --- V18-9 -------------------------------------------------------------------------------------

class CountingEntries(list):
    """A list that counts how often it is iterated (each iteration builds a key set)."""

    def __init__(self, items):
        super().__init__(items)
        self.walks = 0

    def __iter__(self):
        self.walks += 1
        return super().__iter__()


class TocKeyLookup(unittest.TestCase):
    def test_cached_per_toc(self):
        toc = CoreToc("switch", CountingEntries([TocEntry(36, 70, "m"), TocEntry(44, 500, "t")]))
        first = toc.key_set()
        self.assertIs(toc.key_set(), first)
        self.assertIs(MW.toc_keys(toc), first)
        tree = MW.new_material("switch", 70, colours("switch"), 1.0, 0, [(1, 500), (2, 501)])
        for _ in range(50):
            MW.check_material_links(tree, toc)
        self.assertEqual(toc.entries.walks, 1)
        self.assertEqual(toc.keys(), set(first))

    def test_rebuilt_when_entries_change(self):
        toc = CoreToc("switch", [TocEntry(44, 1, "a")])
        first = toc.key_set()
        toc.entries.append(TocEntry(44, 2, "b"))
        self.assertIn((44, 2), toc.key_set())
        toc.entries = [TocEntry(44, 3, "c")]
        self.assertEqual(toc.key_set(), frozenset({(44, 3)}))
        self.assertIsNot(toc.key_set(), first)
        self.assertEqual(CoreToc("switch", [TocEntry(44, 3, "c")]), toc)  # cache is not a field

    def test_prepared_set_not_copied(self):
        prepared = frozenset({(44, 500), (36, 70)})
        self.assertIs(MW.toc_keys(prepared), prepared)
        plain = {(37, 40)}
        self.assertIs(MW.toc_keys(plain), plain)

    def test_shader_map_lookup_uses_cache(self):
        from test_stage15_content import shader_map

        shm = SH.read_shader_map(shader_map("switch", [(2, 0x100, 40)]), "switch")
        toc = CoreToc("switch", CountingEntries([TocEntry(37, 40, "s")]))
        for _ in range(10):
            self.assertEqual(SH.check_shader_map_links(shm, toc), [])
        self.assertEqual(toc.entries.walks, 1)
        self.assertEqual(MW.toc_keys([toc, CoreToc("switch", [TocEntry(1, 2, "x")])]),
                         {(37, 40), (1, 2)})


if __name__ == "__main__":
    unittest.main()
