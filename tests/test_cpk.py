import json
import os
import random
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cpklib
import cpkrebuild  # noqa: F401  (patches CPKWriter)
import prefetch_parser
from cpklib import CPKArchive, CPKWriter
from test_d3cutscene import ogg_page


FILES = {
    "Dir\\A.txt": b"alpha " * 3000,
    "Dir\\B.bin": bytes(range(256)) * 400,
    "Other\\C.dat": b"gamma" * 5000,
}


def build_legacy(path, files=FILES, version=6, endian="big"):
    w = CPKWriter.create_new(version=version, endian=endian)
    for name, data in files.items():
        w.add_file(name, data)
    w.save(path)


def sector_crc(archive):
    return zlib.crc32(archive._data[archive._first_sec:]) & 0xFFFFFFFF


def to_fmt10(src_path, dst_path):
    """Convert a legacy archive to the Patch 2.6.7+ layout (fmt_const 10)."""
    ar = CPKArchive.open(src_path)
    h = ar.header
    data = bytearray(ar._data)
    first = ar._first_sec
    sec0 = bytes(data[first:first + h.comp_sector])
    data[first:first + h.comp_sector] = sec0[6:] + b"\x00" * 6
    struct.pack_into(h.endian + "I", data, 16, 10)
    struct.pack_into(h.endian + "I", data, 28, first)
    crc = zlib.crc32(bytes(data[first:])) & 0xFFFFFFFF
    struct.pack_into(h.endian + "I", data, 56, crc)
    with open(dst_path, "wb") as f:
        f.write(data)


def to_encrypted(src_path, dst_path):
    """Encrypt every chunk of a legacy archive built by build_legacy."""
    ar = CPKArchive.open(src_path)
    h = ar.header
    data = bytearray(ar._data)
    for s in range(ar._comp_sector_count):
        p = ar._first_sec + s * h.comp_sector + (6 if s == 0 else 0)
        end = ar._first_sec + (s + 1) * h.comp_sector
        while p + 6 <= end:
            zsize = struct.unpack_from(h.endian + "H", data, p + 4)[0]
            if zsize == 0:
                break
            data[p + 6:p + 6 + zsize] = cpklib._encrypt_chunk(
                bytes(data[p + 6:p + 6 + zsize]))
            struct.pack_into(h.endian + "H", data, p + 4, zsize | 0x8000)
            p += 6 + zsize
    crc = zlib.crc32(bytes(data[ar._first_sec:])) & 0xFFFFFFFF
    struct.pack_into(h.endian + "I", data, 56, crc)
    with open(dst_path, "wb") as f:
        f.write(data)


def build_cutscene(path, files):
    """Write a raw cutscene archive (fmt_const 0, plain payload)."""
    data_start = 0x10000
    payload = b"".join(files.values())
    sfi, locs = cpklib._BitWriter(), cpklib._BitWriter()
    name_offsets, names = b"", b""
    offset = 0
    for i, (name, content) in enumerate(files.items()):
        sfi.write(cpklib.fnv_hash64(name), 64)
        sfi.write(len(content), 32)
        sfi.write(1, 1)
        sfi.write(i, 8)
        locs.write(offset, 32)
        offset += len(content)
        name_offsets += struct.pack(">I", len(names))
        names += name.encode("latin-1") + b"\x00"
    header = struct.pack(
        ">IIQ11I", CPKArchive.MAGIC, 6, len(payload), 0, len(files), len(files),
        1, 32, 1, 8, 32, 0, 0, 0,
    ).ljust(64, b"\x00")
    tables = header + sfi.finish() + locs.finish() + name_offsets + names
    with open(path, "wb") as f:
        f.write(tables.ljust(data_start, b"\x00") + payload)


class CipherTests(unittest.TestCase):
    @staticmethod
    def bytewise(data, feedback_output):
        """The cipher as specified, one byte at a time."""
        state = cpklib._STREAM_CIPHER_INITIAL_STATE
        out = bytearray()
        for b in data:
            o = b ^ (state & 0xFF)
            out.append(o)
            state = ((o if feedback_output else b) << 56) | (state >> 8)
        return bytes(out)

    def test_block_cipher_matches_bytewise_definition(self):
        rnd = random.Random(4)
        for n in (0, 1, 7, 8, 9, 16, 17, 1000):
            data = rnd.randbytes(n)
            self.assertEqual(cpklib._decrypt_chunk(data), self.bytewise(data, False))
            self.assertEqual(cpklib._encrypt_chunk(data), self.bytewise(data, True))

    def test_observed_prefix_decrypts_to_zlib_header(self):
        # Encrypted chunks in shipped archives start with 19 0D.
        pt = cpklib._decrypt_chunk(bytes.fromhex("190d"))
        self.assertEqual(pt, b"\x78\xda")

    def test_encrypt_decrypt_round_trip(self):
        pt = zlib.compress(b"hello" * 100, 9)
        self.assertEqual(cpklib._decrypt_chunk(cpklib._encrypt_chunk(pt)), pt)


class TempDirTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self._tmp.cleanup)
        self.tmp = self._tmp.name

    def path(self, name):
        return os.path.join(self.tmp, name)

    def tool(self, *args):
        return subprocess.run(
            [sys.executable, os.path.join(ROOT, "cpk_tool.py"), *args],
            capture_output=True, text=True,
        )


class ReaderTests(TempDirTest):
    def _first_chunk(self, ar):
        """Return (chunk data offset, zsize) of the first chunk in sector 0."""
        hdr = ar._first_sec + 6  # legacy preamble with skip == 0
        zsize = struct.unpack(ar.header.endian + "H", ar._data[hdr + 4:hdr + 6])[0]
        return hdr + 6, zsize

    def test_create_round_trip_v6_and_v7(self):
        for version, endian in ((6, "big"), (7, "little")):
            p = self.path(f"v{version}.cpk")
            build_legacy(p, version=version, endian=endian)
            ar = CPKArchive.open(p)
            for name, data in FILES.items():
                self.assertEqual(ar.read(name), data)
            self.assertEqual(ar.header.sector_crc32, sector_crc(ar))

    def test_truncated_stream_keeps_partial_output(self):
        p = self.path("trunc.cpk")
        build_legacy(p, {"A.bin": random.Random(1).randbytes(4000)})
        ar = CPKArchive.open(p)
        data = bytearray(ar._data)
        pos, zsize = self._first_chunk(ar)
        struct.pack_into(">H", data, pos - 2, zsize - 500)
        broken = CPKArchive(bytes(data))
        got = broken.read("A.bin")
        self.assertEqual(len(got), 4000)
        self.assertEqual(got[:3000], ar.read("A.bin")[:3000])
        self.assertEqual(broken.chunk_errors, 0)

    def test_corrupt_chunk_is_counted(self):
        p = self.path("corrupt.cpk")
        build_legacy(p, {"A.bin": b"x" * 4000})
        ar = CPKArchive.open(p)
        data = bytearray(ar._data)
        pos, _ = self._first_chunk(ar)
        data[pos:pos + 2] = b"\xff\xff"
        broken = CPKArchive(bytes(data))
        self.assertEqual(broken.read("A.bin"), b"\x00" * 4000)
        self.assertEqual(broken.chunk_errors, 1)

    def test_raw_cutscene_archive_is_rejected_with_hint(self):
        p = self.path("raw.cpk")
        build_legacy(p)
        ar = CPKArchive.open(p)
        h = ar.header
        data = bytearray(ar._data)
        struct.pack_into(h.endian + "I", data, 16, 0)
        struct.pack_into(h.endian + "Q", data, 8, len(data) - ar._first_sec)
        with self.assertRaisesRegex(ValueError, "d3cutscene_tool"):
            CPKArchive(bytes(data))

    def test_write_entry_rejects_path_traversal(self):
        out = self.path("out")
        for name in ("..\\..\\evil.txt", "a/../../evil.txt", ".."):
            e = cpklib.CPKEntry(name, 1, 0, 0, 1, 0, 0)
            with self.assertRaises(ValueError):
                CPKArchive._write_entry(e, b"x", out, flat=False, sanitize_pipe=True)
        self.assertFalse(os.path.exists(self.path("evil.txt")))

    def test_empty_file_is_not_an_archive(self):
        p = self.path("empty.cpk")
        open(p, "wb").close()
        with self.assertRaisesRegex(ValueError, "Not a CPK"):
            CPKArchive.open(p)

    def test_encrypted_chunks_round_trip(self):
        plain, enc = self.path("p.cpk"), self.path("e.cpk")
        build_legacy(plain)
        to_encrypted(plain, enc)
        ar = CPKArchive.open(enc)
        for name, data in FILES.items():
            self.assertEqual(ar.read(name), data)
        self.assertEqual(ar.chunk_errors, 0)

    def test_close_releases_the_file(self):
        p = self.path("a.cpk")
        build_legacy(p)
        with CPKArchive.open(p) as ar:
            self.assertEqual(ar.read("Dir\\A.txt"), FILES["Dir\\A.txt"])
        with self.assertRaises(ValueError):
            ar.read("Dir\\B.bin")
        os.remove(p)


class LazyReadTests(TempDirTest):
    def setUp(self):
        super().setUp()
        rnd = random.Random(3)
        # Hex text compresses about 2:1, so one file spans several sectors.
        self.files = {
            f"F{i}.bin": rnd.randbytes(15000).hex().encode() for i in range(4)
        }
        self.p = self.path("multi.cpk")
        build_legacy(self.p, self.files)

    def test_read_inflates_only_overlapping_sectors(self):
        ar = CPKArchive.open(self.p)
        with mock.patch.object(
            ar, "_decompress_sector", wraps=ar._decompress_sector
        ) as inflate:
            self.assertEqual(ar.read("F3.bin"), self.files["F3.bin"])
        self.assertIsNone(ar._global_stream)
        self.assertLess(inflate.call_count, ar._comp_sector_count / 2)

    def test_range_read_matches_global_stream(self):
        fmt10 = self.path("fmt10.cpk")
        to_fmt10(self.p, fmt10)
        for path in (self.p, fmt10):
            lazy, full = CPKArchive.open(path), CPKArchive.open(path)
            gs = full.global_stream
            for e in lazy:
                self.assertEqual(lazy.read(e), gs[e.offset:e.offset + e.size])
                self.assertEqual(lazy._read_range(e.offset + 1, 5000),
                                 gs[e.offset + 1:e.offset + 5001])

    def test_unsorted_cs2do_falls_back_to_global_stream(self):
        ar = CPKArchive.open(self.p)
        ar._cs2do_increasing = False
        self.assertEqual(ar.read("F2.bin"), self.files["F2.bin"])
        self.assertIsNotNone(ar._global_stream)

    def test_shared_sector_failure_is_counted_once(self):
        ar = CPKArchive.open(self.p)
        data = bytearray(ar._data)
        pos = ar._first_sec + 12
        data[pos:pos + 2] = b"\xff\xff"
        broken = CPKArchive(bytes(data))
        for _ in range(2):
            broken._sector_cache.clear()
            broken.read("F0.bin")
        self.assertEqual(broken.chunk_errors, 1)


class VerifyTests(TempDirTest):
    def corrupted(self, offset_in_sectors):
        p = self.path("a.cpk")
        build_legacy(p)
        ar = CPKArchive.open(p)
        data = bytearray(ar._data)
        pos = ar._first_sec + offset_in_sectors
        data[pos:pos + 2] = b"\xff\xff"
        out = self.path("bad.cpk")
        with open(out, "wb") as f:
            f.write(data)
        return out

    def test_clean_archives_pass(self):
        p, fmt10 = self.path("a.cpk"), self.path("b.cpk")
        build_legacy(p)
        to_fmt10(p, fmt10)
        for path in (p, fmt10):
            report = CPKArchive.open(path).verify()
            self.assertTrue(report.ok, report.problems)
            self.assertEqual(report.warnings, [])

    def test_crc_mismatch_is_a_problem(self):
        bad = self.corrupted(0x3FFE)  # padding, so every chunk still inflates
        report = CPKArchive.open(bad).verify()
        self.assertFalse(report.ok)
        self.assertEqual(report.failures, [])
        self.assertIn("CRC32", report.problems[0])

    def test_failed_chunk_names_affected_files(self):
        bad = self.corrupted(12)
        report = CPKArchive.open(bad).verify()
        self.assertEqual(len(report.failures), 1)
        self.assertEqual(report.failures[0].sector, 0)
        self.assertEqual(report.affected_files, ["Dir\\A.txt"])

    def test_wrong_cipher_key_is_detected(self):
        # Issue #3: reads "succeed" with zeros, only verify notices.
        plain, enc = self.path("p.cpk"), self.path("e.cpk")
        build_legacy(plain)
        to_encrypted(plain, enc)
        ar = CPKArchive.open(enc)
        with mock.patch.object(cpklib, "_STREAM_CIPHER_KEY", bytes(8)):
            self.assertEqual(ar.read("Dir\\A.txt"), bytes(len(FILES["Dir\\A.txt"])))
            report = ar.verify()
        self.assertFalse(report.ok)
        self.assertEqual(report.crc_actual, report.crc_stored)
        self.assertEqual(sorted(report.affected_files), sorted(FILES))

    def test_truncated_archive_is_a_problem(self):
        p, cut = self.path("a.cpk"), self.path("cut.cpk")
        build_legacy(p)
        with open(p, "rb") as f:
            data = f.read()
        for size in (0x10000 - 1, 0x10000 + 100, len(data) - 0x4000, len(data) - 10):
            with open(cut, "wb") as f:
                f.write(data[:size])
            self.assertFalse(CPKArchive.open(cut).verify().ok, size)

    def test_cli_exit_codes(self):
        p = self.path("a.cpk")
        build_legacy(p)
        ok = self.tool("verify", p)
        self.assertEqual(ok.returncode, 0, ok.stderr)
        self.assertIn("OK", ok.stdout)
        bad = self.tool("verify", self.corrupted(12))
        self.assertEqual(bad.returncode, 1)
        self.assertIn("PROBLEM", bad.stdout)
        self.assertIn("Dir\\A.txt", bad.stdout)


class CutsceneDispatchTests(TempDirTest):
    def setUp(self):
        super().setUp()
        self.good = ogg_page(0x02, 1, 0) + ogg_page(0x04, 1, 1)
        self.p = self.path("enUS_Cutscene.cpk")
        build_cutscene(self.p, {
            "Data_D3\\Cutscenes\\Intro.ogv": self.good,
            "Data_D3\\Cutscenes\\Outro.ogv": self.good * 2,
        })

    def test_read_commands_use_the_cutscene_reader(self):
        info = self.tool("info", self.p)
        self.assertEqual(info.returncode, 0, info.stderr)
        self.assertIn("fmt_const:     0", info.stdout)

        listing = self.tool("list", self.p, "--ext", "ogv", "--iname", "outro")
        self.assertEqual(listing.returncode, 0, listing.stderr)
        self.assertIn("Outro.ogv", listing.stdout)
        self.assertNotIn("Intro.ogv", listing.stdout)

        out = self.path("out")
        extract = self.tool("extract", self.p, "--out", out)
        self.assertEqual(extract.returncode, 0, extract.stderr)
        with open(os.path.join(out, "Data_D3", "Cutscenes", "Intro.ogv"), "rb") as f:
            self.assertEqual(f.read(), self.good)

        verify = self.tool("verify", self.p)
        self.assertEqual(verify.returncode, 0, verify.stderr)

    def test_verify_flags_a_broken_stream(self):
        build_cutscene(self.p, {"Bad.ogv": ogg_page(0x02, 1, 0)})
        self.assertEqual(self.tool("verify", self.p).returncode, 1)

    def test_write_commands_are_refused(self):
        src = self.path("x.bin")
        open(src, "wb").close()
        r = self.tool("add", self.p, "New.ogv", src, "--out", self.path("o.cpk"))
        self.assertEqual(r.returncode, 2)
        self.assertIn("not supported on raw cutscene", r.stderr)


class ExtractorScriptTests(TempDirTest):
    def run_script(self, script, *args):
        r = subprocess.run(
            [sys.executable, os.path.join(ROOT, script), *args],
            capture_output=True, text=True,
        )
        self.assertEqual(r.returncode, 0, r.stderr)
        return r.stdout

    def test_extracts_both_layouts_with_both_entry_points(self):
        legacy, fmt10 = self.path("l.cpk"), self.path("n.cpk")
        build_legacy(legacy)
        to_fmt10(legacy, fmt10)
        for script, src in (("d3cpk_extractor.py", legacy),
                            ("d3cpk_extractor_v7.py", fmt10)):
            out = self.path(script)
            self.assertIn("Complete:  3", self.run_script(script, src, out))
            with open(os.path.join(out, "extracted", "Dir", "A.txt"), "rb") as f:
                self.assertEqual(f.read(), FILES["Dir\\A.txt"])
            self.assertTrue(os.path.exists(os.path.join(out, "extraction_report.txt")))

    def test_failed_chunks_are_dumped(self):
        p, bad = self.path("a.cpk"), self.path("bad.cpk")
        build_legacy(p)
        ar = CPKArchive.open(p)
        data = bytearray(ar._data)
        pos = ar._first_sec + 12
        data[pos:pos + 2] = b"\xff\xff"
        with open(bad, "wb") as f:
            f.write(data)
        out = self.path("out")
        self.assertIn("Partial:   1", self.run_script("d3cpk_extractor.py", bad, out))
        with open(os.path.join(out, "unknown_chunks.json"), encoding="utf-8") as f:
            meta = json.load(f)
        self.assertEqual(meta["total_unknown_chunks"], 1)
        chunk = meta["chunks"][0]
        self.assertEqual(chunk["affected_files"][0]["name"], "Dir\\A.txt")
        self.assertTrue(os.path.exists(
            os.path.join(out, "unknown_chunks", chunk["dump_file"])))


class WriterTests(TempDirTest):
    def test_saving_over_the_source_is_refused(self):
        p = self.path("a.cpk")
        build_legacy(p)
        ar = CPKArchive.open(p)
        with self.assertRaisesRegex(ValueError, "overwrite source"):
            CPKWriter.from_archive(ar).save(p)
        w = CPKWriter.from_archive(ar)
        w.delete_file("Other\\C.dat")
        with self.assertRaisesRegex(ValueError, "overwrite source"):
            w.save(p)
        self.assertEqual(CPKArchive.open(p).read("Other\\C.dat"), FILES["Other\\C.dat"])

    def test_same_size_replace_updates_crc(self):
        p, out = self.path("a.cpk"), self.path("b.cpk")
        build_legacy(p)
        new = b"ALPHA " * 3000
        w = CPKWriter.from_archive(CPKArchive.open(p))
        w.replace_file_same_size("Dir\\A.txt", new)
        w.save(out)
        ar = CPKArchive.open(out)
        self.assertEqual(ar.read("Dir\\A.txt"), new)
        self.assertEqual(ar.header.sector_crc32, sector_crc(ar))

    def test_rebuild_applies_staged_same_size_replacement(self):
        p, out = self.path("a.cpk"), self.path("b.cpk")
        build_legacy(p)
        new = b"ALPHA " * 3000
        w = CPKWriter.from_archive(CPKArchive.open(p))
        w.replace_file_same_size("Dir\\A.txt", new)
        w.delete_file("Other\\C.dat")
        w.save(out)
        ar = CPKArchive.open(out)
        self.assertEqual(ar.read("Dir\\A.txt"), new)
        self.assertIsNone(ar.get("Other\\C.dat"))

    def test_names_are_case_insensitive_for_rebuild(self):
        p, out = self.path("a.cpk"), self.path("b.cpk")
        build_legacy(p)
        w = CPKWriter.from_archive(CPKArchive.open(p))
        w.replace_file("DIR\\A.TXT", b"resized")
        w.delete_file("other\\c.dat")
        w.save(out)
        ar = CPKArchive.open(out)
        self.assertEqual(ar.read("Dir\\A.txt"), b"resized")
        self.assertIsNone(ar.get("Other\\C.dat"))

    def test_fmt10_read_and_rebuild(self):
        legacy, p, out = self.path("l.cpk"), self.path("n.cpk"), self.path("o.cpk")
        build_legacy(legacy)
        to_fmt10(legacy, p)
        ar = CPKArchive.open(p)
        self.assertEqual(ar.header.fmt_const, 10)
        self.assertEqual(ar.read("Dir\\A.txt"), FILES["Dir\\A.txt"])
        w = CPKWriter.from_archive(ar)
        w.add_file("New\\D.txt", b"delta" * 100)
        w.save(out)
        re_ar = CPKArchive.open(out)
        self.assertEqual(re_ar.header.fmt_const, 10)
        self.assertEqual(re_ar.read("New\\D.txt"), b"delta" * 100)
        for name, data in FILES.items():
            self.assertEqual(re_ar.read(name), data)
        self.assertEqual(re_ar.header.sector_crc32, sector_crc(re_ar))

    def test_cli_same_size_replace_falls_back_to_rebuild(self):
        p, out, src = self.path("a.cpk"), self.path("b.cpk"), self.path("new.bin")
        build_legacy(p)
        new = random.Random(2).randbytes(len(FILES["Dir\\A.txt"]))
        with open(src, "wb") as f:
            f.write(new)
        r = subprocess.run(
            [sys.executable, os.path.join(ROOT, "cpk_tool.py"), "replace", p,
             "Dir\\A.txt", src, "--out", out],
            capture_output=True, text=True,
        )
        self.assertEqual(r.returncode, 0, r.stderr)
        ar = CPKArchive.open(out)
        self.assertEqual(ar.read("Dir\\A.txt"), new)
        self.assertEqual(ar.read("Other\\C.dat"), FILES["Other\\C.dat"])


class PrefetchTests(TempDirTest):
    def test_coretoc_handles_may_collide_across_groups(self):
        p = self.path("CoreTOC.dat")
        rec = lambda g, s, n: struct.pack(">II", g, s) + n.ljust(128, b"\x00")
        with open(p, "wb") as f:
            f.write(struct.pack(">I", 2) + rec(1, 7, b"Actor7") + rec(29, 7, b"Power7"))
        toc = prefetch_parser.parse_coretoc(p)
        self.assertEqual(toc[(1, 7)][1], "Actor7")
        self.assertEqual(toc[(29, 7)][1], "Power7")


if __name__ == "__main__":
    unittest.main()
