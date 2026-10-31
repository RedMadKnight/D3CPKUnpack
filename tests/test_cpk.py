import os
import random
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cpklib
import cpkrebuild  # noqa: F401  (patches CPKWriter)
import prefetch_parser
from cpklib import CPKArchive, CPKWriter


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


class CipherTests(unittest.TestCase):
    def test_observed_prefix_decrypts_to_zlib_header(self):
        # Encrypted chunks in shipped archives start with 19 0D.
        pt = cpklib._decrypt_chunk(bytes.fromhex("190d"))
        self.assertEqual(pt, b"\x78\xda")

    def test_encrypt_decrypt_round_trip(self):
        pt = zlib.compress(b"hello" * 100, 9)
        self.assertEqual(cpklib._decrypt_chunk(cpklib._encrypt_chunk(pt)), pt)


class TempDirTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = self._tmp.name

    def path(self, name):
        return os.path.join(self.tmp, name)


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


class WriterTests(TempDirTest):
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
