"""Corrections 28 (``47_corrections28.md``): Y1 (archives written without holding them in memory), Y4 and
Y5 (``apply``), Y6 to Y8 (UV2 table rules), the tests of Y10. Synthetic archives and games built by the
test builders only, in temporary folders outside any git working tree."""

import _libpath  # noqa: F401  (python -I: the library folder on sys.path)
import contextlib
import gc
import hashlib
import io
import json
import mmap
import random
import shutil
import tracemalloc
import unittest
import zlib
from pathlib import Path

import export_fixture as F
import test_stage21 as S
import y1_cases as Y
import d3sno
from d3sno import apply as AP
from d3sno import archive as A
from d3sno import cli
from d3sno import export as E
from d3sno import index_write as IW
from d3sno import uv2_use as U
from d3sno import x360_archive as XA

MB = 1024 * 1024
BOUND = 64 * MB  # Y1: 64 MB plus twice the largest replaced file


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def same_zlib(case) -> None:
    if XA.zlib_implementation() != Y.GOLDEN_ZLIB:
        case.skipTest(f"the 0.21.0 bytes were recorded with {Y.GOLDEN_ZLIB}; this Python has "
                      f"{XA.zlib_implementation()}, whose level-9 bytes differ")


@contextlib.contextmanager
def small_chunks(chunk=4096, window=4096):
    """Y1: tiny copy chunks and read windows, so that every boundary case is hit on small archives."""
    old = A.COPY_CHUNK, A.VIEW_WINDOW
    A.COPY_CHUNK, A.VIEW_WINDOW = chunk, window
    try:
        yield
    finally:
        A.COPY_CHUNK, A.VIEW_WINDOW = old


def mapped(case, path: Path):
    """The archive at ``path`` parsed from a read-only memory map, closed after the test."""
    handle = open(path, "rb")
    view = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
    holder = {"cpk": A.read_archive(view)}

    def close():
        holder.clear()
        gc.collect()
        view.close()
        handle.close()

    case.addCleanup(close)
    return holder["cpk"]


def peak_of(call):
    """(result, peak bytes traced by tracemalloc while ``call`` runs)."""
    gc.collect()
    tracemalloc.start()
    try:
        result = call()
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    return result, peak


def quiet(argv) -> int:
    with contextlib.redirect_stdout(io.StringIO()):
        return cli.main(argv)


def changed_names(source: bytes, written: bytes) -> dict:
    """Every file of ``source`` whose stored bytes differ in ``written``: name -> new bytes."""
    old, new = A.read_archive(source), A.read_archive(written)
    out = {}
    for entry in old.files:
        if old.read_file(entry) != new.read_file(entry.name):
            out[entry.name] = new.read_file(entry.name)
    return out


# --- Y1: output -----------------------------------------------------------------------------------

class Output(unittest.TestCase):
    """Y1 output: the streaming writers give what the byte-returning functions give, and those give
    what 0.21.0 gave."""

    def test_bytes_api_equals_0_21_0(self):
        same_zlib(self)
        for name, call in Y.outputs():
            with self.subTest(name):
                self.assertEqual(Y.sha(call()), Y.GOLDEN_0_21_0[name])

    def test_streamed_equals_bytes(self):
        root = F.temp_root(self)
        for source_name, make in Y.SOURCES.items():
            source = make()
            for change_name, change in Y.CHANGES.items():
                replace, add = change()
                with self.subTest(f"{source_name}/{change_name}"):
                    want = IW.replace_files(source, replace, add)
                    with small_chunks():
                        done = IW.write_changed(A.read_archive(source), None, replace, add)
                    self.assertEqual((done.size, done.sha256), (len(want), sha(want)))
                    if change_name == "both":
                        buffer = io.BytesIO()
                        IW.write_changed(source, buffer, replace, add)
                        self.assertEqual(buffer.getvalue(), want)
                        target = root / f"{source_name}.cpk"
                        on_disk = IW.write_changed(source, target, replace, add)
                        self.assertEqual(target.read_bytes(), want)
                        self.assertEqual(on_disk, done)
                        self.assertEqual([p.name for p in root.iterdir() if p.name.endswith(".part")], [])

    def test_rebuild_and_writers_to(self):
        source = Y.ps3_source()
        want = A.rebuild(source, remove=["Base\\small.txt"]).data
        result = A.rebuild_to(None, source, remove=["Base\\small.txt"])
        self.assertIsNone(result.data)
        self.assertEqual(result.written.sha256, sha(want))
        pieces = [XA.Piece("A\\one.dat", 0, Y.data(5, 90_000)), XA.Piece("B\\two.fsb", 0, Y.data(6, 50_000))]
        whole = XA.write_x360_archive(pieces, read_sector=0x800)
        buffer = io.BytesIO()
        streamed = XA.write_x360_archive_to(buffer, pieces, read_sector=0x800, keep_plans=False)
        self.assertEqual(buffer.getvalue(), whole.data)
        self.assertEqual(streamed.packets, [])
        self.assertEqual(streamed.written.sha256, sha(whole.data))
        self.assertTrue(whole.packets)  # the byte-returning call keeps its packet plans

    def test_x360_slot_closed_short(self):
        """A slot closed with fewer than 7 bytes left (A3) is padded before the next one: a level-0
        piece of 16,364 bytes leaves 3 bytes of a 16,384-byte slot."""
        first, second = Y.data(21, 16_364), Y.data(22, 30_000)
        pieces = [XA.Piece("Sound\\a.fsb", 0, first), XA.Piece("Base\\b.dat", 0, second)]
        whole = XA.write_x360_archive(pieces, read_sector=0x800)
        cpk = A.read_archive(whole.data)
        self.assertEqual(cpk.sector_offsets[:2], [0, len(first)])
        self.assertEqual((cpk.read_file("Sound\\a.fsb"), cpk.read_file("Base\\b.dat")), (first, second))
        self.assertTrue(A.verify_archive(whole.data).ok)
        out = IW.replace_files(whole.data, {"Base\\b.dat": b"short"})
        self.assertEqual(A.read_archive(out).read_file("Sound\\a.fsb"), first)
        self.assertTrue(A.verify_archive(out).ok)

    def test_lazy_pieces_equal_read_pieces(self):
        cpk = A.read_archive(Y.x360_source())
        lazy = XA.source_pieces(cpk, lazy=True)
        read = XA.source_pieces(cpk)
        self.assertEqual([(p.name, p.copy, len(p.data)) for p in lazy], [(p.name, p.copy, len(p.data)) for p in read])
        self.assertEqual([p.data[0:len(p.data)] for p in lazy], [p.data for p in read])


def ps3_game(root: Path) -> Path:
    return F.write_game(root, F.ps3_files(), version=6, order=">")


def x360_game(root: Path) -> Path:
    stl = F.with_handle(x360_text(), F.STL, ">")
    return F.write_game(root, {"Common": {F.name_of("stl", "Words"): stl, "Base\\other.dat": Y.data(7, 300_000)}},
                        version=7, order=">")


def x360_text():
    import builders as B

    return B.build_stl("x360", [(0x1234, b"Hello"), (0x99, b"Second")])


class Games(unittest.TestCase):
    """Y1 output for each build's synthetic game: ``apply`` writes what ``replace_files`` gives for the
    same changes, and on switch what 0.21.0 wrote."""

    def check(self, game: Path, edit) -> dict:
        root = game.parent
        export = root / "export"
        result = E.export_game(game, export, created="X")
        self.assertEqual(result.exit_code, 0, result.message)
        edited = root / "edit"
        shutil.copytree(export, edited)
        edit(edited)
        result = AP.apply_export(game, edited, root / "out")
        self.assertEqual(result.exit_code, 0, result.message)
        self.assertTrue(result.report["archives_written"])
        for rel in result.report["archives_written"]:
            source, written = (game / rel).read_bytes(), (root / "out" / rel).read_bytes()
            changes = changed_names(source, written)
            self.assertTrue(changes)
            self.assertEqual(IW.replace_files(source, changes), written)
            self.assertEqual(result.report["archives"][rel], {"sha256": sha(written), "size": len(written)})
        return result.report

    def test_switch(self):
        setup = S.Setup(self)
        S.three_edits(setup)
        report, out = setup.ok()
        for rel in report["archives_written"]:
            source, written = (setup.game / rel).read_bytes(), (out / rel).read_bytes()
            self.assertEqual(IW.replace_files(source, changed_names(source, written)), written)
        same_zlib(self)
        for rel in report["archives_written"]:
            self.assertEqual(sha((out / rel).read_bytes()), Y.GOLDEN_0_21_0[f"apply_three_edits/{rel}"])

    def test_ps3(self):
        root = F.temp_root(self)
        report = self.check(ps3_game(root / "game"), lambda edit: S.move_vertex(edit / f"meshes/{F.APP}.glb"))
        self.assertEqual(report["counts"]["applied"], 1)

    def test_x360(self):
        root = F.temp_root(self)
        path = f"data/text/default/{F.STL}.json"
        report = self.check(x360_game(root / "game"),
                            lambda edit: S.edit_strings(edit / path, lambda s: s.__setitem__("00001234", "Hallo")))
        self.assertEqual(report["counts"]["applied"], 1)


class Commands(unittest.TestCase):
    """Y1: ``replace`` and ``add`` stream their output; it equals the byte-returning functions."""

    def setUp(self):
        self.root = F.temp_root(self)

    def source(self, name: str) -> Path:
        path = self.root / f"{name}.cpk"
        path.write_bytes(Y.SOURCES[name]())
        return path

    def test_replace(self):
        new = self.root / "new.txt"
        new.write_bytes(b"replaced by the command\n" * 9)
        for name in ("ps3", "switch", "x360"):
            with self.subTest(name):
                source = self.source(name)
                out = self.root / f"{name}_out.cpk"
                self.assertEqual(quiet(["replace", str(source), "Base\\small.txt", str(new), "-o", str(out)]), 0)
                self.assertEqual(out.read_bytes(), IW.replace_file(source.read_bytes(), "Base\\small.txt", new.read_bytes()))

    def test_replace_from_dir(self):
        folder = self.root / "dir"
        (folder / "Base").mkdir(parents=True)
        (folder / "New").mkdir()
        (folder / "Base" / "big.dat").write_bytes(Y.data(11, 400_000))
        (folder / "New" / "x.dat").write_bytes(b"new file")
        for name in ("ps3", "switch_compact", "x360_compact"):
            with self.subTest(name):
                source = self.source(name)
                out = self.root / f"{name}_dir.cpk"
                code = quiet(["replace", str(source), "--from-dir", str(folder), "--add-new", "-o", str(out)])
                self.assertEqual(code, 0)
                want = IW.replace_files(source.read_bytes(), {"Base\\big.dat": Y.data(11, 400_000)},
                                        {"New\\x.dat": b"new file"})
                self.assertEqual(out.read_bytes(), want)

    def test_add(self):
        new = self.root / "add.dat"
        new.write_bytes(Y.data(12, 120_000))
        for name in ("ps3", "switch", "x360"):
            with self.subTest(name):
                source = self.source(name)
                out = self.root / f"{name}_add.cpk"
                self.assertEqual(quiet(["add", str(source), "New\\added.dat", str(new), "-o", str(out)]), 0)
                self.assertEqual(out.read_bytes(), IW.add_file(source.read_bytes(), "New\\added.dat", new.read_bytes()))

    def test_add_sno_with_index(self):
        files = F.switch_files()["Common"]
        source = self.root / "Common.cpk"
        source.write_bytes(A.write_archive(files))
        toc = self.root / "CoreTOC.dat"
        toc.write_bytes(files["CoreTOC.dat"])
        new = self.root / "new.tex"
        new.write_bytes(files[F.name_of("tex", "Stone")])
        name = F.name_of("tex", "Added")
        out, index = self.root / "added.cpk", self.root / "index"
        code = quiet(["add", str(source), name, str(new), "-o", str(out), "--toc", str(toc), "--index-out", str(index)])
        self.assertEqual(code, 0)
        want = IW.add_sno_file(source.read_bytes(), name, new.read_bytes(), {"CoreTOC.dat": toc.read_bytes()})
        self.assertEqual(out.read_bytes(), want.archive)
        self.assertEqual((index / "CoreTOC.dat").read_bytes(), want.index_files["CoreTOC.dat"])
        streamed = IW.add_sno_file(source.read_bytes(), name, new.read_bytes(), {"CoreTOC.dat": toc.read_bytes()},
                                   out=self.root / "again.cpk")
        self.assertEqual(streamed.archive.sha256, sha(want.archive))
        self.assertEqual(streamed.index_files, want.index_files)
        stored = A.read_archive(out.read_bytes()).read_file(name)  # the handle stamped at 0x10 (W14-1)
        self.assertEqual(int.from_bytes(stored[0x10:0x14], "little"), want.identity.handle)
        self.assertEqual(stored[:0x10] + stored[0x14:], new.read_bytes()[:0x10] + new.read_bytes()[0x14:])


# --- Y1: memory ----------------------------------------------------------------------------------

def big_files(seed: int) -> dict:
    rnd = random.Random(seed)
    return {"Big\\a.bin": rnd.randbytes(40 * MB), "Big\\b.bin": rnd.randbytes(25 * MB),
            "Small\\s.txt": b"a small file\n" * 100}


def big_archive(kind: str, path: Path, seed: int = 1) -> dict:
    """A synthetic archive of more than 64 MB in ``kind``'s format (written at a low level, for
    speed); returns its files."""
    files = big_files(seed)
    if kind == "x360":
        pieces = [XA.Piece(n, 0, d) for n, d in files.items()]
        data = XA.write_x360_archive(pieces, compress=lambda d: zlib.compress(d, 1),
                                     compressor=lambda: zlib.compressobj(1)).data
    else:
        version, order = (6, ">") if kind == "ps3" else (7, "<")
        data = A.write_archive(files, version=version, order=order, level=1)
    path.write_bytes(data)
    del data
    return files


class Memory(unittest.TestCase):
    """Y1 memory: an archive of more than 64 MB with one small file replaced is rewritten with a
    traced peak below 64 MB plus twice the replaced file, in each build's archive format."""

    def test_each_format(self):
        root = F.temp_root(self)
        new = b"the replaced file\n" * 300
        for kind in ("ps3", "switch", "x360"):
            with self.subTest(kind):
                source = root / f"{kind}.cpk"
                files = big_archive(kind, source)
                self.assertGreater(source.stat().st_size, 64 * MB)
                cpk = mapped(self, source)
                target = root / f"{kind}_out.cpk"
                done, peak = peak_of(lambda: IW.write_changed(cpk, target, {"Small\\s.txt": new}))
                self.assertLess(peak, BOUND + 2 * len(new), f"{kind}: peak {peak / MB:.1f} MB")
                written = A.read_archive(target.read_bytes())
                self.assertEqual(written.read_file("Small\\s.txt"), new)
                self.assertEqual(sha(written.read_file("Big\\a.bin")), sha(files["Big\\a.bin"]))  # 40 MB: more than one chunk
                self.assertEqual(sha(written.read_file("Big\\b.bin")), sha(files["Big\\b.bin"]))
                self.assertEqual(done.size, target.stat().st_size)
                del files, written


class ApplyMemory(unittest.TestCase):
    """Y1 memory in ``apply``: the peak stays below the bound, and two large archives in one ``apply``
    give no higher peak than one (it does not grow with the number of archives)."""

    def run_apply(self, both: bool):
        root = F.temp_root(self)
        files = F.switch_files()
        files["Common"].update(big_files(3))
        if both:
            files["Act1"].update(big_files(4))
        game = F.write_game(root / "game", files, level=1)
        del files
        result = E.export_game(game, root / "export", created="X")
        self.assertEqual(result.exit_code, 0, result.message)
        shutil.copytree(root / "export", root / "edit")
        S.recolour(root / "edit" / S.TEX_PATH)  # the texture is in Common and in Act1
        real = AP._Apply.write_archives
        writing = []

        def measured(job):  # the memory above what classifying needs: from the start of writing
            base = tracemalloc.get_traced_memory()[0]
            tracemalloc.reset_peak()
            try:
                return real(job)
            finally:
                writing.append(tracemalloc.get_traced_memory()[1] - base)

        AP._Apply.write_archives = measured
        try:
            result, peak = peak_of(lambda: AP.apply_export(game, root / "edit", root / "out"))
        finally:
            AP._Apply.write_archives = real
        self.assertEqual(result.exit_code, 0, result.message)
        self.assertEqual(result.report["archives_written"], ["Act1.cpk", "Common.cpk"])
        for rel, info in result.report["archives"].items():
            data = (root / "out" / rel).read_bytes()
            self.assertEqual(info, {"sha256": sha(data), "size": len(data)})
            self.assertTrue(A.verify_archive(data).ok)
        replaced = len(S.stored(root / "out", "Common", F.name_of("tex", "Stone"))[0])
        return peak, writing[0], replaced

    def test_one_and_two(self):
        one, one_writing, replaced = self.run_apply(False)
        two, two_writing, _ = self.run_apply(True)
        self.assertLess(one, BOUND + 2 * replaced, f"one archive: {one / MB:.1f} MB")
        self.assertLess(two, BOUND + 2 * replaced, f"two archives: {two / MB:.1f} MB")
        # measured on 0.21.1: 8.52 MB and 8.53 MB while writing (the rest is the classification of the
        # export folder and the opened archives' tables)
        self.assertLessEqual(two_writing, one_writing + 256 * 1024,
                             f"writing two archives {two_writing / MB:.2f} MB, one {one_writing / MB:.2f} MB")


class Chunks(unittest.TestCase):
    """Y1 chunks: a stored file larger than one chunk is copied correctly across the boundary."""

    def test_default_chunk(self):
        big = random.Random(5).randbytes(A.COPY_CHUNK + 12345)
        source = A.write_archive({"Big\\c.bin": big, "Small\\s.txt": b"x" * 10}, flags=0)  # copied, not packed
        out = IW.replace_files(source, {"Small\\s.txt": b"y" * 20})
        self.assertEqual(A.read_archive(out).read_file("Big\\c.bin"), big)
        self.assertEqual(out, A.write_archive({"Big\\c.bin": big, "Small\\s.txt": b"y" * 20}, flags=0))

    def test_small_chunks_every_format(self):
        for name, make in Y.SOURCES.items():
            with self.subTest(name):
                source = make()
                with small_chunks(1000, 1500):
                    out = IW.replace_files(source, {"Base\\small.txt": b"z"})
                self.assertEqual(out, IW.replace_files(source, {"Base\\small.txt": b"z"}))
                self.assertEqual(A.read_archive(out).read_file("Base\\big.dat"), A.read_archive(source).read_file("Base\\big.dat"))

    def test_view(self):
        view = A.StreamView([b"abc", b"", b"defgh", b"ij"])
        with small_chunks(2, 3):
            self.assertEqual(view[0:10], b"abcdefghij")
            self.assertEqual([view[i:i + 3] for i in range(0, 10, 3)], [b"abc", b"def", b"ghi", b"j"])
            self.assertEqual(view[2:7], b"cdefg")
            sink = io.BytesIO()
            view.copy_to(sink)
            self.assertEqual(sink.getvalue(), b"abcdefghij")
        with self.assertRaises(TypeError):
            view[3]


# --- Y1 report, Y4, Y5 ------------------------------------------------------------------------------

class Report(unittest.TestCase):
    """Y1 report: ``archives`` gives size and SHA-256 of each written archive; the dry run gives the same
    values and writes no archive."""

    def test_sizes_hashes_and_dry_run(self):
        setup = S.Setup(self)
        S.three_edits(setup)
        report, out = setup.ok()
        self.assertEqual(sorted(report["archives"]), report["archives_written"])
        for rel, info in report["archives"].items():
            data = (out / rel).read_bytes()
            self.assertEqual(info, {"sha256": sha(data), "size": len(data)})
        dry, dry_out = setup.ok("dry", dry_run=True)
        self.assertEqual(dry["archives"], report["archives"])
        self.assertEqual([p.name for p in dry_out.rglob("*")], [AP.REPORT_NAME])
        again, again_out = setup.ok("again")
        self.assertEqual((again_out / AP.REPORT_NAME).read_bytes(), (out / AP.REPORT_NAME).read_bytes())

    def test_no_change_empty(self):
        setup = S.Setup(self)
        report, out = setup.ok()
        self.assertEqual(report["archives"], {})

    def test_stop_leaves_no_archive(self):
        setup = S.Setup(self)
        S.recolour(setup.edit / S.TEX_PATH)  # two archives
        real = IW.write_changed
        calls = []

        def failing(*args, **kw):
            calls.append(1)
            if len(calls) == 2:
                raise OSError("disk full (test)")
            return real(*args, **kw)

        IW.write_changed = failing
        try:
            result, out = setup.apply()
        finally:
            IW.write_changed = real
        self.assertEqual(result.exit_code, AP.EXIT_STOPPED)
        self.assertEqual(result.report["archives"], {})
        self.assertEqual(sorted(p.name for p in out.rglob("*")), [AP.REPORT_NAME])


class CopyUnchanged(unittest.TestCase):
    """Y1 copy and Y4: ``--copy-unchanged`` copies every file not written, archives and others alike,
    in chunks (a file larger than one chunk included)."""

    def test_copy(self):
        setup = S.Setup(self)
        big = random.Random(9).randbytes(A.COPY_CHUNK + 777)
        (setup.game / "data").mkdir()
        (setup.game / "data" / "big.bin").write_bytes(big)
        (setup.game / "readme.txt").write_bytes(b"not an archive\n")
        S.edit_strings(setup.edit / S.TEXT_PATH, lambda s: s.__setitem__("00001234", "Hallo"))
        report, out = setup.ok(copy_unchanged=True)
        self.assertEqual(report["archives_written"], ["Common.cpk"])
        self.assertIn("data/big.bin", report["copied"])
        self.assertIn("readme.txt", report["copied"])
        self.assertEqual((out / "data" / "big.bin").read_bytes(), big)
        self.assertEqual((out / "readme.txt").read_bytes(), b"not an archive\n")
        for path in setup.game.rglob("*"):
            if path.is_file() and path.name != "Common.cpk":
                self.assertEqual((out / path.relative_to(setup.game)).read_bytes(), path.read_bytes(), path)
        self.assertEqual([p.name for p in out.rglob("*.part")], [])


class IndexFiles(unittest.TestCase):
    """Y5: an ``apply`` that changes a file of the archive holding the index files writes none."""

    def test_none_written(self):
        setup = S.Setup(self)
        S.edit_strings(setup.edit / S.TEXT_PATH, lambda s: s.__setitem__("00001234", "Hallo"))
        report, out = setup.ok()
        self.assertEqual(report["archives_written"], ["Common.cpk"])
        self.assertEqual(report["index_files_written"], [])
        self.assertEqual(S.stored(out, "Common", "CoreTOC.dat"), S.stored(setup.game, "Common", "CoreTOC.dat"))
        self.assertFalse(any(p.name.lower().endswith("toc.dat") for p in out.rglob("*")))


# --- Y6 to Y8 ---------------------------------------------------------------------------------------

class Block:
    def __init__(self, textures):
        self.textures = textures

    def texture(self, slot_type):
        return self.textures.get(slot_type)


def uri(handle):
    return f"../textures/{handle}.png"


def rows(*items, known=True):
    return U.Uv2Rows(tuple(U.Uv2Row(f"s{i}", 1, 0, slot, divisor, scale) for i, (slot, divisor, scale) in enumerate(items)),
                     known)


class Uv2(unittest.TestCase):
    def test_y6_layer_with_scale_rows_only(self):
        found = rows((6, 8192, 1))
        self.assertEqual(U.uv2_use(found, False), "layer")
        (entry,) = U.layer_entries(found, Block({6: 42}), uri, "layer")
        self.assertEqual((entry["texcoord"], entry["v_scale"]), (1, "lightmap"))
        self.assertNotIn("v_divide", entry)
        (entry,) = U.layer_entries(found, Block({6: 42}), uri, "lightmap", 3.5)
        self.assertEqual(entry["texcoord"], 1)
        self.assertNotIn("v_divide", entry)
        (entry,) = U.layer_entries(found, Block({6: 42}), uri, None)
        self.assertIsNone(entry["texcoord"])

    def test_y7_slot_39_and_slot_1(self):
        found = rows((39, 8192, 1), (1, 8192, 1))
        block = Block({39: 55, 1: 66})
        slot39, slot1 = sorted(U.layer_entries(found, block, uri, "layer"), key=lambda e: -e["slot_type"])
        self.assertEqual((slot39["slot_type"], slot39["slot_name"], slot39["texture"]), (39, None, None))
        self.assertEqual(slot39["texcoord"], 1)
        self.assertEqual((slot1["slot_name"], slot1["texture"], slot1["also_base_color"]), ("diffuse", uri(66), True))
        self.assertNotIn("also_base_color", slot39)
        self.assertEqual(U.layer_texture_handles(found, block), [66])
        self.assertIsNone(U.slot_name(39))
        self.assertEqual(U.slot_name(42), "layer_42")

    def test_y7_table_rows_of_slot_39(self):
        table = U.tables()
        if table is None:
            self.skipTest("the UV2 tables are not shipped")
        for build in ("ps3", "x360"):
            by_name, _ = table.for_build(build)
            found = [r for group in by_name.values() for r in group if r.slot_type == 39]
            self.assertTrue(found, build)
            self.assertTrue(all(r.lightmap_v_scale == 1 and r.divisor == 8192 for r in found), build)

    def test_y8_empty_slot_type(self):
        layer = rows((None, 8192, 0))
        self.assertEqual(U.uv2_use(layer, False), "layer")
        (entry,) = U.layer_entries(layer, Block({None: 5}), uri, "layer")
        self.assertEqual((entry["slot_type"], entry["slot_name"], entry["texture"]), (None, None, None))
        self.assertEqual(U.layer_texture_handles(layer, Block({None: 5})), [])
        odd = rows((None, 512, 0))
        self.assertEqual(U.uv2_use(odd, False), "open")
        (entry,) = U.layer_entries(odd, None, uri, None)
        self.assertEqual((entry["slot_type"], entry["slot_name"], entry["texture"], entry["divisor"]), (None, None, None, 512))


class Version(unittest.TestCase):
    def test_version(self):
        self.assertEqual(d3sno.__version__, "0.21.1")
        self.assertEqual(d3sno.STAGE, 21)


if __name__ == "__main__":
    unittest.main()
