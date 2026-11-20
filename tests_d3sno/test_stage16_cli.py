"""Stage 16, archive and command-line rules: ``34_modding_tools.md`` W16-12 and W16-13
(``index_write.add_sno_file``). Synthetic archives, TOCs and Prefetch files only.

The ``texture-import`` and ``mesh-import`` wiring is tested with stand-in modules (the real
``png``, ``texture_import`` and ``mesh_import`` modules are written by other agents); the
round-trip tests at the end use the real modules."""

import contextlib
import importlib.util
import io
import struct
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import builders as B
from d3sno import archive as A
from d3sno import cli
from d3sno import index as I
from d3sno import index_write as W
from d3sno import x360_archive as X
from d3sno.binary import FormatError
from d3sno.texture import Image
from test_stage14_index import prefetch_bytes, ps3_toc, sno_file, switch_toc


def run(*args):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli.main([str(a) for a in args])
    return code, out.getvalue(), err.getvalue()


FILES = {
    "Textures\\a.tex": sno_file("<", 3) + bytes(range(200)),
    "Actor\\b.acr": b"actor" * 300,
    "StringList\\c.stl|enUS": b"hello" * 3,
}


def contents(data):
    cpk = A.read_archive(data)
    return {e.name: [cpk.read_file(e, k) for k in range(e.count)] for e in cpk.files}


def make_archive(build):
    if build == "ps3":
        return A.write_archive(FILES, version=6)
    if build == "switch":
        return A.write_archive(FILES, version=7)
    return X.write_x360_archive([X.Piece(n, 0, d) for n, d in FILES.items()]).data


PS3_ROWS = [(44, 3, "a"), (1, 10, "b"), (42, 11, "c")]

X360_GROUPS = {
    "CommonTOC.dat": {44: [(3, "a")]},
    "Act1TOC.dat": {1: [(10, "b")]},
    "Act2TOC.dat": {42: [(11, "c")]},
    "Act3TOC.dat": {},
    "Act4TOC.dat": {},
    "Act5TOC.dat": {44: [(20, "e")]},
}


def x360_tocs(groups=X360_GROUPS, extra=None):
    """Six consistent x360 main TOC files (totals and highest handle over all six)."""
    keys = {(g, h) for rows in groups.values() for g, items in rows.items() for h, _ in items}
    thirds = [0] * 70
    for g, _ in keys:
        thirds[g] += 1
    top = max(h for _, h in keys)
    return {n: switch_toc(rows, ">", thirds, top) for n, rows in groups.items()}


# --- W16-13: add_sno_file ------------------------------------------------------------------------

class AddSnoFile(unittest.TestCase):
    def test_ps3_identity_entry_and_stamp(self):
        data = make_archive("ps3")
        toc = I.parse_core_toc(ps3_toc(PS3_ROWS))
        new = sno_file(">", 0) + b"payload"
        archive, index_files = W.add_sno_file(data, "Textures\\new.tex", new, toc)
        # handle = highest + 1, written at 0x10 in the file's byte order
        got = A.read_archive(archive).read_file("Textures\\new.tex")
        self.assertEqual(struct.unpack_from(">I", got, 0x10)[0], 12)
        self.assertEqual(got[:0x10] + got[0x14:], new[:0x10] + new[0x14:])
        self.assertEqual(index_files, {"CoreTOC.dat": ps3_toc(PS3_ROWS + [(44, 12, "new")])})
        # every other file unchanged, the given TOC unchanged
        after = contents(archive)
        after.pop("Textures\\new.tex")
        self.assertEqual(after, contents(data))
        self.assertEqual(len(toc.entries), 3)
        self.assertTrue(A.verify_archive(archive).ok)

    def test_switch_handle_over_all_tocs_and_pack_names_target(self):
        data = make_archive("switch")
        main = switch_toc({44: [(3, "a")], 1: [(10, "b")]})
        update = switch_toc({42: [(500, "z")]})
        tocs = {"CoreTOC.dat": main, "Update\\CoreTOCUpdate.dat": update}
        with self.assertRaises(FormatError):  # two TOCs, no pack: which one is not clear
            W.add_sno_file(data, "Actor\\new.acr", sno_file("<"), tocs)
        result = W.add_sno_file(data, "Actor\\new.acr", sno_file("<"), tocs, pack="CoreTOC.dat")
        self.assertEqual(result.identity.handle, 501)
        self.assertEqual(list(result.index_files), ["CoreTOC.dat"])
        self.assertEqual(result.index_files["CoreTOC.dat"],
                         switch_toc({44: [(3, "a")], 1: [(10, "b"), (501, "new")]}))
        got = A.read_archive(result.archive).read_file("Actor\\new.acr")
        self.assertEqual(struct.unpack_from("<I", got, 0x10)[0], 501)

    def test_localised_copy(self):
        data = make_archive("ps3")
        result = W.add_sno_file(data, "StringList\\Lines.stl|frFR", sno_file(">"), ps3_toc(PS3_ROWS))
        self.assertEqual(result.identity.entry, I.TocEntry(42, 12, "Lines"))
        self.assertIsNotNone(A.read_archive(result.archive).find("StringList\\Lines.stl|frFR"))

    def test_x360_entry_goes_to_pack_and_all_six_rewritten(self):
        data = make_archive("x360")
        before = x360_tocs()
        tocs = dict(before)
        tocs["Update\\CoreTOC.dat"] = switch_toc({44: [(900, "u")]}, ">", None, 900)
        archive, index_files = W.add_sno_file(data, "Textures\\new.tex", sno_file(">"), tocs, pack="Act2")
        self.assertEqual(sorted(index_files), sorted(before))  # the title-update file is not written
        groups = {n: dict(rows) for n, rows in X360_GROUPS.items()}
        groups["Act2TOC.dat"][44] = [(901, "new")]  # handle over every given TOC
        self.assertEqual(index_files, x360_tocs(groups))
        self.assertNotEqual(index_files["CommonTOC.dat"], before["CommonTOC.dat"])  # totals, top
        got = A.read_archive(archive).read_file("Textures\\new.tex")
        self.assertEqual(struct.unpack_from(">I", got, 0x10)[0], 901)

    def test_x360_pack_from_archive_name(self):
        names = list(X360_GROUPS)
        self.assertEqual(W.x360_pack("Act2.cpk", names), "Act2TOC.dat")
        self.assertEqual(W.x360_pack("C:\\x\\enus_act5.cpk", names), "Act5TOC.dat")
        self.assertEqual(W.x360_pack("Common.cpk", names), "CommonTOC.dat")
        self.assertEqual(W.x360_pack("Data.cpk", names, pack="act1toc.dat"), "Act1TOC.dat")
        for unclear in ("Data.cpk", "CommonLocale.cpk", "Act12.cpk", None):
            with self.assertRaises(FormatError):
                W.x360_pack(unclear, names)
        with self.assertRaises(FormatError):
            W.x360_pack("Data.cpk", names, pack="Act9")

    def test_x360_refusals(self):
        data = make_archive("x360")
        with self.assertRaises(FormatError):  # no pack
            W.add_sno_file(data, "Textures\\new.tex", sno_file(">"), x360_tocs())
        five = dict(x360_tocs())
        del five["Act4TOC.dat"]
        with self.assertRaises(FormatError):  # all six needed
            W.add_sno_file(data, "Textures\\new.tex", sno_file(">"), five, pack="Act1")

    def test_child_and_non_sno_files_get_no_entry(self):
        data = make_archive("switch")
        for name in ("Scene\\x.nav", "Anim\\y.can", "SoundBank\\z.fsb", "Misc\\readme.txt"):
            raw = b"not an sno file"
            archive, index_files = W.add_sno_file(data, name, raw, switch_toc({44: [(3, "a")]}))
            self.assertEqual(index_files, {})
            self.assertEqual(A.read_archive(archive).read_file(name), raw)

    def test_prefetch_parent_only_with_depends(self):
        data = make_archive("ps3")
        pf_items = [(1, 10, [(44, 3)])]
        pf = W.write_prefetch(pf_items, "ps3")
        _, index_files = W.add_sno_file(data, "Actor\\new.acr", sno_file(">"), ps3_toc(PS3_ROWS), pf)
        self.assertEqual(list(index_files), ["CoreTOC.dat"])  # W14-4: no entry needed
        _, index_files = W.add_sno_file(data, "Actor\\new.acr", sno_file(">"), ps3_toc(PS3_ROWS),
                                        {"Prefetch.dat": pf}, ["Textures\\a.tex", (42, 11), "b.acr"])
        self.assertEqual(index_files["Prefetch.dat"], W.write_prefetch(
            pf_items + [(1, 12, [(44, 3), (42, 11), (1, 10)])], "ps3"))

    def test_refusals(self):
        data = make_archive("ps3")
        toc = ps3_toc(PS3_ROWS)
        cases = [
            ("Textures\\a.tex", sno_file(">"), {}),  # name already in the archive
            ("Actor\\B.acr", sno_file(">"), {}),  # name already in the archive (case)
            ("Textures\\A.tex|frFR", sno_file(">"), {}),  # SNO name already in the TOC
            ("Actor\\b.acr|frFR", sno_file(">"), {}),  # SNO name already in the TOC
            ("Actor\\new.tex", sno_file(">"), {}),  # wrong folder for the extension
            ("Actor\\new.acr", b"no signature here..........", {}),  # no SNO header to stamp
            ("Actor\\new.acr", sno_file(">"), {"prefetch": prefetch_bytes([], [], ">"),
                                               "depends": ["Textures\\missing.tex"]}),
        ]
        for name, raw, extra in cases:
            with self.subTest(name=name), self.assertRaises(FormatError):
                W.add_sno_file(data, name, raw, toc, **extra)
        with self.assertRaises(ValueError):  # dependencies without Prefetch
            W.add_sno_file(data, "Actor\\new.acr", sno_file(">"), toc, depends=[(1, 10)])
        with self.assertRaises(ValueError):  # SNO file without a TOC
            W.add_sno_file(data, "Actor\\new.acr", sno_file(">"), None)
        with self.assertRaises(FormatError):  # a TOC of another build
            W.add_sno_file(data, "Actor\\new.acr", sno_file(">"), I.parse_core_toc(switch_toc({}), "switch"))

    def test_replace_files_one_rebuild(self):
        for build in ("ps3", "switch", "x360"):
            data = make_archive(build)
            out = W.replace_files(data, {"Actor\\b.acr": b"new"}, {"Misc\\n.txt": b"added"})
            got = contents(out)
            self.assertEqual(got.pop("Actor\\b.acr"), [b"new"])
            self.assertEqual(got.pop("Misc\\n.txt"), [b"added"])
            before = contents(data)
            del before["Actor\\b.acr"]
            self.assertEqual(got, before, build)
            with self.assertRaises(FormatError):
                W.replace_files(data, {"Actor\\missing.acr": b"x"})


# --- W16-12: replace and add -----------------------------------------------------------------

class ArchiveCommands(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, name, data):
        path = self.dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def test_replace_name_file(self):
        cpk = self.write("a.cpk", make_archive("switch"))
        new = self.write("b.acr", b"grown" * 999)
        out = self.dir / "out.cpk"
        code, text, err = run("replace", cpk, "Actor\\b.acr", new, "-o", out)
        self.assertEqual(code, 0, err)
        got, before = contents(out.read_bytes()), contents(cpk.read_bytes())
        self.assertEqual(got.pop("Actor\\b.acr"), [b"grown" * 999])
        del before["Actor\\b.acr"]
        self.assertEqual(got, before)
        code, _, err = run("replace", cpk, "Actor\\missing.acr", new, "-o", out)
        self.assertEqual(code, 2)

    def test_output_equal_to_input_is_refused(self):
        cpk = self.write("a.cpk", make_archive("ps3"))
        original = cpk.read_bytes()
        new = self.write("b.acr", b"x")
        toc = self.write("toc/CoreTOC.dat", ps3_toc(PS3_ROWS))
        sno = self.write("n.tex", sno_file(">"))
        commands = [
            ("replace", cpk, "Actor\\b.acr", new, "-o", cpk),
            ("replace", cpk, "Actor\\b.acr", new, "-o", new),
            ("replace", cpk, "Actor\\b.acr", new, "-o", self.dir / "." / "a.cpk"),
            ("add", cpk, "Textures\\n.tex", sno, "-o", cpk, "--toc", toc, "--index-out", self.dir / "i"),
            ("add", cpk, "Textures\\n.tex", sno, "-o", toc, "--toc", toc, "--index-out", self.dir / "i"),
            # the index file would land on its input
            ("add", cpk, "Textures\\n.tex", sno, "-o", self.dir / "o.cpk", "--toc", toc,
             "--index-out", toc.parent),
            ("texture-import", sno, new, "-o", new),
            ("mesh-import", new, self.dir / "m.obj", "-o", new),
        ]
        for args in commands:
            with self.subTest(args=args[:1] + args[-2:]):
                code, _, err = run(*args)
                self.assertEqual(code, 2)
                self.assertIn("never overwrite", err)
        self.assertEqual(cpk.read_bytes(), original)
        self.assertEqual(toc.read_bytes(), ps3_toc(PS3_ROWS))
        self.assertFalse((self.dir / "o.cpk").exists())

    def test_replace_from_dir_inverts_extract(self):
        cpk = self.write("a.cpk", make_archive("ps3"))
        tree = self.dir / "tree"
        self.assertEqual(run("extract", cpk, "-o", tree)[0], 0)
        self.assertTrue((tree / "StringList" / "c.stl@enUS").exists())
        # unchanged tree: same contents
        out = self.dir / "same.cpk"
        code, text, err = run("replace", cpk, "--from-dir", tree, "-o", out)
        self.assertEqual(code, 0, err)
        self.assertIn("replaced 3, added 0", text)
        self.assertEqual(contents(out.read_bytes()), contents(cpk.read_bytes()))
        # an edited localised file goes back to its |locale name
        (tree / "StringList" / "c.stl@enUS").write_bytes(b"bonjour")
        code, _, err = run("replace", cpk, "--from-dir", tree, "-o", out)
        self.assertEqual(code, 0, err)
        got = contents(out.read_bytes())
        self.assertEqual(got["StringList\\c.stl|enUS"], [b"bonjour"])
        self.assertEqual(got["Actor\\b.acr"], [FILES["Actor\\b.acr"]])

    def test_replace_from_dir_unknown_files(self):
        cpk = self.write("a.cpk", make_archive("switch"))
        tree = self.dir / "tree"
        self.write("tree/Actor/b.acr", b"changed")
        self.write("tree/Misc/new.txt", b"brand new")
        out = self.dir / "out.cpk"
        code, _, err = run("replace", cpk, "--from-dir", tree, "-o", out)
        self.assertEqual(code, 2)
        self.assertIn("Misc\\new.txt", err)
        self.assertFalse(out.exists())
        code, text, err = run("replace", cpk, "--from-dir", tree, "-o", out, "--add-new")
        self.assertEqual(code, 0, err)
        self.assertIn("replaced 1, added 1", text)
        got = contents(out.read_bytes())
        self.assertEqual(got["Misc\\new.txt"], [b"brand new"])
        self.assertEqual(got["Actor\\b.acr"], [b"changed"])
        self.assertEqual(got["Textures\\a.tex"], [FILES["Textures\\a.tex"]])

    def test_replace_argument_refusals(self):
        cpk = self.write("a.cpk", make_archive("ps3"))
        new = self.write("b.acr", b"x")
        out = self.dir / "o.cpk"
        self.assertEqual(run("replace", cpk, "-o", out)[0], 2)
        self.assertEqual(run("replace", cpk, "Actor\\b.acr", new, "--from-dir", self.dir, "-o", out)[0], 2)
        self.assertEqual(run("replace", cpk, "Actor\\b.acr", new, "--add-new", "-o", out)[0], 2)

    def test_add_ps3_with_toc_and_prefetch(self):
        cpk = self.write("a.cpk", make_archive("ps3"))
        toc = self.write("idx/CoreTOC.dat", ps3_toc(PS3_ROWS))
        pf = self.write("idx/Prefetch.dat", W.write_prefetch([(1, 10, [(44, 3)])], "ps3"))
        sno = self.write("new.acr", sno_file(">"))
        out, idx = self.dir / "out.cpk", self.dir / "out_idx"
        code, text, err = run("add", cpk, "Actor\\new.acr", sno, "-o", out, "--toc", toc,
                              "--prefetch", pf, "--depends", "Textures\\a.tex", "--index-out", idx)
        self.assertEqual(code, 0, err)
        self.assertIn("handle 12", text)
        self.assertEqual((idx / "CoreTOC.dat").read_bytes(), ps3_toc(PS3_ROWS + [(1, 12, "new")]))
        self.assertEqual((idx / "Prefetch.dat").read_bytes(),
                         W.write_prefetch([(1, 10, [(44, 3)]), (1, 12, [(44, 3)])], "ps3"))
        got = A.read_archive(out.read_bytes()).read_file("Actor\\new.acr")
        self.assertEqual(struct.unpack_from(">I", got, 0x10)[0], 12)
        # without --depends Prefetch is not written
        idx2 = self.dir / "out_idx2"
        code, _, err = run("add", cpk, "Actor\\new.acr", sno, "-o", out, "--toc", toc, "--prefetch", pf,
                           "--index-out", idx2)
        self.assertEqual(code, 0, err)
        self.assertEqual(sorted(p.name for p in idx2.iterdir()), ["CoreTOC.dat"])

    def test_add_refusals(self):
        cpk = self.write("a.cpk", make_archive("ps3"))
        toc = self.write("idx/CoreTOC.dat", ps3_toc(PS3_ROWS))
        sno = self.write("new.acr", sno_file(">"))
        out, idx = self.dir / "out.cpk", self.dir / "out_idx"
        cases = {
            "name already in archive": ("Actor\\b.acr", "--toc", toc, "--index-out", idx),
            "SNO already in TOC": ("Actor\\B.acr|frFR", "--toc", toc, "--index-out", idx),
            "no TOC for SNO file": ("Actor\\new.acr",),
            "TOC without index-out": ("Actor\\new.acr", "--toc", toc),
            "depends without prefetch": ("Actor\\new.acr", "--toc", toc, "--index-out", idx,
                                         "--depends", "Textures\\a.tex"),
        }
        for what, (name, *rest) in cases.items():
            with self.subTest(what):
                code, _, err = run("add", cpk, name, sno, "-o", out, *rest)
                self.assertEqual(code, 2)
                self.assertTrue(err.startswith("error:"))
                self.assertFalse(out.exists())

    def test_add_x360_pack(self):
        tocs = x360_tocs()
        paths = [self.write(f"idx/{n}", d) for n, d in tocs.items()]
        sno = self.write("new.tex", sno_file(">"))
        idx = self.dir / "out_idx"
        unclear = self.write("Data.cpk", make_archive("x360"))
        code, _, err = run("add", unclear, "Textures\\new.tex", sno, "-o", self.dir / "o.cpk",
                           "--toc", *paths, "--index-out", idx)
        self.assertEqual(code, 2)
        self.assertIn("--pack", err)
        self.assertFalse(idx.exists())
        code, _, err = run("add", unclear, "Textures\\new.tex", sno, "-o", self.dir / "o.cpk",
                           "--toc", *paths, "--index-out", idx, "--pack", "Act3")
        self.assertEqual(code, 0, err)
        self.assertEqual(I.parse_core_toc((idx / "Act3TOC.dat").read_bytes(), "x360").entries,
                         [I.TocEntry(44, 21, "new")])
        self.assertEqual(sorted(p.name for p in idx.iterdir()), sorted(tocs))
        named = self.write("enUS_Act1.cpk", make_archive("x360"))
        idx3 = self.dir / "out_idx3"
        code, _, err = run("add", named, "Textures\\new.tex", sno, "-o", self.dir / "o2.cpk",
                           "--toc", *paths, "--index-out", idx3)
        self.assertEqual(code, 0, err)
        self.assertIn(I.TocEntry(44, 21, "new"),
                      I.parse_core_toc((idx3 / "Act1TOC.dat").read_bytes(), "x360").entries)

    def test_add_child_file_needs_no_index(self):
        cpk = self.write("a.cpk", make_archive("switch"))
        nav = self.write("x.nav", b"navigation")
        out = self.dir / "o.cpk"
        code, text, err = run("add", cpk, "Scene\\x.nav", nav, "-o", out)
        self.assertEqual(code, 0, err)
        self.assertIn("no CoreTOC entry", text)
        self.assertEqual(A.read_archive(out.read_bytes()).read_file("Scene\\x.nav"), b"navigation")


# --- W16-12: texture-import and mesh-import wiring (stand-in modules) -------------------------

def image(w, h, v=0):
    return Image(w, h, bytes([v]) * (w * h * 4))


class FakeImport:
    """Stand-ins for d3sno.png, d3sno.texture_import and d3sno.mesh_import."""

    def __init__(self):
        self.calls = []
        png = types.ModuleType("d3sno.png")
        png.read_png = lambda data: Image(int(data[0]), int(data[1]), bytes([data[2]]) * (data[0] * data[1] * 4))
        tex = types.ModuleType("d3sno.texture_import")

        def import_texture(template, images, *, levels=None, format=None):
            self.calls.append(("import_texture", images, levels, format))
            return b"TEX"

        def split_strip(img, count):
            if img.height != count * img.width:
                raise FormatError(f"strip of {img.height} rows is not {count} x {img.width}")
            self.calls.append(("split_strip", count))
            size = img.width * img.width * 4
            return [Image(img.width, img.width, img.rgba[i * size:(i + 1) * size]) for i in range(count)]

        tex.import_texture, tex.split_strip = import_texture, split_strip
        mesh = types.ModuleType("d3sno.mesh_import")

        def read_gltf(data, *, base_path=None):
            self.calls.append(("read_gltf", data, base_path))
            return "gltf-doc"

        def read_obj(text):
            self.calls.append(("read_obj", text))
            return "obj-doc"

        def import_mesh_report(template, document, **options):
            self.calls.append(("import_mesh_report", document, options))
            report = types.SimpleNamespace(changed=["sub 0 draw 0"], kept=["sub 1"], warnings=["bone moved"])
            return b"APP", report

        mesh.read_gltf, mesh.read_obj, mesh.import_mesh_report = read_gltf, read_obj, import_mesh_report
        self.modules = {"d3sno.png": png, "d3sno.texture_import": tex, "d3sno.mesh_import": mesh}

    def __enter__(self):
        self.patch = mock.patch.dict(sys.modules, self.modules)
        self.patch.__enter__()
        return self

    def __exit__(self, *exc):
        return self.patch.__exit__(*exc)


def fake_png(w, h, v):
    return bytes((w, h, v))


class ImportCommands(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, name, data):
        path = self.dir / name
        path.write_bytes(data)
        return path

    def tex(self, faces=1, depth=1, w=8, h=8):
        return self.write("t.tex", B.build_tex("ps3", 0, w, h, bytes(w * h * 4 * faces * depth),
                                               faces=faces, depth=depth))

    def test_level_files_sorted_by_number(self):
        names = [f"x_level{n}.png" for n in (0, 1, 10, 11, 2)]
        self.assertEqual(cli.level_order(names), [f"x_level{n}.png" for n in (0, 1, 2, 10, 11)])
        self.assertEqual(cli.level_order(["b.png", "a.png"]), ["b.png", "a.png"])
        with self.assertRaises(ValueError):
            cli.level_order(["a_level1.png", "b_level1.png"])

    def test_texture_import_chain(self):
        template = self.tex()
        paths = [self.write(f"t_level{n}.png", fake_png(8 >> n, 8 >> n, n)) for n in (2, 0, 1)]
        out = self.dir / "o.tex"
        with FakeImport() as fake:
            code, text, err = run("texture-import", template, *paths, "-o", out, "--levels", 3,
                                  "--format", 4)
        self.assertEqual(code, 0, err)
        self.assertEqual(out.read_bytes(), b"TEX")
        kind, images, levels, fmt = fake.calls[-1]
        self.assertEqual((kind, levels, fmt), ("import_texture", 3, 4))
        self.assertEqual([i.width for i in images], [8, 4, 2])
        with FakeImport() as fake:  # one image: passed as one Image
            self.assertEqual(run("texture-import", template, paths[1], "-o", out)[0], 0)
        self.assertIsInstance(fake.calls[-1][1], Image)

    def test_texture_import_cube_faces_and_strip(self):
        template = self.tex(faces=6, w=4, h=4)
        faces = [self.write(f"f{n}.png", fake_png(4, 4, n)) for n in range(6)]
        out = self.dir / "o.tex"
        with FakeImport() as fake:
            code, _, err = run("texture-import", template, *faces, "-o", out)
            self.assertEqual(code, 0, err)
            self.assertEqual([i.rgba[0] for i in fake.calls[-1][1]], list(range(6)))
            # five faces for a cube: refused
            code, _, err = run("texture-import", template, *faces[:5], "-o", out)
            self.assertEqual(code, 2)
            strip = self.write("s.png", bytes((4, 24, 7)))
            code, _, err = run("texture-import", template, strip, "-o", out, "--strip")
            self.assertEqual(code, 0, err)
            self.assertIn(("split_strip", 6), fake.calls)
            self.assertEqual(len(fake.calls[-1][1]), 6)
            # a strip whose height is not 6 x its width: refused by split_strip
            bad = self.write("bad.png", bytes((4, 20, 7)))
            code, _, err = run("texture-import", template, bad, "-o", out, "--strip")
            self.assertEqual(code, 2)
            self.assertIn("strip", err)
            # --strip takes one image, and needs a cube or volume
            self.assertEqual(run("texture-import", template, strip, strip, "-o", out, "--strip")[0], 2)
            flat = self.tex()
            self.assertEqual(run("texture-import", flat, strip, "-o", out, "--strip")[0], 2)

    def test_mesh_import_dispatch_and_report(self):
        template = self.write("t.app", B.build_app_switch())
        out = self.dir / "o.app"
        for name, call in (("m.gltf", "read_gltf"), ("m.GLB", "read_gltf"), ("m.obj", "read_obj")):
            model = self.write(name, b"model")
            with FakeImport() as fake:
                code, text, err = run("mesh-import", template, model, "-o", out, "--flip-v",
                                      "--lightmap-v-scale", 2.5, "--missing", "empty", "--skin",
                                      "nearest", "--skip-low-poly")
            self.assertEqual(code, 0, err)
            self.assertEqual(fake.calls[0][0], call)
            if call == "read_gltf":
                self.assertEqual(Path(fake.calls[0][2]), model.resolve().parent)
            self.assertEqual(fake.calls[1][2], {"flip_v": True, "lightmap_v_scale": 2.5, "missing": "empty",
                                                "skin": "nearest", "low_poly": False})
            self.assertEqual(out.read_bytes(), b"APP")
            self.assertIn("changed sub 0 draw 0", text)
            self.assertIn("kept sub 1", text)
            self.assertIn("warning: bone moved", text)
        with FakeImport() as fake:
            _, _, _ = run("mesh-import", template, self.write("d.gltf", b"x"), "-o", out)
        self.assertEqual(fake.calls[1][2], {"flip_v": False, "lightmap_v_scale": None, "missing": "keep",
                                            "skin": "refuse", "low_poly": True})
        with FakeImport():
            code, _, err = run("mesh-import", template, self.write("m.fbx", b"x"), "-o", out)
        self.assertEqual(code, 2)
        self.assertIn(".obj", err)


# --- round trips with the real import modules (agents A and B) ---------------------------------

def _have(name):
    try:
        return importlib.util.find_spec(name) is not None
    except ImportError:
        return False


class RealRoundTrips(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    @unittest.skipUnless(_have("d3sno.png") and _have("d3sno.texture_import"), "agent A's modules")
    def test_texture_export_import_is_identity(self):
        from d3sno import texture_write as TW
        from test_stage14_texture import exact_image

        template = self.dir / "t.tex"
        template.write_bytes(TW.encode_texture(exact_image(0, 8, 8), "ps3", 0, levels=4))
        png = self.dir / "e.png"
        self.assertEqual(run("texture", template, "-o", png, "--all-levels")[0], 0)
        levels = sorted(self.dir.glob("e_level*.png"), reverse=True)  # the command sorts them
        out = self.dir / "o.tex"
        code, _, err = run("texture-import", template, *levels, "-o", out)
        self.assertEqual(code, 0, err)
        self.assertEqual(out.read_bytes(), template.read_bytes())

    @unittest.skipUnless(_have("d3sno.png") and _have("d3sno.texture_import"), "agent A's modules")
    def test_texture_strip_round_trip_and_bad_strip(self):
        from d3sno import texture_write as TW
        from test_stage14_texture import exact_image

        template = self.dir / "c.tex"
        template.write_bytes(TW.encode_texture([exact_image(0, 4, 4, s) for s in range(6)], "ps3", 0,
                                               levels=1, faces=6))
        strip = self.dir / "s.png"
        self.assertEqual(run("texture", template, "-o", strip, "--strip")[0], 0)
        out = self.dir / "o.tex"
        code, _, err = run("texture-import", template, strip, "-o", out, "--strip")
        self.assertEqual(code, 0, err)
        self.assertEqual(out.read_bytes(), template.read_bytes())
        from d3sno.texture import encode_png
        bad = self.dir / "bad.png"
        bad.write_bytes(encode_png(exact_image(0, 4, 20)))
        self.assertEqual(run("texture-import", template, bad, "-o", out, "--strip")[0], 2)

    @unittest.skipUnless(_have("d3sno.mesh_import"), "agent B's module")
    def test_mesh_export_import_is_identity(self):
        template = self.dir / "t.app"
        template.write_bytes(B.build_app_switch())
        for fmt, name in (("gltf", "m.gltf"), ("obj", "m.obj")):
            model = self.dir / name
            self.assertEqual(run("mesh", template, "-o", model, "--format", fmt)[0], 0)
            out = self.dir / f"o_{fmt}.app"
            code, text, err = run("mesh-import", template, model, "-o", out)
            self.assertEqual(code, 0, err)
            self.assertEqual(out.read_bytes(), template.read_bytes(), fmt)
            self.assertNotIn("changed", text.split("wrote")[0])


if __name__ == "__main__":
    unittest.main()
