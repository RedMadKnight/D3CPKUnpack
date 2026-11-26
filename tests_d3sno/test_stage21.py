"""Stage 21 (``46_stage21_apply.md``): W21-1 to W21-9, the ``apply`` command. Synthetic games built by the
test builders only; every game, export and output folder is a temporary folder outside any git working
tree (one test makes a ``.git`` folder of its own inside its temporary folder, to test the refusal)."""

import _libpath  # noqa: F401  (python -I: the library folder on sys.path)
import hashlib
import json
import shutil
import struct
import unittest
from pathlib import Path

import builders as B
import d3sno
import export_fixture as F
from d3sno import apply as AP
from d3sno import archive as A
from d3sno import cli
from d3sno import export as E
from d3sno import png
from d3sno import text as T
from d3sno import texture
from d3sno.texture import Image

TEXT_PATH = f"data/text/enUS/{F.STL}.json"
TEX_PATH = f"textures/{F.TEX}.png"
MESH_PATH = f"meshes/{F.APP}.glb"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class Setup:
    """A game folder, its export and an editable copy of the export, in one temporary folder."""

    def __init__(self, case, files=None, skip=(), **options):
        self.case = case
        self.root = F.temp_root(case)
        self.files = files if files is not None else F.switch_files()
        self.game = F.write_game(self.root / "game", self.files)
        result = E.export_game(self.game, self.root / "export", created="X", **options)
        case.assertEqual(result.exit_code, 0, result.message)
        self.export = self.root / "export"
        self.manifest = result.manifest
        self.edit = self.root / "edit"
        shutil.copytree(self.export, self.edit, ignore=lambda d, names: [n for n in names
                                                                          if self.rel(Path(d) / n) in skip])

    def rel(self, path: Path) -> str:
        try:
            return path.relative_to(self.export).as_posix()
        except ValueError:
            return ""

    def apply(self, out="out", **options):
        result = AP.apply_export(self.game, self.edit, self.root / out, **options)
        return result, self.root / out

    def ok(self, out="out", **options):
        result, path = self.apply(out, **options)
        self.case.assertEqual(result.exit_code, 0, result.message)
        return result.report, path


# --- edits ------------------------------------------------------------------------------------------

def recolour(path: Path) -> None:
    image = png.read_png(path.read_bytes())
    rgba = bytearray(image.rgba)
    rgba[0:4] = bytes((255 - rgba[0], 7, 9, 255))
    path.write_bytes(texture.encode_png(Image(image.width, image.height, bytes(rgba))))


def glb_parts(data: bytes):
    json_len = struct.unpack_from("<I", data, 12)[0]
    document = json.loads(data[20:20 + json_len])
    rest = data[20 + json_len:]
    binary = rest[8:8 + struct.unpack_from("<I", rest, 0)[0]] if rest else b""
    return document, bytearray(binary)


def glb_join(document: dict, binary: bytes) -> bytes:
    text = json.dumps(document, separators=(",", ":")).encode("utf-8")
    text += b" " * (-len(text) % 4)
    binary = bytes(binary) + bytes(-len(binary) % 4)
    body = struct.pack("<II", len(text), 0x4E4F534A) + text + struct.pack("<II", len(binary), 0x004E4942) + binary
    return struct.pack("<III", 0x46546C67, 2, 12 + len(body)) + body


def move_vertex(path: Path, delta=0.5) -> None:
    """Move the first vertex of the first primitive by ``delta`` in x, with the accessor bounds."""
    document, binary = glb_parts(path.read_bytes())
    accessor = document["accessors"][document["meshes"][0]["primitives"][0]["attributes"]["POSITION"]]
    view = document["bufferViews"][accessor["bufferView"]]
    start = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    stride = view.get("byteStride", 12)
    x = struct.unpack_from("<f", binary, start)[0] + delta
    struct.pack_into("<f", binary, start, x)
    xs = [struct.unpack_from("<f", binary, start + i * stride)[0] for i in range(accessor["count"])]
    accessor["min"][0], accessor["max"][0] = min(xs), max(xs)
    path.write_bytes(glb_join(document, binary))


def edit_strings(path: Path, change) -> None:
    document = json.loads(path.read_text(encoding="utf-8"))
    change(document["strings"])
    path.write_bytes(E.dumps(document))


def three_edits(setup: Setup) -> None:
    recolour(setup.edit / TEX_PATH)
    move_vertex(setup.edit / MESH_PATH)
    edit_strings(setup.edit / TEXT_PATH, lambda s: s.__setitem__("00001234", "Hallo"))


def stored(game: Path, stem: str, name: str) -> list[bytes]:
    cpk = A.read_archive((game / f"{stem}.cpk").read_bytes())
    entry = cpk.find(name)
    return [cpk.read_file(entry, i) for i in range(max(entry.count, 1))]


def changed_rows(report) -> dict:
    return {r["path"]: r for r in report["changed"]}


# --- W21-1 ------------------------------------------------------------------------------------------

class OutputHash(unittest.TestCase):
    def test_every_entry(self):
        s = Setup(self)
        self.assertEqual(s.manifest["format_version"], 4)
        self.assertEqual(E.FORMAT_VERSION, 4)
        self.assertTrue(s.manifest["entries"])
        for entry in s.manifest["entries"]:
            self.assertEqual(entry["output_hash"], sha((s.export / entry["path"]).read_bytes()), entry["path"])

    def test_skip_existing_without_hash(self):
        """An entry without ``output_hash`` counts as not existing: its file is written again."""
        s = Setup(self)
        manifest_path = s.export / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for entry in manifest["entries"]:
            if entry["path"] == TEX_PATH:
                del entry["output_hash"]
        manifest_path.write_bytes(E.dumps(manifest))
        original = (s.export / TEX_PATH).read_bytes()
        (s.export / TEX_PATH).write_bytes(b"stale")
        (s.export / MESH_PATH).write_bytes(b"kept")  # its entry has a hash: kept as it is
        result = E.export_game(s.game, s.export, created="X", skip_existing=True)
        self.assertEqual(result.exit_code, 0)
        self.assertEqual((s.export / TEX_PATH).read_bytes(), original)
        self.assertEqual((s.export / MESH_PATH).read_bytes(), b"kept")
        hashes = {e["path"]: e["output_hash"] for e in result.manifest["entries"]}
        self.assertEqual(hashes[TEX_PATH], sha(original))
        self.assertEqual(hashes[MESH_PATH], {e["path"]: e["output_hash"] for e in s.manifest["entries"]}[MESH_PATH])


# --- W21-2, W21-6 -----------------------------------------------------------------------------------

class NoEdit(unittest.TestCase):
    def test_nothing_changed(self):
        s = Setup(self)
        report, out = s.ok()
        self.assertEqual(report["counts"], {"unchanged": len(s.manifest["entries"]), "changed": 0, "applied": 0,
                                            "missing": 0, "unmanaged": 0, "not_importable": 0, "refused": 0,
                                            "filtered": 0, "no_effect": 0})
        self.assertEqual(report["archives_written"], [])
        self.assertEqual(report["index_files_written"], [])
        self.assertEqual(sorted(p.name for p in out.rglob("*") if p.is_file()), [AP.REPORT_NAME])

    def test_copy_unchanged(self):
        s = Setup(self)
        report, out = s.ok(copy_unchanged=True)
        game = {p.relative_to(s.game).as_posix(): p.read_bytes() for p in s.game.rglob("*") if p.is_file()}
        got = {p.relative_to(out).as_posix(): p.read_bytes() for p in out.rglob("*") if p.is_file()}
        del got[AP.REPORT_NAME]
        self.assertEqual(got, game)
        self.assertEqual(report["copied"], sorted(game, key=str.lower))

    def test_report_deterministic(self):
        s = Setup(self)
        three_edits(s)
        _, first = s.ok("one")
        _, second = s.ok("two")
        self.assertEqual((first / AP.REPORT_NAME).read_bytes(), (second / AP.REPORT_NAME).read_bytes())
        report = json.loads((first / AP.REPORT_NAME).read_text(encoding="utf-8"))
        self.assertEqual((first / AP.REPORT_NAME).read_bytes(), E.dumps(report))  # keys sorted as in ``40``


# --- W21-3 to W21-6 ---------------------------------------------------------------------------------

class OneEditEach(unittest.TestCase):
    def setUp(self):
        self.s = Setup(self)
        three_edits(self.s)
        self.report, self.out = self.s.ok(copy_unchanged=True)

    def test_exactly_three_applied(self):
        rows = changed_rows(self.report)
        self.assertEqual(sorted(rows), sorted([TEX_PATH, MESH_PATH, TEXT_PATH]))
        self.assertEqual({r["result"] for r in rows.values()}, {"applied"})
        self.assertEqual(self.report["counts"]["applied"], 3)
        self.assertEqual(self.report["counts"]["changed"], 3)
        self.assertEqual(rows[TEX_PATH]["archives"], ["Act1", "Common"])  # two equal copies (W21-5)
        self.assertEqual(rows[MESH_PATH]["archives"], ["Common"])
        self.assertEqual(self.report["archives_written"], ["Act1.cpk", "Common.cpk"])

    def test_archives_verify(self):
        for name in self.report["archives_written"]:
            self.assertTrue(A.verify_archive((self.out / name).read_bytes()).ok, name)

    def test_contents(self):
        image = png.read_png((self.s.edit / TEX_PATH).read_bytes())
        for stem in ("Common", "Act1"):
            (data,) = stored(self.out, stem, F.name_of("tex", "Stone"))
            self.assertEqual(texture.decode_image(texture.parse_texture(data), 0, 0, 0).rgba, image.rgba)
        (data,) = stored(self.out, "Common", F.name_of("stl", "Words", "enUS"))
        strings = T.parse_string_list(data)
        self.assertEqual([(e.label_hash, e.text) for e in strings.entries], [(0x1234, "Hallo"), (0x99, "Zdanie ż")])
        # what did not change is kept byte for byte
        for stem, files in self.s.files.items():
            if stem == "enUS_Cutscene":
                continue
            for name, data in files.items():
                if name in (F.name_of("tex", "Stone"), F.name_of("app", "Hero"), F.name_of("stl", "Words", "enUS")):
                    continue
                self.assertEqual(stored(self.out, stem, name), [data], name)

    def test_export_again(self):
        """A new export of the output equals the first one except for the three files and their
        ``output_hash`` and ``source.hash`` (and ``created``)."""
        result = E.export_game(self.out, self.s.root / "again", created="Y")
        self.assertEqual(result.exit_code, 0, result.message)
        again = self.s.root / "again"
        first = {p.relative_to(self.s.export).as_posix(): p.read_bytes() for p in self.s.export.rglob("*") if p.is_file()}
        second = {p.relative_to(again).as_posix(): p.read_bytes() for p in again.rglob("*") if p.is_file()}
        self.assertEqual(sorted(first), sorted(second))
        differ = sorted(k for k in first if first[k] != second[k])
        self.assertEqual(differ, sorted([TEX_PATH, MESH_PATH, TEXT_PATH, "manifest.json"]))
        self.assertEqual(second[TEX_PATH], (self.s.edit / TEX_PATH).read_bytes())
        self.assertEqual(second[TEXT_PATH], (self.s.edit / TEXT_PATH).read_bytes())
        one, two = (json.loads(d) for d in (first["manifest.json"], second["manifest.json"]))
        self.assertEqual((one.pop("created"), two.pop("created")), ("X", "Y"))
        entries_one, entries_two = one.pop("entries"), two.pop("entries")
        self.assertEqual(one, two)
        for a, b in zip(entries_one, entries_two):
            if a["path"] in (TEX_PATH, MESH_PATH, TEXT_PATH):
                self.assertNotEqual(a["output_hash"], b["output_hash"])
                a, b = dict(a), dict(b)
                for e in (a, b):
                    e.pop("output_hash")
                    e["source"] = {k: v for k, v in e["source"].items() if k != "hash"}
            self.assertEqual(a, b)


class Text(unittest.TestCase):
    def test_set_add_remove(self):
        s = Setup(self)

        def change(strings):
            strings["00000099"] = "Inne"
            del strings["00001234"]
            strings["00000005"] = "Nowy"

        edit_strings(s.edit / TEXT_PATH, change)
        report, out = s.ok()
        (data,) = stored(out, "Common", F.name_of("stl", "Words", "enUS"))
        self.assertEqual([(e.label_hash, e.text) for e in T.parse_string_list(data).entries],
                         [(0x99, "Inne"), (0x5, "Nowy")])

    def test_locale_variant(self):
        """Only the exported variant is changed (W21-5)."""
        files = F.switch_files()
        french = F.with_handle(B.build_stl("switch", [(0x1234, "Bonjour".encode())]), F.STL, "<")
        files["Common"][F.name_of("stl", "Words", "frFR")] = french
        s = Setup(self, files)
        self.assertTrue((s.edit / f"data/text/frFR/{F.STL}.json").is_file())
        edit_strings(s.edit / TEXT_PATH, lambda st: st.__setitem__("00001234", "Hallo"))
        report, out = s.ok()
        self.assertEqual(report["counts"]["applied"], 1)
        self.assertEqual(stored(out, "Common", F.name_of("stl", "Words", "frFR")), [french])
        (data,) = stored(out, "Common", F.name_of("stl", "Words", "enUS"))
        self.assertEqual(T.parse_string_list(data).by_hash[0x1234], "Hallo")

    def test_bad_json(self):
        s = Setup(self)
        (s.edit / TEXT_PATH).write_text("{not json", encoding="utf-8")
        report, _ = s.ok()
        row = changed_rows(report)[TEXT_PATH]
        self.assertEqual((row["result"], row["reason"]), ("refused", "bad_json"))


class Copies(unittest.TestCase):
    def test_differing_copies(self):
        files = F.switch_files()
        other = F.with_handle(B.build_tex("switch", 0, 4, 4, bytes(range(64, 128))), F.TEX, "<")
        files["Common"][F.name_of("tex", "Stone")] = other
        s = Setup(self, files)
        recolour(s.edit / TEX_PATH)
        report, out = s.ok()
        row = changed_rows(report)[TEX_PATH]
        self.assertEqual(row["result"], "applied")
        exported = [e for e in s.manifest["entries"] if e["path"] == TEX_PATH][0]["source"]["archive"]
        self.assertEqual(row["archives"], [exported])
        kept = "Common" if exported == "Act1" else "Act1"
        self.assertIn(f"copy_differs_not_changed: archive {kept}", row["warnings"])
        self.assertEqual(report["archives_written"], [f"{exported}.cpk"])
        self.assertFalse((out / f"{kept}.cpk").exists())


class LocaleVariants(unittest.TestCase):
    def test_only_exported_variant(self):
        """A texture stored as ``|enUS`` and ``|frFR`` variants with equal bytes: only the exported
        variant (V27-2: ``enUS``) is changed (W21-5)."""
        files = F.switch_files()
        tex = files["Common"].pop(F.name_of("tex", "Stone"))
        del files["Act1"][F.name_of("tex", "Stone")]
        files["Common"][F.name_of("tex", "Stone", "enUS")] = tex
        files["Common"][F.name_of("tex", "Stone", "frFR")] = tex
        s = Setup(self, files)
        entry = [e for e in s.manifest["entries"] if e["path"] == TEX_PATH][0]
        self.assertEqual(entry["locale"], "enUS")
        recolour(s.edit / TEX_PATH)
        report, out = s.ok()
        self.assertEqual(changed_rows(report)[TEX_PATH]["result"], "applied")
        self.assertEqual(stored(out, "Common", F.name_of("tex", "Stone", "frFR")), [tex])
        self.assertNotEqual(stored(out, "Common", F.name_of("tex", "Stone", "enUS")), [tex])


class Reasons(unittest.TestCase):
    def test_template_changed(self):
        s = Setup(self)
        recolour(s.edit / TEX_PATH)
        changed = F.switch_files()
        other = F.with_handle(B.build_tex("switch", 0, 4, 4, bytes(range(1, 65))), F.TEX, "<")
        changed["Common"][F.name_of("tex", "Stone")] = other
        changed["Act1"][F.name_of("tex", "Stone")] = other
        F.write_game(s.game, changed)
        report, out = s.ok()
        row = changed_rows(report)[TEX_PATH]
        self.assertEqual((row["result"], row["reason"]), ("refused", "template_changed"))
        self.assertEqual(report["counts"]["refused"], 1)
        self.assertEqual(report["archives_written"], [])

    def test_not_importable(self):
        s = Setup(self)
        path = s.edit / f"data/acr/{F.ACR}.json"
        path.write_bytes(path.read_bytes().replace(b"\n}", b',\n "extra": 1\n}', 1))
        anim = s.edit / f"anims/{F.ANI}.glb"
        anim.write_bytes(anim.read_bytes() + b"    ")
        report, _ = s.ok()
        rows = changed_rows(report)
        # ``49`` W22: records and animations are importable now; these two edits are refused as bad JSON / glTF
        self.assertEqual({p: (r["result"], r["reason"]) for p, r in rows.items()},
                         {f"data/acr/{F.ACR}.json": ("refused", "bad_json"),
                          f"anims/{F.ANI}.glb": ("refused", "bad_json")})
        self.assertEqual(report["counts"]["refused"], 2)

    def test_missing_and_unmanaged(self):
        s = Setup(self, skip=(TEX_PATH,))
        (s.edit / "notes.txt").write_text("mine", encoding="utf-8")
        (s.edit / "textures" / "extra.png").write_bytes(b"x")
        report, out = s.ok()
        self.assertEqual(report["missing"], [TEX_PATH])
        self.assertEqual(report["unmanaged"], ["notes.txt", "textures/extra.png"])
        self.assertEqual((report["counts"]["missing"], report["counts"]["unmanaged"], report["counts"]["changed"]),
                         (1, 2, 0))
        self.assertEqual(report["archives_written"], [])  # a missing file is no deletion

    def test_import_failed(self):
        s = Setup(self)
        (s.edit / MESH_PATH).write_bytes(b"glTF not really")
        report, _ = s.ok()
        row = changed_rows(report)[MESH_PATH]
        self.assertEqual((row["result"], row["reason"]), ("refused", "import_failed"))

    def test_image_missing(self):
        """A cube texture: one changed face and one missing face."""
        s = Setup(self, skip=(f"textures/{F.CUBE}_f1.png",))
        recolour(s.edit / f"textures/{F.CUBE}_f0.png")
        report, _ = s.ok()
        row = changed_rows(report)[f"textures/{F.CUBE}_f0.png"]
        self.assertEqual((row["result"], row["reason"]), ("refused", "image_missing"))

    def test_cube_face(self):
        s = Setup(self)
        recolour(s.edit / f"textures/{F.CUBE}_f2.png")
        report, out = s.ok()
        row = changed_rows(report)[f"textures/{F.CUBE}_f2.png"]
        self.assertEqual(row["result"], "applied")
        (data,) = stored(out, "Common", F.name_of("tex", "Sky"))
        tex = texture.parse_texture(data)
        want = png.read_png((s.edit / f"textures/{F.CUBE}_f2.png").read_bytes())
        self.assertEqual(texture.decode_image(tex, 0, 2, 0).rgba, want.rgba)
        for face in (0, 1, 3, 4, 5):
            self.assertEqual(texture.decode_image(tex, 0, face, 0).rgba, F.image(4, 4, face).rgba)

    def test_no_effect(self):
        """A changed file whose import gives the template bytes changes no archive."""
        s = Setup(self)
        path = s.edit / TEXT_PATH
        document = json.loads(path.read_text(encoding="utf-8"))
        document["warnings"] = ["edited elsewhere"]
        path.write_bytes(E.dumps(document))
        report, _ = s.ok()
        self.assertEqual(changed_rows(report)[TEXT_PATH]["result"], "no_effect")
        self.assertEqual(report["archives_written"], [])

    def test_only(self):
        s = Setup(self)
        three_edits(s)
        report, _ = s.ok(only="text")
        rows = changed_rows(report)
        self.assertEqual(rows[TEXT_PATH]["result"], "applied")
        self.assertEqual((rows[TEX_PATH]["result"], rows[MESH_PATH]["result"]), ("filtered", "filtered"))
        self.assertEqual(report["archives_written"], ["Common.cpk"])
        self.assertEqual(report["options"]["only"], ["text"])

    def test_older_manifest(self):
        s = Setup(self)
        path = s.edit / "manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["format_version"] = 3
        for entry in manifest["entries"]:
            del entry["output_hash"]
        manifest["library_version"] = "0.20.1"
        path.write_bytes(E.dumps(manifest))
        result, _ = s.apply()
        self.assertEqual(result.exit_code, 2)
        self.assertIn("library version", result.message)
        manifest["library_version"] = d3sno.__version__  # the same version: compared in memory
        path.write_bytes(E.dumps(manifest))
        recolour(s.edit / TEX_PATH)
        report, out = s.ok()
        self.assertEqual(sorted(changed_rows(report)), [TEX_PATH])
        self.assertEqual(report["counts"]["unchanged"], len(manifest["entries"]) - 1)
        self.assertFalse(any(p.name.endswith(".part") for p in s.root.rglob("*")))


class Refusals(unittest.TestCase):
    def assertRefused(self, result, text):
        self.assertEqual(result.exit_code, 2)
        self.assertIsNone(result.report)
        self.assertIn(text, result.message)

    def test_out_folders(self):
        s = Setup(self)
        for out, text in ((s.game, "game folder"), (s.game / "sub", "game folder"),
                          (s.edit, "export folder"), (s.edit / "sub", "export folder")):
            result = AP.apply_export(s.game, s.edit, out)
            self.assertRefused(result, text)
        self.assertFalse((s.game / "sub").exists())
        self.assertFalse((s.edit / AP.REPORT_NAME).exists())

    def test_git_tree(self):
        s = Setup(self)
        (s.root / "repo" / ".git").mkdir(parents=True)
        self.assertRefused(AP.apply_export(s.game, s.edit, s.root / "repo" / "out"), "git working tree")
        result = AP.apply_export(s.game, s.edit, s.root / "repo" / "out", force=True)
        self.assertEqual(result.exit_code, 0)

    def test_no_manifest(self):
        s = Setup(self, skip=("manifest.json",))
        self.assertRefused(AP.apply_export(s.game, s.edit, s.root / "out"), "manifest.json")

    def test_build_differs(self):
        s = Setup(self)
        path = s.edit / "manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["build"] = "ps3"
        path.write_bytes(E.dumps(manifest))
        self.assertRefused(AP.apply_export(s.game, s.edit, s.root / "out"), "build")
        s2 = Setup(self)
        self.assertRefused(AP.apply_export(s2.game, s2.edit, s2.root / "out", build="ps3"), "--build ps3")

    def test_options(self):
        s = Setup(self)
        self.assertRefused(AP.apply_export(s.game, s.edit, s.root / "out", only="sounds"), "--only")
        self.assertRefused(AP.apply_export(s.game, s.edit, s.root / "out", mesh_options={"missing": "remove"}),
                           "--missing")
        self.assertFalse((s.root / "out").exists())


class DryRun(unittest.TestCase):
    def test_only_report(self):
        s = Setup(self)
        three_edits(s)
        report, out = s.ok(dry_run=True, copy_unchanged=True)
        self.assertEqual(sorted(p.name for p in out.rglob("*") if p.is_file()), [AP.REPORT_NAME])
        self.assertTrue(report["dry_run"])
        self.assertEqual(report["counts"]["applied"], 3)
        self.assertEqual(report["archives_written"], ["Act1.cpk", "Common.cpk"])  # would be written


class Command(unittest.TestCase):
    def test_cli(self):
        s = Setup(self)
        three_edits(s)
        out = s.root / "cli"
        code = cli.main(["apply", str(s.game), str(s.edit), "--out", str(out), "--only", "meshes,text",
                         "--skin", "nearest"])
        self.assertEqual(code, 0)
        report = json.loads((out / AP.REPORT_NAME).read_text(encoding="utf-8"))
        self.assertEqual(report["counts"]["applied"], 2)
        self.assertEqual(report["options"]["mesh"]["skin"], "nearest")
        self.assertEqual(cli.main(["apply", str(s.game), str(s.edit), "--out", str(s.game)]), 2)


class Version(unittest.TestCase):
    def test_version(self):
        self.assertGreaterEqual(tuple(int(x) for x in d3sno.__version__.split(".")), (0, 21, 0))
        self.assertGreaterEqual(d3sno.STAGE, 21)


if __name__ == "__main__":
    unittest.main()
