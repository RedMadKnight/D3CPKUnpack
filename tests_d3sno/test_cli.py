import contextlib
import io
import json
import struct
import tempfile
import unittest
from pathlib import Path

import builders as B
from d3sno import FormatError
from d3sno import archive as A
from d3sno import cli
from d3sno.sno import hash33


def run(*args):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli.main([str(a) for a in args])
    return code, out.getvalue(), err.getvalue()


class Commands(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.dir = Path(self.tmp.name)
        self.cpk = self.dir / "a.cpk"
        self.cpk.write_bytes(A.write_archive({
            "Power\\Foo.pow": b"power data",
            "StringList\\Bar.stl|frFR": b"chaine",
        }))

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, name, data):
        path = self.dir / name
        path.write_bytes(data)
        return path

    def test_info(self):
        code, out, _ = run("info", self.cpk)
        self.assertEqual(code, 0)
        self.assertIn("files          2", out)

    def test_verify_good(self):
        code, out, _ = run("verify", self.cpk)
        self.assertEqual(code, 0)
        self.assertNotIn("FAIL", out)

    def test_verify_bad(self):
        data = bytearray(self.cpk.read_bytes())
        data[-1] ^= 1
        code, out, _ = run("verify", self.write("bad.cpk", bytes(data)))
        self.assertEqual(code, 1)
        self.assertIn("FAIL crc", out)

    def test_list(self):
        code, out, _ = run("list", self.cpk)
        self.assertEqual(code, 0)
        self.assertIn("Power\\Foo.pow", out)
        _, out, _ = run("list", self.cpk, "power*")
        self.assertNotIn("Bar", out)

    def test_extract(self):
        target = self.dir / "out"
        code, out, _ = run("extract", self.cpk, "-o", target)
        self.assertEqual(code, 0)
        self.assertEqual((target / "Power" / "Foo.pow").read_bytes(), b"power data")
        self.assertEqual((target / "StringList" / "Bar.stl@frFR").read_bytes(), b"chaine")

    def test_extract_refuses_escaping_names(self):
        with self.assertRaises(FormatError):
            cli.output_path(Path("root"), "..\\evil")
        with self.assertRaises(FormatError):
            cli.output_path(Path("root"), "C:\\evil")

    def test_toc(self):
        data = struct.pack(">I", 1) + struct.pack(">II", 1, 2) + b"Foo".ljust(128, b"\0")
        code, out, _ = run("toc", self.write("toc.dat", data), "--entries")
        self.assertEqual(code, 0)
        self.assertIn("build ps3, 1 entries", out)
        self.assertIn("Foo", out)

    def test_prefetch(self):
        data = struct.pack("<I4II2I", 1, 3, 4, 1, 0, 1, 5, 6)
        code, out, _ = run("prefetch", self.write("p.dat", data))
        self.assertEqual(code, 0)
        self.assertIn("1 parents, 1 dependencies", out)

    def test_prefetch_with_toc(self):
        data = struct.pack(">I4II2I", 1, 3, 4, 1, 0, 1, 5, 6)
        toc = struct.pack(">I", 1) + struct.pack(">II", 1, 2) + b"Foo".ljust(128, b"\0")
        path = self.write("p.dat", data)
        code, out, _ = run("prefetch", path, "--toc", self.write("t.dat", toc))
        self.assertIn("unresolved in CoreTOC: 1", out)

    def test_power(self):
        code_bytes = B.Code("<").f(1.5).end()
        path = self.write("x.pow", B.build_power(
            "switch", general=[(4, 0x41100, ("1.5", code_bytes)), (0, 2, 3)],
            power_bits=1, formula_count=1,
        ))
        code, out, _ = run("power", path, "--disasm")
        self.assertEqual(code, 0)
        self.assertIn("build switch", out)
        self.assertIn("PUSHF 1.5", out)
        self.assertIn("passable_while_active", out)

    def test_mesh(self):
        path = self.write("x.app", B.build_app_ps3())
        target = self.dir / "x.obj"
        code, out, _ = run("mesh", path, "-o", target)
        self.assertEqual(code, 0)
        self.assertIn("o Body", target.read_text())
        target = self.dir / "x.gltf"
        code, _, _ = run("mesh", path, "-o", target, "--format", "gltf")
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(target.read_text())["asset"]["version"], "2.0")

    def test_texture_summary_and_png(self):
        pixels = bytes((1, 2, 3, 4)) * 4
        path = self.write("x.tex", B.build_tex("ps3", 0, 2, 2, pixels))
        code, out, _ = run("texture", path)
        self.assertEqual(code, 0)
        self.assertIn("size 2x2", out)
        target = self.dir / "x.png"
        code, _, _ = run("texture", path, "-o", target)
        self.assertEqual(code, 0)
        self.assertEqual(target.read_bytes()[:4], b"\x89PNG")

    def test_texture_all_levels(self):
        pixels = bytes(4 * 4) + bytes(4)
        path = self.write("m.tex", B.build_tex("ps3", 0, 2, 2, pixels, mip_max=1))
        run("texture", path, "-o", self.dir / "m.png", "--all-levels")
        self.assertTrue((self.dir / "m_level1.png").exists())

    def test_texture_frame(self):
        frames = [(5, 0.0, 0.0, 0.5, 1.0)]
        path = self.write("f.tex", B.build_tex("ps3", 0, 2, 2, bytes(16), frames=frames))
        code, _, _ = run("texture", path, "-o", self.dir / "f.png", "--frame", 0)
        self.assertEqual(code, 0)

    def test_texture_normal(self):
        block = bytes((200, 100)) + bytes(14)
        path = self.write("n.tex", B.build_tex("ps3", 43, 4, 4, block))
        code, _, _ = run("texture", path, "-o", self.dir / "n.png", "--normal")
        self.assertEqual(code, 0)
        path = self.write("c.tex", B.build_tex("ps3", 12, 4, 4, block))
        code, _, err = run("texture", path, "-o", self.dir / "c.png", "--normal")
        self.assertEqual(code, 2)

    def test_strings(self):
        path = self.write("s.stl", B.build_stl("switch", [(hash33("Foo"), "Fóó".encode())]))
        code, out, _ = run("strings", path)
        self.assertEqual(code, 0)
        self.assertIn("Fóó", out)
        code, out, _ = run("strings", path, "--label", "Foo")
        self.assertEqual(out.strip(), "Fóó")
        code, _, _ = run("strings", path, "--label", "Nope")
        self.assertEqual(code, 1)

    def test_balance(self):
        code, out, _ = run("balance", "--gbid", "axe")
        self.assertEqual(out.strip(), "109694")
        path = self.write("b.gam", B.build_gam("switch", 11, 0xB8, ["Axe"], 32))
        code, out, _ = run("balance", path)
        self.assertEqual(code, 0)
        self.assertIn("109694 Axe", out)

    def test_balance_needs_something(self):
        self.assertEqual(run("balance")[0], 2)

    def test_collision(self):
        mesh = {
            "centre": (0.0, 0.0, 0.0), "vertices": [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
            "triangles": [B.triangle_row("switch", (0, 1, 2))],
            "nodes": [B.tree_leaf("switch", 0, 1)],
        }
        target = self.dir / "c.obj"
        code, out, _ = run("collision", self.write("c.phm", B.build_phm("switch", [mesh])),
                           "-o", target)
        self.assertEqual(code, 0)
        self.assertIn("f 1 2 3", target.read_text())

    def test_level_summary(self):
        entry = B.scene_entry("switch", 5)
        code, out, _ = run("level", self.write("w.wrl", B.build_wrl("switch", [[entry]])))
        self.assertEqual(code, 0)
        self.assertIn("scenes 1", out)

    def test_level_through_archive(self):
        scn = B.build_scene_info("switch", marker_sets=[7])
        mrk = B.build_mrk("switch", [{"position": (1.0, 0.0, 0.0)}])
        cpk = self.write("w.cpk", A.write_archive({
            "Scene\\S.scn": scn, "MarkerSet\\M.mrk": mrk,
        }))
        toc = struct.pack(">I", 2)
        for group, handle, name in ((33, 5, b"S"), (24, 7, b"M")):
            toc += struct.pack(">II", group, handle) + name.ljust(128, b"\0")
        wrl = self.write("w.wrl", B.build_wrl(
            "switch", [[B.scene_entry("switch", 5, (10.0, 0.0, 0.0))]],
        ))
        code, out, _ = run("level", wrl, "--archive", cpk, "--toc", self.write("t.dat", toc))
        self.assertEqual(code, 0)
        self.assertIn("markers 1", out)
        self.assertIn("(11.0, 0.0, 0.0)", out)

    def test_mesh_texture_uri(self):
        data = B.build_app_switch()
        data = B.add_app_materials(data, "switch", [1], [(0x10, [[(1, 44)]])])
        data = B.set_sub_object_material(data, "switch", 0, 0x10)
        target = self.dir / "t.gltf"
        code, _, _ = run("mesh", self.write("t.app", data), "-o", target, "--format", "gltf",
                         "--texture-uri", "tex/{handle}.png")
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(target.read_text())["images"], [{"uri": "tex/44.png"}])

    def test_anim_summary(self):
        perm = B.simple_permutation("switch")
        code, out, _ = run("anim", self.write("a.ani", B.build_ani("switch", [perm])))
        self.assertEqual(code, 0)
        self.assertIn("'Walk'", out)
        self.assertIn("30 fps", out)

    def test_anim_ps3_with_can_and_gltf(self):
        perm = B.simple_permutation("ps3")
        ani = self.write("a.ani", B.build_ani("ps3", [perm]))
        can = self.write("a.can", B.build_can([perm]))
        app = self.write("a.app", B.build_app_ps3(bones=[(0xAA, -1, (0.0, 0.0, 0.0))]))
        target = self.dir / "a.gltf"
        code, _, _ = run("anim", ani, "--can", can, "--app", app, "-o", target)
        self.assertEqual(code, 0)
        self.assertIn("animations", json.loads(target.read_text()))

    def test_malformed_input_gives_exit_code_2(self):
        code, _, err = run("info", self.write("junk.cpk", b"junk" * 40))
        self.assertEqual(code, 2)
        self.assertIn("error", err)

    def test_missing_file(self):
        code, _, err = run("info", self.dir / "absent.cpk")
        self.assertEqual(code, 2)
