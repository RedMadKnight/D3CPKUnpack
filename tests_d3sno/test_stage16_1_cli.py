"""Corrections 21 (``35``): V21-6 x360 pack from the archive name, V21-7 the x360 path of the
``texture`` subcommand. Synthetic data only."""

import tempfile
import unittest
from pathlib import Path

import d3sno
from d3sno import index_write as W
from d3sno import png as P
from d3sno import texture as T
from d3sno import texture_write as TW
from d3sno import x360_texture as XT
from d3sno.binary import FormatError
from test_stage16_cli import X360_GROUPS, run
from test_stage16_texture import random_image, smooth_image, volume_template


class Marker(unittest.TestCase):
    def test_version(self):
        self.assertGreaterEqual(tuple(int(x) for x in d3sno.__version__.split(".")), (0, 16, 1))
        self.assertGreaterEqual(d3sno.STAGE, 16)


class X360PackRule(unittest.TestCase):
    NAMES = list(X360_GROUPS)

    def test_real_name_forms(self):
        cases = {
            "Common.cpk": "CommonTOC.dat", "Act3.cpk": "Act3TOC.dat", "act5": "Act5TOC.dat",
            "CacheCommon.cpk": "CommonTOC.dat", "CacheAct2.cpk": "Act2TOC.dat",
            "enUS_Act3.cpk": "Act3TOC.dat", "enUS_Common.cpk": "CommonTOC.dat",
            "enUS_CacheCommon.cpk": "CommonTOC.dat", "deDE_CacheAct1.cpk": "Act1TOC.dat",
            "C:\\x\\FRFR_ACT4.CPK": "Act4TOC.dat",
        }
        for name, toc in cases.items():
            self.assertEqual(W.x360_pack(name, self.NAMES), toc, name)

    def test_unclear_names_need_pack(self):
        for name in ("Patch.cpk", "enUS_Patch.cpk", "Patch2.cpk", "enUS_Cutscene.cpk", "Act6.cpk",
                     "Act1_enUS.cpk", "Act12.cpk", "CommonLocale.cpk", "Act1x.cpk", "Cache.cpk",
                     "enUS_.cpk", "enUS_enUS_Act1.cpk", "CacheCacheAct1.cpk", "en1S_Act1.cpk", "Data.cpk"):
            with self.assertRaises(FormatError, msg=name):
                W.x360_pack(name, self.NAMES)
        self.assertEqual(W.x360_pack("enUS_Patch.cpk", self.NAMES, pack="Act2"), "Act2TOC.dat")


def x360_cube():
    images = [random_image(8, 8, s) for s in range(6)]
    return TW.encode_texture(images, "x360", 0, levels=1, faces=6)


class X360TextureCli(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, name, data):
        path = self.dir / name
        path.write_bytes(data)
        return path

    def test_strip_of_a_cube_round_trips(self):
        template = self.write("c.tex", x360_cube())
        tex = T.parse_texture(template.read_bytes(), "x360")
        strip = self.dir / "s.png"
        code, _, err = run("texture", template, "-o", strip, "--strip", "--x360-pixels")
        self.assertEqual(code, 0, err)
        image = P.read_png(strip.read_bytes())
        self.assertEqual((image.width, image.height), (8, 48))
        self.assertEqual(image.rgba, b"".join(XT.decode_image(tex, 0, f).rgba for f in range(6)))
        out = self.dir / "o.tex"
        code, _, err = run("texture-import", template, strip, "-o", out, "--strip")
        self.assertEqual(code, 0, err)
        self.assertEqual(out.read_bytes(), template.read_bytes())

    def test_strip_of_a_volume_round_trips(self):
        data, tex = volume_template("x360")
        template = self.write("v.tex", data)
        strip = self.dir / "s.png"
        self.assertEqual(run("texture", template, "-o", strip, "--strip", "--x360-pixels")[0], 0)
        self.assertEqual(P.read_png(strip.read_bytes()).height, 64 * 64)
        out = self.dir / "o.tex"
        code, _, err = run("texture-import", template, strip, "-o", out, "--strip")
        self.assertEqual(code, 0, err)
        self.assertEqual(out.read_bytes(), data)

    def test_frame(self):
        data = TW.encode_texture(smooth_image(16, 16), "x360", 0, levels=2,
                                 frames=[(0, 0.0, 0.0, 0.5, 0.5), (0, 0.5, 0.25, 1.0, 1.0)])
        template = self.write("f.tex", data)
        tex = T.parse_texture(data, "x360")
        for index, level in ((0, 0), (1, 0), (1, 1)):
            out = self.dir / f"f{index}{level}.png"
            code, _, err = run("texture", template, "-o", out, "--frame", index, "--level", level,
                               "--x360-pixels")
            self.assertEqual(code, 0, err)
            rect = tex.frames[index].rect(tex.width, tex.height)
            want = T.crop(XT.decode_image(tex, level), *(v >> level for v in rect))
            self.assertEqual(P.read_png(out.read_bytes()), want)
        self.assertEqual(run("texture", template, "-o", self.dir / "x.png", "--frame", 2, "--x360-pixels")[0], 2)

    def test_normal(self):
        data = TW.encode_texture(random_image(8, 8, 3), "x360", T.FORMAT_NORMAL, levels=1)
        template = self.write("n.tex", data)
        tex = T.parse_texture(data, "x360")
        out = self.dir / "n.png"
        code, _, err = run("texture", template, "-o", out, "--normal", "--x360-pixels")
        self.assertEqual(code, 0, err)
        plain = XT.decode_image(tex, 0)
        got = P.read_png(out.read_bytes()).rgba
        self.assertEqual(got[0::4], plain.rgba[3::4])
        self.assertEqual(got[1::4], plain.rgba[1::4])
        self.assertEqual(set(got[2::4]), {0})
        self.assertEqual(set(got[3::4]), {255})
        other = self.write("o.tex", x360_cube())
        self.assertEqual(run("texture", other, "-o", self.dir / "x.png", "--normal", "--x360-pixels")[0], 2)

    def test_unhonoured_options_refused(self):
        for build in ("ps3", "x360"):
            template = self.write(f"c_{build}.tex", TW.encode_texture(
                [random_image(4, 4, s) for s in range(6)], build, 0, levels=1, faces=6))
            out = self.dir / "r.png"
            for extra in (("--strip", "--all-levels"), ("--strip", "--frame", "0"), ("--frame", "0", "--normal"),
                          ("--strip", "--face", "2"), ("--frame", "0", "--face", "1"), ("--normal", "--slice", "1")):
                code, _, err = run("texture", template, "-o", out, *extra, "--x360-pixels")
                self.assertEqual(code, 2, (build, extra))
                self.assertTrue(err.startswith("error:"), err)
                self.assertFalse(out.exists())

    def test_x360_still_needs_pixels_option(self):
        template = self.write("c.tex", x360_cube())
        self.assertEqual(run("texture", template, "-o", self.dir / "s.png", "--strip")[0], 2)

    def test_other_builds_unchanged(self):
        """The shared path writes what the library functions give, as before."""
        data = TW.encode_texture([random_image(4, 4, s) for s in range(6)], "switch", 0, levels=1, faces=6)
        template = self.write("c.tex", data)
        tex = T.parse_texture(data, "switch")
        out = self.dir / "s.png"
        self.assertEqual(run("texture", template, "-o", out, "--strip")[0], 0)
        self.assertEqual(P.read_png(out.read_bytes()), T.vertical_strip(T.decode_surfaces(tex, 0)))


class MeshImportHelp(unittest.TestCase):
    def test_no_indices_note(self):
        """V21-4: the mesh-import documentation says how to empty a draw from an editor."""
        from d3sno import cli

        text = cli.make_parser()._subparsers._group_actions[0].choices["mesh-import"].format_help()
        self.assertIn("--missing empty", text)
        self.assertIn("no_indices", text)


if __name__ == "__main__":
    unittest.main()
