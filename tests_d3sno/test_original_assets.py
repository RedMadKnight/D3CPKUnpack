"""The original test assets of ``test_assets/``: built for every build with the d3sno writers,
read back in strict mode and written back byte for byte. No game data is used."""

import _libpath  # noqa: F401  (python -I: the library folder on sys.path)
import importlib.util
import os
import unittest

from d3sno import animation as A
from d3sno import anim_write as AW
from d3sno import archive as R
from d3sno import collision as C
from d3sno import collision_write as CW
from d3sno import index as I
from d3sno import index_write as IW
from d3sno import material as M
from d3sno import material_write as MW
from d3sno import sno_layout as L
from d3sno import texture as T
from d3sno import write as W
from d3sno.png import read_png
from d3sno.sno import hash33, parse_header

_PATH = os.path.join(_libpath.LIBRARY, "test_assets", "generate.py")
_SPEC = importlib.util.spec_from_file_location("test_assets_generate", _PATH)
G = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(G)

SOURCES = G.read_sources()
BUILT = G.build_all(SOURCES)


def path(ext):
    return G.identity(ext).path


class Sources(unittest.TestCase):
    def test_committed_sources_equal_the_generator(self):
        made = G.make_sources()
        self.assertEqual(sorted(made), sorted(G.SOURCE_FILES))
        for name in G.SOURCE_FILES:
            with self.subTest(name):
                self.assertTrue(G.same_source(name, made[name], SOURCES[name]))

    def test_generator_is_deterministic(self):
        self.assertEqual(G.make_sources(), G.make_sources())

    def test_png_is_the_checker(self):
        image = read_png(SOURCES["checker_64.png"])
        self.assertEqual((image.width, image.height), (64, 64))
        self.assertEqual(image.rgba, G.checker_pixels().rgba)

    def test_meshes_are_closed_boxes(self):
        for name in ("box.obj", "box_collision.obj"):
            positions, triangles = G.parse_obj(SOURCES[name])
            self.assertEqual((len(positions), len(triangles)), (8, 12))
            edges = {}
            for t in triangles:
                for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
                    edges[(a, b)] = edges.get((a, b), 0) + 1
            # every directed edge once and its reverse once: closed and consistently wound
            self.assertTrue(all(n == 1 and edges.get((b, a)) == 1 for (a, b), n in edges.items()))

    def test_string_table_labels_are_unique(self):
        rows = G.parse_strings(SOURCES["strings.tsv"])
        self.assertEqual(rows, list(G.STRINGS))
        self.assertEqual(len({hash33(label) for label, _ in rows}), len(rows))


class Builds(unittest.TestCase):
    def test_every_build_is_deterministic(self):
        again = G.build_all(G.read_sources())
        for build in G.BUILDS:
            with self.subTest(build):
                self.assertEqual(sorted(again[build]), sorted(BUILT[build]))
                for name, data in BUILT[build].items():
                    self.assertEqual(again[build][name], data, name)

    def test_expected_files(self):
        for build in G.BUILDS:
            names = set(BUILT[build])
            expected = {path("tex"), path("mat"), path("phm"), "CoreTOC.dat", "test_assets.cpk"}
            if build == "switch":
                expected.add(path("ani"))
            self.assertEqual(names, expected, build)

    def test_handles_and_builds(self):
        for build in G.BUILDS:
            for ext in G.BUILDERS:
                data = BUILT[build].get(path(ext))
                if data is None:
                    continue
                with self.subTest(build=build, ext=ext):
                    header = parse_header(data, build)
                    self.assertEqual(header.build, build)
                    self.assertEqual(header.handle, G.HANDLES[ext])


class Texture(unittest.TestCase):
    def test_strict_read_and_byte_round_trip(self):
        for build in G.BUILDS:
            with self.subTest(build):
                data = BUILT[build][path("tex")]
                tex = T.parse_texture(data, build, strict=True)
                self.assertEqual((tex.width, tex.height), (64, 64))
                self.assertEqual(W.write_tex(tex), data)
                self.assertEqual(W.write_tex(T.parse_texture(W.write_tex(tex), build, strict=True)), data)

    def test_top_level_decodes_to_the_source(self):
        for build in ("ps3", "switch"):  # d3sno does not decode the x360 pixel layout
            with self.subTest(build):
                tex = T.parse_texture(BUILT[build][path("tex")], build, strict=True)
                self.assertEqual(T.decode_image(tex).rgba, G.checker_pixels().rgba)


class Material(unittest.TestCase):
    def test_read_and_byte_round_trip(self):
        for build in G.BUILDS:
            with self.subTest(build):
                data = BUILT[build][path("mat")]
                self.assertTrue(L.is_canonical(data, "mat", build))
                self.assertEqual(L.write_tree(MW.material_tree(data, build)), data)
                self.assertEqual(MW.material_zero_problems(data, build), [])
                read = M.parse_material(data, build)
                self.assertEqual(read.header.handle, G.HANDLES["mat"])

    def test_texture_link_resolves_in_the_built_toc(self):
        for build in G.BUILDS:
            with self.subTest(build):
                toc = I.parse_core_toc(BUILT[build]["CoreTOC.dat"], build)
                data = BUILT[build][path("mat")]
                self.assertEqual(MW.check_material_links(data, toc, build), [])
                tree = MW.material_tree(data, build)
                self.assertEqual([e[:2] for e in MW.material_textures(tree)],
                                 [(M.SLOT_DIFFUSE, G.HANDLES["tex"])])


class Collision(unittest.TestCase):
    def test_strict_read_and_byte_round_trip(self):
        for build in G.BUILDS:
            with self.subTest(build):
                data = BUILT[build][path("phm")]
                phm = C.parse_collision(data, build, strict=True)
                self.assertEqual(len(phm.meshes), 1)
                self.assertEqual(len(phm.meshes[0].triangles), 12)
                self.assertEqual(CW.write_collision(phm), data)

    def test_derived_values_agree(self):
        for build in G.BUILDS:
            with self.subTest(build):
                data = BUILT[build][path("phm")]
                self.assertEqual(CW.write_collision(C.parse_collision(data, build, strict=True), derive=True), data)


class Animation(unittest.TestCase):
    def test_strict_read_and_byte_round_trip(self):
        data = BUILT["switch"][path("ani")]
        ani = A.parse_ani(data, "switch", strict=True)
        self.assertEqual(len(ani.permutations), 1)
        perm = ani.permutations[0]
        self.assertEqual((perm.name, perm.keyframes, perm.track_hashes),
                         (G.NAMES["ani"], 31, [hash33("TA_Root")]))
        self.assertEqual(AW.write_ani(ani.permutations, "switch", head=data), data)

    def test_curves_decode_to_the_source_keys(self):
        ani = A.parse_ani(BUILT["switch"][path("ani")], "switch", strict=True)
        curves = A.decode_permutation(ani.permutations[0])
        track = G.json.loads(SOURCES["anim_box_bob.json"])["tracks"][0]
        for kind in ("translation", "rotation", "scale"):
            with self.subTest(kind):
                curve = getattr(curves, kind)[0]
                self.assertEqual(curve.frames, [k[0] for k in track[kind]])
                for got, want in zip(curve.values, (k[1] for k in track[kind])):
                    for g, w in zip(got, want):
                        self.assertAlmostEqual(g, w, places=3)


class IndexAndArchive(unittest.TestCase):
    def test_core_toc_round_trip(self):
        for build in G.BUILDS:
            with self.subTest(build):
                data = BUILT[build]["CoreTOC.dat"]
                toc = I.parse_core_toc(data, build)
                keys = {(e.group, e.handle, e.name) for e in toc.entries}
                for ext in G.BUILDERS:
                    if path(ext) in BUILT[build]:
                        e = G.identity(ext).entry
                        self.assertIn((e.group, e.handle, e.name), keys)
                self.assertEqual(IW.write_core_toc(toc, build), data)

    def test_archive_reads_back_and_rebuilds_byte_for_byte(self):
        for build in G.BUILDS:
            with self.subTest(build):
                data = BUILT[build]["test_assets.cpk"]
                arc = R.read_archive(data, strict=True)
                self.assertEqual(arc.header.build, build)
                self.assertTrue(R.verify_archive(data).ok)
                files = {k: v for k, v in BUILT[build].items() if "\\" in k}
                self.assertEqual({f.name for f in arc.files}, set(files))
                for name, raw in files.items():
                    self.assertEqual(arc.read_file(name), raw)
                self.assertEqual(R.rebuild(data).data, data)


if __name__ == "__main__":
    unittest.main()
