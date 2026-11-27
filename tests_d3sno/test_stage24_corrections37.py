"""Corrections 37 (``59_corrections37.md``): AH1 (appearance links from the appearance decoder) and AH2 (the
affix filler of each build). Synthetic files only; nothing is written to disk."""

import _libpath  # noqa: F401  (python -I: the library folder on sys.path)
import struct
import unittest

import test_stage17_writer as W
import test_stage24 as S
import d3sno
from d3sno import effects as E
from d3sno import links as K
from d3sno import sno_layout as L
from d3sno.binary import build_order
from d3sno.sno import group_by_extension

MAT, TEX = group_by_extension("mat").id, group_by_extension("tex").id
X360_FILLER = 0x73696C00


def appearance(build):
    """A mesh with two looks and one material: look 0 has textures 5 and -1 (empty), look 1 texture 6; the
    builder gives the looks the material handles 900 and 901."""
    if build == "ps3":
        data, _ = W.ps3_file()
    else:
        data, _ = W.switch_skin(">" if build == "x360" else "<")
    return W.with_materials(data, None if build == "switch" else build, looks=(0x61, 0x62),
                            entries=[(0xAAA, [[(1, 5), (2, -1)], [(1, 6)]])], assign=(0xAAA,))


def with_table_rows(data, build, material, texture):
    """The file with a value in every ``snoMaterial`` and ``tMatTex.snoTex`` row of the table (Switch, x360)."""
    tree = L.parse_tree(data, "app", build)
    order = build_order(build)
    rows = {}
    for leaf in E.leaf_table(build, "app"):
        if leaf.name in K.APP_DECODED and leaf.offset is not None:
            rows.setdefault(leaf.record, []).append(leaf)
    hit = 0
    for record in L.tree_records(tree):
        for leaf in rows.get(E._table_record(build, "app", record.name), ()):
            struct.pack_into(order + "i", record.raw, leaf.offset, material if leaf.name == "snoMaterial" else texture)
            hit += 1
    return L.write_tree(tree), hit


class AppLinks(unittest.TestCase):
    """AH1."""

    def test_every_build(self):
        for build in S.BUILDS:
            with self.subTest(build=build):
                found = [(l.path, l.kind, l.target) for l in K.links_of(appearance(build), "app", build)]
                self.assertEqual([f for f in found if f[0].startswith("materials")], [
                    ("materials[0].looks[0].material", "field", ("sno", MAT, 900)),
                    ("materials[0].looks[0].textures[0]", "field", ("sno", TEX, 5)),
                    ("materials[0].looks[1].material", "field", ("sno", MAT, 901)),
                    ("materials[0].looks[1].textures[0]", "field", ("sno", TEX, 6)),
                ])

    def test_table_rows_not_read(self):
        for build in ("switch", "x360"):
            with self.subTest(build=build):
                data, hit = with_table_rows(appearance(build), build, 4444, 5555)
                self.assertGreater(hit, 0)
                found = K.links_of(data, "app", build)
                self.assertFalse([l for l in found if l.path.endswith(K.APP_DECODED)])
                # the rows are the decoder's own fields: each value comes once, from the decoder
                values = [l for l in found if l.target[2] in (4444, 5555)]
                self.assertTrue(values)
                self.assertTrue(all(l.path.startswith("materials") for l in values), values)

    def test_other_rows_still_read(self):
        rows = [r for r in E.leaf_table("switch", "app") if K._link_type(r) and r.name not in K.APP_DECODED]
        self.assertTrue(rows)  # cloth, shader map, particle and surface links stay in the table


class ActorNeeds(unittest.TestCase):
    def test_actor_reaches_textures(self):
        """The check of AH1: an actor's ``needs`` goes through its appearance to its materials and textures."""
        import export_fixture as F
        from d3sno import index_write as IW
        from d3sno.index import TocEntry

        root = F.temp_root(self)
        app = F.with_handle(appearance("switch"), S.APP, "<")
        toc = IW.write_core_toc([TocEntry(S.GID["acr"], S.A1, "Hero"), TocEntry(S.GID["app"], S.APP, "HeroLook"),
                                 TocEntry(TEX, 5, "Skin_D"), TocEntry(TEX, 6, "Skin2_D")], "switch")
        files = {"Common": {F.name_of("acr", "Hero"): S.actor("switch", S.A1, S.APP, []),
                            F.name_of("app", "HeroLook"): app, "CoreTOC.dat": toc}}
        index = K.build_index(F.write_game(root / "game", files))
        reached = {r.node for r in index.needs(S.node("acr", S.A1))}
        self.assertTrue({("sno", S.GID["app"], S.APP), ("sno", MAT, 900), ("sno", MAT, 901), ("sno", TEX, 5),
                         ("sno", TEX, 6)} <= reached, reached)
        self.assertTrue(index.is_resolved(("sno", TEX, 5)))


class Fillers(unittest.TestCase):
    """AH2."""

    def affixes(self, build, values, count):
        found = K.links_of(S.recipe(build, -1, values, count), "rcp", build)
        return [(l.path, l.target[2]) for l in found]

    def test_each_build(self):
        self.assertEqual(K.FILLERS, {"switch": 0x5C000000, "x360": X360_FILLER})
        for build, filler, other in (("switch", K.FILLER, X360_FILLER), ("x360", X360_FILLER, K.FILLER)):
            with self.subTest(build=build):
                self.assertEqual(self.affixes(build, [filler] * 6, 6), [])  # whatever the count holds
                self.assertEqual(self.affixes(build, [filler] * 6, 0), [])
                self.assertEqual(self.affixes(build, [other, filler], 2),
                                 [("record.tSpecificItem.gbidAffixes[0]", other)])  # the other build's word is a link

    def test_value_kept(self):
        data = S.recipe("x360", -1, [X360_FILLER], 1)
        top = E._top_table_record("x360", "rcp")
        leaf = S.row("x360", "rcp", top, "tSpecificItem.gbidAffixes[0]")
        tree = L.parse_tree(data, "rcp", "x360")
        self.assertEqual(struct.unpack_from(">I", tree.top.raw, leaf.offset)[0], X360_FILLER)


class Version(unittest.TestCase):
    def test_version(self):
        self.assertGreaterEqual(tuple(int(p) for p in d3sno.__version__.split(".")), (0, 24, 1))
        self.assertEqual(d3sno.STAGE, 24)


if __name__ == "__main__":
    unittest.main()
