"""Stage 24 (``58_stage24_links.md``): the graph of links between game files (W24-1 to W24-9). Every file is
synthetic, built from the rows of ``f58_layouts.tsv``; game folders and outputs go to a temporary folder outside
any git working tree."""

import _libpath  # noqa: F401  (python -I: the library folder on sys.path)
import contextlib
import io
import json
import os
import struct
import unittest
from pathlib import Path

import builders as B
import export_fixture as F
import test_stage23 as T
import d3sno
from d3sno import balance as G
from d3sno import cli
from d3sno import effects as E
from d3sno import index_write as IW
from d3sno import links as K
from d3sno import sno_layout as L
from d3sno.binary import FormatError, build_order
from d3sno.datafiles import DATA
from d3sno.index import TocEntry
from d3sno.sno import group_by_extension

BUILDS = ("switch", "x360", "ps3")
GID = {ext: group_by_extension(ext).id for ext in ("acr", "app", "glo", "mat", "tex", "rcp", "gam", "ans")}
A1, A2, APP, GLO, MAT, TEX, RCP, GAM_ITEMS, GAM_AFFIX, ANS = 4001, 4002, 4003, 4004, 4005, 4006, 4007, 4008, 4009, 4010
PROJECTILE = 0x1025A  # actor tag "Projectile Sno" (AXETAGTYPE_ACTOR, group 1)
SWORD, FIRE, LOST = "Sword_Of_Tests", "Fire_Affix", "No_Such_Row"


def row(build, ext, record, name):
    return next(r for r in E.leaf_table(build, ext) if r.record == record and r.name == name)


def put(tree, build, record, name, value):
    leaf = row(build, tree.ext, record, name)
    struct.pack_into(build_order(build) + "i", tree.top.raw, leaf.offset, value)


def tag_map(build, records):
    order = build_order(build)
    out = struct.pack(order + "I", len(records))
    for tag, value in records:
        out += struct.pack(order + "IIi", 2, tag, value)
    return out


def slot_at(build, ext, record, offset):
    return next(s for s in L.pair_slots(build, ext, record) if s.offset == offset)


def actor(build, handle, appearance, tags, anim_set=-1):
    tree = T.blank(build, "acr", handle)
    top = E._top_table_record(build, "acr")
    put(tree, build, top, "snoAppearance", appearance)
    put(tree, build, top, "snoAnimSet", anim_set)
    put(tree, build, top, "snoPhysMesh", 0)  # W24-3: 0 is no link
    slot = next(s for s in L.pair_slots(build, "acr", L.top_record(build, "acr")) if s.field.kind == "tagmap")
    tree.top.pairs[slot.offset] = L.PairValue(L.Block(data=tag_map(build, tags)))
    return L.write_tree(tree, derive=True)


def global_file(build, actors):
    tree = T.blank(build, "glo", GLO)
    bases = row(build, "glo", E._top_table_record(build, "glo"), "tBannerParams.arActorBases")
    data = b"".join(struct.pack(build_order(build) + "i", a) for a in actors)
    tree.top.pairs[bases.pair_offset] = L.PairValue(L.Block(data=data))
    return L.write_tree(tree, derive=True)


def material(build, texture):
    tree = T.blank(build, "mat", MAT)
    top = E._top_table_record(build, "mat")
    lst = row(build, "mat", top, "tUberMaterial.ptMatTexList")
    slot = slot_at(build, "mat", L.top_record(build, "mat"), lst.pair_offset)
    element = slot.field.element
    raw = bytearray(L.record_layout(build, "mat", element).size)
    struct.pack_into(build_order(build) + "i", raw, row(build, "mat", lst.element, "tMatTex.snoTex").offset, texture)
    tree.top.pairs[slot.offset] = L.PairValue(L.Block(elements=[L.Record(element, raw)]))
    return L.write_tree(tree, derive=True)


def recipe(build, item, affixes, count):
    tree = T.blank(build, "rcp", RCP)
    top = E._top_table_record(build, "rcp")
    put(tree, build, top, "tSpecificItem.gbidItem", item)
    put(tree, build, top, "tSpecificItem.nNumAffixes", count)
    for k, value in enumerate(affixes):
        put(tree, build, top, f"tSpecificItem.gbidAffixes[{k}]", value)
    return L.write_tree(tree, derive=True)


def gam(build, kind, names, handle):
    _, sw_slot, sw_row, ps3_slot, ps3_row = G.TABLES[kind]
    slot, size = (ps3_slot, ps3_row) if build == "ps3" else (sw_slot, sw_row)
    return F.with_handle(B.build_gam(build, kind, slot, list(names), size), handle, build_order(build))


def game_files():
    """Two actors that name each other by a tag (a cycle), a global file naming both in an array, an actor's
    appearance, a material naming a texture, a recipe naming balance rows (one only in another type's table)."""
    b = "switch"
    toc = IW.write_core_toc([
        TocEntry(GID["acr"], A1, "Hero"), TocEntry(GID["acr"], A2, "Arrow"), TocEntry(GID["app"], APP, "HeroLook"),
        TocEntry(GID["glo"], GLO, "Globals"), TocEntry(GID["mat"], MAT, "Shiny"), TocEntry(GID["tex"], TEX, "Shiny_D"),
        TocEntry(GID["rcp"], RCP, "Forge"), TocEntry(GID["gam"], GAM_ITEMS, "Items"),
        TocEntry(GID["gam"], GAM_AFFIX, "Affixes")], b)
    return {
        "Common": {
            F.name_of("acr", "Hero"): actor(b, A1, APP, [(PROJECTILE, A2)], anim_set=ANS),
            F.name_of("acr", "Arrow"): actor(b, A2, -1, [(PROJECTILE, A1)]),
            F.name_of("glo", "Globals"): global_file(b, [A1, A2, 0]),
            F.name_of("mat", "Shiny"): material(b, TEX),
            F.name_of("tex", "Shiny_D"): b"not read: a type with no links",
            F.name_of("rcp", "Forge"): recipe(b, G.gbid(SWORD), [G.gbid(FIRE), G.gbid(LOST), K.FILLER], 2),
            F.name_of("gam", "Items"): gam(b, 2, [SWORD, FIRE], GAM_ITEMS),
            "CoreTOC.dat": toc,
        },
        "Zeta": {F.name_of("acr", "Hero"): actor(b, A1, -1, [])},  # a second copy: the first archive in order wins
    }


def node(ext, handle):
    return ("sno", GID[ext], handle)


# --- W24-1 to W24-3 ---------------------------------------------------------------------------------------

class Tables(unittest.TestCase):
    def test_f58(self):
        self.assertIn(E.LAYOUT_TABLE, ("f58_layouts.tsv", "f61_layouts.tsv"))  # ``61``: f61 replaces f58
        self.assertEqual(E.ENUM_TABLE, "f58_enums.tsv")
        for name in ("f55_layouts.tsv", "f56_layouts.tsv", "f57_layouts.tsv", "f57_enums.tsv", "f58_enums.tsv"):
            self.assertTrue((DATA / name).is_file(), name)
        self.assertEqual(len(K.link_types()), 59)
        for ext in ("acr", "mon", "mat", "lvl", "scn", "wrl", "mrk", "app", "ani", "ans", "rcp", "phm"):
            self.assertIn(ext, K.link_types())

    def test_link_rows(self):
        """W24-1: 805 / 801 / 640 link rows, 388 of them open on PS3."""
        counts = {}
        for (build, ext), rows in E._rows().items():
            for leaf in rows:
                kind = leaf.element if leaf.is_array else leaf.type
                if kind in K.LINK_TYPES:
                    total, open_ = counts.get(build, (0, 0))
                    counts[build] = (total + 1, open_ + (leaf.offset is None))
        self.assertEqual(counts, {"switch": (805, 0), "x360": (801, 0), "ps3": (640, 388)})

    def test_gbid_rows_stay_numbers_in_effects(self):
        """A ``GBID`` row's ``sno_group`` is a balance type: the effects JSON keeps the number (stage 23)."""
        for build in ("switch", "x360"):
            leaf = row(build, "scg", "SceneGroupItem", "gbidLabel")
            self.assertEqual((leaf.type, leaf.group), ("GBID", 11))
            self.assertIsNone(E._named(leaf).group)
            schema = {(r["record"], r["leaf"]): r for r in E.schema("scg")[build]}
            self.assertIsNone(schema[("SceneGroupItem", "gbidLabel")]["link_group"])
        # an SNO row that gained its group in ``f58`` is a link (``aib`` ``arAIState``)
        leaf = row("switch", "aib", E._top_table_record("switch", "aib"), "arAIState[0]")
        self.assertEqual(E._named(leaf).group, 4)


class LinksOf(unittest.TestCase):
    def test_every_top_record_link_row_is_read(self):
        """Every placed link row of each top record gives one ``field`` link (but rows in a layout conflict)."""
        for build in BUILDS:
            for ext in K.link_types():
                with self.subTest(build=build, ext=ext):
                    tree = T.blank(build, ext, 77)
                    top = E._top_table_record(build, ext)
                    view = E._view(build, ext, top, L.top_record(build, ext))
                    conflicts = {leaf.name for leaf in view.conflicts}
                    rows = [r for r in E.leaf_table(build, ext) if r.record == top and not r.is_array
                            and K._link_type(r) and r.name not in conflicts]
                    for i, leaf in enumerate(rows):
                        struct.pack_into(build_order(build) + "i", tree.top.raw, leaf.offset, 5000 + i)
                    found = K.links_of(L.write_tree(tree, derive=True), ext, build)
                    want = {(f"record.{leaf.name}", "field",
                             ("gbid" if leaf.type == "GBID" else "sno", leaf.group, 5000 + i)) for i, leaf in enumerate(rows)}
                    self.assertEqual({(l.path, l.kind, l.target) for l in found}, want)

    def test_field_array_tag(self):
        for build in BUILDS:
            with self.subTest(build=build):
                found = K.links_of(actor(build, A1, APP, [(PROJECTILE, A2), (0x10007, 5)], anim_set=ANS), "acr", build)
                self.assertIn(K.Link("record.snoAppearance", "field", node("app", APP)), found)
                self.assertIn(K.Link("record.snoAnimSet", "field", node("ans", ANS)), found)
                tags = [l for l in found if l.kind == "tag"]
                self.assertEqual(len(tags), 1)  # the float scale tag is no link
                self.assertEqual(tags[0].target, node("acr", A2))
                self.assertTrue(tags[0].path.startswith("record."))
                self.assertTrue(tags[0].path.endswith(".Projectile Sno"))
                self.assertFalse([l for l in found if l.path.endswith("snoPhysMesh")])  # 0
        found = K.links_of(global_file("switch", [A1, -1, A2]), "glo", "switch")
        self.assertEqual([(l.path, l.kind, l.target) for l in found],
                         [("record.tBannerParams.arActorBases[0]", "array", node("acr", A1)),
                          ("record.tBannerParams.arActorBases[2]", "array", node("acr", A2))])

    def test_element_record(self):
        for build in ("switch", "x360"):  # the PS3 rows have no texture link (none placed)
            with self.subTest(build=build):
                found = K.links_of(material(build, TEX), "mat", build)
                self.assertEqual([(l.path, l.kind, l.target) for l in found],
                                 [("record.tUberMaterial.ptMatTexList[0].tMatTex.snoTex", "field", node("tex", TEX))])
                self.assertEqual(K.links_of(material(build, -1), "mat", build), [])

    def test_empty_and_filler(self):
        """W24-3: −1, 0 and the affix filler are no links (``59`` AH2: in any affix slot, whatever the count)."""
        item, fire = G.gbid(SWORD), G.gbid(FIRE)
        found = K.links_of(recipe("switch", item, [fire, K.FILLER, K.FILLER, -1, 0], 1), "rcp", "switch")
        self.assertEqual([(l.path, l.target) for l in found],
                         [("record.tSpecificItem.gbidItem", ("gbid", 2, item)),
                          ("record.tSpecificItem.gbidAffixes[0]", ("gbid", 8, fire))])
        found = K.links_of(recipe("switch", -1, [K.FILLER] * 6, 0), "rcp", "switch")
        self.assertEqual(found, [])
        found = K.links_of(recipe("switch", 0, [K.FILLER], 1), "rcp", "switch")
        self.assertEqual(found, [])  # ``59`` AH2: before the count too

    def test_no_table_no_links(self):
        self.assertEqual(K.links_of(b"anything", "tex"), [])
        with self.assertRaises(FormatError):
            K.links_of(b"\0" * 8, "acr", "switch")

    def test_shared_block_read_once(self):
        """Q1013: a block two pairs point at is read once, at its first pair."""
        tree = T.blank("switch", "glo", GLO)
        top = E._top_table_record("switch", "glo")
        block = L.Block(data=struct.pack("<2i", A1, A2))
        for name in ("tBannerParams.arActorBases", "tBannerParams.arActorCaps"):
            tree.top.pairs[row("switch", "glo", top, name).pair_offset] = L.PairValue(block)
        reader = K._Reader("switch", "glo")
        reader.record(tree.top, top, ("record",))
        self.assertEqual([l.path for l in reader.out],
                         ["record.tBannerParams.arActorBases[0]", "record.tBannerParams.arActorBases[1]"])


class Building(unittest.TestCase):
    def test_edges_sorted(self):
        """W24-8: edges in a stable order however they were added."""
        index = K.LinkIndex("switch")
        for source, path in ((node("mat", 9), "record.b"), (node("acr", 5), "record.z"), (node("acr", 5), "record.a")):
            index.add_edge(K.Edge(source, path, node("tex", 1), "field"))
        index._finish()
        self.assertEqual([(e.source, e.path) for e in index.edges],
                         [(node("acr", 5), "record.a"), (node("acr", 5), "record.z"), (node("mat", 9), "record.b")])
        self.assertEqual([e.path for e in index.used_by(node("tex", 1))], ["record.a", "record.z", "record.b"])

    def test_once_per_handle(self):
        """W24-6: two names with one handle: the file is read once (the first entry in archive order)."""
        root = F.temp_root(self)
        files = game_files()
        del files["Zeta"]
        files["Common"][F.name_of("acr", "Hero_Copy")] = actor("switch", A1, -1, [(PROJECTILE, A2)])
        index = K.build_index(F.write_game(root / "game", files))
        found = sorted(e.path for e in index.edges if e.source == node("acr", A1))
        one = [sorted(l.path for l in K.links_of(files["Common"][F.name_of("acr", stem)], "acr", "switch"))
               for stem in ("Hero", "Hero_Copy")]
        self.assertIn(found, one)  # one copy's links, not both
        self.assertEqual(index.counts()["files_read"], 6)


# --- W24-4 to W24-8 ---------------------------------------------------------------------------------------

class Index(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import tempfile
        from d3sno.export import in_git_tree

        holder = tempfile.TemporaryDirectory(prefix="d3sno_links_")
        cls.addClassCleanup(holder.cleanup)
        cls.root = Path(holder.name)
        if in_git_tree(cls.root) is not None:
            raise unittest.SkipTest("the temporary folder lies inside a git working tree")
        cls.game = F.write_game(cls.root / "game", game_files())
        cls.index = K.build_index(cls.game)

    def test_edges(self):
        index = self.index
        self.assertEqual(index.build, "switch")
        edges = {(e.source, e.path, e.target, e.kind) for e in index.edges}
        self.assertIn((node("acr", A1), "record.snoAppearance", node("app", APP), "field"), edges)
        self.assertIn((node("glo", GLO), "record.tBannerParams.arActorBases[1]", node("acr", A2), "array"), edges)
        self.assertIn((node("mat", MAT), "record.tUberMaterial.ptMatTexList[0].tMatTex.snoTex", node("tex", TEX),
                       "field"), edges)
        tags = [e for e in index.edges if e.kind == "tag"]
        self.assertEqual({(e.source, e.target) for e in tags}, {(node("acr", A1), node("acr", A2)),
                                                                (node("acr", A2), node("acr", A1))})
        self.assertEqual(len(index.edges), len(edges))

    def test_nodes(self):
        index = self.index
        hero = index.nodes[node("acr", A1)]
        self.assertEqual(hero.name, "Hero")
        self.assertEqual(hero.archives, ["Common", "Zeta"])
        self.assertTrue(hero.read)
        texture = index.nodes[node("tex", TEX)]
        self.assertFalse(texture.read)  # W24-6: a type with no top record is a node without outgoing links
        self.assertEqual(index.uses(node("tex", TEX)), [])
        self.assertTrue(index.is_resolved(node("tex", TEX)))
        self.assertFalse(index.is_resolved(node("ans", ANS)))  # a target not in the CoreTOC
        self.assertIsNone(index.nodes[node("ans", ANS)].name)

    def test_first_copy(self):
        """W24-6: one file per handle, the first copy in sorted archive-name order (Common before Zeta, whose
        copy names nothing)."""
        uses = {e.path for e in self.index.uses(node("acr", A1))}
        self.assertEqual(uses, {"record.snoAppearance", "record.snoAnimSet", "record.hTagMap.Projectile Sno"})
        self.assertEqual([e.source for e in self.index.used_by(node("acr", A1))],
                         [node("acr", A2), node("glo", GLO)])

    def test_gbid(self):
        """W24-4: a GBID resolves through a ``.gam`` row name; another type's table gives ``other_type``."""
        index = self.index
        item = ("gbid", 2, G.gbid(SWORD))
        self.assertTrue(index.is_resolved(item))
        self.assertEqual((index.nodes[item].name, index.nodes[item].gam), (SWORD, GAM_ITEMS))
        edges = {e.path: e for e in index.uses(node("rcp", RCP))}
        self.assertEqual(edges["record.tSpecificItem.gbidItem"].target, item)
        self.assertIsNone(edges["record.tSpecificItem.gbidItem"].other_type)
        fire = edges["record.tSpecificItem.gbidAffixes[0]"]
        self.assertEqual((fire.target, fire.other_type), (("gbid", 2, G.gbid(FIRE)), 2))
        lost = edges["record.tSpecificItem.gbidAffixes[1]"]
        self.assertFalse(index.is_resolved(lost.target))
        self.assertNotIn("record.tSpecificItem.gbidAffixes[2]", edges)  # the filler after the count

    def test_unresolved(self):
        index = self.index
        missing = {(e.source, e.path, e.target) for e in index.unresolved()}
        self.assertEqual(missing, {(node("rcp", RCP), "record.tSpecificItem.gbidAffixes[1]", ("gbid", 8, G.gbid(LOST))),
                                   (node("acr", A1), "record.snoAnimSet", node("ans", ANS))})
        counts = index.counts()
        self.assertEqual((counts["gbid_resolved"], counts["gbid_unresolved"], counts["gbid_other_type"]), (2, 1, 1))
        self.assertEqual(counts["sno_unresolved"], 1)
        self.assertEqual(counts["files_read"], 6)  # 2 actors, global, material, recipe, balance table

    def test_needs_cycle(self):
        index = self.index
        reached = {r.node: r for r in index.needs(node("glo", GLO))}
        self.assertEqual(set(reached), {node("acr", A1), node("acr", A2), node("app", APP), node("ans", ANS)})
        self.assertEqual(reached[node("acr", A1)].depth, 1)
        self.assertEqual(reached[node("app", APP)].depth, 2)
        back = {r.node for r in index.needed_by(node("acr", A1))}
        self.assertEqual(back, {node("acr", A2), node("glo", GLO)})
        cycle = index.needs(node("acr", A2))
        # the cycle back to A2 is visited once; A1's own links follow
        self.assertEqual([r.node for r in cycle], [node("acr", A1), node("ans", ANS), node("app", APP)])
        self.assertEqual([r.node for r in index.needs(node("acr", A2), depth=1)], [node("acr", A1)])
        self.assertEqual(index.needs(node("glo", GLO), depth=1, kinds=["tag"]), [])
        self.assertEqual({r.node for r in index.needs(node("glo", GLO), kinds=["array"])},
                         {node("acr", A1), node("acr", A2)})
        via = reached[node("acr", A1)].via
        self.assertEqual([e.path for e in via], ["record.tBannerParams.arActorBases[0]"])

    def test_find(self):
        index = self.index
        self.assertEqual(index.find("hero"), node("acr", A1))
        self.assertEqual(index.find(A2), node("acr", A2))
        self.assertEqual(index.find(str(MAT)), node("mat", MAT))
        self.assertEqual(index.find("shiny", "mat"), node("mat", MAT))
        self.assertEqual(index.find(SWORD.lower()), ("gbid", 2, G.gbid(SWORD)))
        self.assertEqual(index.find("sno:acr:%d" % A1), node("acr", A1))
        self.assertIsNone(index.find("nobody"))
        self.assertEqual(index.uses("Hero"), index.uses(node("acr", A1)))

    def test_json_stable(self):
        """W24-8: two runs give equal output; nodes by group then handle, edges by source then path."""
        first = self.index.to_json()
        second = K.build_index(self.game).to_json()
        self.assertEqual(first, second)
        self.assertEqual(set(first), {"build", "nodes", "edges", "unresolved", "unread_files", "counts"})
        keys = [(n["kind"], n.get("group_id", n.get("type")), n.get("handle", n.get("value"))) for n in first["nodes"]]
        self.assertEqual(keys, sorted(keys))
        json.dumps(first)
        self.assertEqual(len(first["edges"]), len(self.index.edges))


class Unread(unittest.TestCase):
    def test_broken_file(self):
        root = F.temp_root(self)
        files = game_files()
        files["Common"][F.name_of("acr", "Arrow")] = actor("switch", A2, -1, [])[:0x30]
        del files["Zeta"]
        index = K.build_index(F.write_game(root / "game", files))
        broken = index.nodes[node("acr", A2)]
        self.assertTrue(broken.unread)
        self.assertFalse(broken.read)
        self.assertEqual(index.uses(node("acr", A2)), [])
        self.assertIn(node("acr", A2), {e.target for e in index.used_by(node("acr", A2))} | {node("acr", A2)})
        self.assertEqual(index.counts()["unread"], 1)
        self.assertIn("unread", index.node_json(node("acr", A2)))

    def test_refused(self):
        root = F.temp_root(self)
        with self.assertRaises(K.LinkRefused):
            K.build_index(root / "nothing")
        (root / "empty").mkdir()
        with self.assertRaises(K.LinkRefused):
            K.build_index(root / "empty")


class Command(unittest.TestCase):
    def run_cli(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cli.main(["links", *map(str, argv)])
        return code, out.getvalue(), err.getvalue()

    def test_queries_and_out(self):
        root = F.temp_root(self)
        game = F.write_game(root / "game", game_files())
        target = root / "links.json"
        code, out, err = self.run_cli(game, "--out", target, "--uses", "glo:Globals", "--needed-by", "Hero",
                                      "--unresolved", "--depth", "1")
        self.assertEqual(code, 0, err)
        self.assertIn("SNO links", out.splitlines()[0])
        self.assertIn("GBID links 2 resolve / 1 do not", out.splitlines()[0])
        self.assertIn("record.tBannerParams.arActorBases[0]", out)
        self.assertIn("needed-by sno:acr:%d Hero" % A1, out)
        self.assertIn("unresolved sno:rcp:%d" % RCP, out)
        self.assertEqual(json.loads(target.read_text(encoding="utf-8")), K.build_index(game).to_json())
        code, out, err = self.run_cli(game, "--needs", "nobody")
        self.assertEqual(code, 2)

    def test_git_tree_refused(self):
        root = F.temp_root(self)
        game = F.write_game(root / "game", game_files())
        (root / "repo" / ".git").mkdir(parents=True)
        target = root / "repo" / "links.json"
        code, _, err = self.run_cli(game, "--out", target)
        self.assertEqual(code, 2)
        self.assertIn("git working tree", err)
        self.assertFalse(target.exists())
        code, _, err = self.run_cli(game, "--out", target, "--force")
        self.assertEqual(code, 0, err)
        self.assertTrue(target.exists())


class TestAssets(unittest.TestCase):
    """W24-9: ``links_of`` on the project's own test assets (``test_assets/``) finds the material's texture
    link. The folder is looked for beside the library and above it; without it the test is skipped.

    ``60``: the folder holds only sources and ``generate.py``. The folder is copied to a temporary folder (outside
    any git working tree) and ``generate.py`` is run there, with the library on ``PYTHONPATH`` and the output
    folder as its argument (once more without an argument if it refuses one); the ``.mat`` files are looked for
    in the whole temporary folder (Q1040)."""

    def test_material_texture(self):
        import shutil
        import subprocess
        import sys
        import tempfile

        here = Path(__file__).resolve().parent
        folder = next((p / "test_assets" for p in (here, *here.parents[:3]) if (p / "test_assets").is_dir()), None)
        if folder is None:
            self.skipTest("no test_assets folder")
        work = Path(tempfile.mkdtemp(prefix="d3sno_assets_"))
        self.addCleanup(shutil.rmtree, work, True)
        copy, out = work / "test_assets", work / "out"
        shutil.copytree(folder, copy)
        out.mkdir()
        before = {p for p in work.rglob("*") if p.is_file()}
        script = copy / "generate.py"
        if script.is_file():
            env = dict(os.environ)
            env["PYTHONPATH"] = os.pathsep.join([str(here.parent)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH")
                                                                       else []))
            runs = []
            for argv in ([str(out)], []):
                run = subprocess.run([sys.executable, "-B", str(script), *argv], cwd=copy, env=env,
                                     capture_output=True, text=True, timeout=900)
                runs.append(run)
                if run.returncode == 0:
                    break
            if runs[-1].returncode != 0:
                self.fail(f"generate.py failed: {runs[-1].stderr[-2000:]}")
            if not {p for p in work.rglob("*") if p.is_file()} - before:
                self.fail("generate.py wrote no file into the temporary folder")
        materials = sorted(p for p in work.rglob("*") if p.is_file() and p.suffix.lower() == ".mat")
        if not materials:
            self.skipTest("no material in the generated test assets")
        for path in materials:
            with self.subTest(path=str(path.relative_to(work))):
                if "ps3" in {p.lower() for p in path.relative_to(work).parts}:
                    # Repository note: the PS3 MaterialTextureEntry has no texture row in the layout table yet,
                    # so a PS3 material links no texture (README, known limits). Open for the next corrections.
                    continue
                found = K.links_of(path.read_bytes(), "mat")
                self.assertTrue([l for l in found if l.kind == "field" and l.target[:2] == ("sno", GID["tex"])],
                                found)


class Version(unittest.TestCase):
    def test_version(self):
        self.assertGreaterEqual(tuple(int(p) for p in d3sno.__version__.split(".")), (0, 24, 0))
        self.assertGreaterEqual(d3sno.STAGE, 24)


if __name__ == "__main__":
    unittest.main()
