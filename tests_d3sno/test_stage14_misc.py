"""Stage 14: ``28_derived_values.md`` W14-18 to W14-23 (marker-set special list, world cluster
counter, count fields, actor bounds, sound-bank rows, font kerning order), synthetic files only."""

import _libpath  # noqa: F401  (python -I: the library folder on sys.path)
import struct
import unittest

import builders as B
from d3sno import FormatError
from d3sno import derived as D
from d3sno import sno_layout as L
from d3sno import world as WO
from test_stage11 import actor_file, font_file, header

BODY = 0x10


def path_of(build, ext, record, element):
    return next(s.path for s in L.pair_slots(build, ext, record) if s.field.element == element)


def new_tree(build, ext, order):
    top = L.top_record(build, ext)
    return L.SnoTree(build, ext, header(order), L.new_record(build, ext, top))


# --- W14-20. Count fields ----------------------------------------------------------------------

class CountTable(unittest.TestCase):
    def test_table_is_package_data(self):
        self.assertTrue((L.DATA / L.COUNT_TABLE).is_file())

    def test_every_proven_row_resolves(self):
        proven = sum(1 for links in L._count_table().values() for link in links if link.proven)
        self.assertEqual(proven, 390)
        rows = sum(len(links) for links in L._count_table().values())
        self.assertEqual(rows, 569)

    def test_links_of_a_record(self):
        links = L.count_links("switch", "mrk", "MarkerSetDefinition")
        self.assertEqual([(k.pair_offset, k.count_offset, k.element_size) for k in links], [(0xB4, 0xB0, 2)])
        # by name across types (FloatPath is listed under .prt, laid out under .lit)
        self.assertTrue(L.count_links("switch", "prt", "FloatPath"))
        # HYPOTHESIS rows are not used unless asked for
        self.assertEqual(L.count_links("switch", "ant", "AnimTreeDefinition"), ())
        self.assertEqual(len(L.count_links("switch", "ant", "AnimTreeDefinition", proven_only=False)), 3)


class DeriveCounts(unittest.TestCase):
    def unchanged_files(self):
        yield "actor", actor_file()[0], "acr", "switch"
        yield "font switch", font_file(), "fnt", "switch"
        yield "font ps3", font_file(">", "ps3"), "fnt", "ps3"
        for build in ("switch", "ps3", "x360"):
            markers = [{"type": 0, "links": [(1, 2), (3, 4)]}, {"type": 16}]
            yield f"mrk {build}", B.build_mrk(build, markers, special=(1,)), "mrk", build

    def test_unchanged_trees_are_identical(self):
        for name, data, ext, build in self.unchanged_files():
            plain = L.write_tree(L.parse_tree(data, ext, build))
            tree = L.parse_tree(data, ext, build)
            self.assertEqual(L.write_tree(tree, derive=True), plain, name)
            self.assertEqual(L.derive_counts(L.parse_tree(data, ext, build)), 0, name)

    def test_stale_words_in_unchanged_lists_are_kept(self):
        # font_file fills the top record's count words with 0x01 bytes; nothing changed, nothing set
        data = font_file()
        out = L.write_tree(L.parse_tree(data, "fnt", "switch"), derive=True)
        self.assertEqual(struct.unpack_from("<I", out, BODY + 0xBC)[0], 0x01010101)

    def test_changed_value_list_sets_the_word(self):
        for build in ("switch", "ps3", "x360"):
            data = B.build_mrk(build, [{"type": 16}], special=(0,))
            order = B.order_of(build)
            path = path_of(build, "mrk", L.top_record(build, "mrk"), "WORD")
            tree = L.parse_tree(data, "mrk", build)
            L.set_block(tree, path, struct.pack(order + "3H", 0, 1, 2))
            stale = L.write_tree(tree)
            self.assertEqual(struct.unpack_from(order + "I", stale, BODY + 0xB0)[0], 1)  # default off
            out = L.write_tree(tree, derive=True)
            self.assertEqual(struct.unpack_from(order + "I", out, BODY + 0xB0)[0], 3, build)
            self.assertEqual(WO.parse_marker_set(out, build).special_indices, [0, 1, 2])

    def test_update_counts_on_set_elements(self):
        data = B.build_mrk("switch", [{"type": 0, "links": [(1, 2)]}])
        tree = L.parse_tree(data, "mrk", "switch")
        links = [L.new_record("switch", "mrk", "MarkerLink") for _ in range(3)]
        L.set_elements(tree, "tMarkerSet[0].ptMarkerLinks", links, update_counts=True)
        marker = tree.top.pairs[0xC].block.elements[0]
        self.assertEqual(struct.unpack_from("<i", marker.raw, 0x40)[0], 3)
        parsed = WO.parse_marker_set(L.write_tree(tree), "switch")
        self.assertEqual(len(parsed.markers[0].links), 3)
        # emptied list: count 0
        L.set_elements(tree, "tMarkerSet[0].ptMarkerLinks", [], update_counts=True)
        self.assertEqual(struct.unpack_from("<i", marker.raw, 0x40)[0], 0)

    def test_set_elements_without_update_keeps_the_word(self):
        data = B.build_mrk("switch", [{"type": 0, "links": [(1, 2)]}])
        tree = L.parse_tree(data, "mrk", "switch")
        L.set_elements(tree, "tMarkerSet[0].ptMarkerLinks", [L.new_record("switch", "mrk", "MarkerLink")] * 2)
        marker = tree.top.pairs[0xC].block.elements[0]
        self.assertEqual(struct.unpack_from("<i", marker.raw, 0x40)[0], 1)

    def test_new_tree_and_animation_counts(self):
        # W14-23: animation counts follow W14-20 (AnimPermutation lists share the word at 0x78)
        tree = new_tree("switch", "ani", "<")
        perms_path = path_of("switch", "ani", "AnimationDefinition", "AnimPermutation")
        L.set_elements(tree, perms_path, [L.new_record("switch", "ani", "AnimPermutation")], update_counts=True)
        self.assertEqual(struct.unpack_from("<I", tree.top.raw, 0x20)[0], 1)
        perm = tree.top.pairs[0x18].block.elements[0]

        def fill(element, count):
            path = f"{perms_path}[0].{path_of('switch', 'ani', 'AnimPermutation', element)}"
            L.set_elements(tree, path, [L.new_record("switch", "ani", element) for _ in range(count)])

        for element in ("BoneName", "TranslationCurve", "RotationCurve", "ScaleCurve"):
            fill(element, 2)
        out = L.write_tree(tree, derive=True)
        self.assertEqual(struct.unpack_from("<I", perm.raw, 0x78)[0], 2)
        again = L.parse_tree(out, "ani", "switch")
        self.assertEqual(L.write_tree(again, derive=True), out)
        fill("ScaleCurve", 3)  # lists sharing one word disagree
        with self.assertRaises(FormatError):
            L.write_tree(tree, derive=True)


# --- W14-18. Marker-set special list -----------------------------------------------------------

def acr_bytes(build, kind, gizmo=None):
    order = B.order_of(build)
    img = B.Image(BODY + 448, order)
    img.raw(0, B.sno_head(1, order))
    img.put(BODY + 0x10, "i", kind)
    tags = [(0, 0x10301, gizmo)] if gizmo is not None else []
    img.pair(BODY + 0x58, B.tag_map(tags + [(1, 0x10007, 1.5)], order))
    return img.done()


MARKERS = [
    {"type": 0, "handle": 101},  # gizmo, type 4: special
    {"type": 1},
    {"type": 16},
    {"type": 28},
    {"type": 29},  # switch and x360 only
    {"type": 55},
    {"type": 0, "handle": 102},  # gizmo, type 5: no
    {"type": 0, "handle": 103},  # monster: no
    {"type": 0, "handle": 999},  # unknown actor: no
    {"type": 30},
    {"type": 0, "handle": 104},  # gizmo, type 70: special
]
ACTORS = {101: (2, 4), 102: (2, 5), 103: (1, 4), 104: (2, 70)}
EXPECTED = {"switch": [0, 2, 3, 4, 5, 10], "x360": [0, 2, 3, 4, 5, 10], "ps3": [0, 2, 3, 5, 10]}


class SpecialList(unittest.TestCase):
    def test_indices(self):
        for build, want in EXPECTED.items():
            marker_set = WO.parse_marker_set(B.build_mrk(build, MARKERS), build)
            self.assertEqual(D.special_indices(marker_set, ACTORS), want, build)
            self.assertEqual(D.special_indices(marker_set, ACTORS.get), want, build)

    def test_actor_files_as_lookup(self):
        files = {h: acr_bytes("switch", kind, gizmo) for h, (kind, gizmo) in ACTORS.items()}
        self.assertEqual(D.actor_gizmo_info(files[101]), (2, 4))
        self.assertEqual(D.actor_gizmo_info(acr_bytes("ps3", 2)), (2, None))
        marker_set = WO.parse_marker_set(B.build_mrk("switch", MARKERS), "switch")
        self.assertEqual(D.special_indices(marker_set, files), EXPECTED["switch"])

    def test_written_back(self):
        for build, want in EXPECTED.items():
            data = B.build_mrk(build, MARKERS, special=(1, 2))
            out = D.derive_special_indices(data, ACTORS, build)
            parsed = WO.parse_marker_set(out, build)
            self.assertEqual(parsed.special_indices, want, build)
            self.assertEqual(len(parsed.markers), len(MARKERS))
            self.assertEqual(parsed.bounds, WO.parse_marker_set(data, build).bounds)  # kept (OPEN)

    def test_right_list_is_kept(self):
        data = B.build_mrk("switch", MARKERS, special=EXPECTED["switch"])
        self.assertEqual(D.derive_special_indices(data, ACTORS, "switch"), data)


# --- W14-19. World cluster counter -----------------------------------------------------------

def world(build, clusters):
    return B.build_wrl(build, chunks=[[B.scene_entry(build, 5, cluster=c) for c in clusters]])


class ClusterCounter(unittest.TestCase):
    def test_counter(self):
        self.assertEqual(D.cluster_counter([3, 7, -1]), 8)
        self.assertEqual(D.cluster_counter([-1, -1]), 0)
        self.assertEqual(D.cluster_counter([]), 0)
        self.assertEqual(D.cluster_counter([3], stored=6), 6)  # a larger stored value is kept
        self.assertEqual(D.cluster_counter([9], stored=6), 10)

    def test_written(self):
        for build in ("switch", "ps3"):
            data = world(build, [3, 12, -1])  # builder stores 8
            out = D.derive_cluster_counter(data, build)
            self.assertEqual(WO.parse_world(out, build).server.next_cluster_id, 13)
            self.assertEqual(len(out), len(data))

    def test_larger_stored_value_kept(self):
        data = world("switch", [3, 4])  # max + 1 = 5, stored 8
        self.assertEqual(D.derive_cluster_counter(data, "switch"), data)
        out = D.derive_cluster_counter(data, "switch", keep_larger=False)
        self.assertEqual(WO.parse_world(out, "switch").server.next_cluster_id, 5)
        out = D.derive_cluster_counter(world("switch", [-1]), "switch", keep_larger=False)
        self.assertEqual(WO.parse_world(out, "switch").server.next_cluster_id, 0)

    def test_world_without_scenes_unchanged(self):
        data = B.build_wrl("switch", tiles=[(1, 2, 3, 4)])
        self.assertEqual(D.derive_cluster_counter(data, "switch", keep_larger=False), data)


# --- W14-21. Actor bounds ----------------------------------------------------------------------

def app_bytes(order, centre=(1.0, 2.0, 3.0), radius=4.5, box=(-1.0, -2.0, -3.0, 5.0, 6.0, 7.0)):
    img = B.Image(BODY + 552, order)
    img.raw(0, B.sno_head(1, order))
    img.put(BODY + 0x10 + 0x80, "4f", *centre, radius)
    img.put(BODY + 0x10 + 0x118, "6f", *box)
    return img.done()


def actor_with_bounds(order, radius=1.0, appearance=77):
    img = B.Image(BODY + 448, order)
    img.raw(0, B.sno_head(1, order))
    img.put(BODY + 0x14, "i", appearance)
    img.put(BODY + 0x1C, "5f", 9.0, 9.0, 9.0, 9.0, 9.0)  # cylinder: authored
    img.put(BODY + 0x30, "4f", 0.0, 0.0, 0.0, radius)
    return img.done()


class ActorBounds(unittest.TestCase):
    def test_derived(self):
        for order in ("<", ">"):
            out = D.derive_actor_bounds(actor_with_bounds(order), app_bytes(order))
            self.assertEqual(struct.unpack_from(order + "6f", out, BODY + 0x40), (-1.0, -2.0, -3.0, 5.0, 6.0, 7.0))
            self.assertEqual(struct.unpack_from(order + "4f", out, BODY + 0x30), (1.0, 2.0, 3.0, 6.5))
            self.assertEqual(struct.unpack_from(order + "5f", out, BODY + 0x1C), (9.0,) * 5)

    def test_radius_rounded_to_f32(self):
        out = D.derive_actor_bounds(actor_with_bounds("<"), app_bytes("<", radius=0.1))
        stored = struct.unpack_from("<f", out, BODY + 0x3C)[0]
        f32 = struct.unpack("<f", struct.pack("<f", struct.unpack("<f", struct.pack("<f", 0.1))[0] + 2.0))[0]
        self.assertEqual(stored, f32)

    def test_larger_radius_kept(self):
        actor = actor_with_bounds("<", radius=50.0)
        out = D.derive_actor_bounds(actor, app_bytes("<"))
        self.assertEqual(struct.unpack_from("<f", out, BODY + 0x3C)[0], 50.0)
        out = D.derive_actor_bounds(actor, app_bytes("<"), keep_larger_radius=False)
        self.assertEqual(struct.unpack_from("<f", out, BODY + 0x3C)[0], 6.5)

    def test_every_actor_naming_the_appearance(self):
        actors = {"a": actor_with_bounds("<", appearance=77), "b": actor_with_bounds("<", appearance=78)}
        found = D.update_actors_for_appearance(actors, 77, app_bytes("<"))
        self.assertEqual(list(found), ["a"])
        self.assertEqual(struct.unpack_from("<f", found["a"], BODY + 0x3C)[0], 6.5)


# --- W14-22. Sound-bank rows --------------------------------------------------------------------

def row_sizes(bank, build, index):
    order = B.order_of(build)
    size, decompressed, stored = {"ps3": (40, 0x14, 0x20)}.get(build, (16, 0x08, 0x0C))
    offset = struct.unpack_from(order + "I", bank, 0x1C)[0] + BODY + index * size
    return struct.unpack_from(order + "I", bank, offset + decompressed)[0], \
        struct.unpack_from(order + "I", bank, offset + stored)[0]


class BankRows(unittest.TestCase):
    def test_vorbis(self):
        for build in ("switch", "ps3", "x360"):
            bank = B.build_sbk(build, [(0x11, 5), (0x22, 6)])
            fsb = B.fsb5(15, b"\x02\x00ab" + bytes(28), stereo=True, sample_count=1000)
            out = D.update_bank_row(bank, 1, fsb, build)
            self.assertEqual(row_sizes(out, build, 1), (4000, len(fsb)), build)
            self.assertEqual(row_sizes(out, build, 0), (0, 5), build)
            self.assertEqual(len(out), len(bank))

    def test_mode16_rule(self):
        fsb = B.fsb5(16, bytes(140 * 4), stereo=False, sample_count=1000)
        self.assertEqual(D.fsb_decompressed_size(fsb), 2000)
        bank = D.update_bank_row(B.build_sbk("switch", [(1, 0)]), 0, fsb, "switch")
        self.assertEqual(row_sizes(bank, "switch", 0), (2000, len(fsb)))

    def test_no_file_keeps_the_row(self):
        bank = B.build_sbk("switch", [(1, 77)])
        self.assertEqual(D.update_bank_row(bank, 0, None, "switch"), bank)

    def test_not_fsb5_sets_stored_size_only(self):
        bank = B.build_sbk("switch", [(1, 77)])
        out = D.update_bank_row(bank, 0, b"OggS" + bytes(96), "switch")
        self.assertEqual(row_sizes(out, "switch", 0), (0, 100))
        out = D.update_bank_row(bank, 0, B.fsb5(15, bytes(32)), "switch", decompressed=False)
        self.assertEqual(row_sizes(out, "switch", 0)[0], 0)

    def test_bad_index(self):
        with self.assertRaises(FormatError):
            D.update_bank_row(B.build_sbk("switch", [(1, 0)]), 1, b"x", "switch")


# --- W14-23. Font kerning order ------------------------------------------------------------------

class Kerning(unittest.TestCase):
    def font_tree(self, build):
        order = B.order_of(build)
        tree = new_tree(build, "fnt", order)
        path = path_of(build, "fnt", L.top_record(build, "fnt"), "FontPointSizeData")
        L.set_elements(tree, path, [L.new_record(build, "fnt", "FontPointSizeData")], update_counts=True)
        return tree, order

    def test_inserted_in_order(self):
        for build, count_at in (("switch", 0x38), ("x360", 0x38), ("ps3", 0x30)):
            tree, order = self.font_tree(build)
            for first, second in ((5, 3), (1, 9), (5, 1), (1, 2), (0x3000, 1)):
                D.insert_kerning(tree, 0, first, second, x=first, y=-second)
            keys = [p[:2] for p in D.kerning_pairs(tree, 0)]
            self.assertEqual(keys, [(1, 2), (1, 9), (5, 1), (5, 3), (0x3000, 1)], build)
            self.assertEqual(D.kerning_order_problems(tree), [])
            size = tree.top.pairs[L.count_links(build, "fnt", tree.top.name)[1].pair_offset].block.elements[0]
            self.assertEqual(struct.unpack_from(order + "I", size.raw, count_at)[0], 5, build)
            again = L.parse_tree(L.write_tree(tree), "fnt", build)
            self.assertEqual(D.kerning_pairs(again, 0), D.kerning_pairs(tree, 0))
            self.assertEqual(D.kerning_pairs(again, 0)[0], (1, 2, 1, -2))

    def test_replace(self):
        tree, _ = self.font_tree("switch")
        D.insert_kerning(tree, 0, 4, 4, x=1)
        self.assertEqual(D.insert_kerning(tree, 0, 4, 4, x=7), 0)
        self.assertEqual(D.kerning_pairs(tree, 0), [(4, 4, 7, 0)])
        with self.assertRaises(FormatError):
            D.insert_kerning(tree, 0, 4, 4, replace=False)

    def test_check_reports_unsorted(self):
        tree, order = self.font_tree("switch")
        D.insert_kerning(tree, 0, 1, 1)
        D.insert_kerning(tree, 0, 2, 2)
        size = tree.top.pairs[0xC0].block.elements[0]
        pairs = size.pairs[0x40].block.elements
        pairs.reverse()
        self.assertEqual(len(D.kerning_order_problems(tree)), 1)
        with self.assertRaises(FormatError):
            D.insert_kerning(tree, 0, 3, 3)


if __name__ == "__main__":
    unittest.main()
