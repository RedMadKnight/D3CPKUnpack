"""Stage 14.1: ``29_corrections17.md`` (V16-1 to V16-9), synthetic files only."""

import struct
import unittest

import builders as B
import d3sno
from d3sno import FormatError
from d3sno import collision as C
from d3sno import collision_write as CW
from d3sno import derived as D
from d3sno import index as I
from d3sno import index_write as IW
from d3sno import navigation_derive as ND
from d3sno import sno_layout as L
from d3sno import texture as T
from d3sno import texture_write as TW
from d3sno import world as WO
from test_stage14_collision import grid
from test_stage14_index import switch_toc
from test_stage14_misc import BODY, actor_with_bounds, app_bytes, path_of
from test_stage14_navigation import TILING, ParsedScenes
from test_stage14_texture import exact_image


class Marker(unittest.TestCase):
    def test_version(self):
        self.assertGreaterEqual(tuple(int(x) for x in d3sno.__version__.split(".")), (0, 14, 1))
        self.assertGreaterEqual(d3sno.STAGE, 14)


# --- V16-1 ---------------------------------------------------------------------------------------

def ps3_chain(fmt, w, h, levels):
    tex = T.parse_texture(TW.encode_texture(exact_image(0, w, h), "ps3", fmt, levels=levels), "ps3")
    return tex, sum(T._ps3_level_bytes(tex, m) for m in range(levels))


class Ps3Padding(unittest.TestCase):
    def test_single_face_is_the_bare_chain(self):
        tex, chain = ps3_chain(0, 8, 8, 2)
        self.assertNotEqual(chain % 128, 0)
        self.assertEqual(len(tex.payload), chain)

    def test_cube_pads_every_face(self):
        _, chain = ps3_chain(0, 8, 8, 2)
        stride = -(-chain // 128) * 128
        tex = T.parse_texture(TW.encode_texture([exact_image(0, 8, 8, f) for f in range(6)], "ps3", 0, levels=2,
                                                faces=6), "ps3")
        self.assertEqual(len(tex.payload), 6 * stride)
        self.assertEqual(tex.payload[5 * stride + chain:], bytes(stride - chain))  # the last face too

    def test_rebuild_of_a_bare_chain(self):
        _, chain = ps3_chain(0, 8, 8, 2)
        payload = bytes((i * 7 + 3) & 0xFF for i in range(chain))
        data = B.build_tex("ps3", 0, 8, 8, payload, mip_max=1, frames=[(0, 0.0, 0.0, 1.0, 1.0)], image_ids=[0])
        tex = T.parse_texture(data, "ps3")
        self.assertEqual(TW.rebuild_payload(tex), payload)
        self.assertEqual(TW.ps3_payload(0, 8, 8, 2, TW.texture_elements(tex)), payload)


# --- V16-2 ---------------------------------------------------------------------------------------

LINKS = "tMarkerSet[0].ptMarkerLinks"


def marker_tree(links=2):
    data = B.build_mrk("switch", [{"type": 0, "links": [(1, 2)] * links}])
    tree = L.parse_tree(data, "mrk", "switch")
    marker = tree.top.pairs[0xC].block.elements[0]
    offset = next(s.offset for s in L.pair_slots("switch", "mrk", marker.name) if s.field.element == "MarkerLink")
    return tree, marker, offset


def link_count(data):
    return len(WO.parse_marker_set(data, "switch").markers[0].links), struct.unpack_from(
        "<i", data, BODY + L.parse_tree(data, "mrk", "switch").top.pairs[0xC].stored[0] + 0x40)[0]


class EmptiedLists(unittest.TestCase):
    def test_set_elements_to_empty(self):
        tree, marker, _ = marker_tree()
        L.set_elements(tree, LINKS, [])  # without update_counts
        self.assertEqual(struct.unpack_from("<i", marker.raw, 0x40)[0], 2)
        L.write_tree(tree, derive=True)
        self.assertEqual(struct.unpack_from("<i", marker.raw, 0x40)[0], 0)

    def test_direct_replacement(self):
        for new in (None, 5):
            tree, marker, offset = marker_tree()
            if new is None:
                marker.pairs[offset] = L.PairValue(None)
            else:
                elements = [L.new_record("switch", "mrk", "MarkerLink") for _ in range(new)]
                marker.pairs[offset] = L.PairValue(L.Block(elements=elements))
            self.assertEqual(L.derive_counts(tree), 1)
            self.assertEqual(struct.unpack_from("<i", marker.raw, 0x40)[0], new or 0)

    def test_same_length_is_unchanged(self):
        tree, marker, offset = marker_tree()
        struct.pack_into("<i", marker.raw, 0x40, 7)  # a stale word
        elements = [L.new_record("switch", "mrk", "MarkerLink") for _ in range(2)]
        marker.pairs[offset] = L.PairValue(L.Block(elements=elements))
        self.assertEqual(L.derive_counts(tree), 0)
        self.assertEqual(struct.unpack_from("<i", marker.raw, 0x40)[0], 7)

    def test_value_array_emptied_by_set_block(self):
        for build in ("switch", "ps3", "x360"):
            data = B.build_mrk(build, [{"type": 16}, {"type": 16}], special=(0, 1))
            order = B.order_of(build)
            tree = L.parse_tree(data, "mrk", build)
            L.set_block(tree, path_of(build, "mrk", L.top_record(build, "mrk"), "WORD"), b"")
            out = L.write_tree(tree, derive=True)
            self.assertEqual(struct.unpack_from(order + "I", out, BODY + 0xB0)[0], 0, build)
            self.assertEqual(WO.parse_marker_set(out, build).special_indices, [])

    def test_unchanged_tree_still_identical(self):
        tree, _, _ = marker_tree()
        self.assertEqual(L.write_tree(tree, derive=True), L.write_tree(marker_tree()[0]))


# --- V16-3 ---------------------------------------------------------------------------------------

# triangle 0's edge (0, 1) is non-manifold: triangles 1 and 2 both have the reversed edge (1, 0)
POSITIONS = [(0.0, 0.0, 0.0), (4.0, 0.0, 0.0), (2.0, 3.0, 0.0), (2.0, -3.0, 0.0), (2.0, -2.0, 1.0)]
TRIANGLES = [(0, 1, 2), (1, 0, 3), (1, 0, 4)]


def stored_with_candidate(build):
    """A file whose non-manifold slot holds a candidate (3), as about half of the game's do."""
    mesh = CW.build_collision_mesh(POSITIONS, TRIANGLES, 1, 0, build)
    collision = CW.new_collision_file([mesh], build)
    for k, t in enumerate(mesh.table):
        if t.vertices == (0, 1, 2):
            self_index = k
            mesh.table[k] = C.Triangle(t.vertices, (3,) + tuple(t.neighbours[1:]), t.material, t.flags)
    mesh.triangles = mesh.table[:mesh.triangle_count]
    return CW.write_collision(collision), self_index


def slot0(data, k):
    return C.parse_collision(data).meshes[0].table[k].neighbours[0]


class NonManifoldNeighbours(unittest.TestCase):
    def test_slots(self):
        self.assertEqual(CW.non_manifold_slots(TRIANGLES), {(0, 0), (1, 0), (2, 0)})
        self.assertEqual(CW.triangle_neighbours(TRIANGLES)[0][0], -1)

    def test_proven_keeps_the_stored_candidate(self):
        for build in ("ps3", "switch", "x360"):
            data, k = stored_with_candidate(build)
            self.assertEqual(slot0(data, k), 3)
            self.assertEqual(CW.write_collision(C.parse_collision(data), derive="proven"), data, build)
            # a full rebuild writes -1 there (W14-15)
            rebuilt = C.parse_collision(CW.write_collision(C.parse_collision(data), derive=True)).meshes[0]
            self.assertEqual([t.neighbours[0] for t in rebuilt.table if t.vertices == (0, 1, 2)][0], -1, build)

    def test_changed_triangle_gets_minus_one(self):
        for build in ("ps3", "switch"):
            data, k = stored_with_candidate(build)
            parsed = C.parse_collision(data)
            parsed.meshes[0].positions[2] = (2.0, 3.5, 0.0)  # triangle (0, 1, 2) moves
            self.assertEqual(slot0(CW.write_collision(parsed, derive="proven"), k), -1)

    def test_changed_triangle_on_the_edge_gets_minus_one(self):
        data, k = stored_with_candidate("switch")
        parsed = C.parse_collision(data)
        parsed.meshes[0].positions[4] = (2.0, -2.0, 2.0)  # a candidate on that edge moves
        self.assertEqual(slot0(CW.write_collision(parsed, derive="proven"), k), -1)

    def test_proven_neighbours_without_source(self):
        self.assertEqual(CW.proven_neighbours(TRIANGLES, POSITIONS), CW.triangle_neighbours(TRIANGLES))
        stored = [(3, -1, -1), (-1, -1, -1), (2, -1, -1)]
        kept = CW.proven_neighbours(TRIANGLES, POSITIONS, TRIANGLES, POSITIONS, stored)
        self.assertEqual([n[0] for n in kept], [3, -1, 2])
        self.assertEqual([n[1:] for n in kept], [n[1:] for n in CW.triangle_neighbours(TRIANGLES)])


# --- V16-4 ---------------------------------------------------------------------------------------

def parsed_scene(build="ps3"):
    return ParsedScenes.scene(ParsedScenes(), build, TILING)[0]


class SceneStatuses(unittest.TestCase):
    def test_border_differences_are_supported(self):
        nav = parsed_scene()
        nav.borders[0] = type(nav.borders[0])(nav.borders[0].cell, nav.borders[0].sides ^ 8)
        self.assertEqual(ND.derived_differences(nav, supported=False), [])
        self.assertTrue(all("(SUPPORTED)" in p for p in ND.derived_differences(nav)))
        self.assertTrue(ND.derived_differences(nav))

    def test_cell_entries_are_proven(self):
        nav = parsed_scene()
        n = next(i for i, x in enumerate(nav.neighbours) if x.flags != 0xFFFF)
        nav.neighbours[n] = type(nav.neighbours[n])(nav.neighbours[n].flags ^ 1, nav.neighbours[n].cell)
        self.assertIn("cell-to-cell neighbour entries differ from W14-16", ND.derived_differences(nav, supported=False))

    def test_apply_keeps_stored_borders_by_default(self):
        nav = parsed_scene()
        nav.borders[0] = type(nav.borders[0])(nav.borders[0].cell, nav.borders[0].sides | 8)
        derived = ND.derive_navigation(nav, nav.cells)
        self.assertEqual(ND.apply_derived(nav, derived).borders, nav.borders)
        self.assertEqual(ND.apply_derived(nav, derived, supported=True).borders, derived.borders)
        out = ND.apply_derived(nav, derived)
        self.assertEqual(ND.cell_entries(out.cells, out.neighbours), ND.cell_entries(derived.cells, derived.neighbours))


# --- V16-5, V16-6, V16-7 -------------------------------------------------------------------------

class Counters(unittest.TestCase):
    def test_cluster_counter_keeps_any_value_not_lower(self):
        self.assertEqual(D.cluster_counter([3, -1]), 4)
        self.assertEqual(D.cluster_counter([3, -1], stored=24), 24)
        self.assertEqual(D.cluster_counter([3, -1], stored=2), 4)
        self.assertEqual(D.cluster_counter([-1, -1], stored=20), 20)

    def test_collision_scale_is_consistent(self):
        positions, triangles = grid()
        for build, q in (("switch", 32767), ("ps3", 32766)):
            mesh = CW.build_collision_mesh(positions, triangles, 1, 0, build)
            for k in range(3):
                self.assertEqual(mesh.box_scale[k], CW.f32(mesh.half_extent[k] / q))
                self.assertTrue(all(abs(r[k]) <= mesh.half_extent[k] for r in mesh.vertices))

    def test_actor_radius_kept_for_unchanged_appearance(self):
        actor = actor_with_bounds("<", radius=1.0)
        self.assertEqual(D.derive_actor_bounds(actor, app_bytes("<"), appearance_changed=False), actor)
        out = D.derive_actor_bounds(actor, app_bytes("<"))
        self.assertEqual(struct.unpack_from("<f", out, BODY + 0x3C)[0], 6.5)


# --- V16-9 ---------------------------------------------------------------------------------------

class TitleUpdateToc(unittest.TestCase):
    def test_parsed_unsorted_toc_is_not_sorted(self):
        data = switch_toc({44: [(100, "t"), (3, "tex1")]}, ">", last=100)
        toc = I.parse_core_toc(data, "x360")
        with self.assertRaises(FormatError):
            IW.write_core_toc(toc)
        self.assertEqual(IW.write_core_toc(toc, keep_order=True), data)

    def test_title_update_flag(self):
        data = switch_toc({44: [(3, "tex1"), (100, "t")]}, ">", last=100)
        toc = I.parse_core_toc(data, "x360")
        with self.assertRaises(FormatError):
            IW.write_core_toc(toc, title_update=True)
        self.assertEqual(IW.write_core_toc(toc, title_update=True, keep_order=True), data)
        self.assertEqual(IW.write_core_toc(toc), data)  # sorted already: written as before

    def test_list_of_entries_may_be_sorted(self):
        data = switch_toc({44: [(100, "t"), (3, "tex1")]}, ">", last=100)
        entries = I.parse_core_toc(data, "x360").entries
        self.assertEqual(IW.write_core_toc(entries, "x360"), switch_toc({44: [(3, "tex1"), (100, "t")]}, ">", last=100))


if __name__ == "__main__":
    unittest.main()
