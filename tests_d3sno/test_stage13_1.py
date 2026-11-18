"""Stage 13.1: ``27_corrections16.md`` (V15-1 to V15-4), synthetic files only."""

import struct
import unittest

import builders as B
import d3sno
from d3sno import app_mesh as M
from d3sno import appearance as P
from d3sno import octree as O
from d3sno import sno_layout as L
from test_stage12 import laid_out, switch_rigid
from test_stage13 import RIGID_FLAGGED, flagged_rigid, octree_of, set_mesh, with_root, write_grid

# A float32 root whose child corner differs between float64 and float32 (V15-1):
# 1000.1 and 0.3 as float32; corner + half is 1000.24997559... in float64, 1000.25 in float32.
LOW_X = O.f32(1000.1)
SIZE = O.f32(0.3)
CORNER_64 = LOW_X + SIZE / 2
CORNER_32 = 1000.25
# a triangle reaching x = 1000.24998: inside the float64 child 4, outside the float32 one
NEAR = [(1000.15, 0.05, 0.05), (1000.24998, 0.05, 0.05), (1000.2, 0.1, 0.05)]


def near_app():
    app = P.parse_appearance(flagged_rigid())
    set_mesh(app.sub_objects[0].draws[0], NEAR, [0, 1, 2])
    return app


def two_draw_app():
    """Two flagged rigid sub-objects, the first with two draws (the second a copy of the first)."""
    app = P.parse_appearance(laid_out(B.build_app_switch(pool_spare=0, sub_objects=[
        {"rigid": [(0, 0, 0), (1, 0, 0), (0, 1, 0)], "index_count": 3, "flags": RIGID_FLAGGED},
        {"rigid": [(0, 0, 0), (1, 0, 0), (0, 1, 0)], "index_count": 3, "flags": RIGID_FLAGGED}])))
    sub = app.sub_objects[0]
    sub.draws.append(sub.draws[0])
    return app


def second_form(tree, app):
    """The tree with every primitive of a later draw in the second form (V15-3)."""
    offsets = O.draw_offsets(app)
    out = O.Octree(nodes=list(tree.nodes), leaves=list(tree.leaves), maxima=tree.maxima)
    for user, position in tree.primitives:
        sub, draw = user & 0xFFFF, user >> 16
        out.primitives.append((sub, offsets[sub][draw] + position) if draw else (user, position))
    return out


class Marker(unittest.TestCase):
    def test_version(self):
        self.assertGreaterEqual(tuple(int(x) for x in d3sno.__version__.split(".")), (0, 13, 1))
        self.assertGreaterEqual(d3sno.STAGE, 13)


class FloatCubes(unittest.TestCase):
    def test_child_cube(self):
        (x, y, z), half = O.child_cube((LOW_X, 0.0, 0.0), SIZE, 4)
        self.assertEqual(half, O.f32(SIZE / 2))
        self.assertEqual((x, y, z), (CORNER_32, 0.0, 0.0))
        self.assertNotEqual(CORNER_64, CORNER_32)  # the float64 corner is not the stored one
        for k in range(8):
            low, half = O.child_cube((LOW_X, 0.0, 0.0), SIZE, k)
            self.assertEqual(low, tuple(O.f32(v) for v in low))
            self.assertEqual(low[0], CORNER_32 if k >> 2 & 1 else LOW_X)
            self.assertEqual(low[1], half if k >> 1 & 1 else 0.0)
            self.assertEqual(low[2], half if k & 1 else 0.0)

    def test_builder_uses_float32_children(self):
        app = near_app()
        triangles = O.octree_triangles(app)
        half = O.f32(SIZE / 2)
        # the float64 cube would take the triangle; the float32 one does not
        self.assertTrue(O.triangle_overlaps_cube(triangles[0][2], (CORNER_64, 0.0, 0.0), half))
        self.assertFalse(O.triangle_overlaps_cube(triangles[0][2], (CORNER_32, 0.0, 0.0), half))
        tree = O.build_octree(app, ((1000.1, 0.0, 0.0), 0.3), triangles)  # root rounded to float32
        self.assertEqual(tree.root, ((LOW_X, 0.0, 0.0), SIZE))
        self.assertEqual((tree.leaves[4][0], tree.leaves[4][3]), ((CORNER_32, 0.0, 0.0), 0))
        self.assertEqual(tree.leaves[0][3], 1)
        self.assertEqual(O.leaf_problems(tree, triangles), [])
        self.assertEqual(O.missing_primitives(tree, triangles), [])

    def test_cubes_survive_storage(self):
        app = near_app()
        tree = O.build_octree(app, ((1000.1, 0.0, 0.0), 0.3))
        data = flagged_rigid()
        stored = L.parse_tree(data, "app")
        O.set_octree(stored, tree)
        back = octree_of(L.write_tree(stored))
        self.assertEqual([(tuple(lo), s) for lo, s, _ in back.nodes], [(tuple(lo), s) for lo, s, _ in tree.nodes])
        self.assertEqual([(tuple(lo), s, f, c) for lo, s, f, c in back.leaves],
                         [(tuple(lo), s, f, c) for lo, s, f, c in tree.leaves])

    def test_checks_use_the_float32_cube(self):
        app = near_app()
        triangles = O.octree_triangles(app)
        half = O.f32(SIZE / 2)
        # a leaf given with the float64 corner is checked as stored (float32): the triangle is out
        tree = O.Octree(nodes=[((LOW_X, 0.0, 0.0), SIZE, (O.LEAF_BIT,))],
                        leaves=[((CORNER_64, 0.0, 0.0), half, 0, 1)], primitives=[(0, 0)], maxima=(1, 1, 2))
        self.assertEqual(len(O.leaf_problems(tree, triangles)), 1)
        tree.leaves[0] = ((LOW_X, 0.0, 0.0), half, 0, 1)
        self.assertEqual(O.leaf_problems(tree, triangles), [])

    def test_derived_root_is_float32(self):
        app = P.parse_appearance(flagged_rigid())
        set_mesh(app.sub_objects[0].draws[0], [(0.1, 0.1, 0.1), (3.3, 0.1, 0.1), (0.1, 2.9, 0.1)], [0, 1, 2])
        low, size = O.derived_root(O.octree_triangles(app))
        self.assertEqual(low, tuple(O.f32(v) for v in low))
        out = M.write_app_mesh(app, derive=True)
        again = P.parse_appearance(out)
        tree = octree_of(out)
        triangles = O.octree_triangles(again)
        self.assertEqual(O.missing_primitives(tree, triangles), [])
        self.assertEqual(O.leaf_problems(tree, triangles), [])
        self.assertEqual(O.build_octree(again, tree.root).leaves, [(tuple(lo), s, f, c) for lo, s, f, c in tree.leaves])


class LeafOrder(unittest.TestCase):
    def test_rank(self):
        # the example of V15-2: the first index list has 300 indices
        offsets = {0: [0, 300, 330]}
        self.assertEqual(O.primitive_rank(offsets, 65536, 3 * 4), (0, 300 + 12))
        self.assertEqual(O.primitive_rank(offsets, 0, 9), (0, 9))
        self.assertEqual(O.primitive_rank(offsets, 0, 300 + 12), (0, 312))  # the second form

    def test_offsets(self):
        app = two_draw_app()
        self.assertEqual(O.draw_offsets(app), {0: [0, 3, 6], 1: [0, 3]})

    def test_built_leaves(self):
        app = two_draw_app()
        tree = O.build_octree(app)
        leaf = next(leaf for leaf in tree.leaves if leaf[3] == 3)
        # by (sub-object, position across its draws), not by the user data
        self.assertEqual(tree.primitives[leaf[2]:leaf[2] + 3], [(0, 0), (65536, 0), (1, 0)])
        self.assertEqual(O.octree_problems(app, tree), [])

    def test_check_uses_the_order(self):
        app = two_draw_app()
        tree = O.build_octree(app)
        for n, (_, _, first, count) in enumerate(tree.leaves):
            if count == 3:
                tree.primitives[first:first + 3] = sorted(tree.primitives[first:first + 3])
        self.assertTrue(any("not sorted" in p for p in O.octree_problems(app, tree)))

    def test_single_draws_unchanged(self):
        out, _ = write_grid(4)
        again = P.parse_appearance(out)
        tree = octree_of(out)
        for _, _, first, count in tree.leaves:
            keys = tree.primitives[first:first + count]
            self.assertEqual(list(keys), sorted(keys))
        self.assertEqual(O.octree_problems(again, tree), [])


class SecondForm(unittest.TestCase):
    def test_canonical(self):
        offsets = {0: [0, 300, 330, 360], 1: [0, 9]}
        self.assertEqual(O.canonical_primitive(offsets, 0, 303), (65536, 3))
        self.assertEqual(O.canonical_primitive(offsets, 0, 333), (2 * 65536, 3))
        self.assertEqual(O.canonical_primitive(offsets, 0, 30), (0, 30))
        self.assertEqual(O.canonical_primitive(offsets, 65536, 3), (65536, 3))
        self.assertEqual(O.canonical_primitive(offsets, 1, 6), (1, 6))
        self.assertEqual(O.canonical_primitive(offsets, 5, 6), (5, 6))

    def test_reader_accepts_it(self):
        app = two_draw_app()
        triangles = O.octree_triangles(app)
        tree = second_form(O.build_octree(app), app)
        self.assertNotIn(65536, [u for u, _ in tree.primitives])
        self.assertIn((0, 3), tree.primitives)
        self.assertEqual(O.missing_primitives(tree, triangles, app), [])
        self.assertEqual(O.leaf_problems(tree, triangles, app), [])
        self.assertEqual(O.octree_problems(app, tree), [])
        self.assertEqual(O.missing_primitives(tree, triangles), [(65536, 0)])  # without the mesh

    def test_stored_and_read(self):
        app = two_draw_app()
        tree = second_form(O.build_octree(app), app)
        stored = L.parse_tree(flagged_rigid(), "app")
        O.set_octree(stored, tree)
        back = octree_of(L.write_tree(stored))
        self.assertEqual([tuple(p) for p in back.primitives], tree.primitives)
        self.assertEqual(O.missing_primitives(back, O.octree_triangles(app), app), [])

    def test_writer_uses_the_first_form(self):
        app = two_draw_app()
        self.assertIn((65536, 0), O.build_octree(app).primitives)


class EmptyRootMaxima(unittest.TestCase):
    def test_stored_maxima_kept(self):
        # maxima (1, 1, 2) stored; the leaf maximum is raised to the 8 leaves (V15-5)
        data = with_root(switch_rigid(), (-8.0, -8.0, -8.0), 16.0)
        out, app = write_grid(2, data=data)
        tree = octree_of(out)
        self.assertEqual((len(tree.nodes), len(tree.leaves), tree.primitives), (1, 8, []))
        self.assertEqual(tuple(tree.maxima), (1, 8, 2))
        top = L.parse_tree(out, "app").top
        self.assertEqual(struct.unpack_from("<6I", top.raw, O.COUNTS_AT), (1, 8, 2, 1, 8, 0))
        self.assertEqual(O.octree_problems(P.parse_appearance(out), tree), [])

    def test_changed_mesh_octree(self):
        app = P.parse_appearance(switch_rigid())
        stored = O.Octree(nodes=[((0.0, 0.0, 0.0), 4.0, tuple(O.LEAF_BIT | k for k in range(8)))],
                          leaves=[((0.0, 0.0, 0.0), 2.0, 0, 0)] * 8, maxima=(4, 8, 0))
        self.assertEqual(tuple(M.changed_mesh_octree(app, stored).maxima), (4, 8, 0))
        self.assertIsNone(M.changed_mesh_octree(app, None))

    def test_maxima_below_counts(self):
        out, _ = write_grid(3)
        tree = octree_of(out)
        tree.maxima = (0, len(tree.leaves), len(tree.primitives))
        self.assertTrue(any("nMaxNodes" in p for p in O.octree_problems(P.parse_appearance(out), tree)))


if __name__ == "__main__":
    unittest.main()
