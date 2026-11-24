"""Stage 13.2: ``27_corrections16.md`` V15-5 (octree maxima are not a validity condition),
synthetic files only."""

import _libpath  # noqa: F401  (python -I: the library folder on sys.path)
import struct
import unittest

import d3sno
from d3sno import app_mesh as M
from d3sno import appearance as P
from d3sno import octree as O
from d3sno import sno_layout as L
from test_stage13 import flagged_rigid, octree_of, set_mesh, write_grid


def grid_file():
    out, _ = write_grid(30)
    return out, P.parse_appearance(out), octree_of(out)


class Marker(unittest.TestCase):
    def test_version(self):
        self.assertGreaterEqual(tuple(int(x) for x in d3sno.__version__.split(".")), (0, 13, 2))
        self.assertGreaterEqual(d3sno.STAGE, 13)


class Check(unittest.TestCase):
    def test_other_maxima_are_not_a_problem(self):
        _, app, tree = grid_file()
        lengths = (len(tree.nodes), len(tree.leaves), len(tree.primitives))
        for maxima in (lengths, (lengths[0] + 5, lengths[1] * 2, lengths[2] * 3), (10**6, 10**6, 10**6)):
            tree.maxima = maxima
            self.assertEqual(O.octree_problems(app, tree), [], maxima)

    def test_example_of_v15_5(self):
        # T = 2 with maxima (2, 8, 32): 1 node, 8 leaves, 16 primitives
        app = P.parse_appearance(flagged_rigid())
        set_mesh(app.sub_objects[0].draws[0], [(0, 0, 0), (16, 0, 0), (0, 16, 0), (0, 0, 16)], [0, 1, 2, 0, 1, 3])
        tree = O.build_octree(app, ((0.0, 0.0, 0.0), 32.0))
        self.assertEqual(O.geo0_triangle_count(app), 2)
        tree.maxima = (2, 8, 32)
        self.assertLessEqual(len(tree.primitives), 32)
        self.assertEqual(O.octree_problems(app, tree), [])

    def test_each_maximum_below_its_count(self):
        _, app, tree = grid_file()
        lengths = (len(tree.nodes), len(tree.leaves), len(tree.primitives))
        for k, name in enumerate(("nMaxNodes", "nMaxLeaves", "nMaxPrimitives")):
            maxima = list(lengths)
            maxima[k] -= 1
            tree.maxima = tuple(maxima)
            found = O.octree_problems(app, tree)
            self.assertEqual([p.split()[0] for p in found], [name])

    def test_stored_counts_equal_list_lengths(self):
        out, app, tree = grid_file()
        self.assertEqual(tuple(tree.stored_counts), (len(tree.nodes), len(tree.leaves), len(tree.primitives)))
        self.assertEqual(O.octree_problems(app, tree), [])
        for k in range(3):
            stored = L.parse_tree(out, "app")
            at = O.COUNTS_AT + 12 + 4 * k
            struct.pack_into("<I", stored.top.raw, at, struct.unpack_from("<I", stored.top.raw, at)[0] + 1)
            data = L.write_tree(stored)
            found = O.octree_problems(P.parse_appearance(data), octree_of(data))
            self.assertTrue(any("list lengths" in p for p in found), k)

    def test_built_tree_has_no_stored_counts(self):
        _, app, _ = grid_file()
        self.assertIsNone(O.build_octree(app).stored_counts)


class Writer(unittest.TestCase):
    def test_rule_raised_to_the_counts(self):
        app = P.parse_appearance(flagged_rigid())
        set_mesh(app.sub_objects[0].draws[0], [(0, 0, 0), (16, 0, 0), (0, 16, 0)], [0, 1, 2])
        tree = octree_of(M.write_app_mesh(app, derive=True))
        lengths = (len(tree.nodes), len(tree.leaves), len(tree.primitives))
        self.assertEqual(tuple(tree.maxima), (max(1, lengths[0]), max(1, lengths[1]), max(4, lengths[2])))
        self.assertEqual(tree.maxima[1], 8)

    def test_rule_kept_when_larger(self):
        _, _, tree = grid_file()
        self.assertEqual(tree.maxima[:2], (1800, 1800))  # T above the counts: not raised
        self.assertIn(tree.maxima[2], (3600, 7200))

    def test_raised_maxima(self):
        tree = O.Octree(nodes=[None] * 3, leaves=[None] * 22, primitives=[None] * 40)
        self.assertEqual(O.raised_maxima((2, 8, 32), tree), (3, 22, 40))
        self.assertEqual(O.raised_maxima((5, 30, 50), tree), (5, 30, 50))

    def test_unchanged_tree_keeps_its_maxima(self):
        # an unchanged mesh keeps the stored tree, maxima included
        out, app, tree = grid_file()
        stored = L.parse_tree(out, "app")
        struct.pack_into("<3I", stored.top.raw, O.COUNTS_AT, 10**5, 10**5, 10**5)
        data = L.write_tree(stored)
        again = P.parse_appearance(data)
        self.assertEqual(M.write_app_mesh(again, derive=True), data)


if __name__ == "__main__":
    unittest.main()
