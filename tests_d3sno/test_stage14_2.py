"""Corrections 17, V16-10: duplicated triangles in collision meshes (synthetic data only)."""

import unittest

import d3sno
from d3sno import collision as C
from d3sno import collision_write as CW

# triangle (0, 1, 2) appears twice: every edge of both copies is non-manifold
POSITIONS = [(0.0, 0.0, 0.0), (4.0, 0.0, 0.0), (2.0, 3.0, 0.0), (2.0, -3.0, 0.0)]
TRIANGLES = [(0, 1, 2), (1, 0, 3), (0, 1, 2)]


class Marker(unittest.TestCase):
    def test_version(self):
        self.assertGreaterEqual(tuple(int(x) for x in d3sno.__version__.split(".")), (0, 14, 2))


class ProvenNeighbours(unittest.TestCase):
    def test_copies_keep_their_own_stored_values(self):
        stored = [(1, 7, 8), (-1, -1, -1), (2, 5, 6)]
        kept = CW.proven_neighbours(TRIANGLES, POSITIONS, TRIANGLES, POSITIONS, stored)
        self.assertEqual(kept[0], (1, 7, 8))
        self.assertEqual(kept[2], (2, 5, 6))

    def test_occurrence_order_survives_other_triangles_moving_in_the_table(self):
        stored = [(1, 7, 8), (-1, -1, -1), (2, 5, 6)]
        now = [TRIANGLES[1], TRIANGLES[0], TRIANGLES[2]]  # the copies keep their relative order
        kept = CW.proven_neighbours(now, POSITIONS, TRIANGLES, POSITIONS, stored)
        self.assertEqual(kept[1], (1, 7, 8))
        self.assertEqual(kept[2], (2, 5, 6))

    def test_a_removed_copy_changes_the_edges(self):
        stored = [(1, 7, 8), (-1, -1, -1), (2, 5, 6)]
        now = TRIANGLES[:2]  # edges of (0, 1, 2) are manifold again: W14-15 values
        self.assertEqual(CW.proven_neighbours(now, POSITIONS, TRIANGLES, POSITIONS, stored),
                         CW.triangle_neighbours(now))


def stored_duplicates(build):
    """A file whose two copies of (0, 1, 2) store different neighbours."""
    mesh = CW.build_collision_mesh(POSITIONS, TRIANGLES, 1, 0, build)
    collision = CW.new_collision_file([mesh], build)
    copies = [k for k, t in enumerate(mesh.table[:mesh.triangle_count]) if t.vertices == (0, 1, 2)]
    for value, k in zip((1, 2), copies):
        t = mesh.table[k]
        mesh.table[k] = C.Triangle(t.vertices, (value, value, value), t.material, t.flags)
    if len(mesh.table) > mesh.triangle_count:  # ps3 padding repeats the last row
        mesh.table[mesh.triangle_count:] = [mesh.table[mesh.triangle_count - 1]] * (len(mesh.table) - mesh.triangle_count)
    mesh.triangles = mesh.table[:mesh.triangle_count]
    return CW.write_collision(collision), copies


class WriteCollision(unittest.TestCase):
    def test_proven_rebuilds_duplicates_byte_for_byte(self):
        for build in ("ps3", "switch", "x360"):
            data, copies = stored_duplicates(build)
            self.assertEqual(len(copies), 2, build)
            table = C.parse_collision(data).meshes[0].table
            self.assertEqual([table[k].neighbours for k in copies], [(1, 1, 1), (2, 2, 2)], build)
            self.assertEqual(CW.write_collision(C.parse_collision(data), derive="proven"), data, build)

    def test_full_rebuild_writes_minus_one(self):
        data, _ = stored_duplicates("switch")
        rebuilt = C.parse_collision(CW.write_collision(C.parse_collision(data), derive=True)).meshes[0]
        self.assertEqual({t.neighbours for t in rebuilt.triangles if t.vertices == (0, 1, 2)}, {(-1, -1, -1)})


if __name__ == "__main__":
    unittest.main()
