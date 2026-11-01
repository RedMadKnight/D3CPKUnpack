import struct
import unittest

import builders as B
from d3sno import FormatError
from d3sno import collision as C

BUILDS = ("ps3", "switch")
VERTICES = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 1.0, 0.0)]


def mesh(build, **changes):
    """Two triangles under one internal node."""
    data = {
        "centre": (10.0, 20.0, 30.0),
        "vertices": VERTICES,
        "triangles": [
            B.triangle_row(build, (0, 1, 2), material=4, flags=5),
            B.triangle_row(build, (1, 3, 2)),
        ],
        "nodes": [B.tree_internal(build, 2), B.tree_leaf(build, 0, 1), B.tree_leaf(build, 1, 1)],
    }
    data.update(changes)
    return data


def read(build, **changes):
    return C.parse_collision(B.build_phm(build, [mesh(build, **changes)]))


class Meshes(unittest.TestCase):
    def test_fields(self):
        for build in BUILDS:
            with self.subTest(build):
                parsed = read(build)
                self.assertEqual(parsed.flags, 9)
                m = parsed.meshes[0]
                self.assertEqual(m.centre, (10.0, 20.0, 30.0))
                self.assertEqual(m.half_extent, (1.0, 1.0, 1.0))
                self.assertEqual(m.box_scale, (2.0, 2.0, 2.0))
                self.assertEqual((m.tree_height, m.flags, m.material), (2, 5, 6))
                self.assertEqual(parsed.warnings, [])

    def test_position_is_vertex_plus_centre(self):
        for build in BUILDS:
            self.assertEqual(read(build).meshes[0].positions[1], (11.0, 20.0, 30.0))

    def test_triangle_fields(self):
        for build in BUILDS:
            triangle = read(build).meshes[0].triangles[0]
            self.assertEqual(triangle.vertices, (0, 1, 2))
            self.assertEqual(triangle.neighbours, (-1, -1, -1))
            self.assertEqual((triangle.material, triangle.flags), (4, 5))

    def test_neighbours(self):
        row = B.triangle_row("ps3", (0, 1, 2), (7, 0xFFFF, 8))
        triangles = [row, B.triangle_row("ps3", (1, 3, 2))]
        self.assertEqual(read("ps3", triangles=triangles).meshes[0].triangles[0].neighbours, (7, -1, 8))

    def test_tree_nodes(self):
        nodes = read("switch").meshes[0].nodes
        self.assertEqual(nodes[0].axis, 1)
        self.assertFalse(nodes[0].is_leaf)
        self.assertTrue(nodes[1].is_leaf)
        self.assertEqual((nodes[1].lower, nodes[1].upper), ((0, 0, 0), (1, 1, 1)))

    def test_leaf_with_several_triangles(self):
        for build in BUILDS:
            parsed = read(build, nodes=[B.tree_leaf(build, 0, 2)])
            self.assertEqual(parsed.warnings, [])

    def test_ps3_table_can_be_longer(self):
        extra = mesh("ps3")
        extra["triangles"] = extra["triangles"] + [B.triangle_row("ps3", (99, 99, 99))]
        extra["triangle_count"] = 2
        parsed = C.parse_collision(B.build_phm("ps3", [extra]))
        m = parsed.meshes[0]
        self.assertEqual((len(m.table), len(m.triangles)), (3, 2))
        self.assertEqual(parsed.warnings, [])

    def test_ps3_triangles_come_from_the_tree(self):
        nodes = [
            B.tree_internal("ps3", 2), B.tree_leaf("ps3", 2, 1), B.tree_leaf("ps3", 0, 1),
        ]
        extra = mesh("ps3", nodes=nodes)
        extra["triangles"] = [
            B.triangle_row("ps3", (0, 1, 2)), B.triangle_row("ps3", (9, 9, 9)),
            B.triangle_row("ps3", (1, 3, 2)),
        ]
        extra["triangle_count"] = 2
        m = C.parse_collision(B.build_phm("ps3", [extra])).meshes[0]
        self.assertEqual([t.vertices for t in m.triangles], [(1, 3, 2), (0, 1, 2)])

    def test_no_meshes(self):
        self.assertEqual(C.parse_collision(B.build_phm("switch", [])).meshes, [])

    def test_empty_mesh(self):
        parsed = C.parse_collision(B.build_phm("switch", [mesh(
            "switch", vertices=[], triangles=[], nodes=[],
        )]))
        self.assertEqual(parsed.warnings, [])


class Checks(unittest.TestCase):
    def test_triangle_in_two_leaves(self):
        for build in BUILDS:
            nodes = [B.tree_internal(build, 2), B.tree_leaf(build, 0, 2), B.tree_leaf(build, 1, 1)]
            data = B.build_phm(build, [mesh(build, nodes=nodes)])
            self.assertIn("more than one leaf", C.parse_collision(data).warnings[0])
            with self.assertRaises(FormatError):
                C.parse_collision(data, strict=True)

    def test_triangle_in_no_leaf(self):
        for build in BUILDS:
            parsed = read(build, nodes=[B.tree_leaf(build, 0, 1)])
            self.assertIn("1 of 2", parsed.warnings[0])

    def test_leaf_past_the_table(self):
        for build in BUILDS:
            with self.assertRaises(FormatError):
                read(build, nodes=[B.tree_leaf(build, 0, 5)])

    def test_right_child_past_the_tree(self):
        for build in BUILDS:
            nodes = [B.tree_internal(build, 9), B.tree_leaf(build, 0, 2)]
            with self.assertRaises(FormatError):
                read(build, nodes=nodes)

    def test_right_child_loop(self):
        for build in BUILDS:
            with self.assertRaises(FormatError):
                read(build, nodes=[B.tree_internal(build, 0), B.tree_leaf(build, 0, 2)])

    def test_node_reached_twice(self):
        for build in BUILDS:
            nodes = [
                B.tree_internal(build, 2), B.tree_internal(build, 1), B.tree_leaf(build, 0, 2),
            ]
            with self.assertRaises(FormatError):
                read(build, nodes=nodes)

    def test_vertex_out_of_range(self):
        for build in BUILDS:
            triangles = [B.triangle_row(build, (0, 1, 9)), B.triangle_row(build, (1, 3, 2))]
            with self.assertRaises(FormatError):
                read(build, triangles=triangles)

    def test_mesh_pair_size(self):
        with self.assertRaises(FormatError):
            C.parse_collision(B.build_phm("switch", [mesh("switch")], count=2))

    def test_section_pair_sizes(self):
        for at in (0x50, 0x58, 0x60):
            data = bytearray(B.build_phm("switch", [mesh("switch")]))
            record = struct.unpack_from("<I", data, 0x24)[0] + 0x10
            size = struct.unpack_from("<I", data, record + at + 4)[0]
            struct.pack_into("<I", data, record + at + 4, size + 1)
            with self.assertRaises(FormatError):
                C.parse_collision(bytes(data))

    def test_ps3_entries_below_triangle_count(self):
        data = bytearray(B.build_phm("ps3", [mesh("ps3")]))
        record = struct.unpack_from(">I", data, 0x24)[0] + 0x10
        struct.pack_into(">i", data, record + 0x54, 1)
        with self.assertRaises(FormatError):
            C.parse_collision(bytes(data))


class Export(unittest.TestCase):
    def test_obj(self):
        for build in BUILDS:
            lines = C.to_obj(read(build)).splitlines()
            self.assertIn("o mesh_0", lines)
            self.assertIn("v 11.000000 20.000000 30.000000", lines)
            self.assertIn("f 1 2 3", lines)
            self.assertIn("f 2 4 3", lines)

    def test_obj_indices_continue_across_meshes(self):
        data = B.build_phm("switch", [mesh("switch"), mesh("switch")])
        self.assertIn("f 5 6 7", C.to_obj(C.parse_collision(data)).splitlines())
