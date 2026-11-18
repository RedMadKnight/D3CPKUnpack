"""Stage 14: ``28_derived_values.md`` W14-12 to W14-15 (collision meshes), synthetic data only."""

import math
import struct
import unittest

import builders as B
from d3sno import collision as C
from d3sno import collision_write as W
from d3sno import sno_layout as L

BUILDS = ("ps3", "switch", "x360")
f32 = W.f32


def grid(nx=7, ny=5, step=1.25, origin=(10.0, -3.0, 2.0)):
    """A height field of (nx x ny) quads, two triangles each, consistently wound."""
    seed = 12345
    positions = []
    for j in range(ny + 1):
        for i in range(nx + 1):
            seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
            z = origin[2] + 0.37 * math.sin(i * 0.9) + 0.21 * j + (seed % 1000) / 4000.0
            positions.append((origin[0] + i * step, origin[1] + j * step * 0.8, z))
    triangles = []
    for j in range(ny):
        for i in range(nx):
            a = j * (nx + 1) + i
            b, c, d = a + 1, a + nx + 1, a + nx + 2
            triangles.append((a, b, d))
            triangles.append((a, d, c))
    return positions, triangles


def built_file(build, positions=None, triangles=None, **kw):
    if positions is None:
        positions, triangles = grid()
    mesh = W.build_collision_mesh(positions, triangles, kw.pop("materials", 3), kw.pop("flags", 1), build,
                                  mesh_flags=5, mesh_material=6)
    data = W.write_collision(W.new_collision_file([mesh], build, handle=77, flags=9))
    return mesh, data


def walk(nodes, form):
    """(leaf list of (first, count), parent of each node, depth of each node), checking pre-order."""
    shift = 2 if form == "ps3" else 6
    leaves, depth = [], {0: 1}
    order = []
    pending = [0]
    while pending:
        i = pending.pop()
        order.append(i)
        n = nodes[i]
        if n.is_leaf:
            count = n.a if form == "ps3" else (n.data >> 2) & 15
            leaves.append((n.data >> shift, count))
        else:
            r = i + (n.data >> shift)
            depth[i + 1] = depth[r] = depth[i] + 1
            pending.extend((r, i + 1))
    return leaves, order, depth


def subtree_entries(nodes, form):
    return W.leaf_runs(nodes, form)


class Bounds(unittest.TestCase):  # W14-12
    def test_centre_half_scale_and_vertices(self):
        positions, triangles = grid()
        for build in BUILDS:
            form = W.form_of(build)
            mesh, data = built_file(build)
            parsed = C.parse_collision(data)
            self.assertEqual(parsed.build, build)
            self.assertEqual(parsed.warnings, [])
            m = parsed.meshes[0]
            for a in range(3):
                lo = min(p[a] for p in positions)
                hi = max(p[a] for p in positions)
                c = f32((lo + hi) / 2)
                self.assertEqual(m.centre[a], c)
                raw_max = max(f32(p[a] - c) for p in positions)
                half = f32(raw_max + f32(W.HALF_PAD[form]))
                self.assertEqual(m.half_extent[a], half)
                self.assertEqual(m.box_scale[a], f32(half / (32766 if form == "ps3" else 32767)))
            for row, p in zip(m.vertices, positions):
                self.assertEqual(row[3], 0.0)
                for a in range(3):
                    self.assertEqual(row[a], f32(p[a] - m.centre[a]))
            self.assertEqual(len(m.positions), len(positions))
            self.assertEqual(m.triangle_count, len(triangles))
            self.assertEqual(m.positions, mesh.positions)
            self.assertEqual((m.flags, m.material), (5, 6))

    def test_pads_differ_by_form(self):
        self.assertEqual(W.HALF_PAD, {"switch": 0.164042, "ps3": 0.164})
        _, _, half_s, scale_s = W.mesh_bounds([(0, 0, 0), (2, 4, 6)], "x360")
        _, _, half_p, scale_p = W.mesh_bounds([(0, 0, 0), (2, 4, 6)], "ps3")
        self.assertEqual(half_s[0], f32(1.0 + f32(0.164042)))
        self.assertEqual(half_p[2], f32(3.0 + f32(0.164)))
        self.assertEqual(scale_p[1], f32(half_p[1] / 32766))
        self.assertEqual(scale_s[1], f32(half_s[1] / 32767))

    def test_ps3_table_tail(self):
        positions, triangles = grid()
        mesh, data = built_file("ps3")
        m = C.parse_collision(data).meshes[0]
        n = len(triangles)
        self.assertEqual(len(m.table), n + 3)
        self.assertEqual(m.table[n:], [m.table[n - 1]] * 3)
        raw = struct.unpack_from(">i", data, 0x10 + 0x30 + 0x54)  # first mesh record entry count
        self.assertEqual(raw[0], n + 3)
        for build in ("switch", "x360"):
            self.assertEqual(len(C.parse_collision(built_file(build)[1]).meshes[0].table), n)


class Tree(unittest.TestCase):  # W14-13
    def check_tree(self, m, form, n):
        leaves, order, depth = walk(m.nodes, form)
        self.assertEqual(order, list(range(len(m.nodes))))  # pre-order, left child next
        self.assertEqual(len(m.nodes), 2 * len(leaves) - 1)
        self.assertTrue(all(1 <= count <= 4 for _, count in leaves))
        flat = [e for first, count in leaves for e in range(first, first + count)]
        self.assertEqual(flat, list(range(n)))  # leaf order = table order, each triangle once
        self.assertEqual(m.tree_height, max(depth.values()))
        runs = subtree_entries(m.nodes, form)
        for i, node in enumerate(m.nodes):
            if node.is_leaf:
                self.assertLessEqual(len(runs[i]), 4)
            else:
                self.assertGreater(len(runs[i]), 4)
                if form == "switch":
                    self.assertEqual((node.data >> 2) & 15, 0)

    def test_shape_and_leaf_order(self):
        for build in BUILDS:
            for size in ((1, 1), (2, 1), (7, 5), (13, 9)):
                positions, triangles = grid(*size)
                _, data = built_file(build, positions, triangles)
                m = C.parse_collision(data, strict=True).meshes[0]
                self.check_tree(m, W.form_of(build), len(triangles))

    def test_split_rule(self):
        positions, triangles = grid(11, 6)
        mesh = W.build_collision_mesh(positions, triangles, 0, 0, "switch")
        rows = mesh.vertices
        runs = subtree_entries(mesh.nodes, "switch")
        tri = [t.vertices for t in mesh.table]

        def box(entries):
            pts = [rows[v] for e in entries for v in tri[e]]
            return [min(p[a] for p in pts) for a in range(3)], [max(p[a] for p in pts) for a in range(3)]

        def area(entries):
            lo, hi = box(entries)
            d = [hi[a] - lo[a] for a in range(3)]
            return 2 * (d[0] * d[1] + d[1] * d[2] + d[2] * d[0])

        def centre(e, a):
            vs = [rows[v][a] for v in tri[e]]
            return (min(vs) + max(vs)) / 2

        for i, node in enumerate(mesh.nodes):
            if node.is_leaf:
                continue
            entries = runs[i]
            ext = [max(centre(e, a) for e in entries) - min(centre(e, a) for e in entries) for a in range(3)]
            self.assertEqual(node.axis, ext.index(max(ext)))
            left, right = runs[i + 1], runs[i + (node.data >> 6)]
            self.assertEqual(left + right, entries)
            a = node.axis
            self.assertLessEqual(max(centre(e, a) for e in left), min(centre(e, a) for e in right))
            ordered = sorted(entries, key=lambda e: centre(e, a))
            costs = [area(ordered[:k]) * k + area(ordered[k:]) * (len(ordered) - k) for k in range(1, len(ordered))]
            best = area(left) * len(left) + area(right) * len(right)
            self.assertAlmostEqual(best, min(costs), places=6)

    def test_small_example(self):
        # five thin triangles along x: one split on x
        positions, triangles = [], []
        for i, x in enumerate((0.0, 1.0, 2.0, 10.0, 11.0)):
            positions += [(x, 0.0, 0.0), (x + 0.5, 0.0, 0.0), (x, 0.5, 0.0)]
            triangles.append((3 * i, 3 * i + 1, 3 * i + 2))
        m = W.build_collision_mesh(positions, triangles, 0, 0, "switch")
        self.assertEqual(len(m.nodes), 3)
        self.assertEqual(m.nodes[0].axis, 0)
        self.assertEqual(m.nodes[0].data >> 6, 2)
        self.assertEqual(((m.nodes[1].data >> 2) & 15, (m.nodes[2].data >> 2) & 15), (3, 2))
        self.assertEqual(m.tree_height, 2)
        self.assertEqual([t.vertices for t in m.table], triangles)

    def test_packing(self):
        _, data = built_file("ps3")
        m = C.parse_collision(data).meshes[0]
        for node in m.nodes:
            if node.is_leaf:
                self.assertEqual(node.b, 0)
                self.assertIn(node.a, range(1, 5))
        _, data = built_file("switch")
        m = C.parse_collision(data).meshes[0]
        self.assertTrue(all(n.a is None for n in m.nodes))


class NodeValues(unittest.TestCase):  # W14-14
    def test_switch_bounds_formula(self):
        for build in ("switch", "x360"):
            mesh, data = built_file(build)
            m = C.parse_collision(data).meshes[0]
            runs = subtree_entries(m.nodes, "switch")
            pad = f32(0.0328084)
            for i, node in enumerate(m.nodes):
                pts = [m.vertices[v] for e in runs[i] for v in m.table[e].vertices]
                for a in range(3):
                    s = m.box_scale[a]
                    lo, hi = min(p[a] for p in pts), max(p[a] for p in pts)
                    self.assertEqual(node.lower[a], math.floor(f32(f32(lo - pad) / s)))
                    self.assertEqual(node.upper[a], math.ceil(f32(f32(hi + pad) / s)))
                    self.assertLessEqual(node.lower[a] * s, lo)
                    self.assertGreaterEqual(node.upper[a] * s, hi)
                    self.assertTrue(-32768 <= node.lower[a] <= node.upper[a] <= 32767)

    def test_ps3_split_values(self):
        mesh, data = built_file("ps3")
        m = C.parse_collision(data).meshes[0]
        runs = subtree_entries(m.nodes, "ps3")
        pad = f32(0.0164)
        internal = 0
        for i, node in enumerate(m.nodes):
            if node.is_leaf:
                continue
            internal += 1
            a, s = node.axis, m.box_scale[node.axis]
            left = max(m.vertices[v][a] for e in runs[i + 1] for v in m.table[e].vertices)
            right = min(m.vertices[v][a] for e in runs[i + (node.data >> 2)] for v in m.table[e].vertices)
            sa = struct.unpack(">h", struct.pack(">H", node.a))[0]
            sb = struct.unpack(">h", struct.pack(">H", node.b))[0]
            self.assertEqual(sa, math.ceil(f32(f32(left + pad) / s)))
            self.assertEqual(sb, math.floor(f32(f32(right - pad) / s)))
        self.assertGreater(internal, 5)

    def test_negative_split_is_twos_complement(self):
        a, b = W.ps3_split_values([(-1.0, 0, 0, 0), (-0.5, 0, 0, 0), (0.5, 0, 0, 0)], [(0, 1, 0)], [(2, 2, 2)],
                                  0, (0.001, 1, 1))
        self.assertEqual(a, (-483) & 0xFFFF)  # ceil((-0.5 + 0.0164) / 0.001)
        self.assertEqual(b, 483)


class Neighbours(unittest.TestCase):  # W14-15
    def test_shared_edge_and_border(self):
        tris = [(0, 1, 2), (2, 1, 3)]
        self.assertEqual(W.triangle_neighbours(tris), [(-1, 3, -1), (0, -1, -1)])

    def test_non_manifold_and_same_direction(self):
        # three triangles on edge 1-2: (0,1,2) has two candidates with edge 2->1
        self.assertEqual(W.triangle_neighbours([(0, 1, 2), (2, 1, 3), (2, 1, 4)])[0], (-1, -1, -1))
        # same direction: (0,1,2) and (1,2,5) both use 1->2; (2,1,3) has the reverse
        result = W.triangle_neighbours([(0, 1, 2), (1, 2, 5), (2, 1, 3)])
        self.assertEqual(result[0][1], -1)
        self.assertEqual(result[1][0], -1)
        self.assertEqual(result[2][0], -1)

    def test_in_file(self):
        positions, triangles = grid()
        expected = {t: n for t, n in zip(triangles, W.triangle_neighbours(triangles))}
        for build in BUILDS:
            m = C.parse_collision(built_file(build)[1]).meshes[0]
            for t in m.table:
                self.assertEqual(t.neighbours, expected[t.vertices])
                for j, n in enumerate(t.neighbours):
                    if n != -1:  # the neighbour triangle exists with the reversed edge
                        a, b = t.vertices[j], t.vertices[(j + 1) % 3]
                        self.assertTrue(any(u.vertices in ((b, a, n), (a, n, b), (n, b, a)) for u in m.table))
            self.assertIn(-1, [n for t in m.table for n in t.neighbours])
        data = built_file("ps3")[1]
        self.assertIn(b"\xFF\xFF", data)

    def test_non_manifold_in_file(self):
        positions = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (1, 1, 1)]
        tris = [(0, 1, 2), (2, 1, 3), (2, 1, 4)]
        for build in BUILDS:
            m = C.parse_collision(built_file(build, positions, tris)[1]).meshes[0]
            by = {t.vertices: t.neighbours for t in m.table[:3]}
            self.assertEqual(by[(0, 1, 2)], (-1, -1, -1))
            self.assertEqual(by[(2, 1, 3)], (-1, -1, -1))


class Writer(unittest.TestCase):
    def test_derive_off_byte_identity_on_built(self):
        for build in BUILDS:
            _, data = built_file(build)
            parsed = C.parse_collision(data)
            self.assertEqual(W.write_collision(parsed), data)
            self.assertEqual(W.write_collision(parsed, derive="proven"), data)
            self.assertEqual(W.write_collision(parsed, derive=True), data)
            self.assertTrue(L.is_canonical(data, "phm", build))

    def test_derive_off_keeps_foreign_values(self):
        # a builder file with arbitrary stored values, made canonical by the generic writer
        for build in ("ps3", "switch"):
            order = ">" if build == "ps3" else "<"
            if build == "switch":
                rows = [struct.pack(order + "6iHH", 0, 1, 2, 7, 8, 9, 1, 2)]
                nodes = [struct.pack(order + "6hI", -5, 6, -7, 8, 9, 10, (0 << 6) | (1 << 2) | 3)]
            else:
                rows = [struct.pack(order + "8H", 0, 1, 2, 7, 8, 0xFFFF, 1, 2)] * 4
                nodes = [struct.pack(order + "IHH", 3, 1, 0)]
            source = B.build_phm(build, [{"centre": (1.0, 2.0, 3.0), "vertices": [(0.1, 0.2, 0.3), (1e-9, 4, 5),
                                          (7, 8, 9)], "triangles": rows, "triangle_count": 1, "nodes": nodes}])
            data = L.rebuild_sno(source, "phm", build)
            parsed = C.parse_collision(data)
            self.assertEqual(W.write_collision(parsed), data)
            self.assertNotEqual(W.write_collision(parsed, derive=True), data)

    def test_derive_recomputes_after_edit(self):
        for build in BUILDS:
            _, data = built_file(build)
            parsed = C.parse_collision(data)
            m = parsed.meshes[0]
            m.positions = [(x + 100.0, y, z * 2) for x, y, z in m.positions]
            kept = C.parse_collision(W.write_collision(parsed)).meshes[0]
            self.assertEqual(kept.centre, m.centre)  # stored centre kept with derive off
            new = C.parse_collision(W.write_collision(parsed, derive=True)).meshes[0]
            again = W.build_collision_mesh(m.positions, [t.vertices for t in m.triangles], 3, 1, build,
                                           mesh_flags=5, mesh_material=6)
            self.assertEqual(new.centre, again.centre)
            self.assertNotEqual(new.centre, m.centre)
            self.assertEqual(new.nodes, again.nodes)

    def test_proven_fixes_neighbours(self):
        _, data = built_file("ps3")
        parsed = C.parse_collision(data)
        m = parsed.meshes[0]
        m.table = [C.Triangle(t.vertices, (-1, -1, -1), t.material, t.flags) for t in m.table]
        m.nodes = [n if n.is_leaf else C.TreeNode(n.data, None, None, 0, 0) for n in m.nodes]
        self.assertNotEqual(W.write_collision(parsed), data)
        self.assertEqual(W.write_collision(parsed, derive="proven"), data)

    def test_several_meshes_and_empty(self):
        for build in BUILDS:
            a = W.build_collision_mesh(*grid(3, 2), 1, 2, build)
            b = W.build_collision_mesh(*grid(5, 4, origin=(-50.0, 0.0, 0.0)), [i % 7 for i in range(40)], 0, build)
            data = W.write_collision(W.new_collision_file([a, b], build))
            parsed = C.parse_collision(data, strict=True)
            self.assertEqual(len(parsed.meshes), 2)
            self.assertEqual(sorted(t.material for t in parsed.meshes[1].triangles), sorted(i % 7 for i in range(40)))
            self.assertEqual(W.write_collision(parsed), data)
            empty = W.write_collision(W.new_collision_file([], build))
            self.assertEqual(C.parse_collision(empty).meshes, [])

    def test_errors(self):
        with self.assertRaises(Exception):
            W.build_collision_mesh([(0, 0, 0)], [(0, 1, 2)], 0, 0, "switch")
        with self.assertRaises(Exception):
            W.build_collision_mesh([(0, 0, 0)] * 3, [(0, 1, 2)], [1, 2], 0, "switch")
        with self.assertRaises(ValueError):
            W.derive_mesh(W.build_collision_mesh(*grid(1, 1), 0, 0, "ps3"), "ps3", "all")


if __name__ == "__main__":
    unittest.main()
