"""Stage 13: ``26_corrections15.md`` (W13-1 to W13-7), values the mesh-level ``.app`` writer kept.

There are no game files here. Files come from the test builders laid out by W11; a file that
follows the W13 rules is made by writing a builder file once with ``derive=True``. Expected values
are worked out from the description by hand.
"""

import struct
import unittest

import builders as B
import d3sno
from d3sno import FormatError
from d3sno import app_mesh as M
from d3sno import appearance as P
from d3sno import octree as O
from d3sno import sno_layout as L
from test_stage11 import SKIN
from test_stage12 import BONES, laid_out, ps3, switch_cloth, switch_rigid, switch_skinned, with_octree, x360_skinned

SKELETON_8 = [(0x100 + i, i - 1, (float(i), 0.0, 0.0)) for i in range(8)]
SKELETON_61 = [(0x200 + i, i - 1, (0.0, 0.0, 0.0)) for i in range(61)]
RIGID_FLAGGED = 0x55 | 1 << 15 | O.FLAG_OCTREE


def derived(data, build=None, rules=True):
    """The file as the writer derives it (W13-1, W13-2, W13-4 applied to every sub-object)."""
    return M.write_app_mesh(P.parse_appearance(data, build), derive=rules)


def set_mesh(draw, positions, indices, joints=None, weights=None):
    """Replace a draw's vertices (attributes copied from vertex 0) and indices."""
    n = len(positions)
    draw.positions[:] = [tuple(map(float, p)) for p in positions]
    for name in ("normals", "colours_1", "colours_2", "uv1", "uv2_raw"):
        values = getattr(draw, name)
        values[:] = [values[0]] * n
    if draw.skinned:
        draw.joints[:] = joints if joints is not None else [draw.joints[0]] * n
        draw.weights[:] = weights if weights is not None else [draw.weights[0]] * n
    draw.indices[:] = list(indices)


def grid(cells, z=0.0):
    positions = [(x, y, z) for y in range(cells + 1) for x in range(cells + 1)]
    indices = []
    for y in range(cells):
        for x in range(cells):
            a = y * (cells + 1) + x
            indices += [a, a + 1, a + cells + 1, a + 1, a + cells + 2, a + cells + 1]
    return positions, indices


def flagged_rigid():
    return laid_out(B.build_app_switch(pool_spare=0, sub_objects=[{
        "rigid": [(0, 0, 0), (10, 0, 0), (0, 10, 0)], "index_count": 3, "flags": RIGID_FLAGGED}]))


def with_root(data, low, size, build=None):
    """The file with a stored octree: one root over (low, size) with 8 empty leaves."""
    tree = L.parse_tree(data, "app", build)
    octree = O.Octree(nodes=[(low, size, tuple(O.LEAF_BIT | k for k in range(8)))],
                      leaves=[((0.0, 0.0, 0.0), size / 2, 0, 0)] * 8, maxima=(1, 1, 2))
    O.set_octree(tree, octree)
    return L.write_tree(tree)


def octree_of(data, build=None):
    return O.read_octree(L.parse_tree(data, "app", build))


def skinned_61():
    return laid_out(B.build_app_switch(bones=SKELETON_61, pool_spare=0,
                                       sub_objects=[{"skin": SKIN, "index_count": 3}]))


class Marker(unittest.TestCase):
    def test_version(self):
        self.assertGreaterEqual(tuple(int(x) for x in d3sno.__version__.split(".")), (0, 13, 0))
        self.assertGreaterEqual(d3sno.STAGE, 13)


class Rules(unittest.TestCase):
    def test_rule_set(self):
        self.assertEqual(M.rule_set(True), frozenset(M.RULES))
        self.assertEqual(M.rule_set(False), frozenset())
        self.assertEqual(M.rule_set("W13-1"), frozenset({"W13-1"}))
        self.assertEqual(M.rule_set(["W13-2", "W13-4"]), frozenset({"W13-2", "W13-4"}))
        with self.assertRaisesRegex(FormatError, "unknown rules"):
            M.rule_set({"W13-9"})

    def test_rules_on_unchanged_files(self):
        # a file that follows the rules rebuilds byte-identically with every rule on
        cases = [(switch_rigid(), None), (switch_rigid(">"), "x360"), (switch_skinned(), None),
                 (x360_skinned(), "x360"), (switch_cloth(), None), (with_octree(switch_rigid()), None),
                 (ps3(), "ps3"), (ps3(skinned=True, bones=BONES), "ps3"),
                 (ps3(skinned=True, bones=SKELETON_8, ranges=((48, 240), (48, 96))), "ps3"),
                 (ps3(fat=[(0.0, 0.0, 0.0), (4.0, 0.0, 0.0), (0.0, 4.0, 0.0)], cloth=True), "ps3"),
                 (flagged_rigid(), None)]
        for data, build in cases:
            once = derived(data, build)
            self.assertEqual(P.parse_appearance(once, build).warnings, [], build)
            self.assertEqual(derived(once, build), once, build)
            self.assertEqual(M.write_app_mesh(P.parse_appearance(once, build)), once, build)

    def test_without_the_option_nothing_changes(self):
        data = ps3(skinned=True, bones=BONES)
        self.assertEqual(M.write_app_mesh(P.parse_appearance(data, "ps3")), data)
        self.assertNotEqual(derived(data, "ps3"), data)  # the builder does not follow W13-1/2/4


# --- W13-1 ------------------------------------------------------------------------------------

class WorkBuffer(unittest.TestCase):
    def test_formula(self):
        # P = 16 + 160 + 224 + 208 = 608; Q = 16 + 144 + 96 + 448 = 704 -> 768
        self.assertEqual(M.ps3_skinned_work_size(3, 3, 10, (48, 0, 96, 0)), 768)
        # P = 608 + 160 + 2912 + 208 = 3888; Q = 608 + 1872 + 96 + 448 = 3024 -> 3968
        self.assertEqual(M.ps3_skinned_work_size(100, 300, 10, (48, 0, 96, 0)), 3968)
        # two ranges: span = 480 + 96 - 48 = 528; Q = 16 + 144 + 528 + 448 = 1136 -> 1152
        self.assertEqual(M.ps3_skinned_work_size(3, 3, 10, (48, 480, 48, 96)), 1152)
        # one range: span = size0, offset1 ignored
        self.assertEqual(M.ps3_skinned_work_size(3, 3, 10, (48, 480, 96, 0)), 768)
        # Q = 16 + 144 + 48 + 448 = 656 (16 past 640) -> 768; P = 608
        self.assertEqual(M.ps3_skinned_work_size(8, 3, 10, (48, 0, 48, 0)), 768)

    def test_written(self):
        out = derived(ps3(skinned=True, bones=BONES), "ps3", {"W13-1"})
        record = L.parse_tree(out, "app", "ps3").top.pairs[0x64].block.elements[0].pairs[0x100].block.elements[0]
        ranges = struct.unpack_from(">4H", record.raw, 0x60)  # the builder's ranges (W13-4 off)
        descriptor = record.pairs[0x08].block.data
        self.assertEqual(struct.unpack_from(">I", record.raw, 0x98)[0],
                         M.ps3_skinned_work_size(3, 3, descriptor[1], ranges))

    def test_refused_above_0xc000(self):
        app = P.parse_appearance(ps3(skinned=True, bones=BONES), "ps3")
        draw = app.sub_objects[0].draws[0]
        positions = [(float(i % 30), float(i // 30), 0.0) for i in range(1500)]
        set_mesh(draw, positions, list(range(1500)) * 3)
        with self.assertRaisesRegex(FormatError, r"draw 0: skinned draw work buffer \d+ is above 0xC000 \(W13-1\)"):
            M.write_app_mesh(app, derive=True)


# --- W13-2 ------------------------------------------------------------------------------------

class Steps(unittest.TestCase):
    def test_rule(self):
        # reach = 1000 + 1 + 1 = 1002: 1002 x 128 and x 64 are too large, 1002 x 32 = 32064
        self.assertEqual(M.quantisation_steps((0, 0, 0), (1000, 1000, 1000), (1, 2, 3), False)[0], 1 / 32)
        self.assertEqual(M.quantisation_steps((0, 0, 0), (10, 10, 10), (0, 0, 0), True), (1 / 512,) * 3)
        edge = 32767 / 128 - 1  # reach / step = 32767 exactly: allowed
        self.assertEqual(M.quantisation_steps((0, 0, 0), (edge, 0, 0), (0, 0, 0), False)[0], 1 / 128)
        self.assertEqual(M.quantisation_steps((0, 0, 0), (edge + 0.01, 0, 0), (0, 0, 0), False)[0], 1 / 64)
        # |c - b| counts: centre 100 away from the bias
        self.assertEqual(M.quantisation_steps((100, 0, 0), (0, 0, 0), (0, 0, 0), True)[0], 1 / 256)

    def test_unchanged_file_from_its_box(self):
        out = derived(switch_rigid(), None, {"W13-2"})
        sub = P.parse_appearance(out).sub_objects[0]
        self.assertEqual(sub.scale, (1 / 32,) * 3)  # box half 1000 around 0, bias (1, 2, 3)
        self.assertEqual(sub.bias, (1.0, 2.0, 3.0))  # W2-15: bias kept
        self.assertEqual(sub.draws[0].positions, P.parse_appearance(switch_rigid()).sub_objects[0].draws[0].positions)

    def test_all_zero_box_uses_the_vertices(self):
        out = derived(ps3(zero_bounds=True), "ps3", {"W13-2"})
        sub = P.parse_appearance(out, "ps3").sub_objects[0]
        # decoded vertices (1, 2, 3), (6, 2, 3), (1, 7, 3): reach 2.5 + 2.5 + 1 = 6 -> base 1/128
        self.assertEqual(sub.scale, (1 / 128,) * 3)
        # decoded x 1 .. 1001: centre 501, half 500, reach 500 + 500 + 1 = 1001 -> 1/32
        far = ps3(zero_bounds=True, vertices=((0, 0, 0), (2000, 0, 0), (0, 10, 0)))
        sub = P.parse_appearance(derived(far, "ps3", {"W13-2"}), "ps3").sub_objects[0]
        self.assertEqual(sub.scale, (1 / 32, 1 / 128, 1 / 128))

    def test_changed_mesh(self):
        # W2-3 would take 1/128 (255.9 x 128 = 32755); W13-2 adds one unit: 256.9 x 128 > 32767
        positions = [(-255.9, 0.0, 0.0), (255.9, 0.0, 0.0), (0.0, 1.0, 0.0)]
        for rules, step in ((False, 1 / 128), ({"W13-2"}, 1 / 64)):
            app = P.parse_appearance(switch_rigid())
            set_mesh(app.sub_objects[0].draws[0], positions, [0, 1, 2])
            sub = P.parse_appearance(M.write_app_mesh(app, derive=rules)).sub_objects[0]
            self.assertEqual(sub.bias[0], 0.0)
            self.assertEqual(sub.scale[0], step)


# --- W13-4 ------------------------------------------------------------------------------------

def draw_record(data, k=0):
    return L.parse_tree(data, "app", "ps3").top.pairs[0x64].block.elements[0].pairs[0x100].block.elements[k]


class BoneRanges(unittest.TestCase):
    def test_ranges(self):
        self.assertEqual(M.ps3_bone_ranges({3, 4, 5}), (144, 0, 144, 0))
        self.assertEqual(M.ps3_bone_ranges({1, 2, 7, 8}), (48, 336, 96, 96))
        self.assertEqual(M.ps3_bone_ranges({0, 3, 6}), (0, 144, 48, 192))  # tied gaps: the leftmost
        self.assertEqual(M.ps3_bone_ranges({0, 2, 3, 9}), (0, 432, 192, 48))  # largest gap 2..9 ... 3..9
        self.assertEqual(M.ps3_bone_ranges({5}), (240, 0, 48, 0))
        self.assertEqual(M.ps3_bone_ranges(set()), (0, 0, 0, 0))
        self.assertEqual(M.ranges_palette((48, 336, 96, 96)), [1, 2, 7, 8])
        self.assertEqual(M.ranges_palette(M.ps3_bone_ranges({0, 3, 6})), [0, 3, 4, 5, 6])  # gap bones kept

    def test_unchanged_draw(self):
        # builder ranges: offset0 48, offset1 96, size0 48, size1 48 (bones 1 and 2 as two
        # ranges); the slots name bones {1, 2} -> one range 48 + 96
        data = ps3(skinned=True, bones=BONES)
        before = P.parse_appearance(data, "ps3").sub_objects[0].draws[0]
        out = derived(data, "ps3", {"W13-4"})
        self.assertEqual(struct.unpack_from(">4H", draw_record(out).raw, 0x60), (48, 0, 96, 0))
        after = P.parse_appearance(out, "ps3").sub_objects[0].draws[0]
        self.assertEqual((after.joints, after.weights, after.palette), (before.joints, before.weights, [1, 2]))
        self.assertEqual(after.local_joints, before.local_joints)

    def test_zero_weight_slots_count_when_unchanged(self):
        # zero-weight slots name bone 1 (slot 0); weighted ones bones 1 and 2. A stored
        # zero-weight slot naming bone 7 widens B to {1, 2, 7}
        data = ps3(skinned=True, bones=SKELETON_8, ranges=((48, 336), (96, 48)))
        tree = L.parse_tree(data, "app", "ps3")
        record = tree.top.pairs[0x64].block.elements[0].pairs[0x100].block.elements[0]
        stream_b = bytearray(record.pairs[0x54].block.data)
        stream_b[5] = 2  # vertex 0, slot 2 (weight 0) -> palette position 2 = bone 7
        record.pairs[0x54].block.data = bytes(stream_b)
        data = L.write_tree(tree)
        out = derived(data, "ps3", {"W13-4"})
        self.assertEqual(struct.unpack_from(">4H", draw_record(out).raw, 0x60), (48, 336, 96, 48))
        self.assertEqual(out, data)

    def test_changed_draw(self):
        app = P.parse_appearance(ps3(skinned=True, bones=SKELETON_8), "ps3")
        draw = app.sub_objects[0].draws[0]
        draw.joints[0] = (1, 6, 0, 0)
        out = M.write_app_mesh(app, derive={"W13-4", "W13-1"})
        record = draw_record(out)
        ranges = struct.unpack_from(">4H", record.raw, 0x60)
        self.assertEqual(ranges, (48, 288, 96, 48))  # B = {1, 2, 6}: split after bone 2
        again = P.parse_appearance(out, "ps3").sub_objects[0].draws[0]
        self.assertEqual(again.joints, draw.joints)
        self.assertEqual(again.palette, [1, 2, 6])
        self.assertEqual(again.local_joints[0], (0, 2, 0, 0))  # zero-weight slots 0 (changed draw)
        self.assertEqual(struct.unpack_from(">I", record.raw, 0x98)[0],
                         M.ps3_skinned_work_size(3, 3, record.pairs[0x08].block.data[1], ranges))

    def test_joint_outside_the_skeleton(self):
        app = P.parse_appearance(ps3(skinned=True, bones=BONES), "ps3")
        app.sub_objects[0].draws[0].joints[0] = (9, 2, 0, 0)
        with self.assertRaisesRegex(FormatError, r"joint 9 is not one of the 3 bones \(W13-4\)"):
            M.write_app_mesh(app, derive={"W13-4"})
        app.sub_objects[0].draws[0].joints[0] = (2, 1, 0, 0)  # a skeleton bone outside the stored ranges ...
        out = M.write_app_mesh(app, derive={"W13-4"})       # ... fits once the ranges are derived
        self.assertEqual(P.parse_appearance(out, "ps3").sub_objects[0].draws[0].joints[0], (2, 1, 0, 0))


# --- W13-3 ------------------------------------------------------------------------------------

class Overlap(unittest.TestCase):
    def test_separating_axes(self):
        inside = ((0.2, 0.2, 0.2), (0.8, 0.2, 0.2), (0.2, 0.8, 0.2))
        self.assertTrue(O.triangle_overlaps_cube(inside, (0, 0, 0), 1.0))
        self.assertFalse(O.triangle_overlaps_cube(inside, (2, 0, 0), 1.0))
        far_plane = ((10, 0, 0), (0, 10, 0), (0, 0, 10))  # its box covers the cube, its plane does not
        self.assertFalse(O.triangle_overlaps_cube(far_plane, (0, 0, 0), 1.0))
        through = ((-5, 0.5, 0.5), (5, 0.5, 0.5), (0, 5, 0.5))
        self.assertTrue(O.triangle_overlaps_cube(through, (0, 0, 0), 1.0))
        edge_miss = ((1.6, -0.5, 0.5), (2.5, 0.5, 0.5), (1.6, 0.5, 5.0))  # misses near the x edge
        self.assertFalse(O.triangle_overlaps_cube(edge_miss, (0, 0, 0), 1.0))

    def test_half_open_and_widened(self):
        on_high_face = ((1.0, 0.2, 0.2), (1.0, 0.8, 0.2), (1.0, 0.2, 0.8))
        self.assertFalse(O.triangle_overlaps_cube(on_high_face, (0, 0, 0), 1.0))
        self.assertTrue(O.triangle_overlaps_cube(on_high_face, (1, 0, 0), 1.0))
        just_below = ((-2e-5, 0.2, 0.2), (-2e-5, 0.8, 0.2), (-2e-5, 0.2, 0.8))  # within 3e-5 x size
        self.assertTrue(O.triangle_overlaps_cube(just_below, (0, 0, 0), 1.0))
        further = ((-4e-5, 0.2, 0.2), (-4e-5, 0.8, 0.2), (-4e-5, 0.2, 0.8))
        self.assertFalse(O.triangle_overlaps_cube(further, (0, 0, 0), 1.0))

    def test_degenerate_triangle(self):
        point = ((0.5, 0.5, 0.5),) * 3  # coinciding corners: kept, overlaps where it lies
        self.assertTrue(O.triangle_overlaps_cube(point, (0, 0, 0), 1.0))

    def test_power_of_two(self):
        self.assertEqual([O.power_of_two_at_least(v) for v in (3, 4, 4.5, 0.3, 0)], [4, 4, 8, 0.5, 1])


def write_grid(cells, rules=True, data=None, extra=()):
    app = P.parse_appearance(data or flagged_rigid())
    positions, indices = grid(cells)
    set_mesh(app.sub_objects[0].draws[0], positions, list(indices) + list(extra))
    return M.write_app_mesh(app, derive=rules), app


class Octree(unittest.TestCase):
    def test_primitives(self):
        app = P.parse_appearance(flagged_rigid())
        draw = app.sub_objects[0].draws[0]
        draw.indices[:] = [0, 1, 2, 0, 0, 1, 2, 1, 0]  # the second triangle repeats an index
        self.assertEqual([(u, p) for u, p, _ in O.octree_triangles(app)], [(0, 0), (0, 6)])
        self.assertEqual(O.geo0_triangle_count(app), 3)  # repeated indices included
        app.sub_objects[0].flags &= ~O.FLAG_OCTREE
        self.assertEqual(O.octree_triangles(app), [])

    def test_user_data_of_records(self):
        # Switch: the packed list, then each skinning record; user data = sub + 65536 x draw
        app = P.parse_appearance(laid_out(B.build_app_switch(
            bones=[(0x10, -1, (0.0, 0.0, 0.0))], pool_spare=0,
            sub_objects=[{"rigid": [(0, 0, 0), (10, 0, 0), (0, 10, 0)], "index_count": 3},
                         {"skin": SKIN, "index_count": 3}])))
        for sub in app.sub_objects:
            sub.flags |= O.FLAG_OCTREE
        self.assertEqual([(u, p) for u, p, _ in O.octree_triangles(app)], [(0, 0), (1, 0)])
        app.sub_objects[1].draws.append(app.sub_objects[1].draws[0])
        self.assertEqual([u for u, _, _ in O.octree_triangles(app)], [0, 1, 1 + 65536])

    def test_built_in_the_writer(self):
        out, app = write_grid(30, extra=[0, 0, 1])
        tree = octree_of(out)
        again = P.parse_appearance(out)
        triangles = O.octree_triangles(again)
        self.assertEqual(len(triangles), 1800)  # the repeated-index triangle is not a primitive
        self.assertEqual(tree.maxima[:2], (1801, 1801))
        self.assertEqual(tree.maxima[2], 3602 if 3602 >= len(tree.primitives) else 7204)
        self.assertEqual(O.missing_primitives(tree, triangles), [])
        self.assertEqual(O.leaf_problems(tree, triangles), [])
        self.assertEqual(O.octree_problems(again, tree), [])
        self.assertGreater(len(tree.nodes), 1)  # more than 256 in a child of size >= 2: a node
        # the root: extent 30 -> 32, centred on the box (15, 15, 0)
        self.assertEqual(tree.root, ((-1.0, -1.0, -16.0), 32.0))
        self.assertEqual(O.build_octree(again, tree.root).nodes, tree.nodes)  # rebuilt the same
        self.assertEqual(M.write_app_mesh(again, derive=True), out)
        self.assertEqual(again.warnings, [])

    def test_subdivision(self):
        out, _ = write_grid(30)
        self.assertGreater(len(octree_of(out).nodes), 1)
        tree = octree_of(out)
        for n, (low, size, children) in enumerate(tree.nodes):
            for k, child in enumerate(children):
                half = size / 2
                corner = (low[0] + half * (k >> 2 & 1), low[1] + half * (k >> 1 & 1), low[2] + half * (k & 1))
                if child & O.LEAF_BIT:
                    leaf = tree.leaves[child & ~O.LEAF_BIT]
                    self.assertEqual((tuple(leaf[0]), leaf[1]), (corner, half))
                    self.assertTrue(leaf[3] <= O.SPLIT_LIMIT or half < O.MIN_NODE_SIZE)
                else:
                    self.assertGreater(child, n)  # pre-order
                    self.assertEqual((tuple(tree.nodes[child][0]), tree.nodes[child][1]), (corner, half))
        firsts = [leaf[2] for leaf in tree.leaves]
        self.assertEqual(firsts, sorted(firsts))  # leaves in walk order, primitives consecutive
        running = 0
        for leaf in tree.leaves:
            self.assertEqual(leaf[2], running)  # an empty leaf's first is the running total
            running += leaf[3]

    def test_four_times_maximum(self):
        # one triangle across the whole root lies in several leaves: 2 x 1 < primitives
        app = P.parse_appearance(flagged_rigid())
        set_mesh(app.sub_objects[0].draws[0], [(0, 0, 0), (16, 0, 0), (0, 16, 0)], [0, 1, 2])
        tree = octree_of(M.write_app_mesh(app, derive=True))
        self.assertGreater(len(tree.primitives), 2)
        # nMaxLeaves = T = 1 is raised to the 8 leaves (27_corrections16.md V15-5)
        self.assertEqual(tree.maxima, (1, 8, max(4, len(tree.primitives))))

    def test_stored_root_kept(self):
        data = with_root(flagged_rigid(), (-64.0, -64.0, -64.0), 128.0)
        out, _ = write_grid(20, data=data)
        self.assertEqual(octree_of(out).root, ((-64.0, -64.0, -64.0), 128.0))
        out, _ = write_grid(20, data=with_root(flagged_rigid(), (0.0, 0.0, 1.0), 64.0))  # z = 0 outside
        self.assertEqual(octree_of(out).root, ((-6.0, -6.0, -16.0), 32.0))

    def test_derived_root_on_its_high_face(self):
        # extent 4 -> size 4 would leave the high faces open: doubled to 8 (Q397)
        app = P.parse_appearance(flagged_rigid())
        set_mesh(app.sub_objects[0].draws[0], [(0, 0, 0), (4, 0, 0), (0, 4, 4)], [0, 1, 2])
        tree = octree_of(M.write_app_mesh(app, derive=True))
        self.assertEqual(tree.root, ((-2.0, -2.0, -2.0), 8.0))
        self.assertEqual(O.missing_primitives(tree, O.octree_triangles(P.parse_appearance(M.write_app_mesh(app, derive=True)))), [])

    def test_no_flag(self):
        # no flagged sub-object: a stored tree becomes a root with 8 empty leaves, no tree stays none
        data = with_root(switch_rigid(), (-8.0, -8.0, -8.0), 16.0)
        out, _ = write_grid(2, data=data)
        tree = octree_of(out)
        self.assertEqual((len(tree.nodes), len(tree.leaves), tree.primitives), (1, 8, []))
        self.assertEqual(tree.root, ((-8.0, -8.0, -8.0), 16.0))
        out, _ = write_grid(2, data=switch_rigid())
        self.assertIsNone(octree_of(out))

    def test_without_the_rule_emptied(self):
        out, _ = write_grid(4, rules={"W13-2"})
        self.assertIsNone(octree_of(out))
        top = L.parse_tree(out, "app").top
        self.assertEqual(top.raw[O.COUNTS_AT:O.COUNTS_AT + 24], bytes(24))

    def test_unchanged_triangles_keep_the_tree(self):
        data = with_octree(switch_rigid())
        app = P.parse_appearance(data)
        app.sub_objects[0].draws[0].colours_1[0] = b"\x09\x09\x09\x09"
        out = M.write_app_mesh(app, derive={"W13-3"})
        self.assertEqual(struct.unpack_from("<6I", L.parse_tree(out, "app").top.raw, M.STRUCT_OCTREE), (1, 1, 2, 1, 1, 1))

    def test_encode_read(self):
        out, _ = write_grid(3)
        for build, data in (("switch", out),):
            tree = octree_of(data)
            counts, nodes, leaves, prims = O.encode_octree(tree, "<")
            self.assertEqual(struct.unpack("<6I", counts), tree.counts)
            self.assertEqual((len(nodes[0]), len(leaves[0]), len(prims[0])), (48, 24, 8))


class PsThreeOctree(unittest.TestCase):
    def test_ps3_draw_primitives(self):
        data = ps3()
        tree = L.parse_tree(data, "app", "ps3")
        sub = tree.top.pairs[0x64].block.elements[0]
        struct.pack_into(">I", sub.raw, 0, struct.unpack_from(">I", sub.raw, 0)[0] | O.FLAG_OCTREE)
        app = P.parse_appearance(L.write_tree(tree), "ps3")
        positions, indices = grid(3)
        set_mesh(app.sub_objects[0].draws[0], positions, indices)
        out = M.write_app_mesh(app, derive=True)
        octree = O.read_octree(L.parse_tree(out, "app", "ps3"))
        again = P.parse_appearance(out, "ps3")
        self.assertEqual(sorted(octree.primitives)[:3], [(0, 0), (0, 3), (0, 6)])
        self.assertEqual(len(set(octree.primitives)), 18)
        self.assertEqual(O.octree_problems(again, octree), [])


# --- W13-5 ------------------------------------------------------------------------------------

def skin_records(data, build=None):
    return L.parse_tree(data, "app", build).top.pairs[0x64].block.elements[0].pairs[0xB8].block.elements


class Palette(unittest.TestCase):
    def test_first_use(self):
        joints = [(3, 1, 0, 0), (1, 5, 0, 0), (7, 0, 0, 0)]
        weights = [(200, 55, 0, 0), (100, 155, 0, 0), (0, 0, 0, 0)]
        self.assertEqual(M.first_use_palette(joints, weights), [3, 1, 5])

    def test_changed_record(self):
        data = laid_out(B.build_app_switch(bones=SKELETON_8, pool_spare=0, sub_objects=[{
            "skin": dict(SKIN, palette=[0, 4, 5]), "index_count": 3}]))  # bones 4 and 5: unused tail
        self.assertEqual(M.write_app_mesh(P.parse_appearance(data), derive={"W13-5", "W13-6", "W13-7"}), data)
        app = P.parse_appearance(data)
        draw = app.sub_objects[0].draws[0]
        draw.joints[1] = (6, 0, 0, 0)
        out = M.write_app_mesh(app, derive={"W13-5"})
        skin = skin_records(out)[0]
        self.assertEqual(struct.unpack_from("<2i", skin.pairs[0x04].block.data), (0, 6))  # first use
        self.assertEqual(struct.unpack_from("<i", skin.raw, 0)[0], 2)  # bone count
        vertex = skin.pairs[0x28].block.data
        self.assertEqual(vertex[28:32], bytes((0, 0, 0, 0)))  # the stored zero-weight slot 1 -> 0
        self.assertEqual(vertex[36 + 28:36 + 32], bytes((1, 0, 0, 0)))
        self.assertEqual(P.parse_appearance(out).sub_objects[0].draws[0].joints, draw.joints)

    def test_split_at_45_bones(self):
        app = P.parse_appearance(skinned_61())
        draw = app.sub_objects[0].draws[0]
        positions = [(float(v % 10), float(v // 10), 0.0) for v in range(61)]
        indices = [i for t in range(30) for i in (0, 2 * t + 1, 2 * t + 2)]
        joints = [(v, 0, 0, 0) for v in range(61)]
        set_mesh(draw, positions, indices, joints, [(255, 0, 0, 0)] * 61)
        out = M.write_app_mesh(app, derive=True)
        again = P.parse_appearance(out)
        self.assertEqual(again.warnings, [])
        draws = again.sub_objects[0].draws
        self.assertEqual([len(d.positions) for d in draws], [45, 17])  # 22 triangles, then 8
        self.assertEqual([len(d.indices) for d in draws], [66, 24])
        self.assertEqual(draws[0].palette, list(range(45)))
        self.assertEqual(draws[1].palette, [0] + list(range(45, 61)))
        self.assertEqual(draws[1].indices[:3], [0, 1, 2])  # (0, 45, 46) in the record's vertices
        rebuilt = [tuple(d.joints[i] for i in d.indices[t:t + 3]) for d in draws for t in range(0, len(d.indices), 3)]
        self.assertEqual(rebuilt, [tuple(joints[i] for i in indices[t:t + 3]) for t in range(0, 90, 3)])
        # W13-6: one entry per distinct vertex (vertex 0 is in both records)
        self.assertEqual(len(again.sub_objects[0].influences), 61)
        self.assertEqual(again.sub_objects[0].influences[45], ((45, 255), (0, 0), (0, 0)))

    def test_split_function(self):
        draw = P.parse_appearance(skinned_61()).sub_objects[0].draws[0]
        set_mesh(draw, [(0, 0, 0)] * 4, [0, 1, 2, 1, 2, 3], [(0, 0, 0, 0), (1, 0, 0, 0), (2, 0, 0, 0), (3, 0, 0, 0)],
                 [(255, 0, 0, 0)] * 4)
        pieces = M.split_skinned_draw(draw, limit=3)
        self.assertEqual([ids for _, ids in pieces], [[0, 1, 2], [1, 2, 3]])
        self.assertEqual(M.split_skinned_draw(draw)[0][0] is draw, True)  # fits: whole

    def test_bone_outside_the_skeleton(self):
        app = P.parse_appearance(switch_skinned())
        app.sub_objects[0].draws[0].joints[0] = (3, 0, 0, 0)
        with self.assertRaisesRegex(FormatError, "W13-5"):
            M.write_app_mesh(app, derive={"W13-5"})


# --- W13-6 ------------------------------------------------------------------------------------

class CompactList(unittest.TestCase):
    def test_switch_entries(self):
        app = P.parse_appearance(laid_out(B.build_app_switch(bones=SKELETON_8, pool_spare=0,
                                                              sub_objects=[{"skin": SKIN, "index_count": 3}])))
        draw = app.sub_objects[0].draws[0]
        draw.joints[0], draw.weights[0] = (3, 2, 1, 0), (100, 80, 50, 25)
        draw.joints[1], draw.weights[1] = (0, 5, 6, 0), (0, 200, 55, 0)
        out = M.write_app_mesh(app, derive={"W13-5", "W13-6"})
        influences = P.parse_appearance(out).sub_objects[0].influences
        self.assertEqual(influences[0], ((3, 100), (2, 80), (1, 50)))  # at most three, slot order
        self.assertEqual(influences[1], ((5, 200), (6, 55), (0, 0)))  # weighted slots only
        self.assertEqual(influences[2], ((0, 255), (0, 0), (0, 0)))
        self.assertEqual(M.influence_entry(draw, 2), bytes((0, 255, 0, 0, 0, 0)))

    def test_ps3_presence_and_length(self):
        fat = [(0.0, 0.0, 0.0), (4.0, 0.0, 0.0), (0.0, 4.0, 0.0)]
        # fat (cloth) vertices in a file with bones: 6 x the fat vertex count
        app = P.parse_appearance(ps3(fat=fat, cloth=True, bones=BONES), "ps3")
        app.sub_objects[0].fat_draw.positions[1] = (5.0, 0.0, 0.0)
        out = M.write_app_mesh(app, derive={"W13-6"})
        record = L.parse_tree(out, "app", "ps3").top.pairs[0x64].block.elements[0]
        self.assertEqual(len(M._block_bytes(record.pairs.get(0x10C))), 18)
        # no bones: no list
        app = P.parse_appearance(ps3(fat=fat, cloth=True), "ps3")
        app.sub_objects[0].fat_draw.positions[1] = (5.0, 0.0, 0.0)
        record = L.parse_tree(M.write_app_mesh(app, derive={"W13-6"}), "app", "ps3").top.pairs[0x64].block.elements[0]
        self.assertEqual(M._block_bytes(record.pairs.get(0x10C)), b"")

    def test_ps3_skinned_cloth_length(self):
        fat = [(0.0, 0.0, 0.0), (4.0, 0.0, 0.0), (0.0, 4.0, 0.0), (4.0, 4.0, 0.0)]
        app = P.parse_appearance(ps3(skinned=True, bones=BONES, fat=fat, cloth=True, cloth_draw=True), "ps3")
        app.sub_objects[0].draws[0].weights[0] = (255, 0, 0, 0)
        record = L.parse_tree(M.write_app_mesh(app, derive={"W13-6"}), "app", "ps3").top.pairs[0x64].block.elements[0]
        self.assertEqual(len(M._block_bytes(record.pairs.get(0x10C))), 18)  # 6 x 3 draw vertices

    def test_ps3_skinned_without_cloth_has_none(self):
        tree = L.parse_tree(ps3(skinned=True, bones=BONES), "app", "ps3")
        tree.top.pairs[0x64].block.elements[0].pairs[0x10C] = L.PairValue(L.Block(bytes(18)))
        app = P.parse_appearance(L.write_tree(tree), "ps3")
        app.sub_objects[0].draws[0].weights[0] = (255, 0, 0, 0)
        record = L.parse_tree(M.write_app_mesh(app, derive={"W13-6"}), "app", "ps3").top.pairs[0x64].block.elements[0]
        self.assertEqual(M._block_bytes(record.pairs.get(0x10C)), b"")


# --- W13-7 ------------------------------------------------------------------------------------

def empty(draw, keep_vertices=False):
    if not keep_vertices:
        set_mesh(draw, [(0, 0, 0)], [])
        for name in ("positions", "normals", "colours_1", "colours_2", "uv1", "uv2_raw", "joints", "weights"):
            getattr(draw, name).clear()
    draw.indices.clear()


class EmptyRecords(unittest.TestCase):
    def test_switch_refused_or_dropped(self):
        for keep in (False, True):
            app = P.parse_appearance(switch_skinned())
            empty(app.sub_objects[0].draws[0], keep)
            with self.assertRaisesRegex(FormatError, "W13-7"):
                M.write_app_mesh(app)
            out = M.write_app_mesh(app, derive={"W13-7"})
            tree = L.parse_tree(out, "app")
            self.assertIsNone(tree.top.pairs[0x64].block.elements[0].pairs[0xB8].block)
            self.assertEqual(P.parse_appearance(out).sub_objects[0].draws, [])

    def test_ps3_dropped(self):
        app = P.parse_appearance(ps3(skinned=True, bones=BONES, extra_draw_format=0x83), "ps3")
        empty(app.sub_objects[0].draws[0])
        with self.assertRaisesRegex(FormatError, "empty ps3 draw"):
            M.write_app_mesh(app)
        out = M.write_app_mesh(app, derive={"W13-7"})
        record = L.parse_tree(out, "app", "ps3").top.pairs[0x64].block.elements[0]
        self.assertEqual(struct.unpack_from(">i", record.raw, 0xFC)[0], 1)
        draws = P.parse_appearance(out, "ps3").sub_objects[0].draws
        self.assertEqual([len(d.positions) for d in draws], [3])

    def test_other_records_kept(self):
        data = laid_out(B.build_app_switch(bones=SKELETON_8, pool_spare=0,
                                           sub_objects=[{"skin": SKIN, "index_count": 3}]))
        tree = L.parse_tree(data, "app")
        value = tree.top.pairs[0x64].block.elements[0].pairs[0xB8]
        first = value.block.elements[0]
        second = L.Record(first.name, bytearray(first.raw), dict(first.pairs))
        value.block.elements.append(second)
        data = L.write_tree(tree)
        app = P.parse_appearance(data)
        empty(app.sub_objects[0].draws[0])
        out = M.write_app_mesh(app, derive={"W13-7"})
        self.assertEqual(len(skin_records(out)), 1)
        self.assertEqual(len(P.parse_appearance(out).sub_objects[0].draws[0].positions), 3)


if __name__ == "__main__":
    unittest.main()
