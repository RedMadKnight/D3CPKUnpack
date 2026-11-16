"""Stage 12: ``25_corrections14.md`` (the mesh-level ``.app`` writer W2, the ``.can`` / ``.nav``
writers W3-1 to W3-5, the ps3 globals layout W3-6 to W3-8, the index-pool note).

There are no game files here. ``.app`` files are the test builders' files laid out by W11 (with no
spare index-pool tail, W2-12); the ps3 globals file is laid out by hand from W3-6 to W3-8; child
files come from the builders and from the rules.
"""

import math
import struct
import unittest

import builders as B
import d3sno
from d3sno import FormatError
from d3sno import animation as A
from d3sno import app_mesh as M
from d3sno import appearance as P
from d3sno import children as C
from d3sno import navigation as N
from d3sno import sno_layout as L
from test_stage11 import BODY, SKIN, HandLayout, header

BONES = [(0xAA, -1, (1.0, 2.0, 3.0)), (0xBB, 0, (2.0, 2.0, 3.0)), (0xCC, 1, (2.0, 4.0, 3.0))]
TRI = [(0, 0, 0), (10, 0, 0), (0, 10, 0)]


def laid_out(data, build=None):
    return L.write_tree(L.parse_tree(data, "app", build))


def switch_rigid(order="<"):
    version = 302 if order == ">" else 306
    return laid_out(B.build_app_switch(version=version, order=order, pool_spare=0),
                    "x360" if order == ">" else "switch")


def switch_skinned():
    return laid_out(B.build_app_switch(bones=[(0x10, -1, (0.0, 0.0, 0.0))], pool_spare=0,
                                       sub_objects=[{"skin": SKIN, "index_count": 3}]))


def with_octree(data, build=None):
    """The file with a one-node octree (maxima 1, 1, 2; counts 1, 1, 1) laid out by W11."""
    tree = L.parse_tree(data, "app", build)
    order = "<" if tree.build == "switch" else ">"
    struct.pack_into(order + "6I", tree.top.raw, M.STRUCT_OCTREE, 1, 1, 2, 1, 1, 1)
    for offset, name in zip(M.OCTREE_PAIRS, ("OctreeNode", "OctreeLeaf", "OctreePrimitive")):
        element = L.new_record(tree.build, "app", name)
        element.raw[:] = bytes([0x5A]) * len(element.raw)
        tree.top.pairs[offset] = L.PairValue(L.Block(elements=[element]))
    return L.write_tree(tree)


def x360_skinned():
    return laid_out(B.build_app_switch(version=302, order=">", bones=[(0x10, -1, (0.0, 0.0, 0.0))],
                                       pool_spare=0, sub_objects=[{"skin": SKIN, "index_count": 3}]), "x360")


def switch_cloth():
    return laid_out(B.build_app_switch(sub_objects=[{"index_count": 3, "cloth": True}], pool_spare=0))


def ps3(**options):
    """The ps3 builder's file laid out by W11, with descriptor byte 1 set by W2-9 (the builder
    always writes 8, the value of the rigid format only)."""
    tree = L.parse_tree(B.build_app_ps3(**options), "app", "ps3")
    for record in tree.top.pairs[0x64].block.elements:
        draws = record.pairs.get(0x100)
        for draw in draws.block.elements if draws is not None and draws.block else []:
            descriptor = bytearray(draw.pairs[0x08].block.data)
            descriptor[1] = M.ps3_command_space(descriptor[0], struct.unpack_from(">H", descriptor, 10)[0])
            draw.pairs[0x08].block.data = bytes(descriptor)
    return L.write_tree(tree)


class Marker(unittest.TestCase):
    def test_version(self):
        self.assertTrue(d3sno.__version__.startswith("0.12."))
        self.assertEqual(d3sno.STAGE, 12)


# --- Section 1: the mesh-level .app writer --------------------------------------------------------

class Unchanged(unittest.TestCase):
    def test_source_bytes_back(self):
        cases = [(switch_rigid(), None), (switch_rigid(">"), "x360"), (switch_skinned(), None),
                 (x360_skinned(), "x360"), (with_octree(switch_rigid()), None),
                 (switch_cloth(), None), (ps3(), "ps3"), (ps3(skinned=True, bones=BONES), "ps3"),
                 (ps3(fat=[(0.0, 0.0, 0.0), (4.0, 0.0, 0.0), (0.0, 4.0, 0.0)], cloth=True), "ps3")]
        for data, build in cases:
            app = P.parse_appearance(data, build)
            self.assertEqual(M.write_app_mesh(app), data, build)

    def test_zero_weight_slot_kept(self):
        # SKIN stores slot 1 with weight 0 in each vertex (W2-6: kept as stored)
        app = P.parse_appearance(switch_skinned())
        raw = L.find_pair(L.parse_tree(bytes(app.source), "app"),
                          "tStructure.tGeoSets[0].ptSubObjects[0].ptSkinningInfo[0].serPackVertList")[2].block.data
        self.assertEqual(raw[28:32], bytes((0, 1, 0, 0)))
        self.assertEqual(M.write_app_mesh(app), bytes(app.source))


class Encoders(unittest.TestCase):
    def test_vertex_parts(self):
        self.assertEqual(M.normal_bytes((0.5, 0.0, -0.5)), bytes((192, 128, 64)))
        self.assertEqual(M.normal_bytes((1.5, -2.0, 0.0)), bytes((255, 0, 128)))
        self.assertEqual(M.normal_bytes((0.75, -0.75, 0.25)), bytes((224, 32, 160)))
        self.assertEqual(M.uv1_words((1.0, -0.5)), (0x8200, 0x8000 - 256))
        with self.assertRaises(FormatError):
            M.uv1_words((200.0, 0.0))
        self.assertEqual(M.raw_coordinates((1.5, 2.0, -1.0), (1.0, 2.0, 0.0), (0.5, 0.25, 1 / 128)),
                         (1, 0, -128))
        with self.assertRaises(FormatError):
            M.raw_coordinates((1000.0, 0, 0), (0, 0, 0), (1 / 512,) * 3)

    def test_w2_2_w2_3_fixed_point(self):
        positions = [(-3.2, 0.0, 100.0), (4.6, 1.0, 100.5)]
        bias = M.fixed_point_bias(positions)
        self.assertEqual(bias, (1.0, 0.0, 100.0))
        steps = M.fixed_point_steps(positions, bias, has_bones=True)
        self.assertEqual(steps, (1 / 512, 1 / 512, 1 / 512))
        far = [(0.0, 0.0, 0.0), (200.0, 0.0, 0.0)]
        steps = M.fixed_point_steps(far, (100.0, 0.0, 0.0), has_bones=True)
        self.assertEqual(steps[0], 1 / 256)  # 100 x 512 > 32767, 100 x 256 fits
        self.assertEqual(M.fixed_point_steps(far, (100.0, 0.0, 0.0), has_bones=False)[1], 1 / 128)
        for p in far:
            M.raw_coordinates(p, (100.0, 0.0, 0.0), steps)

    def test_w2_9_command_space(self):
        # w = ceil((i - 1) / 256); n = 28 + w or 36 + w; byte 1 = ceil((n + ceil(n / 32)) / 4)
        self.assertEqual(M.ps3_command_space(P.FORMAT_RIGID, 3), 8)       # n 29
        self.assertEqual(M.ps3_command_space(P.FORMAT_SKINNED, 3), 10)    # n 37
        self.assertEqual(M.ps3_command_space(P.FORMAT_RIGID, 1), 8)       # n 28
        self.assertEqual(M.ps3_command_space(P.FORMAT_RIGID, 258), 8)     # n 30
        self.assertEqual(M.ps3_command_space(P.FORMAT_RIGID, 1000), 9)    # n 32 -> 33
        self.assertEqual(M.ps3_command_space(P.FORMAT_SKINNED, 3000), 13)  # n 48 -> 50

    def test_w2_10_work_size(self):
        # 0x81: align(12 x align(3, 8) + align(6, 16) + 16 x 8 + 192, 128) = align(432, 128)
        self.assertEqual(M.ps3_work_size(P.FORMAT_RIGID, 3, 3), 512)
        # 0x83: max(stored, align(28 x 8 + 16 + 160 + 336, 128) = 768)
        self.assertEqual(M.ps3_work_size(P.FORMAT_SKINNED, 3, 3, stored=640), 768)
        self.assertEqual(M.ps3_work_size(P.FORMAT_SKINNED, 3, 3, stored=1024), 1024)
        with self.assertRaises(FormatError):
            M.ps3_work_size(P.FORMAT_SKINNED, 1600, 3000)


class SwitchChanges(unittest.TestCase):
    def test_small_move_keeps_fixed_point(self):
        app = P.parse_appearance(switch_rigid())
        sub = app.sub_objects[0]
        bias, scale = sub.bias, sub.scale
        draw = sub.draws[0]
        draw.positions[1] = (draw.positions[1][0] + scale[0] * 5, *draw.positions[1][1:])
        out = M.write_app_mesh(app)
        again = P.parse_appearance(out)
        self.assertEqual(again.warnings, [])
        # W2-2: the bias stays while within 1 of the new box centre (the builder's is not)
        centre = [(min(p[k] for p in draw.positions) + max(p[k] for p in draw.positions)) / 2 for k in range(3)]
        want = bias if all(abs(bias[k] - centre[k]) <= 1 for k in range(3)) else M.fixed_point_bias(draw.positions)
        steps = M.fixed_point_steps(draw.positions, want, has_bones=False)
        self.assertEqual((again.sub_objects[0].bias, again.sub_objects[0].scale), (want, steps))
        self.assertEqual(again.sub_objects[0].draws[0].positions, draw.positions)
        self.assertTrue(L.is_canonical(out, "app"))
        # W2-4: the box is the decoded extremes
        xs = [p[0] for p in draw.positions]
        self.assertAlmostEqual(again.sub_objects[0].bounds_centre[0], (min(xs) + max(xs)) / 2, places=5)
        self.assertAlmostEqual(again.sub_objects[0].bounds_half[0], (max(xs) - min(xs)) / 2, places=5)

    def test_w2_13_octree_kept_while_triangles_stay(self):
        app = P.parse_appearance(with_octree(switch_rigid()))
        app.sub_objects[0].draws[0].colours_1[0] = b"\x09\x09\x09\x09"  # not a triangle change
        top = L.parse_tree(M.write_app_mesh(app), "app").top
        self.assertEqual(struct.unpack_from("<6I", top.raw, M.STRUCT_OCTREE), (1, 1, 2, 1, 1, 1))
        self.assertTrue(all(top.pairs[offset].block is not None for offset in M.OCTREE_PAIRS))

    def test_w2_11_w2_13_structure_after_a_move(self):
        app = P.parse_appearance(with_octree(switch_rigid()))
        app.sub_objects[0].draws[0].positions[2] = (0.0, 12.0, 0.0)
        out = M.write_app_mesh(app)
        top = L.parse_tree(out, "app").top
        sub = P.parse_appearance(out).sub_objects[0]
        box = struct.unpack_from("<6f", top.raw, M.STRUCT_BOX)
        for k in range(3):
            self.assertAlmostEqual(box[k], sub.bounds_centre[k], places=5)
            self.assertAlmostEqual(box[k + 3], sub.bounds_half[k], places=5)
        sphere = struct.unpack_from("<4f", top.raw, M.STRUCT_SPHERE)
        self.assertAlmostEqual(sphere[3], math.dist((0, 0, 0), sub.bounds_half), places=4)
        self.assertEqual(top.raw[M.STRUCT_OCTREE:M.STRUCT_OCTREE + 0x18], bytes(0x18))
        for offset in M.OCTREE_PAIRS:
            self.assertIsNone(top.pairs[offset].block)

    def test_new_vertex_and_triangle(self):
        app = P.parse_appearance(switch_rigid())
        draw = app.sub_objects[0].draws[0]
        draw.positions.append((10.0, 10.0, 0.0))
        draw.normals.append((0.0, 0.0, 1.0))
        draw.colours_1.append(b"\x01\x02\x03\x04")
        draw.colours_2.append(b"\x05\x06\x07\x08")
        draw.uv1.append((0.5, 0.5))
        draw.uv2_raw.append((0x8000, 0x8000))
        draw.indices += [1, 3, 2]
        out = M.write_app_mesh(app)
        again = P.parse_appearance(out)
        self.assertEqual(again.warnings, [])
        sub = again.sub_objects[0]
        self.assertEqual((len(sub.draws[0].positions), sub.index_count), (4, 6))  # W2-1
        self.assertEqual(sub.draws[0].indices, [0, 1, 2, 1, 3, 2])
        self.assertEqual(sub.draws[0].uv1[3], (0.5, 0.5))
        self.assertEqual(M.index_pool_problems(again), [])  # W2-12

    def test_far_move_new_fixed_point(self):
        app = P.parse_appearance(switch_rigid())
        draw = app.sub_objects[0].draws[0]
        draw.positions[:] = [(500.0, 0.0, 0.0), (520.0, 0.0, 0.0), (500.0, 30.0, 0.0)]
        out = M.write_app_mesh(app)
        sub = P.parse_appearance(out).sub_objects[0]
        self.assertEqual(sub.bias, (510.0, 15.0, 0.0))  # W2-2: round(box centre)
        self.assertEqual(sub.scale, (1 / 128,) * 3)      # W2-3: no bones, base 1/128 fits
        self.assertEqual(sub.draws[0].positions, draw.positions)

    def test_x360_byte_order(self):
        app = P.parse_appearance(switch_rigid(">"), "x360")
        draw = app.sub_objects[0].draws[0]
        draw.colours_1[0] = b"\x11\x22\x33\x44"
        out = M.write_app_mesh(app)
        again = P.parse_appearance(out, "x360")
        self.assertEqual(again.sub_objects[0].draws[0].colours_1[0], b"\x11\x22\x33\x44")
        raw = L.parse_tree(out, "app", "x360").top.pairs[0x64].block.elements[0].pairs[0xB0].block.data
        self.assertEqual(raw[12:16], b"\x44\x33\x22\x11")  # W2-5: byte-reversed group

    def test_skinned_new_vertex(self):
        app = P.parse_appearance(switch_skinned())
        draw = app.sub_objects[0].draws[0]
        draw.positions.append((5.0, 5.0, 0.0))
        draw.normals.append((0.0, 0.0, 1.0))
        draw.colours_1.append(bytes(4))
        draw.colours_2.append(bytes(4))
        draw.uv1.append((0.0, 0.0))
        draw.uv2_raw.append((0x8000, 0x8000))
        draw.joints.append((0, 0, 0, 0))
        draw.weights.append((255, 0, 0, 0))
        draw.indices += [0, 3, 1]
        out = M.write_app_mesh(app)
        tree = L.parse_tree(out, "app")
        skin = tree.top.pairs[0x64].block.elements[0].pairs[0xB8].block.elements[0]
        self.assertEqual(struct.unpack_from("<i", skin.raw, 0x18)[0], 4)
        self.assertEqual(struct.unpack_from("<i", skin.raw, 0x30)[0], 6)
        self.assertEqual(len(skin.pairs[0x40].block.data), 12)  # W2-1: 2 x count, no padding
        vertices = skin.pairs[0x28].block.data
        self.assertEqual(vertices[3 * 36 + 28:3 * 36 + 32], bytes(4))   # new vertex: slot 0
        self.assertEqual(vertices[0 * 36 + 28:0 * 36 + 32], bytes((0, 1, 0, 0)))  # kept
        again = P.parse_appearance(out)
        self.assertEqual(len(again.sub_objects[0].draws[0].positions), 4)

    def test_x360_skinned_groups_reversed(self):
        app = P.parse_appearance(x360_skinned(), "x360")
        draw = app.sub_objects[0].draws[0]
        draw.weights[0] = (128, 127, 0, 0)
        draw.joints[0] = (0, 0, 0, 0)
        out = M.write_app_mesh(app)
        skin = L.parse_tree(out, "app", "x360").top.pairs[0x64].block.elements[0].pairs[0xB8].block.elements[0]
        vertex, second = skin.pairs[0x28].block.data[:36], skin.pairs[0x28].block.data[36:72]
        self.assertEqual(vertex[28:32], bytes(4))  # both weighted slots name palette position 0
        self.assertEqual(vertex[32:36], bytes((128, 127, 0, 0))[::-1])
        self.assertEqual(second[28:32], bytes((0, 1, 0, 0))[::-1])  # zero-weight slot 1 kept
        self.assertEqual(P.parse_appearance(out, "x360").sub_objects[0].draws[0].weights[0], (128, 127, 0, 0))

    def test_joint_not_in_palette(self):
        app = P.parse_appearance(switch_skinned())
        app.sub_objects[0].draws[0].joints[0] = (2, 0, 0, 0)
        with self.assertRaisesRegex(FormatError, "W2-6"):
            M.write_app_mesh(app)

    def test_fat_vertices(self):
        app = P.parse_appearance(switch_cloth())
        fat = app.sub_objects[0].fat_draw
        fat.positions[0] = (fat.positions[0][0] - 1.0, *fat.positions[0][1:])
        out = M.write_app_mesh(app)
        again = P.parse_appearance(out)
        self.assertEqual(again.sub_objects[0].fat_draw.positions[0], fat.positions[0])
        xs = [p[0] for p in fat.positions]
        self.assertAlmostEqual(again.sub_objects[0].bounds_centre[0], (min(xs) + max(xs)) / 2, places=5)

    def test_sub_object_count_fixed(self):
        app = P.parse_appearance(switch_rigid())
        app.sub_objects.append(app.sub_objects[0])
        with self.assertRaises(FormatError):
            M.write_app_mesh(app)


class Ps3Changes(unittest.TestCase):
    def test_rigid_draw(self):
        app = P.parse_appearance(ps3(), "ps3")
        draw = app.sub_objects[0].draws[0]
        for k in range(6):  # 9 vertices: the work area grows from 16 to 32 bytes
            draw.positions.append((10.0 + k, 10.0, 0.0))
            draw.normals.append((0.0, 0.0, 1.0))
            draw.colours_1.append(bytes(4))
            draw.colours_2.append(bytes(4))
            draw.uv1.append((0.0, 0.0))
            draw.uv2_raw.append((0x8000, 0x8000))
            draw.indices += [1, 3 + k, 2]
        out = M.write_app_mesh(app)
        again = P.parse_appearance(out, "ps3")
        self.assertEqual(again.warnings, [])  # the G1-5 rules hold for the new draw
        record = L.parse_tree(out, "app", "ps3").top.pairs[0x64].block.elements[0] \
            .pairs[0x100].block.elements[0]
        self.assertEqual(struct.unpack_from(">II", record.raw, 0x98), (M.ps3_work_size(0x81, 9, 21), 32))
        descriptor = record.pairs[0x08].block.data
        self.assertEqual(descriptor[1], M.ps3_command_space(0x81, 21))
        self.assertEqual(struct.unpack_from(">HH", descriptor, 8), (9, 21))

    def test_skinned_draw(self):
        app = P.parse_appearance(ps3(skinned=True, bones=BONES), "ps3")
        draw = app.sub_objects[0].draws[0]
        draw.weights[0] = (255, 0, 0, 0)
        out = M.write_app_mesh(app)
        again = P.parse_appearance(out, "ps3")
        self.assertEqual(again.sub_objects[0].draws[0].weights[0], (255, 0, 0, 0))
        # a zero-weight slot keeps its stored value (stream B as stored, W2-6)
        self.assertEqual(again.sub_objects[0].draws[0].local_joints[0][1:],
                         app.sub_objects[0].draws[0].local_joints[0][1:])

    def test_ps3_fat(self):
        fat = [(0.0, 0.0, 0.0), (4.0, 0.0, 0.0), (0.0, 4.0, 0.0)]
        app = P.parse_appearance(ps3(fat=fat, cloth=True), "ps3")
        sub = app.sub_objects[0]
        sub.fat_draw.positions[1] = (5.0, 0.0, 0.0)
        out = M.write_app_mesh(app)
        again = P.parse_appearance(out, "ps3")
        self.assertEqual(again.sub_objects[0].fat_draw.positions[1], (5.0, 0.0, 0.0))
        self.assertEqual(again.sub_objects[0].index_count, 3)


class SharedBlocks(unittest.TestCase):
    def test_w2_14_changed_block_gets_its_own(self):
        app = P.parse_appearance(ps3(), "ps3")
        tree = L.parse_tree(bytes(app.source), "app", "ps3")
        draw_record = tree.top.pairs[0x64].block.elements[0].pairs[0x100].block.elements[0]
        shared = L.PairValue(draw_record.pairs[0x08].block, None, shared=True)
        other = L.Record("x", bytearray(8), {0: shared})
        self.assertTrue(M._set_bytes(draw_record, 0x08, b"N" * 16))
        self.assertIsNot(draw_record.pairs[0x08].block, other.pairs[0].block)
        self.assertNotEqual(other.pairs[0].block.data, b"N" * 16)  # the sharing pair keeps the old
        self.assertFalse(M._set_bytes(draw_record, 0x08, b"N" * 16))  # same bytes: kept


class IndexPool(unittest.TestCase):
    def test_no_spare_tail(self):
        self.assertEqual(M.index_pool_problems(P.parse_appearance(switch_rigid())), [])
        spare = P.parse_appearance(laid_out(B.build_app_switch(pool_spare=2)))
        self.assertTrue(any("W2-12" in p for p in M.index_pool_problems(spare)))
        # the writer lays the pool out without the tail
        out = M.write_app_mesh(spare)
        self.assertEqual(M.index_pool_problems(P.parse_appearance(out)), [])


# --- Section 2: .can and .nav --------------------------------------------------------------------

def perms(build="ps3", count=2):
    return [B.simple_permutation(build, name=f"P{i}") for i in range(count)]


class CanWriter(unittest.TestCase):
    def test_w3_1_round_trip(self):
        for build in ("ps3", "x360"):
            data = B.build_can(perms(build))
            self.assertEqual(C.write_can(A.parse_can(data)), data)
            ani = A.parse_ani(B.build_ani(build, perms(build)), build, can=data)
            self.assertEqual(C.write_can(ani), data)
            self.assertEqual(C.can_count_problems(ani), [])

    def test_w3_2_tables(self):
        perm = A.CanPermutation([1, 2], [], [b"", b""], [b"", b""], [b"", b""], [], [(0.0, 0.0, 0.0)])
        data = C.encode_can_permutation(perm)
        # sizes: hashes 8, blend 0, three tables of 32 (written in full), attachments 0, offsets 12
        self.assertEqual(len(data), (4 + 8) + 4 + 3 * (4 + 32) + 4 + (4 + 12))
        blob_a, blob_b = b"\x01", b"\x02" * 3
        section = C._curve_section([blob_a, b"", blob_b], 3)
        self.assertEqual(struct.unpack_from(">I", section, 0)[0], 48)  # the table only
        table = section[4:52]
        self.assertEqual(struct.unpack_from(">4I", table, 0), (48, 1, 0, 0))
        self.assertEqual(struct.unpack_from(">4I", table, 16), (0, 0, 0, 0))
        self.assertEqual(struct.unpack_from(">4I", table, 32), (56, 3, 0, 0))  # 49 -> next multiple of 8
        self.assertEqual(section[4 + 49:4 + 56], bytes(7))  # gap bytes zero
        self.assertEqual(len(section), 4 + 59)  # ends at the last blob, no alignment

    def test_w3_3_blob_parts(self):
        order = ">"
        blobs = [("translation", B.translation_blob([0, 2], [(1.0, 2.0, 3.0), (3.0, 2.0, 1.0)], 4, 3, order)),
                 ("translation", B.translation_blob([0], [(5, 6, 7)], 2, 3, order)),
                 ("rotation", B.rotation_blob([0, 1, 2], [(1, 2, 3, 4)] * 3, 3, order)),
                 ("scale", B.scale_blob([0, 2], [127, 254], 1, 3, order)),
                 ("translation", B.translation_blob([0, 300], [(1, 2, 3), (4, 5, 6)], 1, 301, order))]
        for kind, blob in blobs:
            keyframes = 301 if len(blob) > 4 and kind == "translation" and blob[:2] == b"\x00\x02" else 3
            parts = C.curve_parts(blob, kind, keyframes)
            encoded = C.encode_curve(parts, keyframes)
            self.assertEqual(encoded, blob + bytes(-len(blob) % 4), kind)
            self.assertEqual(len(encoded) % 4, 0)

    def test_key_width(self):
        self.assertEqual([C.key_width(k) for k in (255, 256, 65535, 65536)], [1, 2, 2, 4])

    def test_w3_5_one_more_keyframe(self):
        for build in ("ps3", "x360"):
            ani_data = B.build_ani(build, perms(build))
            can_data = B.build_can(perms(build))
            ani = A.parse_ani(ani_data, build, can=can_data)
            before = A.decode_permutation(ani.permutations[0])
            C.extend_keyframes(ani.permutations[0])
            new_ani = C.update_ani_counts(ani_data, ani)
            changed = [i for i in range(len(ani_data)) if ani_data[i] != new_ani[i]]
            self.assertTrue(changed and all(0x50 + 0xA8 <= i < 0x50 + 0xAC for i in changed))  # the count only
            new_can = C.write_can(ani)
            again = A.parse_ani(new_ani, build, can=new_can, strict=True)
            self.assertEqual(again.permutations[0].keyframes, 4)
            self.assertEqual(again.warnings, [])
            self.assertEqual(L.is_canonical(new_ani, "ani", build), L.is_canonical(ani_data, "ani", build))
            self.assertEqual(C.write_can(A.parse_can(new_can)), new_can)
            after = A.decode_permutation(again.permutations[0])
            last = A.sample_curve(after.translation[0], 3, (0.0, 0.0, 0.0))
            previous = A.sample_curve(before.translation[0], 2, (0.0, 0.0, 0.0))
            self.assertEqual(last, previous)  # the added keyframe holds the last pose

    def test_w3_5_stretch_past_255(self):
        ani_data = B.build_ani("ps3", perms())
        ani = A.parse_ani(ani_data, "ps3", can=B.build_can(perms()))
        C.extend_keyframes(ani.permutations[1], 300)  # 3 -> 303 keyframes: width 2
        new_can = C.write_can(ani)
        again = A.parse_ani(C.update_ani_counts(ani_data, ani), "ps3", can=new_can, strict=True)
        self.assertEqual(again.permutations[1].keyframes, 303)
        blob = again.permutations[1].translation_blobs[0]
        self.assertEqual(struct.unpack_from(">H", blob, 0)[0], 3)  # count in 2 bytes
        self.assertEqual(A.decode_permutation(again.permutations[1]).translation[0].frames, [0, 2, 302])

    def test_counts_disagree(self):
        ani = A.parse_ani(B.build_ani("ps3", perms()), "ps3", can=B.build_can(perms()))
        ani.permutations[0].nonlinear_offsets.append((0.0, 0.0, 0.0))
        self.assertTrue(C.can_count_problems(ani))


class NavWriter(unittest.TestCase):
    def test_w3_4_round_trip(self):
        for build in ("ps3", "x360"):
            scn, nav_data = B.build_scene(build)
            nav = N.parse_scene(scn, build, nav=nav_data)
            self.assertEqual(C.write_nav(nav), nav_data)

    def test_total_zero(self):
        empty = [b""] * 6
        flat = {0x1BC: 0, 0x1C0: 0}  # no coarse grid
        scn, _ = B.build_scene("ps3", width=3, height=2, top=1, bottom=1, left=1, right=1,
                               sections=empty, body=flat)
        self.assertIsNone(C.write_nav(N.parse_scene(scn, "ps3")))  # ps3: no child
        scn, _ = B.build_scene("x360", width=3, height=2, top=1, bottom=1, left=1, right=1,
                               sections=empty, body=flat)
        self.assertEqual(C.write_nav(N.parse_scene(scn, "x360", nav=b"")), b"")  # x360: 0 bytes

    def test_w3_5_one_more_cell(self):
        for build in ("ps3", "x360"):
            scn, nav_data = B.build_scene(build)
            nav = N.parse_scene(scn, build, nav=nav_data)
            first = nav.cells[0]
            nav.neighbours.append(N.Neighbour(first.flags & 0x0FFF, 0))
            nav.cells.append(N.NavCell((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), 0x0004, 1, len(nav.neighbours) - 1))
            new_scn = C.update_scene_counts(scn, nav)
            new_nav = C.write_nav(nav)
            again = N.parse_scene(new_scn, build, nav=new_nav, strict=True)
            self.assertEqual((len(again.cells), len(again.neighbours)), (len(nav.cells), len(nav.neighbours)))
            self.assertEqual(again.header.cell_count, 3)
            self.assertEqual(L.is_canonical(new_scn, "scn", build), L.is_canonical(scn, "scn", build))
            changed = {i for i in range(len(scn)) if scn[i] != new_scn[i]}
            self.assertTrue(changed <= set(range(0x10 + 0x180, 0x10 + 0x184)) | set(range(0x10 + 0x198, 0x10 + 0x19C)))
            self.assertEqual(C.write_nav(again), new_nav)

    def test_counts_disagree(self):
        scn, nav_data = B.build_scene("ps3")
        nav = N.parse_scene(scn, "ps3", nav=nav_data)
        nav.lookups.pop()
        self.assertTrue(C.nav_count_problems(nav))
        with self.assertRaises(FormatError):
            C.write_nav(nav)


# --- W3-6 to W3-8: ps3 globals --------------------------------------------------------------------

def ps3_globals():
    """A ps3 .glo laid out by hand from W3-6 to W3-8: one server-data record with two actor groups
    and one script variable, two start-location names, and the twelve banner arrays (the +0xC8
    pair sharing the +0xA8 block); every array 8-aligned from the body start."""
    top = bytearray(568)
    struct.pack_into(">I", top, 0x10, 0x1234)
    server = bytearray(424)
    struct.pack_into(">4I", server, 0x20, 1, 2, 3, 4)
    h = HandLayout(header(">", 7), top, ">")
    pairs = []
    sd = h.place(bytes(server), 8)
    pairs.append((BODY + 0x18, sd, 424))
    groups = h.place(b"G" * 68 * 2, 8)
    variables = h.place(b"V" * 40, 8)
    pairs += [(BODY + sd + 0x08, groups, 136), (BODY + sd + 0x18, variables, 40)]
    names = h.place(struct.pack(">I", 1) + b"start".ljust(64, b"\0") + struct.pack(">I", 2) + b"b".ljust(64, b"\0"), 8)
    pairs.append((BODY + 0x30, names, 136))
    banner = 0x88
    sizes = {0x08: 16, 0x20: 32, 0x38: 16, 0x48: 16, 0x60: 16, 0x78: 84, 0x88: 68, 0x98: 4, 0xA8: 8,
             0xB8: 4, 0xD8: 148}
    placed = {}
    for at in (0x08, 0x20, 0x38, 0x48, 0x60, 0x78, 0x88, 0x98, 0xA8, 0xB8, 0xC8, 0xD8):
        if at == 0xC8:
            pairs.append((BODY + banner + at, placed[0xA8], 8))  # W3-7: shares the +0xA8 block
            continue
        placed[at] = h.place(bytes([at]) * sizes[at], 8)
        pairs.append((BODY + banner + at, placed[at], sizes[at]))
    for at, offset, size in pairs:
        h.pair(at, offset, size)
    return bytes(h.out), placed


class Ps3Globals(unittest.TestCase):
    def test_rows_merged(self):
        self.assertIn("glo", L.covered_types("ps3"))
        self.assertEqual(L.top_record("ps3", "glo"), "GlobalDefinition")
        self.assertEqual(L.record_layout("ps3", "glo", "GlobalDefinition").size, 568)
        offsets = [s.offset for s in L.pair_slots("ps3", "glo", "GlobalDefinition")]
        banner = [0x88 + at for at in (0x08, 0x20, 0x38, 0x48, 0x60, 0x78, 0x88, 0x98, 0xA8, 0xB8, 0xC8, 0xD8)]
        self.assertEqual(offsets, [0x18, 0x30] + banner)  # W3-6, W3-8: no pair after 0x170
        self.assertEqual(L.record_layout("ps3", "glo", "GlobalServerData").size, 424)
        self.assertEqual(L.RESIDUE, frozenset())

    def test_w3_6_to_w3_8_rebuilt(self):
        data, placed = ps3_globals()
        tree = L.parse_tree(data, "glo", "ps3")
        self.assertEqual(tree.notes, [])
        self.assertEqual(L.write_tree(tree), data)
        shared = L.shared_pairs(tree)
        self.assertEqual(shared, [("tBannerParams.arActorRibbons", placed[0xA8])])

    def test_w3_8_alignment_and_order(self):
        data, placed = ps3_globals()
        tree = L.parse_tree(data, "glo", "ps3")
        names = L.find_pair(tree, "arStartLocationNames")[2]
        L.set_elements(tree, "arStartLocationNames", names.block.elements[:1])  # 68 bytes
        out = L.write_tree(tree)
        again = L.parse_tree(out, "glo", "ps3")
        first_banner = L.find_pair(again, "tBannerParams.arTexBackgrounds")[2].stored[0]
        self.assertEqual(first_banner % 8, 0)
        self.assertEqual(first_banner, placed[0x08] - 64)  # 136 -> 68 bytes, then a 4-byte gap
        self.assertEqual(L.write_tree(again), out)

    def test_w3_7_added_records(self):
        data, _ = ps3_globals()
        tree = L.parse_tree(data, "glo", "ps3")
        groups_path = "ptServerData[0].arActorGroups"
        groups = L.find_pair(tree, groups_path)[2].block.elements
        extra = L.new_record("ps3", "glo", "ActorGroup")
        extra.raw[:] = b"H" * 68
        L.set_elements(tree, groups_path, groups + [extra])
        out = L.write_tree(tree)
        again = L.parse_tree(out, "glo", "ps3")
        self.assertEqual(len(L.find_pair(again, groups_path)[2].block.elements), 3)
        self.assertEqual(L.write_tree(again), out)


if __name__ == "__main__":
    unittest.main()
