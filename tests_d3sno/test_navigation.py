import struct
import unittest

import builders as B
from d3sno import FormatError
from d3sno import navigation as N

BUILDS = ("ps3", "switch")


def read(build, **changes):
    scn, nav = B.build_scene(build, **changes)
    return N.parse_scene(scn, build, nav)


class Header(unittest.TestCase):
    def test_fields(self):
        for build in BUILDS:
            with self.subTest(build):
                nav = read(build)
                h = nav.header
                self.assertEqual((h.width, h.height, h.type, h.square_count), (6, 5, 1, 30))
                self.assertEqual((h.edge_top, h.edge_bottom), (1, 1))
                self.assertEqual((h.edge_left, h.edge_right), (2, 1))
                self.assertEqual((h.height_min, h.height_max), (10.0, 20.0))
                self.assertEqual((h.edge_height, h.edge_flags), (15.5, 0xBEEF))
                self.assertEqual((h.inner_width, h.inner_height), (3, 3))
                self.assertEqual((h.coarse_feet, h.coarse_squares), (10.0, 4))
                self.assertEqual((h.coarse_width, h.coarse_height), (2, 1))

    def test_build_from_version(self):
        scn, nav = B.build_scene("ps3")
        self.assertEqual(N.parse_scene(scn, nav=nav).build, "ps3")
        scn, _ = B.build_scene("switch")
        self.assertEqual(N.parse_scene(scn).build, "switch")

    def test_version_strict(self):
        scn, _ = B.build_scene("switch", version=1)
        N.parse_scene(scn, "switch")
        with self.assertRaises(FormatError):
            N.parse_scene(scn, "switch", strict=True)

    def test_square_count_warns(self):
        nav = read("switch", body={0x4C: 29})
        self.assertTrue(nav.warnings)

    def test_edges_larger_than_the_mesh(self):
        with self.assertRaises(FormatError):
            read("switch", body={0x58: 9})

    def test_too_short(self):
        with self.assertRaises(FormatError):
            N.parse_scene(B.sno_head(179, "<") + bytes(100), "switch")


class Sections(unittest.TestCase):
    def test_both_builds_agree(self):
        ps3, switch = read("ps3"), read("switch")
        for name in ("grid", "cells", "neighbours", "coarse", "lookups", "borders"):
            self.assertEqual(getattr(ps3, name), getattr(switch, name), name)
        self.assertEqual(ps3.warnings, [])
        self.assertEqual(switch.warnings, [])

    def test_cell(self):
        cell = read("ps3").cells[0]
        self.assertEqual(cell, N.NavCell((0, 1, 2), (3, 4, 5), 0x1001, 2, 0))

    def test_ps3_size_is_exact(self):
        scn, nav = B.build_scene("ps3")
        with self.assertRaises(FormatError):
            N.parse_scene(scn, "ps3", nav + b"\0")
        with self.assertRaises(FormatError):
            N.parse_scene(scn, "ps3", nav[:-1])

    def test_ps3_scene_of_size_zero_has_no_nav(self):
        scn, _ = B.build_scene(
            "ps3", width=2, height=2, top=1, bottom=1, left=1, right=1,
            sections=[b""] * 6, special=b"", body={0x1BC: 0, 0x1C0: 0},
        )
        nav = N.parse_scene(scn, "ps3")
        self.assertTrue(nav.has_sections)
        self.assertEqual(nav.cells, [])

    def test_ps3_scene_with_size_but_no_nav(self):
        scn, _ = B.build_scene("ps3")
        nav = N.parse_scene(scn, "ps3")
        self.assertFalse(nav.has_sections)
        self.assertEqual(nav.header.cell_count, 2)

    def test_switch_pair_sizes(self):
        for at in (0x80, 0x190, 0x1A8, 0x1D0, 0x1E8, 0x200):
            scn = bytearray(B.build_scene("switch")[0])
            size = struct.unpack_from("<I", scn, 0x10 + at + 4)[0]
            struct.pack_into("<I", scn, 0x10 + at + 4, size + 1)
            with self.assertRaises(FormatError):
                N.parse_scene(bytes(scn), "switch")

    def test_switch_pair_out_of_bounds(self):
        scn = bytearray(B.build_scene("switch")[0])
        struct.pack_into("<I", scn, 0x10 + 0x190, 0xFFFFF0)
        with self.assertRaises(FormatError):
            N.parse_scene(bytes(scn), "switch")

    def test_coarse_size_is_not_the_scene_size(self):
        self.assertEqual(len(read("switch").coarse), 2)


class Grid(unittest.TestCase):
    def test_inner_square(self):
        for build in BUILDS:
            nav = read(build)
            square = nav.square(2 + 1, 1 + 2)
            index = 1 + 3 * 2
            self.assertFalse(square.is_edge)
            self.assertEqual(square.flags, 0x100 + index)
            self.assertAlmostEqual(square.height, 10.0 + 10.0 * (index * 6553) / 65535)

    def test_height_range(self):
        nav = read("switch")
        nav.grid = [(65535, 0)] * 9
        self.assertEqual(nav.square(2, 1).height, 20.0)
        nav.grid = [(0, 0)] * 9
        self.assertEqual(nav.square(2, 1).height, 10.0)

    def test_edge_squares(self):
        nav = read("switch")
        for x, y in ((0, 0), (1, 2), (5, 2), (3, 0), (3, 4)):
            square = nav.square(x, y)
            self.assertTrue(square.is_edge, (x, y))
            self.assertEqual((square.height, square.flags), (15.5, 0xBEEF))

    def test_outside_the_mesh(self):
        nav = read("switch")
        for x, y in ((6, 0), (0, 5), (-1, 0)):
            with self.assertRaises(FormatError):
                nav.square(x, y)

    def test_no_grid_reads_as_the_edge_square(self):
        nav = N.parse_scene(B.build_scene("ps3")[0], "ps3")
        square = nav.square(3, 2)
        self.assertTrue(square.is_edge)
        self.assertEqual((square.height, square.flags), (15.5, 0xBEEF))

    def test_height_is_single_precision(self):
        nav = read("switch")
        nav.grid = [(1, 0)] * 9
        height = nav.square(2, 1).height
        self.assertEqual(height, struct.unpack("<f", struct.pack("<f", height))[0])


class Layers(unittest.TestCase):
    def test_special_nibble_becomes_flags(self):
        self.assertEqual(N.special_to_flags(0b0001), 0b00001)
        self.assertEqual(N.special_to_flags(0b0010), 0b01000)
        self.assertEqual(N.special_to_flags(0b0100), 0b10000)
        self.assertEqual(N.special_to_flags(0b1000), 0)
        self.assertEqual(N.special_to_flags(0b0111), 0b11001)

    def test_layer_1_reads_special_flags_with_no_height(self):
        for build in BUILDS:
            nav = read(build)
            square = nav.square(2 + 1, 1, 1)
            self.assertEqual(nav.special_flag(1), 2)
            self.assertEqual((square.flags, square.height), (0b01000, None))

    def test_edge_squares_give_edge_flags_in_both_layers(self):
        nav = read("switch")
        self.assertEqual(nav.square(0, 0, 0).flags, 0xBEEF)
        self.assertEqual(nav.square(0, 0, 1).flags, 0xBEEF)
        self.assertIsNone(nav.square(0, 0, 1).height)


class SubScenes(unittest.TestCase):
    def test_sub_scene_square_is_used_first(self):
        scene, sub = read("switch"), read("switch")
        sub.grid = [(0, 0x77)] * 9
        square = N.square_in_world(
            scene, 7, 6, (0.0, 0.0, 10.0), [(sub, (12.5, 12.5, 15.0))],
        )
        # The offset is 5 squares, so (7, 6) is (2, 1) in the sub-scene, its inner corner.
        self.assertEqual(square.flags, 0x77)
        self.assertAlmostEqual(square.height, 10.0 + 5.0, places=4)

    def test_outside_the_sub_scene_uses_the_scene(self):
        scene, sub = read("switch"), read("switch")
        square = N.square_in_world(scene, 1, 1, (0.0, 0.0, 0.0), [(sub, (12.5, 12.5, 0.0))])
        self.assertEqual(square, scene.square(1, 1))

    def test_layer_1_has_no_height_to_lift(self):
        scene, sub = read("switch"), read("switch")
        square = N.square_in_world(scene, 7, 6, (0.0, 0.0, 0.0), [(sub, (12.5, 12.5, 5.0))], 1)
        self.assertIsNone(square.height)


class SpecialFlags(unittest.TestCase):
    def test_nibbles(self):
        for build in BUILDS:
            nav = read(build)
            self.assertEqual([nav.special_flag(j) for j in range(5)], [1, 2, 3, 4, 5])
            self.assertEqual(nav.special_flag(8), 9)

    def test_ps3_size_rule(self):
        scn, nav = B.build_scene("ps3", special=b"\x21\x43")
        with self.assertRaises(FormatError):
            N.parse_scene(scn, "ps3", nav)

    def test_switch_size_mismatch_warns(self):
        scn, _ = B.build_scene("switch", special=b"\x21\x43")
        self.assertTrue(N.parse_scene(scn, "switch").warnings)

    def test_past_the_table(self):
        with self.assertRaises(FormatError):
            read("switch").special_flag(10)


class Links(unittest.TestCase):
    def sections(self, order="<"):
        return B.nav_sections(order, 9)

    def with_part(self, index, raw):
        parts = self.sections()
        parts[index] = raw
        return read("switch", sections=parts)

    def test_neighbours(self):
        nav = read("ps3")
        cell, border = nav.neighbours
        self.assertIsNone(cell.border_index)
        self.assertEqual(cell.cell, 1)
        self.assertEqual(border.border_index, 0)

    def test_neighbour_flags_mask(self):
        scn_flags = read("switch").cells[1].flags
        self.assertEqual(read("switch").neighbours[0].flags, scn_flags & 0x0FFF)

    def test_neighbour_flags_mismatch(self):
        raw = struct.pack("<HH", 0x777, 1) + struct.pack("<HH", 0xFFFF, 0x8000)
        self.assertTrue(self.with_part(2, raw).warnings)

    def test_neighbour_cell_missing(self):
        raw = struct.pack("<HH", 2, 9) + struct.pack("<HH", 0xFFFF, 0x8000)
        self.assertTrue(self.with_part(2, raw).warnings)

    def test_neighbour_border_missing(self):
        raw = struct.pack("<HH", 2, 1) + struct.pack("<HH", 0xFFFF, 0x8005)
        self.assertTrue(self.with_part(2, raw).warnings)

    def test_cell_neighbours_past_table(self):
        raw = struct.pack("<HH", 2, 1)
        self.assertTrue(self.with_part(2, raw).warnings)

    def test_empty_coarse_square(self):
        coarse = read("switch").coarse
        self.assertFalse(coarse[0].is_empty)
        self.assertTrue(coarse[1].is_empty)

    def test_coarse_lookups_past_table(self):
        raw = struct.pack("<HHH", 1, 5, 0) + struct.pack("<HHH", 0xFFFF, 0, 0)
        self.assertTrue(self.with_part(3, raw).warnings)

    def test_lookup_cell_missing(self):
        raw = struct.pack("<HH", 1, 0) + struct.pack("<HH", 2, 7)
        self.assertTrue(self.with_part(4, raw).warnings)

    def test_border_cell_missing(self):
        self.assertTrue(self.with_part(5, struct.pack("<HH", 7, 1)).warnings)

    def test_strict_raises(self):
        scn, _ = B.build_scene("switch", body={0x4C: 29})
        with self.assertRaises(FormatError):
            N.parse_scene(scn, "switch", strict=True)

    def test_border_sides_keep_extra_bits(self):
        border = read("switch").borders[0]
        self.assertEqual(border.sides, 1 | 8 | 0x40)
        self.assertTrue(border.sides & N.SIDE_X_ZERO)
        self.assertTrue(border.sides & N.SIDE_Y_MAX)
        self.assertFalse(border.sides & N.SIDE_X_MAX)
        self.assertFalse(border.sides & N.SIDE_Y_ZERO)


class Flags(unittest.TestCase):
    def test_names(self):
        self.assertEqual(N.flag_names(0b11), ["walk", "flier"])
        self.assertEqual(N.flag_names(1 << 10), ["projectile"])
        self.assertEqual(N.flag_names(1 << 9), [])
        self.assertEqual(N.flag_names(0x8000), ["rounded_corner_4"])
