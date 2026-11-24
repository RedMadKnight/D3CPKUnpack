"""Stage 14: ``28_derived_values.md`` W14-16 (scene navigation derived sections) and the safe
replacement of W14-17 (cells and heights from a changed grid). Synthetic scenes only."""

import _libpath  # noqa: F401  (python -I: the library folder on sys.path)
import struct
import unittest

from d3sno import children as C
from d3sno import navigation as N
from d3sno import navigation_derive as D
from d3sno.binary import FormatError, to_f32
from builders import build_scene, order_of


def header(width, height, top=0, bottom=0, left=0, right=0, hmin=0.0, hmax=1.0, edge_h=0.0, edge_flags=0):
    return N.SceneHeader(width, height, 1, width * height, top, bottom, left, right, hmin, hmax, edge_h,
                         edge_flags, 0, 0, 0.0, 0.0, 0, 0, 0, 0, 0)


def cell(x0, y0, x1, y1, flags=1, z=(0.0, 0.0)):
    """A cell from square coordinates."""
    return N.NavCell((x0 * 2.5, y0 * 2.5, z[0]), (x1 * 2.5, y1 * 2.5, z[1]), flags, 0, 0)


# 8 x 8 scene, four cells, 2 x 2 coarse squares of 4 squares
TILING = [cell(0, 0, 8, 2, 0x1001), cell(0, 2, 3, 8, 0x0003), cell(3, 2, 8, 5, 0x0007), cell(3, 5, 8, 8, 0x0011)]

# 6 x 6 scene for neighbour order: T1 T2 / L1 M R / L2 (M R) / B
NEIGHBOURS = [
    cell(0, 0, 3, 2, 0xF001), cell(3, 0, 6, 2, 0x0002), cell(0, 2, 2, 3, 0x0004), cell(2, 2, 4, 4, 0x0008),
    cell(4, 2, 6, 4, 0x2010), cell(0, 3, 2, 4, 0x0020), cell(0, 4, 6, 6, 0x0040),
]


def section_bytes(order, nav_rows):
    grid, cells, neighbours, coarse, lookups, borders = nav_rows
    return [
        b"".join(struct.pack(order + "HH", *row) for row in grid),
        b"".join(struct.pack(order + "6fHHI", *c.min, *c.max, c.flags, c.neighbour_count, c.first_neighbour)
                 for c in cells),
        b"".join(struct.pack(order + "HH", n.flags, n.cell) for n in neighbours),
        b"".join(struct.pack(order + "HHH", s.flags, s.count, s.first_lookup) for s in coarse),
        b"".join(struct.pack(order + "HH", x.flags, x.cell) for x in lookups),
        b"".join(struct.pack(order + "HH", b.cell, b.sides) for b in borders),
    ]


def f32_bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


class CoarseSize(unittest.TestCase):
    def test_at_most_three_cells_uses_the_scene_width(self):
        d = D.derive_navigation(header(8, 4), [cell(0, 0, 4, 4), cell(4, 0, 6, 4), cell(6, 0, 8, 4)])
        self.assertEqual((d.coarse_feet, d.coarse_squares, d.coarse_width, d.coarse_height), (20.0, 8, 1, 1))
        self.assertEqual(d.coarse_feet_inverse, to_f32(1 / 20))

    def test_more_than_three_cells_uses_ten_feet(self):
        cells = [cell(0, 0, 4, 4), cell(4, 0, 6, 4), cell(6, 0, 8, 4), cell(0, 4, 8, 5)]
        d = D.derive_navigation(header(9, 5), cells)
        self.assertEqual((d.coarse_feet, d.coarse_squares, d.coarse_width, d.coarse_height), (10.0, 4, 3, 2))
        self.assertEqual(d.coarse_feet_inverse, to_f32(0.1))
        self.assertEqual(len(d.coarse), 6)

    def test_small_scene_taller_than_wide(self):
        d = D.derive_navigation(header(4, 10), [cell(0, 0, 4, 10)])
        self.assertEqual((d.coarse_feet, d.coarse_width, d.coarse_height), (10.0, 1, 3))


class Lookups(unittest.TestCase):
    def test_lookups_running_sums_and_flags(self):
        d = D.derive_navigation(header(8, 8), TILING)
        self.assertEqual(d.lookups, [N.Lookup(f & 0xFFF, i) for f, i in (
            (0x1001, 0), (3, 1), (7, 2), (0x1001, 0), (7, 2), (3, 1), (7, 2), (0x11, 3), (7, 2), (0x11, 3))])
        self.assertEqual(d.coarse, [N.CoarseSquare(1, 3, 0), N.CoarseSquare(1, 2, 3), N.CoarseSquare(1, 3, 5),
                                    N.CoarseSquare(1, 2, 8)])

    def test_order_is_by_clipped_corner_not_cell_index(self):
        cells = [cell(0, 0, 6, 2, 1), cell(6, 0, 8, 8, 5), cell(0, 2, 6, 5, 9), cell(0, 5, 6, 8, 0x11)]
        d = D.derive_navigation(header(8, 8), cells)
        last = d.coarse[3]
        self.assertEqual([x.cell for x in d.lookups[last.first_lookup:last.first_lookup + last.count]], [2, 1, 3])
        self.assertEqual(last.flags, 9 & 0x11 & 5)

    def test_partial_cover_and_empty_square(self):
        cells = [cell(0, 0, 1, 1, 3), cell(1, 0, 2, 1, 3), cell(0, 1, 1, 2, 7), cell(8, 0, 12, 4, 0x105)]
        d = D.derive_navigation(header(12, 4), cells)
        self.assertEqual(d.coarse[0], N.CoarseSquare(3 | 0x20, 3, 0))
        self.assertEqual(d.coarse[1], N.CoarseSquare(0xFFFF, 0, 3))
        self.assertTrue(d.coarse[1].is_empty)
        self.assertEqual(d.coarse[2], N.CoarseSquare(0x105, 1, 3))

    def test_square_past_the_scene_edge_is_clipped(self):
        # 9 x 4: the last coarse column holds one scene column, fully covered: no 0x20
        cells = [cell(0, 0, 4, 4), cell(4, 0, 8, 4), cell(8, 0, 9, 2), cell(8, 2, 9, 4)]
        d = D.derive_navigation(header(9, 4), cells)
        self.assertEqual([s.flags for s in d.coarse], [1, 1, 1])


class Neighbours(unittest.TestCase):
    def setUp(self):
        self.d = D.derive_navigation(header(6, 6), NEIGHBOURS)

    def entries(self, index):
        c = self.d.cells[index]
        return self.d.neighbours[c.first_neighbour:c.first_neighbour + c.neighbour_count]

    def test_order_and_tie_rule(self):
        # M: min-y side T1, T2; then L1 (min-x) and R (max-x) both start at y 2, L1 first; L2; then B
        self.assertEqual([n.cell for n in self.entries(3)], [0, 1, 2, 4, 5, 6])

    def test_flags_are_the_target_low_bits(self):
        self.assertEqual([n.flags for n in self.entries(3)], [0x001, 0x002, 0x004, 0x010, 0x020, 0x040])

    def test_border_entry_comes_last(self):
        entries = self.entries(2)  # L1: T1 above, M right, L2 below, then its border
        self.assertEqual([n.cell for n in entries[:-1]], [0, 3, 5])
        self.assertEqual((entries[-1].flags, entries[-1].border_index), (0xFFFF, 2))
        self.assertTrue(all(n.border_index is None for n in self.entries(3)))

    def test_running_sums(self):
        first = 0
        for c in self.d.cells:
            self.assertEqual(c.first_neighbour, first)
            first += c.neighbour_count
        self.assertEqual(first, len(self.d.neighbours))

    def test_corner_contact_is_no_neighbour(self):
        d = D.derive_navigation(header(2, 2), [cell(0, 0, 1, 1), cell(1, 0, 2, 1), cell(0, 1, 1, 2), cell(1, 1, 2, 2)])
        targets = [n.cell for n in d.neighbours[:d.cells[0].neighbour_count] if n.border_index is None]
        self.assertEqual(targets, [1, 2])


class Borders(unittest.TestCase):
    def test_sides_in_cell_order(self):
        d = D.derive_navigation(header(6, 6), NEIGHBOURS)
        self.assertEqual(d.borders, [N.Border(0, 1 | 4), N.Border(1, 2 | 4), N.Border(2, 1), N.Border(4, 2),
                                     N.Border(5, 1), N.Border(6, 1 | 2 | 8)])

    def test_single_cell_has_all_sides(self):
        d = D.derive_navigation(header(3, 2), [cell(0, 0, 3, 2)])
        self.assertEqual(d.borders, [N.Border(0, 15)])
        self.assertEqual(d.neighbours, [N.Neighbour(0xFFFF, 0x8000)])


class Counts(unittest.TestCase):
    def test_counts(self):
        d = D.derive_navigation(header(6, 6), NEIGHBOURS)
        self.assertEqual(d.counts, {"square_count": 36, "cell_count": 7, "neighbour_count": len(d.neighbours),
                                    "lookup_count": len(d.lookups), "border_count": 6})
        self.assertEqual(len(d.neighbours), sum(c.neighbour_count for c in d.cells))
        self.assertEqual(len(d.lookups), sum(s.count for s in d.coarse))

    def test_off_grid_cell_is_refused(self):
        bad = N.NavCell((0.0, 0.0, 0.0), (3.0, 2.5, 0.0), 1, 0, 0)
        with self.assertRaises(FormatError):
            D.derive_navigation(header(2, 2), [bad])


class ParsedScenes(unittest.TestCase):
    def scene(self, build, cells):
        h = header(8, 8)
        d = D.derive_navigation(h, cells)
        order = order_of(build)
        parts = section_bytes(order, ([(0, 0)] * 36, d.cells, d.neighbours, d.coarse, d.lookups, d.borders))
        body = {0x1B0: f32_bits(d.coarse_feet), 0x1B4: f32_bits(d.coarse_feet_inverse),
                0x1B8: d.coarse_squares, 0x1BC: d.coarse_width, 0x1C0: d.coarse_height}
        scn, nav = build_scene(build, 8, 8, 1, 1, 1, 1, sections=parts, body=body, special=bytes(18))
        return N.parse_scene(scn, build, nav, strict=True), d

    def test_parsed_scene_matches_derivation(self):
        for build in ("ps3", "switch", "x360"):
            nav, d = self.scene(build, TILING)
            self.assertEqual(D.derived_differences(nav), [], build)
            self.assertEqual((nav.cells, nav.lookups, nav.coarse), (d.cells, d.lookups, d.coarse))

    def test_differences_are_reported(self):
        scn, nav_bytes = build_scene("ps3")
        nav = N.parse_scene(scn, "ps3", nav_bytes)
        found = D.derived_differences(nav, D.derive_navigation(nav, [cell(0, 0, 6, 5)]))
        self.assertTrue(found)

    def test_apply_derived_keeps_the_input_and_feeds_the_nav_writer(self):
        nav, d = self.scene("ps3", TILING)
        before = C.write_nav(nav)
        changed = [cell(0, 0, 8, 4, 1), cell(0, 4, 4, 8, 2), cell(4, 4, 8, 6, 3), cell(4, 6, 8, 8, 4), ]
        out = D.apply_derived(nav, D.derive_navigation(nav, changed))
        self.assertEqual(C.write_nav(nav), before)
        self.assertEqual(C.nav_count_problems(out), [])
        self.assertEqual(D.derived_differences(out), [])
        self.assertNotEqual(C.write_nav(out), before)
        # unchanged cells: the derivation reproduces the stored file byte for byte
        self.assertEqual(C.write_nav(D.apply_derived(nav, D.derive_navigation(nav, nav.cells))), before)


class CellsFromGrid(unittest.TestCase):
    def grid_scene(self):
        h = header(6, 5, top=1, bottom=1, left=1, right=1, edge_h=4.0, edge_flags=0x0040)
        flags = [0x1001, 0x0001, 0x0003, 0x0003,
                 0x0001, 0x2001, 0x0003, 0x0003,
                 0x0005, 0x0005, 0x0005, 0x0003]
        heights = [float(i) for i in range(12)]
        return h, list(zip(heights, flags))

    def test_tiling_satisfies_w14_16(self):
        h, grid = self.grid_scene()
        r = D.cells_from_grid(h, grid)
        self.assertEqual(D.tiling_problems(h, r.cells), [])
        self.assertEqual([(c.min[1] / 2.5, c.min[0] / 2.5) for c in r.cells],
                         sorted((c.min[1] / 2.5, c.min[0] / 2.5) for c in r.cells))
        again = D.derive_navigation(h, r.cells)
        self.assertEqual((again.cells, again.neighbours, again.coarse, again.lookups, again.borders),
                         (r.derived.cells, r.derived.neighbours, r.derived.coarse, r.derived.lookups,
                          r.derived.borders))

    def test_merge_flags_and_z(self):
        h, grid = self.grid_scene()
        r = D.cells_from_grid(h, grid)
        cells = r.cells
        self.assertEqual((cells[0].min, cells[0].max, cells[0].flags), ((0.0, 0.0, 4.0), (15.0, 2.5, 4.0), 0x40))
        walk = cells[2]  # the 2 x 2 block of low flags 0x001 at inner (0, 0)
        self.assertEqual((walk.min, walk.max, walk.flags), ((2.5, 2.5, 0.0), (7.5, 7.5, 5.0), 1))
        self.assertTrue(all(c.flags <= 0x0FFF for c in cells))
        self.assertEqual(len(cells), 8)

    def test_height_range_and_requantised_grid(self):
        h, grid = self.grid_scene()
        r = D.cells_from_grid(h, grid)
        self.assertEqual((r.height_min, r.height_max), (0.0, 11.0))
        self.assertEqual(r.grid[0], (0, 0x1001))
        self.assertEqual(r.grid[11], (65535, 0x0003))
        self.assertEqual(r.grid[4][0], round(4 / 11 * 65535))
        # edge square outside the inner range widens it
        h2 = header(6, 5, 1, 1, 1, 1, edge_h=20.0)
        self.assertEqual(D.cells_from_grid(h2, grid).height_max, 20.0)

    def test_no_inner_grid(self):
        h = header(4, 3, top=2, bottom=1, edge_h=7.0, edge_flags=0x8003)
        r = D.cells_from_grid(h, None)
        self.assertEqual((r.height_min, r.height_max), (to_f32(3.4028235e38), to_f32(-3.4028235e38)))
        self.assertEqual(r.height_min, to_f32(3.4028235e38))
        self.assertEqual(r.grid, [])
        self.assertEqual(r.cells, [N.NavCell((0.0, 0.0, 7.0), (10.0, 7.5, 7.0), 3, 1, 0)])
        self.assertEqual(r.derived.borders, [N.Border(0, 15)])
        self.assertEqual(r.derived.coarse_feet, 10.0)

    def test_apply_rebuilt(self):
        h, grid = self.grid_scene()
        r = D.cells_from_grid(h, grid)
        parts = section_bytes(">", ([(0, 0)] * 12, [], [], [N.CoarseSquare(0xFFFF, 0, 0)] * 2, [], []))
        scn, nav_bytes = build_scene("ps3", 6, 5, 1, 1, 1, 1, sections=parts, special=bytes(6))
        nav = N.parse_scene(scn, "ps3", nav_bytes)
        out = D.apply_derived(nav, r.derived, r)
        self.assertEqual((out.header.height_min, out.header.height_max), (0.0, 11.0))
        self.assertEqual(out.grid, r.grid)
        self.assertEqual(C.nav_count_problems(out), [])
        self.assertEqual(D.derived_differences(out), [])

    def test_wrong_grid_size_is_refused(self):
        h, grid = self.grid_scene()
        with self.assertRaises(FormatError):
            D.cells_from_grid(h, grid[:-1])


if __name__ == "__main__":
    unittest.main()
