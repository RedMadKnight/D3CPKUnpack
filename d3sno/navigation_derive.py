"""Stage 14: derived scene navigation sections (W14-16) and the safe cell rebuild (W14-17).

``derive_navigation(scene, cells)`` computes the coarse grid, lookups, neighbours, borders and the
counts from the cells, as data objects of :mod:`d3sno.navigation`. ``cells_from_grid`` is the safe
replacement of W14-17 for a changed grid. ``apply_derived`` puts the result into a copy of a parsed
scene, ready for the ``.nav`` writer; it is never applied by default.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

from .binary import FormatError, to_f32
from .navigation import (
    BORDER_INDEX_BASE, EMPTY_COARSE, NEIGHBOUR_BORDER, SIDE_X_MAX, SIDE_X_ZERO, SIDE_Y_MAX,
    SIDE_Y_ZERO, Border, CoarseSquare, Lookup, NavCell, Navigation, Neighbour, SceneHeader,
)

SQUARE_FEET = 2.5
COARSE_FEET = 10.0
SMALL_SCENE_CELLS = 3
LOW_FLAGS = 0x0FFF
PARTIAL_COVER = 0x20
FLT_MAX = 3.4028234663852886e38  # +3.4028235e38 as stored
NO_GRID_RANGE = (FLT_MAX, -FLT_MAX)


@dataclass
class DerivedNavigation:
    """W14-16 sections and header values. ``cells`` carry the derived neighbour count and first."""
    cells: list[NavCell]
    neighbours: list[Neighbour]
    coarse: list[CoarseSquare]
    lookups: list[Lookup]
    borders: list[Border]
    square_count: int
    coarse_feet: float
    coarse_feet_inverse: float
    coarse_squares: int
    coarse_width: int
    coarse_height: int

    @property
    def counts(self) -> dict[str, int]:
        return {
            "square_count": self.square_count, "cell_count": len(self.cells),
            "neighbour_count": len(self.neighbours), "lookup_count": len(self.lookups),
            "border_count": len(self.borders),
        }

    def header_values(self) -> dict[str, int | float]:
        """The SceneHeader fields W14-16 sets."""
        values: dict[str, int | float] = dict(self.counts)
        values.update(
            coarse_feet=self.coarse_feet, coarse_feet_inverse=self.coarse_feet_inverse,
            coarse_squares=self.coarse_squares, coarse_width=self.coarse_width,
            coarse_height=self.coarse_height,
        )
        return values


@dataclass
class RebuiltNavigation:
    """W14-17 safe replacement: new cells, height range, re-quantised grid and W14-16 sections."""
    height_min: float
    height_max: float
    grid: list[tuple[int, int]]  # (z, flags), inner grid, row-major
    derived: DerivedNavigation
    warnings: list[str] = field(default_factory=list)

    @property
    def cells(self) -> list[NavCell]:
        return self.derived.cells


# --- helpers -------------------------------------------------------------------------------------

def _header(scene) -> SceneHeader:
    if isinstance(scene, Navigation):
        return scene.header
    if isinstance(scene, SceneHeader):
        return scene
    raise TypeError("scene must be a Navigation or a SceneHeader")


def _units(value: float, what: str) -> int:
    """A coordinate in feet as a whole number of 2.5-ft squares."""
    units = round(value / SQUARE_FEET)
    if abs(units * SQUARE_FEET - value) > 1e-3:
        raise FormatError(f"{what} {value} is not on the 2.5-ft grid")
    return units


@dataclass(frozen=True)
class _Rect:
    x0: int
    y0: int
    x1: int
    y1: int


def _rects(cells: list[NavCell]) -> list[_Rect]:
    rects = []
    for index, cell in enumerate(cells):
        r = _Rect(_units(cell.min[0], f"cell {index} min x"), _units(cell.min[1], f"cell {index} min y"),
                  _units(cell.max[0], f"cell {index} max x"), _units(cell.max[1], f"cell {index} max y"))
        if r.x1 <= r.x0 or r.y1 <= r.y0:
            raise FormatError(f"cell {index} has no area")
        rects.append(r)
    return rects


def _overlap(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0))


# --- W14-16 --------------------------------------------------------------------------------------

def coarse_layout(width: int, height: int, cell_count: int) -> tuple[float, float, int, int, int]:
    """(size in feet, f32 inverse, squares per coarse square, coarse width, coarse height)."""
    size = to_f32(width * SQUARE_FEET) if cell_count <= SMALL_SCENE_CELLS else COARSE_FEET
    per = round(size / SQUARE_FEET)
    if per <= 0:
        return size, 0.0, 0, 0, 0
    return size, to_f32(1.0 / size), per, -(-width // per), -(-height // per)


def _coarse_and_lookups(rects, flags, width, height, per, cw, ch):
    coarse, lookups = [], []
    for cy in range(ch):
        y0, y1 = cy * per, min((cy + 1) * per, height)
        for cx in range(cw):
            x0, x1 = cx * per, min((cx + 1) * per, width)
            hits, covered = [], 0
            for index, r in enumerate(rects):
                area = _overlap(r.x0, r.x1, x0, x1) * _overlap(r.y0, r.y1, y0, y1)
                if area > 0:
                    hits.append(((max(r.y0, y0), max(r.x0, x0), index), index))
                    covered += area
            first = len(lookups)
            if not hits:
                coarse.append(CoarseSquare(EMPTY_COARSE, 0, first))
                continue
            hits.sort()
            square_flags = LOW_FLAGS
            for _, index in hits:
                low = flags[index] & LOW_FLAGS
                lookups.append(Lookup(low, index))
                square_flags &= low
            if covered < (x1 - x0) * (y1 - y0):
                square_flags |= PARTIAL_COVER
            coarse.append(CoarseSquare(square_flags, len(hits), first))
    return coarse, lookups


def _sides(r: _Rect, width: int, height: int) -> int:
    return ((SIDE_X_ZERO if r.x0 == 0 else 0) | (SIDE_X_MAX if r.x1 == width else 0)
            | (SIDE_Y_ZERO if r.y0 == 0 else 0) | (SIDE_Y_MAX if r.y1 == height else 0))


def _neighbour_order(rects: list[_Rect], index: int) -> list[int]:
    a = rects[index]
    low_y, sides_x, high_y = [], [], []
    for other, b in enumerate(rects):
        if other == index:
            continue
        if _overlap(a.x0, a.x1, b.x0, b.x1) > 0:
            start_x = max(a.x0, b.x0)
            if b.y1 == a.y0:
                low_y.append((start_x, other))
            elif b.y0 == a.y1:
                high_y.append((start_x, other))
        if _overlap(a.y0, a.y1, b.y0, b.y1) > 0:
            start_y = max(a.y0, b.y0)
            if b.x1 == a.x0:
                sides_x.append((start_y, 0, other))
            elif b.x0 == a.x1:
                sides_x.append((start_y, 1, other))
    return ([o for _, o in sorted(low_y)] + [o for *_, o in sorted(sides_x)]
            + [o for _, o in sorted(high_y)])


def derive_navigation(scene, cells: list[NavCell]) -> DerivedNavigation:
    """W14-16 from ``cells`` (rectangles on the 2.5-ft grid, kept in the given order, which should
    be min y then min x). ``scene`` is a Navigation or SceneHeader; only width and height are used."""
    h = _header(scene)
    width, height = h.width, h.height
    rects = _rects(cells)
    flags = [cell.flags for cell in cells]
    size, inverse, per, cw, ch = coarse_layout(width, height, len(cells))
    coarse, lookups = _coarse_and_lookups(rects, flags, width, height, per, cw, ch)
    if lookups and len(lookups) - coarse[-1].count > 0xFFFF:
        raise FormatError(f"{len(lookups)} lookups: a first lookup does not fit 16 bits")

    borders, border_of = [], {}
    for index, r in enumerate(rects):
        sides = _sides(r, width, height)
        if sides:
            border_of[index] = len(borders)
            borders.append(Border(index, sides))
    if len(borders) > 0xFFFF - BORDER_INDEX_BASE:
        raise FormatError(f"{len(borders)} borders do not fit a neighbour entry")

    neighbours, out_cells = [], []
    for index, cell in enumerate(cells):
        first = len(neighbours)
        for other in _neighbour_order(rects, index):
            neighbours.append(Neighbour(flags[other] & LOW_FLAGS, other))
        if index in border_of:
            neighbours.append(Neighbour(NEIGHBOUR_BORDER, BORDER_INDEX_BASE + border_of[index]))
        count = len(neighbours) - first
        if count > 0xFFFF:
            raise FormatError(f"cell {index}: {count} neighbours")
        out_cells.append(replace(cell, neighbour_count=count, first_neighbour=first))
    return DerivedNavigation(out_cells, neighbours, coarse, lookups, borders, width * height,
                             size, inverse, per, cw, ch)


def cell_entries(cells: list[NavCell], neighbours: list[Neighbour]) -> list[list[Neighbour]]:
    """V16-4: per cell, its cell-to-cell neighbour entries (the PROVEN part of its list, border
    entries left out)."""
    return [[n for n in neighbours[c.first_neighbour:c.first_neighbour + c.neighbour_count]
             if n.flags != NEIGHBOUR_BORDER] for c in cells]


def _border_entries(cells: list[NavCell], neighbours: list[Neighbour]) -> list[list[Neighbour]]:
    return [[n for n in neighbours[c.first_neighbour:c.first_neighbour + c.neighbour_count]
             if n.flags == NEIGHBOUR_BORDER] for c in cells]


def _cell_fields(cell: NavCell) -> NavCell:
    return replace(cell, neighbour_count=0, first_neighbour=0)


PROVEN_VALUES = ("lookup_count", "coarse_feet_inverse", "coarse_squares", "coarse_width", "coarse_height",
                 "square_count", "cell_count")


def derived_differences(nav: Navigation, derived: DerivedNavigation | None = None,
                        supported: bool = True) -> list[str]:
    """Where a parsed scene's sections or counts differ from W14-16 derived from its own cells.
    PROVEN (V16-4): coarse squares, lookups and their count, and the cell-to-cell neighbour entries
    with their order and flags. With ``supported`` (the default) also the SUPPORTED values, marked
    "(SUPPORTED)": the border entries, borders and border count, the neighbour counts, first sums
    and total, and the coarse size."""
    if derived is None:
        derived = derive_navigation(nav, nav.cells)
    found = []
    for name in ("coarse", "lookups"):
        if getattr(nav, name) != getattr(derived, name):
            found.append(f"{name} differ from W14-16")
    if [_cell_fields(c) for c in nav.cells] != [_cell_fields(c) for c in derived.cells]:
        found.append("cells differ from W14-16")
    elif cell_entries(nav.cells, nav.neighbours) != cell_entries(derived.cells, derived.neighbours):
        found.append("cell-to-cell neighbour entries differ from W14-16")
    h = nav.header
    values = derived.header_values()
    for name in PROVEN_VALUES:
        if name in values and getattr(h, name) != values[name]:
            found.append(f"{name}: stored {getattr(h, name)}, derived {values[name]}")
    if supported:
        if nav.borders != derived.borders:
            found.append("borders differ from W14-16 (SUPPORTED)")
        if _border_entries(nav.cells, nav.neighbours) != _border_entries(derived.cells, derived.neighbours):
            found.append("border entries differ from W14-16 (SUPPORTED)")
        if [(c.neighbour_count, c.first_neighbour) for c in nav.cells] != \
                [(c.neighbour_count, c.first_neighbour) for c in derived.cells]:
            found.append("neighbour counts or first neighbours differ from W14-16 (SUPPORTED)")
        for name, value in values.items():
            if name not in PROVEN_VALUES and getattr(h, name) != value:
                found.append(f"{name}: stored {getattr(h, name)}, derived {value} (SUPPORTED)")
    return found


def apply_derived(nav: Navigation, derived: DerivedNavigation,
                  rebuilt: RebuiltNavigation | None = None, supported: bool = False) -> Navigation:
    """A copy of ``nav`` with the derived sections and header values (and, with ``rebuilt``, the new
    grid and height range). The input is not changed; nothing calls this by default.

    V16-4: without ``supported`` (the default) the SUPPORTED border rule is not applied: each cell
    keeps its stored border entries after its derived cell-to-cell entries, the stored borders and
    border count are kept, and the coarse size of the stored header is kept; the neighbour counts,
    first sums and total follow the resulting lists. This needs the same number of cells as stored;
    otherwise, and always with ``rebuilt`` (new cells), the derived borders are used."""
    values = derived.header_values()
    cells, neighbours, borders = list(derived.cells), list(derived.neighbours), list(derived.borders)
    if not supported and rebuilt is None and len(nav.cells) == len(derived.cells):
        stored_borders = _border_entries(nav.cells, nav.neighbours)
        own = cell_entries(derived.cells, derived.neighbours)
        cells, neighbours = [], []
        for cell, entries, border in zip(derived.cells, own, stored_borders):
            first = len(neighbours)
            neighbours += entries + border
            cells.append(replace(cell, neighbour_count=len(neighbours) - first, first_neighbour=first))
        borders = list(nav.borders)
        values.update(neighbour_count=len(neighbours), border_count=len(borders))
        for name in ("coarse_feet",):
            values[name] = getattr(nav.header, name)
    if rebuilt is not None:
        values.update(height_min=rebuilt.height_min, height_max=rebuilt.height_max)
    out = replace(nav, header=replace(nav.header, **values), warnings=list(nav.warnings))
    out.cells, out.neighbours = cells, neighbours
    out.coarse, out.lookups, out.borders = list(derived.coarse), list(derived.lookups), borders
    out.grid = list(rebuilt.grid) if rebuilt is not None else list(nav.grid)
    return out


def tiling_problems(scene, cells: list[NavCell]) -> list[str]:
    """Squares of the scene covered by no cell or by several, and cells outside the scene."""
    h = _header(scene)
    cover = [0] * (h.width * h.height)
    found = []
    for index, r in enumerate(_rects(cells)):
        if r.x0 < 0 or r.y0 < 0 or r.x1 > h.width or r.y1 > h.height:
            found.append(f"cell {index} outside the scene")
            continue
        for y in range(r.y0, r.y1):
            for x in range(r.x0, r.x1):
                cover[x + h.width * y] += 1
    gaps = sum(1 for c in cover if c == 0)
    overlaps = sum(1 for c in cover if c > 1)
    if gaps:
        found.append(f"{gaps} squares in no cell")
    if overlaps:
        found.append(f"{overlaps} squares in several cells")
    return found


# --- W14-17 safe replacement ---------------------------------------------------------------------

def _greedy_merge(width: int, height: int, keys: list[int]) -> list[_Rect]:
    """Row-major greedy merge: grow right, then down while the whole row matches."""
    taken = [False] * (width * height)
    rects = []
    for y in range(height):
        for x in range(width):
            if taken[x + width * y]:
                continue
            key = keys[x + width * y]
            x1 = x + 1
            while x1 < width and not taken[x1 + width * y] and keys[x1 + width * y] == key:
                x1 += 1
            y1 = y + 1
            while y1 < height and all(
                not taken[i + width * y1] and keys[i + width * y1] == key for i in range(x, x1)
            ):
                y1 += 1
            for yy in range(y, y1):
                for xx in range(x, x1):
                    taken[xx + width * yy] = True
            rects.append(_Rect(x, y, x1, y1))
    return rects


def quantise(height: float, low: float, high: float) -> int:
    """A height as a grid z, inverse of ``low + (high - low) * z / 65535``; 0 on a flat range."""
    span = high - low
    if not span > 0:
        return 0
    return min(65535, max(0, round((height - low) / span * 65535)))


def cells_from_grid(scene, grid: list[tuple[float, int]] | None) -> RebuiltNavigation:
    """W14-17 safe replacement for a changed grid.

    ``grid`` is the inner grid as (height in feet, flags) per square, row-major over inner width x
    inner height (None or empty for no inner grid); squares outside it are the edge square of the
    header. Cells are a greedy merge of squares with equal low 12 flag bits (cell flags = those
    bits), z = range of their squares' heights; height range = range over the inner squares and the
    edge square, (+3.4028235e38, -3.4028235e38) with no inner grid; then W14-16; then the grid is
    re-quantised to the new range.
    """
    h = _header(scene)
    width, height = h.width, h.height
    inner_w, inner_h = h.inner_width, h.inner_height
    grid = list(grid or [])
    if grid and len(grid) != inner_w * inner_h:
        raise FormatError(f"grid of {len(grid)} squares for {inner_w} x {inner_h}")
    edge = (to_f32(h.edge_height), h.edge_flags)
    squares = []
    for y in range(height):
        for x in range(width):
            ix, iy = x - h.edge_left, y - h.edge_top
            if grid and 0 <= ix < inner_w and 0 <= iy < inner_h:
                z, flags = grid[ix + inner_w * iy]
                squares.append((to_f32(z), flags))
            else:
                squares.append(edge)
    rects = _greedy_merge(width, height, [flags & LOW_FLAGS for _, flags in squares])
    cells = []
    for r in rects:
        zs = [squares[x + width * y][0] for y in range(r.y0, r.y1) for x in range(r.x0, r.x1)]
        low12 = squares[r.x0 + width * r.y0][1] & LOW_FLAGS
        cells.append(NavCell(
            (to_f32(r.x0 * SQUARE_FEET), to_f32(r.y0 * SQUARE_FEET), min(zs)),
            (to_f32(r.x1 * SQUARE_FEET), to_f32(r.y1 * SQUARE_FEET), max(zs)),
            low12, 0, 0,
        ))
    if grid:
        heights = [to_f32(z) for z, _ in grid] + [edge[0]]
        low, high = min(heights), max(heights)
        quantised = [(quantise(to_f32(z), low, high), flags) for z, flags in grid]
    else:
        low, high = NO_GRID_RANGE
        quantised = []
    return RebuiltNavigation(low, high, quantised, derive_navigation(h, cells))
