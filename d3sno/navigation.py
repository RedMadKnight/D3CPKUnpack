"""Navigation data of a scene, for both builds."""

from __future__ import annotations

from dataclasses import dataclass, field

from .binary import ByteView, FormatError, build_order, to_f32
from .sno import SnoHeader, parse_header, resolve_build

CELL_SIZE = 32
NEIGHBOUR_SIZE = 4
COARSE_SIZE = 6
LOOKUP_SIZE = 4
BORDER_SIZE = 4
GRID_ENTRY_SIZE = 4
SPECIAL_FLAGS_PS3_AT = 0x228
NEIGHBOUR_BORDER = 0xFFFF
BORDER_INDEX_BASE = 0x8000
EMPTY_COARSE = 0xFFFF

SIDE_X_ZERO = 1
SIDE_X_MAX = 2
SIDE_Y_ZERO = 4
SIDE_Y_MAX = 8

FLAG_NAMES = {
    0: "walk", 1: "flier", 2: "spider", 3: "level_area_1", 4: "level_area_2", 5: "sub_scene",
    6: "no_spawn", 7: "special_1", 8: "special_2", 10: "projectile", 11: "ghost",
    12: "rounded_corner_1", 13: "rounded_corner_2", 14: "rounded_corner_3",
    15: "rounded_corner_4",
}


def flag_names(flags: int) -> list[str]:
    return [name for bit, name in FLAG_NAMES.items() if flags >> bit & 1]


def special_to_flags(nibble: int) -> int:
    """Bit 0 stays bit 0, and bits 1 and 2 become the level area bits 3 and 4."""
    return nibble & 1 | (nibble >> 1 & 3) << 3


@dataclass(frozen=True)
class NavCell:
    min: tuple[float, float, float]
    max: tuple[float, float, float]
    flags: int
    neighbour_count: int
    first_neighbour: int


@dataclass(frozen=True)
class Neighbour:
    flags: int
    cell: int

    @property
    def border_index(self) -> int | None:
        """The border this edge leads to, or None for an edge to a cell."""
        if self.flags == NEIGHBOUR_BORDER:
            return self.cell - BORDER_INDEX_BASE
        return None


@dataclass(frozen=True)
class CoarseSquare:
    flags: int
    count: int
    first_lookup: int

    @property
    def is_empty(self) -> bool:
        return self.flags == EMPTY_COARSE and self.count == 0


@dataclass(frozen=True)
class Lookup:
    flags: int
    cell: int


@dataclass(frozen=True)
class Border:
    cell: int
    sides: int


@dataclass(frozen=True)
class GridSquare:
    height: float | None  # not valid in flag layer 1
    flags: int
    is_edge: bool


@dataclass
class SceneHeader:
    width: int
    height: int
    type: int
    square_count: int
    edge_top: int
    edge_bottom: int
    edge_left: int
    edge_right: int
    height_min: float
    height_max: float
    edge_height: float
    edge_flags: int
    cell_count: int
    neighbour_count: int
    coarse_feet: float
    coarse_feet_inverse: float
    coarse_squares: int
    coarse_width: int
    coarse_height: int
    lookup_count: int
    border_count: int

    @property
    def inner_width(self) -> int:
        return self.width - self.edge_left - self.edge_right

    @property
    def inner_height(self) -> int:
        return self.height - self.edge_top - self.edge_bottom

    @property
    def section_sizes(self) -> tuple[int, int, int, int, int, int]:
        """Grid, cells, neighbours, coarse grid, lookups and borders, in bytes."""
        return (
            GRID_ENTRY_SIZE * self.inner_width * self.inner_height,
            CELL_SIZE * self.cell_count,
            NEIGHBOUR_SIZE * self.neighbour_count,
            COARSE_SIZE * self.coarse_width * self.coarse_height,
            LOOKUP_SIZE * self.lookup_count,
            BORDER_SIZE * self.border_count,
        )


@dataclass
class Navigation:
    header_sno: SnoHeader
    build: str
    header: SceneHeader
    has_sections: bool
    grid: list[tuple[int, int]] = field(default_factory=list)  # (z, flags)
    cells: list[NavCell] = field(default_factory=list)
    neighbours: list[Neighbour] = field(default_factory=list)
    coarse: list[CoarseSquare] = field(default_factory=list)
    lookups: list[Lookup] = field(default_factory=list)
    borders: list[Border] = field(default_factory=list)
    special_flags: bytes = b""
    warnings: list[str] = field(default_factory=list)

    def square(self, x: int, y: int, layer: int = 0) -> GridSquare:
        """The square at (x, y) in the full mesh.

        Layer 0 reads the grid, and with no grid every square is the edge square. Layer 1
        reads the special flags instead, and has no height.
        """
        h = self.header
        if not (0 <= x < h.width and 0 <= y < h.height):
            raise FormatError(f"square ({x}, {y}) outside {h.width}x{h.height}")
        inner_x, inner_y = x - h.edge_left, y - h.edge_top
        if not (0 <= inner_x < h.inner_width and 0 <= inner_y < h.inner_height):
            return GridSquare(None if layer else h.edge_height, h.edge_flags, True)
        index = inner_x + h.inner_width * inner_y
        if layer:
            return GridSquare(None, special_to_flags(self.special_flag(index)), False)
        if not self.grid:
            return GridSquare(h.edge_height, h.edge_flags, True)
        if index >= len(self.grid):
            raise FormatError(f"grid entry {index} of {len(self.grid)}")
        z, flags = self.grid[index]
        span = to_f32(h.height_max - h.height_min)
        return GridSquare(to_f32(h.height_min + span * to_f32(z / 65535)), flags, False)

    def special_flag(self, index: int) -> int:
        """The nibble of inner square ``index``."""
        if not 0 <= index >> 1 < len(self.special_flags):
            raise FormatError(f"no special flags for inner square {index}")
        return self.special_flags[index >> 1] >> (4 * (index & 1)) & 0xF


def square_in_world(
    nav: Navigation, x: int, y: int, position: tuple[float, float, float],
    sub_scenes: list[tuple[Navigation, tuple[float, float, float]]], layer: int = 0,
) -> GridSquare:
    """A lookup in a running world: each placed sub-scene is tried first.

    ``position`` is the scene's and each sub-scene comes with its own.
    """
    for sub, sub_position in sub_scenes:
        sub_x = x - int((sub_position[0] - position[0]) / 2.5)
        sub_y = y - int((sub_position[1] - position[1]) / 2.5)
        if 0 <= sub_x < sub.header.width and 0 <= sub_y < sub.header.height:
            square = sub.square(sub_x, sub_y, layer)
            if square.height is None:
                return square
            lifted = to_f32(square.height + (sub_position[2] - position[2]))
            return GridSquare(lifted, square.flags, square.is_edge)
    return nav.square(x, y, layer)


def _read_header(view: ByteView) -> SceneHeader:
    def u(body: int) -> int:
        return view.u32(0x10 + body, "scene header")

    def f(body: int) -> float:
        return view.f32(0x10 + body, "scene header")

    return SceneHeader(
        u(0x40), u(0x44), u(0x48), u(0x4C), u(0x50), u(0x54), u(0x58), u(0x5C), f(0x60), f(0x64),
        f(0x68), u(0x6C), u(0x180), u(0x198), f(0x1B0), f(0x1B4), u(0x1B8), u(0x1BC), u(0x1C0),
        u(0x1D8), u(0x1F0),
    )


def _rows(raw: bytes, order: str, row: str, width: int) -> list[tuple]:
    view = ByteView(raw, order)
    return [view.unpack(row, pos) for pos in range(0, len(raw), width)]


def _fill_sections(nav: Navigation, sections: list[bytes], order: str) -> None:
    grid, cells, neighbours, coarse, lookups, borders = sections
    nav.grid = _rows(grid, order, "HH", GRID_ENTRY_SIZE)
    nav.cells = [
        NavCell(row[:3], row[3:6], row[6], row[7], row[8])
        for row in _rows(cells, order, "6fHHI", CELL_SIZE)
    ]
    nav.neighbours = [Neighbour(*row) for row in _rows(neighbours, order, "HH", NEIGHBOUR_SIZE)]
    nav.coarse = [CoarseSquare(*row) for row in _rows(coarse, order, "HHH", COARSE_SIZE)]
    nav.lookups = [Lookup(*row) for row in _rows(lookups, order, "HH", LOOKUP_SIZE)]
    nav.borders = [Border(*row) for row in _rows(borders, order, "HH", BORDER_SIZE)]


def _switch_sections(view: ByteView, h: SceneHeader) -> list[bytes]:
    sections = []
    for name, body, size in zip(
        ("grid", "cells", "neighbours", "coarse grid", "lookups", "borders"),
        (0x80, 0x190, 0x1A8, 0x1D0, 0x1E8, 0x200),
        h.section_sizes,
    ):
        pair = view.pair(0x10 + body, f"{name} pair")
        if pair.size != size:
            raise FormatError(f"{name}: pair size {pair.size}, expected {size}", 0x10 + body)
        sections.append(view.pair_data(pair, 0x10 + body, name))
    return sections


def _ps3_sections(nav_data: bytes, h: SceneHeader) -> list[bytes]:
    sizes = h.section_sizes
    if len(nav_data) != sum(sizes):
        raise FormatError(f".nav of {len(nav_data)} bytes, expected {sum(sizes)}", 0)
    sections = []
    pos = 0
    for size in sizes:
        sections.append(bytes(nav_data[pos:pos + size]))
        pos += size
    return sections


def _check_links(nav: Navigation) -> None:
    """Report records that point at something missing or disagree with their target."""
    cells, warnings = nav.cells, nav.warnings
    for index, cell in enumerate(cells):
        if cell.first_neighbour + cell.neighbour_count > len(nav.neighbours):
            warnings.append(f"cell {index}: neighbours past the table")
    for index, edge in enumerate(nav.neighbours):
        border = edge.border_index
        if border is not None:
            if not 0 <= border < len(nav.borders):
                warnings.append(f"neighbour {index}: border {border} does not exist")
        elif edge.cell >= len(cells):
            warnings.append(f"neighbour {index}: cell {edge.cell} does not exist")
        elif edge.flags != cells[edge.cell].flags & 0x0FFF:
            warnings.append(f"neighbour {index}: flags differ from the target cell")
    for index, square in enumerate(nav.coarse):
        if square.first_lookup + square.count > len(nav.lookups):
            if not square.is_empty:
                warnings.append(f"coarse square {index}: lookups past the table")
    for index, lookup in enumerate(nav.lookups):
        if lookup.cell >= len(cells):
            warnings.append(f"lookup {index}: cell {lookup.cell} does not exist")
    for index, border in enumerate(nav.borders):
        if border.cell >= len(cells):
            warnings.append(f"border {index}: cell {border.cell} does not exist")


def parse_scene(
    data: bytes, build: str | None = None, nav: bytes | None = None, strict: bool = False,
) -> Navigation:
    """Read a ``.scn``. On ps3, pass the ``.nav`` bytes; a scene of computed size 0 has none."""
    sno = parse_header(data, build)
    resolved = resolve_build("scn", sno, build, strict)
    order = build_order(resolved)
    view = ByteView(data, order)
    h = _read_header(view)
    inner = h.inner_width * h.inner_height
    if h.inner_width < 0 or h.inner_height < 0:
        raise FormatError(f"edges leave an inner grid of {h.inner_width}x{h.inner_height}", 0x60)
    result = Navigation(sno, resolved, h, True)
    if h.square_count != h.width * h.height:
        result.warnings.append(f"square count {h.square_count} for {h.width}x{h.height}")
    if resolved == "switch":
        sections = _switch_sections(view, h)
        result.special_flags = view.pair_data(view.pair(0xA0), 0xA0, "special flags")
    else:
        result.special_flags = view.take(
            SPECIAL_FLAGS_PS3_AT, (inner + 1) // 2, "special flags",
        )
        if nav is None:
            if sum(h.section_sizes):
                result.has_sections = False
                return result
            nav = b""
        sections = _ps3_sections(nav, h)
    if len(result.special_flags) != (inner + 1) // 2:
        result.warnings.append(f"{len(result.special_flags)} special flag bytes for {inner} squares")
    _fill_sections(result, sections, order)
    _check_links(result)
    if strict and result.warnings:
        raise FormatError(result.warnings[0], 0x10)
    return result
