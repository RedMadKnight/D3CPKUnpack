"""``.gam`` game balance tables, game balance ids and the formula table."""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field
from typing import Iterable

from .binary import ByteView, FormatError, Pair
from .formula import ValueContext, first_table_class, table_row_of_class
from .sno import SnoHeader, hash33_ci, parse_header

GBID_ALL = 109305
FORMULA_TYPE = 36
FORMULA_NAME_SIZE = 1024
FORMULA_COLUMNS = 76
_FIRST_SLOT = {"switch": 0x28, "ps3": 0x228}
_FIXED_END = {"switch": 0x248, "ps3": 0x3E8}
_SLOT_SIZE = 16
MONSTER_LEVEL_TYPE = 5
_MONSTER_HEALTH_AT = 0x14
_HEALTH_CLASS_OFFSET = 14

# Type: (table name, switch slot, switch row, ps3 slot, ps3 row). A None slot was not seen on
# that build. A None row size means bytes were seen but the row is not known.
TABLES = {
    1: ("item_types", 0x28, 96, 0x228, 80),
    2: ("items", 0x38, 696, 0x238, 1320),
    3: ("experience", 0x48, 496, 0x248, 316),
    5: ("monster_levels", 0x78, 240, 0x268, 572),
    8: ("affixes", 0x88, 544, 0x278, 512),
    7: ("heroes", 0x98, 264, 0x288, 648),
    10: ("movement_styles", 0xA8, 152, 0x298, 144),
    11: ("labels", 0xB8, 32, 0x2A8, 24),
    12: ("loot_distribution", 0xC8, 92, 0x2B8, 92),
    16: ("rare_item_names", 0xD8, 40, 0x2C8, 32),
    18: ("monster_affixes", 0xE8, 664, 0x2D8, 552),
    19: ("monster_names", 0xF8, 160, 0x2E8, 152),
    21: ("socketed_effects", 0x108, 1176, 0x2F8, 1176),
    25: ("item_drops", 0x118, 904, 0x318, 900),
    26: ("item_level_modifiers", 0x128, 92, 0x328, 92),
    27: ("quality_classes", 0x138, 120, 0x338, 112),
    4: ("hirelings", 0x168, 88, 0x368, 604),
    33: ("set_bonuses", 0x178, 224, 0x378, 224),
    34: ("elite_modifiers", 0x188, 112, 0x388, 104),
    35: ("item_tiers", 0x198, 32, 0x398, 32),
    36: ("formula_table", 0x1A8, 1328, 0x3A8, 1328),
    32: ("recipes", 0x1B8, 96, 0x3B8, 92),
    37: ("achievement_events", 0x1C8, 24, 0x3C8, 40),
    40: ("paragon_bonuses", 0x1E8, 400, 0x3D8, 392),
    24: ("help_codes", 0x68, 640, None, None),
    28: ("handicap_levels", 0x148, 32, 0x348, None),
    29: ("item_salvage_levels", 0x158, 16, 0x358, None),
    39: ("loot_run_quest_tiers", 0x1D8, 280, None, None),
    45: ("legacy_item_conversions", 0x1F8, 40, None, None),
    46: ("enchant_use_count_cost_scalars", 0x208, 8, None, None),
    48: ("alternative_experience", 0x58, 128, None, None),
    49: ("tiered_loot_run_levels", 0x218, 56, None, None),
    50: ("transmute_recipes", 0x228, 136, None, None),
    51: ("currency_conversions", 0x238, 64, None, None),
    23: ("open_23", None, None, 0x308, None),
}


def gbid(name: str) -> int:
    """The game balance id of a name, the case-insensitive hash as signed 32-bit.

    An empty name has id -1. The name is cut to 255 bytes first.
    """
    if not name:
        return -1
    value = hash33_ci(name.encode("latin-1", "replace")[:255])
    return value - (1 << 32) if value >= 1 << 31 else value


@dataclass
class BalanceRow:
    index: int
    name: str | None
    raw: bytes

    @property
    def id(self) -> int | None:
        return None if self.name is None else gbid(self.name)


@dataclass
class FormulaRow:
    name: str
    values: tuple[float, ...]


@dataclass
class BalanceFile:
    header: SnoHeader
    build: str
    type: int
    table: str | None
    row_size: int | None
    rows: list[BalanceRow]
    raw_slots: dict[int, bytes] = field(default_factory=dict)  # by file position
    formula_rows: list[FormulaRow] = field(default_factory=list)


def _row_name(view: ByteView, row: bytes) -> str | None:
    """Read the name pair most rows begin with. Other rows give None."""
    pair = Pair(*struct.unpack_from(view.order + "II", row, 0))
    if pair.size == 0 or pair.file_offset + pair.size > len(view.data):
        return None
    raw = bytes(view.data[pair.file_offset:pair.file_offset + pair.size]).split(b"\0")[0]
    if not raw or any(not 0x20 <= byte < 0x7F for byte in raw):
        return None
    return raw.decode("ascii")


def _read_formula_row(raw: bytes, order: str) -> FormulaRow:
    name = raw[:FORMULA_NAME_SIZE].split(b"\0")[0].decode("latin-1")
    values = struct.unpack_from(f"{order}{FORMULA_COLUMNS}f", raw, FORMULA_NAME_SIZE)
    return FormulaRow(name, values)


def _other_slots(view: ByteView, build: str, skip: int | None) -> dict[int, bytes]:
    """Filled slots outside the selected table. Stale pairs that do not fit are left out."""
    found = {}
    for position in range(_FIRST_SLOT[build], _FIXED_END[build], _SLOT_SIZE):
        pair = view.pair(position)
        if position == skip or pair.size == 0:
            continue
        if pair.file_offset + pair.size <= len(view.data):
            found[position] = bytes(view.data[pair.file_offset:pair.file_offset + pair.size])
    return found


def parse_balance(data: bytes, build: str | None = None) -> BalanceFile:
    header = parse_header(data, build)
    resolved = header.build
    view = ByteView(data, header.order)
    view.need(0, _FIXED_END[resolved], "fixed part")
    kind = view.i32(0x1C, "type word")
    known = TABLES.get(kind)
    if known is None:
        return BalanceFile(
            header, resolved, kind, None, None, [], _other_slots(view, resolved, None),
        )
    name, switch_slot, switch_row, ps3_slot, ps3_row = known
    slot, row_size = (ps3_slot, ps3_row) if resolved == "ps3" else (switch_slot, switch_row)
    if slot is None or row_size is None:
        return BalanceFile(
            header, resolved, kind, name, None, [], _other_slots(view, resolved, None),
        )
    raw = view.pair_data(view.pair(slot), slot, name)
    if len(raw) % row_size:
        raise FormatError(f"{name}: {len(raw)} bytes is not a multiple of {row_size}", slot)
    parts = [raw[i:i + row_size] for i in range(0, len(raw), row_size)]
    result = BalanceFile(
        header, resolved, kind, name, row_size, [], _other_slots(view, resolved, slot),
    )
    if kind == FORMULA_TYPE:
        result.formula_rows = [_read_formula_row(part, view.order) for part in parts]
        result.rows = [
            BalanceRow(i, row.name, part)
            for i, (row, part) in enumerate(zip(result.formula_rows, parts))
        ]
    else:
        result.rows = [
            BalanceRow(i, _row_name(view, part), part) for i, part in enumerate(parts)
        ]
    return result


class GbidResolver:
    """Finds rows by game balance id. A miss can be a family hash, and is not an error."""

    def __init__(self, files: Iterable[BalanceFile] = ()) -> None:
        self._rows: dict[int, tuple[BalanceFile, BalanceRow]] = {}
        for balance in files:
            self.add(balance)

    def add(self, balance: BalanceFile) -> None:
        for row in balance.rows:
            if row.id is not None:
                self._rows.setdefault(row.id, (balance, row))

    def resolve(self, value: int) -> tuple[BalanceFile, BalanceRow] | None:
        signed = value - (1 << 32) if value >= 1 << 31 else value
        return self._rows.get(signed)

    @staticmethod
    def is_sentinel(value: int) -> bool:
        return value == GBID_ALL


def table_value(rows: list[FormulaRow], row: float, column: float) -> float:
    """``Table(row, col)``. Both truncate to integers; out of range gives 0."""
    if not (math.isfinite(row) and math.isfinite(column)):
        return 0.0
    r, c = int(row), int(column)
    if not (0 <= r < len(rows) and 0 <= c < FORMULA_COLUMNS):
        return 0.0
    return rows[r].values[c]


class TableContext(ValueContext):
    """An evaluation context with the formula table.

    Table classes are constants in the evaluator. ``monster_levels`` with ``level`` and
    ``difficulty_scalar`` supply health minimum, class 101 on switch and 77 on ps3.
    """

    def __init__(
        self, rows: list[FormulaRow], build: str, values=None,
        monster_levels: BalanceFile | None = None, level: int = 1,
        difficulty_scalar: float = 1.0,
    ) -> None:
        super().__init__(values or {})
        self.rows = rows
        self.build = build
        self.monster_levels = monster_levels
        self.level = level
        self.difficulty_scalar = difficulty_scalar

    def table(self, a: float, b: float) -> float:
        return table_value(self.rows, a, b)

    def identifier(self, cls: int, a: int, b: int, c: int) -> float:
        health_class = first_table_class(self.build) + _HEALTH_CLASS_OFFSET
        if cls == health_class and self.monster_levels is not None:
            rows = self.monster_levels.rows
            if not 1 <= self.level <= len(rows):
                raise FormatError(f"no monster level row for level {self.level}")
            health = ByteView(rows[self.level - 1].raw, self.monster_levels.header.order)
            return health.f32(_MONSTER_HEALTH_AT) * self.difficulty_scalar
        return super().identifier(cls, a, b, c)
