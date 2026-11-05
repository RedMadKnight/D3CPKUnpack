"""``.gam`` game balance tables, game balance ids and the formula table."""

from __future__ import annotations

import csv
import math
import struct
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Iterable

from .binary import ByBuild, ByteView, FormatError, Pair
from .datafiles import DATA
from .enums import enum_name
from .formula import ValueContext, first_table_class, table_row_of_class
from .sno import SnoHeader, hash33_ci, parse_header, resolve_build

GBID_ALL = 109305
FORMULA_TYPE = 36
FORMULA_NAME_SIZE = 1024
FORMULA_COLUMNS = 76
FIXED_END = ByBuild({"switch": 0x248, "ps3": 0x3E8, "x360": 0x278})  # x360: 37 slots (X7)
_FIRST_SLOT = ByBuild({"switch": 0x28, "ps3": 0x228})
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
    28: ("handicap_levels", 0x148, 32, 0x348, 60),
    29: ("item_salvage_levels", 0x158, 16, 0x358, 28),
    39: ("loot_run_quest_tiers", 0x1D8, 280, None, None),
    45: ("legacy_item_conversions", 0x1F8, 40, None, None),
    46: ("enchant_use_count_cost_scalars", 0x208, 8, None, None),
    48: ("alternative_experience", 0x58, 128, None, None),
    49: ("tiered_loot_run_levels", 0x218, 56, None, None),
    50: ("transmute_recipes", 0x228, 136, None, None),
    51: ("currency_conversions", 0x238, 64, None, None),
    23: ("item_enhancements", None, None, 0x308, 456),
}
# Fixed-part fields: (machine time, ignore-on-load flag); ps3 adds two 256-byte texts.
_MACHINE_TIME_AT = ByBuild({"switch": 0x20, "ps3": 0x220})
_PS3_GBI_AT = 0x20
_PS3_SOURCE_AT = 0x120
_PS3_TEXT_SIZE = 256
NO_TABLE_TYPES = (9, -1)


@dataclass(frozen=True)
class GamPairSlot:
    """A pair in a balance row (``cov_gam_pairs.tsv``)."""

    offset: int
    kind: str  # name, code, text or array
    element_size: int | None = None  # arrays
    nested_code_offsets: tuple[int, ...] = ()  # arrays: code pairs inside each element
    field: str = ""


@lru_cache(maxsize=None)
def _pair_table() -> dict[tuple[str, int], tuple[GamPairSlot, ...]]:
    found: dict[tuple[str, int], list[GamPairSlot]] = {}
    for table in ("cov_gam_pairs.tsv", "x360_gam_pairs.tsv"):
        with open(DATA / table, encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                nested = tuple(
                    int(part, 16) for part in row["nested_code_offsets"].split(",") if part.strip()
                )
                found.setdefault((row["build"], int(row["type"])), []).append(GamPairSlot(
                    int(row["offset"], 16), row["kind"],
                    int(row["element_size"]) if row["element_size"] else None, nested, row["field"],
                ))
    return {key: tuple(sorted(slots, key=lambda s: s.offset)) for key, slots in found.items()}


def gam_pair_slots(build: str, kind: int) -> tuple[GamPairSlot, ...]:
    """The pair slots of a row of table ``kind``, in increasing offset. Empty: no pairs."""
    return _pair_table().get((build, kind), ())


@dataclass(frozen=True)
class X360GamTable:
    """A row of ``x360_gam_tables.tsv``: where a type's table sits on x360 and its row size."""

    type: int
    type_name: str
    table_field: str
    slot: int
    row_struct: str
    row_size: int
    status: str

    @property
    def name(self) -> str:
        """The table name: the Switch table's name where the type has one, else from the type name."""
        if self.type in TABLES:
            return TABLES[self.type][0]
        words = self.type_name.split(" (")[0].lower().split()
        return "_".join(words)


_NO_SWITCH_NAME = " (no name in the Switch enum)"


@lru_cache(maxsize=None)
def x360_gam_tables() -> dict[int, X360GamTable]:
    with open(DATA / "x360_gam_tables.tsv", encoding="utf-8", newline="") as handle:
        return {
            int(row["type"]): X360GamTable(
                int(row["type"]), row["type_name"].replace(_NO_SWITCH_NAME, ""), row["table_field"],
                int(row["slot_file_offset"], 16), row["row_struct"], int(row["row_size"]),
                row["status"],
            )
            for row in csv.DictReader(handle, delimiter="\t")
        }


def balance_type_name(kind: int, build: str = "switch") -> str | None:
    """The type's name: the ``game_balance_type`` enum; on x360 the four x360-only types
    (38, 42, 43, 44) take the name of ``x360_gam_tables.tsv`` (X10)."""
    name = enum_name("game_balance_type", kind)
    if name is None and build == "x360" and kind in x360_gam_tables():
        return x360_gam_tables()[kind].type_name
    return name


@dataclass(frozen=True)
class GamRowField:
    type: int
    row: str
    offset: int
    field: str
    field_type: str
    element: str
    count: int | None
    sno_group: int | None
    value_table: str | None


@lru_cache(maxsize=None)
def gam_row_fields_switch() -> dict[int, tuple[GamRowField, ...]]:
    """Every field of every switch balance row (``cov_gam_row_fields_switch.tsv``), by type."""
    found: dict[int, list[GamRowField]] = {}
    with open(DATA / "cov_gam_row_fields_switch.tsv", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            found.setdefault(int(row["type"]), []).append(GamRowField(
                int(row["type"]), row["row"], int(row["offset"], 16), row["field"],
                row["field_type"], row["element"],
                int(row["count"]) if row["count"] else None,
                int(row["sno_group"]) if row["sno_group"] else None,
                row["value_table"] or None,
            ))
    return {key: tuple(value) for key, value in found.items()}


@dataclass
class GamBlock:
    """The bytes one pair of a row addresses."""

    kind: str
    data: bytes
    source: int | None = None  # file position in the source; equal positions share one copy
    elements: list[dict[int, "GamBlock"]] = field(default_factory=list)  # arrays: nested code


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
    blocks: dict[int, GamBlock] = field(default_factory=dict)  # by pair offset in the row

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
    source: bytes = b""
    slot: int | None = None
    machine_time: int = 0
    ignore_on_load: int = 0
    gbi_name: str | None = None  # ps3: the text at 0x20
    source_path: str | None = None  # ps3: the spreadsheet path at 0x120
    fixed_problems: list[str] = field(default_factory=list)

    @property
    def type_name(self) -> str | None:
        return balance_type_name(self.type, self.build)

    @property
    def pair_slots(self) -> tuple[GamPairSlot, ...]:
        return gam_pair_slots(self.build, self.type)


def _name_text(raw: bytes) -> str | None:
    text = raw.split(b"\0")[0]
    if not text or any(not 0x20 <= byte < 0x7F for byte in text):
        return None
    return text.decode("ascii")


def _read_block(view: ByteView, row: bytes, at: int, base: int, slot: GamPairSlot) -> GamBlock:
    """Read the block of one pair. ``base`` is the file position of ``row``."""
    pair = Pair(*struct.unpack_from(view.order + "II", row, at))
    if pair.size == 0:
        return GamBlock(slot.kind, b"")
    data = view.pair_data(pair, base + at, slot.kind)
    block = GamBlock(slot.kind, data, pair.file_offset)
    if slot.kind == "array":
        size = slot.element_size or 1
        if len(data) % size:
            raise FormatError(f"array of {len(data)} bytes is not a multiple of {size}", base + at)
        nested = GamPairSlot(0, "code")
        for start in range(0, len(data), size):
            block.elements.append({
                offset: _read_block(view, data, start + offset, pair.file_offset + start, nested)
                for offset in slot.nested_code_offsets
            })
    return block


def _fixed_text(view: ByteView, at: int, what: str, problems: list[str]) -> str:
    raw = view.take(at, _PS3_TEXT_SIZE)
    end = raw.find(0)
    if end >= 0 and any(raw[end:]):
        problems.append(f"{what}: bytes after the NUL")
    return view.cstring(at, _PS3_TEXT_SIZE)


def _read_fixed(view: ByteView, result: "BalanceFile") -> None:
    at = _MACHINE_TIME_AT[result.build]
    result.machine_time, result.ignore_on_load = view.unpack("II", at)
    if result.build == "ps3":
        result.gbi_name = _fixed_text(view, _PS3_GBI_AT, "gbi name", result.fixed_problems)
        result.source_path = _fixed_text(view, _PS3_SOURCE_AT, "source path", result.fixed_problems)
    if result.ignore_on_load:
        result.fixed_problems.append(f"ignore-on-load flag is {result.ignore_on_load}")


def _read_formula_row(raw: bytes, order: str) -> FormulaRow:
    name = raw[:FORMULA_NAME_SIZE].split(b"\0")[0].decode("latin-1")
    values = struct.unpack_from(f"{order}{FORMULA_COLUMNS}f", raw, FORMULA_NAME_SIZE)
    return FormulaRow(name, values)


def _other_slots(view: ByteView, build: str, skip: int | None) -> dict[int, bytes]:
    """Filled slots outside the selected table. Stale pairs that do not fit are left out."""
    found = {}
    for position in range(_FIRST_SLOT[build], FIXED_END[build], _SLOT_SIZE):
        pair = view.pair(position)
        if position == skip or pair.size == 0:
            continue
        if pair.file_offset + pair.size <= len(view.data):
            found[position] = bytes(view.data[pair.file_offset:pair.file_offset + pair.size])
    return found


def parse_balance(data: bytes, build: str | None = None) -> BalanceFile:
    header = parse_header(data, build)
    resolved = resolve_build("gam", header, build)
    view = ByteView(data, header.order)
    view.need(0, FIXED_END[resolved], "fixed part")
    kind = view.i32(0x1C, "type word")
    known = TABLES.get(kind)
    if resolved == "x360":  # X7: slot and row size per type from the x360 table
        x360 = x360_gam_tables().get(kind)
        known = None if x360 is None else (x360.name, x360.slot, x360.row_size, None, None)
    if known is None:
        result = BalanceFile(
            header, resolved, kind, None, None, [], _other_slots(view, resolved, None),
            source=bytes(data),
        )
        _read_fixed(view, result)
        return result
    name, switch_slot, switch_row, ps3_slot, ps3_row = known
    slot, row_size = (ps3_slot, ps3_row) if resolved == "ps3" else (switch_slot, switch_row)
    if slot is None or row_size is None:
        result = BalanceFile(
            header, resolved, kind, name, None, [], _other_slots(view, resolved, None),
            source=bytes(data),
        )
        _read_fixed(view, result)
        return result
    raw = view.pair_data(view.pair(slot), slot, name)
    if len(raw) % row_size:
        raise FormatError(f"{name}: {len(raw)} bytes is not a multiple of {row_size}", slot)
    parts = [raw[i:i + row_size] for i in range(0, len(raw), row_size)]
    result = BalanceFile(
        header, resolved, kind, name, row_size, [], _other_slots(view, resolved, slot),
        source=bytes(data), slot=slot,
    )
    _read_fixed(view, result)
    if kind == FORMULA_TYPE:
        result.formula_rows = [_read_formula_row(part, view.order) for part in parts]
        result.rows = [
            BalanceRow(i, row.name, part)
            for i, (row, part) in enumerate(zip(result.formula_rows, parts))
        ]
    else:
        table = view.pair(slot).file_offset
        slots = gam_pair_slots(resolved, kind)
        for i, part in enumerate(parts):
            base = table + i * row_size
            blocks = {s.offset: _read_block(view, part, s.offset, base, s) for s in slots}
            name_block = blocks.get(0)
            row_name = (
                _name_text(name_block.data) if name_block and name_block.kind == "name" else None
            )
            result.rows.append(BalanceRow(i, row_name, part, blocks))
    return result


def code_blocks(file: BalanceFile):
    """Every ``code`` block: ``(row, offset, element or None, block)``, nested ones included."""
    for row in file.rows:
        for offset, block in row.blocks.items():
            if block.kind == "code":
                yield row.index, offset, None, block
            for number, nested in enumerate(block.elements):
                for inner, code in nested.items():
                    yield row.index, inner, number, code


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
