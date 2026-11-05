"""Enum and flag names from ``data/cov_enums_core.tsv`` and ``data/cov_enums_records.tsv``.

Names are looked up; raw values stay with the caller. A value with no row has no name.
Geometry enums come from ``data/cov_enums_geometry.tsv`` (stage 7).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from functools import lru_cache

from .binary import ByBuild
from .datafiles import DATA

CLASS_COUNTS = ByBuild({"switch": 7, "ps3": 5})
EMPTY_NAME = "Normal"
UNNAMED_STATUSES = ("OPEN", "HYPOTHESIS")


def signed32(value: int) -> int:
    value &= 0xFFFFFFFF
    return value - (1 << 32) if value >= 1 << 31 else value


@lru_cache(maxsize=None)
def core_enums() -> dict[str, dict[int, str]]:
    """Every core enum: its values and names."""
    result: dict[str, dict[int, str]] = {}
    with open(DATA / "cov_enums_core.tsv", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            # An empty name is a known value with no display text: ``Normal`` (stage 7, T2).
            name = row["name"] or EMPTY_NAME
            result.setdefault(row["enum"], {}).setdefault(int(row["value"]), name)
    return result


def enum_name(enum: str, value: int) -> str | None:
    """The name of a value of a core enum, such as ``skill_category`` or ``tag_value:gActs``."""
    values = core_enums().get(enum)
    if values is None:
        return None
    return values.get(value, values.get(signed32(value)))


def bit_names(enum: str, value: int) -> tuple[list[str], list[int]]:
    """Names of the set bits of ``value`` from an enum of bit positions, and unnamed bits."""
    names, unknown = [], []
    for bit in range(32):
        if value >> bit & 1:
            name = enum_name(enum, bit)
            if name is None:
                unknown.append(bit)
            else:
                names.append(name)
    return names, unknown


@dataclass(frozen=True)
class RecordValue:
    kind: str  # value or bit
    value: int
    constant: str
    label: str
    status: str

    @property
    def name(self) -> str | None:
        """The constant, else the label; rows with status OPEN or HYPOTHESIS have no name."""
        if self.constant:
            return self.constant
        if self.status in UNNAMED_STATUSES or not self.label:
            return None
        return self.label


def _value_table(name: str) -> dict[tuple[str, str], dict[int, RecordValue]]:
    result: dict[tuple[str, str], dict[int, RecordValue]] = {}
    with open(DATA / name, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="	"):
            entry = RecordValue(
                row["kind"], int(row["value"]), row["constant"], row["label"], row["status"],
            )
            result.setdefault((row["record"], row["field"]), {})[entry.value] = entry
    return result


@lru_cache(maxsize=None)
def record_enums() -> dict[tuple[str, str], dict[int, RecordValue]]:
    return _value_table("cov_enums_records.tsv")


@lru_cache(maxsize=None)
def geometry_enums() -> dict[tuple[str, str], dict[int, RecordValue]]:
    """Enum values and flag bits of the eleven geometry types (stage 7, T8)."""
    return _value_table("cov_enums_geometry.tsv")


def geometry_value(record: str, field: str, value: int) -> RecordValue | None:
    """The row of a geometry enum value; ``record`` is the layout record (``Marker``)."""
    values = geometry_enums().get((record, field), {})
    return values.get(value, values.get(signed32(value)))


def geometry_flags(record: str, field: str, value: int) -> FlagNames:
    """A geometry flags word bit by bit (sub-object bits are positions in ``dwFlags``)."""
    return _flags(geometry_enums().get((record, field), {}), value, 32)


def is_geometry_flags(record: str, field: str) -> bool:
    return any(v.kind == "bit" for v in geometry_enums().get((record, field), {}).values())


def record_value(record: str, field: str, value: int) -> RecordValue | None:
    """The row of an enum value of a gameplay record field."""
    values = record_enums().get((record, field), {})
    return values.get(value, values.get(signed32(value)))


@dataclass(frozen=True)
class FlagNames:
    names: list[str]
    unknown_bits: list[int]
    ignored_bits: list[int]  # leftovers the description says to ignore
    open_bits: list[int] = field(default_factory=list)  # bits with a row but no name (OPEN)
    plain_number: int | None = None  # the word holds a number, not flags


# Three monster records store a plain number in their flags word.
MONSTER_FLAG_NUMBERS = (10000, 9988)


def record_flags(record: str, field: str, value: int, build: str = "switch") -> FlagNames:
    """Read a flags field bit by bit. A bit with no row is unknown.

    In the conversation-list class mask, bits at or above the class count are leftovers.
    """
    bits = record_enums().get((record, field), {})
    value &= 0xFFFFFFFF
    if (record, field) == ("monster", "flags") and value in MONSTER_FLAG_NUMBERS:
        return FlagNames([], [], [], [], value)
    limit = CLASS_COUNTS[build] if (record, field) == ("list_entry", "class_flags") else 32
    return _flags(bits, value, limit)


def _flags(bits: dict[int, RecordValue], value: int, limit: int) -> FlagNames:
    names, unknown, ignored, open_ = [], [], [], []
    value &= 0xFFFFFFFF
    for bit in range(32):
        if not value >> bit & 1:
            continue
        if bit >= limit:
            ignored.append(bit)
        elif bit in bits and bits[bit].name:
            names.append(bits[bit].name)
        elif bit in bits:
            open_.append(bit)
        else:
            unknown.append(bit)
    return FlagNames(names, unknown, ignored, open_)


def is_flags_field(record: str, field: str) -> bool:
    values = record_enums().get((record, field), {})
    return any(v.kind == "bit" for v in values.values())
