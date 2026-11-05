"""Gameplay records driven by ``data/layouts.tsv``, plus tag defaults, attributes and text.

The field tables are data. A record reader walks them, follows every pair, and exposes
fields by their neutral names. A name that repeats in one record is numbered from 1.
"""

from __future__ import annotations

import csv
import math
import re
import struct
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from .binary import ByBuild, ByteView, FormatError, Pair
from .power import TagMap, read_tag_map
from .sno import SnoHeader, hash33, parse_header, resolve_build

DATA = Path(__file__).parent / "data"
TICKS_PER_SECOND = 60
_MAX_DEPTH = 64
_AREA_ALIGNMENT = 8

# File extension: the record a file holds.
RECORDS_BY_EXTENSION = {
    "acr": "actor", "mon": "monster", "qst": "quest", "cnv": "conversation",
    "cnl": "conversation_list", "trs": "treasure_class", "cnd": "condition", "rcp": "recipe",
}
_TABLE_BUILDS = {"nx": "switch", "ps3": "ps3", "x360": "x360"}


@dataclass(frozen=True)
class FieldSpec:
    offset: int
    size: int
    type: str
    count: int | None
    name: str
    meaning: str


@dataclass(frozen=True)
class Layout:
    record: str
    build: str
    size: int
    fields: tuple[FieldSpec, ...]


def _numbered(names: list[str]) -> list[str]:
    totals: dict[str, int] = {}
    for name in names:
        totals[name] = totals.get(name, 0) + 1
    seen: dict[str, int] = {}
    result = []
    for name in names:
        if totals[name] == 1:
            result.append(name)
        else:
            seen[name] = seen.get(name, 0) + 1
            result.append(f"{name}_{seen[name]}")
    return result


@lru_cache(maxsize=None)
def layouts() -> dict[tuple[str, str], Layout]:
    """All record layouts, keyed by (build, record)."""
    sizes: dict[tuple[str, str], int] = {}
    rows: dict[tuple[str, str], list[FieldSpec]] = {}
    table_rows: list[dict] = []
    for table in ("layouts.tsv", "x360_layouts.tsv"):  # x360: seven records of their own (X3)
        with open(DATA / table, encoding="utf-8", newline="") as handle:
            table_rows += csv.DictReader(handle, delimiter="\t")
    for row in table_rows:
        key = (_TABLE_BUILDS[row["build"]], row["record"])
        if row["offset"] == "-":
            sizes[key] = int(row["bytes"])
            continue
        count = int(row["count"]) if row["count"] else None
        rows.setdefault(key, []).append(FieldSpec(
            int(row["offset"], 16), int(row["bytes"]), row["type"], count, row["name"],
            row["meaning"],
        ))
    result = {}
    for key, size in sizes.items():
        specs = sorted(rows.get(key, []), key=lambda spec: spec.offset)
        names = _numbered([spec.name for spec in specs])
        result[key] = Layout(key[1], key[0], size, tuple(
            FieldSpec(s.offset, s.size, s.type, s.count, n, s.meaning) for s, n in zip(specs, names)
        ))
    return result


def layout(build: str, record: str) -> Layout:
    """A record layout. x360 has its own rows for seven records; all others are Switch's (X3)."""
    try:
        found = layouts().get((build, record))
        if found is None and build == "x360":
            found = layouts().get(("switch", record))
        if found is None:
            raise KeyError(record)
        return found
    except KeyError:
        raise FormatError(f"no {record} layout for {build}") from None


@dataclass
class Record:
    type: str
    position: int  # file offset
    raw: bytes
    values: dict[str, object] = field(default_factory=dict)
    build: str | None = None

    def __getitem__(self, name: str):
        return self.values[name]

    def enum(self, name: str):
        """The ``cov_enums_records.tsv`` row of an enum field's value, or None."""
        from .enums import record_value

        value = self.values.get(name)
        return record_value(self.type, _table_field(self.type, name), value) if isinstance(value, int) else None

    def flags(self, name: str):
        """A flags field read bit by bit (``FlagNames``)."""
        from .enums import record_flags

        return record_flags(
            self.type, _table_field(self.type, name), self.values[name], self.build or "switch",
        )

    def named(self) -> dict[str, object]:
        """Every enum and flags field with a table: constant names, raw values kept apart."""
        from .enums import is_flags_field, record_enums

        found: dict[str, object] = {}
        tables = {name for (record, name) in record_enums() if record == self.type}
        for name, value in self.values.items():
            base = _table_field(self.type, name)
            if base not in tables or not isinstance(value, int):
                continue
            if is_flags_field(self.type, base):
                found[name] = self.flags(name)
            else:
                row = self.enum(name)
                found[name] = row.name if row else None
        return found

    def get(self, name: str, default=None):
        return self.values.get(name, default)

    def slots(self, name: str) -> list:
        """Every value of a repeated name, in order: ``lore_1``, ``lore_2`` and so on."""
        if name in self.values:
            return [self.values[name]]
        found = []
        index = 1
        while f"{name}_{index}" in self.values:
            found.append(self.values[f"{name}_{index}"])
            index += 1
        return found


_NUMBERED = re.compile(r"^(.*)_(\d+)$")


def _table_field(record: str, name: str) -> str:
    """The enum-table name of a field; repeated layout names carry a ``_<n>`` suffix."""
    from .enums import record_enums

    if (record, name) in record_enums():
        return name
    match = _NUMBERED.match(name)
    return match.group(1) if match else name


@dataclass
class RecordFile:
    header: SnoHeader
    build: str
    record: Record
    areas: list[tuple[int, int]]  # file offset and size of every block a pair points to
    warnings: list[str] = field(default_factory=list)


_ARRAY = re.compile(r"array_pair\(offset u32, size u32\) of (\w+)")
_COUNT = re.compile(r"i32 count of array at 0x([0-9a-f]+)")
_SCALAR = {"i32": "i", "u32": "I", "f32": "f"}


class _Reader:
    def __init__(self, data: bytes, build: str, order: str) -> None:
        self.view = ByteView(data, order)
        self.build = build
        self.areas: dict[int, int] = {}
        self.warnings: list[str] = []

    def block(self, pair: Pair, position: int, what: str) -> bytes:
        raw = self.view.pair_data(pair, position, what)
        if pair.size:
            previous = self.areas.setdefault(pair.file_offset, pair.size)
            if previous != pair.size:
                self.warnings.append(f"{what}: block at 0x{pair.file_offset:X} has two sizes")
        return raw

    def record(self, name: str, position: int, depth: int) -> Record:
        if depth > _MAX_DEPTH:
            raise FormatError(f"{name} records nested too deeply", position)
        spec = layout(self.build, name)
        raw = self.view.take(position, spec.size, f"{name} record")
        result = Record(name, position, raw, build=self.build)
        counts: list[tuple[str, int, int]] = []
        for item in spec.fields:
            at = position + item.offset
            value = self.value(item, at, depth)
            if value is _SKIP:
                continue
            match = _COUNT.match(item.type)
            if match:
                counts.append((item.name, value, int(match.group(1), 16)))
            if item.type.startswith("file_ref,"):
                result.values.update(zip(item.name.split(","), value))
            else:
                result.values[item.name] = value
        by_offset = {f.offset: f.name for f in spec.fields}
        for name_, value, target in counts:
            array = result.values.get(by_offset.get(target))
            if isinstance(array, (list, bytes)) and len(array) != value:
                self.warnings.append(f"{name}.{name_} is {value}, the array holds {len(array)}")
        return result

    def value(self, item: FieldSpec, at: int, depth: int):
        kind = item.type
        order = self.view.order
        if kind.startswith("pointer_slot"):
            return _SKIP
        if kind.startswith("file_ref,"):
            return self.view.unpack("iiI", at)
        if kind.startswith("typed_ref"):
            return self.view.unpack("ii", at)
        if kind.startswith("char["):
            return self.view.cstring(at, item.size)
        if kind == "zero":
            return self.view.take(at, item.size)
        if kind.startswith("f32["):
            return self.view.f32s(at, item.size // 4)
        if kind.startswith("text_pair"):
            raw = self.block(self.view.pair(at), at, item.name)
            return raw.split(b"\0")[0].decode("latin-1")
        if kind.startswith("tag_map_pair"):
            pair = self.view.pair(at)
            self.block(pair, at, item.name)
            return read_tag_map(self.view, pair, item.name, at, type_7=True)
        match = _ARRAY.match(kind)
        if match:
            return self.array(match.group(1), self.view.pair(at), at, item.name, depth)
        code = "f" if kind.startswith("f32") else "I" if kind.startswith("u32") else "i"
        if item.count:
            return list(self.view.unpack(f"{item.count}{code}", at))
        if item.size != 4:
            raise FormatError(f"field {item.name} of {item.size} bytes as {kind}", at)
        return self.view.unpack(code, at)[0]

    def array(self, element: str, pair: Pair, at: int, what: str, depth: int):
        raw = self.block(pair, at, what)
        if element == "u8":
            return raw
        if element in _SCALAR:
            if len(raw) % 4:
                raise FormatError(f"{what}: {len(raw)} bytes is not a list of 4-byte values", at)
            return list(ByteView(raw, self.view.order).unpack(f"{len(raw) // 4}{_SCALAR[element]}", 0))
        size = layout(self.build, element).size
        if len(raw) % size:
            raise FormatError(f"{what}: {len(raw)} bytes is not a multiple of {size}", at)
        return [
            self.record(element, pair.file_offset + i * size, depth + 1)
            for i in range(len(raw) // size)
        ]


_SKIP = object()


def _check_areas(reader: _Reader, start: int, end: int) -> list[tuple[int, int]]:
    """The data after the record must be filled exactly by the blocks, with alignment gaps."""
    areas = sorted(reader.areas.items())
    data = reader.view.data
    cursor = start
    for offset, size in areas:
        if offset < cursor:
            reader.warnings.append(f"block at 0x{offset:X} overlaps the one before it")
        elif offset - cursor >= _AREA_ALIGNMENT or any(data[cursor:offset]):
            reader.warnings.append(f"gap of {offset - cursor} bytes before 0x{offset:X}")
        cursor = max(cursor, offset + size)
    if cursor != end:
        reader.warnings.append(f"blocks end at 0x{cursor:X}, the file at 0x{end:X}")
    return areas


def parse_record_file(
    data: bytes, kind: str, build: str | None = None, strict: bool = False,
) -> RecordFile:
    """Read an ``.acr``, ``.mon``, ``.qst``, ``.cnv``, ``.cnl``, ``.trs``, ``.cnd`` or ``.rcp``.

    ``kind`` is the extension or the record name.
    """
    extension = kind.lstrip(".").lower()
    record = RECORDS_BY_EXTENSION.get(extension, kind)
    if extension not in RECORDS_BY_EXTENSION:
        extension = next((e for e, r in RECORDS_BY_EXTENSION.items() if r == record), extension)
    header = parse_header(data, build)
    resolved = resolve_build(extension, header, build)
    reader = _Reader(data, resolved, header.order)
    root = reader.record(record, 0x10, 0)
    end = 0x10 + layout(resolved, record).size
    areas = _check_areas(reader, end, len(data))
    if strict and reader.warnings:
        raise FormatError(reader.warnings[0], end)
    return RecordFile(header, resolved, root, areas, reader.warnings)


# --- Monster attributes -------------------------------------------------------------------

@dataclass(frozen=True)
class MonsterAttribute:
    """One game-balance slot of the monster modifier block."""

    index: int
    name: str  # the attribute(s) fed, ``;``-separated; empty where not known
    value_kind: str  # what is stored: always float
    status: str
    switch_offset: int | None = None
    ps3_offset: int | None = None  # None: the slot is past the ps3 block
    slot_name: str = ""
    attribute_kind: str = ""  # int or float, of the receiving attribute
    slot_status: str = ""

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(n for n in self.name.split(";") if n)


def _hex_or_none(text: str) -> int | None:
    return int(text, 16) if text else None


@lru_cache(maxsize=None)
def monster_attribute_indices() -> tuple[MonsterAttribute, ...]:
    with open(DATA / "monster_attribute_indices.tsv", encoding="utf-8", newline="") as handle:
        return tuple(
            MonsterAttribute(
                int(row["index"]), row["attribute_set"], row["value_kind"], row["status"],
                _hex_or_none(row["nx_offset"]), _hex_or_none(row["ps3_offset"]),
                row["slot_name"], row["attribute_kind"], row["slot_status"],
            )
            for row in csv.DictReader(handle, delimiter="\t")
        )


def monster_attributes(monster: Record, order: str = "<") -> dict[str, float | int]:
    """Named attribute values from a monster's modifier slots.

    Slots are always floats. An integer attribute takes the float truncated toward zero.
    A slot feeding two attributes gives both names. ``order`` is kept for old callers.
    """
    floats = monster["attribute_modifiers"]
    result: dict[str, float | int] = {}
    for attribute in monster_attribute_indices():
        if attribute.index >= len(floats) or not attribute.names:
            continue
        value = floats[attribute.index]
        if attribute.attribute_kind == "int":
            value = int(value) if math.isfinite(value) else value
        for name in attribute.names:
            result[name] = value
    return result


def monster_slots(monster: Record) -> dict[str, float]:
    """Every modifier slot by its slot name (``slot_<index>`` where none), as stored."""
    floats = monster["attribute_modifiers"]
    return {
        (a.slot_name or f"slot_{a.index}"): floats[a.index]
        for a in monster_attribute_indices() if a.index < len(floats)
    }


EXPERIENCE_SLOT = 58


def monster_grants_experience(monster: Record) -> bool:
    """Slot 58 is an experience multiplier; zero means no experience."""
    floats = monster["attribute_modifiers"]
    return len(floats) > EXPERIENCE_SLOT and floats[EXPERIENCE_SLOT] != 0


# --- ps3 attributes ------------------------------------------------------------------------

SAFE_STATUSES = ("PROVEN", "SUPPORTED")


@dataclass(frozen=True)
class Ps3Attribute:
    ps3_id: int
    name: str | None
    switch_id: int | None
    status: str
    candidates: tuple[str, ...] = ()  # UNRESOLVED rows: the names the id can be (stage 7)

    @property
    def resolved(self) -> bool:
        return self.status != "UNRESOLVED"


@lru_cache(maxsize=None)
def ps3_attributes() -> dict[int, Ps3Attribute]:
    with open(DATA / "ps3_attributes.tsv", encoding="utf-8", newline="") as handle:
        return {
            int(row["ps3_id"]): Ps3Attribute(
                int(row["ps3_id"]),
                # An UNRESOLVED id gets no single name, only its candidates.
                (row["name"] or None) if row["status"] != "UNRESOLVED" else None,
                int(row["switch_id"]) if row["switch_id"] else None, row["status"],
                tuple(c for c in (row.get("candidates") or "").split(",") if c),
            )
            for row in csv.DictReader(handle, delimiter="\t")
        }


def ps3_attribute(ps3_id: int, safe_only: bool = True) -> Ps3Attribute | None:
    """An attribute by ps3 id. With ``safe_only``, HYPOTHESIS and UNRESOLVED rows give None."""
    found = ps3_attributes().get(ps3_id)
    if found is None or (safe_only and found.status not in SAFE_STATUSES):
        return None
    return found


# --- x360 attributes (X10) -----------------------------------------------------------------

@dataclass(frozen=True)
class X360Attribute:
    """A row of ``x360_attributes.tsv``: the x360 id, its name and the Switch id of that name."""

    id: int
    name: str
    parameter_type: int
    operator: int
    data_type: int
    default_raw: int
    sync_flags: int
    formula_item: str
    formula_character: str
    switch_id: int | None


@lru_cache(maxsize=None)
def x360_attributes() -> dict[int, X360Attribute]:
    with open(DATA / "x360_attributes.tsv", encoding="utf-8", newline="") as handle:
        return {
            int(row["id"]): X360Attribute(
                int(row["id"]), row["name"], int(row["parameter_type"]), int(row["operator"]),
                int(row["data_type"]), int(row["default_raw"]), int(row["sync_flags"]),
                row["formula_item"], row["formula_character"],
                int(row["switch_id"]) if row["switch_id"] not in ("", None) else None,
            )
            for row in csv.DictReader(handle, delimiter="	")
        }


def x360_attribute(x360_id: int) -> X360Attribute | None:
    """An attribute by x360 id. x360 ids are not Switch ids (only 39 coincide)."""
    return x360_attributes().get(x360_id)


def attribute_name(attribute_id: int, build: str) -> str | None:
    """The attribute name of an id on a build: x360 by its own table, ps3 by the ps3 table
    (PROVEN and SUPPORTED rows only), Switch ids are returned as None here (no Switch table)."""
    if build == "x360":
        found = x360_attribute(attribute_id)
        return found.name if found else None
    if build == "ps3":
        found = ps3_attribute(attribute_id)
        return found.name if found else None
    return None


# --- Tag defaults --------------------------------------------------------------------------

@dataclass(frozen=True)
class TagDefault:
    table: str
    tag: int
    value_kind: str  # int, float, formula or row_ref
    raw: str
    form: str  # number, formula, name, none (an empty name) or runtime (set at start-up)
    value: object  # the number, the formula text, the name, or None


def _default_value(kind: str, text: str) -> tuple[str, object]:
    for form in ("formula", "name"):
        prefix = f'{form} "'
        if text.startswith(prefix) and text.endswith('"'):
            return form, text[len(prefix):-1]
    if text.startswith("empty name"):
        return "none", None
    if text.startswith("set at start-up"):
        return "runtime", None
    return "number", float(text) if kind == "float" else int(text)


@lru_cache(maxsize=None)
def tag_defaults() -> dict[tuple[str, int], TagDefault]:
    with open(DATA / "tag_defaults.tsv", encoding="utf-8", newline="") as handle:
        return {
            (row["table"], int(row["tag_hex"], 16)): TagDefault(
                row["table"], int(row["tag_hex"], 16), row["value_kind"], row["default_raw"],
                *_default_value(row["value_kind"], row["default_value"]),
            )
            for row in csv.DictReader(handle, delimiter="\t")
        }


def tag_default(tables: str | tuple[str, ...], tag: int) -> TagDefault | None:
    """The default of a tag in the first table, of those given, that lists it."""
    for table in (tables,) if isinstance(tables, str) else tables:
        found = tag_defaults().get((table, tag))
        if found is not None:
            return found
    return None


POWER_TABLES = ("power_general", "power_tuning")


def default_number(entry: TagDefault | None) -> float | None:
    """A default as a number. An empty name is -1. Other names, and formula text that is not
    a plain number, give None."""
    if entry is None:
        return None
    if entry.form == "number":
        return float(entry.value)
    if entry.form == "none":
        return -1.0
    if entry.form == "formula":
        try:
            return float(entry.value) if entry.value else None
        except ValueError:
            return None
    return None


class DefaultsMixin:
    """Gives ``EvalContext.tag_default`` from the tables. Mix in before the context class."""

    default_tables: tuple[str, ...] = POWER_TABLES

    def tag_default(self, tag: int) -> float:
        number = default_number(tag_default(self.default_tables, tag))
        if number is None:
            self.warnings.append(f"tag 0x{tag:X} has no numeric default, 0 used")
            return 0.0
        return number


# --- Conversations -------------------------------------------------------------------------

NODE_CONDITIONAL = 1
NODE_LINE_GROUP = 3
NODE_RANDOM = 4
NODE_LINE = 5
NODE_PARTICIPANT_CONDITIONAL = 6
LOCALE_NONE = 19
GENDER_ANY = 2


CLASS_COUNT = ByBuild({"switch": 7, "ps3": 5})
LOCALES_WITHOUT_TIMES = (0, 1, 17)


@dataclass(frozen=True)
class Locale:
    index: int
    constant: str
    code: str
    note: str


@lru_cache(maxsize=None)
def display_time_locales() -> dict[int, Locale]:
    """Locale slots of display times (``cov_display_time_locales.tsv``); 19 is no locale."""
    with open(DATA / "cov_display_time_locales.tsv", encoding="utf-8", newline="") as handle:
        return {
            int(row["index"]): Locale(int(row["index"]), row["constant"], row["code"], row["note"])
            for row in csv.DictReader(handle, delimiter="\t")
        }


@dataclass(frozen=True)
class DisplayTime:
    locale: int  # 19 means none
    player_class: int  # the class count means any
    gender: int  # 2 means any
    time: int  # ticks, 60 per second

    @property
    def seconds(self) -> float:
        return self.time / TICKS_PER_SECOND

    @property
    def locale_code(self) -> str | None:
        found = display_time_locales().get(self.locale)
        return found.code or None if found else None

    @property
    def any_gender(self) -> bool:
        return self.gender == GENDER_ANY

    def any_class(self, build: str) -> bool:
        return self.player_class == CLASS_COUNT[build]

    @property
    def carries_a_time(self) -> bool:
        """Slots 0, 1 and 17 never carry a real time."""
        return self.locale not in LOCALES_WITHOUT_TIMES


def decode_display_time(word: int) -> DisplayTime:
    word &= 0xFFFFFFFF
    return DisplayTime(word & 31, word >> 5 & 7, word >> 8 & 3, word >> 10)


def walk_nodes(nodes: list[Record]):
    """Every node of a conversation tree, depth first."""
    for node in nodes:
        yield node
        for branch in ("true_nodes", "false_nodes", "children"):
            yield from walk_nodes(node.get(branch) or [])


def line_groups(conversation: Record) -> list[tuple[int, list[Record]]]:
    """Each kind-3 node's line uid with its kind-5 children."""
    return [
        (node["line_uid"], [c for c in node.get("children") or [] if c["node_kind"] == NODE_LINE])
        for node in walk_nodes(conversation["root_nodes"])
        if node["node_kind"] == NODE_LINE_GROUP
    ]


# --- Conditions ----------------------------------------------------------------------------

@dataclass
class Player:
    gender: int  # 0 male, 1 female
    player_class: int
    level: int = 0
    difficulty: int | None = None  # ps3 difficulty index


@dataclass
class ConditionResult:
    passed: bool
    failed: list[str]
    unchecked: list[str]  # set tests that need game state


_CHECKED = {
    "male", "female", "classes", "difficulties", "player_level_min", "player_level_max",
    "self_handle", "lock_count", "header_flags",
}


def _is_set(name: str, value) -> bool:
    if isinstance(value, str):
        return bool(value)
    if isinstance(value, (list, tuple, bytes)):
        return False
    if name.endswith(("_min", "_max")) and value == 0 and name.startswith("player_level"):
        return False
    return value not in (-1, 0)


def evaluate_condition(condition: Record, player: Player) -> ConditionResult:
    """Every set test must pass. Gender, class, difficulty and player level are checked here."""
    failed = []
    if condition.get("male") and player.gender == 1:
        failed.append("male")
    if condition.get("female") and player.gender == 0:
        failed.append("female")
    classes = condition.get("classes") or []
    if any(classes) and not (0 <= player.player_class < len(classes) and classes[player.player_class]):
        failed.append("classes")
    difficulties = condition.get("difficulties") or []
    if any(difficulties) and player.difficulty is not None:
        if not (0 <= player.difficulty < len(difficulties) and difficulties[player.difficulty]):
            failed.append("difficulties")
    low, high = condition.get("player_level_min", 0), condition.get("player_level_max", 0)
    if low > 0 and player.level < low:
        failed.append("player_level_min")
    if high > 0 and player.level > high:
        failed.append("player_level_max")
    unchecked = [
        name for name, value in condition.values.items()
        if name not in _CHECKED and _is_set(name, value)
    ]
    return ConditionResult(not failed, failed, unchecked)


# --- Treasure ------------------------------------------------------------------------------

DROP_KINDS = {
    0: "nested_treasure_class", 1: "random_item_by_label", 2: "specific_item",
    3: "targeted_legendary", 4: "class_legendary",
}


@dataclass(frozen=True)
class ChooseRule:
    kind: int
    mode: str  # one_of (one modifier by weight), all_of (each by its chance) or none
    shared: bool  # uses the shared multiplayer loot path


_CHOOSE_RULES = {
    -1: ("none", False), 0: ("one_of", False), 1: ("all_of", False),
    2: ("all_of", True), 3: ("one_of", True),
}


def choose_rule(choose_kind: int) -> ChooseRule:
    """How a treasure class picks: 0 one of by weight, 1 roll each, 2 and 3 shared, -1 none."""
    found = _CHOOSE_RULES.get(choose_kind)
    if found is None:
        raise FormatError(f"choose kind {choose_kind}")
    return ChooseRule(choose_kind, *found)


def rolls_every_modifier(choose_kind: int) -> bool:
    """Kinds 1 and 2 roll every modifier by its chance; 0 and 3 pick one by weight."""
    return choose_rule(choose_kind).mode == "all_of"


# --- Objectives ----------------------------------------------------------------------------

@dataclass(frozen=True)
class ObjectiveParams:
    record: str
    kind: int
    kind_name: str
    target_1_group: int | None  # the file group the first target names
    target_1_meaning: str | None
    text_1: str | None  # what the first text names: a trigger, group, flag or event name
    row_param_1: str | None
    params_used: str
    status: str


@lru_cache(maxsize=None)
def objective_params() -> dict[tuple[str, int], ObjectiveParams]:
    """Meaning of the generic objective and failure parameters by kind."""
    def text(value: str) -> str | None:
        return None if value in ("", "-", "no") else value

    found = {}
    with open(DATA / "cov_objective_params.tsv", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            group = row["target_1_group"]
            entry = ObjectiveParams(
                row["record"], int(row["kind"]), row["kind_name"],
                int(group) if group not in ("", "-") else None, text(row["target_1_meaning"]),
                text(row["text_1"]), text(row["row_param_1"]), row["params_used"], row["status"],
            )
            found[(entry.record, entry.kind)] = entry
    return found


def objective_meaning(record: Record) -> ObjectiveParams | None:
    """The parameter meaning of an ``objective`` or ``failure_condition`` record."""
    kind = record.get("objective_kind", record.get("failure_kind"))
    return objective_params().get((record.type, kind)) if isinstance(kind, int) else None


def objective_count(record: Record) -> int:
    """The second integer parameter is a count."""
    return record["param_2"]


def live_affixes(entry: Record) -> list[int]:
    """Affix slots below the affix count. Later slots are junk."""
    return list(entry["affixes"][:max(entry["affix_count"], 0)])


# --- Text labels ---------------------------------------------------------------------------

RUNE_LETTERS = "ABCDE"


def label_hash(label: str) -> int:
    return hash33(label)


def buff_label(power: str, slot: int, description: bool = False) -> str:
    return f"{power}_{slot}" + ("_desc" if description else "")


def attribute_label(attribute: str, parameter: str | None = None) -> str:
    return attribute if parameter is None else f"{attribute}#{parameter}"


def rune_labels(power: str, letter: str) -> dict[str, str]:
    if letter not in RUNE_LETTERS:
        raise FormatError(f"rune letter {letter!r}")
    return {
        "name": f"NameRune_{letter}#{power}",
        "short": f"Simple_Rune_{letter}#{power}",
        "full": f"Rune_{letter}#{power}",
    }


def quest_labels(uid: int, player_class: str = "") -> dict[str, str]:
    return {
        "name": "Name", "splash": f"Step{uid}Splash", "meter": f"Step{uid}Meter",
        "description": f"Step{uid}Desc_{player_class}", "objective": f"Objective{uid}Desc",
    }


# --- Text marks ----------------------------------------------------------------------------

_MARK = re.compile(
    r"\|4([^;|]*);|\|5([^;|]*);|\|3-(\d)\(([^)]*)\)|\[([^\]|]*)\|(\+?)(\d*)(\+?)\|\]",
)
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def default_plural(number: float, forms: int) -> int:
    """The form index for a number. One form for 1, the next for all else."""
    return 0 if number == 1 or forms == 1 else 1


def render_marks(
    text: str, gender: int = 0, value=None, declension: dict | None = None, plural=default_plural,
) -> str:
    """Resolve text marks.

    ``value(expr)`` gives the number of a ``[expr|N|]`` mark. ``declension`` maps
    (word, case) to a form. ``plural(number, form count)`` picks a plural form.
    """
    out: list[str] = []
    pos = 0
    for match in _MARK.finditer(text):
        out.append(text[pos:match.start()])
        pos = match.end()
        plural_forms, gender_forms, case, word, expr, sign_a, decimals, sign_b = match.groups()
        if plural_forms is not None:
            numbers = _NUMBER.findall("".join(out))
            forms = plural_forms.split(":")
            number = float(numbers[-1]) if numbers else 0.0
            out.append(forms[min(plural(number, len(forms)), len(forms) - 1)])
        elif gender_forms is not None:
            forms = gender_forms.split(":")
            out.append(forms[min(1 if gender == 1 else 0, len(forms) - 1)])
        elif case is not None:
            out.append((declension or {}).get((word, int(case)), word))
        else:
            if value is None:
                raise FormatError(f"no value source for [{expr}]")
            number = value(expr)
            spec = f"{'+' if sign_a or sign_b else ''}.{int(decimals or 0)}f"
            out.append(format(number, spec))
    out.append(text[pos:])
    return "".join(out)


# --- Script listeners ----------------------------------------------------------------------

@dataclass(frozen=True)
class ListenerRules:
    attachment_arguments: bool  # handler gets attachment id and kind first
    fire_once: bool
    fire_after_count: bool


ATTACHMENT_KINDS = ("Actor", "Quest", "Scene", "World", "Power")


def listener_rules(flags: int) -> ListenerRules:
    return ListenerRules(not flags & 1, bool(flags & 2), bool(flags & 4))


def handler_arguments(flags: int, attachment_id: int, attachment_kind: str, *event_args) -> tuple:
    if attachment_kind not in ATTACHMENT_KINDS:
        raise FormatError(f"attachment kind {attachment_kind!r}")
    if listener_rules(flags).attachment_arguments:
        return (attachment_id, attachment_kind, *event_args)
    return tuple(event_args)


def ticks_to_seconds(ticks: int) -> float:
    return ticks / TICKS_PER_SECOND
