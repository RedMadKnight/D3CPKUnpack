"""Tag names, the record type each tag must use, and value names (``cov_tags.tsv``)."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache

from .datafiles import DATA, table_rows
from .enums import enum_name
from .power import TagMap

# The table that names the tags of a map depends on the field holding the map.
POWER_MAP_TABLES = {
    "Effect": None, "General": "power_general", "PvpGeneral": "power_general",
    **{f"Contact{i}": "power_tuning" for i in range(4)},
    **{f"PvpContact{i}": "power_tuning" for i in range(4)},
}
OWNER_TABLES = {"actor": "actor", "monster": "monster", "anim": "anim", "marker": "marker",
                "material": "appearance"}
# ps3 tags that were integers there and later became formulas: trust the stored type.
PS3_RETYPED_TAGS = frozenset({0x50210, 0x50230, 0x502D0, 0x502F0, 0x50335})
# x360 (X8): four power tags are integers (record type 0); trust the stored type, as on ps3.
# 0x502D0 is a formula on x360 too.
X360_RETYPED_TAGS = frozenset({0x50210, 0x50230, 0x502F0, 0x50335})
RETYPED_TAGS = {"ps3": PS3_RETYPED_TAGS, "x360": X360_RETYPED_TAGS}
# ps3-only ids with no name in any build (G3-D). Since stage 10 they have rows without a name
# (``s10_tags.tsv``) and are read by those record types; on Switch and x360 they are a problem.
PS3_UNNAMED_TAGS = {"power_general": {0x50281}, "marker": {0x88010, 0x88020}}


@dataclass(frozen=True)
class TagInfo:
    table: str
    tag: int
    name: str
    xml_name: str
    value_type: str
    record_type: int | None  # None means no record type is allowed
    record_kind: str
    sno_group: int | None
    value_enum: str | None
    default_raw: str
    group: str


@lru_cache(maxsize=None)
def tag_table() -> dict[tuple[str, int], TagInfo]:
    result = {}
    for row in table_rows("cov_tags.tsv"):
        info = TagInfo(
            row["table"], int(row["tag_hex"], 16), row["name"], row["xml_name"],
            row["axe_type"], int(row["record_type"]) if row["record_type"] else None,
            row["record_kind"], int(row["sno_group"]) if row["sno_group"] else None,
            row["value_enum"] or None, row["default_raw"], row["tag_group"],
        )
        result[(info.table, info.tag)] = info
    return result


@lru_cache(maxsize=None)
def tag_types() -> dict[str, dict[str, str]]:
    with open(DATA / "cov_tag_types.tsv", encoding="utf-8", newline="") as handle:
        return {row["axe_type"]: row for row in csv.DictReader(handle, delimiter="\t")}


def tag_info(table: str | None, tag: int) -> TagInfo | None:
    return None if table is None else tag_table().get((table, tag))


def tag_name(table: str | None, tag: int) -> str | None:
    info = tag_info(table, tag)
    return None if info is None else info.name or None  # G3-D rows have no name


def value_name(table: str | None, tag: int, value: int) -> str | None:
    """The name of an enum-valued tag's value, from its ``tag_value:`` table."""
    info = tag_info(table, tag)
    if info is None or info.value_enum is None:
        return None
    enum = info.value_enum
    if not enum.startswith("tag_value:"):
        enum = "tag_value:" + enum
    return enum_name(enum, value)


@dataclass(frozen=True)
class NamedRecord:
    """A tag-map record with its names. Raw values stay on ``record``."""

    record: object  # power.TagRecord
    name: str | None
    xml_name: str | None
    value_name: str | None
    type_name: str | None


def name_records(tag_map: TagMap, table: str | None) -> list[NamedRecord]:
    """Name every record of a map from ``table`` (see ``POWER_MAP_TABLES``, ``OWNER_TABLES``)."""
    found = []
    for record in tag_map.records:
        info = tag_info(table, record.tag)
        named_value = None
        if info is not None and isinstance(record.value, int):
            named_value = value_name(table, record.tag, record.value)
        found.append(NamedRecord(
            record, (info.name or None) if info else None, (info.xml_name or None) if info else None,
            named_value,
            enum_name("tag_record_type", record.type),
        ))
    return found


def check_tag_map(tag_map: TagMap, table: str | None, build: str) -> list[str]:
    """Problems with a map's tags: unknown ids and stored types the tag does not allow."""
    problems = []
    if table is None:
        if tag_map.records:
            problems.append(f"map {tag_map.name} has no tag table and should be empty")
        return problems
    unnamed = PS3_UNNAMED_TAGS.get(table, set())
    for record in tag_map.records:
        if record.tag in unnamed and build != "ps3":
            problems.append(f"tag 0x{record.tag:X} occurs only in ps3 data (G3-D)")
            continue
        info = tag_info(table, record.tag)
        if info is None:
            problems.append(f"tag 0x{record.tag:X} is not in table {table}")
            continue
        if info.record_type is None:
            problems.append(f"tag 0x{record.tag:X} ({info.name}) may not be stored")
        elif record.type != info.record_type:
            if record.tag in RETYPED_TAGS.get(build, ()):
                continue
            problems.append(
                f"tag 0x{record.tag:X} ({info.name}) stored as type {record.type}, "
                f"declared {info.record_type}"
            )
    return problems


def power_map_problems(power_file) -> list[str]:
    """``check_tag_map`` over all eleven maps of a parsed ``.pow``."""
    problems = []
    for name, tag_map in power_file.maps.items():
        problems.extend(check_tag_map(tag_map, POWER_MAP_TABLES[name], power_file.build))
    return problems
