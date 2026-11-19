"""``.ans`` anim sets."""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

from .binary import ByBuild, ByteView, FormatError, build_order
from .power import TAG_SNO, TagMap, read_tag_map
from .sno import NONE, SnoHeader, parse_header, resolve_build

_WEAPON_SLOTS = ByBuild({"switch": 28, "ps3": 19})
WEAPON_SLOTS = _WEAPON_SLOTS  # W15-B3a: map count 19 (ps3) or 28 (switch, x360)
_DEFAULT_SLOT = 0x10  # body offset of the default map's pair; weapon map i at 0x20 + 16 i


@dataclass
class AnimSet:
    header: SnoHeader
    build: str
    parent: int
    default: TagMap
    weapons: list[TagMap]
    warnings: list[str] = field(default_factory=list)

    def animation(self, tag: int, weapon: int | None = None) -> int | None:
        """The ``.ani`` handle for a purpose tag, in one weapon map or the default map."""
        tag_map = self.default if weapon is None else self.weapons[weapon]
        record = tag_map.get(tag)
        return None if record is None else record.value


def parse_anim_set(data: bytes, build: str | None = None) -> AnimSet:
    header = parse_header(data, build)
    resolved = resolve_build("ans", header, build)
    view = ByteView(data, header.order)
    default = read_tag_map(view, view.pair(0x20), "default", 0x20, type_7=True)
    weapons = []
    for i in range(_WEAPON_SLOTS[resolved]):
        position = 0x30 + 16 * i
        weapons.append(
            read_tag_map(view, view.pair(position), f"weapon{i}", position, type_7=True),
        )
    result = AnimSet(header, resolved, view.i32(0x1C), default, weapons)
    for tag_map in (default, *weapons):
        if any(record.type != TAG_SNO for record in tag_map.records):
            result.warnings.append(f"map {tag_map.name} holds a record of another type")
    return result


# --- W15-B3a ------------------------------------------------------------------------------------

def anim_set_problems(anim_set: AnimSet, animations=None, anim_sets=None) -> list[str]:
    """W15-B3a: every record of every map is type 2 and (with ``animations``, a set of ``.ani``
    handles) names an existing animation; tags are unique within a map; the map's count word
    equals its records; 19 (ps3) or 28 (switch, x360) weapon maps; the parent word is none (-1)
    or (with ``anim_sets``, a set of ``.ans`` handles) another anim set."""
    found = []
    slots = _WEAPON_SLOTS[anim_set.build]
    if len(anim_set.weapons) != slots:
        found.append(f"{len(anim_set.weapons)} weapon maps, {anim_set.build} has {slots} (W15-B3a)")
    order = build_order(anim_set.build)
    for tag_map in (anim_set.default, *anim_set.weapons):
        seen = set()
        for record in tag_map.records:
            if record.type != TAG_SNO:
                found.append(f"map {tag_map.name}: tag 0x{record.tag:X} has type {record.type} (W15-B3a)")
            elif animations is not None and record.value not in animations:
                found.append(f"map {tag_map.name}: tag 0x{record.tag:X} names no animation "
                             f"({record.value}) (W15-B3a)")
            if record.tag in seen:
                found.append(f"map {tag_map.name}: tag 0x{record.tag:X} repeats (W15-B3a)")
            seen.add(record.tag)
        if tag_map.raw:
            count = struct.unpack_from(order + "I", tag_map.raw, 0)[0]
            if count != len(tag_map.records):
                found.append(f"map {tag_map.name}: count {count} for {len(tag_map.records)} records")
    parent = anim_set.parent
    if parent != NONE:
        if parent == anim_set.header.handle:
            found.append("the parent is the anim set itself (W15-B3a)")
        elif anim_sets is not None and parent not in anim_sets:
            found.append(f"parent {parent} is not an anim set (W15-B3a)")
    return found


def anim_tag_map_bytes(records, build: str) -> bytes:
    """A tag map of type-2 records ``(tag, animation)`` in the given order: count, then
    {type, tag, value} each."""
    order = build_order(build)
    rows = list(records)
    return struct.pack(order + "I", len(rows)) + b"".join(
        struct.pack(order + "IIi", TAG_SNO, tag, value) for tag, value in rows)


def append_anim_tag(data: bytes, tag: int, animation: int, weapon: int | None = None,
                    build: str | None = None) -> bytes:
    """W15-B3a: the ``.ans`` with one type-2 record appended to the default map (``weapon``
    None) or to weapon map ``weapon``; the stored record order is kept and the count word is
    raised by one. The tag must be new in that map. The generic writer lays the file out."""
    from .sno_layout import Block, PairValue, parse_tree, write_tree

    anim_set = parse_anim_set(data, build)
    slots = _WEAPON_SLOTS[anim_set.build]
    if weapon is None:
        tag_map, slot = anim_set.default, _DEFAULT_SLOT
    elif 0 <= weapon < slots:
        tag_map, slot = anim_set.weapons[weapon], 0x20 + 16 * weapon
    else:
        raise FormatError(f"weapon map {weapon}: {anim_set.build} has 0 to {slots - 1}")
    if tag_map.get(tag) is not None:
        raise FormatError(f"tag 0x{tag:X} is already in map {tag_map.name} (W15-B3a)")
    order = build_order(anim_set.build)
    old = tag_map.raw
    count = struct.unpack_from(order + "I", old, 0)[0] if old else 0
    new = struct.pack(order + "I", count + 1) + old[4:] + struct.pack(
        order + "IIi", TAG_SNO, tag, animation)
    tree = parse_tree(data, "ans", anim_set.build)
    tree.top.pairs[slot] = PairValue(Block(new))
    return write_tree(tree)
