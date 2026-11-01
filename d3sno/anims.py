"""``.ans`` anim sets."""

from __future__ import annotations

from dataclasses import dataclass, field

from .binary import ByteView
from .power import TAG_SNO, TagMap, read_tag_map
from .sno import SnoHeader, parse_header

_WEAPON_SLOTS = {"switch": 28, "ps3": 19}


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
    view = ByteView(data, header.order)
    default = read_tag_map(view, view.pair(0x20), "default", 0x20)
    weapons = []
    for i in range(_WEAPON_SLOTS[header.build]):
        position = 0x30 + 16 * i
        weapons.append(read_tag_map(view, view.pair(position), f"weapon{i}", position))
    result = AnimSet(header, header.build, view.i32(0x1C), default, weapons)
    for tag_map in (default, *weapons):
        if any(record.type != TAG_SNO for record in tag_map.records):
            result.warnings.append(f"map {tag_map.name} holds a record of another type")
    return result
