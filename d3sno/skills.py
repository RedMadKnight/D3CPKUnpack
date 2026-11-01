"""``.skl`` skill kits, ``.tme`` timed events and ``.lor`` lore."""

from __future__ import annotations

from dataclasses import dataclass

from .binary import ByteView, FormatError
from .sno import SnoHeader, parse_header

_TRAIT_SIZE = {"switch": 16, "ps3": 12}
_ACTIVE_SIZE = {"switch": 64, "ps3": 40}


@dataclass(frozen=True)
class Trait:
    power: int
    hidden: int
    unlock_level: int


@dataclass(frozen=True)
class ActiveSkill:
    power: int
    category: int
    new_category: int
    unlock_level: int
    granted: int
    rune_unlock_levels: tuple[int, ...]


@dataclass
class SkillKit:
    header: SnoHeader
    build: str
    traits: list[Trait]
    actives: list[ActiveSkill]


def _records(view: ByteView, position: int, size: int, what: str) -> list[bytes]:
    raw = view.pair_data(view.pair(position), position, what)
    if len(raw) % size:
        raise FormatError(f"{what}: {len(raw)} bytes is not a multiple of {size}", position)
    return [raw[i:i + size] for i in range(0, len(raw), size)]


def parse_skill_kit(data: bytes, build: str | None = None) -> SkillKit:
    header = parse_header(data, build)
    view = ByteView(data, header.order)
    traits = [
        Trait(*ByteView(raw, view.order).unpack("iii", 0))
        for raw in _records(view, 0x28, _TRAIT_SIZE[header.build], "traits")
    ]
    actives = []
    for raw in _records(view, 0x38, _ACTIVE_SIZE[header.build], "actives"):
        fields = ByteView(raw, view.order).unpack("10i", 0)
        actives.append(ActiveSkill(*fields[:5], fields[5:]))
    return SkillKit(header, header.build, traits, actives)


@dataclass(frozen=True)
class TimedEvent:
    header: SnoHeader
    build: str
    duration_ticks: int  # the tick rate is open


def parse_timed_event(data: bytes, build: str | None = None) -> TimedEvent:
    header = parse_header(data, build)
    return TimedEvent(header, header.build, ByteView(data, header.order).u32(0x1C, "duration"))


@dataclass(frozen=True)
class Lore:
    header: SnoHeader
    build: str
    flags: int
    category: int
    xp_level: int
    xp_tier: int
    conversation: int
    signature_label_hash: int


def parse_lore(data: bytes, build: str | None = None) -> Lore:
    header = parse_header(data, build)
    fields = ByteView(data, header.order).unpack("6I", 0x1C, "lore fields")
    return Lore(header, header.build, *fields)
