"""``.skl`` skill kits, ``.tme`` timed events and ``.lor`` lore."""

from __future__ import annotations

from dataclasses import dataclass

from .binary import ByBuild, ByteView, FormatError
from .enums import enum_name
from .sno import SnoHeader, parse_header, resolve_build

_TRAIT_SIZE = ByBuild({"switch": 16, "ps3": 12})
_ACTIVE_SIZE = ByBuild({"switch": 64, "ps3": 40})
_FIXED = ByBuild({"switch": 0x50, "ps3": 0x40})
LORE_FIXED = 0x34
TIMED_EVENT_FIXED = 0x20


@dataclass(frozen=True)
class Trait:
    power: int
    hidden: int
    unlock_level: int
    new: int | None = None  # switch only


@dataclass(frozen=True)
class ActiveSkill:
    power: int
    category: int
    new_category: int
    unlock_level: int
    granted: int
    rune_unlock_levels: tuple[int, ...]
    new: int | None = None  # switch only
    rune_new: tuple[int, ...] | None = None  # switch only

    @property
    def category_name(self) -> str | None:
        return enum_name("skill_category", self.category)

    @property
    def new_category_name(self) -> str | None:
        return enum_name("skill_category", self.new_category)


@dataclass
class SkillKit:
    header: SnoHeader
    build: str
    traits: list[Trait]
    actives: list[ActiveSkill]
    power_fallbacks: list[int]  # switch only: Power handles


def _records(view: ByteView, position: int, size: int, what: str) -> list[bytes]:
    raw = view.pair_data(view.pair(position), position, what)
    if len(raw) % size:
        raise FormatError(f"{what}: {len(raw)} bytes is not a multiple of {size}", position)
    return [raw[i:i + size] for i in range(0, len(raw), size)]


def parse_skill_kit(data: bytes, build: str | None = None) -> SkillKit:
    header = parse_header(data, build)
    resolved = resolve_build("skl", header, build)
    view = ByteView(data, header.order)
    view.need(0, 0x10 + _FIXED[resolved] - 0x10, "skill kit fixed part")
    switch = resolved != "ps3"
    traits = []
    for raw in _records(view, 0x28, _TRAIT_SIZE[resolved], "traits"):
        words = ByteView(raw, view.order).unpack("4i" if switch else "3i", 0)
        traits.append(Trait(*words))
    actives = []
    for raw in _records(view, 0x38, _ACTIVE_SIZE[resolved], "actives"):
        r = ByteView(raw, view.order)
        fields = r.unpack("10i", 0)
        extra = r.unpack("6i", 40) if switch else None
        actives.append(ActiveSkill(
            *fields[:5], fields[5:], extra[0] if extra else None, extra[1:] if extra else None,
        ))
    fallbacks = []
    if switch:
        raw = view.pair_data(view.pair(0x48), 0x48, "power fallbacks")
        if len(raw) % 4:
            raise FormatError(f"power fallbacks of {len(raw)} bytes", 0x48)
        fallbacks = list(ByteView(raw, view.order).unpack(f"{len(raw) // 4}i", 0))
    return SkillKit(header, resolved, traits, actives, fallbacks)


@dataclass(frozen=True)
class TimedEvent:
    header: SnoHeader
    build: str
    duration_ticks: int  # 60 ticks per second


def parse_timed_event(data: bytes, build: str | None = None) -> TimedEvent:
    header = parse_header(data, build)
    view = ByteView(data, header.order)
    view.need(0, TIMED_EVENT_FIXED, "timed event")
    return TimedEvent(header, resolve_build("tme", header, build), view.u32(0x1C, "duration"))


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

    @property
    def category_name(self) -> str | None:
        return enum_name("lore_category", self.category)


def parse_lore(data: bytes, build: str | None = None) -> Lore:
    header = parse_header(data, build)
    view = ByteView(data, header.order)
    view.need(0, LORE_FIXED, "lore")
    fields = view.unpack("6I", 0x1C, "lore fields")
    return Lore(header, resolve_build("lor", header, build), *fields)
