"""``.pow`` files: fixed fields, tag maps and the script slice."""

from __future__ import annotations

from dataclasses import dataclass, field

from .binary import ByteView, FormatError, Pair, build_order
from .luachunk import script_slice
from .sno import SnoHeader, parse_header, resolve_build

TAG_INT = 0
TAG_FLOAT = 1
TAG_SNO = 2
TAG_BALANCE = 3
TAG_FORMULA = 4
TAG_ACTOR_GROUP = 5
TAG_ATTRIBUTE = 6
TAG_START_LOCATION = 7
_WIDE_4 = (TAG_INT, TAG_FLOAT, TAG_SNO, TAG_BALANCE, TAG_ACTOR_GROUP, TAG_ATTRIBUTE)

POWER_BIT_NAMES = (
    "passable_while_active", "passable_at_destination", "contact_frame_first",
    "contact_frame_second", "contact_frame_third", "contact_frame_last",
    "needs_walking_path", "moves_forward_without_target", "target_area_grows",
    "walk_between_casts", "radial_targeting", "cone_targeting", "secondary_targeting",
    "prefers_radial", "deprecated", "fires_on_button_up", "target_range_grows",
)

MAP_POSITIONS = [("Effect", 0x20), ("General", 0x30), ("PvpGeneral", 0x40)]
MAP_POSITIONS += [(f"Contact{i}", 0x50 + i * 8) for i in range(4)]
MAP_POSITIONS += [(f"PvpContact{i}", 0x90 + i * 8) for i in range(4)]

BUFF_FILTERS_AT = 0xF8
_FILTER_SLOTS = {"ps3": 8, "switch": 32}
TRAILER_AT = {"ps3": 0x178, "switch": 0x2F8}
_MAX_SLOT = {"ps3": 39, "switch": 63}


@dataclass(frozen=True)
class Formula:
    text: str
    code: bytes


@dataclass(frozen=True)
class TagRecord:
    type: int
    tag: int
    value: int | float | Formula
    offset: int


@dataclass
class TagMap:
    name: str
    pair: Pair
    records: list[TagRecord] = field(default_factory=list)
    raw: bytes = b""
    shares_with: str | None = None  # the earlier map whose block this one uses
    error: str | None = None  # lenient reading: why the records stop early

    def get(self, tag: int) -> TagRecord | None:
        return next((r for r in self.records if r.tag == tag), None)


@dataclass(frozen=True)
class CollisionRecord:
    required_static: int
    prevent_static: int
    dynamic_no_overlap: int
    dynamic_self: int


@dataclass
class BuffFilter:
    pair: Pair
    handles: list[int]
    extra: list[int]  # switch: the second u32 of each entry; empty on ps3
    raw: bytes = b""


@dataclass
class PowerFile:
    header: SnoHeader
    build: str
    maps: dict[str, TagMap]
    power_bits: int
    collision: tuple[CollisionRecord, CollisionRecord]
    word_0f4: int
    buff_filters: list[BuffFilter]
    scripted: bool
    uses_secondary_resource: bool
    formula_count: int
    word_after_count: int
    script_pair: Pair
    quest_handle: int
    padding_end: int  # zero in every file
    script: bytes
    fixed: bytes = b""

    @property
    def general(self) -> TagMap:
        return self.maps["General"]

    @property
    def formula_records(self) -> list[TagRecord]:
        """Every type-4 record in General, live or dead."""
        return [r for r in self.general.records if r.type == TAG_FORMULA]

    def power_bit_names(self) -> list[str]:
        return [n for i, n in enumerate(POWER_BIT_NAMES) if self.power_bits >> i & 1]

    def is_live_slot(self, slot: int) -> bool:
        return slot < self.formula_count

    def slot_record(self, slot: int) -> TagRecord | None:
        return self.general.get(formula_tag(slot, self.build))


def formula_tag(slot: int, build: str) -> int:
    """Tag of script formula slot ``slot``."""
    limit = _MAX_SLOT[build]
    if not 0 <= slot <= limit:
        raise FormatError(f"formula slot {slot} outside 0 to {limit} on {build}")
    return 0x41100 + (slot // 10) * 0x100 + (slot % 10) * 0x10


def _read_formula(view: ByteView, pos: int, end: int) -> tuple[Formula, int]:
    if pos + 32 > end:
        raise FormatError("formula head overruns the map", pos)
    head = view.unpack("8I", pos, "formula head")
    for i in (0, 1, 2, 3, 4, 6):
        if head[i] != 0:
            raise FormatError(f"formula zero word {i} is 0x{head[i]:X}", pos + i * 4)
    text_length, code_length = head[5], head[7]
    if text_length == 0:
        raise FormatError("formula text length omits the NUL", pos + 20)
    if code_length % 4:
        raise FormatError(f"code length {code_length} is not a multiple of 4", pos + 28)
    text_at = pos + 32
    padded = (text_length + 3) & ~3
    code_at = text_at + padded
    if code_at + code_length > end:
        raise FormatError("formula overruns the map", pos)
    text = view.take(text_at, text_length)
    if text[-1] != 0:
        raise FormatError("formula text lacks its NUL", text_at + text_length - 1)
    if any(view.take(text_at + text_length, padded - text_length)):
        raise FormatError("formula text padding is not zero", text_at + text_length)
    formula = Formula(text[:-1].decode("latin-1"), view.take(code_at, code_length))
    return formula, code_at + code_length


def read_tag_map(
    view: ByteView, pair: Pair, name: str = "", position: int = 0, lenient: bool = False,
) -> TagMap:
    """Read a map bounded by its pair. Its declared size must be consumed exactly.

    ``lenient`` is for marker maps: type 7 is then a 4-byte value, and a record that cannot be
    read ends the list in ``error`` without hiding the records before it.
    """
    result = TagMap(name, pair)
    if pair.size == 0:
        return result
    result.raw = view.pair_data(pair, position, f"tag map {name}")
    if not lenient:
        _read_records(view, pair, name, result, False)
        return result
    try:
        _read_records(view, pair, name, result, True)
    except FormatError as exc:
        result.error = exc.message
    return result


def _read_records(view: ByteView, pair: Pair, name: str, result: TagMap, lenient: bool) -> None:
    pos = pair.file_offset
    end = pos + pair.size
    if pair.size < 4:
        raise FormatError(f"tag map {name} too short for its count", pos)
    count = view.u32(pos)
    pos += 4
    seen = set()
    while pos < end:
        if pos + 8 > end:
            raise FormatError(f"record overruns tag map {name}", pos)
        start = pos
        kind, tag = view.unpack("II", pos)
        pos += 8
        if kind == TAG_FORMULA:
            value, pos = _read_formula(view, pos, end)
        elif kind in _WIDE_4 or (lenient and kind == TAG_START_LOCATION):
            if pos + 4 > end:
                raise FormatError(f"record overruns tag map {name}", start)
            value = view.f32(pos) if kind == TAG_FLOAT else view.i32(pos)
            pos += 4
        else:
            raise FormatError(f"tag type {kind} rejected", start)
        if tag in seen:
            raise FormatError(f"tag 0x{tag:X} repeats in map {name}", start)
        seen.add(tag)
        result.records.append(TagRecord(kind, tag, value, start))
    if len(result.records) != count:
        raise FormatError(
            f"tag map {name} declares {count} records, holds {len(result.records)}",
            pair.file_offset,
        )


def _read_buff_filter(view: ByteView, pair: Pair, build: str, position: int) -> BuffFilter:
    raw = view.pair_data(pair, position, "buff filter")
    width = 4 if build == "ps3" else 8
    if len(raw) % width:
        raise FormatError(f"buff filter size {len(raw)} not a multiple of {width}", position)
    words = ByteView(raw, view.order).unpack(f"{len(raw) // 4}I", 0) if raw else ()
    if build == "ps3":
        return BuffFilter(pair, list(words), [], raw)
    return BuffFilter(pair, list(words[0::2]), list(words[1::2]), raw)


def parse_power(data: bytes, build: str | None = None, strict: bool = False) -> PowerFile:
    """Read a ``.pow``. Without ``build`` the version chooses it."""
    header = parse_header(data, build)
    resolved = resolve_build("pow", header, build, strict)
    view = ByteView(data, build_order(resolved))
    trailer = TRAILER_AT[resolved]
    view.need(0, trailer + 0x28, "power fixed part")

    maps = {
        name: read_tag_map(view, view.pair(pos), name, pos) for name, pos in MAP_POSITIONS
    }
    first_user: dict[Pair, str] = {}
    for tag_map in maps.values():
        if tag_map.pair.size:
            root = first_user.setdefault(tag_map.pair, tag_map.name)
            tag_map.shares_with = None if root == tag_map.name else root
    words = view.unpack("9I", 0xD4)
    collision = (CollisionRecord(*words[:4]), CollisionRecord(*words[4:8]))
    filters = []
    for i in range(_FILTER_SLOTS[resolved]):
        pos = BUFF_FILTERS_AT + i * 16
        filters.append(_read_buff_filter(view, view.pair(pos), resolved, pos))

    script_pair = view.pair(trailer + 0x18)
    script = script_slice(data, "pow", resolved)
    if script and script_pair.file_offset + script_pair.size != len(data):
        raise FormatError("script chunk does not end the file", trailer + 0x18)
    return PowerFile(
        header, resolved, maps, view.u32(0xD0), collision, words[8], filters,
        bool(view.u32(trailer)), bool(view.u32(trailer + 4)), view.i32(trailer + 8),
        view.u32(trailer + 0xC), script_pair, view.i32(trailer + 0x20),
        view.i32(trailer + 0x24), script, view.take(0, trailer + 0x28),
    )
