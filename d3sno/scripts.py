"""``.scr`` files: fixed part and message map."""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

from .binary import ByteView, FormatError
from .luachunk import script_slice
from .sno import SnoHeader, hash33_ci, parse_header, resolve_build

_ENTRY = {"switch": 192, "ps3": 176}
_FILTER_SIZE = {"switch": 24, "ps3": 20}
_FILTER_AT = 8
_STRINGS_AT = {"switch": 0x68, "ps3": 0x58}
_HANDLER_AT = {"switch": 0xA8, "ps3": 0x98}
_FLAGS_AT = {"switch": 0xB8, "ps3": 0xA8}
FILTER_COUNT = 4
_NO_VALUE_KINDS = (0, 12, 20, 25)
_FLOAT_KINDS = (2,)
_RANGE_KINDS = (3, 22)
STRING_HASH_KINDS = (7, 15, 21, 24)
# Kind: the SNO group its handle belongs to.
HANDLE_GROUPS = {4: 1, 5: 25, 6: 22, 10: 33, 11: 12, 23: 55, 26: 48, 27: 29, 28: 58, 30: 64}
_INTEGER_KINDS = (1, 8, 9, 29)

# ps3 event id to switch event id. Others are unknown.
EVENT_PS3_TO_SWITCH = {
    **{i: i for i in range(8)},
    8: 9, 9: 10, 10: 11, 11: 12, 12: 13, 13: 14, 14: 15, 16: 17, 17: 19, 18: 20, 19: 21, 24: 26,
    25: 27, 26: 28, 27: 29, 28: 30, 29: 32, 30: 33, 33: 37, 34: 38, 35: 39, 36: 40, 37: 41,
    38: 42, 39: 43, 41: 45, 44: 48, 45: 49, 46: 50, 47: 51, 49: 53, 51: 55,
}


@dataclass(frozen=True)
class MessageFilter:
    kind: int
    value_raw: int
    second_raw: int
    text: str | None
    value: int | float | tuple[float, float] | None  # by kind; raw where the kind is opaque
    group: int | None  # for a handle


@dataclass
class MessageEntry:
    event_id: int
    unique_id: int
    filters: list[MessageFilter]
    handler: str | None
    listener_flags: int
    event_count: int
    raw: bytes = b""
    strings: list[bytes | None] = field(default_factory=list)  # 4 filters, then the handler


@dataclass
class ScriptFile:
    header: SnoHeader
    build: str
    chunk: bytes
    messages: list[MessageEntry]
    uid: int
    quest_handle: int
    warnings: list[str] = field(default_factory=list)
    fixed: bytes = b""


def string_pair_at(build: str, slot: int) -> int:
    """Where a string pair sits in an entry. Slots 0 to 3 are filters, slot 4 the handler."""
    return _STRINGS_AT[build] + slot * 16 if slot < FILTER_COUNT else _HANDLER_AT[build]


def switch_event_id(event_id: int, build: str) -> int | None:
    """The switch id of an event. ps3 ids outside the known map give None."""
    return event_id if build == "switch" else EVENT_PS3_TO_SWITCH.get(event_id)


def _float(word: int, order: str) -> float:
    return struct.unpack(order + "f", struct.pack(order + "I", word))[0]


def _filter_value(kind: int, value: int, second: int, order: str):
    if kind in _NO_VALUE_KINDS:
        return None
    if kind in _FLOAT_KINDS:
        return _float(value, order)
    if kind in _RANGE_KINDS:
        return (_float(value, order), _float(second, order))
    return value


def _text(view: ByteView, position: int) -> str | None:
    raw = view.pair_data(view.pair(position), position, "script string")
    return raw.split(b"\0")[0].decode("latin-1") if raw else None


def _raw_string(view: ByteView, position: int) -> bytes | None:
    """The bytes of a string pair, or None for an empty or stale pair."""
    pair = view.pair(position)
    if pair.size == 0 or pair.file_offset + pair.size > len(view.data):
        return None
    return view.pair_data(pair, position)


def _read_filter(view: ByteView, base: int, build: str, slot: int, warnings: list[str]) -> MessageFilter:
    at = base + _FILTER_AT + slot * _FILTER_SIZE[build]
    if build == "switch":
        value, second = view.unpack("II", at)
        kind = view.u32(at + 16)
    else:
        kind, value, second = view.unpack("III", at)
    text = None
    if kind not in _NO_VALUE_KINDS:
        text = _text(view, base + _STRINGS_AT[build] + slot * 16)
    if kind in STRING_HASH_KINDS and text is not None and hash33_ci(text) != value:
        warnings.append(f"filter string {text!r} does not hash to its value")
    return MessageFilter(
        kind, value, second, text, _filter_value(kind, value, second, view.order),
        HANDLE_GROUPS.get(kind),
    )


def parse_script(data: bytes, build: str | None = None, strict: bool = False) -> ScriptFile:
    """Read a ``.scr``. A filter hash mismatch stays a warning, even with ``strict``."""
    header = parse_header(data, build)
    resolved = resolve_build("scr", header, build, strict)
    view = ByteView(data, header.order)
    view.need(0, 0x10 + 0x48, "script fixed part")
    count = view.i32(0x40)
    size = _ENTRY[resolved]
    pair = view.pair(0x30)
    if count < 0 or pair.size != count * size:
        raise FormatError(f"message map size {pair.size} for {count} entries", 0x30)
    view.pair_data(pair, 0x30, "message map")
    warnings: list[str] = []
    messages = []
    for i in range(count):
        base = pair.file_offset + i * size
        event_id, unique_id = view.unpack("ii", base)
        filters = [_read_filter(view, base, resolved, k, warnings) for k in range(FILTER_COUNT)]
        flags, events = view.unpack("Ii", base + _FLAGS_AT[resolved])
        strings = [
            _raw_string(view, base + _STRINGS_AT[resolved] + k * 16) for k in range(FILTER_COUNT)
        ]
        strings.append(_raw_string(view, base + _HANDLER_AT[resolved]))
        messages.append(MessageEntry(
            event_id, unique_id, filters, _text(view, base + _HANDLER_AT[resolved]), flags, events,
            view.take(base, size), strings,
        ))
    return ScriptFile(
        header, resolved, script_slice(data, "scr", resolved), messages, view.u32(0x44),
        view.i32(0x48), warnings, view.take(0, 0x58),
    )
