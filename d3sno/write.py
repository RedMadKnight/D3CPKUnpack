"""Writers and edit helpers for ``.stl``, ``.pow``, ``.scr`` and ``.tex``.

``write(parsed)`` lays a parsed file out again, so parse then write gives the same bytes. The
fixed part is kept as it was, with its revision and source hash, and only pairs change.
"""

from __future__ import annotations

import struct
from dataclasses import replace
from typing import Mapping

from . import archive, power, scripts, text, texture
from .binary import FormatError, build_order
from .luachunk import LUA_SIGNATURE
from .sno import hash33

_SCRIPT_CHUNK_PAIR = 0x20
_SCRIPT_MAP_PAIR = 0x30
_SCRIPT_COUNT = 0x40
_STL_ENTRIES_PAIR = 0x28


def _align(out: bytearray, alignment: int) -> None:
    out += bytes(-len(out) % alignment)


def _put_pair(out: bytearray, order: str, at: int, position: int, size: int) -> None:
    struct.pack_into(order + "II", out, at, position - 0x10, size)


def _clear_stale_pair(out: bytearray, order: str, at: int) -> None:
    """An emptied block leaves no size behind. An offset with size 0 is left as it was."""
    if struct.unpack_from(order + "I", out, at + 4)[0]:
        struct.pack_into(order + "II", out, at, 0, 0)


def _place_shared(out: bytearray, seen: dict[bytes, int], raw: bytes) -> int:
    """Add ``raw`` 8-aligned, or give the position of its first copy."""
    position = seen.get(raw)
    if position is None:
        _align(out, 8)
        position = len(out)
        out += raw
        seen[raw] = position
    return position


# --- .stl ---------------------------------------------------------------------------------

def write_stl(strings: text.StringList) -> bytes:
    order = strings.header.order
    out = bytearray(strings.fixed)
    table = len(out)
    out += bytes(text.ENTRY_SIZE * len(strings.entries))
    seen: dict[bytes, int] = {}
    for i, entry in enumerate(strings.entries):
        raw = entry.text.encode("utf-8") + b"\0"
        position = _place_shared(out, seen, raw)
        at = table + i * text.ENTRY_SIZE
        out[at:at + 8] = entry.slot_raw
        _put_pair(out, order, at + 8, position, len(raw))
        struct.pack_into(order + "II", out, at + 16, entry.label_hash, entry.zero)
    if strings.entries:
        _put_pair(out, order, _STL_ENTRIES_PAIR, table, text.ENTRY_SIZE * len(strings.entries))
    else:
        _clear_stale_pair(out, order, _STL_ENTRIES_PAIR)
    return bytes(out)


def _label_hash(label: str | int) -> int:
    return label if isinstance(label, int) else hash33(label)


def set_text(strings: text.StringList, label: str | int, value: str) -> None:
    target = _label_hash(label)
    for i, entry in enumerate(strings.entries):
        if entry.label_hash == target:
            strings.entries[i] = replace(entry, text=value)
            strings.by_hash[target] = value
            return
    raise FormatError(f"no label 0x{target:08X} in the list")


def add_entry(strings: text.StringList, label: str | int, value: str) -> None:
    """Add a label. New entries go last."""
    target = _label_hash(label)
    if target in strings.by_hash:
        raise FormatError(f"label 0x{target:08X} is already in the list")
    strings.entries.append(text.StringEntry(target, value))
    strings.by_hash[target] = value


def remove_entry(strings: text.StringList, label: str | int) -> None:
    target = _label_hash(label)
    if target not in strings.by_hash:
        raise FormatError(f"no label 0x{target:08X} in the list")
    strings.entries = [e for e in strings.entries if e.label_hash != target]
    del strings.by_hash[target]


# --- .pow ---------------------------------------------------------------------------------

def write_pow(file: power.PowerFile) -> bytes:
    order = build_order(file.build)
    out = bytearray(file.fixed)
    placed: dict[str, int] = {}
    for name, at in power.MAP_POSITIONS:
        tag_map = file.maps[name]
        if not tag_map.raw:
            continue
        root = tag_map.shares_with or name
        if root not in placed:
            _align(out, 4)
            placed[root] = len(out)
            out += tag_map.raw
        _put_pair(out, order, at, placed[root], len(tag_map.raw))
    for i, buff in enumerate(file.buff_filters):
        if buff.raw:
            _align(out, 8)
            _put_pair(out, order, power.BUFF_FILTERS_AT + i * 16, len(out), len(buff.raw))
            out += buff.raw
    script_at = power.TRAILER_AT[file.build] + 0x18
    if file.script:
        _align(out, 8)
        _put_pair(out, order, script_at, len(out), len(file.script))
        out += file.script
    else:
        _clear_stale_pair(out, order, script_at)
    return bytes(out)


def _formula_bytes(formula: power.Formula, order: str) -> bytes:
    raw = formula.text.encode("latin-1") + b"\0"
    head = struct.pack(order + "8I", 0, 0, 0, 0, 0, len(raw), 0, len(formula.code))
    return head + raw + bytes(-len(raw) % 4) + formula.code


def _record_bytes(record: power.TagRecord, order: str) -> bytes:
    head = struct.pack(order + "II", record.type, record.tag)
    if record.type == power.TAG_FORMULA:
        return head + _formula_bytes(record.value, order)
    code = "f" if record.type == power.TAG_FLOAT else "i"
    try:
        return head + struct.pack(order + code, record.value)
    except (struct.error, TypeError):
        raise FormatError(f"tag 0x{record.tag:X}: bad value {record.value!r}") from None


def _store_map(file: power.PowerFile, tag_map: power.TagMap) -> None:
    """Serialise an edited map into every map that shares its block."""
    order = build_order(file.build)
    raw = struct.pack(order + "I", len(tag_map.records))
    raw += b"".join(_record_bytes(record, order) for record in tag_map.records)
    root = tag_map.shares_with or tag_map.name
    for other in file.maps.values():
        if other.name == root or other.shares_with == root:
            other.records = list(tag_map.records)
            other.raw = raw


def _find_tag(file: power.PowerFile, map_name: str, tag: int) -> tuple[power.TagMap, int]:
    tag_map = file.maps.get(map_name)
    if tag_map is None:
        raise FormatError(f"no tag map {map_name!r}")
    for index, record in enumerate(tag_map.records):
        if record.tag == tag:
            return tag_map, index
    raise FormatError(f"no tag 0x{tag:X} in map {map_name}")


def set_tag(file: power.PowerFile, map_name: str, tag: int, value) -> None:
    """Set a tag of a non-formula type. A shared block changes for all its maps."""
    tag_map, index = _find_tag(file, map_name, tag)
    record = tag_map.records[index]
    if record.type == power.TAG_FORMULA:
        raise FormatError(f"tag 0x{tag:X} is a formula: use set_formula")
    tag_map.records[index] = replace(record, value=value)
    _store_map(file, tag_map)


def set_formula(file: power.PowerFile, map_name: str, tag: int, formula_text: str, code: bytes) -> None:
    """Replace a formula. The game runs the code, so text and code go together."""
    tag_map, index = _find_tag(file, map_name, tag)
    record = tag_map.records[index]
    if record.type != power.TAG_FORMULA:
        raise FormatError(f"tag 0x{tag:X} is not a formula")
    if len(code) % 4:
        raise FormatError(f"code length {len(code)} is not a multiple of 4")
    tag_map.records[index] = replace(record, value=power.Formula(formula_text, bytes(code)))
    _store_map(file, tag_map)


def _check_chunk(chunk: bytes) -> None:
    if chunk and not chunk.startswith(LUA_SIGNATURE):
        raise FormatError("not a Lua chunk")


def replace_script(file: power.PowerFile | scripts.ScriptFile, chunk: bytes) -> None:
    """Replace the script bytes of a ``.pow`` or ``.scr``."""
    _check_chunk(chunk)
    if isinstance(file, power.PowerFile):
        file.script = bytes(chunk)
    else:
        file.chunk = bytes(chunk)


# --- .scr ---------------------------------------------------------------------------------

def write_scr(file: scripts.ScriptFile) -> bytes:
    order = build_order(file.build)
    out = bytearray(file.fixed)
    if file.chunk:
        _align(out, 8)
        _put_pair(out, order, _SCRIPT_CHUNK_PAIR, len(out), len(file.chunk))
        out += file.chunk
    else:
        _clear_stale_pair(out, order, _SCRIPT_CHUNK_PAIR)
    if not file.messages:
        return bytes(out)
    _align(out, 8)
    table = len(out)
    size = len(file.messages[0].raw)
    _put_pair(out, order, _SCRIPT_MAP_PAIR, table, size * len(file.messages))
    struct.pack_into(order + "i", out, _SCRIPT_COUNT, len(file.messages))
    out += b"".join(message.raw for message in file.messages)
    seen: dict[bytes, int] = {}
    for i, message in enumerate(file.messages):
        for slot, raw in enumerate(message.strings):
            if raw is not None:
                at = table + i * size + scripts.string_pair_at(file.build, slot)
                _put_pair(out, order, at, _place_shared(out, seen, raw), len(raw))
    return bytes(out)


def set_handler(file: scripts.ScriptFile, index: int, name: str) -> None:
    """Change the handler of one message map entry."""
    if not 0 <= index < len(file.messages):
        raise FormatError(f"no message map entry {index}")
    message = file.messages[index]
    message.handler = name
    message.strings[scripts.FILTER_COUNT] = name.encode("latin-1") + b"\0"


# --- .tex ---------------------------------------------------------------------------------

def write_tex(tex: texture.TextureFile) -> bytes:
    order = build_order(tex.build)
    out = bytearray(tex.fixed)
    aligned = tex.build == "switch"
    for raw, at in (
        (tex.image_ids_raw, texture.image_id_pair_at(tex.build)),
        (tex.frames_raw, texture.FRAME_PAIR_AT),
    ):
        if aligned:
            _align(out, 8)  # also for an empty block
        if raw:
            _put_pair(out, order, at, len(out), len(raw))
            out += raw
    if tex.payload:
        _put_pair(out, order, texture.PAYLOAD_AT, len(out), len(tex.payload))
        out += tex.payload
    return bytes(out)


def replace_pixels(
    tex: texture.TextureFile, elements: bytes, level: int = 0, face: int = 0, z: int = 0,
) -> None:
    """Replace a surface at the same format and size.

    ``elements`` holds stored elements in row order: DXT blocks as given, or texels in the
    file's own channel order. On switch they go through the block-linear addressing.
    """
    offsets = texture.element_offsets(tex, level, face, z)
    bpp = texture.FORMATS[tex.format][1]
    if len(elements) != len(offsets) * bpp:
        raise FormatError(f"{len(elements)} bytes for {len(offsets)} elements of {bpp}")
    payload = bytearray(tex.payload)
    for i, at in enumerate(offsets):
        payload[at:at + bpp] = elements[i * bpp:(i + 1) * bpp]
    tex.payload = bytes(payload)
    if tex.nv is not None:
        start = tex.nv.image_offset
        tex.nv.image = tex.payload[start:start + len(tex.nv.image)]


# --- Dispatch -----------------------------------------------------------------------------

def write(parsed) -> bytes:
    """Lay out a parsed ``.stl``, ``.pow``, ``.scr`` or ``.tex``."""
    if isinstance(parsed, text.StringList):
        return write_stl(parsed)
    if isinstance(parsed, power.PowerFile):
        return write_pow(parsed)
    if isinstance(parsed, scripts.ScriptFile):
        return write_scr(parsed)
    if isinstance(parsed, texture.TextureFile):
        return write_tex(parsed)
    raise FormatError(f"no writer for {type(parsed).__name__}")


def replace_in_archive(source, files: Mapping[str, object], **options) -> archive.RebuildResult:
    """Rebuild an archive with parsed files from these writers put in by name."""
    return archive.rebuild(
        source, replace={name: write(parsed) for name, parsed in files.items()}, **options,
    )
