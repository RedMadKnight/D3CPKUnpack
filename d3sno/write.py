"""Writers and edit helpers for ``.stl``, ``.pow``, ``.scr``, ``.tex``, ``.gam`` and ``.app``.

``write(parsed)`` lays a parsed file out again, so parse then write gives the same bytes. The
fixed part is kept as it was, with its revision and source hash, and only pairs change.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, replace
from typing import Mapping

from . import appearance, archive, balance, material, power, scripts, text, texture
from .transform import Transform
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


# --- .gam ---------------------------------------------------------------------------------

@dataclass
class _Block:
    position: int  # in the row
    kind: str  # name, formula, empty or array
    data: bytes
    source: int | None  # file offset in the source, which marks a shared block


_NAME_ALIGN = {"switch": 8, "ps3": 4}


def _is_formula(raw: bytes, order: str) -> bool:
    if len(raw) < 32:
        return False
    head = struct.unpack_from(order + "8I", raw, 0)
    if any(head[i] for i in (0, 1, 2, 3, 4, 6)) or head[5] == 0:
        return False
    return 32 + (head[5] + 3) // 4 * 4 + head[7] == len(raw)


def _is_name(raw: bytes) -> bool:
    text_part = raw.split(b"\0")[0]
    return bool(text_part) and raw.endswith(b"\0") and all(0x20 <= b < 0x7F for b in text_part)


def _discover(file: balance.BalanceFile) -> list[list[_Block]]:
    """Find the pairs in each row by what they point at."""
    source, order = file.source, file.header.order
    table = struct.unpack_from(order + "I", source, file.slot)[0] + 0x10
    table_end = table + file.row_size * len(file.rows)
    found = []
    for row in file.rows:
        blocks = []
        position = 0
        while position + 8 <= file.row_size:
            offset, size = struct.unpack_from(order + "II", row.raw, position)
            start = offset + 0x10
            if size and table_end <= start and start + size <= len(source):
                raw = source[start:start + size]
                kind = "name" if position == 0 and _is_name(raw) else (
                    "formula" if _is_formula(raw, order) else "array"
                )
                blocks.append(_Block(position, kind, raw, start))
                position += 8
                continue
            if not size and offset and offset % 4 == 0 and table_end <= start <= len(source):
                blocks.append(_Block(position, "empty", b"", None))
                position += 8
                continue
            position += 4
        found.append(blocks)
    return found


def _lay_out_gam(file: balance.BalanceFile, rows: list[bytes], blocks: list[list[_Block]]) -> bytes:
    order = file.header.order
    out = bytearray(file.source[:balance.FIXED_END[file.build]])
    _align(out, 8)
    table = len(out)
    out += b"".join(rows)
    _put_pair(out, order, file.slot, table, len(out) - table)
    placed: dict[int, int] = {}
    for number, row_blocks in enumerate(blocks):
        row_at = table + number * file.row_size
        for block in row_blocks:
            at = row_at + block.position
            if block.kind == "empty":
                _align(out, 4)
                _put_pair(out, order, at, len(out), 0)
                continue
            if block.source is not None and block.source in placed:
                _put_pair(out, order, at, placed[block.source], len(block.data))
                continue
            alignment = {"name": _NAME_ALIGN[file.build], "formula": 4}.get(block.kind, 8)
            _align(out, alignment)
            if block.source is not None:
                placed[block.source] = len(out)
            _put_pair(out, order, at, len(out), len(block.data))
            out += block.data
    return bytes(out)


def _gam_blocks(file: balance.BalanceFile) -> list[list[_Block]]:
    """The blocks of each row, checked by laying out the source again."""
    if file.row_size is None or file.slot is None:
        raise FormatError(f"type {file.type} has no table this writer knows")
    blocks = getattr(file, "_write_blocks", None)
    if blocks is None:
        blocks = _discover(file)
        if _lay_out_gam(file, [row.raw for row in file.rows], blocks) != file.source:
            raise FormatError("the layout of this .gam is not understood, so it is not written")
        file._write_blocks = blocks
    return blocks


def write_gam(file: balance.BalanceFile) -> bytes:
    """Lay out a ``.gam`` again. Refuses a file whose own layout it cannot reproduce."""
    blocks = _gam_blocks(file)
    return _lay_out_gam(file, [row.raw for row in file.rows], blocks)


def _row_block(file: balance.BalanceFile, row: int, position: int) -> _Block:
    if not 0 <= row < len(file.rows):
        raise FormatError(f"no row {row}")
    for block in _gam_blocks(file)[row]:
        if block.position == position:
            return block
    raise FormatError(f"row {row} has no pair at +0x{position:X}")


def set_row_name(file: balance.BalanceFile, row: int, name: str) -> None:
    """Rename a row. Its game balance id changes with it."""
    block = _row_block(file, row, 0)
    if block.kind != "name":
        raise FormatError(f"row {row} has no name block")
    block.data = name.encode("latin-1") + b"\0"
    block.source = None
    file.rows[row].name = name


def set_row_formula(file: balance.BalanceFile, row: int, position: int, formula_text: str, code: bytes) -> None:
    """Replace a formula held by a row, text and code together."""
    block = _row_block(file, row, position)
    if block.kind not in ("formula", "empty"):
        raise FormatError(f"row {row} +0x{position:X} is not a formula")
    if len(code) % 4:
        raise FormatError(f"code length {len(code)} is not a multiple of 4")
    block.kind = "formula"
    block.data = _formula_bytes(power.Formula(formula_text, bytes(code)), file.header.order)
    block.source = None


def set_row_field(file: balance.BalanceFile, row: int, offset: int, fmt: str, value) -> None:
    """Change a plain field of a row in place, for example ``set_row_field(f, 0, 0x14, "i", 3)``."""
    if not 0 <= row < len(file.rows):
        raise FormatError(f"no row {row}")
    size = struct.calcsize(fmt)
    if offset < 0 or offset + size > file.row_size:
        raise FormatError(f"field +0x{offset:X} is outside the row")
    for block in _gam_blocks(file)[row]:
        if offset < block.position + 8 and block.position < offset + size:
            raise FormatError(f"field +0x{offset:X} overlaps a pair; use the block helpers")
    raw = bytearray(file.rows[row].raw)
    try:
        struct.pack_into(file.header.order + fmt, raw, offset, value)
    except struct.error as exc:
        raise FormatError(f"bad value for +0x{offset:X}: {exc}") from None
    file.rows[row].raw = bytes(raw)


# --- .app ----------------------------------------------------------------------------------

_BONE_SIZE = {"ps3": 176, "switch": 256}
_BONE_TRANSFORMS = {
    "ps3": {"rest": 0x30, "rest_inverse": 0x50, "local": 0x70},
    "switch": {"rest": 0x34, "rest_inverse": 0x54, "local": 0x74, "bind": 0x94, "bind_inverse": 0xB4},
}


def write_app(app: appearance.AppearanceFile) -> bytes:
    """The file with its in-place edits. Edits that would change a size are not supported."""
    return bytes(app.source)


def set_bone_transform(app: appearance.AppearanceFile, index: int, which: str, value: Transform) -> None:
    """Set one of a bone's transforms: rest, rest_inverse, local, and on switch bind, bind_inverse."""
    if not 0 <= index < len(app.bones):
        raise FormatError(f"no bone {index}")
    offsets = _BONE_TRANSFORMS[app.build]
    if which not in offsets:
        raise FormatError(f"bones on {app.build} have no {which} transform")
    order = app.header.order
    table = struct.unpack_from(order + "I", app.source, 0x28)[0] + 0x10
    at = table + index * _BONE_SIZE[app.build] + offsets[which]
    struct.pack_into(order + "8f", app.source, at, *value.q, *value.t, value.s)
    app.bones[index] = replace(app.bones[index], **{which: value})


def set_hardpoint(app: appearance.AppearanceFile, index: int, rotation, position) -> None:
    if not 0 <= index < len(app.hardpoints):
        raise FormatError(f"no hardpoint {index}")
    order = app.header.order
    at = struct.unpack_from(order + "I", app.source, 0xCC)[0] + 0x10 + index * 40 + 12
    struct.pack_into(order + "7f", app.source, at, *rotation, *position)
    app.hardpoints[index] = replace(
        app.hardpoints[index], rotation=tuple(rotation), position=tuple(position),
    )


def set_material_texture(
    app: appearance.AppearanceFile, material_hash: int, look: int, slot_type: int, handle: int,
) -> None:
    """Point one texture slot of one look of a material at another texture."""
    found = app.material_data.find(material_hash)
    if found is None or not 0 <= look < len(found.per_look):
        raise FormatError(f"no material 0x{material_hash:X} with look {look}")
    block = found.per_look[look][0]
    for i, slot in enumerate(block.textures):
        if slot.type == slot_type:
            struct.pack_into(app.header.order + "i", app.source, slot.position, handle)
            block.textures[i] = replace(slot, handle=handle)
            return
    raise FormatError(f"material 0x{material_hash:X} look {look} has no slot type {slot_type}")


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
    if isinstance(parsed, balance.BalanceFile):
        return write_gam(parsed)
    if isinstance(parsed, appearance.AppearanceFile):
        return write_app(parsed)
    raise FormatError(f"no writer for {type(parsed).__name__}")


def replace_in_archive(source, files: Mapping[str, object], **options) -> archive.RebuildResult:
    """Rebuild an archive with parsed files from these writers put in by name."""
    return archive.rebuild(
        source, replace={name: write(parsed) for name, parsed in files.items()}, **options,
    )
