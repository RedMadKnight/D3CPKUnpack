"""Stage 15: switch-form UI dialogs and fonts (``30_create_content.md`` W15-B8 to W15-B11).

"Switch form" is the layout of the switch build; x360 uses the same layout in big-endian, apart
from the 16-byte x360 glyph (W15-B10). Reading and writing share one field table per record
(``UI_RECORDS``, ``font_records(build)``):

- W15-B8: ``parse_ui(data, build)`` / ``write_ui(dialog, build)``. Run-time slots are 8 bytes, a
  list is {i32 count, pair, 12 zero bytes}, every block is aligned to 8 from body start and gaps
  are zero; blocks follow the depth-first walk of W11-2. The dialog is the same nested dict as the
  ps3 reader (``ui_font.parse_ps3_ui``) gives, plus ``header`` (file bytes 0..0x1B, copied) and the
  binding's ``scale`` (f32 at +0x14, an input: 1.0 or 0.6).
- W15-B9: ``ui_key`` (64-bit FNV-1a over the bytes with A..Z lower-cased), ``control_paths``,
  ``derive_ui_keys`` (ordinals, control, parent, style, inline-style and description keys) and
  ``ui_link_problems``.
- W15-B10: ``parse_font(data, build)`` / ``write_font(font, build)``. Deferred blocks (W11-3):
  group 0 every point size's glyph table, group 1 every texture's image then frames; both unaligned
  (blobs, alignment 1). ``custom_data_offset`` is the body offset where the deferred blocks start.
  Image helpers: ``font_image_bytes`` / ``font_image_pixels`` for ps3 (rows), switch (W14-10
  container, one 2D level, pixel format code 1) and x360 (tiled, pitch max(width, 32)).
- W15-B11: ``derive_glyph``, ``line_break_glyph``, ``derive_font_values`` and
  ``font_derived_problems`` (used by ``ui_font.font_problems``).

Corrections 18 (``31_corrections18.md``):

- V18-2: texts are shared. A text equal to an earlier one points at its block; ``parse_ui``
  accepts such pairs in the tiling and records them in ``text_layout``; ``write_ui`` reproduces a
  valid stored layout and otherwise stores each distinct text at its first occurrence in the walk
  (``share_texts``: "proven" fields by default, "all" as HYPOTHESIS, "none").
- V18-4: ordinals are a permutation of 0..n-1, not the list order. ``derive_ui_keys`` keeps
  them, ``append_control`` gives n, ``remove_control`` lowers the ones above (HYPOTHESIS), and the
  check (``ordinal_problems``) tests only the permutation.
- V18-5: glyph characters do not decrease; a repeat is allowed on switch, not on ps3 / x360
  (``glyph_order_problems``). ``derive_font_values`` sorts stably.

Corrections 19 (``32_corrections19.md``):

- V19-1: lists are shared within one field. A list whose block bytes equal a stored list of the
  same field (walk path without indices) points at it; ``parse_ui`` accepts such pairs and
  records them in ``list_layout``; ``write_ui`` follows a stored layout per entry (equal lists
  stored twice stay so) and otherwise shares at the first occurrence (``share_lists``: "proven"
  control fields by default, "all" adds dialog-level styles as HYPOTHESIS, "none").

Reading raises ``FormatError`` only where a read leaves the file; every other rule is reported in
``problems``. A file with no problems is written back byte-identically.
"""

from __future__ import annotations

import copy
import re
import struct
from dataclasses import dataclass

from . import texture as T
from . import texture_write as TW
from . import x360_texture as XT
from .binary import ByteView, FormatError, build_order
from .sno import SNO_SIGNATURE, expected_version, parse_header
from .ui_font import BINDING_PREFIX, TableRead, animation_problems

BODY = 0x10
HEADER_SIZE = 0x1C  # SNO header words; body 0x00..0x0B of the top record
SWITCH_FORM = ("switch", "x360")
ALIGN = 8  # W15-B8 / W11-4: arrays and texts
LIST_SIZE = 24

# V18-2: text sharing scopes of ``write_ui``. PROVEN: control names, parent paths and every text
# of a control's inline style (its name, string texts, child-spec paths and their string texts).
# HYPOTHESIS ("all"): the other texts (dialog styles, constant strings, descriptions, bindings).
TEXT_SHARING = ("proven", "all", "none")
_PROVEN_TEXT = re.compile(r"dialog\.controls\[\d+\](?:\.inline_style(?:\..*)?)?$")

# V19-1: list sharing scopes of ``write_ui``. A list shares only with the same field, the walk
# path with element indices removed (``_list_field``). PROVEN: the control fields (animation
# int, float, vector, description and binding lists; the inline style's string, int, float and
# child-spec lists and the child specs' own lists). HYPOTHESIS ("all"): the same lists of the
# dialog-level styles. The dialog's own lists occur once and never share. ``33`` V20-1: string
# lists are the exception; inline-style, child-spec and dialog-style string lists form one class
# (``STRING_LISTS``) and share across these fields, dialog-style string lists by default.
LIST_SHARING = ("proven", "all", "none")
_PROVEN_LIST = re.compile(r"dialog\.controls\[\d+\]\.(?:anim_\w+|inline_style\.(?:child_specs\[\d+\]\.)?\w+)$")
_STYLE_LIST = re.compile(r"dialog\.styles\[\d+\]\.(?:child_specs\[\d+\]\.)?\w+$")
_INDEX = re.compile(r"\[\d+\]")


STRING_LISTS = "<string lists>"


def _list_field(where: str, record: str | None = None) -> str:
    """V19-1: the field of a list, its walk path without element indices (for example
    ``dialog.controls[].inline_style.int32s``). Lists share only within one field.
    ``33`` V20-1: string-element lists (inline-style, child-spec and dialog-style strings) form
    one sharing class across fields, ``STRING_LISTS``; the walk path does not matter for them."""
    if record == "string":
        return STRING_LISTS
    return _INDEX.sub("[]", where)

# --- Field tables -------------------------------------------------------------------------------
# ("prim", name, offset, struct code) ("char", name, offset, size) ("zero", offset, size)
# ("text", name, offset) pair, align 8 ("list", name, offset, record) 24-byte list
# ("struct", name, offset, record) inline record
# ("array", name, offset, record, count field, alignment, deferred group)
# ("blob", name, offset, alignment, deferred group)

_DATA_LISTS = (("strings", "string"), ("int32s", "int32"), ("float32s", "float32"))

UI_RECORDS: dict[str, tuple[int, tuple]] = {
    "dialog": (0xB8, (
        ("prim", "machine_time", 0x0C, "I"),
        ("list", "controls", 0x10, "control"),
        ("list", "styles", 0x28, "style"),
        ("list", "constant_strings", 0x40, "string"),
        ("list", "constant_int32s", 0x58, "int32"),
        ("list", "constant_float32s", 0x70, "float32"),
        ("zero", 0x88, 0x30),
    )),
    "control": (0x170, (
        ("prim", "ordinal", 0x00, "i"),
        ("zero", 0x04, 4),
        ("prim", "key", 0x08, "Q"),
        ("text", "name", 0x10), ("zero", 0x18, 8),
        ("prim", "parent_key", 0x20, "Q"),
        ("text", "parent_name", 0x28), ("zero", 0x30, 8),
        ("prim", "control_type", 0x38, "i"),
        ("zero", 0x3C, 4),
        ("struct", "inline_style", 0x40, "style"),
        ("list", "anim_int32s", 0xE0, "anim_int32"),
        ("list", "anim_floats", 0xF8, "anim_float"),
        ("list", "anim_vector2s", 0x110, "anim_vector2"),
        ("list", "anim_descriptions", 0x128, "anim_description"),
        ("list", "anim_bindings", 0x140, "anim_binding"),
        ("zero", 0x158, 0x18),
    )),
    "style": (0xA0, (
        ("prim", "key", 0x00, "Q"),
        ("text", "name", 0x08), ("zero", 0x10, 8),
        ("list", "strings", 0x18, "string"),
        ("list", "int32s", 0x30, "int32"),
        ("list", "float32s", 0x48, "float32"),
        ("list", "child_specs", 0x60, "child_spec"),
        ("zero", 0x78, 0x28),
    )),
    "child_spec": (0x70, (
        ("text", "path_to_child", 0x00), ("zero", 0x08, 8),
        ("list", "strings", 0x10, "string"),
        ("list", "int32s", 0x28, "int32"),
        ("list", "float32s", 0x40, "float32"),
        ("zero", 0x58, 0x18),
    )),
    "string": (24, (("prim", "key", 0, "Q"), ("text", "text", 8), ("zero", 0x10, 8))),
    "int32": (16, (("prim", "key", 0, "Q"), ("prim", "value", 8, "i"), ("zero", 0x0C, 4))),
    "float32": (16, (("prim", "key", 0, "Q"), ("prim", "value", 8, "f"), ("zero", 0x0C, 4))),
    "anim_description": (40, (
        ("prim", "key", 0, "Q"), ("text", "name", 8), ("zero", 0x10, 8),
        ("prim", "type", 0x18, "i"), ("prim", "node_range", 0x1C, "2i"), ("zero", 0x24, 4),
    )),
    "anim_binding": (24, (
        ("text", "animation_name", 0), ("zero", 0x08, 8),
        ("prim", "property", 0x10, "i"), ("prim", "scale", 0x14, "f"),
    )),
    "anim_int32": (8, (("prim", "time", 0, "f"), ("prim", "value", 4, "i"))),
    "anim_float": (8, (("prim", "time", 0, "f"), ("prim", "value", 4, "f"))),
    "anim_vector2": (12, (("prim", "time", 0, "f"), ("prim", "x", 4, "f"), ("prim", "y", 8, "f"))),
}

_FONT_COMMON: dict[str, tuple[int, tuple]] = {
    "font": (0xD8, (
        ("prim", "flags", 0x0C, "I"),
        ("char", "font_face", 0x10, 64),
        ("char", "original_font_file", 0x50, 64),
        ("prim", "small_caps", 0x90, "i"),
        ("prim", "hinting_mode", 0x94, "i"),
        ("prim", "hinting_style", 0x98, "i"),
        ("prim", "size_adjustment", 0x9C, "i"),
        ("prim", "line_gap_adjustment", 0xA0, "f"),
        ("prim", "space_scalar", 0xA4, "f"),
        ("array", "textures", 0xA8, "texture", "texture_count", ALIGN, None), ("zero", 0xB0, 8),
        ("prim", "texture_count", 0xB8, "I"),
        ("prim", "point_size_count", 0xBC, "I"),
        ("array", "point_sizes", 0xC0, "point_size", "point_size_count", ALIGN, None), ("zero", 0xC8, 8),
        ("prim", "custom_data_offset", 0xD0, "I"),
        ("zero", 0xD4, 4),
    )),
    "texture": (40, (
        ("prim", "resolution_flags", 0x00, "I"),
        ("prim", "format", 0x04, "i"),
        ("prim", "width", 0x08, "I"),
        ("prim", "height", 0x0C, "I"),
        ("blob", "image", 0x10, 1, 1),
        ("prim", "texture_handle", 0x18, "i"),
        ("prim", "frame_count", 0x1C, "I"),
        ("array", "frames", 0x20, "frame", "frame_count", 1, 1),
    )),
    "frame": (16, (("prim", "u0_v0_u1_v1", 0, "4f"),)),
    "point_size": (88, (
        ("prim", "height", 0x00, "i"), ("prim", "width", 0x04, "i"),
        ("prim", "ascender", 0x08, "i"), ("prim", "max_ascender", 0x0C, "i"),
        ("prim", "descender", 0x10, "i"), ("prim", "line_gap", 0x14, "i"),
        ("prim", "advance_y", 0x18, "i"), ("zero", 0x1C, 4),
        ("prim", "resolution_flags", 0x20, "I"),
        ("prim", "glyph_count", 0x24, "I"),
        ("array", "glyphs", 0x28, "glyph", "glyph_count", 1, 0), ("zero", 0x30, 8),
        ("prim", "kerning_count", 0x38, "I"), ("zero", 0x3C, 4),
        ("array", "kernings", 0x40, "kerning", "kerning_count", ALIGN, None), ("zero", 0x48, 8),
        ("prim", "icon_height", 0x50, "i"),
        ("prim", "point_size", 0x54, "f"),
    )),
    "kerning": (16, (
        ("prim", "previous_char", 0, "I"), ("prim", "char", 4, "I"),
        ("prim", "x", 8, "i"), ("prim", "y", 0x0C, "i"),
    )),
}

GLYPH_RECORDS = {
    "switch": (24, (
        ("prim", "texture_handle", 0x00, "i"), ("prim", "frame", 0x04, "i"),
        ("prim", "bearing_x", 0x08, "h"), ("prim", "bearing_y", 0x0A, "h"),
        ("prim", "height", 0x0C, "h"), ("prim", "advance", 0x0E, "h"),
        ("prim", "texture_index", 0x10, "h"), ("prim", "char", 0x12, "H"),
        ("zero", 0x14, 4),
    )),
    "x360": (16, (
        ("prim", "texture_handle", 0x00, "i"), ("prim", "frame", 0x04, "i"),
        ("prim", "bearing_x", 0x08, "b"), ("prim", "bearing_y", 0x09, "b"),
        ("prim", "height", 0x0A, "B"), ("prim", "advance", 0x0B, "B"),
        ("prim", "texture_index", 0x0C, "h"), ("prim", "char", 0x0E, "H"),
    )),
}


def font_records(build: str) -> dict[str, tuple[int, tuple]]:
    """The switch-form font field table of ``build`` (switch or x360: they differ in the glyph)."""
    _switch_form(build)
    return {**_FONT_COMMON, "glyph": GLYPH_RECORDS[build]}


def record_size(record: str, build: str = "switch") -> int:
    table = UI_RECORDS if record in UI_RECORDS else font_records(build)
    return table[record][0]


def _switch_form(build: str) -> None:
    if build not in SWITCH_FORM:
        raise FormatError(f"build {build!r} has no switch-form layout (switch or x360)")


def _align(value: int, step: int) -> int:
    return (value + step - 1) // step * step


def _default(kind: str, code: str = ""):
    if kind in ("list", "array"):
        return []
    if kind in ("text", "blob"):
        return None
    if kind == "char":
        return ""
    if code == "2i":
        return (0, 0)
    if code == "4f":
        return (0.0, 0.0, 0.0, 0.0)
    return 0.0 if code == "f" else 0


def new_header(build: str, ext: str, handle: int = 0, revision: int = 0, source_hash: int = 0) -> bytes:
    """A fresh 0x1C-byte SNO header (signature, the build's version of ``ext``, revision, source
    hash, handle, two zero run-time words)."""
    version = expected_version(ext, build)
    if version is None:
        raise FormatError(f".{ext} has no version on {build}")
    return struct.pack(build_order(build) + "7I", SNO_SIGNATURE, version, revision, source_hash,
                       handle & 0xFFFFFFFF, 0, 0)


# --- Reader -------------------------------------------------------------------------------------

@dataclass
class _Block:
    group: int | None
    offset: int  # file position
    size: int
    align: int
    name: str


class _Reader:
    def __init__(self, data: bytes, build: str, records: dict):
        self.data = data
        self.order = build_order(build)
        self.view = ByteView(data, self.order)
        self.records = records
        self.blocks: list[_Block] = []
        self.problems: list[str] = []
        # V18-2: texts in walk order; -1 = stored here, j = shares the block of text j
        self.text_layout: list[int] = []
        self._text_pairs: dict[tuple[int, int], int] = {}
        # V19-1: non-empty list pairs in walk order; -1 = stored here, j = shares list j (same
        # field). ``_list_pairs`` maps (field, offset, size) of a stored list to (index, values).
        self.list_layout: list[int] = []
        self._list_pairs: dict[tuple[str, int, int], tuple[int, list]] = {}

    def record(self, record: str, at: int, where: str) -> dict:
        size, fields = self.records[record]
        self.view.need(at, size, where)
        values: dict = {}
        self._fixed(fields, at, where, values)
        self._pairs(fields, at, where, values)
        return values

    def _fixed(self, fields, at, where, values) -> None:
        for f in fields:
            kind = f[0]
            if kind == "prim":
                got = self.view.unpack(f[3], at + f[2], f"{where}.{f[1]}")
                values[f[1]] = got[0] if len(got) == 1 else got
            elif kind == "char":
                raw = self.view.take(at + f[2], f[3], f"{where}.{f[1]}")
                end = raw.find(b"\0")
                if end < 0:
                    self.problems.append(f"{where}.{f[1]}: no NUL in {f[3]} characters")
                    end = len(raw)
                elif any(raw[end:]):
                    self.problems.append(f"{where}.{f[1]}: bytes after the NUL are not zero")
                values[f[1]] = raw[:end].decode("latin-1")
            elif kind == "zero":
                if any(self.view.take(at + f[1], f[2])):
                    self.problems.append(f"{where}: bytes 0x{f[1]:X}..0x{f[1] + f[2]:X} are not zero")
            elif kind == "list":
                pos = at + f[2]
                if any(self.view.take(pos + 12, 12)):
                    self.problems.append(f"{where}.{f[1]}: list slot bytes are not zero")
            elif kind == "struct":
                sub: dict = {}
                self._fixed(self.records[f[3]][1], at + f[2], f"{where}.{f[1]}", sub)
                values[f[1]] = sub

    def _pair(self, pos: int, align: int, group, name: str) -> bytes:
        offset, size = self.view.unpack("II", pos, name)
        if size == 0:
            if offset:
                self.problems.append(f"{name}: empty pair with offset 0x{offset:X}")
            return b""
        self.view.need(BODY + offset, size, name)
        self.blocks.append(_Block(group, BODY + offset, size, align, name))
        return self.data[BODY + offset:BODY + offset + size]

    def _elements(self, record: str, raw: bytes, count, name: str, offset: int) -> list[dict]:
        size = self.records[record][0]
        if count is None or count < 0 or len(raw) != count * size:
            self.problems.append(f"{name}: {len(raw)} bytes for {count} elements of {size}")
            return []
        return [self.record(record, offset + i * size, f"{name}[{i}]") for i in range(count)]

    def _pairs(self, fields, at, where, values) -> None:
        for f in fields:
            kind, name = f[0], f"{where}.{f[1]}" if f[0] != "zero" else ""
            if kind == "text":
                pair = self.view.unpack("II", at + f[2], name)
                earlier = self._text_pairs.get(pair) if pair[1] else None
                if earlier is not None:
                    # V18-2: a pair repeating an earlier text pair shares its block (same bytes,
                    # so the same text); it is not a block of its own and no tiling error
                    self.text_layout.append(earlier)
                    raw = self.data[BODY + pair[0]:BODY + pair[0] + pair[1]]
                    values[f[1]] = raw.split(b"\0")[0].decode("latin-1")
                    continue
                raw = self._pair(at + f[2], ALIGN, None, name)
                if not raw:
                    values[f[1]] = None
                    continue
                self._text_pairs[pair] = len(self.text_layout)
                self.text_layout.append(-1)
                if raw[-1] != 0 or raw.count(0) != 1:
                    self.problems.append(f"{name}: text is not one NUL-terminated string")
                values[f[1]] = raw.split(b"\0")[0].decode("latin-1")
            elif kind == "list":
                pos = at + f[2]
                count = self.view.i32(pos, name)
                pair = self.view.unpack("II", pos + 4, name)
                key = (_list_field(name, f[3]),) + tuple(pair)
                earlier = self._list_pairs.get(key) if pair[1] else None
                if earlier is not None:
                    # V19-1: a pair repeating a stored list pair of the same field shares that
                    # block (same offset and size, so the same bytes); it is not walked again
                    # and is no tiling error. Another field or another size stays a block of
                    # its own, which the tiling reports.
                    self.list_layout.append(earlier[0])
                    size = self.records[f[3]][0]
                    if count is None or count < 0 or pair[1] != count * size:
                        self.problems.append(f"{name}: {pair[1]} bytes for {count} elements of {size}")
                        values[f[1]] = []
                    else:
                        values[f[1]] = copy.deepcopy(earlier[1])
                    continue
                raw = self._pair(pos + 4, ALIGN, None, name)
                offset = self.blocks[-1].offset if raw else 0
                index = len(self.list_layout)
                if raw:
                    self.list_layout.append(-1)
                values[f[1]] = self._elements(f[3], raw, count, name, offset)
                if raw:
                    self._list_pairs[key] = (index, values[f[1]])
            elif kind == "array":
                raw = self._pair(at + f[2], f[5], f[6], name)
                offset = self.blocks[-1].offset if raw else 0
                values[f[1]] = self._elements(f[3], raw, values.get(f[4]), name, offset)
            elif kind == "blob":
                values[f[1]] = self._pair(at + f[2], f[3], f[4], name)
            elif kind == "struct":
                self._pairs(self.records[f[3]][1], at + f[2], name, values[f[1]])

    def tiling(self, start: int) -> int:
        """W11-2 to W11-4: blocks in walk order, then deferred groups, each at the aligned cursor
        with zero gaps; the file ends after the last. Returns the cursor where the walk ended."""
        walk = [b for b in self.blocks if b.group is None]
        groups = sorted({b.group for b in self.blocks if b.group is not None})
        ordered = walk + [b for g in groups for b in self.blocks if b.group == g]
        cursor, walk_end = start, None
        for i, block in enumerate(ordered):
            if i == len(walk):
                walk_end = cursor
            expected = BODY + _align(cursor - BODY, block.align)
            if block.offset != expected:
                self.problems.append(f"tiling: {block.name} at 0x{block.offset:X}, expected 0x{expected:X}")
            elif any(self.data[cursor:expected]):
                self.problems.append(f"tiling: gap before {block.name} is not zero")
            cursor = max(cursor, block.offset + block.size)
        if walk_end is None:
            walk_end = cursor
        if len(self.data) != cursor:
            self.problems.append(f"tiling: file ends at 0x{len(self.data):X}, the last block at 0x{cursor:X}")
        return walk_end


def _read(data: bytes, build: str, records: dict, top: str) -> tuple[TableRead, int]:
    _switch_form(build)
    parse_header(data, build)
    reader = _Reader(bytes(data), build, records)
    root = reader.record(top, BODY, top)
    root["header"] = bytes(data[:HEADER_SIZE])
    if top == "dialog":
        root["text_layout"] = list(reader.text_layout)  # V18-2: the stored sharing, walk order
        root["list_layout"] = list(reader.list_layout)  # V19-1: the same for lists
    walk_end = reader.tiling(BODY + records[top][0])
    blocks = [(b.offset, b.size, b.name) for b in reader.blocks]
    return TableRead(top, root, blocks, reader.problems), walk_end


# --- Writer -------------------------------------------------------------------------------------

@dataclass
class _ListEntry:
    field: str
    at: int  # file position of the block
    size: int
    source: int  # -1 stored here, j shares list j
    value: tuple | None  # normalized items (plan mode only)


def _norm_record(writer, record: str, value: dict) -> tuple:
    """V19-1: the stored content of a record as a comparable value: primitives as their packed
    bytes, texts, nested lists and inline records recursively (missing fields as defaults)."""
    out = []
    for f in writer.records[record][1]:
        kind = f[0]
        if kind == "prim":
            item = value.get(f[1], _default(kind, f[3]))
            try:
                packed = struct.pack(writer.order + f[3], *item) if f[3] in ("2i", "4f") else \
                    struct.pack(writer.order + f[3], item)
            except (struct.error, TypeError):
                packed = repr(item)
            out.append(packed)
        elif kind == "text":
            out.append(value.get(f[1]))
        elif kind == "list":
            out.append(_norm_list(writer, f[3], value.get(f[1]) or []))
        elif kind == "struct":
            out.append(_norm_record(writer, f[3], value.get(f[1]) or {}))
    return tuple(out)


def _norm_list(writer, record: str, items: list) -> tuple:
    return tuple(_norm_record(writer, record, item) for item in items)


class _Writer:
    def __init__(self, build: str, records: dict, derive: bool):
        self.order = build_order(build)
        self.records = records
        self.derive = derive
        self.out = bytearray()
        self.deferred: dict[int, list] = {}
        # V18-2 text sharing: scope (None, "proven" or "all"), an optional stored plan
        # (``text_layout`` of ``parse_ui``), the texts placed so far and the sharing pool
        self.share: str | None = None
        self.plan: list[int] | None = None
        self.plan_ok = True
        self.texts: list[tuple[str, int, int, int]] = []  # (text, file position, size, source)
        self.pool: dict[str, int] = {}
        # V19-1 list sharing: scope (None, "proven" or "all"), an optional stored plan
        # (``list_layout`` of ``parse_ui``), the non-empty lists placed so far and the pool of
        # stored lists by (field, block bytes)
        self.share_lists: str | None = None
        self.list_plan: list[int] | None = None
        self.list_plan_ok = True
        self.lists: list[_ListEntry] = []
        self.list_pool: dict[tuple[str, bytes], int] = {}

    def _state(self) -> tuple:
        return (len(self.out), len(self.texts), dict(self.pool), self.plan_ok,
                len(self.lists), dict(self.list_pool), self.list_plan_ok)

    def _restore(self, state: tuple) -> None:
        out, texts, self.pool, self.plan_ok, lists, self.list_pool, self.list_plan_ok = state
        del self.out[out:]
        del self.texts[texts:]
        del self.lists[lists:]

    def _list(self, record: str, items: list, pos: int, where: str) -> None:
        """V19-1: a list block. With a stored plan each entry is followed (-1 stored, j shares
        list j when j is a stored list of the same field with equal values; a share that is no
        longer valid is stored). Otherwise an eligible list is written, and when its block bytes
        equal a stored list of the same field it is taken back and the pair points there."""
        if not items:
            self._set_pair(pos, None, 0)
            return
        field, i = _list_field(where, record), len(self.lists)
        if self.list_plan is not None:
            if i >= len(self.list_plan):
                self.list_plan_ok = False
            entry = self.list_plan[i] if i < len(self.list_plan) else -1
            value = _norm_list(self, record, items)
            if 0 <= entry < i:
                earlier = self.lists[entry]
                if earlier.source == -1 and earlier.field == field and earlier.value == value:
                    self._set_pair(pos, earlier.at, earlier.size)
                    self.lists.append(_ListEntry(field, earlier.at, earlier.size, entry, value))
                    return
            self.lists.append(_ListEntry(field, 0, 0, -1, value))
            at, size = self._store_list(record, items, pos, where)
            self.lists[i] = _ListEntry(field, at, size, -1, value)
            return
        eligible = (self.share_lists in ("proven", "all") and _PROVEN_LIST.match(where) is not None) or (
            self.share_lists == "all" and _STYLE_LIST.match(where) is not None) or (
            # ``33`` V20-1: string lists of the dialog-level styles take part by default, since
            # inline-style string lists share their blocks (19 / 18 cases in the game files)
            self.share_lists in ("proven", "all") and field == STRING_LISTS
            and _STYLE_LIST.match(where) is not None)
        state = self._state()
        self.lists.append(_ListEntry(field, 0, 0, -1, None))
        at, size = self._store_list(record, items, pos, where)
        raw = bytes(self.out[at:at + size])
        source = self.list_pool.get((field, raw), -1) if eligible else -1
        if source >= 0:
            self._restore(state)
            earlier = self.lists[source]
            self._set_pair(pos, earlier.at, earlier.size)
            self.lists.append(_ListEntry(field, earlier.at, earlier.size, source, None))
            return
        self.lists[i] = _ListEntry(field, at, size, -1, None)
        if eligible:
            self.list_pool.setdefault((field, raw), i)

    def _store_list(self, record: str, items: list, pos: int, where: str) -> tuple[int, int]:
        size = self.records[record][0] * len(items)
        at = self.place(bytes(size), ALIGN)
        self._set_pair(pos, at, size)
        for k, item in enumerate(items):
            self.record(record, item, at + k * self.records[record][0], f"{where}[{k}]")
        return at, size

    def _text(self, text: str | None, pos: int, where: str) -> None:
        if text is None:
            self._set_pair(pos, None, 0)
            return
        raw = text.encode("latin-1") + b"\0"
        i = len(self.texts)
        source = -1
        if self.plan is not None:
            entry = self.plan[i] if i < len(self.plan) else -1
            if i >= len(self.plan):
                self.plan_ok = False
            elif entry >= 0:
                if entry < i and self.texts[entry][0] == text and self.texts[entry][3] == -1:
                    source = entry
                else:
                    self.plan_ok = False
        else:
            eligible = self.share == "all" or (self.share == "proven" and _PROVEN_TEXT.match(where) is not None)
            if eligible:
                source = self.pool.get(text, -1)
                if source < 0:
                    self.pool[text] = i
        if source >= 0:
            at, size = self.texts[source][1], self.texts[source][2]
        else:
            at, size = self.place(raw, ALIGN), len(raw)
        self._set_pair(pos, at, size)
        self.texts.append((text, at, size, source))

    def place(self, raw: bytes, align: int) -> int:
        self.out += bytes(BODY + _align(len(self.out) - BODY, align) - len(self.out))
        at = len(self.out)
        self.out += raw
        return at

    def _pack(self, code: str, at: int, value, name: str) -> None:
        try:
            if code in ("2i", "4f"):
                struct.pack_into(self.order + code, self.out, at, *value)
            else:
                struct.pack_into(self.order + code, self.out, at, value)
        except (struct.error, TypeError) as exc:
            raise FormatError(f"{name}: {value!r} does not fit {code} ({exc})", at) from None

    def record(self, record: str, value: dict, at: int, where: str) -> None:
        fields = self.records[record][1]
        self._fixed(fields, value, at, where)
        self._pairs(fields, value, at, where)

    def _fixed(self, fields, value: dict, at: int, where: str) -> None:
        if self.derive:  # W15-B11: counts are list lengths
            for f in fields:
                if f[0] == "array":
                    value[f[4]] = len(value.get(f[1]) or [])
        for f in fields:
            kind = f[0]
            if kind == "prim":
                self._pack(f[3], at + f[2], value.get(f[1], _default(kind, f[3])), f"{where}.{f[1]}")
            elif kind == "char":
                raw = (value.get(f[1]) or "").encode("latin-1")
                if len(raw) >= f[3]:
                    raise FormatError(f"{where}.{f[1]}: {len(raw)} characters, at most {f[3] - 1}", at + f[2])
                self.out[at + f[2]:at + f[2] + len(raw)] = raw
            elif kind == "list":
                self._pack("i", at + f[2], len(value.get(f[1]) or []), f"{where}.{f[1]}")
            elif kind == "struct":
                self._fixed(self.records[f[3]][1], value.setdefault(f[1], {}), at + f[2], f"{where}.{f[1]}")

    def _set_pair(self, pos: int, at: int | None, size: int) -> None:
        struct.pack_into(self.order + "II", self.out, pos, 0 if not size else at - BODY, size)

    def _array(self, record: str, items: list, pos: int, align: int, where: str) -> None:
        size = self.records[record][0]
        if not items:
            self._set_pair(pos, None, 0)
            return
        at = self.place(bytes(size * len(items)), align)
        self._set_pair(pos, at, size * len(items))
        for i, item in enumerate(items):
            self.record(record, item, at + i * size, f"{where}[{i}]")

    def _blob(self, raw: bytes | None, pos: int, align: int) -> None:
        raw = bytes(raw or b"")
        self._set_pair(pos, self.place(raw, align) if raw else None, len(raw))

    def _pairs(self, fields, value: dict, at: int, where: str) -> None:
        for f in fields:
            kind = f[0]
            if kind == "text":
                self._text(value.get(f[1]), at + f[2], where)
            elif kind == "list":
                self._list(f[3], value.get(f[1]) or [], at + f[2] + 4, f"{where}.{f[1]}")
            elif kind == "array":
                args = (f[3], value.get(f[1]) or [], at + f[2], f[5], f"{where}.{f[1]}")
                if f[6] is None:
                    self._array(*args)
                else:
                    self.deferred.setdefault(f[6], []).append(lambda a=args: self._array(*a))
            elif kind == "blob":
                args = (value.get(f[1]), at + f[2], f[3])
                if f[4] is None:
                    self._blob(*args)
                else:
                    self.deferred.setdefault(f[4], []).append(lambda a=args: self._blob(*a))
            elif kind == "struct":
                self._pairs(self.records[f[3]][1], value[f[1]], at + f[2], f"{where}.{f[1]}")

    def run(self, top: str, value: dict, header: bytes) -> int:
        """Lay the file out; returns the body offset where the deferred blocks start."""
        if len(header) != HEADER_SIZE:
            raise FormatError(f"header of {len(header)} bytes, expected {HEADER_SIZE}")
        self.out = bytearray(header) + bytes(BODY + self.records[top][0] - HEADER_SIZE)
        self.record(top, value, BODY, top)
        walk_end = len(self.out) - BODY
        for group in sorted(self.deferred):
            for job in self.deferred[group]:
                job()
        return walk_end


def _root(value) -> dict:
    return value.root if isinstance(value, TableRead) else value


# --- W15-B8. UI -----------------------------------------------------------------------------------

def parse_ui(data: bytes, build: str) -> TableRead:
    """W15-B8: a dialog of any build. ps3 goes to ``ui_font.parse_ps3_ui``; switch and x360 use
    the switch form. Problems: non-zero zero fields and slots, list sizes, texts, block tiling
    (W11-2 order, 8-alignment, zero gaps, nothing after the last block) and the animation rules
    (V7-T5a). V18-2: a text pair that repeats the offset and size of an earlier text pair shares
    that text and is not a tiling error; ``root["text_layout"]`` records the sharing (texts in
    walk order: -1 stored there, j points at text j) so ``write_ui`` can reproduce it. V19-1: a
    list pair that repeats the offset and size of a stored list pair of the same field shares
    that list (not walked again, no tiling error); a repeat across fields or of another size is
    still a tiling error. ``root["list_layout"]`` records it (non-empty lists in walk order)."""
    if build == "ps3":
        from .ui_font import parse_ps3_ui
        return parse_ps3_ui(data)
    read, _ = _read(data, build, UI_RECORDS, "dialog")
    for i, control in enumerate(read.root.get("controls", [])):
        read.problems += animation_problems(control, f"control {i}")
    return read


def write_ui(dialog, build: str, share_texts: str = "proven", keep_layout: bool = True,
             share_lists: str = "proven") -> bytes:
    """W15-B8: lay a switch-form dialog out. List counts are the list lengths; missing fields are
    zero, missing texts empty pairs; ``header`` (0x1C bytes) defaults to ``new_header``.

    V18-2: a text equal to an earlier text of the dialog is not stored again; its pair repeats the
    earlier pair. Each distinct text is stored at its first occurrence in the W11-2 walk, and
    sharing crosses fields. ``share_texts`` picks the texts that share: "proven" (control names,
    parent paths, inline-style texts; the default), "all" (every text field, HYPOTHESIS) or
    "none" (stage-15 behaviour). With ``keep_layout`` a ``text_layout`` from ``parse_ui`` is
    followed when it is valid for the dialog (same number of texts, every shared text equal to
    the text it points at); otherwise ``share_texts`` decides.

    V19-1: a list whose block bytes equal a stored list of the same field (``_list_field``) is
    not stored again; its pair repeats the earlier pair (first occurrence in the walk; sharing
    never crosses fields). ``share_lists``: "proven" (the control fields; the default), "all"
    (also dialog-level styles, HYPOTHESIS) or "none". With ``keep_layout`` a ``list_layout``
    from ``parse_ui`` is followed per entry when it has one entry per non-empty list: -1 is
    stored (this keeps an equal list stored twice), j shares list j when that is a stored list
    of the same field with equal values, and a share that is no longer valid is stored. A
    layout with another number of entries is dropped and ``share_lists`` decides. The text and
    list layouts are tried together, then the list layout alone, then the text layout alone,
    then neither."""
    _switch_form(build)
    if share_texts not in TEXT_SHARING:
        raise FormatError(f"share_texts {share_texts!r}, expected one of {TEXT_SHARING}")
    if share_lists not in LIST_SHARING:
        raise FormatError(f"share_lists {share_lists!r}, expected one of {LIST_SHARING}")
    root = _root(dialog)
    header = root.get("header") or new_header(build, "ui")
    text_plan = root.get("text_layout") if keep_layout else None
    list_plan = root.get("list_layout") if keep_layout else None
    tried = []
    for plans in ((text_plan, list_plan), (None, list_plan), (text_plan, None), (None, None)):
        if plans in tried:
            continue
        tried.append(plans)
        writer = _Writer(build, UI_RECORDS, derive=False)
        if plans[0] is not None:
            writer.plan = list(plans[0])
        else:
            writer.share = None if share_texts == "none" else share_texts
        if plans[1] is not None:
            writer.list_plan = list(plans[1])
        else:
            writer.share_lists = None if share_lists == "none" else share_lists
        writer.run("dialog", root, header)
        texts_ok = writer.plan is None or (writer.plan_ok and len(writer.texts) == len(writer.plan))
        lists_ok = writer.list_plan is None or (writer.list_plan_ok and len(writer.lists) == len(writer.list_plan))
        if texts_ok and lists_ok:
            return bytes(writer.out)
    raise AssertionError("unreachable: the write without layouts is always valid")


def text_layout(dialog, build: str = "switch", share_texts: str = "proven") -> list[int]:
    """V18-2: the text sharing ``write_ui`` gives the dialog, in walk order (-1 stored, j shares
    the block of text j)."""
    return parse_ui(write_ui(dialog, build, share_texts), build).root["text_layout"]


def list_layout(dialog, build: str = "switch", share_lists: str = "proven",
                share_texts: str = "proven") -> list[int]:
    """V19-1: the list sharing ``write_ui`` gives the dialog: one entry per non-empty list in
    walk order, -1 stored, j shares the block of list j (same field)."""
    return parse_ui(write_ui(dialog, build, share_texts, share_lists=share_lists), build).root["list_layout"]


def verify_ui(data: bytes, build: str) -> list[str]:
    try:
        return parse_ui(data, build).problems
    except FormatError as exc:
        return [f"layout: {exc}"]


# --- W15-B9. UI keys and links ------------------------------------------------------------------

FNV64_BASIS = 0xCBF29CE484222325
FNV64_PRIME = 0x100000001B3
_LOWER = bytes(c + 32 if 65 <= c <= 90 else c for c in range(256))


def ui_key(text: str | bytes) -> int:
    """Hash64: 64-bit FNV-1a over the bytes (latin-1) with A..Z lower-cased."""
    raw = text.encode("latin-1") if isinstance(text, str) else bytes(text)
    value = FNV64_BASIS
    for byte in raw.translate(_LOWER):
        value = ((value ^ byte) * FNV64_PRIME) & 0xFFFFFFFFFFFFFFFF
    return value


def control_path(control: dict) -> str:
    """Parent path + "." + name; a control with an empty parent path (the root) is its name."""
    parent, name = control.get("parent_name") or "", control.get("name") or ""
    return f"{parent}.{name}" if parent else name


def control_paths(dialog) -> list[str]:
    return [control_path(c) for c in _root(dialog).get("controls", [])]


def _keep_ordinals(controls: list[dict]) -> None:
    """V18-4: stored ordinals are kept. A control without a usable ordinal (missing, outside
    0..n-1, or repeating an earlier control's) gets the lowest free value, in list order; so a
    dialog without ordinals gets 0..n-1 and the result is always a permutation."""
    n = len(controls)
    used: set[int] = set()
    missing = []
    for control in controls:
        value = control.get("ordinal")
        if isinstance(value, int) and 0 <= value < n and value not in used:
            used.add(value)
        else:
            missing.append(control)
    free = iter(v for v in range(n) if v not in used)
    for control in missing:
        control["ordinal"] = next(free)


def ordinal_problems(dialog) -> list[str]:
    """V18-4: the stored ordinals are a permutation of 0..n-1 (the only ordinal rule; which
    permutation is OPEN)."""
    controls = _root(dialog).get("controls", [])
    ordinals = [c.get("ordinal") for c in controls]
    if sorted(o for o in ordinals if isinstance(o, int)) == list(range(len(controls))):
        return []
    return [f"ordinals {ordinals} are not a permutation of 0..{len(controls) - 1}"]


def derive_ui_keys(dialog):
    """W15-B9 / V18-4: keep the stored ordinals (only missing, out-of-range or repeated ones get
    the free values, ``_keep_ordinals``); set control keys (Hash64 of the path), parent keys
    (Hash64 of the parent path, 0 for the root), style, inline-style and description keys (Hash64
    of the name). A style or description without a name keeps its key. Returns the dialog."""
    root = _root(dialog)
    _keep_ordinals(root.get("controls", []))
    for control in root.get("controls", []):
        control["key"] = ui_key(control_path(control))
        parent = control.get("parent_name") or ""
        control["parent_key"] = ui_key(parent) if parent else 0
        styles = [control.get("inline_style")] if control.get("inline_style") is not None else []
        for item in styles + list(control.get("anim_descriptions", [])):
            if item.get("name") is not None:
                item["key"] = ui_key(item["name"])
    for style in root.get("styles", []):
        if style.get("name") is not None:
            style["key"] = ui_key(style["name"])
    return dialog


def binding_target(name: str, dialog) -> tuple[int, int] | None:
    """(control index, description index) a binding name "^" + path + "." + description names in
    this dialog, or None."""
    if not name or not name.startswith(BINDING_PREFIX):
        return None
    for i, control in enumerate(_root(dialog).get("controls", [])):
        path = control_path(control) + "."
        if name[1:].startswith(path):
            rest = name[1 + len(path):]
            for d, description in enumerate(control.get("anim_descriptions", [])):
                if description.get("name") == rest:
                    return i, d
    return None


def append_control(dialog, name: str, parent_path: str = "", control_type: int = 0,
                   inline_style: dict | None = None) -> dict:
    """W15-B9 / V18-4: append a control (the stored order is kept) with its keys and ordinal n,
    the number of controls before it. A stored ``text_layout`` and ``list_layout`` are dropped
    (the walk changes; the scopes of ``write_ui`` decide the sharing)."""
    root = _root(dialog)
    control = {"name": name, "parent_name": parent_path or None, "control_type": control_type,
               "inline_style": inline_style or {}}
    controls = root.setdefault("controls", [])
    control["ordinal"] = len(controls)
    controls.append(control)
    root.pop("text_layout", None)
    root.pop("list_layout", None)  # V19-1: the list walk changes too
    control["key"] = ui_key(control_path(control))
    control["parent_key"] = ui_key(parent_path) if parent_path else 0
    style = control["inline_style"]
    if style.get("name") is not None:
        style["key"] = ui_key(style["name"])
    return control


def remove_control(dialog, which: int | str) -> dict:
    """V18-4: remove one control (list index or path) and lower every ordinal above the removed
    one by 1, which keeps the permutation (HYPOTHESIS: the rule for removal is not observed).
    The control's children are not removed: remove them first, or ``ui_link_problems`` reports
    their parent path. A stored ``text_layout`` and ``list_layout`` are dropped. Returns the
    removed control."""
    root = _root(dialog)
    controls = root.get("controls", [])
    if isinstance(which, str):
        paths = control_paths(root)
        if which not in paths:
            raise FormatError(f"no control with path {which!r}")
        which = paths.index(which)
    if not 0 <= which < len(controls):
        raise FormatError(f"control {which} of {len(controls)}")
    removed = controls.pop(which)
    gone = removed.get("ordinal")
    if isinstance(gone, int):
        for control in controls:
            if isinstance(control.get("ordinal"), int) and control["ordinal"] > gone:
                control["ordinal"] -= 1
    root.pop("text_layout", None)
    root.pop("list_layout", None)  # V19-1: the list walk changes too
    return removed


def ui_link_problems(dialog, parents: bool = True, bindings: bool = False) -> list[str]:
    """W15-B9 checks: ordinals a permutation of 0..n-1 (V18-4, not the list order), keys, one
    root, animation ranges (V7-T5a); with ``parents`` every parent path is the path of a control
    of the dialog; with ``bindings`` (SUPPORTED: a few bindings name other dialogs) every binding
    resolves in the dialog."""
    root = _root(dialog)
    found = ordinal_problems(root)
    controls = root.get("controls", [])
    paths = control_paths(root)
    roots = [i for i, c in enumerate(controls) if not c.get("parent_name")]
    if controls and len(roots) != 1:
        found.append(f"{len(roots)} root controls, expected 1")
    for i, control in enumerate(controls):
        where = f"control {i}"
        if control.get("key") != ui_key(paths[i]):
            found.append(f"{where}: key is not Hash64({paths[i]!r})")
        parent = control.get("parent_name") or ""
        if control.get("parent_key") != (ui_key(parent) if parent else 0):
            found.append(f"{where}: parent key is not Hash64({parent!r})")
        if parents and parent and parent not in paths:
            found.append(f"{where}: parent path {parent!r} names no control")
        style = control.get("inline_style") or {}
        if style.get("name") is not None and style.get("key") != ui_key(style["name"]):
            found.append(f"{where}: inline style key is not Hash64({style['name']!r})")
        for d, description in enumerate(control.get("anim_descriptions", [])):
            if description.get("name") is not None and description.get("key") != ui_key(description["name"]):
                found.append(f"{where} description {d}: key is not Hash64({description['name']!r})")
        found += animation_problems(control, where)
        if bindings:
            for b, binding in enumerate(control.get("anim_bindings", [])):
                if binding_target(binding.get("animation_name"), root) is None:
                    found.append(f"{where} binding {b}: {binding.get('animation_name')!r} names no description")
    for s, style in enumerate(root.get("styles", [])):
        if style.get("name") is not None and style.get("key") != ui_key(style["name"]):
            found.append(f"style {s}: key is not Hash64({style['name']!r})")
    return found


# --- W15-B10. Fonts -----------------------------------------------------------------------------

def parse_font(data: bytes, build: str) -> TableRead:
    """W15-B10: a font of any build (ps3 goes to ``ui_font.parse_ps3_font``). Problems: zero
    fields and slots, counts against list sizes, tiling with the deferred groups, the custom-data
    offset, and ``ui_font.font_problems`` (W15-B11 included)."""
    from .ui_font import font_problems, parse_ps3_font
    if build == "ps3":
        return parse_ps3_font(data)
    read, walk_end = _read(data, build, font_records(build), "font")
    root = read.root
    if root["custom_data_offset"] != walk_end - BODY:
        read.problems.append(f"custom-data offset 0x{root['custom_data_offset']:X} is not the first "
                             f"deferred block 0x{walk_end - BODY:X}")
    read.problems += font_problems(root, build=build)
    return read


def write_font(font, build: str, derive: bool = True) -> bytes:
    """W15-B10: lay a switch-form font out. With ``derive`` the counts are the list lengths and
    the custom-data offset is where the deferred blocks start; without, the stored values are
    written. Glyph and kerning order are not changed here (``derive_font_values``)."""
    root = _root(font)
    records = font_records(build)
    header = root.get("header") or new_header(build, "fnt")
    writer = _Writer(build, records, derive)
    walk_end = writer.run("font", root, header)
    if derive:
        root["custom_data_offset"] = walk_end
        struct.pack_into(writer.order + "I", writer.out, BODY + 0xD0, walk_end)
    return bytes(writer.out)


def verify_font(data: bytes, build: str) -> list[str]:
    try:
        return parse_font(data, build).problems
    except FormatError as exc:
        return [f"layout: {exc}"]


# Font images --------------------------------------------------------------------------------------

FONT_TEXTURE_FORMAT = 23  # one byte per pixel; switch container pixel format code 1
SWITCH_PIXEL_FORMAT = 1
X360_TILE = 32


def x360_font_pitch(width: int) -> int:
    """max(width, 32), rounded up to whole 32-pixel tiles."""
    return max(_align(width, X360_TILE), X360_TILE)


def font_image_size(width: int, height: int, build: str) -> int:
    if build == "ps3":
        return width * height
    if build == "x360":
        # the library's surface size (T3): whole 32 x 32 tiles, rounded up to 4096 bytes
        return XT.surface_bytes("a8", 1, x360_font_pitch(width), height)
    if build == "switch":
        return len(TW.switch_container(FONT_TEXTURE_FORMAT, width, height, 1,
                                       bytes(TW.switch_layout(FONT_TEXTURE_FORMAT, width, height, 1).image_size)))
    raise FormatError(f"unknown build {build!r}")


def _x360_offsets(width: int, height: int) -> list[int]:
    pitch = x360_font_pitch(width)
    return [XT.tiled_element(x, y, pitch, 1) for y in range(height) for x in range(width)]


def _switch_offsets(blob: bytes, width: int, height: int) -> tuple[list[int], bytes]:
    nv = T._parse_nv(blob, 0)
    if nv.pixel_format != SWITCH_PIXEL_FORMAT or (nv.width, nv.height, nv.depth) != (width, height, 1):
        raise FormatError(f"container {nv.width}x{nv.height}x{nv.depth} format {nv.pixel_format}, "
                          f"expected {width}x{height} format {SWITCH_PIXEL_FORMAT}")
    if nv.mip_count != 1 or nv.target != T.NV_TARGET_2D:
        raise FormatError(f"container with {nv.mip_count} levels, target {nv.target}")
    bh = T._level_block_height(nv, height)
    gob_width, x_part, y_part = T._gob_tables(width, height, 1, bh)
    offsets = [((row * gob_width + gob_x) * bh + gob_y) * 512 + y_in + x_in
               for row, gob_y, y_in in y_part for gob_x, x_in in x_part]
    return offsets, nv.image


def font_image_bytes(pixels: bytes, width: int, height: int, build: str) -> bytes:
    """W15-B10: the stored image of ``width`` x ``height`` one-byte pixels (rows in order):
    ps3 the rows; switch the W14-10 container (one 2D level, pixel format code 1); x360 the pixels
    at their tiled addresses (pitch ``x360_font_pitch``), zero elsewhere."""
    pixels = bytes(pixels)
    if len(pixels) != width * height:
        raise FormatError(f"{len(pixels)} pixels for {width} x {height}")
    if build == "ps3":
        return pixels
    if build == "switch":
        return TW.switch_payload(FONT_TEXTURE_FORMAT, width, height, 1, {(0, 0): pixels})
    if build == "x360":
        out = bytearray(font_image_size(width, height, build))
        for value, at in zip(pixels, _x360_offsets(width, height)):
            out[at] = value
        return bytes(out)
    raise FormatError(f"unknown build {build!r}")


def font_image_pixels(blob: bytes, width: int, height: int, build: str) -> bytes:
    """The pixels of a stored font image, rows in order (inverse of ``font_image_bytes``)."""
    blob = bytes(blob)
    if build == "ps3":
        if len(blob) != width * height:
            raise FormatError(f"image of {len(blob)} bytes for {width} x {height}")
        return blob
    if build == "switch":
        offsets, image = _switch_offsets(blob, width, height)
    elif build == "x360":
        offsets, image = _x360_offsets(width, height), blob
    else:
        raise FormatError(f"unknown build {build!r}")
    if offsets and max(offsets) >= len(image):
        raise FormatError(f"image of {len(image)} bytes is too short for {width} x {height}")
    return bytes(image[at] for at in offsets)


def font_image_problems(blob: bytes, width: int, height: int, build: str) -> list[str]:
    """The stored image is exactly what ``font_image_bytes`` makes of its pixels."""
    if build == "ps3":
        n = len(blob)
        return [] if n == width * height else [f"image of {n} bytes for {width} x {height}"]
    try:
        pixels = font_image_pixels(blob, width, height, build)
    except FormatError as exc:
        return [f"image: {exc.message}"]
    if font_image_bytes(pixels, width, height, build) != bytes(blob):
        return [f"image of {len(blob)} bytes is not the {build} form of {width} x {height}"]
    return []


# --- W15-B11. Font derived values ---------------------------------------------------------------

LINE_BREAK = {"texture_handle": -2, "texture_index": -1, "frame": -1}


def _whole(value: float, what: str) -> int:
    rounded = round(value)
    if abs(value - rounded) > 1e-3:
        raise FormatError(f"{what} {value} is not a whole number of pixels")
    return int(rounded)


def frame_pixel_size(frame_rect, texture: dict) -> tuple[int, int]:
    """(width, height) in pixels of a frame (u0, v0, u1, v1) on ``texture``."""
    u0, v0, u1, v1 = frame_rect
    return (_whole((u1 - u0) * texture["width"], "frame width"),
            _whole((v1 - v0) * texture["height"], "frame height"))


def glyph_height(frame_rect, texture: dict, build: str = "switch") -> int:
    """Frame pixel height - 1; on ps3 in 26.6 fixed point (times 64)."""
    pixels = frame_pixel_size(frame_rect, texture)[1] - 1
    return pixels * 64 if build == "ps3" else pixels


def derive_glyph(frame_rect, texture: dict, char: int, frame: int, texture_index: int,
                 bearing_x: int = 0, bearing_y: int = 0, advance: int = 0, texture_handle: int = -1,
                 build: str = "switch") -> dict:
    """W15-B11: a glyph whose height follows from its frame; bearings and advance are the
    rasteriser's (inputs, pixels on switch and x360, 26.6 on ps3)."""
    return {"texture_handle": texture_handle, "frame": frame, "bearing_x": bearing_x,
            "bearing_y": bearing_y, "height": glyph_height(frame_rect, texture, build),
            "advance": advance, "texture_index": texture_index, "char": char}


def line_break_glyph(char: int, advance: int = 0, bearing_x: int = 0, bearing_y: int = 0, height: int = 0) -> dict:
    """W15-B11: the one line-break glyph of a point size (texture handle -2, index -1, frame -1)."""
    return {"bearing_x": bearing_x, "bearing_y": bearing_y, "height": height, "advance": advance,
            "char": char, **LINE_BREAK}


def is_line_break(glyph: dict) -> bool:
    return all(glyph.get(k) == v for k, v in LINE_BREAK.items())


def _kerning_key(kerning: dict) -> tuple[int, int]:
    return kerning["previous_char"] & 0xFFFFFFFF, kerning["char"] & 0xFFFFFFFF


def derive_font_values(font) -> dict:
    """W15-B11 / V18-5: glyphs in non-decreasing character order (a stable sort, so repeated
    characters keep their stored order and a non-decreasing list is not touched), kernings
    sorted (W14-23), counts as list lengths. Returns the font dict."""
    root = _root(font)
    textures, sizes = root.setdefault("textures", []), root.setdefault("point_sizes", [])
    for texture in textures:
        texture["frame_count"] = len(texture.get("frames") or [])
    for size in sizes:
        glyphs = list(size.get("glyphs") or [])
        if any(glyphs[k]["char"] < glyphs[k - 1]["char"] for k in range(1, len(glyphs))):
            glyphs.sort(key=lambda g: g["char"])  # list.sort is stable
        size["glyphs"] = glyphs
        size["kernings"] = sorted(size.get("kernings") or [], key=_kerning_key)
        size["glyph_count"], size["kerning_count"] = len(size["glyphs"]), len(size["kernings"])
    root["texture_count"], root["point_size_count"] = len(textures), len(sizes)
    return root


def glyph_order_problems(glyphs: list[dict], build: str, strict: bool | None = None,
                         where: str = "glyphs") -> list[str]:
    """V18-5: glyph characters never go down. A character lower than the one before it is
    always reported. A repeated character is allowed on switch (two glyphs, different frames);
    ps3 and x360 ascend strictly, so with ``strict`` (default: on for ps3 and x360, off for
    switch) a repeat is reported too."""
    if strict is None:
        strict = build != "switch"
    found = []
    chars = [g["char"] for g in glyphs]
    for k in range(1, len(chars)):
        if chars[k] < chars[k - 1]:
            found.append(f"{where}: glyph {k} (char {chars[k]}) is not after char {chars[k - 1]} (lower)")
        elif strict and chars[k] == chars[k - 1]:
            found.append(f"{where}: glyph {k} (char {chars[k]}) is not after char {chars[k - 1]} "
                         f"(repeated; {build} ascends strictly)")
    return found


def font_derived_problems(font, build: str, strict_order: bool | None = None) -> list[str]:
    """W15-B11 checks: counts are list lengths; glyph order (V18-5, ``glyph_order_problems``);
    one line-break glyph per
    point size; every other glyph has a texture index and frame (range: ``ui_font.font_problems``),
    a texture with the point size's resolution flags and height = frame pixel height - 1;
    kernings sorted by (first, second)."""
    root = _root(font)
    found = []
    textures = root.get("textures", [])
    for key, items in (("texture_count", textures), ("point_size_count", root.get("point_sizes", []))):
        if key in root and root[key] != len(items):
            found.append(f"{key} {root[key]} for {len(items)}")
    for i, texture in enumerate(textures):
        if "frame_count" in texture and texture["frame_count"] != len(texture.get("frames", [])):
            found.append(f"texture {i}: frame count {texture['frame_count']} for {len(texture.get('frames', []))}")
    for p, size in enumerate(root.get("point_sizes", [])):
        where = f"point size {p}"
        glyphs, kernings = size.get("glyphs", []), size.get("kernings", [])
        for key, items in (("glyph_count", glyphs), ("kerning_count", kernings)):
            if key in size and size[key] != len(items):
                found.append(f"{where}: {key} {size[key]} for {len(items)}")
        found += glyph_order_problems(glyphs, build, strict_order, where)
        breaks = sum(1 for g in glyphs if is_line_break(g))
        if glyphs and breaks != 1:
            found.append(f"{where}: {breaks} line-break glyphs, expected 1")
        for g, glyph in enumerate(glyphs):
            if is_line_break(glyph):
                continue
            index, frame = glyph["texture_index"], glyph["frame"]
            if index == -1 or frame == -1:
                found.append(f"{where} glyph {g}: no texture index or frame")
                continue
            if not 0 <= index < len(textures):
                continue
            texture = textures[index]
            if texture.get("resolution_flags") != size.get("resolution_flags"):
                found.append(f"{where} glyph {g}: texture {index} has resolution flags "
                             f"{texture.get('resolution_flags')}, the point size {size.get('resolution_flags')}")
            frames = texture.get("frames", [])
            if not 0 <= frame < len(frames):
                continue
            try:
                expected = glyph_height(frames[frame]["u0_v0_u1_v1"], texture, build)
            except FormatError as exc:
                found.append(f"{where} glyph {g}: {exc.message}")
                continue
            if glyph["height"] != expected:
                found.append(f"{where} glyph {g}: height {glyph['height']}, the frame gives {expected}")
        keys = [_kerning_key(k) for k in kernings]
        bad = next((k for k in range(1, len(keys)) if keys[k] < keys[k - 1]), None)
        if bad is not None:
            found.append(f"{where}: kerning pair {bad} {keys[bad]} sorts before {keys[bad - 1]}")
    return found


__all__ = [
    "GLYPH_RECORDS", "UI_RECORDS", "append_control", "binding_target", "control_path",
    "control_paths", "derive_font_values", "derive_glyph", "derive_ui_keys", "font_derived_problems",
    "font_image_bytes", "font_image_pixels", "font_image_problems", "font_image_size", "font_records",
    "frame_pixel_size", "glyph_height", "glyph_order_problems", "is_line_break", "line_break_glyph",
    "new_header", "ordinal_problems", "parse_font", "parse_ui", "record_size", "remove_control",
    "text_layout", "TEXT_SHARING", "ui_key", "ui_link_problems", "verify_font", "verify_ui",
    "write_font", "write_ui", "x360_font_pitch",
]
