"""PS3 ``.ui`` and font files, read field by field from ``data/ps3_ui_font_layout.tsv`` (stage 7).

Offsets in the table are relative to the record start; the top records (``ui_dialog``, ``font``)
start at the body (file position 0x10). Field kinds:

- ``prim``: a value (``u32``, ``i32``, ``u64``, ``f32``, ``char[64]``, ``2i32``, ``4f32``);
- ``struct``: an inline record (the definition header is skipped);
- ``list``: a 16-byte list (count, pair, slot); its element size is the element record's size;
- ``pair``: ``text`` (a NUL-terminated string), ``array:<record>`` (counted by the field
  ``<name>_count``), ``blob`` (raw bytes);
- ``ptr``: a 4-byte run-time slot, must be zero; ``zero``: must be zero;
- ``hole``: alignment bytes holding stale memory; ignored, never checked (T5).

Reading raises ``FormatError`` only where a read leaves the file. Every other rule is reported
in ``problems``.
"""

from __future__ import annotations

import csv
import struct
from dataclasses import dataclass, field
from functools import lru_cache

from .binary import ByteView, FormatError
from .datafiles import DATA
from .sno import parse_header
from .structure import tiling_problems

BODY = 0x10


@dataclass(frozen=True)
class UiFontField:
    record: str
    offset: int
    size: int
    kind: str
    type: str
    name: str
    note: str


# Animation elements have no rows of their own in the table; their notes give the layout.
_EXTRA = {
    "ui_anim_int32": (("time", 0, 4, "f32"), ("value", 4, 4, "i32")),
    "ui_anim_float": (("time", 0, 4, "f32"), ("value", 4, 4, "f32")),
    "ui_anim_vector2": (("time", 0, 4, "f32"), ("x", 4, 4, "f32"), ("y", 8, 4, "f32")),
}


@lru_cache(maxsize=None)
def ui_font_layouts() -> dict[str, tuple[UiFontField, ...]]:
    """Every record of the table, plus the three animation element records."""
    found: dict[str, list[UiFontField]] = {}
    with open(DATA / "ps3_ui_font_layout.tsv", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            found.setdefault(row["record"], []).append(UiFontField(
                row["record"], int(row["offset"], 16), int(row["bytes"]), row["kind"],
                row["type"], row["name"], row["note"] or "",
            ))
    for record, fields in _EXTRA.items():
        found[record] = [UiFontField(record, at, size, "prim", kind, name, "") for name, at, size, kind in fields]
    return {record: tuple(fields) for record, fields in found.items()}


def record_size(record: str) -> int:
    return max(f.offset + f.size for f in ui_font_layouts()[record])


_PRIM = {"u32": "I", "i32": "i", "u64": "Q", "f32": "f", "2i32": "2i", "4f32": "4f"}


@dataclass
class TableRead:
    """A file read by the table: the top record as nested dicts, blocks and problems."""

    record: str
    root: dict
    blocks: list[tuple[int, int, str]] = field(default_factory=list)  # file offset, size, name
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


class _Reader:
    def __init__(self, data: bytes):
        self.data = data
        self.view = ByteView(data, ">")
        self.blocks: list[tuple[int, int, str]] = []
        self.problems: list[str] = []

    def record(self, record: str, at: int, where: str) -> dict:
        self.view.need(at, record_size(record), record)
        values: dict = {}
        for f in ui_font_layouts()[record]:
            pos = at + f.offset
            name = f"{where}.{f.name}"
            if f.kind == "hole" or (f.kind == "struct" and f.name == "definition_header"):
                continue
            if f.kind == "struct":
                values[f.name] = self.record(f.type, pos, name)
            elif f.kind == "prim":
                values[f.name] = self._prim(f, pos, name)
            elif f.kind in ("ptr", "zero"):
                if any(self.view.take(pos, f.size)):
                    self.problems.append(f"{name}: {f.kind} bytes at 0x{pos:X} are not zero")
            elif f.kind == "list":
                values[f.name] = self._list(f.type, pos, name)
            elif f.kind == "pair":
                values[f.name] = self._pair(f, pos, name, values)
        return values

    def _prim(self, f: UiFontField, pos: int, name: str):
        if f.type.startswith("char["):
            raw = self.view.take(pos, f.size, name)
            end = raw.find(b"\0")
            if end < 0:
                self.problems.append(f"{name}: no NUL in {f.size} characters")
                return raw.decode("latin-1")
            if any(raw[end:]):
                self.problems.append(f"{name}: bytes after the NUL are not zero")
            return raw[:end].decode("latin-1")
        values = self.view.unpack(_PRIM[f.type], pos, name)
        return values[0] if len(values) == 1 else values

    def _text(self, pos: int, name: str) -> str | None:
        pair = self.view.pair(pos, name)
        raw = self.view.pair_data(pair, pos, name)
        if not raw:
            return None
        self.blocks.append((pair.file_offset, pair.size, name))
        if raw[-1] != 0 or raw.count(0) != 1:
            self.problems.append(f"{name}: text is not one NUL-terminated string")
        return raw.split(b"\0")[0].decode("latin-1")

    def _elements(self, record: str, raw_at: int, count: int, name: str) -> list[dict]:
        size = record_size(record)
        return [self.record(record, raw_at + i * size, f"{name}[{i}]") for i in range(count)]

    def _list(self, record: str, pos: int, name: str) -> list[dict]:
        count = self.view.i32(pos, name)
        pair = self.view.pair(pos + 4, name)
        raw = self.view.pair_data(pair, pos + 4, name)
        if self.view.u32(pos + 12, name):
            self.problems.append(f"{name}: list slot is not zero")
        size = record_size(record)
        if count < 0 or len(raw) != count * size:
            self.problems.append(f"{name}: {len(raw)} bytes for {count} elements of {size}")
            return []
        if raw:
            self.blocks.append((pair.file_offset, pair.size, name))
        return self._elements(record, pair.file_offset, count, name)

    def _pair(self, f: UiFontField, pos: int, name: str, values: dict):
        if f.type == "text":
            return self._text(pos, name)
        pair = self.view.pair(pos, name)
        raw = self.view.pair_data(pair, pos, name)
        if raw:
            self.blocks.append((pair.file_offset, pair.size, name))
        if f.type == "blob":
            return raw
        record = f.type.split(":", 1)[1]
        size = record_size(record)
        count = values.get(f.name[:-1] + "_count")
        if count is None or len(raw) != count * size:
            self.problems.append(f"{name}: {len(raw)} bytes for {count} elements of {size}")
            return []
        return self._elements(record, pair.file_offset, count, name)


def _top(data: bytes, record: str) -> TableRead:
    parse_header(data, "ps3")
    reader = _Reader(data)
    root = reader.record(record, BODY, record)
    reader.problems += tiling_problems(len(data), BODY + record_size(record), reader.blocks, data)
    return TableRead(record, root, reader.blocks, reader.problems)


def parse_ps3_ui(data: bytes) -> TableRead:
    """A PS3 ``.ui`` dialog: machine time, controls, styles and the three constant lists."""
    read = _top(data, "ui_dialog")
    for i, control in enumerate(read.root.get("controls", [])):
        read.problems += animation_problems(control, f"control {i}")
    return read


# V7-T5a: a description's ``type`` selects one of the control's value lists.
ANIMATION_LISTS = {0: "anim_int32s", 1: "anim_floats", 2: "anim_vector2s"}
BINDING_PREFIX = "^"


def animation_problems(control: dict, where: str = "control") -> list[str]:
    """The animation rules of one control (V7-T5a): each description selects a value list by
    its type; its ``node_range`` (first, end) is a half-open range inside that list; the ranges
    of the descriptions selecting one list cover it exactly once; element times do not decrease
    within a range; binding names start with ``^``. Whether a range starts at time 0.0 is not
    checked (4 of 381 do not)."""
    found = []
    covered: dict[str, list[int]] = {name: [0] * len(control.get(name, [])) for name in ANIMATION_LISTS.values()}
    for d, description in enumerate(control.get("anim_descriptions", [])):
        label = f"{where} description {d}"
        name = ANIMATION_LISTS.get(description.get("type"))
        if name is None:
            found.append(f"{label}: type {description.get('type')} selects no value list")
            continue
        first, end = description["node_range"]
        elements = control.get(name, [])
        if not 0 <= first <= end <= len(elements):
            found.append(f"{label}: range {first}..{end} outside {name} of {len(elements)}")
            continue
        for k in range(first, end):
            covered[name][k] += 1
        times = [elements[k]["time"] for k in range(first, end)]
        if any(b < a for a, b in zip(times, times[1:])):
            found.append(f"{label}: times decrease within the range")
    for name, counts in covered.items():
        if any(count != 1 for count in counts) and control.get(name):
            found.append(f"{where}: descriptions do not cover {name} exactly once")
    for b, binding in enumerate(control.get("anim_bindings", [])):
        text = binding.get("animation_name")
        if text is not None and not text.startswith(BINDING_PREFIX):
            found.append(f"{where} binding {b}: animation name does not start with {BINDING_PREFIX!r}")
    return found


def parse_ps3_font(data: bytes) -> TableRead:
    """A PS3 font: textures with frames, point sizes with glyphs and kernings."""
    read = _top(data, "font")
    read.problems += font_problems(read.root, data)
    return read


def font_problems(font: dict, data: bytes | None = None, build: str = "ps3",
                  derived: bool | None = None) -> list[str]:
    """The font rules of T6 beyond list sizes, slots and tiling, for any build.

    Images: ps3 width x height bytes; switch and x360 the W15-B10 image form
    (``ui_write.font_image_problems``). The ps3 custom-data offset is checked when ``data`` is
    given (the switch-form one by ``ui_write.parse_font``). With ``derived`` the W15-B11 rules
    are added (``ui_write.font_derived_problems``); by default they are on for switch and x360
    and off for ps3, so the stage-7 ps3 result is unchanged."""
    if derived is None:
        derived = build != "ps3"
    found = []
    textures = font.get("textures", [])
    for i, texture in enumerate(textures):
        if build != "ps3":
            from .ui_write import font_image_problems
            found += [f"texture {i}: {p}" for p in font_image_problems(
                texture.get("image") or b"", texture["width"], texture["height"], build)]
        elif len(texture["image"]) != texture["width"] * texture["height"]:
            found.append(f"texture {i}: image of {len(texture['image'])} bytes for "
                         f"{texture['width']} x {texture['height']}")
    for p, size in enumerate(font.get("point_sizes", [])):
        for g, glyph in enumerate(size.get("glyphs", [])):
            index, frame = glyph["texture_index"], glyph["frame"]
            if index == -1:
                continue
            if not 0 <= index < len(textures):
                found.append(f"point size {p} glyph {g}: texture index {index} of {len(textures)}")
            elif frame != -1 and not 0 <= frame < textures[index]["frame_count"]:
                found.append(f"point size {p} glyph {g}: frame {frame} of "
                             f"{textures[index]['frame_count']}")
    if textures and build == "ps3" and data is not None:
        first = ByteView(data, ">").pair(BODY + 0xAC).file_offset  # texture records
        image_offset = ByteView(data, ">").u32(first + 0x10)  # first texture's image pair offset
        if font["custom_data_offset"] != image_offset:
            found.append(f"custom-data offset 0x{font['custom_data_offset']:X} is not the first "
                         f"image block 0x{image_offset:X}")
    if derived:
        from .ui_write import font_derived_problems
        found += font_derived_problems(font, build)
    return found


def verify_ps3_ui(data: bytes) -> list[str]:
    try:
        return parse_ps3_ui(data).problems
    except FormatError as exc:
        return [f"layout: {exc}"]


def verify_ps3_font(data: bytes) -> list[str]:
    try:
        return parse_ps3_font(data).problems
    except FormatError as exc:
        return [f"layout: {exc}"]


def fixed_26_6(value: int) -> float:
    """A 26.6 fixed-point metric in pixels."""
    return value / 64


# Stage 15 (W15-B8 to B11) lives in ``ui_write``; its public names are reachable from here too.
_UI_WRITE_NAMES = frozenset({
    "parse_ui", "write_ui", "verify_ui", "ui_key", "derive_ui_keys", "control_paths",
    "ui_link_problems", "append_control", "binding_target", "parse_font", "write_font",
    "verify_font", "font_image_bytes", "font_image_pixels", "font_image_problems",
    "derive_glyph", "line_break_glyph", "derive_font_values", "font_derived_problems",
    "remove_control", "ordinal_problems", "glyph_order_problems",  # corrections 18
    "list_layout", "LIST_SHARING",  # corrections 19
})


def __getattr__(name: str):
    if name in _UI_WRITE_NAMES:
        from . import ui_write
        return getattr(ui_write, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
