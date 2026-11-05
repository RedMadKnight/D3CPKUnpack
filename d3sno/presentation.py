"""Presentation formats: ps3 records, shader blocks, texture animation, ``.achu`` and FSB5.

Audio is not decoded. Unclear record parts are kept as raw bytes.
"""

from __future__ import annotations

import csv
import math
import random
import struct
from dataclasses import dataclass, field
from functools import lru_cache

from .binary import ByteView, FormatError, Pair, build_order
from .datafiles import DATA
from .material import MaterialBlock, read_material_block
from .sno import SnoHeader, parse_header

PATH_SIZE = 40
PATH_NODES_AT = 32
CUBE_FACES = ("+X", "-X", "+Y", "-Y", "+Z", "-Z")
VERTEX_COLOUR_SETS = {1: 0, 2: 1}  # colour slot to colour set; set 1 white marks cloth anchors


def _body(offset: int) -> int:
    return 0x10 + offset


def _ps3(data: bytes) -> tuple[SnoHeader, ByteView]:
    header = parse_header(data, "ps3")
    return header, ByteView(data, ">")


NODE_SIZES = {"float": 12, "angle": 12, "velocity": 12, "color": 12, "vector": 28}


@dataclass(frozen=True)
class PathNode:
    minimum: float | tuple  # a colour is 4 RGBA bytes, a vector three floats
    maximum: float | tuple
    time: float


PATH_DISTRIBUTIONS = {
    0: "uniform", 1: "low", 2: "mid", 3: "high", 4: "low and high", 5: "smooth low",
    6: "smooth mid", 7: "smooth high", 8: "smooth low and high",
}
PATH_SCALAR_FUNCTIONS = {
    0: "none", 1: "message scalar", 2: "weather intensity", 3: "fractional radial distance",
    4: "attack rate", 5: "actor health percent", 6: "fractional height",
    7: "player close to death", 8: "actor scale", 9: "spawner countdown percent", 10: "random",
    11: "value from bone x", 12: "actor primary resource percent", 13: "gold pickup radius",
}


@dataclass
class Path:
    """An animated value: a 40-byte header and a node pair."""

    raw: bytes
    nodes: list[bytes]
    name: str = ""
    node_kind: str = ""

    def _word(self, at: int, fmt: str):
        return struct.unpack_from(">" + fmt, self.raw, at)[0]

    @property
    def node_count(self) -> int:
        return self._word(0, "i")

    @property
    def loop_start(self) -> float:
        return self._word(4, "f")

    @property
    def loop_end(self) -> float:
        return self._word(8, "f")

    @property
    def distribution(self) -> int:
        """+12, an enum (``PATH_DISTRIBUTIONS``)."""
        return self._word(12, "i")

    @property
    def distribution_name(self) -> str | None:
        return PATH_DISTRIBUTIONS.get(self.distribution)

    @property
    def scalar_function(self) -> int:
        """+16, the function that drives the interpolation scalar (``PATH_SCALAR_FUNCTIONS``)."""
        return self._word(16, "i")

    @property
    def scalar_function_name(self) -> str | None:
        return PATH_SCALAR_FUNCTIONS.get(self.scalar_function)

    @property
    def scalar_lerp_zero(self) -> float:
        """+20, the factor when the function's input is 0."""
        return self._word(20, "f")

    @property
    def scalar_lerp_one(self) -> float:
        """+24, the factor when the function's input is 1."""
        return self._word(24, "f")

    @property
    def interpolation(self) -> tuple[int, float, float]:
        """+16 to +27: the scalar function, then the factors at input 0 and input 1."""
        return self.scalar_function, self.scalar_lerp_zero, self.scalar_lerp_one

    @property
    def padding(self) -> int:
        return self._word(28, "I")

    def decoded(self) -> list[PathNode]:
        """Nodes as min, max and time. Colours are RGBA bytes, vectors xyz floats."""
        found = []
        for node in self.nodes:
            if self.node_kind == "color":
                found.append(PathNode(tuple(node[0:4]), tuple(node[4:8]), struct.unpack(">f", node[8:12])[0]))
            elif len(node) == 28:
                values = struct.unpack(">7f", node)
                found.append(PathNode(values[0:3], values[3:6], values[6]))
            else:
                found.append(PathNode(*struct.unpack(">3f", node)))
        return found

    def problems(self) -> list[str]:
        found = []
        if self.node_count != len(self.nodes):
            found.append(f"{self.name or 'path'}: node count {self.node_count}, {len(self.nodes)} nodes")
        if self.padding:
            found.append(f"{self.name or 'path'}: word +28 is not zero")
        return found


def _read_path(view: ByteView, at: int, node_size: int, name: str = "", node_kind: str = "") -> Path:
    raw = view.take(at, PATH_SIZE, "path")
    pair = view.pair(at + PATH_NODES_AT, "path nodes")
    nodes = view.pair_data(pair, at + PATH_NODES_AT, "path nodes")
    if len(nodes) % node_size:
        raise FormatError(f"path nodes of {len(nodes)} bytes for {node_size}-byte nodes", at)
    return Path(raw, [nodes[i:i + node_size] for i in range(0, len(nodes), node_size)], name, node_kind)


@dataclass(frozen=True)
class PresentationField:
    format: str
    offset: int
    size: int
    name: str
    node_size: int | None
    node_kind: str
    note: str


@lru_cache(maxsize=None)
def presentation_fields() -> dict[str, tuple[PresentationField, ...]]:
    """The named paths and words of ps3 presentation records (``cov_presentation_fields.tsv``)."""
    found: dict[str, list[PresentationField]] = {}
    with open(DATA / "cov_presentation_fields.tsv", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            found.setdefault(row["format"], []).append(PresentationField(
                row["format"], int(row["offset"], 16), int(row["bytes"]), row["name"],
                int(row["node_bytes"]) if row["node_bytes"] else None, row["node_kind"],
                row["note"],
            ))
    return {key: tuple(value) for key, value in found.items()}


def _named_paths(view: ByteView, fmt: str) -> dict[str, Path]:
    return {
        f.name: _read_path(view, _body(f.offset), f.node_size, f.name, f.node_kind)
        for f in presentation_fields()[fmt] if f.note == "path"
    }


def path_problems(paths: dict[str, Path]) -> list[str]:
    """Every path's node count must equal its pair size divided by the node size."""
    return [problem for path in paths.values() for problem in path.problems()]


def _records(view: ByteView, pair_at: int, size: int, what: str, count: int | None = None) -> list[bytes]:
    pair = view.pair(pair_at, what)
    raw = view.pair_data(pair, pair_at, what)
    if len(raw) % size or (count is not None and len(raw) != count * size):
        raise FormatError(f"{what}: {len(raw)} bytes for records of {size}", pair_at)
    return [raw[i:i + size] for i in range(0, len(raw), size)]


# --- Music ---------------------------------------------------------------------------------

MUSIC_FIELDS = (
    "sound_bank", "sample_hash", "volume_db", "start_min", "start_delta", "end", "loop_count",
    "loop_start", "loop_end", "fade_in_offset", "fade_in_time", "fade_out_offset",
    "fade_out_time", "fade_in_shape", "fade_out_shape",
)


@dataclass
class MusicEntry:
    words: dict[str, int]  # raw u32 words; their types are open

    def as_float(self, name: str) -> float:
        return struct.unpack(">f", struct.pack(">I", self.words[name]))[0]


def parse_music(data: bytes) -> list[MusicEntry]:
    """ps3 ``.mus``: 60-byte entries at pair 0x2C, count at 0x34."""
    _, view = _ps3(data)
    count = view.i32(0x34, "music count")
    if count < 0:
        raise FormatError("negative music count", 0x34)
    return [
        MusicEntry(dict(zip(MUSIC_FIELDS, struct.unpack(">15I", raw))))
        for raw in _records(view, 0x2C, 60, "music entries", count)
    ]


# --- Trail, rope, light --------------------------------------------------------------------

@dataclass
class Trail:
    paths: list[Path]
    type: int
    segments: int
    material: MaterialBlock

    @property
    def named_paths(self) -> dict[str, Path]:
        return {p.name: p for p in self.paths}


def parse_trail(data: bytes) -> Trail:
    _, view = _ps3(data)
    paths = list(_named_paths(view, "trl").values())
    return Trail(
        paths, view.i32(_body(0xC8)), view.i32(_body(0xCC)),
        read_material_block(view, _body(0xD0), "ps3"),
    )


@dataclass
class Rope:
    material: MaterialBlock
    physics_raw: bytes
    paths: list[Path]
    floats: tuple[float, float]  # jitter approach speed, expected length
    tail_paths: list[Path]
    events: list[bytes]

    @property
    def named_paths(self) -> dict[str, Path]:
        return {p.name: p for p in self.paths + self.tail_paths}

    @property
    def jitter_approach_speed(self) -> float:
        return self.floats[0]

    @property
    def expected_length(self) -> float:
        return self.floats[1]


def parse_rope(data: bytes) -> Rope:
    """All offsets are body offsets (V25)."""
    _, view = _ps3(data)
    named = list(_named_paths(view, "rop").values())
    floats_at = _body(0x8C) + 14 * PATH_SIZE
    return Rope(
        read_material_block(view, _body(0x30), "ps3"),
        view.take(_body(0x78), 0x8C - 0x78, "rope physics"), named[:14], view.f32s(floats_at, 2),
        named[14:], _records(view, _body(0x38C), 192, "rope events"),
    )


@dataclass
class Light:
    animation: bytes
    texture: int
    duration: float
    priority: int
    distance: float
    states: list[list[bytes]]  # two states of 13 raw 40-byte paths
    paths: dict[str, Path] = field(default_factory=dict)  # state_a.*, state_b.*


def parse_light(data: bytes) -> Light:
    _, view = _ps3(data)
    animation = view.pair_data(view.pair(_body(0x18)), _body(0x18), "light animation")
    states = [
        [view.take(_body(base) + i * PATH_SIZE, PATH_SIZE, "light path") for i in range(13)]
        for base in (0x38, 0x240)
    ]
    return Light(
        animation, view.i32(_body(0x28)), view.f32(_body(0x2C)), view.i32(_body(0x30)),
        view.f32(_body(0x34)), states, _named_paths(view, "lit"),
    )


# --- UI, font, level area ------------------------------------------------------------------

UI_RECORD_SIZES = {
    "control": 272, "style": 112, "child_spec": 76, "string_element": 24, "number_element": 16,
}
UI_DIALOG_SIZE = 0x78
UI_DIALOG_LIST_SIZES = (272, 112, 24, 16, 16)  # controls, styles, strings, int32s, float32s


UI_LIST_SIZE = 16
CONTROL_ANIMATION_SIZES = (8, 8, 12, 40, 20)


@dataclass
class UiList:
    count: int
    raw: bytes
    pointer: int = 0  # run-time slot

    def split(self, size: int) -> list[bytes]:
        if len(self.raw) % size:
            raise FormatError(f"list of {len(self.raw)} bytes for {size}-byte records")
        return [self.raw[i:i + size] for i in range(0, len(self.raw), size)]

    @property
    def element_size(self) -> int | None:
        """Size / count where it divides; None for an empty or uneven list."""
        if self.count <= 0 or len(self.raw) % self.count:
            return None
        return len(self.raw) // self.count

    def fits(self, size: int) -> bool:
        return len(self.raw) == self.count * size


def _ui_list(view: ByteView, position: int) -> UiList:
    count = view.i32(position, "UI list count")
    raw = view.pair_data(view.pair(position + 4), position + 4, "UI list")
    return UiList(count, raw, view.u32(position + 12))


def parse_ui(data: bytes) -> list[UiList]:
    """The five dialog lists (controls, styles, string, integer and float constants), raw.

    ``ui_font.parse_ps3_ui`` reads every field (stage 7, T5).
    """
    _, view = _ps3(data)
    return [_ui_list(view, _body(at)) for at in (0x10, 0x20, 0x30, 0x40, 0x50)]


@dataclass
class UiChildSpec:
    raw: bytes
    element_lists: list[UiList]  # three, at +0x10


@dataclass
class UiStyle:
    raw: bytes
    element_lists: list[UiList]  # three, at +0x18
    child_specs: list[UiChildSpec]  # list at +0x48


@dataclass
class UiControl:
    raw: bytes
    type: int  # +0x38
    style: UiStyle  # inline at +0x40
    animation_lists: list[UiList]  # five, at +0xB0
    problems: list[str] = field(default_factory=list)


def _lists(view: ByteView, at: int, count: int) -> list[UiList]:
    return [_ui_list(view, at + i * UI_LIST_SIZE) for i in range(count)]


def read_ui_style(data: bytes, at: int) -> UiStyle:
    """A 112-byte style at file position ``at``. Element lists keep their bytes."""
    view = ByteView(data, ">")
    raw = view.take(at, UI_RECORD_SIZES["style"], "UI style")
    specs = _ui_list(view, at + 0x48)
    if not specs.fits(UI_RECORD_SIZES["child_spec"]):
        raise FormatError(f"child specs: {len(specs.raw)} bytes for {specs.count}", at + 0x48)
    start = view.pair(at + 0x4C).file_offset
    children = [
        UiChildSpec(
            view.take(start + i * 76, 76), _lists(view, start + i * 76 + 0x10, 3),
        )
        for i in range(specs.count)
    ]
    return UiStyle(raw, _lists(view, at + 0x18, 3), children)


def read_ui_control(data: bytes, at: int) -> UiControl:
    """A 272-byte control at file position ``at``."""
    view = ByteView(data, ">")
    raw = view.take(at, UI_RECORD_SIZES["control"], "UI control")
    animations = _lists(view, at + 0xB0, 5)
    problems = [
        f"animation list {i}: {len(lst.raw)} bytes for {lst.count} of {size}"
        for i, (lst, size) in enumerate(zip(animations, CONTROL_ANIMATION_SIZES))
        if not lst.fits(size)
    ]
    return UiControl(raw, view.i32(at + 0x38), read_ui_style(data, at + 0x40), animations, problems)


@dataclass
class Font:
    textures: list[bytes]  # 40-byte records
    point_sizes: list[bytes]  # 72-byte records


def parse_font(data: bytes) -> Font:
    _, view = _ps3(data)
    return Font(
        _records(view, _body(0xAC), 40, "font textures"),
        _records(view, _body(0xBC), 72, "font point sizes"),
    )


def glyph_metric_pixels(value: int) -> float:
    return value / 64


def level_area_slots(server_block: bytes) -> list[bytes]:
    """26 slots of 352 bytes from server-block 0x18."""
    start = 0x18
    if start + 26 * 352 > len(server_block):
        raise FormatError("server block too short for 26 level area slots", start)
    return [server_block[start + i * 352:start + (i + 1) * 352] for i in range(26)]


# --- Shaders -------------------------------------------------------------------------------

@dataclass
class ShaderRecord:
    raw: bytes
    code: bytes


def parse_shader_block(block: bytes, pass_count: int, build: str) -> list[tuple[ShaderRecord, ShaderRecord]]:
    """A compiled block: 2 x pass count pairs relative to the table end, vertex then pixel."""
    order = build_order(build)  # x360: big-endian (X11)
    view = ByteView(block, order)
    table_end = pass_count * 2 * 8
    view.need(0, table_end, "shader table")
    records = []
    for i in range(pass_count * 2):
        offset, size = view.unpack("II", i * 8)
        raw = view.take(table_end + offset, size, "shader record")
        r = ByteView(raw, order)
        if build == "ps3":
            length = r.u32(0x10, "shader code length")
            if length > size:
                raise FormatError(f"code length {length} in a {size}-byte record", table_end + offset)
            code = raw[size - length:]
        else:
            words = r.unpack("4I", 0, "shader record head")
            if words != (0, 0, size - 16, 0):
                raise FormatError(f"switch shader record head {words}", table_end + offset)
            code = raw[16:]
        records.append(ShaderRecord(raw, code))
    return [(records[i], records[i + 1]) for i in range(0, len(records), 2)]


# --- Texture animation ---------------------------------------------------------------------

@dataclass
class TextureAnimation:
    kind: str  # identity, full, scale, scroll
    scale: tuple[float, float] = (1.0, 1.0)
    speed: tuple[float, float] = (0.0, 0.0)
    type: int | None = None
    matrix: tuple[float, ...] | None = None
    translation: tuple[float, ...] | None = None  # U and V, each initial, speed, delta
    rotation: tuple[float, float, float] | None = None
    flags: tuple[int, ...] = ()
    frame_block: bytes = b""
    animation_flags: int = 0


def parse_texture_animation(blob: bytes, build: str) -> TextureAnimation:
    order = build_order(build)  # x360: big-endian (X11)
    view = ByteView(blob, order)
    if not blob:
        return TextureAnimation("identity")
    if len(blob) == 12:
        kind = view.u32(0)
        if kind not in (0, 1):
            raise FormatError(f"scale animation kind {kind}", 0)
        return TextureAnimation("scale", view.f32s(4, 2))
    if len(blob) == 20:
        kind = view.u32(0)
        if kind != 2:
            raise FormatError(f"scroll animation kind {kind}", 0)
        scale = tuple(s if s > 0 else 1.0 for s in view.f32s(4, 2))
        return TextureAnimation("scroll", scale, view.f32s(12, 2))
    if len(blob) == 144:
        return TextureAnimation(
            "full", type=view.u32(0), matrix=view.f32s(4, 16), translation=view.f32s(68, 6),
            rotation=view.f32s(92, 3), flags=view.unpack("4I", 104),
            frame_block=view.take(120, 20), animation_flags=view.u32(140),
        )
    raise FormatError(f"texture animation of {len(blob)} bytes", 0)


def scroll_offset(animation: TextureAnimation, seconds: float, rng: random.Random) -> tuple[float, float]:
    """A scroll animation's U and V: a random start in [0, 1), moving speed x 60 per second."""
    if animation.kind != "scroll":
        return (0.0, 0.0)
    return tuple(
        math.fmod(rng.random() + speed * 60 * seconds, 1.0) % 1.0 for speed in animation.speed
    )


def lightmap_value(rgba: bytes) -> int:
    """Lightmaps carry their value in red only."""
    return rgba[0]


# --- .achu ---------------------------------------------------------------------------------

@dataclass
class ProtoField:
    number: int
    wire: int
    value: int | bytes


def _varint(data: bytes, pos: int) -> tuple[int, int]:
    value = shift = 0
    while True:
        if pos >= len(data):
            raise FormatError("varint runs past the end", pos)
        byte = data[pos]
        value |= (byte & 0x7F) << shift
        pos += 1
        if not byte & 0x80:
            return value, pos
        shift += 7
        if shift > 63:
            raise FormatError("varint too long", pos)


def parse_protobuf(data: bytes) -> list[ProtoField]:
    """Generic wire format. Nothing is interpreted, and unknown fields are kept."""
    fields = []
    pos = 0
    while pos < len(data):
        key, pos = _varint(data, pos)
        number, wire = key >> 3, key & 7
        if number == 0:
            raise FormatError("field number 0", pos)
        if wire == 0:
            value, pos = _varint(data, pos)
        elif wire == 1 or wire == 5:
            size = 8 if wire == 1 else 4
            if pos + size > len(data):
                raise FormatError("fixed field runs past the end", pos)
            value = int.from_bytes(data[pos:pos + size], "little")
            pos += size
        elif wire == 2:
            length, pos = _varint(data, pos)
            if pos + length > len(data):
                raise FormatError("length-delimited field runs past the end", pos)
            value = bytes(data[pos:pos + length])
            pos += length
        else:
            raise FormatError(f"wire type {wire}", pos)
        fields.append(ProtoField(number, wire, value))
    return fields


ACHIEVEMENT_GROUPS = {1: "categories", 2: "achievements", 3: "criteria", 4: "attributes"}


@dataclass
class Achievements:
    groups: dict[str, list[list[ProtoField]]] = field(default_factory=dict)
    other: list[ProtoField] = field(default_factory=list)


@dataclass(frozen=True)
class AchuField:
    message: str
    number: int
    name: str
    wire: str  # varint, fixed32, bytes, or "message <name>"


@lru_cache(maxsize=None)
def achu_schema() -> dict:
    """``cov_achu.tsv``: fields by (message, number), flag bits, event and condition names."""
    fields: dict[tuple[str, int], AchuField] = {}
    names: dict[str, dict[int, str]] = {}
    with open(DATA / "cov_achu.tsv", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            number = int(row["number_or_value"])
            if row["table"] == "field":
                fields[(row["message_or_enum"], number)] = AchuField(
                    row["message_or_enum"], number, row["name"], row["wire_or_note"],
                )
            else:
                names.setdefault(row["table"], {})[number] = row["name"]
    return {"fields": fields, "names": names}


_WIRE_OF = {"varint": 0, "fixed32": 5, "bytes": 2}


def decode_achu(data: bytes, message: str = "file", depth: int = 0) -> tuple[dict, list[str]]:
    """Walk an ``.achu`` by the schema: {field name: [values]} and the problems found.

    Nested messages become dicts. A field number missing from the schema is a problem and is
    kept under ``#<number>``.
    """
    if depth > 16:
        raise FormatError("achievement messages nest too deep")
    schema = achu_schema()["fields"]
    result: dict[str, list] = {}
    problems: list[str] = []
    for item in parse_protobuf(data):
        spec = schema.get((message, item.number))
        if spec is None:
            problems.append(f"{message}: field {item.number} is not in the table")
            result.setdefault(f"#{item.number}", []).append(item.value)
            continue
        if spec.wire.startswith("message "):
            if item.wire != 2:
                problems.append(f"{message}.{spec.name}: wire type {item.wire}")
                continue
            inner, inner_problems = decode_achu(item.value, spec.wire[8:], depth + 1)
            problems += inner_problems
            result.setdefault(spec.name, []).append(inner)
            continue
        if _WIRE_OF.get(spec.wire) != item.wire:
            problems.append(f"{message}.{spec.name}: wire type {item.wire}, table says {spec.wire}")
        result.setdefault(spec.name, []).append(item.value)
    return result, problems


def achu_flag_names(kind: str, value: int) -> list[str]:
    """Names of set bits; ``kind`` is achievement, category or criteria."""
    names = achu_schema()["names"].get(f"{kind}_flag_bit", {})
    return [names.get(bit, f"bit_{bit}") for bit in range(64) if value >> bit & 1]


def achu_event_name(event_id: int) -> str | None:
    return achu_schema()["names"].get("event_id", {}).get(event_id)


def achu_condition_name(condition_id: int) -> str | None:
    return achu_schema()["names"].get("modifier_condition_id", {}).get(condition_id)


def parse_achievements(data: bytes) -> Achievements:
    result = Achievements({name: [] for name in ACHIEVEMENT_GROUPS.values()})
    for item in parse_protobuf(data):
        name = ACHIEVEMENT_GROUPS.get(item.number)
        if name is not None and item.wire == 2:
            result.groups[name].append(parse_protobuf(item.value))
        else:
            result.other.append(item)
    return result


# --- FSB5 ----------------------------------------------------------------------------------

FSB5_HEADER = 60
CHUNK_LOOP = 3
CHUNK_VORBIS = 11
MODE_VORBIS = 15
MODE_FRAMES = 16
FRAME_SIZE = 140
FRAME_SAMPLES = 256


@dataclass
class FsbChunk:
    type: int
    data: bytes


CHUNK_MARKERS = 4
MARKER_SIZE = 260


@dataclass(frozen=True)
class FsbMarker:
    position: int
    name: str


@dataclass
class FsbSample:
    rate_index: int
    stereo: bool
    data_offset: int
    sample_count: int
    chunks: list[FsbChunk]

    @property
    def loop(self) -> tuple[int, int] | None:
        chunk = next((c for c in self.chunks if c.type == CHUNK_LOOP), None)
        return None if chunk is None or len(chunk.data) < 8 else struct.unpack("<II", chunk.data[:8])

    def markers(self) -> list[FsbMarker]:
        """Type-4 chunks: markers of a 4-byte position and a 256-byte name."""
        found = []
        for chunk in self.chunks:
            if chunk.type != CHUNK_MARKERS:
                continue
            if len(chunk.data) % MARKER_SIZE:
                raise FormatError(f"marker chunk of {len(chunk.data)} bytes")
            for at in range(0, len(chunk.data), MARKER_SIZE):
                name = chunk.data[at + 4:at + MARKER_SIZE].split(b"\0")[0].decode("latin-1")
                found.append(FsbMarker(struct.unpack_from("<I", chunk.data, at)[0], name))
        return found

    @property
    def vorbis_setup(self) -> tuple[int, list[tuple[int, int]]] | None:
        """The setup CRC and the (byte offset, sample) seek pairs."""
        chunk = next((c for c in self.chunks if c.type == CHUNK_VORBIS), None)
        if chunk is None or len(chunk.data) < 4:
            return None
        crc = struct.unpack_from("<I", chunk.data, 0)[0]
        pairs = list(struct.iter_unpack("<II", chunk.data[4:4 + (len(chunk.data) - 4) // 8 * 8]))
        return crc, pairs


@dataclass
class Fsb5:
    version: int
    mode: int
    sample: FsbSample
    header_raw: bytes
    data: bytes

    def packets(self) -> list[bytes]:
        """Mode 15: u16-length-prefixed Vorbis packets. Mode 16: 140-byte frames."""
        if self.mode == MODE_VORBIS:
            out, pos = [], 0
            while pos + 2 <= len(self.data):
                size = struct.unpack_from("<H", self.data, pos)[0]
                if size == 0:
                    break
                if pos + 2 + size > len(self.data):
                    raise FormatError("Vorbis packet runs past the end", pos)
                out.append(self.data[pos + 2:pos + 2 + size])
                pos += 2 + size
            return out
        if self.mode == MODE_FRAMES:
            usable = len(self.data) // FRAME_SIZE * FRAME_SIZE
            return [self.data[i:i + FRAME_SIZE] for i in range(0, usable, FRAME_SIZE)]
        raise FormatError(f"FSB5 mode {self.mode} is not described")


def parse_fsb5(data: bytes) -> Fsb5:
    """An FSB5 file holding one sample. The header field order comes from the public format."""
    view = ByteView(data, "<")
    if view.take(0, 4, "FSB5 magic") != b"FSB5":
        raise FormatError("not an FSB5 file", 0)
    version, count, headers_size, names_size, data_size, mode = view.unpack("6I", 4, "FSB5 header")
    view.need(0, FSB5_HEADER, "FSB5 header")
    if version != 1:
        raise FormatError(f"FSB5 version {version}", 4)
    if count != 1:
        raise FormatError(f"{count} samples, one expected", 8)
    word = view.u64(FSB5_HEADER, "sample header")
    pos = FSB5_HEADER + 8
    chunks = []
    more = bool(word & 1)
    while more:
        chunk_word = view.u32(pos, "chunk header")
        more = bool(chunk_word & 1)
        size = chunk_word >> 1 & 0xFFFFFF
        chunks.append(FsbChunk(chunk_word >> 25 & 0x7F, view.take(pos + 4, size, "chunk")))
        pos += 4 + size
    if pos > FSB5_HEADER + headers_size:
        raise FormatError("sample chunks run past the sample headers", pos)
    sample = FsbSample(
        word >> 1 & 15, bool(word >> 5 & 1), (word >> 6 & ((1 << 28) - 1)) * 16, word >> 34, chunks,
    )
    data_at = FSB5_HEADER + headers_size + names_size
    view.need(data_at, data_size, "FSB5 data")
    start = data_at + sample.data_offset
    return Fsb5(version, mode, sample, view.take(0, FSB5_HEADER), bytes(data[start:data_at + data_size]))
