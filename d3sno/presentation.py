"""Presentation formats: ps3 records, shader blocks, texture animation, ``.achu`` and FSB5.

Audio is not decoded. Unclear record parts are kept as raw bytes.
"""

from __future__ import annotations

import math
import random
import struct
from dataclasses import dataclass, field

from .binary import ByteView, FormatError, Pair
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


@dataclass
class Path:
    raw: bytes
    nodes: list[bytes]


def _read_path(view: ByteView, at: int, node_size: int) -> Path:
    raw = view.take(at, PATH_SIZE, "path")
    pair = view.pair(at + PATH_NODES_AT, "path nodes")
    nodes = view.pair_data(pair, at + PATH_NODES_AT, "path nodes")
    if len(nodes) % node_size:
        raise FormatError(f"path nodes of {len(nodes)} bytes for {node_size}-byte nodes", at)
    return Path(raw, [nodes[i:i + node_size] for i in range(0, len(nodes), node_size)])


def _records(view: ByteView, pair_at: int, size: int, what: str, count: int | None = None) -> list[bytes]:
    pair = view.pair(pair_at, what)
    raw = view.pair_data(pair, pair_at, what)
    if len(raw) % size or (count is not None and len(raw) != count * size):
        raise FormatError(f"{what}: {len(raw)} bytes for records of {size}", pair_at)
    return [raw[i:i + size] for i in range(0, len(raw), size)]


# --- Music ---------------------------------------------------------------------------------

MUSIC_FIELDS = (
    "bank", "sample_hash", "volume", "start_min", "start_delta", "end", "loop_count",
    "loop_start", "loop_end", "fade_in_offset", "fade_in_time", "fade_out_offset",
    "fade_out_time", "word_13", "word_14",
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


def parse_trail(data: bytes) -> Trail:
    _, view = _ps3(data)
    paths = [_read_path(view, _body(0x28) + i * PATH_SIZE, 12) for i in range(4)]
    return Trail(
        paths, view.i32(_body(0xC8)), view.i32(_body(0xCC)),
        read_material_block(view, _body(0xD0), "ps3"),
    )


@dataclass
class Rope:
    material: MaterialBlock
    physics_raw: bytes
    paths: list[Path]
    floats: tuple[float, float]
    tail_paths: list[Path]
    events: list[bytes]


def parse_rope(data: bytes) -> Rope:
    """All offsets are body offsets (V25)."""
    _, view = _ps3(data)
    paths_at = _body(0x8C)
    paths = [
        _read_path(view, paths_at + i * PATH_SIZE, 28 if i == 12 else 12) for i in range(14)
    ]
    floats_at = paths_at + 14 * PATH_SIZE
    tail_at = floats_at + 8
    return Rope(
        read_material_block(view, _body(0x30), "ps3"),
        view.take(_body(0x78), 0x8C - 0x78, "rope physics"), paths, view.f32s(floats_at, 2),
        [_read_path(view, tail_at + i * PATH_SIZE, 12) for i in range(5)],
        _records(view, _body(0x38C), 192, "rope events"),
    )


@dataclass
class Light:
    animation: bytes
    texture: int
    duration: float
    priority: int
    distance: float
    states: list[list[bytes]]  # two states of 13 raw 40-byte paths


def parse_light(data: bytes) -> Light:
    _, view = _ps3(data)
    animation = view.pair_data(view.pair(_body(0x18)), _body(0x18), "light animation")
    states = [
        [view.take(_body(base) + i * PATH_SIZE, PATH_SIZE, "light path") for i in range(13)]
        for base in (0x38, 0x240)
    ]
    return Light(
        animation, view.i32(_body(0x28)), view.f32(_body(0x2C)), view.i32(_body(0x30)),
        view.f32(_body(0x34)), states,
    )


# --- UI, font, level area ------------------------------------------------------------------

UI_RECORD_SIZES = {
    "control": 272, "style": 112, "child_spec": 76, "string_element": 24, "number_element": 16,
}


@dataclass
class UiList:
    count: int
    raw: bytes

    def split(self, size: int) -> list[bytes]:
        if len(self.raw) % size:
            raise FormatError(f"list of {len(self.raw)} bytes for {size}-byte records")
        return [self.raw[i:i + size] for i in range(0, len(self.raw), size)]


def parse_ui(data: bytes) -> list[UiList]:
    """The three dialog lists, as {count, pair, pointer}. Their element types are left raw."""
    _, view = _ps3(data)
    lists = []
    for at in (0x10, 0x20, 0x30):
        position = _body(at)
        lists.append(UiList(
            view.i32(position), view.pair_data(view.pair(position + 4), position + 4, "UI list"),
        ))
    return lists


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
    order = ">" if build == "ps3" else "<"
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
    order = ">" if build == "ps3" else "<"
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
