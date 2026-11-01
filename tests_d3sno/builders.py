"""Synthetic data written from the description, independent of the d3sno writers."""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass, field

SEED = 0x873AE8E53C11D761
MASK = 0xFFFFFFFFFFFFFFFF


def fnv(name: str) -> int:
    value = 0xCBF29CE484222325
    for byte in name.lower().encode("latin-1"):
        value = ((value ^ byte) * 0x100000001B3) & MASK
    return value


def encrypt(plain: bytes) -> bytes:
    state = SEED
    out = bytearray()
    for byte in plain:
        cipher = byte ^ (state & 0xFF)
        out.append(cipher)
        state = ((cipher << 56) | (state >> 8)) & MASK
    return bytes(out)


def packet(plain: bytes, order: str, *, declared=None, encrypted=False, stored=None) -> bytes:
    body = zlib.compress(plain) if stored is None else stored
    if encrypted:
        body = encrypt(body)
    size = len(plain) if declared is None else declared
    word = len(body) | (0x8000 if encrypted else 0)
    return struct.pack(order + "HHH", size & 0xFFFF, size >> 16, word) + body


class Bits:
    def __init__(self):
        self.text = []

    def put(self, value: int, width: int):
        if width:
            self.text.append(format(value, f"0{width}b"))

    def done(self) -> bytes:
        text = "".join(self.text)
        text += "0" * (-len(text) % 8)
        return int(text, 2).to_bytes(len(text) // 8, "big") if text else b""


@dataclass
class CpkSpec:
    """Everything a hand-made archive needs. Slots are given as finished bytes."""

    files: list  # (name, size, count, first)
    locations: list
    sector_offsets: list
    stream_size: int
    slots: list = field(default_factory=list)
    version: int = 7
    order: str = "<"
    flags: int = 2
    read_sector: int = 0x200
    comp_sector: int = 0x1000
    trailing: bytes = b""
    padding_in_blob: int = 0
    # Fault injection, to build the negative cases.
    unsorted: bool = False
    bad_crc: bool = False
    wrong_name_hash: bool = False
    location_bits: int = 32
    sector_offset_bits: int | None = None
    sector_index_bits: int | None = None
    sector_index: list | None = None
    raw_region: bytes | None = None
    name_order_reversed: bool = True


def build_cpk(s: CpkSpec) -> bytes:
    entries = [(fnv(n) ^ (1 if s.wrong_name_hash else 0), size, count, first)
               for n, size, count, first in s.files]
    entries.sort(key=lambda e: e[0], reverse=s.unsorted)

    if s.raw_region is not None:
        region = s.raw_region
    else:
        region = b"".join(slot.ljust(s.comp_sector, b"\0") for slot in s.slots)
    region += s.trailing
    comp_count = (len(region) + s.comp_sector - 1) // s.comp_sector
    decomp_count = (s.stream_size + s.comp_sector - 1) // s.comp_sector

    size_bits, copies_bits, first_bits = 24, 3, 9
    index_bits = comp_count.bit_length() if s.sector_index_bits is None else s.sector_index_bits
    offset_bits = s.location_bits if s.sector_offset_bits is None else s.sector_offset_bits

    tables = bytearray()
    bits = Bits()
    for file_hash, size, count, first in entries:
        bits.put(file_hash, 64)
        bits.put(size, size_bits)
        bits.put(count, copies_bits)
        bits.put(first, first_bits)
    tables += bits.done()
    bits = Bits()
    for location in s.locations:
        bits.put(location, s.location_bits)
    tables += bits.done()
    if s.flags & 2:
        bits = Bits()
        for offset in s.sector_offsets:
            bits.put(offset, s.location_bits)
        tables += bits.done()
        bits = Bits()
        index = s.sector_index
        if index is None:
            index = []
            for k in range(decomp_count):
                index.append(max(sum(1 for o in s.sector_offsets if o <= k * s.comp_sector) - 1, 0))
        for value in index:
            bits.put(value, index_bits)
        tables += bits.done()

    names = [n for n, *_ in s.files]
    if s.name_order_reversed:
        names = names[::-1]
    blob = bytearray()
    for name in names:
        tables += struct.pack(s.order + "I", len(blob))
        blob += name.encode("latin-1") + b"\0"
    tables += blob
    tables += bytes(s.padding_in_blob)

    header_size = 64 if s.version == 6 else 72
    tables_end = header_size + len(tables)
    if s.flags & 8:
        header_sector = tables_end
        start = tables_end
    else:
        header_sector = (tables_end + s.read_sector - 1) // s.read_sector
        start = header_sector * s.read_sector
    crc = zlib.crc32(region) & 0xFFFFFFFF
    if s.bad_crc:
        crc ^= 1
    head = struct.pack(s.order + "II", 0xA1B2C3D4, s.version)
    head += struct.pack(s.order + "Q", s.stream_size)
    head += struct.pack(
        s.order + "11I", s.flags, len(s.files), len(s.locations), header_sector, size_bits,
        copies_bits, first_bits, s.location_bits, offset_bits, index_bits, crc,
    )
    if s.version == 6:
        head += struct.pack(s.order + "I", 0x10000)
    else:
        head += struct.pack(s.order + "III", s.read_sector, s.comp_sector, 0)
    return head + bytes(tables).ljust(start - header_size, b"\0") + region


def simple_slots(stream: bytes, comp_sector: int, order: str, chunk: int = 500, empty_first=False):
    """Pack incompressible-safe chunks. Returns (sector offsets, slots)."""
    offsets, slots = [0], []
    slot = bytearray()
    if empty_first:
        slot += packet(b"", order)
    pos = 0
    while pos < len(stream):
        piece = packet(stream[pos:pos + chunk], order)
        if len(slot) + len(piece) > comp_sector:
            slots.append(bytes(slot))
            slot = bytearray()
            offsets.append(pos)
        slot += piece
        pos += chunk
    slots.append(bytes(slot))
    return offsets, slots


def sno_head(version: int, order: str, handle: int = 7, revision=1, source=0x1234) -> bytes:
    return struct.pack(order + "7I", 0xDEADBEEF, version, revision, source, handle, 0, 0)


class Image:
    """A growing file image. Offsets returned are pair offsets (file position - 0x10)."""

    def __init__(self, size: int, order: str):
        self.b = bytearray(size)
        self.order = order

    def put(self, pos: int, fmt: str, *values):
        struct.pack_into(self.order + fmt, self.b, pos, *values)

    def raw(self, pos: int, data: bytes):
        self.b[pos:pos + len(data)] = data

    def append(self, data: bytes) -> tuple[int, int]:
        position = len(self.b)
        self.b += data
        return position - 0x10, len(data)

    def pair(self, pos: int, data: bytes) -> tuple[int, int]:
        pair = self.append(data)
        self.put(pos, "II", *pair)
        return pair

    def done(self) -> bytes:
        return bytes(self.b)


# --- Tag maps and powers ---------------------------------------------------------------

def formula_value(text: str, code: bytes, order: str) -> bytes:
    raw = text.encode("latin-1") + b"\0"
    head = struct.pack(order + "8I", 0, 0, 0, 0, 0, len(raw), 0, len(code))
    return head + raw + bytes(-len(raw) % 4) + code


def tag_record(kind: int, tag: int, value, order: str) -> bytes:
    head = struct.pack(order + "II", kind, tag)
    if kind == 4:
        return head + formula_value(value[0], value[1], order)
    if kind == 1:
        return head + struct.pack(order + "f", value)
    return head + struct.pack(order + "i", value)


def tag_map(records: list, order: str) -> bytes:
    return struct.pack(order + "I", len(records)) + b"".join(
        tag_record(k, t, v, order) for k, t, v in records
    )


_LAYOUT = {
    "ps3": (">", 1442, 8, 0x178, 0x1A0),
    "switch": ("<", 1722, 32, 0x2F8, 0x320),
}


MAP_POSITIONS = (
    ("Effect", 0x20), ("General", 0x30), ("PvpGeneral", 0x40),
    ("Contact0", 0x50), ("Contact1", 0x58), ("Contact2", 0x60), ("Contact3", 0x68),
    ("PvpContact0", 0x90), ("PvpContact1", 0x98), ("PvpContact2", 0xA0), ("PvpContact3", 0xA8),
)


def build_power(build: str, general=(), contact0=(), buff=(), script=b"", power_bits=0,
                formula_count=0, scripted=None, version=None, all_maps=False,
                shared_contacts=False, revision=1, source=0x1234) -> bytes:
    """``all_maps`` gives every map its own bytes. ``shared_contacts`` lays one contact block."""
    order, default_version, slots, trailer, size = _LAYOUT[build]
    img = Image(size, order)
    img.raw(0, sno_head(version or default_version, order, revision=revision, source=source))
    records = {"General": general, "Contact0": contact0}
    for name, at in MAP_POSITIONS:
        if shared_contacts and name in ("Contact1", "Contact2", "Contact3"):
            img.put(at, "II", *struct.unpack_from(order + "II", img.b, 0x50))
        elif all_maps or records.get(name):
            img.b += bytes(-len(img.b) % 4)
            img.pair(at, tag_map(list(records.get(name, ())), order))
    img.put(0xD0, "I", power_bits)
    img.put(0xD4, "9I", 1, 2, 3, 4, 5, 6, 7, 8, 0x99)
    for i, handles in enumerate(buff):
        if build == "ps3":
            raw = struct.pack(f"{order}{len(handles)}I", *handles)
        else:
            raw = b"".join(struct.pack(order + "II", h, 0) for h in handles)
        img.b += bytes(-len(img.b) % 8)
        img.pair(0xF8 + i * 16, raw)
    img.put(trailer, "II", 1 if (scripted if scripted is not None else bool(script)) else 0, 1)
    img.put(trailer + 8, "i", formula_count)
    img.put(trailer + 0x20, "ii", -1, 0)
    if script:
        img.b += bytes(-len(img.b) % 8)
        img.pair(trailer + 0x18, script)
    return img.done()


# --- Formula bytecode ----------------------------------------------------------------

class Code:
    def __init__(self, order: str):
        self.order = order
        self.out = bytearray()

    def op(self, opcode: int) -> "Code":
        self.out += bytes((opcode, 0, 0, 0))
        return self

    def f(self, value: float) -> "Code":
        self.op(6)
        self.out += struct.pack(self.order + "f", value)
        return self

    def call(self, fn: int) -> "Code":
        self.op(1)
        self.out += bytes((fn, 0, 0, 0))
        return self

    def ident(self, cls: int, a=0, b=0, c=0) -> "Code":
        self.op(5)
        self.out += struct.pack(self.order + "4I", cls, a & 0xFFFFFFFF, b & 0xFFFFFFFF, c)
        return self

    def end(self) -> bytes:
        self.op(0)
        return bytes(self.out)


# --- Lua -------------------------------------------------------------------------------

RETURN_WORD = 30 | (1 << 23)


class LuaWriter:
    def __init__(self, build: str):
        self.build = build
        self.order = ">" if build == "ps3" else "<"
        self.out = bytearray()

    def header(self):
        flag, size_t = (0, 4) if self.build == "ps3" else (1, 8)
        self.out += b"\x1BLuaQ\0" + bytes((flag, 4, size_t, 4, 8, 0))

    def int(self, value):
        self.out += struct.pack(self.order + "i", value)

    def string(self, text):
        fmt = "I" if self.build == "ps3" else "Q"
        if text is None:
            self.out += struct.pack(self.order + fmt, 0)
        else:
            self.out += struct.pack(self.order + fmt, len(text) + 1) + text + b"\0"

    def function(self, code=(RETURN_WORD,), constants=(), protos=(), lines=(0, 0)):
        self.string(None)
        self.int(lines[0])
        self.int(lines[1])
        self.out += bytes((0, 0, 2, 2))
        self.int(len(code))
        for word in code:
            self.out += struct.pack(self.order + "I", word)
        self.int(len(constants))
        for value in constants:
            if value is None:
                self.out.append(0)
            elif isinstance(value, bool):
                self.out += bytes((1, value))
            elif isinstance(value, float):
                self.out.append(3)
                self.out += struct.pack(self.order + "d", value)
            else:
                self.out.append(4)
                self.string(value)
        self.int(len(protos))
        for proto in protos:
            self.function(**proto)
        self.int(0)
        self.int(0)
        self.int(0)


def lua_chunk(build: str, **function) -> bytes:
    writer = LuaWriter(build)
    writer.header()
    writer.function(**function)
    return bytes(writer.out)


# --- Appearance ------------------------------------------------------------------------

def hardpoint_bytes(order, bone) -> bytes:
    return struct.pack(
        order + "IIi4f3f", 0x11223344, 2, bone, 0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 3.0,
    )


def transform_bytes(order, q=(0.0, 0.0, 0.0, 1.0), t=(0.0, 0.0, 0.0), s=1.0) -> bytes:
    return struct.pack(order + "8f", *q, *t, s)


def bone_record(build, name, parent, translation, parent_translation) -> bytes:
    """A bone with no rotation or scale, so its transforms are plain differences."""
    order = ">" if build == "ps3" else "<"
    rest = transform_bytes(order, t=translation)
    inverse = transform_bytes(order, t=tuple(-c for c in translation))
    local = transform_bytes(
        order, t=tuple(a - b for a, b in zip(translation, parent_translation)),
    )
    if build == "ps3":
        record = bytearray(176)
        struct.pack_into(">Ii", record, 0, name, parent)
        for position, raw in ((0x30, rest), (0x50, inverse), (0x70, local)):
            record[position:position + 32] = raw
    else:
        record = bytearray(256)
        struct.pack_into("<Ii", record, 0, name, 0)
        struct.pack_into("<i", record, 8, parent)
        for position, raw in (
            (0x34, rest), (0x54, inverse), (0x74, local), (0x94, rest), (0xB4, inverse),
        ):
            record[position:position + 32] = raw
    return bytes(record)


def bone_records(build, bones) -> bytes:
    out = bytearray()
    for name, parent, translation in bones:
        origin = bones[parent][2] if parent >= 0 else (0.0, 0.0, 0.0)
        out += bone_record(build, name, parent, translation, origin)
    return bytes(out)


def build_app_ps3(version=295, skinned=False, bones=(), index_count=3, bad_descriptor=False,
                  vertices=((0, 0, 0), (10, 0, 0), (0, 10, 0)), pad_a=True, b_extra=0,
                  indices=(0, 1, 2), bounds_half=(1000.0, 1000.0, 1000.0), fmt=None,
                  weights=(200, 55, 0, 0), extra_draw_format=None,
                  ranges=((48, 96), (48, 48))):
    order = ">"
    img = Image(0x300, order)
    img.raw(0, sno_head(version, order))
    n = len(vertices)

    def add_draw(fmt_byte, record_at):
        is_skinned = fmt_byte == 0x83
        descriptor = bytearray(16)
        descriptor[0] = fmt_byte
        struct.pack_into(">HH", descriptor, 8, n + (1 if bad_descriptor else 0), index_count)
        draw = bytearray(160)
        d = Image(0, order)
        d.b = draw
        d.put(0x08, "II", *img.append(bytes(descriptor)))
        d.put(0x10, "I", index_count)
        d.put(0x18, "II", *img.append(struct.pack(f">{len(indices)}H", *indices)))
        d.put(0x24, "I", n)
        d.put(0x60, "HH", *ranges[0])
        d.put(0x64, "HH", *ranges[1])
        if is_skinned:
            a = b"".join(struct.pack(">3h3BB", *v, 192, 128, 64, 0x80) for v in vertices)
            b = bytes((weights[0], 0, weights[1], 1, weights[2], 0, weights[3], 0)) * n
            c = b"".join(
                bytes((1, 2, 3, 4, 5, 6, 7, 8)) + struct.pack("<4H", 0x8000 + 512, 0x8000, 9, 10)
                for _ in vertices
            )
        else:
            a = b"".join(struct.pack(">3h", *v) for v in vertices)
            b = b""
            c = b"".join(
                bytes((192, 128, 64, 0, 1, 2, 3, 4, 5, 6, 7, 8))
                + struct.pack("<4H", 0x8000 + 512, 0x8000, 9, 10)
                for _ in vertices
            )
        if pad_a:
            a += bytes(-len(a) % 16)
        b += bytes(b_extra)
        d.put(0x2C, "II", *img.append(a))
        if b:
            d.put(0x54, "II", *img.append(b))
        d.put(0x6C, "II", *img.append(c))
        img.raw(record_at, bytes(draw))

    formats = [fmt if fmt is not None else (0x83 if skinned else 0x81)]
    if extra_draw_format is not None:
        formats.append(extra_draw_format)

    sub = bytearray(308)
    s = Image(0, order)
    s.b = sub
    s.raw(0x48, b"Body\0")
    s.put(0xD0, "3f", 0.0, 0.0, 0.0)
    s.put(0xDC, "3f", *bounds_half)
    draws_position = len(img.b)
    img.b += bytes(160 * len(formats))
    s.put(0xFC, "III", len(formats), draws_position - 0x10, 160 * len(formats))
    s.put(0x118, "3f", 1.0, 2.0, 3.0)
    s.put(0x124, "3f", 0.5, 0.5, 0.5)
    for i, f in enumerate(formats):
        add_draw(f, draws_position + 160 * i)

    img.put(0x24, "i", len(bones))
    if bones:
        img.pair(0x28, bone_records("ps3", bones))
    img.put(0x70, "i", 1)
    img.pair(0x74, bytes(sub))
    img.put(0xC8, "i", 1)
    img.pair(0xCC, hardpoint_bytes(order, -1))
    return img.done()


def switch_vertex(position, extra=b"") -> bytes:
    return (
        struct.pack("<3hh", *position, 0)
        + bytes((192, 128, 64, 0, 1, 2, 3, 4, 5, 6, 7, 8))
        + struct.pack("<4H", 0x8000 + 512, 0x8000, 9, 10)
        + extra
    )


def build_app_switch(version=306, bones=(), sub_objects=None, pool_spare=2, indices=(0, 1, 2)):
    """``sub_objects`` is a list of dicts: rigid, index_count, skin, influences."""
    order = "<"
    img = Image(0x400, order)
    img.raw(0, sno_head(version, order))
    if sub_objects is None:
        sub_objects = [{"rigid": [(0, 0, 0), (10, 0, 0), (0, 10, 0)], "index_count": 3}]
    records = bytearray(240 * len(sub_objects))
    pool = bytearray()
    for i, spec in enumerate(sub_objects):
        rec = Image(0, order)
        rec.b = records
        base = i * 240
        rec.put(base, "I", 0x55)
        rigid = spec.get("rigid", [])
        index_count = spec.get("index_count", 0)
        rec.put(base + 0x28, "i", index_count)
        rec.put(base + 0xA0, "i", len(rigid))
        pool += struct.pack(f"<{index_count}H", *spec.get("indices", indices[:index_count]))
        if rigid:
            rec.put(base + 0xB0, "II", *img.append(b"".join(switch_vertex(v) for v in rigid)))
        skin = spec.get("skin")
        if skin:
            palette = struct.pack(f"<{len(skin['palette'])}i", *skin["palette"])
            verts = b"".join(
                switch_vertex(v, bytes(skin.get("slots", (0, 1, 0, 0))) + bytes(skin.get("weights", (200, 55, 0, 0))))
                for v in skin["vertices"]
            )
            ind = struct.pack(f"<{len(skin['indices'])}H", *skin["indices"])
            s = Image(104, order)
            s.put(0x00, "i", len(skin["palette"]))
            s.put(0x04, "II", *img.append(palette))
            s.put(0x18, "i", len(skin["vertices"]))
            s.put(0x28, "II", *img.append(verts))
            s.put(0x30, "i", len(skin["indices"]))
            s.put(0x40, "II", *img.append(ind))
            rec.put(base + 0xB8, "II", *img.append(s.done()))
        if spec.get("influences"):
            rec.put(base + 0xC8, "II", *img.append(spec["influences"]))
        rec.put(base + 0x6C, "6f", 0.0, 0.0, 0.0, *spec.get("half", (1000.0, 1000.0, 1000.0)))
        rec.put(base + 0xD8, "3f", 1.0, 2.0, 3.0)
        rec.put(base + 0xE4, "3f", 0.5, 0.5, 0.5)
    img.pair(0x220, bytes(pool) + bytes(pool_spare))
    img.put(0x24, "i", len(bones))
    if bones:
        img.pair(0x28, bone_records("switch", bones))
    img.put(0x70, "i", len(sub_objects))
    img.pair(0x74, bytes(records))
    img.put(0xC8, "i", 0)
    return img.done()


# --- Animation -------------------------------------------------------------------------

def key_width(keyframes: int) -> int:
    return 1 if keyframes < 256 else 2 if keyframes <= 65535 else 4


def _pad_to(out: bytearray, position: int) -> None:
    out += bytes(position - len(out))


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


FRAME_CODES = {1: "B", 2: "H", 4: "I"}
VALUE_CODES = {1: "B", 2: "H", 4: "f"}


def translation_blob(frames, raw, codec, keyframes, order, base=(0.0, 0.0, 0.0), tail=b"") -> bytes:
    width = key_width(keyframes)
    code = order + FRAME_CODES[width]
    out = bytearray(struct.pack(code, len(frames)) + bytes((codec,)))
    if len(frames) >= 2:
        _pad_to(out, _align(width + 1, 4))
        out += struct.pack(order + "3f", *base)
    else:
        _pad_to(out, 2 * width)
    for frame in frames:
        out += struct.pack(code, frame)
    _pad_to(out, _align(len(out), codec))
    for row in raw:
        out += struct.pack(f"{order}3{VALUE_CODES[codec]}", *row)
    return bytes(out) + tail


def rotation_blob(frames, raw, keyframes, order) -> bytes:
    width = key_width(keyframes)
    code = order + FRAME_CODES[width]
    out = bytearray(struct.pack(code, len(frames)))
    for frame in frames:
        out += struct.pack(code, frame)
    _pad_to(out, _align(len(out), 2))
    for row in raw:
        out += struct.pack(order + "4h", *row)
    return bytes(out)


def scale_blob(frames, raw, codec, keyframes, order) -> bytes:
    width = key_width(keyframes)
    code = order + FRAME_CODES[width]
    out = bytearray(struct.pack(code, len(frames)) + bytes((codec,)))
    _pad_to(out, _align(width + 1, width))
    for frame in frames:
        out += struct.pack(code, frame)
    _pad_to(out, _align(len(out), codec))
    for value in raw:
        out += struct.pack(order + VALUE_CODES[codec], value)
    return bytes(out)


def simple_permutation(build: str, name="Walk", **changes) -> dict:
    """Two tracks over three keyframes: the first animated, the second empty."""
    order = ">" if build == "ps3" else "<"
    perm = {
        "name": name, "rate": 0.5, "keyframes": 3, "hashes": [0xAA, 0xBB], "blend": [],
        "translation": [
            translation_blob([0, 2], [(1.0, 2.0, 3.0), (3.0, 2.0, 1.0)], 4, 3, order), b"",
        ],
        "rotation": [rotation_blob([0], [(0, 0, 0, 32767)], 3, order), b""],
        "scale": [scale_blob([0, 2], [127, 254], 1, 3, order), b""],
        "attachments": [],
        "nonlinear": [(0.0, 0.0, float(i)) for i in range(3)],
    }
    perm.update(changes)
    if "keyframes" in changes and "nonlinear" not in changes:
        perm["nonlinear"] = [(0.0, 0.0, float(i)) for i in range(perm["keyframes"])]
    return perm


def _ani_record(perm: dict, order: str) -> bytearray:
    rec = Image(360, order)
    rec.put(0x00, "I", 0x11)
    rec.raw(0x04, perm["name"].encode("latin-1"))
    rec.put(0x48, "f", perm["rate"])
    rec.put(0x5C, "2I", 7, 8)
    rec.put(0x64, "I", 100)
    rec.put(0x68, "2f", 0.75, 0.125)
    rec.put(0x70, "2f", 2.0, 3.0)
    rec.put(0x78, "I", perm.get("bone_count", len(perm["hashes"])))
    rec.put(0x90, "I", len(perm["blend"]))
    rec.put(0xA8, "I", perm["keyframes"])
    rec.put(0xE0, "4f", 1.0, 2.0, 3.0, 4.0)
    rec.put(0xF0, "12f", *[float(i) for i in range(12)])
    rec.put(0x120, "f", 2.0)
    rec.put(0x12C, "I", len(perm["attachments"]))
    rec.put(0x148, "3f", 1.0, 0.0, 0.0)
    rec.put(0x154, "I", 77)
    return rec.b


def build_ani(build: str, perms: list, version=None, playback=2, table_size=None) -> bytes:
    order = ">" if build == "ps3" else "<"
    head = sno_head(version or (123 if build == "ps3" else 129), order)
    table_at = 0x50 if build == "ps3" else 0x48
    img = Image(table_at + 360 * len(perms), order)
    img.raw(0, head)
    img.put(0x1C, "IIi", 0x33, playback, 4242)
    size = 360 * len(perms) if table_size is None else table_size
    img.put(0x28, "II", 0 if build == "ps3" else table_at - 0x10, size)
    shared = {}

    def blob_pair(blob: bytes):
        if not blob:
            return 0, 0
        if blob not in shared:
            shared[blob] = img.append(blob)
        return shared[blob]

    for i, perm in enumerate(perms):
        base = table_at + i * 360
        record = Image(0, order)
        record.b = _ani_record(perm, order)
        if build == "switch":
            hashes = perm["hashes"]
            record.put(0x7C, "II", *img.append(struct.pack(f"{order}{len(hashes)}I", *hashes)))
            record.put(0x98, "II", *img.append(
                b"".join(struct.pack(order + "2f", *row) for row in perm["blend"]),
            ))
            for at, key in ((0xAC, "translation"), (0xC0, "rotation"), (0xD0, "scale")):
                rows = b"".join(
                    struct.pack(order + "II", *blob_pair(blob)) + b"\xCC" * 8
                    for blob in perm[key]
                )
                record.put(at, "II", *img.append(rows))
            record.put(0x124, "II", *img.append(b"".join(perm["attachments"])))
            record.put(0x138, "II", *img.append(
                b"".join(struct.pack(order + "3f", *row) for row in perm["nonlinear"]),
            ))
        img.raw(base, bytes(record.b))
    return img.done()


def can_curves(blobs: list) -> bytes:
    table_size = 16 * len(blobs)
    entries = bytearray()
    body = bytearray()
    cursor = table_size
    for blob in blobs:
        if not blob:
            entries += struct.pack(">4I", 0, 0, 0, 0)
            continue
        offset = _align(cursor, 8)
        body += bytes(offset - cursor) + blob
        entries += struct.pack(">4I", offset, len(blob), 0, 0)
        cursor = offset + len(blob)
    return struct.pack(">I", table_size) + bytes(entries) + bytes(body)


def can_permutation(perm: dict) -> bytes:
    def section(raw: bytes) -> bytes:
        return struct.pack(">I", len(raw)) + raw

    hashes = perm["hashes"]
    return (
        section(struct.pack(f">{len(hashes)}I", *hashes))
        + section(b"".join(struct.pack(">2f", *row) for row in perm["blend"]))
        + can_curves(perm["translation"]) + can_curves(perm["rotation"])
        + can_curves(perm["scale"]) + section(b"".join(perm["attachments"]))
        + section(b"".join(struct.pack(">3f", *row) for row in perm["nonlinear"]))
    )


def build_can(perms: list) -> bytes:
    return b"".join(can_permutation(perm) for perm in perms)


# --- Navigation ------------------------------------------------------------------------

def nav_sections(order: str, inner: int, cell_flags=(0x1001, 0x0002)) -> list:
    grid = b"".join(struct.pack(order + "HH", i * 6553, 0x100 + i) for i in range(inner))
    cells = b"".join(
        struct.pack(order + "6fHHI", 0, 1, 2, 3, 4, 5, flags, 2 if i == 0 else 0, 0)
        for i, flags in enumerate(cell_flags)
    )
    neighbours = struct.pack(order + "HH", cell_flags[1] & 0x0FFF, 1)
    neighbours += struct.pack(order + "HH", 0xFFFF, 0x8000)
    coarse = struct.pack(order + "HHH", 1, 2, 0) + struct.pack(order + "HHH", 0xFFFF, 0, 0)
    lookups = struct.pack(order + "HH", 1, 0) + struct.pack(order + "HH", 2, 1)
    borders = struct.pack(order + "HH", 1, 1 | 8 | 0x40)
    return [grid, cells, neighbours, coarse, lookups, borders]


def build_scene(build: str, width=6, height=5, top=1, bottom=1, left=2, right=1, version=None,
                sections=None, body=None, special=b"\x21\x43\x65\x87\x09") -> tuple:
    """Returns ``(scn, nav)``. ``body`` overrides u32 words at body offsets."""
    order = ">" if build == "ps3" else "<"
    inner_w, inner_h = width - left - right, height - top - bottom
    parts = nav_sections(order, inner_w * inner_h) if sections is None else sections
    img = Image(0x228, order)
    img.raw(0, sno_head(version or (163 if build == "ps3" else 179), order))

    def word(at, *values, fmt="I"):
        img.put(0x10 + at, f"{len(values)}{fmt}", *values)

    word(0x40, width, height, 1, width * height, top, bottom, left, right)
    word(0x60, 10.0, 20.0, 15.5, fmt="f")
    word(0x6C, 0xBEEF)
    word(0x180, len(parts[1]) // 32)
    word(0x198, len(parts[2]) // 4)
    word(0x1B0, 10.0, 0.1, fmt="f")
    word(0x1B8, 4, 2, 1)
    word(0x1D8, len(parts[4]) // 4)
    word(0x1F0, len(parts[5]) // 4)
    nav = b""
    if build == "switch":
        for at, raw in zip((0x80, 0x190, 0x1A8, 0x1D0, 0x1E8, 0x200), parts):
            img.pair(0x10 + at, raw)
        img.pair(0x10 + 0x90, special)
    else:
        img.b += special
        nav = b"".join(parts)
    for at, value in (body or {}).items():
        word(at, value)
    return img.done(), nav


# --- Textures --------------------------------------------------------------------------

def gob_address(x, y, wb, bpp, bh):
    """The block-linear address, written out as in the description."""
    xb = x * bpp
    gob_w = (wb * bpp + 63) // 64
    row = y // (8 * bh)
    gob_x = xb // 64
    gob_y = (y // 8) % bh
    in_gob = (
        (xb % 64 // 32) * 256 + (y % 8 // 2) * 64 + (xb % 32 // 16) * 32 + (y % 2) * 16 + xb % 16
    )
    return (row * gob_w + gob_x) * bh * 512 + gob_y * 512 + in_gob


def level_bytes(wb, hb, bpp, bh):
    gob_w = (wb * bpp + 63) // 64
    return gob_w * ((hb + 8 * bh - 1) // (8 * bh)) * bh * 512


def swizzle(linear: bytes, wb: int, hb: int, bpp: int, bh: int) -> bytes:
    out = bytearray(level_bytes(wb, hb, bpp, bh))
    for y in range(hb):
        for x in range(wb):
            at = gob_address(x, y, wb, bpp, bh)
            out[at:at + bpp] = linear[(y * wb + x) * bpp:(y * wb + x + 1) * bpp]
    return bytes(out)


def address_3d(x, y, z, wb, hb, bpp):
    xb = x * bpp
    gob_w = (wb * bpp + 63) // 64
    row = y // 8
    rows = (hb + 7) // 8
    gob_x = xb // 64
    in_gob = (
        (xb % 64 // 32) * 256 + (y % 8 // 2) * 64 + (xb % 32 // 16) * 32 + (y % 2) * 16 + xb % 16
    )
    return ((z // 16 * rows * gob_w + row * gob_w + gob_x) * 16 + z % 16) * 512 + in_gob


NV_FORMATS = {9: 66, 10: 66, 11: 67, 12: 68, 43: 68, 0: 37, 4: 59, 7: 1, 23: 1}
BPP = {9: 8, 10: 8, 11: 16, 12: 16, 43: 16, 0: 4, 4: 2, 7: 1, 23: 1}


def elements_of(fmt, w, h):
    if fmt in (9, 10, 11, 12, 43):
        return (w + 3) // 4, (h + 3) // 4
    return w, h


def nv_container(fmt, w, h, depth, target, image, mip_offsets, layout, slice_size=0,
                 mip_count=None, kinds=(2, 3, 5),
                 magic=bytes((0x44, 0x46, 0x76, 0x4E)), block_magic=bytes((0x48, 0x42, 0x76, 0x4E)),
                 info_size=120) -> bytes:
    info = bytearray(120)
    offsets = list(mip_offsets) + [0] * (17 - len(mip_offsets))
    count = len(mip_offsets) if mip_count is None else mip_count
    struct.pack_into(
        "<Q8I", info, 0, len(image), 512, w, h, depth, target, NV_FORMATS.get(fmt, 0), count,
        slice_size,
    )
    struct.pack_into("<17I", info, 0x28, *offsets)
    struct.pack_into("<I", info, 0x6C, layout)
    info = bytes(info[:info_size])
    out = bytearray(magic + struct.pack("<III", 16, 1, 1))
    bodies = [info, image, b""]

    def block(kind, data, data_offset):
        return block_magic + struct.pack(
            "<IQQIII", 36, len(data), data_offset, kind, 0, 0,
        )

    for kind, data in zip(kinds, bodies):
        out += block(kind, data, 36) + data
    return bytes(out)


def build_tex(build, fmt, w, h, data, mip_max=0, frames=(), faces=1, depth=1, layout=0,
              version=None, image_ids=None, target=None, slice_size=0, mip_offsets=None,
              payload=None, extra_pair=False, trailing=b"", v_scale=0.0, mip_count=None,
              container=None) -> bytes:
    """``data`` is the ps3 payload. On switch it is level 0 as linear elements, unless
    ``payload`` gives the whole container."""
    order = ">" if build == "ps3" else "<"
    img = Image(0x270 if build == "ps3" else 0x288, order)
    img.raw(0, sno_head(version or (58 if build == "ps3" else 66), order))
    img.put(0x20, "5I", fmt, w, h, faces, depth)
    img.put(0x38, "ii", 0, mip_max)
    ids_at, scale_at = (0x24C, 0x264) if build == "ps3" else (0x260, 0x278)
    img.put(scale_at, "f", v_scale)
    ids = list(range(len(frames))) if image_ids is None else image_ids
    if build == "switch":
        img.b += bytes(-len(img.b) % 8)
    img.pair(ids_at, struct.pack(f"{order}{len(ids)}i", *ids))
    img.put(0x220, "i", len(frames))
    if build == "switch":
        img.b += bytes(-len(img.b) % 8)
    img.pair(0x224, b"".join(struct.pack(order + "I4f", *frame) for frame in frames))
    if extra_pair:
        img.put(0x48, "II", 1, 1)
    if build == "switch" and payload is None:
        wb, hb = elements_of(fmt, w, h)
        image = swizzle(data, wb, hb, BPP[fmt], 1 << layout)
        payload = nv_container(
            fmt, w, h, depth, target or 1, image, mip_offsets or [0], layout, slice_size,
            mip_count, **(container or {}),
        )
    img.pair(0x40, data if build == "ps3" else payload)
    return img.done() + trailing


# --- Text and gameplay -----------------------------------------------------------------

def order_of(build: str) -> str:
    return ">" if build == "ps3" else "<"


def build_stl(build: str, rows: list, array_extra=b"", revision=1, source=0x1234) -> bytes:
    """``rows`` holds (label hash, text bytes without the NUL). Equal text shares bytes."""
    order = order_of(build)
    img = Image(0x38, order)
    img.raw(0, sno_head(1, order, revision=revision, source=source))
    img.pair(0x28, bytes(24 * len(rows)) + array_extra)
    shared = {}
    for i, (label, text) in enumerate(rows):
        if text not in shared:
            img.b += bytes(-len(img.b) % 8)
            shared[text] = img.append(text + b"\0")
        img.put(0x38 + i * 24 + 8, "II", *shared[text])
        img.put(0x38 + i * 24 + 16, "II", label, 0)
    return img.done()


GAM_FIXED = {"switch": 0x248, "ps3": 0x3E8}


def build_gam(build: str, kind: int, slot=None, rows=(), row_size=0, extra_slots=None) -> bytes:
    """``rows`` holds a name (or None) for each row of ``row_size`` bytes."""
    order = order_of(build)
    img = Image(GAM_FIXED[build], order)
    img.raw(0, sno_head(1, order))
    img.put(0x1C, "i", kind)
    table = bytearray()
    for name in rows:
        row = bytearray(row_size)
        if name is not None:
            struct.pack_into(order + "II", row, 0, *img.append(name.encode("latin-1") + b"\0"))
        table += row
    if slot is not None:
        img.pair(slot, bytes(table))
    for position, raw in (extra_slots or {}).items():
        img.pair(position, raw)
    return img.done()


def formula_rows(build: str, rows: list) -> bytes:
    order = order_of(build)
    out = bytearray()
    for name, values in rows:
        padded = list(values) + [0.0] * (76 - len(values))
        out += name.encode("latin-1").ljust(1024, b"\0") + struct.pack(f"{order}76f", *padded)
    return bytes(out)


def build_formula_gam(build: str, rows: list) -> bytes:
    slot = 0x3A8 if build == "ps3" else 0x1A8
    return build_gam(build, 36, slot=None, extra_slots={slot: formula_rows(build, rows)})


def build_ans(build: str, default=(), weapons=None, parent=-1) -> bytes:
    order = order_of(build)
    slots = 19 if build == "ps3" else 28
    img = Image(0x30 + 16 * slots, order)
    img.raw(0, sno_head(1, order))
    img.put(0x1C, "i", parent)
    if default:
        img.pair(0x20, tag_map(list(default), order))
    for i, records in (weapons or {}).items():
        img.pair(0x30 + 16 * i, tag_map(list(records), order))
    return img.done()


def build_skl(build: str, traits=(), actives=()) -> bytes:
    order = order_of(build)
    img = Image(0x48, order)
    img.raw(0, sno_head(1, order))
    trait_size, active_size = (12, 40) if build == "ps3" else (16, 64)
    img.pair(0x28, b"".join(
        struct.pack(f"{order}3i", *row).ljust(trait_size, b"\0") for row in traits
    ))
    img.pair(0x38, b"".join(
        struct.pack(f"{order}10i", *row).ljust(active_size, b"\xEE") for row in actives
    ))
    return img.done()


def build_words(build: str, words: list) -> bytes:
    """A file whose body holds u32 words from body 0x0C, for ``.tme`` and ``.lor``."""
    order = order_of(build)
    return sno_head(1, order)[:0x1C] + struct.pack(f"{order}{len(words)}I", *words)


def build_scr(build: str, entries=(), chunk=b"", uid=5, quest=-1, version=None, count=None) -> bytes:
    order = order_of(build)
    version = version or (86 if build == "ps3" else 94)
    img = Image(0x58, order)
    img.raw(0, sno_head(version, order))
    if chunk:
        img.b += bytes(-len(img.b) % 8)
        img.pair(0x20, chunk)
    size = 176 if build == "ps3" else 192
    filter_size, strings_at, handler_at, flags_at = (
        (20, 0x58, 0x98, 0xA8) if build == "ps3" else (24, 0x68, 0xA8, 0xB8)
    )
    if entries:
        img.b += bytes(-len(img.b) % 8)
    table = len(img.b)
    img.pair(0x30, bytes(size * len(entries)))
    shared = {}

    def string(position, text):
        raw = text.encode("latin-1") + b"\0"
        if raw not in shared:
            img.b += bytes(-len(img.b) % 8)
            shared[raw] = img.append(raw)
        img.put(position, "II", *shared[raw])

    for i, entry in enumerate(entries):
        base = table + i * size
        img.put(base, "ii", entry["event"], entry.get("unique", i))
        for slot, (kind, value, second, text) in enumerate(entry.get("filters", ())):
            at = base + 8 + slot * filter_size
            if build == "switch":
                img.put(at, "II", value, second)
                img.put(at + 16, "I", kind)
            else:
                img.put(at, "III", kind, value, second)
            if text is not None:
                string(base + strings_at + slot * 16, text)
        if entry.get("handler"):
            string(base + handler_at, entry["handler"])
        img.put(base + flags_at, "Ii", entry.get("flags", 3), 7)
    img.put(0x40, "i", len(entries) if count is None else count)
    img.put(0x44, "Ii", uid, quest)
    return img.done()


# --- World -----------------------------------------------------------------------------

def build_scene_info(build: str, marker_sets=(), exclusions=(), inclusions=(), appearance=11,
                     collision=12, version=None, xp=1.5) -> bytes:
    order = order_of(build)
    img = Image(0x10 + 0x218, order)
    img.raw(0, sno_head(version or (163 if build == "ps3" else 179), order))
    img.put(0x20, "6f", 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    img.put(0x38, "6f", 7.0, 8.0, 9.0, 10.0, 11.0, 12.0)
    for at, values in ((0xB0, exclusions), (0xF0, inclusions), (0x130, marker_sets)):
        img.pair(at, struct.pack(f"{order}{len(values)}I", *values))
    img.put(0x170, "I", 0xABCD)
    if build == "switch":
        img.put(0x218, "f", xp)
        img.put(0x21C, "ii", appearance, collision)
    else:
        img.put(0x220, "ii", appearance, collision)
    return img.done()


def marker_bytes(img: Image, order: str, marker: dict) -> bytes:
    record = Image(88, order)
    record.put(0, "I", marker.get("hash", 0x55))
    record.put(8, "I", marker.get("type", 0))
    record.put(0x0C, "4f", *marker.get("rotation", (0.0, 0.0, 0.0, 1.0)))
    record.put(0x1C, "3f", *marker.get("position", (0.0, 0.0, 0.0)))
    record.put(0x28, "ii", marker.get("group", 1), marker.get("handle", 9))
    if marker.get("tags"):
        record.put(0x30, "II", *img.append(marker["tags"]))
    links = marker.get("links", ())
    record.put(0x40, "i", marker.get("link_count", len(links)))
    if links:
        record.put(0x44, "II", *img.append(
            b"".join(struct.pack(order + "II", *link) for link in links),
        ))
    return record.done()


def build_mrk(build: str, markers=(), circles=(), label=-1, special=(), handle=7,
              special_count=None) -> bytes:
    order = order_of(build)
    img = Image(0x10 + 0xC8, order)
    img.raw(0, sno_head(1, order, handle=handle))
    table = b"".join(marker_bytes(img, order, marker) for marker in markers)
    img.pair(0x1C, table)
    img.pair(0x60, b"".join(struct.pack(order + "3f", *circle) for circle in circles))
    img.put(0xA0, "6f", 1.0, 1.0, 1.0, 2.0, 2.0, 2.0)
    img.put(0xB8, "I", 3)
    img.put(0xBC, "i", label)
    img.put(0xC0, "i", len(special) if special_count is None else special_count)
    img.pair(0xC4, struct.pack(f"{order}{len(special)}H", *special))
    return img.done()


def scene_entry(build: str, handle, position=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0, 1.0),
                group=33, cluster=4) -> bytes:
    order = order_of(build)
    record = Image(240 if build == "ps3" else 256, order)
    record.put(0, "iI", group, handle)
    record.put(8, "4f", *rotation)
    record.put(0x18, "3f", *position)
    record.put(0x30, "4i", 1, 2, 3, 4)
    record.put(0x9C if build == "ps3" else 0xAC, "i", cluster)
    return record.done()


def build_wrl(build: str, chunks=(), tiles=None, marker_sets=(), generated=0, block_size=None,
              chunk_count=None) -> bytes:
    """``chunks`` is a list of lists of scene entry bytes. ``tiles`` is generator tile rows."""
    order = order_of(build)
    fixed = 0x100 if build == "ps3" else 0xC0
    img = Image(0x10 + fixed, order)
    img.raw(0, sno_head(1, order))
    img.put(0x1C, "I", generated)
    img.pair(0x30, struct.pack(f"{order}{len(marker_sets)}I", *marker_sets))
    at = (0xD4, 0xD8, 0xDC, 0xE0, 0xE4) if build == "ps3" else (0x84, 0x88, 0x98, 0x9C, 0xA0)
    for number, position in enumerate(at):
        img.put(0x10 + position, "i", 100 + number)
    size = 112 if build == "ps3" else 144
    block = Image(size, order)
    if tiles is not None:
        tile_size = 72 if build == "ps3" else 80
        tile_raw = b"".join(
            struct.pack(order + "4i", *row).ljust(tile_size, b"\0") for row in tiles
        )
        generator = Image(120, order)
        generator.put(0, "II", *img.append(tile_raw))
        generator.put(0x40, "i", 1)
        generator.put(0x44, "II", *img.append(b"cmd!"))
        block.put(0, "II", *img.append(generator.done()))
    lists = bytearray()
    for entries in chunks:
        record = Image(24, order)
        record.put(0, "II", *img.append(b"".join(entries)))
        record.put(8, "i", len(entries) if chunk_count is None else chunk_count)
        lists += record.done()
    if lists:
        block.put(0x10, "II", *img.append(bytes(lists)))
    block.put(0x38, "i", 8)
    block.put(0x40, "i", 1)
    block.put(0x50, "II", *img.append(b"clus"))
    block.put(0x58, "4i", 21, 22, 23, 24)
    block.put(0x68 if build == "ps3" else 0x74, "ii", 31, 32)
    raw = block.done()
    if block_size is not None:
        raw = raw[:block_size]
    img.pair(0x20, raw)
    return img.done()


# --- Collision -------------------------------------------------------------------------

def tree_leaf(build: str, first: int, count: int) -> bytes:
    order = order_of(build)
    if build == "switch":
        data = first << 6 | count << 2 | 3
        return struct.pack(order + "6hI", 0, 0, 0, 1, 1, 1, data)
    return struct.pack(order + "IHH", first << 2 | 3, count, 0)


def tree_internal(build: str, right: int, axis: int = 1) -> bytes:
    order = order_of(build)
    if build == "switch":
        return struct.pack(order + "6hI", 0, 0, 0, 1, 1, 1, right << 6 | axis)
    return struct.pack(order + "IHH", right << 2 | axis, 0, 0)


def triangle_row(build: str, vertices, neighbours=None, material=2, flags=3) -> bytes:
    order = order_of(build)
    if build == "switch":
        return struct.pack(order + "6iHH", *vertices, *(neighbours or (-1, -1, -1)), material, flags)
    return struct.pack(
        order + "8H", *vertices, *(neighbours or (0xFFFF, 0xFFFF, 0xFFFF)), material, flags,
    )


def build_phm(build: str, meshes: list, count=None) -> bytes:
    """Each mesh: centre, vertices, triangles (rows), triangle_count, nodes (rows)."""
    order = order_of(build)
    img = Image(0x2C, order)
    img.raw(0, sno_head(1, order))
    img.put(0x1C, "I", 9)
    fields = (
        (0x18, 0x24, 0x30, 0x3C, 0x40, 0x44, None, 0x48, 0x50, 0x58, 0x60, 0x68, 0x6C)
        if build == "switch"
        else (0x20, 0x2C, 0x38, 0x44, 0x48, 0x4C, 0x54, 0x5C, 0x60, 0x68, 0x78, 0x80, 0x84)
    )
    centre, half, scale, nodes, vertices, tris, entries, height, vp, tp, np, flags, material = fields
    table = bytearray()
    for mesh in meshes:
        record = Image(136 if build == "ps3" else 112, order)
        record.put(centre, "3f", *mesh["centre"])
        record.put(half, "3f", 1.0, 1.0, 1.0)
        record.put(scale, "3f", 2.0, 2.0, 2.0)
        record.put(nodes, "i", len(mesh["nodes"]))
        record.put(vertices, "i", len(mesh["vertices"]))
        record.put(tris, "i", mesh.get("triangle_count", len(mesh["triangles"])))
        if entries is not None:
            record.put(entries, "i", len(mesh["triangles"]))
        record.put(height, "i", 2)
        record.put(vp, "II", *img.append(b"".join(
            struct.pack(order + "4f", *v, 0.0) for v in mesh["vertices"]
        )))
        record.put(tp, "II", *img.append(b"".join(mesh["triangles"])))
        record.put(np, "II", *img.append(b"".join(mesh["nodes"])))
        record.put(flags, "II", 5, 6)
        table += record.done()
    img.put(0x20, "i", len(meshes) if count is None else count)
    img.pair(0x24, bytes(table))
    return img.done()


# --- Materials -------------------------------------------------------------------------

SWITCH_COLOURS = [float(i) for i in range(16)]


def material_block(img: Image, build: str, textures=()) -> bytes:
    """A material block. ``textures`` holds (slot type, handle)."""
    order = order_of(build)
    block = Image(72 if build == "ps3" else 104, order)
    block.put(0, "i", 77)
    if build == "switch":
        block.put(0x04, "16f", *SWITCH_COLOURS)
        block.put(0x44, "fI", 2.5, 8)
        width, animation_at, handle_at, list_at = 32, 0x08, 0x18, 0x4C
    else:
        for i in range(4):
            block.raw(0x04 + i * 8, bytes((i, i + 1, i + 2, i + 3)))
            block.put(0x08 + i * 8, "f", i + 0.5)
        block.put(0x24, "fI", 2.5, 8)
        width, animation_at, handle_at, list_at = 24, 0x04, 0x14, 0x2C
    rows = bytearray()
    for slot_type, handle in textures:
        row = Image(width, order)
        row.put(0, "I", slot_type)
        row.put(animation_at, "II", *img.append(b"anim"))
        row.put(handle_at, "i", handle)
        rows += row.done()
    block.put(list_at, "II", *img.append(bytes(rows)))
    return block.done()


def build_mat(build: str, textures=(), tags=b"", list_size=None) -> bytes:
    order = order_of(build)
    fixed = 0x68 if build == "ps3" else 0x88
    img = Image(0x10 + fixed, order)
    img.raw(0, sno_head(1, order))
    img.put(0x1C, "I", 4)
    if tags:
        img.pair(0x20, tags)
    block = material_block(img, build, textures)
    if list_size is not None:
        at = 0x2C if build == "ps3" else 0x4C
        patched = Image(0, order)
        patched.b = bytearray(block)
        patched.put(at + 4, "I", list_size)
        block = patched.done()
    img.raw(0x30, block)
    return img.done()


def add_app_materials(data: bytes, build: str, looks, materials) -> bytes:
    """Add looks and materials to an ``.app``. ``materials`` holds (hash, per-look textures)."""
    order = order_of(build)
    img = Image(0, order)
    img.b = bytearray(data)
    look_at, count_at, materials_at, looks_at, record_size, look_size, handle_at = (
        (0x188, 0x18C, 0x190, 0x1A0, 20, 184, 0x60) if build == "ps3"
        else (0x178, 0x17C, 0x180, 0x190, 24, 248, 0x80)
    )
    img.put(look_at, "ii", len(looks), len(materials))
    img.pair(looks_at, struct.pack(f"{order}{len(looks)}I", *looks))
    table = bytearray()
    for material_hash, per_look in materials:
        rows = bytearray()
        for number, textures in enumerate(per_look):
            row = Image(look_size, order)
            row.raw(0x18, material_block(img, build, textures))
            row.put(handle_at, "i", 900 + number)
            rows += row.done()
        record = Image(record_size, order)
        record.put(0, "I", material_hash)
        record.put(8, "II", *img.append(bytes(rows)))
        table += record.done()
    img.pair(materials_at, bytes(table))
    return img.done()


def set_sub_object_material(data: bytes, build: str, index: int, material_hash, sub_hash=0x77) -> bytes:
    order = order_of(build)
    out = bytearray(data)
    table = struct.unpack_from(order + "I", out, 0x74)[0] + 0x10
    size, at = (308, 0xC8) if build == "ps3" else (240, 0x64)
    struct.pack_into(order + "II", out, table + index * size + at, material_hash, sub_hash)
    return bytes(out)


# --- Sound -----------------------------------------------------------------------------

def build_sbk(build: str, rows: list, count=None) -> bytes:
    order = order_of(build)
    img = Image(0x40, order)
    img.raw(0, sno_head(46, order))
    size, stored_at, count_at = (40, 0x20, 0x24) if build == "ps3" else (16, 0x0C, 0x30)
    table = bytearray()
    for sample_hash, stored in rows:
        row = Image(size, order)
        row.put(0, "I", sample_hash)
        row.put(stored_at, "I", stored)
        table += row.done()
    img.pair(0x1C, bytes(table))
    img.put(count_at, "i", len(rows) if count is None else count)
    return img.done()


def build_snd(build: str, rows: list, count=None) -> bytes:
    order = order_of(build)
    img = Image(0x70, order)
    img.raw(0, sno_head(1, order))
    size, pair_at, count_at = (44, 0x50, 0x58) if build == "ps3" else (64, 0x54, 0x5C)
    table = bytearray()
    for bank, sample_hash, weight in rows:
        row = Image(size, order)
        row.put(0, "iII", bank, sample_hash, weight)
        table += row.done()
    img.pair(pair_at, bytes(table))
    img.put(count_at, "i", len(rows) if count is None else count)
    return img.done()
