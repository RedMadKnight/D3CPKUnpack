"""``.tex`` files: header, payload layouts, pixel decoding, frames and PNG output."""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass, field

from .binary import ByteView, FormatError, Pair, build_order, to_f32
from .index import Sprite
from .sno import SnoHeader, check_version, parse_header, resolve_build

_ORDER_FIELDS = {
    # image ids pair, cube flag, import flags, machine time, console mip offset, V scale, volume
    "ps3": (0x24C, 0x254, 0x258, 0x25C, 0x260, 0x264, 0x268),
    "switch": (0x260, 0x268, 0x26C, 0x270, 0x274, 0x278, 0x27C),
}
PAYLOAD_AT = 0x40
FRAME_PAIR_AT = 0x224
PS3_FIXED_SIZE = 0x270
_PAIR_SLOTS = 60
_FRAME_SIZE = 20
MAX_MIPS = 17
_MAX_BLOCK_HEIGHT_LOG2 = 8
CUBE_FACES = 6

# Format code: (decoder, bytes per element).
FORMAT_NORMAL = 43
FORMATS = {
    9: ("dxt1", 8),
    10: ("dxt1", 8),
    11: ("dxt3", 16),
    12: ("dxt5", 16),
    43: ("dxt5", 16),
    0: ("argb8888", 4),
    4: ("argb1555", 2),
    7: ("l8", 1),
    23: ("a8", 1),
}
_NV_FORMATS = {9: 66, 10: 66, 11: 67, 12: 68, 43: 68, 0: 37, 4: 59, 7: 1, 23: 1}
NV_FILE_MAGIC = bytes((0x44, 0x46, 0x76, 0x4E))
NV_BLOCK_MAGIC = bytes((0x48, 0x42, 0x76, 0x4E))
_NV_INFO_SIZE = 120
NV_TARGET_2D = 1
NV_TARGET_3D = 2
NV_TARGET_CUBE = 8
_NV_3D_LAYOUT = 0x40
_NV_3D_DEPTH = 16


@dataclass(frozen=True)
class TextureFrame:
    hash: int
    u0: float
    v0: float
    u1: float
    v1: float

    def rect(self, width: int, height: int) -> tuple[int, int, int, int]:
        """Left, top, right, bottom in pixels, from the header's size.

        The product is taken in single precision. Right and bottom are exclusive.
        """
        w, h = to_f32(width), to_f32(height)
        try:
            return (
                int(to_f32(self.u0 * w)), int(to_f32(self.v0 * h)),
                int(to_f32(self.u1 * w)), int(to_f32(self.v1 * h)),
            )
        except (ValueError, OverflowError):
            raise FormatError(f"frame {self.hash:08X} has non-finite coordinates") from None


@dataclass
class NvTexture:
    data_size: int
    alignment: int
    width: int
    height: int
    depth: int
    target: int
    pixel_format: int
    mip_count: int
    slice_size: int
    mip_offsets: tuple[int, ...]
    layout: int
    image: bytes
    image_offset: int = 0  # position of the image data in the payload


@dataclass
class Image:
    width: int
    height: int
    rgba: bytes


@dataclass
class TextureFile:
    header: SnoHeader
    build: str
    format: int
    width: int
    height: int
    faces: int
    depth: int
    mip_min: int
    mip_max: int
    payload_pair: Pair
    payload: bytes
    frames: list[TextureFrame]
    image_ids: list[int]
    cube_flag: int
    import_flags: int
    machine_time: int
    console_mip_offset: int
    lightmap_v_scale: float
    volume_x: int
    volume_y: int
    nv: NvTexture | None = None
    warnings: list[str] = field(default_factory=list)
    fixed: bytes = b""
    image_ids_raw: bytes = b""
    frames_raw: bytes = b""

    @property
    def levels(self) -> int:
        return self.mip_max - self.mip_min + 1

    def level_size(self, level: int) -> tuple[int, int]:
        return max(self.width >> level, 1), max(self.height >> level, 1)

    def level_depth(self, level: int) -> int:
        return max(self.depth >> level, 1)


def _parse_nv(payload: bytes, base: int) -> NvTexture:
    view = ByteView(payload, "<")
    if payload[:4] != NV_FILE_MAGIC:
        raise FormatError("payload lacks the container header", base)
    if view.unpack("III", 4, "container header") != (16, 1, 1):
        raise FormatError("unexpected container header", base + 4)
    pos = 16
    start, info_raw = _read_block(view, pos, 2, base)
    if len(info_raw) != _NV_INFO_SIZE:
        raise FormatError(f"texture info of {len(info_raw)} bytes", base + pos)
    info = ByteView(info_raw, "<")
    fields = info.unpack("Q7II", 0)
    pos = start + _NV_INFO_SIZE
    image_at, image = _read_block(view, pos, 3, base)
    _read_block(view, image_at + fields[0], 5, base)
    return NvTexture(
        fields[0], fields[1], fields[2], fields[3], fields[4], fields[5], fields[6], fields[7],
        fields[8], info.unpack(f"{MAX_MIPS}I", 0x28), info.u32(0x6C), image, image_at,
    )


def _read_block(view: ByteView, pos: int, wanted: int, base: int) -> tuple[int, bytes]:
    """Read the block header at ``pos``. Returns where its data starts, and the data."""
    if view.take(pos, 4, "block header") != NV_BLOCK_MAGIC:
        raise FormatError("expected a block header", base + pos)
    size, data_size, data_offset, kind, _, _ = view.unpack("IQQIII", pos + 4, "block header")
    if size != 36:
        raise FormatError(f"block header size {size}", base + pos + 4)
    if kind != wanted:
        raise FormatError(f"block type {kind}, expected {wanted}", base + pos)
    return pos + data_offset, view.take(pos + data_offset, data_size, "block data")


def _element_grid(tex: TextureFile, level: int) -> tuple[int, int]:
    width, height = tex.level_size(level)
    if FORMATS[tex.format][0].startswith("dxt"):
        return (width + 3) // 4, (height + 3) // 4
    return width, height


def standard_block_height(hb: int) -> int:
    """The block height a level 0 surface of ``hb`` elements should have."""
    n = hb + hb // 2
    for limit, bh in ((128, 16), (64, 8), (32, 4), (16, 2)):
        if n >= limit:
            return bh
    return 1


def _level_block_height(nv: NvTexture, hb: int) -> int:
    bh = 1 << nv.layout
    while bh > 1 and hb <= (bh // 2) * 8:
        bh //= 2
    return bh


def image_id_pair_at(build: str) -> int:
    return _ORDER_FIELDS[build][0]


def _surface_bytes(wb: int, hb: int, bpp: int, bh: int) -> int:
    gob_width = (wb * bpp + 63) // 64
    return gob_width * ((hb + 8 * bh - 1) // (8 * bh)) * bh * 512


def _check_nv(tex: TextureFile) -> None:
    nv = tex.nv
    if nv.target not in (NV_TARGET_2D, NV_TARGET_3D, NV_TARGET_CUBE):
        raise FormatError(f"texture target {nv.target}", tex.payload_pair.file_offset)
    if nv.target != NV_TARGET_3D and nv.layout > _MAX_BLOCK_HEIGHT_LOG2:
        raise FormatError(f"layout word {nv.layout}", tex.payload_pair.file_offset)
    if nv.target == NV_TARGET_3D and nv.layout != _NV_3D_LAYOUT:
        raise FormatError(f"3D layout word 0x{nv.layout:X}", tex.payload_pair.file_offset)
    if (nv.width, nv.height, nv.depth) != (tex.width, tex.height, tex.depth):
        raise FormatError(
            f"container size {nv.width}x{nv.height}x{nv.depth} differs from the header",
            tex.payload_pair.file_offset,
        )
    if _NV_FORMATS.get(tex.format) != nv.pixel_format:
        raise FormatError(
            f"container format {nv.pixel_format} with format {tex.format}",
            tex.payload_pair.file_offset,
        )
    if nv.mip_count != tex.levels:
        tex.warnings.append(f"container has {nv.mip_count} mips, header {tex.levels}")
    if nv.target == NV_TARGET_3D or tex.format not in FORMATS:
        return
    standard = standard_block_height(_element_grid(tex, 0)[1])
    if 1 << nv.layout != standard:
        tex.warnings.append(f"block height {1 << nv.layout}, the standard rule gives {standard}")
    bpp = FORMATS[tex.format][1]
    for level in range(min(nv.mip_count, MAX_MIPS) - 1):
        wb, hb = _element_grid(tex, level)
        size = _surface_bytes(wb, hb, bpp, _level_block_height(nv, hb))
        if nv.mip_offsets[level + 1] < nv.mip_offsets[level] + size:
            tex.warnings.append(f"mip {level + 1} offset overlaps mip {level}")


def parse_texture(data: bytes, build: str | None = None, strict: bool = False) -> TextureFile:
    """Read a ``.tex``. With ``strict``, data warnings become FormatError."""
    header = parse_header(data, build)
    resolved = resolve_build("tex", header, build, strict)
    view = ByteView(data, build_order(resolved))
    fmt, width, height, faces, depth = view.unpack("5I", 0x20, "texture header")
    if width == 0 or height == 0:
        raise FormatError(f"texture size {width}x{height}", 0x24)
    if faces not in (1, CUBE_FACES):
        raise FormatError(f"face count {faces}", 0x2C)
    if depth not in (1, 64):
        raise FormatError(f"depth {depth}", 0x30)
    mip_min, mip_max = view.unpack("ii", 0x38, "mip range")
    if not 1 <= mip_max - mip_min + 1 <= MAX_MIPS:
        raise FormatError(f"mip range {mip_min} to {mip_max}", 0x38)
    warnings = []
    if not check_version("tex", header.version, resolved):
        warnings.append(f"version {header.version} is not this build's; PC files are not covered")
    for slot in range(1, _PAIR_SLOTS):
        if not view.pair(PAYLOAD_AT + slot * 8).is_empty:
            warnings.append(f"payload pair {slot} is not zero")
            break
    payload_pair = view.pair(PAYLOAD_AT, "payload pair")
    payload = view.pair_data(payload_pair, PAYLOAD_AT, "payload")
    if payload and payload_pair.file_offset + payload_pair.size != len(data):
        raise FormatError("payload does not end the file", PAYLOAD_AT)

    frame_count = view.i32(0x220, "frame count")
    frame_pair = view.pair(FRAME_PAIR_AT, "frame pair")
    if frame_count < 0 or frame_pair.size != frame_count * _FRAME_SIZE:
        raise FormatError(
            f"frame pair size {frame_pair.size} for {frame_count} frames", FRAME_PAIR_AT,
        )
    frames_bytes = view.pair_data(frame_pair, FRAME_PAIR_AT, "frames")
    frames_raw = ByteView(frames_bytes, view.order)
    frames = [
        TextureFrame(frames_raw.u32(i * _FRAME_SIZE), *frames_raw.f32s(i * _FRAME_SIZE + 4, 4))
        for i in range(frame_count)
    ]

    ids_at, cube_at, import_at, time_at, console_at, scale_at, volume_at = _ORDER_FIELDS[resolved]
    ids_pair = view.pair(ids_at, "image id pair")
    ids_raw = view.pair_data(ids_pair, ids_at, "image ids")
    if len(ids_raw) % 4:
        raise FormatError(f"image id block of {len(ids_raw)} bytes", ids_at)
    image_ids = list(ByteView(ids_raw, view.order).unpack(f"{len(ids_raw) // 4}i", 0))

    tex = TextureFile(
        header, resolved, fmt, width, height, faces, depth, mip_min, mip_max, payload_pair,
        payload, frames, image_ids, view.u32(cube_at), view.u32(import_at), view.u32(time_at),
        view.u32(console_at), view.f32(scale_at), view.u32(volume_at), view.u32(volume_at + 4),
        warnings=warnings, image_ids_raw=ids_raw, frames_raw=frames_bytes,
    )
    if resolved == "ps3":
        fixed_size = PS3_FIXED_SIZE
    else:
        present = [p.file_offset for p in (ids_pair, frame_pair, payload_pair) if p.size]
        fixed_size = min(present, default=len(data))
    tex.fixed = bytes(data[:fixed_size])
    if resolved == "switch" and payload:
        tex.nv = _parse_nv(payload, payload_pair.file_offset)
        _check_nv(tex)
    if strict and tex.warnings:
        raise FormatError(tex.warnings[0], 0x20)
    return tex


# --- Payload addressing -------------------------------------------------------------------

def _ps3_level_bytes(tex: TextureFile, level: int) -> int:
    wb, hb = _element_grid(tex, level)
    return wb * hb * FORMATS[tex.format][1] * tex.level_depth(level)


def _morton_tables(width: int, height: int, depth: int) -> tuple[list[int], list[int], list[int]]:
    """Per-axis address bits. An axis stops taking bits once its size is used up."""
    sizes = (width, height, depth)
    tables = [[0] * size for size in sizes]
    out_bit = 0
    for bit in range(32):
        for axis in range(3):
            if (1 << bit) < sizes[axis]:
                for value in range(sizes[axis]):
                    if value >> bit & 1:
                        tables[axis][value] |= 1 << out_bit
                out_bit += 1
    return tables[0], tables[1], tables[2]


def _ps3_offsets(tex: TextureFile, level: int, face: int, z: int):
    chain = sum(_ps3_level_bytes(tex, m) for m in range(tex.levels))
    stride = (chain + 127) & ~127
    start = face * stride + sum(_ps3_level_bytes(tex, m) for m in range(level))
    size = _ps3_level_bytes(tex, level)
    if start + size > len(tex.payload):
        raise FormatError(
            f"level {level} runs past the payload", tex.payload_pair.file_offset + start,
        )
    if tex.depth == 1:
        return range(start, start + size, FORMATS[tex.format][1])
    if FORMATS[tex.format][0] != "argb8888":
        raise FormatError(f"volume in format {tex.format}", 0x20)
    width, height = tex.level_size(level)
    x_bits, y_bits, z_bits = _morton_tables(width, height, tex.level_depth(level))
    offsets = []
    for y in range(height):
        row = y_bits[y] | z_bits[z]
        for x in range(width):
            at = start + (x_bits[x] | row) * 4
            if at + 4 > start + size:
                raise FormatError("Morton address past the level", tex.payload_pair.file_offset)
            offsets.append(at)
    return offsets


def _gob_tables(wb: int, hb: int, bpp: int, bh: int):
    gob_width = (wb * bpp + 63) // 64
    x_part = []
    for x in range(wb):
        xb = x * bpp
        in_gob = (xb % 64 // 32) * 256 + (xb % 32 // 16) * 32 + xb % 16
        x_part.append((xb // 64, in_gob))
    y_part = []
    for y in range(hb):
        in_gob = (y % 8 // 2) * 64 + (y % 2) * 16
        y_part.append((y // (8 * bh), y // 8 % bh, in_gob))
    return gob_width, x_part, y_part


def _switch_offsets(tex: TextureFile, level: int, face: int, z: int) -> list[int]:
    nv = tex.nv
    if level >= min(nv.mip_count, MAX_MIPS):
        raise FormatError(f"no level {level} in the container", tex.payload_pair.file_offset)
    bpp = FORMATS[tex.format][1]
    wb, hb = _element_grid(tex, level)
    if wb * hb * bpp > len(nv.image):
        raise FormatError(f"level {level} is larger than the image data", tex.payload_pair.file_offset)
    is_3d = nv.target == NV_TARGET_3D
    bh = 1 if is_3d else _level_block_height(nv, hb)
    gob_width, x_part, y_part = _gob_tables(wb, hb, bpp, bh)
    rows = (hb + 7) // 8
    base = nv.mip_offsets[level] + (0 if is_3d else face * nv.slice_size)
    limit = len(nv.image) - bpp
    offsets = []
    for row, gob_y, y_in_gob in y_part:
        for gob_x, x_in_gob in x_part:
            if is_3d:
                gob = (z // _NV_3D_DEPTH * rows + row) * gob_width + gob_x
                at = base + (gob * _NV_3D_DEPTH + z % _NV_3D_DEPTH) * 512 + y_in_gob + x_in_gob
            else:
                at = (
                    base + ((row * gob_width + gob_x) * bh + gob_y) * 512 + y_in_gob + x_in_gob
                )
            if at > limit:
                raise FormatError(
                    "block-linear address past the image data", tex.payload_pair.file_offset,
                )
            offsets.append(nv.image_offset + at)
    return offsets


def element_offsets(tex: TextureFile, level: int = 0, face: int = 0, z: int = 0):
    """Where each element of a surface sits in the payload, in row order."""
    if tex.format not in FORMATS:
        raise FormatError(f"unknown texture format {tex.format}", 0x20)
    if not tex.payload:
        raise FormatError("texture has no payload", PAYLOAD_AT)
    if not 0 <= level < tex.levels:
        raise FormatError(f"no level {level}: the file has {tex.levels}", 0x38)
    if not 0 <= face < tex.faces:
        raise FormatError(f"no face {face}: the file has {tex.faces}", 0x2C)
    if not 0 <= z < tex.level_depth(level):
        raise FormatError(f"no slice {z} in level {level}", 0x30)
    wb, hb = _element_grid(tex, level)
    if wb * hb * FORMATS[tex.format][1] > len(tex.payload):
        raise FormatError(f"level {level} is larger than the payload", PAYLOAD_AT)
    if tex.build == "ps3":
        return _ps3_offsets(tex, level, face, z)
    return _switch_offsets(tex, level, face, z)


def _read_elements(tex: TextureFile, offsets, bpp: int) -> bytes:
    payload = tex.payload
    if isinstance(offsets, range):
        return payload[offsets.start:offsets.stop]
    return b"".join([payload[at:at + bpp] for at in offsets])


# --- Pixel decoding ------------------------------------------------------------------------

def _rgb565(value: int) -> tuple[int, int, int]:
    r = value >> 11 & 31
    g = value >> 5 & 63
    b = value & 31
    return (r << 3 | r >> 2, g << 2 | g >> 4, b << 3 | b >> 2)


def _colour_palette(c0: int, c1: int, three_colour: bool) -> list[tuple[int, int, int, int]]:
    a = _rgb565(c0)
    b = _rgb565(c1)
    if three_colour:
        mid = tuple((a[i] + b[i]) // 2 for i in range(3))
        return [(*a, 255), (*b, 255), (*mid, 255), (0, 0, 0, 0)]
    one = tuple((2 * a[i] + b[i]) // 3 for i in range(3))
    two = tuple((a[i] + 2 * b[i]) // 3 for i in range(3))
    return [(*a, 255), (*b, 255), (*one, 255), (*two, 255)]


def _alpha_palette(a0: int, a1: int) -> list[int]:
    if a0 > a1:
        return [a0, a1] + [((7 - i) * a0 + i * a1) // 7 for i in range(1, 7)]
    return [a0, a1] + [((5 - i) * a0 + i * a1) // 5 for i in range(1, 5)] + [0, 255]


def _decode_dxt(raw: bytes, width: int, height: int, kind: str) -> bytes:
    wb, hb = (width + 3) // 4, (height + 3) // 4
    block_size = 8 if kind == "dxt1" else 16
    out = bytearray(width * height * 4)
    for by in range(hb):
        rows = min(4, height - by * 4)
        for bx in range(wb):
            at = (by * wb + bx) * block_size
            colour_at = at + (block_size - 8)
            c0, c1, bits = struct.unpack_from("<HHI", raw, colour_at)
            palette = _colour_palette(c0, c1, kind == "dxt1" and c0 <= c1)
            pixels = bytearray(64)
            for i in range(16):
                pixels[i * 4:i * 4 + 4] = bytes(palette[bits >> (2 * i) & 3])
            if kind == "dxt3":
                alpha = int.from_bytes(raw[at:at + 8], "little")
                for i in range(16):
                    pixels[i * 4 + 3] = (alpha >> (4 * i) & 15) * 17
            elif kind == "dxt5":
                alphas = _alpha_palette(raw[at], raw[at + 1])
                codes = int.from_bytes(raw[at + 2:at + 8], "little")
                for i in range(16):
                    pixels[i * 4 + 3] = alphas[codes >> (3 * i) & 7]
            columns = min(4, width - bx * 4)
            for py in range(rows):
                target = ((by * 4 + py) * width + bx * 4) * 4
                out[target:target + columns * 4] = pixels[py * 16:py * 16 + columns * 4]
    return bytes(out)


def _expand5(value: int) -> int:
    return value << 3 | value >> 2


def _decode_argb1555(raw: bytes, count: int, big_endian: bool) -> bytes:
    words = struct.unpack(f"{'>' if big_endian else '<'}{count}H", raw[:count * 2])
    out = bytearray(count * 4)
    for i, word in enumerate(words):
        first = _expand5(word >> 10 & 31)
        middle = _expand5(word >> 5 & 31)
        last = _expand5(word & 31)
        red, blue = (first, last) if big_endian else (last, first)
        out[i * 4:i * 4 + 4] = bytes((red, middle, blue, 255 if word >> 15 else 0))
    return bytes(out)


def _decode_elements(kind: str, build: str, raw: bytes, width: int, height: int) -> bytes:
    count = width * height
    if kind.startswith("dxt"):
        return _decode_dxt(raw, width, height, kind)
    if kind == "argb1555":
        return _decode_argb1555(raw, count, build == "ps3")
    out = bytearray(count * 4)
    if kind == "argb8888":
        if build == "switch":
            return bytes(raw[:count * 4])
        out[0::4] = raw[1::4]
        out[1::4] = raw[2::4]
        out[2::4] = raw[3::4]
        out[3::4] = raw[0::4]
    elif kind == "l8":
        out[0::4] = out[1::4] = out[2::4] = raw[:count]
        out[3::4] = b"\xFF" * count
    else:
        out[3::4] = raw[:count]
    return bytes(out)


def decode_image(tex: TextureFile, level: int = 0, face: int = 0, z: int = 0) -> Image:
    """Decode one level of one face or volume slice to RGBA."""
    offsets = element_offsets(tex, level, face, z)
    width, height = tex.level_size(level)
    kind, bpp = FORMATS[tex.format]
    raw = _read_elements(tex, offsets, bpp)
    return Image(width, height, _decode_elements(kind, tex.build, raw, width, height))


def decode_normal(tex: TextureFile, level: int = 0, face: int = 0) -> Image:
    """Format 43 as a normal map: X (alpha) in red and Y (green) in green, nothing else."""
    if tex.format != FORMAT_NORMAL:
        raise FormatError(f"format {tex.format} does not hold a normal map", 0x20)
    image = decode_image(tex, level, face)
    out = bytearray(len(image.rgba))
    out[0::4] = image.rgba[3::4]
    out[1::4] = image.rgba[1::4]
    out[3::4] = b"\xFF" * (len(out) // 4)
    return Image(image.width, image.height, bytes(out))


def decode_levels(tex: TextureFile, face: int = 0, z: int = 0) -> list[Image]:
    return [decode_image(tex, level, face, z) for level in range(tex.levels)]


def decode_surfaces(tex: TextureFile, level: int = 0) -> list[Image]:
    """Every cube face or volume slice of a level, in stored order."""
    return [
        decode_image(tex, level, face, z)
        for face in range(tex.faces)
        for z in range(tex.level_depth(level))
    ]


def vertical_strip(images: list[Image]) -> Image:
    if not images or len({image.width for image in images}) != 1:
        raise FormatError("a strip needs images of one width")
    return Image(
        images[0].width, sum(image.height for image in images),
        b"".join(image.rgba for image in images),
    )


def crop(image: Image, left: int, top: int, right: int, bottom: int) -> Image:
    if not 0 <= left < right <= image.width or not 0 <= top < bottom <= image.height:
        raise FormatError(f"rectangle {left},{top},{right},{bottom} outside the image")
    rows = [
        image.rgba[(y * image.width + left) * 4:(y * image.width + right) * 4]
        for y in range(top, bottom)
    ]
    return Image(right - left, bottom - top, b"".join(rows))


def frame_image(tex: TextureFile, index: int, level: int = 0) -> Image:
    """One frame, cropped. The rectangle comes from the header's size, then scales by level."""
    if not 0 <= index < len(tex.frames):
        raise FormatError(f"no frame {index}: the file has {len(tex.frames)}", 0x220)
    image = decode_image(tex, level)
    rect = tex.frames[index].rect(tex.width, tex.height)
    return crop(image, *(value >> level for value in rect))


def sprite_image(tex: TextureFile, sprite: Sprite, level: int = 0) -> Image:
    """The frame a sprite table record names. Its hash must equal the frame's."""
    if not 0 <= sprite.frame < len(tex.frames):
        raise FormatError(f"no frame {sprite.frame}: the file has {len(tex.frames)}", 0x220)
    if tex.frames[sprite.frame].hash != sprite.name_hash:
        raise FormatError(f"sprite hash differs from frame {sprite.frame}", 0x224)
    return frame_image(tex, sprite.frame, level)


def _png_chunk(kind: bytes, body: bytes) -> bytes:
    crc = zlib.crc32(kind + body) & 0xFFFFFFFF
    return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", crc)


def encode_png(image: Image) -> bytes:
    """An 8-bit RGBA PNG. Stored values go in unchanged, with no gamma chunk."""
    stride = image.width * 4
    rows = bytearray()
    for y in range(image.height):
        rows.append(0)
        rows += image.rgba[y * stride:(y + 1) * stride]
    head = struct.pack(">IIBBBBB", image.width, image.height, 8, 6, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n" + _png_chunk(b"IHDR", head)
        + _png_chunk(b"IDAT", zlib.compress(bytes(rows))) + _png_chunk(b"IEND", b"")
    )
