"""x360 texture payloads (``20_corrections9.md`` G2-T), an optional decoder.

``texture.parse_texture`` and the ``.tex`` writer keep the X5 rule: the payload is copied as one
block. This module adds, on request, the pixel layout of that block:

- T1: elements are 4 x 4 DXT blocks (8 or 16 bytes) or texels (4, 2 or 1 bytes). DXT and ARGB1555
  data have the two bytes of every 16-bit word swapped; after the swap a DXT block is the standard
  little-endian block. ARGB1555 is then read as a big-endian stored word (A1 R5 G5 B5). ARGB8888 is
  A, R, G, B. L8 and A8 are single bytes.
- T2: a 2D surface (or a cube face) is tiled; ``tiled_element`` gives the element number.
- T3: levels with a smaller side over 16 texels are surfaces of their own; from the first level
  whose smaller side is 16 or less (only when the texture has more than one level), every level
  shares one tail surface. Surfaces are (elements rounded up to 32) per side, rounded to 4,096.
- T4: packed level p sits at ``tail_offset`` inside the tail (V9-T4: from p = 3 on, based on
  the first packed level's longer side ``L`` instead of 16).
- T5: cube faces follow each other within a level (and within the tail).
- T6: volumes (64 x 64 x 64 ARGB8888) use the bit permutation ``VOLUME_BITS``.

``build_payload`` applies the rules in reverse (place, then swap; padding is zero).
"""

from __future__ import annotations

from dataclasses import dataclass

from .binary import FormatError
from .texture import FORMATS, Image, TextureFile, _decode_elements

TILE = 32
SURFACE_ALIGN = 4096
PACK_LIMIT = 16  # texels: a level whose smaller side is at most this is packed into the tail
VOLUME_SIDE = 64
VOLUME_BYTES = VOLUME_SIDE ** 3 * 4
SWAPPED_KINDS = ("dxt1", "dxt3", "dxt5", "argb1555")

# T6 (``x360_tex_volume_bits.tsv``): element-index bit -> the coordinate bits xored into it.
VOLUME_BITS = (
    ("x0",), ("x1",), ("y0",), ("x2",), ("x3",), ("x4", "y3", "z2"), ("y1",), ("y2",), ("z0",),
    ("y3", "z2"), ("z1",), ("x5",), ("y4",), ("y5",), ("z2",), ("z3",), ("z4",), ("z5",),
)


def _round(value: int, step: int) -> int:
    return (value + step - 1) // step * step


def _log2(value: int) -> int:
    if value & (value - 1):
        raise FormatError(f"{value} bytes per element is not a power of two")
    return value.bit_length() - 1


def _ceil_log2(value: int) -> int:
    return (value - 1).bit_length()


def tiled_element(x: int, y: int, pitch: int, bytes_per_element: int) -> int:
    """T2: the element number of element (x, y) of a tiled surface ``pitch`` elements wide
    (``pitch`` a multiple of 32)."""
    b = _log2(bytes_per_element)
    macro = ((x >> 5) + (y >> 5) * (pitch >> 5)) << (b + 7)
    micro = ((x & 7) + ((y & 6) << 2)) << b
    t = macro + ((micro & ~15) << 1) + (micro & 15) + ((y & 8) << (3 + b)) + ((y & 1) << 4)
    u = (((t & ~511) << 3) + ((t & 448) << 2) + (t & 63) + ((y & 16) << 7)
         + (((((y & 8) >> 2) + (x >> 3)) & 3) << 6))
    return u >> b


def volume_element(x: int, y: int, z: int) -> int:
    """T6: the element number of texel (x, y, z) of a 64 x 64 x 64 volume."""
    bits = {f"{axis}{i}": (value >> i) & 1 for axis, value in (("x", x), ("y", y), ("z", z))
            for i in range(6)}
    element = 0
    for position, sources in enumerate(VOLUME_BITS):
        bit = 0
        for source in sources:
            bit ^= bits[source]
        element |= bit << position
    return element


@dataclass(frozen=True)
class Surface:
    """Where one level of one face lies in the payload (T3, T4, T5)."""

    level: int
    face: int
    width: int  # texels
    height: int
    offset: int  # payload offset of the surface (own or tail)
    pitch: int  # elements, a multiple of 32
    origin: tuple[int, int]  # element offset inside the surface (tail slot; (0, 0) otherwise)
    packed: bool
    packed_index: int | None


def _elements(kind: str, width: int, height: int) -> tuple[int, int]:
    if kind.startswith("dxt"):
        return (width + 3) // 4, (height + 3) // 4
    return width, height


def surface_bytes(kind: str, bytes_per_element: int, width: int, height: int) -> int:
    """T3: the size of a surface of its own for a level of ``width`` x ``height`` texels."""
    we, he = _elements(kind, width, height)
    return _round(_round(we, TILE) * _round(he, TILE) * bytes_per_element, SURFACE_ALIGN)


def tail_offset(packed_index: int, wide: bool, longer: int = 16) -> tuple[int, int]:
    """T4 with V9-T4: the texel offset of packed level ``packed_index`` inside the tail.

    Levels 0 to 2 follow ``x360_tex_tail_slots.tsv``. From level 3 on the offset is
    ``longer >> (p - 2)``, where ``longer`` is the longer side of the first packed level rounded
    up to a power of two (``tail_longer_side``); the table holds the case ``longer`` = 16."""
    p = packed_index
    if p < 3:
        return (0, 16 >> p) if wide else (16 >> p, 0)
    step = longer >> (p - 2)
    return (step, 0) if wide else (0, step)


def tail_longer_side(first_w: int, first_h: int) -> int:
    """V9-T4: ``L``, the longer side of the first packed level rounded up to a power of two."""
    return 1 << _ceil_log2(max(first_w, first_h, 1))


def _format(tex_format: int) -> tuple[str, int]:
    if tex_format not in FORMATS:
        raise FormatError(f"unknown texture format {tex_format}", 0x20)
    return FORMATS[tex_format]


def first_packed_level(width: int, height: int, levels: int) -> int | None:
    """T3: the first level stored in the tail, or None (one level is never packed)."""
    if levels <= 1:
        return None
    for level in range(levels):
        if min(max(width >> level, 1), max(height >> level, 1)) <= PACK_LIMIT:
            return level
    return None


def layout(tex_format: int, width: int, height: int, levels: int, faces: int = 1,
           depth: int = 1) -> tuple[list[Surface], int]:
    """Every (level, face) surface and the payload size the rules give."""
    kind, bpe = _format(tex_format)
    if depth != 1:
        if (kind, width, height, depth, levels, faces) != ("argb8888", 64, 64, 64, 1, 1):
            raise FormatError(f"x360 volume {width}x{height}x{depth}, format {tex_format}, "
                              f"{levels} levels: only 64 x 64 x 64 ARGB8888 with one level is known")
        return [Surface(0, 0, 64, 64, 0, 0, (0, 0), False, None)], VOLUME_BYTES
    packed_from = first_packed_level(width, height, levels)
    found: list[Surface] = []
    offset = 0
    own = levels if packed_from is None else packed_from
    for level in range(own):
        w, h = max(width >> level, 1), max(height >> level, 1)
        size = surface_bytes(kind, bpe, w, h)
        pitch = _round(_elements(kind, w, h)[0], TILE)
        for face in range(faces):
            found.append(Surface(level, face, w, h, offset, pitch, (0, 0), False, None))
            offset += size
    if packed_from is not None:
        first_w, first_h = max(width >> packed_from, 1), max(height >> packed_from, 1)
        size = surface_bytes(kind, bpe, first_w, first_h)
        pitch = _round(_elements(kind, first_w, first_h)[0], TILE)
        wide = _ceil_log2(first_w) > _ceil_log2(first_h)
        longer = tail_longer_side(first_w, first_h)
        divisor = 4 if kind.startswith("dxt") else 1
        for face in range(faces):
            for level in range(packed_from, levels):
                p = level - packed_from
                ox, oy = tail_offset(p, wide, longer)
                found.append(Surface(
                    level, face, max(width >> level, 1), max(height >> level, 1), offset, pitch,
                    (ox // divisor, oy // divisor), True, p,
                ))
            offset += size
    return found, offset


def tex_layout(tex: TextureFile) -> tuple[list[Surface], int]:
    return layout(tex.format, tex.width, tex.height, tex.levels, tex.faces, tex.depth)


def payload_problems(tex: TextureFile) -> list[str]:
    """T3 as a check: the stored payload size against the rule."""
    _, size = tex_layout(tex)
    if len(tex.payload) != size:
        return [f"payload of {len(tex.payload)} bytes, the x360 layout gives {size}"]
    return []


def _surface(tex: TextureFile, level: int, face: int) -> Surface:
    surfaces, size = tex_layout(tex)
    if len(tex.payload) < size:
        raise FormatError(f"payload of {len(tex.payload)} bytes, the x360 layout needs {size}",
                          tex.payload_pair.file_offset)
    for surface in surfaces:
        if surface.level == level and surface.face == face:
            return surface
    raise FormatError(f"no level {level}, face {face}", 0x38)


def element_offsets(tex: TextureFile, level: int = 0, face: int = 0, z: int = 0) -> list[int]:
    """Payload offsets of the elements of one surface (or volume slice), in row order."""
    if tex.build != "x360":
        raise FormatError(f"not an x360 texture ({tex.build})")
    kind, bpe = _format(tex.format)
    if not 0 <= level < tex.levels or not 0 <= face < tex.faces:
        raise FormatError(f"no level {level}, face {face}", 0x38)
    if tex.depth != 1:
        tex_layout(tex)
        if not 0 <= z < VOLUME_SIDE:
            raise FormatError(f"no slice {z}", 0x30)
        if len(tex.payload) < VOLUME_BYTES:
            raise FormatError("volume payload is short", tex.payload_pair.file_offset)
        return [volume_element(x, y, z) * 4 for y in range(VOLUME_SIDE) for x in range(VOLUME_SIDE)]
    surface = _surface(tex, level, face)
    we, he = _elements(kind, surface.width, surface.height)
    ox, oy = surface.origin
    return [surface.offset + tiled_element(x + ox, y + oy, surface.pitch, bpe) * bpe
            for y in range(he) for x in range(we)]


def swap16(raw: bytes) -> bytes:
    """Swap the two bytes of every 16-bit word."""
    out = bytearray(raw)
    out[0::2], out[1::2] = raw[1::2], raw[0::2]
    return bytes(out)


def read_elements(tex: TextureFile, level: int = 0, face: int = 0, z: int = 0) -> bytes:
    """The elements of a surface in row order, as stored (no swap)."""
    _, bpe = _format(tex.format)
    payload = tex.payload
    return b"".join(payload[at:at + bpe] for at in element_offsets(tex, level, face, z))


def standard_elements(tex: TextureFile, level: int = 0, face: int = 0, z: int = 0) -> bytes:
    """The elements in row order in the standard byte order: DXT blocks little-endian (swapped),
    ARGB1555 as little-endian words (swapped), other formats as stored."""
    kind, _ = _format(tex.format)
    raw = read_elements(tex, level, face, z)
    return swap16(raw) if kind in SWAPPED_KINDS else raw


def decode_image(tex: TextureFile, level: int = 0, face: int = 0, z: int = 0) -> Image:
    """Decode one level of one face (or one volume slice) of an x360 texture to RGBA."""
    kind, _ = _format(tex.format)
    width, height = (VOLUME_SIDE, VOLUME_SIDE) if tex.depth != 1 else tex.level_size(level)
    raw = read_elements(tex, level, face, z)
    if kind.startswith("dxt"):
        return Image(width, height, _decode_elements(kind, "switch", swap16(raw), width, height))
    # ARGB1555: the stored word is big-endian (ps3 rule); ARGB8888 A, R, G, B; L8, A8 bytes.
    return Image(width, height, _decode_elements(kind, "ps3", raw, width, height))


def decode_levels(tex: TextureFile, face: int = 0, z: int = 0) -> list[Image]:
    return [decode_image(tex, level, face, z) for level in range(tex.levels)]


def build_payload(tex_format: int, width: int, height: int, levels: int,
                  elements: dict[tuple[int, int], bytes], faces: int = 1, depth: int = 1) -> bytes:
    """T8 in reverse: place standard-order elements (row order; DXT blocks little-endian) of each
    (level, face) into a payload, then swap. Padding is zero. A volume takes (0, 0) with its
    texels in x, y, z order."""
    kind, bpe = _format(tex_format)
    surfaces, size = layout(tex_format, width, height, levels, faces, depth)
    out = bytearray(size)
    if depth != 1:
        raw = elements[(0, 0)]
        for z in range(VOLUME_SIDE):
            for y in range(VOLUME_SIDE):
                for x in range(VOLUME_SIDE):
                    at = ((z * VOLUME_SIDE + y) * VOLUME_SIDE + x) * 4
                    target = volume_element(x, y, z) * 4
                    out[target:target + 4] = raw[at:at + 4]
        return bytes(out)
    for surface in surfaces:
        raw = elements[(surface.level, surface.face)]
        if kind in SWAPPED_KINDS:
            raw = swap16(raw)
        we, he = _elements(kind, surface.width, surface.height)
        if len(raw) != we * he * bpe:
            raise FormatError(f"level {surface.level} face {surface.face}: {len(raw)} bytes for "
                              f"{we} x {he} elements of {bpe}")
        ox, oy = surface.origin
        for y in range(he):
            for x in range(we):
                target = surface.offset + tiled_element(x + ox, y + oy, surface.pitch, bpe) * bpe
                if target + bpe > len(out):
                    raise FormatError("tiled address past the surface")
                at = (y * we + x) * bpe
                out[target:target + bpe] = raw[at:at + bpe]
    return bytes(out)


def block_difference(a: Image, b: Image, block: int = 4) -> float:
    """T7: the mean absolute difference of block-averaged RGBA, for comparing with Switch."""
    if (a.width, a.height) != (b.width, b.height):
        raise FormatError("images of different sizes")
    total, count = 0.0, 0
    for by in range(0, a.height, block):
        for bx in range(0, a.width, block):
            sums = [[0, 0, 0, 0], [0, 0, 0, 0]]
            n = 0
            for y in range(by, min(by + block, a.height)):
                for x in range(bx, min(bx + block, a.width)):
                    at = (y * a.width + x) * 4
                    for c in range(4):
                        sums[0][c] += a.rgba[at + c]
                        sums[1][c] += b.rgba[at + c]
                    n += 1
            total += sum(abs(sums[0][c] - sums[1][c]) / n for c in range(4)) / 4
            count += 1
    return total / count if count else 0.0


__all__ = [
    "Surface", "block_difference", "build_payload", "decode_image", "decode_levels",
    "element_offsets", "first_packed_level", "layout", "payload_problems", "read_elements",
    "standard_elements", "surface_bytes", "swap16", "tail_longer_side", "tail_offset", "tex_layout", "tiled_element",
    "volume_element",
]
