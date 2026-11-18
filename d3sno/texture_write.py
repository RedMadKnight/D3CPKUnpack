"""Stage 14 texture encoder (``28_derived_values.md`` W14-7 to W14-11).

``encode_texture`` turns RGBA images into a complete ``.tex`` file for ps3, switch or x360:

- W14-7: header values derived from the images (size, faces, depth, cube flag, volume slices,
  image ids, frame block, word 0x34, mip range); the other fields come from ``template`` or zero.
- W14-8: the mip chain rule; a level count above the rule is refused.
- W14-9: ``ps3_payload``, W14-10: ``switch_payload`` (the container), W14-11: ``x360_payload``
  (``x360_texture.build_payload``).

Payload builders take *elements*: a dict ``{(level, face): bytes}`` with each surface's elements in
row order, in the build's element form (``element_form``); a volume takes ``(0, 0)`` with its
texels in x, then y, then z order. ``texture_elements`` reads them back from a parsed file, so a
payload can be rebuilt from a parsed texture and compared with the stored one.

DXT blocks are made by a small deterministic block encoder (bounding-box endpoints, nearest
palette entry); uncompressed formats encode exactly (ARGB1555 exactly for values its 5-bit
expansion can give).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

from . import texture as T
from . import x360_texture as XT
from .binary import FormatError, build_order
from .sno import expected_version
from .texture import FORMATS, Image, TextureFile, TextureFrame

BUILDS = ("ps3", "switch", "x360")
DXT_KINDS = ("dxt1", "dxt3", "dxt5")
VOLUME_SIDE = 64
VOLUME_SLICES = (8, 8)
FLAT_SLICES = (1, 1)
PS3_FACE_ALIGN = 128
FIXED_SIZE = {"ps3": T.PS3_FIXED_SIZE, "switch": 0x288, "x360": 0x288}
_FIELDS = {  # image ids pair, cube flag, import flags, machine time, console mip offset, V scale, volume
    "ps3": (0x24C, 0x254, 0x258, 0x25C, 0x260, 0x264, 0x268),
    "switch": (0x260, 0x268, 0x26C, 0x270, 0x274, 0x278, 0x27C),
}
_FIELDS["x360"] = _FIELDS["switch"]
WORD_0x34 = 0x34

# W14-10 container constants.
NV_ALIGNMENT = 512
NV_IMAGE_AT = 512
NV_INFO_AT = 52
NV_TAIL_SIZE = 60
NV_INFO_TAIL = bytes((0x07, 0x00, 0x01, 0x00)) + bytes(4)
NV_3D_LAYOUT = 0x40


# --- W14-8: mip chain -----------------------------------------------------------------------

def _is_pow2(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def max_levels(tex_format: int, width: int, height: int, depth: int = 1) -> int:
    """W14-8: the largest level count the rule allows. Sides that are not powers of two, and
    volumes, allow one level. DXT ends at the level whose smaller side is 4, other formats at 1x1."""
    if depth != 1 or not (_is_pow2(width) and _is_pow2(height)):
        return 1
    kind = _kind(tex_format)
    if kind in DXT_KINDS:
        count = min(width, height).bit_length() - 1 - 1
    else:
        count = max(width, height).bit_length() - 1 + 1
    return max(1, min(count, T.MAX_MIPS))


def level_sizes(width: int, height: int, levels: int) -> list[tuple[int, int]]:
    """W14-8: level m is max(W >> m, 1) by max(H >> m, 1)."""
    return [(max(width >> m, 1), max(height >> m, 1)) for m in range(levels)]


def check_levels(tex_format: int, width: int, height: int, levels: int, depth: int = 1) -> None:
    limit = max_levels(tex_format, width, height, depth)
    if not 1 <= levels <= limit:
        raise FormatError(
            f"{levels} levels for a {width}x{height}x{depth} texture of format {tex_format}: "
            f"the mip chain rule allows 1 to {limit}"
        )


# --- Element forms --------------------------------------------------------------------------

def _kind(tex_format: int) -> str:
    if tex_format not in FORMATS:
        raise FormatError(f"unknown texture format {tex_format}", 0x20)
    return FORMATS[tex_format][0]


def _grid(kind: str, width: int, height: int) -> tuple[int, int]:
    if kind in DXT_KINDS:
        return (width + 3) // 4, (height + 3) // 4
    return width, height


def _to5(value: int) -> int:
    return (value * 31 + 127) // 255


def _to565(r: int, g: int, b: int) -> int:
    return _to5(r) << 11 | (g * 63 + 127) // 255 << 5 | _to5(b)


def _dist(a, b) -> int:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2


def _block_pixels(rgba: bytes, width: int, height: int, bx: int, by: int) -> list[tuple]:
    """The 16 pixels of a block; positions past the image repeat the last row or column."""
    pixels = []
    for py in range(4):
        y = min(by * 4 + py, height - 1)
        for px in range(4):
            x = min(bx * 4 + px, width - 1)
            at = (y * width + x) * 4
            pixels.append(tuple(rgba[at:at + 4]))
    return pixels


def _colour_block(pixels: list[tuple], punch_through: bool) -> bytes:
    """8 bytes: c0, c1, 2-bit indices (little-endian). With ``punch_through`` (DXT1), texels with
    alpha below 128 take index 3 of the three-colour mode."""
    opaque = [p for p in pixels if not punch_through or p[3] >= 128]
    transparent = punch_through and len(opaque) < 16
    if not opaque:
        return struct.pack("<HHI", 0, 0, 0xFFFFFFFF)
    # Endpoints: the two texels farthest apart (first such pair in texel order).
    colours = sorted({p[:3] for p in opaque})
    lo = hi = colours[0]
    best = -1
    for i, a in enumerate(colours):
        for b in colours[i:]:
            d = _dist(a, b)
            if d > best:
                best, lo, hi = d, a, b
    c_hi, c_lo = _to565(*hi), _to565(*lo)
    if c_hi < c_lo:
        c_hi, c_lo = c_lo, c_hi
    if transparent:
        c0, c1 = c_lo, c_hi  # c0 <= c1: three-colour mode
    else:
        c0, c1 = c_hi, c_lo
    three = punch_through and c0 <= c1
    palette = T._colour_palette(c0, c1, three)
    choices = range(3) if three else range(4)
    bits = 0
    for i, p in enumerate(pixels):
        if transparent and p[3] < 128:
            index = 3
        else:
            index = min(choices, key=lambda k: (_dist(p, palette[k]), k))
        bits |= index << (2 * i)
    return struct.pack("<HHI", c0, c1, bits)


def _alpha_block5(pixels: list[tuple]) -> bytes:
    alphas = [p[3] for p in pixels]
    a0, a1 = max(alphas), min(alphas)
    palette = T._alpha_palette(a0, a1) if a0 != a1 else [a0] * 8
    codes = 0
    for i, a in enumerate(alphas):
        index = 0 if a0 == a1 else min(range(8), key=lambda k: (abs(a - palette[k]), k))
        codes |= index << (3 * i)
    return bytes((a0, a1)) + codes.to_bytes(6, "little")


def _alpha_block3(pixels: list[tuple]) -> bytes:
    value = 0
    for i, p in enumerate(pixels):
        value |= ((p[3] * 15 + 127) // 255) << (4 * i)
    return value.to_bytes(8, "little")


def compress_dxt(image: Image, kind: str) -> bytes:
    """DXT blocks in row order, standard little-endian form."""
    wb, hb = _grid(kind, image.width, image.height)
    out = bytearray()
    for by in range(hb):
        for bx in range(wb):
            pixels = _block_pixels(image.rgba, image.width, image.height, bx, by)
            if kind == "dxt1":
                out += _colour_block(pixels, True)
            elif kind == "dxt3":
                out += _alpha_block3(pixels) + _colour_block(pixels, False)
            else:
                out += _alpha_block5(pixels) + _colour_block(pixels, False)
    return bytes(out)


def element_form(image: Image, tex_format: int, build: str) -> bytes:
    """Encode one surface to the elements a payload builder takes, in row order.

    ps3: DXT standard, ARGB8888 A R G B, ARGB1555 big-endian A1 R5 G5 B5, L8, A8.
    switch: DXT standard, ARGB8888 R G B A, ARGB1555 little-endian A1 B5 G5 R5, L8, A8.
    x360 (standard order, before ``build_payload``'s swap): DXT standard, ARGB8888 A R G B,
    ARGB1555 little-endian A1 R5 G5 B5, L8, A8.
    L8 takes the red channel, A8 the alpha channel.
    """
    kind = _kind(tex_format)
    if build not in BUILDS:
        raise FormatError(f"unknown build {build!r}")
    rgba = image.rgba
    count = image.width * image.height
    if len(rgba) != count * 4:
        raise FormatError(f"{len(rgba)} bytes of RGBA for {image.width}x{image.height}")
    if kind in DXT_KINDS:
        return compress_dxt(image, kind)
    if kind == "argb8888":
        out = bytearray(count * 4)
        if build == "switch":
            return bytes(rgba)
        out[0::4], out[1::4], out[2::4], out[3::4] = rgba[3::4], rgba[0::4], rgba[1::4], rgba[2::4]
        return bytes(out)
    if kind == "argb1555":
        words = []
        for i in range(count):
            r, g, b, a = rgba[i * 4:i * 4 + 4]
            first, last = (_to5(b), _to5(r)) if build == "switch" else (_to5(r), _to5(b))
            words.append((a >= 128) << 15 | first << 10 | _to5(g) << 5 | last)
        return struct.pack(f"{'>' if build == 'ps3' else '<'}{count}H", *words)
    if kind == "l8":
        return bytes(rgba[0::4])
    return bytes(rgba[3::4])  # a8


# --- Mip generation ------------------------------------------------------------------------

def downsample(image: Image) -> Image:
    """The next level: each texel the rounded mean of its 2 x 2 (or 2 x 1) source texels."""
    width, height = max(image.width >> 1, 1), max(image.height >> 1, 1)
    sx = 2 if image.width > 1 else 1
    sy = 2 if image.height > 1 else 1
    src = image.rgba
    out = bytearray(width * height * 4)
    n = sx * sy
    for y in range(height):
        for x in range(width):
            for c in range(4):
                total = 0
                for dy in range(sy):
                    for dx in range(sx):
                        total += src[((y * sy + dy) * image.width + x * sx + dx) * 4 + c]
                out[(y * width + x) * 4 + c] = (total + n // 2) // n
    return Image(width, height, bytes(out))


# --- Payload builders ----------------------------------------------------------------------

def _level_bytes_ps3(kind: str, bpp: int, width: int, height: int, depth: int) -> int:
    wb, hb = _grid(kind, width, height)
    return wb * hb * bpp * depth


def _check_elements(raw: bytes, expected: int, level: int, face: int) -> None:
    if len(raw) != expected:
        raise FormatError(f"level {level} face {face}: {len(raw)} bytes of elements, {expected} needed")


def ps3_payload(tex_format: int, width: int, height: int, levels: int,
                elements: dict[tuple[int, int], bytes], faces: int = 1, depth: int = 1) -> bytes:
    """W14-9 as corrected by V16-1: per face, the levels tightly packed. Only a cube (6 faces)
    pads each face, the last included, with zeros to 128 bytes; a single-face texture (2D or
    volume) is the bare chain. A volume (ARGB8888) is placed in 3D Morton order into a zero
    buffer."""
    kind, bpp = FORMATS[tex_format] if tex_format in FORMATS else (_kind(tex_format), 0)
    chain = sum(
        _level_bytes_ps3(kind, bpp, w, h, max(depth >> m, 1))
        for m, (w, h) in enumerate(level_sizes(width, height, levels))
    )
    stride = -(-chain // PS3_FACE_ALIGN) * PS3_FACE_ALIGN if faces == 6 else chain
    out = bytearray(faces * stride)
    for face in range(faces):
        at = face * stride
        for level, (w, h) in enumerate(level_sizes(width, height, levels)):
            d = max(depth >> level, 1)
            size = _level_bytes_ps3(kind, bpp, w, h, d)
            raw = elements[(level, face)]
            _check_elements(raw, size, level, face)
            if depth == 1:
                out[at:at + size] = raw
            else:
                if kind != "argb8888":
                    raise FormatError(f"ps3 volume in format {tex_format}")
                xs, ys, zs = T._morton_tables(w, h, d)
                i = 0
                for z in range(d):
                    for y in range(h):
                        row = ys[y] | zs[z]
                        for x in range(w):
                            target = at + (xs[x] | row) * 4
                            out[target:target + 4] = raw[i:i + 4]
                            i += 4
            at += size
    return bytes(out)


@dataclass
class SwitchLayout:
    """W14-10: what the container's information block holds."""

    level_sizes: list[int]
    mip_offsets: list[int]
    block_heights: list[int]
    chain: int  # S
    image_size: int  # D
    layout_word: int
    target: int


def switch_layout(tex_format: int, width: int, height: int, levels: int, faces: int = 1,
                  depth: int = 1) -> SwitchLayout:
    kind, bpp = FORMATS[tex_format] if tex_format in FORMATS else (_kind(tex_format), 0)
    if depth != 1:
        if (width, height, depth, levels, faces) != (VOLUME_SIDE,) * 3 + (1, 1):
            raise FormatError("a switch volume is 64 x 64 x 64 with one level")
        size = VOLUME_SIDE ** 3 * bpp
        return SwitchLayout([size], [0], [1], size, size, NV_3D_LAYOUT, T.NV_TARGET_3D)
    hb0 = _grid(kind, width, height)[1]
    bh = T.standard_block_height(hb0)
    layout_word = bh.bit_length() - 1
    sizes, offsets, heights = [], [], []
    running = 0
    for w, h in level_sizes(width, height, levels):
        wb, hb = _grid(kind, w, h)
        while bh > 1 and hb <= (bh // 2) * 8:
            bh //= 2
        size = T._surface_bytes(wb, hb, bpp, bh)
        offsets.append(running)
        sizes.append(size)
        heights.append(bh)
        running += size
    target = T.NV_TARGET_CUBE if faces == T.CUBE_FACES else T.NV_TARGET_2D
    return SwitchLayout(sizes, offsets, heights, running, faces * running, layout_word, target)


def _block_header(data_size: int, data_offset: int, kind: int, index: int) -> bytes:
    return T.NV_BLOCK_MAGIC + struct.pack("<IQQIII", 36, data_size, data_offset, kind, index, 0)


def switch_container(tex_format: int, width: int, height: int, levels: int, image: bytes,
                     faces: int = 1, depth: int = 1) -> bytes:
    """W14-10 bytes around an image data block of D bytes."""
    lay = switch_layout(tex_format, width, height, levels, faces, depth)
    if len(image) != lay.image_size:
        raise FormatError(f"image data of {len(image)} bytes, the layout gives {lay.image_size}")
    offsets = lay.mip_offsets + [0] * (T.MAX_MIPS - len(lay.mip_offsets))
    info = struct.pack(
        "<Q8I", lay.image_size, NV_ALIGNMENT, width, height, depth, lay.target,
        T._NV_FORMATS[tex_format], levels, lay.chain,
    ) + struct.pack(f"<{T.MAX_MIPS}II", *offsets, lay.layout_word) + NV_INFO_TAIL
    out = bytearray(T.NV_FILE_MAGIC + struct.pack("<III", 16, 1, 1))
    out += _block_header(len(info), 36, 2, 0) + info
    out += _block_header(lay.image_size, NV_IMAGE_AT - len(out), 3, 1)
    out += bytes(NV_IMAGE_AT - len(out))
    out += image
    out += _block_header(24, 36, 5, 2) + bytes(8) + struct.pack("<II", 1, 1) + bytes(8)
    return bytes(out)


def switch_image(tex_format: int, width: int, height: int, levels: int,
                 elements: dict[tuple[int, int], bytes], faces: int = 1, depth: int = 1) -> bytes:
    """W14-10 image data: every element at its block-linear address, zeros elsewhere."""
    kind, bpp = FORMATS[tex_format]
    lay = switch_layout(tex_format, width, height, levels, faces, depth)
    out = bytearray(lay.image_size)
    if depth != 1:
        raw = elements[(0, 0)]
        _check_elements(raw, lay.image_size, 0, 0)
        gob_w, x_part, y_part = T._gob_tables(width, height, bpp, 1)
        rows = (height + 7) // 8
        i = 0
        for z in range(depth):
            for row, _, y_in in y_part:
                for gob_x, x_in in x_part:
                    gob = (z // 16 * rows + row) * gob_w + gob_x
                    at = (gob * 16 + z % 16) * 512 + y_in + x_in
                    out[at:at + bpp] = raw[i:i + bpp]
                    i += bpp
        return bytes(out)
    for face in range(faces):
        for level, (w, h) in enumerate(level_sizes(width, height, levels)):
            wb, hb = _grid(kind, w, h)
            bh = lay.block_heights[level]
            raw = elements[(level, face)]
            _check_elements(raw, wb * hb * bpp, level, face)
            gob_w, x_part, y_part = T._gob_tables(wb, hb, bpp, bh)
            base = face * lay.chain + lay.mip_offsets[level]
            i = 0
            for row, gob_y, y_in in y_part:
                for gob_x, x_in in x_part:
                    at = base + ((row * gob_w + gob_x) * bh + gob_y) * 512 + y_in + x_in
                    out[at:at + bpp] = raw[i:i + bpp]
                    i += bpp
    return bytes(out)


def switch_payload(tex_format: int, width: int, height: int, levels: int,
                   elements: dict[tuple[int, int], bytes], faces: int = 1, depth: int = 1) -> bytes:
    """W14-10: the whole switch payload (container and image data)."""
    image = switch_image(tex_format, width, height, levels, elements, faces, depth)
    return switch_container(tex_format, width, height, levels, image, faces, depth)


def x360_payload(tex_format: int, width: int, height: int, levels: int,
                 elements: dict[tuple[int, int], bytes], faces: int = 1, depth: int = 1) -> bytes:
    """W14-11: exactly ``x360_texture.build_payload``."""
    return XT.build_payload(tex_format, width, height, levels, elements, faces, depth)


PAYLOAD_BUILDERS = {"ps3": ps3_payload, "switch": switch_payload, "x360": x360_payload}


def build_payload(build: str, tex_format: int, width: int, height: int, levels: int,
                  elements: dict[tuple[int, int], bytes], faces: int = 1, depth: int = 1) -> bytes:
    if build not in PAYLOAD_BUILDERS:
        raise FormatError(f"unknown build {build!r}")
    return PAYLOAD_BUILDERS[build](tex_format, width, height, levels, elements, faces, depth)


def texture_elements(tex: TextureFile) -> dict[tuple[int, int], bytes]:
    """Every surface's elements of a parsed texture, in the form the payload builders take."""
    _, bpp = FORMATS[tex.format]
    out = {}
    if tex.depth != 1:
        parts = []
        for z in range(tex.depth):
            if tex.build == "x360":
                parts.append(XT.read_elements(tex, 0, 0, z))
            else:
                parts.append(T._read_elements(tex, T.element_offsets(tex, 0, 0, z), bpp))
        out[(0, 0)] = b"".join(parts)
        return out
    for face in range(tex.faces):
        for level in range(tex.levels):
            if tex.build == "x360":
                out[(level, face)] = XT.standard_elements(tex, level, face)
            else:
                out[(level, face)] = T._read_elements(tex, T.element_offsets(tex, level, face), bpp)
    return out


def rebuild_payload(tex: TextureFile) -> bytes:
    """The payload the rules give for a parsed texture's own elements (W14-9 to W14-11)."""
    return build_payload(tex.build, tex.format, tex.width, tex.height, tex.levels,
                         texture_elements(tex), tex.faces, tex.depth)


# --- W14-7: header and file ----------------------------------------------------------------

def _frame(value) -> TextureFrame:
    if isinstance(value, TextureFrame):
        return value
    return TextureFrame(int(value[0]), *(float(v) for v in value[1:5]))


def derived_image_ids(frame_count: int, faces: int) -> list[int]:
    """W14-7: 0 .. n-1 for n frames; six zeros for a cube."""
    if faces == T.CUBE_FACES:
        return [0] * T.CUBE_FACES
    return list(range(frame_count))


def assemble_texture(build: str, tex_format: int, width: int, height: int, payload: bytes,
                     levels: int, faces: int = 1, depth: int = 1, frames=None,
                     template: TextureFile | None = None) -> bytes:
    """A whole ``.tex`` around a finished payload: W14-7 header values, then the layout of the
    ``.tex`` writer (ps3 tight from 0x270; switch and x360 image ids and frames 8-aligned).

    Kept from ``template`` (else zero / the build's version): SNO header words, the word at
    0x1C, mip min, import flags, machine time, console mip offset, light-map V scale, other fixed
    bytes (same build only), frames when ``frames`` is None, image ids when the frame count and
    faces are the template's, and volume slices of a 2D template for a 2D texture."""
    from .write import write_tex  # the stage 4 layout rule

    if build not in BUILDS:
        raise FormatError(f"unknown build {build!r}")
    if faces not in (1, T.CUBE_FACES) or depth not in (1, VOLUME_SIDE):
        raise FormatError(f"faces {faces}, depth {depth}")
    order = build_order(build)
    same = template is not None and template.build == build
    size = FIXED_SIZE[build]
    fixed = bytearray(size)
    if same:
        kept = template.fixed[:size]
        fixed[:len(kept)] = kept
    # SNO header (W14-1 handle at 0x10; runtime words zero).
    if template is not None:
        head = template.header
        version = head.version if same else expected_version("tex", build)
        words = (0xDEADBEEF, version, head.revision, head.source_hash, head.handle, 0, 0)
    else:
        words = (0xDEADBEEF, expected_version("tex", build), 0, 0, 0, 0, 0)
    struct.pack_into(order + "7I", fixed, 0, *words)
    # Pairs are laid out again by the writer; the 60 payload pairs are zero except the first.
    fixed[T.PAYLOAD_AT:0x220] = bytes(0x220 - T.PAYLOAD_AT)
    fixed[T.FRAME_PAIR_AT:T.FRAME_PAIR_AT + 8] = bytes(8)
    ids_at, cube_at, import_at, time_at, console_at, scale_at, volume_at = _FIELDS[build]
    fixed[ids_at:ids_at + 8] = bytes(8)

    mip_min = template.mip_min if template is not None else 0
    struct.pack_into(order + "5I", fixed, 0x20, tex_format, width, height, faces, depth)
    struct.pack_into(order + "I", fixed, WORD_0x34, 0)
    struct.pack_into(order + "ii", fixed, 0x38, mip_min, mip_min + levels - 1)

    if frames is None:
        frame_list = list(template.frames) if template is not None else [TextureFrame(0, 0.0, 0.0, 1.0, 1.0)]
    else:
        frame_list = [_frame(f) for f in frames]
    if faces == T.CUBE_FACES and len(frame_list) != 1:
        raise FormatError(f"a cube has one frame, not {len(frame_list)}")
    struct.pack_into(order + "i", fixed, 0x220, len(frame_list))
    frames_raw = b"".join(struct.pack(order + "I4f", f.hash, f.u0, f.v0, f.u1, f.v1) for f in frame_list)

    if (template is not None and template.faces == faces
            and len(template.frames) == len(frame_list)):
        ids_raw = template.image_ids_raw if same else struct.pack(
            f"{order}{len(template.image_ids)}i", *template.image_ids)
    else:
        ids = derived_image_ids(len(frame_list), faces)
        ids_raw = struct.pack(f"{order}{len(ids)}i", *ids)

    if depth == VOLUME_SIDE:
        slices = VOLUME_SLICES
    elif template is not None and template.depth == 1:
        slices = (template.volume_x, template.volume_y)
    else:
        slices = FLAT_SLICES
    values = (
        (cube_at, "I", 1 if faces == T.CUBE_FACES else 0),
        (import_at, "I", template.import_flags if template is not None else 0),
        (time_at, "I", template.machine_time if template is not None else 0),
        (console_at, "I", template.console_mip_offset if template is not None else 0),
        (scale_at, "f", template.lightmap_v_scale if template is not None else 0.0),
        (volume_at, "II", *slices),
    )
    for at, fmt, *vals in values:
        struct.pack_into(order + fmt, fixed, at, *vals)

    shell = TextureFile(
        header=None, build=build, format=tex_format, width=width, height=height, faces=faces,
        depth=depth, mip_min=mip_min, mip_max=mip_min + levels - 1, payload_pair=None,
        payload=payload, frames=frame_list, image_ids=[], cube_flag=0, import_flags=0,
        machine_time=0, console_mip_offset=0, lightmap_v_scale=0.0, volume_x=0, volume_y=0,
        fixed=bytes(fixed), image_ids_raw=ids_raw, frames_raw=frames_raw,
    )
    return write_tex(shell)


def _surfaces(images, faces: int, depth: int) -> list:
    if isinstance(images, Image):
        images = [images]
    images = list(images)
    wanted = depth if depth != 1 else faces
    if len(images) != wanted:
        raise FormatError(f"{len(images)} images for {faces} faces and depth {depth}: {wanted} needed")
    return images


def encode_texture(images, build: str, format: int, levels: int | None = None, faces: int = 1,
                   depth: int = 1, frames=None, template: TextureFile | None = None) -> bytes:
    """Encode RGBA images to a complete ``.tex`` (W14-7 to W14-11).

    ``images``: one ``Image`` (2D), six (a cube, faces +X, -X, +Y, -Y, +Z, -Z) or 64 slices of
    64 x 64 (a volume, ARGB8888, one level). A face may instead be a list of Images, its whole
    mip chain; otherwise lower levels are made with ``downsample``. ``levels`` defaults to the
    given chain's length, or to the full chain of W14-8; a count above W14-8 is refused."""
    if build not in BUILDS:
        raise FormatError(f"unknown build {build!r}")
    kind = _kind(format)
    surfaces = _surfaces(images, faces, depth)
    chains = [s if isinstance(s, (list, tuple)) else [s] for s in surfaces]
    top = chains[0][0]
    width, height = top.width, top.height
    for chain in chains:
        if (chain[0].width, chain[0].height) != (width, height):
            raise FormatError("every face or slice must have the size of the first")
    if depth != 1:
        if (kind, width, height, depth, faces) != ("argb8888", VOLUME_SIDE, VOLUME_SIDE, VOLUME_SIDE, 1):
            raise FormatError("a volume is 64 x 64 x 64 ARGB8888 with one face")
    if levels is None:
        given = len(chains[0])
        levels = given if given > 1 else max_levels(format, width, height, depth)
    check_levels(format, width, height, levels, depth)
    sizes = level_sizes(width, height, levels)
    elements: dict[tuple[int, int], bytes] = {}
    if depth != 1:
        elements[(0, 0)] = b"".join(element_form(chain[0], format, build) for chain in chains)
    else:
        for face, chain in enumerate(chains):
            chain = list(chain)
            while len(chain) < levels:
                chain.append(downsample(chain[-1]))
            for level in range(levels):
                image = chain[level]
                if (image.width, image.height) != sizes[level]:
                    raise FormatError(
                        f"face {face} level {level} is {image.width}x{image.height}, "
                        f"{sizes[level][0]}x{sizes[level][1]} needed")
                elements[(level, face)] = element_form(image, format, build)
    payload = build_payload(build, format, width, height, levels, elements, faces, depth)
    return assemble_texture(build, format, width, height, payload, levels, faces, depth,
                            frames, template)


__all__ = [
    "SwitchLayout", "assemble_texture", "build_payload", "check_levels", "compress_dxt",
    "derived_image_ids", "downsample", "element_form", "encode_texture", "level_sizes",
    "max_levels", "ps3_payload", "rebuild_payload", "switch_container", "switch_image",
    "switch_layout", "switch_payload", "texture_elements", "x360_payload",
]
