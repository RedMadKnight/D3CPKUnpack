"""PNG reader (``34_modding_tools.md`` W16-1), standard library only.

``read_png`` follows the public PNG specification: it checks the signature, every chunk CRC and
the chunk order, inflates the IDAT stream, undoes filters 0 to 4 and Adam7 interlacing, and
converts every colour type and bit depth to the ``texture.Image`` form (RGBA8, top row first).

- Grey (types 0, 4): (L, L, L, A). Palette (type 3): PLTE entries, alpha from tRNS else 255.
  tRNS on grey or RGB: the matching colour (compared at the stored depth) gets alpha 0.
- Depths below 8: ``v * 255 // (2**d - 1)``. 16-bit samples: ``(v * 255 + 32767) // 65535``.
- Ancillary chunks other than tRNS are ignored; no gamma is applied. An unknown critical chunk is
  refused.

Every failure raises ``FormatError`` with the byte offset of the problem.
"""

from __future__ import annotations

import struct
import zlib

from .binary import FormatError
from .texture import Image

SIGNATURE = b"\x89PNG\r\n\x1a\n"
_MAX_LENGTH = 0x7FFFFFFF
_KNOWN_CRITICAL = (b"IHDR", b"PLTE", b"IDAT", b"IEND")
# Colour type -> (allowed bit depths, samples per pixel).
COLOUR_TYPES = {
    0: ((1, 2, 4, 8, 16), 1),
    2: ((8, 16), 3),
    3: ((1, 2, 4, 8), 1),
    4: ((8, 16), 2),
    6: ((8, 16), 4),
}
# Adam7: x start, y start, x step, y step.
ADAM7 = ((0, 0, 8, 8), (4, 0, 8, 8), (0, 4, 4, 8), (2, 0, 4, 4), (0, 2, 2, 4), (1, 0, 2, 2),
         (0, 1, 1, 2))


def _chunks(data: bytes):
    """Yield (offset, type, body) for every chunk, checking lengths, names and CRCs."""
    pos = len(SIGNATURE)
    while pos < len(data):
        if pos + 8 > len(data):
            raise FormatError("truncated chunk header", pos)
        length, kind = struct.unpack_from(">I4s", data, pos)
        if length > _MAX_LENGTH:
            raise FormatError(f"chunk length {length} above 2^31 - 1", pos)
        if not all(65 <= c <= 90 or 97 <= c <= 122 for c in kind):
            raise FormatError(f"chunk type {kind!r} is not four letters", pos + 4)
        end = pos + 12 + length
        if end > len(data):
            raise FormatError(f"chunk {kind.decode('latin-1')} runs past the end of the file", pos)
        body = data[pos + 8:pos + 8 + length]
        (crc,) = struct.unpack_from(">I", data, pos + 8 + length)
        if zlib.crc32(kind + body) & 0xFFFFFFFF != crc:
            raise FormatError(f"chunk {kind.decode('latin-1')} has a bad CRC", pos + 8 + length)
        yield pos, kind, body
        pos = end


class _Header:
    def __init__(self, body: bytes, at: int):
        if len(body) != 13:
            raise FormatError(f"IHDR of {len(body)} bytes, 13 needed", at)
        (self.width, self.height, self.depth, self.colour, compression, filtering,
         self.interlace) = struct.unpack(">IIBBBBB", body)
        if not 1 <= self.width <= _MAX_LENGTH or not 1 <= self.height <= _MAX_LENGTH:
            raise FormatError(f"image size {self.width}x{self.height}", at + 8)
        if self.colour not in COLOUR_TYPES:
            raise FormatError(f"colour type {self.colour}", at + 17)
        depths, self.samples = COLOUR_TYPES[self.colour]
        if self.depth not in depths:
            raise FormatError(f"bit depth {self.depth} with colour type {self.colour}", at + 16)
        if compression != 0:
            raise FormatError(f"compression method {compression}", at + 18)
        if filtering != 0:
            raise FormatError(f"filter method {filtering}", at + 19)
        if self.interlace not in (0, 1):
            raise FormatError(f"interlace method {self.interlace}", at + 20)
        self.bits = self.depth * self.samples  # bits per pixel
        self.bpp = max(1, self.bits // 8)  # filter unit in bytes

    def row_bytes(self, width: int) -> int:
        return (width * self.bits + 7) // 8

    def passes(self):
        """(x0, y0, dx, dy, width, height) of each non-empty pass."""
        if self.interlace == 0:
            return [(0, 0, 1, 1, self.width, self.height)]
        out = []
        for x0, y0, dx, dy in ADAM7:
            w = (self.width - x0 + dx - 1) // dx if self.width > x0 else 0
            h = (self.height - y0 + dy - 1) // dy if self.height > y0 else 0
            if w and h:
                out.append((x0, y0, dx, dy, w, h))
        return out


def _paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    return b if pb <= pc else c


def _unfilter(raw: bytes, pos: int, width_bytes: int, height: int, bpp: int,
              at: int) -> tuple[list[bytes], int]:
    """Undo the filters of one pass. Returns the rows and the position after them."""
    rows = []
    prior = bytes(width_bytes)
    for _ in range(height):
        kind = raw[pos]
        line = bytearray(raw[pos + 1:pos + 1 + width_bytes])
        if kind == 0:
            pass
        elif kind == 1:
            for i in range(bpp, width_bytes):
                line[i] = (line[i] + line[i - bpp]) & 255
        elif kind == 2:
            for i in range(width_bytes):
                line[i] = (line[i] + prior[i]) & 255
        elif kind == 3:
            for i in range(width_bytes):
                left = line[i - bpp] if i >= bpp else 0
                line[i] = (line[i] + ((left + prior[i]) >> 1)) & 255
        elif kind == 4:
            for i in range(width_bytes):
                if i >= bpp:
                    line[i] = (line[i] + _paeth(line[i - bpp], prior[i], prior[i - bpp])) & 255
                else:
                    line[i] = (line[i] + prior[i]) & 255
        else:
            raise FormatError(f"filter type {kind} in the image data (inflated byte {pos})", at)
        prior = bytes(line)
        rows.append(prior)
        pos += 1 + width_bytes
    return rows, pos


def _samples(row: bytes, count: int, depth: int) -> list[int]:
    """``count`` samples of ``depth`` bits from a row."""
    if depth == 8:
        return list(row[:count])
    if depth == 16:
        return list(struct.unpack(f">{count}H", row[:count * 2]))
    per = 8 // depth
    mask = (1 << depth) - 1
    out = []
    for i in range(count):
        out.append(row[i // per] >> (8 - depth * (i % per + 1)) & mask)
    return out


def _scale(depth: int):
    if depth == 16:
        return lambda v: (v * 255 + 32767) // 65535
    if depth == 8:
        return lambda v: v
    top = (1 << depth) - 1
    return lambda v: v * 255 // top


def _pixels(head: _Header, rows: list[bytes], width: int, palette, trns, at: int) -> list[bytes]:
    """Convert one pass's rows to lists of RGBA8 pixels (4 bytes each)."""
    scale = _scale(head.depth)
    out = []
    n = head.samples
    for row in rows:
        values = _samples(row, width * n, head.depth)
        pixels = []
        for x in range(width):
            s = values[x * n:(x + 1) * n]
            if head.colour == 0:
                v = scale(s[0])
                alpha = 0 if trns is not None and s[0] == trns[0] else 255
                pixels.append(bytes((v, v, v, alpha)))
            elif head.colour == 2:
                alpha = 0 if trns is not None and tuple(s) == trns else 255
                pixels.append(bytes((scale(s[0]), scale(s[1]), scale(s[2]), alpha)))
            elif head.colour == 3:
                if s[0] >= len(palette):
                    raise FormatError(f"palette index {s[0]} past {len(palette)} entries", at)
                pixels.append(palette[s[0]])
            elif head.colour == 4:
                v = scale(s[0])
                pixels.append(bytes((v, v, v, scale(s[1]))))
            else:
                pixels.append(bytes(scale(v) for v in s))
        out.append(pixels)
    return out


def read_png(data: bytes) -> Image:
    """Read a PNG file into an RGBA8 ``Image`` (W16-1)."""
    data = bytes(data)
    if data[:8] != SIGNATURE:
        raise FormatError("not a PNG file: bad signature", 0)
    head = None
    plte = plte_at = None
    trns = trns_at = None
    idat = []
    idat_at = None
    idat_done = False
    ended = False
    for at, kind, body in _chunks(data):
        name = kind.decode("latin-1")
        if ended:
            raise FormatError(f"chunk {name} after IEND", at)
        if head is None and kind != b"IHDR":
            raise FormatError(f"first chunk is {name}, not IHDR", at)
        if kind == b"IHDR":
            if head is not None:
                raise FormatError("second IHDR", at)
            head = _Header(body, at)
            continue
        if kind == b"IDAT":
            if idat_done:
                raise FormatError("IDAT chunks are not contiguous", at)
            if idat_at is None:
                idat_at = at
            idat.append(body)
            continue
        if idat:
            idat_done = True
        if kind == b"IEND":
            if body:
                raise FormatError(f"IEND of {len(body)} bytes", at)
            ended = True
        elif kind == b"PLTE":
            if plte is not None:
                raise FormatError("second PLTE", at)
            if idat:
                raise FormatError("PLTE after IDAT", at)
            if trns is not None:
                raise FormatError("PLTE after tRNS", at)
            if head.colour in (0, 4):
                raise FormatError(f"PLTE in a colour type {head.colour} image", at)
            if not body or len(body) % 3 or len(body) // 3 > 256:
                raise FormatError(f"PLTE of {len(body)} bytes", at)
            if head.colour == 3 and len(body) // 3 > 1 << head.depth:
                raise FormatError(f"{len(body) // 3} palette entries for bit depth {head.depth}", at)
            plte, plte_at = body, at
        elif kind == b"tRNS":
            if trns is not None:
                raise FormatError("second tRNS", at)
            if idat:
                raise FormatError("tRNS after IDAT", at)
            trns, trns_at = body, at
        elif not kind[0] & 0x20:
            raise FormatError(f"unknown critical chunk {name}", at)
        # other ancillary chunks (gAMA, sRGB, iCCP, text, time, ...) are ignored
    end_at = len(data)
    if head is None:
        raise FormatError("no IHDR chunk", 8)
    if not ended:
        raise FormatError("no IEND chunk at the end", end_at)
    if not idat:
        raise FormatError("no IDAT chunk", end_at)
    if head.colour == 3 and plte is None:
        raise FormatError("palette image without PLTE", idat_at)

    palette = None
    if head.colour == 3:
        alphas = trns or b""
        palette = [
            bytes((plte[i * 3], plte[i * 3 + 1], plte[i * 3 + 2], alphas[i] if i < len(alphas) else 255))
            for i in range(len(plte) // 3)
        ]
    key = None
    if trns is not None:
        if head.colour in (4, 6):
            raise FormatError(f"tRNS in a colour type {head.colour} image", trns_at)
        if head.colour == 3:
            if len(trns) > len(palette):
                raise FormatError(f"tRNS of {len(trns)} entries for {len(palette)} palette entries",
                                  trns_at)
        elif head.colour == 0:
            if len(trns) != 2:
                raise FormatError(f"grey tRNS of {len(trns)} bytes", trns_at)
            key = struct.unpack(">H", trns)
        else:
            if len(trns) != 6:
                raise FormatError(f"RGB tRNS of {len(trns)} bytes", trns_at)
            key = struct.unpack(">3H", trns)

    inflater = zlib.decompressobj()
    try:
        raw = inflater.decompress(b"".join(idat)) + inflater.flush()
    except zlib.error as exc:
        raise FormatError(f"image data does not inflate: {exc}", idat_at) from None
    if not inflater.eof:
        raise FormatError("image data stream is truncated", idat_at)
    if inflater.unused_data:
        raise FormatError(f"{len(inflater.unused_data)} bytes after the image data stream", idat_at)
    passes = head.passes()
    expected = sum((1 + head.row_bytes(w)) * h for _, _, _, _, w, h in passes)
    if len(raw) != expected:
        raise FormatError(f"image data inflates to {len(raw)} bytes, {expected} expected", idat_at)

    width, height = head.width, head.height
    out = [[b""] * width for _ in range(height)]
    pos = 0
    for x0, y0, dx, dy, w, h in passes:
        rows, pos = _unfilter(raw, pos, head.row_bytes(w), h, head.bpp, idat_at)
        for j, pixels in enumerate(_pixels(head, rows, w, palette, key, idat_at)):
            target = out[y0 + j * dy]
            for i, pixel in enumerate(pixels):
                target[x0 + i * dx] = pixel
    return Image(width, height, b"".join(b"".join(row) for row in out))


__all__ = ["ADAM7", "COLOUR_TYPES", "SIGNATURE", "read_png"]
