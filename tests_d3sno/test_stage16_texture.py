"""Stage 16, textures: ``34_modding_tools.md`` W16-1 (PNG reader) and W16-2 to W16-4 (texture
import onto a template). Synthetic data only."""

import random
import struct
import unittest
import zlib

from d3sno import png as P
from d3sno import texture as T
from d3sno import texture_import as TI
from d3sno import texture_write as W
from d3sno import x360_texture as XT
from d3sno.binary import FormatError
from d3sno.texture import Image

BUILDS = ("ps3", "switch", "x360")
FORMATS = (0, 4, 7, 23, 9, 10, 11, 12, 43)
DXT = (9, 10, 11, 12, 43)
_EXP5 = [v << 3 | v >> 2 for v in range(32)]


# --- A PNG writer for tests (any colour type, depth, filter, interlace) --------------------------

def chunk(kind, body):
    return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF)


def _paeth(a, b, c):
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    return b if pb <= pc else c


def _pack_row(samples, depth):
    if depth == 16:
        return struct.pack(f">{len(samples)}H", *samples)
    if depth == 8:
        return bytes(samples)
    out = bytearray()
    per = 8 // depth
    for i in range(0, len(samples), per):
        byte = 0
        for j, s in enumerate(samples[i:i + per]):
            byte |= s << (8 - depth * (j + 1))
        out.append(byte)
    return bytes(out)


def _filter_rows(rows, bpp, filters):
    out = bytearray()
    prior = bytes(len(rows[0])) if rows else b""
    for y, row in enumerate(rows):
        kind = filters[y % len(filters)]
        out.append(kind)
        for i, x in enumerate(row):
            a = row[i - bpp] if i >= bpp else 0
            b = prior[i]
            c = prior[i - bpp] if i >= bpp else 0
            pred = (0, a, b, (a + b) >> 1, _paeth(a, b, c))[kind]
            out.append((x - pred) & 255)
        prior = row
    return bytes(out)


def make_png(width, height, colour, depth, pixels, filters=(0,), interlace=0, plte=None, trns=None,
             before_idat=(), after_idat=(), idat_parts=1, raw_override=None):
    """``pixels[y][x]`` is a tuple of samples."""
    samples = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[colour]
    bpp = max(1, depth * samples // 8)
    if interlace:
        passes = [(x0, y0, dx, dy) for x0, y0, dx, dy in P.ADAM7]
    else:
        passes = [(0, 0, 1, 1)]
    raw = bytearray()
    for x0, y0, dx, dy in passes:
        rows = []
        for y in range(y0, height, dy):
            flat = [s for x in range(x0, width, dx) for s in pixels[y][x]]
            if flat:
                rows.append(_pack_row(flat, depth))
        if rows:
            raw += _filter_rows(rows, bpp, filters)
    if raw_override is not None:
        raw = raw_override
    stream = zlib.compress(bytes(raw))
    out = P.SIGNATURE + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, depth, colour, 0, 0, interlace))
    if plte is not None:
        out += chunk(b"PLTE", plte)
    if trns is not None:
        out += chunk(b"tRNS", trns)
    for c in before_idat:
        out += c
    step = -(-len(stream) // idat_parts)
    for i in range(idat_parts):
        out += chunk(b"IDAT", stream[i * step:(i + 1) * step])
    for c in after_idat:
        out += c
    return out + chunk(b"IEND", b"")


def expected_rgba(colour, depth, pixels, palette=None, trns_key=None):
    """W16-1 conversion, written independently of the reader."""
    def scale(v):
        if depth == 16:
            return (v * 255 + 32767) // 65535
        return v * 255 // ((1 << depth) - 1)
    out = bytearray()
    for row in pixels:
        for s in row:
            if colour == 0:
                out += bytes((scale(s[0]),) * 3 + (0 if trns_key == s else 255,))
            elif colour == 2:
                out += bytes(tuple(scale(v) for v in s) + (0 if trns_key == s else 255,))
            elif colour == 3:
                out += palette[s[0]]
            elif colour == 4:
                out += bytes((scale(s[0]),) * 3 + (scale(s[1]),))
            else:
                out += bytes(scale(v) for v in s)
    return bytes(out)


def random_pixels(rng, width, height, samples, depth):
    top = (1 << depth) - 1
    return [[tuple(rng.randint(0, top) for _ in range(samples)) for _ in range(width)]
            for _ in range(height)]


# --- W16-1 ------------------------------------------------------------------------------------

class ReadPng(unittest.TestCase):
    def test_encode_png_reads_back(self):
        rng = random.Random(1)
        for w, h in ((1, 1), (7, 5), (16, 3), (33, 9)):
            image = Image(w, h, bytes(rng.randrange(256) for _ in range(w * h * 4)))
            self.assertEqual(P.read_png(T.encode_png(image)), image)

    def test_every_colour_type_depth_filter_and_interlace(self):
        rng = random.Random(2)
        sizes = ((9, 7), (1, 1), (3, 2), (8, 8), (13, 1))
        for colour, (depths, samples) in P.COLOUR_TYPES.items():
            for depth in depths:
                for interlace in (0, 1):
                    for w, h in sizes:
                        pixels = random_pixels(rng, w, h, samples, depth)
                        palette = None
                        plte = None
                        if colour == 3:
                            entries = 1 << depth
                            palette = [bytes((rng.randrange(256), rng.randrange(256), rng.randrange(256), 255))
                                       for _ in range(entries)]
                            plte = b"".join(p[:3] for p in palette)
                        data = make_png(w, h, colour, depth, pixels, filters=(0, 1, 2, 3, 4),
                                        interlace=interlace, plte=plte)
                        got = P.read_png(data)
                        self.assertEqual((got.width, got.height), (w, h))
                        self.assertEqual(got.rgba, expected_rgba(colour, depth, pixels, palette),
                                         (colour, depth, interlace, w, h))

    def test_each_filter_alone(self):
        rng = random.Random(3)
        pixels = random_pixels(rng, 11, 6, 4, 8)
        for kind in range(5):
            for depth in (8, 16):
                px = pixels if depth == 8 else [[tuple(v * 257 for v in p) for p in row] for row in pixels]
                got = P.read_png(make_png(11, 6, 6, depth, px, filters=(kind,)))
                self.assertEqual(got.rgba, expected_rgba(6, depth, px), (kind, depth))

    def test_paeth_ties(self):
        rng = random.Random(5)
        pixels = random_pixels(rng, 48, 48, 4, 8)
        got = P.read_png(make_png(48, 48, 6, 8, pixels, filters=(4,)))
        self.assertEqual(got.rgba, expected_rgba(6, 8, pixels))

    def test_low_depth_scaling(self):
        for depth, values in ((1, (0, 1)), (2, (0, 1, 2, 3)), (4, (0, 5, 15))):
            pixels = [[(v,) for v in values]]
            got = P.read_png(make_png(len(values), 1, 0, depth, pixels))
            self.assertEqual(list(got.rgba[0::4]), [v * 255 // ((1 << depth) - 1) for v in values])
        self.assertEqual(P.read_png(make_png(3, 1, 0, 2, [[(0,), (1,), (3,)]])).rgba[0::4], bytes((0, 85, 255)))

    def test_16_bit_reduction(self):
        values = (0, 128, 129, 32767, 65535, 257 * 200)
        got = P.read_png(make_png(len(values), 1, 0, 16, [[(v,) for v in values]]))
        self.assertEqual(list(got.rgba[0::4]), [(v * 255 + 32767) // 65535 for v in values])
        self.assertEqual(list(got.rgba[0::4]), [0, 0, 1, 127, 255, 200])

    def test_grey_and_alpha(self):
        got = P.read_png(make_png(2, 1, 4, 8, [[(10, 20), (30, 40)]]))
        self.assertEqual(got.rgba, bytes((10, 10, 10, 20, 30, 30, 30, 40)))

    def test_palette_with_trns(self):
        plte = bytes((1, 2, 3, 4, 5, 6, 7, 8, 9))
        got = P.read_png(make_png(3, 1, 3, 8, [[(0,), (1,), (2,)]], plte=plte, trns=bytes((0, 77))))
        self.assertEqual(got.rgba, bytes((1, 2, 3, 0, 4, 5, 6, 77, 7, 8, 9, 255)))
        got = P.read_png(make_png(3, 1, 3, 8, [[(0,), (1,), (2,)]], plte=plte))
        self.assertEqual(got.rgba[3::4], bytes((255, 255, 255)))

    def test_trns_grey_and_rgb(self):
        got = P.read_png(make_png(2, 1, 0, 4, [[(5,), (6,)]], trns=struct.pack(">H", 5)))
        self.assertEqual(got.rgba[3::4], bytes((0, 255)))
        got = P.read_png(make_png(2, 1, 0, 16, [[(1000,), (1001,)]], trns=struct.pack(">H", 1001)))
        self.assertEqual(got.rgba[3::4], bytes((255, 0)))
        got = P.read_png(make_png(2, 1, 2, 8, [[(1, 2, 3), (1, 2, 4)]], trns=struct.pack(">3H", 1, 2, 3)))
        self.assertEqual(got.rgba, bytes((1, 2, 3, 0, 1, 2, 4, 255)))
        got = P.read_png(make_png(1, 1, 2, 16, [[(300, 2, 3)]], trns=struct.pack(">3H", 300, 2, 3)))
        self.assertEqual(got.rgba[3], 0)

    def test_ancillary_chunks_ignored(self):
        pixels = [[(9, 8, 7, 6)]]
        plain = P.read_png(make_png(1, 1, 6, 8, pixels))
        extras = [chunk(b"gAMA", struct.pack(">I", 45455)), chunk(b"sRGB", b"\x00"),
                  chunk(b"iCCP", b"x\x00\x00" + zlib.compress(b"p")), chunk(b"tEXt", b"a\x00b"),
                  chunk(b"tIME", bytes(7)), chunk(b"pHYs", bytes(9))]
        got = P.read_png(make_png(1, 1, 6, 8, pixels, before_idat=extras, after_idat=[chunk(b"zTXt", b"k\x00\x00")]))
        self.assertEqual(got, plain)
        # PLTE in an RGB image is a suggestion only
        got = P.read_png(make_png(1, 1, 6, 8, pixels, plte=bytes(3)))
        self.assertEqual(got, plain)

    def test_split_idat(self):
        rng = random.Random(4)
        pixels = random_pixels(rng, 20, 10, 3, 8)
        got = P.read_png(make_png(20, 10, 2, 8, pixels, idat_parts=5, filters=(4, 1)))
        self.assertEqual(got.rgba, expected_rgba(2, 8, pixels))


class ReadPngRefusals(unittest.TestCase):
    def good(self):
        return make_png(2, 2, 6, 8, [[(1, 2, 3, 4)] * 2] * 2)

    def refused(self, data, offset=None, text=None):
        with self.assertRaises(FormatError) as ctx:
            P.read_png(data)
        if offset is not None:
            self.assertEqual(ctx.exception.offset, offset)
        if text is not None:
            self.assertIn(text, ctx.exception.message)
        return ctx.exception

    def test_signature(self):
        self.refused(b"\x89PNX" + self.good()[4:], 0, "signature")
        self.refused(b"", 0)

    def test_crc(self):
        data = bytearray(self.good())
        data[8 + 8 + 13] ^= 1  # first byte of IHDR's CRC
        self.refused(bytes(data), 8 + 8 + 13, "CRC")
        data = bytearray(self.good())
        data[8 + 8 + 2] ^= 1  # IHDR body byte
        self.refused(bytes(data), 8 + 8 + 13, "CRC")

    def test_ihdr_first(self):
        good = self.good()
        ihdr = good[8:8 + 25]
        rest = good[8 + 25:]
        self.refused(P.SIGNATURE + chunk(b"tEXt", b"a\x00b") + ihdr + rest, 8, "IHDR")
        self.refused(P.SIGNATURE + rest, 8, "IHDR")
        self.refused(P.SIGNATURE + ihdr + ihdr + rest, 8 + 25, "second IHDR")

    def test_no_idat(self):
        good = self.good()
        self.refused(P.SIGNATURE + good[8:8 + 25] + chunk(b"IEND", b""), None, "no IDAT")

    def test_idat_contiguous(self):
        pixels = [[(1, 2, 3, 4)] * 2] * 2
        data = make_png(2, 2, 6, 8, pixels, idat_parts=2)
        # put a tEXt between the two IDAT chunks
        first_at = 8 + 25
        (length,) = struct.unpack_from(">I", data, first_at)
        second_at = first_at + 12 + length
        broken = data[:second_at] + chunk(b"tEXt", b"a\x00b") + data[second_at:]
        self.refused(broken, second_at + 12 + 3, "contiguous")

    def test_iend_last(self):
        good = self.good()
        self.refused(good + chunk(b"tEXt", b"a\x00b"), len(good), "after IEND")
        self.refused(good[:-12], len(good) - 12, "IEND")
        self.refused(good[:-12] + chunk(b"IEND", b"x"), len(good) - 12, "IEND of 1")
        self.refused(good + b"\x00", len(good), "truncated")

    def test_unknown_critical_chunk(self):
        data = make_png(1, 1, 6, 8, [[(1, 2, 3, 4)]], before_idat=[chunk(b"QUUX", b"")])
        self.refused(data, 8 + 25, "unknown critical chunk QUUX")
        # the same name as ancillary is fine
        P.read_png(make_png(1, 1, 6, 8, [[(1, 2, 3, 4)]], before_idat=[chunk(b"qUUX", b"")]))

    def test_bad_chunk_names_and_lengths(self):
        good = self.good()
        self.refused(good[:-12] + struct.pack(">I", 0) + b"IE1D" + bytes(4), len(good) - 12 + 4, "letters")
        self.refused(good[:-12] + struct.pack(">I", 0x80000000) + b"IEND", len(good) - 12, "2^31")
        self.refused(good[:-12] + struct.pack(">I", 50) + b"IEND" + bytes(4), len(good) - 12, "past the end")

    def test_header_values(self):
        for colour, depth in ((0, 3), (2, 4), (3, 16), (4, 1), (6, 2), (5, 8), (1, 8)):
            self.refused(P.SIGNATURE + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, depth, colour, 0, 0, 0)),
                         8 + 16 if colour in P.COLOUR_TYPES else 8 + 17)
        for extra, at in (((1, 0, 0), 8 + 18), ((0, 1, 0), 8 + 19), ((0, 0, 2), 8 + 20)):
            self.refused(P.SIGNATURE + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 6, *extra)), at)
        self.refused(P.SIGNATURE + chunk(b"IHDR", struct.pack(">IIBBBBB", 0, 1, 8, 6, 0, 0, 0)), 16, "size")
        self.refused(P.SIGNATURE + chunk(b"IHDR", bytes(12)), 8, "13")

    def test_filter_type(self):
        raw = bytes((5, 1, 2, 3, 4))
        data = make_png(1, 1, 6, 8, [[(1, 2, 3, 4)]], raw_override=raw)
        self.refused(data, 8 + 25, "filter type 5")

    def test_image_data_size(self):
        self.refused(make_png(1, 1, 6, 8, [[(1, 2, 3, 4)]], raw_override=bytes(6)), 8 + 25, "inflates")
        self.refused(make_png(1, 1, 6, 8, [[(1, 2, 3, 4)]], raw_override=bytes(4)), 8 + 25, "inflates")
        good = self.good()
        stream = zlib.compress(bytes(10))
        base = P.SIGNATURE + good[8:8 + 25]
        self.refused(base + chunk(b"IDAT", stream[:-3]) + chunk(b"IEND", b""), 8 + 25, "truncated")
        self.refused(base + chunk(b"IDAT", stream + b"xx") + chunk(b"IEND", b""), 8 + 25, "after")
        self.refused(base + chunk(b"IDAT", b"garbage") + chunk(b"IEND", b""), 8 + 25, "inflate")

    def test_palette_rules(self):
        pixels = [[(0,), (2,)]]
        self.refused(make_png(2, 1, 3, 8, pixels), 8 + 25, "without PLTE")
        self.refused(make_png(2, 1, 3, 8, pixels, plte=bytes(6)), 8 + 25 + 18, "palette index 2")
        self.refused(make_png(2, 1, 3, 8, pixels, plte=bytes(4)), 8 + 25, "PLTE of 4")
        self.refused(make_png(2, 1, 3, 1, [[(0,), (1,)]], plte=bytes(9)), 8 + 25, "palette entries")
        self.refused(make_png(1, 1, 0, 8, [[(0,)]], plte=bytes(3)), 8 + 25, "colour type 0")
        self.refused(make_png(1, 1, 4, 8, [[(0, 0)]], plte=bytes(3)), 8 + 25, "colour type 4")
        self.refused(make_png(1, 1, 3, 8, [[(0,)]], plte=bytes(3), before_idat=[chunk(b"PLTE", bytes(3))]),
                     8 + 25 + 15, "second PLTE")
        self.refused(make_png(1, 1, 3, 8, [[(0,)]], plte=bytes(3), trns=bytes(2)), 8 + 25 + 15, "tRNS of 2")
        data = make_png(1, 1, 2, 8, [[(0, 0, 0)]], after_idat=[chunk(b"PLTE", bytes(3))])
        self.refused(data, None, "PLTE after IDAT")
        data = make_png(1, 1, 3, 8, [[(0,)]], trns=b"\x00", before_idat=[chunk(b"PLTE", bytes(3))])
        self.refused(data, 8 + 25 + 13, "PLTE after tRNS")

    def test_trns_rules(self):
        self.refused(make_png(1, 1, 6, 8, [[(1, 2, 3, 4)]], trns=bytes(6)), 8 + 25, "colour type 6")
        self.refused(make_png(1, 1, 4, 8, [[(1, 2)]], trns=bytes(2)), 8 + 25, "colour type 4")
        self.refused(make_png(1, 1, 0, 8, [[(1,)]], trns=bytes(3)), 8 + 25, "grey tRNS")
        self.refused(make_png(1, 1, 2, 8, [[(1, 2, 3)]], trns=bytes(2)), 8 + 25, "RGB tRNS")
        self.refused(make_png(1, 1, 2, 8, [[(1, 2, 3)]], trns=bytes(6), before_idat=[chunk(b"tRNS", bytes(6))]),
                     8 + 25 + 18, "second tRNS")
        self.refused(make_png(1, 1, 2, 8, [[(1, 2, 3)]], after_idat=[chunk(b"tRNS", bytes(6))]), None,
                     "tRNS after IDAT")


# --- Texture helpers ---------------------------------------------------------------------------

def random_image(w, h, seed):
    rng = random.Random(seed * 7919 + w * 31 + h)
    return Image(w, h, bytes(rng.randrange(256) for _ in range(w * h * 4)))


def smooth_image(w, h, seed=0):
    out = bytearray()
    for y in range(h):
        for x in range(w):
            out += bytes(((x * 255 // max(w - 1, 1) + seed * 40) % 256, y * 255 // max(h - 1, 1),
                          (128 + seed * 17) % 256, (x + y) * 255 // max(w + h - 2, 1)))
    return Image(w, h, bytes(out))


def parse(data, build):
    return T.parse_texture(data, build, strict=True)


def png_trip(image):
    return P.read_png(T.encode_png(image))


def export_chains(tex):
    """Every level of every face through PNG, as the exporter writes them."""
    return [[png_trip(TI.decode_template(tex, m, f)) for m in range(tex.levels)] for f in range(tex.faces)]


def export_slices(tex):
    return [png_trip(TI.decode_template(tex, 0, 0, z)) for z in range(tex.depth)]


def template_for(build, fmt, w=16, h=16, faces=1, levels=None, seed=1, smooth=False):
    make = (lambda s: smooth_image(w, h, s)) if smooth else (lambda s: random_image(w, h, s))
    images = [make(seed + f) for f in range(faces)]
    data = W.encode_texture(images if faces > 1 else images[0], build, fmt, levels=levels, faces=faces)
    return data, parse(data, build)


def edit_colour(fmt):
    return {0: (10, 20, 30, 40), 4: (_EXP5[3], _EXP5[7], _EXP5[9], 255), 7: (99, 99, 99, 255),
            23: (0, 0, 0, 77)}.get(fmt, (_EXP5[4], 40, _EXP5[20], 255))


def edit_area(image, x0, y0, colour):
    raw = bytearray(image.rgba)
    for y in range(y0, y0 + 4):
        for x in range(x0, x0 + 4):
            at = (y * image.width + x) * 4
            raw[at:at + 4] = bytes(colour)
    return Image(image.width, image.height, bytes(raw))


_VOLUMES = {}


def volume_template(build):
    if build not in _VOLUMES:
        rng = random.Random(9)
        slices = [Image(64, 64, bytes(rng.randrange(256) for _ in range(64 * 64 * 4))) for _ in range(64)]
        data = W.encode_texture(slices, build, 0, depth=64)
        _VOLUMES[build] = (data, parse(data, build))
    return _VOLUMES[build]


# --- W16-2 / W16-3: round trips ---------------------------------------------------------------

class RoundTrip(unittest.TestCase):
    def test_every_format_and_build_full_chain(self):
        for build in BUILDS:
            for fmt in FORMATS:
                data, tex = template_for(build, fmt, 16, 16)
                chains = export_chains(tex)
                self.assertEqual(TI.import_texture(tex, chains[0]), data, (build, fmt))
                # only level 0 given: the lower levels come from the template
                self.assertEqual(TI.import_texture(tex, chains[0][0]), data, (build, fmt, "top"))

    def test_shapes(self):
        shapes = ((32, 8, None), (8, 32, None), (12, 10, None), (16, 16, 1), (4, 4, None), (64, 16, 2))
        for build in BUILDS:
            for fmt in (0, 4, 9, 12):
                for w, h, levels in shapes:
                    data, tex = template_for(build, fmt, w, h, levels=levels, seed=w + h)
                    chain = export_chains(tex)[0]
                    self.assertEqual(TI.import_texture(tex, chain), data, (build, fmt, w, h))
                    self.assertEqual(TI.import_texture(tex, [chain[0]]), data, (build, fmt, w, h))

    def test_cube(self):
        for build in BUILDS:
            for fmt in (0, 4, 7, 9, 11, 12):
                data, tex = template_for(build, fmt, 16, 16, faces=6, seed=fmt)
                chains = export_chains(tex)
                self.assertEqual(TI.import_texture(tex, chains), data, (build, fmt))
                self.assertEqual(TI.import_texture(tex, [c[0] for c in chains]), data, (build, fmt))
                # through a strip of level 0, as the exporter's strip option writes it
                strip = png_trip(T.vertical_strip([TI.decode_template(tex, 0, f) for f in range(6)]))
                self.assertEqual(TI.import_texture(tex, TI.split_strip(strip, 6)), data, (build, fmt))

    def test_volume(self):
        for build in BUILDS:
            data, tex = volume_template(build)
            slices = [TI.decode_template(tex, 0, 0, z) for z in range(64)]
            self.assertEqual(TI.import_texture(tex, slices), data, build)
            strip = T.vertical_strip(slices)
            self.assertEqual(TI.import_texture(tex, TI.split_strip(strip, 64)), data, build)
        data, tex = volume_template("ps3")
        self.assertEqual(TI.import_texture(tex, export_slices(tex)), data)

    def test_foreign_dxt_blocks_kept(self):
        # blocks another encoder made (random endpoints and indices) survive the round trip
        from d3sno.write import write_tex
        import dataclasses
        rng = random.Random(11)
        for build in BUILDS:
            for fmt in DXT:
                data, tex = template_for(build, fmt, 16, 16, smooth=True)
                bpp = T.FORMATS[fmt][1]
                elements = {(m, 0): bytes(rng.randrange(256) for _ in range(len(raw)))
                            for (m, _), raw in W.texture_elements(tex).items()}
                foreign = write_tex(dataclasses.replace(tex, payload=TI._patch_payload(tex, elements)))
                ftex = parse(foreign, build)
                self.assertEqual(W.texture_elements(ftex)[(0, 0)], elements[(0, 0)])
                self.assertEqual(TI.import_texture(ftex, export_chains(ftex)[0]), foreign, (build, fmt))
                self.assertEqual(TI.import_texture(ftex, png_trip(TI.decode_template(ftex, 0))), foreign)
                edited = edit_area(TI.decode_template(ftex, 0), 0, 0, edit_colour(fmt))
                new = W.texture_elements(parse(TI.import_texture(ftex, edited), build))[(0, 0)]
                self.assertEqual(new[bpp:], elements[(0, 0)][bpp:], (build, fmt))
                del bpp

    def test_template_header_values_kept(self):
        for build in BUILDS:
            base = W.encode_texture(random_image(16, 16, 5), build, 12, levels=2)
            raw = bytearray(base)
            order = ">" if build != "switch" else "<"
            ids_at, cube_at, import_at, time_at, console_at, scale_at, volume_at = W._FIELDS[build]
            struct.pack_into(order + "4I", raw, 8, 3, 0xABCD, 0x1234, 0)
            struct.pack_into(order + "ii", raw, 0x38, 2, 3)
            struct.pack_into(order + "IIIf", raw, import_at, 0x55, 0x66, 3, 0.5)
            struct.pack_into(order + "II", raw, volume_at, 0, 0)
            struct.pack_into(order + "I", raw, 0x34, 7)  # a stored oddity is kept as well
            raw[0x100] = 0x5A
            data = bytes(raw)
            tex = T.parse_texture(data, build)
            chain = export_chains(tex)[0]
            self.assertEqual(TI.import_texture(tex, chain), data, build)
            self.assertEqual(TI.import_texture(tex, chain[0]), data, build)

    def test_stray_payload_bytes_kept(self):
        # ps3 cube: the last face's padding holds a stray byte; switch: a byte between surfaces
        data, tex = template_for("ps3", 9, 16, 16, faces=6)
        raw = bytearray(data)
        raw[-1] = 0x77
        tex = T.parse_texture(bytes(raw), "ps3")
        self.assertEqual(TI.import_texture(tex, [c[0] for c in export_chains(tex)]), bytes(raw))
        data, tex = template_for("x360", 0, 16, 16)
        raw = bytearray(data)
        raw[-1] = 0x66  # tiled padding of the tail surface
        tex = T.parse_texture(bytes(raw), "x360")
        self.assertEqual(TI.import_texture(tex, export_chains(tex)[0]), bytes(raw))


class EditOneArea(unittest.TestCase):
    def test_edit_changes_only_that_area(self):
        for build in BUILDS:
            for fmt in FORMATS:
                data, tex = template_for(build, fmt, 16, 16, smooth=fmt in DXT, seed=2)
                kind, bpp = T.FORMATS[fmt]
                top = png_trip(TI.decode_template(tex, 0))
                edited = edit_area(top, 4, 8, edit_colour(fmt))
                out = TI.import_texture(tex, edited)
                new = parse(out, build)  # strict
                self.assertEqual((new.levels, new.format, new.width), (tex.levels, fmt, 16))
                old_el, new_el = W.texture_elements(tex), W.texture_elements(new)
                changed = [i for i in range(len(old_el[(0, 0)]) // bpp)
                           if old_el[(0, 0)][i * bpp:(i + 1) * bpp] != new_el[(0, 0)][i * bpp:(i + 1) * bpp]]
                if fmt in DXT:
                    self.assertEqual(changed, [2 * 4 + 1], (build, fmt))  # block (1, 2)
                else:
                    area = [y * 16 + x for y in range(8, 12) for x in range(4, 8)]
                    self.assertTrue(set(changed) <= set(area) and changed, (build, fmt))
                # level 0 decodes to the edited image
                self.assertEqual(TI.decode_template(new, 0), edited, (build, fmt))
                # lower levels are made again from the edited image
                chain = [edited]
                for m in range(1, new.levels):
                    chain.append(W.downsample(chain[-1]))
                    w, h = new.level_size(m)
                    want = TI.decode_form(kind, build, W.element_form(chain[m], fmt, build), w, h)
                    self.assertEqual(TI.decode_template(new, m).rgba, want, (build, fmt, m))

    def test_edit_cube_face(self):
        for build in BUILDS:
            data, tex = template_for(build, 9, 16, 16, faces=6, seed=3, smooth=True)
            faces = [png_trip(TI.decode_template(tex, 0, f)) for f in range(6)]
            faces[4] = edit_area(faces[4], 12, 0, edit_colour(9))
            new = parse(TI.import_texture(tex, faces), build)
            old_el, new_el = W.texture_elements(tex), W.texture_elements(new)
            for (level, face), raw in old_el.items():
                if face != 4:
                    self.assertEqual(new_el[(level, face)], raw, (build, level, face))
            diff = [i for i in range(16) if old_el[(0, 4)][i * 8:i * 8 + 8] != new_el[(0, 4)][i * 8:i * 8 + 8]]
            self.assertEqual(diff, [3])

    def test_edit_volume_slice(self):
        for build in BUILDS:
            data, tex = volume_template(build)
            slices = [TI.decode_template(tex, 0, 0, z) for z in range(64)]
            slices[10] = edit_area(slices[10], 0, 0, (1, 2, 3, 4))
            out = TI.import_texture(tex, slices)
            new = parse(out, build)
            self.assertEqual(TI.decode_template(new, 0, 0, 10), slices[10])
            changed = sum(a != b for a, b in zip(new.payload, tex.payload))
            self.assertLessEqual(changed, 16 * 4)
            self.assertGreater(changed, 0)


class MissingLevels(unittest.TestCase):
    def odd_chain_template(self, build, fmt):
        chain = [random_image(16 >> m, 16 >> m, 40 + m) for m in range(W.max_levels(fmt, 16, 16))]
        data = W.encode_texture([chain], build, fmt)
        return data, parse(data, build)

    def test_unchanged_top_takes_template_levels(self):
        for build in BUILDS:
            for fmt in (0, 12):
                data, tex = self.odd_chain_template(build, fmt)
                self.assertEqual(TI.import_texture(tex, TI.decode_template(tex, 0)), data)

    def test_changed_top_downsamples(self):
        for build in BUILDS:
            data, tex = self.odd_chain_template(build, 0)
            edited = edit_area(TI.decode_template(tex, 0), 0, 0, (1, 1, 1, 1))
            new = parse(TI.import_texture(tex, edited), build)
            self.assertEqual(TI.decode_template(new, 1), W.downsample(edited))
            self.assertNotEqual(TI.decode_template(new, 2), TI.decode_template(tex, 2))

    def test_partial_chain(self):
        for build in BUILDS:
            data, tex = self.odd_chain_template(build, 0)
            top, one = TI.decode_template(tex, 0), TI.decode_template(tex, 1)
            one_edit = edit_area(one, 0, 0, (5, 5, 5, 5))
            new = parse(TI.import_texture(tex, [top, one_edit], levels=tex.levels), build)
            self.assertEqual(TI.decode_template(new, 1), one_edit)
            for m in range(2, tex.levels):  # still the template's: the top is unchanged
                self.assertEqual(W.texture_elements(new)[(m, 0)], W.texture_elements(tex)[(m, 0)])


class LevelCount(unittest.TestCase):
    def test_levels_argument(self):
        for build in BUILDS:
            data, tex = template_for(build, 0, 16, 16)
            new = parse(TI.import_texture(tex, TI.decode_template(tex, 0), levels=2), build)
            self.assertEqual(new.levels, 2)
            for m in range(2):
                self.assertEqual(W.texture_elements(new)[(m, 0)], W.texture_elements(tex)[(m, 0)])
            self.assertEqual((new.import_flags, new.mip_min), (tex.import_flags, tex.mip_min))

    def test_chain_length(self):
        data, tex = template_for("ps3", 0, 16, 16, levels=1)
        chain = [random_image(16 >> m, 16 >> m, m) for m in range(3)]
        new = parse(TI.import_texture(tex, chain), "ps3")
        self.assertEqual(new.levels, 3)
        self.assertEqual(TI.decode_template(new, 2), chain[2])

    def test_template_count_for_new_size(self):
        for build in BUILDS:
            data, tex = template_for(build, 0, 16, 16, levels=3)
            new = parse(TI.import_texture(tex, random_image(32, 32, 1)), build)
            self.assertEqual(new.levels, 3)

    def test_full_chain_when_template_count_not_allowed(self):
        for build in BUILDS:
            data, tex = template_for(build, 0, 16, 16)  # 5 levels
            new = parse(TI.import_texture(tex, random_image(4, 4, 1)), build)
            self.assertEqual(new.levels, 3)
            new = parse(TI.import_texture(tex, random_image(12, 4, 1)), build)
            self.assertEqual(new.levels, 1)

    def test_size_change_keeps_header(self):
        for build in BUILDS:
            base = W.encode_texture(random_image(16, 16, 5), build, 0, levels=2,
                                    frames=[(0x99, 0.0, 0.0, 0.5, 1.0), (0x98, 0.5, 0.0, 1.0, 1.0)])
            raw = bytearray(base)
            order = ">" if build != "switch" else "<"
            ids_at, cube_at, import_at, time_at, console_at, scale_at, volume_at = W._FIELDS[build]
            struct.pack_into(order + "IIIf", raw, import_at, 0x55, 0x66, 3, 0.25)
            struct.pack_into(order + "ii", raw, 0x38, 1, 2)
            tex = T.parse_texture(bytes(raw), build)
            image = random_image(32, 32, 6)
            new = parse(TI.import_texture(tex, image), build)
            self.assertEqual((new.width, new.height, new.levels, new.mip_min), (32, 32, 2, 1))
            self.assertEqual((new.import_flags, new.machine_time, new.console_mip_offset), (0x55, 0x66, 3))
            self.assertEqual(new.lightmap_v_scale, 0.25)
            self.assertEqual(new.frames, tex.frames)
            self.assertEqual(new.image_ids, tex.image_ids)
            self.assertEqual(new.frames[1].rect(new.width, new.height), (16, 0, 32, 32))
            self.assertEqual(TI.decode_template(new, 0), image)
            self.assertEqual(TI.decode_template(new, 1), W.downsample(image))
            self.assertEqual(new.header.handle, tex.header.handle)

    def test_format_argument(self):
        for build in BUILDS:
            data, tex = template_for(build, 9, 16, 16, smooth=True)
            new = parse(TI.import_texture(tex, TI.decode_template(tex, 0), format=0), build)
            self.assertEqual((new.format, new.levels), (0, tex.levels))
            for m in range(tex.levels):  # missing levels from the template, stored exactly
                self.assertEqual(TI.decode_template(new, m), TI.decode_template(tex, m))
            self.assertEqual(new.import_flags, tex.import_flags)


class Strip(unittest.TestCase):
    def test_split(self):
        images = [random_image(4, 4, s) for s in range(3)]
        self.assertEqual(TI.split_strip(T.vertical_strip(images), 3), images)
        self.assertEqual(TI.split_strip(images[0], 1), [images[0]])

    def test_refusals(self):
        strip = T.vertical_strip([random_image(4, 4, s) for s in range(3)])
        with self.assertRaises(FormatError):
            TI.split_strip(strip, 2)
        with self.assertRaises(FormatError):
            TI.split_strip(T.vertical_strip([random_image(4, 2, s) for s in range(6)]), 6)
        for bad in (0, -1, 1.5, True, "3"):
            with self.assertRaises(ValueError):
                TI.split_strip(strip, bad)
        with self.assertRaises(ValueError):
            TI.split_strip(b"png", 1)
        with self.assertRaises(ValueError):
            TI.split_strip(Image(4, 12, bytes(10)), 3)


class ImportRefusals(unittest.TestCase):
    def test_bad_arguments(self):
        data, tex = template_for("ps3", 0, 16, 16)
        image = random_image(16, 16, 1)
        with self.assertRaises(ValueError):
            TI.import_texture(data, image)  # not parsed
        with self.assertRaises(ValueError):
            TI.import_texture(tex, image, levels=0)
        with self.assertRaises(ValueError):
            TI.import_texture(tex, image, levels="2")
        with self.assertRaises(ValueError):
            TI.import_texture(tex, image, format=99)
        with self.assertRaises(ValueError):
            TI.import_texture(tex, Image(16, 16, bytes(5)))
        with self.assertRaises(ValueError):
            TI.import_texture(tex, "image.png")
        with self.assertRaises(ValueError):
            TI.import_texture(tex, [])
        with self.assertRaises(ValueError):  # chain longer than the level count
            TI.import_texture(tex, [image, W.downsample(image), W.downsample(W.downsample(image))], levels=2)

    def test_level_rule(self):
        data, tex = template_for("ps3", 9, 16, 16)
        with self.assertRaises(FormatError):
            TI.import_texture(tex, random_image(16, 16, 1), levels=4)  # DXT 16x16 allows 3
        with self.assertRaises(FormatError):
            TI.import_texture(tex, random_image(12, 12, 1), levels=2)  # not a power of two

    def test_chain_sizes(self):
        data, tex = template_for("switch", 0, 16, 16)
        with self.assertRaises(FormatError):
            TI.import_texture(tex, [random_image(16, 16, 1), random_image(4, 4, 2)])

    def test_cube_faces(self):
        data, tex = template_for("ps3", 0, 8, 8, faces=6)
        with self.assertRaises(ValueError):
            TI.import_texture(tex, random_image(8, 8, 1))
        with self.assertRaises(ValueError):
            TI.import_texture(tex, [random_image(8, 8, s) for s in range(5)])
        faces = [random_image(8, 8, s) for s in range(6)]
        faces[3] = random_image(4, 4, 3)
        with self.assertRaises(FormatError):
            TI.import_texture(tex, faces)

    def test_volume(self):
        data, tex = volume_template("switch")
        slices = [TI.decode_template(tex, 0, 0, z) for z in range(64)]
        with self.assertRaises(ValueError):
            TI.import_texture(tex, slices[:63])
        with self.assertRaises(ValueError):
            TI.import_texture(tex, [[s, W.downsample(s)] for s in slices])
        with self.assertRaises(FormatError):
            TI.import_texture(tex, [random_image(32, 32, z) for z in range(64)])
        with self.assertRaises(FormatError):
            TI.import_texture(tex, slices, format=9)


if __name__ == "__main__":
    unittest.main()
