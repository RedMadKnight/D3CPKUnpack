"""Stage 14, textures: ``28_derived_values.md`` W14-7 to W14-11 (the encoder), synthetic data only."""

import random
import struct
import unittest

import builders as B
from d3sno import texture as T
from d3sno import texture_write as W
from d3sno import x360_texture as XT
from d3sno.binary import FormatError
from d3sno.texture import Image

BUILDS = ("ps3", "switch", "x360")
UNCOMPRESSED = (0, 4, 7, 23)
DXT = (9, 10, 11, 12, 43)
_EXP5 = [v << 3 | v >> 2 for v in range(32)]


def exact_image(fmt, w, h, seed=1):
    """An image every uncompressed format stores exactly (values the format can hold)."""
    rng = random.Random(seed * 1000 + w * 31 + h)
    out = bytearray()
    for _ in range(w * h):
        if fmt == 4:
            out += bytes((rng.choice(_EXP5), rng.choice(_EXP5), rng.choice(_EXP5), rng.choice((0, 255))))
        elif fmt == 7:
            v = rng.randrange(256)
            out += bytes((v, v, v, 255))
        elif fmt == 23:
            out += bytes((0, 0, 0, rng.randrange(256)))
        else:
            out += bytes(rng.randrange(256) for _ in range(4))
    return Image(w, h, bytes(out))


def smooth_image(w, h, alpha=True):
    out = bytearray()
    for y in range(h):
        for x in range(w):
            out += bytes((x * 255 // max(w - 1, 1), y * 255 // max(h - 1, 1), 128,
                          (x + y) * 255 // max(w + h - 2, 1) if alpha else 255))
    return Image(w, h, bytes(out))


def parse(data, build):
    return T.parse_texture(data, build, strict=True)


def decode(tex, level=0, face=0, z=0):
    if tex.build == "x360":
        return XT.decode_image(tex, level, face, z)
    return T.decode_image(tex, level, face, z)


def max_diff(a, b):
    return max(abs(x - y) for x, y in zip(a.rgba, b.rgba))


def volume_slices(seed=3):
    rng = random.Random(seed)
    return [Image(64, 64, bytes(rng.randrange(256) for _ in range(64 * 64 * 4))) for _ in range(64)]


_VOLUME = {}


def volume_file(build):
    if build not in _VOLUME:
        slices = volume_slices()
        _VOLUME[build] = (slices, W.encode_texture(slices, build, 0, depth=64))
    return _VOLUME[build]


# --- W14-7 ------------------------------------------------------------------------------------

class HeaderValues(unittest.TestCase):
    def test_2d_defaults(self):
        for build in BUILDS:
            data = W.encode_texture(exact_image(0, 16, 8), build, 0)
            tex = parse(data, build)
            self.assertEqual((tex.width, tex.height, tex.faces, tex.depth), (16, 8, 1, 1))
            self.assertEqual(tex.cube_flag, 0)
            self.assertEqual((tex.volume_x, tex.volume_y), (1, 1))
            self.assertEqual(tex.image_ids, [0])
            self.assertEqual(tex.frames, [T.TextureFrame(0, 0.0, 0.0, 1.0, 1.0)])
            order = ">" if build != "switch" else "<"
            self.assertEqual(struct.unpack_from(order + "I", data, 0x34)[0], 0)
            self.assertEqual(struct.unpack_from(order + "I", data, 0x224 + 4)[0], 20)
            self.assertEqual((tex.mip_min, tex.mip_max), (0, 4))
            self.assertEqual(tex.header.version, {"ps3": 58, "switch": 66, "x360": 61}[build])
            self.assertEqual(tex.warnings, [])

    def test_cube(self):
        faces = [exact_image(0, 8, 8, seed=f) for f in range(6)]
        for build in BUILDS:
            tex = parse(W.encode_texture(faces, build, 0, levels=2, faces=6), build)
            self.assertEqual((tex.faces, tex.cube_flag, tex.levels), (6, 1, 2))
            self.assertEqual(tex.image_ids, [0] * 6)
            self.assertEqual(len(tex.frames), 1)
            self.assertEqual((tex.volume_x, tex.volume_y), (1, 1))

    def test_cube_has_one_frame(self):
        faces = [exact_image(0, 8, 8)] * 6
        with self.assertRaises(FormatError):
            W.encode_texture(faces, "ps3", 0, faces=6, frames=[(1, 0, 0, 1, 1), (2, 0, 0, 1, 1)])

    def test_volume(self):
        for build in BUILDS:
            _, data = volume_file(build)
            tex = parse(data, build)
            self.assertEqual((tex.depth, tex.faces, tex.cube_flag, tex.levels), (64, 1, 0, 1))
            self.assertEqual((tex.volume_x, tex.volume_y), (8, 8))
            self.assertEqual(tex.image_ids, [0])

    def test_frames_and_ids(self):
        frames = [(0x11, 0.0, 0.0, 0.5, 1.0), (0x22, 0.5, 0.0, 1.0, 1.0), (0x33, 0.0, 0.0, 1.0, 0.5)]
        for build in BUILDS:
            tex = parse(W.encode_texture(exact_image(0, 8, 8), build, 0, frames=frames), build)
            self.assertEqual(tex.image_ids, [0, 1, 2])
            self.assertEqual([f.hash for f in tex.frames], [0x11, 0x22, 0x33])
            self.assertEqual(tex.frames[1].rect(tex.width, tex.height), (4, 0, 8, 8))
            if build != "x360":
                self.assertEqual(T.frame_image(tex, 1).width, 4)

    def test_template_values_kept(self):
        for build in BUILDS:
            base = W.encode_texture(exact_image(0, 8, 8), build, 0, levels=1)
            raw = bytearray(base)
            order = ">" if build != "switch" else "<"
            ids_at, cube_at, import_at, time_at, console_at, scale_at, volume_at = W._FIELDS[build]
            struct.pack_into(order + "4I", raw, 8, 3, 0xABCD, 0x1234, 0)  # revision, source, handle
            struct.pack_into(order + "ii", raw, 0x38, 2, 2)  # mip min 2
            struct.pack_into(order + "IIIf", raw, import_at, 0x55, 0x66, 3, 0.5)
            struct.pack_into(order + "II", raw, volume_at, 0, 0)  # a 2D file with (0, 0)
            template = parse(bytes(raw), build)
            tex = parse(W.encode_texture(exact_image(0, 16, 16), build, 0, levels=3, template=template), build)
            self.assertEqual((tex.header.revision, tex.header.source_hash, tex.header.handle), (3, 0xABCD, 0x1234))
            self.assertEqual((tex.mip_min, tex.mip_max), (2, 4))
            self.assertEqual((tex.import_flags, tex.machine_time, tex.console_mip_offset), (0x55, 0x66, 3))
            self.assertEqual(tex.lightmap_v_scale, 0.5)
            self.assertEqual((tex.volume_x, tex.volume_y), (0, 0))
            self.assertEqual((tex.width, tex.height), (16, 16))

    def test_template_image_ids_kept(self):
        # SUPPORTED rule: a template's ids are kept when its frame count matches
        frames = [(i, 0.0, 0.0, 1.0, 1.0) for i in range(4)]
        for build in BUILDS:
            order = ">" if build != "switch" else "<"
            data = B.build_tex(build, 0, 4, 4, bytes(64) if build == "ps3" else bytes(64),
                               frames=frames, image_ids=[0],
                               payload=bytes(4096) if build == "x360" else None)
            template = T.parse_texture(data, build)
            self.assertEqual(template.image_ids, [0])
            tex = parse(W.encode_texture(exact_image(0, 4, 4), build, 0, levels=1, template=template), build)
            self.assertEqual(tex.image_ids, [0])
            fresh = parse(W.encode_texture(exact_image(0, 4, 4), build, 0, levels=1, frames=frames), build)
            self.assertEqual(fresh.image_ids, [0, 1, 2, 3])
            del order

    def test_whole_file_rebuilds_from_parsed(self):
        cases = [(fmt, 1, 1) for fmt in UNCOMPRESSED + DXT] + [(0, 6, 1), (12, 6, 1)]
        for build in BUILDS:
            for fmt, faces, depth in cases:
                images = [exact_image(fmt, 16, 16, seed=f) for f in range(faces)]
                data = W.encode_texture(images, build, fmt, faces=faces)
                tex = parse(data, build)
                again = W.assemble_texture(build, fmt, tex.width, tex.height, W.rebuild_payload(tex),
                                           tex.levels, tex.faces, tex.depth, template=tex)
                self.assertEqual(again, data, (build, fmt, faces))
                if fmt in UNCOMPRESSED:
                    chains = [[decode(tex, m, f) for m in range(tex.levels)] for f in range(faces)]
                    self.assertEqual(W.encode_texture(chains, build, fmt, faces=faces, template=tex), data)


# --- W14-8 ------------------------------------------------------------------------------------

class MipChain(unittest.TestCase):
    def test_rule(self):
        self.assertEqual(W.max_levels(9, 64, 16), 3)  # DXT ends at smaller side 4
        self.assertEqual(W.max_levels(12, 256, 256), 7)
        self.assertEqual(W.max_levels(0, 64, 16), 7)  # others end at 1 x 1
        self.assertEqual(W.max_levels(7, 1, 1), 1)
        self.assertEqual(W.max_levels(0, 24, 16), 1)  # not a power of two
        self.assertEqual(W.max_levels(9, 4, 4), 1)
        self.assertEqual(W.max_levels(0, 64, 64, 64), 1)
        self.assertEqual(W.level_sizes(64, 16, 7)[-3:], [(4, 1), (2, 1), (1, 1)])

    def test_last_level(self):
        for build in BUILDS:
            dxt = parse(W.encode_texture(smooth_image(64, 16), build, 9), build)
            self.assertEqual(dxt.levels, 3)
            self.assertEqual(min(dxt.level_size(dxt.levels - 1)), 4)
            plain = parse(W.encode_texture(exact_image(0, 64, 16), build, 0), build)
            self.assertEqual(plain.levels, 7)
            self.assertEqual(plain.level_size(plain.levels - 1), (1, 1))

    def test_refuses_more_levels(self):
        for build in BUILDS:
            W.encode_texture(smooth_image(16, 16), build, 12, levels=3)
            with self.assertRaises(FormatError):
                W.encode_texture(smooth_image(16, 16), build, 12, levels=4)
            with self.assertRaises(FormatError):
                W.encode_texture(exact_image(0, 24, 16), build, 0, levels=2)
            with self.assertRaises(FormatError):
                W.encode_texture(exact_image(0, 8, 8), build, 0, levels=5)

    def test_given_chain(self):
        top = exact_image(0, 8, 8)
        chain = [top, exact_image(0, 4, 4, 2), exact_image(0, 2, 2, 3)]
        for build in BUILDS:
            tex = parse(W.encode_texture([chain], build, 0), build)
            self.assertEqual(tex.levels, 3)
            self.assertEqual(decode(tex, 1).rgba, chain[1].rgba)
            with self.assertRaises(FormatError):
                W.encode_texture([[top, exact_image(0, 2, 2)]], build, 0)

    def test_downsample(self):
        image = Image(2, 2, bytes((0, 0, 0, 0, 10, 20, 30, 40, 0, 0, 0, 0, 10, 20, 30, 40)))
        self.assertEqual(W.downsample(image).rgba, bytes((5, 10, 15, 20)))
        self.assertEqual(W.downsample(Image(4, 1, bytes(range(16)))).width, 2)


# --- Pixels back through the existing decoders --------------------------------------------------

class RoundTrip(unittest.TestCase):
    def test_uncompressed_exact(self):
        for build in BUILDS:
            for fmt in UNCOMPRESSED:
                image = exact_image(fmt, 16, 8)
                tex = parse(W.encode_texture(image, build, fmt), build)
                self.assertEqual(decode(tex).rgba, image.rgba, (build, fmt))
                small = W.downsample(W.downsample(image))
                if fmt == 4:  # the mean is not always a 5-bit value: close, alpha one bit
                    got = decode(tex, 2).rgba
                    self.assertLessEqual(max(abs(got[i] - small.rgba[i]) for i in range(len(got)) if i % 4 != 3), 4)
                    self.assertEqual(got[3::4], bytes(255 if a >= 128 else 0 for a in small.rgba[3::4]))
                else:
                    self.assertEqual(decode(tex, 2).rgba, small.rgba, (build, fmt))

    def test_dxt_close(self):
        image = smooth_image(32, 32)
        for build in BUILDS:
            for fmt in DXT:
                tex = parse(W.encode_texture(image, build, fmt), build)
                got = decode(tex)
                # DXT1: texels with alpha below 128 become transparent black
                kept = [i for i in range(len(image.rgba)) if i % 4 != 3
                        and (fmt not in (9, 10) or image.rgba[i | 3] >= 128)]
                rgb = max(abs(got.rgba[i] - image.rgba[i]) for i in kept)
                self.assertLessEqual(rgb, 24, (build, fmt))
                if fmt in (9, 10):
                    self.assertEqual(got.rgba[3::4], bytes(255 if a >= 128 else 0 for a in image.rgba[3::4]))
                if fmt in (11, 12, 43):
                    alpha = max(abs(got.rgba[i] - image.rgba[i]) for i in range(3, len(image.rgba), 4))
                    self.assertLessEqual(alpha, 18, (build, fmt))

    def test_dxt_two_colour_blocks_exact(self):
        a, b = (_EXP5[31], 255, 0, 255), (0, 0, _EXP5[31], 255)
        rgba = b"".join(bytes(a if (x + y) % 3 else b) for y in range(8) for x in range(8))
        image = Image(8, 8, rgba)
        for build in BUILDS:
            for fmt in DXT:
                tex = parse(W.encode_texture(image, build, fmt, levels=1), build)
                self.assertEqual(decode(tex).rgba, rgba, (build, fmt))

    def test_dxt1_punch_through(self):
        rgba = b"".join(bytes((200, 40, 40, 255 if x < 2 else 0)) for y in range(4) for x in range(4))
        for build in BUILDS:
            got = decode(parse(W.encode_texture(Image(4, 4, rgba), build, 10), build))
            self.assertEqual([got.rgba[i * 4 + 3] for i in range(16)], [255, 255, 0, 0] * 4)

    def test_odd_size(self):
        image = exact_image(0, 6, 5)
        for build in BUILDS:
            tex = parse(W.encode_texture(image, build, 0), build)
            self.assertEqual(tex.levels, 1)
            self.assertEqual(decode(tex).rgba, image.rgba)
            dxt = parse(W.encode_texture(smooth_image(6, 5), build, 12), build)
            self.assertEqual((decode(dxt).width, decode(dxt).height), (6, 5))

    def test_cube_faces(self):
        faces = [exact_image(4, 8, 8, seed=f) for f in range(6)]
        for build in BUILDS:
            tex = parse(W.encode_texture(faces, build, 4, faces=6), build)
            for f in range(6):
                self.assertEqual(decode(tex, 0, f).rgba, faces[f].rgba, (build, f))

    def test_volume_slices(self):
        for build in BUILDS:
            slices, data = volume_file(build)
            tex = parse(data, build)
            for z in (0, 17, 63):
                self.assertEqual(decode(tex, 0, 0, z).rgba, slices[z].rgba, (build, z))


# --- W14-9 ------------------------------------------------------------------------------------

class Ps3Payload(unittest.TestCase):
    def test_size(self):
        for fmt, w, h, levels, faces in ((0, 8, 8, 2, 1), (9, 16, 16, 3, 1), (7, 4, 4, 3, 6), (12, 8, 8, 1, 6)):
            chain = sum(T._ps3_level_bytes(
                parse(W.encode_texture(exact_image(0, w, h), "ps3", fmt, levels=levels), "ps3"), m)
                for m in range(levels))
            imgs = [exact_image(0, w, h, f) for f in range(faces)]
            tex = parse(W.encode_texture(imgs, "ps3", fmt, levels=levels, faces=faces), "ps3")
            # V16-1 (29_corrections17.md): only a cube pads its faces
            self.assertEqual(len(tex.payload), 6 * (-(-chain // 128) * 128) if faces == 6 else chain)
        self.assertEqual(len(parse(volume_file("ps3")[1], "ps3").payload), 64 ** 3 * 4)

    def test_padding_is_zero(self):
        tex = parse(W.encode_texture([exact_image(0, 8, 8, f) for f in range(6)], "ps3", 0, levels=2, faces=6), "ps3")
        # chain 256 + 64 = 320 -> stride 384
        for f in range(6):
            self.assertEqual(tex.payload[f * 384 + 320:(f + 1) * 384], bytes(64))

    def test_rebuild_builder_file(self):
        rng = random.Random(5)
        for fmt, w, h, mip_max, faces in ((0, 8, 8, 1, 1), (9, 8, 8, 0, 1), (4, 4, 4, 2, 6)):
            tex0 = parse(W.encode_texture([exact_image(0, w, h)] * faces, "ps3", fmt, levels=mip_max + 1,
                                          faces=faces), "ps3")
            chain = sum(T._ps3_level_bytes(tex0, m) for m in range(tex0.levels))
            stride = -(-chain // 128) * 128 if faces == 6 else chain  # V16-1
            payload = b"".join(bytes(rng.randrange(256) for _ in range(chain)) + bytes(stride - chain)
                               for _ in range(faces))
            data = B.build_tex("ps3", fmt, w, h, payload, mip_max=mip_max, faces=faces,
                               frames=[(0, 0.0, 0.0, 1.0, 1.0)], image_ids=[0] * (6 if faces == 6 else 1))
            tex = parse(data, "ps3")
            self.assertEqual(W.rebuild_payload(tex), tex.payload)
            self.assertEqual(W.ps3_payload(fmt, w, h, tex.levels, W.texture_elements(tex), faces), payload)

    def test_volume_rebuild(self):
        tex = parse(volume_file("ps3")[1], "ps3")
        self.assertEqual(W.rebuild_payload(tex), tex.payload)


# --- W14-10 -----------------------------------------------------------------------------------

def check_container(test, payload, fmt, w, h, depth, faces, levels):
    lay = W.switch_layout(fmt, w, h, levels, faces, depth)
    d = lay.image_size
    test.assertEqual(len(payload), 512 + d + 60)
    test.assertEqual(payload[:16], bytes((0x44, 0x46, 0x76, 0x4E)) + struct.pack("<III", 16, 1, 1))
    test.assertEqual(payload[16:52], bytes((0x48, 0x42, 0x76, 0x4E)) + struct.pack("<IQQIII", 36, 120, 36, 2, 0, 0))
    info = payload[52:172]
    target = 2 if depth == 64 else (8 if faces == 6 else 1)
    test.assertEqual(struct.unpack_from("<Q8I", info, 0),
                     (d, 512, w, h, depth, target, T._NV_FORMATS[fmt], levels, lay.chain))
    offsets = struct.unpack_from("<17I", info, 40)
    test.assertEqual(list(offsets[:levels]), lay.mip_offsets)
    test.assertEqual(offsets[levels:], (0,) * (17 - levels))
    test.assertEqual(struct.unpack_from("<I", info, 108)[0], lay.layout_word)
    test.assertEqual(info[112:], bytes((7, 0, 1, 0, 0, 0, 0, 0)))
    test.assertEqual(payload[172:208], bytes((0x48, 0x42, 0x76, 0x4E)) + struct.pack("<IQQIII", 36, d, 340, 3, 1, 0))
    test.assertEqual(payload[208:512], bytes(304))
    tail = payload[512 + d:]
    test.assertEqual(tail, bytes((0x48, 0x42, 0x76, 0x4E)) + struct.pack("<IQQIII", 36, 24, 36, 5, 2, 0)
                     + bytes(8) + struct.pack("<II", 1, 1) + bytes(8))
    test.assertEqual(d, faces * lay.chain)
    return lay


class SwitchContainer(unittest.TestCase):
    def test_2d_levels(self):
        tex = parse(W.encode_texture(smooth_image(64, 64), "switch", 12), "switch")
        lay = check_container(self, tex.payload, 12, 64, 64, 1, 1, 5)
        # 16 x 16 blocks: n = 24 -> block height 2; layout word 1
        self.assertEqual(lay.layout_word, 1)
        self.assertEqual(lay.block_heights, [2, 1, 1, 1, 1])
        sizes = [T._surface_bytes(wb, wb, 16, bh) for wb, bh in ((16, 2), (8, 1), (4, 1), (2, 1), (1, 1))]
        self.assertEqual(lay.level_sizes, sizes)
        self.assertEqual(lay.mip_offsets, [sum(sizes[:m]) for m in range(5)])

    def test_tall_texture_block_height(self):
        lay = W.switch_layout(0, 8, 256, 9)
        self.assertEqual(lay.layout_word, 4)  # n = 384 -> 16
        self.assertEqual(lay.block_heights[:5], [16, 16, 8, 4, 2])

    def test_cube_and_volume(self):
        tex = parse(W.encode_texture([exact_image(0, 16, 16, f) for f in range(6)], "switch", 0, faces=6), "switch")
        check_container(self, tex.payload, 0, 16, 16, 1, 6, 5)
        self.assertEqual(tex.nv.slice_size * 6, tex.nv.data_size)
        vol = parse(volume_file("switch")[1], "switch")
        lay = check_container(self, vol.payload, 0, 64, 64, 64, 1, 1)
        self.assertEqual((lay.layout_word, lay.chain, lay.image_size, lay.mip_offsets), (0x40, 64 ** 3 * 4, 64 ** 3 * 4, [0]))

    def test_rebuild_from_parsed(self):
        for fmt, faces in ((9, 1), (11, 1), (0, 6), (4, 1), (23, 6), (43, 1)):
            imgs = [smooth_image(32, 16)] * faces
            tex = parse(W.encode_texture(imgs, "switch", fmt, faces=faces), "switch")
            self.assertEqual(W.rebuild_payload(tex), tex.payload, fmt)
        vol = parse(volume_file("switch")[1], "switch")
        self.assertEqual(W.rebuild_payload(vol), vol.payload)

    def test_image_data_of_builder_file(self):
        # The builder places level 0 with the description's address formula; the standard block height
        rng = random.Random(9)
        for fmt, w, h in ((0, 16, 40), (12, 64, 64), (7, 32, 8), (9, 128, 128)):
            wb, hb = B.elements_of(fmt, w, h)
            layout = T.standard_block_height(hb).bit_length() - 1
            linear = bytes(rng.randrange(256) for _ in range(wb * hb * B.BPP[fmt]))
            tex = parse(B.build_tex("switch", fmt, w, h, linear, layout=layout,
                                    frames=[(0, 0.0, 0.0, 1.0, 1.0)]), "switch")
            image = W.switch_image(fmt, w, h, 1, W.texture_elements(tex))
            self.assertEqual(image, tex.nv.image[:len(image)], fmt)
            self.assertEqual(W.texture_elements(tex)[(0, 0)], linear)


# --- W14-11 -----------------------------------------------------------------------------------

class X360Payload(unittest.TestCase):
    def test_is_build_payload(self):
        for fmt in (0, 4, 7, 9, 11, 12):
            image = smooth_image(64, 32)
            data = W.encode_texture(image, "x360", fmt, levels=4)
            tex = parse(data, "x360")
            chain = [image]
            while len(chain) < 4:
                chain.append(W.downsample(chain[-1]))
            elements = {(m, 0): W.element_form(chain[m], fmt, "x360") for m in range(4)}
            self.assertEqual(tex.payload, XT.build_payload(fmt, 64, 32, 4, elements), fmt)
            self.assertEqual(len(tex.payload), XT.layout(fmt, 64, 32, 4)[1])
            self.assertEqual(XT.payload_problems(tex), [])

    def test_rebuild_from_parsed(self):
        for fmt, faces in ((10, 1), (0, 6), (12, 6), (23, 1)):
            tex = parse(W.encode_texture([smooth_image(32, 32)] * faces, "x360", fmt, faces=faces), "x360")
            self.assertEqual(W.rebuild_payload(tex), tex.payload, fmt)
        vol = parse(volume_file("x360")[1], "x360")
        self.assertEqual(len(vol.payload), XT.VOLUME_BYTES)
        self.assertEqual(W.rebuild_payload(vol), vol.payload)

    def test_swap(self):
        # DXT and ARGB1555 words are byte-swapped in the stored payload
        image = exact_image(4, 32, 32)
        tex = parse(W.encode_texture(image, "x360", 4, levels=1), "x360")
        std = W.element_form(image, 4, "x360")
        first = XT.element_offsets(tex)[0]
        self.assertEqual(tex.payload[first:first + 2], std[1::-1])


class Refusals(unittest.TestCase):
    def test_bad_inputs(self):
        with self.assertRaises(FormatError):
            W.encode_texture(exact_image(0, 8, 8), "pc", 0)
        with self.assertRaises(FormatError):
            W.encode_texture(exact_image(0, 8, 8), "ps3", 99)
        with self.assertRaises(FormatError):
            W.encode_texture([exact_image(0, 8, 8)] * 5, "ps3", 0, faces=6)
        with self.assertRaises(FormatError):
            W.encode_texture([exact_image(0, 32, 32)] * 64, "ps3", 0, depth=64)
        with self.assertRaises(FormatError):
            W.encode_texture([exact_image(0, 8, 8), exact_image(0, 4, 4)] + [exact_image(0, 8, 8)] * 4,
                             "switch", 0, faces=6)


if __name__ == "__main__":
    unittest.main()
