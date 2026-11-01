import struct
import unittest
import zlib

import builders as B
from d3sno import FormatError
from d3sno import texture as T
from d3sno.index import Sprite
from d3sno.sno import hash33


def pixels(image):
    rgba = image.rgba
    return [tuple(rgba[i:i + 4]) for i in range(0, len(rgba), 4)]


def dxt1_block(c0, c1, indices=0):
    return struct.pack("<HHI", c0, c1, indices)


RED, BLUE = 0xF800, 0x001F


class Header(unittest.TestCase):
    def test_fields(self):
        for build in ("ps3", "switch"):
            with self.subTest(build):
                frames = [(7, 0.0, 0.0, 1.0, 1.0)]
                tex = T.parse_texture(B.build_tex(build, 0, 2, 2, bytes(16), frames=frames,
                                                  v_scale=0.5))
                self.assertEqual(tex.build, build)
                self.assertEqual((tex.format, tex.width, tex.height), (0, 2, 2))
                self.assertEqual((tex.faces, tex.depth, tex.levels), (1, 1, 1))
                self.assertEqual(tex.frames[0].hash, 7)
                self.assertEqual(tex.image_ids, [0])
                self.assertEqual(tex.lightmap_v_scale, 0.5)

    def test_level_sizes(self):
        tex = T.parse_texture(B.build_tex("ps3", 0, 8, 3, bytes(8 * 3 * 4 + 4 * 1 * 4 + 2 * 4 + 4),
                                          mip_max=3))
        self.assertEqual(tex.levels, 4)
        self.assertEqual([tex.level_size(m) for m in range(4)], [(8, 3), (4, 1), (2, 1), (1, 1)])

    def test_version_picks_the_build(self):
        data = B.build_tex("ps3", 0, 2, 2, bytes(16), version=66)
        with self.assertRaises(FormatError):
            T.parse_texture(data)

    def test_version_strict(self):
        data = B.build_tex("ps3", 0, 2, 2, bytes(16), version=59)
        self.assertIn("not this build", T.parse_texture(data).warnings[0])
        with self.assertRaises(FormatError):
            T.parse_texture(data, strict=True)

    def test_faces(self):
        with self.assertRaises(FormatError):
            T.parse_texture(B.build_tex("ps3", 0, 2, 2, bytes(16), faces=2))

    def test_depth(self):
        with self.assertRaises(FormatError):
            T.parse_texture(B.build_tex("ps3", 0, 2, 2, bytes(16), depth=2))

    def test_zero_size(self):
        with self.assertRaises(FormatError):
            T.parse_texture(B.build_tex("ps3", 0, 0, 2, bytes(16)))

    def test_mip_range(self):
        with self.assertRaises(FormatError):
            T.parse_texture(B.build_tex("ps3", 0, 2, 2, bytes(16), mip_max=-1))
        with self.assertRaises(FormatError):
            T.parse_texture(B.build_tex("ps3", 0, 2, 2, bytes(16), mip_max=17))

    def test_second_pair_warns(self):
        data = B.build_tex("ps3", 0, 2, 2, bytes(16), extra_pair=True)
        self.assertTrue(T.parse_texture(data).warnings)
        with self.assertRaises(FormatError):
            T.parse_texture(data, strict=True)

    def test_payload_ends_the_file(self):
        with self.assertRaises(FormatError):
            T.parse_texture(B.build_tex("ps3", 0, 2, 2, bytes(16), trailing=b"\0"))

    def test_payload_out_of_bounds(self):
        data = bytearray(B.build_tex("ps3", 0, 2, 2, bytes(16)))
        struct.pack_into(">I", data, 0x44, 0xFFFFFF)
        with self.assertRaises(FormatError):
            T.parse_texture(bytes(data))

    def test_frame_pair_size(self):
        data = bytearray(B.build_tex("ps3", 0, 2, 2, bytes(16), frames=[(1, 0, 0, 1, 1)]))
        struct.pack_into(">I", data, 0x220, 2)
        with self.assertRaises(FormatError):
            T.parse_texture(bytes(data))

    def test_image_id_size(self):
        data = bytearray(B.build_tex("ps3", 0, 2, 2, bytes(16)))
        struct.pack_into(">I", data, 0x250, 3)
        with self.assertRaises(FormatError):
            T.parse_texture(bytes(data))

    def test_extra_or_too_few_image_ids_are_fine(self):
        frames = [(1, 0, 0, 1, 1)]
        extra = T.parse_texture(B.build_tex("ps3", 0, 2, 2, bytes(16), frames=frames,
                                            image_ids=[1, 2, 3]))
        self.assertEqual((extra.image_ids, extra.warnings), ([1, 2, 3], []))
        few = T.parse_texture(B.build_tex("ps3", 0, 2, 2, bytes(16), frames=frames, image_ids=[]))
        self.assertEqual((few.image_ids, few.warnings), ([], []))


class Dxt(unittest.TestCase):
    def decode(self, fmt, block, w=4, h=4):
        return T.decode_image(T.parse_texture(B.build_tex("ps3", fmt, w, h, block)))

    def test_four_colour_mode(self):
        bits = 0b11_10_01_00
        image = pixels(self.decode(9, dxt1_block(RED, BLUE, bits)))
        self.assertEqual(image[0], (255, 0, 0, 255))
        self.assertEqual(image[1], (0, 0, 255, 255))
        self.assertEqual(image[2], (170, 0, 85, 255))
        self.assertEqual(image[3], (85, 0, 170, 255))

    def test_three_colour_mode(self):
        bits = 0b11_10_01_00
        image = pixels(self.decode(9, dxt1_block(BLUE, RED, bits)))
        self.assertEqual(image[2], (127, 0, 127, 255))
        self.assertEqual(image[3], (0, 0, 0, 0))

    def test_equal_colours_are_three_colour_mode(self):
        image = pixels(self.decode(9, dxt1_block(RED, RED, 3)))
        self.assertEqual(image[0], (0, 0, 0, 0))

    def test_format_10_decodes_like_9(self):
        block = dxt1_block(BLUE, RED, 0b11_10_01_00)
        self.assertEqual(self.decode(10, block), self.decode(9, block))

    def test_dxt3_alpha(self):
        alpha = struct.pack("<Q", 0xF | (0x8 << 4))
        image = pixels(self.decode(11, alpha + dxt1_block(BLUE, RED, 0b01)))
        self.assertEqual(image[0], (255, 0, 0, 255))
        self.assertEqual(image[1], (0, 0, 255, 0x88))
        self.assertEqual(image[2], (0, 0, 255, 0))

    def test_dxt3_colour_is_always_four_colour(self):
        image = pixels(self.decode(11, bytes(8) + dxt1_block(BLUE, RED, 0b10)))
        self.assertEqual(image[0], (85, 0, 170, 0))

    def test_dxt5_eight_alpha_mode(self):
        codes = (0 | 1 << 3 | 2 << 6 | 7 << 9).to_bytes(6, "little")
        block = bytes((210, 70)) + codes + dxt1_block(RED, BLUE)
        alphas = [p[3] for p in pixels(self.decode(12, block))]
        self.assertEqual(alphas[:4], [210, 70, 190, 90])

    def test_dxt5_six_alpha_mode(self):
        codes = (2 | 5 << 3 | 6 << 6 | 7 << 9).to_bytes(6, "little")
        block = bytes((50, 100)) + codes + dxt1_block(RED, BLUE)
        alphas = [p[3] for p in pixels(self.decode(12, block))]
        self.assertEqual(alphas[:4], [60, 90, 0, 255])

    def test_format_43_is_dxt5(self):
        block = bytes((210, 70)) + bytes(6) + dxt1_block(RED, BLUE)
        self.assertEqual(self.decode(43, block), self.decode(12, block))

    def test_normal_is_alpha_then_green(self):
        block = bytes((200, 100)) + bytes(6) + dxt1_block(0x0400, 0)
        tex = T.parse_texture(B.build_tex("ps3", 43, 4, 4, block))
        full = pixels(T.decode_image(tex))[0]
        normal = pixels(T.decode_normal(tex))[0]
        self.assertEqual(normal, (full[3], full[1], 0, 255))
        self.assertEqual(normal[0], 200)

    def test_normal_needs_format_43(self):
        tex = T.parse_texture(B.build_tex("ps3", 12, 4, 4, bytes(16)))
        with self.assertRaises(FormatError):
            T.decode_normal(tex)

    def test_small_surface_crops_the_block(self):
        image = self.decode(9, dxt1_block(RED, BLUE, 0b0100), 2, 1)
        self.assertEqual(pixels(image), [(255, 0, 0, 255), (0, 0, 255, 255)])

    def test_blocks_in_row_order(self):
        data = dxt1_block(RED, BLUE) + dxt1_block(BLUE, 0)
        image = pixels(self.decode(9, data, 8, 4))
        self.assertEqual(image[0], (255, 0, 0, 255))
        self.assertEqual(image[4], (0, 0, 255, 255))


class Ps3Payload(unittest.TestCase):
    def decode(self, fmt, data, w=1, h=1, **kw):
        return T.decode_image(T.parse_texture(B.build_tex("ps3", fmt, w, h, data)), **kw)

    def test_argb8888(self):
        self.assertEqual(pixels(self.decode(0, bytes((4, 1, 2, 3)))), [(1, 2, 3, 4)])

    def test_argb1555_big_endian(self):
        word = 1 << 15 | 31 << 10 | 0 << 5 | 8
        self.assertEqual(pixels(self.decode(4, struct.pack(">H", word))), [(255, 0, 66, 255)])
        self.assertEqual(pixels(self.decode(4, struct.pack(">H", 0))), [(0, 0, 0, 0)])

    def test_l8_and_a8(self):
        self.assertEqual(pixels(self.decode(7, b"\x40")), [(0x40, 0x40, 0x40, 255)])
        self.assertEqual(pixels(self.decode(23, b"\x40")), [(0, 0, 0, 0x40)])

    def test_top_row_stays_first(self):
        image = self.decode(7, b"\x01\x02", 1, 2)
        self.assertEqual(pixels(image)[0][0], 1)

    def test_levels_are_tightly_packed(self):
        data = bytes(range(4)) + b"\x7F"
        tex = T.parse_texture(B.build_tex("ps3", 7, 2, 2, data, mip_max=1))
        self.assertEqual(pixels(T.decode_image(tex, 1)), [(0x7F, 0x7F, 0x7F, 255)])

    def test_dxt_level_size(self):
        data = dxt1_block(RED, BLUE) * 4 + dxt1_block(BLUE, 0)
        tex = T.parse_texture(B.build_tex("ps3", 9, 5, 5, data, mip_max=1))
        self.assertEqual(pixels(T.decode_image(tex, 1))[0], (0, 0, 255, 255))

    def test_level_past_payload(self):
        tex = T.parse_texture(B.build_tex("ps3", 7, 2, 2, bytes(4), mip_max=1))
        with self.assertRaises(FormatError):
            T.decode_image(tex, 1)

    def test_cube_stride_rounds_chain_to_128(self):
        chain = 64  # 4 x 4 ARGB8888
        faces = b"".join(bytes((f, f, f, f)) * 16 + bytes(128 - chain) for f in range(6))
        tex = T.parse_texture(B.build_tex("ps3", 0, 4, 4, faces, faces=6))
        for face in range(6):
            self.assertEqual(pixels(T.decode_image(tex, face=face))[0], (face,) * 4)

    def test_cube_each_face_has_its_chain(self):
        size = 4 + 1
        stride = 128
        face = bytes((1, 1, 1, 1)) + b"\x09"
        data = (face + bytes(stride - size)) * 5 + face + bytes(3)
        tex = T.parse_texture(B.build_tex("ps3", 7, 2, 2, data, faces=6, mip_max=1))
        self.assertEqual(pixels(T.decode_image(tex, 1, 5))[0][0], 9)

    def morton(self, x, y, z):
        return x & 1 | (y & 1) << 1 | (z & 1) << 2 | (z >> 1) << 3

    def test_volume_morton(self):
        data = bytearray(2 * 2 * 64 * 4)
        for z in range(64):
            for y in range(2):
                for x in range(2):
                    at = self.morton(x, y, z) * 4
                    data[at:at + 4] = bytes((200, x, y, z))
        tex = T.parse_texture(B.build_tex("ps3", 0, 2, 2, bytes(data), depth=64))
        image = pixels(T.decode_image(tex, z=37))
        self.assertEqual(image[0], (0, 0, 37, 200))
        self.assertEqual(image[1], (1, 0, 37, 200))
        self.assertEqual(image[2], (0, 1, 37, 200))
        self.assertEqual(image[3], (1, 1, 37, 200))

    def test_volume_needs_argb8888(self):
        tex = T.parse_texture(B.build_tex("ps3", 7, 2, 2, bytes(4 * 64), depth=64))
        with self.assertRaises(FormatError):
            T.decode_image(tex)


class SwitchSwizzle(unittest.TestCase):
    def gradient(self, wb, hb, bpp):
        out = bytearray()
        for y in range(hb):
            for x in range(wb):
                out += bytes(((x + y * wb) >> (8 * i)) & 0xFF for i in range(bpp))
        return bytes(out)

    def test_address_examples_from_the_formula(self):
        self.assertEqual(B.gob_address(0, 0, 64, 4, 1), 0)
        self.assertEqual(B.gob_address(1, 0, 64, 4, 1), 4)
        self.assertEqual(B.gob_address(4, 0, 64, 4, 1), 32)
        self.assertEqual(B.gob_address(8, 0, 64, 4, 1), 256)
        self.assertEqual(B.gob_address(0, 1, 64, 4, 1), 16)
        self.assertEqual(B.gob_address(0, 2, 64, 4, 1), 64)
        self.assertEqual(B.gob_address(16, 0, 64, 4, 1), 512)

    def test_block_heights(self):
        for w, h, layout in ((16, 16, 0), (32, 32, 1), (20, 40, 2), (70, 9, 0)):
            with self.subTest((w, h, layout)):
                linear = self.gradient(w, h, 4)
                tex = T.parse_texture(B.build_tex("switch", 0, w, h, linear, layout=layout))
                self.assertEqual(T.decode_image(tex).rgba, linear)

    def test_layout_comes_from_the_header(self):
        """A tall surface whose block height was not chosen by the usual height rule."""
        linear = self.gradient(8, 64, 4)
        for layout in (0, 1, 3):
            tex = T.parse_texture(B.build_tex("switch", 0, 8, 64, linear, layout=layout))
            self.assertEqual(T.decode_image(tex).rgba, linear)
            self.assertEqual(any("standard rule" in w for w in tex.warnings), layout != 3)

    def test_dxt_elements(self):
        wb, hb = 8, 8
        linear = b"".join(
            dxt1_block(RED if (x + y) % 2 else BLUE, 0) for y in range(hb) for x in range(wb)
        )
        tex = T.parse_texture(B.build_tex("switch", 9, 32, 32, linear, layout=1))
        image = pixels(T.decode_image(tex))
        self.assertEqual(image[0], (0, 0, 255, 255))
        self.assertEqual(image[4], (255, 0, 0, 255))
        self.assertEqual(image[32 * 4], (255, 0, 0, 255))

    def test_bytes_per_element(self):
        for fmt, bpp in ((7, 1), (4, 2)):
            linear = self.gradient(40, 20, bpp)
            tex = T.parse_texture(B.build_tex("switch", fmt, 40, 20, linear))
            self.assertEqual(len(T.decode_image(tex).rgba), 40 * 20 * 4)

    def test_channels(self):
        tex = T.parse_texture(B.build_tex("switch", 0, 1, 1, bytes((1, 2, 3, 4))))
        self.assertEqual(pixels(T.decode_image(tex)), [(1, 2, 3, 4)])

    def test_argb1555_little_endian(self):
        word = 1 << 15 | 8 << 10 | 0 << 5 | 31
        tex = T.parse_texture(B.build_tex("switch", 4, 1, 1, struct.pack("<H", word)))
        self.assertEqual(pixels(T.decode_image(tex)), [(255, 0, 66, 255)])

    def test_standard_block_height(self):
        cases = (
            (1, 1), (10, 1), (11, 2), (21, 2), (22, 4), (42, 4), (43, 8), (85, 8), (86, 16),
            (4096, 16),
        )
        for hb, bh in cases:
            self.assertEqual(T.standard_block_height(hb), bh, hb)

    def test_height_just_above_a_power_of_two_differs_from_the_plain_rule(self):
        # The smallest bh with 8 * bh >= 17 is 4. Here n is 25, which gives 2.
        self.assertEqual(T.standard_block_height(17), 2)

    def test_matching_block_height_gives_no_warning(self):
        linear = self.gradient(32, 32, 4)
        tex = T.parse_texture(B.build_tex("switch", 0, 32, 32, linear, layout=2))
        self.assertEqual(tex.warnings, [])

    def test_block_height_uses_elements_not_texels(self):
        # 32 texels are 8 elements, so n is 12 and the rule gives 1. Texels would give 4.
        linear = bytes(8 * 8 * 8)
        tex = T.parse_texture(B.build_tex("switch", 9, 32, 32, linear, layout=0))
        self.assertEqual(tex.warnings, [])

    def test_block_height_mismatch_warns_and_header_wins(self):
        linear = self.gradient(32, 32, 4)
        data = B.build_tex("switch", 0, 32, 32, linear, layout=1)
        tex = T.parse_texture(data)
        self.assertIn("standard rule", tex.warnings[0])
        self.assertEqual(T.decode_image(tex).rgba, linear)
        with self.assertRaises(FormatError):
            T.parse_texture(data, strict=True)

    def test_3d_is_not_held_to_the_rule(self):
        payload = B.nv_container(0, 4, 4, 64, 2, bytes(0x10000), [0], 0x40)
        tex = T.parse_texture(B.build_tex("switch", 0, 4, 4, b"", depth=64, payload=payload))
        self.assertEqual(tex.warnings, [])

    def test_block_height_halves_for_small_levels(self):
        nv = T.NvTexture(0, 512, 0, 0, 0, 1, 37, 1, 0, (), 3, b"")
        self.assertEqual(T._level_block_height(nv, 64), 8)
        self.assertEqual(T._level_block_height(nv, 32), 4)
        self.assertEqual(T._level_block_height(nv, 16), 2)
        self.assertEqual(T._level_block_height(nv, 8), 1)
        self.assertEqual(T._level_block_height(nv, 1), 1)

    def mip_file(self, layout=2):
        w, h = 32, 32
        top = self.gradient(w, h, 4)
        second = self.gradient(w // 2, h // 2, 4)
        bh0 = T._level_block_height(T.NvTexture(0, 0, 0, 0, 0, 0, 0, 0, 0, (), layout, b""), h)
        bh1 = T._level_block_height(T.NvTexture(0, 0, 0, 0, 0, 0, 0, 0, 0, (), layout, b""), h // 2)
        level0 = B.swizzle(top, w, h, 4, bh0)
        level1 = B.swizzle(second, w // 2, h // 2, 4, bh1)
        payload = B.nv_container(0, w, h, 1, 1, level0 + level1, [0, len(level0)], layout)
        return B.build_tex("switch", 0, w, h, b"", mip_max=1, payload=payload), top, second

    def test_mips_use_header_offsets(self):
        data, top, second = self.mip_file()
        tex = T.parse_texture(data)
        self.assertEqual(tex.warnings, [])
        self.assertEqual(T.decode_image(tex, 0).rgba, top)
        self.assertEqual(T.decode_image(tex, 1).rgba, second)

    def test_mip_offsets_are_checked(self):
        linear = bytes(32 * 32 * 4)
        image = B.swizzle(linear, 32, 32, 4, 1) * 2
        payload = B.nv_container(0, 32, 32, 1, 1, image, [0, 100], 0)
        tex = T.parse_texture(B.build_tex("switch", 0, 32, 32, b"", mip_max=1, payload=payload))
        self.assertTrue(any("overlaps" in w for w in tex.warnings))

    def test_cube_uses_slice_size(self):
        faces = [self.gradient(16, 16, 4)[::-1] if f % 2 else self.gradient(16, 16, 4) for f in range(6)]
        images = [B.swizzle(face, 16, 16, 4, 1) for face in faces]
        size = len(images[0]) + 0x200
        image = b"".join(i.ljust(size, b"\0") for i in images)
        payload = B.nv_container(0, 16, 16, 1, 8, image, [0], 0, slice_size=size)
        tex = T.parse_texture(B.build_tex("switch", 0, 16, 16, b"", faces=6, payload=payload))
        self.assertEqual(T.decode_image(tex, face=3).rgba, faces[3])
        self.assertEqual(T.decode_image(tex, face=4).rgba, faces[4])

    def test_3d(self):
        w, h, depth = 8, 4, 64
        image = bytearray(B.address_3d(7, 3, 63, w, h, 4) + 4)
        for z in range(depth):
            for y in range(h):
                for x in range(w):
                    at = B.address_3d(x, y, z, w, h, 4)
                    image[at:at + 4] = bytes((x, y, z, 9))
        payload = B.nv_container(0, w, h, depth, 2, bytes(image), [0], 0x40)
        tex = T.parse_texture(B.build_tex("switch", 0, w, h, b"", depth=depth, payload=payload))
        for z in (0, 15, 16, 63):
            image_out = pixels(T.decode_image(tex, z=z))
            self.assertEqual(image_out[0], (0, 0, z, 9))
            self.assertEqual(image_out[3 * w + 7], (7, 3, z, 9))

    def test_3d_layout_word(self):
        payload = B.nv_container(0, 4, 4, 64, 2, bytes(0x10000), [0], 1)
        with self.assertRaises(FormatError):
            T.parse_texture(B.build_tex("switch", 0, 4, 4, b"", depth=64, payload=payload))


class Container(unittest.TestCase):
    def build(self, **container):
        return B.build_tex("switch", 0, 1, 1, bytes(4), container=container)

    def test_info_fields(self):
        nv = T.parse_texture(self.build()).nv
        self.assertEqual((nv.alignment, nv.width, nv.height, nv.depth), (512, 1, 1, 1))
        self.assertEqual((nv.target, nv.pixel_format, nv.mip_count), (1, 37, 1))
        self.assertEqual(nv.data_size, len(nv.image))
        self.assertEqual(len(nv.mip_offsets), 17)

    def test_blocks_follow_their_data(self):
        tex = T.parse_texture(self.build())
        self.assertEqual(pixels(T.decode_image(tex)), [(0, 0, 0, 0)])
        self.assertEqual(tex.nv.image_offset, 16 + 36 + 120 + 36)

    def test_next_block_comes_from_the_info_data_size(self):
        payload = bytearray(B.nv_container(0, 1, 1, 1, 1, bytes(512), [0], 0))
        struct.pack_into("<Q", payload, 16 + 36, 500)
        with self.assertRaises(FormatError):
            T.parse_texture(B.build_tex("switch", 0, 1, 1, b"", payload=bytes(payload)))

    def test_unknown_block_type_is_fatal(self):
        with self.assertRaises(FormatError):
            T.parse_texture(self.build(kinds=(2, 3, 7)))

    def test_magic_bytes_on_disk(self):
        payload = B.nv_container(0, 1, 1, 1, 1, bytes(512), [0], 0)
        self.assertEqual(payload[:4], bytes.fromhex("4446764E"))
        self.assertEqual(payload[16:20], bytes.fromhex("4842764E"))

    def test_magic_words_are_not_read_as_ascii_names(self):
        with self.assertRaises(FormatError):
            T.parse_texture(self.build(magic=b"NvFD"))
        with self.assertRaises(FormatError):
            T.parse_texture(self.build(block_magic=b"NvBH"))

    def test_file_magic(self):
        with self.assertRaises(FormatError):
            T.parse_texture(self.build(magic=b"XXXX"))

    def test_block_magic(self):
        with self.assertRaises(FormatError):
            T.parse_texture(self.build(block_magic=b"XXXX"))

    def test_block_sequence(self):
        with self.assertRaises(FormatError):
            T.parse_texture(self.build(kinds=(2, 5, 3)))

    def test_info_size(self):
        with self.assertRaises(FormatError):
            T.parse_texture(self.build(info_size=112))

    def test_target(self):
        payload = B.nv_container(0, 1, 1, 1, 4, bytes(512), [0], 0)
        with self.assertRaises(FormatError):
            T.parse_texture(B.build_tex("switch", 0, 1, 1, b"", payload=payload))

    def test_format_mismatch_is_fatal(self):
        linear = bytes(32)
        data = B.build_tex("switch", 9, 8, 8, linear)
        tex = T.parse_texture(data)
        self.assertEqual(tex.warnings, [])
        payload = B.nv_container(0, 8, 8, 1, 1, bytes(512), [0], 0)
        with self.assertRaises(FormatError):
            T.parse_texture(B.build_tex("switch", 9, 8, 8, b"", payload=payload))

    def test_size_mismatch_is_fatal(self):
        for size in ((2, 1, 1), (1, 2, 1), (1, 1, 64)):
            payload = B.nv_container(0, *size, 1, bytes(512), [0], 0)
            with self.assertRaises(FormatError):
                T.parse_texture(B.build_tex("switch", 0, 1, 1, b"", payload=payload))

    def test_mip_count_mismatch_warns(self):
        payload = B.nv_container(0, 1, 1, 1, 1, bytes(512), [0], 0, mip_count=3)
        tex = T.parse_texture(B.build_tex("switch", 0, 1, 1, b"", payload=payload))
        self.assertTrue(any("3 mips" in w for w in tex.warnings))

    def test_image_too_small(self):
        payload = B.nv_container(0, 64, 64, 1, 1, bytes(512), [0], 0)
        tex = T.parse_texture(B.build_tex("switch", 0, 64, 64, b"", payload=payload))
        with self.assertRaises(FormatError):
            T.decode_image(tex)

    def test_address_past_image(self):
        payload = B.nv_container(0, 4, 4, 1, 1, bytes(512), [500], 0)
        tex = T.parse_texture(B.build_tex("switch", 0, 4, 4, b"", payload=payload))
        with self.assertRaises(FormatError):
            T.decode_image(tex)


class Decoding(unittest.TestCase):
    def test_unknown_format(self):
        tex = T.parse_texture(B.build_tex("ps3", 99, 1, 1, bytes(4)))
        with self.assertRaises(FormatError):
            T.decode_image(tex)

    def test_no_payload(self):
        tex = T.parse_texture(B.build_tex("ps3", 0, 1, 1, b""))
        with self.assertRaises(FormatError):
            T.decode_image(tex)

    def test_bad_arguments(self):
        tex = T.parse_texture(B.build_tex("ps3", 0, 1, 1, bytes(4)))
        for kw in ({"level": 1}, {"face": 1}, {"z": 1}, {"level": -1}):
            with self.assertRaises(FormatError):
                T.decode_image(tex, **kw)

    def test_all_levels(self):
        tex = T.parse_texture(B.build_tex("ps3", 7, 2, 2, bytes(5), mip_max=1))
        self.assertEqual([(i.width, i.height) for i in T.decode_levels(tex)], [(2, 2), (1, 1)])

    def test_surfaces_and_strip(self):
        faces = (bytes(16) + bytes(112)) * 6
        tex = T.parse_texture(B.build_tex("ps3", 0, 2, 2, faces, faces=6))
        surfaces = T.decode_surfaces(tex)
        self.assertEqual(len(surfaces), 6)
        strip = T.vertical_strip(surfaces)
        self.assertEqual((strip.width, strip.height), (2, 12))

    def test_strip_needs_one_width(self):
        with self.assertRaises(FormatError):
            T.vertical_strip([T.Image(1, 1, bytes(4)), T.Image(2, 1, bytes(8))])
        with self.assertRaises(FormatError):
            T.vertical_strip([])


class Frames(unittest.TestCase):
    def tex(self, frames):
        # A, R, G, B on ps3, so R holds x and G holds y.
        data = b"".join(bytes((255, x, y, 0)) for y in range(8) for x in range(8))
        return T.parse_texture(B.build_tex("ps3", 0, 8, 8, data, frames=frames))

    def test_rect_truncates(self):
        frame = T.TextureFrame(0, 0.13, 0.25, 0.5, 0.99)
        self.assertEqual(frame.rect(8, 8), (1, 2, 4, 7))

    def test_rect_is_single_precision(self):
        u0 = struct.unpack("<f", struct.pack("<f", 0.7))[0]
        self.assertEqual(int(u0 * 10), 6)
        self.assertEqual(T.TextureFrame(0, u0, 0.0, 1.0, 1.0).rect(10, 10)[0], 7)

    def test_rect_uses_the_header_size_on_a_lower_level(self):
        data = bytes(8 * 8 + 4 * 4)
        tex = T.parse_texture(B.build_tex(
            "ps3", 7, 8, 8, data, mip_max=1, frames=[(1, 0.25, 0.0, 0.75, 1.0)],
        ))
        image = T.frame_image(tex, 0, 1)
        self.assertEqual((image.width, image.height), (2, 4))

    def test_crop(self):
        tex = self.tex([(1, 0.25, 0.5, 0.75, 1.0)])
        image = T.frame_image(tex, 0)
        self.assertEqual((image.width, image.height), (4, 4))
        self.assertEqual(pixels(image)[0], (2, 4, 0, 255))
        self.assertEqual(pixels(image)[-1], (5, 7, 0, 255))

    def test_no_such_frame(self):
        with self.assertRaises(FormatError):
            T.frame_image(self.tex([]), 0)

    def test_rect_outside_image(self):
        with self.assertRaises(FormatError):
            T.frame_image(self.tex([(1, 0.0, 0.0, 2.0, 1.0)]), 0)

    def test_empty_rect(self):
        with self.assertRaises(FormatError):
            T.frame_image(self.tex([(1, 0.5, 0.0, 0.5, 1.0)]), 0)

    def test_nan_rect(self):
        with self.assertRaises(FormatError):
            T.frame_image(self.tex([(1, float("nan"), 0.0, 1.0, 1.0)]), 0)

    def test_sprite(self):
        tex = self.tex([(0, 0.0, 0.0, 1.0, 1.0), (hash33("Icon"), 0.0, 0.0, 0.5, 0.5)])
        image = T.sprite_image(tex, Sprite(hash33("Icon"), 5, 1))
        self.assertEqual((image.width, image.height), (4, 4))

    def test_sprite_hash_must_match(self):
        tex = self.tex([(1, 0.0, 0.0, 1.0, 1.0)])
        with self.assertRaises(FormatError):
            T.sprite_image(tex, Sprite(2, 5, 0))

    def test_sprite_frame_must_exist(self):
        with self.assertRaises(FormatError):
            T.sprite_image(self.tex([]), Sprite(2, 5, 0))


class Png(unittest.TestCase):
    def chunks(self, data):
        self.assertEqual(data[:8], b"\x89PNG\r\n\x1a\n")
        pos = 8
        found = []
        while pos < len(data):
            length = struct.unpack_from(">I", data, pos)[0]
            kind = data[pos + 4:pos + 8]
            body = data[pos + 8:pos + 8 + length]
            crc = struct.unpack_from(">I", data, pos + 8 + length)[0]
            self.assertEqual(crc, zlib.crc32(kind + body))
            found.append((kind, body))
            pos += 12 + length
        return found

    def test_structure(self):
        image = T.Image(2, 2, bytes(range(16)))
        chunks = self.chunks(T.encode_png(image))
        self.assertEqual([c[0] for c in chunks], [b"IHDR", b"IDAT", b"IEND"])
        self.assertEqual(struct.unpack(">IIBBBBB", chunks[0][1]), (2, 2, 8, 6, 0, 0, 0))
        rows = zlib.decompress(chunks[1][1])
        self.assertEqual(rows, b"\0" + bytes(range(8)) + b"\0" + bytes(range(8, 16)))

    def test_no_gamma_chunk(self):
        chunks = self.chunks(T.encode_png(T.Image(1, 1, bytes(4))))
        self.assertNotIn(b"gAMA", [c[0] for c in chunks])
