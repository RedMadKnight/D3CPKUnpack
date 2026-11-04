import random
import struct
import unittest

import builders as B
from d3sno import FormatError
from d3sno import presentation as V


class Music(unittest.TestCase):
    def build(self, entries, count=None):
        img = B.ps3_file(0x40)
        raw = b"".join(struct.pack(">15I", *entry) for entry in entries)
        img.put(0x2C, "II", *img.append(raw))
        img.put(0x34, "i", len(entries) if count is None else count)
        return img.done()

    def test_entries(self):
        volume = struct.unpack(">I", struct.pack(">f", 0.5))[0]
        entries = V.parse_music(self.build([(7, 0xABC, volume, *range(12))]))
        self.assertEqual(entries[0].words["bank"], 7)
        self.assertEqual(entries[0].words["sample_hash"], 0xABC)
        self.assertEqual(entries[0].as_float("volume"), 0.5)
        self.assertEqual(entries[0].words["word_14"], 11)

    def test_count(self):
        with self.assertRaises(FormatError):
            V.parse_music(self.build([tuple(range(15))], count=2))


class Paths(unittest.TestCase):
    def test_trail(self):
        img = B.ps3_file(0x10 + 0x118)
        for i in range(4):
            B.path_bytes(img, 0x10 + 0x28 + i * 40, [bytes(12)] * (i + 1))
        img.put(0x10 + 0xC8, "ii", 3, 9)
        img.raw(0x10 + 0xD0, B.material_block(img, "ps3", [(1, 55)]))
        trail = V.parse_trail(img.done())
        self.assertEqual([len(p.nodes) for p in trail.paths], [1, 2, 3, 4])
        self.assertEqual((trail.type, trail.segments), (3, 9))
        self.assertEqual(trail.material.texture(1), 55)

    def test_path_node_size(self):
        img = B.ps3_file(0x10 + 0x118)
        B.path_bytes(img, 0x10 + 0x28, [bytes(12)])
        struct.pack_into(">I", img.b, 0x10 + 0x28 + 36, 13)
        img.raw(0x10 + 0xD0, B.material_block(img, "ps3"))
        with self.assertRaises(FormatError):
            V.parse_trail(img.done())

    def test_rope(self):
        img = B.ps3_file(0x3A4)
        img.raw(0x40, B.material_block(img, "ps3", [(1, 66)]))
        for i in range(14):
            B.path_bytes(img, 0x9C + i * 40, [bytes(28 if i == 12 else 12)])
        img.put(0x9C + 14 * 40, "2f", 1.5, 2.5)
        for i in range(5):
            B.path_bytes(img, 0x9C + 14 * 40 + 8 + i * 40, [])
        img.put(0x39C, "II", *img.append(bytes(192 * 2)))
        rope = V.parse_rope(img.done())
        self.assertEqual(rope.material.texture(1), 66)
        self.assertEqual(len(rope.paths[12].nodes[0]), 28)
        self.assertEqual(rope.floats, (1.5, 2.5))
        self.assertEqual((len(rope.tail_paths), len(rope.events)), (5, 2))

    def test_rope_events_pair_past_the_end(self):
        img = B.ps3_file(0x3A4)
        img.raw(0x40, B.material_block(img, "ps3"))
        for i in range(19):
            B.path_bytes(img, 0x9C + i * 40 + (8 if i >= 14 else 0), [])
        img.put(0x39C, "II", 0xFFFF, 192)
        with self.assertRaises(FormatError):
            V.parse_rope(img.done())

    def test_light(self):
        img = B.ps3_file(0x10 + 0x240 + 13 * 40)
        img.put(0x10 + 0x18, "II", *img.append(b"anim"))
        img.put(0x10 + 0x28, "ifif", 77, 2.0, 3, 9.5)
        light = V.parse_light(img.done())
        self.assertEqual(light.animation, b"anim")
        self.assertEqual((light.texture, light.duration, light.priority, light.distance), (77, 2.0, 3, 9.5))
        self.assertEqual([len(s) for s in light.states], [13, 13])


class Ui(unittest.TestCase):
    def test_lists(self):
        img = B.ps3_file(0x50)
        for i, at in enumerate((0x10, 0x20, 0x30)):
            img.put(0x10 + at, "i", i + 1)
            img.put(0x10 + at + 4, "II", *img.append(bytes(24 * (i + 1))))
        lists = V.parse_ui(img.done())
        self.assertEqual([l.count for l in lists], [1, 2, 3])
        self.assertEqual(len(lists[2].split(24)), 3)
        with self.assertRaises(FormatError):
            lists[0].split(16)

    def test_font(self):
        img = B.ps3_file(0x10 + 0xC4)
        img.put(0x10 + 0xAC, "II", *img.append(bytes(80)))
        img.put(0x10 + 0xBC, "II", *img.append(bytes(72)))
        font = V.parse_font(img.done())
        self.assertEqual((len(font.textures), len(font.point_sizes)), (2, 1))
        self.assertEqual(V.glyph_metric_pixels(96), 1.5)

    def test_level_area_slots(self):
        self.assertEqual(len(V.level_area_slots(bytes(0x18 + 26 * 352))), 26)
        with self.assertRaises(FormatError):
            V.level_area_slots(bytes(100))


class Shaders(unittest.TestCase):
    def test_ps3(self):
        records = [bytes(0x10) + struct.pack(">I", 4) + b"\0" * 12 + b"CODE", bytes(0x10) + struct.pack(">I", 2) + b"PX"]
        table = b""
        offset = 0
        for raw in records:
            table += struct.pack(">II", offset, len(raw))
            offset += len(raw)
        passes = V.parse_shader_block(table + b"".join(records), 1, "ps3")
        self.assertEqual((passes[0][0].code, passes[0][1].code), (b"CODE", b"PX"))

    def test_switch(self):
        records = [struct.pack("<4I", 0, 0, 4, 0) + b"VERT", struct.pack("<4I", 0, 0, 2, 0) + b"PX"]
        table = struct.pack("<II", 0, 20) + struct.pack("<II", 20, 18)
        passes = V.parse_shader_block(table + b"".join(records), 1, "switch")
        self.assertEqual((passes[0][0].code, passes[0][1].code), (b"VERT", b"PX"))

    def test_switch_head(self):
        raw = struct.pack("<4I", 1, 0, 4, 0) + b"VERT"
        block = struct.pack("<II", 0, 20) * 2 + raw
        with self.assertRaises(FormatError):
            V.parse_shader_block(block, 1, "switch")

    def test_record_past_the_block(self):
        with self.assertRaises(FormatError):
            V.parse_shader_block(struct.pack("<II", 0, 99) * 2, 1, "switch")


class TextureAnimation(unittest.TestCase):
    def test_identity(self):
        self.assertEqual(V.parse_texture_animation(b"", "ps3").kind, "identity")

    def test_scale(self):
        animation = V.parse_texture_animation(struct.pack("<I2f", 1, 2.0, 3.0), "switch")
        self.assertEqual((animation.kind, animation.scale), ("scale", (2.0, 3.0)))

    def test_scroll_scale_defaults(self):
        animation = V.parse_texture_animation(struct.pack(">I4f", 2, -1.0, 2.0, 0.5, 0.25), "ps3")
        self.assertEqual(animation.scale, (1.0, 2.0))
        self.assertEqual(animation.speed, (0.5, 0.25))

    def test_scroll_offset(self):
        animation = V.parse_texture_animation(struct.pack("<I4f", 2, 1.0, 1.0, 0.01, 0.0), "switch")
        start = random.Random(1).random()
        offset = V.scroll_offset(animation, 1.0, random.Random(1))
        self.assertAlmostEqual(offset[0], (start + 0.6) % 1.0)
        self.assertLess(offset[0], 1.0)

    def test_full(self):
        blob = struct.pack("<I", 4) + struct.pack("<16f", *range(16)) + struct.pack("<9f", *range(9))
        blob += struct.pack("<4I", 1, 2, 3, 4) + bytes(20) + struct.pack("<I", 7)
        animation = V.parse_texture_animation(blob, "switch")
        self.assertEqual((animation.kind, animation.type, animation.animation_flags), ("full", 4, 7))
        self.assertEqual(animation.matrix[15], 15.0)
        self.assertEqual(animation.rotation, (6.0, 7.0, 8.0))

    def test_bad_sizes_and_kinds(self):
        for blob in (bytes(7), struct.pack("<I2f", 5, 1, 1), struct.pack("<I4f", 1, 1, 1, 1, 1)):
            with self.assertRaises(FormatError):
                V.parse_texture_animation(blob, "switch")

    def test_constants(self):
        self.assertEqual(V.CUBE_FACES[3], "-Y")
        self.assertEqual(V.lightmap_value(bytes((9, 1, 2, 3))), 9)


class Protobuf(unittest.TestCase):
    def test_generic(self):
        data = B.proto(1, 300) + B.proto(2, b"abc") + bytes((0x1D,)) + struct.pack("<I", 7)
        fields = V.parse_protobuf(data)
        self.assertEqual([(f.number, f.value) for f in fields], [(1, 300), (2, b"abc"), (3, 7)])

    def test_achievements(self):
        data = B.proto(2, B.proto(1, 5) + B.proto(9, b"x")) + B.proto(4, b"") + B.proto(7, 1)
        result = V.parse_achievements(data)
        self.assertEqual(len(result.groups["achievements"]), 1)
        self.assertEqual(result.groups["achievements"][0][1].number, 9)
        self.assertEqual(len(result.groups["attributes"]), 1)
        self.assertEqual(result.other[0].number, 7)

    def test_truncated(self):
        for data in (B.proto(2, b"abc")[:-1], bytes((0x80,)), bytes((0x0F,)), bytes((0x00,))):
            with self.assertRaises(FormatError):
                V.parse_protobuf(data)


class Fsb(unittest.TestCase):
    def test_header_and_vorbis_packets(self):
        packets = struct.pack("<H", 3) + b"abc" + struct.pack("<H", 1) + b"d"
        setup = struct.pack("<I", 0xC0FFEE) + struct.pack("<2I", 0, 0) + struct.pack("<2I", 100, 512)
        data = B.fsb5(15, packets, chunks=[(3, struct.pack("<II", 10, 900)), (11, setup)])
        fsb = V.parse_fsb5(data)
        self.assertEqual((fsb.mode, fsb.sample.rate_index, fsb.sample.stereo), (15, 9, True))
        self.assertEqual(fsb.sample.sample_count, 1000)
        self.assertEqual(fsb.sample.loop, (10, 900))
        crc, pairs = fsb.sample.vorbis_setup
        self.assertEqual((crc, pairs), (0xC0FFEE, [(0, 0), (100, 512)]))
        self.assertEqual(fsb.packets(), [b"abc", b"d"])

    def test_frames(self):
        fsb = V.parse_fsb5(B.fsb5(16, bytes(140 * 3)))
        self.assertEqual(len(fsb.packets()), 3)
        self.assertIsNone(fsb.sample.loop)

    def test_checks(self):
        with self.assertRaises(FormatError):
            V.parse_fsb5(b"RIFF" + bytes(80))
        with self.assertRaises(FormatError):
            V.parse_fsb5(B.fsb5(15, b"", version=0))
        with self.assertRaises(FormatError):
            V.parse_fsb5(B.fsb5(15, b"", count=2))
        with self.assertRaises(FormatError):
            V.parse_fsb5(B.fsb5(15, b"abc")[:-1])

    def test_other_modes_are_not_described(self):
        with self.assertRaises(FormatError):
            V.parse_fsb5(B.fsb5(2, b"")).packets()
