import math
import struct
import unittest

import builders as B
from d3sno import FormatError
from d3sno import animation as A
from d3sno import appearance as P
from d3sno.transform import Transform, quat_multiply, quat_normalise, rotate, transforms_match

BUILDS = (("ps3", ">"), ("switch", "<"))
SQRT = math.sqrt(0.5)
Z_90 = (0.0, 0.0, SQRT, SQRT)


class Transforms(unittest.TestCase):
    def test_hamilton_product(self):
        i, j, k = (1.0, 0, 0, 0), (0, 1.0, 0, 0), (0, 0, 1.0, 0)
        self.assertEqual(quat_multiply(i, j), k)
        self.assertEqual(quat_multiply(j, i), (0, 0, -1.0, 0))

    def test_rotate(self):
        x, y, z = rotate(Z_90, (1.0, 0.0, 0.0))
        self.assertAlmostEqual(x, 0.0)
        self.assertAlmostEqual(y, 1.0)
        self.assertAlmostEqual(z, 0.0)

    def test_compose_formula(self):
        parent = Transform(Z_90, (1.0, 2.0, 3.0), 2.0)
        local = Transform(Z_90, (1.0, 0.0, 0.0), 3.0)
        result = parent.compose(local)
        self.assertAlmostEqual(result.t[0], 1.0)
        self.assertAlmostEqual(result.t[1], 4.0)
        self.assertAlmostEqual(result.t[2], 3.0)
        self.assertEqual(result.s, 6.0)
        self.assertAlmostEqual(result.q[2], 1.0)

    def test_matrix_agrees_with_apply(self):
        transform = Transform(Z_90, (1.0, 2.0, 3.0), 2.0)
        m = transform.matrix()
        p = (0.5, -1.0, 2.0)
        result = tuple(m[i] * p[0] + m[4 + i] * p[1] + m[8 + i] * p[2] + m[12 + i] for i in range(3))
        for got, want in zip(result, transform.apply(p)):
            self.assertAlmostEqual(got, want)

    def test_zero_quaternion_normalises_to_identity(self):
        self.assertEqual(quat_normalise((0.0, 0.0, 0.0, 0.0)), (0.0, 0.0, 0.0, 1.0))

    def test_match_ignores_quaternion_sign(self):
        a = Transform((0.0, 0.0, 1.0, 0.0))
        b = Transform((0.0, 0.0, -1.0, 0.0))
        self.assertTrue(transforms_match(a, b))
        self.assertFalse(transforms_match(a, Transform()))


class Ani(unittest.TestCase):
    def test_header_and_record(self):
        for build, _ in BUILDS:
            with self.subTest(build):
                data = B.build_ani(build, [B.simple_permutation(build)])
                ani = A.parse_ani(data)
                self.assertEqual(ani.build, build)
                self.assertEqual((ani.flags, ani.playback_mode), (0x33, 2))
                self.assertEqual(ani.appearance_handle, 4242)
                perm = ani.permutations[0]
                self.assertEqual((perm.flags, perm.name), (0x11, "Walk"))
                self.assertEqual((perm.fps, perm.keyframes, perm.duration), (30.0, 3, 2 / 30))
                self.assertEqual(perm.blend_times, (7, 8))
                self.assertEqual((perm.weight, perm.speed_min, perm.speed_delta), (100, 0.75, 0.125))
                self.assertEqual(perm.ragdoll_factors, (2.0, 3.0))
                self.assertEqual(perm.bone_count, 2)
                self.assertEqual(perm.contact_keyframes, (1.0, 2.0, 3.0, 4.0))
                self.assertEqual(perm.contact_offsets[3], (9.0, 10.0, 11.0))
                self.assertEqual(perm.earliest_interrupt, 2.0)
                self.assertEqual(perm.average_velocity, (1.0, 0.0, 0.0))
                self.assertEqual(perm.hardpoint_link, 77)

    def test_ps3_size_is_exact(self):
        data = B.build_ani("ps3", [B.simple_permutation("ps3")])
        self.assertEqual(len(data), 0x50 + 360)
        with self.assertRaises(FormatError):
            A.parse_ani(data + b"\0")

    def test_ps3_too_short(self):
        with self.assertRaises(FormatError):
            A.parse_ani(B.sno_head(123, ">"))

    def test_ps3_table_size_mismatch_warns(self):
        data = B.build_ani("ps3", [B.simple_permutation("ps3")], table_size=720)
        self.assertTrue(A.parse_ani(data).warnings)
        with self.assertRaises(FormatError):
            A.parse_ani(data, strict=True)

    def test_switch_table_size_not_multiple(self):
        data = B.build_ani("switch", [B.simple_permutation("switch")], table_size=359)
        with self.assertRaises(FormatError):
            A.parse_ani(data)

    def test_switch_table_past_end(self):
        data = B.build_ani("switch", [], table_size=360)
        with self.assertRaises(FormatError):
            A.parse_ani(data)

    def test_version_strict(self):
        data = B.build_ani("switch", [], version=1)
        self.assertEqual(A.parse_ani(data).build, "switch")
        with self.assertRaises(FormatError):
            A.parse_ani(data, strict=True)

    def test_switch_data(self):
        perm = B.simple_permutation(
            "switch", blend=[(1.0, 2.0), (3.0, 4.0)], attachments=[bytes(range(192))],
        )
        ani = A.parse_ani(B.build_ani("switch", [perm]))
        read = ani.permutations[0]
        self.assertTrue(read.has_data)
        self.assertEqual(read.track_hashes, [0xAA, 0xBB])
        self.assertEqual(read.blend_weights[1], (3.0, 4.0))
        self.assertEqual(read.attachments, [bytes(range(192))])
        self.assertEqual(read.nonlinear_offsets[2], (0.0, 0.0, 2.0))
        self.assertEqual(read.translation_blobs, perm["translation"])

    def test_identical_blobs_can_be_shared(self):
        blob = B.translation_blob([0, 2], [(1.0, 2.0, 3.0), (3.0, 2.0, 1.0)], 4, 3, "<")
        perm = B.simple_permutation("switch", translation=[blob, blob])
        data = B.build_ani("switch", [perm])
        self.assertEqual(data.count(blob), 1)
        self.assertEqual(A.parse_ani(data).permutations[0].translation_blobs, [blob, blob])

    def test_switch_pair_sizes_are_checked(self):
        for at in (0x7C, 0xAC, 0xC0, 0xD0, 0x138):
            data = bytearray(B.build_ani("switch", [B.simple_permutation("switch")]))
            size = struct.unpack_from("<I", data, 0x48 + at + 4)[0]
            struct.pack_into("<I", data, 0x48 + at + 4, size + 1)
            with self.assertRaises(FormatError):
                A.parse_ani(bytes(data))

    def test_blend_count_is_zero_or_bones(self):
        perm = B.simple_permutation("switch", blend=[(1.0, 2.0)])
        with self.assertRaises(FormatError):
            A.parse_ani(B.build_ani("switch", [perm]))

    def test_blob_past_end(self):
        data = bytearray(B.build_ani("switch", [B.simple_permutation("switch")]))
        table = struct.unpack_from("<I", data, 0x48 + 0xAC)[0] + 0x10
        struct.pack_into("<I", data, table + 4, 0xFFFFFF)
        with self.assertRaises(FormatError):
            A.parse_ani(bytes(data))

    def test_missing_can_is_reported_when_decoding(self):
        ani = A.parse_ani(B.build_ani("ps3", [B.simple_permutation("ps3")]))
        self.assertFalse(ani.permutations[0].has_data)
        with self.assertRaises(FormatError):
            A.decode_permutation(ani.permutations[0])


class Can(unittest.TestCase):
    def perms(self):
        return [
            B.simple_permutation("ps3"),
            B.simple_permutation(
                "ps3", name="Run", blend=[(1.0, 2.0), (3.0, 4.0)],
                attachments=[bytes(192)], keyframes=3,
            ),
        ]

    def test_walks_to_end_of_file(self):
        can = A.parse_can(B.build_can(self.perms()))
        self.assertEqual(len(can.permutations), 2)
        second = can.permutations[1]
        self.assertEqual(second.track_hashes, [0xAA, 0xBB])
        self.assertEqual(second.blend_weights[0], (1.0, 2.0))
        self.assertEqual(len(second.attachments), 1)
        self.assertEqual(len(second.nonlinear_offsets), 3)
        self.assertEqual(second.translation_blobs, self.perms()[1]["translation"])

    def test_empty_file(self):
        self.assertEqual(A.parse_can(b"").permutations, [])

    def test_blobs_start_at_multiples_of_8_from_the_table(self):
        blobs = [b"\x01" * 5, b"", b"\x02" * 3, b"\x03" * 8]
        section = B.can_curves(blobs)
        offsets = [struct.unpack_from(">I", section, 4 + i * 16)[0] for i in range(4)]
        self.assertEqual(offsets, [64, 0, 72, 80])
        perm = B.simple_permutation(
            "ps3", hashes=[1, 2, 3, 4], translation=blobs, rotation=[b""] * 4, scale=[b""] * 4,
        )
        read = A.parse_can(B.build_can([perm])).permutations[0]
        self.assertEqual(read.translation_blobs, blobs)

    def test_next_section_follows_the_last_blob_with_no_padding(self):
        perm = B.simple_permutation(
            "ps3", translation=[b"\x01" * 3, b""], rotation=[b""] * 2, scale=[b""] * 2,
        )
        self.assertEqual(len(A.parse_can(B.build_can([perm])).permutations), 1)

    def test_truncated(self):
        data = B.build_can(self.perms())
        for cut in range(1, len(data)):
            try:
                can = A.parse_can(data[:cut])
            except FormatError:
                continue
            self.assertEqual(len(can.permutations), 1)
            self.assertEqual(cut, len(B.build_can(self.perms()[:1])))

    def test_trailing_bytes(self):
        with self.assertRaises(FormatError):
            A.parse_can(B.build_can(self.perms()) + b"\0\0\0")

    def test_section_size_multiples(self):
        with self.assertRaises(FormatError):
            A.parse_can(struct.pack(">I", 3) + bytes(3))

    def corrupt(self, position, value):
        data = bytearray(B.build_can(self.perms()[:1]))
        struct.pack_into(">I", data, position, value)
        return bytes(data)

    def test_table_size_must_be_16_per_track(self):
        table_word = 4 + 8 + 4
        with self.assertRaises(FormatError):
            A.parse_can(self.corrupt(table_word, 16))

    def test_reserved_words_must_be_zero(self):
        with self.assertRaises(FormatError):
            A.parse_can(self.corrupt(4 + 8 + 4 + 4 + 8, 1))

    def test_blob_offset_is_checked(self):
        with self.assertRaises(FormatError):
            A.parse_can(self.corrupt(4 + 8 + 4 + 4, 40))

    def test_merge(self):
        perms = self.perms()
        ani = A.parse_ani(B.build_ani("ps3", perms), can=B.build_can(perms))
        self.assertTrue(ani.permutations[1].has_data)
        self.assertEqual(len(ani.permutations[1].attachments), 1)
        self.assertEqual(A.decode_permutation(ani.permutations[0]).translation[0].frames, [0, 2])

    def test_merge_permutation_count(self):
        perms = self.perms()
        with self.assertRaises(FormatError):
            A.parse_ani(B.build_ani("ps3", perms), can=B.build_can(perms[:1]))

    def test_merge_counts_must_agree(self):
        one = B.simple_permutation("ps3")
        other = B.simple_permutation("ps3", keyframes=4)
        with self.assertRaises(FormatError):
            A.parse_ani(B.build_ani("ps3", [other]), can=B.build_can([one]))


class Curves(unittest.TestCase):
    def test_key_width(self):
        self.assertEqual(A._key_width(255), 1)
        self.assertEqual(A._key_width(256), 2)
        self.assertEqual(A._key_width(65535), 2)
        self.assertEqual(A._key_width(65536), 4)

    def test_translation_codecs(self):
        for _, order in BUILDS:
            for codec, raw, want in ((1, 254, 1.0), (2, 65534, 1.0)):
                blob = B.translation_blob(
                    [0, 2], [(0, raw, 0), (127 if codec == 1 else 32767, 0, 0)], codec, 3, order,
                    base=(10.0, 20.0, 30.0),
                )
                curve = A.decode_translation(blob, 3, order)
                self.assertEqual(curve.frames, [0, 2])
                self.assertEqual(curve.base, (10.0, 20.0, 30.0))
                self.assertAlmostEqual(curve.values[0][0], 9.0)
                self.assertAlmostEqual(curve.values[0][1], 20.0 + want)
                self.assertAlmostEqual(curve.values[1][0], 10.0)

    def test_translation_float_codec_adds_base(self):
        blob = B.translation_blob([0, 2], [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)], 4, 3, "<",
                                  base=(1.0, 1.0, 1.0))
        self.assertEqual(A.decode_translation(blob, 3, "<").values[1], (5.0, 6.0, 7.0))

    def test_translation_one_key_has_base_zero_and_frames_at_2s(self):
        blob = B.translation_blob([0], [(1.5, 2.5, 3.5)], 4, 3, "<")
        self.assertEqual(blob[:2], b"\x01\x04")
        self.assertEqual(A.decode_translation(blob, 3, "<").values, [(1.5, 2.5, 3.5)])
        blob = B.translation_blob([0], [(127, 254, 0)], 1, 3, "<")
        values = A.decode_translation(blob, 3, "<").values[0]
        self.assertAlmostEqual(values[0], 0.0)
        self.assertAlmostEqual(values[1], 1.0)
        self.assertAlmostEqual(values[2], -1.0)

    def test_translation_wide_frames(self):
        for keyframes in (300, 70000):
            blob = B.translation_blob(
                [0, keyframes - 1], [(1.0, 1.0, 1.0), (2.0, 2.0, 2.0)], 4, keyframes, ">",
            )
            curve = A.decode_translation(blob, keyframes, ">")
            self.assertEqual(curve.frames, [0, keyframes - 1])

    def test_rotation(self):
        for keyframes, order in ((3, "<"), (300, ">")):
            blob = B.rotation_blob(
                [0, keyframes - 1], [(0, 0, 0, 32767), (32767, 0, 0, -32767)], keyframes, order,
            )
            curve = A.decode_rotation(blob, keyframes, order)
            self.assertEqual(curve.values[0], (0.0, 0.0, 0.0, 1.0))
            self.assertEqual(curve.values[1], (1.0, 0.0, 0.0, -1.0))

    def test_rotation_values_align_to_2(self):
        blob = B.rotation_blob([0, 1, 2], [(0, 0, 0, 1)] * 3, 3, "<")
        self.assertEqual(len(blob), 4 + 24)
        blob = B.rotation_blob([0, 2], [(0, 0, 0, 1)] * 2, 3, "<")
        self.assertEqual(blob[3], 0)
        self.assertEqual(len(A.decode_rotation(blob, 3, "<").values), 2)

    def test_scale_codecs(self):
        curve = A.decode_scale(B.scale_blob([0, 2], [127, 254], 1, 3, "<"), 3, "<")
        self.assertEqual(curve.values, [(1.0,), (2.0,)])
        curve = A.decode_scale(B.scale_blob([0, 2], [32767, 65534], 2, 3, ">"), 3, ">")
        self.assertEqual(curve.values, [(1.0,), (2.0,)])
        curve = A.decode_scale(B.scale_blob([0, 2], [1.5, 2.5], 4, 3, "<"), 3, "<")
        self.assertEqual(curve.values, [(1.5,), (2.5,)])

    def test_scale_frames_align_to_s(self):
        curve = A.decode_scale(B.scale_blob([0, 299], [1.0, 2.0], 4, 300, "<"), 300, "<")
        self.assertEqual(curve.frames, [0, 299])

    def test_scale_keys_are_not_clamped(self):
        curve = A.decode_scale(B.scale_blob([0, 2], [0.0, 500.0], 4, 3, "<"), 3, "<")
        self.assertEqual(curve.values, [(0.0,), (500.0,)])

    def test_padding_to_4_bytes(self):
        blob = B.scale_blob([0, 2], [127, 254], 1, 3, "<")
        padded = blob + bytes(-len(blob) % 4)
        self.assertEqual(A.decode_scale(padded, 3, "<").values, [(1.0,), (2.0,)])

    def test_too_long_blob(self):
        blob = B.scale_blob([0, 2], [1.0, 2.0], 4, 3, "<") + bytes(4)
        with self.assertRaises(FormatError):
            A.decode_scale(blob, 3, "<")

    def test_truncated_blob(self):
        blob = B.rotation_blob([0, 2], [(0, 0, 0, 1)] * 2, 3, "<")
        with self.assertRaises(FormatError):
            A.decode_rotation(blob[:-1], 3, "<")

    def test_codec_other_than_1_or_2_reads_as_floats(self):
        blob = bytes((1, 9, 0, 0)) + struct.pack("<f", 2.5)
        self.assertEqual(A.decode_scale(blob, 3, "<").values, [(2.5,)])
        blob = bytes((1, 0, 0, 0)) + struct.pack("<3f", 1.0, 2.0, 3.0)
        self.assertEqual(A.decode_translation(blob, 3, "<").values, [(1.0, 2.0, 3.0)])

    def test_first_key_must_be_zero(self):
        with self.assertRaises(FormatError):
            A.decode_rotation(B.rotation_blob([1, 2], [(0, 0, 0, 1)] * 2, 3, "<"), 3, "<")

    def test_keys_must_ascend(self):
        with self.assertRaises(FormatError):
            A.decode_rotation(B.rotation_blob([0, 2, 2], [(0, 0, 0, 1)] * 3, 3, "<"), 3, "<")

    def test_last_key_must_be_k_minus_1(self):
        with self.assertRaises(FormatError):
            A.decode_rotation(B.rotation_blob([0, 1], [(0, 0, 0, 1)] * 2, 3, "<"), 3, "<")

    def test_zero_key_count(self):
        with self.assertRaises(FormatError):
            A.decode_rotation(b"\0\0\0\0", 3, "<")

    def test_empty_blob_is_an_empty_curve(self):
        for decode in (A.decode_translation, A.decode_rotation, A.decode_scale):
            self.assertEqual(decode(b"", 3, "<").frames, [])


class Sampling(unittest.TestCase):
    CURVE = A.Curve([0, 4, 10], [(0.0,), (4.0,), (1.0,)])

    def test_keys_and_segments(self):
        self.assertEqual(A.sample_curve(self.CURVE, 0, (9.0,)), (0.0,))
        self.assertEqual(A.sample_curve(self.CURVE, 2, (9.0,)), (2.0,))
        self.assertEqual(A.sample_curve(self.CURVE, 4, (9.0,)), (4.0,))
        self.assertEqual(A.sample_curve(self.CURVE, 7, (9.0,)), (2.5,))

    def test_outside_the_keys(self):
        self.assertEqual(A.sample_curve(self.CURVE, -1, (9.0,)), (0.0,))
        self.assertEqual(A.sample_curve(self.CURVE, 11, (9.0,)), (1.0,))

    def test_one_key(self):
        self.assertEqual(A.sample_curve(A.Curve([0], [(5.0,)]), 3, (9.0,)), (5.0,))

    def test_empty_gives_default(self):
        self.assertEqual(A.sample_curve(A.Curve([], []), 3, (9.0,)), (9.0,))

    def curves(self, rotation):
        return A.DecodedPermutation(
            [A.Curve([], [])], [rotation], [A.Curve([], [])],
        )

    def test_empty_track_defaults(self):
        transform = A.sample_track(self.curves(A.Curve([], [])), 0, 1)
        self.assertEqual(transform, Transform())

    def test_rotation_lerp_then_normalise(self):
        rotation = A.Curve([0, 2], [(0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 1.0, 0.0)])
        q = A.sample_track(self.curves(rotation), 0, 1).q
        self.assertAlmostEqual(q[2], SQRT)
        self.assertAlmostEqual(q[3], SQRT)

    def test_no_shortest_arc_flip(self):
        rotation = A.Curve([0, 2], [(0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.6, -0.8)])
        q = A.sample_track(self.curves(rotation), 0, 1).q
        length = math.sqrt(0.3 ** 2 + 0.1 ** 2)
        self.assertAlmostEqual(q[2], 0.3 / length)
        self.assertAlmostEqual(q[3], 0.1 / length)


class Scale(unittest.TestCase):
    def curves(self, scale):
        return A.DecodedPermutation([A.Curve([], [])], [A.Curve([], [])], [scale])

    def test_single_key_is_unclamped(self):
        transform = A.sample_track(self.curves(A.Curve([0], [(0.0,)])), 0, 1)
        self.assertEqual(transform.s, 0.0)
        transform = A.sample_track(self.curves(A.Curve([0], [(500.0,)])), 0, 1)
        self.assertEqual(transform.s, 500.0)

    def test_result_between_keys_is_clamped(self):
        curve = A.Curve([0, 2], [(0.0,), (500.0,)])
        self.assertEqual(A.sample_track(self.curves(curve), 0, 0).s, 0.01)
        self.assertEqual(A.sample_track(self.curves(curve), 0, 2).s, 100.0)
        self.assertEqual(A.sample_track(self.curves(curve), 0, 1).s, 100.0)

    def test_non_finite_result_is_fatal(self):
        curve = A.Curve([0, 2], [(math.inf,), (1.0,)])
        with self.assertRaises(FormatError):
            A.sample_track(self.curves(curve), 0, 1)


class Rotation(unittest.TestCase):
    def sample(self, curve, frame):
        curves = A.DecodedPermutation([A.Curve([], [])], [curve], [A.Curve([], [])])
        return A.sample_track(curves, 0, frame).q

    def test_keys_are_normalised_before_the_lerp(self):
        q = self.sample(A.Curve([0, 2], [(0.0, 0.0, 0.0, 10.0), (0.0, 0.0, 1.0, 0.0)]), 1)
        self.assertAlmostEqual(q[2], SQRT)
        self.assertAlmostEqual(q[3], SQRT)

    def test_single_key_is_normalised(self):
        self.assertEqual(self.sample(A.Curve([0], [(0.0, 0.0, 0.0, 2.0)]), 0), (0.0, 0.0, 0.0, 1.0))

    def test_decoding_uses_the_single_precision_unit(self):
        blob = B.rotation_blob([0], [(3, 0, 0, 0)], 3, "<")
        unit = struct.unpack("<f", struct.pack("<f", 1 / 32767))[0]
        expected = struct.unpack("<f", struct.pack("<f", 3 * unit))[0]
        self.assertEqual(A.decode_rotation(blob, 3, "<").values[0][0], expected)


class Time(unittest.TestCase):
    def perm(self, keyframes=4, rate=0.5):
        return A.parse_ani(B.build_ani("switch", [B.simple_permutation(
            "switch", keyframes=keyframes, rate=rate,
            translation=[b"", b""], rotation=[b"", b""], scale=[b"", b""],
        )])).permutations[0]

    def test_stop_mode_stops_at_the_end(self):
        perm = self.perm()
        self.assertEqual(A.play_time(perm, 0.05, A.MODE_STOP), 0.05)
        self.assertEqual(A.play_time(perm, 9.0, A.MODE_STOP), perm.duration)

    def test_wrap_mode_subtracts_the_end(self):
        perm = self.perm()
        end = perm.duration
        self.assertEqual(A.play_time(perm, end, A.MODE_WRAP), end)
        self.assertAlmostEqual(A.play_time(perm, end + 0.01, A.MODE_WRAP), 0.01)
        self.assertAlmostEqual(A.play_time(perm, 2 * end, A.MODE_WRAP), end)

    def test_other_modes_are_fatal(self):
        with self.assertRaises(FormatError):
            A.play_time(self.perm(), 0.0, 1)

    def test_frame(self):
        self.assertEqual(A.frame_at_time(self.perm(), 0.05), 1.5)

    def test_even_frames_from_20_fps(self):
        perm = self.perm()
        self.assertEqual(A.frame_at_time(perm, 0.1, even_frames=True), 2.0)
        self.assertEqual(A.frame_at_time(perm, 0.13, even_frames=True), 2.0)
        slow = self.perm(rate=0.25)
        self.assertEqual(A.frame_at_time(slow, 0.2, even_frames=True), 3.0)


class NoTables(unittest.TestCase):
    def test_no_tables_on_both_builds(self):
        for build, _ in BUILDS:
            perm = B.simple_permutation(build, translation=[], rotation=[], scale=[])
            can = B.build_can([perm]) if build == "ps3" else None
            parsed = A.parse_ani(B.build_ani(build, [perm]), can=can).permutations[0]
            curves = A.decode_permutation(parsed)
            self.assertEqual(A.sample_track(curves, 1, 0), Transform())


class Pose(unittest.TestCase):
    BONES = [(0xAA, -1, (1.0, 0.0, 0.0)), (0xBB, 0, (1.0, 2.0, 0.0)), (0xCC, 1, (1.0, 2.0, 3.0))]

    def rig(self, build="switch"):
        data = B.build_app_switch(bones=self.BONES) if build == "switch" \
            else B.build_app_ps3(bones=self.BONES)
        return P.parse_appearance(data).bones

    def permutation(self, build, **changes):
        order = ">" if build == "ps3" else "<"
        perm = B.simple_permutation(
            build,
            hashes=[0xBB, 0xDEAD],
            translation=[B.translation_blob([0], [(0.0, 5.0, 0.0)], 4, 3, order), b""],
            rotation=[B.rotation_blob([0], [(0, 0, 0, 32767)], 3, order), b""],
            scale=[b"", b""],
        )
        perm.update(changes)
        data = B.build_ani(build, [perm])
        can = B.build_can([perm]) if build == "ps3" else None
        parsed = A.parse_ani(data, can=can).permutations[0]
        return parsed, A.decode_permutation(parsed)

    def test_rest_pose_when_no_track_matches(self):
        bones = self.rig()
        perm, curves = self.permutation("switch", hashes=[1, 2])
        pose = A.pose_at(bones, perm, curves, 0)
        self.assertEqual(pose.world[2].t, (1.0, 2.0, 3.0))
        for transform in pose.skin:
            for got, want in zip(transform.t, (0.0, 0.0, 0.0)):
                self.assertAlmostEqual(got, want)
        self.assertFalse(pose.bind_is_rest_inverse)

    def test_matching_track_replaces_local(self):
        bones = self.rig()
        perm, curves = self.permutation("switch")
        pose = A.pose_at(bones, perm, curves, 0)
        self.assertEqual(pose.local[1].t, (0.0, 5.0, 0.0))
        self.assertEqual(pose.world[1].t, (1.0, 5.0, 0.0))
        self.assertEqual(pose.world[2].t, (1.0, 5.0, 3.0))
        self.assertEqual(pose.skin[1].t, (0.0, 3.0, 0.0))

    def test_track_binds_to_the_first_bone_with_its_hash(self):
        data = B.build_app_switch(bones=[
            (0xBB, -1, (1.0, 0.0, 0.0)), (0xBB, -1, (2.0, 0.0, 0.0)),
        ])
        bones = P.parse_appearance(data).bones
        perm, curves = self.permutation("switch")
        pose = A.pose_at(bones, perm, curves, 0)
        self.assertEqual(pose.local[0].t, (0.0, 5.0, 0.0))
        self.assertEqual(pose.local[1].t, (2.0, 0.0, 0.0))

    def test_ps3_flags_the_bind(self):
        bones = self.rig("ps3")
        perm, curves = self.permutation("ps3")
        pose = A.pose_at(bones, perm, curves, 0)
        self.assertTrue(pose.bind_is_rest_inverse)
        self.assertEqual(pose.skin[1].t, (0.0, 3.0, 0.0))

    def test_skin_vertex(self):
        skin = [Transform(t=(1.0, 0.0, 0.0)), Transform(t=(0.0, 2.0, 0.0))]
        result = A.skin_vertex((1.0, 1.0, 1.0), (0, 1, 0, 0), (51, 204, 0, 0), skin)
        self.assertAlmostEqual(result[0], 1.2)
        self.assertAlmostEqual(result[1], 2.6)
        self.assertAlmostEqual(result[2], 1.0)

    def test_weight_zero_joint_is_ignored(self):
        skin = [Transform()]
        self.assertEqual(A.skin_vertex((1.0, 0.0, 0.0), (0, 99), (255, 0), skin), (1.0, 0.0, 0.0))


class Gltf(unittest.TestCase):
    def app(self):
        return P.parse_appearance(B.build_app_switch(bones=[
            (0xAA, -1, (0.0, 0.0, 0.0)), (0xBB, 0, (1.0, 0.0, 0.0)),
        ]))

    def ani(self, **changes):
        return A.parse_ani(B.build_ani("switch", [B.simple_permutation("switch", **changes)]))

    def test_channels(self):
        doc = A.to_gltf_animated(self.app(), self.ani())
        animation = doc["animations"][0]
        self.assertEqual(animation["name"], "Walk")
        paths = sorted((c["target"]["node"], c["target"]["path"]) for c in animation["channels"])
        self.assertEqual(paths, [
            (0, "rotation"), (0, "scale"), (0, "translation"),
            (1, "rotation"), (1, "scale"), (1, "translation"),
        ])
        first = animation["samplers"][animation["channels"][0]["sampler"]]
        self.assertEqual(first["interpolation"], "LINEAR")
        accessor = doc["accessors"][first["input"]]
        self.assertEqual(accessor["count"], 2)
        self.assertAlmostEqual(accessor["max"][0], 2 / 30, places=6)

    def test_unmatched_track_is_ignored(self):
        doc = A.to_gltf_animated(self.app(), self.ani(hashes=[0xAA, 0xDEAD]))
        nodes = {c["target"]["node"] for c in doc["animations"][0]["channels"]}
        self.assertEqual(nodes, {0})

    def test_no_match_gives_no_animation(self):
        doc = A.to_gltf_animated(self.app(), self.ani(hashes=[1, 2]))
        self.assertNotIn("animations", doc)

    def test_no_such_permutation(self):
        with self.assertRaises(FormatError):
            A.to_gltf_animated(self.app(), self.ani(), 5)

    def test_zero_frame_rate(self):
        with self.assertRaises(FormatError):
            A.to_gltf_animated(self.app(), self.ani(rate=0.0))

    def test_json_is_valid(self):
        import json
        self.assertIn("animations", json.loads(A.to_gltf_animated_json(self.app(), self.ani())))
