"""Stage 15, animation (30_create_content.md W15-B1, B2, B3, B3a). Synthetic data only."""

import _libpath  # noqa: F401  (python -I: the library folder on sys.path)
import math
import struct
import unittest

import builders as B
from d3sno import anim_write as W
from d3sno import animation as A
from d3sno import anims
from d3sno import children as C
from d3sno import sno_layout as L
from d3sno.binary import FormatError

BUILDS = ("ps3", "switch", "x360")


def order(build):
    return "<" if build == "switch" else ">"


def half_up(x):
    return int(math.floor(x + 0.5))


def padded(perm):
    """A builder permutation with its blobs ending at a multiple of 4 (W3-3)."""
    for kind in ("translation", "rotation", "scale"):
        perm[kind] = [b + bytes(-len(b) % 4) for b in perm[kind]]
    return perm


def unit(*q):
    n = math.sqrt(sum(c * c for c in q))
    return tuple(c / n for c in q)


def two_track_curves():
    return A.DecodedPermutation(
        [A.Curve([0, 4], [(1.0, 2.0, 3.0), (1.5, 2.0, 2.5)]), A.Curve([], [])],
        [A.Curve([0, 4], [(0.0, 0.0, 0.0, 1.0), unit(0.0, 0.0, 0.5, 1.0)]),
         A.Curve([0, 4], [(0.0, 0.0, 0.0, 1.0), unit(0.0, 0.0, 0.5, 1.0)])],
        [A.Curve([0], [(1.0,)]), A.Curve([], [])],
    )


def new_switch_perm(name="Idle", hashes=(0x11, 0x22), curves=None):
    perm = W.new_permutation("switch", name, 5, 0.5, list(hashes))
    W.encode_animation([(perm, curves or two_track_curves())], "switch")
    return perm


class CurveEncoding(unittest.TestCase):
    """W15-B1."""

    def test_translation_codecs_store_offsets_from_first_key(self):
        values = [(1.0, 2.0, 3.0), (1.5, 2.0, 2.5)]
        for build in BUILDS:
            o = order(build)
            expected = {
                1: [(127, 127, 127), (half_up(1.5 * 127), 127, half_up(0.5 * 127))],
                2: [(32767, 32767, 32767), (half_up(1.5 * 32767), 32767, half_up(0.5 * 32767))],
                4: [(0.0, 0.0, 0.0), (0.5, 0.0, -0.5)],
            }
            for codec, raw in expected.items():
                blob = W.encode_curve_values("translation", [0, 4], values, 5, build, codec=codec)
                self.assertEqual(blob, C.encode_curve(C.CurveParts(
                    "translation", [0, 4], raw, codec, (1.0, 2.0, 3.0)), 5, o))
                stored = B.translation_blob([0, 4], raw, codec, 5, o, base=(1.0, 2.0, 3.0))
                self.assertEqual(blob, stored + bytes(-len(stored) % 4))

    def test_single_key_translation_has_no_base(self):
        blob = W.encode_curve_values("translation", [0], [(0.5, -0.5, 0.0)], 3, "ps3", codec=1)
        self.assertEqual(blob[:2], b"\x01\x01")
        self.assertEqual(len(blob), 2 + 3 + 3)  # count, codec, frame + pad to 2S, 3 bytes, pad 4
        self.assertEqual(list(blob[3:6]), [191, 64, 127])
        curve = A.decode_translation(blob, 3, ">")
        self.assertEqual(curve.base, (0.0, 0.0, 0.0))
        with self.assertRaises(FormatError):
            W.encode_curve_values("translation", [0], [(0.5, 0.5, 0.5)], 3, "ps3", 1, base=(1, 0, 0))

    def test_rotation_and_scale(self):
        q = unit(0.1, 0.2, 0.3, 0.9)
        blob = W.encode_curve_values("rotation", [0, 2], [q, (0.0, 0.0, 0.0, 1.0)], 3, "switch")
        raw = [tuple(int(math.floor(c * 32767 + 0.5)) for c in q), (0, 0, 0, 32767)]
        self.assertEqual(blob, B.rotation_blob([0, 2], raw, 3, "<"))
        for codec, raw in ((1, [127, 254]), (2, [32767, 65534]), (4, [1.0, 2.0])):
            blob = W.encode_curve_values("scale", [0, 2], [(1.0,), (2.0,)], 3, "ps3", codec=codec)
            self.assertEqual(blob, B.scale_blob([0, 2], raw, codec, 3, ">") + bytes(-len(
                B.scale_blob([0, 2], raw, codec, 3, ">")) % 4))

    def test_frame_width_and_frame_rules(self):
        blob = W.encode_curve_values("scale", [0, 299], [(1.0,), (1.0,)], 300, "ps3", codec=4)
        self.assertEqual(struct.unpack_from(">H", blob, 0)[0], 2)  # S = 2 from K = 300
        self.assertEqual(A.decode_scale(blob, 300, ">").frames, [0, 299])
        for frames in ([1, 4], [0, 3], [0, 0, 4], [0, 5]):
            with self.assertRaises(FormatError):
                W.encode_curve_values("scale", frames, [(1.0,)] * len(frames), 5, "ps3", codec=4)
        self.assertEqual(W.encode_curve_values("scale", [], [], 5, "ps3"), b"")
        with self.assertRaises(FormatError):
            W.encode_curve_values("translation", [0, 4], [(1.0, 2.0), (1.0, 2.0)], 5, "ps3")
        with self.assertRaises(FormatError):
            W.encode_curve_values("translation", [0, 4], [(0.0,) * 3, (3.0, 0, 0)], 5, "ps3", codec=1)

    def test_stored_blobs_re_encode_identically(self):
        k = 3
        for build in BUILDS:
            o = order(build)
            blobs = {
                "translation": [
                    B.translation_blob([0, 2], [(1, 2, 3), (250, 0, 127)], 1, k, o, base=(0.5, -2.0, 7.25)),
                    B.translation_blob([0, 2], [(1, 65535, 3), (2, 0, 32767)], 2, k, o, base=(1e3, 0.0, -3.0)),
                    B.translation_blob([0, 1, 2], [(0.25, -0.0, 1.5), (0.0, 1.0, -0.0), (2.0, 3.0, 4.0)],
                                       4, k, o, base=(1.0, 2.0, 3.0)),
                    B.translation_blob([0], [(1.0, -2.5, 3.0)], 4, k, o),
                ],
                "rotation": [B.rotation_blob([0, 2], [(1, -2, 3, 32000), (-32767, 0, 0, 5)], k, o)],
                "scale": [B.scale_blob([0, 2], [3, 255], 1, k, o), B.scale_blob([0, 2], [7, 65535], 2, k, o),
                          B.scale_blob([0], [1.5], 4, k, o)],
            }
            decoders = {"translation": A.decode_translation, "rotation": A.decode_rotation,
                        "scale": A.decode_scale}
            for kind, items in blobs.items():
                for blob in items:
                    blob = blob + bytes(-len(blob) % 4)
                    curve = decoders[kind](blob, k, o)
                    self.assertEqual(W.encode_curve_object(curve, kind, k, build), blob, (build, kind))

    def test_negative_zero_offset_is_kept_from_raw(self):
        blob = B.translation_blob([0, 2], [(-0.0, 0.0, 0.0), (1.0, -0.0, 0.0)], 4, 3, ">", base=(1.0, 1.0, 1.0))
        curve = A.decode_translation(blob, 3, ">")
        plain = W.encode_curve_values("translation", curve.frames, curve.values, 3, "ps3", 4,
                                      base=curve.base)
        self.assertNotEqual(plain, blob)  # quantising again loses the signed zeros
        self.assertEqual(W.encode_curve_values("translation", curve.frames, curve.values, 3, "ps3", 4,
                                               base=curve.base, raw=curve.raw), blob)

    def test_edited_values_ignore_raw(self):
        blob = B.translation_blob([0, 2], [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)], 4, 3, ">", base=(1.0, 1.0, 1.0))
        curve = A.decode_translation(blob, 3, ">")
        curve.values[1] = (3.0, 1.0, 1.0)
        out = A.decode_translation(W.encode_curve_object(curve, "translation", 3, "ps3"), 3, ">")
        self.assertEqual(out.values, [(1.0, 1.0, 1.0), (3.0, 1.0, 1.0)])

    def test_encode_animation_sets_every_blob(self):
        perm = new_switch_perm()
        self.assertEqual(len(perm.translation_blobs), 2)
        self.assertEqual(perm.translation_blobs[1], b"")
        decoded = A.decode_permutation(perm)
        self.assertEqual(decoded.translation[0].values, [(1.0, 2.0, 3.0), (1.5, 2.0, 2.5)])
        self.assertEqual(decoded.translation[0].codec, 4)
        with self.assertRaises(FormatError):
            W.encode_animation([(W.new_permutation("ps3", "x", 5, 0.5, [1]), two_track_curves())], "ps3")


class CodecChoice(unittest.TestCase):
    """W15-B3 codec for new ps3 / x360 translation data (HYPOTHESIS, off by default)."""

    def test_hypothesis_rule(self):
        small = [(0.0, 0.0, 0.0), (0.5, -0.9, 1.0)]
        big = [(0.0, 0.0, 0.0), (1.5, 0.0, 0.0)]
        self.assertEqual(W.hypothesis_codec([0, 4], small), 2)
        self.assertEqual(W.hypothesis_codec([0, 4], big), 4)
        self.assertEqual(W.hypothesis_codec([0], [(1.5, 0.0, 0.0)]), 4)
        self.assertEqual(W.hypothesis_codec([0, 4], small, max_error=1 / 254), 1)
        self.assertEqual(W.hypothesis_codec([0, 4], small, max_error=1 / 300), 2)
        # offsets, not values, decide with 2 or more keys
        self.assertEqual(W.hypothesis_codec([0, 4], [(10.0, 0, 0), (10.5, 0, 0)]), 2)

    def test_rule_is_an_option(self):
        values = [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0)]
        for build in ("ps3", "x360"):
            default = W.encode_curve_values("translation", [0, 4], values, 5, build)
            self.assertEqual(default[1], 4)
            chosen = W.encode_curve_values("translation", [0, 4], values, 5, build, codec_rule="hypothesis")
            self.assertEqual(chosen[1], 2)
        self.assertEqual(W.encode_curve_values("translation", [0, 4], values, 5, "switch",
                                               codec_rule="hypothesis")[1], 4)
        self.assertEqual(W.encode_curve_values("scale", [0], [(0.5,)], 5, "ps3", codec_rule="hypothesis")[1], 4)
        with self.assertRaises(FormatError):
            W.encode_curve_values("translation", [0, 4], values, 5, "ps3", codec_rule="guess")


class SwitchAssembly(unittest.TestCase):
    """W15-B2 (switch)."""

    def test_new_file_is_canonical_and_reads_back(self):
        perm = new_switch_perm()
        data = W.write_ani([perm])
        self.assertTrue(L.is_canonical(data, "ani"))
        ani = A.parse_ani(data, strict=True)
        self.assertEqual(ani.build, "switch")
        read = ani.permutations[0]
        self.assertEqual((read.name, read.bone_count, read.keyframes), ("Idle", 2, 5))
        self.assertEqual(read.track_hashes, [0x11, 0x22])
        self.assertEqual(read.translation_blobs, perm.translation_blobs)
        self.assertEqual(read.nonlinear_offsets, [(0.0, 0.0, 0.0)] * 5)
        self.assertEqual(struct.unpack_from("<I", data, 0x30)[0], 1)  # permutation count word
        self.assertEqual(ani.appearance_handle, -1)

    def test_identical_blobs_are_written_once(self):
        one, two = new_switch_perm("A"), new_switch_perm("B", hashes=(0x33, 0x44))
        data = W.write_ani([one, two])
        blob = one.rotation_blobs[0]
        self.assertEqual(data.count(blob), 1)
        tree = L.parse_tree(data, "ani")
        self.assertEqual(len(L.shared_pairs(tree)), 5)  # rotation x3, translation, scale
        ani = A.parse_ani(data)
        self.assertEqual(ani.permutations[1].rotation_blobs, [blob, blob])
        apart = W.write_ani([one, two], head=data, share_blobs=False)
        self.assertEqual(apart.count(blob), 4)
        self.assertEqual(A.parse_ani(apart).permutations[1].rotation_blobs, [blob, blob])

    def test_unchanged_file_round_trips(self):
        data = W.write_ani([new_switch_perm("A"), new_switch_perm("B")])
        ani = A.parse_ani(data)
        self.assertEqual(W.write_ani(ani.permutations, head=data), data)
        for perm in ani.permutations:  # decoded and re-encoded curves
            W.encode_animation([(perm, A.decode_permutation(perm))], "switch")
        self.assertEqual(W.write_ani(ani.permutations, head=data), data)
        self.assertEqual(L.rebuild_sno(data, "ani"), data)

    def test_entry_tails_and_head_from_ani_file(self):
        data = B.build_ani("switch", [B.simple_permutation("switch")])  # 0xCC entry tails
        ani = A.parse_ani(data)
        self.assertTrue(ani.permutations[0].curve_tails)
        out = W.write_ani(ani.permutations, head=ani)
        self.assertTrue(L.is_canonical(out, "ani"))
        tree = L.parse_tree(out, "ani")
        _, _, value = L.find_pair(tree, "ptPermutations[0].ptTranslationCurves")
        self.assertEqual(bytes(value.block.elements[0].raw[8:]), b"\xCC" * 8)
        again = A.parse_ani(out)
        self.assertEqual((again.flags, again.playback_mode, again.appearance_handle), (0x33, 2, 4242))
        self.assertEqual(again.permutations[0].translation_blobs, ani.permutations[0].translation_blobs)

    def test_counts_follow_the_data(self):
        perm = new_switch_perm()
        perm.attachments = [bytes(range(192))]
        perm.blend_weights = [(0.5, 0.5), (1.0, 0.0)]
        perm.keyframes = 5
        ani = A.parse_ani(W.write_ani([perm]))
        read = ani.permutations[0]
        self.assertEqual((read.attachment_count, read.blend_weight_count), (1, 2))
        self.assertEqual(read.attachments, [bytes(range(192))])
        perm.nonlinear_offsets = perm.nonlinear_offsets[:4]
        with self.assertRaises(FormatError):
            W.write_ani([perm])

    def test_other_builds_refuse(self):
        with self.assertRaises(FormatError):
            W.write_ani([], "ps3")


class ChildAssembly(unittest.TestCase):
    """W15-B2 (ps3, x360): write_can + update_ani_counts / update_ani_records."""

    def test_unchanged_child_and_parent(self):
        for build in ("ps3", "x360"):
            perms = [padded(B.simple_permutation(build)), padded(B.simple_permutation(build, name="Run"))]
            ani_data, can = B.build_ani(build, perms), B.build_can(perms)
            ani = A.parse_ani(ani_data, build, can=can)
            for perm in ani.permutations:
                W.encode_animation([(perm, A.decode_permutation(perm))], build)
            self.assertEqual(C.write_can(ani), can)
            self.assertEqual(W.update_ani_records(ani_data, ani), ani_data)

    def test_new_curves_and_fields(self):
        perms = [B.simple_permutation("ps3")]
        ani_data, can = B.build_ani("ps3", perms), B.build_can(perms)
        ani = A.parse_ani(ani_data, can=can)
        perm = ani.permutations[0]
        curves = A.decode_permutation(perm)
        curves.translation[1] = A.Curve([0, 2], [(0.0, 0.0, 0.0), (0.25, 0.5, -0.5)])
        W.encode_animation([(perm, curves)], "ps3", codec_rule="hypothesis")
        W.set_in_place(perm)
        perm.attachments = [bytes(192)]
        new_can = C.write_can(ani)
        new_ani = W.update_ani_records(ani_data, ani)
        self.assertEqual(C.can_count_problems(A.parse_ani(new_ani), A.parse_can(new_can)), [])
        read = A.parse_ani(new_ani, can=new_can).permutations[0]
        self.assertEqual(read.average_velocity, (0.0, 0.0, 0.0))
        self.assertEqual(read.contact_keyframes, (-1.0, 1.0, 0.0, 1.0))
        self.assertEqual(read.attachment_count, 1)
        self.assertEqual(read.translation_blobs[1][1], 2)  # hypothesis codec 2
        for got, want in zip(A.decode_permutation(read).translation[1].values[1], (0.25, 0.5, -0.5)):
            self.assertAlmostEqual(got, want, delta=1 / 32767)
        changed = [i for i in range(len(ani_data)) if ani_data[i] != new_ani[i]]
        record = 0x50
        allowed = set(range(record + 0xE0, record + 0x120)) | set(range(record + 0x12C, record + 0x130)) \
            | set(range(record + 0x148, record + 0x154))
        self.assertTrue(set(changed) <= allowed, [hex(i) for i in changed])


class InputsAndDefaults(unittest.TestCase):
    """W15-B3."""

    def test_new_permutation_defaults(self):
        perm = W.new_permutation("switch", "Stand", 4, 0.5, [1, 2, 3])
        self.assertEqual(perm.average_velocity, (0.0, 0.0, 0.0))
        self.assertEqual(perm.nonlinear_offsets, [(0.0, 0.0, 0.0)] * 4)
        self.assertEqual(perm.contact_keyframes, (-1.0, 1.0, 0.0, 1.0))
        self.assertEqual(perm.contact_offsets, ((0.0, 0.0, 0.0),) * 4)
        raw = perm.raw
        self.assertEqual(struct.unpack_from("<4f", raw, 0xE0), (-1.0, 1.0, 0.0, 1.0))
        self.assertEqual(struct.unpack_from("<f", raw, 0x48)[0], 0.5)
        self.assertEqual(struct.unpack_from("<I", raw, 0x78)[0], 3)
        self.assertEqual(struct.unpack_from("<I", raw, 0xA8)[0], 4)
        self.assertEqual(raw[4:10], b"Stand\0")
        self.assertEqual(A._read_permutation(raw, "<", "switch").name, "Stand")
        self.assertEqual(W.permutation_problems(perm), [])

    def test_set_in_place(self):
        perm = A.parse_ani(B.build_ani("switch", [B.simple_permutation("switch")])).permutations[0]
        self.assertNotEqual(perm.average_velocity, (0.0, 0.0, 0.0))
        W.set_in_place(perm)
        self.assertEqual(perm.nonlinear_offsets, [(0.0, 0.0, 0.0)] * perm.keyframes)
        self.assertEqual(perm.average_velocity, (0.0, 0.0, 0.0))
        self.assertEqual(perm.contact_keyframes, W.IN_PLACE_CONTACT)

    def test_pack_keeps_unchanged_bytes(self):
        raw = bytearray(B._ani_record(B.simple_permutation("ps3"), ">"))
        raw[0x68:0x6C] = b"\x7f\xa0\x00\x01"  # a signalling NaN payload
        raw[0x6C:0x70] = b"\x80\x00\x00\x00"  # negative zero
        perm = A._read_permutation(bytes(raw), ">", "ps3")
        self.assertEqual(W.pack_permutation(perm), bytes(raw))
        perm.weight = 5
        out = W.pack_permutation(perm)
        self.assertEqual(struct.unpack_from(">I", out, 0x64)[0], 5)
        self.assertEqual(out[0x68:0x70], bytes(raw[0x68:0x70]))

    def test_rotation_unit_and_hemisphere(self):
        good = [(0.0, 0.0, 0.0, 1.0), unit(0, 0, 0.3, 1)]
        self.assertEqual(W.rotation_problems(good), [])
        flipped = [good[0], tuple(-c for c in good[1])]
        self.assertEqual(len(W.rotation_problems(flipped)), 1)
        self.assertIn("hemisphere", W.rotation_problems(flipped)[0])
        self.assertEqual(W.rotation_problems(W.align_hemispheres(flipped)), [])
        self.assertIn("norm", W.rotation_problems([(0.0, 0.0, 0.0, 0.9)])[0])
        with self.assertRaises(FormatError):
            W.encode_curve_values("rotation", [0], [(0.0, 0.0, 0.0, 0.9)], 1, "ps3")

    def test_permutation_problems(self):
        perm = new_switch_perm()
        curves = A.decode_permutation(perm)
        self.assertEqual(W.permutation_problems(perm, curves), [])
        curves.rotation[0].values[1] = tuple(-c for c in curves.rotation[0].values[1])
        self.assertTrue(any("hemisphere" in p for p in W.permutation_problems(perm, curves)))
        perm.nonlinear_offsets = [(1.0, 0.0, 0.0)] + perm.nonlinear_offsets[1:]
        self.assertTrue(any("row 0" in p for p in W.permutation_problems(perm)))
        perm.scale_blobs[0] = W.encode_curve_values("scale", [0], [(1.0,)], 5, "switch", codec=1)
        self.assertTrue(any("codec 1" in p for p in W.permutation_problems(perm)))
        perm.blend_weights = [(1.0, 0.0)]
        self.assertTrue(any("blend" in p for p in W.permutation_problems(perm)))


class AnimSets(unittest.TestCase):
    """W15-B3a."""

    def test_clean_set(self):
        for build in BUILDS:
            data = B.build_ans(build, [(2, 5, 700), (2, 1, 701)], {1: [(2, 6, 800)]}, parent=33)
            read = anims.parse_anim_set(data, build)
            self.assertEqual(len(read.weapons), 19 if build == "ps3" else 28)
            self.assertEqual(anims.anim_set_problems(read, {700, 701, 800}, {33}), [])

    def test_problems(self):
        read = anims.parse_anim_set(B.build_ans("switch", [(7, 5, 700), (2, 6, 999)], parent=44))
        found = anims.anim_set_problems(read, {700}, {33})
        self.assertTrue(any("type 7" in p for p in found))
        self.assertTrue(any("names no animation" in p for p in found))
        self.assertTrue(any("not an anim set" in p for p in found))
        read = anims.parse_anim_set(B.build_ans("ps3", [(2, 5, 700)], parent=7))  # handle 7
        self.assertTrue(any("itself" in p for p in anims.anim_set_problems(read)))
        read.weapons.pop()
        self.assertTrue(any("weapon maps" in p for p in anims.anim_set_problems(read)))
        read.parent = -1
        read.default.records.append(read.default.records[0])
        found = anims.anim_set_problems(read)
        self.assertTrue(any("repeats" in p for p in found))
        self.assertTrue(any("count 1 for 2" in p for p in found))

    def test_append_keeps_stored_order(self):
        for build in BUILDS:
            data = L.rebuild_sno(B.build_ans(build, [(2, 9, 700), (2, 3, 701)], {2: [(2, 6, 800)]}), "ans", build)
            out = anims.append_anim_tag(data, 5, 702, build=build)
            read = anims.parse_anim_set(out, build)
            self.assertEqual([(r.tag, r.value) for r in read.default.records], [(9, 700), (3, 701), (5, 702)])
            self.assertEqual([(r.tag, r.value) for r in read.weapons[2].records], [(6, 800)])
            self.assertEqual(anims.anim_set_problems(read), [])
            out = anims.append_anim_tag(out, 4, 900, weapon=0, build=build)
            read = anims.parse_anim_set(out, build)
            self.assertEqual([(r.tag, r.value) for r in read.weapons[0].records], [(4, 900)])
            self.assertTrue(L.is_canonical(out, "ans", build))
            with self.assertRaises(FormatError):
                anims.append_anim_tag(out, 9, 1, build=build)
            with self.assertRaises(FormatError):
                anims.append_anim_tag(out, 1, 1, weapon=len(read.weapons), build=build)

    def test_untouched_bytes_stay(self):
        data = L.rebuild_sno(B.build_ans("switch", [(2, 9, 700)], {27: [(2, 6, 800)]}, parent=12), "ans")
        out = anims.append_anim_tag(data, 1, 2, weapon=27)
        self.assertEqual(len(out), len(data) + 12)  # the last map grows at the end
        size_word = 0x10 + 0x20 + 16 * 27 + 4
        map_at = struct.unpack_from("<I", data, size_word - 4)[0] + 0x10
        changed = {i for i in range(len(data)) if out[i] != data[i]}
        self.assertEqual(changed, {size_word, map_at})  # the pair size and the count word
        self.assertEqual(struct.unpack_from("<I", out, map_at)[0], 2)

    def test_tag_map_bytes(self):
        self.assertEqual(anims.anim_tag_map_bytes([(5, 700)], "ps3"),
                         struct.pack(">IIIi", 1, 2, 5, 700))


if __name__ == "__main__":
    unittest.main()
