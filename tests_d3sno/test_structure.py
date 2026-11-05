"""Stage 6, part B: table-driven layouts, packed words and the R3/R4 checks."""

import struct
import unittest

import builders as B
from d3sno import FormatError
from d3sno import structure as S

SIZES = {  # record sizes stated in the description
    ("switch", "AppearanceDefinition"): 552, ("ps3", "AppearanceDefinition"): 568,
    ("switch", "AnimationDefinition"): 56, ("ps3", "AnimationDefinition"): 64,
    ("switch", "SceneDefinition"): 536, ("ps3", "SceneDefinition"): 536,
    ("switch", "PhysMeshDefinition"): 48, ("ps3", "PhysMeshDefinition"): 48,
    ("switch", "CollisionMesh"): 112, ("ps3", "CollisionMesh"): 136,
    ("ps3", "TextureDefinition"): 0x260,
    ("switch", "SoundDefinition"): 168, ("ps3", "SoundDefinition"): 124,
    ("switch", "SoundBankDefinition"): 40, ("ps3", "SoundBankDefinition"): 28,
    ("switch", "WorldDefinition"): 192, ("ps3", "WorldDefinition"): 256,
    ("switch", "Marker"): 88, ("ps3", "Marker"): 88,
    ("switch", "MaterialDefinition"): 136, ("ps3", "MaterialDefinition"): 104,
    ("switch", "UberMaterial"): 104, ("ps3", "UberMaterial"): 72,
    ("switch", "BoneStructure"): 256, ("ps3", "BoneStructure"): 176,
    ("switch", "CollisionShape"): 80, ("ps3", "CollisionShape"): 68,
    ("switch", "ConstraintParameters"): 236, ("ps3", "ConstraintParameters"): 228,
    ("switch", "SubObject"): 240, ("ps3", "SubObjectP3"): 308,
    ("switch", "ClothStructure"): 144, ("ps3", "ClothStructureP3"): 100,
    ("switch", "AppearanceMaterial"): 24, ("ps3", "AppearanceMaterial"): 20,
    ("switch", "SubObjectAppearance"): 248, ("ps3", "SubObjectAppearance"): 184,
    ("switch", "SoundSample"): 16, ("ps3", "SoundSampleP3"): 40,
    ("ps3", "SoundPermutationP3"): 44,
}


def field_at(build, record, name):
    return next(f for f in S.layout(build, record).fields if f.name == name)


class Table(unittest.TestCase):
    def test_sizes_from_the_description(self):
        for (build, record), size in SIZES.items():
            self.assertEqual(S.layout(build, record).size, size, (build, record))

    def test_every_type_has_a_top_record(self):
        for extension, record in S.TOP_RECORDS.items():
            for build in ("ps3", "switch"):
                self.assertEqual(S.layout(build, record).root_ext, extension)
        self.assertEqual(len(S.EXTENSIONS), 11)

    def test_no_overlap_and_inside(self):
        for (build, record), lay in S.layouts().items():
            self.assertEqual(S.overlapping_fields(build, record), [], (build, record))
            for f in lay.fields:
                self.assertLessEqual(f.offset + f.size, lay.size, (build, record, f.name))

    def test_referenced_records_exist(self):
        for (build, record), lay in S.layouts().items():
            for f in lay.fields:
                if f.kind == "struct":
                    self.assertTrue(S.layout(build, f.type))
                if f.kind == "pair" and f.points_to.startswith("array:"):
                    element = f.points_to[6:]
                    self.assertTrue(element in S.VALUE_ARRAYS or S.layout(build, element))

    def test_ps3_positions_from_the_description(self):
        # Body offsets in the description; the table is relative to the record (body) start.
        app = {"nNumLooks": 0x178, "nNumMaterials": 0x17C, "serAppearanceMaterials": 0x180,
               "serAppearanceLooks": 0x190, "dwMachineTime": 0x1D0, "dwFlags": 0x230}
        for name, offset in app.items():
            self.assertEqual(field_at("ps3", "AppearanceDefinition", name).offset, offset, name)
        sound = {"serPermutations": 0x40, "nPermutationCount": 0x48, "eType": 0x50,
                 "serDspEffects": 0x5C, "dwCooldownTime": 0x68}
        for name, offset in sound.items():
            self.assertEqual(field_at("ps3", "SoundDefinition", name).offset, offset, name)
        self.assertEqual(field_at("ps3", "SoundDefinition", "tSoundPlayHistory").kind, "zero")
        self.assertEqual(field_at("ps3", "WorldDefinition", "rgbaSkyColor").offset, 0x60)

    def test_slots_of_four_bytes_on_ps3(self):
        self.assertEqual(field_at("ps3", "SoundBankDefinition", "ptSoundSamples").size, 4)
        self.assertEqual(field_at("switch", "SoundBankDefinition", "ptSoundSamples").size, 8)


class Bits(unittest.TestCase):
    def test_mesh_node(self):
        word = 2 | (5 << 2) | (1234 << 6)
        self.assertEqual(S.unpack_bits("dmMeshNode", word),
                         {"axis": 2, "triangleCount": 5, "index": 1234})

    def test_sound_sample_word(self):
        word = 1 | (0 << 1) | (7 << 2) | (300 << 12) | (11 << 22)
        self.assertEqual(S.unpack_bits("CompAndFormatUnion", word), {
            "bIsVerifiedVO": 1, "bIsTemporaryVO": 0, "eCompression": 7,
            "nCompressionQuality": 300, "eFormat": 11,
        })

    def test_appearance_sample_words(self):
        self.assertEqual(S._packed("AppearanceSample", "u0", 0x123456_80),
                         {"dwZFraction": 0x80, "snoSurface": 0x123456})
        self.assertEqual(S._packed("AppearanceSample", "u8", (0xABCDEF << 8) | (3 << 4) | 9),
                         {"dwClutterDensity": 9, "dwMask": 3, "dwColor": 0xABCDEF})


def sbk(build, samples, extra=b"", slot=0, gap=0):
    """A sound bank laid out as the description states."""
    order = ">" if build == "ps3" else "<"
    top = 28 if build == "ps3" else 40
    data = bytearray(B.sno_head(46, order) + bytes(top - 12))
    data += bytes(gap)
    start = len(data)
    data += b"".join(samples) + extra
    struct.pack_into(order + "II", data, 0x1C, start - 0x10, len(b"".join(samples)))
    struct.pack_into(order + "i", data, 0x10 + (0x14 if build == "ps3" else 0x20), len(samples))
    if slot:
        struct.pack_into(order + "I", data, 0x10 + 0x18, slot)
    return bytes(data)


def switch_sample(sample_hash, word, decompressed, stored):
    return struct.pack("<4I", sample_hash, word, decompressed, stored)


def ps3_sample(sample_hash, compression, fmt, stored):
    return struct.pack(">10I", sample_hash, 1, 0, compression, 50, 1000, fmt, 0, stored, 0)


class SoundBanks(unittest.TestCase):
    def test_switch(self):
        word = 1 | (3 << 2) | (11 << 22)
        data = sbk("switch", [switch_sample(0xAA, word, 900, 400)])
        parsed = S.parse_structure(data, "sbk")
        sample = parsed.root["serSoundSamples"][0]
        self.assertEqual(sample["dwHash"], 0xAA)
        bits = sample["tCompAndFormatUnion"]["u0_bits"]
        self.assertEqual((bits["eCompression"], bits["eFormat"], bits["bIsVerifiedVO"]), (3, 11, 1))
        self.assertEqual(parsed.root["nSoundSampleCount"], 1)
        self.assertTrue(S.verify_structure(data, "sbk").ok)

    def test_ps3(self):
        data = sbk("ps3", [ps3_sample(0xBB, 2, 5, 77), ps3_sample(0xCC, 0, 0, 1)])
        parsed = S.parse_structure(data, "sbk")
        self.assertEqual(parsed.build, "ps3")
        self.assertEqual(len(parsed.root["serSoundSamples"]), 2)
        report = S.verify_structure(data, "sbk")
        self.assertEqual(report.problems, [])

    def test_run_time_slot_must_be_zero(self):
        report = S.verify_structure(sbk("ps3", [ps3_sample(1, 0, 0, 1)], slot=5), "sbk")
        self.assertEqual(len(report.problems), 1)
        self.assertIn("R3", report.problems[0])

    def test_uncovered_byte_must_be_zero(self):
        data = bytearray(sbk("switch", [switch_sample(1, 0, 0, 0)]))
        data[0x10 + 0x14] = 1  # between the pair and the slot
        report = S.verify_structure(bytes(data), "sbk")
        self.assertTrue(any("R3" in p for p in report.problems))

    def test_tiling(self):
        sample = [switch_sample(1, 0, 0, 0)]
        self.assertTrue(S.verify_structure(sbk("switch", sample, gap=8), "sbk").ok)
        self.assertFalse(S.verify_structure(sbk("switch", sample, gap=16), "sbk").ok)
        self.assertTrue(S.verify_structure(sbk("switch", sample, extra=bytes(4)), "sbk").ok)
        self.assertFalse(S.verify_structure(sbk("switch", sample, extra=b"\1"), "sbk").ok)
        self.assertFalse(S.verify_structure(sbk("switch", sample, extra=bytes(16)), "sbk").ok)
        bad_gap = bytearray(sbk("switch", sample, gap=8))
        bad_gap[0x10 + 40] = 1
        self.assertFalse(S.verify_structure(bytes(bad_gap), "sbk").ok)

    def test_tiling_rule(self):
        data = bytes(64)
        self.assertEqual(S.tiling_problems(64, 16, [(16, 16, "a"), (32, 32, "b")], data), [])
        self.assertEqual(S.tiling_problems(64, 16, [(16, 32, "a"), (16, 32, "b"), (48, 16, "c")], data), [])
        self.assertTrue(S.tiling_problems(64, 16, [(16, 32, "a"), (40, 24, "b")], data))

    def test_layout_errors(self):
        whole = sbk("switch", [switch_sample(1, 0, 0, 0)])
        broken = bytearray(whole)
        struct.pack_into("<I", broken, 0x20, 15)  # not a whole sample
        with self.assertRaises(FormatError):
            S.parse_structure(bytes(broken), "sbk")
        self.assertFalse(S.verify_structure(bytes(broken), "sbk").ok)
        with self.assertRaises(FormatError):
            S.parse_structure(whole[:0x30], "sbk")
        with self.assertRaises(FormatError):
            S.parse_structure(whole, "gam")


class OtherTypes(unittest.TestCase):
    def test_markers(self):
        for build in ("ps3", "switch"):
            parsed = S.parse_structure(B.build_mrk(build, circles=[(1.0, 2.0, 3.0)]), "mrk")
            self.assertEqual(parsed.record, "MarkerSetDefinition")
            self.assertEqual(S.verify_structure(B.build_mrk(build), "mrk").problems, [])

    def test_anim_set_tag_maps(self):
        for build in ("ps3", "switch"):
            data = B.build_ans(build, [(2, 5, 700)])
            parsed = S.parse_structure(data, "ans")
            self.assertEqual(parsed.root["serTagMapAnimDefault"].get(5).value, 700)
            self.assertTrue(S.verify_structure(data, "ans").ok)

    def test_material(self):
        for build in ("ps3", "switch"):
            data = B.build_mat(build)
            self.assertEqual(S.parse_structure(data, "mat").record, "MaterialDefinition")
            self.assertTrue(S.verify_structure(data, "mat").ok)

    def test_fuzz_raises_only_format_errors(self):
        import random
        rng = random.Random(6)
        base = sbk("switch", [switch_sample(1, 3, 4, 5)] * 2)
        for _ in range(300):
            data = bytearray(base)
            for _ in range(3):
                data[rng.randrange(len(data))] = rng.randrange(256)
            try:
                S.verify_structure(bytes(data), "sbk")
            except FormatError:
                pass


if __name__ == "__main__":
    unittest.main()
