"""Stage 7 (``16_corrections6.md``): names, paths, packing, geometry enums, DSP, sub-object rules."""

import struct
import unittest

import builders as B
import d3sno
from d3sno import appearance as P
from d3sno import enums as E
from d3sno import records as R
from d3sno import scripts
from d3sno import structure as S
from d3sno import tags as T


class Names(unittest.TestCase):
    """A1, T1 to T3."""

    def test_label_when_no_constant(self):
        row = E.record_value("conversation", "conversation_kind", 12)
        self.assertEqual((row.constant, row.name), ("", "Quest Float"))
        record = R.Record("conversation", 0, b"", {"conversation_kind": 12}, "ps3")
        self.assertEqual(record.named()["conversation_kind"], "Quest Float")

    def test_constant_wins(self):
        row = E.record_value("treasure_class", "choose_kind", 0)
        self.assertEqual(row.name, row.constant)
        self.assertTrue(row.name)

    def test_open_and_hypothesis_rows_stay_unnamed(self):
        self.assertIsNone(E.record_value("objective", "flags", 1).name)
        flags = E.record_flags("objective", "flags", 0b10010, "switch")
        self.assertEqual((flags.names, flags.open_bits), ([], [1, 4]))
        dynamic = E.record_flags("actor", "coll_dynamic_no_overlap", 1 << 17)
        self.assertEqual(dynamic.open_bits, [17])
        self.assertEqual(E.record_flags("actor_event", "pin_flags", 1).open_bits, [0])

    def test_labelled_bit_is_named(self):
        labelled = [
            (key, v) for key, values in E.record_enums().items() for v in values.values()
            if v.kind == "bit" and not v.constant and v.status not in E.UNNAMED_STATUSES
        ]
        self.assertTrue(labelled)
        (record, field), row = labelled[0]
        flags = E.record_flags(record, field, 1 << row.value, "switch")
        self.assertEqual((flags.names, flags.open_bits), ([row.label], []))

    def test_only_four_rows_unnamed(self):
        unnamed = [
            (key, v.value) for key, values in E.record_enums().items() for v in values.values()
            if v.name is None
        ]
        self.assertEqual(sorted(unnamed), sorted([
            (("actor", "coll_dynamic_no_overlap"), 17), (("objective", "flags"), 1),
            (("objective", "flags"), 4), (("actor_event", "pin_flags"), 0),
        ]))

    def test_empty_core_name_is_normal(self):
        self.assertEqual(E.enum_name("tag_value:gMonsterRarity", 0), "Normal")
        self.assertEqual(T.value_name("marker", 0x82018, 0), "Normal")

    def test_ps3_event_minus_one(self):
        self.assertEqual(scripts.switch_event_id(-1, "ps3"), -1)
        read = scripts.parse_script(B.build_scr("ps3", [{"event": -1}], B.lua_chunk("ps3")))
        self.assertEqual(read.messages[0].event_name("ps3"), "Invalid")

    def test_stage_marked(self):
        self.assertGreaterEqual(d3sno.STAGE, 7)  # stage 8 keeps every stage-7 rule


class Packing(unittest.TestCase):
    """T7 and T12."""

    def read(self, build, record, raw):
        reader = S._Reader(bytes(0x10) + raw, build, ">" if build == "ps3" else "<")
        return reader.record(record, 0x10, record), reader.problems

    def test_ps3_node_has_its_own_packing(self):
        word = 3 | (1000 << 2)  # leaf, first triangle entry 1000
        values, problems = self.read("ps3", "dmMeshNodeP3", struct.pack(">IHH", word, 5, 0))
        self.assertEqual(values["data_bits"], {"axis": 3, "index": 1000})
        self.assertEqual((values["a"], values["b"], problems), (5, 0, []))
        # The Switch packing would see a count of 10 and an index of 15.
        self.assertNotEqual(S.unpack_bits("dmMeshNode", word)["index"], 1000)

    def test_switch_node_bounds_are_values(self):
        raw = struct.pack("<6hI", -5, 2, 3, 400, -1, 7, 2 | 3 << 2 | 9 << 6)
        values, problems = self.read("switch", "dmMeshNode", raw)
        self.assertEqual([values[k] for k in ("lowerX", "lowerY", "lowerZ", "upperX", "upperY", "upperZ")],
                         [-5, 2, 3, 400, -1, 7])
        self.assertEqual(values["uc_bits"], {"axis": 2, "triangleCount": 3, "index": 9})
        self.assertEqual(problems, [])

    def test_known_raw_rows_are_read(self):
        kinds = {(b, r, f.name): (f.kind, f.type) for (b, r), lay in S.layouts().items() for f in lay.fields}
        for build in ("switch", "ps3"):
            self.assertEqual(kinds[(build, "dmSubEdge", "twinOffset")], ("prim", "b"))
            self.assertEqual(kinds[(build, "AppearanceDefinition", "tUnwrapTimeStamp")], ("prim", "q"))
            self.assertEqual(kinds[(build, "DspEffect", "u8")][0], "dsp")
        self.assertEqual(kinds[("ps3", "DrawP3", "u34")][0], "raw")
        values, _ = self.read("switch", "dmSubEdge", b"\xfe" + bytes(S.layout("switch", "dmSubEdge").size - 1))
        self.assertEqual(values["twinOffset"], -2)

    def test_sample_handles(self):
        self.assertIsNone(S.sample_handle(0))
        self.assertIsNone(S.sample_handle(0xFFFFFF))
        self.assertEqual(S.sample_handle(0x123), 0x123)
        self.assertEqual(S._packed("AppearanceSample", "u4", (0xABC << 8) | 0x40),
                         {"dwZFraction": 0x40, "snoParticle": 0xABC})


def dsp(build, kind, slots, tail=b""):
    order = ">" if build == "ps3" else "<"
    raw = struct.pack(order + "ii", kind, 0)
    for minimum, delta in slots:
        raw += struct.pack(order + "ffIi", minimum, delta, 0, 0)
    raw += tail
    return raw.ljust(264, b"\0")


class Dsp(unittest.TestCase):
    """T8: DSP parameter names."""

    def read(self, build, raw):
        reader = S._Reader(bytes(0x10) + raw, build, ">" if build == "ps3" else "<")
        return reader.record("DspEffect", 0x10, "fx"), reader.problems

    def test_table(self):
        params = S.dsp_params()
        counts = {name: len(p) for name, p in params.values()}
        self.assertEqual(counts, {
            "chorus": 8, "compressor": 4, "distortion": 1, "echo": 5, "flange": 4, "highpass": 2,
            "lowpass": 2, "lowpass_simple": 1, "param_eq": 3, "sfx_reverb": 15,
        })
        self.assertEqual(params[3][1][1], "decay_ratio")

    def test_named_slots(self):
        for build in ("switch", "ps3"):
            values, problems = self.read(build, dsp(build, 5, [(100.0, 2.0), (1.0, 0.5)]))
            self.assertEqual(problems, [])
            self.assertEqual(sorted(values["parameters"]), ["cutoff", "resonance"])
            self.assertEqual(values["parameters"]["cutoff"],
                             {"minimum": 100.0, "delta": 2.0, "flags": 0, "source_scaler": 0})

    def test_unused_slots_must_be_zero(self):
        _, problems = self.read("switch", dsp("switch", 2, [(1.0, 0.0), (5.0, 0.0)]))
        self.assertEqual(len(problems), 1)
        _, problems = self.read("ps3", dsp("ps3", -1, [(1.0, 0.0)]))
        self.assertEqual(len(problems), 1)
        values, problems = self.read("ps3", dsp("ps3", -1, []))
        self.assertEqual((values["parameters"], problems), ({}, []))

    def test_type_names(self):
        self.assertEqual(E.geometry_value("DspEffect", "eDspType", -1).name, "no effect")
        self.assertTrue(E.geometry_value("DspEffect", "eDspType", 9).name)


class GeometryEnums(unittest.TestCase):
    """T8: geometry enum and flag names."""

    def test_table(self):
        rows = [v for values in E.geometry_enums().values() for v in values.values()]
        self.assertEqual(len(rows), 439)
        unnamed = [v for v in rows if v.name is None]
        self.assertTrue(all(v.status in E.UNNAMED_STATUSES for v in unnamed))
        self.assertEqual(sorted(v.value for v in unnamed), [76, 77, 78, 79, 80, 81, 82, 83, 84, 87, 88,
                                                            91, 98, 99, 100, 101])

    def test_every_record_has_a_layout(self):
        for record, field in E.geometry_enums():
            builds = [b for b in ("switch", "ps3") if (b, record) in S.layouts()]
            self.assertTrue(builds, record)
            members = {m.name for m in S.bitfields().get(record, ())}
            for build in builds:
                fields = {f.name for f in S.layout(build, record).fields}
                if record == "TextureDefinition" and field == "eUsage" and build == "ps3":
                    continue  # the ps3 texture record has no usage field
                self.assertIn(field, fields | members, (build, record, field))

    def test_markers(self):
        for build in ("switch", "ps3"):
            data = B.build_mrk(build, markers=[{"type": 50}, {"type": 76}, {"type": 999}])
            named = S.parse_structure(data, "mrk").named()
            types = [names["eType"] for record, _, names in named if record == "Marker"]
            self.assertEqual(types, ["MARKERTYPE_GIZMOLOC", None, None])

    def test_sub_object_flags(self):
        flags = E.geometry_flags("SubObject", "dwFlags", 1 << 8 | 1 << 15 | 1 << 20)
        self.assertEqual(flags.names, ["SUBOBJECT_CLOTH_BIT", "SUBOBJECT_PACK_GEOMETRY_BIT"])
        self.assertEqual(flags.unknown_bits, [20])
        self.assertEqual(S.geometry_names("SubObjectP3", {"dwFlags": 1 << 13})["dwFlags"].names,
                         ["SUBOBJECT_EDGE_GEOMETRY_BIT"])

    def test_packed_members(self):
        names = S.geometry_names("CompAndFormatUnion", {"u0": 0, "u0_bits": {"eCompression": 0, "eFormat": 0}})
        self.assertEqual(set(names), {"eCompression", "eFormat"})
        self.assertTrue(all(isinstance(v, str) for v in names.values()))


class SubObjects(unittest.TestCase):
    """T10, T11 and T13."""

    RIGID = [(0, 0, 0), (10, 0, 0), (0, 10, 0)]

    def test_flags_follow_the_data(self):
        app = P.parse_appearance(B.build_app_switch(), strict=True)
        self.assertEqual(app.warnings, [])

    def test_bit_15_without_packed_vertices(self):
        data = B.build_app_switch(sub_objects=[{"rigid": self.RIGID, "index_count": 3, "flags": 0}])
        self.assertIn("15/16", P.parse_appearance(data).warnings[0])

    def test_bit_16_without_skinning(self):
        data = B.build_app_switch(sub_objects=[
            {"rigid": self.RIGID, "index_count": 3, "flags": 1 << 15 | 1 << 16},
        ])
        warnings = P.parse_appearance(data).warnings
        self.assertTrue(any("bit 16" in w for w in warnings))

    def test_cloth_implies_bit_8_not_the_reverse(self):
        data = B.build_app_switch(sub_objects=[
            {"rigid": self.RIGID, "index_count": 3, "cloth": bytes(144), "flags": 1 << 15},
        ])
        self.assertIn("bit 8", P.parse_appearance(data).warnings[0])
        data = B.build_app_switch(sub_objects=[
            {"rigid": self.RIGID, "index_count": 3, "flags": 1 << 15 | 1 << 8},
        ])
        self.assertEqual(P.parse_appearance(data).warnings, [])

    def test_cloth_without_vertex_list_is_not_damage(self):
        data = B.build_app_switch(sub_objects=[
            {"index_count": 3, "cloth": bytes(144), "flags": 1 << 8},
        ])
        app = P.parse_appearance(data, strict=True)
        self.assertEqual(app.warnings, [])
        self.assertTrue(app.sub_objects[0].cloth_only)
        self.assertEqual(app.skipped_sub_objects, [0])
        self.assertIn("cloth", app.notes[0])

    def test_indices_without_vertices_or_cloth_stay_a_warning(self):
        data = B.build_app_switch(sub_objects=[{"index_count": 3, "flags": 0}])
        app = P.parse_appearance(data)
        self.assertIn("no vertices", app.warnings[0])
        self.assertFalse(app.sub_objects[0].cloth_only)

    def test_ps3_bounds_is_a_note(self):
        data = B.build_app_ps3(bounds_half=(1.0, 1.0, 1.0))
        app = P.parse_appearance(data, strict=True)
        self.assertEqual(app.warnings, [])
        self.assertEqual(len(app.notes), 1)

    def test_switch_bounds_still_strict(self):
        data = B.build_app_switch(sub_objects=[
            {"rigid": self.RIGID, "index_count": 3, "half": (0.5, 0.5, 0.5)},
        ])
        with self.assertRaises(d3sno.FormatError):
            P.parse_appearance(data, strict=True)


class Ps3Attributes(unittest.TestCase):
    """Part C."""

    def test_counts(self):
        statuses = [a.status for a in R.ps3_attributes().values()]
        self.assertEqual(len(statuses), 231)
        self.assertEqual({s: statuses.count(s) for s in set(statuses)},
                         {"PROVEN": 17, "SUPPORTED": 204, "HYPOTHESIS": 2, "UNRESOLVED": 8})

    def test_unresolved_have_candidates_and_no_name(self):
        for ps3_id in (78, 87, 128, 150, 199):
            row = R.ps3_attribute(ps3_id, safe_only=False)
            self.assertEqual(row.status, "UNRESOLVED")
            self.assertIsNone(row.name)
            self.assertFalse(row.resolved)
            self.assertGreaterEqual(len(row.candidates), 2)
            self.assertIsNone(R.ps3_attribute(ps3_id))
        self.assertEqual(R.ps3_attribute(150, safe_only=False).candidates,
                         ("Movement_Scalar", "Walking_Rate"))

    def test_unknown_names(self):
        for ps3_id in (31, 32, 33):
            row = R.ps3_attribute(ps3_id, safe_only=False)
            self.assertEqual((row.name, row.candidates), (None, ()))

    def test_hypotheses(self):
        self.assertEqual(R.ps3_attribute(13, safe_only=False).name, "Vitality")
        self.assertEqual(R.ps3_attribute(13, safe_only=False).status, "HYPOTHESIS")
        self.assertEqual(R.ps3_attribute(0).status, "SUPPORTED")


if __name__ == "__main__":
    unittest.main()
