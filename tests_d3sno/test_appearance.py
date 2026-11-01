import base64
import json
import struct
import unittest

import builders as B
from d3sno import FormatError
from d3sno import appearance as P

BONES = [(0xAA, -1, (1.0, 2.0, 3.0)), (0xBB, 0, (2.0, 2.0, 3.0)), (0xCC, 1, (2.0, 4.0, 3.0))]
SKIN = {
    "palette": [2, 1],
    "vertices": [(0, 0, 0), (10, 0, 0), (0, 10, 0)],
    "indices": [0, 1, 2],
}


class Ps3(unittest.TestCase):
    def test_rigid(self):
        app = P.parse_appearance(B.build_app_ps3())
        self.assertEqual(app.build, "ps3")
        sub = app.sub_objects[0]
        self.assertEqual(sub.name, "Body")
        draw = sub.draws[0]
        self.assertFalse(draw.skinned)
        self.assertEqual(draw.positions[1], (6.0, 2.0, 3.0))
        self.assertEqual(draw.normals[0], (0.5, 0.0, -0.5))
        self.assertEqual(draw.uv1[0], (1.0, 0.0))
        self.assertEqual(draw.uv2_raw[0], (9, 10))
        self.assertEqual(draw.colours_1[0], bytes((1, 2, 3, 4)))
        self.assertEqual(draw.colours_2[0], bytes((5, 6, 7, 8)))
        self.assertEqual(draw.indices, [0, 1, 2])
        point = app.hardpoints[0]
        self.assertEqual((point.hash, point.flag), (0x11223344, 2))
        self.assertEqual((point.bone, point.position), (-1, (1.0, 2.0, 3.0)))
        self.assertEqual(point.rotation, (0.0, 0.0, 0.0, 1.0))

    def test_skinned(self):
        app = P.parse_appearance(B.build_app_ps3(skinned=True, bones=BONES))
        draw = app.sub_objects[0].draws[0]
        self.assertTrue(draw.skinned)
        self.assertEqual(draw.palette, [1, 2])
        self.assertEqual(draw.weights[0], (200, 55, 0, 0))
        self.assertEqual(draw.local_joints[0], (0, 1, 0, 0))
        self.assertEqual(draw.joints[0], (1, 2, 0, 0))
        self.assertEqual(draw.normals[0], (0.5, 0.0, -0.5))
        self.assertEqual(draw.colours_2[0], bytes((5, 6, 7, 8)))
        self.assertEqual(draw.uv1[0], (1.0, 0.0))
        self.assertEqual(app.warnings, [])

    def test_bones(self):
        app = P.parse_appearance(B.build_app_ps3(bones=BONES))
        self.assertEqual(app.bones[2].parent, 1)
        self.assertEqual(app.bones[1].name_hash, 0xBB)
        self.assertEqual(app.bones[1].translation, (2.0, 2.0, 3.0))

    def test_bone_transforms(self):
        bone = P.parse_appearance(B.build_app_ps3(bones=BONES)).bones[1]
        self.assertEqual(bone.rest.t, (2.0, 2.0, 3.0))
        self.assertEqual(bone.rest_inverse.t, (-2.0, -2.0, -3.0))
        self.assertEqual(bone.local.t, (1.0, 0.0, 0.0))
        self.assertEqual(bone.rest.q, (0.0, 0.0, 0.0, 1.0))
        self.assertIsNone(bone.bind)
        self.assertEqual(bone.skin_inverse, bone.rest_inverse)

    def test_bone_chain_is_checked(self):
        data = bytearray(B.build_app_ps3(bones=BONES))
        record = struct.unpack_from(">I", data, 0x28)[0] + 0x10 + 176
        struct.pack_into(">f", data, record + 0x70 + 16, 9.0)
        app = P.parse_appearance(bytes(data))
        self.assertIn("bone 1", app.warnings[0])
        with self.assertRaises(FormatError):
            P.parse_appearance(bytes(data), strict=True)

    def test_bone_range_two_ranges_in_order(self):
        data = B.build_app_ps3(skinned=True, bones=BONES, ranges=((96, 0), (48, 48)))
        draw = P.parse_appearance(data).sub_objects[0].draws[0]
        self.assertEqual(draw.palette, [2, 0])
        self.assertEqual(draw.joints[0], (2, 0, 0, 0))

    def test_bone_range_size_zero_is_unused(self):
        data = B.build_app_ps3(skinned=True, bones=BONES, ranges=((0, 480), (96, 0)),
                               weights=(200, 55, 0, 0))
        self.assertEqual(P.parse_appearance(data).sub_objects[0].draws[0].palette, [0, 1])

    def test_bone_range_multiple_of_48(self):
        for ranges in (((47, 0), (48, 0)), ((48, 0), (47, 0))):
            with self.assertRaises(FormatError):
                P.parse_appearance(B.build_app_ps3(skinned=True, bones=BONES, ranges=ranges))

    def test_bone_range_past_skeleton(self):
        with self.assertRaises(FormatError):
            P.parse_appearance(B.build_app_ps3(skinned=True, bones=BONES, ranges=((96, 0), (96, 0))))

    def test_local_joint_past_palette(self):
        with self.assertRaises(FormatError):
            P.parse_appearance(B.build_app_ps3(skinned=True, bones=BONES, ranges=((48, 0), (48, 0))))

    def test_weight_zero_needs_no_joint(self):
        data = B.build_app_ps3(skinned=True, bones=BONES, ranges=((48, 0), (48, 0)),
                               weights=(255, 0, 0, 0))
        self.assertEqual(P.parse_appearance(data).sub_objects[0].draws[0].joints[0], (1, 0, 0, 0))

    def test_uv2(self):
        draw = P.parse_appearance(B.build_app_ps3()).sub_objects[0].draws[0]
        draw.uv2_raw = [(0x8000 + 8192, 0x8000 + 4096)]
        self.assertEqual(draw.uv2(), [(1.0, 0.5)])
        self.assertEqual(draw.uv2(0.25), [(1.0, 0.125)])

    def test_stream_a_padding(self):
        P.parse_appearance(B.build_app_ps3(pad_a=False))

    def test_stream_b_padding(self):
        data = B.build_app_ps3(skinned=True, bones=BONES, b_extra=8)
        draw = P.parse_appearance(data).sub_objects[0].draws[0]
        self.assertEqual(draw.weights, [(200, 55, 0, 0)] * 3)

    def test_stream_b_padded_too_far(self):
        with self.assertRaises(FormatError):
            P.parse_appearance(B.build_app_ps3(skinned=True, bones=BONES, b_extra=16))

    def test_stream_b_too_short(self):
        data = bytearray(B.build_app_ps3(skinned=True, bones=BONES))
        sub = struct.unpack_from(">I", data, 0x74)[0] + 0x10
        draw = struct.unpack_from(">I", data, sub + 0x100)[0] + 0x10
        struct.pack_into(">I", data, draw + 0x58, 23)
        with self.assertRaises(FormatError):
            P.parse_appearance(bytes(data))

    def test_stream_b_must_be_empty_when_rigid(self):
        with self.assertRaises(FormatError):
            P.parse_appearance(B.build_app_ps3(b_extra=8))

    def test_bad_format_byte(self):
        with self.assertRaises(FormatError):
            P.parse_appearance(B.build_app_ps3(fmt=0x99))

    def test_formats_never_mix(self):
        with self.assertRaises(FormatError):
            P.parse_appearance(B.build_app_ps3(bones=BONES, extra_draw_format=0x83))

    def test_skinned_needs_bones(self):
        with self.assertRaises(FormatError):
            P.parse_appearance(B.build_app_ps3(skinned=True))

    def test_descriptor_counts(self):
        with self.assertRaises(FormatError):
            P.parse_appearance(B.build_app_ps3(bad_descriptor=True))

    def test_index_count_divisible_by_three(self):
        with self.assertRaises(FormatError):
            P.parse_appearance(B.build_app_ps3(index_count=2))

    def test_index_past_vertices(self):
        with self.assertRaises(FormatError):
            P.parse_appearance(B.build_app_ps3(indices=(0, 1, 3)))

    def test_bounds_warning_and_strict(self):
        data = B.build_app_ps3(bounds_half=(1.0, 1.0, 1.0))
        app = P.parse_appearance(data)
        self.assertTrue(app.warnings)
        with self.assertRaises(FormatError):
            P.parse_appearance(data, strict=True)

    def test_bounds_tolerance_is_two_raw_units(self):
        # The furthest vertex is at 6.0 on x with scale 0.5, so 5.0 passes with tolerance 1.
        vertices = ((10, 0, 0),)
        data = B.build_app_ps3(vertices=vertices, indices=(), index_count=0,
                               bounds_half=(5.0, 5.0, 5.0))
        self.assertEqual(P.parse_appearance(data).warnings, [])
        data = B.build_app_ps3(vertices=vertices, indices=(), index_count=0,
                               bounds_half=(4.9, 5.0, 5.0))
        self.assertTrue(P.parse_appearance(data).warnings)

    def test_weights_checked(self):
        data = B.build_app_ps3(skinned=True, bones=BONES, weights=(55, 200, 0, 0))
        self.assertTrue(P.parse_appearance(data).warnings)
        data = B.build_app_ps3(skinned=True, bones=BONES, weights=(100, 55, 0, 0))
        self.assertTrue(P.parse_appearance(data).warnings)


class Switch(unittest.TestCase):
    def test_rigid(self):
        app = P.parse_appearance(B.build_app_switch())
        self.assertEqual(app.build, "switch")
        draw = app.sub_objects[0].draws[0]
        self.assertEqual(draw.positions[1], (6.0, 2.0, 3.0))
        self.assertEqual(draw.normals[0], (0.5, 0.0, -0.5))
        self.assertEqual(draw.indices, [0, 1, 2])
        self.assertEqual(app.sub_objects[0].format_bits, 0x55)

    def test_running_index_sum(self):
        vertices = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)]
        app = P.parse_appearance(B.build_app_switch(sub_objects=[
            {"rigid": vertices, "index_count": 3, "indices": (0, 1, 2)},
            {"rigid": vertices, "index_count": 6, "indices": (3, 2, 1, 0, 1, 2)},
        ]))
        self.assertEqual(app.sub_objects[1].draws[0].indices, [3, 2, 1, 0, 1, 2])

    def test_index_only_sub_object_is_skipped_but_counted(self):
        vertices = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
        app = P.parse_appearance(B.build_app_switch(sub_objects=[
            {"index_count": 3, "indices": (9, 9, 9)},
            {"rigid": vertices, "index_count": 3, "indices": (2, 1, 0)},
        ]))
        self.assertEqual(app.skipped_sub_objects, [0])
        self.assertEqual(app.sub_objects[0].draws, [])
        self.assertEqual(app.sub_objects[1].draws[0].indices, [2, 1, 0])
        self.assertTrue(app.warnings)

    def test_pool_too_short(self):
        data = bytearray(B.build_app_switch(pool_spare=0))
        offset, size = struct.unpack_from("<II", data, 0x220)
        struct.pack_into("<II", data, 0x220, offset, size - 2)
        with self.assertRaises(FormatError):
            P.parse_appearance(bytes(data))

    def test_skinned(self):
        app = P.parse_appearance(B.build_app_switch(bones=BONES, sub_objects=[{"skin": SKIN}]))
        draw = app.sub_objects[0].draws[0]
        self.assertTrue(draw.skinned)
        self.assertEqual(draw.palette, [2, 1])
        self.assertEqual(draw.joints[0], (2, 1, 0, 0))
        self.assertEqual(draw.weights[0], (200, 55, 0, 0))
        self.assertEqual(draw.indices, [0, 1, 2])

    def test_palette_index_past_count(self):
        skin = dict(SKIN, slots=(5, 0, 0, 0))
        with self.assertRaises(FormatError):
            P.parse_appearance(B.build_app_switch(bones=BONES, sub_objects=[{"skin": skin}]))

    def test_palette_entry_not_a_bone(self):
        skin = dict(SKIN, palette=[9, 1])
        with self.assertRaises(FormatError):
            P.parse_appearance(B.build_app_switch(bones=BONES, sub_objects=[{"skin": skin}]))

    def test_influences(self):
        raw = bytes((1, 200, 2, 55, 0, 0)) * 3
        app = P.parse_appearance(B.build_app_switch(sub_objects=[
            {"rigid": [(0, 0, 0), (1, 0, 0), (0, 1, 0)], "index_count": 3, "influences": raw},
        ]))
        self.assertEqual(app.sub_objects[0].influences[0], ((1, 200), (2, 55), (0, 0)))

    def test_influence_count_is_its_own(self):
        app = P.parse_appearance(B.build_app_switch(sub_objects=[
            {"rigid": [(0, 0, 0), (1, 0, 0), (0, 1, 0)], "index_count": 3,
             "influences": bytes(6)},
        ]))
        self.assertEqual(len(app.sub_objects[0].influences), 1)
        self.assertEqual(app.warnings, [])

    def test_bounds(self):
        app = P.parse_appearance(B.build_app_switch())
        self.assertEqual(app.sub_objects[0].bounds_half, (1000.0, 1000.0, 1000.0))

    def test_bounds_warning(self):
        data = B.build_app_switch(sub_objects=[
            {"rigid": [(0, 0, 0), (10, 0, 0), (0, 10, 0)], "index_count": 3,
             "half": (0.5, 0.5, 0.5)},
        ])
        self.assertIn("outside bounds", P.parse_appearance(data).warnings[0])
        with self.assertRaises(FormatError):
            P.parse_appearance(data, strict=True)

    def test_influence_size(self):
        with self.assertRaises(FormatError):
            P.parse_appearance(B.build_app_switch(sub_objects=[
                {"rigid": [(0, 0, 0)], "influences": b"\1\2\3"},
            ]))

    def test_bones_at_switch_offsets(self):
        app = P.parse_appearance(B.build_app_switch(bones=BONES))
        self.assertEqual(app.bones[1].parent, 0)
        self.assertEqual(app.bones[2].translation, (2.0, 4.0, 3.0))
        self.assertEqual(app.bones[2].local.t, (0.0, 2.0, 0.0))
        self.assertEqual(app.bones[2].bind.t, (2.0, 4.0, 3.0))
        self.assertEqual(app.bones[2].skin_inverse.t, (-2.0, -4.0, -3.0))


class TopLevel(unittest.TestCase):
    def test_pair_size_must_match_count(self):
        data = bytearray(B.build_app_ps3())
        struct.pack_into(">I", data, 0x70, 2)
        with self.assertRaises(FormatError):
            P.parse_appearance(bytes(data))

    def test_bone_pair_size(self):
        data = bytearray(B.build_app_ps3(bones=BONES))
        struct.pack_into(">i", data, 0x24, 2)
        with self.assertRaises(FormatError):
            P.parse_appearance(bytes(data))

    def test_hardpoint_bone_out_of_range_warns(self):
        data = bytearray(B.build_app_ps3())
        record = struct.unpack_from(">I", data, 0xCC)[0] + 0x10
        struct.pack_into(">i", data, record + 8, 5)
        self.assertIn("hardpoint 0", P.parse_appearance(bytes(data)).warnings[0])

    def second_list(self, data, order):
        """Point the second list at the first list's records."""
        out = bytearray(data)
        count, offset, size = struct.unpack_from(order + "iII", out, 0x70)
        struct.pack_into(order + "iII", out, 0x88, count, offset, size)
        return bytes(out)

    def test_second_sub_object_list(self):
        app = P.parse_appearance(self.second_list(B.build_app_ps3(), ">"))
        self.assertEqual(len(app.extra_sub_objects), 1)
        self.assertEqual(app.extra_sub_objects[0].name, "Body")
        self.assertEqual(len(app.sub_objects), 1)

    def test_second_list_is_left_out_of_the_mesh_export(self):
        app = P.parse_appearance(self.second_list(B.build_app_ps3(), ">"))
        self.assertEqual(P.to_obj(app).count("o Body"), 1)

    def test_second_list_on_switch_continues_the_index_pool(self):
        data = B.build_app_switch(pool_spare=6)
        app = P.parse_appearance(self.second_list(data, "<"))
        self.assertEqual(app.extra_sub_objects[0].draws[0].indices, [0, 0, 0])

    def test_second_list_size(self):
        data = bytearray(B.build_app_ps3())
        struct.pack_into(">i", data, 0x88, 1)
        with self.assertRaises(FormatError):
            P.parse_appearance(bytes(data))

    def test_empty_ps3_name_stays_empty(self):
        data = bytearray(B.build_app_ps3())
        table = struct.unpack_from(">I", data, 0x74)[0] + 0x10
        data[table + 0x48:table + 0x48 + 5] = bytes(5)
        data[table + 0x4C:table + 0x4C + 4] = b"Old\0"
        self.assertIsNone(P.parse_appearance(bytes(data)).sub_objects[0].name)

    def test_hardpoint_pair_size(self):
        data = bytearray(B.build_app_ps3())
        struct.pack_into(">i", data, 0xC8, 2)
        with self.assertRaises(FormatError):
            P.parse_appearance(bytes(data))

    def test_parent_must_be_earlier(self):
        with self.assertRaises(FormatError):
            P.parse_appearance(B.build_app_ps3(bones=[(1, 0, (0.0, 0.0, 0.0))]))

    def test_pair_out_of_bounds(self):
        data = bytearray(B.build_app_switch())
        struct.pack_into("<I", data, 0x78, 0xFFFFF0)
        struct.pack_into("<I", data, 0x74, 0xFFFFF0)
        with self.assertRaises(FormatError):
            P.parse_appearance(bytes(data))

    def test_position_outside_float32(self):
        data = bytearray(B.build_app_switch())
        offset = struct.unpack_from("<I", data, 0x74)[0] + 0x10
        struct.pack_into("<f", data, offset + 0xE4, 3e38)
        with self.assertRaises(FormatError):
            P.parse_appearance(bytes(data))

    def test_bone_translation_nan(self):
        with self.assertRaises(FormatError):
            P.parse_appearance(B.build_app_ps3(bones=[(1, -1, (float("nan"), 0.0, 0.0))]))

    def test_build_from_version(self):
        self.assertEqual(P.parse_appearance(B.build_app_ps3(version=295)).build, "ps3")
        with self.assertRaises(FormatError):
            P.parse_appearance(B.build_app_ps3(version=999), strict=True)


class Export(unittest.TestCase):
    def test_obj(self):
        text = P.to_obj(P.parse_appearance(B.build_app_ps3()))
        lines = text.splitlines()
        self.assertIn("o Body", lines)
        self.assertEqual(sum(l.startswith("v ") for l in lines), 3)
        self.assertEqual(sum(l.startswith("vn ") for l in lines), 3)
        self.assertEqual(sum(l.startswith("vt ") for l in lines), 3)
        self.assertIn("f 1/1/1 2/2/2 3/3/3", lines)

    def test_obj_indices_continue_across_objects(self):
        vertices = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
        app = P.parse_appearance(B.build_app_switch(sub_objects=[
            {"rigid": vertices, "index_count": 3},
            {"rigid": vertices, "index_count": 3},
        ]))
        lines = P.to_obj(app).splitlines()
        self.assertIn("o subobject_1", lines)
        self.assertIn("f 4/4/4 5/5/5 6/6/6", lines)

    def test_flip_v_defaults_off(self):
        app = P.parse_appearance(B.build_app_switch())
        self.assertIn("vt 1.000000 0.000000", P.to_obj(app).splitlines())
        self.assertIn("vt 1.000000 1.000000", P.to_obj(app, flip_v=True).splitlines())

    def accessor_values(self, doc, index, fmt):
        accessor = doc["accessors"][index]
        view = doc["bufferViews"][accessor["bufferView"]]
        uri = doc["buffers"][0]["uri"]
        blob = base64.b64decode(uri.split(",", 1)[1])
        raw = blob[view["byteOffset"]:view["byteOffset"] + view["byteLength"]]
        return list(struct.iter_unpack("<" + fmt, raw))

    def test_gltf_structure(self):
        doc = json.loads(P.to_gltf_json(P.parse_appearance(B.build_app_ps3())))
        self.assertEqual(doc["asset"]["version"], "2.0")
        primitive = doc["meshes"][0]["primitives"][0]
        positions = self.accessor_values(doc, primitive["attributes"]["POSITION"], "3f")
        self.assertEqual(positions[1], (6.0, 2.0, 3.0))
        accessor = doc["accessors"][primitive["attributes"]["POSITION"]]
        self.assertEqual(accessor["min"], [1.0, 2.0, 3.0])
        self.assertEqual(accessor["max"], [6.0, 7.0, 3.0])
        self.assertEqual(self.accessor_values(doc, primitive["indices"], "H"), [(0,), (1,), (2,)])
        for view in doc["bufferViews"]:
            self.assertEqual(view["byteOffset"] % 4, 0)
        self.assertEqual(
            doc["buffers"][0]["byteLength"],
            len(base64.b64decode(doc["buffers"][0]["uri"].split(",", 1)[1])),
        )

    def test_gltf_normals_are_unit(self):
        doc = P.to_gltf(P.parse_appearance(B.build_app_ps3()))
        primitive = doc["meshes"][0]["primitives"][0]
        for n in self.accessor_values(doc, primitive["attributes"]["NORMAL"], "3f"):
            self.assertAlmostEqual(sum(c * c for c in n), 1.0, places=5)

    def test_gltf_skeleton_and_skin_on_switch(self):
        app = P.parse_appearance(B.build_app_switch(bones=BONES, sub_objects=[{"skin": SKIN}]))
        doc = P.to_gltf(app)
        self.assertEqual(doc["nodes"][0]["translation"], [1.0, 2.0, 3.0])
        self.assertEqual(doc["nodes"][0]["rotation"], [0.0, 0.0, 0.0, 1.0])
        self.assertEqual(doc["nodes"][1]["translation"], [1.0, 0.0, 0.0])
        self.assertEqual(doc["nodes"][1]["children"], [2])
        self.assertEqual(doc["skins"][0]["joints"], [0, 1, 2])
        self.assertEqual(doc["nodes"][3]["skin"], 0)
        attributes = doc["meshes"][0]["primitives"][0]["attributes"]
        self.assertEqual(self.accessor_values(doc, attributes["JOINTS_0"], "4H")[0], (2, 1, 0, 0))
        weights = self.accessor_values(doc, attributes["WEIGHTS_0"], "4f")[0]
        self.assertAlmostEqual(sum(weights), 1.0, places=5)
        matrices = self.accessor_values(doc, doc["skins"][0]["inverseBindMatrices"], "16f")
        self.assertEqual(matrices[0][12:15], (-1.0, -2.0, -3.0))

    def test_gltf_ps3_skin_uses_the_skeleton_and_flags_the_bind(self):
        app = P.parse_appearance(B.build_app_ps3(skinned=True, bones=BONES))
        doc = P.to_gltf(app)
        self.assertTrue(doc["skins"][0]["extras"]["bind_from_rest_inverse"])
        attributes = doc["meshes"][0]["primitives"][0]["attributes"]
        self.assertEqual(self.accessor_values(doc, attributes["JOINTS_0"], "4H")[0], (1, 2, 0, 0))
        matrices = self.accessor_values(doc, doc["skins"][0]["inverseBindMatrices"], "16f")
        self.assertEqual(matrices[1][12:15], (-2.0, -2.0, -3.0))
        self.assertEqual(len(doc["nodes"]), 4)

    def test_gltf_nodes_carry_parent_relative_transforms(self):
        doc = P.to_gltf(P.parse_appearance(B.build_app_ps3(bones=BONES)))
        node = doc["nodes"][2]
        self.assertEqual(node["translation"], [0.0, 2.0, 0.0])
        self.assertEqual(node["rotation"], [0.0, 0.0, 0.0, 1.0])
        self.assertEqual(node["scale"], [1.0, 1.0, 1.0])

    def test_gltf_uv2_on_request(self):
        app = P.parse_appearance(B.build_app_ps3())
        plain = P.to_gltf(app)["meshes"][0]["primitives"][0]["attributes"]
        self.assertNotIn("TEXCOORD_1", plain)
        doc = P.to_gltf(app, lightmap_v_scale=0.5)
        attributes = doc["meshes"][0]["primitives"][0]["attributes"]
        self.assertEqual(
            self.accessor_values(doc, attributes["TEXCOORD_1"], "2f")[0],
            ((9 - 0x8000) / 8192, (10 - 0x8000) / 8192 * 0.5),
        )

    def test_gltf_rejects_a_transform_it_cannot_hold(self):
        data = bytearray(B.build_app_ps3(bones=BONES))
        record = struct.unpack_from(">I", data, 0x28)[0] + 0x10
        struct.pack_into(">f", data, record + 0x70 + 16, float("nan"))
        with self.assertRaises(FormatError):
            P.to_gltf(P.parse_appearance(bytes(data)))

    def test_gltf_empty(self):
        app = P.parse_appearance(B.build_app_switch(sub_objects=[]))
        doc = P.to_gltf(app)
        self.assertNotIn("meshes", doc)
        self.assertEqual(doc["scenes"], [{"nodes": []}])
