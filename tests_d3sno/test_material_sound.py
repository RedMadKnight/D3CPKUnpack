import base64
import struct
import unittest

import builders as B
from d3sno import FormatError
from d3sno import animation, appearance as P
from d3sno import material as M
from d3sno import sound as S

BUILDS = ("ps3", "switch")


class Materials(unittest.TestCase):
    def test_fields(self):
        for build in BUILDS:
            read = M.parse_material(B.build_mat(build, [(1, 50), (2, 51)], tags=b"tags"))
            self.assertEqual((read.flags, read.tag_map_raw), (4, b"tags"))
            block = read.block
            self.assertEqual(block.shader_map, 77)
            self.assertEqual((block.texture(1), block.texture(2), block.texture(3)), (50, 51, None))
            self.assertEqual(block.textures[0].name, "diffuse")
            self.assertEqual(block.textures[1].animation_raw, b"anim")
            self.assertEqual(block.power, 2.5)
            self.assertEqual(block.sort_priority_raw, 8)

    def test_switch_colours(self):
        block = M.parse_material(B.build_mat("switch")).block
        self.assertEqual(block.ambient, M.Colour((0.0, 1.0, 2.0, 3.0), None))
        self.assertEqual(block.emissive.rgba, (12.0, 13.0, 14.0, 15.0))

    def test_ps3_colours(self):
        block = M.parse_material(B.build_mat("ps3")).block
        self.assertEqual(block.ambient, M.Colour((0, 1, 2, 3), 0.5))
        self.assertEqual(block.specular, M.Colour((2, 3, 4, 5), 2.5))

    def test_slot_names(self):
        self.assertEqual(M.SLOT_NAMES[19], "alpha")

    def test_fixed_part(self):
        with self.assertRaises(FormatError):
            M.parse_material(B.build_mat("switch")[:0x60])

    def test_texture_list_size(self):
        with self.assertRaises(FormatError):
            M.parse_material(B.build_mat("switch", [(1, 50)], list_size=31))

    def test_texture_list_out_of_bounds(self):
        with self.assertRaises(FormatError):
            M.parse_material(B.build_mat("ps3", [(1, 50)], list_size=24 * 100))


def app_with_materials(build):
    base = B.build_app_ps3() if build == "ps3" else B.build_app_switch()
    data = B.add_app_materials(base, build, [0xA, 0xB], [
        (0x10, [[(2, 40), (1, 41)], [(1, 42)]]),
        (0x20, [[(3, 43)], []]),
    ])
    return B.set_sub_object_material(data, build, 0, 0x10, 0x99)


class AppMaterials(unittest.TestCase):
    def test_read(self):
        for build in BUILDS:
            app = P.parse_appearance(app_with_materials(build))
            self.assertEqual(app.warnings, [])
            self.assertEqual(app.material_data.looks, [0xA, 0xB])
            self.assertEqual(len(app.material_data.materials), 2)
            self.assertEqual(app.material_data.find(0x10).per_look[1][1], 901)
            self.assertEqual(app.sub_objects[0].material_hash, 0x10)
            self.assertEqual(app.sub_objects[0].sub_hash, 0x99)

    def test_sub_object_texture(self):
        for build in BUILDS:
            app = P.parse_appearance(app_with_materials(build))
            sub = app.sub_objects[0]
            self.assertEqual(M.sub_object_texture(app, sub), 41)
            self.assertEqual(M.sub_object_texture(app, sub, 1), 42)
            self.assertEqual(M.sub_object_texture(app, sub, 0, M.SLOT_LIGHTMAP), 40)
            self.assertIsNone(M.sub_object_texture(app, sub, 1, M.SLOT_LIGHTMAP))
            self.assertIsNone(M.sub_object_texture(app, sub, 5))

    def test_unknown_hash(self):
        data = B.set_sub_object_material(app_with_materials("switch"), "switch", 0, 0x55)
        app = P.parse_appearance(data)
        self.assertIsNone(M.sub_object_texture(app, app.sub_objects[0]))

    def test_no_materials(self):
        app = P.parse_appearance(B.build_app_switch())
        self.assertEqual(app.material_data.materials, [])

    def test_bad_material_table_leaves_the_mesh_usable(self):
        data = bytearray(app_with_materials("switch"))
        struct.pack_into("<i", data, 0x17C, 9)
        app = P.parse_appearance(bytes(data))
        self.assertIn("materials not read", app.warnings[0])
        self.assertEqual(len(app.sub_objects[0].draws), 1)
        with self.assertRaises(FormatError):
            P.parse_appearance(bytes(data), strict=True)

    def test_look_count_checked(self):
        data = bytearray(app_with_materials("ps3"))
        struct.pack_into(">i", data, 0x188, 9)
        self.assertIn("materials not read", P.parse_appearance(bytes(data)).warnings[0])


class GltfMaterials(unittest.TestCase):
    def doc(self, build="switch", resolver=lambda handle: f"tex/{handle}.png", look=0):
        app = P.parse_appearance(app_with_materials(build))
        return P.to_gltf(app, texture_resolver=resolver, look=look)

    def test_no_resolver_no_material(self):
        doc = P.to_gltf(P.parse_appearance(app_with_materials("switch")))
        self.assertNotIn("materials", doc)
        self.assertNotIn("material", doc["meshes"][0]["primitives"][0])

    def test_uri(self):
        for build in BUILDS:
            doc = self.doc(build)
            self.assertEqual(doc["images"], [{"uri": "tex/41.png"}])
            self.assertEqual(doc["textures"], [{"source": 0, "sampler": 0}])
            self.assertEqual(doc["meshes"][0]["primitives"][0]["material"], 0)
            pbr = doc["materials"][0]["pbrMetallicRoughness"]
            self.assertEqual(pbr["baseColorTexture"], {"index": 0})

    def test_look(self):
        self.assertEqual(self.doc(look=1)["images"], [{"uri": "tex/42.png"}])

    def test_uv_wraps(self):
        self.assertEqual(self.doc()["samplers"], [{"wrapS": 10497, "wrapT": 10497}])

    def test_embedded_bytes(self):
        doc = self.doc(resolver=lambda handle: b"\x89PNGdata")
        image = doc["images"][0]
        self.assertEqual(image["mimeType"], "image/png")
        view = doc["bufferViews"][image["bufferView"]]
        blob = base64.b64decode(doc["buffers"][0]["uri"].split(",", 1)[1])
        start = view["byteOffset"]
        self.assertEqual(blob[start:start + view["byteLength"]], b"\x89PNGdata")

    def test_unresolved_texture(self):
        self.assertNotIn("materials", self.doc(resolver=lambda handle: None))

    def test_resolver_called_once_per_handle(self):
        calls = []

        def resolver(handle):
            calls.append(handle)
            return "a.png"

        vertices = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
        data = B.build_app_switch(sub_objects=[
            {"rigid": vertices, "index_count": 3}, {"rigid": vertices, "index_count": 3},
        ])
        data = B.add_app_materials(data, "switch", [1], [(0x10, [[(1, 44)]])])
        data = B.set_sub_object_material(data, "switch", 0, 0x10)
        data = B.set_sub_object_material(data, "switch", 1, 0x10)
        doc = P.to_gltf(P.parse_appearance(data), texture_resolver=resolver)
        self.assertEqual(calls, [44])
        self.assertEqual(len(doc["materials"]), 1)

    def test_animated_export_carries_materials(self):
        app = P.parse_appearance(B.add_app_materials(
            B.set_sub_object_material(
                B.build_app_switch(bones=[(0xAA, -1, (0.0, 0.0, 0.0))]), "switch", 0, 0x10,
            ), "switch", [1], [(0x10, [[(1, 44)]])],
        ))
        ani = animation.parse_ani(B.build_ani("switch", [B.simple_permutation("switch")]))
        doc = animation.to_gltf_animated(app, ani, texture_resolver=lambda handle: "a.png")
        self.assertIn("materials", doc)
        self.assertIn("animations", doc)


class Banks(unittest.TestCase):
    def test_rows(self):
        for build in BUILDS:
            bank = S.parse_sound_bank(B.build_sbk(build, [(0x11, 100), (0x22, 200)]))
            self.assertEqual(bank.rows[1].hash, 0x22)
            self.assertEqual(bank.rows[1].stored_size, 200)
            self.assertEqual(bank.find(0x11).index, 0)
            self.assertIsNone(bank.find(0x33))

    def test_child_name(self):
        row = S.parse_sound_bank(B.build_sbk("switch", [(1, 2)] * 13)).rows[12]
        self.assertEqual(row.child_name("Boss"), "SoundBank\\Boss\\0012.fsb")
        self.assertEqual(row.child_name("Boss", "ogg"), "SoundBank\\Boss\\0012.ogg")

    def test_count_must_match(self):
        with self.assertRaises(FormatError):
            S.parse_sound_bank(B.build_sbk("ps3", [(1, 2)], count=2))

    def test_out_of_bounds(self):
        data = bytearray(B.build_sbk("switch", [(1, 2)]))
        struct.pack_into("<I", data, 0x1C, 0xFFFFFF)
        with self.assertRaises(FormatError):
            S.parse_sound_bank(bytes(data))


class Sounds(unittest.TestCase):
    def test_permutations(self):
        for build in BUILDS:
            read = S.parse_sound(B.build_snd(build, [(5, 0x11, 100)]))
            self.assertEqual(read.permutations[0].bank, 5)
            self.assertEqual(read.permutations[0].sample_hash, 0x11)
            self.assertEqual(read.permutations[0].weight, 100)

    def test_count_must_match(self):
        with self.assertRaises(FormatError):
            S.parse_sound(B.build_snd("switch", [(5, 1, 100)], count=3))

    def test_resolve(self):
        banks = {5: S.parse_sound_bank(B.build_sbk("switch", [(0x11, 9)]))}
        hit, wrong_hash, wrong_bank = S.parse_sound(
            B.build_snd("switch", [(5, 0x11, 100), (5, 0x12, 100), (6, 0x11, 100)]),
        ).permutations
        self.assertEqual(S.resolve_sample(hit, banks).stored_size, 9)
        self.assertIsNone(S.resolve_sample(wrong_hash, banks))
        self.assertIsNone(S.resolve_sample(wrong_bank, banks))
