"""Damaged input must end in FormatError or a soft result, never another exception."""

import random
import struct
import unittest

import builders as B
from d3sno import FormatError
from d3sno import (
    animation, appearance, archive, formula, index, luachunk, navigation, power, texture,
)

ROUNDS = 400


def damage(data: bytes, rng: random.Random) -> bytes:
    out = bytearray(data)
    kind = rng.randrange(3)
    if kind == 0 and out:
        return bytes(out[:rng.randrange(len(out))])
    for _ in range(rng.randrange(1, 6)):
        if not out:
            break
        position = rng.randrange(len(out))
        if kind == 1:
            out[position] = rng.randrange(256)
        else:
            value = rng.choice((0, 1, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF))
            position &= ~3
            if position + 4 <= len(out):
                struct.pack_into(rng.choice(("<I", ">I")), out, position, value)
    return bytes(out)


def toc_switch():
    records = struct.pack("<iii", 1, 5, 0) + b"Name\0"
    counts = [0] * 70
    counts[1] = 1
    return struct.pack("<70i70i70iI", *counts, *([0] * 70), *counts, 0) + records


def prefetch():
    return struct.pack("<I4II2I", 1, 1, 2, 1, 0, 1, 3, 4)


def run_tag_formulas(data):
    parsed = power.parse_power(data)
    for record in parsed.formula_records:
        formula.disassemble(record.value.code, parsed.build)
        formula.evaluate_tag(parsed, record.tag, formula.ValueContext({}))


def export_mesh(data):
    app = appearance.parse_appearance(data)
    appearance.to_obj(app)
    appearance.to_gltf_json(app)


class Fuzz(unittest.TestCase):
    def survive(self, parse, sample: bytes, seed: int):
        rng = random.Random(seed)
        for _ in range(ROUNDS):
            try:
                parse(damage(sample, rng))
            except FormatError:
                pass

    def test_archive_reader(self):
        def read_all(data):
            arc = archive.read_archive(data, strict=False)
            for entry in arc.files:
                arc.read_file(entry, strict=False)

        for options in ({}, {"version": 6}, {"flags": 0}):
            sample = archive.write_archive({"a.bin": b"abc" * 300, "b.bin": b"xyz"}, **options)
            self.survive(read_all, sample, 1)

    def test_archive_verifier_reports_instead_of_raising(self):
        for options in ({}, {"version": 6}, {"flags": 0}, {"flags": 10}):
            sample = archive.write_archive({"a.bin": b"abc" * 300, "b.bin": b"xyz"}, **options)
            rng = random.Random(2)
            for _ in range(ROUNDS):
                archive.verify_archive(damage(sample, rng)).format()

    def test_rebuild(self):
        sample = archive.write_archive({"a.bin": b"abc" * 300, "b.bin": b"xyz"})
        self.survive(archive.rebuild, sample, 3)

    def test_index_files(self):
        self.survive(index.parse_core_toc, toc_switch(), 4)
        ps3 = struct.pack(">I", 1) + struct.pack(">II", 1, 2) + bytes(128)
        self.survive(index.parse_core_toc, ps3, 5)
        self.survive(index.parse_prefetch, prefetch(), 6)
        self.survive(index.parse_manifest, struct.pack("<I2I", 0xAABB0004, 1, 2), 7)
        self.survive(index.parse_sprite_table, struct.pack("<I3I", 0xBBCC0005, 1, 2, 0), 8)
        self.survive(index.parse_subtitles, b"00:00:01:00 - 00:00:02:00\r\nx\r\n", 9)
        descriptor = struct.pack("<I", 0xAABBCCDD) + bytes(1092)
        self.survive(index.parse_descriptor_sizes, descriptor, 10)

    def test_power_and_formulas(self):
        code = B.Code("<").f(2).f(3).op(11).ident(1).op(13).call(0).end()
        sample = B.build_power(
            "switch", general=[(4, 0x41100, ("text", code)), (0, 2, 3)], buff=[[1, 2]],
            script=B.lua_chunk("switch"),
        )
        self.survive(run_tag_formulas, sample, 11)
        code = B.Code(">").f(2).ident(1).op(11).end()
        sample = B.build_power("ps3", general=[(4, 0x41100, ("text", code))], buff=[[1]])
        self.survive(run_tag_formulas, sample, 12)

    def test_bytecode(self):
        sample = B.Code("<").f(2).f(3).op(11).call(1).ident(1).op(17).end()
        self.survive(
            lambda code: formula.evaluate(code, "switch", formula.ValueContext({1: 1})),
            sample, 13,
        )

    def test_appearance(self):
        bones = [(1, -1, (0.0, 0.0, 0.0)), (2, 0, (1.0, 0.0, 0.0))]
        skin = {"palette": [0, 1], "vertices": [(0, 0, 0)] * 3, "indices": [0, 1, 2]}
        self.survive(export_mesh, B.build_app_ps3(), 14)
        self.survive(export_mesh, B.build_app_ps3(skinned=True, bones=bones), 15)
        self.survive(export_mesh, B.build_app_switch(), 16)
        self.survive(export_mesh, B.build_app_switch(bones=bones, sub_objects=[{"skin": skin}]), 17)

    def test_animation(self):
        bones = [(0xAA, -1, (0.0, 0.0, 0.0))]
        app = appearance.parse_appearance(B.build_app_switch(bones=bones))

        def use(data):
            ani = animation.parse_ani(data)
            animation.to_gltf_animated_json(app, ani)
            perm = ani.permutations[0]
            curves = animation.decode_permutation(perm)
            animation.pose_at(app.bones, perm, curves, 1.5)

        self.survive(use, B.build_ani("switch", [B.simple_permutation("switch")]), 18)

    def test_ps3_animation(self):
        perm = B.simple_permutation("ps3")
        ani = B.build_ani("ps3", [perm])

        def use(data):
            parsed = animation.parse_ani(ani, can=data)
            animation.decode_permutation(parsed.permutations[0])

        self.survive(use, B.build_can([perm]), 19)
        self.survive(lambda d: animation.parse_ani(d, can=B.build_can([perm])), ani, 23)

    def test_navigation(self):
        for build in ("ps3", "switch"):
            scn, nav = B.build_scene(build)

            def use(data):
                parsed = navigation.parse_scene(data, build, nav)
                parsed.square(1, 1)
                parsed.special_flag(0)

            self.survive(use, scn, 20)
            if nav:
                self.survive(lambda d: navigation.parse_scene(scn, build, d), nav, 21)

    def test_texture(self):
        samples = [
            B.build_tex("ps3", 12, 8, 8, bytes(range(64))),
            B.build_tex("ps3", 0, 2, 2, bytes(16), frames=[(1, 0.0, 0.0, 1.0, 1.0)]),
            B.build_tex("ps3", 0, 4, 4, bytes(128 * 6), faces=6),
            B.build_tex("switch", 9, 8, 8, bytes(32), layout=0),
            B.build_tex("switch", 0, 4, 4, bytes(64)),
        ]

        def use(data):
            tex = texture.parse_texture(data)
            texture.encode_png(texture.decode_image(tex))
            if tex.frames:
                texture.frame_image(tex, 0)

        for number, sample in enumerate(samples):
            self.survive(use, sample, 30 + number)

    def test_gameplay(self):
        from d3sno import anims, balance, scripts, skills, text
        for build in ("ps3", "switch"):
            order = ">" if build == "ps3" else "<"
            self.survive(text.parse_string_list, B.build_stl(build, [(1, b"a"), (2, b"bc")]), 40)
            slot, size = (0x2A8, 24) if build == "ps3" else (0xB8, 32)
            self.survive(balance.parse_balance, B.build_gam(build, 11, slot, ["Axe"], size), 41)
            self.survive(balance.parse_balance, B.build_formula_gam(build, [("Row", [1.0])]), 42)
            self.survive(anims.parse_anim_set, B.build_ans(build, [(2, 5, 7)]), 43)
            self.survive(skills.parse_skill_kit, B.build_skl(build, [(1, 2, 3)], [tuple(range(10))]), 44)
            filters = [(7, 1, 0, "name"), (2, 0, 0, None)]
            entry = {"event": 1, "handler": "Fn", "filters": filters}
            sample = B.build_scr(build, [entry], B.lua_chunk(build))
            self.survive(scripts.parse_script, sample, 45)
        self.survive(text.parse_declension, b"\xEF\xBB\xBFa\t1\tb\r\n", 46)

    def test_world(self):
        from d3sno import collision, material, sound, world
        for build in ("ps3", "switch"):
            tag = B.tag_map([(1, 0x80000, 2.0)], ">" if build == "ps3" else "<")
            marker = {"tags": tag, "links": [(1, 2)]}
            self.survive(world.parse_marker_set, B.build_mrk(build, [marker], [(1.0, 2.0, 3.0)], special=[1]), 50)
            self.survive(world.parse_scene_info, B.build_scene_info(build, [1]), 51)
            sample = B.build_wrl(build, [[B.scene_entry(build, 5)]], tiles=[(1, 2, 3, 4)])
            self.survive(world.parse_world, sample, 52)
            mesh = {
                "centre": (0.0, 0.0, 0.0), "vertices": [(0.0, 0.0, 0.0)] * 3,
                "triangles": [B.triangle_row(build, (0, 1, 2))] * 2,
                "nodes": [B.tree_internal(build, 2), B.tree_leaf(build, 0, 1), B.tree_leaf(build, 1, 1)],
            }

            def use_phm(data):
                collision.to_obj(collision.parse_collision(data, build))

            self.survive(use_phm, B.build_phm(build, [mesh]), 53)
            self.survive(material.parse_material, B.build_mat(build, [(1, 5)]), 54)
            self.survive(sound.parse_sound_bank, B.build_sbk(build, [(1, 2)]), 55)
            self.survive(sound.parse_sound, B.build_snd(build, [(1, 2, 100)]), 56)

    def test_app_materials(self):
        data = B.build_app_switch()
        data = B.add_app_materials(data, "switch", [1], [(0x10, [[(1, 44)]])])
        data = B.set_sub_object_material(data, "switch", 0, 0x10)

        def use(sample):
            app = appearance.parse_appearance(sample)
            appearance.to_gltf_json(app, texture_resolver=lambda handle: b"png")

        self.survive(use, data, 57)

    def test_lua(self):
        inner = {"lines": (1, 2)}
        for build in ("ps3", "switch"):
            sample = B.lua_chunk(build, constants=(1.5, b"s", True, None), protos=(inner,))
            self.survive(luachunk.parse_chunk, sample, 22)
