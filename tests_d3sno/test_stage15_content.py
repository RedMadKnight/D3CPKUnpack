"""Stage 15, content: Lua chunks, script embedding, materials, shaders (``30`` W15-A8..A14).
Synthetic data only."""

import struct
import unittest

import builders as B
from d3sno import FormatError
from d3sno import luachunk as L
from d3sno import lua_write as LW
from d3sno import material as M
from d3sno import material_write as MW
from d3sno import power as P
from d3sno import scripts as S
from d3sno import shader as SH
from d3sno import sno_layout as T
from d3sno import write as W
from d3sno.index import CoreToc, TocEntry
from d3sno.sno import expected_version

BUILDS = ("ps3", "switch", "x360")
ORDER = {"ps3": ">", "switch": "<", "x360": ">"}
E = LW.encode_abc
X = LW.encode_abx


def fn(code, constants=(), protos=(), lines=(0, 0), upvalues=0, params=0, vararg=0, stack=2,
       source=None, **extra):
    return L.LuaFunction(source, lines[0], lines[1], upvalues, params, vararg, stack, list(code),
                         list(constants), list(protos), **extra)


def good_tree():
    """A chunk in the stored form and standard code shape, defining OnStart and OnEnd."""
    on_start = fn([X("GETGLOBAL", 1, 0), E("MOVE", 2, 0), E("CALL", 1, 2, 1), E("RETURN", 0, 1)],
                  [b"print"], lines=(1, 3), params=1, stack=3)
    on_end = fn([E("GETUPVAL", 0, 0), E("TEST", 0, 0, 0), LW.encode_asbx("JMP", 0, 1),
                 X("LOADK", 0, 0), E("RETURN", 0, 1)], [2.5], lines=(5, 9), upvalues=1)
    return fn([X("CLOSURE", 0, 0), X("SETGLOBAL", 0, 0), X("LOADK", 0, 1), X("CLOSURE", 1, 1),
               E("MOVE", 0, 0), X("SETGLOBAL", 1, 2), E("RETURN", 0, 1)],
              [b"OnStart", 42.0, b"OnEnd"], [on_start, on_end], vararg=2)


def tree_with_every_part():
    """Every constant type and the unstripped parts, for the round trip."""
    inner = fn([E("RETURN", 0, 1)], lines=(2, 4), vararg=7, source=b"@x.lua",
               line_info=[4], locals=[L.LocalVar(b"a", 0, 1)], upvalue_names=[b"u", None])
    return fn([E("RETURN", 0, 1)], [None, True, False, -0.0, 1e300, b"", b"text\x00x"],
              [inner, fn([E("RETURN", 0, 1)], lines=(5, 5))], vararg=2)


# --- W15-A8 ----------------------------------------------------------------------------------------

class ChunkForm(unittest.TestCase):
    def test_dump_parse_dump_is_identity(self):
        for build in BUILDS:
            for tree in (good_tree(), tree_with_every_part()):
                with self.subTest(build):
                    chunk = LW.dump_chunk(tree, build)
                    self.assertEqual(LW.dump_chunk(L.parse_chunk(chunk, build), build), chunk)
                    self.assertEqual(L.parse_chunk(chunk, build).trailing_bytes, 0)

    def test_builder_chunks_round_trip(self):
        for build in BUILDS:
            inner = {"code": (B.RETURN_WORD,), "lines": (3, 9)}
            chunk = B.lua_chunk(build, constants=(None, True, 2.5, b"hello"), protos=(inner,))
            self.assertEqual(LW.dump_chunk(L.parse_chunk(chunk, build), build), chunk)

    def test_header_bytes(self):
        tree = good_tree()
        self.assertEqual(LW.dump_chunk(tree, "switch")[:12], b"\x1bLuaQ\x00\x01\x04\x08\x04\x08\x00")
        self.assertEqual(LW.dump_chunk(tree, "ps3")[:12], b"\x1bLuaQ\x00\x00\x04\x04\x04\x08\x00")
        self.assertEqual(LW.dump_chunk(tree, "x360"), LW.dump_chunk(tree, "ps3"))

    def test_string_and_size_t(self):
        tree = fn([X("LOADK", 0, 0), E("RETURN", 0, 1)], [b"ab"], vararg=2)
        sw = LW.dump_chunk(tree, "switch")
        self.assertIn(struct.pack("<Q", 3) + b"ab\0", sw)
        self.assertIn(struct.pack(">I", 3) + b"ab\0", LW.dump_chunk(tree, "ps3"))
        self.assertEqual(sw[12:20], bytes(8))  # no source name: length 0

    def test_form_problems(self):
        self.assertEqual(LW.form_problems(LW.dump_chunk(good_tree(), "ps3")), [])
        problems = LW.form_problems(L.parse_chunk(LW.dump_chunk(tree_with_every_part(), "ps3")))
        self.assertTrue(any("not stripped" in p for p in problems))
        bad = good_tree()
        bad.param_count, bad.vararg_flags, bad.line_defined = 1, 0, 3
        self.assertEqual(len(LW.form_problems(bad)), 3)
        self.assertTrue(LW.form_problems(LW.dump_chunk(good_tree(), "ps3") + b"\0"))

    def test_bad_values_refused(self):
        with self.assertRaises(FormatError):
            LW.dump_chunk(fn([E("RETURN", 0, 1)], [object()]), "ps3")
        with self.assertRaises(FormatError):
            LW.dump_chunk(fn([E("RETURN", 0, 1)], stack=300), "ps3")
        with self.assertRaises(FormatError):
            LW.dump_chunk(good_tree(), "pc")


# --- W15-A9 ----------------------------------------------------------------------------------------

class CodeShape(unittest.TestCase):
    def test_good_tree(self):
        self.assertEqual(LW.shape_problems(good_tree()), [])
        LW.check_shape(good_tree())

    def broken(self, change):
        tree = good_tree()
        change(tree)
        return LW.shape_problems(tree)

    def test_each_rule(self):
        cases = {
            "RETURN": lambda t: t.code.__setitem__(-1, E("RETURN", 0, 2)),
            "repeat": lambda t: t.constants.__setitem__(2, b"OnStart"),
            "not used": lambda t: t.constants.append(b"spare"),
            "upvalue": lambda t: t.code.__setitem__(4, E("LOADNIL", 0, 0)),
            "jump": lambda t: t.protos[1].code.__setitem__(2, E("MOVE", 0, 0)),
            "nil load": lambda t: t.protos[1].code.__setitem__(0, E("LOADNIL", 0, 0)),
            "source order": lambda t: t.protos.reverse(),
            "max stack": lambda t: setattr(t, "max_stack", 1),
            "vararg": lambda t: setattr(t.protos[0], "vararg_flags", 2),
        }
        for word, change in cases.items():
            with self.subTest(word):
                problems = self.broken(change)
                self.assertTrue(any(word in p for p in problems), problems)
        with self.assertRaises(FormatError):
            LW.check_shape(fn([E("RETURN", 0, 2)]))

    def test_vararg_forms(self):
        uses = fn([E("VARARG", 0, 2), E("RETURN", 0, 1)], vararg=7, lines=(1, 1))
        self.assertTrue(any("vararg 7" in p for p in LW.shape_problems(fn(
            [X("CLOSURE", 0, 0), E("RETURN", 0, 1)], protos=[uses], vararg=2))))
        unused = fn([E("RETURN", 0, 1)], vararg=7, lines=(1, 1))
        self.assertEqual(LW.shape_problems(fn([X("CLOSURE", 0, 0), E("RETURN", 0, 1)],
                                              protos=[unused], vararg=2)), [])

    def test_nil_load_of_a_parameter_is_standard(self):
        f = fn([E("LOADNIL", 0, 0), E("RETURN", 0, 1)], params=1, lines=(1, 1))
        self.assertEqual(LW.shape_problems(fn([X("CLOSURE", 0, 0), E("RETURN", 0, 1)],
                                              protos=[f], vararg=2)), [])

    def test_retarget(self):
        tree = good_tree()
        chunks = {b: LW.dump_chunk(tree, b) for b in BUILDS}
        for source in BUILDS:
            for target in BUILDS:
                with self.subTest((source, target)):
                    self.assertEqual(LW.retarget_chunk(chunks[source], source, target), chunks[target])
        with self.assertRaises(FormatError):
            LW.retarget_chunk(chunks["ps3"], "switch", "ps3")  # header names another build
        with self.assertRaises(FormatError):
            LW.retarget_chunk(chunks["ps3"] + b"\0", "ps3", "switch")

    def test_retargeted_tree_is_equal(self):
        chunk = LW.retarget_chunk(LW.dump_chunk(good_tree(), "switch"), "switch", "ps3")
        self.assertEqual(L.parse_chunk(chunk, "ps3").main, good_tree())


# --- W15-A10 ---------------------------------------------------------------------------------------

def tree_file(build, ext, chunk=b""):
    """A .con or .cnv laid out by the generic writer: every top-level list holds one element."""
    order = ORDER[build]
    header = struct.pack(order + "4I", 0xDEADBEEF, expected_version(ext, build) or 1, 1, 0x1234)
    top_name = T.top_record(build, ext)
    top = T.new_record(build, ext, top_name)
    for i in range(12, len(top.raw)):
        top.raw[i] = (i * 7) % 251
    script_at = L.SCRIPT_PAIRS[ext][0 if build == "ps3" else 1] - 0x10
    for slot in T.pair_slots(build, ext, top_name):
        if slot.offset == script_at:
            if chunk:
                top.pairs[slot.offset] = T.PairValue(T.Block(chunk))
        elif slot.field.is_record_array:
            element = T.new_record(build, ext, slot.field.element)
            element.raw[:] = bytes((i * 3 + 1) % 256 for i in range(len(element.raw)))
            top.pairs[slot.offset] = T.PairValue(T.Block(elements=[element]))
        elif slot.field.kind == "array":
            top.pairs[slot.offset] = T.PairValue(T.Block(bytes(8)))
    data = T.write_tree(T.SnoTree(build, ext, header, top))
    return T.write_tree(T.parse_tree(data, ext, build))


def masked(data, order, at):
    out = bytearray(data)
    out[at:at + 8] = bytes(8)
    return bytes(out)


class ChunkEmbedding(unittest.TestCase):
    def chunks(self, build):
        return B.lua_chunk(build, lines=(1, 2)), LW.dump_chunk(good_tree(), build)

    def test_pow(self):
        for build in BUILDS:
            with self.subTest(build):
                small, large = self.chunks(build)
                data = B.build_power(build, general=[(0, 0x1000, 5)], script=small)
                parsed = P.parse_power(data, build)
                self.assertEqual(W.write_pow(parsed), data)
                W.replace_script(parsed, large)
                out = W.write_pow(parsed)
                read = P.parse_power(out, build)
                self.assertEqual(read.script, large)
                at = L.SCRIPT_PAIRS["pow"][0 if build == "ps3" else 1]
                start = read.script_pair.file_offset
                self.assertEqual(start % 8, 0)
                self.assertEqual(start + len(large), len(out))  # the last block
                self.assertEqual(masked(out, ORDER[build], at)[:start],
                                 masked(data, ORDER[build], at)[:start])

    def test_scr(self):
        for build in BUILDS:
            with self.subTest(build):
                small, large = self.chunks(build)
                data = B.build_scr(build, [{"event": 1, "handler": "OnStart"},
                                           {"event": 2, "handler": "OnEnd"}], chunk=small)
                parsed = S.parse_script(data, build)
                self.assertEqual(W.write_scr(parsed), data)
                W.replace_script(parsed, large)
                out = W.write_scr(parsed)
                read = S.parse_script(out, build)
                self.assertEqual(read.chunk, large)
                pair = struct.unpack_from(ORDER[build] + "II", out, 0x20)
                self.assertEqual(pair[0] + 0x10, 0x58)  # the first block
                self.assertEqual([m.handler for m in read.messages], ["OnStart", "OnEnd"])
                self.assertEqual([m.event_id for m in read.messages], [1, 2])
                self.assertEqual(masked(masked(out, "", 0x20), "", 0x30)[:0x58],
                                 masked(masked(data, "", 0x20), "", 0x30)[:0x58])
                table = struct.unpack_from(ORDER[build] + "I", out, 0x30)[0] + 0x10
                self.assertEqual(table % 8, 0)

    def test_con_and_cnv(self):
        for ext in ("con", "cnv"):
            for build in BUILDS:
                with self.subTest((ext, build)):
                    small, large = self.chunks(build)
                    data = tree_file(build, ext, small)
                    tree = T.parse_tree(data, ext, build)
                    self.assertEqual(T.write_tree(tree), data)
                    self.assertEqual(L.script_slice(data, ext, build), small)
                    W.replace_script(tree, large)
                    out = W.write(tree)
                    self.assertEqual(L.script_slice(out, ext, build), large)
                    self.assertEqual(W.tree_script(T.parse_tree(out, ext, build)), large)
                    at = L.SCRIPT_PAIRS[ext][0 if build == "ps3" else 1]
                    start = struct.unpack_from(ORDER[build] + "I", out, at)[0] + 0x10
                    self.assertEqual(start % 8, 0)
                    self.assertEqual(start + len(large), len(out))  # the last block
                    self.assertEqual(masked(out, "", at)[:start], masked(data, "", at)[:start])

    def test_no_chunk_is_an_empty_pair(self):
        for build in BUILDS:
            tree = T.parse_tree(tree_file(build, "con", B.lua_chunk(build)), "con", build)
            W.replace_script(tree, b"")
            out = T.write_tree(tree)
            self.assertEqual(struct.unpack_from(ORDER[build] + "II", out, 0x48), (0, 0))
            parsed = P.parse_power(B.build_power(build, script=B.lua_chunk(build)), build)
            W.replace_script(parsed, b"")
            at = L.SCRIPT_PAIRS["pow"][0 if build == "ps3" else 1]
            self.assertEqual(struct.unpack_from(ORDER[build] + "II", W.write_pow(parsed), at), (0, 0))

    def test_tree_accepted_and_build_checked(self):
        parsed = P.parse_power(B.build_power("switch"), "switch")
        W.replace_script(parsed, good_tree())
        self.assertEqual(parsed.script, LW.dump_chunk(good_tree(), "switch"))
        with self.assertRaises(FormatError):
            W.replace_script(parsed, LW.dump_chunk(good_tree(), "ps3"))
        with self.assertRaises(FormatError):
            W.replace_script(parsed, LW.dump_chunk(good_tree(), "switch") + b"\0")
        with self.assertRaises(FormatError):
            W.replace_script(T.parse_tree(tree_file("switch", "con"), "con", "switch").__class__(
                "switch", "mat", b"", T.new_record("switch", "mat", "MaterialDefinition")), b"")

    def test_handlers_are_global_functions(self):
        chunk = LW.dump_chunk(good_tree(), "switch")
        self.assertEqual(LW.defined_globals(chunk, "switch"), {"OnStart", "OnEnd"})
        self.assertEqual(LW.defined_globals(b""), set())
        data = B.build_scr("switch", [{"event": 1, "handler": "OnStart"},
                                      {"event": 2, "handler": "OnMissing"}], chunk=chunk)
        parsed = S.parse_script(data, "switch")
        problems = S.handler_problems(parsed)
        self.assertEqual(len(problems), 1)
        self.assertIn("OnMissing", problems[0])
        with self.assertRaises(FormatError):
            W.replace_script(parsed, chunk, check_handlers=True)
        W.set_handler(parsed, 1, "OnEnd")
        self.assertEqual(S.handler_problems(parsed), [])
        W.replace_script(parsed, chunk, check_handlers=True)

    def test_global_needs_a_closure(self):
        # A string stored in a global is not a function definition.
        tree = fn([X("LOADK", 0, 0), X("SETGLOBAL", 0, 1), E("RETURN", 0, 1)], [b"v", b"OnStart"], vararg=2)
        self.assertEqual(LW.defined_globals(tree), set())
        moved = fn([X("CLOSURE", 1, 0), E("MOVE", 0, 1), X("SETGLOBAL", 0, 0), E("RETURN", 0, 1)],
                   [b"OnStart"], [fn([E("RETURN", 0, 1)], lines=(1, 1))], vararg=2)
        self.assertEqual(LW.defined_globals(moved), {"OnStart"})


# --- W15-A11 / A12 ----------------------------------------------------------------------------------

def colours(build):
    if build == "ps3":
        return [((i, i + 1, i + 2, 255), 0.5 + i) for i in range(4)]
    return [(0.1 * i, 0.2, 0.3, 1.0) for i in range(4)]


def sample_material(build, entries=((1, 500), (3, 501, bytes(12)))):
    return MW.new_material(build, 70, colours(build), 8.0, -2, entries,
                           [(2, 0x10, 900), (0, 0x20, 5)], handle=1234)


class MaterialFields(unittest.TestCase):
    def test_new_material_reads_back(self):
        for build in BUILDS:
            with self.subTest(build):
                data = T.write_tree(sample_material(build))
                read = M.parse_material(data, build)
                self.assertEqual(read.flags, 0)
                block = read.block
                self.assertEqual((block.shader_map, block.power, block.sort_priority_raw),
                                 (70, 8.0, 0xFFFFFFFE))
                self.assertEqual([(t.type, t.handle) for t in block.textures], [(1, 500), (3, 501)])
                self.assertEqual(block.textures[1].animation_raw, bytes(12))
                if build == "ps3":
                    self.assertEqual(block.specular.rgba, (2, 3, 4, 255))
                    self.assertEqual(block.specular.scale, 2.5)
                else:
                    self.assertAlmostEqual(block.specular.rgba[0], 0.2, places=6)
                self.assertEqual(len(data) - 0x10 >= (0x68 if build == "ps3" else 0x88), True)
                self.assertEqual(struct.unpack_from(ORDER[build] + "i", data, 0x10)[0], 1234)
                self.assertEqual(MW.material_zero_problems(data, build), [])

    def test_rebuilds_through_the_generic_tree(self):
        for build in BUILDS:
            data = T.write_tree(sample_material(build))
            self.assertTrue(T.is_canonical(data, "mat", build))
            self.assertEqual(T.write_tree(MW.material_tree(data, build)), data)

    def test_zero_fields(self):
        for build in BUILDS:
            data = bytearray(T.write_tree(sample_material(build)))
            block = 0x30
            at = block + (0x60 if build != "ps3" else 0x40)
            data[at] = 1
            self.assertTrue(MW.material_zero_problems(bytes(data), build))
            data = bytearray(T.write_tree(sample_material(build)))
            data[0x1C + 3] = 1
            self.assertTrue(MW.material_zero_problems(bytes(data), build))

    def test_appended_entry(self):
        for build in BUILDS:
            with self.subTest(build):
                tree = sample_material(build)
                before = M.parse_material(T.write_tree(tree), build)
                MW.add_texture(tree, 5, 777, bytes(20))
                after = M.parse_material(T.write_tree(tree), build)
                self.assertEqual([(t.type, t.handle, t.animation_raw) for t in after.block.textures],
                                 [(1, 500, b""), (3, 501, bytes(12)), (5, 777, bytes(20))])
                self.assertEqual(after.tag_map_raw, before.tag_map_raw)
                for name in ("shader_map", "ambient", "diffuse", "specular", "emissive", "power",
                             "sort_priority_raw"):
                    self.assertEqual(getattr(after.block, name), getattr(before.block, name))

    def test_animation_sizes(self):
        for size in (0, 12, 20, 144):
            MW.new_material("switch", -1, colours("switch"), 1.0, 0, [(1, 1, bytes(size))])
        with self.assertRaises(FormatError):
            MW.new_material("switch", -1, colours("switch"), 1.0, 0, [(1, 1, bytes(4))])
        with self.assertRaises(FormatError):
            MW.new_material("switch", -1, colours("switch")[:3], 1.0, 0)

    def test_set_texture_in_place(self):
        for build in BUILDS:
            tree = sample_material(build)
            before = T.write_tree(tree)
            self.assertEqual(MW.set_texture(tree, 3, -1), 501)
            after = T.write_tree(tree)
            self.assertEqual(len(after), len(before))
            changed = [i for i in range(len(after)) if after[i] != before[i]]
            self.assertEqual(len(changed), 4)  # 501 -> -1: one handle word
            read = M.parse_material(after, build)
            self.assertEqual([(t.type, t.handle) for t in read.block.textures], [(1, 500), (3, -1)])
            self.assertEqual(changed[0], read.block.textures[1].position)
            with self.assertRaises(FormatError):
                MW.set_texture(tree, 9, 1)

    def test_allowed_slots(self):
        tree = sample_material("switch")
        with self.assertRaises(FormatError):
            MW.add_texture(tree, 7, 1, allowed_slots={1, 3})
        MW.add_texture(tree, 1, 1, allowed_slots={1, 3})


class MaterialLinks(unittest.TestCase):
    def toc(self, *keys):
        return CoreToc("switch", [TocEntry(g, h, f"n{h}") for g, h in keys])

    def test_links(self):
        for build in BUILDS:
            tree = sample_material(build)
            good = self.toc((36, 70), (44, 500), (44, 501), (43, 900))
            self.assertEqual(MW.check_material_links(tree, good), [])
            self.assertEqual(MW.check_material_links(T.write_tree(tree), good, build), [])
            problems = MW.check_material_links(tree, self.toc((44, 500)))
            self.assertEqual(len(problems), 3)  # shader map, texture 501, the type-2 tag
            self.assertTrue(any("shader map" in p for p in problems))
            self.assertTrue(any("501" in p for p in problems))
            self.assertTrue(any("0x10" in p for p in problems))

    def test_minus_one_and_other_targets(self):
        tree = MW.new_material("switch", -1, colours("switch"), 1.0, 0, [(1, -1)], [(2, 0x10, 33)])
        self.assertEqual(MW.check_material_links(tree, {(27, 33)}), [])  # a particle
        self.assertEqual(MW.check_material_links(tree, {(37, 33)}), [])  # a shader
        self.assertEqual(len(MW.check_material_links(tree, {(44, 33)})), 1)

    def test_block_and_file(self):
        data = B.build_mat("switch", [(1, 50)])
        parsed = M.parse_material(data, "switch")
        self.assertEqual(len(MW.check_material_links(parsed.block, set())), 2)


# --- W15-A13 / A14 ----------------------------------------------------------------------------------

def program(build, size, fill=0):
    """One record of a compiled block."""
    order = ORDER[build]
    if build == "ps3":
        code = bytes([fill + 1]) * size
        return bytes(0x10) + struct.pack(">I", len(code)) + bytes(12) + code
    if build == "switch":
        body = bytearray([fill]) * (0x50 + size)
        struct.pack_into("<II", body, 0x44, len(body), SH.SWITCH_PROGRAM_MARK)
        return struct.pack("<4I", 0, 0, len(body), 0) + bytes(body)
    return struct.pack(">4I", 0, 3, 8, 16) + bytes([fill]) * (8 + 16 + 8 * size)


PASS_AT = {"ps3": (0x10, 0x14, 0x20, 0x90, 0x74, 0xC0), "switch": (0x14, 0x18, 0x28, 0x70, 0x84, 0xD8)}
PASS_AT["x360"] = PASS_AT["switch"]


def shader_file(build, passes, sizes, fill=0, key_base=100):
    """A .shd laid out by the generic writer: ``passes`` stage-slot lists, one per pass."""
    order = ORDER[build]
    count_at, passes_at, block_at, stages_at, tags_at, keys_at = PASS_AT[build]
    header = struct.pack(order + "4I", 0xDEADBEEF, expected_version("shd", build) or 1, 1, 0x99)
    top_name = T.top_record(build, "shd")
    top = T.new_record(build, "shd", top_name)
    struct.pack_into(order + "II", top.raw, 0x0C, 5, len(passes))
    if count_at != 0x10:
        struct.pack_into(order + "I", top.raw, count_at, len(passes))
    slots = {s.offset: s for s in T.pair_slots(build, "shd", top_name)}
    pass_name = slots[passes_at].field.element
    inner = {s.offset: s for s in T.pair_slots(build, "shd", pass_name)}
    records = []
    for k, stage_slots in enumerate(passes):
        record = T.new_record(build, "shd", pass_name)
        for i in range(0x08, 0x68):
            record.raw[i] = (i + k) % 200
        struct.pack_into(order + "I", record.raw, 0x6C, len(stage_slots))
        struct.pack_into(order + "II", record.raw, keys_at, key_base + 2 * k, key_base + 2 * k + 1)
        stages = []
        for slot_type in stage_slots:
            stage = T.new_record(build, "shd", inner[stages_at].field.element)
            struct.pack_into(order + "5If", stage.raw, 0, slot_type, 1, 2, 3, 4, -0.5)
            stages.append(stage)
        if stages:
            record.pairs[stages_at] = T.PairValue(T.Block(elements=stages))
        record.pairs[tags_at] = T.PairValue(T.Block(struct.pack(order + "IIIi", 1, 2, 0x50, 7)))
        records.append(record)
    top.pairs[passes_at] = T.PairValue(T.Block(elements=records))
    raws = [program(build, s, fill) for s in sizes]
    top.pairs[block_at] = T.PairValue(T.Block(SH.shader_block(raws, build)))
    return T.write_tree(T.SnoTree(build, "shd", header, top))


class ShaderContainer(unittest.TestCase):
    def test_read(self):
        for build in BUILDS:
            with self.subTest(build):
                data = shader_file(build, [[1, 3], [2]], [1, 2, 3, 4])
                shader = SH.read_shader(data, build)
                self.assertEqual(shader.problems, [])
                self.assertEqual((shader.flags, shader.pass_count, len(shader.passes)), (5, 2, 2))
                self.assertEqual([s.slot_type for s in shader.passes[0].stages], [1, 3])
                self.assertEqual(shader.passes[0].stages[0].address_modes, (1, 2, 3))
                self.assertEqual(shader.passes[0].stages[0].mip_bias, -0.5)
                self.assertEqual((shader.passes[1].vertex_key, shader.passes[1].pixel_key), (102, 103))
                self.assertEqual([(p.pass_index, p.stage) for p in shader.programs],
                                 [(0, "vertex"), (0, "pixel"), (1, "vertex"), (1, "pixel")])
                self.assertTrue(data.endswith(shader.block))
                self.assertEqual(SH.write_shader(shader), data)
                self.assertEqual(SH.sampled_slot_types(shader), {1, 2, 3})
                if build == "x360":
                    self.assertEqual(shader.programs[0].head, (0, 3, 8, 16))
                if build == "ps3":
                    self.assertEqual(shader.programs[2].code, bytes([1]) * 3)

    def test_x360_heads(self):
        raws = [program("x360", 1), struct.pack(">4I", 0, 9, 0, 0) + bytes(16)]
        found = SH.shader_records(SH.shader_block(raws, "x360"), 1, "x360")
        self.assertEqual(found[1].head, (0, 9, 0, 0))
        bad = [program("x360", 1), struct.pack(">4I", 0, 9, 4, 0) + bytes(16)]
        with self.assertRaises(FormatError):
            SH.shader_records(SH.shader_block(bad, "x360"), 1, "x360")

    def test_contiguous_records(self):
        first, second = program("switch", 1), program("switch", 1)
        block = struct.pack("<4I", 0, len(first), len(first) + 8, len(second)) + first + bytes(8) + second
        with self.assertRaises(FormatError):
            SH.shader_records(bytes(block), 1, "switch")
        SH.shader_records(bytes(block), 1, "switch", strict=False)
        tail = SH.shader_block([program("switch", 1), program("switch", 1)], "switch") + bytes(8)
        with self.assertRaises(FormatError):
            SH.shader_records(tail, 1, "switch")

    def test_switch_program_words(self):
        raw = bytearray(program("switch", 1))
        struct.pack_into("<I", raw, 16 + 0x48, 0x2D1)
        with self.assertRaises(FormatError):
            SH.shader_records(SH.shader_block([bytes(raw), program("switch", 1)], "switch"), 1, "switch")
        raw = bytearray(program("switch", 1))
        struct.pack_into("<I", raw, 16 + 0x44, 4)
        with self.assertRaises(FormatError):
            SH.shader_records(SH.shader_block([bytes(raw), program("switch", 1)], "switch"), [0], "switch")

    def test_counts_checked(self):
        data = bytearray(shader_file("switch", [[1, 3]], [1, 2]))
        struct.pack_into("<I", data, 0x10 + 0x14, 3)
        self.assertTrue(any("pass count" in p for p in SH.read_shader(bytes(data), "switch").problems))
        passes = struct.unpack_from("<I", data, 0x10 + 0x18)[0] + 0x10
        struct.pack_into("<I", data, passes + 0x6C, 1)
        self.assertTrue(any("stage count" in p for p in SH.read_shader(bytes(data), "switch").problems))


class ShaderReplacement(unittest.TestCase):
    def test_replace(self):
        for build in BUILDS:
            with self.subTest(build):
                data = shader_file(build, [[1], [2]], [1, 1, 1, 1])
                donor_data = shader_file(build, [[4, 5], [6]], [3, 2, 5, 4], fill=9, key_base=500)
                shader = SH.read_shader(data, build)
                donor = SH.read_shader(donor_data, build)
                out = SH.replace_shader_programs(shader, donor)
                self.assertTrue(out.endswith(donor.block))
                self.assertEqual(shader.block, donor.block)
                self.assertEqual(shader.problems, [])
                self.assertEqual([(p.vertex_key, p.pixel_key) for p in shader.passes],
                                 [(500, 501), (502, 503)])
                self.assertEqual([s.slot_type for s in shader.passes[0].stages], [1])
                block_pair = 0x10 + PASS_AT[build][2]
                start = struct.unpack_from(ORDER[build] + "I", out, block_pair)[0] + 0x10
                self.assertEqual(start, struct.unpack_from(ORDER[build] + "I", data, block_pair)[0] + 0x10)
                passes = struct.unpack_from(ORDER[build] + "I", data, 0x10 + PASS_AT[build][1])[0] + 0x10
                keys = PASS_AT[build][5]
                size = 200 if build == "ps3" else 224
                skip = set(range(block_pair + 4, block_pair + 8))
                for k in range(2):
                    skip |= set(range(passes + k * size + keys, passes + k * size + keys + 8))
                diff = [i for i in range(start) if out[i] != data[i] and i not in skip]
                self.assertEqual(diff, [])

    def test_options(self):
        data = shader_file("switch", [[1], [2]], [1, 1, 1, 1])
        donor = SH.read_shader(shader_file("switch", [[4, 5], [6]], [2, 2, 2, 2], 9, 500), "switch")
        shader = SH.read_shader(data, "switch")
        SH.replace_shader_programs(shader, donor, copy_stages=True, copy_render_params=True)
        self.assertEqual([s.slot_type for s in shader.passes[0].stages], [4, 5])
        self.assertEqual(shader.passes[0].stage_count, 2)
        self.assertEqual(shader.passes[1].record.raw[0x08:0x68], donor.passes[1].record.raw[0x08:0x68])
        self.assertEqual(shader.problems, [])
        ps3 = SH.read_shader(shader_file("ps3", [[1]], [1, 1]), "ps3")
        with self.assertRaises(FormatError):
            SH.replace_shader_programs(ps3, SH.read_shader(shader_file("ps3", [[1]], [2, 2]), "ps3"),
                                       copy_render_params=True)

    def test_refused(self):
        one = SH.read_shader(shader_file("switch", [[1]], [1, 1]), "switch")
        two = SH.read_shader(shader_file("switch", [[1], [2]], [1, 1, 1, 1]), "switch")
        with self.assertRaises(FormatError):
            SH.replace_shader_programs(one, two)
        with self.assertRaises(FormatError):
            SH.replace_shader_programs(one, SH.read_shader(shader_file("x360", [[1]], [1, 1]), "x360"))

    def test_unused_key(self):
        self.assertEqual(SH.unused_pass_key({1, 2, 4}), 3)


def shader_map(build, records):
    order = ORDER[build]
    header = struct.pack(order + "4I", 0xDEADBEEF, expected_version("shm", build) or 1, 1, 0x77)
    top = T.new_record(build, "shm", T.top_record(build, "shm"))
    top.pairs[0x10] = T.PairValue(T.Block(MW.encode_tag_map(records, build)))
    return T.write_tree(T.SnoTree(build, "shm", header, top))


class ShaderMaps(unittest.TestCase):
    def test_set_link(self):
        for build in BUILDS:
            with self.subTest(build):
                data = shader_map(build, [(2, 0x100, 40), (2, 0x200, 41), (0, 0x300, 9)])
                shm = SH.read_shader_map(data, build)
                self.assertEqual(T.write_tree(shm), data)
                self.assertEqual(SH.shader_map_links(shm), {0x100: 40, 0x200: 41})
                self.assertEqual(SH.set_shader_map_link(shm, 0x200, 77), 41)
                out = T.write_tree(shm)
                self.assertEqual(len(out), len(data))
                diff = [i for i in range(len(out)) if out[i] != data[i]]
                self.assertEqual(len(diff) <= 4 and len(diff) > 0, True)
                self.assertEqual(SH.shader_map_links(SH.read_shader_map(out, build)), {0x100: 40, 0x200: 77})
                with self.assertRaises(FormatError):
                    SH.set_shader_map_link(shm, 0x300, 1)
                with self.assertRaises(FormatError):
                    SH.set_shader_map_link(shm, 0x400, 1)

    def test_links_resolve(self):
        shm = SH.read_shader_map(shader_map("switch", [(2, 0x100, 40), (2, 0x200, -1)]), "switch")
        self.assertEqual(SH.check_shader_map_links(shm, {(37, 40)}), [])
        self.assertEqual(len(SH.check_shader_map_links(shm, {(36, 40)})), 1)


if __name__ == "__main__":
    unittest.main()
