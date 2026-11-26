"""Corrections 29 (``48_corrections29.md``): Z1 to Z5. Synthetic data only; every file goes to a temporary
folder outside any git working tree. Test shader names are our own except where the shipped tables are
checked (those tests are skipped when a table is missing, V26-8)."""

import _libpath  # noqa: F401  (python -I: the library folder on sys.path)
import io
import struct
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import builders as B  # noqa: F401  (the fixtures below use its layouts through test_stage20)
import export_fixture as F
import test_stage20 as T20
import test_stage21 as S
import test_stage21_corrections28 as C28
import y1_cases as Y
import d3sno
from d3sno import apply as AP
from d3sno import appearance as P
from d3sno import archive as A
from d3sno import export as E
from d3sno import index_write as IW
from d3sno import mesh_import as I
from d3sno import shader as SH
from d3sno import sno_layout as L
from d3sno import uv2_use as U
from d3sno import x360_archive as XA
from d3sno.sno import expected_version
from test_stage15_content import PASS_AT, ORDER, program
from test_stage18_export import glb_json
from test_stage19 import export, primitives
from test_stage20 import LAYER_TEX, S as SCALE, make_tables, need_uv2_tables, six_mesh

CLAMP, REPEAT = 33071, 10497
TRANSFORM = {"KHR_texture_transform": {"offset": [0.0, 0.0], "rotation": 0.0, "scale": [16.0, 16.0]}}


# --- Z1: the switch rows and list -----------------------------------------------------------------

ROWS = [("ps3", "s_both", 0, 1, 19, 512, 0), ("switch", "s_both", 0, 1, 19, 8192, 0),
        ("switch", "s_only", 1, 0, 12, 8192, 0), ("ps3", "s_ps3row", 0, 0, 12, 8192, 0)]
KNOWN = ["s_both", "s_ps3row", "s_listed"]
KNOWN_SWITCH = ["s_both", "s_only", "s_ps3row", "s_listed"]


class SwitchRows(unittest.TestCase):
    """Z1 on the library, with a test table."""

    def setUp(self):
        self.table = U.load_tables(make_tables(self, rows=ROWS, known=KNOWN, known_switch=KNOWN_SWITCH))

    def use(self, names, build):
        return U.uv2_use(U.rows_for(names, self.table, build), False)

    def test_own_row_over_ps3_row(self):
        switch = U.rows_for(("s_both",), self.table, "switch")
        self.assertEqual([(r.build, r.divisor) for r in switch.rows], [("switch", 8192)])
        self.assertEqual([(r.build, r.divisor) for r in U.rows_for(("s_both",), self.table, "ps3").rows],
                         [("ps3", 512)])
        self.assertEqual(U.layer_entries(switch, None, str, "layer")[0]["divisor"], 8192)

    def test_listed_without_row_is_unused(self):
        self.assertEqual(self.use(("s_listed",), "switch"), "unused")
        self.assertEqual(self.use(("s_ps3row",), "switch"), "unused")  # a ps3 row does not serve switch
        self.assertEqual(self.use(("s_ps3row",), "ps3"), "layer")

    def test_in_neither_is_unknown(self):
        self.assertEqual(self.use(("s_nowhere",), "switch"), "unknown")
        self.assertEqual(self.use(("s_only",), "ps3"), "unknown")  # in the switch list only
        self.assertEqual(self.use(("s_only",), "switch"), "layer")

    def test_table_build(self):
        self.assertEqual([U.table_build(b) for b in ("ps3", "switch", "x360", None)], ["ps3", "switch", "x360", "ps3"])
        self.assertEqual(U.KNOWN_TABLES["switch"], "shader_known_switch.txt")


class MissingSwitchList(unittest.TestCase):
    """Z1: without ``shader_known_switch.txt`` every switch primitive that needs it is ``"unknown"``."""

    def setUp(self):
        self.table = U.load_tables(make_tables(self, rows=ROWS, known=KNOWN, known_switch=KNOWN_SWITCH,
                                               skip=(U.KNOWN_TABLE_SWITCH,)))

    def test_library(self):
        self.assertEqual(set(self.table.known), {"ps3", "x360"})
        self.assertIsNone(self.table.for_build("switch"))
        for names in (("s_both",), ("s_only",), ("s_listed",), ("s_nowhere",)):
            self.assertEqual(U.uv2_use(U.rows_for(names, self.table, "switch"), False), "unknown", names)
        self.assertEqual(U.uv2_use(U.rows_for(("s_both",), self.table, "switch"), True), "lightmap")
        self.assertEqual(U.uv2_use(U.rows_for(("s_both",), self.table, "ps3"), False), "layer")

    def test_export_run(self):
        with mock.patch.object(U, "tables", lambda: self.table):
            run = SimpleNamespace(build="switch", shader_names=lambda handle: ("s_both",))
            self.assertEqual(U.uv2_use(E._Run.uv2_rows(run, SimpleNamespace(shader_map=1)), False), "unknown")

    def test_skip_message(self):
        path = str(U.DATA_DIR / U.KNOWN_TABLE_SWITCH)
        self.assertIn(path, [str(p) for p in T20.TABLE_PATHS])
        with mock.patch.object(T20, "MISSING_TABLES", (path,)):
            with self.assertRaises(unittest.SkipTest) as caught:
                need_uv2_tables()
        self.assertIn("V26-8", str(caught.exception))
        self.assertIn(U.KNOWN_TABLE_SWITCH, str(caught.exception))


class ShippedSwitchTable(unittest.TestCase):
    """Z1 on the package data."""

    def setUp(self):
        need_uv2_tables()
        self.table = U.load_tables()

    def test_package_data(self):
        self.assertEqual(set(self.table.rows), {"ps3", "switch", "x360"})
        self.assertEqual(set(self.table.known), {"ps3", "switch", "x360"})
        rows, known = self.table.for_build("switch")
        self.assertGreater(len(known), 1000)
        for group in rows.values():
            for row in group:
                self.assertEqual(row.build, "switch")
                self.assertIn(row.shader.lower(), known, row)

    def test_named_differences(self):
        name = "assaultBeast_deathTransition"
        rows = {b: [(r.pass_index, r.stage, r.divisor) for r in U.rows_for((name,), self.table, b).rows]
                for b in ("ps3", "switch", "x360")}
        self.assertEqual(rows, {"ps3": [(1, 1, 512)], "switch": [(1, 1, 8192)], "x360": [(1, 1, 8192)]})
        trails = "trails_additive_cm2x_bloom"  # no UV2 on switch
        self.assertEqual(U.uv2_use(U.rows_for((trails,), self.table, "switch"), False), "unused")
        self.assertEqual(U.uv2_use(U.rows_for((trails,), self.table, "ps3"), False), "layer")


# --- Z2: divisor-512 rows ------------------------------------------------------------------------

ODD_ROWS = [("t_odd", 0, 2, 12, 512, 0), ("t_odd1", 0, 0, 1, 512, 0), ("t_layer", 0, 1, 19, 8192, 0)]
ODD_KNOWN = ["t_odd", "t_odd1", "t_layer", "t_plain"]
ODD_MAPS = {1: ("t_odd",), 2: ("t_odd", "t_layer"), 3: ("t_plain",), 4: ("t_odd1",), 5: ("t_odd",), 6: ("t_odd",)}
CONST = [(0x9000, 0x9000)] * 3
VARY = [(0x8000, 0x8000), (0xA000, 0x9000), (0x7000, 0x8800)]
ODD_LIGHT, ODD_MISSING = 610, 611
# sub-object: odd only; odd + layer; unused; odd on slot 1; odd with a missing lightmap; odd, constant UV2
ODD_TEXTURES = [[(12, 620)], [(12, 620), (19, LAYER_TEX)], [], [], [(2, ODD_MISSING)], [(12, 621)]]
ODD_UV2 = [VARY, VARY, VARY, VARY, VARY, CONST]


def odd_document(case, build="ps3"):
    table = U.load_tables(make_tables(case, rows=ODD_ROWS, known=ODD_KNOWN))
    source = six_mesh(ODD_TEXTURES, [1, 2, 3, 4, 5, 6], ODD_UV2)
    doc = P.to_gltf(P.parse_appearance(source), texture_resolver=lambda h: f"../textures/{h}.png",
                    lightmap_resolver=lambda h: None,
                    uv2_rows=lambda block: U.rows_for(ODD_MAPS.get(block.shader_map) if block else None, table, build))
    return source, doc


def entries(doc, prim):
    return doc["materials"][prim["material"]].get("extras", {}).get("d3sno_export", {}).get("uv2_layers")


def check_document(case, doc):
    """The references a glTF 2.0 validator checks among textures, samplers, images and materials, the
    sampler values, and the extension lists (``KHR_texture_transform`` used, never required)."""
    case.assertNotIn("extensionsRequired", doc)
    for texture in doc.get("textures", []):
        case.assertLess(texture["source"], len(doc["images"]))
        case.assertLess(texture["sampler"], len(doc["samplers"]))
    for sampler in doc.get("samplers", []):
        case.assertLessEqual(set(sampler), {"wrapS", "wrapT", "magFilter", "minFilter"})
        for key in ("wrapS", "wrapT"):
            if key in sampler:
                case.assertIn(sampler[key], (CLAMP, 33648, REPEAT))
    for material in doc.get("materials", []):
        base = material.get("pbrMetallicRoughness", {}).get("baseColorTexture")
        if base is not None:
            case.assertLess(base["index"], len(doc["textures"]))
        for entry in material.get("extras", {}).get("d3sno_export", {}).get("uv2_layers", ()):
            for name in entry.get("texture_info", {}).get("extensions", {}):
                case.assertIn(name, doc.get("extensionsUsed", ()))
    used = doc.get("extensionsUsed", [])
    case.assertEqual(len(used), len(set(used)))


class OddRows(unittest.TestCase):
    """Z2 on ``to_gltf``."""

    def setUp(self):
        self.source, self.doc = odd_document(self)
        self.prims = primitives(self.doc)
        self.glb = I.gltf_to_glb(self.doc)

    def info(self, i):
        return self.prims[i]["extras"]["d3sno_export"]

    def test_uses(self):
        self.assertEqual([self.info(i).get("uv2_use") for i in range(6)],
                         ["layer", "layer", "unused", "layer", "lightmap", None])
        self.assertEqual(self.info(0)["uv2_scale"], "layer")

    def test_texcoord_1_at_8192(self):
        for i in (0, 1, 3, 4, 5):  # Z2: also with a missing lightmap and with a constant UV2
            rows = C28_accessor(self.glb, self.prims[i]["attributes"]["TEXCOORD_1"])
            raw = ODD_UV2[i]
            for got, (u, v) in zip(rows, raw):
                self.assertAlmostEqual(got[0], (u - 0x8000) / 8192, places=6)
                self.assertAlmostEqual(got[1], (v - 0x8000) / 8192, places=6)
        self.assertNotIn("TEXCOORD_1", self.prims[2]["attributes"])
        self.assertIn("_D3_UV2", self.prims[0]["attributes"])

    def test_entry(self):
        (entry,) = entries(self.doc, self.prims[0])
        self.assertEqual(entry, {"shader": "t_odd", "pass": 0, "stage": 2, "slot_type": 12, "slot_name": "layer_12",
                                 "divisor": 512, "texture": "../textures/620.png", "texcoord": 1, "shader_scale": 16,
                                 "in_game": "open", "texture_info": {"texCoord": 1, "extensions": TRANSFORM}})

    def test_layer_entry_beside_it_has_no_transform(self):
        got = {e["shader"]: e for e in entries(self.doc, self.prims[1])}
        self.assertEqual(got["t_odd"]["shader_scale"], 16)
        self.assertNotIn("shader_scale", got["t_layer"])
        self.assertNotIn("texture_info", got["t_layer"])
        self.assertNotIn("in_game", got["t_layer"])

    def test_slot_1(self):
        """A divisor-512 slot-1 entry; the base colour stays on ``TEXCOORD_0`` without a transform."""
        (entry,) = entries(self.doc, self.prims[3])
        self.assertEqual((entry["also_base_color"], entry["texcoord"], entry["shader_scale"]), (True, 1, 16))
        base = self.doc["materials"][self.prims[3]["material"]]["pbrMetallicRoughness"]["baseColorTexture"]
        self.assertEqual(base, {"index": base["index"]})

    def test_extension_lists(self):
        self.assertEqual(self.doc["extensionsUsed"], ["KHR_texture_transform"])
        check_document(self, self.doc)
        I.read_gltf(self.glb)
        self.assertEqual(I.import_mesh(P.parse_appearance(self.source), I.read_gltf(self.glb)), self.source)

    def test_no_odd_row_no_extension(self):
        table = U.load_tables(make_tables(self, rows=ODD_ROWS, known=ODD_KNOWN))
        doc = P.to_gltf(P.parse_appearance(six_mesh(ODD_TEXTURES, [3] * 6, ODD_UV2)),
                        texture_resolver=lambda h: f"../textures/{h}.png", lightmap_resolver=lambda h: None,
                        uv2_rows=lambda block: U.rows_for(ODD_MAPS.get(block.shader_map) if block else None, table))
        self.assertNotIn("extensionsUsed", doc)

    def test_texcoord_rows(self):
        row = U.Uv2Row("x", 0, 0, 12, 512, 0)
        self.assertEqual(U.entry_texcoord(row, "layer", SCALE), (1, None))
        self.assertEqual(U.entry_texcoord(row, "lightmap", SCALE), (1, SCALE))
        self.assertEqual(U.entry_texcoord(row, None, SCALE), (None, None))
        (entry,) = U.layer_entries(U.Uv2Rows((row,), True), None, str, None)
        self.assertEqual((entry["texcoord"], entry["shader_scale"], entry["in_game"]), (None, 16, "open"))
        self.assertNotIn("texture_info", entry)  # no ``TEXCOORD_1`` to transform
        self.assertEqual(U.uv2_use(U.Uv2Rows((row,), True), False), "layer")


def C28_accessor(glb, index):
    return I.read_gltf(glb).accessor(index)


class ExceptedPass(unittest.TestCase):
    """Z2: the x360 and switch rows of the one pass that differs from ps3 are divisor 8192: no transform."""

    def test_shipped_rows(self):
        need_uv2_tables()
        table = U.load_tables()
        name = "assaultBeast_deathTransition"
        for build, odd in (("ps3", True), ("switch", False), ("x360", False)):
            (entry,) = U.layer_entries(U.rows_for((name,), table, build), None, str, "layer")
            self.assertEqual(("texture_info" in entry, "shader_scale" in entry), (odd, odd), build)
            self.assertEqual(entry["texcoord"], 1)


# --- Z3: wrap modes --------------------------------------------------------------------------------

class Samplers(unittest.TestCase):
    """Z3 on ``sampler_for`` and ``to_gltf``."""

    def test_modes(self):
        self.assertEqual(P.sampler_for([("s", 0, 0, 1, 1)]), ({"wrapS": CLAMP, "wrapT": CLAMP}, None))
        self.assertEqual(P.sampler_for([("s", 0, 0, 2, 2)]), ({"wrapS": REPEAT, "wrapT": REPEAT}, None))
        self.assertEqual(P.sampler_for([("s", 0, 0, 1, 2)]), ({"wrapS": CLAMP, "wrapT": REPEAT}, None))
        self.assertEqual(P.sampler_for([]), ({"wrapS": REPEAT, "wrapT": REPEAT}, None))
        self.assertEqual(P.sampler_for(None), ({"wrapS": REPEAT, "wrapT": REPEAT}, None))

    def test_unknown_mode(self):
        sampler, listed = P.sampler_for([("s", 0, 1, 3, 2)])
        self.assertEqual(sampler, {"wrapT": REPEAT})
        self.assertEqual(listed, [{"shader": "s", "pass": 0, "stage": 1, "address_modes": [3, 2], "used": True}])
        sampler, _ = P.sampler_for([("s", 0, 1, 0, 7)])
        self.assertEqual(sampler, {})

    def test_first_pass_decides(self):
        sampler, listed = P.sampler_for([("a", 0, 0, 1, 1), ("a", 1, 2, 2, 2), ("b", 0, 0, 1, 1)])
        self.assertEqual(sampler, {"wrapS": CLAMP, "wrapT": CLAMP})
        self.assertEqual(listed, [{"shader": "a", "pass": 1, "stage": 2, "address_modes": [2, 2], "used": False}])

    def document(self, modes_of):
        source = six_mesh([[]] * 6, [1, 2, 3, 4, 5, 6])
        return P.to_gltf(P.parse_appearance(source), texture_resolver=lambda h: f"../textures/{h}.png",
                         stage_modes=lambda block: modes_of.get(block.shader_map if block else None, []))

    def test_document(self):
        doc = self.document({1: [("c", 0, 0, 1, 1)], 2: [("w", 0, 0, 2, 2)], 3: [("c2", 1, 0, 1, 1)],
                             4: [("u", 0, 0, 5, 1)], 5: []})
        prims = primitives(doc)

        def sampler(i):
            texture = doc["textures"][doc["materials"][prims[i]["material"]]["pbrMetallicRoughness"]
                                      ["baseColorTexture"]["index"]]
            return doc["samplers"][texture["sampler"]]

        self.assertEqual([sampler(i) for i in range(6)],
                         [{"wrapS": CLAMP, "wrapT": CLAMP}, {"wrapS": REPEAT, "wrapT": REPEAT},
                          {"wrapS": CLAMP, "wrapT": CLAMP}, {"wrapT": CLAMP},
                          {"wrapS": REPEAT, "wrapT": REPEAT}, {"wrapS": REPEAT, "wrapT": REPEAT}])
        self.assertEqual(len(doc["samplers"]), 3)  # equal settings shared
        self.assertEqual(len(doc["images"]), 1)  # one image, a texture per sampler
        self.assertEqual(len(doc["textures"]), 3)
        unknown = doc["materials"][prims[3]["material"]]["extras"]["d3sno_export"]["address_modes"]
        self.assertEqual(unknown, [{"shader": "u", "pass": 0, "stage": 0, "address_modes": [5, 1], "used": True}])
        for i in (0, 1, 2, 4, 5):
            self.assertNotIn("extras", doc["materials"][prims[i]["material"]])
        check_document(self, doc)

    def test_without_stage_modes(self):
        source = six_mesh([[]] * 6, [1, 2, 3, 4, 5, 6])
        doc = P.to_gltf(P.parse_appearance(source), texture_resolver=lambda h: f"../textures/{h}.png")
        self.assertEqual(doc["samplers"], [{"wrapS": REPEAT, "wrapT": REPEAT}])


def shader_with_modes(build, passes):
    """A .shd laid out by the generic writer (as ``test_stage15_content.shader_file``); ``passes`` hold
    (slot type, U mode, V mode) per stage."""
    order = ORDER[build]
    count_at, passes_at, block_at, stages_at, tags_at, keys_at = PASS_AT[build]
    header = struct.pack(order + "4I", 0xDEADBEEF, expected_version("shd", build) or 1, 1, 0x99)
    top_name = L.top_record(build, "shd")
    top = L.new_record(build, "shd", top_name)
    struct.pack_into(order + "II", top.raw, 0x0C, 5, len(passes))
    if count_at != 0x10:
        struct.pack_into(order + "I", top.raw, count_at, len(passes))
    slots = {s.offset: s for s in L.pair_slots(build, "shd", top_name)}
    pass_name = slots[passes_at].field.element
    inner = {s.offset: s for s in L.pair_slots(build, "shd", pass_name)}
    records = []
    for k, stages_of in enumerate(passes):
        record = L.new_record(build, "shd", pass_name)
        struct.pack_into(order + "I", record.raw, 0x6C, len(stages_of))
        struct.pack_into(order + "II", record.raw, keys_at, 100 + 2 * k, 101 + 2 * k)
        stages = []
        for slot_type, u, v in stages_of:
            stage = L.new_record(build, "shd", inner[stages_at].field.element)
            struct.pack_into(order + "5If", stage.raw, 0, slot_type, u, v, 3, 4, 0.0)
            stages.append(stage)
        if stages:
            record.pairs[stages_at] = L.PairValue(L.Block(elements=stages))
        record.pairs[tags_at] = L.PairValue(L.Block(struct.pack(order + "IIIi", 1, 2, 0x50, 7)))
        records.append(record)
    top.pairs[passes_at] = L.PairValue(L.Block(elements=records))
    top.pairs[block_at] = L.PairValue(L.Block(SH.shader_block([program(build, 4) for _ in range(2 * len(passes))], build)))
    return L.write_tree(L.SnoTree(build, "shd", header, top))


class SamplersInExport(unittest.TestCase):
    """Z3 in the export: the address modes of the shaders' stages that bind the base colour slot."""

    def setUp(self):
        need_uv2_tables()
        files = T20.export_files()
        # shader 9001 (map 3001): pass 0 binds slot 1 clamped, pass 1 binds it wrapped (listed)
        data = shader_with_modes("switch", [[(1, 1, 1), (19, 2, 2)], [(1, 2, 2)]])
        self.assertEqual([[s.address_modes[:2] for s in p.stages] for p in SH.read_shader(data, "switch").passes],
                         [[(1, 1), (2, 2)], [(2, 2)]])
        files["Act1"][F.name_of("shd", T20.SHADERS[9001])] = F.with_handle(data, 9001, "<")
        self.result, out = export(self, files)
        self.glb = (out / f"meshes/{F.PLAIN_APP}.glb").read_bytes()
        self.doc = glb_json(self.glb)
        self.prims = primitives(self.doc)

    def sampler(self, i):
        material = self.doc["materials"][self.prims[i]["material"]]
        texture = self.doc["textures"][material["pbrMetallicRoughness"]["baseColorTexture"]["index"]]
        return self.doc["samplers"][texture["sampler"]], material.get("extras", {}).get("d3sno_export", {})

    def test_samplers(self):
        for i in (0, 1, 2):  # map 3001
            sampler, extras = self.sampler(i)
            self.assertEqual(sampler, {"wrapS": CLAMP, "wrapT": CLAMP}, i)
            self.assertEqual(extras["address_modes"], [{"shader": T20.SHADERS[9001], "pass": 1, "stage": 0,
                                                        "address_modes": [2, 2], "used": False}])
        for i in (3, 4, 5):  # map 3002 (its shader is not in the archives), -1, a map not in the archives
            sampler, extras = self.sampler(i)
            self.assertEqual(sampler, {"wrapS": REPEAT, "wrapT": REPEAT}, i)
            self.assertNotIn("address_modes", extras)
        check_document(self, self.doc)

    def test_shaders_not_exported(self):
        self.assertNotIn("shd", self.result.coverage["groups"])
        self.assertEqual(I.import_mesh(P.parse_appearance(six_mesh(T20.EXPORT_TEXTURES, T20.EXPORT_MAPS)),
                                       I.read_gltf(self.glb)), six_mesh(T20.EXPORT_TEXTURES, T20.EXPORT_MAPS))


# --- Z4: faster archive writing --------------------------------------------------------------------

class Cipher(unittest.TestCase):
    """Z4: the packet cipher computed on whole integers gives the bytes of the byte-by-byte rule."""

    @staticmethod
    def slow(data, encrypt):
        state, out = A.CIPHER_SEED, bytearray(len(data))
        for i, b in enumerate(data):
            out[i] = b ^ (state & 0xFF)
            c = out[i] if encrypt else b
            state = ((c << 56) | (state >> 8)) & 0xFFFFFFFFFFFFFFFF
        return bytes(out)

    def test_both_ways(self):
        for n in (0, 1, 7, 8, 9, 16, 17, 255, 4099):
            data = Y.data(n + 7, n)
            self.assertEqual(A.encrypt_packet(data), self.slow(data, True), n)
            self.assertEqual(A.decrypt_packet(data), self.slow(data, False), n)
            self.assertEqual(A.decrypt_packet(A.encrypt_packet(data)), data)


class Deflate(unittest.TestCase):
    """Z4: a compressor fed in chunks gives the bytes of ``zlib.compress``; the head is its prefix."""

    def test_head(self):
        import zlib
        for seed, size in ((1, 0), (2, 5000), (3, 70_000), (4, 200_000)):
            data = Y.data(seed, size)
            whole = zlib.compress(data, 9)
            out, complete = A.deflate_head(data, len(whole) + 10)
            self.assertEqual((out, complete), (whole, True))
            for room in (0, 100, 4000):
                out, complete = A.deflate_head(data, room)
                if not complete:
                    self.assertGreater(len(out), room)
                    self.assertEqual(out, whole[:len(out)])
                    self.assertLess(len(out), len(whole))  # stopped before the end


def all_outputs(jobs):
    for src_name, make in Y.SOURCES.items():
        source = make()
        for change_name, change in Y.CHANGES.items():
            replace, add = change()
            yield f"{src_name}/{change_name}", IW.write_changed(source, None, replace, add, jobs=jobs).sha256


class Jobs(unittest.TestCase):
    """Z4: ``jobs`` 1, 2 and 8 give the bytes of 0.21.0 / 0.21.1 for every archive format."""

    def test_expected_outputs(self):
        C28.same_zlib(self)
        for jobs in (1, 2, 8):
            for key, digest in all_outputs(jobs):
                self.assertEqual(digest, Y.GOLDEN_0_21_0[key], (key, jobs))

    def test_rebuild(self):
        C28.same_zlib(self)
        for jobs in (1, 8):
            buffer = io.BytesIO()
            A.rebuild_to(buffer, Y.ps3_source(), remove=["Base\\small.txt"], jobs=jobs)
            self.assertEqual(C28.sha(buffer.getvalue()), Y.GOLDEN_0_21_0["ps3/rebuild_remove"])

    def test_work_ahead_is_used(self):
        """The paths that work ahead run (and the bytes stay the same). ``51`` AB7: ps3 and switch packets
        are no longer compressed ahead; ``52`` AC5: nor are x360 piece starts; source sectors are inflated
        ahead."""
        calls = {"heads": 0, "sectors": 0}
        heads, batch = XA._head_ahead, A._SectorsAhead._batch

        def count(name, fn):
            def wrapper(*args):
                calls[name] += 1
                return fn(*args)
            return wrapper

        with mock.patch.object(XA, "_head_ahead", count("heads", heads)), \
                mock.patch.object(A._SectorsAhead, "_batch", count("sectors", batch)):
            compressible = {f"Base\\f{i}.dat": (b"%05d " % i) * 6000 for i in range(12)}
            compressible["Base\\noise.dat"] = Y.data(5, 900_000)  # many slots: sectors read ahead
            compressible["Base\\zeros.dat"] = bytes(1_500_000)  # x360: its start ahead passes no room
            for kind in ("ps3", "x360"):
                if kind == "x360":
                    source = XA.write_x360_archive([XA.Piece(n, 0, d) for n, d in compressible.items()]).data
                else:
                    source = A.write_archive(compressible, version=6, order=">")
                one = IW.write_changed(source, None, {"Base\\f3.dat": b"new"}, jobs=1)
                many = IW.write_changed(source, None, {"Base\\f3.dat": b"new"}, jobs=4)
                self.assertEqual(one, many, kind)
                self.assertEqual(IW.replace_file(source, "Base\\f3.dat", b"new"), self.written(source))
        self.assertEqual(calls["heads"], 0, calls)  # ``52`` AC5: no compression ahead of the packer
        self.assertGreater(calls["sectors"], 0, calls)

    @staticmethod
    def written(source):
        buffer = io.BytesIO()
        IW.write_changed(source, buffer, {"Base\\f3.dat": b"new"}, jobs=3)
        return buffer.getvalue()

    def test_forced_misses(self):
        """Work compressed ahead that never matches (its key is never asked for) gives the same bytes."""
        C28.same_zlib(self)
        with mock.patch.object(A, "_ahead_key", lambda position: -1 - position):
            for key, digest in all_outputs(4):
                self.assertEqual(digest, Y.GOLDEN_0_21_0[key], key)

    def test_small_windows(self):
        """Small read windows and chunks (``47`` Y1) with workers."""
        C28.same_zlib(self)
        with C28.small_chunks():
            for key, digest in all_outputs(3):
                self.assertEqual(digest, Y.GOLDEN_0_21_0[key], key)

    def test_sectors_read_ahead(self):
        """Sectors inflated ahead are those the reader asks for; the rebuild counts the same packets."""
        source = A.read_archive(Y.switch_source())
        count = len(source.sector_offsets)
        self.assertGreater(count, 3 * A.SECTOR_BATCH)
        expected = [source._decode_sector(k) for k in range(count)]
        with A.Workers(3) as workers, A.reading_ahead(source, workers):
            self.assertIsNotNone(source._ahead)
            for k in (*range(count), 0, 5, 30, 31, 2):
                self.assertEqual(source.sector(k), expected[k], k)
        self.assertIsNone(source._ahead)
        results = [A.rebuild_to(io.BytesIO(), Y.switch_source(), jobs=jobs) for jobs in (1, 4)]
        self.assertEqual([r.encrypted_packets_dropped for r in results], [results[0].encrypted_packets_dropped] * 2)
        self.assertGreater(results[0].encrypted_packets_dropped, 0)
        self.assertEqual(results[0].written, results[1].written)

    def test_in_flight_bound(self):
        workers = A.Workers(3)
        try:
            futures = [workers.submit(lambda: 1) for _ in range(8)]
            self.assertEqual(sum(f is not None for f in futures), 6)  # 2 x jobs
            self.assertEqual(workers.take(futures[0]), 1)
            self.assertIsNotNone(workers.submit(lambda: 2))
        finally:
            workers.close()
        self.assertFalse(A.Workers(1))
        self.assertIsNone(A.Workers(1).submit(lambda: 1))
        self.assertEqual(A.default_jobs(), __import__("os").cpu_count() or 1)


class WorkerFault(unittest.TestCase):
    """Z4: an exception in a worker stops the run and leaves no partial output file."""

    def setUp(self):
        self.root = F.temp_root(self)

    def check(self, patch, source):
        target = self.root / "out.cpk"
        with patch, self.assertRaises(RuntimeError):
            IW.write_changed(source, target, {"Base\\small.txt": b"x"}, jobs=4)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_sector_worker(self):
        def fail(self, batch):
            raise RuntimeError("worker fault")

        for name in ("ps3", "switch"):  # sectors are read ahead in the counting pass of a rebuild
            with self.subTest(name):
                self.check(mock.patch.object(A._SectorsAhead, "_batch", fail), Y.SOURCES[name]())

    def test_compression_worker(self):
        """``52`` AC5: x360 piece starts are no longer compressed ahead; a fault of the worker that reads
        the x360 source ahead stops the run."""
        def fail(self, batch):
            raise RuntimeError("worker fault")

        self.check(mock.patch.object(A._SectorsAhead, "_batch", fail), Y.SOURCES["x360"]())

    def test_apply(self):
        files = F.switch_files()
        for pack in files.values():  # enough slots in every archive for sectors to be read ahead
            pack["Base\\noise.dat"] = Y.data(6, 600_000)
        setup = S.Setup(self, files)
        S.three_edits(setup)

        def fail(self, batch):
            raise RuntimeError("worker fault")

        with mock.patch.object(A._SectorsAhead, "_batch", fail):
            result, out = setup.apply("faulty", jobs=4)
        self.assertNotEqual(result.exit_code, 0)
        self.assertEqual([p for p in out.rglob("*.cpk")] + [p for p in out.rglob("*.part")], [])


class JobsOption(unittest.TestCase):
    """Z4: ``--jobs`` of ``apply``, ``replace`` and ``add``; the outputs do not depend on it."""

    def setUp(self):
        self.root = F.temp_root(self)

    def test_parse(self):
        from d3sno import cli
        parser = cli.make_parser()
        args = parser.parse_args(["replace", "a.cpk", "n", "f", "-o", "o.cpk"])
        self.assertEqual(args.jobs, A.default_jobs())
        self.assertEqual(parser.parse_args(["replace", "a.cpk", "n", "f", "-o", "o.cpk", "--jobs", "3"]).jobs, 3)
        with self.assertRaises(SystemExit), mock.patch("sys.stderr", io.StringIO()):
            parser.parse_args(["replace", "a.cpk", "n", "f", "-o", "o.cpk", "--jobs", "0"])
        for command in (["apply", "g", "e", "--out", "o"], ["add", "a.cpk", "n", "f", "-o", "o"]):
            self.assertEqual(parser.parse_args(command).jobs, A.default_jobs())

    def test_replace_and_add(self):
        new = self.root / "new.txt"
        new.write_bytes(b"replaced\n" * 50)
        for name in ("ps3", "switch", "x360"):
            source = self.root / f"{name}.cpk"
            source.write_bytes(Y.SOURCES[name]())
            outs = []
            for jobs in ("1", "5"):
                out = self.root / f"{name}_{jobs}.cpk"
                self.assertEqual(C28.quiet(["replace", str(source), "Base\\small.txt", str(new), "-o", str(out),
                                            "--jobs", jobs]), 0)
                added = self.root / f"{name}_{jobs}_add.cpk"
                self.assertEqual(C28.quiet(["add", str(source), "New\\x.dat", str(new), "-o", str(added),
                                            "--jobs", jobs]), 0)
                outs.append((out.read_bytes(), added.read_bytes()))
            self.assertEqual(outs[0], outs[1], name)
            self.assertEqual(outs[0][0], IW.replace_file(source.read_bytes(), "Base\\small.txt", new.read_bytes()))

    def test_apply(self):
        setup = S.Setup(self)
        S.three_edits(setup)
        one, out_one = setup.ok("one", jobs=1)
        many, out_many = setup.ok("many", jobs=6)
        self.assertEqual(one["archives"], many["archives"])
        for rel in one["archives_written"]:
            self.assertEqual((out_one / rel).read_bytes(), (out_many / rel).read_bytes())
        self.assertNotIn("jobs", one["options"])  # the report does not depend on it
        refused = AP.apply_export(setup.game, setup.edit, setup.root / "bad", jobs=0)
        self.assertEqual(refused.exit_code, AP.EXIT_REFUSED)


class JobsMemory(unittest.TestCase):
    """Z4: with 8 workers the Y1 bound grows by at most 8 pieces of work (here: 8 x 2 MB)."""

    def test_peak(self):
        root = F.temp_root(self)
        new = b"the replaced file\n" * 300
        for kind in ("ps3", "x360"):
            with self.subTest(kind):
                source = root / f"{kind}.cpk"
                C28.big_archive(kind, source)
                cpk = C28.mapped(self, source)
                done, peak = C28.peak_of(lambda: IW.write_changed(cpk, root / f"{kind}_out.cpk",
                                                                  {"Small\\s.txt": new}, jobs=8))
                self.assertLess(peak, C28.BOUND + 8 * 2 * C28.MB)
                self.assertEqual(done.size, (root / f"{kind}_out.cpk").stat().st_size)
                one = IW.write_changed(cpk, None, {"Small\\s.txt": new}, jobs=1)
                self.assertEqual(one, done)


class Version(unittest.TestCase):
    def test_version(self):
        self.assertGreaterEqual(tuple(int(x) for x in d3sno.__version__.split(".")), (0, 21, 2))
        self.assertGreaterEqual(d3sno.STAGE, 21)


if __name__ == "__main__":
    unittest.main()
