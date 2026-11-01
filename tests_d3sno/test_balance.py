import struct
import unittest

import builders as B
from d3sno import FormatError
from d3sno import balance as G
from d3sno import formula as F


class Ids(unittest.TestCase):
    def test_examples(self):
        self.assertEqual(G.gbid("axe"), 109694)
        self.assertEqual(G.gbid("AXE"), 109694)
        self.assertEqual(G.gbid("all"), G.GBID_ALL)

    def test_empty_name(self):
        self.assertEqual(G.gbid(""), -1)

    def test_name_is_cut_to_255_bytes(self):
        self.assertEqual(G.gbid("a" * 255 + "b"), G.gbid("a" * 255))
        self.assertNotEqual(G.gbid("a" * 254 + "b"), G.gbid("a" * 254))

    def test_signed(self):
        self.assertLess(G.gbid("a long name that wraps past the sign bit"), 1 << 31)
        self.assertEqual(G.gbid("\xFF"), -1)


class Tables(unittest.TestCase):
    def test_every_table_on_both_builds(self):
        for kind, (name, sw_slot, sw_row, ps3_slot, ps3_row) in G.TABLES.items():
            if kind == G.FORMULA_TYPE:
                continue
            for build, slot, row in (("switch", sw_slot, sw_row), ("ps3", ps3_slot, ps3_row)):
                with self.subTest((kind, build)):
                    if slot is None or row is None:
                        read = G.parse_balance(B.build_gam(build, kind))
                        self.assertEqual((read.table, read.rows), (name, []))
                        continue
                    read = G.parse_balance(B.build_gam(build, kind, slot, ["Axe", None], row))
                    self.assertEqual((read.type, read.table, read.row_size), (kind, name, row))
                    self.assertEqual(read.rows[0].name, "Axe")
                    self.assertEqual(read.rows[0].id, 109694)
                    self.assertIsNone(read.rows[1].name)
                    self.assertEqual(len(read.rows[1].raw), row)

    def test_table_count(self):
        self.assertEqual(len(G.TABLES), 35)

    def test_table_seen_on_ps3_as_bytes_only(self):
        read = G.parse_balance(B.build_gam("ps3", 28, extra_slots={0x348: bytes(600)}))
        self.assertEqual((read.table, read.row_size), ("handicap_levels", None))
        self.assertEqual(len(read.raw_slots[0x348]), 600)

    def test_named_rows_of_added_tables_resolve(self):
        table = G.parse_balance(B.build_gam("switch", 50, 0x228, ["Axe"], 136))
        self.assertEqual(G.GbidResolver([table]).resolve(109694)[1].name, "Axe")

    def test_size_must_be_a_multiple_of_row(self):
        data = bytearray(B.build_gam("switch", 11, 0xB8, ["Axe"], 32))
        struct.pack_into("<I", data, 0xBC, 31)
        with self.assertRaises(FormatError):
            G.parse_balance(bytes(data))

    def test_table_out_of_bounds(self):
        data = bytearray(B.build_gam("switch", 11, 0xB8, ["Axe"], 32))
        struct.pack_into("<I", data, 0xB8, 0xFFFFFF)
        with self.assertRaises(FormatError):
            G.parse_balance(bytes(data))

    def test_fixed_part(self):
        with self.assertRaises(FormatError):
            G.parse_balance(B.build_gam("ps3", 11)[:0x3E0])

    def test_empty_types(self):
        for kind in (9, -1):
            read = G.parse_balance(B.build_gam("switch", kind))
            self.assertEqual((read.table, read.rows), (None, []))

    def test_other_slots_are_raw(self):
        read = G.parse_balance(B.build_gam("switch", 11, 0xB8, [], 32, {0x1D8: b"raw!"}))
        self.assertEqual(read.raw_slots, {0x1D8: b"raw!"})

    def test_stale_slot_is_not_an_error(self):
        data = bytearray(B.build_gam("switch", 11, 0xB8, [], 32))
        struct.pack_into("<II", data, 0x1D8, 0xFFFFFFF, 0xFFFF)
        self.assertEqual(G.parse_balance(bytes(data)).raw_slots, {})

    def test_name_that_is_not_text_is_no_name(self):
        data = bytearray(B.build_gam("switch", 11, 0xB8, ["Axe"], 32))
        table = struct.unpack_from("<I", data, 0xB8)[0] + 0x10
        name_at = struct.unpack_from("<I", data, table)[0] + 0x10
        data[name_at] = 0x01
        self.assertIsNone(G.parse_balance(bytes(data)).rows[0].name)


class Resolver(unittest.TestCase):
    def test_resolve(self):
        table = G.parse_balance(B.build_gam("switch", 11, 0xB8, ["Axe"], 32))
        resolver = G.GbidResolver([table])
        self.assertEqual(resolver.resolve(109694)[1].name, "Axe")

    def test_unsigned_value(self):
        name = "\xFF"
        table = G.parse_balance(B.build_gam("switch", 11, 0xB8, [], 32))
        resolver = G.GbidResolver([table])
        table.rows = [G.BalanceRow(0, name, b"")]
        resolver.add(table)
        self.assertIsNotNone(resolver.resolve(0xFFFFFFFF))

    def test_miss_is_not_an_error(self):
        self.assertIsNone(G.GbidResolver().resolve(5))

    def test_sentinel(self):
        self.assertTrue(G.GbidResolver.is_sentinel(109305))


class FormulaTable(unittest.TestCase):
    ROWS = [(f"Row{i}", [float(i), 1.0, i * 10.0]) for i in range(16)]

    def rows(self, build="switch"):
        return G.parse_balance(B.build_formula_gam(build, self.ROWS)).formula_rows

    def test_row(self):
        for build in ("ps3", "switch"):
            rows = self.rows(build)
            self.assertEqual(rows[3].name, "Row3")
            self.assertEqual(rows[3].values[0], 3.0)
            self.assertEqual(len(rows[3].values), 76)

    def test_row_size(self):
        self.assertEqual(len(B.formula_rows("ps3", self.ROWS[:1])), 1328)

    def test_table_truncates(self):
        self.assertEqual(G.table_value(self.rows(), 3.9, 2.9), 30.0)
        self.assertEqual(G.table_value(self.rows(), 3.9, 0.0), 3.0)

    def test_out_of_range_gives_zero(self):
        rows = self.rows()
        self.assertEqual(G.table_value(rows, 0, 76), 0.0)
        self.assertEqual(G.table_value(rows, 16, 0), 0.0)
        self.assertEqual(G.table_value(rows, -1, 0), 0.0)
        self.assertEqual(G.table_value(rows, float("nan"), 0), 0.0)

    def test_class_rows(self):
        cases = (
            (87, 0), (93, 6), (94, 7), (95, 8), (96, 9), (97, 10), (100, 13), (102, 14),
            (103, 15), (110, 22),
        )
        for cls, row in cases:
            self.assertEqual(G.table_row_of_class(cls, "switch"), row)
            self.assertEqual(G.table_row_of_class(cls - 24, "ps3"), row)
        self.assertIsNone(G.table_row_of_class(101, "switch"))
        self.assertIsNone(G.table_row_of_class(77, "ps3"))
        self.assertIsNone(G.table_row_of_class(86, "switch"))
        self.assertIsNone(G.table_row_of_class(111, "switch"))

    def test_context_runs_a_table_formula(self):
        ctx = G.TableContext(self.rows(), "switch", {1: 2})
        code = B.Code("<").ident(90).ident(1).call(11).end()
        self.assertEqual(F.evaluate(code, "switch", ctx), 30.0)

    def test_context_passes_other_classes_on(self):
        ctx = G.TableContext(self.rows(), "switch", {101: 7})
        self.assertEqual(F.evaluate(B.Code("<").ident(101).end(), "switch", ctx), 7)
        with self.assertRaises(FormatError):
            F.evaluate(B.Code("<").ident(2).end(), "switch", ctx, strict=True)

    def monster_levels(self, health):
        data = bytearray(B.build_gam("switch", 5, 0x78, [None, None], 240))
        table = struct.unpack_from("<I", data, 0x78)[0] + 0x10
        struct.pack_into("<f", data, table + 240 + 0x14, health)
        return G.parse_balance(bytes(data))

    def test_health_minimum_reads_monster_levels(self):
        ctx = G.TableContext(
            self.rows(), "switch", monster_levels=self.monster_levels(50.0), level=2,
            difficulty_scalar=3.0,
        )
        self.assertEqual(F.evaluate(B.Code("<").ident(101).end(), "switch", ctx), 150)

    def test_health_minimum_level_out_of_range_fails_the_formula(self):
        ctx = G.TableContext(
            self.rows(), "switch", monster_levels=self.monster_levels(50.0), level=9,
        )
        self.assertEqual(F.evaluate(B.Code("<").ident(101).end(), "switch", ctx), 0)
        self.assertTrue(ctx.warnings)
