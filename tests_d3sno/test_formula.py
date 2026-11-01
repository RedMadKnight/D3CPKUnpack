import math
import struct
import unittest

import builders as B
from d3sno import FormatError
from d3sno import formula as F
from d3sno import power as P


def code(build="switch"):
    return B.Code(">" if build == "ps3" else "<")


def run(c, build="switch", ctx=None, strict=False):
    return F.evaluate(c, build, ctx, strict=strict)


def nan():
    return float("nan")


class Words(F.EvalContext):
    """A context with a fixed list of random words."""

    def __init__(self, *words, mode=None):
        super().__init__()
        self.words = list(words)
        self.drawn = 0
        self.random_mode = mode

    def random_word(self):
        self.drawn += 1
        return self.words.pop(0)


class Decode(unittest.TestCase):
    def test_opcode_is_first_byte_on_both_builds(self):
        for build in ("ps3", "switch"):
            ins = F.decode(code(build).f(2.5).end(), build)
            self.assertEqual([i.name for i in ins], ["PUSHF", "END"])
            self.assertEqual(ins[0].operands, (2.5,))
            self.assertEqual(ins[1].offset, 8)

    def test_pushid_operands_in_build_order(self):
        ins = F.decode(code("ps3").ident(1, 2, -2, 4).end(), "ps3")[0]
        self.assertEqual(ins.operands, (1, 2, 0xFFFFFFFE, 4))

    def test_call_id_in_first_byte(self):
        self.assertEqual(F.decode(code("ps3").call(7).end(), "ps3")[0].operands, (7,))

    def test_pushid1(self):
        c = code()
        c.op(4)
        c.out += b"\x05\0\0\0"
        self.assertEqual(F.decode(c.end(), "switch")[0].operands, (5,))

    def test_upper_bytes_are_ignored(self):
        self.assertEqual(F.decode(b"\x00\x01\x02\x03", "ps3")[0].name, "END")

    def test_stops_at_end(self):
        data = code().f(1).end() + b"\xFF\xFF\xFF"
        self.assertEqual([i.name for i in F.decode(data, "switch")], ["PUSHF", "END"])

    def test_unknown_opcode_ends_the_list(self):
        ins = F.decode(bytes((18, 0, 0, 0)) + code().end(), "ps3")
        self.assertEqual([i.name for i in ins], ["OP18"])

    def test_opcode_15_is_listed(self):
        self.assertEqual(F.decode(bytes((15, 0, 0, 0)), "ps3")[0].name, "OP15")

    def test_truncated_operand(self):
        ins = F.decode(bytes((5, 0, 0, 0)) + bytes(12), "ps3")
        self.assertTrue(ins[0].truncated)
        self.assertIn("past the code", F.disassemble(bytes((5, 0, 0, 0)), "ps3"))

    def test_disassembly(self):
        c = code("ps3").ident(0, 0, -2, 0).f(1.5).call(1).op(11).end()
        text = F.disassemble(c, "ps3")
        self.assertIn("class=0 a=0 b=-2 c=0 ; Attribute", text)
        self.assertIn("PUSHF 1.5", text)
        self.assertIn("CALL Max", text)


class Arithmetic(unittest.TestCase):
    def test_binary_order(self):
        self.assertEqual(run(code().f(7).f(2).op(12).end()), 5)
        self.assertEqual(run(code().f(8).f(2).op(14).end()), 4)
        self.assertEqual(run(code().f(3).f(2).op(13).end()), 6)
        self.assertEqual(run(code().f(3).f(2).op(11).end()), 5)

    def test_divide_by_zero(self):
        self.assertEqual(run(code().f(8).f(0).op(14).end()), 0)

    def test_compare(self):
        self.assertEqual(run(code().f(1).f(2).op(7).end()), 1.0)
        self.assertEqual(run(code().f(2).f(1).op(7).end()), 0.0)
        self.assertEqual(run(code().f(2).f(1).op(8).end()), 1.0)

    def test_equality_epsilon(self):
        self.assertEqual(run(code().f(1).f(1.0000001).op(9).end()), 1.0)
        self.assertEqual(run(code().f(1).f(1.001).op(9).end()), 0.0)
        self.assertEqual(run(code().f(1).f(1.001).op(10).end()), 1.0)
        self.assertEqual(run(code().f(1).f(1.0000001).op(10).end()), 0.0)

    def test_neg(self):
        self.assertEqual(run(code().f(3).op(16).end()), -3)

    def test_select(self):
        self.assertEqual(run(code().f(1).f(10).f(20).op(17).end()), 10)
        self.assertEqual(run(code().f(0).f(10).f(20).op(17).end()), 20)
        self.assertEqual(run(code().f(-5).f(10).f(20).op(17).end()), 20)

    def test_select_threshold(self):
        self.assertEqual(run(code().f(1e-7).f(10).f(20).op(17).end()), 20)

    def test_single_precision(self):
        self.assertAlmostEqual(run(code().f(0.1).end()), 0.1, places=7)
        self.assertNotEqual(run(code().f(0.1).end()), 0.1)

    def test_overflow_is_infinite(self):
        self.assertEqual(run(code().f(3e38).f(3e38).op(13).end()), math.inf)


class Functions(unittest.TestCase):
    def test_floor_pow_log(self):
        self.assertEqual(run(code().f(2.7).call(5).end()), 2)
        self.assertEqual(run(code().f(-2.5).call(5).end()), -3)
        self.assertEqual(run(code().f(2).f(3).call(7).end()), 8)
        self.assertAlmostEqual(run(code().f(math.e).call(8).end()), 1, places=6)

    def test_min_max_nan_picks_the_second_argument(self):
        self.assertEqual(run(code().f(1).f(2).call(0).end()), 1)
        self.assertEqual(run(code().f(1).f(2).call(1).end()), 2)
        self.assertEqual(run(code().f(nan()).f(2).call(0).end()), 2)
        self.assertEqual(run(code().f(nan()).f(2).call(1).end()), 2)
        self.assertTrue(math.isnan(run(code().f(2).f(nan()).call(0).end())))

    def test_pin(self):
        self.assertEqual(run(code().f(5).f(1).f(3).call(2).end()), 3)
        self.assertEqual(run(code().f(0).f(1).f(3).call(2).end()), 1)
        self.assertEqual(run(code().f(2).f(1).f(3).call(2).end()), 2)

    def test_pin_low_test_comes_first(self):
        self.assertEqual(run(code().f(0).f(5).f(3).call(2).end()), 5)
        self.assertEqual(run(code().f(10).f(5).f(3).call(2).end()), 3)
        self.assertEqual(run(code().f(4).f(5).f(3).call(2).end()), 5)

    def test_diminishing(self):
        self.assertAlmostEqual(run(code().f(1).f(3).f(9).call(6).end()), 3, places=5)
        self.assertAlmostEqual(run(code().f(22).f(3).f(9).call(6).end()), 9, places=5)

    def test_diminishing_has_no_clamp(self):
        self.assertAlmostEqual(run(code().f(106).f(3).f(9).call(6).end()), 15, places=4)

    def test_diminishing_small_sum(self):
        self.assertEqual(run(code().f(-7).f(3).f(9).call(6).end()), 0)

    def test_range_int(self):
        ctx = Words(17)
        self.assertEqual(run(code().f(10).f(5).call(3).end(), ctx=ctx), 15)
        self.assertEqual(ctx.drawn, 1)

    def test_range_int_truncates(self):
        self.assertEqual(run(code().f(10.9).f(5.9).call(3).end(), ctx=Words(17)), 15)

    def test_range_int_converts_as_unsigned(self):
        value = run(code().f(-20).f(5).call(3).end(), ctx=Words(0))
        self.assertEqual(value, 4294967296.0)

    def test_range_int_wrapped_modulus_uses_the_word_whole(self):
        self.assertEqual(run(code().f(0).f(-1).call(3).end(), ctx=Words(7)), 7)

    def test_random_int(self):
        self.assertEqual(run(code().f(10).f(15).call(4).end(), ctx=Words(17)), 15)

    def test_random_int_hi_below_lo(self):
        ctx = Words()
        self.assertEqual(run(code().f(5.5).f(3).call(4).end(), ctx=ctx), 5.5)
        self.assertEqual(ctx.drawn, 0)
        self.assertTrue(ctx.warnings)

    def test_range_float(self):
        self.assertEqual(run(code().f(10).f(2).call(9).end(), ctx=Words(1 << 31)), 11)

    def test_random_float(self):
        self.assertEqual(run(code().f(10).f(14).call(10).end(), ctx=Words(1 << 31)), 12)

    def test_random_float_hi_below_lo(self):
        ctx = Words()
        self.assertEqual(run(code().f(5).f(3).call(10).end(), ctx=ctx), 5)
        self.assertTrue(ctx.warnings)

    def test_flags(self):
        cases = (
            (9, 10.5, 2, "maximum", 12.5, 0), (9, 10.5, 2, "minimum", 10.5, 0),
            (9, 10.5, 2, "draw_maximum", 12.5, 1), (9, 10.5, 2, "draw_minimum", 10.5, 1),
            (3, 10.5, 2.5, "maximum", 12, 0), (3, 10.5, 2.5, "minimum", 10, 0),
            (4, 10.5, 14.5, "maximum", 14, 0), (4, 10.5, 14.5, "draw_minimum", 10, 1),
            (10, 10, 14, "draw_maximum", 14, 1),
        )
        for fn, a, b, mode, want, drawn in cases:
            with self.subTest((fn, mode)):
                ctx = Words(1, mode=mode)
                self.assertEqual(run(code().f(a).f(b).call(fn).end(), ctx=ctx), want)
                self.assertEqual(ctx.drawn, drawn)

    def test_both_select_branches_consume_random_numbers(self):
        ctx = Words(1 << 31, 0)
        c = code().f(1).f(0).f(1).call(10).f(0).f(1).call(10).op(17).end()
        self.assertEqual(run(c, ctx=ctx), 0.5)
        self.assertEqual(ctx.drawn, 2)

    def test_table_comes_from_the_context(self):
        class Ctx(F.EvalContext):
            def table(self, a, b):
                return a * b

        self.assertEqual(run(code().f(10).f(2).call(11).end(), ctx=Ctx()), 20)

    def test_missing_source_fails_the_formula(self):
        for fn in (3, 4, 9, 10, 11):
            self.assertEqual(run(code().f(1).f(2).call(fn).end()), 0)
            with self.assertRaises(FormatError):
                run(code().f(1).f(2).call(fn).end(), strict=True)


class Failures(unittest.TestCase):
    def check_failed(self, c):
        ctx = F.EvalContext()
        self.assertEqual(run(c, ctx=ctx), 0)
        self.assertTrue(ctx.warnings)
        with self.assertRaises(FormatError):
            run(c, strict=True)

    def test_end_on_empty_stack(self):
        self.check_failed(code().end())

    def test_invalid_opcodes(self):
        for op in (2, 3, 18, 200):
            self.check_failed(code().f(1).op(op).end())

    def test_no_end(self):
        self.check_failed(bytes(code().f(1).out))

    def test_empty_code(self):
        self.check_failed(b"")

    def test_operand_past_the_code(self):
        self.check_failed(bytes((6, 0, 0, 0)))
        self.check_failed(bytes((5, 0, 0, 0)) + bytes(8))

    def test_identifier_failure_ends_the_run(self):
        self.check_failed(code().f(5).ident(1).op(11).end())

    def test_opcode_15_is_fatal(self):
        with self.assertRaises(FormatError):
            run(code().f(1).op(15).end())

    def test_opcode_15_is_fatal_from_a_nested_run(self):
        tag = P.formula_tag(0, "switch")
        power = P.parse_power(B.build_power("switch", general=[
            (4, tag, ("", code().op(15).end())),
        ]))
        with self.assertRaises(FormatError):
            F.evaluate(code().ident(23).end(), "switch", F.EvalContext(), power)

    def test_failed_nested_formula_is_0_and_the_outer_goes_on(self):
        tag = P.formula_tag(0, "switch")
        power = P.parse_power(B.build_power("switch", general=[(4, tag, ("", code().end()))]))
        c = code().f(5).ident(23).op(11).end()
        self.assertEqual(F.evaluate(c, "switch", F.EvalContext(), power), 5)

    def test_pop_from_empty_stack_gives_0_and_continues(self):
        ctx = F.EvalContext()
        self.assertEqual(run(code().f(1).op(11).end(), ctx=ctx), 1)
        self.assertTrue(ctx.warnings)
        with self.assertRaises(FormatError):
            run(code().f(1).op(11).end(), strict=True)

    def test_push_on_full_stack_is_dropped(self):
        c = code()
        for i in range(65):
            c.f(i)
        ctx = F.EvalContext()
        self.assertEqual(run(c.end(), ctx=ctx), 63)
        self.assertTrue(ctx.warnings)

    def test_stack_holds_64(self):
        c = code()
        for i in range(64):
            c.f(i)
        ctx = F.EvalContext()
        self.assertEqual(run(c.end(), ctx=ctx), 63)
        self.assertEqual(ctx.warnings, [])

    def test_unknown_function_pops_nothing_and_pushes_0(self):
        ctx = F.EvalContext()
        self.assertEqual(run(code().f(5).f(2).call(12).op(11).end(), ctx=ctx), 2)
        self.assertTrue(ctx.warnings)

    def test_upper_bytes_of_opcode_words_are_ignored(self):
        self.assertEqual(run(b"\x06\xFF\xFF\xFF" + struct.pack("<f", 3) + b"\x00\xAA\xBB\xCC"), 3)

    def test_bytes_after_end_are_never_looked_at(self):
        self.assertEqual(run(code().f(3).end() + b"\xFF\xFF"), 3)


class Identifiers(unittest.TestCase):
    def test_context_values(self):
        ctx = F.ValueContext({1: 4, (0, 0x41, 0xFFFFFFFE, 0): 9})
        self.assertEqual(run(code().ident(1).end(), ctx=ctx), 4)
        self.assertEqual(run(code().ident(0, 0x41, -2).end(), ctx=ctx), 9)

    def test_op_4_is_op_5_with_zero_operands(self):
        c = code()
        c.op(4)
        c.out += struct.pack("<I", 1)
        ctx = F.ValueContext({(1, 0, 0, 0): 4})
        self.assertEqual(run(c.end(), ctx=ctx), 4)

    def test_lost_literal_is_not_an_error(self):
        ctx = F.ValueContext({(0, 0, 0xFFFFFFFE, 0): 0})
        self.assertEqual(run(code("ps3").ident(0, 0, -2, 0).end(), "ps3", ctx, strict=True), 0)

    def test_class_5_fails_even_with_a_value(self):
        ctx = F.ValueContext({5: 9})
        self.assertEqual(run(code().f(1).ident(5).op(11).end(), ctx=ctx), 0)

    def test_classes_above_the_last_fail(self):
        ctx = F.ValueContext({114: 9, 90: 9, 113: 4, 89: 4})
        self.assertEqual(run(code().ident(114).end(), ctx=ctx), 0)
        self.assertEqual(run(code().ident(113).end(), ctx=ctx), 4)
        self.assertEqual(run(code("ps3").ident(90).end(), "ps3", ctx), 0)
        self.assertEqual(run(code("ps3").ident(89).end(), "ps3", ctx), 4)

    def test_table_classes_are_constants(self):
        cases = ((87, 0), (93, 6), (100, 13), (102, 14), (103, 15), (110, 22))
        for cls, row in cases:
            self.assertEqual(run(code().ident(cls).end(), strict=True), row)
            self.assertEqual(run(code("ps3").ident(cls - 24).end(), "ps3", strict=True), row)

    def test_non_table_classes_go_to_the_context(self):
        ctx = F.ValueContext({101: 1, 111: 2, 112: 3, 113: 4})
        for cls, value in ((101, 1), (111, 2), (112, 3), (113, 4)):
            self.assertEqual(run(code().ident(cls).end(), ctx=ctx), value)

    def test_class_names(self):
        for build, value in (("ps3", 63), ("switch", 87)):
            self.assertEqual(F.class_name(value, build), "DmgTier1")
        self.assertEqual(F.class_name(70, "ps3"), "Healing")
        self.assertEqual(F.class_name(94, "switch"), "Healing")
        self.assertEqual(F.class_name(77, "ps3"), "HealthMin")
        self.assertEqual(F.class_name(101, "switch"), "HealthMin")
        self.assertEqual(F.class_name(78, "ps3"), "LegendaryProcDmg")
        self.assertEqual(F.class_name(102, "switch"), "LegendaryProcDmg")
        self.assertEqual(F.class_name(103, "switch"), "OffhandDmgMinMin")
        self.assertEqual(F.class_name(111, "switch"), "AreaDamageMin")
        self.assertEqual(F.class_name(89, "ps3"), "TraitActive")
        self.assertEqual(F.class_name(6, "ps3"), "MonsterDamagePhysicalMin")
        self.assertEqual(F.class_name(7, "ps3"), "MonsterDamagePhysicalDelta")
        self.assertEqual(F.class_name(16, "ps3"), "MonsterDamageArcaneMin")
        self.assertEqual(F.class_name(23, "ps3"), "ScriptSlot0")
        self.assertEqual(F.class_name(93, "switch"), "DmgTier7")
        self.assertEqual(F.class_name(71, "ps3"), "WDCost")
        self.assertEqual(F.class_name(96, "switch"), "RuneDamageBonus")

    def test_slot_ranges(self):
        self.assertEqual(F.slot_of_class(62, "ps3"), 39)
        self.assertIsNone(F.slot_of_class(63, "ps3"))
        self.assertEqual(F.slot_of_class(86, "switch"), 63)
        self.assertIsNone(F.slot_of_class(87, "switch"))
        self.assertIsNone(F.slot_of_class(22, "switch"))


class Slots(unittest.TestCase):
    def power(self, general, build="switch", contact0=()):
        return P.parse_power(B.build_power(build, general=general, contact0=contact0), build)

    def tag(self, n):
        return P.formula_tag(n, "switch")

    def test_missing_tag_is_zero(self):
        self.assertEqual(F.evaluate_slot(self.power([]), 3), 0)

    def test_int_float_and_formula(self):
        p = self.power([
            (0, self.tag(0), 5), (1, self.tag(1), 2.5),
            (4, self.tag(2), ("", code().f(9).end())),
        ])
        self.assertEqual(F.evaluate_slot(p, 0), 5.0)
        self.assertEqual(F.evaluate_slot(p, 1), 2.5)
        self.assertEqual(F.evaluate_slot(p, 2), 9)

    def test_other_type_warns(self):
        p = self.power([(2, self.tag(0), 5)])
        ctx = F.EvalContext()
        self.assertEqual(F.evaluate_slot(p, 0, ctx), 0)
        self.assertEqual(len(ctx.warnings), 1)

    def test_slot_above_the_limit_gives_0_and_a_log(self):
        ctx = F.EvalContext()
        self.assertEqual(F.evaluate_slot(self.power([]), 64, ctx), 0)
        self.assertTrue(ctx.warnings)
        with self.assertRaises(FormatError):
            F.evaluate_slot(self.power([]), 64, strict=True)

    def test_identifier_reaches_slot_in_this_power(self):
        p = self.power([
            (0, self.tag(4), 21),
            (4, self.tag(5), ("", code().ident(23 + 4).f(2).op(13).end())),
        ])
        self.assertEqual(F.evaluate_slot(p, 5), 42)

    def test_self_reference_ending_in_plus_1_gives_32(self):
        p = self.power([(4, self.tag(0), ("", code().ident(23).f(1).op(11).end()))])
        ctx = F.EvalContext()
        self.assertEqual(F.evaluate_slot(p, 0, ctx), 32)
        self.assertEqual(len(ctx.warnings), 1)

    def test_class_22_with_minus_1_is_the_current_power(self):
        p = self.power([(1, 0x99, 6.0), (4, 0x98, ("", code().ident(22, -1, 0x99).end()))])
        self.assertEqual(F.evaluate_tag(p, 0x98), 6)

    def test_class_22_other_power(self):
        other = self.power([(4, 0x99, ("", code().f(6).end()))])

        class Ctx(F.EvalContext):
            def power(self, handle):
                return other if handle == 1234 else None

        self.assertEqual(run(code().ident(22, 1234, 0x99).end(), ctx=Ctx()), 6)
        self.assertEqual(run(code().ident(22, 1234, 0x98).end(), ctx=Ctx()), 0)

    def test_tag_outside_general_reads_the_first_contact_map(self):
        p = self.power([], contact0=[(1, 0x99, 6.0)])
        self.assertEqual(F.evaluate_tag(p, 0x99), 6)

    def test_pvp_map_wins_in_a_pvp_game(self):
        p = self.power([(1, 0x99, 6.0)])
        p.maps["PvpGeneral"].records.append(P.TagRecord(1, 0x99, 8.0, 0))
        self.assertEqual(F.evaluate_tag(p, 0x99), 6)
        ctx = F.EvalContext()
        ctx.pvp = True
        self.assertEqual(F.evaluate_tag(p, 0x99, ctx), 8)

    def test_missing_tag_takes_its_default(self):
        class Ctx(F.EvalContext):
            def tag_default(self, tag):
                return -1.0

        self.assertEqual(F.evaluate_tag(self.power([]), 0x99, Ctx()), -1)

    def test_nested_run_does_not_inherit_the_skill_level(self):
        p = P.parse_power(B.build_power("switch", general=[
            (4, self.tag(0), ("", code().ident(1).end())),
        ]))
        c = code().ident(1).ident(23).op(11).end()
        ctx = F.ValueContext({1: 5})
        self.assertEqual(F.evaluate(c, "switch", ctx, p), 5)
        ctx = F.ValueContext({1: 5, ("skill", p.header.handle): 2})
        self.assertEqual(F.evaluate(c, "switch", ctx, p), 7)

    def chain(self, depth):
        powers = {}
        for i in range(depth):
            powers[i] = self.power([(4, 1, ("", code().ident(22, i + 1, 1).end()))])
        powers[depth] = self.power([(1, 1, 7.0)])
        return powers

    def test_read_past_the_limit_gives_0(self):
        class Ctx(F.EvalContext):
            def __init__(self, powers):
                super().__init__()
                self.powers = powers

            def power(self, handle):
                return self.powers[handle]

        self.assertEqual(F.evaluate_tag(self.chain(31)[0], 1, Ctx(self.chain(31))), 7)
        ctx = Ctx(self.chain(32))
        self.assertEqual(F.evaluate_tag(self.chain(32)[0], 1, ctx), 0)
        self.assertTrue(ctx.warnings)
