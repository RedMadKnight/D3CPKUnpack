"""Stage 15, formulas (W15-A1 to A7): ``d3sno.formula_write``. Synthetic data only."""

import random
import struct
import unittest

from d3sno import formula, formula_write as fw, power, write
from d3sno.binary import ByteView, FormatError, Pair

BUILDS = ("ps3", "switch", "x360")
NAMES = fw.FormulaNames(
    attributes={"Strength": 5, "Damage_Bonus": 9, "sLevel": 77, "Zero_Attr": 0},
    powers={"Bash": 0x1234, "Cleave": 0x2345}, extra={"OldPower": 0x999},
    power_tags={"Cooldown Time": 0x30513},
)


def order(build):
    return ">" if build in ("ps3", "x360") else "<"


def op(code):
    return bytes((code, 0, 0, 0))


def pushid(build, cls, a, b, c):
    return op(5) + struct.pack(order(build) + "4I", cls, a & 0xFFFFFFFF, b & 0xFFFFFFFF, c)


def pushf(build, value=None, bits=None):
    if bits is None:
        bits = struct.unpack("<I", struct.pack("<f", value))[0]
    return op(6) + struct.pack(order(build) + "I", bits)


def call(fn):
    return op(1) + bytes((fn, 0, 0, 0))


def ops(code):
    return [ins.opcode for ins in formula.decode(code, "switch")]


UNKNOWN = (0, 0, 0xFFFFFFFE, 0)


class RecordFrame(unittest.TestCase):
    """W15-A1."""

    def test_layout(self):
        for build in BUILDS:
            code = pushf(build, 1.0) + op(0)
            raw = fw.formula_record("1.0", code, build)
            head = struct.unpack(order(build) + "8I", raw[:32])
            self.assertEqual(head, (0, 0, 0, 0, 0, 4, 0, 12))
            self.assertEqual(raw[32:36], b"1.0\0")
            self.assertEqual(raw[36:], code)

    def test_padding_and_read_back(self):
        for build in BUILDS:
            for text in ("", "a", "ab", "abc", "abcd", "sLevel * 2"):
                code = fw.compile_formula(text, build) if text else b""
                raw = fw.formula_record(text, code, build)
                pad = -(len(text) + 1) % 4
                self.assertEqual(raw[32 + len(text):32 + len(text) + 1 + pad], bytes(1 + pad))
                read, end = power._read_formula(ByteView(raw, order(build)), 0, len(raw))
                self.assertEqual((read.text, read.code, end), (text, code, len(raw)))
                self.assertEqual(raw, write._formula_bytes(power.Formula(text, code), order(build)))

    def test_rejects(self):
        with self.assertRaises(FormatError):
            fw.formula_record("x", b"\0\0\0", "switch")
        with self.assertRaises(FormatError):
            fw.formula_record("a\0b", b"", "switch")


class Assembler(unittest.TestCase):
    """W15-A2."""

    def test_encoding(self):
        code = fw.assemble("PUSHID 0 5 -2 0\nPUSHF 0x3F800000\nCALL Min\nPUSHID1 7\nSELECT\nEND", "ps3")
        self.assertEqual(code[:20], op(5) + bytes.fromhex("00000000 00000005 FFFFFFFE 00000000"))
        self.assertEqual(code[20:28], op(6) + bytes.fromhex("3F800000"))
        self.assertEqual(code[28:36], op(1) + bytes(4))
        self.assertEqual(code[36:44], op(4) + bytes.fromhex("00000007"))
        self.assertEqual(code[44:], op(17) + op(0))
        le = fw.assemble("PUSHID 0 5 -2 0\nPUSHF 1.0\nCALL table\nEND", "switch")
        self.assertEqual(le[4:20], bytes.fromhex("00000000 05000000 FEFFFFFF 00000000"))
        self.assertEqual(le[24:28], bytes.fromhex("0000803F"))
        self.assertEqual(le[28:36], op(1) + bytes((11, 0, 0, 0)))

    def test_opcode_table(self):
        names = ["END", "CALL", "PAREN", "COMMA", "PUSHID1", "PUSHID", "PUSHF", "LT", "GT", "EQ",
                 "NE", "ADD", "SUB", "MUL", "DIV", "POW", "NEG", "SELECT"]
        for number, name in enumerate(names):
            if name in ("CALL", "PUSHID", "PUSHID1", "PUSHF"):
                continue
            self.assertEqual(fw.assemble(name.lower(), "switch"), op(number))
        arities = [2, 2, 3, 2, 2, 1, 3, 2, 1, 2, 2, 2]
        self.assertEqual([formula.FUNCTIONS[i][1] for i in range(12)], arities)

    def test_exact_bits(self):
        for build in BUILDS:
            for bits in (0x7FC00001, 0x7F800001, 0x80000000, 0x00000001, 0x408D1EB9):
                code = pushf(build, bits=bits) + op(0)
                text = fw.disassemble(code, build)
                self.assertIn(f"0x{bits:08X}", text)
                self.assertEqual(fw.assemble(text, build), code)

    def test_round_trip_compiled(self):
        texts = ["sLevel * 2.5 + Min(iLevel, 3)", "a ? b : c", "-Strength#Fire / 0.016",
                 'PowerTag.Bash."Script Formula 3" < 4.41', "sLevel 2"]
        for build in BUILDS:
            for text in texts:
                code = fw.compile_formula(text, build, NAMES)
                self.assertEqual(fw.assemble(fw.disassemble(code, build), build), code)

    def test_round_trip_any_bytes(self):
        rng = random.Random(15)
        for build in BUILDS:
            for _ in range(300):
                words = []
                for _ in range(rng.randrange(0, 12)):
                    kind = rng.random()
                    if kind < 0.6:
                        words.append(op(rng.randrange(0, 20)))
                    elif kind < 0.7:
                        words.append(bytes((1, 0, rng.randrange(256), 0)))
                    else:
                        words.append(rng.randbytes(4))
                code = b"".join(words) + rng.randbytes(rng.choice((0, 0, 1, 3)))
                self.assertEqual(fw.assemble(fw.disassemble(code, build), build), code)

    def test_existing_disassembler_untouched(self):
        code = pushf("switch", 1.0) + op(0)
        self.assertTrue(formula.disassemble(code, "switch").startswith("0000  PUSHF"))

    def test_assembler_errors(self):
        for bad in ("FOO", "PUSHID 1 2", "CALL 300", "PUSHF x", "BYTES zz"):
            with self.assertRaises(FormatError):
                fw.assemble(bad, "switch")


class Tokens(unittest.TestCase):
    """W15-A3."""

    def test_dot_number_by_build(self):
        self.assertEqual(fw._tokenize(".5", "switch"), [("num", ".5")])
        self.assertEqual(fw._tokenize(".5", "x360"), [("num", ".5")])
        self.assertEqual(fw._tokenize(".5", "ps3"), [("name", ".5")])
        self.assertEqual(fw.compile_formula(".5", "switch"), pushf("switch", 0.5) + op(0))
        self.assertEqual(fw.compile_formula(".5", "ps3"), pushid("ps3", *UNKNOWN) + op(0))
        self.assertEqual(fw.compile_formula("1 + .5", "x360"), pushf("x360", 1.5) + op(0))

    def test_numbers(self):
        self.assertEqual(fw._tokenize("12.5.5", "switch"), [("num", "12.5"), ("num", ".5")])
        self.assertEqual(fw._tokenize("12.5.5", "ps3"), [("num", "12.5"), ("name", ".5")])
        self.assertEqual(fw._tokenize("3.", "ps3"), [("num", "3.")])
        self.assertEqual(fw._tokenize("3abc", "ps3"), [("num", "3"), ("name", "abc")])

    def test_names(self):
        self.assertEqual(fw._tokenize('PowerTag.X_1."Cool down" + a#Fire', "ps3"),
                         [("name", 'PowerTag.X_1."Cool down"'), ("op", "+"), ("name", "a#Fire")])
        self.assertEqual(fw._tokenize('a."open ( end', "switch"), [("name", 'a."open ( end')])
        self.assertEqual(fw._tokenize("_x.y#2", "switch"), [("name", "_x.y#2")])

    def test_operators_and_skipping(self):
        self.assertEqual([t for _, t in fw._tokenize("a==b!=c<d>e", "switch")],
                         ["a", "==", "b", "!=", "c", "<", "d", ">", "e"])
        self.assertEqual(fw._tokenize(" \t1 $ @ ! = 2\n", "switch"), [("num", "1"), ("num", "2")])
        self.assertEqual([t for _, t in fw._tokenize("+-*/?:(),", "ps3")], list("+-*/?:(),"))


class Grammar(unittest.TestCase):
    """W15-A4."""

    def compiled(self, text, build="switch"):
        return fw.compile_formula(text, build, NAMES)

    def test_ternary_level(self):
        a = pushid("switch", 1, 0, 0, 0)
        b = pushid("switch", 2, 0, 0, 0)
        c = pushid("switch", 3, 0, 0, 0)
        n = pushid("switch", 23, 0, 0, 0)
        self.assertEqual(self.compiled("SF_0 + sLevel ? iLevel : mLevel"),
                         n + a + b + c + op(17) + op(11) + op(0))
        self.assertEqual(self.compiled("sLevel ? iLevel : mLevel - SF_0"),
                         a + b + c + op(17) + n + op(12) + op(0))
        self.assertEqual(self.compiled("sLevel < iLevel ? mLevel : SF_0"),
                         a + b + c + n + op(17) + op(7) + op(0))
        self.assertEqual(self.compiled("sLevel * iLevel ? mLevel : SF_0"),
                         a + b + op(13) + c + n + op(17) + op(0))

    def test_ternary_left_associative(self):
        code = self.compiled("sLevel ? iLevel : mLevel ? SF_0 : SF_1")
        self.assertEqual(ops(code), [5, 5, 5, 17, 5, 5, 17, 0])
        self.assertEqual(code[-28:-4], pushid("switch", 24, 0, 0, 0) + op(17))
        self.assertEqual(code[60:64], op(17))  # the first SELECT closes a ? b : c

    def test_left_associative_and_precedence(self):
        self.assertEqual(ops(self.compiled("sLevel - iLevel - mLevel")), [5, 5, 12, 5, 12, 0])
        self.assertEqual(ops(self.compiled("sLevel + iLevel * mLevel")), [5, 5, 5, 13, 11, 0])
        self.assertEqual(ops(self.compiled("sLevel / iLevel / mLevel")), [5, 5, 14, 5, 14, 0])
        self.assertEqual(ops(self.compiled("sLevel + iLevel > mLevel == SF_0")),
                         [5, 5, 11, 5, 8, 5, 9, 0])
        self.assertEqual(ops(self.compiled("sLevel != iLevel")), [5, 5, 10, 0])

    def test_unary_and_parentheses(self):
        self.assertEqual(ops(self.compiled("--sLevel")), [5, 16, 16, 0])
        self.assertEqual(self.compiled("+(sLevel)"), self.compiled("sLevel"))
        self.assertEqual(ops(self.compiled("-sLevel * iLevel")), [5, 16, 5, 13, 0])
        self.assertEqual(self.compiled("(sLevel + (iLevel"), self.compiled("sLevel + iLevel"))
        with self.assertRaises(FormatError):
            self.compiled("(sLevel iLevel")

    def test_calls(self):
        code = self.compiled("MIN(sLevel, pin(iLevel, 1, 2)) + floor(2.5)")
        self.assertEqual(ops(code), [5, 5, 6, 6, 1, 1, 6, 1, 11, 0])
        fns = [ins.operands[0] for ins in formula.decode(code, "switch") if ins.opcode == 1]
        self.assertEqual(fns, [2, 0, 5])
        self.assertEqual(ops(self.compiled("RandomFloat(1, 2")), [6, 6, 1, 0])
        with self.assertRaises(FormatError):
            self.compiled("Nope(1)")
        with self.assertRaises(FormatError):
            fw.compile_formula("Min(1)", "switch", NAMES, check_arity=True)

    def test_remaining_tokens(self):
        self.assertEqual(self.compiled("sLevel 2"),
                         pushid("switch", 1, 0, 0, 0) + pushf("switch", 2.0) + op(0))

    def test_constant_folding(self):
        self.assertEqual(self.compiled("1 + 2 * 3"), pushf("switch", 7.0) + op(0))
        self.assertEqual(self.compiled("5 / 0 + 1"), pushf("switch", 1.0) + op(0))
        self.assertEqual(self.compiled("-2"), pushf("switch", -2.0) + op(0))
        # comparison is the loosest level: 1 < (2 ? 3 : 4)
        self.assertEqual(self.compiled("1 < 2 ? 3 : 4"), pushf("switch", 1.0) + op(0))
        self.assertEqual(self.compiled("5 < 2 ? 3 : 4"), pushf("switch", 0.0) + op(0))
        self.assertEqual(self.compiled("1 2"), pushf("switch", 2.0) + op(0))
        # float32 rounding after each step: 16777216 + 1 stays 16777216, then - 16777216 = 0.
        self.assertEqual(self.compiled("16777216 + 1 - 16777216"), pushf("switch", 0.0) + op(0))
        # a name or a call stops folding
        self.assertEqual(ops(self.compiled("1 + 2 + sLevel")), [6, 6, 11, 5, 11, 0])
        self.assertEqual(ops(self.compiled("Floor(1.5)")), [6, 1, 0])

    def test_empty(self):
        self.assertEqual(fw.compile_formula("  ", "ps3"), b"")
        with self.assertRaises(FormatError):
            fw.compile_formula("sLevel ? 1", "ps3")

    def test_byte_order(self):
        be = fw.compile_formula("sLevel + 1.5", "ps3")
        le = fw.compile_formula("sLevel + 1.5", "switch")
        self.assertEqual(be, pushid("ps3", 1, 0, 0, 0) + pushf("ps3", 1.5) + op(11) + op(0))
        self.assertEqual(le, pushid("switch", 1, 0, 0, 0) + pushf("switch", 1.5) + op(11) + op(0))
        self.assertNotEqual(be, le)


class Literals(unittest.TestCase):
    """W15-A5."""

    def bits(self, value):
        return struct.unpack("<I", struct.pack("<f", value))[0]

    def test_rule(self):
        self.assertEqual(fw.formula_literal("2.5"), 2.5)
        self.assertEqual(fw.formula_literal("10"), 10.0)
        self.assertEqual(fw.formula_literal("3."), 3.0)
        self.assertEqual(fw.formula_literal(".25"), 0.25)
        # differs by one unit in the last place from correct rounding
        self.assertEqual(fw.literal_bits("4.41"), 0x408D1EB9)
        self.assertEqual(self.bits(float("4.41")), 0x408D1EB8)
        self.assertEqual(fw.literal_bits("0.016"), 0x3C83126E)
        self.assertEqual(self.bits(float("0.016")), 0x3C83126F)
        self.assertEqual(fw.literal_bits("0.1"), self.bits(0.1))
        self.assertEqual(fw.formula_literal("9" * 50), float("inf"))
        with self.assertRaises(FormatError):
            fw.formula_literal("1e5")

    def test_compiled_literal(self):
        self.assertEqual(fw.compile_formula("4.41", "ps3"), pushf("ps3", bits=0x408D1EB9) + op(0))

    def test_token_search(self):
        self.assertEqual(fw.literal_token(self.bits(2.5)), "2.5")
        self.assertEqual(fw.literal_token(0x408D1EB9), "4.41")
        token = fw.literal_token(0x408D1EB8)
        self.assertIsNotNone(token)
        self.assertNotEqual(token, "4.41")
        self.assertEqual(fw.literal_bits(token), 0x408D1EB8)
        self.assertEqual(fw.literal_token(0), "0")
        for bits in (0x80000000, 0xBF800000, 0x7F800000, 0x7FC00000, 0x00000001):
            self.assertIsNone(fw.literal_token(bits))


class Names(unittest.TestCase):
    """W15-A6."""

    def test_rules(self):
        for build, first in (("ps3", 63), ("switch", 87), ("x360", 87)):
            r = lambda name: NAMES.resolve(name, build)
            self.assertEqual(r('PowerTag.Bash."Cooldown Time"'), (22, 0x1234, 0x30513, 0))
            self.assertEqual(r('powertag.bash."script formula 12"'), (22, 0x1234, 0x41220, 0))
            self.assertEqual(r('PowerTag.OldPower."Script Formula 0'), (22, 0x999, 0x41100, 0))
            self.assertEqual(r("SLEVEL"), (1, 0, 0, 0))
            self.assertEqual(r("iLevel"), (2, 0, 0, 0))
            self.assertEqual(r("mLevel"), (3, 0, 0, 0))
            self.assertEqual(r("SF_0"), (23, 0, 0, 0))
            self.assertEqual(r("sf_12"), (35, 0, 0, 0))
            self.assertEqual(r("mDamageMin#Physical"), (6, 0, 0, 0))
            self.assertEqual(r("mDamageDelta#siege"), (21, 0, 0, 0))
            self.assertEqual(r("mDamageMin#Fire"), (8, 0, 0, 0))
            self.assertEqual(r("DmgTier1"), (first, 0, 0, 0))
            self.assertEqual(r("Healing"), (first + 7, 0, 0, 0))
            self.assertEqual(r("mHealthMin"), (first + 14, 0, 0, 0))
            self.assertEqual(r("HealthMin"), (first + 14, 0, 0, 0))
            self.assertEqual(r("mAreaDamageMin"), (first + 24, 0, 0, 0))
            self.assertEqual(r("mAreaDamageDelta"), (first + 25, 0, 0, 0))
            self.assertEqual(r("bTraitActive"), (first + 26, 0, 0, 0))
            self.assertEqual(r("strength"), (0, 5, 0xFFFFFFFE, 0))
            self.assertEqual(r("Strength#Cold"), (0, 5, 3, 0))
            self.assertEqual(r("Damage_Bonus#Cleave"), (0, 9, 0x2345, 0))
            self.assertEqual(r("Damage_Bonus#OldPower"), (0, 9, 0x999, 0))
            self.assertEqual(r("Damage_Bonus#Whatever"), (0, 9, 0xFFFFFFFF, 0))
            self.assertEqual(r("Mystery"), UNKNOWN)

    def test_first_match_wins(self):
        # an attribute named like a level name still gives the level (rule 2 before rule 6)
        self.assertEqual(NAMES.resolve("sLevel", "switch"), (1, 0, 0, 0))
        # unknown power: falls through to rule 7, or raises when asked
        self.assertEqual(NAMES.resolve('PowerTag.Gone."Cooldown Time"', "switch"), UNKNOWN)
        strict = fw.FormulaNames(powers={"Bash": 1}, unknown_power_tag="error")
        with self.assertRaises(FormatError):
            strict.resolve('PowerTag.Gone."Script Formula 1"', "switch")

    def test_function_and_tables(self):
        self.assertEqual(fw.resolve("Strength#Fire", "ps3", {"Strength": 5}), (0, 5, 1, 0))
        self.assertEqual(
            fw.resolve('PowerTag.P."Script Formula 63"', "switch", powers={"P": 3}),
            (22, 3, 0x41100 + 6 * 0x100 + 3 * 0x10, 0))
        self.assertEqual(fw.resolve('PowerTag.Q."Script Formula 1"', "switch", powers={},
                                    extra={"Q": 8}), (22, 8, 0x41110, 0))
        self.assertEqual(fw.library_power_tags()["Script Formula 0"], 0x41100)

    def test_core_toc_powers(self):
        from d3sno.index import CoreToc, TocEntry
        toc = CoreToc("switch", [TocEntry(29, 0x77, "Zap"), TocEntry(1, 0x88, "NotAPower")])
        self.assertEqual(fw.power_handles(toc), {"Zap": 0x77})
        names = fw.FormulaNames(powers=toc)
        self.assertEqual(names.resolve('PowerTag.Zap."Script Formula 2"', "switch"),
                         (22, 0x77, 0x41120, 0))
        self.assertEqual(names.resolve('PowerTag.NotAPower."Script Formula 2"', "switch"), UNKNOWN)

    def test_compiled_pushid(self):
        code = fw.compile_formula("Strength#Fire", "ps3", NAMES)
        self.assertEqual(code, pushid("ps3", 0, 5, 1, 0) + op(0))
        code = fw.compile_formula("x", "switch", lambda name, build: (40, 1, 2, 3))
        self.assertEqual(code, pushid("switch", 40, 1, 2, 3) + op(0))


class RoundTrips(unittest.TestCase):
    """W15-A7."""

    TEXTS = [
        "sLevel * 2.5 + Min(iLevel, 3)", "SF_1 + sLevel ? iLevel : mLevel",
        "sLevel ? iLevel : mLevel - SF_2", "sLevel ? 1 : 2 ? 3 : 4",
        'PowerTag.Bash."Script Formula 12" / 4.41 - 0.016',
        'PowerTag.OldPower."Cooldown Time" == Damage_Bonus#Cleave',
        "-(Strength#Fire) * mDamageDelta#Arcane", "Table(sLevel, 2) < Diminishing(1, 2, iLevel)",
        "mHealthMin + bTraitActive + Healing + Unknown_Thing", "sLevel 4", "(Strength) 2 != 1",
        "4.41", "-4.41", "0", "1 / 3", "Zero_Attr#Bash", "Strength#Nope",
    ]

    def test_text_code_text(self):
        for build in BUILDS:
            for text in self.TEXTS:
                code = fw.compile_formula(text, build, NAMES)
                back = fw.decompile(code, build, NAMES)
                self.assertEqual(fw.compile_formula(back, build, NAMES), code, (build, text, back))

    def test_fully_parenthesised(self):
        code = fw.compile_formula("SF_1 + sLevel ? iLevel : mLevel * 2", "switch", NAMES)
        self.assertEqual(fw.decompile(code, "switch", NAMES), "(SF_1 + (sLevel ? iLevel : (mLevel * 2)))")
        code = fw.compile_formula("sLevel ? 1 : 2 ? 3 : 4", "ps3", NAMES)
        self.assertEqual(fw.decompile(code, "ps3", NAMES), "((sLevel ? 1 : 2) ? 3 : 4)")

    def test_exact_bits_escape(self):
        for build in BUILDS:
            # a constant no decimal token reaches, next to a name (no folding)
            code = pushid(build, 1, 0, 0, 0) + pushf(build, bits=0x408D1EB8) + op(13) + op(0)
            text = fw.decompile(code, build, NAMES)
            self.assertIn(fw.literal_token(0x408D1EB8), text)  # a longer decimal reaches it
            self.assertEqual(fw.compile_formula(text, build, NAMES), code)
            for bits in (0x00000001, 0xBF800000, 0x7FC00000):
                code = pushid(build, 1, 0, 0, 0) + pushf(build, bits=bits) + op(11) + op(0)
                text = fw.decompile(code, build, NAMES)
                self.assertIn(f"__f32_{bits:08X}", text)
                self.assertEqual(fw.compile_formula(text, build, NAMES), code)
            lone = pushf(build, bits=0x7FC00123) + op(0)
            self.assertEqual(fw.compile_formula(fw.decompile(lone, build), build), lone)
            negative = pushf(build, -2.5) + op(0)
            self.assertEqual(fw.decompile(negative, build), "(-2.5)")
            self.assertEqual(fw.compile_formula("(-2.5)", build), negative)

    def test_operand_escape(self):
        for build in BUILDS:
            code = pushid(build, 0, 1234, 0xFFFFFFFE, 0) + pushid(build, 99, 1, 2, 3) + op(9) + op(0)
            text = fw.decompile(code, build)  # no tables: escapes
            self.assertIn("__id_0_4D2_FFFFFFFE_0", text)
            self.assertEqual(fw.compile_formula(text, build), code)
            plain = fw.compile_formula(text, build, escapes=False, names=lambda n, b: (7, 0, 0, 0))
            self.assertEqual(plain, pushid(build, 7, 0, 0, 0) * 2 + op(9) + op(0))

    def test_gam_like_code(self):
        # .gam code: no text; code -> text -> code
        rng = random.Random(7)
        for build in BUILDS:
            for _ in range(200):
                code = self.random_code(rng, build)
                text = fw.decompile(code, build, NAMES)
                self.assertEqual(fw.compile_formula(text, build, NAMES), code, (build, text))

    def random_code(self, rng, build):
        def leaf():
            if rng.random() < 0.5:
                return pushid(build, *rng.choice([(1, 0, 0, 0), (0, 5, 3, 0), (22, 0x2345, 0x41130, 0),
                                                   (0, 77, 0xFFFFFFFE, 0), (50, 1, 2, 3)]))
            return pushf(build, bits=rng.choice([0x3F800000, 0x408D1EB8, 0x3DCCCCCD, rng.getrandbits(32)]))

        def tree(depth):
            if depth == 0 or rng.random() < 0.3:
                return leaf()
            kind = rng.randrange(4)
            if kind == 0:
                return tree(depth - 1) + op(16)
            if kind == 1:
                return tree(depth - 1) + tree(depth - 1) + tree(depth - 1) + op(17)
            if kind == 2:
                fn = rng.randrange(12)
                return b"".join(tree(depth - 1) for _ in range(formula.FUNCTIONS[fn][1])) + call(fn)
            return tree(depth - 1) + tree(depth - 1) + op(rng.choice([7, 8, 9, 10, 11, 12, 13, 14]))

        # a name first keeps the code unfolded, as compiled code with names is
        return pushid(build, 2, 0, 0, 0) + tree(3) + op(11) + op(0)

    def test_two_values(self):
        for build in BUILDS:
            code = pushid(build, 1, 0, 0, 0) + pushid(build, 2, 0, 0, 0) + op(0)
            text = fw.decompile(code, build, NAMES)
            self.assertEqual(text, "(sLevel) (iLevel)")
            self.assertEqual(fw.compile_formula(text, build, NAMES), code)

    def test_decompile_rejects(self):
        for code in (op(2) + op(0), op(4) + bytes(4) + op(0), op(11) + op(0), op(0) + op(0),
                     pushf("switch", 1.0), call(20) + op(0), op(15) + op(0)):
            with self.assertRaises(FormatError):
                fw.decompile(code, "switch")
        self.assertEqual(fw.decompile(b"", "switch"), "")

    def test_store_with_set_formula(self):
        """A writer stores text and code together (set_formula), and the record frame is the
        same bytes the power writer makes."""
        for build in BUILDS:
            text = "sLevel * 2"
            code = fw.compile_formula(text, build)
            record = power.TagRecord(4, 0x41100, power.Formula(text, code), 0)
            raw = write._record_bytes(record, order(build))
            self.assertEqual(raw[8:], fw.formula_record(text, code, build))
            self.assertEqual(formula.evaluate(code, build, formula.ValueContext({1: 3})), 6.0)


if __name__ == "__main__":
    unittest.main()
