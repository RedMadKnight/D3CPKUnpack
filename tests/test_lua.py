import os
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cpklib
import cpkrebuild  # noqa: F401  (patches CPKWriter)
import lua51
import luadec

K = 256  # RK operands at or above this index the constant table


def abc(op, a=0, b=0, c=0):
    return lua51.OPCODES.index(op) | a << 6 | c << 14 | b << 23


def abx(op, a, bx):
    return lua51.OPCODES.index(op) | a << 6 | bx << 14


def sbx(op, a, offset):
    return abx(op, a, offset + 131071)


def function(endian, code, constants=(), protos=(), params=0, upvalues=0, vararg=0):
    u32 = lambda v: struct.pack(endian + "I", v)

    def string(text):
        data = text.encode("latin-1") + b"\x00"
        return u32(len(data)) + data

    out = u32(0) + u32(0) + u32(0)
    out += bytes((upvalues, params, vararg, 20))
    out += u32(len(code)) + b"".join(u32(word) for word in code)
    out += u32(len(constants))
    for value in constants:
        if value is None:
            out += b"\x00"
        elif isinstance(value, bool):
            out += b"\x01" + bytes((value,))
        elif isinstance(value, str):
            out += b"\x04" + string(value)
        else:
            out += b"\x03" + struct.pack(endian + "d", value)
    out += u32(len(protos)) + b"".join(protos)
    return out + u32(0) + u32(0) + u32(0)


def chunk(body, endian=">"):
    little = 1 if endian == "<" else 0
    return b"\x1bLuaQ\x00" + bytes((little, 4, 4, 4, 8, 0)) + body


def call_stmt(reg, name_k, value_k, hydra_k=0):
    """Hydra.<name>(<value>) with no results, using registers reg, reg + 1."""
    return [
        abx("GETGLOBAL", reg, hydra_k),
        abc("GETTABLE", reg, reg, K + name_k),
        abx("LOADK", reg + 1, value_k),
        abc("CALL", reg, 2, 1),
    ]


def decompile(*args, **kwargs):
    return luadec.decompile(lua51.parse_chunk(chunk(function(">", *args, **kwargs))))


class ChunkParserTests(unittest.TestCase):
    SIMPLE = [abx("LOADK", 0, 0), abc("RETURN", 0, 1, 0)]

    def test_big_and_little_endian(self):
        for endian in "<>":
            blob = chunk(function(endian, self.SIMPLE, ("text", 2.5, True, None)), endian)
            parsed = lua51.parse_chunk(blob)
            self.assertEqual(parsed.endian, endian)
            self.assertEqual(parsed.size, len(blob))
            self.assertEqual(parsed.main.constants, ["text", 2.5, True, None])
            self.assertEqual([i.op for i in parsed.main.code], ["LOADK", "RETURN"])
            self.assertTrue(parsed.stripped)

    def test_nested_functions_are_walked(self):
        inner = function(">", self.SIMPLE, ("x",))
        outer = function(">", self.SIMPLE, ("y",), protos=[inner, inner])
        self.assertEqual(len(list(lua51.parse_chunk(chunk(outer)).main.walk())), 3)

    def test_find_chunks_skips_false_signatures(self):
        blob = chunk(function(">", self.SIMPLE, ("x",)))
        data = b"\x1bLua" + bytes(30) + blob + b"tail"
        found = list(lua51.find_chunks(data))
        self.assertEqual([offset for offset, _ in found], [34])

    def test_bad_chunks_are_rejected(self):
        blob = chunk(function(">", self.SIMPLE, ("x",)))
        for bad in (blob[:-5], b"nope" + blob, blob[:4] + b"\x52" + blob[5:]):
            with self.assertRaises(ValueError):
                lua51.parse_chunk(bad)

    def test_disassembly_resolves_constants(self):
        listing = lua51.disassemble(function_of(self.SIMPLE, ("hello",)))
        self.assertTrue(any("LOADK" in text and '"hello"' in text for text in listing))


def function_of(code, constants):
    return lua51.parse_chunk(chunk(function(">", code, constants))).main


class DecompilerTests(unittest.TestCase):
    RET = abc("RETURN", 0, 1, 0)

    def test_call_folds_into_one_statement(self):
        text = decompile(call_stmt(0, 1, 2) + [self.RET], ("Hydra", "Wait", 1.0))
        self.assertEqual(text, "Hydra.Wait(1)\n")

    def test_closure_set_as_global_is_a_named_function(self):
        inner = function(">", [self.RET], params=2)
        text = decompile(
            [abx("CLOSURE", 0, 0), abx("SETGLOBAL", 0, 0), self.RET],
            ("OnStart",), protos=[inner])
        self.assertEqual(text, "function OnStart(a0, a1)\nend\n")

    def test_if_else(self):
        then_part = call_stmt(0, 2, 4)
        else_part = call_stmt(0, 3, 4)
        code = [abx("GETGLOBAL", 0, 1), abc("TEST", 0, 0, 0),
                sbx("JMP", 0, len(then_part) + 1)]
        code += then_part + [sbx("JMP", 0, len(else_part))] + else_part + [self.RET]
        text = decompile(code, ("Hydra", "flag", "First", "Second", 1.0))
        self.assertEqual(text, "if flag then\n    Hydra.First(1)\n"
                               "else\n    Hydra.Second(1)\nend\n")

    def test_and_condition(self):
        code = [
            abc("LT", 0, K + 0, 0), sbx("JMP", 0, 6),
            abc("LE", 0, 0, K + 1), sbx("JMP", 0, 4),
        ] + call_stmt(1, 3, 1, hydra_k=2) + [self.RET]
        text = luadec.decompile(lua51.parse_chunk(chunk(function(
            ">", code, (0.5, 0.75, "Hydra", "Go"), params=1))))
        self.assertIn("if (0.5 < a0) and (a0 <= 0.75) then", text)

    def test_numeric_for(self):
        body = call_stmt(4, 1, 3, hydra_k=0)
        code = [abx("LOADK", 0, 3), abx("LOADK", 1, 2), abx("LOADK", 2, 3),
                sbx("FORPREP", 0, len(body))] + body + \
            [sbx("FORLOOP", 0, -len(body) - 1), self.RET]
        text = decompile(code, ("Hydra", "Tick", 5.0, 1.0))
        self.assertIn("for r3 = 1, 5 do\n    Hydra.Tick(1)\nend", text)

    def test_if_else_inside_generic_for(self):
        # The JMP that ends the then-branch lands on TFORLOOP. It must not
        # read as the start of another loop, which duplicated the else-branch.
        then_part = call_stmt(6, 2, 4)
        else_part = call_stmt(6, 3, 4)
        body = [abx("GETGLOBAL", 6, 5), abc("TEST", 6, 0, 0),
                sbx("JMP", 0, len(then_part) + 1)]
        body += then_part + [sbx("JMP", 0, len(else_part))] + else_part
        code = [abx("GETGLOBAL", 0, 1), abx("GETGLOBAL", 1, 6),
                abc("CALL", 0, 2, 4), sbx("JMP", 0, len(body))] + body + \
            [abc("TFORLOOP", 0, 0, 2), sbx("JMP", 0, -len(body) - 2), self.RET]
        text = decompile(
            code, ("Hydra", "ipairs", "First", "Second", 1.0, "flag", "list"))
        self.assertEqual(text.count("Hydra.Second(1)"), 1)
        self.assertEqual(text.count("for "), 1)
        self.assertIn("for r3, r4 in ipairs(list) do", text)
        self.assertIn("else", text)

    def test_upvalues_do_not_collide_with_locals(self):
        inner = function(">", [
            abc("GETUPVAL", 0, 0), abx("LOADK", 1, 0),
            abc("CALL", 0, 2, 2), abx("SETGLOBAL", 0, 1), self.RET,
        ], (1.0, "result"), upvalues=1)
        text = decompile([
            abx("GETGLOBAL", 0, 0), abx("CLOSURE", 1, 0), abc("MOVE", 0, 0, 0),
            abx("SETGLOBAL", 1, 1), self.RET,
        ], ("helper", "Run"), protos=[inner])
        self.assertIn("up0_r0 = helper", text)
        self.assertIn("result = up0_r0(1)", text)

    def test_named_local_is_not_folded_into_a_call(self):
        text = decompile([
            abx("GETGLOBAL", 0, 0), abc("CALL", 0, 1, 2),
            abx("GETGLOBAL", 1, 1), abc("MOVE", 2, 0), abc("CALL", 1, 2, 1),
            self.RET,
        ], ("first", "second"))
        self.assertEqual(text, "local r0\nr0 = first()\nsecond(r0)\n")

    def test_folding_keeps_calls_in_order(self):
        # x = first(); y = second(); total = y * x
        text = decompile([
            abx("GETGLOBAL", 0, 0), abc("CALL", 0, 1, 2),
            abx("GETGLOBAL", 1, 1), abc("CALL", 1, 1, 2),
            abc("MUL", 2, 1, 0), abx("SETGLOBAL", 2, 2), self.RET,
        ], ("first", "second", "total"))
        self.assertLess(text.index("first()"), text.index("second()"))

    def test_return_of_one_call_result_stays_truncated(self):
        text = decompile([
            abx("GETGLOBAL", 0, 0), abc("CALL", 0, 1, 2),
            abc("RETURN", 0, 2, 0), self.RET,
        ], ("first",))
        self.assertIn("return (first())", text)

    def test_unstructured_jump_prints_goto(self):
        code = [abc("TEST", 0, 0, 0), sbx("JMP", 0, -2), self.RET]
        text = luadec.decompile(lua51.parse_chunk(chunk(function(
            ">", code, params=1))))
        self.assertIn("goto L0", text)
        self.assertIn("::L0::", text)

    def test_unsupported_code_falls_back_to_disassembly(self):
        text = decompile([abc("TESTSET", 0, 1, 0), self.RET])
        self.assertIn("-- could not decompile", text)
        self.assertIn("TESTSET", text)


class LuaToolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def run_tool(self, *args):
        r = subprocess.run([sys.executable, os.path.join(ROOT, "lua_tool.py"), *args],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        return r.stdout

    def test_extract_writes_scripts_and_skips_empty_ones(self):
        code = call_stmt(0, 1, 2) + [abc("RETURN", 0, 1, 0)]
        script = chunk(function(">", code, ("Hydra", "Wait", 1.0)))
        empty = chunk(function(">", [abc("RETURN", 0, 1, 0)]))
        archive = os.path.join(self.tmp.name, "a.cpk")
        w = cpklib.CPKWriter.create_new()
        w.add_file("Power\\Fire.pow", bytes(64) + script)
        w.add_file("Conversation\\Hi.cnv", bytes(16) + empty)
        w.add_file("Textures\\T.tex", script)
        w.save(archive)

        self.assertIn(".pow       1 scripts", self.run_tool("info", archive))
        out = os.path.join(self.tmp.name, "out")
        self.assertIn("written: 1 scripts", self.run_tool(
            "extract", archive, "--out", out, "--asm", "--raw"))
        stem = os.path.join(out, "Power", "Fire.pow")
        with open(stem + ".lua", encoding="utf-8") as f:
            self.assertEqual(f.read(), "Hydra.Wait(1)\n")
        with open(stem + ".luac", "rb") as f:
            self.assertEqual(f.read(), script)
        self.assertTrue(os.path.exists(stem + ".lst"))
        self.assertFalse(os.path.exists(os.path.join(out, "Textures")))


if __name__ == "__main__":
    unittest.main()
