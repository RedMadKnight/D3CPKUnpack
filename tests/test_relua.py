import os
import re
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cpklib
import cpkrebuild  # noqa: F401  (patches CPKWriter)
import lua51
import lua_tool
import relua
from test_lua import K, abc, abx, chunk, function, sbx


def source(code, constants=(), **kwargs):
    parsed = lua51.parse_chunk(chunk(function(">", code, constants, **kwargs)))
    return relua.render(relua.decompile(parsed))


class ReluaTests(unittest.TestCase):
    def test_value_left_on_the_stack_is_a_local(self):
        text = source([
            abx("LOADK", 0, 0),
            abx("GETGLOBAL", 1, 1),
            abc("MOVE", 2, 0),
            abc("CALL", 1, 2, 1),
            abc("RETURN", 0, 1),
        ], ("value", "print"))
        self.assertIn('local v', text)
        self.assertIn('"value"', text)
        self.assertRegex(text, r"print\(v\w*\)")

    def test_assignment_before_a_test_is_kept(self):
        # local x = g.k; if x == nil then x = g.m end; return x
        text = source([
            abx("GETGLOBAL", 0, 0),
            abc("GETTABLE", 1, 0, K + 1),
            abc("EQ", 0, 1, K + 3),
            sbx("JMP", 0, 1),
            abc("GETTABLE", 1, 0, K + 2),
            abc("RETURN", 1, 2),
            abc("RETURN", 0, 1),
        ], ("g", "k", "m", None))
        self.assertIn(".k", text)
        self.assertIn(".m", text)
        self.assertRegex(text, r"local v\w* = v\w*\.k")

    def test_value_read_twice_is_a_local(self):
        # local x = f(); x = x * x
        text = source([
            abx("GETGLOBAL", 0, 0),
            abc("CALL", 0, 1, 2),
            abc("MUL", 0, 0, 0),
            abc("RETURN", 0, 1),
        ], ("f",))
        self.assertRegex(text, r"local (v\w*) = f\(\)\n\1 = \1 \* \1")

    def test_no_goto_in_a_loop(self):
        text = source([
            abc("LOADBOOL", 0, 1, 0),
            abc("TEST", 0, 0, 0),
            sbx("JMP", 0, 1),
            sbx("JMP", 0, -3),
            abc("RETURN", 0, 1),
        ])
        self.assertNotIn("goto", text)

    def test_non_ascii_string_stays_ascii(self):
        text = source([abx("LOADK", 0, 0), abc("RETURN", 0, 2), abc("RETURN", 0, 1)], ("caf\xe9",))
        self.assertTrue(text.isascii(), text)


def shape(text):
    """Locals and parameters renamed a, b, c... in order of appearance."""
    names = {}
    return re.sub(r"\bv\d+\b", lambda m: names.setdefault(m.group(), "abcdefgh"[len(names)]), text)


class RoundTripTests(unittest.TestCase):
    """Each chunk below is what stock luac 5.1 (stripped) emits for the expected
    source, so getting that source back means it recompiles to the same
    instructions and registers."""

    def test_multi_name_local_with_operand_kept(self):
        # local a, b = f(), g(); a = a * b; print(a)
        # A bare `local b` after `local a = f() * g()` would add LOADNIL 1 1.
        code = [
            abx("GETGLOBAL", 0, 0),
            abc("CALL", 0, 1, 2),
            abx("GETGLOBAL", 1, 1),
            abc("CALL", 1, 1, 2),
            abc("MUL", 0, 0, 1),
            abx("GETGLOBAL", 2, 2),
            abc("MOVE", 3, 0),
            abc("CALL", 2, 2, 1),
            abc("RETURN", 0, 1),
        ]
        self.assertEqual(shape(source(code, ("f", "g", "print"))),
                         "local a, b = f(), g()\na = a * b\nprint(a)\n")

    def test_multi_name_local_with_operand_first(self):
        # local a, b = f(), g(); a = b * a; print(a)
        code = [
            abx("GETGLOBAL", 0, 0),
            abc("CALL", 0, 1, 2),
            abx("GETGLOBAL", 1, 1),
            abc("CALL", 1, 1, 2),
            abc("MUL", 0, 1, 0),
            abx("GETGLOBAL", 2, 2),
            abc("MOVE", 3, 0),
            abc("CALL", 2, 2, 1),
            abc("RETURN", 0, 1),
        ]
        self.assertEqual(shape(source(code, ("f", "g", "print"))),
                         "local a, b = f(), g()\na = b * a\nprint(a)\n")

    def test_valueless_local_at_function_start(self):
        # function(p) local x; if p == 1 then x = 2 end; return x end
        # No LOADNIL: Lua 5.1 drops it at pc 0.
        code = [
            abc("EQ", 0, 0, K + 0),
            sbx("JMP", 0, 1),
            abx("LOADK", 1, 1),
            abc("RETURN", 1, 2),
            abc("RETURN", 0, 1),
        ]
        self.assertEqual(shape(source(code, (1.0, 2.0), params=1)),
                         "local a\nif b == 1 then\n    a = 2\nend\nreturn a\n")

    def test_valueless_local_at_function_start_test(self):
        # function(p) local x; if p then x = 2 end; print(x) end
        code = [
            abc("TEST", 0, 0, 0),
            sbx("JMP", 0, 1),
            abx("LOADK", 1, 0),
            abx("GETGLOBAL", 2, 1),
            abc("MOVE", 3, 1),
            abc("CALL", 2, 2, 1),
            abc("RETURN", 0, 1),
        ]
        self.assertEqual(shape(source(code, (2.0, "print"), params=1)),
                         "local a\nif b then\n    a = 2\nend\nprint(a)\n")

    def test_valueless_local_with_loadnil_stays_in_place(self):
        # function(p) print(p); local x; if p then x = 2 end; return x end
        code = [
            abx("GETGLOBAL", 1, 1),
            abc("MOVE", 2, 0),
            abc("CALL", 1, 2, 1),
            abc("LOADNIL", 1, 1),
            abc("TEST", 0, 0, 0),
            sbx("JMP", 0, 1),
            abx("LOADK", 1, 0),
            abc("RETURN", 1, 2),
            abc("RETURN", 0, 1),
        ]
        self.assertEqual(shape(source(code, (2.0, "print"), params=1)),
                         "print(a)\nlocal b\nif a then\n    b = 2\nend\nreturn b\n")


    # Pattern A must not fire for the last argument of a call followed by a
    # for loop that reuses the call's registers.

    def test_arith_last_argument_before_for_loop(self):
        # function(o, a, b, c, n) o.F(a, -1 * b * math.floor(c / 2))
        #     for i = 1, n do print(i) end end
        code = [
            abc("GETTABLE", 5, 0, K + 0),
            abc("MOVE", 6, 1),
            abc("MUL", 7, K + 1, 2),
            abx("GETGLOBAL", 8, 2),
            abc("GETTABLE", 8, 8, K + 3),
            abc("DIV", 9, 3, K + 4),
            abc("CALL", 8, 2, 2),
            abc("MUL", 7, 7, 8),
            abc("CALL", 5, 3, 1),
            abx("LOADK", 5, 5),
            abc("MOVE", 6, 4),
            abx("LOADK", 7, 5),
            sbx("FORPREP", 5, 3),
            abx("GETGLOBAL", 9, 6),
            abc("MOVE", 10, 8),
            abc("CALL", 9, 2, 1),
            sbx("FORLOOP", 5, -4),
            abc("RETURN", 0, 1),
        ]
        text = source(code, ("F", -1.0, "math", "floor", 2.0, 1.0, "print"), params=5)
        self.assertEqual(shape(text),
                         "a.F(b, -1 * c * math.floor(d / 2))\n"
                         "for e = 1, f do\n    print(e)\nend\n")

    def test_arith_difference_argument_before_for_loop(self):
        # function(a, x, y, n) G(a, x / 2 * y - x / 2 * y)
        #     for i = 1, n do print(i) end end
        code = [
            abx("GETGLOBAL", 4, 0),
            abc("MOVE", 5, 0),
            abc("DIV", 6, 1, K + 1),
            abc("MUL", 6, 6, 2),
            abc("DIV", 7, 1, K + 1),
            abc("MUL", 7, 7, 2),
            abc("SUB", 6, 6, 7),
            abc("CALL", 4, 3, 1),
            abx("LOADK", 4, 2),
            abc("MOVE", 5, 3),
            abx("LOADK", 6, 2),
            sbx("FORPREP", 4, 3),
            abx("GETGLOBAL", 8, 3),
            abc("MOVE", 9, 7),
            abc("CALL", 8, 2, 1),
            sbx("FORLOOP", 4, -4),
            abc("RETURN", 0, 1),
        ]
        text = source(code, ("G", 2.0, 1.0, "print"), params=4)
        self.assertEqual(shape(text),
                         "G(a, b / 2 * c - b / 2 * c)\n"
                         "for d = 1, e do\n    print(d)\nend\n")

    def test_open_call_keeps_every_argument(self):
        # A CALL with B = 0 whose argument registers already hold locals (as in
        # a degraded decode): every register up to the multiple results is an
        # argument, so "Name" (in R2) must not be dropped.
        code = [
            abc("GETTABLE", 1, 0, K + 0),
            abx("LOADK", 2, 1),
            abc("GETTABLE", 3, 0, K + 2),
            abc("MOVE", 4, 0),
            abc("CALL", 3, 2, 0),
            abc("CALL", 1, 0, 1),
            abc("RETURN", 0, 1),
        ]
        text = shape(source(code, ("SetStoredInt", "Name", "CreateCursor"), params=3))
        self.assertEqual(text, 'a = b.SetStoredInt\nc = "Name"\na(c, b.CreateCursor(b))\n')


class ExactOptionTests(unittest.TestCase):
    def test_extract_writes_exact_source(self):
        script = chunk(function(">", [
            abx("LOADK", 0, 0),
            abx("GETGLOBAL", 1, 1),
            abc("MOVE", 2, 0),
            abc("CALL", 1, 2, 1),
            abc("RETURN", 0, 1),
        ], ("value", "print")))
        body = bytes(0x20) + script
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "t.cpk")
            w = cpklib.CPKWriter.create_new()
            w.add_file("Script\\T.scr", body)
            w.save(path)
            out = os.path.join(d, "out")
            subprocess.run(
                [sys.executable, os.path.join(ROOT, "lua_tool.py"), "extract", path,
                 "--out", out, "--exact"], check=True, capture_output=True)
            with open(os.path.join(out, "Script", "T.scr.lua"), encoding="utf-8") as f:
                self.assertIn("local v", f.read())

    def test_unknown_pattern_returns_none(self):
        class Broken:
            main = None
        original = relua.decompile
        def fail(_):
            raise relua.Fail("no pattern")
        relua.decompile = fail
        try:
            self.assertIsNone(lua_tool.exact_source(Broken()))
        finally:
            relua.decompile = original


if __name__ == "__main__":
    unittest.main()
