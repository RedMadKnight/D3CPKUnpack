import os
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
