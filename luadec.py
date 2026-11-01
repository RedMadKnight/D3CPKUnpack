#!/usr/bin/env python3
"""
Decompiler from Lua 5.1 bytecode (lua51.py) to readable pseudo-Lua.

The shipped chunks are stripped, so local names are gone: parameters print as
a0, a1, ..., other locals as r<register>. Global, field and string names
survive. Single-use temporaries fold back into expressions, and if/else,
while and for loops are recovered. A jump that fits none of those prints as
`goto L<pc>`, with a Lua 5.2 style label, so the control flow stays exact.
A function that cannot be decompiled falls back to its disassembly.
"""
from __future__ import annotations

import re

from lua51 import LuaChunk, LuaFunction, disassemble, format_constant

_RK_CONSTANT = 256
_ARITH = {"ADD": "+", "SUB": "-", "MUL": "*", "DIV": "/", "MOD": "%", "POW": "^"}
_COMPARE = {"EQ": "==", "LT": "<", "LE": "<="}
_TESTS = ("EQ", "LT", "LE", "TEST")
_NOT_EXPRESSIONS = ("JMP", "RETURN", "FORPREP", "FORLOOP", "TFORLOOP", "CLOSURE",
                    "TAILCALL", "SETGLOBAL", "SETUPVAL", "SETTABLE")
_ONE_OPERAND = ("TEST", "TESTSET", "SETGLOBAL", "SETUPVAL", "UNM", "NOT", "LEN")
_SIMPLE = re.compile(r"^[\w.]+$")
_CALL = re.compile(r"^[\w.:]+\((?:[^()]|\([^()]*\))*\)$")
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_KEYWORDS = {
    "and", "break", "do", "else", "elseif", "end", "false", "for", "function",
    "if", "in", "local", "nil", "not", "or", "repeat", "return", "then", "true",
    "until", "while",
}


def _strip_parens(text: str) -> str:
    """Drop one pair of parentheses that encloses the whole expression."""
    if not (text.startswith("(") and text.endswith(")")):
        return text
    depth = 0
    quoted = False
    for pos, ch in enumerate(text):
        if ch == '"' and text[pos - 1] != "\\":
            quoted = not quoted
        elif not quoted:
            depth += (ch == "(") - (ch == ")")
            if depth == 0 and pos < len(text) - 1:
                return text
    return text[1:-1]


def _negate(text: str) -> str:
    bare = _strip_parens(text)
    if bare.startswith("not (") and _strip_parens(bare[4:]) != bare[4:]:
        return bare[4:]
    if _SIMPLE.match(bare) or (bare.endswith(")") and _CALL.match(bare)):
        return f"not {bare}"
    return f"not ({bare})"


class _Decompiler:
    def __init__(self, fn: LuaFunction, upvalues: list[str], depth: int = 0):
        self.fn = fn
        self.code = fn.code
        self.upvalues = upvalues
        self.depth = depth
        self.labels: set[int] = set()
        self._analyse()

    # --- register use ---

    def _accesses(self, i, multi):
        """(reads, writes, first register clobbered) of an instruction."""
        rk = lambda v: [v] if v < _RK_CONSTANT else []
        span = lambda first, last: list(range(first, last + 1))
        op = i.op
        if op == "MOVE":
            return [i.b], [i.a], None
        if op in ("LOADK", "GETGLOBAL", "GETUPVAL", "NEWTABLE", "LOADBOOL",
                  "CLOSURE"):
            return [], [i.a], None
        if op == "LOADNIL":
            return [], span(i.a, i.b), None
        if op == "GETTABLE":
            return [i.b] + rk(i.c), [i.a], None
        if op in ("SETGLOBAL", "SETUPVAL", "TEST"):
            return [i.a], [], None
        if op == "SETTABLE":
            return [i.a] + rk(i.b) + rk(i.c), [], None
        if op == "SELF":
            return [i.b] + rk(i.c), [i.a, i.a + 1], None
        if op in _ARITH:
            return rk(i.b) + rk(i.c), [i.a], None
        if op in ("UNM", "NOT", "LEN"):
            return [i.b], [i.a], None
        if op == "CONCAT":
            return span(i.b, i.c), [i.a], None
        if op in _COMPARE:
            return rk(i.b) + rk(i.c), [], None
        if op == "TESTSET":
            return [i.b], [i.a], None
        if op in ("CALL", "TAILCALL"):
            last = i.a + i.b - 1 if i.b else multi
            writes = span(i.a, i.a + i.c - 2) if i.c else [i.a]
            return span(i.a, last), writes, i.a
        if op == "RETURN":
            return span(i.a, i.a + i.b - 2 if i.b else multi), [], None
        if op == "SETLIST":
            return span(i.a, i.a + i.b if i.b else multi), [i.a], None
        if op == "FORPREP":
            return span(i.a, i.a + 2), [], None
        if op == "VARARG":
            return [], span(i.a, i.a + i.b - 2) if i.b else [i.a], None
        return [], [], None

    def _analyse(self):
        code = self.code
        self.leaders: set[int] = set()
        self.loop_ends: dict[int, int] = {}
        self.captured: set[int] = set()
        self.pseudo: set[int] = set()
        self.reads, self.writes, self.kills = [], [], []
        multi = 0
        for i in code:
            reads, writes, kill = self._accesses(i, multi)
            self.reads.append(reads)
            self.writes.append(writes)
            self.kills.append(kill)
            if i.op in ("CALL", "TAILCALL") and i.c == 0 or \
                    i.op == "VARARG" and i.b == 0:
                multi = i.a
            if i.op == "CLOSURE":
                for k in range(self.fn.protos[i.bx].num_upvalues):
                    capture = code[i.pc + 1 + k]
                    self.pseudo.add(capture.pc)
                    if capture.op == "MOVE":
                        self.captured.add(capture.b)
        for i in code:
            if self._is_bool_value(i.pc):
                continue
            if i.op in ("JMP", "FORPREP", "FORLOOP"):
                target = i.pc + 1 + i.sbx
                self.leaders.add(target)
                if i.op == "JMP" and target <= i.pc:
                    previous = code[i.pc - 1].op if i.pc else ""
                    if previous not in _TESTS + ("TFORLOOP",):
                        self.loop_ends[target] = max(
                            i.pc, self.loop_ends.get(target, 0))
            elif i.op == "LOADBOOL" and i.c and \
                    not self._is_bool_value(i.pc - 2):
                self.leaders.add(i.pc + 2)
        for pc in self.pseudo:
            self.writes[pc], self.kills[pc] = [], None

    def _is_bool_value(self, pc: int) -> bool:
        """test; JMP +1; LOADBOOL x 0 1; LOADBOOL x 1 0"""
        code = self.code
        if pc < 0 or pc + 3 >= len(code) or code[pc].op not in _TESTS:
            return False
        jmp, low, high = code[pc + 1], code[pc + 2], code[pc + 3]
        return (jmp.op == "JMP" and jmp.sbx == 1
                and low.op == "LOADBOOL" and (low.b, low.c) == (0, 1)
                and high.op == "LOADBOOL" and (high.b, high.c) == (1, 0)
                and high.a == low.a)

    def _single_use(self, pc: int, reg: int) -> bool:
        """True when `reg`, set at pc, is read once, soon, and then dead.

        Folding must keep calls in order: text reads left to right, so nothing
        called in between may land to the left of the folded value.
        """
        called = False
        for q in range(pc + 1, len(self.code)):
            if q in self.leaders:
                return False
            if reg in self.reads[q]:
                # Only a named local is read through MOVE. Keeping it stops a
                # call from moving past the others in the same statement.
                if self.reads[q].count(reg) > 1 or self.code[q].op == "MOVE":
                    return False
                i = self.code[q]
                if called and (i.op in _ONE_OPERAND or (
                        (i.op in _ARITH or i.op in _COMPARE)
                        and i.c == reg and reg < i.b < _RK_CONSTANT)):
                    return False
                if reg in self.captured and reg not in self.writes[q] \
                        and not self._clobbers(q, reg):
                    return False
                return self._dead_after(q, reg)
            if reg in self.writes[q]:
                return False
            if self.kills[q] is not None:
                if self.kills[q] < reg:
                    return False
                called = True
        return False

    def _dead_after(self, q: int, reg: int) -> bool:
        if reg in self.writes[q] or self._clobbers(q, reg):
            return True
        for s in range(q + 1, len(self.code)):
            if reg in self.reads[s]:
                return False
            if reg in self.writes[s] or self._clobbers(s, reg):
                return True
        return True

    def _clobbers(self, pc: int, reg: int) -> bool:
        return self.kills[pc] is not None and reg >= self.kills[pc]

    # --- output ---

    def local_name(self, reg: int) -> str:
        kind = "a" if reg < self.fn.num_params else "r"
        prefix = f"up{self.depth}_" if reg in self.captured else ""
        return f"{prefix}{kind}{reg}"

    def name(self, reg: int) -> str:
        if reg >= self.fn.num_params and reg not in self.loop_vars:
            self.declared.add(reg)
        return self.local_name(reg)

    def line(self, text: str):
        self.out.append("    " * self.indent + text)

    def flush(self, everything: bool = True):
        for reg in sorted(self.pending):
            expr, constant = self.pending[reg]
            if expr is None or (constant and not everything):
                continue
            del self.pending[reg]
            self.line(f"{self.name(reg)} = {_strip_parens(expr)}")
            self.statements += 1

    def emit(self, text: str):
        self.flush(everything=False)
        self.line(text)
        self.statements += 1

    def set(self, pc: int, reg: int, expr: str, constant: bool = False):
        self.pending.pop(reg, None)
        if self._single_use(pc, reg):
            self.pending[reg] = (expr, constant)
        else:
            self.emit(f"{self.name(reg)} = {_strip_parens(expr)}")

    def read(self, reg: int):
        if reg in self.pending:
            return self.pending.pop(reg)[0]
        return self.name(reg)

    def rk(self, value: int):
        if value >= _RK_CONSTANT:
            return format_constant(self.fn.constants[value - _RK_CONSTANT])
        return self.read(value)

    def index(self, obj: str, key: str) -> str:
        if key.startswith('"') and key.endswith('"'):
            bare = key[1:-1]
            if _IDENT.match(bare) and bare not in _KEYWORDS:
                return f"{obj}.{bare}"
        return f"{obj}[{key}]"

    def arguments(self, first: int, count: int) -> list[str]:
        """Read registers first.. as a list; count -1 means up to the top."""
        last = first + count - 1 if count >= 0 else self.multi
        values = [self.read(r) for r in range(first, last + 1)]
        return [_strip_parens(v) for v in values if v is not None]

    def call(self, i) -> str:
        args = self.arguments(i.a + 1, i.b - 1 if i.b else -1)
        return f"{self.read(i.a)}({', '.join(args)})"

    # --- straight-line instructions ---

    def step(self, i) -> int:
        op, a = i.op, i.a
        constants = self.fn.constants
        if op == "MOVE":
            self.set(i.pc, a, self.read(i.b))
        elif op == "LOADK":
            self.set(i.pc, a, format_constant(constants[i.bx]), constant=True)
        elif op == "LOADBOOL":
            self.set(i.pc, a, "true" if i.b else "false", constant=True)
            if i.c:
                self.flush()
                self.goto(i.pc + 2)
        elif op == "LOADNIL":
            for reg in range(a, i.b + 1):
                self.set(i.pc, reg, "nil", constant=True)
        elif op == "GETUPVAL":
            self.set(i.pc, a, self.upvalues[i.b])
        elif op == "GETGLOBAL":
            self.set(i.pc, a, constants[i.bx])
        elif op == "GETTABLE":
            key = self.rk(i.c)
            self.set(i.pc, a, self.index(self.read(i.b), key))
        elif op == "SETGLOBAL":
            self.emit(f"{constants[i.bx]} = {_strip_parens(self.read(a))}")
        elif op == "SETUPVAL":
            self.emit(f"{self.upvalues[i.b]} = {_strip_parens(self.read(a))}")
        elif op == "SETTABLE":
            value = _strip_parens(self.rk(i.c))
            key = _strip_parens(self.rk(i.b))
            if a in self.pending:
                self.flush_reg(a)
            self.emit(f"{self.index(self.name(a), key)} = {value}")
        elif op == "NEWTABLE":
            self.set(i.pc, a, "{}")
        elif op == "SELF":
            key = self.rk(i.c)
            obj = self.read(i.b)
            self.pending.pop(a, None)
            self.pending[a] = (f"{obj}:{key[1:-1]}", False)
            self.pending[a + 1] = (None, True)
        elif op in _ARITH:
            right = self.rk(i.c)
            self.set(i.pc, a, f"({self.rk(i.b)} {_ARITH[op]} {right})")
        elif op == "UNM":
            self.set(i.pc, a, f"-{self.read(i.b)}")
        elif op == "NOT":
            self.set(i.pc, a, f"not {self.read(i.b)}")
        elif op == "LEN":
            self.set(i.pc, a, f"#{self.read(i.b)}")
        elif op == "CONCAT":
            parts = [self.read(r) for r in range(i.b, i.c + 1)]
            self.set(i.pc, a, f"({' .. '.join(parts)})")
        elif op == "CALL":
            self.finish_call(i, self.call(i))
        elif op == "TAILCALL":
            self.emit(f"return {self.call(i)}")
            return i.pc + 2
        elif op == "RETURN":
            values = self.arguments(a, i.b - 1 if i.b else -1)
            last = self.code[i.pc - 1]
            if i.b == 2 and i.pc and last.op == "CALL" and last.a == a \
                    and last.c == 2:
                values = [f"({values[0]})"]  # keeps it to one result
            if values or i.pc != len(self.code) - 1:
                self.emit(f"return {', '.join(values)}".rstrip())
        elif op == "SETLIST":
            values = self.arguments(a + 1, i.b if i.b else -1)
            if self.pending.get(a, (None,))[0] == "{}" and i.c <= 1:
                self.pending.pop(a)
                self.set(i.pc, a, "{" + ", ".join(values) + "}")
            else:
                self.flush_reg(a)
                for n, value in enumerate(values, (i.c - 1) * 50 + 1):
                    self.emit(f"{self.name(a)}[{n}] = {value}")
        elif op == "CLOSURE":
            return self.closure(i)
        elif op == "VARARG":
            count = i.b - 1 if i.b else 1
            if count == 1:
                self.multi = a
                self.set(i.pc, a, "...")
            else:
                names = ", ".join(self.name(a + n) for n in range(count))
                self.emit(f"{names} = ...")
        elif op == "CLOSE":
            pass
        else:
            raise ValueError(f"unexpected {op} at pc {i.pc}")
        return i.pc + 1

    def flush_reg(self, reg: int):
        if reg in self.pending:
            expr = self.pending.pop(reg)[0]
            self.line(f"{self.name(reg)} = {_strip_parens(expr)}")
            self.statements += 1

    def finish_call(self, i, text: str):
        if i.c == 1:
            self.emit(text)
        elif i.c == 2:
            self.set(i.pc, i.a, text)
        elif i.c == 0:
            self.multi = i.a
            self.pending[i.a] = (text, False)
        else:
            names = ", ".join(self.name(i.a + n) for n in range(i.c - 1))
            self.emit(f"{names} = {text}")

    def closure(self, i) -> int:
        proto = self.fn.protos[i.bx]
        names = []
        for k in range(proto.num_upvalues):
            capture = self.code[i.pc + 1 + k]
            if capture.op == "MOVE":
                names.append(self.name(capture.b))
            else:
                names.append(self.upvalues[capture.b])
        next_pc = i.pc + 1 + proto.num_upvalues
        params, body = _function_text(proto, names, self.indent + 1)
        follower = self.code[next_pc] if next_pc < len(self.code) else None
        if follower is not None and follower.op == "SETGLOBAL" \
                and follower.a == i.a:
            title = self.fn.constants[follower.bx]
            self.emit(f"function {title}({params})")
            self.out.extend(body)
            self.line("end")
            return next_pc + 1
        close = "    " * self.indent + "end"
        text = "\n".join([f"function({params})", *body, close])
        self.set(i.pc, i.a, text)
        return next_pc

    # --- control flow ---

    def goto(self, target: int):
        if target not in self.labels:
            self.labels.add(target)
            self.changed = True
        self.line(f"goto L{target}")
        self.statements += 1

    def conditions(self, i) -> tuple[str, str]:
        """(condition under which the JMP is taken, the opposite)."""
        if i.op == "TEST":
            value = self.read(i.a)
            return (value, _negate(value)) if i.c else (_negate(value), value)
        right = self.rk(i.c)
        left = self.rk(i.b)
        positive = f"({left} {_COMPARE[i.op]} {right})"
        if i.op == "EQ":
            negative = f"({left} ~= {right})"
        else:
            negative = f"not {positive}"
        return (positive, negative) if i.a else (negative, positive)

    def chain_next(self, pc: int, end: int, target: int):
        """Evaluate up to the next test+JMP of the same condition, if any.

        Only expressions may lie in between, and the test must share the jump
        target or jump over it into the body. Otherwise nothing is consumed.
        """
        code = self.code
        saved = dict(self.pending), len(self.out), self.statements, self.multi
        q = pc
        while q is not None and q < end and code[q].op not in _TESTS:
            if code[q].op in _NOT_EXPRESSIONS or q in self.leaders \
                    or q in self.loop_ends:
                q = None
                break
            q = self.step(code[q])
            if self.statements != saved[2]:
                q = None
        if q is not None and q + 1 < end and code[q].op in _TESTS \
                and code[q + 1].op == "JMP" and q not in self.leaders \
                and target in (q + 2 + code[q + 1].sbx, q + 2):
            return q
        self.pending, self.statements, self.multi = saved[0], saved[2], saved[3]
        del self.out[saved[1]:]
        return None

    def conditional(self, pc: int, end: int) -> int:
        code = self.code
        if self._is_bool_value(pc):
            taken, _ = self.conditions(code[pc])
            self.set(pc + 3, code[pc + 2].a, taken)
            return pc + 4
        if code[pc + 1].op != "JMP":
            raise ValueError(f"test without JMP at pc {pc}")
        chain = []
        at = pc
        while True:
            taken, fall = self.conditions(code[at])
            target = at + 2 + code[at + 1].sbx
            chain.append((taken, fall, target, at + 2))
            at = self.chain_next(at + 2, end, target)
            if at is None:
                break
        final = chain[-1][2]
        body = chain[-1][3]
        condition = chain[-1][1]
        for taken, fall, target, _ in reversed(chain[:-1]):
            if target == body:
                condition = f"({taken} or {condition})"
            else:
                condition = f"({fall} and {condition})"
        taken_all = _negate(condition) if len(chain) > 1 else chain[-1][0]
        self.flush()
        if final == self.loop_exit:
            self.line(f"if {_strip_parens(taken_all)} then")
            self.line("    break")
            self.line("end")
            self.statements += 1
            return body
        if final <= pc or final > end:
            self.line(f"if {_strip_parens(taken_all)} then")
            self.indent += 1
            self.goto(final)
            self.indent -= 1
            self.line("end")
            return body
        self.line(f"if {_strip_parens(condition)} then")
        stop = final
        else_end = None
        if final - 1 >= body and code[final - 1].op == "JMP":
            exit_pc = final + code[final - 1].sbx
            if final < exit_pc <= end and exit_pc != self.loop_exit \
                    and (code[exit_pc].op != "TFORLOOP"
                         or exit_pc in self.open_for):
                stop, else_end = final - 1, exit_pc
        self.nested(body, stop)
        if else_end is not None:
            self.line("else")
            mark = len(self.out)
            self.nested(final, else_end)
            self.merge_elseif(mark)
        self.line("end")
        return else_end if else_end is not None else final

    def merge_elseif(self, mark: int):
        """Turn `else` + a body that is one if statement into `elseif`."""
        pad = "    " * self.indent
        inner = pad + "    "
        body = self.out[mark:]
        tops = [text for text in body if text.startswith(inner)
                and not text.startswith(inner + " ")]
        if len(tops) < 2 or not tops[0].startswith(inner + "if ") \
                or body[-1] != inner + "end":
            return
        if any(not text.startswith(inner + ("else", "end", "if ")[k])
               for text in tops[1:-1] for k in [text.lstrip()[:2] != "el" and 1 or 0]):
            return
        self.out[mark - 1:] = [pad + "else" + body[0].lstrip()] + [
            text[4:] for text in body[1:-1]]

    def nested(self, start: int, end: int):
        self.indent += 1
        self.block(start, end)
        self.flush()
        self.indent -= 1

    def while_loop(self, head: int) -> int:
        back = self.loop_ends[head]
        self.flush()
        self.active.add(head)
        saved = self.loop_exit
        self.loop_exit = back + 1
        self.line("while true do")
        mark = len(self.out)
        self.nested(head, back)
        pad = "    " * (self.indent + 1)
        if len(self.out) >= mark + 3 and self.out[mark].startswith(pad + "if ") \
                and self.out[mark + 1] == pad + "    break" \
                and self.out[mark + 2] == pad + "end":
            test = self.out[mark].strip()[3:-5]
            self.out[mark - 1:mark + 3] = [
                "    " * self.indent + f"while {_strip_parens(_negate(test))} do"]
        self.line("end")
        self.loop_exit = saved
        return back + 1

    def numeric_for(self, i) -> int:
        close = i.pc + 1 + i.sbx
        step = self.read(i.a + 2)
        limit = self.read(i.a + 1)
        start = self.read(i.a)
        self.flush()
        self.loop_vars.add(i.a + 3)
        var = self.name(i.a + 3)
        text = f"{var} = {_strip_parens(start)}, {_strip_parens(limit)}"
        if step != "1":
            text += f", {_strip_parens(step)}"
        self.line(f"for {text} do")
        saved = self.loop_exit
        self.loop_exit = close + 1
        self.nested(i.pc + 1, close)
        self.loop_vars.discard(i.a + 3)
        self.loop_exit = saved
        self.line("end")
        return close + 1

    def generic_for(self, i, target: int) -> int:
        loop = self.code[target]
        self.flush()
        variables = {loop.a + 3 + n for n in range(loop.c)}
        self.loop_vars |= variables
        names = ", ".join(self.name(reg) for reg in sorted(variables))
        source = ", ".join(self.name(loop.a + n) for n in range(3))
        pattern = f"{source} = "
        if self.out and self.out[-1].lstrip().startswith(pattern):
            source = self.out.pop().split(" = ", 1)[1]
        self.line(f"for {names} in {source} do")
        saved = self.loop_exit
        self.loop_exit = target + 2
        self.open_for.add(target)
        self.nested(i.pc + 1, target)
        self.open_for.discard(target)
        self.loop_vars -= variables
        self.loop_exit = saved
        self.line("end")
        return target + 2

    def block(self, pc: int, end: int):
        code = self.code
        while pc < end:
            if pc in self.labels:
                self.flush()
                self.line(f"::L{pc}::")
            i = code[pc]
            if pc in self.loop_ends and pc not in self.active:
                pc = self.while_loop(pc)
            elif i.op in _TESTS:
                pc = self.conditional(pc, end)
            elif i.op == "FORPREP":
                pc = self.numeric_for(i)
            elif i.op == "JMP":
                target = pc + 1 + i.sbx
                if code[target].op == "TFORLOOP" \
                        and target not in self.open_for:
                    pc = self.generic_for(i, target)
                    continue
                self.flush()
                if target == self.loop_exit:
                    self.line("break")
                    self.statements += 1
                else:
                    self.goto(target)
                pc += 1
            else:
                pc = self.step(i)

    def run(self) -> list[str]:
        self.changed = True
        while self.changed:
            self.changed = False
            self.out = []
            self.pending = {}
            self.declared = set()
            self.active = set()
            self.open_for = set()
            self.loop_vars = set()
            self.loop_exit = None
            self.multi = 0
            self.statements = 0
            self.indent = self.depth
            self.block(0, len(self.code))
            self.flush()
        text = "\n".join(self.out)
        names = ", ".join(
            self.local_name(r) for r in sorted(self.declared)
            if re.search(rf"\b{self.local_name(r)}\b", text))
        if names:
            self.out.insert(0, "    " * self.depth + f"local {names}")
        return self.out


def _function_text(fn: LuaFunction, upvalues: list[str], depth: int):
    worker = None
    try:
        worker = _Decompiler(fn, upvalues, depth)
        body = worker.run()
    except (ValueError, IndexError, KeyError) as ex:
        pad = "    " * depth
        body = [f"{pad}-- could not decompile: {ex or type(ex).__name__}"]
        body += [f"{pad}-- {t}" for t in disassemble(fn)]
    names = [worker.local_name(n) if worker else f"a{n}"
             for n in range(fn.num_params)]
    if fn.is_vararg:
        names.append("...")
    return ", ".join(names), body


def decompile(chunk: LuaChunk) -> str:
    """Pseudo-Lua source for a parsed chunk."""
    _, body = _function_text(chunk.main, [], 0)
    return "\n".join(body) + "\n"
