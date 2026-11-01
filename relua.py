#!/usr/bin/env python3
"""
Lua 5.1 bytecode to compilable Lua 5.1 source.

Unlike luadec.py this recovers real local scopes from register allocation and
never prints `goto`. It reads the code the way the reference compiler wrote it:
temporaries live above the active locals, so a value left on the stack when a
statement ends must have been a `local`.

    tree = relua.decompile(chunk)      # AST, names unset
    text = relua.render(tree)

Chunks are stripped, so locals print as v<id> unless their `name` is set first.
"""
from __future__ import annotations

import re

_RK = 256
_ARITH = {"ADD": "+", "SUB": "-", "MUL": "*", "DIV": "/", "MOD": "%", "POW": "^"}
_TESTS = ("EQ", "LT", "LE", "TEST", "TESTSET")
_KEYWORDS = {
    "and", "break", "do", "else", "elseif", "end", "false", "for", "function",
    "if", "in", "local", "nil", "not", "or", "repeat", "return", "then", "true",
    "until", "while",
}
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class Bail(Exception):
    """A sandboxed expression run reached something that is a statement."""


class Fail(Exception):
    """Code that fits no known compiler pattern."""


# --- AST ---------------------------------------------------------------------

class Var:
    __slots__ = ("reg", "name", "kind", "hints", "reads", "writes", "fn", "uid", "literal", "assigned")
    count = 0

    def __init__(self, reg, kind="local", fn=None):
        self.reg, self.kind, self.fn = reg, kind, fn
        self.name = None
        self.hints = []
        self.reads = self.writes = 0
        self.literal = None
        self.assigned = 0
        Var.count += 1
        self.uid = Var.count


class Expr:
    prec = 100


class Const(Expr):
    def __init__(self, value):
        self.value = value
        if isinstance(value, float) and value < 0:
            self.prec = 8


class Global(Expr):
    def __init__(self, name):
        self.name = name


class Ref(Expr):
    """A local or upvalue."""
    def __init__(self, var):
        self.var = var


class Index(Expr):
    def __init__(self, obj, key):
        self.obj, self.key = obj, key


class Method(Expr):
    def __init__(self, obj, name):
        self.obj, self.name = obj, name


class Call(Expr):
    def __init__(self, func, args, nres):
        self.func, self.args, self.nres = func, args, nres


class Vararg(Expr):
    def __init__(self, nres):
        self.nres = nres


class Binop(Expr):
    _PREC = {"or": 1, "and": 2, "<": 3, "<=": 3, ">": 3, ">=": 3, "==": 3, "~=": 3,
             "..": 5, "+": 6, "-": 6, "*": 7, "/": 7, "%": 7, "^": 10}

    def __init__(self, op, left, right):
        self.op, self.left, self.right = op, left, right
        self.prec = self._PREC[op]


class Unop(Expr):
    prec = 8

    def __init__(self, op, operand):
        self.op, self.operand = op, operand


class Table(Expr):
    def __init__(self, narr, nhash):
        self.narr, self.nhash = narr, nhash
        self.array = []            # values in order
        self.hash = []             # (array position, key, value)


class Func(Expr):
    def __init__(self, proto, upvars):
        self.proto, self.upvars = proto, upvars
        self.fn = None


class SelfSlot(Expr):
    pass


class MultiSlot(Expr):
    pass


class Stat:
    pass


class Local(Stat):
    def __init__(self, vars, exprs):
        self.vars, self.exprs = vars, exprs


class LocalFunction(Stat):
    def __init__(self, var, func):
        self.var, self.func = var, func


class Assign(Stat):
    def __init__(self, targets, exprs):
        self.targets, self.exprs = targets, exprs


class CallStat(Stat):
    def __init__(self, call):
        self.call = call


class Return(Stat):
    def __init__(self, exprs):
        self.exprs = exprs


class Break(Stat):
    pass


class If(Stat):
    def __init__(self, cond, then, orelse):
        self.cond, self.then, self.orelse = cond, then, orelse


class While(Stat):
    def __init__(self, cond, body):
        self.cond, self.body = cond, body


class Repeat(Stat):
    def __init__(self, body, cond):
        self.body, self.cond = body, cond


class ForNum(Stat):
    def __init__(self, var, start, stop, step, body):
        self.var, self.start, self.stop, self.step, self.body = var, start, stop, step, body


class ForIn(Stat):
    def __init__(self, vars, exprs, body):
        self.vars, self.exprs, self.body = vars, exprs, body


class Do(Stat):
    def __init__(self, body):
        self.body = body


def neg(e):
    """Logical negation that keeps the bytecode of the original test."""
    if isinstance(e, Unop) and e.op == "not":
        return e.operand
    if isinstance(e, Binop):
        if e.op == "and":
            return Binop("or", neg(e.left), neg(e.right))
        if e.op == "or":
            return Binop("and", neg(e.left), neg(e.right))
        if e.op == "==":
            return Binop("~=", e.left, e.right)
        if e.op == "~=":
            return Binop("==", e.left, e.right)
    return Unop("not", e)


def _fb2int(x):
    e = (x >> 3) & 31
    return x if e == 0 else ((x & 7) + 8) << (e - 1)


# --- register use ------------------------------------------------------------

class _Flow:
    """Reads, writes and liveness per instruction."""

    def __init__(self, proto):
        self.code = code = proto.code
        n = len(code)
        self.pseudo = set()
        self.reads, self.writes, self.kills = [], [], []
        multi = 0
        for i in code:
            r, w, k = self._access(i, multi)
            self.reads.append(r)
            self.writes.append(w)
            self.kills.append(k)
            if i.op in ("CALL", "TAILCALL") and i.c == 0 or i.op == "VARARG" and i.b == 0:
                multi = i.a
            if i.op == "CLOSURE":
                for k in range(proto.protos[i.bx].num_upvalues):
                    self.pseudo.add(i.pc + 1 + k)
        for pc in self.pseudo:
            self.writes[pc], self.kills[pc] = [], None
            if code[pc].op != "MOVE":
                self.reads[pc] = []
        succ = []
        for i in code:
            pc = i.pc
            if i.op == "JMP":
                s = [pc + 1 + i.sbx]
            elif i.op == "FORLOOP":
                s = [pc + 1 + i.sbx, pc + 1]
            elif i.op == "FORPREP":
                s = [pc + 1 + i.sbx]
            elif i.op in _TESTS or i.op == "TFORLOOP":
                s = [pc + 1, pc + 2]
            elif i.op == "LOADBOOL" and i.c:
                s = [pc + 2]
            elif i.op == "RETURN":
                s = []
            else:
                s = [pc + 1]
            succ.append([t for t in s if 0 <= t < n])
        use = [sum(1 << r for r in set(x)) for x in self.reads]
        kill = []
        for pc in range(n):
            m = sum(1 << r for r in set(self.writes[pc]))
            if self.kills[pc] is not None:
                m |= ~((1 << self.kills[pc]) - 1) & ((1 << 260) - 1)
            kill.append(m)
        out = [0] * n
        inn = [0] * n
        changed = True
        while changed:
            changed = False
            for pc in range(n - 1, -1, -1):
                o = 0
                for t in succ[pc]:
                    o |= inn[t]
                x = (o & ~kill[pc]) | use[pc]
                if o != out[pc] or x != inn[pc]:
                    out[pc], inn[pc] = o, x
                    changed = True
        self.live_out, self.live_in = out, inn

    @staticmethod
    def _access(i, multi):
        rk = lambda v: [v] if v < _RK else []
        span = lambda a, b: list(range(a, b + 1))
        op = i.op
        if op == "MOVE":
            return [i.b], [i.a], None
        if op in ("LOADK", "GETGLOBAL", "GETUPVAL", "NEWTABLE", "LOADBOOL", "CLOSURE"):
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
        if op in ("EQ", "LT", "LE"):
            return rk(i.b) + rk(i.c), [], None
        if op == "TESTSET":
            return [i.b], [i.a], None
        if op in ("CALL", "TAILCALL"):
            last = i.a + i.b - 1 if i.b else multi
            w = span(i.a, i.a + i.c - 2) if i.c else [i.a]
            return span(i.a, last), w, i.a
        if op == "RETURN":
            return span(i.a, i.a + i.b - 2 if i.b else multi), [], None
        if op == "SETLIST":
            return span(i.a, i.a + i.b if i.b else multi), [], None
        if op == "FORPREP":
            return span(i.a, i.a + 2), [i.a], None
        if op == "FORLOOP":
            return span(i.a, i.a + 2), [i.a, i.a + 3], None
        if op == "TFORLOOP":
            return span(i.a, i.a + 2), span(i.a + 3, i.a + 2 + i.c), i.a + 3
        if op == "VARARG":
            return [], span(i.a, i.a + i.b - 2) if i.b else [i.a], None
        return [], [], None


# --- decompiler --------------------------------------------------------------

class _Item:
    __slots__ = ("c", "target", "end", "op", "reg", "kept")

    def __init__(self, c, target, end, op=None, reg=None):
        self.c, self.target, self.end, self.op, self.reg = c, target, end, op, reg


class Fn:
    def __init__(self, proto, upvars):
        self.proto = proto
        self.code = proto.code
        self.upvars = upvars
        self.flow = _Flow(proto)
        self.params = [Var(r, "param", self) for r in range(proto.num_params)]
        self.vararg = bool(proto.is_vararg)
        self.pending = {}
        self.tags = {}
        self.regvar = {v.reg: v for v in self.params}
        self.nact = proto.num_params
        if proto.is_vararg & 1:
            arg = Var(proto.num_params, "arg", self)
            arg.name = "arg"
            self.regvar[arg.reg] = arg
            self.nact += 1
        self.out = []
        self.pure = 0
        self.dest = None
        self.block_end = len(self.code)
        self.loop_exit = None
        self.rep = None
        self.until = None
        self.active = set()
        self.open_whiles = {}
        self._find_loops()

    def _find_loops(self):
        code = self.code
        self.whiles, self.repeats = {}, {}
        conditional = []
        for i in code:
            if i.op != "JMP" or i.pc + 1 + i.sbx > i.pc:
                continue
            head = i.pc + 1 + i.sbx
            prev = code[i.pc - 1].op if i.pc else ""
            if prev == "TFORLOOP":
                continue
            if prev in _TESTS:
                conditional.append((head, i.pc))
            else:
                self.whiles[head] = [max(self.whiles.get(head, [0]) + [i.pc])]
        for head, pc in conditional:
            if head in self.whiles and pc < max(self.whiles[head]):
                continue
            self.repeats[head] = max(self.repeats.get(head, 0), pc + 1)
        self.backs = {pc for backs in self.whiles.values() for pc in backs}
        for backs in self.whiles.values():
            backs.reverse()         # outermost first

    # state

    def snapshot(self):
        return dict(self.pending), dict(self.tags), self.nact, dict(self.regvar), len(self.out)

    def restore(self, s):
        self.pending, self.tags, self.nact, self.regvar = dict(s[0]), dict(s[1]), s[2], dict(s[3])
        del self.out[s[4]:]

    def free(self):
        return max(self.pending) + 1 if self.pending else self.nact

    def emit(self, stat):
        if self.pure:
            raise Bail
        self.out.append(stat)

    def new_var(self, reg):
        v = Var(reg, "local", self)
        self.regvar[reg] = v
        return v

    def promote(self, top):
        """Everything on the stack up to `top` is a local."""
        if self.pure:
            raise Bail
        for reg in sorted(self.pending):
            if reg > top:
                break
            e = self.pending[reg]
            if isinstance(e, (SelfSlot, MultiSlot, Method)) or \
                    isinstance(e, (Call, Vararg)) and e.nres != 1:
                raise Fail(f"cannot make R{reg} a local")
            for gap in range(self.nact, reg):
                self.out.append(Local([self.new_var(gap)], []))
            del self.pending[reg]
            self.tags.pop(reg, None)
            var = self.new_var(reg)
            self.nact = reg + 1
            if isinstance(e, Const) and e.value is None and self.tags.get(reg) != "keep":
                self.out.append(Local([var], []))
            else:
                self.out.append(Local([var], [e]))

    def promote_all(self):
        if self.pending:
            self.promote(max(self.pending))

    def declare_to(self, reg):
        """Locals with no write: `local a, b` at the top of a function."""
        if self.pure:
            raise Bail
        self.promote_all()
        for gap in range(self.nact, reg + 1):
            self.out.append(Local([self.new_var(gap)], []))
        self.nact = max(self.nact, reg + 1)

    def live_after(self, r, pc):
        return bool(self.flow.live_out[pc] >> r & 1)

    def assigned_in_branches(self, r, pc):
        """r is written in an `if` and read after it, so it outlives the `if`."""
        code = self.code
        j = code[pc + 1]
        if j.op != "JMP" or j.sbx <= 0:
            return False
        false = pc + 2 + j.sbx
        escape = code[false - 1]
        if escape.op != "JMP" or escape.sbx <= 0 or false >= self.block_end:
            return False
        after = false + escape.sbx
        return after < len(code) and bool(self.flow.live_in[after] >> r & 1) \
            and any(r in self.flow.writes[q] for q in range(pc + 2, false))

    def stack_keeps(self, r, pc):
        """True when later temporaries sit above r, so r stayed a local."""
        if self.code[pc].op in _TESTS and self.assigned_in_branches(r, pc):
            return True
        for q in range(pc + 1, self.block_end):
            if q in self.flow.pseudo:
                continue
            w = [x for x in self.flow.writes[q] if x >= r]
            if self.flow.kills[q] is not None and self.flow.kills[q] >= r:
                w.append(self.flow.kills[q])
            if w:
                return min(w) > r
        return False

    def must_be_local(self, r, pc, dest, direct):
        tag = self.tags.get(r)
        if direct and tag in ("copy", "const"):
            return True
        if max(self.pending) > r:
            return True
        if dest is not None and dest != r and dest >= self.nact and dest > r and dest != self.dest:
            return True
        if dest == r:
            return False
        if self.live_after(r, pc):
            return True
        if dest is None or dest < self.nact:
            return self.stack_keeps(r, pc)
        return False

    def use(self, r, pc, dest=None, direct=True):
        if r in self.pending:
            if self.must_be_local(r, pc, dest, direct):
                self.promote(r)
                return Ref(self.regvar[r])
            e = self.pending.pop(r)
            self.tags.pop(r, None)
            return e
        var = self.regvar.get(r) if r < self.nact else None
        if var is None:
            if pc == 0 and r == self.nact and not self.live_after(r, pc):
                return Const(None)      # the compiler drops LOADNIL at pc 0
            self.declare_to(r)
            var = self.regvar[r]
        return Ref(var)

    def rk(self, v, pc, dest=None):
        if v >= _RK:
            return Const(self.proto.constants[v - _RK])
        return self.use(v, pc, dest)

    def pop(self, r):
        """A register that can only be a temporary."""
        if r in self.pending:
            self.tags.pop(r, None)
            return self.pending.pop(r)
        if r < self.nact and r in self.regvar:
            return Ref(self.regvar[r])
        raise Fail(f"R{r} is not on the stack")

    def put(self, pc, a, expr, tag="val"):
        if a == self.dest:
            self.pending[a] = expr
            return
        if a < self.nact:
            self.store_local(pc, a, expr)
            return
        if a in self.pending:
            self.promote(a)
            self.store_local(pc, a, expr)
            return
        if a > self.free():
            self.declare_to(a - 1)
        self.pending[a] = expr
        self.tags[a] = tag

    def store_local(self, pc, a, expr):
        self.promote_all()
        var = self.regvar.get(a)
        if var is None:
            raise Fail(f"store to unknown R{a}")
        self.emit(Assign([Ref(var)], [expr]))

    # straight-line code

    def step(self, i):
        """Run one instruction; returns the next pc."""
        op, a, pc = i.op, i.a, i.pc
        k = self.proto.constants
        if op == "MOVE":
            return self.do_move(i)
        if op == "LOADK":
            self.put(pc, a, Const(k[i.bx]), "const" if i.bx < _RK else "val")
        elif op == "LOADBOOL":
            if i.c:
                raise Fail("LOADBOOL skip outside a value")
            self.put(pc, a, Const(bool(i.b)), "const")
        elif op == "LOADNIL":
            for r in range(a, i.b + 1):
                self.put(pc, r, Const(None), "const")
        elif op == "GETUPVAL":
            self.put(pc, a, Ref(self.upvars[i.b]))
        elif op == "GETGLOBAL":
            self.put(pc, a, Global(k[i.bx]))
        elif op == "GETTABLE":
            if i.c < _RK and i.c < i.b and i.c in self.pending:
                self.promote(i.c)
            key = self.rk(i.c, pc, a)
            obj = self.use(i.b, pc, a)
            self.put(pc, a, Index(obj, key))
        elif op == "SETGLOBAL":
            self.store(i, Global(k[i.bx]))
        elif op == "SETUPVAL":
            self.store(i, Ref(self.upvars[i.b]))
        elif op == "SETTABLE":
            return self.do_settable(i)
        elif op == "NEWTABLE":
            self.put(pc, a, Table(_fb2int(i.b), _fb2int(i.c)))
        elif op == "SELF":
            key = self.rk(i.c, pc, a)
            obj = self.use(i.b, pc, a, direct=False)
            self.pending[a] = Method(obj, key.value)
            self.pending[a + 1] = SelfSlot()
        elif op in _ARITH:
            if i.b < _RK and i.c < i.b and i.c in self.pending:
                self.promote(i.c)
            c = self.rk(i.c, pc, a)
            b = self.rk(i.b, pc, a)
            self.put(pc, a, Binop(_ARITH[op], b, c))
        elif op in ("UNM", "NOT", "LEN"):
            b = self.use(i.b, pc, a)
            self.put(pc, a, Unop({"UNM": "-", "NOT": "not", "LEN": "#"}[op], b))
        elif op == "CONCAT":
            parts = [self.pop(r) for r in range(i.b, i.c + 1)]
            e = parts[-1]
            for p in reversed(parts[:-1]):
                e = Binop("..", p, e)
            self.put(pc, a, e)
        elif op == "SETLIST":
            self.do_setlist(i)
        elif op == "CLOSURE":
            return self.do_closure(i)
        elif op == "VARARG":
            return self.do_vararg(i)
        elif op == "CALL":
            return self.do_call(i)
        elif op == "TAILCALL":
            call = self.make_call(i, -1)
            self.promote_all()
            self.emit(Return([call]))
            return pc + 2
        elif op == "RETURN":
            self.do_return(i)
        elif op == "CLOSE":
            pass
        else:
            raise Fail(f"unexpected {op} at {pc}")
        return pc + 1

    def do_move(self, i):
        a, b, pc = i.a, i.b, i.pc
        if a in self.pending and a != self.dest and b > a:
            v = self.pop(b)         # `local a, b` then `a = f()`
            self.promote(a)
            self.store_local(pc, a, v)
            return pc + 1
        if a < self.nact or a == self.dest:
            if a == self.dest and b in self.pending and b > a:
                self.pending[a] = self.pop(b)
                return pc + 1
            v = self.use(b, pc, a if a == self.dest else None, direct=False)
            if a == self.dest:
                self.pending[a] = v
                return pc + 1
            if self.multi_store(i, v, Ref(self.regvar[a])):
                return self.after_multi
            self.store_local(pc, a, v)
            return pc + 1
        if b in self.pending:
            if b > a:
                self.pending[a] = self.pop(b)
                return pc + 1
            self.promote(b)
        self.put(pc, a, self.use(b, pc, a, direct=False), "copy")
        return pc + 1

    def store(self, i, target):
        pc = i.pc
        v = self.use(i.a, pc, None, direct=False)
        if self.multi_store(i, v, target):
            return
        self.promote_all()
        self.emit(Assign([target], [v]))

    after_multi = 0

    def multi_store(self, i, value, target):
        """`a, b = x, y` stores from the top of the stack down."""
        code = self.code
        reg = i.a if i.op != "MOVE" else i.b
        below = reg - 1
        q = i.pc + 1
        if below not in self.pending or q >= len(code):
            return False
        n = code[q]
        if not (n.op in ("SETGLOBAL", "SETUPVAL") and n.a == below
                or n.op == "MOVE" and n.b == below and n.a < self.nact):
            return False
        if self.live_after(below, q):
            return False
        targets, values = [target], [value]
        while True:
            n = code[q]
            if n.op == "SETGLOBAL":
                t = Global(self.proto.constants[n.bx])
                r = n.a
            elif n.op == "SETUPVAL":
                t = Ref(self.upvars[n.b])
                r = n.a
            elif n.op == "MOVE" and n.a < self.nact and n.b >= self.nact:
                t = Ref(self.regvar[n.a])
                r = n.b
            else:
                break
            if r not in self.pending or r != below:
                break
            targets.insert(0, t)
            values.insert(0, self.pop(r))
            below -= 1
            q += 1
            if q >= len(code):
                break
        self.promote_all()
        self.emit(Assign(targets, values))
        self.after_multi = q
        self.skip_to = q
        return True

    skip_to = None

    def do_settable(self, i):
        a, pc = i.a, i.pc
        t = self.pending.get(a)
        if isinstance(t, Table) and len(t.hash) < t.nhash:
            v = self.rk(i.c, pc, i.c)
            key = self.rk(i.b, pc, i.b)
            t.hash.append((self.array_position(a, t), key, v))
            return pc + 1
        v = self.rk(i.c, pc, None)
        key = self.rk(i.b, pc, None)
        if isinstance(self.pending.get(a), Table):
            self.promote(a)
        obj = self.use(a, pc, None)
        self.promote_all()
        self.emit(Assign([Index(obj, key)], [v]))
        return pc + 1

    def array_position(self, a, t):
        return len(t.array) + sum(1 for r in self.pending if r > a)

    def do_setlist(self, i):
        a = i.a
        t = self.pending.get(a)
        if not isinstance(t, Table):
            raise Fail("SETLIST on a non-table")
        if i.b:
            regs = range(a + 1, a + i.b + 1)
        else:
            regs = sorted(r for r in self.pending if r > a)
        for r in regs:
            t.array.append(self.pop(r))

    def do_closure(self, i):
        proto = self.proto.protos[i.bx]
        pc = i.pc
        ups = []
        recursive = False
        for n in range(proto.num_upvalues):
            cap = self.code[pc + 1 + n]
            if cap.op == "MOVE":
                if cap.b == i.a and i.a >= self.nact:
                    recursive = True
                    if self.pure:
                        raise Bail
                    self.promote_all()
                    var = self.new_var(i.a)
                    self.nact = i.a + 1
                    ups.append(var)
                    continue
                if cap.b in self.pending:
                    self.promote(cap.b)
                ups.append(self.regvar[cap.b])
            else:
                ups.append(self.upvars[cap.b])
        func = Func(proto, ups)
        next_pc = pc + 1 + proto.num_upvalues
        if recursive:
            self.emit(LocalFunction(self.regvar[i.a], func))
        else:
            self.put(pc, i.a, func)
        return next_pc

    def do_vararg(self, i):
        a, pc = i.a, i.pc
        if i.b == 0:
            self.pending[a] = Vararg(-1)
        elif i.b == 2:
            self.put(pc, a, Vararg(1))
        else:
            self.promote_all()
            vs = [self.new_var(a + n) for n in range(i.b - 1)]
            self.nact = a + i.b - 1
            self.emit(Local(vs, [Vararg(i.b - 1)]))
        return pc + 1

    def make_call(self, i, nres):
        a = i.a
        if i.b:
            regs = list(range(a + 1, a + i.b))
        else:
            regs = sorted(r for r in self.pending if r > a)
        args = [self.pop(r) for r in regs]
        f = self.pop(a)
        if isinstance(f, Method):
            if args and isinstance(args[0], SelfSlot):
                args = args[1:]
        return Call(f, [x for x in args if not isinstance(x, SelfSlot)], nres)

    def do_call(self, i):
        a, pc, c = i.a, i.pc, i.c
        call = self.make_call(i, c - 1)
        if c == 1:
            self.promote_all()
            self.emit(CallStat(call))
        elif c == 2:
            self.put(pc, a, call)
        elif c == 0:
            self.pending[a] = call
        else:
            n = self.code[pc + 1]
            if n.op == "JMP" and self.code[pc + 2 + n.sbx].op == "TFORLOOP" \
                    and self.code[pc + 2 + n.sbx].a == a:
                self.pending[a] = call
                for r in range(a + 1, a + c - 1):
                    self.pending[r] = MultiSlot()
                return pc + 1
            top = a + c - 2
            if n.op in ("SETGLOBAL", "SETUPVAL") and n.a == top \
                    or n.op == "MOVE" and n.b == top and n.a < self.nact:
                return self.multi_result(i, call)
            self.promote_all()
            vs = [self.new_var(a + n) for n in range(c - 1)]
            self.nact = a + c - 1
            self.emit(Local(vs, [call]))
        return pc + 1

    def multi_result(self, i, call):
        """`a, b = f()`"""
        q = i.pc + 1
        top = i.a + i.c - 2
        targets = []
        for r in range(top, i.a - 1, -1):
            n = self.code[q]
            if n.op == "SETGLOBAL" and n.a == r:
                targets.insert(0, Global(self.proto.constants[n.bx]))
            elif n.op == "SETUPVAL" and n.a == r:
                targets.insert(0, Ref(self.upvars[n.b]))
            elif n.op == "MOVE" and n.b == r and n.a < self.nact:
                targets.insert(0, Ref(self.regvar[n.a]))
            else:
                raise Fail("multiple assignment")
            q += 1
        self.promote_all()
        self.emit(Assign(targets, [call]))
        return q

    def do_return(self, i):
        a, b, pc = i.a, i.b, i.pc
        if b == 1:
            if pc == len(self.code) - 1:
                return
            self.promote_all()
            self.emit(Return([]))
            return
        if b == 2:
            e = self.pending.get(a)
            if isinstance(e, Call) and e.nres == 1:
                self.promote(a)     # `local x = f(); return x`, not `return (f())`
            v = self.use(a, pc, None, direct=False)
            self.promote_all()
            self.emit(Return([v]))
            return
        regs = list(range(a, a + b - 1)) if b else sorted(r for r in self.pending if r >= a)
        vals = [self.pop(r) for r in regs]
        self.promote_all()
        self.emit(Return(vals))

    # conditions

    def test_item(self, pc, value_mode):
        i = self.code[pc]
        j = self.code[pc + 1]
        if j.op != "JMP":
            raise Fail("test without JMP")
        target = pc + 2 + j.sbx
        reg = None
        if i.op == "TEST":
            if value_mode and i.a in self.pending:
                v = self.pop(i.a)
                reg = i.a
            else:
                v = self.use(i.a, pc, None, direct=False)
            c = v if i.c else neg(v)
        elif i.op == "TESTSET":
            if i.b in self.pending:
                v = self.pop(i.b)
            else:
                v = self.use(i.b, pc, None, direct=False)
            reg = i.a
            c = v if i.c else neg(v)
        else:
            if i.b < _RK and i.c < _RK and i.b > i.c:
                left = self.rk(i.b, pc)
                right = self.rk(i.c, pc)
            else:
                right = self.rk(i.c, pc)
                left = self.rk(i.b, pc)
            if i.op == "EQ":
                e = Binop("==", left, right)
                c = e if i.a else neg(e)
            else:
                swap = i.b >= _RK and i.c < _RK or (i.b < _RK and i.c < _RK and i.b > i.c >= self.nact)
                if swap:
                    e = Binop(">" if i.op == "LT" else ">=", right, left)
                else:
                    e = Binop("<" if i.op == "LT" else "<=", left, right)
                c = e if i.a else Unop("not", e)
        return _Item(c, target, pc + 2, None, reg)

    def same_place(self, target, pc):
        """Jumps to a JMP take that JMP's target: the compiler threads them."""
        if target == pc:
            return True
        if pc is None or pc >= len(self.code):
            return False
        i = self.code[pc]
        return i.op == "JMP" and (not pc or self.code[pc - 1].op not in _TESTS) \
            and pc + 1 + i.sbx == target

    def resolve(self, target, lo, end):
        """Where in [lo, end] a jump lands, undoing threading. None if nowhere."""
        if lo <= target <= end:
            return target
        for q in range(end, lo - 1, -1):
            if q < len(self.code) and self.same_place(target, q) and q != target:
                return q
        return None

    def merge(self, a, b):
        if self.same_place(a.target, b.end):
            return _Item(Binop("and", neg(a.c), b.c), b.target, b.end, b.op, b.reg)
        if a.target == b.target:
            return _Item(Binop("or", a.c, b.c), b.target, b.end, b.op, b.reg)
        return None

    def reduce_if(self, items):
        items = list(items)
        while len(items) > 1:
            for n in range(len(items) - 1):
                m = self.merge(items[n], items[n + 1])
                if m:
                    items[n:n + 2] = [m]
                    break
            else:
                return None
        return items[0]

    def collect(self, pc, end):
        """Test items from pc, each with the state after it."""
        items = []
        states = []
        items.append(self.test_item(pc, False))
        self.promote_all()      # an `if` starts a statement
        states.append(self.snapshot())
        q = pc + 2
        self.pure += 1
        try:
            while q < end:
                while self.code[q].op not in _TESTS:
                    op = self.code[q].op
                    if op in ("JMP", "RETURN", "TAILCALL", "FORPREP", "FORLOOP", "TFORLOOP", "CLOSE") \
                            or q in self.whiles or q in self.repeats:
                        raise Bail
                    q = self.step(self.code[q])
                    if q >= end:
                        raise Bail
                it = self.test_item(q, False)
                if self.code[q + 1].op != "JMP" or self.pending:
                    raise Bail      # anything left on the stack is a local of the body
                items.append(it)
                states.append(self.snapshot())
                q += 2
        except (Bail, Fail, KeyError, IndexError):
            pass
        finally:
            self.pure -= 1
        return items, states

    def rep_head(self):
        return self.rep[0] if self.rep else None

    def try_value(self, pc, end):
        """`x = a and b or c`, comparisons as values. None if this is no value."""
        code = self.code
        s = self.snapshot()
        saved_dest = self.dest
        self.pure += 1
        try:
            return self._value(pc, end)
        except (Bail, Fail, KeyError, IndexError):
            self.restore(s)
            return None
        finally:
            self.pure -= 1
            self.dest = saved_dest

    def _value(self, pc, end):
        code = self.code
        items = []
        q = pc
        reg = None
        tail_bool = None
        temp_test = None
        while True:
            if q == pc and self.pure == 1:
                self.pure = 0
                try:
                    it = self.test_item(q, True)
                finally:
                    self.pure = 1
            else:
                it = self.test_item(q, True)
            if it.target <= q:
                raise Bail
            if code[q].op == "TESTSET":
                if reg is not None and reg != it.reg:
                    raise Bail
                reg = it.reg
                self.dest = reg
            elif code[q].op == "TEST" and it.reg is not None:
                temp_test = it.reg
            items.append(it)
            q += 2
            limit = max(x.target for x in items)
            if limit > end:
                raise Bail
            if q == limit and tail_bool is None and code[q - 1].op != "LOADBOOL":
                # nothing after the last test
                if not (code[q].op == "LOADBOOL"):
                    raise Bail
            found = False
            while q < limit:
                i = code[q]
                if i.op in _TESTS:
                    found = True
                    break
                if i.op == "JMP" and i.sbx == 2 and q + 3 >= limit \
                        and code[q + 1].op == "LOADBOOL" and code[q + 2].op == "LOADBOOL":
                    tail_bool = (q + 1, q + 2, q + 3, True)
                    q += 3
                    break
                if i.op == "LOADBOOL" and i.b == 0 and i.c == 1 and q + 2 >= limit \
                        and code[q + 1].op == "LOADBOOL" and code[q + 1].b == 1 and code[q + 1].c == 0:
                    tail_bool = (q, q + 1, q + 2, False)
                    q += 2
                    break
                if i.op in ("JMP", "RETURN", "TAILCALL", "FORPREP", "FORLOOP", "TFORLOOP", "CLOSE"):
                    raise Bail
                if reg is None and temp_test is not None:
                    self.dest = temp_test
                q = self.step(i)
            if found:
                continue
            break
        L = q
        if tail_bool:
            r2 = code[tail_bool[0]].a
            if reg is not None and reg != r2:
                raise Bail
            reg = r2
        if reg is None:
            reg = temp_test
        if reg is None:
            raise Bail
        if temp_test is not None and temp_test != reg:
            raise Bail
        if not tail_bool and reg == temp_test and reg >= self.nact \
                and not (L < len(code) and self.flow.live_in[L] >> reg & 1):
            raise Bail
        exits = {L}
        ops = {}
        if tail_bool:
            exits |= {tail_bool[0], tail_bool[1]}
        for n, it in enumerate(items):
            i = code[it.end - 2]
            if it.target == L:
                if i.op not in ("TEST", "TESTSET"):
                    raise Bail
                it.op = "or" if i.c else "and"
            elif tail_bool and it.target == tail_bool[0]:
                it.op = "and"
            elif tail_bool and it.target == tail_bool[1]:
                it.op = "or"
        # the last comparison of a boolean value is the tail itself
        tail = self.pending.pop(reg, None)
        self.tags.pop(reg, None)
        if tail is None or (tail_bool and not tail_bool[3]):
            last = items[-1]
            if not tail_bool or last.end != tail_bool[0] or last.target != tail_bool[1]:
                raise Bail
            tail = last.c
            items.pop()
        while items:
            merged = False
            for n in range(len(items) - 1):
                a, b = items[n], items[n + 1]
                if a.target == b.end:
                    items[n:n + 2] = [_Item(Binop("and", neg(a.c), b.c), b.target, b.end, b.op, b.reg)]
                    merged = True
                    break
                if a.target == b.target and a.op == b.op:
                    items[n:n + 2] = [_Item(Binop("or", a.c, b.c), b.target, b.end, b.op, b.reg)]
                    merged = True
                    break
            if merged:
                continue
            last = items.pop()
            if last.target not in exits:
                raise Bail
            kept = last.c if last.op == "or" else neg(last.c)
            tail = Binop(last.op, kept, tail)
        self.value_reg = reg
        self.value_expr = tail
        self.value_next = L
        return True

    def body_block(self, start, end, nactvar=None):
        """A nested block; returns its statements."""
        saved = self.out, self.nact, dict(self.regvar), self.block_end, self.pending, self.tags
        self.out, self.pending, self.tags = [], {}, {}
        self.block_end = end
        if nactvar is not None:
            self.nact = nactvar
        self.block(start, end)
        body = self.out
        self.out, self.nact, self.regvar, self.block_end, self.pending, self.tags = saved
        return body

    def conditional(self, pc, end):
        code = self.code
        if self.try_value(pc, end):
            reg, expr = self.value_reg, self.value_expr
            if reg < self.nact:
                self.store_local(pc, reg, expr)
            else:
                self.pending[reg] = expr
                self.tags[reg] = "val"
            return self.value_next
        items, states = self.collect(pc, end)
        for k in range(len(items), 0, -1):
            r = self.reduce_if(items[:k])
            if r is None:
                continue
            if self.rep and r.target == self.rep[0] and r.end == self.rep[1]:
                self.restore(states[k - 1])
                self.promote_all()
                self.until = neg(r.c)
                return r.end
            false = self.resolve(r.target, r.end, end)
            if false is None:
                continue
            self.restore(states[k - 1])
            self.promote_all()
            return self.if_statement(neg(r.c), r.end, false, end)
        raise Fail(f"condition at {pc}")

    def if_statement(self, cond, body, false, end):
        code = self.code
        stop, after = false, false
        has_else = False
        if false > body and code[false - 1].op == "JMP":
            j = code[false - 1]
            prev = code[false - 2].op if false - 2 >= body else ""
            if prev not in _TESTS and false - 1 not in self.backs \
                    and not self.same_place(false + j.sbx, self.loop_exit):
                exit_pc = self.resolve(false + j.sbx, false, end)
                if exit_pc is not None:
                    has_else = True
                    stop, after = false - 1, exit_pc
        then = self.body_block(body, stop)
        orelse = self.body_block(false, after) if has_else else None
        self.emit(If(cond, then, orelse))
        return after

    def loop_body(self, start, end, exit_pc, nactvar=None):
        saved = self.loop_exit, self.rep
        self.loop_exit, self.rep = exit_pc, None
        body = self.body_block(start, end, nactvar)
        self.loop_exit, self.rep = saved
        return body

    def next_back(self, head):
        backs = self.whiles.get(head)
        n = self.open_whiles.get(head, 0)
        return backs[n] if backs and n < len(backs) else None

    def while_statement(self, head, end):
        code = self.code
        back = self.next_back(head)
        exit_pc = back + 1
        self.open_whiles[head] = self.open_whiles.get(head, 0) + 1
        s = self.snapshot()
        cond, start = None, head
        self.pure += 1
        try:
            q = head
            while code[q].op not in _TESTS:
                if q in self.repeats or code[q].op in ("JMP", "RETURN", "TAILCALL", "FORPREP"):
                    raise Bail
                q = self.step(code[q])
        except (Bail, Fail, KeyError):
            q = None
        self.pure -= 1
        if q is not None and q < back:
            self.pure += 1
            try:
                items, _ = self.collect_while(q, back)
            except (Bail, Fail, KeyError):
                items = []
            self.pure -= 1
            for k in range(len(items), 0, -1):
                r = self.reduce_if(items[:k])
                if r is not None and self.same_place(r.target, exit_pc):
                    cond, start = neg(r.c), r.end
                    break
        self.restore(s)
        body = self.loop_body(start, back, exit_pc)
        self.emit(While(cond if cond is not None else Const(True), body))
        return exit_pc

    def collect_while(self, q, back):
        items, states = [], []
        items.append(self.test_item(q, False))
        states.append(None)
        q += 2
        try:
            while q < back:
                while self.code[q].op not in _TESTS:
                    op = self.code[q].op
                    if op in ("JMP", "RETURN", "TAILCALL", "FORPREP", "FORLOOP", "TFORLOOP", "CLOSE"):
                        raise Bail
                    q = self.step(self.code[q])
                    if q >= back:
                        raise Bail
                it = self.test_item(q, False)
                if self.pending:
                    raise Bail
                items.append(it)
                states.append(None)
                q += 2
        except (Bail, Fail, KeyError, IndexError):
            pass
        return items, states

    def repeat_statement(self, head, end):
        e = self.repeats[head]
        self.active.add(head)
        saved = self.loop_exit, self.rep, self.until
        self.loop_exit, self.rep, self.until = e, (head, e), None
        s = self.out, self.nact, dict(self.regvar), self.block_end, self.pending, self.tags
        self.out, self.pending, self.tags = [], {}, {}
        self.block_end = e
        self.block(head, e)
        body, cond = self.out, self.until
        self.out, self.nact, self.regvar, self.block_end, self.pending, self.tags = s
        self.loop_exit, self.rep, self.until = saved
        if cond is None:
            raise Fail("repeat without until")
        self.emit(Repeat(body, cond))
        return e

    def numeric_for(self, i):
        a, pc = i.a, i.pc
        close = pc + 1 + i.sbx
        step = self.pop(a + 2)
        stop = self.pop(a + 1)
        start = self.pop(a)
        self.promote_all()
        saved = dict(self.regvar)
        var = self.new_var(a + 3)
        body = self.loop_body(pc + 1, close, close + 1, a + 4)
        self.regvar = saved
        self.emit(ForNum(var, start, stop, step, body))
        return close + 1

    def generic_for(self, i, target):
        loop = self.code[target]
        a = loop.a
        exprs = []
        for r in range(a, a + 3):
            e = self.pending.get(r)
            if e is None:
                raise Fail("generic for setup")
            if not isinstance(e, MultiSlot):
                exprs.append(e)
            del self.pending[r]
            self.tags.pop(r, None)
        while len(exprs) > 1 and isinstance(exprs[-1], Const) and exprs[-1].value is None:
            exprs.pop()
        self.promote_all()
        saved = dict(self.regvar)
        vs = [self.new_var(a + 3 + n) for n in range(loop.c)]
        body = self.loop_body(i.pc + 1, target, target + 2, a + 3 + loop.c)
        self.regvar = saved
        self.emit(ForIn(vs, exprs, body))
        return target + 2

    def block(self, pc, end):
        code = self.code
        start_nact = self.nact
        while pc < end:
            i = code[pc]
            back = self.next_back(pc)
            if back is not None and back < end \
                    and (pc not in self.repeats or back >= self.repeats[pc]):
                self.promote_all()
                pc = self.while_statement(pc, end)
            elif pc in self.repeats and pc not in self.active and self.repeats[pc] <= end:
                self.promote_all()
                pc = self.repeat_statement(pc, end)
            elif i.op in _TESTS:
                pc = self.conditional(pc, end)
            elif i.op == "FORPREP":
                pc = self.numeric_for(i)
            elif i.op == "JMP":
                target = pc + 1 + i.sbx
                if code[target].op == "TFORLOOP" and code[target].a in self.pending:
                    pc = self.generic_for(i, target)
                elif self.loop_exit is not None and self.same_place(target, self.loop_exit):
                    self.promote_all()
                    self.emit(Break())
                    pc += 1
                elif target > pc and target <= end:
                    # `if false then`
                    self.promote_all()
                    pc = self.if_statement(Const(False), pc + 1, target, end)
                else:
                    raise Fail(f"stray JMP at {pc}")
            else:
                self.skip_to = None
                n = self.step(i)
                pc = self.skip_to if self.skip_to is not None else n
                self.skip_to = None
        self.promote_all()

    def run(self):
        self.block(0, len(self.code))
        self.body = self.out
        return self


def walk_exprs(e):
    """Yield e and everything below it, without entering functions."""
    yield e
    if isinstance(e, Index):
        yield from walk_exprs(e.obj)
        yield from walk_exprs(e.key)
    elif isinstance(e, Method):
        yield from walk_exprs(e.obj)
    elif isinstance(e, Call):
        yield from walk_exprs(e.func)
        for a in e.args:
            yield from walk_exprs(a)
    elif isinstance(e, Binop):
        yield from walk_exprs(e.left)
        yield from walk_exprs(e.right)
    elif isinstance(e, Unop):
        yield from walk_exprs(e.operand)
    elif isinstance(e, Table):
        for a in e.array:
            yield from walk_exprs(a)
        for _, k, v in e.hash:
            yield from walk_exprs(k)
            yield from walk_exprs(v)


def stat_exprs(s):
    if isinstance(s, Local):
        return list(s.exprs)
    if isinstance(s, LocalFunction):
        return [s.func]
    if isinstance(s, Assign):
        return list(s.targets) + list(s.exprs)
    if isinstance(s, CallStat):
        return [s.call]
    if isinstance(s, Return):
        return list(s.exprs)
    if isinstance(s, (If, While)):
        return [s.cond]
    if isinstance(s, Repeat):
        return [s.cond]
    if isinstance(s, ForNum):
        return [s.start, s.stop, s.step]
    if isinstance(s, ForIn):
        return list(s.exprs)
    return []


def stat_blocks(s):
    if isinstance(s, If):
        return [s.then] + ([s.orelse] if s.orelse is not None else [])
    if isinstance(s, (While, Repeat, ForNum, ForIn, Do)):
        return [s.body]
    return []


def walk_stats(body):
    for s in body:
        yield s
        for b in stat_blocks(s):
            yield from walk_stats(b)


def decompile_fn(proto, upvars):
    fn = Fn(proto, upvars).run()
    for s in walk_stats(fn.body):
        for e in stat_exprs(s):
            for x in walk_exprs(e):
                if isinstance(x, Func) and x.fn is None:
                    x.fn = decompile_fn(x.proto, x.upvars)
    return fn


def decompile(chunk):
    return decompile_fn(chunk.main, [])


# --- output ------------------------------------------------------------------

def number(v):
    if v != v or v in (float("inf"), float("-inf")):
        raise Fail("number with no literal")
    if v == int(v) and abs(v) < 1e15:
        return ("-0" if v == 0 and str(v)[0] == "-" else str(int(v)))
    return repr(v)


def string(s):
    out = []
    n = len(s)
    for p, ch in enumerate(s):
        o = ord(ch)
        if ch == "\\":
            out.append("\\\\")
        elif ch == '"':
            out.append('\\"')
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\r":
            out.append("\\r")
        elif ch == "\t":
            out.append("\\t")
        elif o < 32 or o > 126:
            digit = p + 1 < n and s[p + 1].isdigit()
            out.append(f"\\{o:03d}" if digit else f"\\{o}")
        else:
            out.append(ch)
    return '"' + "".join(out) + '"'


class Renderer:
    def __init__(self, annotate=None):
        self.annotate = annotate

    def name(self, var):
        return var.name or f"v{var.uid}"

    def prefix(self, e, ind):
        t = self.expr(e, ind)
        if isinstance(e, (Global, Ref, Index, Call)) and not (isinstance(e, Call) and t.startswith("(")):
            return t
        return f"({t})"

    def key(self, obj, key, ind):
        if isinstance(key, Const) and isinstance(key.value, str) \
                and _IDENT.match(key.value) and key.value not in _KEYWORDS:
            return f"{obj}.{key.value}"
        return f"{obj}[{self.expr(key, ind)}]"

    def args(self, args, ind):
        out = []
        for n, a in enumerate(args):
            t = self.expr(a, ind)
            if n == len(args) - 1 and isinstance(a, (Call, Vararg)) and a.nres == 1:
                t = f"({t})"
            out.append(t)
        return ", ".join(out)

    def values(self, targets, exprs, ind):
        if len(targets) <= len(exprs):
            return ", ".join(self.expr(x, ind) for x in exprs)
        return self.args(exprs, ind)

    def expr(self, e, ind):
        if isinstance(e, Const):
            v = e.value
            if v is None:
                return "nil"
            if v is True:
                return "true"
            if v is False:
                return "false"
            if isinstance(v, float):
                return number(v)
            return string(v)
        if isinstance(e, Global):
            return e.name
        if isinstance(e, Ref):
            return self.name(e.var)
        if isinstance(e, Index):
            return self.key(self.prefix(e.obj, ind), e.key, ind)
        if isinstance(e, Method):
            return f"{self.prefix(e.obj, ind)}:{e.name}"
        if isinstance(e, Call):
            return f"{self.prefix(e.func, ind) if not isinstance(e.func, Method) else self.expr(e.func, ind)}({self.args(e.args, ind)})"
        if isinstance(e, Vararg):
            return "..."
        if isinstance(e, Binop):
            return self.binop(e, ind)
        if isinstance(e, Unop):
            t = self.expr(e.operand, ind)
            if e.operand.prec < 8 or isinstance(e.operand, Const) and e.op == "-" and e.operand.prec == 8:
                t = f"({t})"
            gap = " " if e.op == "not" else ""
            if e.op == "-" and t.startswith("-"):
                t = f"({t})"
            return f"{e.op}{gap}{t}"
        if isinstance(e, Table):
            return self.table(e, ind)
        if isinstance(e, Func):
            return self.function(e, ind)
        raise Fail(f"cannot print {type(e).__name__}")

    def binop(self, e, ind):
        p = e.prec
        left = self.expr(e.left, ind)
        right = self.expr(e.right, ind)
        right_assoc = e.op in ("..", "^")
        lp = e.left.prec
        rp = e.right.prec
        if lp < p or (lp == p and right_assoc):
            left = f"({left})"
        same_logic = e.op in ("and", "or") and isinstance(e.right, Binop) and e.right.op == e.op
        if rp < p or (rp == p and not right_assoc and not same_logic):
            right = f"({right})"
        return f"{left} {e.op} {right}"

    def table(self, t, ind):
        parts = []
        hashes = list(t.hash)
        pos = 0
        def flush(limit):
            while hashes and hashes[0][0] <= limit:
                _, k, v = hashes.pop(0)
                if isinstance(k, Const) and isinstance(k.value, str) \
                        and _IDENT.match(k.value) and k.value not in _KEYWORDS:
                    parts.append(f"{k.value} = {self.expr(v, ind + 1)}")
                else:
                    parts.append(f"[{self.expr(k, ind + 1)}] = {self.expr(v, ind + 1)}")
        for n, v in enumerate(t.array):
            flush(n)
            x = self.expr(v, ind + 1)
            if n == len(t.array) - 1 and isinstance(v, (Call, Vararg)) and v.nres == 1 and not hashes:
                x = f"({x})"
            parts.append(x)
        flush(10 ** 9)
        if not parts:
            return "{}"
        one = "{" + ", ".join(parts) + "}"
        if len(one) <= 90 and "\n" not in one:
            return one
        pad = "    " * (ind + 1)
        return "{\n" + "".join(f"{pad}{p},\n" for p in parts) + "    " * ind + "}"

    def params(self, fn):
        names = [self.name(v) for v in fn.params]
        if fn.vararg:
            names.append("...")
        return ", ".join(names)

    def function(self, e, ind):
        body = self.block(e.fn.body, ind + 1, e.fn)
        return "\n".join([f"function({self.params(e.fn)})", *body, "    " * ind + "end"])

    def block(self, body, ind, fn):
        out = []
        for n, s in enumerate(body):
            lines = self.stat(s, ind, fn)
            if ind == 0 and n and (isinstance(s, LocalFunction) or isinstance(s, Assign)
                                   and isinstance(s.exprs[0], Func)):
                out.append("")
            if isinstance(s, (Return, Break)) and n != len(body) - 1:
                lines = ["    " * ind + "do", *["    " + x for x in lines], "    " * ind + "end"]
            out.extend(lines)
        return out

    def stat(self, s, ind, fn):
        pad = "    " * ind
        out = []
        above, trail = self.annotate(s) if self.annotate else (None, None)
        if above:
            out.extend(pad + "-- " + x for x in above)
        mark = len(out)
        self.stat_lines(s, ind, fn, out, pad)
        if trail:
            if len(out) == mark + 1 and len(out[mark]) + len(trail) < 116:
                out[mark] += "  -- " + trail
            else:
                out.insert(mark, pad + "-- " + trail)
        return out

    def stat_lines(self, s, ind, fn, out, pad):
        if isinstance(s, Local):
            names = ", ".join(self.name(v) for v in s.vars)
            if s.exprs:
                out.append(f"{pad}local {names} = {self.values(s.vars, s.exprs, ind)}")
            else:
                out.append(f"{pad}local {names}")
        elif isinstance(s, LocalFunction):
            body = self.block(s.func.fn.body, ind + 1, s.func.fn)
            out += [f"{pad}local function {self.name(s.var)}({self.params(s.func.fn)})", *body, pad + "end"]
        elif isinstance(s, Assign):
            if len(s.targets) == 1 and isinstance(s.exprs[0], Func) and isinstance(s.targets[0], (Global, Index)) \
                    and self.dotted(s.targets[0]):
                f = s.exprs[0]
                body = self.block(f.fn.body, ind + 1, f.fn)
                out += [f"{pad}function {self.expr(s.targets[0], ind)}({self.params(f.fn)})", *body, pad + "end"]
            else:
                t = ", ".join(self.expr(x, ind) for x in s.targets)
                out.append(f"{pad}{t} = {self.values(s.targets, s.exprs, ind)}")
        elif isinstance(s, CallStat):
            out.append(pad + self.expr(s.call, ind))
        elif isinstance(s, Return):
            out.append(f"{pad}return {self.args(s.exprs, ind)}".rstrip())
        elif isinstance(s, Break):
            out.append(pad + "break")
        elif isinstance(s, If):
            self.if_lines(s, ind, fn, out, "if")
            out.append(pad + "end")
        elif isinstance(s, While):
            out.append(f"{pad}while {self.expr(s.cond, ind)} do")
            out += self.block(s.body, ind + 1, fn)
            out.append(pad + "end")
        elif isinstance(s, Repeat):
            out.append(pad + "repeat")
            out += self.block(s.body, ind + 1, fn)
            out.append(f"{pad}until {self.expr(s.cond, ind)}")
        elif isinstance(s, ForNum):
            r = f"{self.expr(s.start, ind)}, {self.expr(s.stop, ind)}"
            if not (isinstance(s.step, Const) and s.step.value == 1.0):
                r += f", {self.expr(s.step, ind)}"
            out.append(f"{pad}for {self.name(s.var)} = {r} do")
            out += self.block(s.body, ind + 1, fn)
            out.append(pad + "end")
        elif isinstance(s, ForIn):
            names = ", ".join(self.name(v) for v in s.vars)
            out.append(f"{pad}for {names} in {self.args(s.exprs, ind)} do")
            out += self.block(s.body, ind + 1, fn)
            out.append(pad + "end")
        elif isinstance(s, Do):
            out.append(pad + "do")
            out += self.block(s.body, ind + 1, fn)
            out.append(pad + "end")
        else:
            raise Fail(f"cannot print {type(s).__name__}")

    def dotted(self, t):
        if isinstance(t, Global):
            return True
        return isinstance(t, Index) and isinstance(t.key, Const) and isinstance(t.key.value, str) \
            and _IDENT.match(t.key.value) and t.key.value not in _KEYWORDS \
            and isinstance(t.obj, (Global, Ref, Index)) and (not isinstance(t.obj, Index) or self.dotted(t.obj))

    def if_lines(self, s, ind, fn, out, word):
        pad = "    " * ind
        out.append(f"{pad}{word} {self.expr(s.cond, ind)} then")
        out.extend(self.block(s.then, ind + 1, fn))
        if s.orelse is None:
            return
        if len(s.orelse) == 1 and isinstance(s.orelse[0], If):
            self.if_lines(s.orelse[0], ind, fn, out, "elseif")
            return
        out.append(pad + "else")
        out.extend(self.block(s.orelse, ind + 1, fn))


def render(fn, annotate=None):
    r = Renderer(annotate)
    return "\n".join(r.block(fn.body, 0, fn)) + "\n"
