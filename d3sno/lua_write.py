"""Lua 5.1 chunk writer, code-shape checker and re-targeting (``30`` W15-A8, W15-A9).

``dump_chunk`` serialises a function tree (``luachunk.LuaFunction``, or a ``LuaChunk``) with a
build's header, byte order and size_t width; ``luachunk.parse_chunk`` reads it back, so
``dump_chunk(parse_chunk(c)) == c`` for every chunk that has nothing after its main function.
``retarget_chunk`` ports a chunk to another build by re-serialisation (W15-A9).
``shape_problems`` lists where a tree departs from the standard Lua 5.1 code shape that stored
chunks have (W15-A9) and ``form_problems`` where it departs from the stored chunk form (W15-A8).
``defined_globals`` gives the global functions a chunk's main function defines (W15-A10).

No Lua compiler is included: byte identity with a reference compiler is untested (``30``).
"""

from __future__ import annotations

import struct

from .binary import FormatError, build_order
from .luachunk import (
    LUA_SIGNATURE, OPCODES, LuaChunk, LuaFunction, _HEADERS, check_header, opcode_of, parse_chunk,
)

__all__ = [
    "dump_chunk", "retarget_chunk", "shape_problems", "form_problems", "check_shape",
    "defined_globals", "encode_abc", "encode_abx", "encode_asbx", "decode",
]

_OP = {name: i for i, name in enumerate(OPCODES)}
MAX_SBX = (1 << 17) - 1  # 131071
BITRK = 1 << 8

# Opcodes whose B / C operands are RK (a register, or a constant at index - 256).
_RK_B = frozenset(_OP[n] for n in ("SETTABLE", "ADD", "SUB", "MUL", "DIV", "MOD", "POW",
                                   "EQ", "LT", "LE"))
_RK_C = frozenset(_OP[n] for n in ("GETTABLE", "SETTABLE", "SELF", "ADD", "SUB", "MUL", "DIV",
                                   "MOD", "POW", "EQ", "LT", "LE"))
_BX_CONSTANT = frozenset(_OP[n] for n in ("LOADK", "GETGLOBAL", "SETGLOBAL"))
_TESTS = frozenset(_OP[n] for n in ("EQ", "LT", "LE", "TEST", "TESTSET", "TFORLOOP"))
# Opcodes that do not write register A (so a CLOSURE held there survives them).
_KEEPS_A = frozenset(_OP[n] for n in (
    "SETGLOBAL", "SETUPVAL", "SETTABLE", "JMP", "EQ", "LT", "LE", "TEST", "RETURN", "SETLIST",
    "CLOSE", "TAILCALL",
))


# --- Instructions ----------------------------------------------------------------------------

def encode_abc(op: str | int, a: int, b: int = 0, c: int = 0) -> int:
    code = _OP[op] if isinstance(op, str) else op
    return code | a << 6 | c << 14 | b << 23


def encode_abx(op: str | int, a: int, bx: int) -> int:
    code = _OP[op] if isinstance(op, str) else op
    return code | a << 6 | bx << 14


def encode_asbx(op: str | int, a: int, sbx: int) -> int:
    return encode_abx(op, a, sbx + MAX_SBX)


def decode(word: int) -> tuple[int, int, int, int, int]:
    """(opcode, A, B, C, Bx) of an instruction."""
    return word & 0x3F, (word >> 6) & 0xFF, (word >> 23) & 0x1FF, (word >> 14) & 0x1FF, word >> 14


# --- Writer (W15-A8) ---------------------------------------------------------------------------

class _Writer:
    def __init__(self, build: str):
        build_order(build)
        if build not in _HEADERS:
            raise FormatError(f"no Lua header for build {build!r}")
        self.build = build
        self.order = build_order(build)
        self.size_t = "Q" if _HEADERS[build][2] == 8 else "I"
        self.out = bytearray(LUA_SIGNATURE + b"\0" + bytes(_HEADERS[build]))

    def int(self, value: int) -> None:
        self.out += struct.pack(self.order + "i", value)

    def byte(self, value: int, what: str) -> None:
        if not 0 <= value <= 0xFF:
            raise FormatError(f"{what} {value} does not fit in a byte")
        self.out.append(value)

    def string(self, value: bytes | str | None) -> None:
        if value is None:
            self.out += struct.pack(self.order + self.size_t, 0)
            return
        raw = value.encode("latin-1") if isinstance(value, str) else bytes(value)
        self.out += struct.pack(self.order + self.size_t, len(raw) + 1) + raw + b"\0"

    def function(self, f: LuaFunction, depth: int = 0) -> None:
        if depth > 200:
            raise FormatError("functions nested too deeply")
        self.string(f.source)
        self.int(f.line_defined)
        self.int(f.last_line_defined)
        self.byte(f.upvalue_count, "upvalue count")
        self.byte(f.param_count, "parameter count")
        self.byte(f.vararg_flags, "vararg flags")
        self.byte(f.max_stack, "max stack")
        self.int(len(f.code))
        for word in f.code:
            if opcode_of(word) >= len(OPCODES):
                raise FormatError(f"opcode {opcode_of(word)} outside Lua 5.1")
            self.out += struct.pack(self.order + "I", word)
        self.int(len(f.constants))
        for value in f.constants:
            if value is None:
                self.out.append(0)
            elif isinstance(value, bool):
                self.out += bytes((1, int(value)))
            elif isinstance(value, (int, float)):
                self.out.append(3)
                self.out += struct.pack(self.order + "d", float(value))
            elif isinstance(value, (bytes, bytearray, str)):
                self.out.append(4)
                self.string(value)
            else:
                raise FormatError(f"constant {value!r} has no Lua type")
        self.int(len(f.protos))
        for proto in f.protos:
            self.function(proto, depth + 1)
        self.int(len(f.line_info))
        for line in f.line_info:
            self.int(line)
        self.int(len(f.locals))
        for local in f.locals:
            self.string(local.name)
            self.int(local.start_pc)
            self.int(local.end_pc)
        self.int(len(f.upvalue_names))
        for name in f.upvalue_names:
            self.string(name)


def _main(function_tree) -> LuaFunction:
    if isinstance(function_tree, LuaChunk):
        return function_tree.main
    if isinstance(function_tree, LuaFunction):
        return function_tree
    raise FormatError(f"not a Lua function tree: {type(function_tree).__name__}")


def dump_chunk(function_tree, build: str) -> bytes:
    """W15-A8: serialise a function tree (``LuaFunction`` or ``LuaChunk``) as ``build``'s chunk:
    header, then the main function; nothing follows it."""
    writer = _Writer(build)
    writer.function(_main(function_tree))
    return bytes(writer.out)


def retarget_chunk(chunk: bytes, from_build: str, to_build: str) -> bytes:
    """W15-A9: the same function tree serialised with another build's header, byte order and
    size_t width. Bytes after the main function are refused (W15-A8: nothing follows it)."""
    parsed = parse_chunk(chunk, from_build)
    if parsed.trailing_bytes:
        raise FormatError(f"{parsed.trailing_bytes} bytes follow the main function", parsed.consumed)
    return dump_chunk(parsed.main, to_build)


# --- Form and shape (W15-A8, W15-A9) -------------------------------------------------------------

def form_problems(chunk) -> list[str]:
    """W15-A8: departures from the stored chunk form. ``chunk`` is chunk bytes or a ``LuaChunk``:
    stripped functions (no source, line info, locals or upvalue names; line spans kept), main
    function with 0 parameters, vararg 2 and lines 0/0, nothing after it."""
    parsed = parse_chunk(chunk) if isinstance(chunk, (bytes, bytearray)) else chunk
    problems = []
    if isinstance(parsed, LuaChunk) and parsed.trailing_bytes:
        problems.append(f"{parsed.trailing_bytes} bytes follow the main function")
    main = _main(parsed)
    if main.param_count:
        problems.append(f"main function has {main.param_count} parameters")
    if main.vararg_flags != 2:
        problems.append(f"main function vararg flags {main.vararg_flags}, expected 2")
    if main.line_span != (0, 0):
        problems.append(f"main function lines {main.line_span}, expected (0, 0)")
    for path, f in _functions(main):
        if not f.stripped:
            problems.append(f"{path}: not stripped")
    return problems


def _functions(main: LuaFunction, path: str = "main"):
    yield path, main
    for i, proto in enumerate(main.protos):
        yield from _functions(proto, f"{path}.{i}")


def _constant_key(value):
    return (type(value).__name__, value)


def _used_constants(f: LuaFunction) -> set[int]:
    used = set()
    for word in f.code:
        op, _, b, c, bx = decode(word)
        if op in _BX_CONSTANT:
            used.add(bx)
        if op in _RK_B and b & BITRK:
            used.add(b & ~BITRK)
        if op in _RK_C and c & BITRK:
            used.add(c & ~BITRK)
    return used


def _function_shape(f: LuaFunction, path: str, is_main: bool) -> list[str]:
    problems = []
    code = f.code
    if not code or opcode_of(code[-1]) != _OP["RETURN"] or decode(code[-1])[2] != 1:
        problems.append(f"{path}: last instruction is not RETURN with B = 1")
    keys = [_constant_key(v) for v in f.constants]
    if len(set(keys)) != len(keys):
        problems.append(f"{path}: constants repeat")
    used = _used_constants(f)
    unused = [i for i in range(len(f.constants)) if i not in used]
    if unused:
        problems.append(f"{path}: constants {unused} are not used")
    if any(i >= len(f.constants) for i in used):
        problems.append(f"{path}: an instruction names a constant past the list")
    pc = 0
    has_vararg = False
    while pc < len(code):
        op, a, b, c, bx = decode(code[pc])
        if op == _OP["VARARG"]:
            has_vararg = True
        if op == _OP["CLOSURE"]:
            if bx >= len(f.protos):
                problems.append(f"{path}: CLOSURE at pc {pc} names prototype {bx} of {len(f.protos)}")
            else:
                for k in range(f.protos[bx].upvalue_count):
                    at = pc + 1 + k
                    if at >= len(code) or decode(code[at])[0] not in (_OP["MOVE"], _OP["GETUPVAL"]):
                        problems.append(f"{path}: CLOSURE at pc {pc} lacks upvalue instruction {k}")
                        break
                pc += f.protos[bx].upvalue_count
        elif op in _TESTS:
            if pc + 1 >= len(code) or opcode_of(code[pc + 1]) != _OP["JMP"]:
                problems.append(f"{path}: {OPCODES[op]} at pc {pc} is not followed by a jump")
        pc += 1
    if code and opcode_of(code[0]) == _OP["LOADNIL"] and decode(code[0])[1] >= f.param_count:
        problems.append(f"{path}: nil load at pc 0")
    lines = [p.line_defined for p in f.protos]
    if lines != sorted(lines):
        problems.append(f"{path}: prototypes are not in source order")
    if f.max_stack < 2:
        problems.append(f"{path}: max stack {f.max_stack} < 2")
    if not is_main:
        if f.vararg_flags == 7 and has_vararg:
            problems.append(f"{path}: vararg 7 but the function uses ...")
        elif f.vararg_flags == 3 and not has_vararg:
            problems.append(f"{path}: vararg 3 but the function does not use ...")
        elif f.vararg_flags not in (0, 3, 7):
            problems.append(f"{path}: vararg flags {f.vararg_flags}")
        elif f.vararg_flags == 0 and has_vararg:
            problems.append(f"{path}: VARARG in a function without ...")
    return problems


def shape_problems(function_tree) -> list[str]:
    """W15-A9: departures from standard Lua 5.1 code generation, in every function: last
    instruction RETURN with B = 1; constants unique and all used; upvalue pseudo-instructions
    after CLOSURE; tests followed by a jump; no nil load at pc 0 (of a non-parameter register);
    prototypes in source order; max stack >= 2; vararg 0, or 7 for a function with ``...`` that
    does not use it (3, the standard flags of one that does, is also accepted)."""
    main = _main(function_tree)
    problems = []
    for path, f in _functions(main):
        problems += _function_shape(f, path, f is main)
    return problems


def check_shape(function_tree) -> None:
    """Raise FormatError on the first ``shape_problems`` entry."""
    problems = shape_problems(function_tree)
    if problems:
        raise FormatError(problems[0])


# --- Globals defined by the main function (W15-A10) ------------------------------------------------

def defined_globals(chunk, build: str | None = None) -> set[str]:
    """The global names the main function assigns a closure to (``function Name() ... end`` at
    top level). ``chunk`` is chunk bytes, a ``LuaChunk`` or a ``LuaFunction``; empty bytes
    define nothing."""
    if isinstance(chunk, (bytes, bytearray)):
        if not chunk:
            return set()
        chunk = parse_chunk(bytes(chunk), build)
    main = _main(chunk)
    holds: dict[int, bool] = {}
    names: set[str] = set()
    code = main.code
    pc = 0
    while pc < len(code):
        op, a, b, _, bx = decode(code[pc])
        if op == _OP["CLOSURE"]:
            holds[a] = True
            if bx < len(main.protos):
                pc += main.protos[bx].upvalue_count  # skip the pseudo-instructions
        elif op == _OP["MOVE"]:
            holds[a] = holds.get(b, False)
        elif op == _OP["SETGLOBAL"]:
            if holds.get(a) and bx < len(main.constants) and isinstance(main.constants[bx], bytes):
                names.add(main.constants[bx].decode("latin-1"))
        elif op == _OP["LOADNIL"]:
            for r in range(a, b + 1):
                holds[r] = False
        elif op not in _KEEPS_A:
            holds[a] = False
        pc += 1
    return names


def chunk_header_build(chunk: bytes) -> str:
    """The build whose header a chunk carries (x360 and ps3 share one; ps3 is returned)."""
    return check_header(chunk)
