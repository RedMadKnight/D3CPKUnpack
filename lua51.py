#!/usr/bin/env python3
"""
Reader and disassembler for Lua 5.1 bytecode chunks.

Diablo III embeds compiled scripts in several SNO file types. They are standard
Lua 5.1 chunks written big-endian on the v6 console archives, so this module
reads the format as the reference compiler defines it and honours whatever
sizes and byte order the chunk header declares.

    chunk = lua51.parse_chunk(blob)
    print("\\n".join(lua51.disassemble(chunk.main)))
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Iterator, Optional, Union

SIGNATURE = b"\x1bLua"
VERSION = 0x51
Constant = Union[None, bool, float, str]

OPCODES = (
    "MOVE", "LOADK", "LOADBOOL", "LOADNIL", "GETUPVAL", "GETGLOBAL", "GETTABLE",
    "SETGLOBAL", "SETUPVAL", "SETTABLE", "NEWTABLE", "SELF", "ADD", "SUB", "MUL",
    "DIV", "MOD", "POW", "UNM", "NOT", "LEN", "CONCAT", "JMP", "EQ", "LT", "LE",
    "TEST", "TESTSET", "CALL", "TAILCALL", "RETURN", "FORLOOP", "FORPREP",
    "TFORLOOP", "SETLIST", "CLOSE", "CLOSURE", "VARARG",
)
_ABX_OPCODES = {"LOADK", "GETGLOBAL", "SETGLOBAL", "CLOSURE"}
_ASBX_OPCODES = {"JMP", "FORLOOP", "FORPREP"}
_MAXARG_SBX = (1 << 18) // 2 - 1
_RK_CONSTANT = 256


@dataclass(frozen=True)
class Instruction:
    pc: int
    op: str
    a: int
    b: int
    c: int
    bx: int

    @property
    def sbx(self) -> int:
        return self.bx - _MAXARG_SBX

    @classmethod
    def decode(cls, pc: int, word: int) -> "Instruction":
        code = word & 0x3F
        if code >= len(OPCODES):
            raise ValueError(f"unknown opcode {code} at pc {pc}")
        return cls(
            pc=pc, op=OPCODES[code], a=(word >> 6) & 0xFF,
            c=(word >> 14) & 0x1FF, b=(word >> 23) & 0x1FF, bx=word >> 14,
        )


@dataclass
class LuaFunction:
    source: Optional[str]
    line_defined: int
    last_line_defined: int
    num_upvalues: int
    num_params: int
    is_vararg: int
    max_stack: int
    code: list[Instruction] = field(default_factory=list)
    constants: list[Constant] = field(default_factory=list)
    protos: list["LuaFunction"] = field(default_factory=list)
    line_info: list[int] = field(default_factory=list)
    locals: list[tuple[str, int, int]] = field(default_factory=list)
    upvalues: list[str] = field(default_factory=list)

    def walk(self) -> Iterator["LuaFunction"]:
        yield self
        for proto in self.protos:
            yield from proto.walk()


@dataclass
class LuaChunk:
    main: LuaFunction
    size: int            # bytes the chunk occupies in the buffer
    endian: str

    @property
    def stripped(self) -> bool:
        return not any(f.line_info or f.locals for f in self.main.walk())


class _Reader:
    def __init__(self, data: bytes, pos: int):
        self.data = data
        self.pos = pos
        if data[pos:pos + 4] != SIGNATURE:
            raise ValueError("not a Lua chunk: bad signature")
        if len(data) < pos + 12:
            raise ValueError("truncated Lua chunk header")
        version, fmt, little, self.int_size, self.size_t_size, instr_size, \
            self.number_size, integral = data[pos + 4:pos + 12]
        if version != VERSION or fmt != 0:
            raise ValueError(f"unsupported Lua chunk: version {version:#x}, format {fmt}")
        if instr_size != 4 or integral or self.number_size != 8:
            raise ValueError("unsupported Lua number or instruction layout")
        if self.int_size not in (4, 8) or self.size_t_size not in (4, 8):
            raise ValueError("unsupported Lua integer size")
        self.endian = "<" if little else ">"
        self.pos = pos + 12

    def take(self, n: int) -> bytes:
        if self.pos + n > len(self.data):
            raise ValueError(f"truncated Lua chunk at {self.pos:#x}")
        out = self.data[self.pos:self.pos + n]
        self.pos += n
        return out

    def byte(self) -> int:
        return self.take(1)[0]

    def _unsigned(self, size: int) -> int:
        code = "I" if size == 4 else "Q"
        return struct.unpack(self.endian + code, self.take(size))[0]

    def int(self) -> int:
        return self._unsigned(self.int_size)

    def string(self) -> Optional[str]:
        size = self._unsigned(self.size_t_size)
        if size == 0:
            return None
        return self.take(size)[:-1].decode("latin-1")

    def function(self, parent_source: Optional[str]) -> LuaFunction:
        source = self.string() or parent_source
        fn = LuaFunction(
            source=source, line_defined=self.int(), last_line_defined=self.int(),
            num_upvalues=self.byte(), num_params=self.byte(),
            is_vararg=self.byte(), max_stack=self.byte(),
        )
        fn.code = [
            Instruction.decode(pc, self._unsigned(4))
            for pc in range(self.int())
        ]
        fn.constants = [self.constant() for _ in range(self.int())]
        fn.protos = [self.function(source) for _ in range(self.int())]
        fn.line_info = [self.int() for _ in range(self.int())]
        fn.locals = [
            (self.string() or "", self.int(), self.int())
            for _ in range(self.int())
        ]
        fn.upvalues = [self.string() or "" for _ in range(self.int())]
        return fn

    def constant(self) -> Constant:
        kind = self.byte()
        if kind == 0:
            return None
        if kind == 1:
            return self.byte() != 0
        if kind == 3:
            return struct.unpack(self.endian + "d", self.take(8))[0]
        if kind == 4:
            return self.string()
        raise ValueError(f"unknown Lua constant type {kind}")


def parse_chunk(data: bytes, offset: int = 0) -> LuaChunk:
    """Parse the chunk at `offset`. Raises ValueError if it is not valid."""
    reader = _Reader(data, offset)
    main = reader.function(None)
    return LuaChunk(main=main, size=reader.pos - offset, endian=reader.endian)


def find_chunks(data: bytes) -> Iterator[tuple[int, LuaChunk]]:
    """Yield (offset, chunk) for each complete Lua 5.1 chunk in a buffer."""
    pos = data.find(SIGNATURE)
    while pos != -1:
        try:
            chunk = parse_chunk(data, pos)
        except (ValueError, struct.error):
            pos = data.find(SIGNATURE, pos + 1)
            continue
        yield pos, chunk
        pos = data.find(SIGNATURE, pos + chunk.size)


def format_constant(value: Constant) -> str:
    if value is None:
        return "nil"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() else repr(value)
    escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    return f'"{escaped}"'


def _rk(fn: LuaFunction, value: int) -> str:
    if value >= _RK_CONSTANT:
        return format_constant(fn.constants[value - _RK_CONSTANT])
    return f"R{value}"


def _comment(fn: LuaFunction, i: Instruction) -> str:
    if i.op in ("LOADK", "GETGLOBAL", "SETGLOBAL"):
        return format_constant(fn.constants[i.bx])
    if i.op in _ASBX_OPCODES:
        return f"to {i.pc + 1 + i.sbx}"
    if i.op == "CLOSURE":
        return f"function {i.bx}"
    if i.op in ("GETTABLE", "SELF"):
        return _rk(fn, i.c)
    if i.op in ("SETTABLE", "ADD", "SUB", "MUL", "DIV", "MOD", "POW",
                "EQ", "LT", "LE"):
        return f"{_rk(fn, i.b)} {_rk(fn, i.c)}"
    return ""


def disassemble(fn: LuaFunction, name: str = "main") -> list[str]:
    """Listing of a function and its nested functions, like `luac -l`."""
    lines = [
        f"function {name}: {fn.num_params} params, {fn.num_upvalues} upvalues, "
        f"{fn.max_stack} slots, {len(fn.code)} instructions, "
        f"{len(fn.constants)} constants, {len(fn.protos)} functions"
    ]
    for i in fn.code:
        if i.op in _ABX_OPCODES:
            args = f"{i.a} {i.bx}"
        elif i.op in _ASBX_OPCODES:
            args = f"{i.a} {i.sbx}"
        else:
            args = f"{i.a} {i.b} {i.c}"
        comment = _comment(fn, i)
        text = f"  {i.pc:5d}  {i.op:<10} {args}"
        lines.append(f"{text:<36} ; {comment}" if comment else text)
    for n, proto in enumerate(fn.protos):
        lines.append("")
        lines.extend(disassemble(proto, f"{name}.{n}"))
    return lines
