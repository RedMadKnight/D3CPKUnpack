"""Lua 5.1 binary chunks: header check, whole-chunk parser and script slices."""

from __future__ import annotations

from dataclasses import dataclass, field

from .binary import ByteView, FormatError, build_order
from .sno import parse_header

LUA_SIGNATURE = b"\x1BLuaQ"
HEADER_SIZE = 12
MAX_NESTING = 200

# endianness flag, int size, size_t size, instruction size, number size, integral flag
_HEADERS = {
    "ps3": (0, 4, 4, 4, 8, 0),
    "switch": (1, 4, 8, 4, 8, 0),
}

# Position of the script pair per file type, (ps3, switch).
SCRIPT_PAIRS = {
    "pow": (0x190, 0x310),
    "scr": (0x20, 0x20),
    "con": (0x48, 0x48),
    "cnv": (0x1E8, 0xE8),
}

OPCODES = (
    "MOVE", "LOADK", "LOADBOOL", "LOADNIL", "GETUPVAL", "GETGLOBAL", "GETTABLE", "SETGLOBAL",
    "SETUPVAL", "SETTABLE", "NEWTABLE", "SELF", "ADD", "SUB", "MUL", "DIV", "MOD", "POW",
    "UNM", "NOT", "LEN", "CONCAT", "JMP", "EQ", "LT", "LE", "TEST", "TESTSET", "CALL",
    "TAILCALL", "RETURN", "FORLOOP", "FORPREP", "TFORLOOP", "SETLIST", "CLOSE", "CLOSURE",
    "VARARG",
)
_RETURN = OPCODES.index("RETURN")


def opcode_of(instruction: int) -> int:
    return instruction & 0x3F


def opcode_name(instruction: int) -> str:
    return OPCODES[opcode_of(instruction)]


@dataclass
class LocalVar:
    name: bytes | None
    start_pc: int
    end_pc: int


@dataclass
class LuaFunction:
    source: bytes | None
    line_defined: int
    last_line_defined: int
    upvalue_count: int
    param_count: int
    vararg_flags: int
    max_stack: int
    code: list[int]
    constants: list  # None, bool, float or bytes
    protos: list["LuaFunction"]
    line_info: list[int] = field(default_factory=list)
    locals: list[LocalVar] = field(default_factory=list)
    upvalue_names: list[bytes | None] = field(default_factory=list)

    @property
    def stripped(self) -> bool:
        return not (self.source or self.line_info or self.locals or self.upvalue_names)

    @property
    def line_span(self) -> tuple[int, int]:
        return self.line_defined, self.last_line_defined

    def walk(self):
        yield self
        for proto in self.protos:
            yield from proto.walk()


@dataclass
class LuaChunk:
    build: str
    main: LuaFunction
    consumed: int
    trailing_bytes: int

    @property
    def is_empty_script(self) -> bool:
        """An empty script is one function holding one RETURN."""
        main = self.main
        return (
            len(main.code) == 1
            and opcode_of(main.code[0]) == _RETURN
            and not main.protos
        )

    @property
    def stripped(self) -> bool:
        return all(f.stripped for f in self.main.walk())


def check_header(data: bytes, build: str | None = None) -> str:
    """Validate the 12-byte header and return the build. Values must match exactly."""
    if len(data) < HEADER_SIZE:
        raise FormatError("chunk shorter than its header", 0)
    if bytes(data[:5]) != LUA_SIGNATURE:
        raise FormatError("bad Lua signature", 0)
    if data[5] != 0:
        raise FormatError(f"Lua format {data[5]}, expected 0", 5)
    values = tuple(data[6:HEADER_SIZE])
    if build is not None:
        build_order(build)
        expected = _HEADERS[build]
        for i, (got, want) in enumerate(zip(values, expected)):
            if got != want:
                raise FormatError(f"chunk header byte {6 + i}: {got}, {build} needs {want}", 6 + i)
        return build
    for name, expected in _HEADERS.items():
        if values == expected:
            return name
    raise FormatError(f"chunk header {values} matches no build", 6)


class _ChunkReader:
    def __init__(self, data: bytes, build: str):
        self.view = ByteView(data, build_order(build))
        self.pos = HEADER_SIZE
        self.size_t = "I" if build == "ps3" else "Q"

    def byte(self) -> int:
        value = self.view.u8(self.pos, "chunk byte")
        self.pos += 1
        return value

    def int(self) -> int:
        value = self.view.i32(self.pos, "chunk int")
        self.pos += 4
        return value

    def count(self, width: int, what: str) -> int:
        value = self.int()
        if value < 0:
            raise FormatError(f"negative {what} count", self.pos - 4)
        self.view.need(self.pos, value * width, what)
        return value

    def string(self) -> bytes | None:
        length = self.view.unpack(self.size_t, self.pos, "string length")[0]
        self.pos += 4 if self.size_t == "I" else 8
        if length == 0:
            return None
        raw = self.view.take(self.pos, length, "string")
        if raw[-1] != 0:
            raise FormatError("string lacks its terminator", self.pos + length - 1)
        self.pos += length
        return raw[:-1]

    def function(self, depth: int) -> LuaFunction:
        if depth > MAX_NESTING:
            raise FormatError("functions nested too deeply", self.pos)
        source = self.string()
        line_defined = self.int()
        last_line_defined = self.int()
        upvalues = self.byte()
        params = self.byte()
        vararg = self.byte()
        max_stack = self.byte()

        count = self.count(4, "code")
        code = list(self.view.unpack(f"{count}I", self.pos))
        for i, word in enumerate(code):
            if opcode_of(word) >= len(OPCODES):
                raise FormatError(f"opcode {opcode_of(word)} outside Lua 5.1", self.pos + i * 4)
        self.pos += count * 4

        constants = []
        for _ in range(self.count(1, "constants")):
            kind = self.byte()
            if kind == 0:
                constants.append(None)
            elif kind == 1:
                constants.append(self.byte() != 0)
            elif kind == 3:
                constants.append(self.view.unpack("d", self.pos, "number")[0])
                self.pos += 8
            elif kind == 4:
                constants.append(self.string())
            else:
                raise FormatError(f"constant type {kind}", self.pos - 1)

        protos = [
            self.function(depth + 1) for _ in range(self.count(1, "prototypes"))
        ]

        count = self.count(4, "line info")
        line_info = list(self.view.unpack(f"{count}i", self.pos))
        self.pos += count * 4
        local_vars = []
        for _ in range(self.count(1, "locals")):
            name = self.string()
            start_pc = self.int()
            local_vars.append(LocalVar(name, start_pc, self.int()))
        upvalue_names = [self.string() for _ in range(self.count(1, "upvalue names"))]
        return LuaFunction(
            source, line_defined, last_line_defined, upvalues, params, vararg, max_stack,
            code, constants, protos, line_info, local_vars, upvalue_names,
        )


def parse_chunk(data: bytes, build: str | None = None) -> LuaChunk:
    """Parse a whole chunk. ``trailing_bytes`` counts what follows the main function."""
    resolved = check_header(data, build)
    reader = _ChunkReader(data, resolved)
    main = reader.function(0)
    return LuaChunk(resolved, main, reader.pos, len(data) - reader.pos)


def script_slice(data: bytes, file_type: str, build: str | None = None) -> bytes:
    """Return a file's script chunk, or empty bytes for an empty pair."""
    extension = file_type.lstrip(".")
    if extension not in SCRIPT_PAIRS:
        raise FormatError(f"no script pair known for .{extension}")
    resolved = build or parse_header(data).build
    view = ByteView(data, build_order(resolved))
    position = SCRIPT_PAIRS[extension][0 if resolved == "ps3" else 1]
    pair = view.pair(position, "script pair")
    chunk = view.pair_data(pair, position, "script pair")
    if chunk and not chunk.startswith(LUA_SIGNATURE):
        raise FormatError("script pair does not address a Lua chunk", pair.file_offset)
    return chunk
