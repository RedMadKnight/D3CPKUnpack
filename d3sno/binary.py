"""Bounds-checked binary access shared by every reader."""

from __future__ import annotations

import struct
from dataclasses import dataclass

BUILDS = ("ps3", "switch")
_ORDERS = {"ps3": ">", "switch": "<"}


class FormatError(Exception):
    """Malformed input. ``offset`` is the byte position of the problem."""

    def __init__(self, message: str, offset: int = 0):
        super().__init__(f"{message} (offset 0x{offset:X})")
        self.message = message
        self.offset = offset


def build_order(build: str) -> str:
    """Return the struct byte-order prefix of a build name."""
    try:
        return _ORDERS[build]
    except KeyError:
        raise FormatError(f"unknown build {build!r}") from None


def order_build(order: str) -> str:
    """Return the build name of a struct byte-order prefix."""
    return "ps3" if order == ">" else "switch"


@dataclass(frozen=True)
class Pair:
    """``{u32 offset, u32 size}``. The data is at file position offset + 0x10."""

    offset: int
    size: int

    @property
    def file_offset(self) -> int:
        return self.offset + 0x10

    @property
    def is_empty(self) -> bool:
        return self.offset == 0 and self.size == 0


class ByteView:
    """Reads values at explicit positions and never past the end of the data."""

    __slots__ = ("data", "order")

    def __init__(self, data: bytes, order: str):
        self.data = data
        self.order = order

    def need(self, pos: int, size: int, what: str = "read") -> None:
        if pos < 0 or size < 0 or pos + size > len(self.data):
            raise FormatError(f"{what}: {size} bytes run past the end", pos)

    def unpack(self, fmt: str, pos: int, what: str = "read") -> tuple:
        full = self.order + fmt
        self.need(pos, struct.calcsize(full), what)
        return struct.unpack_from(full, self.data, pos)

    def u8(self, pos: int, what: str = "read") -> int:
        self.need(pos, 1, what)
        return self.data[pos]

    def u16(self, pos: int, what: str = "read") -> int:
        return self.unpack("H", pos, what)[0]

    def u32(self, pos: int, what: str = "read") -> int:
        return self.unpack("I", pos, what)[0]

    def i32(self, pos: int, what: str = "read") -> int:
        return self.unpack("i", pos, what)[0]

    def u64(self, pos: int, what: str = "read") -> int:
        return self.unpack("Q", pos, what)[0]

    def f32(self, pos: int, what: str = "read") -> float:
        return self.unpack("f", pos, what)[0]

    def f32s(self, pos: int, count: int, what: str = "read") -> tuple[float, ...]:
        return self.unpack(f"{count}f", pos, what)

    def take(self, pos: int, size: int, what: str = "read") -> bytes:
        self.need(pos, size, what)
        return bytes(self.data[pos:pos + size])

    def cstring(self, pos: int, limit: int, what: str = "string") -> str:
        """Read latin-1 text up to a NUL or up to ``limit`` bytes, whichever is first."""
        raw = self.take(pos, limit, what)
        end = raw.find(0)
        return raw[: end if end >= 0 else limit].decode("latin-1")

    def pair(self, pos: int, what: str = "pair") -> Pair:
        return Pair(*self.unpack("II", pos, what))

    def pair_data(self, pair: Pair, pos: int = 0, what: str = "pair") -> bytes:
        """Return a pair's bytes. ``pos`` is where the pair itself sits, for the error."""
        if pair.size == 0:
            if pair.offset != 0 and pair.file_offset > len(self.data):
                raise FormatError(f"{what}: offset past the end", pos)
            return b""
        if pair.file_offset + pair.size > len(self.data):
            raise FormatError(
                f"{what}: data at 0x{pair.file_offset:X} size 0x{pair.size:X} "
                f"runs past the end of a 0x{len(self.data):X} byte file",
                pos,
            )
        return bytes(self.data[pair.file_offset:pair.file_offset + pair.size])


_F32 = struct.Struct("<f")


def to_f32(value: float) -> float:
    """Round to single precision. Overflow gives infinity, as a C float would."""
    try:
        return _F32.unpack(_F32.pack(value))[0]
    except OverflowError:
        return float("inf") if value > 0 else float("-inf")


def bit_length(value: int) -> int:
    return value.bit_length()


class BitReader:
    """Reads MSB-first bit fields. Earlier bits are more significant."""

    __slots__ = ("data", "bit", "base")

    def __init__(self, data: bytes, pos: int):
        self.data = data
        self.base = pos
        self.bit = pos * 8

    def read(self, width: int) -> int:
        if width == 0:
            return 0
        start = self.bit >> 3
        stop_bit = self.bit + width
        stop = (stop_bit + 7) >> 3
        if stop > len(self.data):
            raise FormatError("bit field runs past the end", start)
        value = int.from_bytes(self.data[start:stop], "big")
        value >>= stop * 8 - stop_bit
        self.bit = stop_bit
        return value & ((1 << width) - 1)

    def align(self) -> int:
        """Skip to the next byte boundary and return the byte position."""
        self.bit = (self.bit + 7) & ~7
        return self.bit >> 3


class BitWriter:
    """Writes MSB-first bit fields, the inverse of :class:`BitReader`."""

    __slots__ = ("out", "acc", "count")

    def __init__(self) -> None:
        self.out = bytearray()
        self.acc = 0
        self.count = 0

    def write(self, value: int, width: int) -> None:
        if value < 0 or value >> width:
            raise FormatError(f"value {value} does not fit in {width} bits")
        self.acc = (self.acc << width) | value
        self.count += width
        while self.count >= 8:
            self.count -= 8
            self.out.append((self.acc >> self.count) & 0xFF)
        self.acc &= (1 << self.count) - 1

    def finish(self) -> bytes:
        if self.count:
            self.out.append((self.acc << (8 - self.count)) & 0xFF)
            self.acc = 0
            self.count = 0
        return bytes(self.out)
