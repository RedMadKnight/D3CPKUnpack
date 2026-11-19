"""``.sbk`` sound banks and ``.snd`` sounds."""

from __future__ import annotations

import struct
from dataclasses import dataclass

from .binary import ByBuild, ByteView, FormatError, build_order
from .sno import SnoHeader, parse_header, resolve_build

_ROW = ByBuild({"switch": 16, "ps3": 40})
_STORED_AT = ByBuild({"switch": 0x0C, "ps3": 0x20})
_SAMPLE_COUNT_AT = ByBuild({"switch": 0x30, "ps3": 0x24})
_PERMUTATION = ByBuild({"switch": (0x54, 0x5C, 64), "ps3": (0x50, 0x58, 44)})  # pair, count, size


@dataclass(frozen=True)
class SampleRow:
    index: int
    hash: int
    stored_size: int
    raw: bytes

    def child_name(self, bank: str, extension: str = "fsb") -> str:
        """The archive name of this row's sound file, which may not exist."""
        return f"SoundBank\\{bank}\\{self.index:04d}.{extension}"


@dataclass
class SoundBank:
    header: SnoHeader
    build: str
    rows: list[SampleRow]

    def find(self, sample_hash: int) -> SampleRow | None:
        return next((row for row in self.rows if row.hash == sample_hash), None)


def parse_sound_bank(data: bytes, build: str | None = None, strict: bool = False) -> SoundBank:
    header = parse_header(data, build)
    resolved = resolve_build("sbk", header, build, strict)
    view = ByteView(data, header.order)
    count = view.i32(_SAMPLE_COUNT_AT[resolved], "sample count")
    pair = view.pair(0x1C, "sample table pair")
    size = _ROW[resolved]
    if count < 0 or pair.size != count * size:
        raise FormatError(f"sample table of {pair.size} bytes for {count}", 0x1C)
    raw = view.pair_data(pair, 0x1C, "sample table")
    rows = []
    for i in range(count):
        row = raw[i * size:(i + 1) * size]
        r = ByteView(row, view.order)
        rows.append(SampleRow(i, r.u32(0), r.u32(_STORED_AT[resolved]), row))
    return SoundBank(header, resolved, rows)


@dataclass(frozen=True)
class SoundPermutation:
    bank: int
    sample_hash: int
    weight: int
    raw: bytes


@dataclass
class SoundFile:
    header: SnoHeader
    build: str
    permutations: list[SoundPermutation]


def parse_sound(data: bytes, build: str | None = None) -> SoundFile:
    header = parse_header(data, build)
    resolved = resolve_build("snd", header, build)
    view = ByteView(data, header.order)
    pair_at, count_at, size = _PERMUTATION[resolved]
    count = view.i32(count_at, "permutation count")
    pair = view.pair(pair_at, "permutations pair")
    if count < 0 or pair.size != count * size:
        raise FormatError(f"permutations of {pair.size} bytes for {count}", pair_at)
    raw = view.pair_data(pair, pair_at, "permutations")
    permutations = []
    for i in range(count):
        row = raw[i * size:(i + 1) * size]
        r = ByteView(row, view.order)
        permutations.append(SoundPermutation(r.i32(0), r.u32(4), r.u32(8), row))
    return SoundFile(header, resolved, permutations)


def resolve_sample(permutation: SoundPermutation, banks: dict[int, SoundBank]) -> SampleRow | None:
    """The bank row of a permutation. Some dialogue rows do not resolve."""
    bank = banks.get(permutation.bank)
    return None if bank is None else bank.find(permutation.sample_hash)


# --- W15-B5. Row values ----------------------------------------------------------------------
# switch / x360 16-byte row: hash, packed word (verified bit 0, temporary bit 1, compression bits
# 2..11, quality bits 12..21, format bits 22..31), decompressed size, stored size.
# ps3 40-byte row: hash, verified, temporary, compression, quality, decompressed size, format,
# zero, stored size, zero (15_coverage.md). x360 is taken to be the switch row, big-endian (Q503).

@dataclass(frozen=True)
class BankRowValues:
    hash: int
    compression: int
    format: int
    quality: int = 0
    decompressed_size: int = 0
    stored_size: int = 0
    verified: bool = False
    temporary: bool = False


def bank_row_values(raw: bytes, build: str) -> BankRowValues:
    """The fields of one ``.sbk`` sample row."""
    order = build_order(build)
    if len(raw) != _ROW[build]:
        raise FormatError(f"{build} sample row of {len(raw)} bytes, {_ROW[build]} expected")
    if build == "ps3":
        h, ver, tmp, comp, quality, dec, fmt, _, stored, _ = struct.unpack(order + "10I", raw)
        return BankRowValues(h, comp, fmt, quality, dec, stored, bool(ver), bool(tmp))
    h, word, dec, stored = struct.unpack(order + "4I", raw)
    return BankRowValues(h, word >> 2 & 0x3FF, word >> 22 & 0x3FF, word >> 12 & 0x3FF, dec, stored,
                         bool(word & 1), bool(word >> 1 & 1))


def pack_bank_row(values: BankRowValues, build: str) -> bytes:
    """One ``.sbk`` sample row from its fields (zero words zero)."""
    order = build_order(build)
    v = values
    if build == "ps3":
        return struct.pack(order + "10I", v.hash, int(v.verified), int(v.temporary), v.compression,
                           v.quality, v.decompressed_size, v.format, 0, v.stored_size, 0)
    for name, value in (("compression", v.compression), ("quality", v.quality), ("format", v.format)):
        if not 0 <= value < 1 << 10:
            raise FormatError(f"{name} {value} does not fit 10 bits")
    word = int(v.verified) | int(v.temporary) << 1 | v.compression << 2 | v.quality << 12 | v.format << 22
    return struct.pack(order + "4I", v.hash, word, v.decompressed_size, v.stored_size)
