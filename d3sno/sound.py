"""``.sbk`` sound banks and ``.snd`` sounds."""

from __future__ import annotations

from dataclasses import dataclass

from .binary import ByteView, FormatError
from .sno import SnoHeader, parse_header, resolve_build

_ROW = {"switch": 16, "ps3": 40}
_STORED_AT = {"switch": 0x0C, "ps3": 0x20}
_SAMPLE_COUNT_AT = {"switch": 0x30, "ps3": 0x24}
_PERMUTATION = {"switch": (0x54, 0x5C, 64), "ps3": (0x50, 0x58, 44)}  # pair, count, size


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
    view = ByteView(data, header.order)
    pair_at, count_at, size = _PERMUTATION[header.build]
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
    return SoundFile(header, header.build, permutations)


def resolve_sample(permutation: SoundPermutation, banks: dict[int, SoundBank]) -> SampleRow | None:
    """The bank row of a permutation. Some dialogue rows do not resolve."""
    bank = banks.get(permutation.bank)
    return None if bank is None else bank.find(permutation.sample_hash)
