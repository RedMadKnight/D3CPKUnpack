"""CPK archives, versions 6 and 7: reader, verifier, writer, rebuild and overlay lookup."""

from __future__ import annotations

import bisect
import struct
import zlib
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from .binary import BitReader, BitWriter, ByteView, FormatError

CPK_MAGIC = 0xA1B2C3D4
FLAG_COMPRESSED = 0x02
FLAG_COMPACT = 0x08
FLAG_RUNTIME_MASK = 0x35
V6_READ_SECTOR = 0x10000
V6_COMP_SECTOR = 0x4000
MAX_READ_SECTOR = 0x10000
MAX_COMP_SECTOR = 0x4000
MAX_PACKET_PLAIN = 0x10000
ZLIB_LEVEL = 9
_BIT_14 = 0x4000
CIPHER_SEED = 0x873AE8E53C11D761
_MASK64 = 0xFFFFFFFFFFFFFFFF
_MIN_SLOT_ROOM = 64

# Archive priorities on switch, for callers that overlay archives.
SWITCH_PRIORITIES = {
    "hotfix": 12,
    "CoreCommon": 8,
    "CoreCommon.locale": 9,
    "Common": 6,
    "Common.locale": 7,
    "ClientCommon": -4,
    "ServerCommon": -4,
    "act": -4,
    "ClientCommon.locale": -3,
    "ServerCommon.locale": -3,
    "act.locale": -3,
    "cutscene": -6,
}


def fnv1a64(data: bytes) -> int:
    value = 0xCBF29CE484222325
    for byte in data:
        value = ((value ^ byte) * 0x100000001B3) & _MASK64
    return value


def _lower_ascii(name: str) -> str:
    return "".join(chr(ord(c) + 32) if "A" <= c <= "Z" else c for c in name)


def name_hash(name: str) -> int:
    """FNV-1a 64 of the lower-cased name, ``|locale`` suffix included."""
    return fnv1a64(_lower_ascii(name).encode("latin-1"))


def decrypt_packet(stored: bytes) -> bytes:
    state = CIPHER_SEED
    out = bytearray(len(stored))
    for i, c in enumerate(stored):
        out[i] = c ^ (state & 0xFF)
        state = ((c << 56) | (state >> 8)) & _MASK64
    return bytes(out)


def encrypt_packet(plain: bytes) -> bytes:
    state = CIPHER_SEED
    out = bytearray(len(plain))
    for i, p in enumerate(plain):
        c = p ^ (state & 0xFF)
        out[i] = c
        state = ((c << 56) | (state >> 8)) & _MASK64
    return bytes(out)


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


@dataclass
class CpkHeader:
    order: str
    version: int
    stream_size: int
    flags: int
    file_count: int
    location_count: int
    header_sector: int
    size_bits: int
    copies_bits: int
    first_location_bits: int
    location_bits: int
    sector_offset_bits: int
    sector_index_bits: int
    crc: int
    read_sector: int
    comp_sector: int
    reserved: int
    # v6 only: the word at 0x3C. Its meaning is not settled; see QUESTIONS.md.
    word_3c: int
    file_size: int

    @property
    def header_size(self) -> int:
        return 64 if self.version == 6 else 72

    @property
    def compressed(self) -> bool:
        return bool(self.flags & FLAG_COMPRESSED)

    @property
    def start(self) -> int:
        """File position of the sector region."""
        if self.flags & FLAG_COMPACT:
            return self.header_sector
        return self.header_sector * self.read_sector

    @property
    def comp_sector_count(self) -> int:
        return _ceil_div(max(self.file_size - self.start, 0), self.comp_sector)

    @property
    def decomp_sector_count(self) -> int:
        return _ceil_div(self.stream_size, self.comp_sector)


@dataclass
class CpkFile:
    hash: int
    size: int
    count: int
    first: int
    name: str | None = None


@dataclass
class Packet:
    slot: int
    offset: int
    declared: int
    stored: int
    encrypted: bool
    inflated: int
    cut: bool
    padded: int  # zero bytes added because a cut packet stopped short
    excess: int  # bytes inflated past the declared size and dropped
    underrun: int  # a finished stream gave this many bytes fewer than declared


@dataclass
class SectorData:
    data: bytes
    packets: list[Packet]
    expected: int

    @property
    def length_ok(self) -> bool:
        return len(self.data) == self.expected


def read_header(data: bytes) -> CpkHeader:
    head = bytes(data[:4])
    if head == b"\xA1\xB2\xC3\xD4":
        order = ">"
    elif head == b"\xD4\xC3\xB2\xA1":
        order = "<"
    else:
        raise FormatError("bad CPK magic", 0)
    view = ByteView(data, order)
    version = view.u32(4, "version")
    if version not in (6, 7):
        raise FormatError(f"unknown CPK version {version}", 4)
    view.need(0, 64 if version == 6 else 72, "CPK header")
    fields = view.unpack("11I", 0x10, "CPK header")
    word_3c = view.u32(0x3C)
    if version == 6:
        read_sector, comp_sector, reserved = V6_READ_SECTOR, V6_COMP_SECTOR, 0
    else:
        read_sector, comp_sector, reserved = word_3c, view.u32(0x40), view.u32(0x44)
        if read_sector == 0:
            raise FormatError("read sector size is zero", 0x3C)
        if comp_sector == 0:
            raise FormatError("comp sector size is zero", 0x40)
    return CpkHeader(
        order, version, view.u64(8), fields[0], fields[1], fields[2], fields[3], fields[4],
        fields[5], fields[6], fields[7], fields[8], fields[9], fields[10],
        read_sector, comp_sector, reserved, word_3c, len(data),
    )


def header_problems(h: CpkHeader) -> list[str]:
    """Header sanity checks that do not depend on the tables."""
    problems = []
    if h.flags & FLAG_RUNTIME_MASK:
        problems.append(f"runtime flag bits set on disk: 0x{h.flags:X}")
    if h.flags not in (0, 2, 10):
        problems.append(f"unexpected flags value {h.flags}")
    if (h.version, h.order) not in ((6, ">"), (7, "<")):
        problems.append(f"version {h.version} with byte order {h.order!r}")
    if h.read_sector > MAX_READ_SECTOR:
        problems.append(f"read sector 0x{h.read_sector:X} above 0x10000")
    if h.comp_sector > MAX_COMP_SECTOR:
        problems.append(f"comp sector 0x{h.comp_sector:X} above 0x4000")
    if h.reserved:
        problems.append(f"reserved word is 0x{h.reserved:X}")
    return problems


def width_problems(h: CpkHeader) -> list[str]:
    problems = []
    if h.compressed:
        if h.sector_offset_bits != h.location_bits:
            problems.append(
                f"sectorOffsetBits {h.sector_offset_bits} != locationBits {h.location_bits}"
            )
        want = h.comp_sector_count.bit_length()
        if h.sector_index_bits != want:
            problems.append(f"sectorIndexBits {h.sector_index_bits} != bit_length {want}")
    return problems


@dataclass
class CpkArchive:
    data: bytes
    header: CpkHeader
    files: list[CpkFile]
    locations: list[int]
    sector_offsets: list[int]
    sector_index: list[int]
    name_offsets: list[int]
    names: list[str]
    tables_end: int
    by_hash: dict[int, CpkFile] = field(default_factory=dict)
    _cache: OrderedDict = field(default_factory=OrderedDict, repr=False)

    def find(self, name: str | int) -> CpkFile | None:
        return self.by_hash.get(name if isinstance(name, int) else name_hash(name))

    def read_file(self, file: str | int | CpkFile, copy: int = 0, strict: bool = True) -> bytes:
        entry = file if isinstance(file, CpkFile) else self.find(file)
        if entry is None:
            raise FormatError(f"no such file: {file!r}")
        if not 0 <= copy < entry.count:
            raise FormatError(f"file has no copy {copy}")
        index = entry.first + copy
        if index >= len(self.locations):
            raise FormatError(f"location {index} past the location table")
        return self.read_stream(self.locations[index], entry.size, strict)

    def read_stream(self, offset: int, size: int, strict: bool = True) -> bytes:
        h = self.header
        if offset < 0 or size < 0 or offset + size > h.stream_size:
            raise FormatError(f"stream range 0x{offset:X}+0x{size:X} outside stream", offset)
        if size == 0:
            return b""
        if not h.compressed:
            pos = h.start + offset
            if pos + size > len(self.data):
                raise FormatError("raw payload runs past the end", pos)
            return bytes(self.data[pos:pos + size])
        k = bisect.bisect_right(self.sector_offsets, offset) - 1
        if k < 0:
            raise FormatError("no sector holds this stream offset", offset)
        parts = []
        remaining = size
        cursor = offset
        while remaining:
            if k >= len(self.sector_offsets):
                raise FormatError("stream range past the last sector", cursor)
            sector = self.sector(k)
            if strict and not sector.length_ok:
                raise FormatError(
                    f"sector {k} inflated to {len(sector.data)} bytes, "
                    f"its offset gap is {sector.expected}",
                    self.slot_position(k),
                )
            inner = cursor - self.sector_offsets[k]
            part = sector.data[inner:inner + remaining]
            if inner >= 0 and part:
                parts.append(part)
                cursor += len(part)
                remaining -= len(part)
            k += 1
        return b"".join(parts)

    def slot_position(self, slot: int) -> int:
        return self.header.start + slot * self.header.comp_sector

    def sector_length(self, slot: int) -> int:
        """Expected inflated length: the gap to the next sector offset."""
        offsets = self.sector_offsets
        stop = offsets[slot + 1] if slot + 1 < len(offsets) else self.header.stream_size
        return stop - offsets[slot]

    def sector(self, slot: int) -> SectorData:
        cached = self._cache.get(slot)
        if cached is not None:
            self._cache.move_to_end(slot)
            return cached
        result = self._decode_sector(slot)
        self._cache[slot] = result
        if len(self._cache) > 8:
            self._cache.popitem(last=False)
        return result

    def _decode_sector(self, slot: int) -> SectorData:
        h = self.header
        data = self.data
        pos = self.slot_position(slot)
        end = min(pos + h.comp_sector, len(data))
        parts: list[bytes] = []
        packets: list[Packet] = []
        while end - pos >= 7:
            low, high, word = struct.unpack_from(h.order + "HHH", data, pos)
            stored = word & 0x7FFF
            if stored == 0:
                if low or high or packets:
                    break
                pos += 6  # an empty header before any real data is hopped
                continue
            encrypted = bool(word & 0x8000)
            declared = low | (high << 16)
            payload = pos + 6
            if encrypted:
                stored_end = min(payload + stored, end)
                feed = decrypt_packet(bytes(data[payload:stored_end]))
            else:
                stored_end = end
                feed = bytes(data[payload:end])
            inflater = zlib.decompressobj()
            try:
                out = inflater.decompress(feed)
            except zlib.error as exc:
                raise FormatError(f"packet does not inflate: {exc}", pos) from None
            cut = not inflater.eof
            if cut:
                next_pos = stored_end if encrypted else end
            else:
                next_pos = payload + len(feed) - len(inflater.unused_data)
            excess = max(len(out) - declared, 0)
            if excess:
                out = out[:declared]
            padded = underrun = 0
            if len(out) < declared:
                if cut:
                    padded = declared - len(out)
                    out += bytes(padded)
                else:
                    underrun = declared - len(out)
            parts.append(out)
            packets.append(Packet(
                slot, pos, declared, stored, encrypted, len(out) - padded,
                cut, padded, excess, underrun,
            ))
            pos = next_pos
        joined = b"".join(parts)
        expected = self.sector_length(slot)
        padded_total = sum(p.padded for p in packets)
        if padded_total and len(joined) != expected and len(joined) - padded_total == expected:
            joined = joined[:expected]
        return SectorData(joined, packets, expected)


def read_archive(data: bytes, strict: bool = True) -> CpkArchive:
    """Parse the tables. With ``strict`` a bad header or width raises FormatError."""
    h = read_header(data)
    if strict:
        problems = header_problems(h) + width_problems(h)
        if problems:
            raise FormatError(problems[0], 0x10)
    if h.start > len(data):
        raise FormatError(f"sector region 0x{h.start:X} past the end of the file", 0x1C)
    if h.location_count and h.location_bits == 0:
        raise FormatError("locations declared with width 0", 0x2C)
    if h.compressed and h.decomp_sector_count and h.sector_index_bits == 0:
        raise FormatError("sector index declared with width 0", 0x34)

    bits = BitReader(data, h.header_size)
    files = []
    for _ in range(h.file_count):
        file_hash = bits.read(64)
        size = bits.read(h.size_bits)
        count = bits.read(h.copies_bits)
        first = bits.read(h.first_location_bits)
        files.append(CpkFile(file_hash, size, count, first))
    bits.align()
    locations = [bits.read(h.location_bits) for _ in range(h.location_count)]
    bits.align()
    sector_offsets: list[int] = []
    sector_index: list[int] = []
    if h.compressed:
        sector_offsets = [bits.read(h.location_bits) for _ in range(h.comp_sector_count)]
        bits.align()
        sector_index = [bits.read(h.sector_index_bits) for _ in range(h.decomp_sector_count)]
        bits.align()
    pos = bits.bit >> 3
    view = ByteView(data, h.order)
    view.need(pos, h.file_count * 4, "name offsets")
    name_offsets = list(view.unpack(f"{h.file_count}I", pos, "name offsets"))
    blob_start = pos + h.file_count * 4
    if blob_start > h.start:
        raise FormatError("tables run into the sector region", blob_start)
    blob = bytes(data[blob_start:h.start])
    names = []
    for offset in name_offsets:
        end = blob.find(b"\0", offset)
        if offset >= len(blob) or end < 0:
            raise FormatError(f"name offset {offset} has no name", blob_start + offset)
        names.append(blob[offset:end].decode("latin-1"))
    by_hash: dict[int, CpkFile] = {}
    for entry in files:
        by_hash.setdefault(entry.hash, entry)
    for name in names:
        entry = by_hash.get(name_hash(name))
        if entry is not None:
            entry.name = name
    return CpkArchive(
        data, h, files, locations, sector_offsets, sector_index, name_offsets, names,
        blob_start, by_hash,
    )


# --- Verification ---------------------------------------------------------------------

CHECK_NAMES = (
    "header", "widths", "order", "names", "crc", "sector_tables", "packets",
    "sector_lengths", "stream", "locations", "copies",
)


@dataclass
class CheckResult:
    name: str
    problems: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    skipped: bool = False

    @property
    def ok(self) -> bool:
        return not self.problems


@dataclass
class VerifyReport:
    checks: dict[str, CheckResult]
    packets: int = 0
    encrypted_packets: int = 0
    cut_packets: int = 0
    padded_bytes: int = 0

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks.values())

    def format(self, limit: int = 10) -> str:
        lines = []
        for check in self.checks.values():
            state = "SKIP" if check.skipped else "OK" if check.ok else "FAIL"
            lines.append(f"{state:4} {check.name}")
            lines.extend(f"     ! {p}" for p in check.problems[:limit])
            if len(check.problems) > limit:
                lines.append(f"     ! ... {len(check.problems) - limit} more")
            lines.extend(f"     - {n}" for n in check.notes[:limit])
        lines.append(
            f"packets {self.packets}, encrypted {self.encrypted_packets}, "
            f"cut {self.cut_packets}, zero-padded bytes {self.padded_bytes}"
        )
        return "\n".join(lines)


def verify_archive(data: bytes, allow_gaps: bool = False) -> VerifyReport:
    """Run every check separately. Stream gaps become notes with ``allow_gaps``."""
    report = VerifyReport({name: CheckResult(name) for name in CHECK_NAMES})
    checks = report.checks
    try:
        archive = read_archive(data, strict=False)
    except FormatError as exc:
        checks["header"].problems.append(str(exc))
        for name in CHECK_NAMES[1:]:
            checks[name].skipped = True
        return report
    h = archive.header
    checks["header"].problems.extend(header_problems(h))
    excess = (h.file_size - h.start) % h.comp_sector
    if excess:
        checks["header"].notes.append(
            f"file does not end on a slot boundary ({excess} bytes into the last slot)"
        )
    checks["widths"].problems.extend(width_problems(h))
    _check_order(archive, checks["order"])
    _check_names(archive, checks["names"])
    crc = zlib.crc32(memoryview(data)[h.start:], 0) & 0xFFFFFFFF
    if crc != h.crc:
        checks["crc"].problems.append(f"crc 0x{crc:08X}, header says 0x{h.crc:08X}")
    _check_sector_tables(archive, checks["sector_tables"])
    sectors_ok = _check_sectors(archive, report)
    _check_stream(archive, checks["stream"], allow_gaps)
    locations_ok = _check_locations(archive, checks["locations"])
    if sectors_ok and locations_ok:
        _check_copies(archive, checks["copies"])
    else:
        checks["copies"].skipped = True
    return report


def _check_order(archive: CpkArchive, check: CheckResult) -> None:
    previous = None
    for index, entry in enumerate(archive.files):
        if previous is not None and entry.hash <= previous:
            kind = "duplicate" if entry.hash == previous else "out of order"
            check.problems.append(f"entry {index}: hash 0x{entry.hash:016X} {kind}")
        previous = entry.hash


def _check_names(archive: CpkArchive, check: CheckResult) -> None:
    for name in archive.names:
        if name_hash(name) not in archive.by_hash:
            check.problems.append(f"name {name!r} matches no file entry")
    for entry in archive.files:
        if entry.name is None:
            check.problems.append(f"file 0x{entry.hash:016X} has no name")


def _check_sector_tables(archive: CpkArchive, check: CheckResult) -> None:
    h = archive.header
    if not h.compressed:
        check.skipped = True
        return
    offsets = archive.sector_offsets
    if offsets and offsets[0] != 0:
        check.problems.append(f"first sector offset is {offsets[0]}")
    if len(offsets) >= 2 and offsets[0] == offsets[1] == 0:
        check.notes.append("slot 0 is an empty stub")
    for k in range(1, len(offsets)):
        if offsets[k] < offsets[k - 1]:
            check.problems.append(f"sector offset {k} goes backwards")
        elif offsets[k] == offsets[k - 1] and k > 1:
            check.problems.append(f"sector offset {k} repeats")
        if offsets[k] > h.stream_size:
            check.problems.append(f"sector offset {k} past the stream")
    for k, value in enumerate(archive.sector_index):
        true_sector = bisect.bisect_right(offsets, k * h.comp_sector) - 1
        if value > true_sector:
            check.problems.append(f"sector index {k}: {value} is past true sector {true_sector}")


def _check_sectors(archive: CpkArchive, report: VerifyReport) -> bool:
    packets_check = report.checks["packets"]
    lengths_check = report.checks["sector_lengths"]
    if not archive.header.compressed:
        packets_check.skipped = lengths_check.skipped = True
        return True
    good = True
    for slot in range(len(archive.sector_offsets)):
        try:
            sector = archive._decode_sector(slot)
        except FormatError as exc:
            packets_check.problems.append(f"slot {slot}: {exc}")
            good = False
            continue
        for number, packet in enumerate(sector.packets):
            if number and packet.stored & _BIT_14:
                packets_check.problems.append(
                    f"slot {slot}: packet at 0x{packet.offset:X} after the first has bit 14 set"
                )
            report.packets += 1
            report.encrypted_packets += packet.encrypted
            report.cut_packets += packet.cut
            report.padded_bytes += packet.padded
            if packet.padded:
                packets_check.notes.append(
                    f"slot {slot}: cut packet zero-padded by {packet.padded} bytes"
                )
            if packet.underrun:
                packets_check.problems.append(
                    f"slot {slot}: packet at 0x{packet.offset:X} is {packet.underrun} "
                    "bytes short of its declared size"
                )
                good = False
        if not sector.length_ok:
            lengths_check.problems.append(
                f"slot {slot}: inflated {len(sector.data)}, offset gap {sector.expected}"
            )
            good = False
    return good


def _check_stream(archive: CpkArchive, check: CheckResult, allow_gaps: bool) -> None:
    h = archive.header
    report = check.notes if allow_gaps else check.problems
    total = sum(entry.size * entry.count for entry in archive.files)
    if total != h.stream_size:
        report.append(f"sum of size * count is {total}, streamSize is {h.stream_size}")
    spans = []
    for entry in archive.files:
        for index in range(entry.first, min(entry.first + entry.count, len(archive.locations))):
            spans.append((archive.locations[index], entry.size))
    spans.sort()
    cursor = 0
    for location, size in spans:
        if location > cursor:
            report.append(f"gap in the stream at 0x{cursor:X}")
        elif location < cursor and size:
            report.append(f"overlap in the stream at 0x{location:X}")
        cursor = max(cursor, location + size)
    if cursor < h.stream_size:
        report.append(f"gap at the end of the stream, from 0x{cursor:X}")


def _check_locations(archive: CpkArchive, check: CheckResult) -> bool:
    h = archive.header
    count = len(archive.locations)
    for entry in archive.files:
        if entry.first + entry.count > count:
            check.problems.append(f"file 0x{entry.hash:016X}: locations past the table")
            continue
        for index in range(entry.first, entry.first + entry.count):
            location = archive.locations[index]
            if location + entry.size > h.stream_size:
                check.problems.append(
                    f"file 0x{entry.hash:016X}: location 0x{location:X} outside the stream"
                )
    return check.ok


def _check_copies(archive: CpkArchive, check: CheckResult) -> None:
    single_only = archive.header.version == 7
    for entry in archive.files:
        if entry.count == 1:
            continue
        if single_only:
            check.problems.append(f"file 0x{entry.hash:016X} has {entry.count} locations on v7")
        first = archive.read_file(entry, 0, strict=False)
        masked = first[:8] + first[16:]
        for copy in range(1, entry.count):
            other = archive.read_file(entry, copy, strict=False)
            if other[:8] + other[16:] != masked:
                check.problems.append(
                    f"file 0x{entry.hash:016X}: copy {copy} differs outside bytes 8 to 15"
                )


# --- Writing ------------------------------------------------------------------------------

@dataclass
class InputFile:
    name: str
    copies: list[bytes]


@dataclass
class Widths:
    size: int = 0
    copies: int = 0
    first: int = 0
    location: int = 0


def _make_packet(plain: bytes, order: str, encrypted: bool, level: int = ZLIB_LEVEL) -> bytes:
    stored = zlib.compress(plain, level)
    if encrypted:
        stored = encrypt_packet(stored)
    word = len(stored) | (0x8000 if encrypted else 0)
    size = len(plain)
    return struct.pack(order + "HHH", size & 0xFFFF, size >> 16, word) + stored


def _cut_to_room(stored: bytes, room: int) -> tuple[bytes, int]:
    """Cut a zlib stream to ``room`` bytes. Returns them and the plain bytes they give."""
    kept = stored[:room]
    return kept, len(zlib.decompressobj().decompress(kept))


def _pack_stream(
    stream: bytes, comp_sector: int, order: str, empty_first: bool, encrypt,
    level: int = ZLIB_LEVEL, run_past_slot: bool = True, max_packet: int = MAX_PACKET_PLAIN,
) -> tuple[list[int], bytes]:
    """Split the stream into packets.

    With ``run_past_slot`` a slot's last packet is cut at the slot end, and declares what the
    cut stream gives. The next slot goes on from there. Otherwise every packet is complete.
    """
    offsets: list[int] = []
    slots: list[bytes] = []
    slot = bytearray()
    open_slot = False
    pos = 0
    number = 0
    if empty_first:
        offsets.append(0)
        slot += _make_packet(b"", order, False, level)
        open_slot = True
    while pos < len(stream):
        if not open_slot:
            offsets.append(pos)
            open_slot = True
        room = comp_sector - len(slot) - 6
        size = min(max_packet, len(stream) - pos)
        stored = None
        if room >= _MIN_SLOT_ROOM:
            while True:
                stored = zlib.compress(stream[pos:pos + size], level)
                if len(stored) <= room:
                    break
                if run_past_slot:
                    stored, size = _cut_to_room(stored, room)
                    if size == 0:
                        stored = None
                    break
                if size == 1:
                    stored = None
                    break
                size = max(1, min(size - 1, int(size * room / len(stored) * 0.95)))
        if stored is None:
            slots.append(bytes(slot).ljust(comp_sector, b"\0"))
            slot = bytearray()
            open_slot = False
            continue
        encrypted = bool(encrypt and encrypt(number))
        if encrypted:
            stored = encrypt_packet(stored)
        word = len(stored) | (0x8000 if encrypted else 0)
        slot += struct.pack(order + "HHH", size & 0xFFFF, size >> 16, word) + stored
        pos += size
        number += 1
    if open_slot:
        slots.append(bytes(slot).ljust(comp_sector, b"\0"))
    return offsets, b"".join(slots)


def write_archive(
    files: Iterable[InputFile] | Mapping[str, bytes],
    *,
    version: int = 7,
    order: str | None = None,
    flags: int = 2,
    read_sector: int | None = None,
    comp_sector: int | None = None,
    empty_first_packet: bool | None = None,
    widths: Widths | None = None,
    name_order: Sequence[str] | None = None,
    encrypt=None,
    word_3c: int | None = None,
    level: int = ZLIB_LEVEL,
    run_past_slot: bool = True,
    max_packet: int = MAX_PACKET_PLAIN,
) -> bytes:
    """Build an archive. ``encrypt(n)`` picks data packets to encrypt, for tests and parity."""
    if version not in (6, 7):
        raise FormatError(f"cannot write CPK version {version}")
    if flags not in (0, 2, 10):
        raise FormatError(f"cannot write flags {flags}")
    order = order or (">" if version == 6 else "<")
    if version == 6:
        if read_sector not in (None, V6_READ_SECTOR) or comp_sector not in (None, V6_COMP_SECTOR):
            raise FormatError("v6 sector sizes are fixed")
    read_sector = read_sector or V6_READ_SECTOR
    comp_sector = comp_sector or V6_COMP_SECTOR
    if read_sector > MAX_READ_SECTOR:
        raise FormatError(f"read sector 0x{read_sector:X} above 0x10000")
    if comp_sector > MAX_COMP_SECTOR:
        raise FormatError(f"comp sector 0x{comp_sector:X} above 0x4000")
    if comp_sector < 0x100:
        raise FormatError(f"comp sector 0x{comp_sector:X} too small to hold a packet")
    if empty_first_packet is None:
        empty_first_packet = version == 6

    if isinstance(files, Mapping):
        inputs = [InputFile(name, [content]) for name, content in files.items()]
    else:
        inputs = list(files)

    # The stream and the location table follow the input order.
    stream_parts: list[bytes] = []
    locations: list[int] = []
    entries: list[CpkFile] = []
    cursor = 0
    seen: set[int] = set()
    for item in inputs:
        if "\0" in item.name:
            raise FormatError(f"NUL in name {item.name!r}")
        try:
            file_hash = name_hash(item.name)
        except UnicodeEncodeError:
            raise FormatError(f"name {item.name!r} is not latin-1") from None
        if file_hash in seen:
            raise FormatError(f"duplicate name or hash: {item.name!r}")
        seen.add(file_hash)
        if not item.copies:
            raise FormatError(f"file {item.name!r} has no copies")
        size = len(item.copies[0])
        if any(len(copy) != size for copy in item.copies):
            raise FormatError(f"copies of {item.name!r} differ in size")
        entries.append(CpkFile(file_hash, size, len(item.copies), len(locations), item.name))
        for copy in item.copies:
            locations.append(cursor)
            stream_parts.append(copy)
            cursor += size
    stream = b"".join(stream_parts)
    entries.sort(key=lambda e: e.hash)

    sector_offsets: list[int] = []
    if flags & FLAG_COMPRESSED:
        sector_offsets, region = _pack_stream(
            stream, comp_sector, order, empty_first_packet, encrypt, level, run_past_slot,
            max_packet,
        )
    else:
        region = stream.ljust(_ceil_div(len(stream), comp_sector) * comp_sector, b"\0")
    comp_count = len(region) // comp_sector
    decomp_count = _ceil_div(len(stream), comp_sector)
    index_bits = comp_count.bit_length()

    w = widths or Widths()
    size_bits = max(w.size, max((e.size for e in entries), default=0).bit_length(), 1)
    copies_bits = max(w.copies, max((e.count for e in entries), default=0).bit_length(), 1)
    first_bits = max(w.first, max((e.first for e in entries), default=0).bit_length(), 1)
    location_bits = max(w.location, len(stream).bit_length(), 1)

    bits = BitWriter()
    for entry in entries:
        bits.write(entry.hash, 64)
        bits.write(entry.size, size_bits)
        bits.write(entry.count, copies_bits)
        bits.write(entry.first, first_bits)
    tables = bytearray(bits.finish())
    bits = BitWriter()
    for location in locations:
        bits.write(location, location_bits)
    tables += bits.finish()
    if flags & FLAG_COMPRESSED:
        bits = BitWriter()
        for offset in sector_offsets:
            bits.write(offset, location_bits)
        tables += bits.finish()
        bits = BitWriter()
        for k in range(decomp_count):
            sector = max(bisect.bisect_right(sector_offsets, k * comp_sector) - 1, 0)
            bits.write(sector, index_bits)
        tables += bits.finish()

    ordered_names = list(name_order) if name_order is not None else [i.name for i in inputs]
    if sorted(map(name_hash, ordered_names)) != [e.hash for e in entries]:
        raise FormatError("name order does not match the file list")
    blob = bytearray()
    for name in ordered_names:
        tables += struct.pack(order + "I", len(blob))
        blob += name.encode("latin-1") + b"\0"
    tables += blob

    header_size = 64 if version == 6 else 72
    tables_end = header_size + len(tables)
    if flags & FLAG_COMPACT:
        header_sector = tables_end
        start = tables_end
    else:
        header_sector = _ceil_div(tables_end, read_sector)
        start = header_sector * read_sector

    crc = zlib.crc32(region, 0) & 0xFFFFFFFF
    header = struct.pack(
        order + "IIQ12I",
        CPK_MAGIC, version, len(stream), flags, len(entries), len(locations), header_sector,
        size_bits, copies_bits, first_bits, location_bits, location_bits, index_bits, crc,
        read_sector if version == 7 or word_3c is None else word_3c,
    )
    if version == 7:
        header += struct.pack(order + "II", comp_sector, 0)
    return header + bytes(tables).ljust(start - header_size, b"\0") + region


@dataclass
class RebuildResult:
    data: bytes
    encrypted_packets_dropped: int = 0
    notes: list[str] = field(default_factory=list)


def rebuild(
    source: bytes | CpkArchive,
    replace: Mapping[str, bytes] | None = None,
    remove: Iterable[str] = (),
    add: Mapping[str, bytes] | None = None,
    single_copy: bool = False,
    level: int = ZLIB_LEVEL,
    run_past_slot: bool = True,
) -> RebuildResult:
    """Rewrite an archive with the source's format. A replacement updates every copy."""
    archive = source if isinstance(source, CpkArchive) else read_archive(source)
    h = archive.header
    replacements = {name_hash(name): data for name, data in (replace or {}).items()}
    removals = {name_hash(name) for name in remove}
    for file_hash in list(replacements) + list(removals):
        if file_hash not in archive.by_hash:
            raise FormatError(f"no file with hash 0x{file_hash:016X} in the source")

    inputs: list[InputFile] = []
    for entry in sorted(archive.files, key=lambda e: e.first):
        if entry.hash in removals:
            continue
        if entry.name is None:
            raise FormatError(f"file 0x{entry.hash:016X} has no name, cannot rebuild it")
        count = 1 if single_copy else entry.count
        if entry.hash in replacements:
            copies = [replacements[entry.hash]] * count
        else:
            copies = [archive.read_file(entry, copy) for copy in range(count)]
        inputs.append(InputFile(entry.name, copies))
    names = [n for n in archive.names if name_hash(n) not in removals]
    for name, content in (add or {}).items():
        inputs.append(InputFile(name, [content]))
        names.append(name)

    encrypted = 0
    empty_first = h.version == 6
    if h.compressed:
        for slot in range(len(archive.sector_offsets)):
            packets = archive.sector(slot).packets
            encrypted += sum(p.encrypted for p in packets)
            if slot == 0 and packets and packets[0].declared == 0:
                empty_first = True
    data = write_archive(
        inputs,
        version=h.version,
        order=h.order,
        flags=h.flags,
        read_sector=h.read_sector,
        comp_sector=h.comp_sector,
        empty_first_packet=empty_first,
        widths=Widths(h.size_bits, h.copies_bits, h.first_location_bits, h.location_bits),
        name_order=names,
        word_3c=h.word_3c,
        level=level,
        run_past_slot=run_past_slot,
    )
    result = RebuildResult(data, encrypted)
    if encrypted:
        result.notes.append(f"{encrypted} encrypted packets were written unencrypted")
    return result


# --- Overlay lookup ----------------------------------------------------------------------

class ArchiveOverlay:
    """Game lookup order over several archives."""

    def __init__(self) -> None:
        self._entries: list[tuple[int, int, CpkArchive]] = []

    def add(self, archive: CpkArchive, priority: int) -> None:
        self._entries.append((priority, len(self._entries), archive))

    def _best(self, name: str) -> tuple[CpkArchive, CpkFile] | None:
        file_hash = name_hash(name)
        hits = [e for e in self._entries if file_hash in e[2].by_hash]
        if not hits:
            return None
        archive = max(hits, key=lambda e: (e[0], e[1]))[2]
        return archive, archive.by_hash[file_hash]

    def resolve(
        self, name: str, display_locale: str | None = None, audio_locale: str | None = None,
    ) -> tuple[CpkArchive, CpkFile] | None:
        attempts = []
        if display_locale:
            attempts.append(f"{name}|{display_locale}")
        if audio_locale and audio_locale != display_locale:
            attempts.append(f"{name}|{audio_locale}")
        attempts.append(name)
        for attempt in attempts:
            hit = self._best(attempt)
            if hit is not None:
                return hit
        return None
