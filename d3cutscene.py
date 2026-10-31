#!/usr/bin/env python3
"""
Reader for Diablo III raw cutscene CPK archives.

Most Diablo III CPK archives store encrypted/zlib-compressed sectors. The
cutscene archives observed on console builds use the same header, bit-packed
SortedFileInfo table, Locations table, and name table, but their payload region
is a raw concatenation of complete Ogg/Theora/Vorbis files. This module handles
that raw variant without relying on any game executable behavior.
"""
from __future__ import annotations

import os
import shutil
import struct
from dataclasses import dataclass
from typing import BinaryIO, Iterable, Optional


def read_bits(buf: bytes, bit_pos: int, bit_count: int) -> int:
    """Read an MSB-first bit field from a byte string."""
    result = 0
    for i in range(bit_count):
        pos = bit_pos + i
        byte_pos = pos // 8
        byte_bit = 7 - (pos % 8)
        result <<= 1
        if buf[byte_pos] & (1 << byte_bit):
            result |= 1
    return result


class Reader:
    def __init__(self, f: BinaryIO, endian: str):
        self.f = f
        self.endian = endian

    def tell(self) -> int:
        return self.f.tell()

    def seek(self, pos: int) -> None:
        self.f.seek(pos)

    def read(self, size: int) -> bytes:
        data = self.f.read(size)
        if len(data) != size:
            raise EOFError(f"short read: wanted {size}, got {len(data)}")
        return data

    def read_u32(self) -> int:
        return struct.unpack(f"{self.endian}I", self.read(4))[0]

    def read_u64(self) -> int:
        return struct.unpack(f"{self.endian}Q", self.read(8))[0]

    def read_cstr(self) -> str:
        data = bytearray()
        while True:
            b = self.f.read(1)
            if not b or b == b"\x00":
                break
            data += b
        return data.decode("latin1")


@dataclass(frozen=True)
class CutsceneEntry:
    index: int
    name: str
    name_hash: int
    size: int
    loc_count: int
    loc_index: int
    offset: int

    @property
    def end(self) -> int:
        return self.offset + self.size

    @property
    def basename(self) -> str:
        return self.name.replace("\\", "/").rsplit("/", 1)[-1]

    @property
    def ext(self) -> str:
        return self.basename.rsplit(".", 1)[1].lower() if "." in self.basename else ""


@dataclass(frozen=True)
class OggPage:
    pos: int
    flags: int
    serial: int
    sequence: int
    granule: int
    size: int
    segments: int


@dataclass(frozen=True)
class OggGroup:
    index: int
    start: int
    end: int
    pages: int
    serials: frozenset[int]

    @property
    def size(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class OggValidation:
    ok: bool
    pages: int
    streams: frozenset[int]
    status: str


def parse_ogg_page(f: BinaryIO, pos: int) -> Optional[OggPage]:
    f.seek(pos)
    header = f.read(27)
    if len(header) != 27 or header[:4] != b"OggS":
        return None
    if header[4] != 0:
        return None
    flags = header[5]
    granule = struct.unpack_from("<q", header, 6)[0]
    serial = struct.unpack_from("<I", header, 14)[0]
    sequence = struct.unpack_from("<I", header, 18)[0]
    segment_count = header[26]
    segment_table = f.read(segment_count)
    if len(segment_table) != segment_count:
        return None
    size = 27 + segment_count + sum(segment_table)
    return OggPage(pos, flags, serial, sequence, granule, size, segment_count)


def scan_ogg_groups(path: str, start: int) -> list[OggGroup]:
    groups: list[OggGroup] = []
    with open(path, "rb") as f:
        file_size = f.seek(0, os.SEEK_END)
        pos = start
        group_start = pos
        active: set[int] = set()
        serials: set[int] = set()
        pages = 0
        while pos < file_size:
            page = parse_ogg_page(f, pos)
            if page is None:
                break
            if page.flags & 0x02:
                active.add(page.serial)
                serials.add(page.serial)
            if page.flags & 0x04:
                active.discard(page.serial)
            pages += 1
            pos += page.size
            if not active:
                groups.append(
                    OggGroup(
                        len(groups), group_start, pos, pages, frozenset(serials)
                    )
                )
                group_start = pos
                serials.clear()
                pages = 0
    return groups


def validate_ogg_range(
    f: BinaryIO, start: int, size: int, require_eos: bool = True
) -> OggValidation:
    end = start + size
    pos = start
    pages = 0
    active: set[int] = set()
    serials: set[int] = set()
    while pos < end:
        page = parse_ogg_page(f, pos)
        if page is None:
            return OggValidation(
                False, pages, frozenset(serials), f"bad Ogg page at {pos:#x}"
            )
        if pos + page.size > end:
            return OggValidation(
                False,
                pages,
                frozenset(serials),
                f"Ogg page at {pos:#x} extends past range end",
            )
        if page.flags & 0x02:
            active.add(page.serial)
            serials.add(page.serial)
        if page.flags & 0x04:
            active.discard(page.serial)
        pages += 1
        pos += page.size
    if pos != end:
        return OggValidation(False, pages, frozenset(serials), "range under-read")
    if pages == 0:
        return OggValidation(False, pages, frozenset(serials), "empty range")
    if require_eos and active:
        return OggValidation(
            False, pages, frozenset(serials), "missing EOS for one or more streams"
        )
    return OggValidation(True, pages, frozenset(serials), "ok")


class CutsceneCPK:
    MAGIC = 0xA1B2C3D4

    def __init__(self, path: str):
        self.path = path
        with open(path, "rb") as f:
            self.file_size = f.seek(0, os.SEEK_END)
            f.seek(0)
            self._read_header(f)
            self._read_tables(f)

    def _read_header(self, f: BinaryIO) -> None:
        magic_be = struct.unpack(">I", f.read(4))[0]
        if magic_be == self.MAGIC:
            endian = ">"
        elif magic_be == 0xD4C3B2A1:
            endian = "<"
        else:
            raise ValueError(f"not a Diablo III CPK archive: {magic_be:08x}")

        f.seek(0)
        r = Reader(f, endian)
        self.endian = endian
        self.magic = r.read_u32()
        self.version = r.read_u32()
        self.decomp_size = r.read_u64()
        self.fmt_const = r.read_u32()
        self.file_count = r.read_u32()
        self.loc_count = r.read_u32()
        self.header_sector = r.read_u32()
        self.size_bits = r.read_u32()
        self.loc_count_bits = r.read_u32()
        self.loc_index_bits = r.read_u32()
        self.loc_bits = r.read_u32()
        self.cs2do_bits = r.read_u32()
        self.ds2cs_bits = r.read_u32()
        self.sector_crc32 = r.read_u32()

        if self.version >= 7:
            self.read_sector = r.read_u32()
            self.comp_sector = r.read_u32()
            self.reserved = r.read_u32()
            self.header_size = 0x48
        else:
            self.read_sector = 0x10000
            self.comp_sector = 0x4000
            self.reserved = 0
            self.header_size = 0x40

        if self.fmt_const == 10:
            self.data_start = self.header_sector
        else:
            self.data_start = self.header_sector * self.read_sector

        self.is_raw_cutscene_archive = (
            self.fmt_const == 0
            and self.file_size - self.data_start == self.decomp_size
        )

    def _read_tables(self, f: BinaryIO) -> None:
        f.seek(self.header_size)
        record_bits = (
            64 + self.size_bits + self.loc_count_bits + self.loc_index_bits
        )
        sorted_info_size = (record_bits * self.file_count + 7) // 8
        sorted_info = f.read(sorted_info_size)

        records = []
        for i in range(self.file_count):
            bit_pos = i * record_bits
            name_hash = read_bits(sorted_info, bit_pos, 64)
            bit_pos += 64
            size = read_bits(sorted_info, bit_pos, self.size_bits)
            bit_pos += self.size_bits
            loc_count = read_bits(sorted_info, bit_pos, self.loc_count_bits)
            bit_pos += self.loc_count_bits
            loc_index = read_bits(sorted_info, bit_pos, self.loc_index_bits)
            records.append((name_hash, size, loc_count, loc_index))

        loc_table_size = (self.loc_bits * self.loc_count + 7) // 8
        loc_table = f.read(loc_table_size)
        locations = [
            read_bits(loc_table, i * self.loc_bits, self.loc_bits)
            for i in range(self.loc_count)
        ]

        if not self.is_raw_cutscene_archive:
            raise ValueError(
                "archive does not look like the raw cutscene CPK variant "
                f"(fmt_const={self.fmt_const}, data_start={self.data_start:#x}, "
                f"payload={self.file_size - self.data_start}, "
                f"declared={self.decomp_size})"
            )

        raw_offsets = f.read(4 * self.file_count)
        name_offsets = list(
            struct.unpack(f"{self.endian}{self.file_count}I", raw_offsets)
        )
        names_base = f.tell()

        names = []
        r = Reader(f, self.endian)
        for off in name_offsets:
            r.seek(names_base + off)
            names.append(r.read_cstr())

        self.sorted_info_offset = self.header_size
        self.locations_offset = self.header_size + sorted_info_size
        self.name_offsets_offset = names_base - 4 * self.file_count
        self.names_base = names_base
        self.locations = locations
        self.entries: list[CutsceneEntry] = []

        for i, (name_hash, size, loc_count, loc_index) in enumerate(records):
            if loc_count and loc_index < len(locations):
                offset = self.data_start + locations[loc_index]
            else:
                offset = 0
            self.entries.append(
                CutsceneEntry(i, names[i], name_hash, size, loc_count, loc_index, offset)
            )

    def scan_ogg_groups(self) -> list[OggGroup]:
        return scan_ogg_groups(self.path, self.data_start)

    def validate_entry(self, entry: CutsceneEntry) -> OggValidation:
        with open(self.path, "rb") as f:
            return validate_ogg_range(f, entry.offset, entry.size)

    def iter_entries(self, name_contains: Optional[str] = None) -> Iterable[CutsceneEntry]:
        if not name_contains:
            yield from self.entries
            return
        needle = name_contains.lower()
        for entry in self.entries:
            if needle in entry.name.lower():
                yield entry

    def extract_entry(
        self,
        entry: CutsceneEntry,
        out_dir: str,
        *,
        flat: bool = False,
        chunk_size: int = 1024 * 1024,
    ) -> str:
        out_path = output_path_for_entry(out_dir, entry, flat=flat)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(self.path, "rb") as src, open(out_path, "wb") as dst:
            src.seek(entry.offset)
            remaining = entry.size
            while remaining:
                data = src.read(min(chunk_size, remaining))
                if not data:
                    raise EOFError(f"short read while extracting {entry.name}")
                dst.write(data)
                remaining -= len(data)
        shutil.copystat(self.path, out_path, follow_symlinks=True)
        return out_path


def output_path_for_entry(out_dir: str, entry: CutsceneEntry, *, flat: bool) -> str:
    if flat:
        parts = [entry.basename]
    else:
        normalized = entry.name.replace("\\", "/")
        parts = [part for part in normalized.split("/") if part]
    if not parts:
        raise ValueError(f"empty output path for entry {entry.index}")
    for part in parts:
        if part in (".", "..") or ":" in part:
            raise ValueError(f"unsafe path component {part!r} in {entry.name!r}")
    return os.path.join(out_dir, *parts)
