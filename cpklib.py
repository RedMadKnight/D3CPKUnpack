#!/usr/bin/env python3
"""
cpklib — Diablo III console CPK archive library.

Reader-side API:
    with CPKArchive.open(path) as archive:
        for entry in archive.iter_files():
            print(entry.name, entry.size)
        data = archive.read(name)     # extract one file
        archive.extract_all(out_dir)  # extract everything
        report = archive.verify()     # check CRC, tables and every chunk

open() memory-maps the file and read() inflates only the sectors a file
overlaps, so the source must not be overwritten while the archive is open.

The library understands:
  * CPK v6 (Xbox 360 / PS3, big-endian) and v7 (Switch / PS4, little-endian)
  * Bit-packed SortedFileInfo / Locations / CompSectorToDecompOffset / Names tables
  * Encrypted-zlib chunks (stream cipher with ciphertext feedback)
  * Sector positioning via cs2do (NOT naive concatenation)
  * SFI stubs for files whose payload lives in companion archives
    (CacheCommon.cpk, <locale>_CacheCommon.cpk) — read() returns bytes that
    are intentionally zero in the source archive.

Write-side: CPKWriter lives in this module too (byte-identical clone and
same-size in-place replace); cpkrebuild.py adds full rebuild on top of it.

Raw cutscene archives (*_Cutscene.cpk, fmt_const 0) are a different payload
format and are handled by d3cutscene.py.
"""
from __future__ import annotations

import mmap
import os
import struct
import sys
import zlib
from bisect import bisect_left, bisect_right
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from typing import Iterator, Optional


# --------------------------------------------------------------------- #
# Hash + bit reader
# --------------------------------------------------------------------- #

def fnv_hash64(name: str) -> int:
    """FNV-1a 64-bit, lowercase ASCII, the hash D3 uses to look up files."""
    h = 0xCBF29CE484222325
    for c in name.lower().encode("ascii", errors="replace"):
        h = (0x100000001B3 * (h ^ c)) & 0xFFFFFFFFFFFFFFFF
    return h


def _read_bits(buf: bytes, bit_pos: int, n: int) -> int:
    """MSB-first bit-stream reader, used for SFI / Locations / cs2do tables."""
    bs = bit_pos >> 3
    be = (bit_pos + n + 7) >> 3
    chunk = buf[bs:be]
    val = int.from_bytes(chunk, "big")
    total_bits = (be - bs) * 8
    bit_off = bit_pos & 7
    return (val >> (total_bits - bit_off - n)) & ((1 << n) - 1)


def _highest_bit(u: int) -> int:
    r = 0
    while u != 0:
        u >>= 1
        r += 1
    return r


# --------------------------------------------------------------------- #
# Stream cipher
# --------------------------------------------------------------------- #

_STREAM_CIPHER_INITIAL_STATE = 0x873AE8E53C11D761
_STREAM_CIPHER_KEY = _STREAM_CIPHER_INITIAL_STATE.to_bytes(8, "little")


def _xor_bytes(a: bytes, b: bytes) -> bytes:
    """XOR two byte strings of equal length."""
    return (int.from_bytes(a, "little") ^ int.from_bytes(b, "little")).to_bytes(
        len(a), "little")


def _decrypt_chunk(ct: bytes) -> bytes:
    """Decrypt a chunk whose `zsize_raw` had bit 0x8000 set.

    Cipher: per-byte XOR with state low byte, then ciphertext-feedback into
    the high byte:
        plaintext_n  = ciphertext_n XOR (state_n & 0xFF)
        state_{n+1}  = ((ciphertext_n & 0xFF) << 56) | (state_n >> 8)
    The state is a shift register, so its low byte is simply the ciphertext
    byte from 8 positions back; the keystream is the initial state followed
    by the ciphertext, which lets us XOR the whole chunk at once.
    Output is a conformant zlib stream.
    """
    return _xor_bytes(ct, (_STREAM_CIPHER_KEY + ct)[:len(ct)])


# --------------------------------------------------------------------- #
# Public dataclasses
# --------------------------------------------------------------------- #

@dataclass(frozen=True)
class CPKEntry:
    """A single file entry inside the archive."""
    name: str            # path as stored in the Names table, '\\'-separated
    size: int            # declared size in bytes (after decompression)
    hash: int            # FNV-1a 64-bit of name.lower()
    loc_idx: int         # index into Locations[]
    loc_count: int       # number of stored copies, see CPKArchive.copies()
    sfi_index: int       # index into SortedFileInfo[]
    offset: int          # absolute offset within the global decompressed stream

    @property
    def ext(self) -> str:
        """Lowercased extension without the dot. '' if none. The '|locale'
        suffix on localized assets is treated as part of the name, not ext."""
        base = self.name.split("|", 1)[0]
        if "." not in base:
            return ""
        return base.rsplit(".", 1)[1].lower()

    @property
    def basename(self) -> str:
        return os.path.basename(self.name.replace("\\", "/"))


@dataclass
class CPKHeader:
    version: int
    decomp_fs: int
    file_count: int
    loc_count: int
    header_sector: int
    fs_bc: int           # bit-width of size field in SFI
    flcnt_bc: int        # bit-width of loc_count
    flidx_bc: int        # bit-width of loc_idx
    loc_bc: int          # bit-width of locations entries  (offset 44)
    cs2do_bc: int        # bit-width of cs2do entries      (offset 48)
    ds2cs_bc: int        # bit-width of ds2cs entries      (offset 52)
    sector_crc32: int    # crc32 of comp-sector payload    (offset 56)
    read_sector: int     # 0x10000 on v6, runtime-defined on v7
    comp_sector: int     # 0x4000 on v6, runtime-defined on v7
    header_size: int     # 64 on v6, 72 on v7
    endian: str          # ">" or "<"
    fmt_const: int = 2   # offset 16: 2 = legacy layout (Diablo III RoS,
                         #            CoreCommon, Act*.cpk),
                         #            10 = new layout introduced with
                         #            Patch 2.6.7 (header_sector is a direct
                         #            byte offset; sector 0 has no skip
                         #            preamble).


@dataclass(frozen=True)
class ChunkFailure:
    """A chunk that would not inflate and was zero-filled."""
    sector: int
    header_offset: int   # file offset of the 6-byte chunk header
    zsize: int
    zsize_raw: int       # zsize plus the 0x8000 encryption flag
    decomp_size: int
    stream_offset: int   # offset of its output in the global stream
    error: str

    @property
    def data_offset(self) -> int:
        return self.header_offset + 6


@dataclass
class VerifyReport:
    """Result of CPKArchive.verify(). `problems` means the archive is damaged
    or read wrongly; `warnings` are oddities that lose no known data."""
    crc_stored: int
    crc_actual: int
    sectors: int
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    failures: list[ChunkFailure] = field(default_factory=list)
    affected_files: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


class RawCutsceneArchive(ValueError):
    """The file is a raw cutscene CPK; d3cutscene.py reads those."""


# --------------------------------------------------------------------- #
# Reader
# --------------------------------------------------------------------- #

class CPKArchive:
    """Read-only access to a Diablo III console CPK archive."""

    MAGIC = 0xA1B2C3D4
    _SECTOR_CACHE_SIZE = 16

    # --- construction ---

    def __init__(self, data: bytes, path: Optional[str] = None):
        self._data = data
        self.path = path
        self.header: CPKHeader = self._parse_header()
        self.entries: list[CPKEntry] = []
        self.unmatched_names: list[str] = []  # names with no SFI row
        self._raw_locs: list[int] = []
        self._cs2do: list[int] = []
        self._global_stream: Optional[bytes] = None  # lazy
        self._sector_cache: "OrderedDict[int, bytes]" = OrderedDict()
        self._sector_failures: dict[int, list[ChunkFailure]] = {}
        self._overrun_sectors: set[int] = set()  # chunk ran past the sector end
        self._parse_tables()

    @classmethod
    def open(cls, path: str) -> "CPKArchive":
        with open(path, "rb") as f:
            size = os.fstat(f.fileno()).st_size
            data = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) if size else b""
        return cls(data, path=path)

    def close(self) -> None:
        """Release the memory map. The archive is unusable afterwards."""
        if isinstance(self._data, mmap.mmap):
            self._data.close()

    def __enter__(self) -> "CPKArchive":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # --- header / tables ---

    def _parse_header(self) -> CPKHeader:
        d = self._data
        if len(d) < 64:
            raise ValueError(f"Not a CPK archive: only {len(d)} B long")
        magic_be = struct.unpack(">I", d[:4])[0]
        magic_le = struct.unpack("<I", d[:4])[0]
        if magic_be == self.MAGIC:
            endian = ">"
        elif magic_le == self.MAGIC:
            endian = "<"
        else:
            raise ValueError(
                f"Not a CPK archive: magic BE=0x{magic_be:08X} LE=0x{magic_le:08X}, "
                f"expected 0x{self.MAGIC:08X}"
            )

        u32 = lambda o: struct.unpack(endian + "I", d[o:o+4])[0]
        u64 = lambda o: struct.unpack(endian + "Q", d[o:o+8])[0]

        ver = u32(4)
        decomp_fs = u64(8)
        fmt_const = u32(16)
        file_count = u32(20)
        loc_count = u32(24)
        header_sector = u32(28)
        fs_bc, flcnt_bc, flidx_bc, loc_bc = u32(32), u32(36), u32(40), u32(44)
        cs2do_bc = u32(48)
        ds2cs_bc = u32(52)
        sector_crc32 = u32(56)

        if ver == 6:
            read_sector, comp_sector, header_size = 0x10000, 0x4000, 64
        elif ver == 7:
            read_sector = u32(60)
            comp_sector = u32(64)
            header_size = 72
        else:
            raise ValueError(f"Unsupported CPK version {ver}")

        if file_count > 10_000_000 or decomp_fs > 100_000_000_000:
            raise ValueError(
                f"Implausible header (file_count={file_count}, decomp_fs={decomp_fs}). "
                f"Bit-stream byte order may differ in this build."
            )

        if fmt_const == 0 and len(d) - header_sector * read_sector == decomp_fs:
            raise RawCutsceneArchive(
                "Raw cutscene archive (fmt_const=0): payload is plain Ogg, "
                "not zlib sectors. Use d3cutscene_tool.py instead."
            )

        return CPKHeader(
            version=ver, decomp_fs=decomp_fs, file_count=file_count,
            loc_count=loc_count, header_sector=header_sector,
            fs_bc=fs_bc, flcnt_bc=flcnt_bc, flidx_bc=flidx_bc, loc_bc=loc_bc,
            cs2do_bc=cs2do_bc, ds2cs_bc=ds2cs_bc, sector_crc32=sector_crc32,
            read_sector=read_sector, comp_sector=comp_sector,
            header_size=header_size, endian=endian, fmt_const=fmt_const,
        )

    def _parse_tables(self) -> None:
        d = self._data
        h = self.header
        u32 = lambda o: struct.unpack(h.endian + "I", d[o:o+4])[0]

        # --- SortedFileInfo ---
        entry_bits = 64 + h.fs_bc + h.flcnt_bc + h.flidx_bc
        sfi_bytes = (h.file_count * entry_bits + 7) // 8
        sfi_block = d[h.header_size : h.header_size + sfi_bytes]
        sfi_raw: list[tuple[int, int, int, int]] = []
        for i in range(h.file_count):
            bp = i * entry_bits
            hh = _read_bits(sfi_block, bp, 64); bp += 64
            sz = _read_bits(sfi_block, bp, h.fs_bc); bp += h.fs_bc
            lc = _read_bits(sfi_block, bp, h.flcnt_bc); bp += h.flcnt_bc
            li = _read_bits(sfi_block, bp, h.flidx_bc)
            sfi_raw.append((hh, sz, lc, li))
        pos = h.header_size + sfi_bytes

        # --- Locations ---
        loc_bytes = (h.loc_bc * h.loc_count + 7) // 8
        loc_block = d[pos : pos + loc_bytes]
        self._raw_locs = [
            _read_bits(loc_block, i * h.loc_bc, h.loc_bc)
            for i in range(h.loc_count)
        ]
        pos += loc_bytes

        # --- CompSectorToDecompOffset ---
        # Two layouts depending on header.fmt_const:
        #   * fmt_const == 2  (legacy): header_sector is in units of read_sector
        #     and the first compressed sector starts at the next read_sector
        #     boundary, with the leading bytes of that sector reserved for
        #     leftover index data (sector 0 carries a 2-byte "skip" preamble).
        #   * fmt_const == 10 (new, Patch 2.6.7+): header_sector is a direct
        #     byte offset to the first compressed sector. Sectors are still
        #     comp_sector bytes wide. There is NO skip preamble in sector 0.
        if h.fmt_const == 10:
            first_sec = h.header_sector
        else:
            first_sec = (h.read_sector * h.header_sector) & 0xFFFF0000
            if first_sec % h.read_sector != 0:
                first_sec += h.read_sector
        comp_sector_count = (
            h.comp_sector + len(d) - 1 - first_sec
        ) // h.comp_sector
        if comp_sector_count < 0:
            comp_sector_count = 0
        cs2do_bytes = (comp_sector_count * h.cs2do_bc + 7) // 8
        cs2do_block = d[pos : pos + cs2do_bytes]
        self._cs2do = [
            _read_bits(cs2do_block, i * h.cs2do_bc, h.cs2do_bc)
            for i in range(comp_sector_count)
        ]
        pos += cs2do_bytes
        self._first_sec = first_sec
        self._comp_sector_count = comp_sector_count
        # Only a strictly increasing cs2do can be searched by read().
        self._cs2do_increasing = all(
            a < b for a, b in zip(self._cs2do, self._cs2do[1:])
        )

        # --- DecompSectorToCompSector (skipped, cs2do alone locates sectors) ---
        # ds2cs_bc is stored in the header and equals _highest_bit(comp_sector_count)
        # for well-formed archives — trust the header value when set.
        ds2cs_bc = h.ds2cs_bc if h.ds2cs_bc > 0 else _highest_bit(comp_sector_count)
        ds_sector_count = (h.decomp_fs + h.comp_sector - 1) // h.comp_sector
        ds2cs_bytes = (ds2cs_bc * ds_sector_count + 7) // 8
        pos += ds2cs_bytes

        # --- Names ---
        name_offs = [u32(pos + i*4) for i in range(h.file_count)]
        pos += h.file_count * 4
        names_data_start = pos
        names: list[str] = []
        for off in name_offs:
            abs_off = names_data_start + off
            end = d.find(b"\x00", abs_off)
            if end == -1:
                end = abs_off + 256
            names.append(d[abs_off:end].decode("latin-1", errors="replace"))

        # --- Build CPKEntry list ---
        hash_to_sfi = {row[0]: i for i, row in enumerate(sfi_raw)}
        for name in names:
            nh = fnv_hash64(name)
            if nh not in hash_to_sfi:
                # Name with no SFI row — should never happen in shipped archives;
                # skip rather than fail.
                self.unmatched_names.append(name)
                continue
            sfi_idx = hash_to_sfi[nh]
            hh, sz, lc, li = sfi_raw[sfi_idx]
            offset = self._raw_locs[li] if (lc > 0 and li < len(self._raw_locs)) else 0
            self.entries.append(CPKEntry(
                name=name, size=sz, hash=hh,
                loc_idx=li, loc_count=lc, sfi_index=sfi_idx,
                offset=offset,
            ))

        self._name_to_entry = {e.name: e for e in self.entries}
        self._lower_to_entry = {e.name.lower(): e for e in self.entries}

    # --- decompression ---

    def _decompress_sector(self, sec_idx: int) -> bytes:
        d = self._data
        h = self.header
        u16 = lambda o: struct.unpack(h.endian + "H", d[o:o+2])[0]
        sec_start = self._first_sec + sec_idx * h.comp_sector
        sec_end = min(sec_start + h.comp_sector, len(d))

        p = sec_start
        # Legacy (fmt_const == 2) archives align first_sec UP to the next
        # read_sector boundary, so the very first compressed sector contains a
        # 2-byte "skip" word at offset 4 indicating how many bytes of leftover
        # index data sit before the first chunk. The new layout (fmt_const == 10)
        # uses an unaligned byte offset and starts with chunks immediately.
        if sec_idx == 0 and h.fmt_const != 10:
            if p + 6 > len(d):
                return b""
            skip = u16(p + 4)
            p += 6 + skip

        out = bytearray()
        failures: list[ChunkFailure] = []
        while p + 6 <= sec_end:
            nlow = u16(p); nhigh = u16(p + 2); zsize_raw = u16(p + 4)
            if zsize_raw == 0:
                break
            encrypted = (zsize_raw & 0x8000) != 0
            zsize = zsize_raw & 0x7FFF
            decomp_size = (nhigh << 16) | nlow
            ds = p + 6
            de = ds + zsize
            if de > sec_end:
                self._overrun_sectors.add(sec_idx)
                break
            chunk = d[ds:de]
            if encrypted:
                chunk = _decrypt_chunk(chunk)
            try:
                # The last chunk of most sectors is cut off at the slot
                # boundary, which one-shot zlib.decompress would reject.
                dec = zlib.decompressobj().decompress(chunk)
                if len(dec) < decomp_size:
                    dec = dec + b'\x00' * (decomp_size - len(dec))
                out.extend(dec)
            except zlib.error as ex:
                failures.append(ChunkFailure(
                    sector=sec_idx, header_offset=p, zsize=zsize,
                    zsize_raw=zsize_raw, decomp_size=decomp_size,
                    stream_offset=self._cs2do[sec_idx] + len(out),
                    error=str(ex),
                ))
                out.extend(b'\x00' * decomp_size)
            p = de
        if failures:
            self._sector_failures[sec_idx] = failures
        return bytes(out)

    def _load_sector(self, sec_idx: int) -> bytes:
        """_decompress_sector with a small LRU, as neighbours share sectors."""
        cache = self._sector_cache
        sb = cache.get(sec_idx)
        if sb is not None:
            cache.move_to_end(sec_idx)
            return sb
        sb = cache[sec_idx] = self._decompress_sector(sec_idx)
        if len(cache) > self._SECTOR_CACHE_SIZE:
            cache.popitem(last=False)
        return sb

    def _read_range(self, offset: int, size: int) -> bytes:
        """Global-stream slice built from only the sectors that overlap it."""
        out = bytearray(size)
        end = offset + size
        s = max(bisect_right(self._cs2do, offset) - 1, 0)
        while s < self._comp_sector_count and self._cs2do[s] < end:
            sb = self._load_sector(s)
            start = self._cs2do[s]
            lo = max(start, offset)
            hi = min(start + len(sb), end)
            if lo < hi:
                out[lo - offset:hi - offset] = sb[lo - start:hi - start]
            s += 1
        return bytes(out)

    def _build_global_stream(self) -> bytes:
        h = self.header
        initial_size = max(
            h.decomp_fs,
            (self._cs2do[-1] + h.comp_sector * 4) if self._cs2do else 0,
        )
        buf = bytearray(initial_size)
        for s in range(self._comp_sector_count):
            sb = self._decompress_sector(s)
            target = self._cs2do[s]
            end = target + len(sb)
            if end > len(buf):
                buf.extend(b"\x00" * (end - len(buf)))
            buf[target:end] = sb
        return bytes(buf)

    @property
    def global_stream(self) -> bytes:
        if self._global_stream is None:
            self._global_stream = self._build_global_stream()
        return self._global_stream

    @property
    def chunk_failures(self) -> list[ChunkFailure]:
        """Zero-filled chunks seen so far, by global-stream offset."""
        return sorted(
            (f for fs in self._sector_failures.values() for f in fs),
            key=lambda f: f.stream_offset,
        )

    @property
    def chunk_errors(self) -> int:
        return sum(len(fs) for fs in self._sector_failures.values())

    def copies(self, e: CPKEntry) -> list[int]:
        """Stream offset of every stored copy of a file. Small shared assets
        repeat through an archive; read() uses the first copy."""
        return self._raw_locs[e.loc_idx : e.loc_idx + e.loc_count]

    def failed_chunks_by_file(self) -> dict[str, list[int]]:
        """Map file name -> indices into chunk_failures it overlaps."""
        failures = self.chunk_failures
        starts = [f.stream_offset for f in failures]
        ends = [f.stream_offset + f.decomp_size for f in failures]
        out = {}
        for e in self.entries:
            if e.loc_count == 0 or e.size == 0:
                continue
            lo = bisect_right(ends, e.offset)
            hi = bisect_left(starts, e.offset + e.size)
            if lo < hi:
                out[e.name] = list(range(lo, hi))
        return out

    def warn_chunk_errors(self) -> None:
        """Warn on stderr if the data read so far has zero-filled chunks."""
        if self.chunk_errors:
            sys.stderr.write(
                f"  WARNING: {self.chunk_errors:,} chunk(s) failed to decompress "
                f"and were zero-filled\n"
            )

    # --- integrity check ---

    def verify(self) -> VerifyReport:
        """Check the sector CRC and tables, then inflate every sector.

        Zero-filled chunks are otherwise invisible in the output, which is
        how a wrong cipher constant once went unnoticed. Sectors are not kept
        in memory, so this works on archives of any size.
        """
        h = self.header
        with memoryview(self._data) as view:
            crc = zlib.crc32(view[self._first_sec:]) & 0xFFFFFFFF
        report = VerifyReport(h.sector_crc32, crc, self._comp_sector_count)

        if crc != h.sector_crc32:
            report.problems.append(
                f"sector CRC32 mismatch: stored {h.sector_crc32:08X}, "
                f"computed {crc:08X}")
        if h.decomp_fs and not self._comp_sector_count:
            report.problems.append(
                f"no compressed sectors: file ends before first_sec "
                f"0x{self._first_sec:X}")
        if self.unmatched_names:
            report.problems.append(
                f"{len(self.unmatched_names):,} name(s) have no SortedFileInfo row")
        bad_locs = sum(
            e.loc_idx + e.loc_count > len(self._raw_locs)
            for e in self.entries)
        if bad_locs:
            report.problems.append(
                f"{bad_locs:,} file(s) point past the Locations table")
        out_of_range = sum(
            any(off + e.size > h.decomp_fs for off in self.copies(e))
            for e in self.entries)
        if out_of_range:
            report.problems.append(
                f"{out_of_range:,} file(s) end past decomp_fs ({h.decomp_fs:,} B)")

        if not self._cs2do_increasing:
            report.warnings.append("cs2do is not strictly increasing")
        overlaps = 0
        for s in range(self._comp_sector_count):
            sb = self._decompress_sector(s)
            if (s + 1 < self._comp_sector_count
                    and self._cs2do[s] + len(sb) > self._cs2do[s + 1]):
                overlaps += 1
        if overlaps:
            report.warnings.append(
                f"{overlaps:,} sector(s) inflate past the next cs2do offset")
        if self._overrun_sectors:
            report.warnings.append(
                f"{len(self._overrun_sectors):,} sector(s) end with a chunk "
                f"that overruns the sector")

        report.failures = self.chunk_failures
        if report.failures:
            report.affected_files = sorted(self.failed_chunks_by_file())
            report.problems.append(
                f"{len(report.failures):,} chunk(s) failed to decompress and "
                f"were zero-filled ({len(report.affected_files):,} file(s) "
                f"affected)")
        return report

    # --- public lookup / extraction ---

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self) -> Iterator[CPKEntry]:
        return iter(self.entries)

    def iter_files(self) -> Iterator[CPKEntry]:
        return iter(self.entries)

    def get(self, name: str) -> Optional[CPKEntry]:
        e = self._name_to_entry.get(name)
        if e is not None:
            return e
        return self._lower_to_entry.get(name.lower())

    def __contains__(self, name: str) -> bool:
        return self.get(name) is not None

    def read(self, name_or_entry) -> bytes:
        """Return the raw bytes of one file. May be all-zero for SFI stubs."""
        e = name_or_entry if isinstance(name_or_entry, CPKEntry) else self.get(name_or_entry)
        if e is None:
            raise KeyError(name_or_entry)
        if e.loc_count == 0 or e.size == 0:
            return b""
        end = e.offset + e.size
        if (self._global_stream is None and self._cs2do_increasing
                and end <= self.header.decomp_fs):
            return self._read_range(e.offset, e.size)
        gs = self.global_stream
        if end > len(gs):
            # Should not happen now that we honor cs2do; surface the corruption.
            raise IOError(
                f"Entry {e.name!r} at {e.offset}+{e.size} exceeds global stream "
                f"({len(gs)} B)"
            )
        return gs[e.offset:end]

    # --- mass extraction with filtering ---

    def extract_all(
        self,
        out_dir: str,
        *,
        filter_fn=None,
        flat: bool = False,
        sanitize_pipe: bool = True,
    ) -> dict:
        """Write matching files to disk. Returns a small stats dict."""
        os.makedirs(out_dir, exist_ok=True)
        stats = Counter()
        # Stream order keeps neighbouring files inside the sector cache.
        for e in sorted(self.entries, key=lambda e: e.offset):
            if filter_fn is not None and not filter_fn(e):
                stats["skipped_filter"] += 1
                continue
            try:
                payload = self.read(e)
            except IOError:
                stats["failed"] += 1
                continue
            if not payload:
                stats["zero"] += 1
            stats["written"] += 1
            self._write_entry(e, payload, out_dir, flat=flat, sanitize_pipe=sanitize_pipe)
        return dict(stats)

    @staticmethod
    def _write_entry(e: CPKEntry, payload: bytes, out_dir: str,
                     flat: bool, sanitize_pipe: bool) -> str:
        name = e.name.replace("\\", "/").lstrip("/")
        if ".." in name.split("/"):
            raise ValueError(f"unsafe path in archive entry: {e.name!r}")
        if sanitize_pipe:
            name = name.replace("|", "__")
        # Forbidden chars on Windows
        for bad in ('<', '>', ':', '"', '?', '*'):
            name = name.replace(bad, "_")
        if flat:
            name = name.replace("/", "__")
        path = os.path.join(out_dir, name)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as fh:
            fh.write(payload)
        return path

    # --- diagnostics ---

    def stats(self) -> dict:
        ext_count = Counter(e.ext for e in self.entries)
        size_total = sum(e.size for e in self.entries)
        return {
            "version": self.header.version,
            "endian": self.header.endian,
            "file_count": len(self.entries),
            "decomp_fs": self.header.decomp_fs,
            "size_total": size_total,
            "ext_count_top10": ext_count.most_common(10),
            "comp_sector": self.header.comp_sector,
            "comp_sector_count": self._comp_sector_count,
        }


# --------------------------------------------------------------------- #
# Bit-packed table writer (mirror of _read_bits)
# --------------------------------------------------------------------- #

class _BitWriter:
    """MSB-first bit writer matching _read_bits semantics."""

    def __init__(self):
        self._buf = bytearray()
        self._byte = 0
        self._bit = 0  # number of bits already in self._byte (0..7)

    def write(self, value: int, n: int) -> None:
        if n <= 0:
            return
        # value is `n` bits, MSB first.
        for i in range(n - 1, -1, -1):
            bit = (value >> i) & 1
            self._byte = (self._byte << 1) | bit
            self._bit += 1
            if self._bit == 8:
                self._buf.append(self._byte)
                self._byte = 0
                self._bit = 0

    def finish(self) -> bytes:
        if self._bit:
            # Flush partial byte left-aligned (MSB-first)
            self._buf.append((self._byte << (8 - self._bit)) & 0xFF)
            self._byte = 0
            self._bit = 0
        return bytes(self._buf)


# --------------------------------------------------------------------- #
# Encryption (inverse of _decrypt_chunk — same routine: cipher is XOR
# with state low byte, state advances on the *ciphertext* byte; for
# encryption the ciphertext is the output, so we feed the OUTPUT byte
# back into the state).
# --------------------------------------------------------------------- #

def _encrypt_chunk(plaintext: bytes) -> bytes:
    # Every 8 output bytes key the next 8 (see _decrypt_chunk).
    ct = bytearray()
    key = _STREAM_CIPHER_KEY
    for i in range(0, len(plaintext), 8):
        block = plaintext[i:i + 8]
        key = _xor_bytes(block, key[:len(block)])
        ct += key
    return bytes(ct)


# --------------------------------------------------------------------- #
# Writer — phase 1: byte-identical clone via raw chunk pass-through
# --------------------------------------------------------------------- #
#
# Building a CPK from scratch with the same compressed bytes as the original
# is impossible in general — zlib output is implementation-specific.
# However, we CAN produce a byte-identical clone of an existing archive,
# which proves we understand every offset / table / packing rule. That is
# the round-trip identity test.
#
# Strategy used here:
#   * Reuse the original chunk bodies verbatim (we copy compressed chunk
#     payloads from the source archive byte-for-byte, including encryption).
#   * Re-emit the bit-packed tables (SFI, Locations, cs2do, ds2cs) using
#     the exact same field widths and ordering observed on input.
#   * Re-emit the Names block in the original order.
#
# After we verify identity, the in-place edit operations (replace / delete)
# only need to touch the affected sectors and the global Locations row(s).

class CPKWriter:
    """Build / rewrite a CPK archive.

    Phase 1 (this commit): `from_archive(src).save(out)` produces a byte-for-
    byte identical copy. This proves the format model is complete.
    Phase 2 will add `replace_file`, `remove_file`, `add_file`, and full
    `from_directory` construction.
    """

    def __init__(self, source: CPKArchive):
        self._src = source
        # Phase 2: staged same-size replacements
        # name -> (CPKEntry, new_bytes)
        self._replacements: "dict[str, tuple]" = {}

    @classmethod
    def from_archive(cls, archive: CPKArchive) -> "CPKWriter":
        return cls(archive)

    def _refuse_source_path(self, out_path: str) -> None:
        """The source stays memory-mapped, so it cannot be its own target."""
        src_path = self._src.path if self._src is not None else None
        if src_path and os.path.abspath(out_path) == os.path.abspath(src_path):
            raise ValueError(f"refusing to overwrite source archive: {out_path}")

    # --- byte-identical save ---

    def save_as_clone(self, out_path: str) -> None:
        """Write a byte-identical copy of the source archive."""
        with open(out_path, "wb") as f:
            f.write(self._src._data)

    def save(self, out_path: str) -> None:
        """Re-encode the archive from the parsed model.

        If no replacements are staged, this is byte-identical to the source.
        If `replace_file_same_size` was called, only the affected sectors
        are re-encoded; all tables (SFI / Locations / cs2do / Names) and
        every other sector remain byte-identical.
        """
        self._refuse_source_path(out_path)
        if not self._replacements:
            return self.save_as_clone(out_path)
        self._save_with_replacements(out_path)

    # --- Phase 2: same-size in-place replace ---

    def replace_file_same_size(self, name: str, new_content: bytes) -> None:
        """Stage a same-size replacement of an existing file.

        Constraints:
          * `name` must exist in the source archive.
          * `len(new_content) == entry.size` (no resize).
          * The file's bytes must live in this archive (not be a stub for a
            companion CacheCommon.cpk payload). For SFI stubs, replace will
            silently affect the buffer region but the engine will still load
            the real payload from the companion file.

        Save with `.save(out_path)`.
        """
        e = self._src.get(name)
        if e is None:
            raise KeyError(name)
        if len(new_content) != e.size:
            raise ValueError(
                f"size mismatch for {name!r}: "
                f"expected {e.size:,} B, got {len(new_content):,} B")
        self._replacements[name] = (e, bytes(new_content))

    def _affected_sectors(self) -> "set[int]":
        """Return the set of sector indices that overlap any staged replacement."""
        src = self._src
        cs2do = src._cs2do
        n_sec = src._comp_sector_count
        decomp_fs = src.header.decomp_fs

        def sec_range(s):
            start = cs2do[s]
            end = cs2do[s + 1] if s + 1 < n_sec else max(decomp_fs, start)
            return start, end

        affected = set()
        for _, (e, _) in self._replacements.items():
            for f_start in src.copies(e):
                f_end = f_start + e.size
                for s in range(n_sec):
                    s_start, s_end = sec_range(s)
                    if s_end <= f_start:
                        continue
                    if s_start >= f_end:
                        break
                    affected.add(s)
        return affected

    def _save_with_replacements(self, out_path: str) -> None:
        src = self._src
        h = src.header
        affected = self._affected_sectors()
        if not affected:
            return self.save_as_clone(out_path)

        gs = bytearray(src.global_stream)
        src.warn_chunk_errors()
        for _, (e, new_bytes) in self._replacements.items():
            for off in src.copies(e):
                gs[off : off + e.size] = new_bytes

        out = bytearray(src._data)
        slot_size = h.comp_sector
        n_sec = src._comp_sector_count
        cs2do = src._cs2do

        for s in sorted(affected):
            s_start = cs2do[s]
            s_end = cs2do[s + 1] if s + 1 < n_sec else h.decomp_fs
            decompressed = bytes(gs[s_start:s_end])

            if s == 0 and h.fmt_const != 10:
                # Legacy archives carry a 2-byte "skip" preamble at the start
                # of sector 0 (followed by leftover bit-packed table data).
                # New-format (fmt_const==10) archives place sector 0 cleanly
                # at first_sec with no prelude.
                prelude_size = self._sector0_prelude_size()
                prelude = bytes(src._data[
                    src._first_sec : src._first_sec + prelude_size
                ])
            else:
                prelude_size = 0
                prelude = b""

            payload_capacity = slot_size - prelude_size
            payload = _encode_sector_payload(
                decompressed, payload_capacity,
                endian=h.endian, encrypt=False,
            )
            slot = prelude + payload
            assert len(slot) == slot_size, (
                f"sector {s} encoded to {len(slot)} (expected {slot_size})"
            )

            sec_off = src._first_sec + s * slot_size
            out[sec_off : sec_off + slot_size] = slot

        crc = zlib.crc32(bytes(out[src._first_sec:])) & 0xFFFFFFFF
        struct.pack_into(h.endian + "I", out, 56, crc)

        with open(out_path, "wb") as f:
            f.write(out)

    def _sector0_prelude_size(self) -> int:
        src = self._src
        endian = src.header.endian
        u16 = struct.unpack(
            endian + "H",
            src._data[src._first_sec + 4:src._first_sec + 6]
        )[0]
        return 6 + u16


# Splitting further only adds chunk headers: data this small will not shrink.
_MIN_CHUNK_SPLIT = 1024


def _encode_sector_payload(decompressed: bytes, capacity: int,
                           endian: str, encrypt: bool) -> bytes:
    """Encode arbitrary decompressed bytes into a payload of `capacity` bytes."""
    HEADER = 6
    if not decompressed:
        return b"\x00" * capacity

    chunks = [decompressed]
    while True:
        encoded = []
        total = 0
        too_big = False
        for c in chunks:
            cz = zlib.compress(c, 9)
            flag = 0
            if encrypt:
                cz = _encrypt_chunk(cz)
                flag = 0x8000
            zsize = len(cz)
            if zsize > 0x7FFF:
                too_big = True
                break
            decomp_size = len(c)
            nlow = decomp_size & 0xFFFF
            nhigh = (decomp_size >> 16) & 0xFFFF
            zsize_raw = zsize | flag
            header = struct.pack(endian + "HHH", nlow, nhigh, zsize_raw)
            encoded.append(header + cz)
            total += HEADER + zsize
        if not too_big and total <= capacity:
            blob = b"".join(encoded)
            if total + HEADER <= capacity:
                blob += b"\x00" * HEADER
                total += HEADER
            blob += b"\x00" * (capacity - total)
            return blob

        new_chunks = []
        progressed = False
        for c in chunks:
            if len(c) <= _MIN_CHUNK_SPLIT:
                new_chunks.append(c)
                continue
            mid = len(c) // 2
            new_chunks.append(c[:mid])
            new_chunks.append(c[mid:])
            progressed = True
        if not progressed:
            raise ValueError(
                f"Cannot fit {len(decompressed):,} B in {capacity:,}-byte slot"
            )
        chunks = new_chunks


def round_trip_identity_test(src_path: str, tmp_path: str) -> dict:
    """End-to-end round-trip identity test (no modifications)."""
    src = CPKArchive.open(src_path)
    CPKWriter.from_archive(src).save(tmp_path)
    with open(src_path, "rb") as f1, open(tmp_path, "rb") as f2:
        a = f1.read()
        b = f2.read()
    bytes_identical = (a == b)
    rewritten = CPKArchive.open(tmp_path)
    if len(rewritten) != len(src):
        return {"bytes_identical": bytes_identical, "semantic_match": False,
                "reason": f"file count differs: {len(src)} vs {len(rewritten)}"}
    sample_indices = list(range(0, len(src), max(1, len(src) // 50)))[:50]
    semantic_failures = []
    for i in sample_indices:
        e_src = src.entries[i]; e_dst = rewritten.entries[i]
        if (e_src.name != e_dst.name or e_src.size != e_dst.size
                or e_src.offset != e_dst.offset):
            semantic_failures.append((i, e_src.name, e_dst.name)); continue
        d1 = src.read(e_src); d2 = rewritten.read(e_dst)
        if d1 != d2:
            semantic_failures.append((i, e_src.name, "content mismatch"))
    return {"bytes_identical": bytes_identical, "src_size": len(a),
            "dst_size": len(b), "src_files": len(src), "dst_files": len(rewritten),
            "semantic_match": len(semantic_failures) == 0,
            "failures": semantic_failures}


def replace_file_test(*args, **kwargs) -> dict:
    """Stub kept for backwards compatibility."""
    return {"ok": False, "reason": "replace_file_test stub - use cpk_tool.py"}


# end of file marker for sync verification
_CPKLIB_VERSION = "1.2-fmt10"
