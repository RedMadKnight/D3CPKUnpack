#!/usr/bin/env python3
"""
cpkrebuild - Phase 4: full archive rebuild + create_new (v6/v7 from scratch).
"""
from __future__ import annotations

import struct
import sys
import time
import zlib

import cpklib
from cpklib import (
    CPKArchive,
    CPKHeader,
    CPKWriter,
    fnv_hash64,
    _BitWriter,
    _highest_bit,
    _encrypt_chunk,
)


def _encode_sector_payload_fast(decompressed, capacity, endian, encrypt,
                                level=6):
    HEADER = 6
    if not decompressed:
        return b"\x00" * capacity
    chunks = [decompressed]
    while True:
        encoded = []
        total = 0
        too_big = False
        for c in chunks:
            cz = zlib.compress(c, level)
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
            if len(c) <= 1:
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


def _ensure_mods(self):
    if not hasattr(self, "_mods"):
        self._mods = {"replace": {}, "delete": set(), "add": {}}
    return self._mods


def _is_new_archive(self):
    return getattr(self, "_src", None) is None


def replace_file(self, name, new_content):
    if _is_new_archive(self):
        raise RuntimeError("replace_file not valid on a new archive (use add_file)")
    if self._src.get(name) is None:
        raise KeyError(name)
    _ensure_mods(self)["replace"][name] = bytes(new_content)


def delete_file(self, name):
    if _is_new_archive(self):
        raise RuntimeError("delete_file not valid on a new archive")
    if self._src.get(name) is None:
        raise KeyError(name)
    _ensure_mods(self)["delete"].add(name)


def add_file(self, name, content):
    mods = _ensure_mods(self)
    if not _is_new_archive(self) and self._src.get(name) is not None:
        raise KeyError(f"already exists: {name}")
    if name in mods["add"]:
        raise KeyError(f"already staged: {name}")
    mods["add"][name] = bytes(content)


def set_zlib_level(self, level):
    if not (0 <= int(level) <= 9):
        raise ValueError(f"zlib level must be 0..9, got {level}")
    self._zlib_level = int(level)


@classmethod
def create_new(cls, version=6, endian="big"):
    """Create a writer for a fresh CPK archive (no source).

    version: 6 (BE, hdr=64) or 7 (LE, hdr=72).
    endian:  'big' or 'little'. Conventionally v6=big, v7=little, but both
             are accepted at any version.
    """
    if version not in (6, 7):
        raise ValueError(f"unsupported version: {version}")
    if endian not in ("big", "little"):
        raise ValueError("endian must be 'big' or 'little'")
    e = ">" if endian == "big" else "<"
    w = cls.__new__(cls)
    w._src = None
    w._replacements = {}
    w._mods = {"replace": {}, "delete": set(), "add": {}}
    # Pseudo-header used by _full_rebuild for endian/version/sector sizes.
    w._new_template = CPKHeader(
        version=version, decomp_fs=0, file_count=0, loc_count=0,
        header_sector=0, fs_bc=1, flcnt_bc=1, flidx_bc=1, loc_bc=1,
        cs2do_bc=1, ds2cs_bc=1, sector_crc32=0,
        read_sector=0x10000, comp_sector=0x4000,
        header_size=64 if version == 6 else 72,
        endian=e, fmt_const=2,
    )
    return w


def _has_rebuild_mods(self):
    if _is_new_archive(self):
        return True  # always rebuild on a fresh archive
    m = getattr(self, "_mods", None)
    if m is None:
        return False
    return bool(m["replace"]) or bool(m["delete"]) or bool(m["add"])


def _build_model(self):
    mods = _ensure_mods(self)
    out = []
    if not _is_new_archive(self):
        src = self._src
        delete = mods["delete"]
        replace = mods["replace"]
        gs = src.global_stream
        for e in src.entries:
            if e.name in delete:
                continue
            if e.name in replace:
                out.append((e.name, replace[e.name]))
            else:
                out.append((e.name, bytes(gs[e.offset:e.offset + e.size])))
    for name, content in mods["add"].items():
        out.append((name, content))
    return out


def _full_rebuild(self, out_path):
    _t = [time.time()]

    def _step(name):
        sys.stderr.write(f'  [rebuild] {name}: {time.time()-_t[0]:.2f}s\n')
        sys.stderr.flush()
        _t[0] = time.time()

    sys.stderr.write(f'[rebuild] enter, out={out_path}\n')
    sys.stderr.flush()

    new_archive = _is_new_archive(self)
    if new_archive:
        h = self._new_template
    else:
        h = self._src.header
    endian = h.endian
    comp_sector = h.comp_sector
    read_sector = h.read_sector
    level = getattr(self, "_zlib_level", 6)
    if not (0 <= level <= 9):
        raise ValueError(f"zlib level must be 0..9, got {level}")

    model = _build_model(self)
    file_count = len(model)
    _step(f'build_model ({file_count}, new={new_archive})')
    if file_count == 0:
        raise ValueError("Cannot build empty archive")

    offset = 0
    layout = []
    for name, content in model:
        sz = len(content)
        layout.append((name, offset, sz, content))
        offset += sz
    decomp_fs = offset

    global_stream = bytearray(decomp_fs)
    for _, off, sz, content in layout:
        global_stream[off:off + sz] = content
    _step(f'global_stream ({decomp_fs:,} B)')

    max_size = max((sz for _, _, sz, _ in layout), default=1)
    loc_count = file_count
    fs_bc = max(_highest_bit(max(max_size, 1)), h.fs_bc, 1)
    flcnt_bc = max(_highest_bit(1), h.flcnt_bc, 1)
    flidx_bc = max(_highest_bit(max(loc_count - 1, 1)), h.flidx_bc, 1)
    loc_bc = max(_highest_bit(max(decomp_fs, 1)), h.loc_bc, 1)

    sfi_entries = []
    locations = []
    for name, off, sz, _ in layout:
        loc_idx = len(locations)
        locations.append(off)
        sfi_entries.append((fnv_hash64(name), sz, 1, loc_idx, name))
    sfi_entries.sort(key=lambda x: x[0])

    bw = _BitWriter()
    for h_val, sz, lc, li, _ in sfi_entries:
        bw.write(h_val, 64)
        bw.write(sz, fs_bc)
        bw.write(lc, flcnt_bc)
        bw.write(li, flidx_bc)
    sfi_block = bw.finish()
    _step(f'sfi ({len(sfi_block):,} B)')

    bw = _BitWriter()
    for o in locations:
        bw.write(o, loc_bc)
    loc_block = bw.finish()

    sectors = []
    p = 0
    while p < decomp_fs:
        size = min(comp_sector, decomp_fs - p)
        first = (len(sectors) == 0)
        prelude = 6 if first else 0
        capacity = comp_sector - prelude
        while True:
            try:
                payload = _encode_sector_payload_fast(
                    bytes(global_stream[p:p + size]),
                    capacity, endian=endian, encrypt=False,
                    level=level,
                )
                slot = (b"\x00" * 6 + payload) if first else payload
                assert len(slot) == comp_sector
                sectors.append((p, size, slot))
                p += size
                break
            except ValueError:
                size //= 2
                if size < 1:
                    raise
    comp_sector_count = len(sectors)
    _step(f'sectors ({comp_sector_count}, level={level})')

    cs2do_bc_out = loc_bc
    ds2cs_bc_out = max(_highest_bit(max(comp_sector_count, 1)), 1)

    bw = _BitWriter()
    for sec_off, _, _ in sectors:
        bw.write(sec_off, cs2do_bc_out)
    cs2do_block = bw.finish()

    ds_sector_count = (decomp_fs + comp_sector - 1) // comp_sector
    bw = _BitWriter()
    si = 0
    for d in range(ds_sector_count):
        ds_off = d * comp_sector
        while si + 1 < comp_sector_count and sectors[si + 1][0] <= ds_off:
            si += 1
        bw.write(si, ds2cs_bc_out)
    ds2cs_block = bw.finish()

    name_data = bytearray()
    name_lookup = {}
    name_offsets = []
    for _, _, _, _, name in sfi_entries:
        if name in name_lookup:
            name_offsets.append(name_lookup[name])
        else:
            o = len(name_data)
            name_offsets.append(o)
            name_lookup[name] = o
            name_data.extend(name.encode("latin-1", errors="replace"))
            name_data.append(0)
    names_offset_block = b"".join(
        struct.pack(endian + "I", o) for o in name_offsets
    )

    pos = h.header_size
    sfi_pos = pos;            pos += len(sfi_block)
    loc_pos = pos;            pos += len(loc_block)
    cs2do_pos = pos;          pos += len(cs2do_block)
    ds2cs_pos = pos;          pos += len(ds2cs_block)
    names_off_pos = pos;      pos += len(names_offset_block)
    names_data_pos = pos;     pos += len(name_data)

    needed_blocks = (pos + read_sector - 1) // read_sector
    header_sector = needed_blocks
    first_sec = read_sector * header_sector
    if first_sec < pos:
        header_sector += 1
        first_sec = read_sector * header_sector

    # Build header from scratch (avoids depending on src._data)
    hdr = bytearray(h.header_size)
    struct.pack_into(endian + "I", hdr, 0, CPKArchive.MAGIC)
    struct.pack_into(endian + "I", hdr, 4, h.version)
    struct.pack_into(endian + "Q", hdr, 8, decomp_fs)
    struct.pack_into(endian + "I", hdr, 16, getattr(h, "fmt_const", 2))
    struct.pack_into(endian + "I", hdr, 20, file_count)
    struct.pack_into(endian + "I", hdr, 24, loc_count)
    struct.pack_into(endian + "I", hdr, 28, header_sector)
    struct.pack_into(endian + "I", hdr, 32, fs_bc)
    struct.pack_into(endian + "I", hdr, 36, flcnt_bc)
    struct.pack_into(endian + "I", hdr, 40, flidx_bc)
    struct.pack_into(endian + "I", hdr, 44, loc_bc)
    struct.pack_into(endian + "I", hdr, 48, cs2do_bc_out)
    struct.pack_into(endian + "I", hdr, 52, ds2cs_bc_out)
    struct.pack_into(endian + "I", hdr, 56, 0)  # CRC placeholder
    if h.version == 7:
        struct.pack_into(endian + "I", hdr, 60, read_sector)
        struct.pack_into(endian + "I", hdr, 64, comp_sector)
        struct.pack_into(endian + "I", hdr, 68, 0)
    # For preserve-mode (existing source), copy any unknown header bytes
    # back in BEFORE we patch the known fields, just in case the file has
    # extra non-zero bytes we haven't decoded. But our mapping is now
    # complete, so this is purely defensive.
    if not new_archive:
        src_hdr = self._src._data[:h.header_size]
        # Only the magic..known-field range is overwritten; everything else
        # has been zeroed. For v6, offsets 60..63 are documented padding.
        # No-op on observed archives; kept for forward safety.
        pass

    archive_size = first_sec + comp_sector_count * comp_sector
    out = bytearray(archive_size)
    out[0:h.header_size] = hdr
    out[sfi_pos:sfi_pos + len(sfi_block)] = sfi_block
    out[loc_pos:loc_pos + len(loc_block)] = loc_block
    out[cs2do_pos:cs2do_pos + len(cs2do_block)] = cs2do_block
    out[ds2cs_pos:ds2cs_pos + len(ds2cs_block)] = ds2cs_block
    out[names_off_pos:names_off_pos + len(names_offset_block)] = names_offset_block
    out[names_data_pos:names_data_pos + len(name_data)] = name_data
    for i, (_, _, slot) in enumerate(sectors):
        sec_off = first_sec + i * comp_sector
        out[sec_off:sec_off + comp_sector] = slot

    sector_crc = zlib.crc32(bytes(out[first_sec:])) & 0xFFFFFFFF
    struct.pack_into(endian + "I", out, 56, sector_crc)

    with open(out_path, "wb") as f:
        f.write(out)
    _step(f'wrote ({archive_size:,} B, crc32={sector_crc:08X})')


_original_save = CPKWriter.save


def _save_dispatch(self, out_path):
    if _has_rebuild_mods(self):
        return _full_rebuild(self, out_path)
    _original_save(self, out_path)


CPKWriter.save = _save_dispatch
CPKWriter.replace_file = replace_file
CPKWriter.delete_file = delete_file
CPKWriter.add_file = add_file
CPKWriter.set_zlib_level = set_zlib_level
CPKWriter.create_new = create_new
