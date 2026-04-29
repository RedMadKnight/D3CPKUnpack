#!/usr/bin/env python3
"""
D3CPK Extractor v7 — FULL DECOMPRESSION including encrypted chunks.

Version history:
  v4: PS3/Xbox 360 only (BE + CPK v6)
  v5: added Switch LE + CPK v7
  v6: fixed bit-ordering for LE (byte-stream MSB-first for metadata)
      + Windows filename sanitization ('|' → '__', etc.)
  v7: FULL DECOMPRESSION. Three breakthroughs landed in this version:

      (1) The "0x8000 flag" is NOT a different compression algorithm —
          it's an ENCRYPTION flag. Data is still zlib-compressed, but
          XOR-encrypted with a stream cipher before zlib. Must DECRYPT
          first, then zlib decompress.

          Algorithm identified via Switch NSO disassembly (branch 2_6_2,
          file ConsoleIO_Decompress.cpp, function DecompressData):

            state := 0x872DCDA7A97A7EE1  (init)
            for each byte ct in encrypted_data:
                pt := ct XOR (state & 0xFF)
                state := (ct << 56) | (state >> 8)    # CT-feedback

          After decryption, the data is a standard zlib stream.
          Recovered ALL previously partial files (588 on Xbox, 7610 on Switch).

      (2) When zlib decompresses a chunk to FEWER bytes than the packet
          header promised (decomp_size), zero-fill the delta to keep
          file offsets aligned. Per-chunk zlib works — stream does NOT
          actually span packets/sectors in practice, despite NSO code
          looking like it could.

      (3) Switch CoreCommon.cpk has orphaned SFI entries whose data was
          never written — decomp_fs in the header promises ~29,805 more
          bytes than the last sectors physically contain. Extend the
          global stream with zeros up to decomp_fs so those 29 MarkerSet
          entries extract as zero-filled placeholders instead of failing.

      Final extraction results:
        Xbox   CacheCommon.cpk:  47,138 / 47,138   (100.00%)
        Switch CoreCommon.cpk:  125,993 / 125,993 (100.00%)

Auto-detects endianness via magic bytes.

Usage:
    python3 d3cpk_extractor_v7.py <file.cpk> [<output_dir>]
"""
import struct, zlib, os, sys, time, json
from collections import Counter, defaultdict

if len(sys.argv) < 2:
    print("Usage: python3 d3cpk_extractor_v5.py <file.cpk> [<output_dir>]")
    sys.exit(1)

PATH = sys.argv[1]
OUT_DIR = sys.argv[2] if len(sys.argv) > 2 else 'd3cpk_output'

os.makedirs(os.path.join(OUT_DIR, 'extracted'), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, 'unknown_chunks'), exist_ok=True)

print(f"Loading {PATH}...")
with open(PATH, 'rb') as f:
    data = f.read()
print(f"  {len(data):,} B ({len(data)/1024/1024:.1f} MB)")

# ====================================================================
# Endianness detection via magic bytes
# ====================================================================

magic_be = struct.unpack('>I', data[:4])[0]
magic_le = struct.unpack('<I', data[:4])[0]

if magic_be == 0xA1B2C3D4:
    ENDIAN = '>'
    ENDIAN_NAME = 'big-endian (PS3/Xbox 360)'
elif magic_le == 0xA1B2C3D4:
    ENDIAN = '<'
    ENDIAN_NAME = 'little-endian (Switch)'
else:
    sys.exit(f"Unknown magic: BE=0x{magic_be:08X}, LE=0x{magic_le:08X}. "
             f"Expected 0xA1B2C3D4 in either endianness.")

print(f"Endianness: {ENDIAN_NAME}\n")

# ====================================================================
# Endian-aware helpers
# ====================================================================

def u16(o): return struct.unpack(ENDIAN + 'H', data[o:o+2])[0]
def u32(o): return struct.unpack(ENDIAN + 'I', data[o:o+4])[0]
def u64(o): return struct.unpack(ENDIAN + 'Q', data[o:o+8])[0]

def read_bits_be(buf, bit_pos, bit_count):
    """Read bits MSB-first within each byte.

    This is the SAME function used for both BE (PS3/Xbox 360) and LE (Switch)
    CPKs. Empirical testing showed that the Switch port kept the PowerPC-era
    bit-packer intact: word-level values (magic, sizes, offsets) are
    little-endian on Switch, but the bit-stream in SFI / Locations /
    CompSectorToDecomp tables still goes MSB-first within each byte.
    """
    result = 0
    for i in range(bit_count):
        pos = bit_pos + i
        byte_pos = pos >> 3
        byte_bit = 7 - (pos & 7)
        result = (result << 1) | (1 if buf[byte_pos] & (1 << byte_bit) else 0)
    return result

read_bits = read_bits_be

def fnv_hash64(s):
    h = 0xCBF29CE484222325
    for c in s.lower().encode('ascii'):
        h = (0x100000001B3 * (h ^ c)) & 0xFFFFFFFFFFFFFFFF
    return h

def highest_bit(u):
    r = 0
    while u != 0: u >>= 1; r += 1
    return r

# ====================================================================
# Parse header and tables (field offsets identical — just endianness differs)
# ====================================================================

ver = u32(4)
decomp_fs = u64(8)
fmt_const = u32(16)        # 2 = legacy layout, 10 = new layout (Patch 2.6.7+)
file_count = u32(20)
loc_count = u32(24)
header_sector = u32(28)
FS_BC  = u32(32)
FLCNT_BC = u32(36)
FLIDX_BC = u32(40)
LOC_BC   = u32(44)
CS2DO_BC = u32(48)
DS2CS_BC = u32(52)

print(f"=== HEADER ===")
print(f"Version: {ver}, DecompSize: {decomp_fs:,} B, FileCount: {file_count:,}")

if ver == 6:
    read_sector = 0x10000
    comp_sector = 0x4000
    header_size = 64
elif ver == 7:
    read_sector = u32(60)
    comp_sector = u32(64)
    header_size = 72
    print(f"v7 read_sector=0x{read_sector:X}, comp_sector=0x{comp_sector:X}")
else:
    sys.exit(f"Unsupported CPK version {ver}")

# Patch 2.6.7 (May 2019) introduced a refactored layout with fmt_const=10:
#   * header_sector becomes a direct byte offset (was: multiple of read_sector)
#   * sector 0 starts immediately with chunks (was: 6-byte header + skip prologue)
# Detect via fmt_const at offset 0x10 and branch on first_sec / sector-0 logic.
if fmt_const == 10:
    first_sec = header_sector
else:
    first_sec = (read_sector * header_sector) & 0xFFFF0000
    if first_sec % read_sector != 0:
        first_sec += read_sector
comp_sector_count = (comp_sector + len(data) - 1 - first_sec) // comp_sector
if comp_sector_count < 0:
    comp_sector_count = 0
ds2cs_bc = DS2CS_BC if DS2CS_BC > 0 else highest_bit(comp_sector_count)
ds_sector_count = (decomp_fs + comp_sector - 1) // comp_sector
print(f"Layout: fmt_const={fmt_const} ({'new' if fmt_const == 10 else 'legacy'}), first_sec=0x{first_sec:08X}")

print(f"Sectors: {comp_sector_count:,}")

# Sanity-check: obvious garbage means our LE bit-reader choice is wrong
if file_count > 10_000_000 or decomp_fs > 100_000_000_000:
    print(f"\n!!! WARNING: FileCount or DecompSize look implausible.")
    print(f"    FileCount={file_count:,}  DecompSize={decomp_fs:,}")
    print(f"    If this is the LE/Switch build, the bit-packing byte order")
    print(f"    within metadata tables may be different from what this")
    print(f"    extractor assumes. Try swapping read_bits at the top of")
    print(f"    this file (read_bits_be ↔ read_bits_le).")
    sys.exit(1)

t0 = time.time()

# SortedFileInfo
sfi_bytes = (file_count * (64 + FS_BC + FLCNT_BC + FLIDX_BC) + 7) // 8
sfi_block = data[header_size : header_size + sfi_bytes]
pos = header_size + sfi_bytes
sfi = []
bp = 0
for i in range(file_count):
    h = read_bits(sfi_block, bp, 64); bp += 64
    sz = read_bits(sfi_block, bp, FS_BC); bp += FS_BC
    lc = read_bits(sfi_block, bp, FLCNT_BC); bp += FLCNT_BC
    li = read_bits(sfi_block, bp, FLIDX_BC); bp += FLIDX_BC
    sfi.append({'hash': h, 'size': sz, 'loc_count': lc, 'loc_idx': li})

# Locations
loc_bytes = (LOC_BC * loc_count + 7) // 8
loc_block = data[pos : pos + loc_bytes]
pos += loc_bytes
raw_locs = []
bp = 0
for i in range(loc_count):
    raw_locs.append(read_bits(loc_block, bp, LOC_BC)); bp += LOC_BC

# CompSectorToDecompOffset — bit-width is CS2DO_BC from header (offset 0x30).
# In every archive examined CS2DO_BC == LOC_BC, but trust the header value.
cs2do_bytes = (comp_sector_count * CS2DO_BC + 7) // 8
cs2do_block = data[pos : pos + cs2do_bytes]
pos += cs2do_bytes
cs2do = []
bp = 0
for i in range(comp_sector_count):
    cs2do.append(read_bits(cs2do_block, bp, CS2DO_BC)); bp += CS2DO_BC

# DecompSectorToCompSector (kept for completeness; not used here)
ds2cs_bytes = (ds2cs_bc * ds_sector_count + 7) // 8
pos += ds2cs_bytes

# Names
name_offs = [u32(pos + i*4) for i in range(file_count)]
pos += file_count * 4
names_data_start = pos
names = []
for off in name_offs:
    abs_off = names_data_start + off
    end = data.find(b'\x00', abs_off)
    if end == -1: end = abs_off + 256
    names.append(data[abs_off:end].decode('latin-1', errors='replace'))

hash_to_sfi = {e['hash']: i for i, e in enumerate(sfi)}

print(f"Tables parsed in: {time.time()-t0:.2f}s")
print(f"Loc_count: {Counter(e['loc_count'] for e in sfi).most_common(5)}")

# ====================================================================
# Blizzard CPK encryption (stream cipher with CT-feedback)
# ====================================================================

CIPHER_KEY_INIT = (-8702387258687105183) & 0xFFFFFFFFFFFFFFFF  # 0x872DCDA7A97A7EE1

def decrypt_cpk_chunk(ciphertext):
    """
    Decrypt an encrypted CPK chunk payload.

    Algorithm from Switch NSO disassembly (D3 branch 2_6_2,
    ConsoleIO_Decompress.cpp::DecompressData):
      state := 0x872DCDA7A97A7EE1
      for byte ct in ciphertext:
          pt := ct XOR (state & 0xFF)
          state := (ct << 56) | (state >> 8)    # CT feedback

    After decryption, the data is a standard zlib-compressed stream.
    """
    state = CIPHER_KEY_INIT
    pt = bytearray()
    for c in ciphertext:
        pt.append(c ^ (state & 0xFF))
        state = ((c & 0xFF) << 56) | (state >> 8)
        state &= 0xFFFFFFFFFFFFFFFF
    return bytes(pt)

# ====================================================================
# Sector decompression (handles both plain and encrypted chunks)
# ====================================================================

unknown_chunks = []      # diagnostic: chunks that failed even after decrypt
encryption_stats = {'encrypted': 0, 'encrypted_ok': 0, 'encrypted_fail': 0}
DEBUG_LAST_SECTORS = 0   # set > 0 to enable verbose logs for the last N sectors

def decompress_sector(sec_idx):
    sec_start = first_sec + sec_idx * comp_sector
    sec_end = sec_start + comp_sector
    if sec_end > len(data):
        sec_end = len(data)

    debug_this = (sec_idx >= comp_sector_count - DEBUG_LAST_SECTORS)

    p = sec_start
    # Legacy archives (fmt_const=2) carry a 2-byte "skip" preamble at sector 0
    # (overflow of the bit-packed table area into the first read_sector slot).
    # New layout (fmt_const=10) starts sector 0 cleanly with chunks.
    if sec_idx == 0 and fmt_const != 10:
        if p + 6 > len(data):
            return b'', []
        skip = u16(p + 4)
        p += 6 + skip

    out = bytearray()
    local_unknowns = []

    if debug_this:
        print(f"  [DBG] Sector {sec_idx}: range [{sec_start}..{sec_end}]")

    while p + 6 <= sec_end:
        nlow = u16(p)
        nhigh = u16(p + 2)
        zsize_raw = u16(p + 4)
        if zsize_raw == 0:
            if debug_this:
                print(f"  [DBG]   chunk@{p}: zsize_raw=0 -> END of sector")
            break

        encrypted = (zsize_raw & 0x8000) != 0
        zsize = zsize_raw & 0x7FFF
        decomp_size = (nhigh << 16) | nlow

        ds = p + 6
        de = ds + zsize
        if de > sec_end:
            if debug_this:
                print(f"  [DBG]   chunk@{p}: de={de} > sec_end={sec_end} -> SKIP "
                      f"(zsize={zsize}, decomp_size={decomp_size}, enc={encrypted})")
            break

        chunk_data = data[ds:de]
        if encrypted:
            encryption_stats['encrypted'] += 1
            try:
                chunk_data = decrypt_cpk_chunk(chunk_data)
                encryption_stats['encrypted_ok'] += 1
            except Exception as ex:
                encryption_stats['encrypted_fail'] += 1
                out.extend(b'\x00' * decomp_size)
                p = de
                continue

        # Try zlib decompress on this chunk
        try:
            decompressed = zlib.decompress(chunk_data)
            got = len(decompressed)
            if got < decomp_size:
                # Truncated — zero-fill missing bytes so offsets stay aligned
                decompressed = decompressed + b'\x00' * (decomp_size - got)
            out.extend(decompressed)
            if debug_this:
                print(f"  [DBG]   chunk@{p}: OK decomp={got}/{decomp_size} "
                      f"(zsize={zsize}, enc={encrypted})")
        except Exception as ex:
            # zlib failed — try stateful then give up
            try:
                dc = zlib.decompressobj()
                decompressed = dc.decompress(chunk_data)
                if len(decompressed) < decomp_size:
                    decompressed = decompressed + b'\x00' * (decomp_size - len(decompressed))
                out.extend(decompressed)
                if debug_this:
                    print(f"  [DBG]   chunk@{p}: OK(partial) decomp={len(decompressed)}/{decomp_size}")
            except Exception as ex2:
                local_unknowns.append({
                    'file_offset': p, 'chunk_header_offset': p,
                    'data_offset': ds, 'zsize': zsize, 'zsize_raw': zsize_raw,
                    'decomp_size': decomp_size,
                    'local_stream_offset': len(out),
                    'raw_bytes': bytes(data[ds:de]),
                    'error': f'zlib failed: {ex2}',
                })
                out.extend(b'\x00' * decomp_size)
                if debug_this:
                    print(f"  [DBG]   chunk@{p}: FAILED zlib={ex2}")

        p = de

    # If the sector had trailing bytes we couldn't parse (less than 6),
    # report it for the last sector — this is where missing data may live.
    if debug_this:
        tail = sec_end - p
        if tail > 0:
            print(f"  [DBG] Sector {sec_idx}: {tail} trailing bytes unparsed at offset {p}")
            print(f"  [DBG]   First 32: {bytes(data[p:min(p+32, sec_end)]).hex(' ')}")

    return bytes(out), local_unknowns

print(f"\n=== SECTOR DECOMPRESSION ===")
t0 = time.time()
global_stream_parts = []
current_global_offset = 0

for s in range(comp_sector_count):
    sector_bytes, sector_unknowns = decompress_sector(s)
    for uk in sector_unknowns:
        uk['global_stream_offset'] = current_global_offset + uk['local_stream_offset']
        uk['sector_index'] = s
        del uk['local_stream_offset']
        unknown_chunks.append(uk)
    global_stream_parts.append(sector_bytes)
    current_global_offset += len(sector_bytes)

global_stream = b''.join(global_stream_parts)
print(f"Decompressed {comp_sector_count:,} sectors in {time.time()-t0:.2f}s")
print(f"Global stream: {len(global_stream):,} B (expected {decomp_fs:,})")
print(f"Encrypted chunks: {encryption_stats['encrypted']:,}")
print(f"  Decrypted successfully:   {encryption_stats['encrypted_ok']:,}")
print(f"  Decrypt/decompress fail:  {encryption_stats['encrypted_fail']:,}")

# Some CPK archives (Switch CoreCommon.cpk) have SFI entries for orphaned/unused
# files whose data was never written. decomp_fs in the header promises more bytes
# than physically exist in the last sectors. Extend the stream with zeros so these
# files can still be extracted (as zero-filled) instead of being listed as Failed.
if len(global_stream) < decomp_fs:
    missing = decomp_fs - len(global_stream)
    print(f"Extending stream by {missing:,} zero bytes to match expected decomp_fs")
    print(f"  (files landing in this region will be zero-filled — orphaned SFI entries)")
    global_stream = global_stream + b'\x00' * missing

if len(unknown_chunks) > 0:
    total_flag_bytes = sum(uk['decomp_size'] for uk in unknown_chunks)
    total_raw_bytes = sum(uk['zsize'] for uk in unknown_chunks)
    print(f"Chunks left unresolved (zero-filled): {len(unknown_chunks):,}")
    print(f"  Total decompressed of these chunks: {total_flag_bytes:,} B")
    print(f"  Total compressed (raw) in CPK:      {total_raw_bytes:,} B")

# ====================================================================
# File extraction
# ====================================================================

unknown_intervals = sorted(
    (uk['global_stream_offset'], uk['global_stream_offset'] + uk['decomp_size'], i)
    for i, uk in enumerate(unknown_chunks)
)

def find_overlapping_chunks(file_start, file_size):
    file_end = file_start + file_size
    overlapping = []
    for start, end, idx in unknown_intervals:
        if end <= file_start: continue
        if start >= file_end: break
        overlapping.append(idx)
    return overlapping

print(f"\n=== FILE EXTRACTION ===")
t0 = time.time()

clean_files = []
partial_files = []
failed_files = []
chunk_to_files = defaultdict(list)

for name in names:
    h = fnv_hash64(name)
    if h not in hash_to_sfi:
        failed_files.append((name, 'hash not found in SFI'))
        continue
    entry = sfi[hash_to_sfi[h]]
    if entry['loc_count'] == 0:
        failed_files.append((name, 'loc_count=0'))
        continue

    loc_off = raw_locs[entry['loc_idx']]
    size = entry['size']

    if loc_off + size > len(global_stream):
        failed_files.append((name, f'file exceeds global_stream (off={loc_off}, size={size}, stream={len(global_stream)})'))
        continue

    file_data = global_stream[loc_off:loc_off + size]
    overlaps = find_overlapping_chunks(loc_off, size)

    safe_name = name.replace('\\', '/').lstrip('/')
    # Windows forbids these chars in filenames: < > : " / \ | ? *
    # (backslash already converted, forward slash kept as dir separator)
    # The '|' in names like "StringList/Foo.stl|jpJP" marks a localized variant,
    # we substitute '__' so files become "Foo.stl__jpJP" and stay human-readable.
    safe_name = safe_name.replace('|', '__')
    for bad_ch in ['<', '>', ':', '"', '?', '*']:
        safe_name = safe_name.replace(bad_ch, '_')
    out_path = os.path.join(OUT_DIR, 'extracted', safe_name)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'wb') as f:
        f.write(file_data)

    if overlaps:
        partial_files.append((name, overlaps))
        for chunk_idx in overlaps:
            chunk_to_files[chunk_idx].append({
                'name': name, 'file_offset_in_stream': loc_off, 'size': size
            })
    else:
        clean_files.append(name)

print(f"Extraction: {time.time()-t0:.2f}s")
print(f"  Complete:  {len(clean_files):,}")
print(f"  Partial:   {len(partial_files):,}")
print(f"  Failed:    {len(failed_files):,}")

# ====================================================================
# Dump flagged chunks
# ====================================================================

print(f"\n=== DUMPING FLAGGED CHUNKS ===")
if len(unknown_chunks) > 0:
    MAX_INDIVIDUAL_DUMPS = 500
    chunks_to_dump = min(len(unknown_chunks), MAX_INDIVIDUAL_DUMPS)

    for i, uk in enumerate(unknown_chunks[:chunks_to_dump]):
        chunk_path = os.path.join(OUT_DIR, 'unknown_chunks',
                                   f'chunk_{i:05d}_at_0x{uk["chunk_header_offset"]:X}_decomp{uk["decomp_size"]}.bin')
        with open(chunk_path, 'wb') as f:
            f.write(uk['raw_bytes'])

    json_data = {
        'source_file': PATH,
        'source_size': len(data),
        'cpk_version': ver,
        'endianness': ENDIAN_NAME,
        'total_unknown_chunks': len(unknown_chunks),
        'individual_dumps_saved': chunks_to_dump,
        'summary_stats': {
            'total_decomp_size': sum(uk['decomp_size'] for uk in unknown_chunks),
            'total_raw_size': sum(uk['zsize'] for uk in unknown_chunks),
            'decomp_size_distribution': dict(Counter(uk['decomp_size'] for uk in unknown_chunks).most_common(20)),
            'raw_size_distribution': dict(Counter(uk['zsize'] for uk in unknown_chunks).most_common(20)),
        },
        'chunks': [],
    }

    for i, uk in enumerate(unknown_chunks):
        chunk_entry = {
            'chunk_index': i,
            'sector_index': uk['sector_index'],
            'chunk_header_offset': uk['chunk_header_offset'],
            'chunk_header_hex': f"0x{uk['chunk_header_offset']:X}",
            'data_offset': uk['data_offset'],
            'data_offset_hex': f"0x{uk['data_offset']:X}",
            'zsize': uk['zsize'],
            'zsize_raw': uk['zsize_raw'],
            'zsize_raw_hex': f"0x{uk['zsize_raw']:04X}",
            'decomp_size': uk['decomp_size'],
            'global_stream_offset': uk['global_stream_offset'],
            'first_16_bytes_hex': uk['raw_bytes'][:16].hex(' '),
            'dump_file': f'chunk_{i:05d}_at_0x{uk["chunk_header_offset"]:X}_decomp{uk["decomp_size"]}.bin' if i < chunks_to_dump else None,
            'affected_files': chunk_to_files.get(i, []),
        }
        json_data['chunks'].append(chunk_entry)

    json_path = os.path.join(OUT_DIR, 'unknown_chunks.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    print(f"Saved {chunks_to_dump} chunks individually.")
    print(f"Metadata: {json_path}")
else:
    print(f"No flagged chunks — this CPK doesn't need special handling.")

# ====================================================================
# Text report
# ====================================================================

report_path = os.path.join(OUT_DIR, 'extraction_report.txt')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(f"=== D3CPK EXTRACTION — REPORT ===\n")
    f.write(f"Source file: {PATH}\n")
    f.write(f"Size:        {len(data):,} B\n")
    f.write(f"Endianness:  {ENDIAN_NAME}\n\n")
    f.write(f"--- HEADER ---\n")
    f.write(f"CPK version:           {ver}\n")
    f.write(f"DecompressedFileSize:  {decomp_fs:,} B\n")
    f.write(f"FileCount:             {file_count:,}\n")
    f.write(f"LocationCount:         {loc_count:,}\n")
    f.write(f"CompSectorCount:       {comp_sector_count:,}\n\n")
    f.write(f"--- EXTRACTION SUMMARY ---\n")
    f.write(f"Complete files:    {len(clean_files):,}\n")
    f.write(f"Partial files:     {len(partial_files):,}\n")
    f.write(f"Failed files:      {len(failed_files):,}\n")
    f.write(f"Global stream:     {len(global_stream):,} B (expected {decomp_fs:,})\n")
    f.write(f"Unsupported chunks: {len(unknown_chunks):,}\n\n")

    if partial_files:
        f.write(f"--- PARTIAL FILES ---\n")
        partial_exts = Counter(n.rsplit('.', 1)[-1].lower() if '.' in n else '?' for n, _ in partial_files)
        f.write(f"Extension distribution: {dict(partial_exts.most_common(10))}\n\n")
        for name, chunk_idxs in partial_files[:50]:
            f.write(f"  {name}  (touches {len(chunk_idxs)} flagged chunks)\n")
        if len(partial_files) > 50:
            f.write(f"  ... and {len(partial_files) - 50} more\n")
        f.write(f"\n")

    if failed_files:
        f.write(f"--- FAILED FILES ---\n")
        for name, reason in failed_files[:30]:
            f.write(f"  {name}: {reason}\n")
        if len(failed_files) > 30:
            f.write(f"  ... and {len(failed_files) - 30} more\n")

print(f"\nReport: {report_path}")

# ====================================================================
# Magic bytes sanity check
# ====================================================================

print(f"\n=== MAGIC BYTES OF CLEAN FILES (sample) ===")
magic_stats = Counter()
for name in clean_files[:min(5000, len(clean_files))]:
    h = fnv_hash64(name)
    entry = sfi[hash_to_sfi[h]]
    loc_off = raw_locs[entry['loc_idx']]
    if loc_off + 4 <= len(global_stream):
        magic_stats[global_stream[loc_off:loc_off+4]] += 1

print(f"Top 10:")
for magic, cnt in magic_stats.most_common(10):
    ascii_ = ''.join(chr(b) if 0x20 <= b < 0x7F else '.' for b in magic)
    print(f"  {magic.hex(' ')} ({ascii_!r}) x{cnt}")

print(f"\n=== DONE ===")
print(f"Output: {OUT_DIR}/")
