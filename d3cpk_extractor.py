#!/usr/bin/env python3
"""
D3CPK Extractor — extract a whole CPK archive and report on it, as used for
the console releases of Diablo III. All parsing lives in cpklib.py; this
script adds the output tree, the text report and dumps of unreadable chunks:

    <output_dir>/extracted/              every file in the archive
    <output_dir>/extraction_report.txt   summary
    <output_dir>/unknown_chunks/         raw chunks that failed to inflate
    <output_dir>/unknown_chunks.json     metadata for those chunks

Reading the shipped archives depends on four details not present in naive
zlib-only CPK readers:

  (1) Chunks whose `zsize_raw` has bit 0x8000 set are not compressed
      with an alternative algorithm — they are encrypted zlib. The
      pipeline for encrypted chunks is: decrypt, then zlib.decompress.

  (2) When zlib decompresses a chunk to fewer bytes than the chunk
      header's `decomp_size` declares, the shortfall must be
      zero-padded so that global-stream offsets stay aligned with
      the Locations table. Chunks do not span sector boundaries.

  (3) Some archives declare a `decomp_fs` in the header that exceeds
      the sum of physically present chunk payloads. Extend the global
      stream with zeros up to `decomp_fs` so that orphaned
      SortedFileInfo entries extract as zero-filled placeholders
      rather than producing errors.

  (4) Decompressed sectors do NOT pack tightly into the global
      stream. Each compressed sector slot is `comp_sector` bytes
      (e.g. 0x4000) on disk, but a single sector can decompress to
      anywhere from ~15K to ~50K bytes. The CompSectorToDecompOffset
      (cs2do) table gives the authoritative offset of each sector's
      output in the global stream; sectors must be written at those
      offsets, with gaps zero-filled. Naively concatenating sector
      outputs (assuming a fixed ratio) drifts file positions and is
      what produced the "weird / cut out / sometimes empty" strings
      reported in issue #1 against CoreCommon.cpk.

Usage:
    python3 d3cpk_extractor.py <file.cpk> [<output_dir>]
"""
import json
import os
import sys
import time
from collections import Counter, defaultdict

import cpklib


MAX_INDIVIDUAL_DUMPS = 500


def dump_failed_chunks(ar, out_dir, chunk_files):
    failures = ar.chunk_failures
    print(f"\n=== DUMPING FLAGGED CHUNKS ===")
    if not failures:
        print("No flagged chunks - this CPK doesn't need special handling.")
        return

    def dump_name(i, f):
        return f'chunk_{i:05d}_at_0x{f.header_offset:X}_decomp{f.decomp_size}.bin'

    def raw_bytes(f):
        return bytes(ar._data[f.data_offset:f.data_offset + f.zsize])

    dumps = min(len(failures), MAX_INDIVIDUAL_DUMPS)
    for i, f in enumerate(failures[:dumps]):
        with open(os.path.join(out_dir, 'unknown_chunks', dump_name(i, f)), 'wb') as fh:
            fh.write(raw_bytes(f))

    json_data = {
        'source_file': ar.path,
        'source_size': len(ar._data),
        'cpk_version': ar.header.version,
        'endianness': 'big-endian' if ar.header.endian == '>' else 'little-endian',
        'total_unknown_chunks': len(failures),
        'individual_dumps_saved': dumps,
        'summary_stats': {
            'total_decomp_size': sum(f.decomp_size for f in failures),
            'total_raw_size': sum(f.zsize for f in failures),
            'decomp_size_distribution': dict(Counter(f.decomp_size for f in failures).most_common(20)),
            'raw_size_distribution': dict(Counter(f.zsize for f in failures).most_common(20)),
        },
        'chunks': [
            {
                'chunk_index': i,
                'sector_index': f.sector,
                'chunk_header_offset': f.header_offset,
                'chunk_header_hex': f"0x{f.header_offset:X}",
                'data_offset': f.data_offset,
                'data_offset_hex': f"0x{f.data_offset:X}",
                'zsize': f.zsize,
                'zsize_raw': f.zsize_raw,
                'zsize_raw_hex': f"0x{f.zsize_raw:04X}",
                'decomp_size': f.decomp_size,
                'global_stream_offset': f.stream_offset,
                'first_16_bytes_hex': raw_bytes(f)[:16].hex(' '),
                'dump_file': dump_name(i, f) if i < dumps else None,
                'affected_files': chunk_files.get(i, []),
                'error': f.error,
            }
            for i, f in enumerate(failures)
        ],
    }
    json_path = os.path.join(out_dir, 'unknown_chunks.json')
    with open(json_path, 'w', encoding='utf-8') as fh:
        json.dump(json_data, fh, indent=2, ensure_ascii=False)
    print(f"Saved {dumps} chunks individually.")
    print(f"Metadata: {json_path}")


def write_report(ar, out_dir, clean, partial, failed):
    h = ar.header
    report_path = os.path.join(out_dir, 'extraction_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"=== D3CPK EXTRACTION — REPORT ===\n")
        f.write(f"Source file: {ar.path}\n")
        f.write(f"Size:        {len(ar._data):,} B\n")
        f.write(f"Endianness:  {'big' if h.endian == '>' else 'little'}-endian\n\n")
        f.write(f"--- HEADER ---\n")
        f.write(f"CPK version:           {h.version}\n")
        f.write(f"DecompressedFileSize:  {h.decomp_fs:,} B\n")
        f.write(f"FileCount:             {h.file_count:,}\n")
        f.write(f"LocationCount:         {h.loc_count:,}\n")
        f.write(f"CompSectorCount:       {ar.stats()['comp_sector_count']:,}\n\n")
        f.write(f"--- EXTRACTION SUMMARY ---\n")
        f.write(f"Complete files:    {len(clean):,}\n")
        f.write(f"Partial files:     {len(partial):,}\n")
        f.write(f"Failed files:      {len(failed):,}\n")
        f.write(f"Global stream:     {len(ar.global_stream):,} B (expected {h.decomp_fs:,})\n")
        f.write(f"Unsupported chunks: {ar.chunk_errors:,}\n\n")

        if partial:
            f.write(f"--- PARTIAL FILES ---\n")
            exts = Counter(e.ext or '?' for e in partial)
            f.write(f"Extension distribution: {dict(exts.most_common(10))}\n\n")
            for e in partial[:50]:
                f.write(f"  {e.name}\n")
            if len(partial) > 50:
                f.write(f"  ... and {len(partial) - 50} more\n")
            f.write(f"\n")

        if failed:
            f.write(f"--- FAILED FILES ---\n")
            for name, reason in failed[:30]:
                f.write(f"  {name}: {reason}\n")
            if len(failed) > 30:
                f.write(f"  ... and {len(failed) - 30} more\n")
    print(f"\nReport: {report_path}")


def print_magic_stats(ar, clean):
    print(f"\n=== MAGIC BYTES OF CLEAN FILES (sample) ===")
    magic_stats = Counter(ar.read(e)[:4] for e in clean[:5000] if e.size >= 4)
    print(f"Top 10:")
    for magic, cnt in magic_stats.most_common(10):
        ascii_ = ''.join(chr(b) if 0x20 <= b < 0x7F else '.' for b in magic)
        print(f"  {magic.hex(' ')} ({ascii_!r}) x{cnt}")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print(f"Usage: python3 {os.path.basename(sys.argv[0])} <file.cpk> [<output_dir>]")
        sys.exit(1)
    path = argv[0]
    out_dir = argv[1] if len(argv) > 1 else 'd3cpk_output'

    print(f"Loading {path}...")
    try:
        ar = cpklib.CPKArchive.open(path)
    except ValueError as ex:
        sys.exit(str(ex))
    h = ar.header
    print(f"  {len(ar._data):,} B ({len(ar._data)/1024/1024:.1f} MB)")
    print(f"Version: {h.version} ({'big' if h.endian == '>' else 'little'}-endian), "
          f"fmt_const: {h.fmt_const}, files: {len(ar):,}, "
          f"sectors: {ar.stats()['comp_sector_count']:,}")

    os.makedirs(os.path.join(out_dir, 'extracted'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'unknown_chunks'), exist_ok=True)

    print(f"\n=== SECTOR DECOMPRESSION ===")
    t0 = time.time()
    stream_size = len(ar.global_stream)
    print(f"Decompressed in {time.time()-t0:.2f}s")
    print(f"Global stream: {stream_size:,} B (expected {h.decomp_fs:,})")
    print(f"Chunks left unresolved (zero-filled): {ar.chunk_errors:,}")

    print(f"\n=== FILE EXTRACTION ===")
    t0 = time.time()
    by_file = ar.failed_chunks_by_file()
    clean = []
    partial = []
    failed = [(name, 'hash not found in SFI') for name in ar.unmatched_names]
    chunk_files = defaultdict(list)
    for e in ar:
        if e.loc_count == 0:
            failed.append((e.name, 'loc_count=0'))
            continue
        try:
            payload = ar.read(e)
            ar._write_entry(e, payload, os.path.join(out_dir, 'extracted'),
                            flat=False, sanitize_pipe=True)
        except (IOError, ValueError) as ex:
            failed.append((e.name, str(ex)))
            continue
        if e.name in by_file:
            partial.append(e)
            for i in by_file[e.name]:
                chunk_files[i].append({
                    'name': e.name, 'file_offset_in_stream': e.offset, 'size': e.size,
                })
        else:
            clean.append(e)
    print(f"Extraction: {time.time()-t0:.2f}s")
    print(f"  Complete:  {len(clean):,}")
    print(f"  Partial:   {len(partial):,}")
    print(f"  Failed:    {len(failed):,}")

    dump_failed_chunks(ar, out_dir, chunk_files)
    write_report(ar, out_dir, clean, partial, failed)
    print_magic_stats(ar, clean)

    print(f"\n=== DONE ===")
    print(f"Output: {out_dir}/")


if __name__ == "__main__":
    main()
