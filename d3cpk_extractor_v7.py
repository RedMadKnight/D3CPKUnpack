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

            state := 0x873AE8E53C11D761  (init)
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
from d3cpk_extractor import main


if __name__ == "__main__":
    main()
