# Diablo III — SNO Asset System and CPK Archive Format

## A file-format reference for interoperability

---

### Disclaimer and scope

This document is a specification of the on-disk file formats and data-layout
conventions used by the console release of Diablo III.

All structures, constants, layouts, and algorithms described here were derived
exclusively from:

- binary inspection of shipped archive and asset files (archives,
  `CoreTOC.dat`, `Prefetch.dat`, and samples extracted from them);
- black-box behavioural testing of the game when presented with valid,
  modified, or truncated inputs;
- correlation across multiple sample datasets;
- cross-reference against publicly available open-source projects describing
  related Blizzard formats (see §19.2).

No executable binaries, symbol tables, debug information, assertion strings,
source paths, or disassembly artefacts were used in preparing this
specification. Every internal runtime structure discussed below is either
(a) directly visible in the data files, or (b) a plausible working model
named by this document's author for descriptive purposes — flagged as such
where it appears. Community-established terms ("SNO", "CPK", "CoreTOC",
"Prefetch") are retained because they have entered public vocabulary
independently of any official source.

This document is intended solely to support interoperability — specifically,
writing one's own parsers for the data files — as permitted by Directive
2009/24/EC Article 6 (Polish implementation: UoPA Art. 75 §2–3). Users are
expected to possess a lawful copy of the game.

---

## Quickstart

The repository ships self-contained Python files (stdlib only — no external
dependencies; Python 3.10+):

```
cpklib.py       — reader + writer library (CPKArchive + CPKWriter,
                  byte-identical round-trip, same-size in-place replace)
cpkrebuild.py   — extension layer: replace_file / delete_file / add_file /
                  set_zlib_level / CPKWriter.create_new(version=6|7, endian=…)
cpk_tool.py     — command-line front-end: info / list / extract / verify /
                  replace / delete / add / create
d3cpk_extractor.py
                — extract everything, with a report and dumps of unreadable
                  chunks (d3cpk_extractor_v7.py is the same tool)
d3cutscene.py   — reader for the raw cutscene CPK variant
d3cutscene_tool.py
                — command-line front-end for cutscene info / list / validate /
                  extract
coretoc_parser.py
                — parse CoreTOC.dat into CSV / JSON (§4)
prefetch_parser.py
                — parse Prefetch.dat into a dependency graph (§5)
lua_tool.py     — recover embedded Lua scripts as pseudo-Lua (§14)
lua51.py, luadec.py
                — Lua 5.1 chunk reader / disassembler, and decompiler
```

The files work as they are, or `pip install .` adds the commands `cpk-tool`,
`d3cutscene-tool`, `d3cpk-extractor` and `lua-tool`.

Common operations:

```bash
# inspect an archive
python3 cpk_tool.py info Act3.cpk

# check it: CRC, tables, and that every chunk really inflates
python3 cpk_tool.py verify Act3.cpk

# list contents — filter by extension, sort by uncompressed size
python3 cpk_tool.py list Act3.cpk --ext stl --sort size --reverse

# extract every texture file to a directory
python3 cpk_tool.py extract Common.cpk --ext tex --out ./textures

# replace a single asset (same size — fast, no rebuild)
python3 cpk_tool.py replace Act3.cpk 'AnimSet\Goblin.ans' new.ans --out modded.cpk

# resize-replace, delete, or add a file (forces full rebuild; --level 0..9)
python3 cpk_tool.py add Act3.cpk 'Test\Note.txt' note.txt --out modded.cpk --level 9
python3 cpk_tool.py delete Act3.cpk 'Some\Old.tex' --out modded.cpk
python3 cpk_tool.py replace Act3.cpk 'Anim\bigger.ans' bigger.ans --out modded.cpk --resize --level 6

# build a fresh archive from a directory tree
python3 cpk_tool.py create ./assets --out fresh.cpk --version 6 --endian big

# extract everything, with a report and dumps of unreadable chunks
python3 d3cpk_extractor.py Act3.cpk ./output

# parse the two index files
python3 coretoc_parser.py CoreTOC.dat ./toc
python3 prefetch_parser.py Prefetch.dat CoreTOC.dat ./prefetch

# recover the embedded Lua scripts, with listings and the original chunks
# (add --exact for source that recompiles to the same instructions)
python3 lua_tool.py extract Common.cpk CacheAct1.cpk --out ./scripts --asm --raw

# inspect the raw Ogg cutscene archive
python3 d3cutscene_tool.py info enUS_Cutscene.cpk --scan-ogg

# list class creation and Act 1 wizard interstitial movies
python3 d3cutscene_tool.py list enUS_Cutscene.cpk --long --groups \
  --iname 'Hero_Create|Interstitial_Act1_00_Wiz_M'

# validate every declared cutscene as a complete Ogg stream
python3 d3cutscene_tool.py validate enUS_Cutscene.cpk --progress

# extract the full Data_D3/Locale/... tree
python3 d3cutscene_tool.py extract enUS_Cutscene.cpk --out ./cutscenes \
  --validate --progress

# or extract only basenames, useful for quick media inspection
python3 d3cutscene_tool.py extract enUS_Cutscene.cpk --out ./hero --flat \
  --iname Hero_Create --validate
```

`replace` patches same-size replacements in-place and falls back to a full
rebuild via the extension layer when the new data no longer fits in its sector
slots. A replacement of a different size needs `--resize`, which always
rebuilds. Write commands produce a new file: `--out` cannot be the source
archive.

An in-place `replace` updates every stored copy of a file (§3.5). A full
rebuild writes each file once, in plain chunks, so the output differs from the
source even where the content is identical; rebuilding a 600 MB archive takes
one to three minutes.

`extract` and `read()` memory-map the archive and inflate only the sectors a
file overlaps, located with a binary search of `cs2do`. A chunk that will not
inflate is zero-filled so that offsets stay aligned, which looks like success
in the output; `verify` is the check for that. It exits with status 1 and
names the affected files.

`info`, `list`, `extract` and `verify` in `cpk_tool.py` recognise raw cutscene
archives and hand them to `d3cutscene.py`, where `verify` validates each Ogg
stream. `d3cutscene_tool.py` offers the Ogg-specific options.

Tests use synthetic archives only (no game data). CI runs them on Ubuntu and
Windows with Python 3.10 and 3.14:

```
python3 -m unittest discover -s tests
```

The cutscene tool targets the raw cutscene CPK layout observed in localized
`*_Cutscene.cpk` archives. These archives reuse the normal CPK header, bit-packed
file table, Locations table, and filename table, but the payload region is a raw
concatenation of complete Ogg/Theora/Vorbis files rather than encrypted zlib
sectors. The extractor derives byte ranges from the table data, preserves the
stored `Data_D3\Locale\...\Cutscenes\*.ogv` paths by default, and can validate
each range as an Ogg stream before writing it.

---

## 1. Build under observation

| Field | Value | Source of identification |
|---|---|---|
| Game | Diablo III | Packaging |
| Target platform | Console release (7th generation), and a later handheld release | File naming, packaging, archive variant |
| Archive variant examined | CPK v6 (big-endian) | Magic + version field in the archive header |

The observations in this document apply to the big-endian console CPK v6
data distribution. A v7 little-endian variant is also supported by the
reference reader (see §3.3.1).

Two builds are covered. **Build A** is the big-endian v6 build: 28 CPK archives,
217,396 files. **Build B** is a later little-endian v7 build with Act 5: 23 archives,
347,833 files. Counts called corpus-wide mean build A, and other counts name what they
cover. Sections describe build A unless they say otherwise. Nothing in this document
that comes from data alone has been tested in the game.

---

## 2. Asset identifier — the SNO key

*"SNO" is a community-established abbreviation for the asset-identifier system
used in multiple Blizzard titles. It is retained here as a term of art.*

Every asset in the shipped data is uniquely addressed by the pair
`(snoGroup, snoHandle)`. These two fields appear verbatim in `CoreTOC.dat`,
in `Prefetch.dat`, and throughout SNO file bodies as cross-references.

### 2.1 Two encodings observed on disk

**Expanded encoding** (seen in file headers, cross-reference fields, and
`CoreTOC.dat` entries):

```
struct SNOKey_Expanded {
    int32  snoGroup;    // small non-negative integer; -1 = invalid
    int32  snoHandle;   // up to 2^24 − 1; -1 = invalid
};  // 8 bytes
```

**Packed encoding** (seen in compact key fields inside SNO file bodies and in
archive index entries):

```
uint32 packed = (snoHandle << 8) | (snoGroup & 0xFF)
```

The low byte carries the group; the upper three bytes carry the handle.

### 2.2 Range limits inferred from format geometry

- The packed encoding allocates 8 bits to the group → at most 256 groups
  representable; the empirically observed populated range is 0..69 with
  five gaps (§4.4).
- The packed encoding allocates 24 bits to the handle → at most
  16,777,216 assets per group.
- Combined theoretical address space: ≈ 1.17 billion slots.

Values `snoGroup == -1` and `snoHandle == -1` consistently mark "no asset"
entries; the game treats any record containing them as unresolved.

---

## 3. CPK archive format (Blizzard's variant)

> **Not to be confused with CRI Middleware's CPK.** Blizzard's archive
> container reuses the acronym but is unrelated. The CRI format uses magic
> `"CPK "` (`0x43504B20`); Blizzard's uses `0xA1B2C3D4`.

### 3.1 Magic and version

| Field | Value | Notes |
|---|---|---|
| Magic | `0xA1B2C3D4` | First 4 bytes of the file, in the archive's byte order |
| Format version | 6 (BE) or 7 (LE on observed builds) | `uint32` right after the magic, at `0x04` |

### 3.2 General archive layout

The archive consists of a fixed header followed by **six** index tables (four
of them bit-packed) and a fixed-width sector region for compressed payloads. The layout —
confirmed by a round-trip parser exercised on every archive in the shipped
distribution — is:

```
┌─────────────────────────┐
│ Header (64 B v6 / 72 B  │  magic, version, decomp_fs, format constants,
│  v7; bit-packed widths) │  per-field bit widths, sector CRC32
├─────────────────────────┤
│ SortedFileInfo[]        │  per-file entries: FNV-1a name hash, decomp size,
│  (bit-packed)           │  Locations-range count + first-index
├─────────────────────────┤
│ Locations[]             │  decompressed-stream offset of every stored
│  (bit-packed)           │  copy of a file (§3.5)
├─────────────────────────┤
│ CompSectorToDecompOff[] │  decompressed-stream offset for every compressed
│  (bit-packed, "cs2do")  │  sector — used to locate sector boundaries
├─────────────────────────┤
│ DecompSectorToCompSec[] │  inverse map: comp-sector index for every
│  (bit-packed, "ds2cs")  │  decompressed sector — used for fast seeks
├─────────────────────────┤
│ FileName offsets[]      │  uint32 array, one entry per SFI record,
│  (1 × uint32 per file)  │  pointing into FileName data
├─────────────────────────┤
│ FileName data           │  null-terminated backslash-separated paths
│  (latin-1, NUL-term.)   │
├─────────────────────────┤
│ (alignment padding to   │  zero-filled up to first_sec; legacy layout
│  next read_sector)      │  only (§3.3.1.1)
├─────────────────────────┤
│ Compressed sector data  │  comp_sector_count × comp_sector bytes; chunks
│  (fixed-width slots)    │  of zlib data, optionally encrypted
└─────────────────────────┘
```

Differences vs. earlier descriptions of this format in the wild:

- **`ds2cs`** (DecompSectorToCompSector) gives `O(1)` random access into the
  global decompressed stream by mapping an offset straight to its compressed
  sector. A reader can do without it at `O(log n)` by binary-searching
  `cs2do`, which is what `cpklib.py` does.
- The `FileName` section is two tables, not one: a `uint32` offset array
  followed by a NUL-terminated string blob.

The four bit-packed index tables (`SortedFileInfo`, `Locations`, `cs2do`,
`ds2cs`) use per-field bit widths declared in the header (§3.3.1). Each table
starts on a byte boundary; unused bits in its last byte are padding.

### 3.3 Format parameters

| Parameter | Value |
|---|---|
| Endianness (word-level) | big-endian (v6) / little-endian (v7) |
| Magic bytes on disk | `A1 B2 C3 D4` (BE) / `D4 C3 B2 A1` (LE) |
| `read_sector` (read unit size) | `0x10000` |
| `comp_sector` (compressed sector size) | `0x4000` |
| Header size | 64 bytes (v6) / 72 bytes (v7) |

**Bit-packer note.** The `SortedFileInfo`, `Locations`, `cs2do` and `ds2cs`
tables are MSB-first within each byte — the same
convention used by Luigi Auriemma's QuickBMS script for this format.

### 3.3.1 Header byte-level layout

Every byte of the 64-byte (v6) / 72-byte (v7) header is accounted for. Field
widths, contents, and (where applicable) algebraic relationships to other
fields:

| Off. | Size | Field | Meaning / value |
|---:|---:|---|---|
| `0x00` | 4 | `magic` | `0xA1B2C3D4` (byte order on disk reveals endianness) |
| `0x04` | 4 | `version` | `6` (v6, BE) or `7` (v7, LE on observed builds) |
| `0x08` | 8 | `decomp_fs` | total bytes in the decompressed global stream |
| `0x10` | 4 | `fmt_const` | layout discriminator. `2` = legacy layout (RoS, Act\*.cpk, CoreCommon.cpk). `10` = new layout introduced with **Patch 2.6.7** (May 2019). `0` = raw cutscene archive (plain Ogg payload, no zlib sectors). All 23 archives of build B use `2`, cutscenes included. See §3.3.1.1 |
| `0x14` | 4 | `file_count` | number of `SortedFileInfo` entries |
| `0x18` | 4 | `loc_count` | number of `Locations` entries: `file_count` plus any extra copies of files (§3.5) |
| `0x1C` | 4 | `header_sector` | legacy (`fmt_const=2`): `(first_compressed_sector_offset / read_sector)` so `first_sec = header_sector × read_sector`, always on a `read_sector` boundary. New (`fmt_const=10`): direct byte offset (`first_sec = header_sector`). |
| `0x20` | 4 | `fs_bc` | bit-width of size field in `SortedFileInfo` |
| `0x24` | 4 | `flcnt_bc` | bit-width of copy-count field in `SortedFileInfo` (8 in `CacheAct1.cpk`, 2 in `Common.cpk`) |
| `0x28` | 4 | `flidx_bc` | bit-width of first-Locations-index field in `SortedFileInfo` |
| `0x2C` | 4 | `loc_bc` | bit-width of `Locations` entries; equal to `bit_length(decomp_fs)` |
| `0x30` | 4 | `cs2do_bc` | bit-width of `cs2do` entries; equal to `loc_bc` (both index the decompressed stream) |
| `0x34` | 4 | `ds2cs_bc` | bit-width of `ds2cs` entries; equal to `bit_length(comp_sector_count)` |
| `0x38` | 4 | `sector_crc32` | CRC32 (zlib polynomial) of the entire compressed-sector region — every byte from `first_sec` to end-of-file. See §3.5.3 |
| `0x3C` | 4 | `read_sector` (v7 only; zero-padding on v6) | `0x00010000` |
| `0x40` | 4 | `comp_sector` (v7 only) | `0x00004000` |
| `0x44` | 4 | reserved (v7 only) | always zero on observed data |

v6 archives end the header at offset `0x40` (64 bytes); v7 extends to `0x48`
(72 bytes) to make `read_sector` and `comp_sector` explicit, since both are
runtime-defined on v7 (in practice every observed v7 archive uses the same
v6 values).

#### 3.3.1.1 Two layouts: `fmt_const` discriminator

Patch 2.6.7 (Necromancer balance pass) introduced a refactored archive
layout that ships side-by-side with the legacy one. The **only** byte that
distinguishes them is `fmt_const` at `0x10`, and it changes the meaning of
`header_sector`:

| Aspect | Legacy (`fmt_const = 2`) | New (`fmt_const = 10`) |
|---|---|---|
| Meaning of `header_sector` | count of `read_sector` units; `first_sec` always falls on a `read_sector` boundary | direct **byte offset** to the first compressed chunk; no rounding |

Everything else — table order, bit-packing, FNV-1a 64-bit hashing, the
encrypted-zlib chunk format, sector-CRC32 location, name-string layout —
is unchanged.

**Sector 0 holds ordinary chunks.** On build A its first chunk declares size 0
and stores 8 bytes, an empty zlib stream. On build B it is real data, in 20 of 21
non-empty archives. A reader that skips it reads wrong bytes with no error, which
is why `verify` compares each sector's inflated size with its `cs2do` gap.

Observed archives by layout:

| Archive group | `fmt_const` |
|---|:-:|
| `Act1.cpk` … `Act4.cpk`, `CacheActN.cpk`, `Common.cpk`, `CacheCommon.cpk`, `CoreCommon.cpk` | `2` |
| `Patch2_6_7.cpk`, `Patch2_6_8.cpk`, `Patch2_7_3.cpk` … `Patch2_7_7.cpk` | `10` |
| `<locale>_Cutscene.cpk` (raw Ogg payload, read with `d3cutscene.py`) | `0` |
| All of build B, including `<locale>_*Cutscene.cpk`, which hold zlib-packed `.ogv` files | `2` |

### 3.4 Chunk compression and encryption

All compressed content in CPK is **zlib**. There is no second compression
algorithm. Some chunks carry an additional **encryption** layer; the
`0x8000` bit in the `zsize_raw` field distinguishes the two cases:

| Flag | Pipeline to recover a file |
|---|---|
| `zsize_raw & 0x8000 == 0` | `zlib.decompress(chunk)` |
| `zsize_raw & 0x8000 != 0` | `zlib.decompress(stream_cipher_decrypt(chunk))` |

Each sector slot holds a run of chunks. Every chunk starts with a 6-byte
header of three `uint16` fields in the archive's byte order:

| Off. | Field | Meaning |
|---|---|---|
| `+0` | `decomp_low` | low 16 bits of the decompressed size |
| `+2` | `decomp_high` | high 16 bits of the decompressed size |
| `+4` | `zsize_raw` | bits 0–14: stored chunk size; bit 15 (`0x8000`): encrypted |

A `zsize_raw` of zero ends the sector's list of chunks. Declared sizes can exceed
16 bits: the largest seen is 2,495,344 B.

**Stream cipher** — derived by analysing paired ciphertext / known-plaintext
samples (zlib streams have highly constrained initial bytes, making the
keystream recoverable for the first tens of bytes and then extending
algorithmically). The recovered scheme is:

```
CIPHER_INIT_STATE = 0x873AE8E53C11D761   # 64-bit constant, global to all chunks

def decrypt_chunk(ciphertext: bytes) -> bytes:
    state = CIPHER_INIT_STATE
    out = bytearray()
    for c in ciphertext:
        out.append(c ^ (state & 0xFF))
        # Ciphertext-feedback step: each new ciphertext byte
        # becomes the most-significant byte of the next state.
        state = ((c & 0xFF) << 56) | (state >> 8)
        state &= 0xFFFFFFFFFFFFFFFF
    return bytes(out)
```

Cipher properties inferred from the observed behaviour:

- **Stream cipher with ciphertext feedback (CFB-like).** The update uses
  the *ciphertext* byte, not the plaintext — confirmed by verifying the
  identity `state_{n+1} = (c_n << 56) | (state_n >> 8)` against several
  chunks of known plaintext.
- **No IV / nonce.** The same initial state is used for every chunk in
  every archive.
- **Output is a standard zlib stream**, decompressable by any conformant
  zlib implementation.

This explains two earlier red herrings. The observed Shannon entropy of
"encrypted" chunks (≈ 7.84 bits/byte) is close to maximum because zlib
output itself is near-maximum entropy, and XOR with a keystream preserves
that. Similarly, the frequently seen prefix `19 0d` on encrypted chunks is
simply a `78 DA` zlib header XOR-ed with the two low bytes of the initial
state (`0x78 ⊕ 0x61 = 0x19`, `0xDA ⊕ 0xD7 = 0x0D`).

Because of the feedback, the initial state only influences the first 8
bytes of a chunk — but those include the zlib header, so with a wrong
constant every encrypted chunk fails to inflate as a whole.

Encryption is the exception: 1,316 of 73,644 chunks in `CacheAct1.cpk` and 635
of 65,734 in `Common.cpk`.

Known-plaintext attack via cross-platform corpus diff: the same logical
asset appears in both big-endian and little-endian archive variants. A
subset of chunks is stored as raw zlib on one variant and as the encrypted
form on the other. Aligning a known-plain chunk with its encrypted
counterpart yields the keystream by a single XOR operation over the full
chunk length. Validating the recovered keystream against additional chunks
confirms (a) whether the cipher carries internal state feedback, and
(b) whether any IV or per-chunk nonce is in use.

### 3.5 Additional format invariants worth noting

1. **Zero-fill on zlib short output.** Some chunks decompress to fewer
   bytes than `decomp_size` declares. The shortfall must be zero-padded
   so that byte offsets in the global decompressed stream stay in sync
   with the `Locations` table. Chunks do not span sector boundaries in
   practice. No such chunk occurs in `Common.cpk` or `CacheAct1.cpk`.
2. **Orphaned SortedFileInfo entries.** Some archives declare a
   `decomp_fs` in the header that exceeds the sum of physically present
   chunks by a small margin (low tens of kilobytes), leaving a handful
   of file entries that address a region the archive never actually
   filled. A correct extractor zero-fills the global stream up to
   `decomp_fs`, producing those entries as zero-filled placeholders
   rather than erroring.
3. **Chunks cut off at the slot boundary.** The last chunk of nearly every
   sector stops exactly at the end of the slot, before its zlib stream ends
   (35,621 of 35,640 sectors in `CacheAct1.cpk`, 37,346 of 37,357 in
   `Common.cpk`). One-shot `zlib.decompress` rejects these chunks. A streaming
   `decompressobj` returns exactly `decomp_size` bytes, and each sector's
   output then matches the distance to the next `cs2do` entry. A few sectors
   instead end in a stub of one or two zlib header bytes with `decomp_size` 0.
4. **Repeated files.** A `SortedFileInfo` row can own several consecutive
   `Locations` entries, each one a complete copy of the file, not a fragment:
   `decomp_fs` equals the sum of `size × copies`, with no gaps. `CacheAct1.cpk`
   has 38,039 locations for 22,915 files, with 2,820 files stored more than
   once, while `Common.cpk` has two such files and no other non-cache archive
   has any, and build B stores every file once. Repetition follows file type: 12 extensions repeat in the five
   `Cache*` archives, from `.acr` (85% of files) and `.mat` (72%) down to `.con`
   (10%), while `.snd`, `.prt`, `.tex`, `.ani`, `.app` and `.pow` never do. Only
   files listed as `Prefetch.dat` dependencies repeat (0 of the 11,462 others).
   The largest repeated file is 22,676 B, and extra copies cost 0.5 to 1.6% of a
   stream. None has a locale tag. Copies match byte for byte, except that in 869
   files they differ in the `flags` and `fileHash` header fields (§7.1). With
   all copies, 0.90 to 0.93 of a prefetch group's dependencies of these types
   lie within 1 MiB of each other, against 0.67 to 0.72 with one copy. The rule
   that decides when a copy is written has not been found, and the seek saving
   has not been measured. Tools that write in place must update every copy.
   Build B instead repeats 15,559 names across archives, all byte-identical.

### 3.5.3 Sector CRC32

The 4 bytes at header offset `0x38` carry a CRC32 (standard zlib
polynomial, `crc32` in any common library) computed over the contiguous
compressed-sector payload — i.e. `data[first_sec:]` where
`first_sec` is the start of the first compressed sector (§3.3.1.1).

Verified on every shipped archive examined: stored value matches recomputed
CRC exactly. This implies the engine validates archive integrity at load
time; tools that rebuild an archive must recompute and rewrite this field,
otherwise the rebuilt file will differ from a freshly-built archive in this
single 4-byte position even when sector contents are otherwise valid.

Reference (Python):

```python
import struct, zlib
with open("Archive.cpk", "rb") as f:
    data = f.read()
# … parse header to recover header_sector and read_sector …
first_sec = header_sector if fmt_const == 10 else header_sector * read_sector
assert struct.unpack(endian + "I", data[0x38:0x3C])[0] \
       == zlib.crc32(data[first_sec:]) & 0xFFFFFFFF
```

### 3.6 Archive packaging

Archive filenames in the shipped distribution exhibit a clear per-act
partitioning. Alongside act-specific archives (one set per act, I–V) there
is a "common" archive carrying cross-act shared content, and cutscene
archives follow the same partitioning. This is observable directly from
the shipped file listing; it is not derived from any internal
enumeration.

---

## 4. CoreTOC.dat

`CoreTOC.dat` is a global index mapping every `(snoGroup, snoHandle)`
pair to a human-readable asset name.

### 4.1 Binary format

Two forms. Build A is big-endian and flat, with a single count:

```
uint32 BE:  count
for each of `count` entries (136 bytes each):
    uint32 BE:  snoGroup
    uint32 BE:  snoHandle
    char[128]:  name (null-padded ASCII)
```

Build B is little-endian and segmented (211,835 entries):

```
int32 count[70]; int32 offset[70]; int32 count2[70]; uint32 unknown    // 844 bytes
for each group g, at 844 + offset[g]:
    { int32 group; int32 handle; int32 nameOffset; }[count[g]]         // 12 bytes
    NUL-terminated names; nameOffset counts from the end of the records
```

`count2` equals `count` in all 70 slots. The trailing word is unexplained.
`coretoc_parser.py` reads both forms.

### 4.2 Comparison with the PC variant

| Aspect | Build A | Build B |
|---|---|---|
| Endianness | big-endian | little-endian |
| Header | 4 bytes (count only) | 844 bytes (per-group entry counts, offsets, reserved fields) |
| Layout | flat, single record type | segmented per group |
| Record size | 136 B (fixed) | 12 B, plus names |

The PC form has the same segments. The `jybp/casc` table describes an 844-byte header as
above, while WhiteoutLib reads a 1,120-byte header with a fourth 70-entry array. Neither
PC form was tested here.

### 4.3 Groups absent from the PC mapping

The `CoreTOC.dat` examined here contains groups absent from the PC
`SnoExtensions` table (see the jybp/casc project). The most directly
verifiable example is `snoGroup = 52` → `TreasureClass` / `.trs`, which
is not present in any published PC mapping.

### 4.4 Canonical mapping `snoGroup → (name, extension)`

Compiled by correlating names in `CoreTOC.dat` with asset-name strings
embedded elsewhere in the data and with the publicly available
`SnoExtensions` table from the PC modding community.

**PC-shared core** (confirmed against the `SnoExtensions` table in the
jybp/casc project, i.e. cross-referenced against a third-party
open-source artefact):

| ID | Name | Ext. | ID | Name | Ext. |
|:-:|:--|:-:|:-:|:--|:-:|
| 0 | *(empty)* | — | 37 | Shaders | `.shd` |
| 1 | Actor | `.acr` | 38 | Shakes | `.shk` |
| 2 | Adventure | `.adv` | 39 | SkillKit | `.skl` |
| 5 | AmbientSound | `.ams` | 40 | Sound | `.snd` |
| 6 | Anim | `.ani` | 41 | SoundBank | `.sbk` |
| 7 | Anim2D | `.an2` | 42 | StringList | `.stl` |
| 8 | AnimSet | `.ans` | 43 | Surface | `.srf` |
| 9 | Appearance | `.app` | 44 | Textures | `.tex` |
| 11 | Cloth | `.clt` | 45 | Trail | `.trl` |
| 12 | Conversation | `.cnv` | 46 | UI | `.ui` |
| 14 | EffectGroup | `.efg` | 47 | Weather | `.wth` |
| 15 | Encounter | `.enc` | 48 | Worlds | `.wrl` |
| 17 | Explosion | `.xpl` | 49 | Recipe | `.rcp` |
| 19 | Font | `.fnt` | 51 | Condition | `.cnd` |
| 20 | GameBalance | `.gam` | 56 | Act | `.act` |
| 21 | Globals | `.glo` | 57 | Material | `.mat` |
| 22 | LevelArea | `.lvl` | 58 | QuestRange | `.qsr` |
| 23 | Light | `.lit` | 59 | Lore | `.lor` |
| 24 | MarkerSet | `.mrk` | 60 | Reverb | `.rev` |
| 25 | Monster | `.mon` | 61 | PhysMesh | `.phm` |
| 26 | Observer | `.obs` | 62 | Music | `.mus` |
| 27 | Particle | `.prt` | 63 | Tutorial | `.tut` |
| 28 | Physics | `.phy` | 64 | BossEncounter | `.bos` |
| 29 | Power | `.pow` | 66 | Accolade | `.aco` |
| 31 | Quest | `.qst` | | | |
| 32 | Rope | `.rop` | | | |
| 33 | Scene | `.scn` | | | |
| 34 | SceneGroup | `.scg` | | | |
| 36 | ShaderMap | `.shm` | | | |

**Groups outside that table.** Names come from joining the word at +0x10 of each
file to `CoreTOC.dat`, then comparing the file stem and the archive directory.
IDs 1 to 48 also run alphabetically, which fits 13 and 35.

| ID | Name | Ext. | Status | Evidence |
|:-:|:--|:-:|:--|:--|
| 3 | AiBehavior | `.aib` | confirmed | `ConstantAttack`, `PoltahrEscortFollow`, `SkeletonKing` |
| 4 | AiState | `.ais` | confirmed | `*_Attack`, `*_Idle` |
| 13 | ConversationList | `.cnl` | confirmed | 246 files; +0x30 holds a Conversation handle in 240 |
| 18 | FlagSet | `.flg` | confirmed | `GameFlags`, `PlayerFlags` |
| 35 | Script | `.scr` | confirmed | `Script\` directory; a Lua chunk in 794 of 794 files |
| 52 | TreasureClass | `.trs` | confirmed | `MonsterDropSpellRune_Normal`, `LootClicky_Type1` |
| 54 | Conductor | `.con` | confirmed | `Conductor\` directory; a Lua chunk in 250 of 250 files |
| 55 | TimedEvent | `.tme` | confirmed | `TimedEvent\` directory; 32-byte files such as `ZK Random Dungeon` |
| 65 | ControlScheme | `.ctr` | confirmed | Only entry is the placeholder `Axe Bad Data`, whose one file has handle 0. On build B it is 460 B, exactly the listed `ControlScheme` struct (444 B) plus the header (§7.3) |
| 67 | AnimTree | `.ant` | confirmed | `AnimTree\` directory; `NPC`, `Player`, `NPC_AdditiveFlinch` |
| 68 | Vibrations | `.vib` | confirmed | `Vibrations\` directory; `Console_LowHealth`, `Barbarian_WeaponThrow`. Rumble curves is an inference |
| 69 | DungeonFinder | `.dfn` | confirmed | Build B only: `DungeonFinder\` directory, 4 files |

**Summary:**

- 53 IDs come from the PC-shared core: 52 named groups plus the empty slot 0.
- 12 more groups are named here from files on disk.
- **65 distinct group IDs identified.** IDs 10, 16, 30, 50 and 53 within the
  0..69 observed range have no entries in `CoreTOC.dat`.

---

## 5. Prefetch.dat

`Prefetch.dat` encodes the assertion *"when asset X is loaded, the following
assets Y, Z, W, … should also be preloaded"*. The layout is the same on both
builds, in the build's byte order, and every byte of both files is accounted for.

### 5.1 Binary format

```
uint32:  parent_count
parent records (16 bytes each):
    uint32:  snoGroup            # parent group
    uint32:  snoHandle           # parent handle
    uint32:  dep_count           # number of this parent's dependencies
    uint32:  first_dep           # index of its first dependency
uint32:  total_deps_count        # = Σ dep_count
dependency records (8 bytes each):
    uint32:  snoGroup
    uint32:  snoHandle
```

Parent *i* owns records `[first_dep, first_dep + dep_count)`. The ranges partition
the list without gaps or overlaps, but not in parent order.

### 5.2 Format invariants (empirically validated)

- Build A: 77,186 parents and 376,385 dependencies. Build B: 147,638 and 844,045.
- **Parent `snoGroup` and `snoHandle` resolve in `CoreTOC.dat`** for every record.
- **`total_deps_count` equals Σ dep_count.**
- **Not every dependency resolves:** 420 on build A and 24,872 on build B do not.
- On build B, parents are sorted by the low 16 bits of the handle.

### 5.3 Semantic examples (verified end-to-end)

- `AnimSet\Monk_Male` → 406 animations.
- `Power\EmoteWait` → conversation `Hero_Wait`.
- `Globals\globals` → 362 banner textures and 9 actors.

---

## 6. Asset Registry (runtime model, working name)

> The term **"Asset Registry"** is used in this document as a working name
> for the logical structure that the game uses to index loaded assets by
> `(snoGroup, snoHandle)`. It is not claimed as an official identifier.

The invariants observable from the data files constrain any plausible
runtime model:

- **Approximately 60–70 group slots.** Observable as the populated range
  of `snoGroup` values in `CoreTOC.dat` entries and in `Prefetch.dat`
  parent/dependency records. Exact cardinality cannot be pinned from data
  alone because gaps exist in the populated range.
- **Per-group handle namespaces.** `snoHandle` values collide across groups
  and are unique only within a group.
- **Stable cross-references.** Every `(snoGroup, snoHandle)` appearing as a
  dependency in `Prefetch.dat` or as a reference inside an SNO file body
  resolves to a `CoreTOC.dat` entry, so the registry must support
  lookup by `(snoGroup, snoHandle)`.
- **Stable name lookup.** The game is observed to resolve asset names to
  handles (for example, when user-supplied strings in diagnostic commands
  or developer-mode interfaces reach the asset subsystem), so a
  name → handle index exists in some form.

Runtime implementation details beyond these invariants — such as the number
of parallel TOCs held in memory, the presence of a "pending act" slot, or a
developer-mode loose-file override manifest — are not observable from the
shipped data files and are not asserted here.

---

## 7. SNO file header

Every SNO file on disk begins with a small fixed-size header. The same header
format is shared across every observed group type (see §11).

### 7.1 Header fields

Observable by parsing the first bytes of any SNO file extracted from a CPK
archive:

| Offset | Field | Type | Meaning |
|---|---|---|---|
| +0x00 | `magic` | uint32 | `0xDEADBEEF` |
| +0x04 | `version` | uint32 | Same for every file of a group, e.g. Power `0x5A2`, Actor `0x113`, Monster `0x4D5`, Textures `0x3A` (see §11) |
| +0x08 | `flags` | uint32 | Mostly 1; up to 7 on build A and 17 on build B; meaning not determined from data alone |
| +0x0C | `fileHash` | uint32 | Working name. Zero exactly when `flags` is zero. Input unknown (below) |
| +0x10 | `snoHandle` | uint32 | Matches `CoreTOC.dat` in 133,131 of 133,132 files; the exception is a stub with handle 0 |
| +0x14, +0x18 | | uint32 | Zero in all 133,132 files |

The first four big-endian uint32s form a 16-byte universal prefix, and offsets
inside an SNO body count from its end, +0x10. The body type decides what follows
+0x1C. Files listed in §7.3 lack this header.

**Why identical copies differ, on build A.** `flags` and `fileHash` are zero in every copy
that crosses a compressed-sector boundary: 17,382 of 17,382 across the five
`Cache*` archives. Each of the 869 files with differing copies has exactly one
stamped state and one zero state. Another 463 zero copies lie inside one sector,
and all of them touch a sector edge or end within 4 bytes of one. As 418 stamped
copies also touch an edge, the exact rule is open. A rebuild changes which copies
cross a sector, and nothing here shows whether the game reads either field. Build B has
no zero state: all 274,414 SNO files are stamped, including 27,512 that cross a sector.

**What `fileHash` is not.** Unseeded CRC, Adler-32, MD5, SHA-1, FNV, Jenkins and
Murmur values of the body or the name do not match it, and it does not cover the
visible bytes alone:
479 identical bodies carry more than one hash, and 24 hashes span different
bodies. It does not depend on platform bytes either: of 85,222 assets shared by both
builds with a stamped hash, 47,703 have equal hashes though none is byte-identical. An
equal hash predicts equal logic: 868 of 870 such scripts have the same instructions or
differ only in constants, and no formula text changed in 7,677 records of 377 such powers.
So it is most likely a hash of the source asset before cooking, and it cannot be
recomputed from the archives.

### 7.2 Payload organisation

The remainder of an SNO file after its 16-byte header is a
type-specific body (see §12 and §13 for concrete examples). The body is
read in a single pass for most asset types; for types with auxiliary buffers
(notably `.app` files, §13) the body references further buffers whose sizes
are given in the fixed section.

### 7.3 Files without the SNO header

| File | Layout |
|---|---|
| `.can` | Build A only; build B stores this inside `.ani`. `Anim\<stem>\0000.can` pairs with `Anim\<stem>.ani` (9,140 of 9,140), and `.ani` +0xC8 gives its track count. Starts with `u32 tableBytes`, then `tableBytes / 4` track-name hashes, then three curve sections and attachments. Each section is zero padding, a `u32` table size, then its table. A curve table is 16 B per track, `{offset, size, 0, 0}`, with offsets from the table's start. A blob's byte 0 is its key count and byte 1 its codec. Sections walk in 8,332 of 9,136 files |
| `.nav` | Build A only; build B stores this inside `.scn`. `Scene\<stem>\0000.nav` pairs with `Scene\<stem>.scn` (1,730 of 1,747 scenes have one). Six sections in a row: a grid of 4 B squares, 32 B cells, 4 B records, 6 B grid squares, and two lists of 4 B records. The sum matches the file size in 2,135 of 2,147 scenes shared with build B. The rule for a short grid is open |
| `.dat`, magic `0xAABB0004` | SNO manifest: `{u32 group, u32 sno}` records after the magic. On build A, `Misc\ActN.app.dat` and `Misc\ActN.game.dat` together list exactly the SNOs of that act's archive and its locale archives. That does not hold on build B |
| `.dat`, magic `0xBBCC0005` | Sprite table: `{u32 nameHash, u32 texSno, u32 frame}` records after the magic. The hash is §12's, case-sensitive |
| `.sbt` | UTF-8 subtitle text with CRLF line ends, cue lines `HH:MM:SS:cc - HH:MM:SS:cc` |
| `.achu` | Protobuf wire format; no schema recovered |
| `.fsb` | Plain FSB5 sound banks |
| `.ogg`, `.ogv` | Plain Ogg |
| `ComplexTypeDescriptorSizes.dat` | Build B only: `u32 0xAABBCCDD`, then 396 records of `{char name[64], u32 size, u32 fieldOffset[256]}`, on 64-bit layout. It lists every SNO type's struct, and its name, size and offsets guided §12 and §13's build B layouts |

`CoreTOC.dat` and `Prefetch.dat` have no magic: their first words are counts.

---

## 8. Lookup, hashing, and the FNV-1a algorithm

### 8.1 Overall lookup behaviour

From observing how the game resolves cross-references, a two-level lookup
model is consistent with all observed behaviour:

```
function Lookup(snoGroup, snoHandle):
    # Fast path: small, densely populated handles use direct indexing.
    if snoHandle < FAST_TABLE_SIZE:
        return fastTable[snoHandle]
    # Invalid sentinels short-circuit.
    if snoHandle == -1 or snoGroup == -1: return None
    if snoGroup outside the observed populated range: return None
    # Slow path: per-group hash table keyed on snoHandle.
    bucket = fnv1a_32_uint32(snoHandle) & mask_for(snoGroup)
    return hash_tables[snoGroup].find_in_chain(bucket, snoHandle)
```

The choice of FNV-1a is directly verifiable from the CPK archive
(see §8.2).

### 8.2 FNV-1a constants (verifiable from CPK name hashes)

The `SortedFileInfo` table (§3.2) stores a **64-bit** FNV-1a hash for every
path in the `FileName` section. Hashing the lower-cased ASCII path, separators
as stored, yields the stored value for every entry (cross-validated across
every archive examined), which establishes both the algorithm and its
constants:

```
FNV1A64_OFFSET_BASIS = 0xCBF29CE484222325
FNV1A64_PRIME        = 0x100000001B3
```

The lookup model in §8.1 uses the **32-bit** variant with its standard
constants. No CPK table stores a 32-bit hash, so this part of the model cannot
be confirmed from the archives:

```
FNV1A_OFFSET_BASIS = 0x811C9DC5
FNV1A_PRIME        = 0x01000193   # 16,777,619
```

Reference implementation:

```python
def fnv1a_32(data: bytes) -> int:
    h = 0x811C9DC5
    for b in data:
        h = ((h ^ b) * 0x01000193) & 0xFFFFFFFF
    return h
```

For lookup purposes, `fnv1a_32_uint32(handle)` is the four-byte
little-endian representation of the handle fed into the same function.

---

## 9. Asset sourcing and locales (observations)

Two kinds of information are sometimes associated with an asset beyond the
SNO key itself:

1. **Source selector** — whether the asset should be fetched from inside
   an archive or from a loose file on disk. Distinct sourcing paths for
   these two cases are evidenced by the fact that the developer workflow
   supports loose-file overrides, but retail builds use only the archive
   path. The specific encoding used to record this preference inside the
   game is not determinable from data files, and a clean-room parser of
   the retail distribution does not need it: every retail asset comes
   from an archive.
2. **Locale tag** — per-asset locale distinguishes language variants for
   text and voice. Locale tagging is evidenced in asset naming (e.g.
   trailing `|xxYY` suffixes in archive entry names such as
   `StringList\Foo.stl|jpJP`) and in the per-locale audio packaging.
   See §17 for the set of locales observed.

Neither a runtime manifest structure combining these attributes with the
SNO key nor its specific binary layout is part of the data-file format
— this is deliberately not described here.

---

## 10. Group metadata

Each observed `snoGroup` has a canonical textual name. These names appear
verbatim in `CoreTOC.dat` entries (the 128-byte name field in each record;
see §4.1) and also in asset path prefixes inside CPK archives (e.g. paths
of the form `Actor\...`, `Power\...`, `Monster\...`). The table in §4.4
compiles those names for every identified group.

Beyond the name, the game appears to associate additional metadata with
each group — the set of valid schema versions, whether the group carries
a particular domain classification, and the reflection handler used to
deserialize its file body. This metadata is **not** stored in any shipped
data file and is deliberately out of scope for this document; clean-room
parsers do not need it to read the on-disk format.

One behavioural observation is worth stating: the authoring infrastructure
appears to support a user-generated-content data domain in addition to
the shipped "title" data, but no UGC ever shipped to consoles and this
mode produces no observable content in retail builds.

---

## 11. Shared deserialization framework (observation)

Every SNO file begins with the same 16-byte universal header (§7.1), and
every attempt to load a file whose schema fingerprint (`version` —
the second word of the header) disagrees with the value the runtime
expects for its group produces a refusal rather than a partial load.
These two facts — identical header layout across groups and consistent
version validation — are consistent with a single, schema-driven
deserializer parameterised by a per-group type descriptor rather than
per-group hand-written parsers.

The practical consequences for a clean-room parser are:

1. **Validate the universal header first.** Reject a file immediately if
   its `magic` or `version` does not match the value established for its group
   from known-good samples.
2. **Expect the body layout to be stable within a patch.** The header's
   `version` functions as a compatibility token: a
   single patch ships a single body layout per group.
3. **Expect body layouts to vary across patches.** An archive from patch
   A may not parse under a schema recovered from patch B even if both
   share group IDs and extensions.

Beyond these invariants, specifics of the runtime deserialization
machinery (in-memory type descriptors, dispatch mechanics, and similar
runtime-only artefacts) are not observable from data files and are not
asserted here.

---

## 12. Worked example: `.pow` (Power) file layout

All 1,021 `.pow` files in the corpus share one layout. The fixed section is
0x1A0 = 416 bytes including the 16-byte header. Tag maps, an optional list of
Power SNOs and the script follow it, so the smallest file is 628 B. Each pair is
`{offset, size}`, and file position = offset + 0x10.

| File offset | Content | Notes |
|---|---|---|
| 0x000 | File header (§7.1) | |
| 0x020, 0x030, 0x040 | Tag maps `M0`, `General`, `PVPGeneral`: pair + 8 zero bytes | `M0` and `PVPGeneral` always empty |
| 0x050 to 0x068 | Tag maps `Contact0` to `Contact3`: four pairs | |
| 0x090 to 0x0A8 | Tag maps `PVPContact0` to `3`: four pairs | Always empty; `PVP` names are guesses |
| 0x0D0 | `powerFlags`, uint32 | Bits 0 to 16 used; meanings open |
| 0x0D4, 0x0E4 | Two collision blocks of four uint32 | Named `ActorCollisionFlags` in build B's type list |
| 0x0F8 to 0x177 | Eight 16-byte slots, each a pair then 8 zero bytes, for optional lists of uint32 Power SNOs | Slots 0 and 2 in use, in 2 files and 1 file |
| 0x178 | `scripted`, boolean | |
| 0x17C | `useSecondaryResource`, boolean | |
| 0x180 | `formulaCount`, int32 | 0 to 40 |
| 0x190 | `compiledScriptRange` pair | Non-zero in 595 files, 3 of them empty chunks |
| 0x198 | `questMetaDataRef` | Always −1 |

All other bytes of the fixed section are zero, and it holds no float field. The
Lua chunk (§14) is at `offset + 0x10` and always ends the file.

**Build B** keeps this layout, little-endian, with 32 slots at 0x0F8, of which each list
item is 8 B. Everything after the slots moves up by 0x180, so the fixed section is
0x320 = 800 B and the script pair is at 0x310. All 3,006 files parse. Tag IDs rose
to 871, and five tags changed from type 0 to type 4, so use each record's own type.

**Tag maps.** A map is a `uint32 count` followed by records of
`uint32 type, uint32 tag, value`:

| Type | Value | Records |
|---|---|---|
| 0 | int32: enum, flag or hash | 13,608 |
| 1 | float32 | 88 |
| 2 | SNO handle, 4 bytes | 872 |
| 3 | 32-bit hash or −1 | 48 |
| 4 | Formula | 13,866 |
| 6 | Attribute id | 49 |

Attribute numbers differ by build, so none can be carried across.

Bytecode keeps each opcode in the first byte of its word, with operands in the build's
byte order. A formula value is five zero uint32s, `textLength`, a zero uint32 and
`codeLength`, then the source text padded to 4 bytes, then stack bytecode. The
text is readable, for example `Rune_A ? 3 : 1`. Of 13,456 formulas recompiled,
13,452 gave the stored bytecode.

**Script formulas.** `Hydra.PowerGetFormulaValue(n)` reads the type-4 record in
`General` with tag `0x41100 + (n / 10) * 0x100 + (n % 10) * 0x10`. Of 2,933 such
calls tested, 2,917 reach a non-empty formula. `n` runs to 39 on build A and 63 on
build B.

**String hash.** Several tags and the sprite tables (§7.3) use
`h = (h * 33 + byte) mod 2^32`, seed 0, on case-sensitive names. The player class
tag `0x507B0` applies it to the lower-cased name. Most hash values are unresolved.

> Note. A runtime representation of an active power (on-screen, in combat)
> does not appear in the on-disk `.pow` file and is outside the scope of
> this document. The `.pow` file encodes the *definition*; the instantiated
> state lives only in memory during play.

---

## 13. Worked example: `.app` (Appearance) file layout

`.app` files encode a 3D model: one or more meshes, skinning data, and
packed vertex/index buffers.

### 13.1 Top-level structure

The file begins with the 16-byte universal header (§7.1), followed by a
per-type fixed section. All offsets count from +0x10. Sizes below hold across
the corpus: 6,033 files, 114,846 sub-objects and 126,953 draws. Only the listed
fields are decoded.

| Offset | Field | Notes |
|---|---|---|
| +0x24 | bone count | |
| +0x28 / +0x2C | bone array offset / size | 176 bytes per bone |
| +0x70 / +0x74 / +0x78 | sub-object count / offset / size | 308 bytes each |
| +0xC8 / +0xCC / +0xD0 | hardpoint count / offset / size | 40 bytes each |
| +0x18C / +0x190 / +0x194 | material count / offset / size, build A | 20 bytes each |

### 13.2 Sub-object record — 308 bytes each

| Offset | Field | Notes |
|---|---|---|
| +0x4C | name string | Seen on a sample file only |
| +0xD0 | bounds centre | float × 3 |
| +0xDC | bounds half-extent | float × 3 |
| +0xFC / +0x100 / +0x104 | draw count / offset / size | 160 bytes each |
| +0x118 | position bias | float × 3 |
| +0x124 | position scale | float × 3 |

### 13.3 Draw record — 160 bytes each

| Offset | Field | Notes |
|---|---|---|
| +0x08 / +0x0C | descriptor offset / size | 16 bytes. Byte 0 is the format; bytes 8 to 11 repeat the vertex and index counts as uint16 |
| +0x10 | index count | Divisible by 3 in every draw, which points to triangle lists |
| +0x18 / +0x1C | index buffer offset / size | |
| +0x24 | vertex count | |
| +0x2C / +0x30 | stream A offset / size | May be padded to 16 bytes |
| +0x54 / +0x58 | stream B offset / size | Size 0 when rigid |
| +0x6C / +0x70 | stream C offset / size | |

### 13.4 Vertex and index widths

Indices are 16 bits wide. Vertices are split into streams, and no layout has a
32-byte vertex. Format 0x81 occurs only in files without bones and format 0x83
only in files with bones. Stream sizes match these strides in all 126,953 draws.

**Rigid, format 0x81: stream A 6 bytes, stream C 20 bytes**

| Stream | Offset | Type | Field |
|---|---|---|---|
| A | +0 | int16 BE × 3 | position |
| C | +0 | uint8 × 3 | normal |
| C | +3 | uint8 | always 0 |
| C | +4, +8 | uint8 × 4, twice | colour-like values, meaning unknown |
| C | +12 | uint16 **LE** × 2 | UV1 |
| C | +16 | uint16 **LE** × 2 | UV2, scale unknown |

**Skinned, format 0x83: stream A 10 bytes, stream B 8 bytes, stream C 16 bytes**

| Stream | Offset | Type | Field |
|---|---|---|---|
| A | +0 | int16 BE × 3 | position |
| A | +6 | uint8 × 3 | normal |
| A | +9 | uint8 | always 0x80 |
| B | +0 to +7 | (uint8, uint8) × 4 | (bone weight, bone index) pairs |
| C | +0, +4 | uint8 × 4, twice | colour-like values, meaning unknown |
| C | +8 | uint16 **LE** × 2 | UV1 |
| C | +12 | uint16 **LE** × 2 | UV2, scale unknown |

Decoding:

- **Position** is `raw × scale + bias`. The bias is required: decoded positions
  fit the bounds tightly only with it, in all 17,030 sub-objects tested.
- **Normal** is `(byte − 128) / 128`, axes X, Y, Z. Triangles are counter-clockwise
  around it in index order.
- **UV1** is `(raw − 0x8000) / 512`. The evidence is strong, but no decoded UV has
  been checked against a texture.
- **Bone weights** sum to 255 in descending order. At most three were non-zero
  across the 227,439 records checked. Indices stay below the bone count and are
  spatially coherent, but they have not been matched to skeleton positions.
- **Tangents** are not stored: every byte has another field.
- **Bones:** word 0 of a bone record is its name hash, and the parent index follows at
  +4: -1 or an earlier bone, in 250 of 250 files. Model-space translation is at +0x40.
  Skinned bone indices are **local to the draw**, not skeleton indices. Where build A
  keeps the map to the skeleton has not been found.

### 13.5 Build B

All 13,403 files parse. Offsets of the top-level fields are unchanged. Bone records are
256 B, translation at +0x44, parent at +8. Sub-objects are 240 B, and there are no draw
records. Vertices are interleaved and little-endian:

| Layout | Stride | Content |
|---|---|---|
| Rigid | 28 | Position int16 × 3, 2 B padding, normal, colour × 2, UV1, UV2 |
| Skinned | 36 | The same, then 4 bone indices and 4 weights |

| Sub-object field | Offset |
|---|---|
| Vertex count | +0xA0 |
| Vertex offset / size | +0xB0 / +0xB4 |
| Skinning record offset / size | +0xB8 / +0xBC |
| Influence stream offset / size, 6 B per vertex | +0xC8 / +0xCC |
| Position bias, scale | +0xD8, +0xE4 |
| Index count | +0x28 |

Rigid indices are 16-bit, in one pool at fixed-section body +0x210 / +0x214, and each
sub-object starts at the running sum of earlier counts. A skinning record is 104 B:
palette count +0, palette +4 / +8, vertex count +0x18, vertices +0x28 / +0x2C, index
count +0x30, indices +0x40 / +0x44. Indices go through that palette.

Between builds the geometry is unchanged: in 1,300 of 1,300 rigid sub-objects compared,
triangles and vertices match, in a different order.

### 13.6 Alignment

The buffers have no 32-byte alignment. In 18,967 draws from 900 files, index and
stream A offsets are 16-byte aligned in 8,953 and at least 4-byte aligned in all
but 7, and many sub-object arrays are 4-byte aligned only. Tools that edit `.app`
files should keep the offsets they find.

---

## 14. Embedded scripting — Lua 5.1

### 14.1 Version identification

The compiled script blobs embedded in SNO files begin with the
standard Lua chunk signature `1B 4C 75 61 51` (`\x1BLuaQ`), which
identifies **Lua 5.1** specifically. The opcode set used in these blobs
enumerates 38 opcodes in the exact Lua 5.1 order (distinguishable from
5.2, 5.3, and LuaJIT by both count and ordering). The chunk header declares big-endian data, 4-byte integers, 4-byte
`size_t` and 8-byte floating-point numbers. Every chunk is stripped: there are
no source names, line tables or local variable names, while global names, field names
and strings survive. Each function's `linedefined` and `lastlinedefined` do survive.
Build B differs in being little-endian with an 8-byte `size_t`, so chunks do not carry
across builds.

Scripts occur in four file types. Each stores an offset + size pair at a fixed
position, and the offset counts from the end of the 16-byte file header:

| Type | Pair at | Blobs in `Common.cpk` + `CacheAct1.cpk` | Content |
|---|---|---|---|
| `.pow` (Power) | `0x190` | 108 | Skill logic; most `.pow` files have no blob |
| `.scr` (Script) | `0x20` | 261 | Actor and event scripts |
| `.con` (Conductor) | `0x48` | 101 | Scripted sequences, as `Phase_<n>_Sequence_<n>_Start` handlers |
| `.cnv` (Conversation) | `0x1E8` | 655 | Nearly always empty: 653 chunks hold only `RETURN` |

On build B the pairs are at `0x310` for `.pow` and `0xE8` for `.cnv`, and unchanged for the
other two. It has 8,145 chunks, 3,913 of them distinct.

Scripts reach the engine through one global table, `Hydra`, of which 597 names
are used in these two archives, such as `Hydra.PowerGetFormulaValue` or
`Hydra.ActorPlaySimpleConversation`. Across all 28 archives, scripts use 581
`Hydra` functions and 458 named constants. The engine calls back into functions the
script defines as globals, such as `OnStart` and `OnContact0`.

### 14.2 Practical consequences for tooling

- `lua_tool.py extract` finds the chunks by parsing them, decompiles each one
  and skips the empty ones. Paths follow the internal names, so
  `Power\Foo.pow` becomes `<out>/Power/Foo.pow.lua`. `--asm` adds a `.lst`
  disassembly and `--raw` the original `.luac` chunk, which standard Lua 5.1
  decompilers such as `unluac` and `luadec` can also read.
- The output is pseudo-Lua: logic and names are recovered, but because the
  chunks are stripped, parameters print as `a0`, other locals as `r<register>`
  and captured variables as `up<depth>_a<n>` or `up<depth>_r<register>`. A jump that fits no
  structure prints as `goto L<pc>` with a Lua 5.2 style label.
- The 4,755 chunks of the corpus reduce to 1,501 distinct ones, of which one is
  empty. Recompiling the text of the 1,571 script files with non-empty chunks and
  comparing each function's call sequence with the original bytecode gives:

  | Result | Files |
  |---|---|
  | Compiles, call order matches | 1,444 |
  | Uses `goto`, compiles as Lua 5.2, not compared | 86 |
  | Compiles, call order differs in places | 22 |
  | `goto` into a nested block, does not compile | 19 |

  7,505 of 7,527 compiled functions match. This covers call order, not the full
  logic: the pseudo-Lua can lose a local assignment that no call depends on. Treat
  `.lst` as authoritative where the two seem to disagree.
- `lua_tool.py extract --exact` writes real Lua 5.1 source instead, through
  `relua.py`. It recovers `local` scopes from register allocation and never prints
  `goto`, at the cost of plain names (`v<id>`). All 1,571 files recompile with
  Lua 5.1 to the same instructions, registers, jumps and constant values as the
  original, ignoring the order of the constant table and stack size. Where the
  bytecode cannot tell two forms apart, one is chosen: `if a then if b` prints as
  `if a and b`. On build B, 3,909 of 3,913 distinct chunks match in the same way. The
  other four differ in a jump or an extra `LOADNIL`. The scripts have only been
  compiled, never executed.

### 14.3 Parameter tagging

Small-integer tags are observed on script-bound parameter fields in SNO
bodies, distinguishing integer, float, string, and nil-valued parameters.
Each script-bound parameter carries a type tag alongside its value, so the
reader must dispatch on that tag. §12 lists the record types of `.pow` tag maps.

### 14.4 Event parameters

Compiled Lua chunks contain string constants naming the kinds of values
the engine pushes into a script as event parameters. These names are
directly observable as string literals in the bytecode constants pool
after decompiling a representative sample of `.pow` files. They describe
common game-object references (actors, monsters, worlds, players,
quests, markers, …), time/range values, and hashed-string keys. A
clean-room Lua analyzer can enumerate the full set empirically from a
corpus of decompiled chunks; no enum definition needs to be imported.

---

## 15. Field schema — tag taxonomy (concept)

The shared deserialization framework (§11) is driven by per-field type
tags. Integer tag values are directly observable in SNO file bodies as
leading bytes of individually tagged substructures; their symbolic
semantics are inferred by correlating how fields of a given tag behave
across many samples (numeric range, SNO-cross-reference resolution,
enum-like clustering, …).

From the shipped data, the schema exhibits three distinct dimensions:

- **A large vocabulary of field-tag values** spanning primitives
  (integer, float, boolean, angle, tick, velocity, …), SNO-reference
  tags (actor, monster, power, world, quest, boss encounter, …),
  asset-reference tags (texture, sound, particle, effect group,
  animation, …), and specialised runtime tags.
- **A secondary grouping of tags by thematic category**, used by the
  authoring tool (actor categories, power categories, animation
  categories, …). This grouping is visible from the layout of tagged
  substructures in data but exact cardinality cannot be fixed without
  access to a schema definition.
- **Disjoint 64 K-slot ID spaces** for tunable IDs. IDs drawn from the
  shipped data consistently fall in ranges of the form
  `0x10000..0x1FFFF`, `0x20000..0x2FFFF`, and so on; each range
  clusters with IDs of a specific thematic type (actor tuning, power
  tuning, shader map, power effect, surface, attribute, marker,
  appearance, animation, world/DRLG grouping, monster, …). A
  clean-room parser can reliably partition unknown IDs into categories
  by range alone, without needing to know the full enumeration.

Full enumeration of every tag's symbolic name is not required for
interoperability: a parser needs only (a) correct tag-to-field-width
mappings for primitives, and (b) recognition of the SNO-reference tag
width: 4 bytes in `.pow` tag records (§12).

---

## 16. Act structure

Five main acts plus an act-independent "common" partition are directly
visible in archive naming, in `Prefetch.dat`, and in asset categorisation:

- Acts I through V (V = Reaper of Souls).
- "Common" — assets shared across all acts (UI, globals, shared monster
  families, …).

The engine internally separates assets into loading-schedule categories
that distinguish client-side from server-side requirements (so that a
dedicated-server configuration can skip textures, UI, and similar
client-only assets). The specific enumeration used internally is not
stored in any data file and is not needed by a clean-room parser; the
observation that such a split exists is sufficient to explain why some
asset classes appear only in certain packaging subsets.

---

## 17. Localization

Assets can exist in locale-variant form, typically for text and voice.
This is observable directly in archive entry names: localized variants
are tagged with a `|xxYY` suffix (e.g. `StringList\Foo.stl|jpJP`) that
marks the locale code. Per-locale audio packaging is visible in the
distribution as separate locale subsets.

The set of locale codes observed in packaging is:

```
ENGLISH_EU, ENGLISH_SOUTHEAST_ASIA, SPANISH_EU, SPANISH_LATAM, FRENCH,
ITALIAN, GERMAN, KOREAN, PORTUGUESE_BRAZIL, PORTUGUESE_PORTUGAL, RUSSIAN,
CHINESE_CHINA, CHINESE_TAIWAN, TURKISH, POLISH, JAPANESE
```

Sixteen audio locales total — this matches the number of locale-specific
packaging subsets and is independently verifiable from the shipped
distribution.

---

## 18. Developer-only features (not applicable to retail builds)

Various artefacts in the data suggest the engine supports developer-time
workflows — hot-reload of modified assets, loose-file overrides of archived
content, and an RPC protocol between clients and an authoring server.
These features appear to be gated behind a command-line flag that is not
enabled in retail builds; they do not produce observable behaviour in the
shipped game and are out of scope for this document.

---

## 19. Tools

### 19.1 Reference parsers (RedMadKnight, publishable)

Built and exercised against shipped data files during the preparation of
this document:

| Tool | Role | Input / output | Status |
|---|---|---|---|
| `cpklib.py` | Reference Python library: archive reader (full parser for v6/v7), `CPKWriter` with byte-identical round-trip, same-size in-place file replacement | `*.cpk` ↔ Python objects | Round-trip verified on every observed archive |
| `cpkrebuild.py` | Extension layer adding full-rebuild operations: `replace_file` (size-changing), `delete_file`, `add_file`, configurable zlib level (`set_zlib_level(0..9)`), and `CPKWriter.create_new(version=6\|7, endian=…)` for from-scratch construction | live `CPKWriter` | Validated on archives from 4 MB to 80 MB; from-scratch archives reopen and round-trip cleanly |
| `cpk_tool.py` | Command-line front-end: `info` / `list` / `extract` (with extension/regex/size/locale filters) / `verify` / `replace` / `delete` / `add` / `create` (build new archive from a directory tree) | shell | All subcommands round-trip-verified |
| `d3cpk_extractor.py` | Whole-archive extraction with a text report and dumps of chunks that fail to inflate; built on `cpklib.py` (`d3cpk_extractor_v7.py` is the same tool) | `*.cpk` → directory tree | Run on `Common.cpk`: all 28,394 files extracted |
| `d3cutscene.py` / `d3cutscene_tool.py` | Reader and command-line front-end for raw cutscene archives (`fmt_const = 0`): info / list / validate / extract | `*_Cutscene.cpk` → `.ogv` | Contributed; not re-verified here against shipped data |
| `lua51.py` / `luadec.py` / `relua.py` / `lua_tool.py` | Lua 5.1 chunk reader and disassembler, decompilers to pseudo-Lua and to compilable source, and command-line front-end | `*.cpk` → `.lua` / `.lst` / `.luac` | Run on all 4,755 chunks of the corpus (§14.2) |
| `d3sno/` | Library and command line for both builds: archives, index files, powers and formulas, meshes with OBJ and glTF export, animation, navigation, textures to PNG, text, balance tables, worlds, collision, materials, sound, and writers for `.stl`, `.pow`, `.scr` and `.tex` | `*.cpk` and files inside them | Reads all 54 archives of both builds; `.stl`, `.pow`, `.scr`, `.tex` write back byte-identical |
| `coretoc_parser.py` | Global asset index parser, both forms | `CoreTOC.dat` | All observed groups identified |
| `prefetch_parser.py` | Preload dependency graph parser, both byte orders | `Prefetch.dat` + `CoreTOC.dat` | Every byte accounted for on both builds |

`verify`, on-demand reading, the faster decryption, the sector packer and the
current `d3cpk_extractor.py` were checked on two shipped archives, `Common.cpk`
and `CacheAct1.cpk` (v6, big-endian, `fmt_const = 2`): CRC and every chunk pass,
each file read on demand matches the global stream, a save without changes is
byte-identical, and rebuilt archives hold the same content. On v7 and
`fmt_const = 10` archives they have only run against the synthetic test suite.
`verify` also passed on all 28 archives of build A (§1), and after the sector 0 fix on the
6 archives rechecked from both builds. Agreement with a second reader was checked on
those, not on all 51.
Nobody has yet loaded a rebuilt archive in the game.

### 19.2 Reference open-source projects (cross-reference)

Publicly available projects covering related Blizzard formats, used to
cross-check independent observations:

| Project | Author | Relevance |
|---|---|---|
| `diablo3_xbox.bms` | Luigi Auriemma | QuickBMS script demonstrating CPK extraction with `zlib_noerror` on magic `\xA1\xB2\xC3\xD4` |
| CPKReaderWV | zeroKilo | CPK header parser (header-info only); full extraction is in this repository (§19.1) |
| jybp/casc | jybp | Open-source `SnoExtensions` table for the PC CASC build — shares its core with the console mapping (see §4.4) |
| StormLib | Ladislav Zezula | Reference MPQ implementation — not directly applicable |
| unluac, luadec | tehtmi; viruscamp and others | General-purpose Lua 5.1 decompilers, usable on the chunks from `lua_tool.py --raw`. No public project found that targets the Diablo III scripts themselves |
| CascLib | Ladislav Zezula | Reference CASC implementation — useful for cross-checking hypotheses about compression algorithms (its `CascDecompress` is zlib-only) |

---

## 20. Structural reference — consolidated

All format-level sizes, constants, and algorithmic parameters in one place:

| Item | Value | Source |
|---|---|---|
| SNO key (expanded encoding) | 8 bytes | Format layout |
| SNO key (packed encoding) | 4 bytes | Format layout |
| SNO file header | 16 bytes, magic `0xDEADBEEF` | On-disk prefix |
| `.pow` fixed section | 416 bytes (build B: 800), including the header | §12 |
| `.app` sub-object record | 308 bytes (build B: 240) | §13.2, §13.5 |
| `.app` draw record | 160 bytes (build B: none) | §13.3 |
| `.app` rigid vertex streams | 6 + 20 bytes | §13.4 |
| `.app` skinned vertex streams | 10 + 8 + 16 bytes | §13.4 |
| `.app` build B vertex | 28 bytes rigid, 36 bytes skinned | §13.5 |
| `.app` index width | 16 bits | §13.4 |
| CPK magic | `0xA1B2C3D4` | Empirical |
| CPK format version | 6 (BE) / 7 (LE) | Empirical |
| CPK cipher initial state | `0x873AE8E53C11D761` | Recovered by known-plaintext attack against zlib prefixes (see §3.4) |
| CPK encryption flag | bit `0x8000` in `zsize_raw` | Empirical + cipher verification |
| CPK `read_sector` (read unit size) | `0x10000` | Header (implicit on v6, explicit on v7 at offset `0x3C`) |
| CPK `comp_sector` (compressed sector size) | `0x4000` | Header (implicit on v6, explicit on v7 at offset `0x40`) |
| CPK header size | 64 bytes (v6) / 72 bytes (v7) | Header |
| CPK header field `fmt_const` at `0x10` | `0x2` (legacy) / `0xA` (new layout, Patch 2.6.7+) / `0x0` (raw cutscene) — see §3.3.1.1 | §3.3.1 |
| CPK header field `cs2do_bc` at `0x30` | = `loc_bc` = `bit_length(decomp_fs)` | §3.3.1 |
| CPK header field `ds2cs_bc` at `0x34` | = `bit_length(comp_sector_count)` | §3.3.1 |
| CPK header field `sector_crc32` at `0x38` | CRC32 of `data[first_sec:]` (zlib polynomial) | §3.5.3 |
| FNV-1a 32-bit offset basis | `0x811C9DC5` | Standard value; lookup model (§8.1) |
| FNV-1a 32-bit prime | `0x01000193` (16,777,619) | Standard value; lookup model (§8.1) |
| FNV-1a 64-bit offset basis | `0xCBF29CE484222325` | Cross-validated on CPK `SortedFileInfo` hashes |
| FNV-1a 64-bit prime | `0x100000001B3` | Cross-validated on CPK `SortedFileInfo` hashes |
| Lua bytecode signature | `1B 4C 75 61 51` | On-disk prefix of script blobs |
| Lua opcode set | Lua 5.1 (38 opcodes, standard order) | Decoded script bytecode |
| Audio locales | 16 | Packaging directory names |
| `CoreTOC.dat` record size | 136 bytes (build B: 12, plus names) | Empirical |
| `Prefetch.dat` parent record | 16 bytes | Empirical |
| `Prefetch.dat` dependency record | 8 bytes | Empirical |

---

## 21. Coverage and open issues

### 21.1 Solved

- **CPK archive format** — compression (zlib) and encryption (stream
  cipher with fixed initial state `0x873AE8E53C11D761`) both recovered
  and round-trip-verified against shipped archives.
- **Complete byte-level header layout** (v6: 64 B, v7: 72 B) — every byte
  accounted for; see §3.3.1.
- **Sector-payload CRC32** at header offset `0x38` recovered and verified
  across all shipped archives; rebuild tooling must recompute it
  (§3.5.3).
- **Six-table archive layout** — SortedFileInfo, Locations, cs2do, ds2cs,
  FileName-offsets, FileName-data — confirmed by round-trip parser
  (§3.2).
- **Full archive rebuild + from-scratch construction** — `cpkrebuild.py`
  supports `replace_file`, `delete_file`, `add_file`, configurable zlib
  level, and `CPKWriter.create_new(version=6|7, endian=…)`. From-scratch
  archives reopen and round-trip cleanly through the reference reader.
- **`CoreTOC.dat`** — both forms fully parsed.
- **`Prefetch.dat`** — fully parsed on both builds, format invariants validated
  against `CoreTOC.dat`.
- **`snoGroup → (name, extension)` mapping** — every populated group
  observed in the shipped data has an entry (via the PC-shared core for
  most groups, plus 12 more named from files on disk; §4.4).
- **SNO key representation** (expanded / packed) and its range limits.
- **Universal SNO file header** (16 bytes, §7.1).
- **Shared deserialization framework as an observation** — verified by
  identical header layout across groups and consistent version
  validation.
- **Embedded scripting identified as Lua 5.1** — signature and opcode
  count both match the 5.1 baseline exactly, making the shipped
  bytecode decompilable with off-the-shelf tools.
- **Script recovery** — every Lua chunk in `.pow`, `.scr`, `.con` and `.cnv`
  files located and decompiled to pseudo-Lua, and the result recompiled and
  compared with the bytecode across the corpus (§14.2). `--exact` output matches
  the original instructions in all 1,571 files.
- **`.pow` layout** — fixed section, tag maps, record types and formula lookup,
  checked on all 1,021 files (§12).
- **`.app` vertex layout** — record sizes and stream strides on all 126,953
  draws, with position, normal and bone data decoded, and build B's layout
  on 13,403 files (§13).
- **Why copies of a file differ** — header zeroing at sector crossings (§7.1).
- **Hashing algorithm** — 64-bit FNV-1a with the standard constants,
  verified against the `FileName` and `SortedFileInfo` sections of CPK
  archives.
- **Bit-packer invariant** — MSB-first within each byte in the
  `SortedFileInfo`, `Locations`, `cs2do` and `ds2cs` tables.
- **Two on-disk layouts** discriminated by `fmt_const` at header
  offset `0x10` (legacy `2` vs new `10`, the latter introduced with
  Patch 2.6.7). Differences confined to (a) the encoding of
  `header_sector` — multiple of `read_sector` vs direct byte offset —
  and (b) the absence of a sector-0 skip preamble in the new layout.
  Reader and writer in `cpklib.py` branch on `fmt_const`; round-trip
  identity holds for both layouts on every shipped archive tested
  (Act\*.cpk, CoreCommon.cpk, Patch2_6_7…2_7_7.cpk).

### 21.2 Open

- **`.pow` details** — `powerFlags` bits, collision blocks, and the names of most
  of the 559 tags and hash values (§12).
- **`flags` and `fileHash` in the SNO header** (§7.1) — the meaning of `flags`, the
  input of the hash, and the exact zeroing rule at sector edges on build A.
- **Why files are repeated** inside an archive (§3.5) — the load unit behind the
  copies, and whether the game accepts a rebuilt archive that stores each file
  once.
- **`.app` details** (§13) — the colour-like values, UV2, the UV1 divisor and V
  direction, the source of tangents if any, and where build A keeps its bone
  palette.
- **Non-SNO files** (§7.3) — `.can` codec payloads and the 804 files that do not
  walk, and the `.nav` short-grid rule.
- **`.tex`** — not decoded.

### 21.3 Highest-leverage next steps

1. **Test in the game** — load a rebuilt archive and a changed script. This is the
   only way to learn whether the header fields are read, and every result here
   comes from data alone.
2. **Decode `.tex`** — this would also test the UV1 divisor against real
   images.

---

## Appendix A — Asset load sequence (observable effects)

From observing the order and timing of file opens, reads, and completion
events at the file-system layer (and via packet capture where the game
issues network requests for patchable content), the asset-load pipeline
has the following observable stages:

```
Startup:
    → read Prefetch.dat                   (build dependency graph in memory)
    → read CoreTOC.dat                    (build (group, handle) → name table)
On asset request (snoGroup, snoHandle):
    → resolve to an asset descriptor
      · fast path: direct-indexed table
      · slow path: per-group hash (FNV-1a of handle; §8)
    → if descriptor has no payload yet:
        · allocate payload record
        · resolve the (CPK archive, byte range) for this asset
        · read the 16-byte file header
        · validate magic and version (§7, §11)
        · read the body (single pass for most types)
    → if the type has auxiliary buffers (e.g. `.app`):
        · read each buffer at the offset/size declared in the fixed section
    → mark asset as loaded; fire any registered listeners
```

This ordering is observable externally (via file-system traces) and does
not rely on any runtime-internal information.

---

## Appendix B — Working-name glossary

For readers cross-referencing this document against community resources
or their own parsers, the following table collects the working names used
here against the public format-level names that are independently
observable:

| Working name (this document) | Observable counterpart |
|---|---|
| SNO key (expanded) | 8-byte `(snoGroup, snoHandle)` pair in `CoreTOC.dat` / `Prefetch.dat` |
| SNO key (packed) | 4-byte compact encoding inside SNO bodies and archive index |
| SNO file header | 16-byte prefix of an SNO file on disk (§7.1) |
| Power file layout | `.pow` file fixed section (§12) |
| Tag map | `uint32 count` plus typed tag records in `.pow` (§12) |
| Appearance file layout | `.app` file top-level structure (§13) |
| Sub-object record | 308-byte mesh sub-object, 240 on build B (§13.2, §13.5) |
| Draw record | 160-byte draw descriptor with its vertex streams (§13.3) |
| Shared deserialization framework | inferred schema-driven deserializer (§11) |
| Field tag | per-field type tag observable in SNO bodies (§15) |
| Tag ID space | one of the observed 64 K-stride ID ranges (§15) |

Community-established terms retained as-is:

- **SNO** — the asset-identifier system used in several Blizzard titles.
- **CPK** — Blizzard's archive format (distinct from CRI's CPK; the
  community Luigi Auriemma script, CPKReaderWV, and similar tools use
  this term).
- **CoreTOC**, **Prefetch** — the shipped filenames of the two index
  files, directly visible in the distribution.
- **SortedFileInfo**, **Locations**, **cs2do**, **ds2cs** — table
  names used by the open-source CPK-reading community (Auriemma's
  QuickBMS script and CPKReaderWV) and by `cpklib.py` in this
  repository.
