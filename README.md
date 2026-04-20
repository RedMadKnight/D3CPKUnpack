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
  related Blizzard formats (see §20).

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

## 1. Build under observation

| Field | Value | Source of identification |
|---|---|---|
| Game | Diablo III | Packaging |
| Target platform | Console release (7th generation) | File naming, packaging, archive variant |
| Archive variant examined | CPK v6 (big-endian) | Magic + version byte in the archive header |

The observations in this document apply to the big-endian console CPK v6
data distribution.

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
  representable; the empirically observed populated range is 0..68 with
  some gaps (roughly 55 groups carry entries in `CoreTOC.dat` plus five
  more slots observed but not definitively named — see §4).
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
| Magic | `0xA1B2C3D4` (big-endian) | First 4 bytes of the file |
| Format version | 6 | First byte after magic, in the header |

### 3.2 General archive layout

```
┌─────────────────────────┐
│ Header (bit-packed)     │  metadata + per-field bit widths
├─────────────────────────┤
│ SortedFileInfo[]        │  per-file entries: name hash, size, location range
├─────────────────────────┤
│ Locations[]             │  offsets into the global decompressed stream
├─────────────────────────┤
│ CompressedSectorChunk[] │  chunk metadata: offset, size, flags
├─────────────────────────┤
│ FileName[]              │  hash → file name (forward-slash paths)
├─────────────────────────┤
│ Chunk data (binary)     │  compressed / encrypted payload
└─────────────────────────┘
```

The three index tables (`SortedFileInfo`, `Locations`,
`CompressedSectorChunk`) are packed into a single contiguous bit-stream
using per-field bit widths declared in the header.

### 3.3 Format parameters

| Parameter | Value |
|---|---|
| Endianness (word-level) | big-endian |
| Magic bytes on disk | `A1 B2 C3 D4` |
| `read_sector` (read unit size) | `0x10000` |
| `comp_sector` (compressed sector size) | `0x4000` |
| Header size | 64 bytes |

**Bit-packer note.** The bit-stream covering `SortedFileInfo` /
`Locations` / `CompressedSectorChunk` is MSB-first within each byte —
the same convention used by Luigi Auriemma's QuickBMS script for this
format.

### 3.4 Chunk compression and encryption

All compressed content in CPK is **zlib**. There is no second compression
algorithm. Some chunks carry an additional **encryption** layer; the
`0x8000` bit in the `zsize_raw` field distinguishes the two cases:

| Flag | Pipeline to recover a file |
|---|---|
| `zsize_raw & 0x8000 == 0` | `zlib.decompress(chunk)` |
| `zsize_raw & 0x8000 != 0` | `zlib.decompress(stream_cipher_decrypt(chunk))` |

**Stream cipher** — derived by analysing paired ciphertext / known-plaintext
samples (zlib streams have highly constrained initial bytes, making the
keystream recoverable for the first tens of bytes and then extending
algorithmically). The recovered scheme is:

```
CIPHER_INIT_STATE = 0x872DCDA7A97A7EE1   # 64-bit constant, global to all chunks

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
an artefact of XOR-ing a typical `78 9C` zlib header with the first two
keystream bytes derived from the fixed initial state
(`0x78 ⊕ 0xE1 = 0x99`, subsequently flipped by the sign of the
`zsize_raw` field, yielding `0x19`).

### 3.5 Two additional format invariants worth noting

1. **Zero-fill on zlib short output.** Some chunks decompress to fewer
   bytes than `decomp_size` declares. The shortfall must be zero-padded
   so that byte offsets in the global decompressed stream stay in sync
   with the `Locations` table. Chunks do not span sector boundaries in
   practice.
2. **Orphaned SortedFileInfo entries.** Some archives declare a
   `decomp_fs` in the header that exceeds the sum of physically present
   chunks by a small margin (low tens of kilobytes), leaving a handful
   of file entries that address a region the archive never actually
   filled. A correct extractor zero-fills the global stream up to
   `decomp_fs`, producing those entries as zero-length placeholders
   rather than erroring.

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

- Endianness: big-endian throughout.
- Structure: flat, no auxiliary header beyond a single count.

```
uint32 BE:  count
for each of `count` entries (136 bytes each):
    uint32 BE:  snoGroup
    uint32 BE:  snoHandle
    char[128]:  name (null-padded ASCII)
```

### 4.2 Differences from the PC variant

| Aspect | This variant | PC (CASC) |
|---|---|---|
| Endianness | big-endian | little-endian |
| Header | 4 bytes (count only) | 844 bytes (per-group entry counts, offsets, reserved fields) |
| Layout | flat, single record type | segmented per group |
| Record size | 136 B (fixed) | variable |

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

**Console-only groups** (identified empirically by inspecting the names in
sample entries of the console `CoreTOC.dat`):

| ID | Name | Ext. | Status | Example names |
|:-:|:--|:-:|:--|:--|
| 3 | AiBehavior | `.aib` | confirmed | `ConstantAttack`, `PoltahrEscortFollow`, `SkeletonKing` |
| 4 | AiState | `.ais` | confirmed | `*_Attack`, `*_Idle` |
| 13 | NpcRole | `.npc` | confirmed | `A1C1DeckardCain`, `A1C1LostAdventurersLeader` |
| 18 | FlagSet | `.flg` | confirmed | `GameFlags`, `PlayerFlags` |
| 52 | TreasureClass | `.trs` | confirmed | `MonsterDropSpellRune_Normal`, `LootClicky_Type1` |
| 55 | Dungeon | `.dun` | confirmed | `ZK Random Dungeon`, `Jar Of Souls`, `AlcarnusRitual` |
| 35 | *(working name:* `Encounter2`*)* | `.en2` | ID confirmed, semantics inferred | alternate encounter definitions (console-only range) |
| 54 | *(working name:* `NpcRole2`*)* | `.np2` | ID confirmed, semantics inferred | `A2C2GreedyMiner`, `A2C1Poltahr` |
| 65 | *(unknown)* | `.u65` | ID present, semantics not determined | — |
| 67 | *(working name:* `NpcExtra`*)* | `.npx` | ID confirmed, semantics inferred | `NPC`, `Player`, `NPC_AdditiveFlinch` |
| 68 | *(mixed)* | `.m68` | ID present, heterogeneous content | sound-related samples: `StaticShortHigh`, `Barbarian_WeaponThrow` |

**Summary:**

- 49 groups are fully named via the PC-shared core.
- 6 additional console-only groups have confirmed names and extensions.
- 5 groups (`35`, `54`, `65`, `67`, `68`) have confirmed IDs but working
  or absent names.
- **60 distinct group IDs identified.** Several IDs within the 0..68
  observed range have no entries in `CoreTOC.dat` and appear to be empty
  or retired.

---

## 5. Prefetch.dat

`Prefetch.dat` encodes the assertion *"when asset X is loaded, the following
assets Y, Z, W, … should also be preloaded"*. The format was reverse-engineered
empirically against the big-endian shipped file and validated end-to-end
against `CoreTOC.dat`.

### 5.1 Binary format (big-endian)

```
Section 1 — parent records (16 bytes each):
    uint32 BE:  unknown_32          # the first uint32 of the file also
                                    #   functions as Section 1 count
    uint32 BE:  snoGroup            # parent group
    uint32 BE:  snoHandle           # parent handle
    uint32 BE:  dep_count           # number of this parent's dependencies

Section 2 — header (8 bytes):
    uint32 BE:  unknown_32
    uint32 BE:  total_deps_count    # = Σ dep_count from Section 1

Section 2 — body (8 bytes per record):
    uint32 BE:  snoGroup            # dependency group
    uint32 BE:  snoHandle           # dependency handle
```

Dependencies are consumed sequentially: parent *i* owns the dependency
records in the range
`[Σ dep_counts[0..i−1], Σ dep_counts[0..i])`.

### 5.2 Format invariants (empirically validated)

On the shipped file, every record satisfies:

- **Parent `snoGroup` matches an existing group in `CoreTOC.dat`** for every
  record.
- **`total_deps_count` equals Σ dep_count.**
- **Every dependency `(snoGroup, snoHandle)` resolves to a name in
  `CoreTOC.dat`.**

### 5.3 Semantic examples (verified end-to-end)

- `AnimSet\Monk_Male` → 406 dependencies (textures, actors, sounds for the
  full monk character package).
- `Power\EmoteWait` → 1 dependency: `Textures\Overlay_sandDeath`.
- `Globals\globals` → 371 dependencies (lighting, textures).

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
| +0x00 | `snoHandle` | uint32 | Handle component of the asset's SNO key |
| +0x04 | `version` | uint32 | Changes between patches |
| +0x08 | `flags` | uint32 | Per-asset flags; bit semantics not determined from data alone |
| +0x0C | `schemaHash` | uint32 | Schema fingerprint (see §11) |

The four uint32s constitute a 16-byte universal prefix.

### 7.2 Payload organisation

The remainder of an SNO file after its 16-byte header is a
type-specific body (see §13 and §14 for concrete examples). The body is
read in a single pass for most asset types; for types with auxiliary buffers
(notably `.app` files, §14) the body references further buffers whose sizes
are given in the fixed section.

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

CPK archives store `hash → file name` mappings in the `FileName` section
(§3.2). Computing the 32-bit FNV-1a over the forward-slash path yields the
stored hash for every tested entry, establishing both the algorithm and
its constants beyond reasonable doubt:

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
   `StringList/Foo.stl|jpJP`) and in the per-locale audio packaging.
   See §17 for the set of locales observed.

Neither a runtime manifest structure combining these attributes with the
SNO key nor its specific binary layout is part of the data-file format
— this is deliberately not described here.

---

## 10. Group metadata

Each observed `snoGroup` has a canonical textual name. These names appear
verbatim in `CoreTOC.dat` entries (the 128-byte name field in each record;
see §4.1) and also in asset path prefixes inside CPK archives (e.g. paths
of the form `Actor/...`, `Power/...`, `Monster/...`). The table in §4.4
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
every attempt to load a file whose schema fingerprint (`schemaHash` —
the fourth word of the header) disagrees with the value the runtime
expects for its group produces a refusal rather than a partial load.
These two facts — identical header layout across groups and consistent
schema-hash validation — are consistent with a single, schema-driven
deserializer parameterised by a per-group type descriptor rather than
per-group hand-written parsers.

The practical consequences for a clean-room parser are:

1. **Validate the universal header first.** Reject a file immediately if
   its `schemaHash` does not match the value established for its group
   from known-good samples.
2. **Expect the body layout to be stable within a patch.** The header's
   `version` / `schemaHash` pair functions as a compatibility token: a
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

**File size of the fixed section:** 0x310 = 784 bytes.

Working names are used below for substructures; `PowerBody` is a
placeholder for the cluster of fields describing a single power's
gameplay parameters, pending further per-field analysis.

```
struct PowerFileLayout {              // 784 bytes, fixed section
    uint8              header[16];    // file header (§7.1)
    PowerBody          body;          // gameplay parameters (sub-layout TBD)
    uint32             scripted;      // boolean
    uint32             useSecondaryResource;  // boolean
    int32              formulaCount;
    uint64             compiledScriptField;   // 8-byte field; zero on disk
    SerializeData      compiledScriptRange;   // offset + size of bytecode blob
    SNOKey_Expanded    questMetaDataRef;      // 8 bytes
    SNOKey_Expanded    scriptRef;             // 8 bytes
};
```

The 8-byte `compiledScriptField` is a placeholder that stores zero on
disk and is overwritten by the loader when the asset is brought into
memory; the actual location of the bytecode blob is carried by the
adjacent `compiledScriptRange` pair (offset + size), which is what a
clean-room parser should read.

**Auxiliary buffer referenced from the fixed section:**

- A trailing Lua 5.1 bytecode chunk addressed by `compiledScriptRange`
  (see §14).
- A per-power formula table — a 16-byte descriptor whose payload is an
  array of formula entries referenced by its own offset + size pair:

```
struct FormulaTable {                 // 16 bytes
    SerializeData      entriesRange;  // offset + size of the entry array
    uint64             entriesField;  // 8-byte placeholder, zero on disk
};
```

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
per-type fixed section. The fixed section carries two count + descriptor
pairs addressing two sub-object arrays — a natural fit for a
level-of-detail split (e.g. hi-res vs. proxy mesh) or a variant split;
the exact semantics of each slot are not determined from data alone.

### 13.2 Sub-object record — 240 bytes each

The sub-object record is a fixed-size descriptor carrying counts and
placeholder fields that the loader patches with in-memory addresses at
load time. On disk, the 8-byte "pointer" fields are zero; the actual
payload is located via adjacent offset + size pairs.

| Offset | Field | Size | Notes |
|---|---|---|---|
| +0x04 | vertex count | int32 | |
| +0x10 | vertex list placeholder | uint64 | zero on disk |
| +0x28 | index count | int32 | |
| +0x38 | index list placeholder | uint64 | zero on disk |
| +0xA8 | packed vertex list placeholder | uint64 | zero on disk |
| +0xB0 | packed vertex offset | int32 | offset to packed vertex blob |
| +0xB4 | packed vertex size | int32 | size of packed vertex blob |
| +0xBC | skinning-record count | int32 | |
| +0xC0 | skinning-record list placeholder | uint64 | zero on disk |

Vertex and index arrays are located by walking the file past the fixed
section; the offset + size pairs in skinning records (§13.3) and in the
sub-object itself allow direct location of each buffer.

### 13.3 Skinning record — 104 bytes each

| Offset | Field | Notes |
|---|---|---|
| +0x20 | packed vertex list placeholder | zero on disk |
| +0x28 | packed vertex offset | |
| +0x2C | packed vertex size | |
| +0x38 | packed index list placeholder | zero on disk |
| +0x40 | packed index offset | |
| +0x44 | packed index size | |

### 13.4 Vertex and index widths

- **Vertex stride:** 32 bytes. Inferred from `32 × vertex_count` matching
  the size of the vertex-list region for multiple sample files.
- **Index width:** 16 bits. Inferred from `2 × index_count` matching the
  size of the index-list region.

The exact layout within a 32-byte vertex (position, normal, UV, bone
indices / weights, tangent frames) is **not** determinable from the
`.app` file alone; it is implicit in the renderer's vertex declaration
and would require shader analysis to recover. A clean-room parser can
successfully round-trip positions, indices, and topology without the
per-field breakdown.

### 13.5 Alignment

Sub-object and skinning buffers are 32-byte-aligned on disk. Tools that
emit `.app` files for interoperability should preserve this alignment.

---

## 14. Embedded scripting — Lua 5.1

### 14.1 Version identification

The compiled script blobs embedded in `.pow` files (§12) begin with the
standard Lua chunk signature `1B 4C 75 61 51` (`\x1BLuaQ`), which
identifies **Lua 5.1** specifically. The opcode set used in these blobs
enumerates 38 opcodes in the exact Lua 5.1 order (distinguishable from
5.2, 5.3, and LuaJIT by both count and ordering). Compiled chunks carry
the debug information that the reference Lua 5.1 compiler emits by
default.

### 14.2 Practical consequences for tooling

- Extract the bytecode blob referenced from `.pow` fixed-section fields
  (`compiledScriptRange`; see §12).
- Feed it to any open-source Lua 5.1 decompiler (`unluac`, `luadec`) —
  no custom decoder is required.
- Expect per-skill source-level logic to be recoverable as readable Lua,
  including variable and function names where the compiler preserved them.

### 14.3 Parameter tagging

Small-integer tags are observed on script-bound parameter fields in SNO
bodies, distinguishing integer, float, string, and nil-valued parameters.
The specific numeric encoding is not needed for a clean-room parser of
the data format itself — the observation simply states that each
script-bound parameter carries a type tag alongside its value, so the
reader must dispatch on that tag.

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
width (an `SNOKey_Expanded`, 8 bytes).

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
are tagged with a `|xxYY` suffix (e.g. `StringList/Foo.stl|jpJP`) that
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

### 19.1 Reference parsers (author's, publishable)

Built and exercised against shipped data files during the preparation of
this document:

| Tool | Role | Input | Status |
|---|---|---|---|
| `d3cpk_extractor.py` | CPK archive extractor (handles both big-endian and little-endian variants, zlib + stream cipher) | `*.cpk` | All observed archives extracted |
| `coretoc_parser.py` | Global asset index parser | `CoreTOC.dat` (big-endian variant) | All observed groups identified |
| `prefetch_parser.py` | Preload dependency graph parser | `Prefetch.dat` + `CoreTOC.dat` | Format invariants validated |

### 19.2 Reference open-source projects (cross-reference)

Publicly available projects covering related Blizzard formats, used to
cross-check independent observations:

| Project | Author | Relevance |
|---|---|---|
| `diablo3_xbox.bms` | Luigi Auriemma | QuickBMS script demonstrating CPK extraction with `zlib_noerror` on magic `\xA1\xB2\xC3\xD4` |
| CPKReaderWV | RedMadKnight | CPK header parser (header-info only) |
| jybp/casc | jybp | Open-source `SnoExtensions` table for the PC CASC build — shares its core with the console mapping (see §4.4) |
| StormLib | Ladislav Zezula | Reference MPQ implementation — not directly applicable, but useful for cross-checking hypotheses about compression algorithms (its `CascDecompress` is zlib-only) |
| CascLib | Ladislav Zezula | Reference CASC implementation |

---

## 20. Structural reference — consolidated

All format-level sizes, constants, and algorithmic parameters in one place:

| Item | Value | Source |
|---|---|---|
| SNO key (expanded encoding) | 8 bytes | Format layout |
| SNO key (packed encoding) | 4 bytes | Format layout |
| SNO file header | 16 bytes | On-disk prefix |
| `.pow` fixed section | 784 bytes | §12 |
| `.app` sub-object record | 240 bytes | §13.2 |
| `.app` skinning record | 104 bytes | §13.3 |
| `.app` vertex stride | 32 bytes | §13.4 |
| `.app` index width | 16 bits | §13.4 |
| CPK magic | `0xA1B2C3D4` | Empirical |
| CPK format version | 6 | Empirical |
| CPK cipher initial state | `0x872DCDA7A97A7EE1` | Recovered by known-plaintext attack against zlib prefixes (see §3.4) |
| CPK encryption flag | bit `0x8000` in `zsize_raw` | Empirical + cipher verification |
| CPK `read_sector` (read unit size) | `0x10000` | Header (implicit) |
| CPK `comp_sector` (compressed sector size) | `0x4000` | Header (implicit) |
| CPK header size | 64 bytes | Header |
| FNV-1a offset basis | `0x811C9DC5` | Cross-validated on CPK `FileName` entries |
| FNV-1a prime | `0x01000193` (16,777,619) | Cross-validated on CPK `FileName` entries |
| Lua bytecode signature | `1B 4C 75 61 51` | On-disk prefix of script blobs |
| Lua opcode set | Lua 5.1 (38 opcodes, standard order) | Decoded script bytecode |
| Audio locales | 16 | Packaging directory names |
| `CoreTOC.dat` record size | 136 bytes | Empirical |
| `Prefetch.dat` parent record | 16 bytes | Empirical |
| `Prefetch.dat` dependency record | 8 bytes | Empirical |

---

## 21. Coverage and open issues

### 21.1 Solved

- **CPK archive format** — compression (zlib) and encryption (stream
  cipher with fixed initial state `0x872DCDA7A97A7EE1`) both recovered
  and round-trip-verified against shipped archives.
- **`CoreTOC.dat`** — fully parsed.
- **`Prefetch.dat`** — fully parsed, format invariants validated
  against `CoreTOC.dat`.
- **`snoGroup → (name, extension)` mapping** — every populated group
  observed in the shipped data is named (via the PC-shared core for most
  groups, plus console-only extensions identified empirically from the
  sample names in `CoreTOC.dat`).
- **SNO key representation** (expanded / packed) and its range limits.
- **Universal SNO file header** (16 bytes, §7.1).
- **Shared deserialization framework as an observation** — verified by
  identical header layout across groups and consistent schema-hash
  validation.
- **Embedded scripting identified as Lua 5.1** — signature and opcode
  count both match the 5.1 baseline exactly, making the shipped
  bytecode decompilable with off-the-shelf tools.
- **Hashing algorithm** (FNV-1a with the standard 32-bit constants)
  verified against the `FileName` section of CPK archives.
- **Bit-packer invariant** — MSB-first within each byte for the
  bit-stream covering `SortedFileInfo` / `Locations` /
  `CompressedSectorChunk`.

### 21.2 Open

- **Semantics of several populated group IDs** whose names are not yet
  confirmed (working names only in §4.4). IDs are certain; names and
  precise content types require cross-validation against a larger body
  of sample assets.
- **Internal layout of the `.pow` body** (the `PowerBody` placeholder
  in §12) — requires further sample-driven per-field analysis.
- **Per-field breakdown of the 32-byte vertex in `.app` files** —
  positions and indices are recoverable, but the exact layout of
  normals, UVs, tangent frames, and skinning weights within the 32-byte
  stride is not determinable from `.app` data alone (it would require
  shader analysis).

### 21.3 Highest-leverage next steps

1. **Lua bytecode recovery.** Extract the compiled-script blob from each
   `.pow` file and run it through `unluac`. Result: readable logic for
   all shipped skills.
2. **Extend the corpus** across all shipped CPK archives (per-act
   archives, language packs) — broader sampling is the cheapest way to
   close most open items.
3. **Cross-platform corpus diff** — compare PC CASC-resident assets
   (via the public `SnoExtensions` table) against the console mapping
   in §4.4 to confirm the working names for the remaining console-only
   groups.

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
        · validate magic/version and schema fingerprint (§7, §11)
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
| Power body | placeholder name for the `.pow` body's core gameplay-parameter cluster |
| Formula table | 16-byte formula-array descriptor referenced from `.pow` |
| Appearance file layout | `.app` file top-level structure (§13) |
| Sub-object record | 240-byte mesh sub-object (§13.2) |
| Skinning record | 104-byte skinning sub-descriptor (§13.3) |
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
- **SortedFileInfo**, **Locations**, **CompressedSectorChunk** — table
  names used by the open-source CPK-reading community (Auriemma's
  QuickBMS script and CPKReaderWV) and retained here for consistency.
