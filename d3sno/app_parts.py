"""Record rules for adding and removing mesh parts (``37_mesh_parts.md`` W17-1 to W17-9).

This module holds the pieces of stage 17 that do not need the mesh encoders of ``app_mesh``:

- W17-1: the content bits of the sub-object flags word (``sub_object_flags``);
- W17-2: the sub-object (and look / material) name hash (``name_hash``) and free names (W17-12);
- W17-3, W17-4: sub-object records built from zero, with the copied fields taken from a template part
  (``new_sub_record``, ``rebuilt_sub_raw``);
- W17-5a: the ps3 draw size budget (``draw_budget``, ``BUDGET``);
- W17-6: ps3 descriptor and vertex-format blocks shared by equal bytes (``share_ps3_blocks``);
- W17-7: skinning records from zero (``skin_record_words``);
- W17-8: the material table (``material_entries``, ``new_material_hash``, ``copied_material_entry``,
  ``set_material_entries``);
- the report of added and removed parts (``PartsReport``).

The writer itself (W17-10), the move of a sub-object between files (W17-11) and the rebuild of every
record from zero are in ``app_mesh`` (``write_app_mesh``, ``transplant``, ``rebuild_from_zero``).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

from . import sno_layout as L
from .appearance import FORMAT_RIGID, FORMAT_SKINNED, PS3_DESCRIPTOR_FIXED, PS3_FORMAT_BLOCKS, AppearanceFile
from .binary import FormatError, build_order

# --- W17-2: names ----------------------------------------------------------------------------------

PS3_NAME_SIZE = 128


def name_hash(name) -> int:
    """W17-2: h = 0; for each byte c, ``A``-``Z`` lowered to ``a``-``z``, h = (h x 33 + c) mod 2^32.
    ``name`` is a str (ASCII) or bytes."""
    if isinstance(name, str):
        try:
            data = name.encode("ascii")
        except UnicodeEncodeError:
            raise FormatError(f"name {name!r} is not ASCII (W17-2)") from None
    else:
        data = bytes(name)
    h = 0
    for c in data:
        if 0x41 <= c <= 0x5A:
            c += 0x20
        h = (h * 33 + c) & 0xFFFFFFFF
    return h


def ps3_name_field(name: str) -> bytes:
    """W17-2: the 128-byte ps3 name field: the ASCII name, NUL-terminated, the rest zero (at most
    127 bytes, W17-12)."""
    if not isinstance(name, str) or not name:
        raise FormatError(f"ps3 sub-object name {name!r}: a non-empty name is needed (W17-2)")
    try:
        data = name.encode("ascii")
    except UnicodeEncodeError:
        raise FormatError(f"ps3 sub-object name {name!r} is not ASCII (W17-12)") from None
    if len(data) > PS3_NAME_SIZE - 1 or b"\0" in data:
        raise FormatError(f"ps3 sub-object name {name!r} is longer than 127 bytes (W17-12)")
    return data + bytes(PS3_NAME_SIZE - len(data))


def free_name(base: str, taken_hashes) -> str:
    """W17-12: ``base`` when its hash is not taken, else ``base_2``, ``base_3``, ...: the smallest
    free number."""
    taken = set(taken_hashes)
    if name_hash(base) not in taken:
        return base
    k = 2
    while name_hash(f"{base}_{k}") in taken:
        k += 1
    return f"{base}_{k}"


# --- W17-1: flag bits ------------------------------------------------------------------------------

BIT_ALWAYS = 0
BIT_FAT = 8  # ps3: owns fat vertices; switch / x360: cloth record or fat vertices (or kept)
BIT_DRAWS = 13  # ps3
BIT_SKINNED_DRAW = 14  # ps3
BIT_PACKED = 15  # switch / x360: packed rigid list or skinning record
BIT_SKIN = 16  # switch / x360: skinning record
CONTENT_MASK = (1 << BIT_ALWAYS) | (1 << BIT_FAT) | (1 << BIT_DRAWS) | (1 << BIT_SKINNED_DRAW) \
    | (1 << BIT_PACKED) | (1 << BIT_SKIN)
LOW_BITS = (1 << 17) - 1  # bits 17..31 are clear in every record


def sub_object_flags(build: str, stored: int, *, fat: bool = False, cloth: bool = False, draws: int = 0,
                     skinned_draws: int = 0, rigid: bool = False, skin_records: int = 0,
                     keep_bit8: bool = False) -> int:
    """W17-1: the flags word of a sub-object. The content bits come from what it holds; every other
    bit of ``stored`` (bits 1..16 outside the content bits) is copied; bits 17..31 are clear.
    ``keep_bit8``: switch / x360 only, keep a stored bit 8 on an existing rigid sub-object."""
    flags = (stored & LOW_BITS & ~CONTENT_MASK) | (1 << BIT_ALWAYS)
    if build == "ps3":
        flags |= (fat << BIT_FAT) | (bool(draws) << BIT_DRAWS) | (bool(skinned_draws) << BIT_SKINNED_DRAW)
    else:
        flags |= (bool(cloth or fat or keep_bit8) << BIT_FAT) | (bool(rigid or skin_records) << BIT_PACKED) \
            | (bool(skin_records) << BIT_SKIN)
    return flags


# --- W17-3, W17-4: sub-object records --------------------------------------------------------------

SWITCH_FIELDS = {"flags": 0x00, "fat": 0x04, "index": 0x28, "cloth": 0x40, "physics": 0x50, "fade_group": 0x54,
                 "fade_alpha": 0x58, "name_info": 0x5C, "material": 0x64, "hash": 0x68, "box": 0x6C,
                 "shape_count": 0x84, "shapes": 0x88, "packed_count": 0xA0, "packed": 0xB0, "skins": 0xB8,
                 "compact": 0xC8, "bias": 0xD8, "scale": 0xE4}
PS3_FIELDS = {"flags": 0x00, "fat": 0x04, "index": 0x20, "cloth": 0x30, "physics": 0x3C, "fade_group": 0x40,
              "fade_alpha": 0x44, "name": 0x48, "material": 0xC8, "hash": 0xCC, "box": 0xD0, "shape_count": 0xE8,
              "shapes": 0xEC, "draw_count": 0xFC, "draws": 0x100, "compact": 0x10C, "bias": 0x118,
              "scale": 0x124}
# W17-3 "copied" words (4 bytes each) besides the copied flag bits.
COPIED = {"switch": ("physics", "fade_group", "fade_alpha", "name_info"),
          "ps3": ("physics", "fade_group", "fade_alpha")}
SHAPE_RECORD = {"switch": 80, "x360": 80, "ps3": 68}
SUB_RECORD_NAME = {"switch": "SubObject", "x360": "SubObject", "ps3": "SubObjectP3"}
SKIN_RECORD_NAME = "SkinningInfo"
DRAW_RECORD_NAME = "DrawP3"
MATERIAL_RECORD_NAME = "AppearanceMaterial"
# Structure geo sets (body offsets): count word, sub-object pair.
GEO_COUNTS = (0x60, 0x78)
GEO_PAIRS = (0x64, 0x7C)


def sub_fields(build: str) -> dict[str, int]:
    return PS3_FIELDS if build == "ps3" else SWITCH_FIELDS


def copied_names(build: str) -> tuple[str, ...]:
    return COPIED["ps3" if build == "ps3" else "switch"]


def new_sub_record(build: str, template_raw: bytes, material_hash: int, sub_hash: int,
                   name: str | None = None) -> L.Record:
    """W17-3: a sub-object record from zero. Copied from ``template_raw`` (the template part's
    record, W17-10): the flag bits other than the content bits (W17-1; bit 8 is a content bit, so
    a new part has it only by content), the physics surface handle, fade group and alpha and (switch,
    x360) the name info word (W17-4). Set: the material hash, the sub-object hash and the ps3 name.
    Every other word is 0 and every pair empty; the writer fills the content."""
    order = build_order(build)
    record = L.new_record(build, "app", SUB_RECORD_NAME[build])
    fields = sub_fields(build)
    raw = record.raw
    stored_flags = struct.unpack_from(order + "I", template_raw, fields["flags"])[0]
    struct.pack_into(order + "I", raw, fields["flags"], sub_object_flags(build, stored_flags))
    for name_ in copied_names(build):
        at = fields[name_]
        raw[at:at + 4] = template_raw[at:at + 4]
    struct.pack_into(order + "II", raw, fields["material"], material_hash & 0xFFFFFFFF, sub_hash & 0xFFFFFFFF)
    if build == "ps3":
        at = fields["name"]
        raw[at:at + PS3_NAME_SIZE] = ps3_name_field(name)
    return record


def pair_size(record: L.Record, offset: int) -> int:
    value = record.pairs.get(offset)
    if value is None or value.block is None:
        return 0
    return value.pair_size()


def shape_count(build: str, record: L.Record) -> int:
    """W17-3: the number of collision shapes in the shape pair."""
    return pair_size(record, sub_fields(build)["shapes"]) // SHAPE_RECORD[build]


# --- W17-5a: ps3 draw budget -----------------------------------------------------------------------

BUDGET = {FORMAT_RIGID: (21, 55998), FORMAT_SKINNED: (46, 58168)}  # per-vertex weight, limit


def draw_budget(fmt: int, vertex_count: int, index_count: int) -> int:
    """W17-5a: the budget value of a ps3 draw (21 nv + ni for 0x81, 46 nv + ni for 0x83)."""
    return BUDGET[fmt][0] * vertex_count + index_count


def within_budget(fmt: int, vertex_count: int, index_count: int) -> bool:
    return draw_budget(fmt, vertex_count, index_count) <= BUDGET[fmt][1]


def new_draw_record(fmt: int) -> L.Record:
    """W17-5: a ps3 draw record from zero with the vertex-format blocks of its format (G1-6). The
    writer sets the descriptor, index list, streams and every word (W2-9, W2-10, W13-1, W13-4)."""
    record = L.new_record("ps3", "app", DRAW_RECORD_NAME)
    record.pairs[0x80] = L.PairValue(L.Block(PS3_FORMAT_BLOCKS[fmt][0]))
    record.pairs[0x90] = L.PairValue(L.Block(PS3_FORMAT_BLOCKS[fmt][1]))
    return record


def ps3_descriptor(fmt: int, command_space: int, vertex_count: int, index_count: int) -> bytes:
    """W2-9 / G1-5: format, command space, the fixed bytes, the two counts (u16) and four 0xFF."""
    return bytes([fmt, command_space]) + PS3_DESCRIPTOR_FIXED[fmt] \
        + struct.pack(">HH", vertex_count, index_count) + b"\xFF" * 4


# --- W17-6: ps3 shared blocks ----------------------------------------------------------------------

PS3_SHARED_PAIRS = (0x08, 0x80, 0x90)  # descriptor, vertex format A, vertex format B


def _block_bytes(value) -> bytes:
    if value is None or value.block is None:
        return b""
    block = value.block
    if block.elements is None:
        return block.data or b""
    return b"".join(bytes(e.raw) for e in block.elements) + block.extra


def ps3_draw_records(tree: L.SnoTree) -> list[L.Record]:
    """The ps3 draw records in the layout walk order: geo set 0, then geo set 1, sub-objects and
    draws in order (W17-6)."""
    out = []
    for at in GEO_PAIRS:
        value = tree.top.pairs.get(at)
        for sub in value.block.elements if value is not None and value.block else []:
            draws = sub.pairs.get(PS3_FIELDS["draws"])
            out.extend(draws.block.elements if draws is not None and draws.block else [])
    return out


def share_ps3_blocks(tree: L.SnoTree) -> int:
    """W17-6: a draw's descriptor block is shared with an earlier draw of the file exactly when
    their bytes are equal; the same for the vertex-format blocks. Returns the number of shared
    pairs."""
    seen: dict[tuple[int, bytes], L.Block] = {}
    shared = 0
    for record in ps3_draw_records(tree):
        for offset in PS3_SHARED_PAIRS:
            value = record.pairs.get(offset)
            data = _block_bytes(value)
            if not data:
                continue
            key = (0 if offset == 0x08 else 1, data)  # descriptors among themselves, format blocks
            if key in seen:
                record.pairs[offset] = L.PairValue(seen[key], None, shared=True)
                shared += 1
            else:
                block = value.block if value.block.elements is None else L.Block(data)
                if value.shared or block is not value.block:
                    record.pairs[offset] = L.PairValue(block)
                seen[key] = block
    return shared


# --- W17-7: skinning records -----------------------------------------------------------------------

SKIN_WORDS = {"bones": 0x00, "palette": 0x04, "vertex_count": 0x18, "vertices": 0x28, "index_count": 0x30,
              "indices": 0x40}
PACKED_SKINNED = 36


def skin_record_words(record: L.Record, order: str) -> bytearray:
    """W17-7: the skinning record from zero: palette count, vertex count and index count from its
    blocks, every other word 0."""
    raw = bytearray(len(record.raw))
    struct.pack_into(order + "i", raw, SKIN_WORDS["bones"], pair_size(record, SKIN_WORDS["palette"]) // 4)
    struct.pack_into(order + "i", raw, SKIN_WORDS["vertex_count"],
                     pair_size(record, SKIN_WORDS["vertices"]) // PACKED_SKINNED)
    struct.pack_into(order + "i", raw, SKIN_WORDS["index_count"], pair_size(record, SKIN_WORDS["indices"]) // 2)
    return raw


# --- W17-8: material table -------------------------------------------------------------------------

# Body offsets: look count, material count, materials pair, looks pair.
MATERIAL_FIELDS = {"switch": (0x168, 0x16C, 0x170, 0x180), "x360": (0x168, 0x16C, 0x170, 0x180),
                   "ps3": (0x178, 0x17C, 0x180, 0x190)}
PER_LOOK_PAIR = 0x08


@dataclass(frozen=True)
class MaterialSource:
    """Where a new material entry copies its per-look records from (W17-8): the entry ``hash`` of
    ``app`` (a donor file of the same build), or of the file being written when ``app`` is None."""

    hash: int
    app: AppearanceFile | None = None


def material_source(value) -> MaterialSource | None:
    """A ``SubObject.material_source`` value as a ``MaterialSource``: None, an int (an entry of the
    same file), a ``MaterialSource`` or a pair (donor file, hash)."""
    if value is None or isinstance(value, MaterialSource):
        return value
    if isinstance(value, bool):
        raise FormatError(f"material source {value!r}")
    if isinstance(value, int):
        return MaterialSource(value)
    if isinstance(value, tuple) and len(value) == 2:
        return MaterialSource(value[1], value[0])
    raise FormatError(f"material source {value!r}: an int hash, a MaterialSource or (file, hash)")


def material_entries(tree: L.SnoTree) -> list[L.Record]:
    value = tree.top.pairs.get(MATERIAL_FIELDS[tree.build][2])
    return list(value.block.elements) if value is not None and value.block and value.block.elements else []


def entry_hash(record: L.Record, order: str) -> int:
    return struct.unpack_from(order + "I", record.raw, 0)[0]


def look_count(tree: L.SnoTree) -> int:
    return struct.unpack_from(build_order(tree.build) + "i", tree.top.raw, MATERIAL_FIELDS[tree.build][0])[0]


def set_material_entries(tree: L.SnoTree, entries: list[L.Record]) -> None:
    """The material table pair and its count word (W17-8)."""
    _, count_at, pair_at, _ = MATERIAL_FIELDS[tree.build]
    tree.top.pairs[pair_at] = L.PairValue(L.Block(elements=list(entries)) if entries else None)
    struct.pack_into(build_order(tree.build) + "i", tree.top.raw, count_at, len(entries))


def new_material_hash(app: AppearanceFile, name: str) -> int:
    """W17-8: the hash of a new material: W17-2 of ``name``; refused when it is 0 or already in
    the file's material table."""
    value = name_hash(name)
    if value == 0:
        raise FormatError(f"material name {name!r} gives hash 0 (W17-8)")
    if app.material_data.find(value) is not None:
        raise FormatError(f"material hash 0x{value:08X} of {name!r} is already in the table (W17-8)")
    return value


def copy_record(record: L.Record, memo: dict | None = None) -> L.Record:
    """A deep copy of a record and every block below it. Blocks shared inside the copy stay shared
    (the first pair in walk order owns the copy); nothing is shared with the original."""
    memo = {} if memo is None else memo
    pairs = {}
    for offset, value in record.pairs.items():
        if value is None or value.block is None:
            pairs[offset] = L.PairValue(None, value.size if value is not None else None)
            continue
        block = value.block
        if id(block) in memo:
            pairs[offset] = L.PairValue(memo[id(block)], value.size, shared=True)
            continue
        if block.elements is None:
            copy = L.Block(block.data, extra=block.extra)
        else:
            copy = L.Block(elements=[], extra=block.extra)
        memo[id(block)] = copy
        if block.elements is not None:
            copy.elements = [copy_record(e, memo) for e in block.elements]
        pairs[offset] = L.PairValue(copy, value.size)
    return L.Record(record.name, bytearray(record.raw), pairs)


def copied_material_entry(source: L.Record, new_hash: int, order: str, looks: int) -> L.Record:
    """W17-8: a new entry with hash ``new_hash`` copying the per-look records of ``source`` (an
    entry of the same file or of a donor file of the same build). When the source has another
    number of per-look records than the file has looks, its first per-look record is repeated for
    every look (HYPOTHESIS for the game)."""
    entry = copy_record(source)
    struct.pack_into(order + "I", entry.raw, 0, new_hash & 0xFFFFFFFF)
    value = source.pairs.get(PER_LOOK_PAIR)
    per_look = list(value.block.elements) if value is not None and value.block and value.block.elements else []
    if len(per_look) != looks:
        if not per_look and looks:
            raise FormatError(f"material 0x{entry_hash(source, order):08X} has no per-look record to "
                              f"repeat for {looks} looks (W17-8)")
        repeated = [copy_record(per_look[0]) for _ in range(looks)]
        entry.pairs[PER_LOOK_PAIR] = L.PairValue(L.Block(elements=repeated) if repeated else None)
    return entry


# --- Report ----------------------------------------------------------------------------------------

@dataclass(frozen=True)
class PartChange:
    """One added, removed or skipped part. ``kind``: "sub-object", "draw" (ps3), "skinning record",
    "rigid list" (switch, x360) or "material". ``label`` names the part: a removed part by its place
    in the template, an added part by its place in the output."""

    action: str  # "added", "removed" or "skipped" (an empty new part, not written)
    kind: str
    label: str


@dataclass
class PartsReport:
    changes: list[PartChange] = field(default_factory=list)

    def add(self, action: str, kind: str, label: str) -> None:
        self.changes.append(PartChange(action, kind, label))

    @property
    def added(self) -> list[str]:
        return [c.label for c in self.changes if c.action == "added"]

    @property
    def removed(self) -> list[str]:
        return [c.label for c in self.changes if c.action == "removed"]

    @property
    def skipped(self) -> list[str]:
        return [c.label for c in self.changes if c.action == "skipped"]

    def __bool__(self) -> bool:
        return bool(self.changes)


__all__ = [
    "BUDGET", "CONTENT_MASK", "GEO_COUNTS", "GEO_PAIRS", "MATERIAL_FIELDS", "MaterialSource", "PartChange",
    "PartsReport", "PS3_FIELDS", "SWITCH_FIELDS", "copied_material_entry", "copy_record", "draw_budget",
    "free_name", "material_entries", "material_source", "name_hash", "new_draw_record", "new_material_hash",
    "new_sub_record", "ps3_descriptor", "ps3_draw_records", "ps3_name_field", "set_material_entries",
    "shape_count", "share_ps3_blocks", "skin_record_words", "sub_object_flags", "within_budget",
]
