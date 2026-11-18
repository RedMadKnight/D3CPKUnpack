"""The generic SNO writer of ``22_corrections11.md`` (W11-1 to W11-8), driven by ``s11_layouts.tsv``.

A file is parsed into a tree (``parse_tree``): the 16-byte header, every record's bytes (pair
fields included; the writer overwrites the pairs), every non-empty block and, for every pair,
whether it points at the same block as an earlier pair (W11-6: such pairs hold the same
``Block`` object). ``write_tree`` lays the tree out again:

- W11-2: a cursor starts at the end of the top record (file 0x10 + its size). The top record's
  fields are walked in table order; inline records in place, a pair's block is placed and, when it
  is an array of records, each element is walked before the parent's next field.
- W11-3: fields with a deferred group are placed after the walk, group 0, 1, 2, each in walk order.
- W11-4: before a block the cursor is padded with zero bytes to the field's alignment, counted
  from file 0x10.
- W11-5: an empty pair is {0, 0}; formula code (kind ``code``) aligns to 4 and stores
  {cursor - 0x10, 0}; ``empty_form`` cursor stores {cursor - 0x10, 0} without aligning.
- W11-6: a pair sharing an earlier block stores that block's offset and its own size; nothing is
  written and the block's elements are not walked again.
- W11-7: the file ends at the cursor.
- W11-8: ps3 ``.glo`` was not described as rebuilt; ``25`` W3-6 to W3-8 give its rows, merged
  into the bundled table, so ``RESIDUE`` is now empty.

Records are looked up by (build, extension, record) and else by (build, record): the table lists
a record once per build under the first type that uses it (ps3: once per type).
"""

from __future__ import annotations

import csv
import struct
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Iterator

from .binary import FormatError, build_order
from .datafiles import DATA
from .sno import parse_header, resolve_build

TABLE = "s11_layouts.tsv"
BODY = 0x10
HEADER_SIZE = 0x10
PAIR_SIZE = 8
VALUE_ELEMENTS = {"BYTE": 1, "WORD": 2, "INT": 4, "SNO": 4, "VECTOR3D": 12}
PAIR_KINDS = ("array", "text", "tagmap", "code", "blob")
CODE_ALIGN = 4
# W11-4 does not give an alignment for fields that are empty in every file ("none (always
# empty)"); a non-empty block there is placed with this one and noted (Q347).
ALWAYS_EMPTY_ALIGN = {"switch": 8, "x360": 8, "ps3": 4}
RESIDUE: frozenset = frozenset()  # W11-8; ps3 .glo is described by W3-6 to W3-8 (stage 12)


@dataclass(frozen=True)
class LayoutField:
    order: int
    name: str
    offset: int  # in the record
    kind: str  # inline, array, text, tagmap, code, blob
    element: str  # record name, BYTE / WORD / INT / SNO / VECTOR3D, or ""
    count: int | None  # inline: element count; array: element size
    alignment: int | None  # None: "none (always empty)"
    empty_form: str  # zero or cursor
    deferred_group: int | None

    @property
    def is_pair(self) -> bool:
        return self.kind != "inline"

    @property
    def is_record_array(self) -> bool:
        return self.kind == "array" and self.element not in VALUE_ELEMENTS


@dataclass(frozen=True)
class RecordLayout:
    build: str
    ext: str
    name: str
    size: int
    fields: tuple[LayoutField, ...]


def _int(text: str) -> int | None:
    text = (text or "").strip()
    return None if not text else int(text, 0)


@lru_cache(maxsize=None)
def _table() -> tuple[dict, dict, dict]:
    records: dict[tuple[str, str, str], RecordLayout] = {}
    by_name: dict[tuple[str, str], tuple[str, str, str]] = {}
    tops: dict[tuple[str, str], str] = {}
    pending: dict[tuple[str, str, str], list[LayoutField]] = {}
    sizes: dict[tuple[str, str, str], int] = {}
    with open(DATA / TABLE, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            key = (row["build"], row["ext"], row["record"])
            if not row["field"]:
                tops.setdefault((row["build"], row["ext"]), row["record"])
                sizes[key] = int(row["record_size"])
                pending.setdefault(key, [])
                by_name.setdefault((row["build"], row["record"]), key)
                continue
            align = row["alignment"].strip()
            pending.setdefault(key, []).append(LayoutField(
                int(row["order"]), row["field"], int(row["pair_offset"], 16), row["kind"],
                row["element"], _int(row["element_size_or_count"]),
                None if align.startswith("none") or not align else int(align),
                row["empty_form"] or "zero", _int(row["deferred_group"]),
            ))
    for key, fields in pending.items():
        records[key] = RecordLayout(*key, sizes[key], tuple(sorted(fields, key=lambda f: f.order)))
    return records, by_name, tops


def covered_types(build: str) -> tuple[str, ...]:
    """The extensions with a layout for ``build`` (section 1)."""
    return tuple(sorted(ext for b, ext in _table()[2] if b == build))


def top_record(build: str, ext: str) -> str:
    try:
        return _table()[2][(build, ext)]
    except KeyError:
        raise FormatError(f"no s11 layout for {build} .{ext}") from None


def record_layout(build: str, ext: str, name: str) -> RecordLayout:
    records, by_name, _ = _table()
    key = (build, ext, name)
    if key not in records:
        key = by_name.get((build, name))
        if key is None:
            raise FormatError(f"no s11 layout for record {name} ({build})")
    return records[key]


@dataclass(frozen=True)
class PairSlot:
    """A pair field of a record, with inline records expanded (W11-2: in place)."""

    offset: int  # in the outer record
    field: LayoutField
    path: str


@lru_cache(maxsize=None)
def pair_slots(build: str, ext: str, name: str) -> tuple[PairSlot, ...]:
    """The pair fields of a record in walk order. Inline members that share bytes (unions such
    as ``DspEffect``) give each pair offset once, at its first place."""
    found: list[PairSlot] = []
    seen: set[int] = set()

    def expand(layout: RecordLayout, base: int, prefix: str) -> None:
        for f in layout.fields:
            if f.kind == "inline":
                inner = record_layout(build, ext, f.element)
                count = f.count or 1
                for i in range(count):
                    label = f"{prefix}{f.name}[{i}]." if count > 1 else f"{prefix}{f.name}."
                    expand(inner, base + f.offset + i * inner.size, label)
            elif base + f.offset not in seen:
                seen.add(base + f.offset)
                found.append(PairSlot(base + f.offset, f, prefix + f.name))

    expand(record_layout(build, ext, name), 0, "")
    return tuple(found)


# --- Tree ----------------------------------------------------------------------------------------

@dataclass(eq=False)
class Block:
    """One block. ``elements`` is set for an array of records, ``data`` otherwise. ``extra`` holds
    bytes after the last whole element."""

    data: bytes | None = None
    elements: list["Record"] | None = None
    extra: bytes = b""
    source: int | None = None  # body offset in the parsed file

    def size(self) -> int:
        if self.elements is None:
            return len(self.data or b"")
        return sum(len(e.raw) for e in self.elements) + len(self.extra)

    def is_empty(self) -> bool:
        return self.size() == 0


@dataclass(eq=False)
class PairValue:
    block: Block | None  # None: an empty pair
    size: int | None = None  # the pair's own size when it differs from the block (W11-6)
    stored: tuple[int, int] = (0, 0)  # as parsed
    shared: bool = False  # points at the block of an earlier pair (W11-6)

    def pair_size(self) -> int:
        return self.block.size() if self.size is None else self.size


@dataclass(eq=False)
class Record:
    name: str
    raw: bytearray  # the whole record; pair fields are overwritten when written (W11-1)
    pairs: dict[int, PairValue] = field(default_factory=dict)  # by slot offset
    # pair sizes when parsed, by slot offset (29_corrections17.md V16-2): a list whose element
    # count differs from this has changed, however it was replaced; a new record has none (0)
    parsed: dict[int, int] = field(default_factory=dict)


@dataclass(eq=False)
class SnoTree:
    build: str
    ext: str
    header: bytes
    top: Record
    trailing: bytes = b""  # bytes after the last block (W11-7: not written)
    notes: list[str] = field(default_factory=list)


class _Walk:
    """The W11-2/W11-3 traversal shared by the reader and the writer."""

    def __init__(self, build: str, ext: str):
        self.build, self.ext = build, ext
        self.groups: dict[int, list] = {}

    def run(self, top: Record, top_pos: int) -> None:
        self.record(top, top_pos, "")
        done = -1
        while True:
            waiting = sorted(g for g, items in self.groups.items() if items)
            if not waiting:
                return
            group = waiting[0]
            if group < done:
                raise FormatError(f"deferred group {group} found after group {done} was placed")
            done = group
            items = self.groups[group]
            while items:
                self.slot(*items.pop(0))

    def record(self, record: Record, pos: int, path: str) -> None:
        for slot in pair_slots(self.build, self.ext, record.name):
            if slot.field.deferred_group is not None:
                self.groups.setdefault(slot.field.deferred_group, []).append((record, pos, slot, path))
            else:
                self.slot(record, pos, slot, path)

    def slot(self, record: Record, pos: int, slot: PairSlot, path: str) -> None:
        raise NotImplementedError


class _Reader(_Walk):
    def __init__(self, data: bytes, build: str, ext: str):
        super().__init__(build, ext)
        self.data, self.order = data, build_order(build)
        self.seen: dict[int, Block] = {}
        self.ends: list[int] = []
        self.notes: list[str] = []

    def slot(self, record: Record, pos: int, slot: PairSlot, path: str) -> None:
        f = slot.field
        at = pos + slot.offset
        offset, size = struct.unpack_from(self.order + "II", self.data, at)
        where = f"{path}{slot.path}"
        record.parsed[slot.offset] = size
        if size == 0:
            record.pairs[slot.offset] = PairValue(None, 0, (offset, 0))
            want_cursor = f.kind == "code" or f.empty_form == "cursor"
            if not want_cursor and offset != 0:
                self.notes.append(f"{where}: empty pair with offset 0x{offset:X} (W11-5 writes 0)")
            return
        if offset in self.seen:
            block = self.seen[offset]
            own = None if size == block.size() else size
            record.pairs[slot.offset] = PairValue(block, own, (offset, size), shared=True)
            return
        start = BODY + offset
        if start + size > len(self.data):
            raise FormatError(f"{where}: block at 0x{start:X} size 0x{size:X} past the end", at)
        self.ends.append(start + size)
        if f.is_record_array:
            inner = record_layout(self.build, self.ext, f.element)
            count, rest = divmod(size, inner.size)
            elements = [Record(inner.name, bytearray(self.data[start + i * inner.size:
                                                              start + (i + 1) * inner.size]))
                        for i in range(count)]
            block = Block(elements=elements, extra=bytes(self.data[start + count * inner.size:start + size]),
                          source=offset)
            if rest:
                self.notes.append(f"{where}: {rest} bytes after the last whole {inner.name}")
        else:
            block = Block(bytes(self.data[start:start + size]), source=offset)
        self.seen[offset] = block
        record.pairs[slot.offset] = PairValue(block, None, (offset, size))
        if block.elements is not None:
            inner_size = record_layout(self.build, self.ext, f.element).size
            for i, element in enumerate(block.elements):
                self.record(element, start + i * inner_size, f"{where}[{i}].")


class _Writer(_Walk):
    def __init__(self, tree: SnoTree):
        super().__init__(tree.build, tree.ext)
        self.order = build_order(tree.build)
        self.out = bytearray(tree.header) + tree.top.raw
        self.placed: dict[int, int] = {}  # id(block) -> body offset

    def _align(self, alignment: int) -> None:
        out = self.out
        out += bytes(-(len(out) - BODY) % alignment)

    def _pair(self, at: int, offset: int, size: int) -> None:
        struct.pack_into(self.order + "II", self.out, at, offset, size)

    def slot(self, record: Record, pos: int, slot: PairSlot, path: str) -> None:
        f = slot.field
        at = pos + slot.offset
        value = record.pairs.get(slot.offset)
        block = value.block if value is not None else None
        if block is None or block.is_empty():
            if f.kind == "code":
                self._align(CODE_ALIGN)
                self._pair(at, len(self.out) - BODY, 0)
            elif f.empty_form == "cursor":
                self._pair(at, len(self.out) - BODY, 0)
            else:
                self._pair(at, 0, 0)
            return
        if id(block) in self.placed:
            self._pair(at, self.placed[id(block)], value.pair_size())
            return
        alignment = f.alignment if f.alignment is not None else ALWAYS_EMPTY_ALIGN[self.build]
        self._align(alignment)
        start = len(self.out)
        self.placed[id(block)] = start - BODY
        self._pair(at, start - BODY, value.pair_size())
        if block.elements is None:
            self.out += block.data
            return
        for element in block.elements:
            self.out += element.raw
        self.out += block.extra
        position = start
        for i, element in enumerate(block.elements):
            self.record(element, position, f"{path}{slot.path}[{i}].")
            position += len(element.raw)


def parse_tree(data: bytes, ext: str, build: str | None = None) -> SnoTree:
    """W11-1: read a file of a covered type into a tree."""
    ext = ext.lower().lstrip(".")
    header = parse_header(data, build)
    build = resolve_build(ext, header, build)
    top = record_layout(build, ext, top_record(build, ext))
    end = BODY + top.size
    if len(data) < end:
        raise FormatError(f"{top.name} of {top.size} bytes runs past the end", BODY)
    tree = SnoTree(build, ext, bytes(data[:HEADER_SIZE]), Record(top.name, bytearray(data[BODY:end])))
    reader = _Reader(bytes(data), build, ext)
    reader.run(tree.top, BODY)
    last = max(reader.ends, default=end)
    last = max(last, end)
    tree.trailing = bytes(data[last:])
    tree.notes = reader.notes
    if tree.trailing:
        tree.notes.append(f"{len(tree.trailing)} bytes after the last block (W11-7: not written)")
    if (build, ext) in RESIDUE:
        tree.notes.append(f"{build} .{ext}: not described as rebuilt (W11-8)")
    return tree


def write_tree(tree: SnoTree, derive: bool = False) -> bytes:
    """W11-1 to W11-7: lay a tree out.

    With ``derive`` (W14-20), every PROVEN count word of ``s14_count_links.tsv`` whose list has a
    different element count than when the tree was parsed is set first (in the tree itself);
    unchanged lists keep their stored words, so an unchanged tree gives the same bytes."""
    record_layout(tree.build, tree.ext, tree.top.name)
    if derive:
        derive_counts(tree)
    writer = _Writer(tree)
    writer.run(tree.top, BODY)
    return bytes(writer.out)


def rebuild_sno(data: bytes, ext: str, build: str | None = None) -> bytes:
    return write_tree(parse_tree(data, ext, build))


def is_canonical(data: bytes, ext: str, build: str | None = None) -> bool:
    """True when the file is laid out by W11 (its rebuild is the same bytes)."""
    try:
        return rebuild_sno(data, ext, build) == bytes(data)
    except FormatError:
        return False


# --- Access ----------------------------------------------------------------------------------

def iter_pairs(tree: SnoTree) -> Iterator[tuple[str, Record, PairSlot, PairValue | None]]:
    """Every pair of the tree in walk order (deferred ones in place), with its path, e.g.
    ``tStructure.tGeoSets[0].ptSubObjects[1].serPackVertList``. Shared blocks are not entered."""
    visited: set[int] = set()

    def walk(record: Record, path: str):
        for slot in pair_slots(tree.build, tree.ext, record.name):
            value = record.pairs.get(slot.offset)
            yield f"{path}{slot.path}", record, slot, value
            if value is None or value.block is None or value.block.elements is None:
                continue
            if value.shared or id(value.block) in visited:
                continue
            visited.add(id(value.block))
            for i, element in enumerate(value.block.elements):
                yield from walk(element, f"{path}{slot.path}[{i}].")

    yield from walk(tree.top, "")


def shared_pairs(tree: SnoTree) -> list[tuple[str, int]]:
    """W11-6: the pairs that point at an earlier pair's block, with that block's source offset."""
    return [(path, value.block.source) for path, _, _, value in iter_pairs(tree)
            if value is not None and value.shared]


def find_pair(tree: SnoTree, path: str) -> tuple[Record, PairSlot, PairValue | None]:
    for found, record, slot, value in iter_pairs(tree):
        if found == path:
            return record, slot, value
    raise KeyError(path)


def set_block(tree: SnoTree, path: str, data: bytes, update_counts: bool = False) -> None:
    """Give one pair new bytes (a blob, text, tag map, code or value array). The pair gets a block
    of its own; pairs that shared the old block keep it. With ``update_counts`` (W14-20) the
    PROVEN count word linked to a value array is set to its new element count."""
    record, slot, _ = find_pair(tree, path)
    if slot.field.is_record_array:
        raise FormatError(f"{path} is an array of records: use set_elements")
    if slot.field.kind == "array":
        unit = VALUE_ELEMENTS[slot.field.element]
        if len(data) % unit:
            raise FormatError(f"{path}: {len(data)} bytes is not a whole number of {slot.field.element}")
    record.pairs[slot.offset] = PairValue(Block(bytes(data)) if data else None)
    if update_counts:
        update_count_words(tree, record, slot.offset)


def set_elements(tree: SnoTree, path: str, elements: list[Record], update_counts: bool = False) -> None:
    """Give an array of records new elements (records of the field's element type). With
    ``update_counts`` (W14-20) the PROVEN count word linked to this list is set."""
    record, slot, _ = find_pair(tree, path)
    if not slot.field.is_record_array:
        raise FormatError(f"{path} is not an array of records")
    size = record_layout(tree.build, tree.ext, slot.field.element).size
    for element in elements:
        if element.name != slot.field.element or len(element.raw) != size:
            raise FormatError(f"{path}: an element is not a {slot.field.element} of {size} bytes")
    record.pairs[slot.offset] = PairValue(Block(elements=list(elements)) if elements else None)
    if update_counts:
        update_count_words(tree, record, slot.offset)


# --- Count fields (28_derived_values.md W14-20) -------------------------------------------------

COUNT_TABLE = "s14_count_links.tsv"
PROVEN = "PROVEN"


@dataclass(frozen=True)
class CountLink:
    """A 32-bit word of a record that equals the element count of one of its lists (W14-20).
    Offsets are in the record, inline members expanded as in ``pair_slots``."""

    pair_offset: int
    count_offset: int
    minus_one: bool  # the word is the element count minus one
    element_size: int
    status: str

    @property
    def proven(self) -> bool:
        return self.status == PROVEN


@lru_cache(maxsize=None)
def _count_table() -> dict[tuple[str, str, str], tuple[CountLink, ...]]:
    """Rows keyed by the layout record they resolve to (as ``record_layout`` resolves names).
    Rows naming no array pair of the table are skipped."""
    records, by_name, _ = _table()
    found: dict[tuple[str, str, str], list[CountLink]] = {}
    with open(DATA / COUNT_TABLE, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            key = (row["build"], row["ext"], row["record"])
            if key not in records:
                key = by_name.get((row["build"], row["record"]))
                if key is None:
                    continue
            offset = int(row["pair_offset"], 16)
            slot = next((s for s in pair_slots(*key) if s.offset == offset), None)
            if slot is None or slot.field.kind != "array":
                continue
            size = slot.field.count or (VALUE_ELEMENTS.get(slot.field.element)
                                        or record_layout(key[0], key[1], slot.field.element).size)
            found.setdefault(key, []).append(CountLink(
                offset, int(row["count_offset"], 16), row["count_is"].strip() == "elements-1", size,
                row["status"].strip(),
            ))
    return {key: tuple(links) for key, links in found.items()}


def count_links(build: str, ext: str, name: str, proven_only: bool = True) -> tuple[CountLink, ...]:
    """The W14-20 count links of a record (by default the PROVEN ones only)."""
    layout = record_layout(build, ext, name)
    links = _count_table().get((layout.build, layout.ext, layout.name), ())
    return tuple(link for link in links if link.proven or not proven_only)


def _elements(value: PairValue | None, size: int) -> int:
    if value is None or value.block is None:
        return 0
    return value.pair_size() // size


def _parsed_elements(record: Record, pair_offset: int, size: int) -> int:
    """V16-2: the element count of a list when the file was parsed (0 for a new record), kept on
    the record so that a list emptied or replaced through any setter counts as changed."""
    return record.parsed.get(pair_offset, 0) // size


def _set_words(tree: SnoTree, record: Record, forced: int | None, strict: bool) -> int:
    """Set the linked words of one record whose list changed; returns the words written."""
    groups: dict[int, list[CountLink]] = {}
    for link in count_links(tree.build, tree.ext, record.name):
        groups.setdefault(link.count_offset, []).append(link)
    order = build_order(tree.build)
    written = 0
    for word, links in groups.items():
        changed = [link for link in links if link.pair_offset == forced or (
            forced is None and _elements(record.pairs.get(link.pair_offset), link.element_size)
            != _parsed_elements(record, link.pair_offset, link.element_size))]
        if not changed:
            continue
        values = {_elements(record.pairs.get(link.pair_offset), link.element_size) - link.minus_one
                  for link in (links if strict else changed)}
        if len(values) > 1:
            raise FormatError(f"{record.name}: lists linked to the count word at +0x{word:X} have "
                              f"different counts {sorted(values)} (W14-20)")
        value = values.pop()
        struct.pack_into(order + ("i" if value < 0 else "I"), record.raw, word, value)
        written += 1
    return written


def update_count_words(tree: SnoTree, record: Record, pair_offset: int) -> int:
    """W14-20: set the PROVEN count words linked to one list of ``record`` to its element count."""
    return _set_words(tree, record, pair_offset, strict=False)


def tree_records(tree: SnoTree) -> list[Record]:
    """Every record of the tree that is reached by the walk, top first, each once."""
    found: dict[int, Record] = {id(tree.top): tree.top}
    for _, record, _, value in iter_pairs(tree):
        found.setdefault(id(record), record)
        if value is not None and value.block is not None and value.block.elements:
            for element in value.block.elements:
                found.setdefault(id(element), element)
    return list(found.values())


def derive_counts(tree: SnoTree) -> int:
    """W14-20 for a whole tree: for every PROVEN link whose list's element count differs from the
    count when parsed (a new list counts from 0), set the linked word. Lists sharing one word
    must then agree. Returns the number of words written."""
    return sum(_set_words(tree, record, None, strict=True) for record in tree_records(tree))


def new_record(build: str, ext: str, name: str) -> Record:
    """A zero-filled record of a table type, for building trees."""
    return Record(name, bytearray(record_layout(build, ext, name).size))


__all__ = [
    "Block", "CountLink", "LayoutField", "PairSlot", "PairValue", "Record", "RecordLayout", "SnoTree",
    "count_links", "covered_types", "derive_counts", "find_pair", "is_canonical", "iter_pairs",
    "new_record", "pair_slots", "parse_tree", "rebuild_sno", "record_layout", "set_block",
    "set_elements", "shared_pairs", "top_record", "tree_records", "update_count_words", "write_tree",
]
