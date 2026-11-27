"""Stage 23 (``55_stage23_effect_types.md``): named export and import of the 31 remaining SNO types
(W23-1 to W23-9).

The files are read through the generic tree (``sno_layout``, ``15``): each record of the tree is named by
the rows of ``data/f55_layouts.tsv`` for its build, group and record (``leaf_table``). A record's JSON holds
its named leaves (dotted names, W23-1), its curves (W23-3), its pairs (record arrays as lists of records,
value arrays as lists of numbers) and, under ``"unnamed"``, every byte no named leaf and no pair covers
(W23-6). ``decode`` gives the JSON of W23-7; ``encode`` writes an edited JSON onto its template by the rules
of stage 22 (W23-8): only differing leaves are written, lists change length, and the file is laid out with
``write_tree(derive=True)``.

W23-2: a table row that does not lie inside its record, overlaps a pair or another row, or (an ``ARRAY:``
row) does not fall on a pair of the tree, is a ``layout_conflict``: reported, its bytes left unnamed.

Stage 25 (``60_stage25_create.md``): every type with rows in the table is read (W25-1); the JSON lists the pairs
that share a block (``"shared"``, W25-3); ``create`` builds a file from its JSON alone (W25-4, W25-5) and
``add_new`` adds it to an archive as a new SNO file (W25-6).
"""

from __future__ import annotations

import copy
import csv
import json
import struct
from dataclasses import dataclass, field
from functools import lru_cache

from . import sno_layout as L
from .binary import FormatError, build_order
from .datafiles import DATA
from .sno import group_by_id, parse_header

__all__ = ["GROUPS", "Leaf", "Decoded", "leaf_table", "decode", "decode_full", "encode", "encode_result",
           "schema", "enum_names", "FINDINGS", "types", "shared_list", "create", "create_result", "add_new",
           "NewFile", "ALWAYS_SHARED"]

# ``57``: the corrected tables replace ``f56_*`` and ``f55_*``; ``58``: ``f58_*`` (59 types) replace ``f57_*``;
# ``61`` AI3: ``f61_layouts.tsv`` replaces ``f58_layouts.tsv`` (the enums stay). The older tables stay, unused.
LAYOUT_TABLE = "f61_layouts.tsv"
ENUM_TABLE = "f58_enums.tsv"
OWN_HANDLE = "tHeader.snoHandle"  # ``57`` AG2: the top record's own handle, not changed by the import
GROUPS = ("prt", "efg", "lit", "mus", "ams", "aib", "ais", "adv", "qsr", "xpl", "dfn", "wth", "trl", "rop", "clt",
          "shk", "srf", "obs", "phy", "rev", "tut", "bos", "aco", "act", "enc", "an2", "scg", "flg", "ctr", "ant",
          "vib")
FINDINGS = ("layout_conflict", "curve_count_differs", "link_unresolved")
FLOATS = frozenset({"FLOAT", "ANGLE", "VELOCITY", "IMPULSE", "ACCEL", "MASS", "ANGULARVELOCITY"})
NODES = "ptArrayNodes"
PATH_HEADER = "PathHeader"  # ``55`` W23-3; ``f56`` has none (``56`` AF3)
# W23-3: the curve header, by the leaf name after the curve's prefix
HEADER = {
    "tHeader.nNodeCount": "count", "tHeader.tLoopPointStart": "loop_start", "tHeader.tLoopPointEnd": "loop_end",
    "tHeader.nDistributionFunction": "distribution", "tHeader.tInterpScalar.nScalarFunction": "function",
    "tHeader.tInterpScalar.flScalarLerpZero": "at_zero", "tHeader.tInterpScalar.flScalarLerpOne": "at_one",
}
HEADER_ENUMS = {"distribution": "DistributionFunction", "function": "ScalarFunction"}
HEADER_CODES = {"count": "i", "loop_start": "f", "loop_end": "f", "distribution": "i", "function": "i",
                "at_zero": "f", "at_one": "f"}
NODE_PARTS = {"tTime": "time", "tValueMin": "min", "tValueMax": "max"}
UNNAMED = "unnamed"
SHARED = "shared"  # ``60`` W25-3
NON_FINITE = ("nan", "inf", "-inf")  # ``export.plain``: non-finite floats as strings
VALUE_CODES = {"BYTE": "B", "WORD": "H", "INT": "i", "SNO": "i"}


# --- the tables (W23-1) ---------------------------------------------------------------------------------

@dataclass(frozen=True)
class Leaf:
    """One row of ``f55_layouts.tsv``. ``offset`` is None for a PS3 ``open`` row; ``group`` is the SNO group
    number of a link, else None."""

    build: str
    ext: str
    record: str
    name: str
    offset: int | None
    size: int
    type: str
    unit: str
    enum: str
    flags: str
    group: int | None
    status: str
    pair_offset: int | None = None  # ``56`` AF2: where an ``ARRAY:`` row's pair is, from the record's start

    @property
    def is_array(self) -> bool:
        return self.type.startswith("ARRAY:")

    @property
    def element(self) -> str:
        return self.type[6:] if self.is_array else ""


@lru_cache(maxsize=None)
def _rows() -> dict:
    return load_layouts(DATA / LAYOUT_TABLE)


def load_layouts(path) -> dict:
    """The rows of a layout table by (build, ext). ``61`` AI3: a leaf placed twice in one record is an error that
    names the build, type, record and leaf."""
    out: dict[tuple[str, str], list[Leaf]] = {}
    placed: dict[tuple, tuple] = {}
    with open(path, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            offset = row["offset"].strip()
            group = row["sno_group"].strip()
            pair = (row.get("pair_offset") or "").strip()
            out.setdefault((row["build"], row["ext"]), []).append(Leaf(
                row["build"], row["ext"], row["record"], row["leaf"], int(offset) if offset else None,
                int(row["bytes"]), row["type"], row["unit"], row["enum"], row["flag_bits"],
                int(group) if group else None, row["status"], int(pair) if pair else None))
            if offset:
                # rows of one leaf at the same place (a copy of a row, the union rows of ``con``) are one place
                key = (row["build"], row["ext"], row["record"], row["leaf"])
                place = (int(offset), int(row["bytes"]))
                if placed.setdefault(key, place) != place:
                    raise FormatError(f"{path}: {row['build']} .{row['ext']} {row['record']}: leaf {row['leaf']!r} "
                                      f"is placed twice (offsets {placed[key][0]} and {place[0]})")
    return {key: tuple(rows) for key, rows in out.items()}


def leaf_table(build: str, ext: str) -> list[Leaf]:
    """W23-9: the rows of the layout table for ``build`` and ``ext``, in table order."""
    return list(_rows().get((build, ext.lower().lstrip(".")), ()))


@lru_cache(maxsize=None)
def _enum_table() -> dict:
    out: dict[str, dict] = {}
    with open(DATA / ENUM_TABLE, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            entry = out.setdefault(row["enum"], {"value": {}, "bit": {}})
            kind = "bit" if row["kind"] == "bit number" else "value"
            entry[kind].setdefault(int(row["value"], 0), row["name"])
    return out


def enum_names(enum: str, kind: str = "value") -> dict:
    """value (or bit, ``kind`` "bit") -> name, from ``f55_enums.tsv``."""
    return dict(_enum_table().get(enum, {}).get(kind, {}))


@lru_cache(maxsize=None)
def _by_record(build: str, ext: str) -> dict:
    out: dict[str, list[Leaf]] = {}
    for leaf in _rows().get((build, ext), ()):
        out.setdefault(leaf.record, []).append(leaf)
    return out


def _table_record(build: str, ext: str, tree_name: str | None) -> str | None:
    """The table record of a tree record: its own name (``56`` AF3: PS3 rows name the tree's records, ``anon:``
    ones included), else without a ``#n`` suffix."""
    if not tree_name:
        return None
    records = _by_record(build, ext)
    if tree_name in records:
        return tree_name
    if tree_name.startswith("anon:"):
        return None
    for name in (tree_name.split("#")[0], tree_name.split("@")[0]):
        if name in records:
            return name
    return None


@lru_cache(maxsize=None)
def _top_table_record(build: str, ext: str) -> str | None:
    """The table record of the top record: the tree's own name when the table has it, else the one record no
    ``ARRAY:`` row points to that ends in ``Definition`` (x360 names some top records otherwise)."""
    top = L.top_record(build, ext)
    records = _by_record(build, ext)
    if top in records:
        return top
    targets = {leaf.element for rows in records.values() for leaf in rows if leaf.is_array}
    free = [name for name in records if name not in targets and name != PATH_HEADER]
    named = [name for name in free if name.endswith("Definition")]
    return (named or free or [None])[0]


# --- record views (W23-2) -------------------------------------------------------------------------------

@dataclass(frozen=True)
class _Named:
    """A placed leaf of a record: its JSON name, offset in the record and stored form."""

    name: str
    offset: int
    size: int
    code: str  # struct code, or "rgba"
    enum: str = ""
    flags: str = ""
    group: int | None = None
    leaf: Leaf | None = None


@dataclass
class _Pair:
    slot: L.PairSlot
    name: str
    element: str | None  # the table record of its elements


@dataclass
class _Curve:
    name: str  # "" when the record itself is a curve
    pair: _Pair
    parts: dict  # part -> _Named


@dataclass
class _View:
    size: int
    plain: list
    pairs: list
    curves: list
    unnamed: list  # (offset, size)
    conflicts: list  # Leaf
    open: int  # PS3 open rows of this record


def _code(leaf_type: str, size: int, name: str) -> str:
    if leaf_type == "RGBACOLOR":
        return "rgba"
    if leaf_type == "CHARARRAY":  # ``56`` AF4
        return "text"
    if leaf_type in FLOATS:
        return "f"
    if size == 8:  # ``61`` AI1: INT64, one signed 64-bit integer
        return "q"
    if size == 2:
        return "h" if leaf_type == "TIME_SHORT" else "H"
    if size == 1:
        return "B"
    if not leaf_type:  # a PathHeader row: the header's own form (W23-3)
        part = HEADER.get("tHeader." + name)
        return HEADER_CODES.get(part, "i")
    return "i"


def _named(leaf: Leaf, name: str | None = None, offset: int | None = None) -> _Named:
    name = leaf.name if name is None else name
    enum = leaf.enum
    part = next((p for suffix, p in HEADER.items() if name == suffix or name.endswith("." + suffix)), None)
    if not enum and part in HEADER_ENUMS:
        enum = HEADER_ENUMS[part]
    code = _code(leaf.type, leaf.size, leaf.name) if leaf.type else HEADER_CODES.get(part, "i")
    # ``58`` W24-1: a ``GBID`` row's ``sno_group`` is a balance type, not an SNO group: no link form here
    group = None if leaf.type == "GBID" else leaf.group
    return _Named(name, leaf.offset if offset is None else offset, leaf.size, code, enum, leaf.flags, group, leaf)


def _same_element(row: Leaf, slot: L.PairSlot) -> bool:
    """An ``ARRAY:`` row falls on a pair whose elements the tree names alike (or does not name), or whose element
    record has the row's element size (``56``: the PS3 tree names the nodes of the emitter curves ``FloatPath``,
    12 bytes, where the table has ``FloatNode``)."""
    element = slot.field.element or ""
    if element.startswith("anon") or element == row.element or \
            element.split("#")[0].split("@")[0] == row.element.split("#")[0].split("@")[0]:
        return True
    if not slot.field.is_record_array:
        return False
    try:
        return L.record_layout(row.build, row.ext, element).size == L.record_layout(row.build, row.ext, row.element).size
    except FormatError:
        return False


@lru_cache(maxsize=None)
def _view(build: str, ext: str, table: str | None, tree_name: str) -> _View:
    layout = L.record_layout(build, ext, tree_name)
    size = layout.size
    slots = L.pair_slots(build, ext, tree_name)
    by_offset = {slot.offset: slot for slot in slots}
    rows = list(_by_record(build, ext).get(table, ())) if table else []
    open_rows = sum(1 for r in rows if r.offset is None)
    placed: list[tuple[_Named, Leaf]] = [(_named(r), r) for r in rows if r.offset is not None and not r.is_array]
    arrays = [r for r in rows if r.offset is not None and r.is_array]
    conflicts: list[Leaf] = []
    # ``56`` AF2: an ``ARRAY:`` row's pair is at its ``pair_offset`` (the pointer, where the build has one, at
    # ``offset``); a row whose pair the tree does not have there, or holds other elements, is a conflict
    at_pair: dict[int, Leaf] = {}
    for row in arrays:
        slot = by_offset.get(row.pair_offset) if row.pair_offset is not None else None
        if slot is not None and slot.field.is_pair and row.pair_offset not in at_pair and _same_element(row, slot):
            at_pair[row.pair_offset] = row
    names = {slot.offset: at_pair[slot.offset].name if slot.offset in at_pair else slot.path for slot in slots}
    # ``56`` AF3: PS3 curve headers are rows of the parent record (``f56``); no PathHeader rows any more
    placed_rows = {id(row) for row in at_pair.values()}
    conflicts.extend(row for row in arrays if id(row) not in placed_rows)
    pair_spans = [(slot.offset, slot.offset + 8) for slot in slots]
    bad: set[int] = set()
    for i, (named, row) in enumerate(placed):
        start, end = named.offset, named.offset + named.size
        if start < 0 or end > size or any(start < b and a < end for a, b in pair_spans):
            bad.add(i)
            continue
        for j in range(i + 1, len(placed)):
            other = placed[j][0]
            if start < other.offset + other.size and other.offset < end:
                bad.update((i, j))
    for i in sorted(bad):
        conflicts.append(placed[i][1])
    good = [named for i, (named, _) in enumerate(placed) if i not in bad]
    pairs = []
    taken = {named.name for named in good} | {UNNAMED}
    # rows' names first, then the tree's; a name another key already has takes the pair's offset (keys stay unique)
    for slot in sorted(slots, key=lambda s: (s.offset not in at_pair, slots.index(s))):
        row = at_pair.get(slot.offset)
        element = None
        if slot.field.is_record_array:
            # ``56`` AF3: the table names PS3 records as the tree does (``FloatPath#1``): the tree's own element
            # name when the table has it, else the row's element
            records = _by_record(build, ext)
            if slot.field.element in records:
                element = slot.field.element
            elif row is not None:
                element = row.element
            else:
                element = _table_record(build, ext, slot.field.element)
        elif row is not None:
            element = row.element
        name = row.name if row is not None else slot.path
        if name in taken:
            name = f"{name}@{slot.offset:X}"
        taken.add(name)
        pairs.append(_Pair(slot, name, element))
    pairs.sort(key=lambda pair: slots.index(pair.slot))
    curves = []
    used: set[str] = set()
    for pair in pairs:
        if not pair.name.endswith(NODES) or not pair.slot.field.is_record_array:
            continue
        prefix = pair.name[:-len(NODES)]
        parts = {}
        for named in good:
            part = HEADER.get(named.name[len(prefix):]) if named.name.startswith(prefix) else None
            if part is None or part in parts:
                continue  # ``61`` AI3: the PS3 inline curves have their count row too (``f61``)
            parts[part] = named
            used.add(named.name)
        curves.append(_Curve(prefix.rstrip("."), pair, parts))
    plain = []
    names: set = set()
    for named in good:
        if named.name in used and any(named is p for c in curves for p in c.parts.values()):
            continue
        if named.name in names or named.name in {c.name for c in curves if c.name}:
            continue  # a name taken twice: its bytes stay unnamed (``61`` AI3)
        names.add(named.name)
        plain.append(named)
    exported = plain + [p for c in curves for p in c.parts.values()]
    # ``61`` AI3: only what the export writes covers bytes; every other word is an unnamed word
    covered = bytearray(size)
    for named in exported:
        covered[named.offset:named.offset + named.size] = b"\1" * named.size
    for a, b in pair_spans:
        covered[a:min(b, size)] = b"\1" * (min(b, size) - a)
    unnamed = []
    at = 0
    while at < size:
        if covered[at]:
            at += 1
            continue
        end = at
        while end < size and not covered[end] and (end == at or end % 4):
            end += 1
        unnamed.append((at, end - at))
        at = end
    return _View(size, plain, pairs, curves, unnamed, conflicts, open_rows)


# --- values (W23-3 to W23-5) ----------------------------------------------------------------------------

def _read(raw, named: _Named, order: str):
    if named.code == "rgba":
        return list(raw[named.offset:named.offset + 4])
    if named.code == "text":
        return bytes(raw[named.offset:named.offset + named.size])
    return struct.unpack_from(order + named.code, raw, named.offset)[0]


def _float(value):
    from .export import plain

    return plain(value)


@dataclass
class Decoded:
    """``decode_full``: the JSON and what coverage reports of one file."""

    json: dict
    named: int = 0
    unnamed: int = 0
    conflicts: list = field(default_factory=list)  # (record, leaf)
    counts_differ: list = field(default_factory=list)  # JSON paths
    links: list = field(default_factory=list)  # (path, group id, handle)
    open: int = 0


class _Ctx:
    def __init__(self, build: str, ext: str, order: str, out: Decoded | None = None) -> None:
        self.build, self.ext, self.order = build, ext, order
        self.out = out if out is not None else Decoded({})
        self.seen_conflicts: set = set()


def _value(ctx: _Ctx, raw, named: _Named, path: tuple):
    value = _read(raw, named, ctx.order)
    ctx.out.named += 1
    if named.code == "text":
        return _text_json(value)
    if named.code == "f":
        return _float(value)
    if named.enum:
        return {"value": value, "name": enum_names(named.enum).get(value)}
    if named.flags:
        bits = enum_names(named.flags, "bit")
        mask = value & ((1 << (8 * named.size)) - 1)
        return {"value": value, "names": [bits.get(k, f"bit{k}") for k in range(8 * named.size) if mask >> k & 1]}
    if named.group is not None:
        if value == -1:  # ``60`` W25-2: null is -1; a stored 0 keeps its value
            return None
        group = group_by_id(named.group)
        if value != 0:  # W24-3: 0 is no link
            ctx.out.links.append((path, named.group, value))
        return {"group": group.extension if group else str(named.group), "handle": value}
    return value


def _text_json(raw: bytes):
    """``56`` AF4: the bytes up to the first zero, as a string; a field with other bytes after it (or bytes that
    are not ASCII) as ``{"text", "raw"}`` (``raw``: the hex of all n bytes)."""
    text, _, rest = raw.partition(b"\0")
    if not any(rest) and all(b < 0x80 for b in text):
        return text.decode("ascii")
    return {"text": text.decode("latin-1"), "raw": raw.hex().upper()}


def _text_bytes(new, old, size: int, path: tuple) -> bytes:
    """``56`` AF4: a string as its bytes, then zeros to the field's length (n bytes or more, or not ASCII:
    ``value_out_of_range``); a ``{"text", "raw"}`` field from ``raw`` unless its text changed."""
    if isinstance(new, dict):
        if set(new) != {"text", "raw"} or not isinstance(new["text"], str) or not isinstance(new["raw"], str):
            raise _refused(path, "bad_json", "a text field is a string or {\"text\", \"raw\"}")
        if not (isinstance(old, dict) and old.get("text") != new["text"]):
            try:
                raw = bytes.fromhex(new["raw"])
            except ValueError:
                raise _refused(path + ("raw",), "bad_json", "not hex") from None
            if len(raw) != size:
                raise _refused(path + ("raw",), "value_out_of_range", f"{len(raw)} bytes for a field of {size}")
            return raw
        new = new["text"]
    if not isinstance(new, str):
        raise _refused(path, "bad_json", "a string")
    if any(ord(c) >= 0x80 for c in new) or "\0" in new:
        raise _refused(path, "value_out_of_range", "only ASCII text without zero bytes is written")
    raw = new.encode("ascii")
    if len(raw) >= size:
        raise _refused(path, "value_out_of_range", f"{len(raw)} bytes for a text field of {size} (with its zero)")
    return raw + bytes(size - len(raw))


def _view_of(ctx: _Ctx, table: str | None, record: L.Record) -> _View:
    view = _view(ctx.build, ctx.ext, table, record.name)
    for leaf in view.conflicts:
        key = (leaf.record, leaf.name)
        if key not in ctx.seen_conflicts:
            ctx.seen_conflicts.add(key)
            ctx.out.conflicts.append(key)
    return view


def _record_json(ctx: _Ctx, record: L.Record, table: str | None, path: tuple, node: bool = False) -> dict:
    view = _view_of(ctx, table, record)
    ctx.out.open += view.open
    out: dict = {}
    raw = record.raw
    if node:
        parts: dict[str, list] = {}
        for named in view.plain:
            head = named.name.split(".")[0]
            part = NODE_PARTS.get(head)
            if part is None:
                out[named.name] = _value(ctx, raw, named, path + (named.name,))
            else:
                parts.setdefault(part, []).append(named)
        for part, items in parts.items():
            values = [_value(ctx, raw, n, path + (part,)) for n in sorted(items, key=lambda n: n.offset)]
            out[part] = values[0] if len(items) == 1 and items[0].name == _node_head(part) else values
    else:
        for named in view.plain:
            out[named.name] = _value(ctx, raw, named, path + (named.name,))
    for curve in view.curves:
        where = path + ((curve.name,) if curve.name else ())
        body = _curve_json(ctx, record, curve, where)
        if curve.name:
            out[curve.name] = body
        else:
            out.update(body)
    curve_pairs = {id(c.pair) for c in view.curves}
    for pair in view.pairs:
        if id(pair) not in curve_pairs:
            out[pair.name] = _pair_json(ctx, record, pair, path + (pair.name,))
    if view.unnamed:
        out[UNNAMED] = {_word_key(o): int.from_bytes(raw[o:o + n], "big" if ctx.order == ">" else "little")
                        for o, n in view.unnamed}
        ctx.out.unnamed += len(view.unnamed)
    return out


def _node_head(part: str) -> str:
    return {v: k for k, v in NODE_PARTS.items()}[part]


def _word_key(offset: int) -> str:
    return f"0x{offset:04X}"


def _curve_json(ctx: _Ctx, record: L.Record, curve: _Curve, path: tuple) -> dict:
    raw = record.raw

    def part(name):
        named = curve.parts.get(name)
        if named is None:
            return None
        value = _value(ctx, raw, named, path + (name,))
        return value

    nodes = _nodes(ctx, record, curve.pair, path + ("nodes",))
    out = {"loop": [part("loop_start"), part("loop_end")], "distribution": part("distribution"),
           "scalar": {"function": part("function"), "at_zero": part("at_zero"), "at_one": part("at_one")},
           "nodes": nodes}
    count = curve.parts.get("count")
    if count is not None:
        stored = _read(raw, count, ctx.order)
        if stored != len(nodes):  # W23-3: reported, kept
            out["stored_count"] = stored
            ctx.out.counts_differ.append(path)
    return out


def _elements(record: L.Record, pair: _Pair) -> list:
    value = record.pairs.get(pair.slot.offset)
    if value is None or value.block is None or value.block.elements is None:
        return []
    return value.block.elements


def _nodes(ctx: _Ctx, record: L.Record, pair: _Pair, path: tuple) -> list:
    return [_record_json(ctx, element, pair.element, path + (i,), node=True)
            for i, element in enumerate(_elements(record, pair))]


def _pair_json(ctx: _Ctx, record: L.Record, pair: _Pair, path: tuple):
    f = pair.slot.field
    value = record.pairs.get(pair.slot.offset)
    block = value.block if value is not None else None
    if f.is_record_array:
        return [_record_json(ctx, element, pair.element, path + (i,)) for i, element in enumerate(_elements(record, pair))]
    data = bytes(block.data or b"") if block is not None and block.elements is None else b""
    if f.kind == "array" and f.element in VALUE_CODES:
        code = VALUE_CODES[f.element]
        unit = struct.calcsize(code)
        if len(data) % unit == 0:
            return [struct.unpack_from(ctx.order + code, data, k)[0] for k in range(0, len(data), unit)]
    if f.kind == "array" and f.element == "VECTOR3D" and len(data) % 12 == 0:
        return [[_float(v) for v in struct.unpack_from(ctx.order + "3f", data, k)] for k in range(0, len(data), 12)]
    return {"bytes": data.hex().upper()}


def types(build: str | None = None) -> tuple[str, ...]:
    """``60`` W25-1: the types with rows in the layout table (on ``build``, or on any build)."""
    return tuple(sorted({e for b, e in _rows() if build is None or b == build}))


def _check_type(ext: str, build: str) -> None:
    if (build, ext) not in _rows():
        raise FormatError(f".{ext} has no rows in {LAYOUT_TABLE} on {build}")


def decode_full(data: bytes, ext: str, build: str | None = None) -> Decoded:
    """W23-7 with what coverage reports (W23-2 to W23-6); ``60`` W25-1: every type with rows in the table on the
    file's build; W25-3: the top-level ``"shared"`` list."""
    ext = ext.lower().lstrip(".")
    if ext not in types():
        raise FormatError(f".{ext} has no rows in {LAYOUT_TABLE}")
    tree = L.parse_tree(data, ext, build)
    build = tree.build
    _check_type(ext, build)
    header = parse_header(data, build)
    out = Decoded({})
    ctx = _Ctx(build, ext, build_order(build), out)
    body = _record_json(ctx, tree.top, _top_table_record(build, ext), ("record",))
    out.json = {"header": {"handle": header.handle, "version": header.version, "revision": header.revision,
                           "source_hash": header.source_hash,
                           "byte_order": "big" if header.order == ">" else "little"},
                "build": build, "group": ext, "record": body, SHARED: shared_list(tree)}
    return out


def decode(data: bytes, ext: str, build: str | None = None) -> dict:
    """W23-9: the JSON of W23-7."""
    return decode_full(data, ext, build).json


# --- the manifest schema (W23-7) ------------------------------------------------------------------------

def schema(ext: str) -> dict:
    """``schemas.<ext>``: per build, the table's rows (record, leaf, stored type, unit, enum, flags, link
    group), so a reader needs no copy of the table."""
    out = {}
    for build in ("ps3", "switch", "x360"):
        rows = []
        for leaf in leaf_table(build, ext):
            group = group_by_id(leaf.group) if leaf.group is not None and leaf.type != "GBID" else None
            rows.append({"record": leaf.record, "leaf": leaf.name, "type": leaf.type, "unit": leaf.unit or None,
                         "enum": leaf.enum or None, "flags": leaf.flags or None,
                         "link_group": (group.extension if group else str(leaf.group)) if leaf.group is not None
                         and leaf.type != "GBID" else None, "offset": leaf.offset, "status": leaf.status})
        out[build] = rows
    return out


# --- import (W23-8) ---------------------------------------------------------------------------------------

def _refused(path, reason, message=""):
    from .json_import import Refused

    return Refused(path, reason, message)


class _Enc:
    def __init__(self, tree: L.SnoTree, ext: str, known=None) -> None:
        self.tree, self.ext, self.build = tree, ext, tree.build
        self.order = build_order(tree.build)
        self.refusals: list = []
        self.checks: list = []  # (path, value) a re-export must give (W22-5)
        self.rows: list = []
        self.warnings: list = []
        self.known = known
        self.ctx = _Ctx(self.build, ext, self.order)
        # ``56`` AF1: blocks are changed where they are, so pairs sharing one (W11-6) keep sharing it; a block
        # changed through one path is not changed again differently through another
        self.refs: dict[int, int] = {}
        for _, _, _, value in L.iter_pairs(tree):
            if value is not None and value.block is not None:
                self.refs[id(value.block)] = self.refs.get(id(value.block), 0) + 1
        self.blocks: dict[int, tuple] = {}  # id(block) -> (path, new JSON)
        self.resized: dict[int, tuple] = {}  # id(node block) -> (old length, new length)
        self.creating = False  # ``60`` W25-5: built on zero records; every leaf is written as given
        self.placed: dict = {}  # ``create``: JSON path -> (record, offset) of a written leaf

    def claim(self, block, path: tuple, new) -> bool:
        """False when ``block`` was already written through another path with the same value."""
        if block is None:
            return True
        earlier = self.blocks.get(id(block))
        if earlier is not None:
            if not _same(earlier[1], new):
                raise _refused(path, "conflicting_values", f"{_path_text(earlier[0])} shares this block and sets it "
                                                           "to something else")
            return False
        self.blocks[id(block)] = (path, new)
        if self.refs.get(id(block), 0) > 1:
            self.warnings.append(f"shared_block: {_path_text(path)} shares its block with other pairs (W11-6); "
                                 "they change too")
        return True


def _same(a, b) -> bool:
    from .json_import import _same_json

    return _same_json(a, b)


def _write_named(enc: _Enc, record: L.Record, named: _Named, old, new, path: tuple) -> None:
    from .json_import import Change, Place, _enum_value, _flags_value, _pack

    if path == ("record", OWN_HANDLE) and record is enc.tree.top:
        raise _refused(path, "field_fixed", "the file's own handle is not changed by the import (as the header)")

    enc.rows.append({"path": _path_text(path), "old": old, "new": new})
    if enc.creating:
        enc.placed[path] = (record, named.offset)
    if named.code == "text":
        _put(record, named, _text_bytes(new, old, named.size, path), path)
        if isinstance(new, str):
            enc.checks.append((path, new))
        return
    if named.code == "rgba":
        if not (isinstance(new, list) and len(new) == 4 and all(
                isinstance(c, int) and not isinstance(c, bool) and 0 <= c <= 255 for c in new)):
            raise _refused(path, "value_out_of_range", "a colour is [r, g, b, a], each 0 to 255")
        _put(record, named, bytes(new), path)
        enc.checks.append((path, new))
        return
    if named.enum:
        # ``60`` W25-5: in ``create`` the name (when given) and the value are checked against each other only
        change = Change(path, None if enc.creating else old, new,
                        Place("enum", names={n: v for v, n in enum_names(named.enum).items()}))
        value = _enum_value(change, None)
    elif named.flags and enc.creating:
        value = _created_flags(named, new, path)
    elif named.flags:
        bits = {n: b for b, n in enum_names(named.flags, "bit").items()}
        for k in range(8 * named.size):
            bits.setdefault(f"bit{k}", k)
        value = _flags_value(Change(path, old, new, Place("flags", names=bits)), None)
        if named.code.islower() and isinstance(value, int) and value >= 1 << (8 * named.size - 1):
            value -= 1 << (8 * named.size)  # a set top bit of a signed word
    elif named.group is not None:
        if new is None:
            value = -1
        else:
            group = group_by_id(named.group)
            want = group.extension if group else str(named.group)
            if not isinstance(new, dict) or set(new) != {"group", "handle"}:
                raise _refused(path, "bad_json", "a link is {\"group\", \"handle\"} or null")
            if new["group"] != want:
                raise _refused(path, "wrong_group", f"group {new['group']!r}; the field links {want!r}")
            value = new["handle"]
            if enc.known is not None and group is not None and isinstance(value, int) and not enc.known(group.id, value):
                enc.warnings.append(f"dangling_link: {_path_text(path)} names {want} {value}, which is not in the "
                                    "game folder's index")
        enc.checks.append((path, new))
    else:
        value = new
        if enc.creating and named.code == "f" and new in NON_FINITE:  # ``60``: as the export writes them
            value = float(new)
        enc.checks.append((path, new))
    _put(record, named, _pack(named.code, value, path, enc.order), path)


def _put(record: L.Record, named: _Named, data: bytes, path: tuple) -> None:
    """``61`` AI1: a leaf is written with exactly as many bytes as its row has; the record keeps its length."""
    if len(data) != named.size or named.offset + named.size > len(record.raw):
        raise _refused(path, "value_out_of_range", f"{len(data)} bytes for a row of {named.size} at "
                                                   f"+{named.offset} of a {len(record.raw)}-byte record")
    record.raw[named.offset:named.offset + named.size] = data


def _created_flags(named: _Named, new, path: tuple) -> int:
    """``60`` W25-5: a flags field of ``create``: its value, or the bits of its names; both given must agree (over
    the field's bits, so a set top bit of a signed word agrees with its negative value)."""
    if not isinstance(new, dict) or set(new) - {"value", "names"} or not ("value" in new or "names" in new):
        raise _refused(path, "bad_json", "flags are {\"value\", \"names\"}")
    width = 8 * named.size
    full = (1 << width) - 1
    value = new.get("value")
    if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
        raise _refused(path, "value_out_of_range", f"{value!r} is not an integer")
    names = new.get("names")
    if names is not None:
        if not isinstance(names, list):
            raise _refused(path, "bad_json", "flag names are a list")
        bits = {n: b for b, n in enum_names(named.flags, "bit").items()}
        for k in range(width):
            bits.setdefault(f"bit{k}", k)
        unknown = [n for n in names if n not in bits]
        if unknown:
            raise _refused(path, "unknown_name", f"{unknown[0]!r} is not a flag of this field")
        from_names = sum(1 << bits[n] for n in set(names)) & full
        if value is not None and value & full != from_names:
            raise _refused(path, "conflicting_values", f"value {value} and names {names} disagree")
        if value is None:
            value = from_names
    if named.code.islower() and value >= 1 << (width - 1):
        value -= 1 << width
    return value


def _path_text(path) -> str:
    from .json_import import path_text

    return path_text(path)


def _keys(old, new, path) -> None:
    if not isinstance(new, dict):
        raise _refused(path, "bad_json", "an object")
    if set(old) != set(new):
        raise _refused(path, "bad_json", "keys were added or removed")


def _guard(enc: _Enc, fn, *args) -> None:
    from .json_import import Refused, _failed

    try:
        fn(*args)
    except Refused as refusal:
        enc.refusals.append(refusal)
    except (FormatError, struct.error, TypeError, ValueError, KeyError, IndexError) as exc:
        enc.refusals.append(_failed(args[-1] if args and isinstance(args[-1], tuple) else (), exc))


def _encode_record(enc: _Enc, record: L.Record, table: str | None, old: dict, new, path: tuple,
                   node: bool = False) -> None:
    if enc.creating and isinstance(new, dict):  # ``60``: a record that is a curve may give its stored count
        _keys({k: 0 for k in old if k != "stored_count"}, {k: 0 for k in new if k != "stored_count"}, path)
    else:
        _keys(old, new, path)
    view = _view(enc.build, enc.ext, table, record.name)
    if node:
        groups: dict[str, list] = {}
        for named in view.plain:
            part = NODE_PARTS.get(named.name.split(".")[0])
            if part is None:
                if not _same(old[named.name], new[named.name]):
                    _guard(enc, _write_named, enc, record, named, old[named.name], new[named.name], path + (named.name,))
            else:
                groups.setdefault(part, []).append(named)
        for part, items in groups.items():
            items = sorted(items, key=lambda n: n.offset)
            if _same(old[part], new[part]):
                continue
            single = len(items) == 1 and items[0].name == _node_head(part)
            if single:
                _guard(enc, _write_named, enc, record, items[0], old[part], new[part], path + (part,))
            elif not isinstance(new[part], list) or len(new[part]) != len(items):
                enc.refusals.append(_refused(path + (part,), "value_out_of_range", f"{len(items)} components"))
            else:
                for k, named in enumerate(items):
                    if not _same(old[part][k], new[part][k]):
                        _guard(enc, _write_named, enc, record, named, old[part][k], new[part][k], path + (part, k))
    else:
        for named in view.plain:
            if not _same(old[named.name], new[named.name]):
                _guard(enc, _write_named, enc, record, named, old[named.name], new[named.name], path + (named.name,))
    for curve in view.curves:
        if curve.name:
            if curve.name not in new:
                continue
            _guard(enc, _encode_curve, enc, record, curve, old[curve.name], new[curve.name], path + (curve.name,))
        else:
            _guard(enc, _encode_curve, enc, record, curve, old, new, path)
    curve_pairs = {id(c.pair) for c in view.curves}
    for pair in view.pairs:
        if id(pair) not in curve_pairs and not _same(old[pair.name], new[pair.name]):
            _guard(enc, _encode_pair, enc, record, pair, old[pair.name], new[pair.name], path + (pair.name,))
    if UNNAMED in old and not _same(old[UNNAMED], new[UNNAMED]):
        _guard(enc, _encode_unnamed, enc, record, view, old[UNNAMED], new[UNNAMED], path + (UNNAMED,))


def _encode_unnamed(enc: _Enc, record: L.Record, view: _View, old: dict, new, path: tuple) -> None:
    _keys(old, new, path)
    for offset, size in view.unnamed:
        key = _word_key(offset)
        if _same(old[key], new[key]):
            continue
        value = new[key]
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 1 << (8 * size):
            enc.refusals.append(_refused(path + (key,), "value_out_of_range", f"an unsigned {8 * size}-bit word"))
            continue
        record.raw[offset:offset + size] = value.to_bytes(size, "big" if enc.order == ">" else "little")
        enc.rows.append({"path": _path_text(path + (key,)), "old": old[key], "new": value})
        enc.checks.append((path + (key,), value))


def _encode_curve(enc: _Enc, record: L.Record, curve: _Curve, old: dict, new, path: tuple) -> None:
    if not isinstance(new, dict):
        raise _refused(path, "bad_json", "a curve is an object")
    keys = ("loop", "distribution", "scalar", "nodes") + (() if enc.creating else ("stored_count",))
    if {k for k in old if k in keys} != {k for k in new if k in keys}:
        raise _refused(path, "bad_json", "keys were added or removed")
    parts = [("loop_start", ("loop", 0)), ("loop_end", ("loop", 1)), ("distribution", ("distribution",)),
             ("function", ("scalar", "function")), ("at_zero", ("scalar", "at_zero")), ("at_one", ("scalar", "at_one"))]
    for container, size in (("loop", 2), ("scalar", None)):
        value = new[container]
        if container == "loop" and not (isinstance(value, list) and len(value) == 2):
            raise _refused(path + ("loop",), "bad_json", "a loop is [start, end]")
        if container == "scalar" and not (isinstance(value, dict) and set(value) == set(old["scalar"])):
            raise _refused(path + ("scalar",), "bad_json", "a scalar is {\"function\", \"at_zero\", \"at_one\"}")
    for part, where in parts:
        a, b = old, new
        for key in where:
            a, b = a[key], b[key]
        if _same(a, b):
            continue
        named = curve.parts.get(part)
        if named is None:
            enc.refusals.append(_refused(path + where, "field_fixed", "this part of the curve is not named on this "
                                                                     "build"))
            continue
        _guard(enc, _write_named, enc, record, named, a, b, path + where)
    nodes_old, nodes_new = old["nodes"], new["nodes"]
    changed_length = isinstance(nodes_new, list) and len(nodes_new) != len(nodes_old)
    _encode_list(enc, record, curve.pair, nodes_old, nodes_new, path + ("nodes",), node=True)
    if changed_length:
        value = record.pairs.get(curve.pair.slot.offset)
        if value is not None and value.block is not None:
            enc.resized.setdefault(id(value.block), (len(nodes_old), len(nodes_new)))
    count = curve.parts.get("count")
    if enc.creating:  # ``60`` W25-5: the count is the node count, or the stored count the JSON gives
        if "stored_count" in new and count is None:
            raise _refused(path + ("stored_count",), "field_fixed", "this curve has no count on this build")
        if count is not None:
            if "stored_count" in new:
                _write_named(enc, record, count, None, new["stored_count"], path + ("stored_count",))
                # kept as stored: ``write_tree(derive=True)`` does not set it again from the node count
                value = record.pairs.get(curve.pair.slot.offset)
                record.parsed[curve.pair.slot.offset] = value.pair_size() if value and value.block else 0
            else:
                struct.pack_into(enc.order + "i", record.raw, count.offset, len(nodes_new))
        return
    if count is not None and changed_length:
        record.raw[count.offset:count.offset + 4] = struct.pack(enc.order + "i", len(nodes_new))  # W23-3, W23-8
        if "stored_count" in old:
            enc.refusals.append(_refused(path + ("stored_count",), "derived_field",
                                         "the count of a curve whose nodes change follows its node list"))
    elif count is not None and "stored_count" in old and not _same(old["stored_count"], new["stored_count"]):
        _guard(enc, _write_named, enc, record, count, old["stored_count"], new["stored_count"],
               path + ("stored_count",))


def _zero_record(enc: _Enc, name: str) -> L.Record:
    return L.Record(name, bytearray(L.record_layout(enc.build, enc.ext, name).size))


def _encode_list(enc: _Enc, record: L.Record, pair: _Pair, old: list, new, path: tuple, node: bool = False) -> None:
    if not isinstance(new, list):
        raise _refused(path, "bad_json", "a list")
    elements = list(_elements(record, pair))
    bases: list = list(old)
    if len(new) != len(old):
        # W23-8, W22-5: elements added (a copy of the last one, or a zero record) or removed at the end
        value = record.pairs.get(pair.slot.offset)
        block = value.block if value is not None else None
        if not enc.claim(block, path, new):
            return
        enc.rows.append({"path": _path_text(path), "old": {"length": len(old)}, "new": {"length": len(new)}})
        kept = elements[:len(new)]
        for i in range(len(old), len(new)):
            if elements:
                base = copy.deepcopy(elements[-1])
                base_json = old[-1]
            else:
                base = _zero_record(enc, pair.slot.field.element)
                base_json = json.loads(json.dumps(_record_json(_Ctx(enc.build, enc.ext, enc.order), base, pair.element,
                                                               path + (i,), node=node)))
            kept.append(base)
            bases.append(json.loads(json.dumps(base_json)))
        if block is not None:  # ``56`` AF1: the block itself, so the file is laid out as from scratch
            block.elements = kept
        elif kept:
            record.pairs[pair.slot.offset] = L.PairValue(L.Block(elements=kept))
        L.update_count_words(enc.tree, record, pair.slot.offset)
        elements = kept
    for i, element in enumerate(elements[:len(new)]):
        if not _same(bases[i], new[i]):
            _guard(enc, _encode_record, enc, element, pair.element, bases[i], new[i], path + (i,), node)


def _encode_pair(enc: _Enc, record: L.Record, pair: _Pair, old, new, path: tuple) -> None:
    f = pair.slot.field
    if f.is_record_array:
        _encode_list(enc, record, pair, old, new, path)
        return
    if isinstance(old, dict) or (enc.creating and isinstance(new, dict)):  # ``60``: either form in ``create``
        if not isinstance(new, dict) or set(new) != {"bytes"} or not isinstance(new["bytes"], str):
            raise _refused(path, "bad_json", "raw bytes are {\"bytes\": \"<hex>\"}")
        try:
            data = bytes.fromhex(new["bytes"])
        except ValueError:
            raise _refused(path, "bad_json", "not hex") from None
    else:
        if not isinstance(new, list):
            raise _refused(path, "bad_json", "a list")
        from .json_import import _pack

        if f.element == "VECTOR3D":
            data = b"".join(_pack("f", c, path + (i,), enc.order) for i, v in enumerate(new)
                            for c in (v if isinstance(v, list) and len(v) == 3 else [None] * 3))
        else:
            data = b"".join(_pack(VALUE_CODES[f.element], v, path + (i,), enc.order) for i, v in enumerate(new))
    value = record.pairs.get(pair.slot.offset)
    block = value.block if value is not None else None
    if not enc.claim(block, path, new):
        return
    if block is not None and block.elements is None:  # ``56`` AF1: the block itself (an empty one: no block)
        block.data = data
    else:
        record.pairs[pair.slot.offset] = L.PairValue(L.Block(data) if data else None)
    L.update_count_words(enc.tree, record, pair.slot.offset)
    enc.rows.append({"path": _path_text(path), "old": old, "new": new})
    if not (isinstance(new, dict) and not isinstance(old, dict)):  # ``60``: the bytes form is written as given
        enc.checks.append((path, new))


def _sharer_counts(enc: _Enc, record: L.Record, table: str | None, seen: set) -> None:
    """``56`` AF1: a curve whose node block changed length through another curve (W11-6) takes the new count,
    when its count was that block's old length (a count that differed stays as stored)."""
    if id(record) in seen:
        return
    seen.add(id(record))
    view = _view(enc.build, enc.ext, table, record.name)
    for curve in view.curves:
        count = curve.parts.get("count")
        value = record.pairs.get(curve.pair.slot.offset)
        if count is None or value is None or value.block is None or id(value.block) not in enc.resized:
            continue
        before, after = enc.resized[id(value.block)]
        if struct.unpack_from(enc.order + "i", record.raw, count.offset)[0] == before:
            struct.pack_into(enc.order + "i", record.raw, count.offset, after)
    for pair in view.pairs:
        if pair.slot.field.is_record_array:
            for element in _elements(record, pair):
                _sharer_counts(enc, element, pair.element, seen)


def encode_result(template: bytes, ext: str, build: str | None, edited: dict, known=None):
    """W23-8: (new bytes, changed rows, warnings). Raises ``json_import.Refused`` (``refusals`` holds every
    refusal) when any leaf is refused; nothing is written then."""
    from .json_import import Refused, _close, _json_round, get_path

    ext = ext.lower().lstrip(".")
    old = _json_round(decode(template, ext, build))
    build = old["build"]
    if not isinstance(edited, dict):
        raise Refused((), "bad_json", "an object")
    # ``60`` W25-3: onto a template the tree's own sharing stays (``56`` AF1); the list is not read
    old.pop(SHARED, None)
    edited = {k: v for k, v in edited.items() if k != SHARED}
    if _same(old, edited):
        return bytes(template), [], []
    refusals = []
    for key in ("header", "build", "group"):
        if not _same(old.get(key), edited.get(key)):
            refusals.append(Refused((key,), "field_fixed", "the header, build and group are not changed by the import"))
    if set(edited) != set(old):
        refusals.append(Refused((), "bad_json", "keys were added or removed"))
    tree = L.parse_tree(template, ext, build)
    enc = _Enc(tree, ext, known)
    enc.refusals.extend(refusals)
    if not refusals:
        _guard(enc, _encode_record, enc, tree.top, _top_table_record(build, ext), old["record"], edited["record"],
               ("record",))
    data = None
    if not enc.refusals and enc.resized:
        _sharer_counts(enc, tree.top, _top_table_record(build, ext), set())
    if not enc.refusals:
        try:
            data = L.write_tree(tree, derive=True)
            again = _json_round(decode(data, ext, build))
        except FormatError as exc:
            enc.refusals.append(Refused((), "value_out_of_range", f"the written file does not read back: {exc}"))
        else:
            for path, value in enc.checks:  # W22-5
                try:
                    found = get_path(again, path)
                except (KeyError, IndexError, TypeError):
                    enc.refusals.append(Refused(path, "derived_field", "not in the written file's export"))
                    continue
                if not _close(found, value):
                    enc.refusals.append(Refused(path, "value_out_of_range", f"reads back as {json.dumps(found)}"))
    if enc.refusals:
        error = enc.refusals[0]
        error.refusals = enc.refusals
        raise error
    return data, enc.rows, enc.warnings


def encode(template: bytes, ext: str, build: str | None, edited: dict) -> bytes:
    """W23-9: the new bytes of ``template`` with the edited JSON written (W23-8)."""
    return encode_result(template, ext, build, edited)[0]


# --- creating files from their JSON alone (``60`` W25-1 to W25-8) -----------------------------------------

# ``60`` §1, W25-3: the builds and types whose stored files always point equal blocks (same bytes, same element
# type) at one block. x360 is taken as Switch (Q1033); PS3 ``prt`` has exceptions and is not pooled (Q1032).
_POOLED = frozenset(GROUPS) | {"acr", "mon", "mat", "lvl", "qst", "cnv", "glo", "mrk"}
ALWAYS_SHARED = {"switch": _POOLED, "x360": _POOLED, "ps3": _POOLED - {"mrk", "prt"}}
TOP_KEYS = ("header", "build", "group", "record", SHARED)
CURVE_KEYS = ("loop", "distribution", "scalar", "nodes", "stored_count")


def _sites(build: str, ext: str, record: L.Record, table: str | None, path: tuple, inside: bool = False):
    """Every pair of a record and of its element records, in the order of the JSON: ``(JSON path, record, pair,
    value, inside)``; ``inside`` is True below a pair that shares an earlier block (W11-6)."""
    view = _view(build, ext, table, record.name)
    curve_pairs = {id(c.pair) for c in view.curves}
    for curve in view.curves:
        where = path + ((curve.name,) if curve.name else ()) + ("nodes",)
        yield from _site(build, ext, record, curve.pair, where, inside)
    for pair in view.pairs:
        if id(pair) not in curve_pairs:
            yield from _site(build, ext, record, pair, path + (pair.name,), inside)


def _site(build, ext, record, pair, where, inside):
    value = record.pairs.get(pair.slot.offset)
    yield where, record, pair, value, inside
    if pair.slot.field.is_record_array and value is not None and value.block is not None and value.block.elements:
        for i, element in enumerate(value.block.elements):
            yield from _sites(build, ext, element, pair.element, where + (i,), inside or value.shared)


def shared_list(tree: L.SnoTree) -> list:
    """``60`` W25-3: ``[path, first_path]`` for every pair that points at a block first written at ``first_path``
    (traversal order of the writer), in JSON order. Pairs inside a shared block's elements are not listed (they
    come with it)."""
    from .json_import import path_text

    first: dict[int, tuple] = {}
    found = []
    for where, _, _, value, inside in _sites(tree.build, tree.ext, tree.top,
                                             _top_table_record(tree.build, tree.ext), ("record",)):
        if inside or value is None or value.block is None or value.block.is_empty():
            continue
        if value.shared:  # ``61`` AI4: also a block first written by a pair of another element type
            found.append((where, value.alias_of if value.alias_of is not None else value.block))
        else:
            first.setdefault(id(value.block), where)
    return [[path_text(where), path_text(first[id(block)])] for where, block in found if id(block) in first]


@lru_cache(maxsize=None)
def _zero_text(build: str, ext: str, table: str | None, name: str, node: bool) -> str:
    from .json_import import _json_round

    record = L.new_record(build, ext, name)
    return json.dumps(_json_round(_record_json(_Ctx(build, ext, build_order(build)), record, table, (), node)))


def _zero_json(build: str, ext: str, table: str | None, name: str, node: bool = False) -> dict:
    """The JSON of a zero record (``60`` W25-5 step 1): what a key the JSON leaves out stays."""
    return json.loads(_zero_text(build, ext, table, name, node))


class _Filler:
    """``60`` W25-5: a JSON written by hand may leave keys out; they keep the zero record's value. A key the
    table does not have is refused."""

    def __init__(self, build: str, ext: str) -> None:
        self.build, self.ext = build, ext
        self.refusals: list = []

    def refuse(self, path, reason, message) -> None:
        self.refusals.append(_refused(path, reason, message))

    def record(self, table: str | None, name: str, new, path: tuple, node: bool = False) -> dict:
        zero = _zero_json(self.build, self.ext, table, name, node)
        if not isinstance(new, dict):
            self.refuse(path, "bad_json", "a record is an object")
            return zero
        view = _view(self.build, self.ext, table, name)
        curves = {curve.name: curve for curve in view.curves}
        own = curves.get("")  # the record is itself a curve
        extra = ("stored_count",) if own is not None and own.parts.get("count") is not None else ()
        for key in new:
            if key not in zero and key not in extra:
                self.refuse(path + (key,), "unknown_name", f"{table or name} has no leaf {key!r} on {self.build}")
        curve_pairs = {id(c.pair) for c in view.curves}
        arrays = {pair.name: pair for pair in view.pairs
                  if pair.slot.field.is_record_array and id(pair) not in curve_pairs}
        out = {}
        for key, base in zero.items():
            if key not in new:
                out[key] = base
            elif key == UNNAMED:
                out[key] = self.unnamed(view, base, new[key], path + (key,))
            elif key in curves:
                out[key] = self.curve(curves[key], base, new[key], path + (key,))
            elif key in arrays:
                out[key] = self.elements(arrays[key], new[key], path + (key,))
            elif own is not None and key in CURVE_KEYS:
                out[key] = self.curve_part(own, key, base, new[key], path + (key,))
            else:
                out[key] = new[key]
        for key in extra:
            if key in new:
                out[key] = new[key]
        return out

    def elements(self, pair: _Pair, new, path: tuple, node: bool = False) -> list:
        if not isinstance(new, list):
            self.refuse(path, "bad_json", "a list")
            return []
        return [self.record(pair.element, pair.slot.field.element, item, path + (i,), node)
                for i, item in enumerate(new)]

    def curve(self, curve: _Curve, base: dict, new, path: tuple) -> dict:
        if not isinstance(new, dict):
            self.refuse(path, "bad_json", "a curve is an object")
            return base
        for key in new:
            if key not in CURVE_KEYS:
                self.refuse(path + (key,), "unknown_name", f"a curve has no {key!r}")
        out = dict(base)
        for key in CURVE_KEYS:
            if key in new:
                out[key] = self.curve_part(curve, key, base.get(key), new[key], path + (key,))
        return out

    def curve_part(self, curve: _Curve, key: str, base, new, path: tuple):
        if key == "nodes":
            return self.elements(curve.pair, new, path, node=True)
        if key == "scalar" and isinstance(base, dict) and isinstance(new, dict):
            for name in new:
                if name not in base:
                    self.refuse(path + (name,), "unknown_name", f"a curve's scalar has no {name!r}")
            return {name: new.get(name, value) for name, value in base.items()}
        return new

    def unnamed(self, view: _View, base: dict, new, path: tuple) -> dict:
        if not isinstance(new, dict):
            self.refuse(path, "bad_json", "unnamed words are an object")
            return base
        named = bytearray(view.size)
        for leaf in list(view.plain) + [part for curve in view.curves for part in curve.parts.values()]:
            named[leaf.offset:leaf.offset + leaf.size] = b"\1" * leaf.size
        for pair in view.pairs:
            named[pair.slot.offset:pair.slot.offset + 8] = b"\1" * 8
        for key in new:
            if key in base:
                continue
            try:
                offset = int(key, 16)
            except ValueError:
                offset = None
            if offset is not None and 0 <= offset < view.size and any(named[offset:offset + 4]):
                self.refuse(path + (key,), "conflicting_values", "an unnamed word over a named leaf or pair")
            else:
                self.refuse(path + (key,), "unknown_name", "no unnamed word starts here")
        return {key: new.get(key, value) for key, value in base.items()}


def _content(block: L.Block, build: str, ext: str, cache: dict):
    """The bytes a block stands for, with its element records' pairs replaced by their blocks' contents (so two
    blocks are equal when the file would store them alike)."""
    if block.elements is None:
        return bytes(block.data or b"")
    key = cache.get(id(block))
    if key is None:
        items = []
        for element in block.elements:
            raw = bytearray(element.raw)
            children = []
            for slot in L.pair_slots(build, ext, element.name):
                raw[slot.offset:slot.offset + 8] = bytes(8)
                value = element.pairs.get(slot.offset)
                if value is None or value.block is None or value.block.is_empty():
                    children.append(None)
                else:
                    children.append((slot.field.kind, slot.field.element, value.pair_size(),
                                     _content(value.block, build, ext, cache)))
            items.append((element.name, bytes(raw), tuple(children)))
        key = (tuple(items), bytes(block.extra))
        cache[id(block)] = key
    return key


class _Pooler(L._Walk):
    """W25-3: in the writer's order, a pair whose block has the same bytes and element type as an earlier pair's
    takes that block."""

    def __init__(self, tree: L.SnoTree) -> None:
        super().__init__(tree.build, tree.ext)
        self.blocks: dict = {}
        self.placed: set = set()
        self.cache: dict = {}
        self.pooled = 0

    def slot(self, record: L.Record, pos: int, slot: L.PairSlot, path: str) -> None:
        value = record.pairs.get(slot.offset)
        if value is None or value.block is None or value.block.is_empty():
            return
        block = value.block
        if id(block) in self.placed or value.alias_of is not None:
            return
        key = (slot.field.kind, slot.field.element, value.pair_size(),
               _content(block, self.build, self.ext, self.cache))
        earlier = self.blocks.get(key)
        if earlier is not None:
            record.pairs[slot.offset] = L.PairValue(earlier, shared=True)
            self.pooled += 1
            return
        self.blocks[key] = block
        self.placed.add(id(block))
        for element in block.elements or ():
            self.record(element, 0, path)


def _apply_shared(filler: _Filler, tree: L.SnoTree, table: str | None, record_json: dict, shared) -> list:
    """W25-3: each listed pair gets the block of its ``first_path``. ``61`` AI4: a pair of another element type
    keeps its own block (built from its JSON) and points at ``first_path``'s; the two must have the same bytes, which
    the read-back of the written file checks. Returns those entries: ``(k, path, first)``."""
    from .json_import import get_path, path_text

    across = []
    if shared is None:
        return across
    if not isinstance(shared, list):
        filler.refuse((SHARED,), "bad_json", "a list of [path, first_path]")
        return across
    sites = {path_text(where): (where, record, pair)
             for where, record, pair, _, _ in _sites(tree.build, tree.ext, tree.top, table, ("record",))}
    for k, entry in enumerate(shared):
        where = (SHARED, k)
        if not (isinstance(entry, list) and len(entry) == 2 and all(isinstance(p, str) for p in entry)):
            filler.refuse(where, "bad_json", "an entry is [path, first_path]")
            continue
        found = [sites.get(p) for p in entry]
        if None in found:
            filler.refuse(where, "bad_json", f"no pair at {entry[found.index(None)]!r}")
            continue
        (path, record, pair), (first, first_record, first_pair) = found
        value = first_record.pairs.get(first_pair.slot.offset)
        if value is None or value.block is None or value.block.is_empty():
            filler.refuse(where, "bad_json", f"{entry[1]} holds no block")
            continue
        if (pair.slot.field.kind, pair.slot.field.element) != (first_pair.slot.field.kind,
                                                                first_pair.slot.field.element):
            own = record.pairs.get(pair.slot.offset)
            size = own.pair_size() if own is not None and own.block is not None else 0
            if size != value.pair_size():
                filler.refuse(where, "conflicting_values", f"{entry[0]} ({size} bytes) and {entry[1]} "
                                                           f"({value.pair_size()} bytes) hold different bytes")
                continue
            record.pairs[pair.slot.offset] = L.PairValue(own.block, shared=True, alias_of=value.block)
            across.append((k, path, first))
            continue
        if not _same(get_path(record_json, path[1:]), get_path(record_json, first[1:])):
            filler.refuse(where, "conflicting_values", f"{entry[0]} shares the block of {entry[1]} but its JSON "
                                                       "differs")
            continue
        record.pairs[pair.slot.offset] = L.PairValue(value.block, shared=True)
    return across


def _word(header: dict, key: str, default, filler: _Filler, signed: bool = False):
    value = header.get(key, default)
    if value is None:
        return default
    low = -(1 << 31) if signed else 0
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value < 1 << 32:
        filler.refuse(("header", key), "value_out_of_range", "a 32-bit word")
        return default
    return value


def create_result(document: dict, ext: str, build: str | None = None, *, pool: bool = True):
    """``60`` W25-5: ``(bytes, warnings)`` of a file built from its JSON alone. Raises ``json_import.Refused``
    (``refusals`` holds every refusal) on a leaf the table does not have, an unnamed word over a named one or a
    value out of range; nothing is written then."""
    from .json_import import Refused, _close, _json_round, get_path
    from .sno import SNO_SIGNATURE, expected_version

    ext = ext.lower().lstrip(".")
    if not isinstance(document, dict):
        raise Refused((), "bad_json", "an object")
    unknown = [key for key in document if key not in TOP_KEYS]
    if unknown:
        raise Refused((unknown[0],), "unknown_name", f"the JSON has {', '.join(TOP_KEYS)}")
    given = document.get("build")
    if build is None:
        build = given
    if build not in ("ps3", "switch", "x360"):
        raise Refused(("build",), "bad_json", "the build (ps3, switch or x360) is not given")
    if given is not None and given != build:
        raise Refused(("build",), "conflicting_values", f"the JSON is of {given}, not {build}")
    if document.get("group", ext) != ext:
        raise Refused(("group",), "conflicting_values", f"the JSON is of .{document.get('group')}, not .{ext}")
    if (build, ext) not in _rows():
        raise Refused(("group",), "unsupported_value", f".{ext} has no rows in {LAYOUT_TABLE} on {build}")
    header = document.get("header") or {}
    if not isinstance(header, dict):
        raise Refused(("header",), "bad_json", "an object")
    order = build_order(build)
    filler = _Filler(build, ext)
    warnings = []
    # W25-4: the version of the type on the build; revision and source hash from the JSON (1 and 0 without)
    version = expected_version(ext, build)
    stated = _word(header, "version", None, filler)
    if version is None:
        version = stated
        if version is None:
            filler.refuse(("header", "version"), "bad_json", f"the version tables have no .{ext} on {build}: "
                                                             "give header.version")
            version = 0
    elif stated is not None and stated != version:
        warnings.append(f"version: header.version {stated} is not the {build} version {version} of .{ext}; "
                        f"{version} is written")
    revision = _word(header, "revision", 1, filler)
    source_hash = _word(header, "source_hash", 0, filler)
    handle = _word(header, "handle", None, filler, signed=True)
    byte_order = header.get("byte_order")
    if byte_order is not None and byte_order != ("big" if order == ">" else "little"):
        filler.refuse(("header", "byte_order"), "conflicting_values", f"{build} files are not {byte_order}-endian")
    top = L.top_record(build, ext)
    table = _top_table_record(build, ext)
    tree = L.SnoTree(build, ext, struct.pack(order + "4I", SNO_SIGNATURE, version, revision, source_hash),
                     L.new_record(build, ext, top))
    record_json = filler.record(table, top, document.get("record", {}), ("record",))
    zero = _zero_json(build, ext, table, top)
    # W25-4: the handle is the record's first word (``tHeader.snoHandle``)
    if OWN_HANDLE in record_json:
        own = record_json[OWN_HANDLE]
        if handle is None and isinstance(own, int) and not isinstance(own, bool):
            handle = own
        elif own != zero[OWN_HANDLE] and own != handle:
            filler.refuse(("record", OWN_HANDLE), "conflicting_values", f"{own} is not header.handle {handle}")
        record_json[OWN_HANDLE] = zero[OWN_HANDLE]
    enc = _Enc(tree, ext)
    enc.creating = True
    enc.refusals.extend(filler.refusals)
    filler.refusals = enc.refusals
    if not enc.refusals:
        _guard(enc, _encode_record, enc, tree.top, table, zero, record_json, ("record",))
    if handle is not None:
        struct.pack_into(order + ("i" if handle < 0 else "I"), tree.top.raw, 0, handle)
    across = []
    if not enc.refusals:
        across = _apply_shared(filler, tree, table, record_json, document.get(SHARED))
    if not enc.refusals and pool and ext in ALWAYS_SHARED[build]:
        _Pooler(tree).run(tree.top, L.BODY)
    data = None
    if not enc.refusals:
        try:
            data = L.write_tree(tree, derive=True)
            again = _json_round(decode(data, ext, build))
        except FormatError as exc:
            enc.refusals.append(_refused((), "value_out_of_range", f"the written file does not read back: {exc}"))
        else:
            for k, path, first in across:  # ``61`` AI4: the pair reads back as its own JSON from the shared bytes
                if not _close(_json_round(get_path(record_json, path[1:])), get_path(again, path)):
                    enc.refusals.append(_refused((SHARED, k), "conflicting_values",
                                                 f"{_path_text(path)} and {_path_text(first)} hold different bytes"))
            for path, value in enc.checks:  # W22-5
                try:
                    found = get_path(again, path)
                except (KeyError, IndexError, TypeError):
                    enc.refusals.append(_refused(path, "derived_field", "not in the written file's export"))
                    continue
                if _close(found, value):
                    continue
                record, offset = enc.placed.get(path, (None, None))
                if record is not None and any(link.count_offset == offset for link in
                                              L.count_links(build, ext, record.name)):
                    # W14-20: a count word follows its list (as ``encode`` does)
                    warnings.append(f"derived: {_path_text(path)} is {json.dumps(found)}, the count of its list")
                    continue
                enc.refusals.append(_refused(path, "value_out_of_range", f"reads back as {json.dumps(found)}"))
    if enc.refusals:
        error = enc.refusals[0]
        error.refusals = list(enc.refusals)
        raise error
    return data, warnings + enc.warnings


def create(document: dict, ext: str, build: str | None = None, *, pool: bool = True) -> bytes:
    """``60`` W25-5: the bytes of a file of ``ext`` built from its JSON alone."""
    return create_result(document, ext, build, pool=pool)[0]


# --- adding a new file (W25-6, W25-8) ---------------------------------------------------------------------

class NewFile(tuple):
    """``60`` W25-6: what ``add_sno_file`` returns (``(archive, index_files)``, ``identity``), with ``data`` (the
    file as added), ``links`` (``links_of``), ``depends`` (the SNO targets that resolve in the TOCs: its Prefetch
    dependencies), ``unresolved`` (SNO links no TOC lists) and ``gbids`` (GBID links: the TOCs cannot resolve
    them, W25-8), and ``warnings`` of ``create``."""

    def __new__(cls, added, data, links, depends, unresolved, gbids, warnings):
        self = super().__new__(cls, tuple(added))
        self.identity = added.identity
        self.data, self.links, self.depends = data, links, depends
        self.unresolved, self.gbids, self.warnings = unresolved, gbids, warnings
        return self

    @property
    def archive(self):
        return self[0]

    @property
    def index_files(self) -> dict:
        return self[1]

    @property
    def handle(self):
        return self.identity.handle if self.identity is not None else None


def add_new(archive, name: str, document: dict, *, tocs, prefetch=None, build: str | None = None,
            pool: bool = True, pack: str | None = None, out=None, jobs: int | None = 1,
            segment: bool = False) -> NewFile:
    """``60`` W25-6: build ``document`` with ``create``, add it to ``archive`` as ``name`` with a new handle and
    its index entries (``index_write.add_sno_file``, W16-12), and give it (when ``prefetch`` is given) a Prefetch
    parent whose dependencies are the SNO targets of its links that resolve in ``tocs`` (W24-4: a TOC lists
    ``(group, handle)``). Unresolved links are not refused (W25-8)."""
    from . import index_write as IW
    from . import links as LK

    cpk = IW._read(archive)
    archive_build = cpk.header.build
    if build is not None and build != archive_build:
        raise FormatError(f"the archive is of {archive_build}, not {build}")
    _, _, ext, _ = IW.split_archive_name(name)
    if not ext:
        raise FormatError(f"{name!r} has no extension")
    data, warnings = create_result(document, ext, archive_build, pool=pool)
    links = LK.links_of(data, ext, archive_build)
    listed = set()
    for toc in IW._named_tocs(tocs, archive_build).values():
        for entry in toc.entries:
            listed.add((entry.group, _signed(entry.handle)))
    depends, unresolved, gbids = [], [], []
    for link in links:
        kind, group, value = link.target
        if kind != "sno":
            gbids.append(link)
        elif (group, _signed(value)) in listed:
            if (group, value) not in depends:
                depends.append((group, value))
        else:
            unresolved.append(link)
    added = IW.add_sno_file(cpk, name, data, tocs, prefetch, depends if prefetch is not None else (), pack=pack,
                            out=out, jobs=jobs, segment=segment)
    stamped = added.identity.stamp(data) if added.identity is not None else data
    return NewFile(added, stamped, links, depends, unresolved, gbids, warnings)


def _signed(value: int) -> int:
    value &= 0xFFFFFFFF
    return value - (1 << 32) if value >= 1 << 31 else value
