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
           "schema", "enum_names", "FINDINGS"]

# ``57``: the corrected tables replace ``f56_*`` and ``f55_*`` (which stay in the package data, unused)
LAYOUT_TABLE = "f57_layouts.tsv"
ENUM_TABLE = "f57_enums.tsv"
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
    out: dict[tuple[str, str], list[Leaf]] = {}
    with open(DATA / LAYOUT_TABLE, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            offset = row["offset"].strip()
            group = row["sno_group"].strip()
            pair = (row.get("pair_offset") or "").strip()
            out.setdefault((row["build"], row["ext"]), []).append(Leaf(
                row["build"], row["ext"], row["record"], row["leaf"], int(offset) if offset else None,
                int(row["bytes"]), row["type"], row["unit"], row["enum"], row["flag_bits"],
                int(group) if group else None, row["status"], int(pair) if pair else None))
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
    return _Named(name, leaf.offset if offset is None else offset, leaf.size, code, enum, leaf.flags, leaf.group,
                  leaf)


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
            if part is None or (part == "count" and build == "ps3" and prefix):
                continue  # ``56`` AF3: a PS3 inline curve has no count; a PS3 path element (its own curve) has one
            parts[part] = named
            used.add(named.name)
        curves.append(_Curve(prefix.rstrip("."), pair, parts))
    plain = [named for named in good if named.name not in used]
    covered = bytearray(size)
    for named in good:
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
        if value in (-1, 0):
            return None
        group = group_by_id(named.group)
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


def decode_full(data: bytes, ext: str, build: str | None = None) -> Decoded:
    """W23-7 with what coverage reports (W23-2 to W23-6)."""
    ext = ext.lower().lstrip(".")
    if ext not in GROUPS:
        raise FormatError(f".{ext} is not a group of stage 23")
    tree = L.parse_tree(data, ext, build)
    build = tree.build
    header = parse_header(data, build)
    out = Decoded({})
    ctx = _Ctx(build, ext, build_order(build), out)
    body = _record_json(ctx, tree.top, _top_table_record(build, ext), ("record",))
    out.json = {"header": {"handle": header.handle, "version": header.version, "revision": header.revision,
                           "source_hash": header.source_hash,
                           "byte_order": "big" if header.order == ">" else "little"},
                "build": build, "group": ext, "record": body}
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
            group = group_by_id(leaf.group) if leaf.group is not None else None
            rows.append({"record": leaf.record, "leaf": leaf.name, "type": leaf.type, "unit": leaf.unit or None,
                         "enum": leaf.enum or None, "flags": leaf.flags or None,
                         "link_group": (group.extension if group else str(leaf.group)) if leaf.group is not None
                         else None, "offset": leaf.offset, "status": leaf.status})
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
    if named.code == "text":
        record.raw[named.offset:named.offset + named.size] = _text_bytes(new, old, named.size, path)
        if isinstance(new, str):
            enc.checks.append((path, new))
        return
    if named.code == "rgba":
        if not (isinstance(new, list) and len(new) == 4 and all(
                isinstance(c, int) and not isinstance(c, bool) and 0 <= c <= 255 for c in new)):
            raise _refused(path, "value_out_of_range", "a colour is [r, g, b, a], each 0 to 255")
        record.raw[named.offset:named.offset + 4] = bytes(new)
        enc.checks.append((path, new))
        return
    if named.enum:
        change = Change(path, old, new, Place("enum", names={n: v for v, n in enum_names(named.enum).items()}))
        value = _enum_value(change, None)
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
        enc.checks.append((path, new))
    record.raw[named.offset:named.offset + named.size] = _pack(named.code, value, path, enc.order)


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
    keys = ("loop", "distribution", "scalar", "nodes", "stored_count")
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
    if isinstance(old, dict):
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
