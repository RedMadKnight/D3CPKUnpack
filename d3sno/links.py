"""Stage 24 (``58_stage24_links.md``): the graph of links between game files (W24-1 to W24-9).

``links_of`` reads the links of one file through the generic tree (``sno_layout``) and the rows of
``data/f58_layouts.tsv`` (as the effects module of stage 23 names every leaf): link leaves of type ``SNO``,
``SNOName`` and ``GBID`` with a ``sno_group`` (W24-1), the elements of ``ARRAY:SNO`` / ``ARRAY:SNOName`` /
``ARRAY:GBID`` rows, and the tag records of every tag map whose tag type has a ``sno_group`` in
``cov_tag_types.tsv`` (W24-2). −1, 0 and the affix filler 0x5C000000 are no links (W24-3).

``build_index`` reads a whole game folder into a ``LinkIndex`` (W24-6): nodes ``("sno", group, handle)`` and
``("gbid", balance type, value)``, edges ``(source, path, target, kind)``, resolved against the CoreTOC and the
rows of the ``.gam`` files (W24-4). The queries are those of W24-7; ``to_json`` gives W24-8.

The output names the user's game files: it stays on the user's computer (``40``).
"""

from __future__ import annotations

import os
import struct
from collections import deque
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from . import effects as E
from . import sno_layout as L
from .binary import ByteView, FormatError, Pair, build_order
from .sno import group_by_extension, group_by_id, parse_header
from .tags import OWNER_TABLES, tag_table, tag_types

__all__ = ["Link", "Edge", "Reach", "LinkIndex", "LinkRefused", "links_of", "build_index", "link_types",
           "node_id", "LINK_TYPES", "FILLER", "FILLERS", "APP_DECODED"]

LINK_TYPES = ("SNO", "SNOName", "GBID")  # W24-1, and their ``ARRAY:`` forms
FILLER = 0x5C000000  # W24-3: the affix filler of ``ItemSpecifierData`` (Switch)
# ``59`` AH2: the filler of each build ("sil" and a zero on x360); PS3 has none (its affix rows are open)
FILLERS = {"switch": FILLER, "x360": 0x73696C00}
# ``59`` AH1: ``.app`` rows not read; the appearance decoder gives these links on every build
APP_DECODED = ("snoMaterial", "tMatTex.snoTex")
AFFIX_COUNT = "nNumAffixes"
KINDS = ("field", "array", "tag")
# W24-2: the tag table of a group's tag maps (``tags.OWNER_TABLES``; an animation set's maps hold animation tags)
MAP_TABLES = {"acr": OWNER_TABLES["actor"], "mon": OWNER_TABLES["monster"], "ani": OWNER_TABLES["anim"],
              "ans": OWNER_TABLES["anim"], "mrk": OWNER_TABLES["marker"], "app": OWNER_TABLES["material"],
              "mat": OWNER_TABLES["material"]}


class LinkRefused(Exception):
    """W24-8: a refused run (the rules of the export, W18-1)."""


@dataclass(frozen=True)
class Link:
    """W24-9: one link value of a file, not resolved. ``target`` is ``("sno", group id, handle)`` or
    ``("gbid", balance type, value)``."""

    path: str
    kind: str  # field, array or tag
    target: tuple


# --- where links are (W24-1 to W24-3) -------------------------------------------------------------------

@lru_cache(maxsize=None)
def link_types() -> tuple[str, ...]:
    """The extensions with a top record in the layout table (W24-1: 59 types)."""
    return tuple(sorted({ext for _, ext in E._rows()}))


def _link_type(leaf) -> str | None:
    """``SNO``, ``SNOName`` or ``GBID`` for a link row (its own type, or its ``ARRAY:`` element), else None.
    A row without a ``sno_group`` names no group and is no link (``tHeader.snoHandle``, the ``snoname`` rows)."""
    if leaf is None or leaf.group is None or leaf.offset is None:
        return None
    kind = leaf.element if leaf.is_array else leaf.type
    return kind if kind in LINK_TYPES else None


def _target(link_type: str, group: int, value: int) -> tuple:
    return ("gbid", group, value) if link_type == "GBID" else ("sno", group, value)


@lru_cache(maxsize=None)
def _tag_groups() -> dict:
    """W24-2: (table, tag) -> SNO group, for the tags whose tag type has a ``sno_group``; and tag -> group for a
    tag id whose rows in every table agree."""
    by_type = {name: int(row["sno_group"]) for name, row in tag_types().items() if row.get("sno_group")}
    own: dict = {}
    anywhere: dict = {}
    for (table, tag), info in tag_table().items():
        group = by_type.get(info.value_type)
        own[(table, tag)] = (group, info.name or None)
        anywhere.setdefault(tag, set()).add((group, info.name or None))
    single = {tag: next(iter(found)) for tag, found in anywhere.items() if len(found) == 1}
    return {"own": own, "single": single}


def _tag_group(ext: str, tag: int):
    """The SNO group and name of a tag of a map of ``ext``: from the group's own tag table, else from the one
    meaning the tag has in every table."""
    tables = _tag_groups()
    table = MAP_TABLES.get(ext)
    if table is not None and (table, tag) in tables["own"]:
        return tables["own"][(table, tag)]
    return tables["single"].get(tag, (None, None))


def _path_text(path) -> str:
    out = ""
    for part in path:
        out += f"[{part}]" if isinstance(part, int) else ("." if out else "") + str(part)
    return out


def _is_filler(view, named, value: int, build: str) -> bool:
    """W24-3, ``59`` AH2: an affix slot ``<prefix>.gbidAffixes[k]`` of an ``ItemSpecifierData`` (a record with
    ``<prefix>.nNumAffixes``) that holds the build's filler, whatever the count holds."""
    if build not in FILLERS or value != FILLERS[build] or not named.name.endswith("]"):
        return False
    head, _, index = named.name[:-1].rpartition("[")
    prefix = head.rpartition(".")[0]
    count_name = f"{prefix}.{AFFIX_COUNT}" if prefix else AFFIX_COUNT
    return index.isdigit() and any(n.name == count_name for n in view.plain)


class _Reader:
    def __init__(self, build: str, ext: str) -> None:
        self.build, self.ext = build, ext
        self.order = build_order(build)
        self.out: list[Link] = []
        self.visited: set[int] = set()

    def add(self, path: tuple, kind: str, target: tuple) -> None:
        if target[2] in (-1, 0):  # W24-3: empty
            return
        self.out.append(Link(_path_text(path), kind, target))

    def record(self, record: L.Record, table: str | None, path: tuple) -> None:
        view = E._view(self.build, self.ext, table, record.name)
        named_leaves = list(view.plain) + [n for curve in view.curves for n in curve.parts.values()]
        for named in sorted(named_leaves, key=lambda n: n.offset):
            link_type = _link_type(named.leaf)
            if link_type is None or named.size != 4:
                continue
            if self.ext == "app" and named.name in APP_DECODED:  # ``59`` AH1
                continue
            value = struct.unpack_from(self.order + "i", record.raw, named.offset)[0]
            if link_type == "GBID" and _is_filler(view, named, value, self.build):
                continue
            self.add(path + (named.name,), "field", _target(link_type, named.leaf.group, value))
        rows = E._by_record(self.build, self.ext).get(table, ()) if table else ()
        curves = {id(curve.pair): curve for curve in view.curves}
        for pair in view.pairs:
            self.pair(record, pair, rows, curves.get(id(pair)), path)

    def pair(self, record: L.Record, pair, rows, curve, path: tuple) -> None:
        slot = pair.slot
        value = record.pairs.get(slot.offset)
        if value is None or value.block is None:
            return
        block = value.block
        if value.shared or id(block) in self.visited:  # a block is read once (Q1013)
            return
        self.visited.add(id(block))
        f = slot.field
        if curve is not None:
            where = path + ((curve.name,) if curve.name else ()) + ("nodes",)
        else:
            where = path + (pair.name,)
        if f.kind == "tagmap":
            self.tag_map(bytes(block.data or b""), where)
        elif block.elements is not None:
            for i, element in enumerate(block.elements):
                self.record(element, pair.element, where + (i,))
        elif f.kind == "array":
            row = next((r for r in rows if r.is_array and r.pair_offset == slot.offset and r.name == pair.name), None)
            link_type = _link_type(row)
            data = bytes(block.data or b"")
            if link_type is None:
                return
            for i in range(len(data) // 4):
                item = struct.unpack_from(self.order + "i", data, 4 * i)[0]
                if link_type == "GBID" and item == FILLERS.get(self.build):
                    continue
                self.add(where + (i,), "array", _target(link_type, row.group, item))

    def app_materials(self, data: bytes) -> None:
        """``59`` AH1: the material of each look and the texture slots of its block, from the appearance decoder
        (``material.read_app_materials``). A file whose materials do not read has none of these links, as the
        appearance decoder keeps the mesh (Q1025)."""
        from .material import read_app_materials

        try:
            found = read_app_materials(ByteView(data, self.order), self.build)
        except (FormatError, struct.error):
            return
        mat, tex = group_by_extension("mat").id, group_by_extension("tex").id
        for i, material in enumerate(found.materials):
            for j, (block, handle) in enumerate(material.per_look):
                where = ("materials", i, "looks", j)
                self.add(where + ("material",), "field", ("sno", mat, handle))
                for k, slot in enumerate(block.textures):
                    self.add(where + ("textures", k), "field", ("sno", tex, slot.handle))

    def tag_map(self, data: bytes, path: tuple) -> None:
        from .power import read_tag_map

        if len(data) < 4:
            return
        found = read_tag_map(ByteView(data, self.order), Pair(-0x10, len(data)), _path_text(path), lenient=True)
        for record in found.records:
            if not isinstance(record.value, int):
                continue
            group, name = _tag_group(self.ext, record.tag)
            if group is None:
                continue
            self.add(path + (name or f"0x{record.tag:X}",), "tag", ("sno", group, record.value))


def links_of(data: bytes, ext: str, build: str | None = None) -> list[Link]:
    """W24-9: the links of one file, as ``Link(path, kind, target)``, in walk order, with no resolving. A type
    with no top record in the table has none. Raises ``FormatError`` on a file the generic tree cannot read."""
    ext = ext.lower().lstrip(".")
    if ext not in link_types():
        return []
    tree = L.parse_tree(data, ext, build)
    reader = _Reader(tree.build, ext)
    reader.record(tree.top, E._top_table_record(tree.build, ext), ("record",))
    if ext == "app":
        reader.app_materials(data)
    return reader.out


# --- the index (W24-6, W24-7) ---------------------------------------------------------------------------

def node_id(node: tuple) -> str:
    """A node's text: ``sno:<ext>:<handle>`` or ``gbid:<type>:<value>``."""
    if node[0] == "sno":
        group = group_by_id(node[1])
        return f"sno:{group.extension if group else node[1]}:{node[2]}"
    return f"gbid:{node[1]}:{node[2]}"


def _parse_id(text: str):
    kind, _, rest = text.partition(":")
    group, _, value = rest.partition(":")
    try:
        number = int(value)
    except ValueError:
        return None
    if kind == "sno":
        found = group_by_extension(group) if not group.lstrip("-").isdigit() else group_by_id(int(group))
        return ("sno", found.id, number) if found else None
    if kind == "gbid" and group.lstrip("-").isdigit():
        return ("gbid", int(group), number)
    return None


@dataclass(frozen=True)
class Edge:
    """W24-6: one link value. ``other_type``: a GBID found only in a table of another balance type (W24-4)."""

    source: tuple
    path: str
    target: tuple
    kind: str
    other_type: int | None = None

    def key(self):
        return (_order(self.source), self.path, _order(self.target), self.kind)


@dataclass(frozen=True)
class Reach:
    """W24-7: a node reached by ``needs`` / ``needed_by``, with its depth and the edges that first reached it."""

    node: tuple
    depth: int
    via: tuple  # of Edge, from the start


def _order(node: tuple):
    return (node[0], node[1], node[2])


@dataclass
class _Node:
    kind: str
    group: int
    value: int
    name: str | None = None
    archives: list = field(default_factory=list)
    unread: str | None = None
    read: bool = False
    resolved: bool = False
    gam: int | None = None  # a GBID row: the handle of its ``.gam``


class LinkIndex:
    """W24-6, W24-7: the nodes and edges of a game folder, with the queries."""

    def __init__(self, build: str | None) -> None:
        self.build = build
        self.nodes: dict[tuple, _Node] = {}
        self.edges: list[Edge] = []
        self._out: dict[tuple, list[Edge]] = {}
        self._in: dict[tuple, list[Edge]] = {}
        self.unread_files: list[dict] = []  # files that could not be read and have no handle
        self.files_read = 0
        self.toc: dict[tuple[int, int], str] = {}  # (group, handle) -> CoreTOC name (W24-4)

    # --- building ---------------------------------------------------------------------------------------
    def node(self, key: tuple) -> _Node:
        found = self.nodes.get(key)
        if found is None:
            found = self.nodes[key] = _Node(key[0], key[1], key[2])
            if key[0] == "sno" and (key[1], key[2]) in self.toc:  # W24-4: listed in the index
                found.name, found.resolved = self.toc[(key[1], key[2])], True
        return found

    def add_edge(self, edge: Edge) -> None:
        self.node(edge.source)
        self.node(edge.target)
        self.edges.append(edge)
        self._out.setdefault(edge.source, []).append(edge)
        self._in.setdefault(edge.target, []).append(edge)

    def _finish(self) -> None:
        self.edges.sort(key=Edge.key)
        for table in (self._out, self._in):
            for edges in table.values():
                edges.sort(key=Edge.key)

    # --- queries (W24-7) --------------------------------------------------------------------------------
    def _key(self, node) -> tuple:
        if isinstance(node, tuple):
            return node
        if isinstance(node, _Node):
            return (node.kind, node.group, node.value)
        found = self.find(node)
        if found is None:
            raise KeyError(node)
        return found

    def uses(self, node) -> list[Edge]:
        """The edges out of ``node``."""
        return list(self._out.get(self._key(node), ()))

    def used_by(self, node) -> list[Edge]:
        """The edges into ``node``."""
        return list(self._in.get(self._key(node), ()))

    def _walk(self, start, depth, kinds, forward: bool) -> list[Reach]:
        start = self._key(start)
        table = self._out if forward else self._in
        kinds = None if kinds is None else set(kinds)
        seen = {start}
        out: list[Reach] = []
        queue = deque([(start, 0, ())])
        while queue:
            node, level, via = queue.popleft()
            if depth is not None and level >= depth:
                continue
            for edge in table.get(node, ()):
                if kinds is not None and edge.kind not in kinds:
                    continue
                other = edge.target if forward else edge.source
                if other in seen:  # a cycle, or a node already reached: visited once
                    continue
                seen.add(other)
                reach = Reach(other, level + 1, via + (edge,))
                out.append(reach)
                queue.append((other, level + 1, reach.via))
        return out

    def needs(self, node, *, depth: int | None = None, kinds=None) -> list[Reach]:
        """Every node reachable from ``node`` (breadth first), with the edges that first reached each."""
        return self._walk(node, depth, kinds, True)

    def needed_by(self, node, *, depth: int | None = None, kinds=None) -> list[Reach]:
        """The same, backwards."""
        return self._walk(node, depth, kinds, False)

    def is_resolved(self, node: tuple) -> bool:
        found = self.nodes.get(node)
        return bool(found and found.resolved)

    def unresolved(self) -> list[Edge]:
        """Every edge whose target does not resolve."""
        return [edge for edge in self.edges if not self.is_resolved(edge.target)]

    def find_all(self, name_or_handle, group=None) -> list[tuple]:
        """Every node with that CoreTOC (or row) name, case-insensitive, or that handle; ``group`` an extension,
        a group id or ``"gbid"``."""
        if isinstance(name_or_handle, tuple):
            return [name_or_handle] if name_or_handle in self.nodes else []
        if isinstance(name_or_handle, str):
            parsed = _parse_id(name_or_handle)
            if parsed is not None:
                return [parsed] if parsed in self.nodes else []
        want_kind, want_group = None, None
        if group is not None:
            if group == "gbid":
                want_kind = "gbid"
            else:
                found = group_by_id(group) if isinstance(group, int) else group_by_extension(str(group))
                if found is None:
                    raise ValueError(f"{group!r} is not an SNO group")
                want_kind, want_group = "sno", found.id
        number = None
        if isinstance(name_or_handle, int):
            number = name_or_handle
        elif isinstance(name_or_handle, str) and name_or_handle.lstrip("-").isdigit():
            number = int(name_or_handle)
        text = str(name_or_handle).lower()
        out = []
        for key in sorted(self.nodes, key=_order):
            node = self.nodes[key]
            if want_kind is not None and key[0] != want_kind:
                continue
            if want_group is not None and key[1] != want_group:
                continue
            if (number is not None and key[2] == number) or (node.name is not None and node.name.lower() == text):
                out.append(key)
        return out

    def find(self, name_or_handle, group=None) -> tuple | None:
        """A node by CoreTOC name (case-insensitive) or handle; None when there is none; ``ValueError`` when
        several groups have it (give ``group``)."""
        found = self.find_all(name_or_handle, group)
        if len(found) > 1:
            raise ValueError(f"{name_or_handle!r} names {len(found)} nodes: "
                             f"{', '.join(node_id(k) for k in found[:8])}; give the group")
        return found[0] if found else None

    # --- output (W24-8) ---------------------------------------------------------------------------------
    def counts(self) -> dict:
        out = {"files_read": self.files_read, "unread": sum(1 for n in self.nodes.values() if n.unread)
               + len(self.unread_files), "nodes": len(self.nodes), "edges": len(self.edges),
               "sno_resolved": 0, "sno_unresolved": 0, "gbid_resolved": 0, "gbid_unresolved": 0,
               "gbid_other_type": 0}
        for edge in self.edges:
            state = "resolved" if self.is_resolved(edge.target) else "unresolved"
            out[f"{edge.target[0]}_{state}"] += 1
            if edge.other_type is not None:
                out["gbid_other_type"] += 1
        out["by_kind"] = {kind: sum(1 for e in self.edges if e.kind == kind) for kind in KINDS}
        return out

    def node_json(self, key: tuple) -> dict:
        node = self.nodes[key]
        if key[0] == "sno":
            group = group_by_id(key[1])
            out = {"id": node_id(key), "kind": "sno", "group": group.extension if group else str(key[1]),
                   "group_id": key[1], "handle": key[2], "name": node.name, "archives": list(node.archives),
                   "read": node.read, "resolved": node.resolved}
        else:
            out = {"id": node_id(key), "kind": "gbid", "type": key[1], "value": key[2], "name": node.name,
                   "gam": node.gam, "resolved": node.resolved}
        if node.unread:
            out["unread"] = node.unread
        return out

    @staticmethod
    def edge_json(edge: Edge) -> dict:
        out = {"source": node_id(edge.source), "path": edge.path, "target": node_id(edge.target), "kind": edge.kind}
        if edge.other_type is not None:
            out["other_type"] = edge.other_type
        return out

    def to_json(self) -> dict:
        """W24-8: nodes and edges in a stable order (group, handle, path)."""
        return {"build": self.build, "nodes": [self.node_json(k) for k in sorted(self.nodes, key=_order)],
                "edges": [self.edge_json(e) for e in self.edges],
                "unresolved": [self.edge_json(e) for e in self.unresolved()],
                "unread_files": list(self.unread_files), "counts": self.counts()}

    def summary(self) -> str:
        """The summary line of the command (the counts of W24-5)."""
        c = self.counts()
        return (f"{self.build}: {c['files_read']} files read, {c['unread']} unread; SNO links {c['sno_resolved']} "
                f"resolve / {c['sno_unresolved']} do not; GBID links {c['gbid_resolved']} resolve / "
                f"{c['gbid_unresolved']} do not ({c['gbid_other_type']} in a table of another type); "
                f"{c['nodes']} nodes, {c['edges']} edges")


# --- building the index from a game folder (W24-6) ------------------------------------------------------

def _ext_of(name: str) -> str:
    base = name.replace("/", "\\").rpartition("\\")[2]
    return base.rpartition(".")[2].lower() if "." in base else ""


def build_index(game_dir, *, build: str | None = None, archives=None) -> LinkIndex:
    """W24-6: read every file of the 59 types and every ``.gam`` of ``game_dir`` once per handle (the first copy
    in sorted archive-name order, as the export does), resolve every link (W24-4). ``archives``: base-name
    patterns, as the export's ``--archives``. Raises ``LinkRefused`` for a folder the export would refuse."""
    from . import archive
    from . import export as X
    from .balance import gbid, parse_balance

    game = Path(os.path.abspath(game_dir))
    if not game.is_dir():
        raise LinkRefused(f"{game_dir}: not a directory")
    if build is not None and build not in ("ps3", "switch", "x360"):
        raise LinkRefused(f"--build {build!r}: give ps3, switch or x360")
    run = X._Run(game, game, archives, [], False, False, build)
    try:
        try:
            run.open_archives()
            run.read_indexes()
        except X.ExportRefused as exc:
            raise LinkRefused(str(exc)) from None
        index = LinkIndex(run.build)
        index.unread_files.extend(run.unread)
        for (group, handle), name in run.names.items():
            index.toc[(group, handle - (1 << 32) if handle >= 1 << 31 else handle)] = name
        # the files of each name, copies in sorted archive order (locale rank first, V27-2)
        items: dict[tuple, list] = {}
        order: list[tuple] = []
        for stem, cpk in run.archives:
            for entry in cpk.files:
                if entry.name is None:
                    continue
                name = entry.name.partition("|")[0]
                ext = _ext_of(name)
                if group_by_extension(ext) is None:
                    continue
                key = (ext, name.lower())
                if key not in items:
                    items[key] = []
                    order.append(key)
                items[key].append(X._Copy(stem, cpk, entry))
        wanted = set(link_types())
        done: set[tuple] = set()
        gbids: dict[tuple, tuple] = {}  # (type, value) -> (name, gam handle)
        pending: list[tuple] = []  # (source, Link)
        for key in order:
            ext = key[0]
            copies = sorted(items[key], key=X._locale_rank)
            group = group_by_extension(ext)
            stems = sorted({c.archive for c in copies}, key=str.lower)
            if ext not in wanted and ext != "gam":
                handle = run.name_handles.get((group.id, key[1].replace("/", "\\").partition("\\")[2]
                                               .rpartition(".")[0]))
                if handle is None:
                    continue  # a file of a type with no links and no index name: no node (Q1015)
                signed = handle - (1 << 32) if handle >= 1 << 31 else handle
                node = index.node(("sno", group.id, signed))
                node.archives = sorted(set(node.archives) | set(stems), key=str.lower)
                continue
            data, copy, error = None, None, None
            for c in copies:
                try:
                    data, copy = c.cpk.read_file(c.entry), c
                    break
                except FormatError as exc:
                    error = error or str(exc)
            handle, file_build = None, None
            if data is not None:
                try:
                    file_build = archive.file_build(copy.cpk.header, data, ext)
                    handle = parse_header(data).handle
                except FormatError as exc:
                    error = str(exc)
            if handle is None:
                stem = key[1].replace("/", "\\").partition("\\")[2].rpartition(".")[0]
                handle = run.name_handles.get((group.id, stem))
                if handle is None:
                    index.unread_files.append({"file": copies[0].entry.name, "archive": copies[0].archive,
                                               "message": error or "no handle"})
                    continue
                handle = handle - (1 << 32) if handle >= 1 << 31 else handle
                node = index.node(("sno", group.id, handle))
                node.archives = sorted(set(node.archives) | set(stems), key=str.lower)
                node.unread = node.unread or (error or "not read")
                continue
            source = ("sno", group.id, handle)
            node = index.node(source)
            node.archives = sorted(set(node.archives) | set(stems), key=str.lower)
            if source in done:
                continue  # once per handle
            done.add(source)
            try:
                if ext == "gam":
                    table = parse_balance(data, file_build)
                    for row in table.rows:
                        if row.name:
                            gbids.setdefault((table.type, gbid(row.name)), (row.name, handle))
                    found = []
                else:
                    found = links_of(data, ext, file_build)
            except (FormatError, struct.error, ValueError, KeyError, IndexError) as exc:
                node.unread = str(exc) or type(exc).__name__
                continue
            node.read = True
            index.files_read += 1
            pending.extend((source, link) for link in found)
        # W24-4: resolving
        by_value: dict[int, list] = {}
        for (kind, value), row in sorted(gbids.items()):
            by_value.setdefault(value, []).append(kind)
            target = index.node(("gbid", kind, value))
            target.name, target.gam, target.resolved = row[0], row[1], True
        for source, link in pending:
            target, other = link.target, None
            if target[0] == "gbid" and (target[1], target[2]) not in gbids and by_value.get(target[2]):
                other = by_value[target[2]][0]
                target = ("gbid", other, target[2])
            index.add_edge(Edge(source, link.path, target, link.kind, other))
        index._finish()
        return index
    finally:
        run.close()
