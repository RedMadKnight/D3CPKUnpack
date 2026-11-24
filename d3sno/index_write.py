"""Index-file writers and new-file helpers (``28_derived_values.md`` W14-1 to W14-6).

- W14-1: ``new_handle``, ``sno_identity`` (handle, CoreTOC entry, offset-0x10 word, archive path).
- W14-2/3: ``write_core_toc`` (ps3, switch and x360 forms), ``add_toc_entry``, ``write_x360_tocs``.
- W14-4: ``write_prefetch``, ``add_prefetch_parent``.
- W14-5: ``write_manifest``, ``write_sprite_table``, ``write_descriptor_sizes``, ``write_slo``.
- W14-6: ``add_file``, ``replace_file`` (``replace_files``) on ps3 / switch / x360 archives.
- W16-12 / W16-13: ``add_sno_file`` (archive entry, identity and index files in one step),
  ``x360_pack``.

Every writer accepts what the matching parser of ``index.py`` returns, so parse -> write gives
the same bytes for a file that follows the rules.
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from .binary import FormatError, build_order
from .index import (
    DESCRIPTOR_MAGIC, GROUP_SLOTS, MANIFEST_MAGIC, SPRITE_MAGIC, SWITCH_TOC_HEADER, CoreToc,
    DescriptorSize, Manifest, Prefetch, PrefetchParent, Sprite, SpriteTable, TocEntry,
)
from .sno import SNO_HEADER_SIZE, SNO_SIGNATURE, group_by_extension, group_by_id, hash33

__all__ = [
    "SLO_MAGIC", "X360_MAIN_TOCS", "SnoIdentity", "new_handle", "sno_identity", "write_core_toc",
    "add_toc_entry", "write_x360_tocs", "prefetch_bucket", "write_prefetch", "add_prefetch_parent",
    "write_manifest", "write_sprite_table", "write_descriptor_sizes", "write_slo", "add_file",
    "replace_file", "replace_files", "AddedFile", "split_archive_name", "x360_pack", "add_sno_file",
]

SLO_MAGIC = 0xACBF0004
PS3_NAME_FIELD = 128
DESCRIPTOR_NAME_FIELD = 64
DESCRIPTOR_WORDS = 256
X360_MAIN_TOCS = ("commontoc.dat", "act1toc.dat", "act2toc.dat", "act3toc.dat", "act4toc.dat",
                  "act5toc.dat")
CHILD_EXTENSIONS = ("nav", "can", "fsb")
_U32 = 0xFFFFFFFF


def _order(order_or_build: str) -> str:
    if order_or_build in ("<", ">"):
        return order_or_build
    return build_order(order_or_build)


def _u32(value: int) -> int:
    if not -0x80000000 <= value <= _U32:
        raise FormatError(f"value {value} does not fit 32 bits")
    return value & _U32


def _name_bytes(name: str) -> bytes:
    if "\0" in name:
        raise FormatError(f"NUL in name {name!r}")
    try:
        return name.encode("latin-1")
    except UnicodeEncodeError:
        raise FormatError(f"name {name!r} is not latin-1") from None


# --- W14-1 -----------------------------------------------------------------------------------

def _entries_of(toc) -> list[TocEntry]:
    if isinstance(toc, CoreToc):
        return toc.entries
    return [e if isinstance(e, TocEntry) else TocEntry(*e) for e in toc]


def new_handle(tocs) -> int:
    """W14-1: highest handle over every given TOC (and every stored highest-handle word) + 1.

    ``tocs`` is one CoreToc, one entry list, or an iterable of them."""
    if isinstance(tocs, CoreToc) or (isinstance(tocs, list) and tocs
                                     and isinstance(tocs[0], (TocEntry, tuple))):
        tocs = [tocs]
    highest = 0
    for toc in tocs:
        for entry in _entries_of(toc):
            highest = max(highest, entry.handle & _U32)
        if isinstance(toc, CoreToc) and toc.unknown is not None:
            highest = max(highest, toc.unknown & _U32)
    if highest + 1 > _U32:
        raise FormatError("no free handle above the highest one")
    return highest + 1


@dataclass(frozen=True)
class SnoIdentity:
    group: int
    name: str
    handle: int
    entry: TocEntry
    path: str  # archive path, ``|<locale>`` included for a localised copy

    def stamp(self, data: bytes) -> bytes:
        """Write the handle as the word at file offset 0x10, in the file's byte order (taken
        from the 0xDEADBEEF signature)."""
        head = bytes(data[:4])
        if head == SNO_SIGNATURE.to_bytes(4, "big"):
            order = ">"
        elif head == SNO_SIGNATURE.to_bytes(4, "little"):
            order = "<"
        else:
            raise FormatError("not an SNO file: no 0xDEADBEEF signature", 0)
        if len(data) < SNO_HEADER_SIZE + 4:
            raise FormatError("SNO file too short for the handle word", SNO_HEADER_SIZE)
        out = bytearray(data)
        struct.pack_into(order + "I", out, SNO_HEADER_SIZE, self.handle & _U32)
        return bytes(out)


def sno_identity(group: int, name: str, handle: int, locale: str | None = None) -> SnoIdentity:
    """W14-1: the CoreTOC entry, the offset-0x10 handle and the archive path
    ``<group directory>\\<name>.<ext>`` (``|<locale>`` appended for a localised copy)."""
    info = group_by_id(group)
    if info is None:
        raise FormatError(f"group {group} has no directory (unknown group)")
    if info.extension in CHILD_EXTENSIONS:
        raise FormatError(f".{info.extension} files have no CoreTOC entry")
    raw = _name_bytes(name)
    if not raw:
        raise FormatError("empty name")
    if len(raw) > PS3_NAME_FIELD:
        raise FormatError(f"name of {len(raw)} bytes does not fit the 128-byte ps3 field")
    if any(c in name for c in "\\/|"):
        raise FormatError(f"name {name!r} holds a path or locale separator")
    _u32(handle)
    path = f"{info.name}\\{name}.{info.extension}"
    if locale:
        path += f"|{locale}"
    return SnoIdentity(group, name, handle, TocEntry(group, handle, name), path)


# --- W14-2 / W14-3 ---------------------------------------------------------------------------

def _write_toc_ps3(entries: list[TocEntry]) -> bytes:
    handles = [e.handle & _U32 for e in entries]
    order = sorted(range(len(entries)), key=lambda i: handles[i])
    for a, b in zip(order, order[1:]):
        if handles[a] == handles[b]:
            raise FormatError(f"handle {handles[a]} appears twice (ps3 needs strictly ascending)")
    out = bytearray(struct.pack(">I", len(entries)))
    for i in order:
        e = entries[i]
        raw = _name_bytes(e.name)
        if len(raw) > PS3_NAME_FIELD:
            raise FormatError(f"name {e.name!r} longer than 128 bytes")
        out += struct.pack(">II", _u32(e.group), handles[i]) + raw.ljust(PS3_NAME_FIELD, b"\0")
    return bytes(out)


def _group_blocks(entries: list[TocEntry], keep_order: bool) -> list[list[TocEntry]]:
    blocks: list[list[TocEntry]] = [[] for _ in range(GROUP_SLOTS)]
    for e in entries:
        if not 0 <= e.group < GROUP_SLOTS:
            raise FormatError(f"group {e.group} outside 0..69")
        blocks[e.group].append(e)
    if not keep_order:
        for g, block in enumerate(blocks):
            block.sort(key=lambda e: e.handle & _U32)
            for a, b in zip(block, block[1:]):
                if a.handle & _U32 == b.handle & _U32:
                    raise FormatError(f"group {g}: handle {a.handle} appears twice")
    return blocks


def _write_toc_switch(entries: list[TocEntry], order: str, totals, top, keep_order: bool) -> bytes:
    blocks = _group_blocks(entries, keep_order)
    counts = [len(b) for b in blocks]
    if totals is None:
        totals = counts
    totals = list(totals)
    if len(totals) != GROUP_SLOTS:
        raise FormatError(f"{len(totals)} third values, expected {GROUP_SLOTS}")
    if top is None:
        top = max((e.handle & _U32 for e in entries), default=0)
    offsets = [0] * GROUP_SLOTS
    body = bytearray()
    for g, block in enumerate(blocks):
        if not block:
            # The position the next block would get; group 0 stores 0 (equal anyway).
            offsets[g] = 0 if g == 0 else len(body)
            continue
        offsets[g] = len(body)
        names = bytearray()
        records = bytearray()
        for e in block:
            records += struct.pack(order + "III", _u32(e.group), _u32(e.handle), len(names))
            names += _name_bytes(e.name) + b"\0"
        body += records + names
        body += bytes(-len(body) % 4)
    header = struct.pack(
        order + f"{GROUP_SLOTS}I{GROUP_SLOTS}I{GROUP_SLOTS}II",
        *counts, *offsets, *(_u32(t) for t in totals), _u32(top))
    assert len(header) == SWITCH_TOC_HEADER
    return header + bytes(body)


def write_core_toc(entries, build: str | None = None, totals: Sequence[int] | None = None,
                   top: int | None = None, keep_order: bool = False, title_update: bool = False) -> bytes:
    """W14-2 / W14-3. ``entries`` is a CoreToc (as parsed) or a list of TocEntry / tuples.

    ps3: count, then 136-byte entries sorted by handle. switch / x360: 844-byte header, then a
    block per group with entries sorted by handle (``keep_order`` keeps the given order, for
    the x360 title-update TOC files). Third values: ``totals`` (default: own counts); last word:
    ``top`` (default: own highest handle). A parsed x360 CoreToc supplies its stored totals and
    word unless given; for the six x360 main files use ``write_x360_tocs``.

    V16-9: an x360 title-update TOC written alone must keep its stored order: the caller passes
    ``keep_order=True``. ``title_update=True`` without ``keep_order`` is refused, and so is
    sorting a parsed x360 CoreToc whose entries are not sorted by handle inside their group (the
    form of a title-update file): sorting would change the file."""
    if title_update and not keep_order:
        raise FormatError("an x360 title-update TOC keeps its stored order: pass keep_order=True (V16-9)")
    if isinstance(entries, CoreToc) and entries.build == "x360" and not keep_order:
        by_group: dict[int, list[int]] = {}
        for item in _entries_of(entries):
            by_group.setdefault(item.group, []).append(item.handle)
        if any(handles != sorted(handles) for handles in by_group.values()):
            raise FormatError("this x360 TOC is not sorted by handle inside its groups (a title-update "
                              "TOC): pass keep_order=True, or a list of entries to sort it (V16-9)")
    if isinstance(entries, CoreToc):
        build = build or entries.build
        if build == "x360" and entries.build == "x360":
            totals = entries.totals if totals is None else totals
            top = entries.unknown if top is None else top
    if build is None:
        build = "ps3"
    items = _entries_of(entries)
    if build == "ps3":
        if totals is not None or top is not None or keep_order:
            raise FormatError("ps3 CoreTOC has no totals, highest-handle word or kept order")
        return _write_toc_ps3(items)
    if build not in ("switch", "x360"):
        raise FormatError(f"unknown build {build!r}")
    return _write_toc_switch(items, build_order(build), totals, top, keep_order)


def add_toc_entry(toc, group: int, handle: int, name: str, keep_order: bool = False):
    """W14-2 / W14-3: insert an entry by handle (ps3: over the file; switch / x360: inside its
    group). With ``keep_order`` (x360 title-update files) append it after its group's entries.

    ``toc`` is a CoreToc or a list; it is changed in place and returned. A handle already used
    in this TOC is refused (handles are unique over all groups, W14-1)."""
    items = toc.entries if isinstance(toc, CoreToc) else toc
    build = toc.build if isinstance(toc, CoreToc) else "ps3"
    _name_bytes(name)
    _u32(handle)
    if any(e.handle & _U32 == handle & _U32 for e in _entries_of(items)):
        raise FormatError(f"handle {handle} is already in this CoreTOC")
    if build != "ps3" and not 0 <= group < GROUP_SLOTS:
        raise FormatError(f"group {group} outside 0..69")
    new = TocEntry(group, handle, name)
    entries = _entries_of(items)
    if build == "ps3":
        at = len(entries)
        for i, e in enumerate(entries):
            if e.handle & _U32 > handle & _U32:
                at = i
                break
    elif keep_order:
        at = 0
        for i, e in enumerate(entries):
            if e.group <= group:
                at = i + 1
    else:
        at = len(entries)
        for i, e in enumerate(entries):
            if e.group > group or (e.group == group and e.handle & _U32 > handle & _U32):
                at = i
                break
    items.insert(at, new)
    return toc


def _is_main_toc(name: str) -> bool:
    return os.path.basename(name.replace("\\", "/")).lower() in X360_MAIN_TOCS


def write_x360_tocs(tocs: Mapping[str, object], title_update: Iterable[str] = ()) -> dict[str, bytes]:
    """W14-3, x360: rewrite all TOC files at once. ``tocs`` maps a file name to its entries
    (CoreToc or list). Names whose base name is ``CommonTOC.dat`` or ``Act1TOC.dat`` ..
    ``Act5TOC.dat`` are the main files: sorted by handle, third value = distinct (group, handle)
    of the group over all main files, last word = highest handle over them. Other names, and
    every name in ``title_update``, are title-update files: own counts, own highest handle,
    stored order kept."""
    forced = set(title_update)
    main = {n: _entries_of(e) for n, e in tocs.items() if _is_main_toc(n) and n not in forced}
    keys: set[tuple[int, int]] = set()
    for entries in main.values():
        for e in entries:
            if not 0 <= e.group < GROUP_SLOTS:
                raise FormatError(f"group {e.group} outside 0..69")
            keys.add((e.group, e.handle & _U32))
    totals = [0] * GROUP_SLOTS
    for group, _ in keys:
        totals[group] += 1
    top = max((h for _, h in keys), default=0)
    out = {}
    for name, entries in tocs.items():
        if name in main:
            out[name] = write_core_toc(main[name], "x360", totals, top)
        else:
            out[name] = write_core_toc(_entries_of(entries), "x360", None, None, keep_order=True)
    return out


# --- W14-4 -----------------------------------------------------------------------------------

def prefetch_bucket(handle: int) -> int:
    """x360 main Prefetch: FNV-1a-32 of the handle's 4 little-endian bytes, & 0x3FFF."""
    value = 0x811C9DC5
    for byte in struct.pack("<I", handle & _U32):
        value = ((value ^ byte) * 0x01000193) & _U32
    return value & 0x3FFF


def _prefetch_items(parents) -> list[tuple[int, int, list[tuple[int, int]]]]:
    if isinstance(parents, Prefetch):
        return [(p.group, p.handle, list(parents.dependencies(p))) for p in parents.parents]
    if isinstance(parents, Mapping):
        return [(g, h, list(d)) for (g, h), d in parents.items()]
    return [(g, h, list(d)) for g, h, d in parents]


def _prefetch_layout(items, build: str, keep_order: bool):
    """Parent list order and block order of W14-4. Returns (parents, deps)."""
    seen = set()
    for g, h, _ in items:
        if (g, h) in seen:
            raise FormatError(f"parent ({g}, {h}) appears twice")
        seen.add((g, h))
    if build == "ps3":
        listed = sorted(items, key=lambda it: (it[1] & 0xFF, it[0], -it[1]))
    elif build == "switch":
        listed = sorted(items, key=lambda it: (it[1] & 0xFFFF, -it[1]))
    elif build == "x360":
        listed = list(items) if keep_order else sorted(items, key=lambda it: prefetch_bucket(it[1]))
    else:
        raise FormatError(f"unknown build {build!r}")
    if build == "x360":
        blocks = listed
    else:
        blocks = sorted(items, key=lambda it: (it[1], it[0]))
    first: dict[tuple[int, int], int] = {}
    deps: list[tuple[int, int]] = []
    for g, h, d in blocks:
        first[(g, h)] = len(deps)
        deps.extend((dg, dh) for dg, dh in d)
    parents = [PrefetchParent(g, h, len(d), first[(g, h)]) for g, h, d in listed]
    return parents, deps


def write_prefetch(parents, build: str | None = None, keep_order: bool = False) -> bytes:
    """W14-4. ``parents`` is a parsed Prefetch, a list of (group, handle, deps) or a mapping
    {(group, handle): deps}. The parent list and the blocks follow the build's order rule;
    ``keep_order`` (x360 title-update files) keeps the given list order and appends.

    A Prefetch parsed without a build reads x360 files as ps3 (same byte order): pass
    ``build="x360"`` for them."""
    if build is None:
        build = parents.build if isinstance(parents, Prefetch) else None
    if build is None:
        raise FormatError("pass a build")
    if keep_order and build != "x360":
        raise FormatError("only x360 title-update Prefetch files keep their order")
    order = build_order(build)
    listed, deps = _prefetch_layout(_prefetch_items(parents), build, keep_order)
    out = bytearray(struct.pack(order + "I", len(listed)))
    for p in listed:
        out += struct.pack(order + "4I", _u32(p.group), _u32(p.handle), p.dep_count, p.first_dep)
    out += struct.pack(order + "I", len(deps))
    for g, h in deps:
        out += struct.pack(order + "II", _u32(g), _u32(h))
    return bytes(out)


def add_prefetch_parent(prefetch: Prefetch, group: int, handle: int, deps,
                        build: str | None = None, keep_order: bool = False) -> Prefetch:
    """W14-4: add a parent and its dependencies. The Prefetch is relaid out in place as the
    writer orders it (x360 main: at the end of its bucket; x360 title update with
    ``keep_order``: appended) and returned. An existing (group, handle) parent is refused."""
    build = build or prefetch.build
    if keep_order and build != "x360":
        raise FormatError("only x360 title-update Prefetch files keep their order")
    items = _prefetch_items(prefetch)
    if any(g == group and h == handle for g, h, _ in items):
        raise FormatError(f"parent ({group}, {handle}) is already listed")
    _u32(group)
    _u32(handle)
    items.append((group, handle, [(g, h) for g, h in deps]))
    prefetch.parents, prefetch.deps = _prefetch_layout(items, build, keep_order)
    prefetch.build = build
    return prefetch


# --- W14-5 -----------------------------------------------------------------------------------

def write_manifest(manifest, order: str | None = None) -> bytes:
    """W14-5: magic, then {group, handle} in the given order. ``order`` is "<", ">" or a build;
    a parsed Manifest supplies its own."""
    if isinstance(manifest, Manifest):
        order = order or manifest.order
        entries = manifest.entries
    else:
        entries = manifest
    if order is None:
        raise FormatError("pass a byte order or build")
    o = _order(order)
    out = bytearray(struct.pack(o + "I", MANIFEST_MAGIC))
    for g, h in entries:
        out += struct.pack(o + "II", _u32(g), _u32(h))
    return bytes(out)


def write_sprite_table(table, order: str | None = None) -> bytes:
    """W14-5: magic, then {name hash, texture handle, frame} sorted by (texture handle, frame).
    Entries may be Sprite, (hash, handle, frame) tuples or (name str, handle, frame) tuples (the
    name is hashed with ``hash33``). Also for the x360 ``.itemic``, ``.mapic``, ``.uiic`` files."""
    if isinstance(table, SpriteTable):
        order = order or table.order
        sprites = table.sprites
    else:
        sprites = table
    if order is None:
        raise FormatError("pass a byte order or build")
    o = _order(order)
    rows = []
    for s in sprites:
        if not isinstance(s, Sprite):
            name_hash, texture, frame = s
            if isinstance(name_hash, str):
                name_hash = hash33(name_hash)
            s = Sprite(name_hash, texture, frame)
        rows.append(s)
    rows.sort(key=lambda s: (s.texture_handle & _U32, s.frame & _U32))
    out = bytearray(struct.pack(o + "I", SPRITE_MAGIC))
    for s in rows:
        out += struct.pack(o + "III", _u32(s.name_hash), _u32(s.texture_handle), _u32(s.frame))
    return bytes(out)


def write_descriptor_sizes(entries: Sequence[DescriptorSize], order: str = "switch") -> bytes:
    """W14-5: magic, then {name[64] NUL-padded, size, 256 words}. The parser does not keep the
    byte order: pass "x360" (or ">") for the big-endian x360 file."""
    o = _order(order)
    out = bytearray(struct.pack(o + "I", DESCRIPTOR_MAGIC))
    for e in entries:
        if not isinstance(e, DescriptorSize):
            e = DescriptorSize(*e)
        raw = _name_bytes(e.name)
        if len(raw) > DESCRIPTOR_NAME_FIELD:
            raise FormatError(f"descriptor name {e.name!r} longer than 64 bytes")
        if len(e.words) != DESCRIPTOR_WORDS:
            raise FormatError(f"descriptor {e.name!r}: {len(e.words)} words, expected 256")
        out += raw.ljust(DESCRIPTOR_NAME_FIELD, b"\0") + struct.pack(o + "I", _u32(e.size))
        out += struct.pack(o + f"{DESCRIPTOR_WORDS}I", *(_u32(w) for w in e.words))
    return bytes(out)


def write_slo() -> bytes:
    """W14-5: an x360 ``.slo`` file is the magic 0xACBF0004 alone (big-endian)."""
    return struct.pack(">I", SLO_MAGIC)


# --- W14-6 -----------------------------------------------------------------------------------

def _read(archive):
    from .archive import CpkArchive, read_archive

    return archive if isinstance(archive, CpkArchive) else read_archive(archive)


def _x360_write(cpk, out, replace: Mapping[str, bytes], add: Mapping[str, bytes]):
    from .archive import name_hash
    from .x360_archive import Piece, source_pieces, write_x360_archive_to

    h = cpk.header
    pieces = source_pieces(cpk, lazy=True)  # Y1: a piece that is not replaced is read while written
    wanted = {name_hash(n): d for n, d in replace.items()}
    pieces = [Piece(p.name, p.copy, wanted.get(name_hash(p.name), p.data)) for p in pieces]
    pieces += [Piece(name, 0, data) for name, data in add.items()]
    written = write_x360_archive_to(out, pieces, flags=h.flags, read_sector=h.read_sector,
                                    comp_sector=h.comp_sector, keep_plans=False).written
    cpk._cache.clear()
    return written


def _change(archive, replace: Mapping[str, bytes], add: Mapping[str, bytes]) -> bytes:
    import io

    buffer = io.BytesIO()
    _change_to(archive, buffer, replace, add)
    return buffer.getvalue()


def _change_to(archive, out, replace: Mapping[str, bytes], add: Mapping[str, bytes]):
    """``47`` Y1: the changed archive into ``out`` (a binary file object or ``archive.Output``);
    returns its ``archive.Written`` (size and SHA-256)."""
    from .archive import as_output, name_hash, rebuild_to

    out = as_output(out)
    cpk = _read(archive)
    for name in add:
        if cpk.find(name) is not None:
            raise FormatError(f"{name!r} is already in the archive; use replace_file")
    for name in replace:
        if cpk.find(name) is None:
            raise FormatError(f"{name!r} is not in the archive; use add_file")
    if len({name_hash(n) for n in add}) != len(add):
        raise FormatError("two added names share a hash")
    if cpk.header.version == 7 and cpk.header.order == ">":
        return _x360_write(cpk, out, replace, add)
    return rebuild_to(out, cpk, replace=dict(replace), add=dict(add)).written


def add_file(archive, name: str, data: bytes) -> bytes:
    """W14-6: the archive (bytes or CpkArchive) with a new file ``name`` (one copy). ps3 and
    switch go through ``archive.rebuild``, x360 through ``write_x360_archive`` with an added
    piece at the end of the stream. Encrypted source packets are written unencrypted."""
    return _change(archive, {}, {name: bytes(data)})


def replace_file(archive, name: str, data: bytes) -> bytes:
    """W14-6: the archive with every copy of ``name`` replaced (its size may change)."""
    return _change(archive, {name: bytes(data)}, {})


def replace_files(archive, replace: Mapping[str, bytes], add: Mapping[str, bytes] | None = None) -> bytes:
    """W14-6 / W16-12: ``replace_file`` and ``add_file`` for several names in one rebuild (every
    replaced name must be in the archive, every added one must not)."""
    return _change(archive, {n: bytes(d) for n, d in replace.items()},
                   {n: bytes(d) for n, d in (add or {}).items()})


def write_changed(archive, target, replace: Mapping[str, bytes], add: Mapping[str, bytes] | None = None):
    """``47`` Y1: what ``replace_files`` returns, streamed to ``target``: a path (written under a
    temporary name and renamed when complete), a binary file object, or None (only counted and
    hashed: the dry run). Files that are not replaced are read from the source archive in pieces
    and nothing of the new archive is kept; give a memory-mapped source (``read_archive`` of an
    ``mmap``) to keep the source out of memory too. Returns ``archive.Written`` (size, SHA-256)."""
    from .archive import stream_to

    replace = {n: bytes(d) for n, d in replace.items()}
    add = {n: bytes(d) for n, d in (add or {}).items()}
    return stream_to(target, lambda out: _change_to(archive, out, replace, add))


# --- W16-12 / W16-13: add_sno_file ------------------------------------------------------------

class AddedFile(tuple):
    """``(archive, index_files)``: the new archive bytes and ``{name: bytes}`` of every index
    file that changed. ``identity`` is the SnoIdentity given to the file (None for a non-SNO or
    child file)."""

    def __new__(cls, archive: bytes, index_files: dict, identity: SnoIdentity | None = None):
        self = super().__new__(cls, (archive, index_files))
        self.identity = identity
        return self

    @property
    def archive(self) -> bytes:
        return self[0]

    @property
    def index_files(self) -> dict:
        return self[1]


def split_archive_name(name: str) -> tuple[str, str, str, str | None]:
    """``Group\\name.ext|locale`` -> (folder, stem, lower-case extension, locale or None)."""
    locale = None
    if "|" in name:
        name, locale = name.split("|", 1)
        if not locale or "|" in locale:
            raise FormatError(f"bad locale suffix in {name!r}")
    folder, _, base = name.replace("/", "\\").rpartition("\\")
    stem, dot, ext = base.rpartition(".")
    if not dot or not stem:
        return folder, base, "", locale
    return folder, stem, ext.lower(), locale


def _toc_key(name: str) -> str:
    return os.path.basename(name.replace("\\", "/")).lower()


# ``35`` V21-6: the stems that name a pack after the locale and ``Cache`` are removed
X360_PACK_NAMES = ("common", "act1", "act2", "act3", "act4", "act5")


def x360_pack(archive_name: str | None, toc_names: Iterable[str], pack: str | None = None) -> str:
    """W16-12: the main x360 TOC (one of ``toc_names``) of the pack an archive belongs to.

    ``pack`` (a TOC name such as ``Act1TOC.dat`` or a pack name such as ``Act1``) wins.
    Otherwise the archive's base name decides (``35`` V21-6, replacing Q722): take the stem with
    case ignored, remove a leading four-ASCII-letter locale and ``_`` (``enUS_``, V22-4), then a leading
    ``Cache``; the rest must be exactly ``Common`` or ``Act1`` .. ``Act5``. Anything else, the
    ``Patch`` and ``Cutscene`` archives included, is unclear and refused."""
    mains = [n for n in toc_names if _is_main_toc(n)]
    if pack is not None:
        wanted = pack.lower()
        if not wanted.endswith(".dat"):
            wanted += "toc.dat"
        found = [n for n in mains if _toc_key(n) == _toc_key(wanted)]
        if len(found) != 1:
            raise FormatError(f"pack {pack!r} names no single main x360 TOC of {sorted(mains)}")
        return found[0]
    if not archive_name:
        raise FormatError("the x360 pack is not known: pass pack (--pack)")
    stem = os.path.basename(archive_name.replace("\\", "/")).lower()
    stem = stem.rsplit(".", 1)[0] if "." in stem else stem
    # ``36`` V22-4: the locale is four ASCII letters
    if len(stem) > 5 and stem[4] == "_" and stem[:4].isascii() and stem[:4].isalpha():
        stem = stem[5:]
    if stem.startswith("cache"):
        stem = stem[len("cache"):]
    found = []
    if stem in X360_PACK_NAMES:
        found = [n for n in mains if _toc_key(n)[:-len("toc.dat")] == stem]
    if len(found) != 1:
        raise FormatError(f"the pack of archive {archive_name!r} is not clear: pass pack (--pack)")
    return found[0]


def _copy_toc(toc, build: str) -> CoreToc:
    if isinstance(toc, (bytes, bytearray, memoryview)):
        from .index import parse_core_toc

        return parse_core_toc(bytes(toc), build)
    if isinstance(toc, CoreToc):
        if toc.build != build:
            raise FormatError(f"a {toc.build} CoreTOC does not fit a {build} archive")
        return CoreToc(toc.build, list(toc.entries), toc.unknown, toc.name_padding_problems, toc.totals)
    return CoreToc(build, list(_entries_of(toc)))


def _named_tocs(tocs, build: str) -> dict[str, CoreToc]:
    if tocs is None:
        return {}
    if isinstance(tocs, Mapping):
        items = list(tocs.items())
    elif isinstance(tocs, (CoreToc, bytes, bytearray, memoryview)):
        items = [("CoreTOC.dat", tocs)]
    else:
        tocs = list(tocs)
        if tocs and isinstance(tocs[0], (TocEntry, tuple)):
            items = [("CoreTOC.dat", tocs)]
        elif len(tocs) == 1:
            items = [("CoreTOC.dat", tocs[0])]
        else:
            items = [(f"CoreTOC{i}.dat", t) for i, t in enumerate(tocs)]
    seen = set()
    for name, _ in items:
        if _toc_key(name) in seen:
            raise FormatError(f"two TOC files named {os.path.basename(name)!r}")
        seen.add(_toc_key(name))
    return {name: _copy_toc(t, build) for name, t in items}


def _resolve_dependency(dep, named: Mapping[str, CoreToc]) -> tuple[int, int]:
    if not isinstance(dep, str):
        g, h = dep
        return int(g), int(h)
    folder, stem, ext, _ = split_archive_name(dep)
    group = group_by_extension(ext) if ext else None
    if group is None:
        raise FormatError(f"dependency {dep!r} has no SNO extension")
    if folder and folder.lower() != group.name.lower():
        raise FormatError(f"dependency {dep!r}: .{ext} files are in {group.name}\\")
    for toc in named.values():
        for e in toc.entries:
            if e.group == group.id and e.name.lower() == stem.lower():
                return e.group, e.handle & _U32
    raise FormatError(f"dependency {dep!r} is in no given CoreTOC")


def add_sno_file(archive, name: str, data: bytes, tocs, prefetch=None, depends=(), *,
                 pack: str | None = None, out=None) -> AddedFile:
    """W16-12 / W16-13: add ``name`` to the archive and give an SNO file its identity (W14-1).

    - ``tocs``: one CoreToc (parsed, bytes or an entry list), or ``{file name: toc}``; on x360
      all six main TOC files (``CommonTOC.dat``, ``Act1TOC.dat`` .. ``Act5TOC.dat``), title-update
      TOC files may be added for the handle. Given TOCs are not changed.
    - SNO extension (group by extension, folder must be the group's): a new handle over every
      given TOC (``new_handle``), the CoreTOC entry (x360: in the TOC of ``pack``, see
      ``x360_pack``; ps3 / switch: the one TOC, or ``pack`` naming it), the handle at 0x10.
    - ``.nav`` / ``.can`` / ``.fsb`` and names without an SNO extension: archive only.
    - ``prefetch`` (Prefetch, bytes or ``{file name: prefetch}``) with ``depends`` (``(group,
      handle)`` pairs or SNO names such as ``Textures\\x.tex``, looked up in the TOCs) adds a
      Prefetch parent (W14-4). Without ``depends`` Prefetch is unchanged and not returned.
    - Refused: a name already in the archive, an SNO name (group, name) already in a TOC.

    Returns ``AddedFile(archive, index_files)`` (unpacks as a pair); index file names are the
    keys of ``tocs`` / ``prefetch`` (default ``CoreTOC.dat`` / ``Prefetch.dat``).

    ``47`` Y1: with ``out`` (a path or a binary file object) the archive is streamed there as
    ``write_changed`` does, and ``archive`` of the result is its ``archive.Written``."""
    from .index import parse_prefetch

    cpk = _read(archive)
    build = cpk.header.build
    if cpk.find(name) is not None:
        raise FormatError(f"{name!r} is already in the archive; use replace")
    depends = list(depends or ())
    if depends and prefetch is None:
        raise ValueError("dependencies need the Prefetch file (--prefetch)")
    folder, stem, ext, locale = split_archive_name(name)
    group = group_by_extension(ext) if ext and ext not in CHILD_EXTENSIONS else None
    if group is None:
        if depends:
            raise ValueError(f"{name!r} is no SNO file with a CoreTOC entry: it takes no dependencies")
        return AddedFile(_added(cpk, name, data, out), {}, None)
    if folder.lower() != group.name.lower():
        raise FormatError(f"{name!r}: .{ext} files belong in {group.name}\\")
    named = _named_tocs(tocs, build)
    if not named:
        raise ValueError(f"{name!r} is an SNO file: pass its CoreTOC (--toc)")
    for toc_name, toc in named.items():
        for e in toc.entries:
            if e.group == group.id and e.name.lower() == stem.lower():
                raise FormatError(f"SNO {group.name}\\{stem} is already in {os.path.basename(toc_name)} "
                                  f"(handle {e.handle})")
    if build == "x360":
        mains = sorted(_toc_key(n) for n in named if _is_main_toc(n))
        if mains != sorted(X360_MAIN_TOCS):
            raise FormatError("x360: pass all six main TOC files (CommonTOC.dat, Act1TOC.dat .. "
                              "Act5TOC.dat), W14-3")
        target = x360_pack(None, list(named), pack) if pack is not None else None
        if target is None:
            raise FormatError("x360: the pack of the archive is not known: pass pack (--pack)")
    elif pack is not None:
        found = [n for n in named if _toc_key(n) in (_toc_key(pack), _toc_key(pack) + "toc.dat")]
        if len(found) != 1:
            raise FormatError(f"pack {pack!r} names no single given TOC")
        target = found[0]
    elif len(named) == 1:
        target = next(iter(named))
    else:
        raise FormatError("several TOC files: pass pack to name the one that gets the entry")
    handle = new_handle(list(named.values()))
    ident = sno_identity(group.id, stem, handle, locale)
    stamped = ident.stamp(data)
    add_toc_entry(named[target], group.id, handle, stem)
    if build == "x360":
        written = write_x360_tocs(named)
        index_files = {n: d for n, d in written.items() if _is_main_toc(n)}
    else:
        index_files = {target: write_core_toc(named[target], build)}
    if depends:
        if isinstance(prefetch, Mapping):
            if len(prefetch) != 1:
                raise ValueError("pass one Prefetch file")
            pf_name, pf = next(iter(prefetch.items()))
        else:
            pf_name, pf = "Prefetch.dat", prefetch
        if isinstance(pf, Prefetch):
            pf = Prefetch(pf.build, list(pf.parents), list(pf.deps))
        else:
            pf = parse_prefetch(bytes(pf), build)
        deps = [_resolve_dependency(d, named) for d in depends]
        add_prefetch_parent(pf, group.id, handle, deps, build)
        index_files[pf_name] = write_prefetch(pf, build)
    return AddedFile(_added(cpk, name, stamped, out), index_files, ident)


def _added(cpk, name: str, data: bytes, out):
    return add_file(cpk, name, data) if out is None else write_changed(cpk, out, {}, {name: data})
