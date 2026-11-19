"""New materials, texture links and material reference checks (``30`` W15-A11, W15-A12).

A ``.mat`` is built and edited as a generic tree (``sno_layout``, W11) and laid out with
``sno_layout.write_tree``: body flags 0x0C (zero), tag map pair 0x10, material block 0x20 (file
0x1C, 0x20, 0x30). There is no count, size or hash besides the pairs (W15-A11).

Block (switch, x360 104 bytes; ps3 72): shader map handle or -1 at 0; four colours (ambient,
diffuse, specular, emissive; switch four floats each, ps3 four bytes plus a float scale each);
power; sort priority (signed); texture list pair; zero pointer slot; switch words +0x60 and
+0x64 zero; ps3 12 zero bytes at +0x3C. Texture entry (32 / 24 bytes): slot type, switch zero
pad, animation blob pair (blob of 0, 12, 20 or 144 bytes), zero pointer slot, texture handle or
-1, switch zero pad.

``31`` V18-6: within one material block, entries with byte-equal animation blobs point at one
stored blob; ``new_material`` and ``add_texture`` share equal blobs this way. V18-7: the only
unresolved links are 88 texture handles in 22 material blocks of 12 x360 appearance files.
V18-9: ``toc_keys`` does not copy a prepared set and uses ``CoreToc.key_set()`` (cached).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import AbstractSet, Iterable, Sequence

from . import material, sno_layout
from .binary import ByBuild, ByteView, FormatError, Pair, build_order
from .power import TAG_FLOAT, TAG_SNO, read_tag_map
from .sno import SNO_SIGNATURE, expected_version

__all__ = [
    "ANIMATION_SIZES", "MaterialLinks", "new_material", "set_texture", "add_texture",
    "material_tree", "material_textures", "check_material_links", "material_zero_problems",
    "encode_tag_map", "toc_keys", "GROUP_SHADER_MAP", "GROUP_TEXTURE", "TAG_TARGET_GROUPS",
]

ANIMATION_SIZES = (0, 12, 20, 144)
GROUP_SHADER_MAP = 36
GROUP_SHADERS = 37
GROUP_TEXTURE = 44
GROUP_SURFACE = 43
GROUP_PARTICLE = 27
TAG_TARGET_GROUPS = (GROUP_SURFACE, GROUP_PARTICLE, GROUP_SHADERS)  # W15-A12: type-2 values

FLAGS_AT = 0x0C  # body (record) offsets
TAGS_AT = 0x10
BLOCK_AT = 0x20
_BLOCK_SIZE = ByBuild({"switch": 104, "ps3": 72})
_LIST_AT = ByBuild({"switch": 0x4C, "ps3": 0x2C})
_TAIL_AT = ByBuild({"switch": 0x44, "ps3": 0x24})  # power, then sort priority
_ENTRY_SIZE = ByBuild({"switch": 32, "ps3": 24})
_ANIM_AT = ByBuild({"switch": 0x08, "ps3": 0x04})
_HANDLE_AT = ByBuild({"switch": 0x18, "ps3": 0x14})
# Zero ranges (start, end) in the block and in a texture entry (Q: watch, see notes).
_BLOCK_ZERO = ByBuild({"switch": ((0x54, 0x68),), "ps3": ((0x34, 0x38), (0x3C, 0x48))})
_ENTRY_ZERO = ByBuild({"switch": ((0x04, 0x08), (0x10, 0x18), (0x1C, 0x20)), "ps3": ((0x0C, 0x14),)})


# --- Helpers -----------------------------------------------------------------------------------

def encode_tag_map(records, order_or_build: str) -> bytes:
    """A tag map from (type, tag, value) records: u32 count, then {u32 type, u32 tag, 4-byte
    value} each (type 1 a float, else an int). Bytes are returned as given."""
    if isinstance(records, (bytes, bytearray)):
        return bytes(records)
    order = order_or_build if order_or_build in "<>" else build_order(order_or_build)
    records = list(records or ())
    if not records:
        return b""
    out = bytearray(struct.pack(order + "I", len(records)))
    seen = set()
    for kind, tag, value in records:
        if tag in seen:
            raise FormatError(f"tag 0x{tag:X} repeats")
        seen.add(tag)
        out += struct.pack(order + "II", kind, tag)
        out += struct.pack(order + ("f" if kind == TAG_FLOAT else "i"), value)
    return bytes(out)


def _tag_records(raw: bytes, order: str):
    if not raw:
        return []
    view = ByteView(bytes(0x10) + bytes(raw), order)
    return read_tag_map(view, Pair(0, len(raw)), "material", 0x10, type_7=True).records


def _toc_key_set(toc):
    key_set = getattr(toc, "key_set", None)
    if callable(key_set):
        return key_set()  # CoreToc: built once per object (V18-9)
    return {(e.group, e.handle) for e in toc.entries}


def toc_keys(toc) -> AbstractSet[tuple[int, int]]:
    """(group, handle) keys of a ``CoreToc``, a list of them, TOC entries or a set of keys.

    ``31`` V18-9: the result is for lookups only and is not copied per call. A set or frozenset
    is returned as it is (a prepared set), and a ``CoreToc`` gives its cached ``key_set()``, so a
    check over many handles does not rebuild the key set. Other inputs are built each call:
    prepare them once with ``toc_keys`` and pass the result."""
    if isinstance(toc, (set, frozenset)):
        return toc
    if hasattr(toc, "entries"):
        return _toc_key_set(toc)
    keys: set[tuple[int, int]] = set()
    for item in toc:
        if hasattr(item, "entries"):
            keys |= _toc_key_set(item)
        elif hasattr(item, "group"):
            keys.add((item.group, item.handle))
        else:
            keys.add(tuple(item))
    return keys


def _colour_bytes(colour, build: str, order: str) -> bytes:
    if isinstance(colour, material.Colour):
        rgba, scale = colour.rgba, colour.scale
    elif build == "ps3":
        rgba, scale = colour
    else:
        rgba, scale = colour, None
    if build == "ps3":
        if len(rgba) != 4 or any(not 0 <= v <= 255 for v in rgba):
            raise FormatError(f"ps3 colour {rgba!r} is not four bytes")
        return bytes(rgba) + struct.pack(order + "f", 1.0 if scale is None else scale)
    if len(rgba) != 4:
        raise FormatError(f"colour {rgba!r} is not four floats")
    return struct.pack(order + "4f", *rgba)


def _entry(build: str, element: str, slot: int, handle: int, animation: bytes,
           blobs: dict[bytes, sno_layout.Block] | None = None) -> sno_layout.Record:
    """One texture entry. ``blobs`` (V18-6): the animation blocks already in the material block,
    by their bytes; an equal blob reuses that block (the writer then stores one offset)."""
    animation = bytes(animation)
    if len(animation) not in ANIMATION_SIZES:
        raise FormatError(f"animation blob of {len(animation)} bytes (W15-A11: 0, 12, 20 or 144)")
    order = build_order(build)
    raw = bytearray(_ENTRY_SIZE[build])
    struct.pack_into(order + "I", raw, 0, slot)
    struct.pack_into(order + "i", raw, _HANDLE_AT[build], handle)
    record = sno_layout.Record(element, raw)
    if animation:  # an empty blob has no block: its pair is {0, 0} and is never shared
        block = blobs.get(animation) if blobs is not None else None
        if block is not None:
            record.pairs[_ANIM_AT[build]] = sno_layout.PairValue(block, shared=True)
        else:
            block = sno_layout.Block(animation)
            if blobs is not None:
                blobs[animation] = block
            record.pairs[_ANIM_AT[build]] = sno_layout.PairValue(block)
    return record


def _animation_blocks(build: str, rows: Iterable[sno_layout.Record]) -> dict[bytes, sno_layout.Block]:
    """V18-6: the non-empty animation blocks of texture entries, by bytes (the first of equal ones)."""
    found: dict[bytes, sno_layout.Block] = {}
    for row in rows:
        value = row.pairs.get(_ANIM_AT[build])
        if value is None or value.block is None or value.block.is_empty() or value.block.data is None:
            continue
        if value.size is not None and value.size != value.block.size():
            continue  # a pair with its own size (W11-6) is not the whole block
        found.setdefault(bytes(value.block.data), value.block)
    return found


def _entry_args(entry) -> tuple[int, int, bytes]:
    if isinstance(entry, material.TextureSlot):
        return entry.type, entry.handle, entry.animation_raw
    if len(entry) == 2:
        return entry[0], entry[1], b""
    return entry[0], entry[1], bytes(entry[2])


def _list_slot(tree: sno_layout.SnoTree) -> sno_layout.PairSlot:
    if tree.ext != "mat":
        raise FormatError(f"not a material tree (.{tree.ext})")
    offset = BLOCK_AT + _LIST_AT[tree.build]
    for slot in sno_layout.pair_slots(tree.build, tree.ext, tree.top.name):
        if slot.offset == offset:
            return slot
    raise FormatError(f"{tree.build} .mat: no texture list pair at +0x{offset:X}")


def _entries(tree: sno_layout.SnoTree) -> list[sno_layout.Record]:
    value = tree.top.pairs.get(_list_slot(tree).offset)
    if value is None or value.block is None or value.block.elements is None:
        return []
    return value.block.elements


# --- API ---------------------------------------------------------------------------------------

def material_tree(data: bytes, build: str | None = None) -> sno_layout.SnoTree:
    """Read a ``.mat`` as a generic tree (every ``.mat`` rebuilds through it, W15-A11)."""
    return sno_layout.parse_tree(data, "mat", build)


def new_material(build: str, shader_map: int, colours: Sequence, power: float, sort: int,
                 entries: Iterable = (), tags=None, *, handle: int = 0, version: int | None = None,
                 revision: int = 1, source_hash: int = 0,
                 share_animations: bool = True) -> sno_layout.SnoTree:
    """W15-A11: a new ``.mat`` as a generic tree; ``sno_layout.write_tree`` lays it out.

    ``colours``: ambient, diffuse, specular, emissive; switch and x360 four floats each, ps3
    ((r, g, b, a) bytes, float scale) each (``material.Colour`` also accepted). ``entries``:
    (slot type, texture handle) or (slot type, handle, animation blob) or ``TextureSlot``.
    ``tags``: tag-map bytes or (type, tag, value) records (type 2 values are SNO handles).
    ``handle`` is the file's SNO handle (body word 0); flags are 0.

    ``31`` V18-6 (PROVEN): entries whose animation blobs are byte-equal point at one stored blob
    (the first one's offset), so a material rebuilt from its fields equals the stored file.
    ``share_animations=False`` stores every blob on its own (not seen in the game files). An
    empty blob is the pair {0, 0} and is not shared."""
    order = build_order(build)
    colours = list(colours)
    if len(colours) != 4:
        raise FormatError(f"{len(colours)} colours, a material has 4")
    top_name = sno_layout.top_record(build, "mat")
    top = sno_layout.new_record(build, "mat", top_name)
    struct.pack_into(order + "i", top.raw, 0, handle)
    struct.pack_into(order + "I", top.raw, FLAGS_AT, 0)
    at = BLOCK_AT
    struct.pack_into(order + "i", top.raw, at, shader_map)
    width = 8 if build == "ps3" else 16
    for i, colour in enumerate(colours):
        top.raw[at + 4 + i * width:at + 4 + (i + 1) * width] = _colour_bytes(colour, build, order)
    struct.pack_into(order + "fi", top.raw, at + _TAIL_AT[build], power, sort)
    tag_raw = encode_tag_map(tags, order)
    if tag_raw:
        _tag_records(tag_raw, order)  # must read back
        top.pairs[TAGS_AT] = sno_layout.PairValue(sno_layout.Block(tag_raw))
    header =struct.pack(order + "4I", SNO_SIGNATURE, version or expected_version("mat", build) or 1,
                         revision, source_hash)
    tree = sno_layout.SnoTree(build, "mat", header, top)
    slot = _list_slot(tree)
    blobs: dict[bytes, sno_layout.Block] | None = {} if share_animations else None
    rows = [_entry(build, slot.field.element, *_entry_args(e), blobs) for e in entries]
    if rows:
        top.pairs[slot.offset] = sno_layout.PairValue(sno_layout.Block(elements=rows))
    return tree


def set_texture(material_tree_: sno_layout.SnoTree, slot: int, handle: int) -> int:
    """W15-A12: point the first texture entry of slot type ``slot`` at ``handle`` (or -1), in
    place; the slot type stays. Returns the old handle."""
    order = build_order(material_tree_.build)
    at = _HANDLE_AT[material_tree_.build]
    for entry in _entries(material_tree_):
        if struct.unpack_from(order + "I", entry.raw, 0)[0] == slot:
            old = struct.unpack_from(order + "i", entry.raw, at)[0]
            struct.pack_into(order + "i", entry.raw, at, handle)
            return old
    raise FormatError(f"the material has no texture of slot type {slot}")


def add_texture(material_tree_: sno_layout.SnoTree, slot: int, handle: int, animation: bytes = b"",
                allowed_slots: Iterable[int] | None = None, share_animations: bool = True) -> None:
    """Append a texture entry. ``allowed_slots`` (for example ``shader.sampled_slot_types`` of
    the shader map's default shader, W15-A12) refuses other slot types.

    ``31`` V18-6: when an entry of the same material block already has a byte-equal animation
    blob, the new entry points at that blob instead of a copy (``share_animations=False``:
    store a copy)."""
    if allowed_slots is not None and slot not in set(allowed_slots):
        raise FormatError(f"slot type {slot} is not sampled by the shader")
    tree = material_tree_
    list_slot = _list_slot(tree)
    old = list(_entries(tree))
    blobs = _animation_blocks(tree.build, old) if share_animations else None
    rows = old + [_entry(tree.build, list_slot.field.element, slot, handle, animation, blobs)]
    tree.top.pairs[list_slot.offset] = sno_layout.PairValue(sno_layout.Block(elements=rows))


def material_textures(material_tree_: sno_layout.SnoTree) -> list[tuple[int, int, bytes]]:
    """(slot type, handle, animation blob) of every texture entry."""
    tree = material_tree_
    order = build_order(tree.build)
    found = []
    for entry in _entries(tree):
        value = entry.pairs.get(_ANIM_AT[tree.build])
        blob = b"" if value is None or value.block is None else bytes(value.block.data or b"")
        found.append((struct.unpack_from(order + "I", entry.raw, 0)[0],
                      struct.unpack_from(order + "i", entry.raw, _HANDLE_AT[tree.build])[0], blob))
    return found


@dataclass
class MaterialLinks:
    shader_map: int
    textures: list[tuple[int, int]]  # (slot type, handle)
    tag_handles: list[tuple[int, int]]  # (tag, handle) of type-2 records


def _links(material_, build: str | None) -> MaterialLinks:
    tags = b""
    order = None
    if isinstance(material_, sno_layout.SnoTree):
        material_ = material.parse_material(sno_layout.write_tree(material_), material_.build)
    elif isinstance(material_, (bytes, bytearray)):
        material_ = material.parse_material(bytes(material_), build)
    if isinstance(material_, material.MaterialFile):
        tags, order, block = material_.tag_map_raw, material_.header.order, material_.block
    elif isinstance(material_, material.MaterialBlock):
        block = material_
    else:
        raise FormatError(f"not a material: {type(material_).__name__}")
    tag_handles = [(r.tag, r.value) for r in _tag_records(tags, order or ">") if r.type == TAG_SNO]
    return MaterialLinks(block.shader_map, [(t.type, t.handle) for t in block.textures], tag_handles)


def _tag_group(tag: int) -> int | None:
    from .tags import tag_info

    info = tag_info("appearance", tag)  # tags.OWNER_TABLES["material"]
    return None if info is None else info.sno_group


def check_material_links(material_, toc, build: str | None = None) -> list[str]:
    """W15-A12: the handles a material names that do not resolve in ``toc``: the shader map
    handle (group 36, or -1), every texture handle (group 44, or -1) and every tag-map type-2
    value (a surface, particle or shader; the tag's own group when the tag table gives one).
    ``material_`` is a ``.mat`` tree, ``MaterialFile``, ``MaterialBlock`` or file bytes.

    ``toc`` is a ``CoreToc`` (its key set is cached on the object), a list of them, or a set of
    (group, handle) keys prepared once with ``toc_keys`` (used as it is, not copied; V18-9).

    Expected exceptions (``31`` V18-7, correcting W15-A12): 88 texture handles in 22 material
    blocks of 12 x360 appearance files (the player-class male and female appearances, 4 or 8
    handles each) do not resolve; every other material link resolves in all three builds. They
    are reported here like any other unresolved handle."""
    links = _links(material_, build)
    keys = toc_keys(toc)
    problems = []
    if links.shader_map != -1 and (GROUP_SHADER_MAP, links.shader_map) not in keys:
        problems.append(f"shader map {links.shader_map} is not in the TOC")
    for slot, handle in links.textures:
        if handle != -1 and (GROUP_TEXTURE, handle) not in keys:
            problems.append(f"texture {handle} (slot {slot}) is not in the TOC")
    for tag, handle in links.tag_handles:
        if handle == -1:
            continue
        group = _tag_group(tag)
        groups = (group,) if group in TAG_TARGET_GROUPS else TAG_TARGET_GROUPS
        if not any((g, handle) in keys for g in groups):
            problems.append(f"tag 0x{tag:X}: handle {handle} names no surface, particle or shader")
    return problems


def material_zero_problems(data: bytes, build: str | None = None) -> list[str]:
    """W15-A11: the fields that are zero in every ``.mat``: flags, the block's pointer slot and
    padding, every entry's pads and pointer slot; animation blobs of 0, 12, 20 or 144 bytes."""
    parsed = material.parse_material(bytes(data), build)
    build = parsed.build
    view = ByteView(bytes(data), parsed.header.order)
    problems = []
    if parsed.flags:
        problems.append(f"flags 0x{parsed.flags:X}")
    block_at = 0x10 + BLOCK_AT
    problems += block_zero_problems(view, block_at, build)
    return problems


def block_zero_problems(view: ByteView, block_at: int, build: str) -> list[str]:
    """The zero fields of one material block at file ``block_at`` (``.mat`` or ``.app``)."""
    problems = []
    for start, end in _BLOCK_ZERO[build]:
        if any(view.take(block_at + start, end - start)):
            problems.append(f"block bytes +0x{start:X}..0x{end:X} are not zero")
    pair = view.pair(block_at + _LIST_AT[build])
    width = _ENTRY_SIZE[build]
    for i in range(pair.size // width):
        at = pair.file_offset + i * width
        for start, end in _ENTRY_ZERO[build]:
            if any(view.take(at + start, end - start)):
                problems.append(f"texture entry {i} bytes +0x{start:X}..0x{end:X} are not zero")
        size = view.pair(at + _ANIM_AT[build]).size
        if size not in ANIMATION_SIZES:
            problems.append(f"texture entry {i}: animation blob of {size} bytes")
    return problems
