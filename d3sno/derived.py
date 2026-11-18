"""Stage 14 derived values of marker sets, worlds, actors, sound banks and fonts
(``28_derived_values.md`` W14-18 to W14-23).

- W14-18 ``special_indices``: the marker-set special list; ``set_special_indices`` /
  ``derive_special_indices`` write it back through the generic tree (count word by W14-20).
- W14-19 ``cluster_counter`` / ``derive_cluster_counter``: the world server block +0x38.
- W14-20 count fields live in ``sno_layout`` (``write_tree(derive=True)``,
  ``set_elements(..., update_counts=True)``); ``count_links`` and ``derive_counts`` are re-exported.
- W14-21 ``derive_actor_bounds``: actor box, sphere centre and radius from the appearance.
- W14-22 ``update_bank_row``: a sound-bank row's stored and decompressed sizes.
- W14-23 ``insert_kerning`` / ``kerning_order_problems``: kerning pairs in (first, second) order.
  Animation counts follow W14-20; root-motion values are authored and kept (nothing to derive).

Every function returns new bytes (or changes the given tree); nothing is derived unless called.
"""

from __future__ import annotations

import bisect
import struct
from typing import Iterable, Mapping

from . import sno_layout as L
from .binary import ByteView, FormatError, build_order
from .power import read_tag_map
from .sno import parse_header, resolve_build
from .sno_layout import count_links, derive_counts, update_count_words  # noqa: F401  (W14-20)
from .world import MarkerSet, SceneEntry, parse_marker_set, parse_world

BODY = 0x10

# --- W14-18. Marker-set special list --------------------------------------------------------

SPECIAL_TYPES = frozenset({16, 28})
SPECIAL_TYPE_FROM = 50  # types >= 50 (gizmo locations, locations)
SPECIAL_TYPE_SWITCH_FORM = 29  # switch and x360 only
ACTOR_KIND_GIZMO = 2
GIZMO_TYPE_TAG = 0x10301
SPECIAL_GIZMO_TYPES = frozenset({2, 4, 12, 20, 66, 67, 68, 70})
SPECIAL_COUNT_AT = 0xB0  # body
SPECIAL_PAIR_AT = 0xB4  # body
ACTOR_KIND_AT = 0x10  # actor body
ACTOR_APPEARANCE_AT = 0x14
ACTOR_TAG_MAP_AT = 0x58

ActorInfo = tuple  # (actor kind, gizmo type or None)


def actor_gizmo_info(actor_data: bytes, build: str | None = None) -> tuple[int, int | None]:
    """An ``.acr`` file's actor kind (body 0x10) and its gizmo-type tag 0x10301 (None if absent)."""
    header = parse_header(actor_data, build)
    view = ByteView(actor_data, header.order)
    kind = view.i32(BODY + ACTOR_KIND_AT, "actor kind")
    tags = read_tag_map(view, view.pair(BODY + ACTOR_TAG_MAP_AT, "actor tag map"), "actor",
                        BODY + ACTOR_TAG_MAP_AT, lenient=True)
    record = tags.get(GIZMO_TYPE_TAG)
    return kind, (None if record is None else record.value)


def _actor_info(actor_lookup, handle: int):
    if actor_lookup is None:
        return None
    if callable(actor_lookup) and not isinstance(actor_lookup, Mapping):
        info = actor_lookup(handle)
    else:
        info = actor_lookup.get(handle)
    if isinstance(info, (bytes, bytearray)):
        info = actor_gizmo_info(bytes(info))
    return info


def is_special_marker(marker_type: int, build: str, actor: ActorInfo | None = None) -> bool:
    """W14-18 for one marker: ``actor`` is (actor kind, gizmo type) of an actor marker's actor."""
    if marker_type in SPECIAL_TYPES or marker_type >= SPECIAL_TYPE_FROM:
        return True
    if marker_type == SPECIAL_TYPE_SWITCH_FORM and build != "ps3":
        return True
    if marker_type == 0 and actor is not None:
        kind, gizmo = actor[0], actor[1]
        return kind == ACTOR_KIND_GIZMO and gizmo in SPECIAL_GIZMO_TYPES
    return False


def special_indices(marker_set: MarkerSet | bytes, actor_lookup=None, build: str | None = None) -> list[int]:
    """W14-18: the ascending indices of the special markers of a marker set.

    ``actor_lookup`` maps an actor handle (marker +0x2C) to (actor kind, gizmo type), to the
    ``.acr`` bytes, or to None; it may be a mapping or a callable. An actor it does not know makes
    its marker not special."""
    if isinstance(marker_set, (bytes, bytearray)):
        marker_set = parse_marker_set(bytes(marker_set), build)
    found = [
        i for i, marker in enumerate(marker_set.markers)
        if is_special_marker(marker.type, marker_set.build,
                             _actor_info(actor_lookup, marker.handle) if marker.type == 0 else None)
    ]
    if found and found[-1] > 0xFFFF:
        raise FormatError(f"marker index {found[-1]} does not fit the u16 special list")
    return found


def _slot(tree: L.SnoTree, record: L.Record, offset: int) -> L.PairSlot:
    for slot in L.pair_slots(tree.build, tree.ext, record.name):
        if slot.offset == offset:
            return slot
    raise FormatError(f"{record.name} has no pair at +0x{offset:X}")


def set_special_indices(data: bytes, indices: Iterable[int], build: str | None = None) -> bytes:
    """Write a special list into a marker set (u16 list at body 0xB4, count at 0xB0) through the
    generic tree; the file is laid out again by W11. The bounds at body 0x90 are kept (OPEN)."""
    tree = L.parse_tree(data, "mrk", build)
    values = sorted(indices)
    order = build_order(tree.build)
    slot = _slot(tree, tree.top, SPECIAL_PAIR_AT)
    L.set_block(tree, slot.path, struct.pack(f"{order}{len(values)}H", *values))
    struct.pack_into(order + "I", tree.top.raw, SPECIAL_COUNT_AT, len(values))
    return L.write_tree(tree)


def derive_special_indices(data: bytes, actor_lookup=None, build: str | None = None) -> bytes:
    """W14-18 recomputed and written back. A set whose list is already right is returned as is."""
    marker_set = parse_marker_set(data, build)
    values = special_indices(marker_set, actor_lookup)
    if values == marker_set.special_indices:
        return bytes(data)
    return set_special_indices(data, values, build)


# --- W14-19. World cluster counter -----------------------------------------------------------

SERVER_PAIR_AT = 0x10  # world body
NEXT_CLUSTER_AT = 0x38  # server block


def cluster_counter(entries: Iterable[SceneEntry | int], stored: int | None = None) -> int:
    """W14-19 as corrected by V16-5: highest cluster id of the scene entries + 1 (0 when all are
    -1 or there are none); a ``stored`` value that is not lower than this is kept, however much
    larger (values up to 20 above were seen)."""
    ids = [e if isinstance(e, int) else e.cluster_id for e in entries]
    value = max(ids, default=-1) + 1
    value = max(value, 0)
    return value if stored is None else max(value, stored)


def derive_cluster_counter(world_data: bytes, build: str | None = None, keep_larger: bool = True) -> bytes:
    """W14-19 (SUPPORTED) written into the server block +0x38. A world without a server block or
    without scene entries is returned unchanged."""
    world = parse_world(world_data, build)
    if world.server is None or not world.scene_entries:
        return bytes(world_data)
    stored = world.server.next_cluster_id
    value = cluster_counter(world.scene_entries, stored if keep_larger else None)
    if value == stored:
        return bytes(world_data)
    view = ByteView(world_data, world.header.order)
    block = view.pair(BODY + SERVER_PAIR_AT).file_offset
    out = bytearray(world_data)
    struct.pack_into(world.header.order + "i", out, block + NEXT_CLUSTER_AT, value)
    return bytes(out)


# --- W14-21. Actor bounds from the appearance ------------------------------------------------

APP_SPHERE_AT = 0x10 + 0x80  # appearance body: structure (inline at 0x10) +0x80
APP_BOX_AT = 0x10 + 0x118  # structure +0x118
ACTOR_SPHERE_AT = 0x30  # actor body: centre, radius at 0x3C
ACTOR_BOX_AT = 0x40  # 6 floats, 0x40..0x57
RADIUS_MARGIN = 2.0


def _f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def actor_appearance(actor_data: bytes, build: str | None = None) -> int:
    """The appearance handle an actor names (body 0x14)."""
    header = parse_header(actor_data, build)
    return ByteView(actor_data, header.order).i32(BODY + ACTOR_APPEARANCE_AT, "actor appearance")


def derive_actor_bounds(actor_data: bytes, appearance: bytes, build: str | None = None,
                        keep_larger_radius: bool = True, appearance_changed: bool = True) -> bytes:
    """W14-21 (SUPPORTED): actor body 0x40..0x57 = the appearance's box (structure +0x118), the
    sphere centre (body 0x30) = the appearance sphere centre (structure +0x80), the radius =
    f32(appearance radius + 2.0), a larger stored radius kept unless ``keep_larger_radius`` is
    False. The cylinder (0x1C..0x2F) is authored and not touched.

    V16-7: with ``appearance_changed=False`` the actor is returned unchanged (stored radius, box and
    centre kept: some differ from the rule by 1 ulp)."""
    if not appearance_changed:
        return bytes(actor_data)
    a_header = parse_header(actor_data, build)
    p_header = parse_header(appearance, build)
    a_order, p_order = a_header.order, p_header.order
    pv = ByteView(appearance, p_order)
    pv.need(0, BODY + APP_BOX_AT + 24, "appearance structure")
    ByteView(actor_data, a_order).need(0, BODY + ACTOR_BOX_AT + 24, "actor bounds")
    out = bytearray(actor_data)
    if a_order == p_order:  # copy the floats bit for bit
        out[BODY + ACTOR_BOX_AT:BODY + ACTOR_BOX_AT + 24] = appearance[BODY + APP_BOX_AT:BODY + APP_BOX_AT + 24]
        out[BODY + ACTOR_SPHERE_AT:BODY + ACTOR_SPHERE_AT + 12] = \
            appearance[BODY + APP_SPHERE_AT:BODY + APP_SPHERE_AT + 12]
    else:
        struct.pack_into(a_order + "6f", out, BODY + ACTOR_BOX_AT, *pv.f32s(BODY + APP_BOX_AT, 6))
        struct.pack_into(a_order + "3f", out, BODY + ACTOR_SPHERE_AT, *pv.f32s(BODY + APP_SPHERE_AT, 3))
    radius = _f32(pv.f32(BODY + APP_SPHERE_AT + 12) + RADIUS_MARGIN)
    stored = struct.unpack_from(a_order + "f", actor_data, BODY + ACTOR_SPHERE_AT + 12)[0]
    if keep_larger_radius and stored > radius:
        radius = stored
    if radius != stored or not keep_larger_radius:
        struct.pack_into(a_order + "f", out, BODY + ACTOR_SPHERE_AT + 12, radius)
    return bytes(out)


def update_actors_for_appearance(actors: Mapping, appearance_handle: int, appearance: bytes,
                                 keep_larger_radius: bool = True) -> dict:
    """W14-21: every actor (a mapping of any key to ``.acr`` bytes) that names the appearance,
    with its bounds derived; actors that do not name it are left out of the result."""
    return {
        key: derive_actor_bounds(data, appearance, keep_larger_radius=keep_larger_radius)
        for key, data in actors.items() if actor_appearance(data) == appearance_handle
    }


# --- W14-22. Sound-bank rows -----------------------------------------------------------------

# Row offsets: (row size, decompressed size, stored size)
_BANK_ROW = {"switch": (16, 0x08, 0x0C), "x360": (16, 0x08, 0x0C), "ps3": (40, 0x14, 0x20)}
_BANK_TABLE_AT = 0x0C  # body: sample table pair
FSB_MODE_VORBIS = 15
FSB_MODE_FRAMES = 16


def fsb_decompressed_size(fsb_bytes: bytes) -> int | None:
    """W14-22: 2 x samples x channels from the FSB5 header for modes 15 and 16 (mode 16: the
    original sample length is not in the file, so up to 510 bytes per channel too large);
    None for anything else (not FSB5, e.g. a switch ``.ogg``, or another mode)."""
    if len(fsb_bytes) < 0x44 or fsb_bytes[:4] != b"FSB5":
        return None
    mode = struct.unpack_from("<I", fsb_bytes, 24)[0]
    if mode not in (FSB_MODE_VORBIS, FSB_MODE_FRAMES):
        return None
    word = struct.unpack_from("<Q", fsb_bytes, 60)[0]
    channels = 2 if word >> 5 & 1 else 1
    return 2 * (word >> 34) * channels


def update_bank_row(bank: bytes, index: int, fsb_bytes: bytes | None, build: str | None = None,
                    decompressed: bool = True) -> bytes:
    """W14-22: row ``index`` of a ``.sbk`` gets stored size = ``len(fsb_bytes)`` (PROVEN) and, with
    ``decompressed``, the decompressed size of ``fsb_decompressed_size``. With no sample file
    (``fsb_bytes`` None) the row is kept. Hash, compression, quality and format are inputs."""
    header = parse_header(bank, build)
    resolved = resolve_build("sbk", header, build)
    size, decompressed_at, stored_at = _BANK_ROW[resolved]
    view = ByteView(bank, header.order)
    pair = view.pair(BODY + _BANK_TABLE_AT, "sample table pair")
    if not 0 <= index < pair.size // size:
        raise FormatError(f"row {index} of {pair.size // size}")
    if fsb_bytes is None:
        return bytes(bank)
    at = pair.file_offset + index * size
    out = bytearray(bank)
    struct.pack_into(header.order + "I", out, at + stored_at, len(fsb_bytes))
    value = fsb_decompressed_size(fsb_bytes) if decompressed else None
    if value is not None:
        struct.pack_into(header.order + "I", out, at + decompressed_at, value)
    return bytes(out)


# --- W14-23. Font kerning order --------------------------------------------------------------

KERNING_RECORD = "FontConsoleKerning"
POINT_SIZE_RECORD = "FontPointSizeData"


def _array_slot(tree: L.SnoTree, record: L.Record, element: str) -> L.PairSlot:
    for slot in L.pair_slots(tree.build, tree.ext, record.name):
        if slot.field.kind == "array" and slot.field.element == element:
            return slot
    raise FormatError(f"{record.name} has no list of {element}")


def _point_sizes(tree: L.SnoTree) -> tuple[L.PairSlot, list[L.Record]]:
    slot = _array_slot(tree, tree.top, POINT_SIZE_RECORD)
    value = tree.top.pairs.get(slot.offset)
    elements = [] if value is None or value.block is None else (value.block.elements or [])
    return slot, elements


def _kernings(tree: L.SnoTree, size: L.Record) -> tuple[L.PairSlot, list[L.Record]]:
    slot = _array_slot(tree, size, KERNING_RECORD)
    value = size.pairs.get(slot.offset)
    elements = [] if value is None or value.block is None else (value.block.elements or [])
    return slot, elements


def kerning_key(record: L.Record, order: str) -> tuple[int, int]:
    """(first code, second code) of a 16-byte kerning record (codes, then X and Y offsets)."""
    return struct.unpack_from(order + "II", record.raw, 0)


def kerning_pairs(tree: L.SnoTree, point_size: int) -> list[tuple[int, int, int, int]]:
    """(first, second, x, y) of every kerning pair of one point size, in stored order."""
    order = build_order(tree.build)
    _, sizes = _point_sizes(tree)
    _, pairs = _kernings(tree, sizes[point_size])
    return [struct.unpack_from(order + "IIii", p.raw, 0) for p in pairs]


def insert_kerning(tree: L.SnoTree, point_size: int, first: int, second: int, x: int = 0, y: int = 0,
                   replace: bool = True) -> int:
    """W14-23: add a kerning pair to a font tree (``sno_layout.parse_tree(data, "fnt")``) at its
    (first, second) place; an existing pair with the same codes gets the new offsets (or, without
    ``replace``, a FormatError). The kerning count word is set (W14-20). Returns the index."""
    order = build_order(tree.build)
    size_slot, sizes = _point_sizes(tree)
    if not 0 <= point_size < len(sizes):
        raise FormatError(f"point size {point_size} of {len(sizes)}")
    size = sizes[point_size]
    slot, pairs = _kernings(tree, size)
    keys = [kerning_key(p, order) for p in pairs]
    if any(b < a for a, b in zip(keys, keys[1:])):
        raise FormatError(f"point size {point_size}: kerning pairs are not in (first, second) order")
    key = (first & 0xFFFFFFFF, second & 0xFFFFFFFF)
    place = bisect.bisect_left(keys, key)
    record = L.new_record(tree.build, tree.ext, KERNING_RECORD)
    struct.pack_into(order + "IIii", record.raw, 0, key[0], key[1], x, y)
    pairs = list(pairs)
    if place < len(keys) and keys[place] == key:
        if not replace:
            raise FormatError(f"point size {point_size}: pair {key} exists")
        pairs[place] = record
    else:
        pairs.insert(place, record)
    path = f"{size_slot.path}[{point_size}].{slot.path}"
    L.set_elements(tree, path, pairs, update_counts=True)
    return place


def kerning_order_problems(tree: L.SnoTree) -> list[str]:
    """W14-23 check: point sizes whose kerning pairs are not sorted by (first, second)."""
    order = build_order(tree.build)
    found = []
    _, sizes = _point_sizes(tree)
    for i, size in enumerate(sizes):
        _, pairs = _kernings(tree, size)
        keys = [kerning_key(p, order) for p in pairs]
        bad = next((k for k in range(1, len(keys)) if keys[k] < keys[k - 1]), None)
        if bad is not None:
            found.append(f"point size {i}: kerning pair {bad} {keys[bad]} sorts before {keys[bad - 1]}")
    return found


__all__ = [
    "actor_appearance", "actor_gizmo_info", "cluster_counter", "count_links", "derive_actor_bounds",
    "derive_cluster_counter", "derive_counts", "derive_special_indices", "fsb_decompressed_size",
    "insert_kerning", "is_special_marker", "kerning_key", "kerning_order_problems", "kerning_pairs",
    "set_special_indices", "special_indices", "update_actors_for_appearance", "update_bank_row",
    "update_count_words",
]
