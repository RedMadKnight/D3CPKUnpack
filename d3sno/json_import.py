"""Stage 22 (``49_stage22_json_import.md``): importing an edited JSON record, world or collision file onto
its template (W22-1 to W22-9).

W22-1: the template is exported again in memory (``template_json``), compared with the edited JSON leaf by
leaf (``compare``), and only the leaves that differ are applied.

W22-2: ``places(ext, data, build)`` gives, for a JSON path of the template's export, the stored place of
the leaf (a ``Place``: file offset and stored type) or nothing (a derived leaf). Each place builder below
reads the file with the same parser and the same offsets as the exporter of its group (``export``), and
``classify`` names every leaf of an export as stored, derived or fixed, so a test can check that the two
do not drift apart.

The stored changes are written through the generic tree (``sno_layout``): each place is a file offset of
the template; ``TreePatch`` finds the tree record or block that holds it, writes the new bytes there and
lays the file out with ``write_tree(derive=True)``. W22-5: a changed derived leaf is accepted only when a
new export of the written file gives the same value (it was edited consistently).

Collision (W22-9) goes through ``collision_write``; navigation in a separate ``.nav`` through
``children.write_nav`` (W22-8). Animations (W22-10) are in ``anim_import``.
"""

from __future__ import annotations

import copy
import json
import math
import struct
from dataclasses import dataclass, field

from . import collision, collision_write, effects, enums, export, material, navigation, records, world
from . import sno_layout as L
from .binary import ByteView, FormatError, build_order
from .power import TAG_FLOAT
from .sno import group_by_extension, group_by_id
from .transform import quat_normalise

__all__ = [
    "GROUPS", "LIST_RULES", "Change", "ImportResult", "Place", "REASONS", "Refused", "classify", "compare",
    "import_json", "path_text", "places", "template_json",
]

GROUPS = ("acr", "mon", "mat", "lvl", "scn", "wrl", "mrk", "phm") + effects.GROUPS  # ``55`` W23-8
# W22-3 to W22-10, W22-11: the reasons of a refused leaf or entry
REASONS = ("import_failed", "value_out_of_range", "unknown_name", "conflicting_values", "wrong_group", "list_length_fixed",
           "derived_field", "navigation_geometry", "skeleton_mismatch", "duration_changed",
           "keyframe_count_mismatch", "permutation_count", "field_fixed", "missing_shader", "bad_json",
           "unsupported_value", "not_canonical")
FIELDS_LISTED = 200  # W22-11: changed paths listed per entry
FLOAT_TOLERANCE = 1e-6
QUATERNION_TOLERANCE = 1e-6
_CODES = {"i": "i", "u": "I", "f": "f", "h": "e", "b": "B", "s": "h", "w": "H"}


# --- paths and differences (W22-1) ----------------------------------------------------------------

def path_text(path) -> str:
    """``a.b[3].c`` (W22-1)."""
    out = ""
    for part in path:
        if isinstance(part, int):
            out += f"[{part}]"
        else:
            out += ("." if out else "") + str(part)
    return out


@dataclass
class Place:
    """W22-2: where a leaf (or a compound value: enum, flags, link, rotation, bytes, tag map) is stored.

    ``kind``: ``num`` (``code``, a struct code), ``enum`` / ``flags`` (``code``, ``names``: name ->
    value or bit), ``link`` (``group``: the field's group extension; ``group_at``: the offset of a stored
    group word, or None), ``rotation`` (4 float32, xyzw), ``string`` (``size`` bytes), ``bytes``
    (``size`` bytes), ``text`` (a text pair at ``at``), ``tagmap`` (a tag map pair at ``at``;
    ``resizable``), ``fixed`` (stored, not to be changed here: ``field_fixed``), ``count`` (a count word,
    derived), ``nav`` (navigation: ``field`` names what), ``geometry`` (navigation geometry)."""

    kind: str
    at: int = -1
    code: str = "i"
    group: str | None = None
    group_at: int | None = None
    size: int = 0
    names: dict | None = None
    resizable: bool = False
    field: str = ""
    message: str = ""

    @property
    def compound(self) -> bool:
        return self.kind in ("enum", "flags", "link", "rotation", "bytes", "tagmap", "string", "text", "fixed") or (
            self.kind == "nav" and self.field == "cell_flags")


@dataclass
class Change:
    path: tuple
    old: object
    new: object
    place: Place | None = None
    kind: str = "leaf"  # leaf (or compound) / list


class Refused(Exception):
    def __init__(self, path, reason: str, message: str = ""):
        super().__init__(message or reason)
        self.path, self.reason, self.message = tuple(path), reason, message

    def as_row(self) -> dict:
        row = {"path": path_text(self.path), "reason": self.reason}
        if self.message:
            row["message"] = self.message
        return row


def compare(old, new, places: dict, path: tuple = (), ignore=None) -> list[Change]:
    """W22-1, W22-4: the changes from ``old`` (the template's export) to ``new`` (the edited JSON). A
    compound place is compared as one value; lists of another length are one ``list`` change.
    ``51`` AB2: ``ignore(path)`` names the export diagnostics, never compared (present or not)."""
    out: list[Change] = []

    def walk(a, b, at: tuple) -> None:
        place = places.get(at)
        if place is not None and place.compound:
            if not _same_json(a, b):
                out.append(Change(at, a, b, place))
            return
        if isinstance(a, dict) and isinstance(b, dict):
            keys_a = {k for k in a if ignore is None or not ignore(at + (k,))}
            keys_b = {k for k in b if ignore is None or not ignore(at + (k,))}
            if keys_a != keys_b:
                out.append(Change(at, sorted(keys_a), sorted(keys_b), None, "keys"))
                return
            for key in sorted(keys_a):
                walk(a[key], b[key], at + (key,))
            return
        if isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                out.append(Change(at, a, b, place, "list"))
                return
            for i, (x, y) in enumerate(zip(a, b)):
                walk(x, y, at + (i,))
            return
        if not _same_json(a, b):
            out.append(Change(at, a, b, place))

    walk(old, new, path)
    return out


def _same_json(a, b) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a == b and _same_zero(a, b)
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return set(a) == set(b) and all(_same_json(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(_same_json(x, y) for x, y in zip(a, b))
    return a == b


def _same_zero(a, b) -> bool:
    """``61`` AI2: two floats are the same value only when their float32 bits are equal, so 0.0 and -0.0 differ
    (an integer 0 is +0)."""
    if a == 0 and b == 0:
        return math.copysign(1.0, a) == math.copysign(1.0, b)
    return True


def _close(a, b, tolerance: float = FLOAT_TOLERANCE) -> bool:
    """JSON values equal, floats as float32 (``export.plain``) within ``tolerance``."""
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b and type(a) is type(b)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if isinstance(a, int) and isinstance(b, int):
            return a == b
        fa, fb = float(export.plain(float(a))), float(export.plain(float(b)))
        if not _same_zero(fa, fb):
            return False  # ``61`` AI2: the sign of a zero must match
        return fa == fb or abs(fa - fb) <= tolerance * max(1.0, abs(fa), abs(fb))
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(_close(a[k], b[k], tolerance) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_close(x, y, tolerance) for x, y in zip(a, b))
    return a == b


def get_path(document, path):
    value = document
    for part in path:
        value = value[part]
    return value


# --- writing through the generic tree (W22-2) --------------------------------------------------------

class TreePatch:
    """The template's tree with its records and blocks located by file offset. ``write`` puts bytes
    at a template offset; ``set_pair`` gives the pair field at a template offset a new block;
    ``elements`` gives the record array behind a pair field new elements. ``bytes`` lays the tree out
    with ``write_tree(derive=True)``. A file that the tree does not rebuild byte for byte
    (``canonical`` False) is patched in place instead (same-length writes only)."""

    def __init__(self, data: bytes, ext: str, build: str) -> None:
        self.data = bytes(data)
        self.flat = bytearray(data)
        self.tree = L.parse_tree(self.data, ext, build)
        self.canonical = L.write_tree(self.tree) == self.data
        self.tree = L.parse_tree(self.data, ext, build)
        self.segments: list[tuple[int, int, object]] = [(L.BODY, len(self.tree.top.raw), self.tree.top)]
        self.owners: dict[int, L.Block] = {}  # id(holder) -> the block it lies in
        self.copies: list[L.Record] = []  # ``51`` AB4: new list elements sharing their base's blocks
        self.refs: dict[int, int] = {}  # ``53`` AD2: id(block) -> the pairs of the template that point at it
        seen: set[int] = set()
        self.pairs: dict[int, tuple[L.Record, int]] = {}  # file offset of a pair field -> (record, slot)

        def walk(record: L.Record, position: int) -> None:
            for slot in L.pair_slots(self.tree.build, self.tree.ext, record.name):
                self.pairs[position + slot.offset] = (record, slot.offset)
                value = record.pairs.get(slot.offset)
                if value is not None and value.block is not None:
                    self.refs[id(value.block)] = self.refs.get(id(value.block), 0) + 1
                if value is None or value.block is None or value.shared or id(value.block) in seen:
                    continue
                block = value.block
                seen.add(id(block))
                if block.source is None:
                    continue
                start = L.BODY + block.source
                if block.elements is None:
                    self.segments.append((start, len(block.data or b""), block))
                    self.owners[id(block)] = block
                    continue
                at = start
                for element in block.elements:
                    self.segments.append((at, len(element.raw), element))
                    self.owners[id(element)] = block
                    walk(element, at)
                    at += len(element.raw)

        walk(self.tree.top, L.BODY)
        self.structural = False

    def holder(self, at: int, size: int):
        for start, length, holder in self.segments:
            if start <= at and at + size <= start + length:
                return holder, at - start
        raise FormatError(f"no stored record or block holds 0x{at:X}+{size}")

    def write(self, at: int, data: bytes) -> None:
        self.flat[at:at + len(data)] = data
        if not self.canonical:
            return  # written in place (``bytes``)
        holder, rel = self.holder(at, len(data))
        owner = self.owners.get(id(holder))
        if owner is not None:  # ``51`` AB4: a copy that shares this block keeps the block as it was
            for record in self.copies:
                for slot, value in list(record.pairs.items()):
                    if value is not None and value.shared and value.block is owner:
                        record.pairs[slot] = L.PairValue(copy.deepcopy(owner), value.size, value.stored)
        if isinstance(holder, L.Record):
            if rel + len(data) > len(holder.raw):  # ``61`` AI1: a record keeps its length
                raise FormatError(f"a write of {len(data)} bytes at +{rel} runs past a {len(holder.raw)}-byte record")
            holder.raw[rel:rel + len(data)] = data
        else:
            raw = bytearray(holder.data)
            raw[rel:rel + len(data)] = data
            holder.data = bytes(raw)

    def set_pair(self, at: int, data: bytes) -> None:
        record, slot = self.pairs[at]
        record.pairs[slot] = L.PairValue(L.Block(bytes(data)) if data else None)
        self.structural = True

    def element_record(self, at: int) -> L.Record:
        holder, rel = self.holder(at, 1)
        if not isinstance(holder, L.Record) or rel:
            raise FormatError(f"0x{at:X} is not the start of a record")
        return holder

    def elements(self, pair_at: int, elements: list) -> None:
        """The record array behind a pair: the new elements, then the bytes the stored array holds after its
        last whole element (``52`` AC4: kept the same way on every path, so a list change undone gives the
        stored bytes); every offset and count word follows from the new content."""
        record, slot = self.pairs[pair_at]
        before = record.pairs.get(slot)
        extra = before.block.extra if before is not None and before.block is not None else b""
        record.pairs[slot] = L.PairValue(L.Block(elements=list(elements), extra=extra) if elements else None)
        L.update_count_words(self.tree, record, slot)
        self.structural = True

    def bytes(self) -> bytes:
        if not self.canonical:
            if self.structural:
                raise Refused((), "not_canonical", "the template is not laid out as the tree writer lays it "
                                                  "out: only same-length values can be written")
            return bytes(self.flat)
        return L.write_tree(self.tree, derive=True)


# --- place builders (W22-2), one per exporter ------------------------------------------------------

def _record_places(record: records.Record, data: bytes, order: str, prefix: tuple, out: dict) -> None:
    """``export._record``: every field of a layout record by its name."""
    spec = records.layout(record.build or "switch", record.type)
    view = ByteView(data, order)
    for item in spec.fields:
        at = record.position + item.offset
        kind = item.type
        if kind.startswith("pointer_slot"):
            continue
        if kind.startswith("file_ref,"):
            for k, name in enumerate(item.name.split(",")):
                out[prefix + (name,)] = Place("num", at + 4 * k, "I" if k == 2 else "i")
            continue
        path = prefix + (item.name,)
        ref = export._REF.match(kind)
        if kind.startswith("typed_ref"):
            out[path] = Place("link", at + 4, "i", None, group_at=at)
        elif kind.startswith("char["):
            out[path] = Place("string", at, size=item.size)
        elif kind == "zero":
            out[path] = Place("bytes", at, size=item.size)
        elif kind.startswith("text_pair"):
            out[path] = Place("text", at)
        elif kind.startswith("tag_map_pair"):
            out[path] = Place("tagmap", at, resizable=record.type in ("actor", "monster"))
        elif kind.startswith("array_pair"):
            element = export_array_element(kind)
            pair = view.pair(at)
            value = record.values.get(item.name)
            if element == "u8":
                out[path] = Place("bytes", pair.file_offset, size=pair.size)
            elif element in ("i32", "u32", "f32"):
                code = {"i32": "i", "u32": "I", "f32": "f"}[element]
                for i in range(pair.size // 4):
                    out[path + (i,)] = Place("num", pair.file_offset + 4 * i, code)
            else:
                for i, sub in enumerate(value or ()):
                    _record_places(sub, data, order, path + (i,), out)
        elif _COUNT_WORD.match(kind):
            out[path] = Place("count", at, "i")
        elif ref and item.size == 4:
            group = group_by_id(int(ref.group(1)))
            out[path] = Place("link", at, "i", group.extension if group else ref.group(1))
        elif "enum" in kind:
            out[path] = Place("enum", at, "i", names=_enum_names(record.type, item.name))
        elif "flags" in kind:
            out[path] = Place("flags", at, "I", names=_flag_bits(record.type, item.name))
        else:
            code = "f" if kind.startswith("f32") else "I" if kind.startswith("u32") else "i"
            if item.count:
                for i in range(item.count):
                    out[path + (i,)] = Place("num", at + 4 * i, code)
            else:
                out[path] = Place("num", at, code)


_COUNT_WORD = records._COUNT


def export_array_element(kind: str) -> str:
    match = records._ARRAY.match(kind)
    return match.group(1) if match else ""


def _enum_names(record: str, name: str) -> dict:
    values = enums.record_enums().get((record, records._table_field(record, name)), {})
    return {row.name: value for value, row in values.items() if getattr(row, "name", None)}


def _flag_bits(record: str, name: str) -> dict:
    bits = enums.record_enums().get((record, records._table_field(record, name)), {})
    return {row.name: bit for bit, row in bits.items() if getattr(row, "name", None)
            and getattr(row, "status", None) != enums.NOT_KNOWABLE}


def _alias(out: dict, alias: tuple, target: tuple) -> None:
    """A second JSON path of a stored value (the export shows some fields twice)."""
    for path, place in list(out.items()):
        if path[:len(target)] == target:
            out[alias + path[len(target):]] = place


def _actor_places(data: bytes, build: str) -> dict:
    found = records.parse_record_file(data, "acr", build)
    out: dict = {}
    _record_places(found.record, data, found.header.order, ("fields",), out)
    for alias, name in (("appearance", "appearance_ref"), ("anim_set", "anim_set_ref"),
                        ("collision", "physics_mesh_ref"), ("monster", "monster_ref")):
        _alias(out, ("links", alias), ("fields", name))
    for name in ("bound_cylinder_base", "bound_cylinder_height", "bound_cylinder_radius", "bound_sphere_center",
                 "bound_sphere_radius", "bound_box_min", "bound_box_extent"):
        _alias(out, ("bounds", name), ("fields", name))
    _alias(out, ("actor_type",), ("fields", "actor_kind"))
    return out


def _monster_places(data: bytes, build: str) -> dict:
    found = records.parse_record_file(data, "mon", build)
    out: dict = {}
    _record_places(found.record, data, found.header.order, ("fields",), out)
    _alias(out, ("links", "actor"), ("fields", "actor_ref"))
    return out


def _material_places(data: bytes, build: str) -> dict:
    """W22-6: texture links, numeric parameters and the shader link; the rest is ``fixed``."""
    mat = material.parse_material(data, build)
    block = mat.block
    at = 0x10 + material._BLOCK_AT
    out: dict = {("shader",): Place("link", at, "i", "shm")}
    if mat.build != "ps3":
        for i, name in enumerate(("ambient", "diffuse", "specular", "emissive")):
            for k in range(4):
                out[("parameters", name, "rgba", k)] = Place("num", at + 0x04 + i * 16 + 4 * k, "f")
            out[("parameters", name, "scale")] = Place("fixed", message="switch and x360 store no colour scale")
        tail = 0x44
    else:
        for i, name in enumerate(("ambient", "diffuse", "specular", "emissive")):
            for k in range(4):
                out[("parameters", name, "rgba", k)] = Place("num", at + 0x04 + i * 8 + k, "B")
            out[("parameters", name, "scale")] = Place("num", at + 0x08 + i * 8, "f")
        tail = 0x24
    out[("parameters", "power")] = Place("num", at + tail, "f")
    out[("parameters", "sort_priority_raw")] = Place("num", at + tail + 4, "I")
    for i, slot in enumerate(block.textures):
        out[("textures", i, "texture")] = Place("link", slot.position, "i", "tex")
        out[("textures", i, "slot")] = Place("fixed", message="W22-6: only the texture of a slot may change")
        out[("textures", i, "animation_raw")] = Place("fixed", message="W22-6")
    out[("flags",)] = Place("fixed", message="W22-6")
    out[("tag_map",)] = Place("fixed", message="W22-6")
    return out


MARKER_TYPE_NAMES = {name: value for value, name in world.MARKER_TYPES.items()}


def _marker_places(at: int, out: dict, prefix: tuple) -> None:
    """``export._marker_json`` of the marker stored at ``at``."""
    out[prefix + ("name_hash",)] = Place("num", at, "I")
    out[prefix + ("type",)] = Place("enum", at + 8, "I", names=MARKER_TYPE_NAMES)
    out[prefix + ("rotation",)] = Place("rotation", at + 0x0C)
    for k in range(3):
        out[prefix + ("position", k)] = Place("num", at + 0x1C + 4 * k, "f")
    out[prefix + ("link",)] = Place("link", at + 0x2C, "i", None, group_at=at + 0x28)
    out[prefix + ("tags",)] = Place("tagmap", at + 0x30)


def _marker_set_places(data: bytes, build: str) -> dict:
    found = world.parse_marker_set(data, build)
    view = ByteView(data, found.header.order)
    out: dict = {}
    markers = view.pair(0x10 + 0x0C)
    for i in range(len(found.markers)):
        at = markers.file_offset + i * world.MARKER_SIZE
        _marker_places(at, out, ("markers", i))
        links = view.pair(at + 0x44)
        for k in range(links.size // world.LINK_SIZE):
            for j in range(2):
                out[("markers", i, "links", k, j)] = Place("num", links.file_offset + 8 * k + 4 * j, "I")
    circles = view.pair(0x10 + 0x50)
    for i in range(circles.size // world.CIRCLE_SIZE):
        for k in range(3):
            out[("circles", i, k)] = Place("num", circles.file_offset + 12 * i + 4 * k, "f")
    for side in range(2):
        for k in range(3):
            out[("bounds", side, k)] = Place("num", 0x10 + 0x90 + 12 * side + 4 * k, "f")
    out[("flags",)] = Place("num", 0x10 + 0xA8, "I")
    out[("label",)] = Place("num", 0x10 + 0xAC, "i")
    special = view.pair(0x10 + 0xB4)
    for i in range(special.size // 2):
        out[("special_indices", i)] = Place("num", special.file_offset + 2 * i, "H")
    return out


def _u32_list_places(view: ByteView, body: int, out: dict, path: tuple, group: str | None = None) -> None:
    pair = view.pair(0x10 + body)
    for i in range(pair.size // 4):
        at = pair.file_offset + 4 * i
        out[path + (i,)] = Place("link", at, "I", group) if group else Place("num", at, "I")


def _scene_places(data: bytes, build: str, nav: bytes | None) -> dict:
    info = world.parse_scene_info(data, build)
    view = ByteView(data, info.header.order)
    out: dict = {}
    for name, body in (("bounds", 0x10), ("marker_bounds", 0x28)):
        for side in range(2):
            for k in range(3):
                out[(name, side, k)] = Place("num", 0x10 + body + 12 * side + 4 * k, "f")
    _u32_list_places(view, 0xA0, out, ("exclusions",))
    _u32_list_places(view, 0xE0, out, ("inclusions",))
    _u32_list_places(view, 0x120, out, ("marker_sets",), "mrk")
    out[("look_hash",)] = Place("num", 0x10 + 0x160, "I")
    if info.build == "switch":
        out[("xp_density",)] = Place("num", 0x10 + 0x208, "f")
        out[("appearance",)] = Place("link", 0x10 + 0x20C, "i", "app")
        out[("collision",)] = Place("link", 0x10 + 0x210, "i", "phm")
    else:
        out[("appearance",)] = Place("link", 0x10 + 0x210, "i", "app")
        out[("collision",)] = Place("link", 0x10 + 0x214, "i", "phm")
    navi = navigation.parse_scene(data, info.build, nav)
    for i, cell in enumerate(navi.cells):
        out[("navigation", "cells", i, "flags")] = Place("nav", i, field="cell_flags")
        for j in range(cell.neighbour_count):
            out[("navigation", "cells", i, "neighbours", j, "cell")] = Place("nav", cell.first_neighbour + j,
                                                                             field="neighbour_cell")
    for k in range(len(navi.neighbours)):
        out[("navigation", "neighbours", k, "cell")] = Place("nav", k, field="neighbour_cell")
    return out


def _world_places(data: bytes, build: str) -> dict:
    wrl = world.parse_world(data, build)
    order = wrl.header.order
    view = ByteView(data, order)
    out: dict = {("is_generated",): Place("num", 0x10 + 0x0C, "I")}
    _u32_list_places(view, 0x20, out, ("marker_sets",), "mrk")
    skybox, music, ambient, reverb, weather = (
        (0x84, 0x88, 0x98, 0x9C, 0xA0) if wrl.build != "ps3" else (0xD4, 0xD8, 0xDC, 0xE0, 0xE4))
    if wrl.build != "ps3":
        out[("skybox_actor",)] = Place("link", 0x10 + skybox, "i", "acr")
    out[("skybox_word",)] = Place("num", 0x10 + skybox, "i")
    for name, at, group in (("music", music, "mus"), ("ambient", ambient, "amb"), ("reverb", reverb, "rev"),
                            ("weather", weather, "wth")):
        out[(name,)] = Place("link", 0x10 + at, "i", _ext_of(name, group))
    server = view.pair(0x10 + 0x10)
    if server.size:
        base = server.file_offset
        gens = view.pair(base)
        tile_size = world._TILE_SIZE[wrl.build]
        for g in range(gens.size // world._GENERATOR_SIZE):
            at = gens.file_offset + g * world._GENERATOR_SIZE
            prefix = ("server", "generators", g)
            out[prefix + ("raw",)] = Place("bytes", at, size=world._GENERATOR_SIZE)
            out[prefix + ("command_count",)] = Place("num", at + 0x40, "i")
            commands = view.pair(at + 0x44)
            out[prefix + ("commands_raw",)] = Place("bytes", commands.file_offset, size=commands.size)
            tiles = view.pair(at)
            for t in range(tiles.size // tile_size):
                for j, name in enumerate(("exits", "tile_set", "scene", "weight")):
                    out[prefix + ("tiles", t, name)] = Place("num", tiles.file_offset + t * tile_size + 4 * j, "i")
        lists = view.pair(base + 0x10)
        flat = 0
        size = world._ENTRY_SIZE[wrl.build]
        for j in range(lists.size // world._LIST_SIZE):
            chunk = view.pair(lists.file_offset + j * world._LIST_SIZE)
            for k in range(chunk.size // size):
                at = chunk.file_offset + k * size
                entry: dict = {}
                _entry_places(at, wrl.build, entry)
                for path, place in entry.items():
                    out[("server", "scene_lists", j, k) + path] = place
                    out[("scene_entries", flat) + path] = place
                flat += 1
        out[("server", "next_cluster_id")] = Place("num", base + 0x38, "i")
        out[("server", "cluster_count")] = Place("num", base + 0x40, "i")
        clusters = view.pair(base + 0x50)
        out[("server", "clusters_raw")] = Place("bytes", clusters.file_offset, size=clusters.size)
        for k in range(4):
            out[("server", "navigation_powers", k)] = Place("num", base + 0x58 + 4 * k, "i")
        script_at, seed_at = (0x74, 0x78) if wrl.build != "ps3" else (0x68, 0x6C)
        out[("server", "script")] = Place("link", base + script_at, "i", group_by_id(35).extension)
        out[("server", "seed")] = Place("num", base + seed_at, "i")
    return out


def _ext_of(name: str, fallback: str) -> str:
    groups = {"music": 62, "ambient": 5, "reverb": 60, "weather": 47}
    found = group_by_id(groups[name])
    return found.extension if found else fallback


def _entry_places(at: int, build: str, out: dict) -> None:
    """``export._entry_json`` of the scene entry stored at ``at``."""
    out[("scene",)] = Place("link", at + 4, "I", None, group_at=at)
    out[("rotation",)] = Place("rotation", at + 8)
    for k in range(3):
        out[("position", k)] = Place("num", at + 0x18 + 4 * k, "f")
    for k in range(4):
        out[("level_areas", k)] = Place("num", at + 0x30 + 4 * k, "i")
    out[("cluster_id",)] = Place("num", at + (0xAC if build != "ps3" else 0x9C), "i")


def places(ext: str, data: bytes, build: str, nav: bytes | None = None) -> dict:
    """W22-2: JSON path -> ``Place`` for the stored leaves of the template's export."""
    if ext == "acr":
        return _actor_places(data, build)
    if ext == "mon":
        return _monster_places(data, build)
    if ext == "mat":
        return _material_places(data, build)
    if ext == "mrk":
        return _marker_set_places(data, build)
    if ext == "scn":
        return _scene_places(data, build, nav)
    if ext == "wrl":
        return _world_places(data, build)
    if ext == "lvl":
        return {}
    raise FormatError(f"no JSON import for .{ext}")


# W22-2, W22-5: the derived leaves of each export, by path prefix (anything placed is stored)
DERIVED = {
    "*": (("header",), ("build",), ("warnings",)),
    "acr": (),
    "mon": (("attributes",),),
    "mat": (),
    "lvl": (("body",), ("size",)),
    "mrk": (("markers", "*", "scale"), ("always_active",)),
    "scn": (("navigation",), ("xp_density",)),  # ps3 and x360 store no XP density (null)
    "wrl": (("layout",), ("scene_entries", "*", "note"), ("server", "scene_lists", "*", "*", "note")),
}


# ``51`` AB2: the export diagnostics of each group, by path prefix: lists and leaves the export writes only
# to report on the file. They are never compared, written or refused on import; a row that a new export of
# the written file adds is reported as ``export_warning``.
DIAGNOSTICS = {
    "*": (("warnings",),),
    "scn": (("navigation", "warnings"),),
    "wrl": (("scene_entries", "*", "note"), ("server", "scene_lists", "*", "*", "note")),
}


def _matches(path: tuple, pattern: tuple) -> bool:
    return len(path) >= len(pattern) and all(p == "*" or p == q for p, q in zip(pattern, path))


def is_diagnostic(ext: str, path: tuple) -> bool:
    """``51`` AB2: whether ``path`` is (inside) an export diagnostic of group ``ext``."""
    return any(_matches(path, p) for p in (*DIAGNOSTICS["*"], *DIAGNOSTICS.get(ext, ())))


def _diagnostic_values(ext: str, document) -> dict:
    """``51`` AB2: path -> value of every export diagnostic of a document."""
    out: dict = {}

    def walk(value, at: tuple, pattern: tuple) -> None:
        if len(at) == len(pattern):
            out[at] = value
            return
        part = pattern[len(at)]
        if part == "*":
            items = enumerate(value) if isinstance(value, list) else (value.items() if isinstance(value, dict) else ())
            for key, sub in items:
                walk(sub, at + (key,), pattern)
        elif isinstance(value, dict) and part in value:
            walk(value[part], at + (part,), pattern)

    for pattern in (*DIAGNOSTICS["*"], *DIAGNOSTICS.get(ext, ())):
        walk(document, (), pattern)
    return out


def export_warnings(ext: str, before, after) -> list[str]:
    """``51`` AB2: ``export_warning: <path>: <text>`` for each diagnostic row of ``after`` (a new export of
    the written file) that the template's export ``before`` does not have."""
    old = _diagnostic_values(ext, before)
    found = []
    for path, value in sorted(_diagnostic_values(ext, after).items(), key=lambda kv: path_text(kv[0])):
        rows = value if isinstance(value, list) else [value]
        known = old.get(path)
        known = [json.dumps(r, sort_keys=True) for r in (known if isinstance(known, list) else [known])]
        for row in rows:
            text = json.dumps(row, sort_keys=True)
            if row is None or text in known:
                if text in known:
                    known.remove(text)
                continue
            found.append(f"export_warning: {path_text(path)}: {row if isinstance(row, str) else text}")
    return found


def classify(ext: str, path: tuple, found: dict) -> str:
    """``stored``, ``fixed``, ``derived``, ``diagnostic`` (``51`` AB2) or ``unknown`` (a leaf no builder
    names: a drift)."""
    if is_diagnostic(ext, path):
        return "diagnostic"
    for length in range(len(path), -1, -1):
        place = found.get(path[:length])
        if place is not None:
            if place.kind in ("fixed",):
                return "fixed"
            if place.kind in ("count", "geometry"):
                return "derived"
            return "stored"
    if any(_matches(path, p) for p in (*DERIVED["*"], *DERIVED.get(ext, ()))):
        return "derived"
    return "unknown"


# --- the template's export (W22-1) ------------------------------------------------------------------

def template_json(ext: str, data: bytes, build: str, nav: bytes | None = None, scenes=None, sets=None) -> dict:
    """The JSON the export writes for the template (``scenes(h)`` / ``sets(h)``: parsed scenes and marker
    sets of the game, for a world's layout)."""
    if ext == "acr":
        return export.plain(export._actor_json(data, build))
    if ext == "mon":
        return export.plain(export._monster_json(data, build))
    if ext == "mat":
        return export.plain(export._material_json(data, build))
    if ext == "lvl":
        return export.plain(export._level_area_json(data, build))
    if ext == "mrk":
        return export.plain(export._marker_set_json(data, build))
    if ext == "phm":
        return export.plain(export._collision_json(data, build))
    if ext == "scn":
        return export.plain(export._scene_json(data, build, nav)[0])
    if ext == "wrl":
        return export.plain(export._world_json(data, build, scenes or (lambda h: None), sets or (lambda h: None))[0])
    if ext in effects.GROUPS:  # ``55`` W23-7
        return export.plain(effects.decode(data, ext, build))
    raise FormatError(f"no JSON export for .{ext}")


def _json_round(document):
    """As the export writes it and ``json.loads`` reads it back."""
    return json.loads(export.dumps(document).decode("utf-8"))


# --- values (W22-3) ---------------------------------------------------------------------------------

def _pack(code: str, value, path, order: str) -> bytes:
    if code in "fe":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise Refused(path, "value_out_of_range", f"{value!r} is not a number")
        try:
            out = struct.pack(order + code, float(value))
        except (OverflowError, struct.error):
            raise Refused(path, "value_out_of_range", f"{value!r} does not fit a float{'32' if code == 'f' else '16'}")
        if not math.isfinite(struct.unpack(order + code, out)[0]) and math.isfinite(float(value)):
            raise Refused(path, "value_out_of_range", f"{value!r} does not fit a float{'32' if code == 'f' else '16'}")
        return out
    if isinstance(value, bool) or not isinstance(value, int):
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        else:
            raise Refused(path, "value_out_of_range", f"{value!r} is not an integer")
    try:
        return struct.pack(order + code, value)
    except struct.error:
        raise Refused(path, "value_out_of_range", f"{value} does not fit the stored {struct.calcsize(code) * 8}-bit "
                                                  f"{'signed' if code.islower() else 'unsigned'} field")


@dataclass
class _Context:
    build: str
    order: str
    known: object  # (group id, handle) -> bool, or None
    warnings: list = field(default_factory=list)
    nav_changes: dict = field(default_factory=dict)  # ("cell_flags", i) / ("neighbour_cell", k) -> (value, path)
    writes: dict = field(default_factory=dict)  # place -> (bytes, path)


def _enum_value(change: Change, ctx: _Context):
    old, new, place, path = change.old, change.new, change.place, change.path
    if not isinstance(new, dict) or set(new) - {"value", "name"} or "value" not in new:
        raise Refused(path, "bad_json", "an enum is {\"value\", \"name\"}")
    value_changed = new.get("value") != (old or {}).get("value")
    name_changed = new.get("name") != (old or {}).get("name")
    names = place.names or {}
    if name_changed:
        if new.get("name") not in names:
            raise Refused(path, "unknown_name", f"{new.get('name')!r} is not a name of this enum")
        by_name = names[new["name"]]
        if value_changed and new["value"] != by_name:
            raise Refused(path, "conflicting_values", f"value {new['value']} and name {new['name']!r} ({by_name}) disagree")
        return by_name
    return new["value"]


def _flags_value(change: Change, ctx: _Context):
    old, new, place, path = change.old, change.new, change.place, change.path
    if not isinstance(new, dict) or "value" not in new or not isinstance(new.get("names", []), list):
        raise Refused(path, "bad_json", "flags are {\"value\", \"names\"}")
    old = old or {}
    value_changed = new.get("value") != old.get("value")
    names_changed = sorted(new.get("names", [])) != sorted(old.get("names", []))
    if set(new) - {"value", "names"} != set(old) - {"value", "names"} or any(
            new[k] != old.get(k) for k in set(new) - {"value", "names"}):
        raise Refused(path, "derived_field", "only value and names of a flags field may change")
    if not names_changed:
        return new["value"]
    bits = place.names or {}
    unknown = [n for n in new["names"] if n not in bits]
    if unknown:
        raise Refused(path, "unknown_name", f"{unknown[0]!r} is not a flag of this field")
    mask = 0
    for bit in bits.values():
        mask |= 1 << bit
    base = new["value"] if value_changed else old.get("value", 0)
    if not isinstance(base, int) or isinstance(base, bool):
        raise Refused(path, "value_out_of_range", f"{base!r} is not an integer")
    value = (base & ~mask) | sum(1 << bits[n] for n in set(new["names"]))
    if value_changed and value != new["value"]:
        raise Refused(path, "conflicting_values", f"value {new['value']} and names {new['names']} disagree")
    return value


def _link_value(change: Change, ctx: _Context, place: Place, stored_group: int | None):
    """(group id or None, handle) of a link edit."""
    new, path = change.new, change.path
    if new is None:
        return None, -1
    if not isinstance(new, dict) or set(new) != {"group", "handle"}:
        raise Refused(path, "bad_json", "a link is {\"group\", \"handle\"} or null")
    field_group = place.group
    if field_group is None:
        old = change.old if isinstance(change.old, dict) else None
        found = group_by_id(stored_group & 0xFFFFFFFF) if isinstance(stored_group, int) and stored_group not in (-1,) else None
        field_group = (old or {}).get("group") or (found.extension if found else None)
    if field_group is not None and new["group"] != field_group:
        raise Refused(path, "wrong_group", f"group {new['group']!r}; the field links {field_group!r}")
    handle = new["handle"]
    group = group_by_extension(field_group) if field_group else None
    if ctx.known is not None and group is not None and isinstance(handle, int) and not isinstance(handle, bool):
        if not ctx.known(group.id, handle):
            ctx.warnings.append(f"dangling_link: {path_text(path)} names {field_group} {handle}, which is not in "
                                "the game folder's index")
    return (group.id if group else None), handle


def _rotation_value(change: Change, ctx: _Context) -> tuple:
    old, new, path = change.old, change.new, change.path
    if not isinstance(new, dict) or set(new) != {"quaternion", "raw"}:
        raise Refused(path, "bad_json", "a rotation is {\"quaternion\", \"raw\"}")
    for key in ("quaternion", "raw"):
        if not (isinstance(new[key], list) and len(new[key]) == 4
                and all(isinstance(c, (int, float)) and not isinstance(c, bool) for c in new[key])):
            raise Refused(path + (key,), "bad_json", "four numbers, x y z w")
    q_changed = not _same_json(new["quaternion"], old["quaternion"])
    raw_changed = not _same_json(new["raw"], old["raw"])
    if q_changed and raw_changed:
        if any(not math.isfinite(c) for c in new["raw"]) or not any(new["raw"]):
            raise Refused(path, "value_out_of_range", "raw is not a rotation")
        a, b = quat_normalise(tuple(new["quaternion"])), quat_normalise(tuple(new["raw"]))
        if any(abs(x - y) > QUATERNION_TOLERANCE for x, y in zip(a, b)):
            raise Refused(path, "conflicting_values", "quaternion and raw disagree")
        return tuple(float(c) for c in new["raw"])
    if raw_changed:
        return tuple(float(c) for c in new["raw"])
    q = new["quaternion"]
    norm = math.sqrt(sum(c * c for c in q))
    if not norm or not math.isfinite(norm):
        raise Refused(path, "value_out_of_range", "a zero or non-finite quaternion")
    q = tuple(c / norm for c in q)
    previous = quat_normalise(tuple(old["raw"]))
    dot = sum(x * y for x, y in zip(q, previous))
    # the stored sign is kept within 90 degrees of the old rotation (|angle| < 90: dot > cos 45)
    if dot < 0 and abs(dot) > math.cos(math.radians(45)):
        q = tuple(-c for c in q)
    return q


def _tag_records(value, path, resizable: bool, old_value):
    if not isinstance(value, dict) or not isinstance(value.get("records"), list):
        raise Refused(path, "bad_json", "a tag map is {\"records\": [...]}")
    if "error" in value or "error" in (old_value or {}):
        raise Refused(path, "unsupported_value", "a tag map that could not be read is not written")
    if set(value) != set(old_value or {}):
        raise Refused(path, "bad_json", "a tag map keeps its keys")
    if not resizable and len(value["records"]) != len(old_value["records"]):
        raise Refused(path + ("records",), "list_length_fixed", "this tag map keeps its number of records")
    out = []
    for i, row in enumerate(value["records"]):
        if not isinstance(row, dict) or set(row) != {"tag", "type", "value"}:
            raise Refused(path + ("records", i), "bad_json", "a tag record is {\"tag\", \"type\", \"value\"}")
        kind = row["type"].get("value") if isinstance(row["type"], dict) else row["type"]
        if not isinstance(kind, int) or isinstance(kind, bool):
            raise Refused(path + ("records", i, "type"), "bad_json", "the tag type is a number")
        if not isinstance(row["value"], (int, float)) or isinstance(row["value"], bool):
            raise Refused(path + ("records", i, "value"), "unsupported_value", "only 4-byte tag values are written")
        if not isinstance(row["tag"], int) or isinstance(row["tag"], bool):
            raise Refused(path + ("records", i, "tag"), "value_out_of_range", "a tag is a number")
        if kind != TAG_FLOAT and isinstance(row["value"], float) and not row["value"].is_integer():
            raise Refused(path + ("records", i, "value"), "value_out_of_range", "an integer tag value")
        out.append((kind, row["tag"], row["value"] if kind == TAG_FLOAT else int(row["value"])))
    return out


class _Writes:
    """The writes of one entry: two paths that write different bytes at one place conflict (W22-3)."""

    def __init__(self, target, ctx: _Context, path: tuple, key) -> None:
        self.target, self.ctx, self.path, self.key = target, ctx, path, key
        self.data = target.data

    def write(self, at: int, data: bytes) -> None:
        self._claim(("at", self.key, at), data)
        self.target.write(at, data)

    def set_pair(self, at: int, data: bytes) -> None:
        self._claim(("pair", self.key, at), data)
        self.target.set_pair(at, data)

    def _claim(self, key, data: bytes) -> None:
        earlier = self.ctx.writes.get(key)
        if earlier is not None and earlier[0] != bytes(data):
            raise Refused(self.path, "conflicting_values", f"{path_text(earlier[1])} sets the same stored value "
                                                           "to something else")
        self.ctx.writes[key] = (bytes(data), self.path)


def _write_change(change: Change, patch, ctx: _Context) -> None:
    place, path, new = change.place, change.path, change.new
    order = ctx.order
    if place.kind == "link" and place.group == "shm" and ctx.known is not None and isinstance(new, dict)             and isinstance(new.get("handle"), int) and not ctx.known(group_by_extension("shm").id, new["handle"]):
        raise Refused(path, "missing_shader", f"no shader file {new['handle']} in the game folder's index (W22-6)")
    if place.kind == "fixed":
        raise Refused(path, "field_fixed", place.message or "this field is not changed by the import")
    if place.kind == "count":
        raise Refused(path, "derived_field", "a count word follows its list")
    if place.kind == "num":
        patch.write(place.at, _pack(place.code, new, path, order))
    elif place.kind == "enum":
        patch.write(place.at, _pack(place.code, _enum_value(change, ctx), path, order))
    elif place.kind == "flags":
        patch.write(place.at, _pack(place.code, _flags_value(change, ctx), path, order))
    elif place.kind == "link":
        stored_group = None
        if place.group_at is not None:
            stored_group = struct.unpack_from(order + "i", patch.data, place.group_at)[0]
        _, handle = _link_value(change, ctx, place, stored_group)
        patch.write(place.at, _pack(place.code if handle != -1 else "i", handle, path, order))
    elif place.kind == "rotation":
        q = _rotation_value(change, ctx)
        patch.write(place.at, b"".join(_pack("f", c, path, order) for c in q))
    elif place.kind == "string":
        if not isinstance(new, str):
            raise Refused(path, "bad_json", "a string")
        raw = new.encode("utf-8")
        if len(raw) > place.size:
            raise Refused(path, "value_out_of_range", f"{len(raw)} bytes for a field of {place.size}")
        patch.write(place.at, raw.ljust(place.size, b"\0"))
    elif place.kind == "text":
        if not isinstance(new, str):
            raise Refused(path, "bad_json", "a string")
        old_block = _pair_data(patch.data, place.at, order)
        unit = 4 if len(old_block) % 4 == 0 and len(old_block) else 1
        raw = new.encode("utf-8") + b"\0"
        patch.set_pair(place.at, raw + bytes(-len(raw) % unit) if new else b"")
    elif place.kind == "bytes":
        if not isinstance(new, dict) or set(new) != {"bytes"} or not isinstance(new["bytes"], str):
            raise Refused(path, "bad_json", "raw bytes are {\"bytes\": \"<hex>\"}")
        try:
            raw = bytes.fromhex(new["bytes"])
        except ValueError:
            raise Refused(path, "bad_json", "not hex") from None
        if len(raw) != place.size:
            raise Refused(path, "value_out_of_range", f"{len(raw)} bytes for a field of {place.size}")
        patch.write(place.at, raw)
    elif place.kind == "tagmap":
        from .material_write import encode_tag_map

        stored = _pair_data(patch.data, place.at, order)
        old_records = _tag_records(change.old, path, True, change.old)
        if encode_tag_map(old_records, order) != stored:
            raise Refused(path, "unsupported_value", "the stored tag map is not one the writer reproduces")
        patch.set_pair(place.at, encode_tag_map(_tag_records(new, path, place.resizable, change.old), order))
    elif place.kind == "nav":
        if isinstance(new, bool) or not isinstance(new, (int, dict)):
            raise Refused(path, "value_out_of_range", "an integer")
        key = (place.field, place.at)
        if place.field == "cell_flags":
            value = _flags_value(Change(path, change.old, new, Place("flags", names={
                n: b for b, n in navigation.FLAG_NAMES.items()})), ctx)
        else:
            value = new
        if key in ctx.nav_changes and ctx.nav_changes[key][0] != value:
            # ``52`` AC2: the cell's neighbour list and the flat list are two views of one value
            raise Refused(path, "derived_field", f"{path_text(ctx.nav_changes[key][1])} gives "
                                                 f"{json.dumps(ctx.nav_changes[key][0])} for this value")
        ctx.nav_changes[key] = (value, path)
    else:
        raise Refused(path, "derived_field", "not a stored value")


def _pair_data(data: bytes, at: int, order: str) -> bytes:
    view = ByteView(data, order)
    return view.pair_data(view.pair(at), at)


# --- lists (W22-4) ------------------------------------------------------------------------------------

# group -> list path pattern -> what it is
LIST_RULES = {"mrk": {("markers",): "markers"}, "wrl": {("scene_entries",): "scene_entries"}}


def _list_rule(ext: str, path: tuple):
    for pattern, name in LIST_RULES.get(ext, {}).items():
        if len(pattern) == len(path) and _matches(path, pattern):
            return name
    return None


def _content_key(ext: str, path: tuple, element) -> str:
    """``52`` AC3: the whole content of a list row (its export diagnostics left out, ``51`` AB2)."""
    if isinstance(element, dict):
        element = {k: v for k, v in element.items() if not is_diagnostic(ext, path + (0, k))}
    return json.dumps(element, sort_keys=True)


def _nested_shape(rule: str, element) -> tuple:
    """The lengths of an element's nested lists (a marker's links)."""
    if rule == "markers" and isinstance(element, dict) and isinstance(element.get("links"), list):
        return (len(element["links"]),)
    return ()


def _match_elements(old: list, new: list, ext: str = "", path: tuple = ()) -> list:
    """W22-4 with ``52`` AC3: the rows of the edited list paired with the template's rows, never by a key such as
    a name hash. For each new row ``(k, own)``: the template row whose stored record it is written from (None:
    a new row, copied from a base, W22-4), ``own`` when that record itself is used (else a copy of it that
    shares its blocks, ``51`` AB4). In this order:
    1. a row equal to the template row at its position is that row, unchanged;
    2. a row equal to another template row (a row moved or copied) is written from that row's stored
       record: the record itself while no other row has it (the nearest first), else a copy;
    3. any other row, in order, is the template row no other row has (its own position, or one of its nested
       lists' lengths) from which it differs in the fewest leaves, with its leaves changed: its own position
       on a tie (so a row changed in place is that row, whether the list changed length or not; a changed
       length of its links is then refused, ``50`` AA2), else the nearest;
    4. the rest are new rows (a copy of their base, W22-4); template rows no row has are removed.
    The pairing depends only on the two lists, so an edit undone pairs every row with the row it came from."""
    rule = _list_rule(ext, path) or ""
    keys_old = [_content_key(ext, path, x) for x in old]
    keys_new = [_content_key(ext, path, x) for x in new]
    plan: list = [None] * len(new)
    used: set[int] = set()
    for j in range(min(len(old), len(new))):
        if keys_new[j] == keys_old[j]:
            plan[j] = (j, True)
            used.add(j)
    where: dict[str, list[int]] = {}
    for k, key in enumerate(keys_old):
        where.setdefault(key, []).append(k)
    for j, key in enumerate(keys_new):
        if plan[j] is not None or key not in where:
            continue
        found = sorted(where[key], key=lambda k: (abs(k - j), k))
        free = [k for k in found if k not in used]
        if free:
            plan[j] = (free[0], True)
            used.add(free[0])
        else:
            plan[j] = (found[0], False)
    for j in range(len(new)):
        if plan[j] is not None:
            continue
        shape = _nested_shape(rule, new[j])
        free = [k for k in range(len(old))
                if k not in used and (k == j or _nested_shape(rule, old[k]) == shape)]
        if not free:
            continue
        # the free template row it differs from in the fewest leaves: its own row on a tie, else the nearest
        k = min(free, key=lambda k: (_difference(old[k], new[j]), abs(k - j), k))
        plan[j] = (k, True)
        used.add(k)
    return plan


def _difference(a, b) -> int:
    """``52`` AC3: the number of leaves in which two rows differ (a list of another length: all its leaves)."""
    if isinstance(a, dict) and isinstance(b, dict):
        return sum(_difference(a.get(k), b.get(k)) for k in set(a) | set(b))
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return sum(_difference(x, y) for x, y in zip(a, b))
    if a == b:
        return 0
    return max(_size(a), _size(b))


def _size(value) -> int:
    if isinstance(value, dict):
        return max(1, sum(_size(v) for v in value.values()))
    if isinstance(value, list):
        return max(1, sum(_size(v) for v in value))
    return 1


def _apply_list(ext: str, change: Change, patch: TreePatch, ctx: _Context, template: dict, found: dict) -> list:
    """Markers of a ``mrk`` or scene entries of a ``wrl`` (W22-4, ``52`` AC3). Returns the leaf changes left
    to write."""
    rule = _list_rule(ext, change.path)
    if rule is None:
        raise Refused(change.path, "list_length_fixed", f"{len(change.old)} elements, {len(change.new)} given")
    if not patch.canonical:
        raise Refused(change.path, "not_canonical", "the template is not laid out as the tree writer lays it out")
    old, new = change.old, change.new
    if not all(isinstance(x, dict) for x in new):
        raise Refused(change.path, "bad_json", "elements are objects")
    plan = _match_elements(old, new, ext, change.path)
    order = ctx.order
    view = ByteView(patch.data, order)
    if rule == "markers":
        pair_at = 0x10 + 0x0C
        size = world.MARKER_SIZE
        starts = [view.pair(pair_at).file_offset + i * size for i in range(len(old))]
    else:
        server = view.pair(0x10 + 0x10)
        lists = view.pair(server.file_offset + 0x10)
        if lists.size != world._LIST_SIZE:
            raise Refused(change.path, "list_length_fixed", "a world of more than one scene list keeps its entries")
        pair_at = lists.file_offset
        size = world._ENTRY_SIZE[ctx.build]
        starts = [view.pair(pair_at).file_offset + i * size for i in range(len(old))]
    kept = [p[0] if p is not None else None for p in plan]
    records_out, left = [], []
    for j, element in enumerate(new):
        if plan[j] is not None:
            source, own = plan[j]
        else:
            source = _base_index(rule, old, element, j, kept)
            if source is None:
                raise Refused(change.path + (j,), "list_length_fixed", "no element to copy")
            own = False
        base_json, base_at = old[source], starts[source]
        record = patch.element_record(base_at)
        if not own:
            record = _share_copy(record, patch)
            patch.copies.append(record)
        prefix = change.path + (j,)
        for sub in compare(base_json, element, _element_places(rule, base_at, ctx.build, patch.data, order), (),
                           ignore=lambda p, prefix=prefix: is_diagnostic(ext, prefix + p)):
            sub.path = change.path + (j,) + sub.path
            # ``50`` AA2: a template row's own record is written at its template places (its record and blocks
            # are in the tree); a copy through ``_Redirect``
            left.append((sub, None if own else record, base_at))
        records_out.append(record)
    patch.elements(pair_at, records_out)  # with its count word (the scene list's entry count, W14-20)
    return left


def _base_index(rule: str, old: list, element: dict, j: int, kept: list):
    if not old:
        return None
    if rule == "scene_entries":
        return len(old) - 1
    earlier = [kept[k] for k in range(j) if kept[k] is not None]
    want = (element.get("type") or {}).get("value") if isinstance(element.get("type"), dict) else None
    shape = _nested_shape(rule, element)

    def candidates():
        for index in sorted(set(earlier), reverse=True):
            yield index
        for index in range(len(old) - 1, -1, -1):
            if not earlier or index <= max(earlier):
                yield index

    typed = [i for i in candidates() if (old[i].get("type") or {}).get("value") == want]
    # ``51`` AB4: the nearest such marker whose links have the new marker's length, then any marker of its
    # type with that length (nearest first), else the nearest of its type (W22-4)
    centre = max(earlier) if earlier else 0
    for index in typed + sorted(range(len(old)), key=lambda i: (abs(i - centre), i)):
        if (old[index].get("type") or {}).get("value") == want and _nested_shape(rule, old[index]) == shape:
            return index
    return typed[0] if typed else 0


def _share_copy(record: L.Record, patch: "TreePatch | None" = None) -> L.Record:
    """``51`` AB4, ``53`` AD2: a new list element as a copy of its base record. A block the template shares
    between rows (W11-6) is shared by the copy too, until a value in it changes (``_Redirect`` copies that
    block first); a block of the base's own is copied, so the copy's equal block is never pooled with it."""
    tag_maps = set()
    if patch is not None and patch.tree.ext == "mrk":
        tag_maps = {s.offset for s in L.pair_slots(patch.tree.build, patch.tree.ext, record.name)
                    if s.field.kind == "tagmap"}

    def placed(slot, value):
        if value is None:
            return None
        shared = value.block is not None and (patch is None or patch.refs.get(id(value.block), 0) > 1)
        if slot in tag_maps and value.block is not None and not shared:
            # ``54`` AE1: a new row's tag map follows the build (``sno_layout.MAPS_POOLED``): its own block on
            # ps3, the first equal map's block on switch and x360 (a map the template shares stays shared)
            block = copy.deepcopy(value.block)
            block.fresh = True
            return L.PairValue(block, value.size, value.stored)
        # ``53`` AD2: a block the template shares between rows stays shared (W11-6); a block of its own in the
        # template is a block of its own in the copy too (never pooled with the base's equal block)
        if shared:
            return L.PairValue(value.block, value.size, value.stored, shared=True)
        return L.PairValue(copy.deepcopy(value.block), value.size, value.stored)

    pairs = {slot: placed(slot, value) for slot, value in record.pairs.items()}
    return L.Record(record.name, bytearray(record.raw), pairs, dict(record.parsed))


def _element_places(rule: str, at: int, build: str, data: bytes, order: str) -> dict:
    """The places of one list element stored at ``at`` (``50`` AA2: a marker's links included)."""
    out: dict = {}
    if rule == "markers":
        _marker_places(at, out, ())
        links = ByteView(data, order).pair(at + 0x44)
        for k in range(links.size // world.LINK_SIZE):
            for j in range(2):
                out[("links", k, j)] = Place("num", links.file_offset + 8 * k + 4 * j, "I")
    else:
        _entry_places(at, build, out)
    return out


class _Redirect:
    """Writes of a list element: template offsets of the base element into the element's record."""

    def __init__(self, patch: TreePatch, record: L.Record, base_at: int) -> None:
        self.patch, self.record, self.base_at, self.data = patch, record, base_at, patch.data
        self.canonical = True

    def write(self, at: int, data: bytes) -> None:
        rel = at - self.base_at
        if 0 <= rel and rel + len(data) <= len(self.record.raw):
            self.record.raw[rel:rel + len(data)] = data
            return
        for slot, value in list(self.record.pairs.items()):  # a block of the copy (a marker's links)
            block = value.block if value is not None else None
            if block is None or block.source is None:
                continue
            start = L.BODY + block.source
            if not start <= at < start + block.size():
                continue
            if value.shared:  # ``51`` AB4: still the base's block: the copy gets its own first
                block = copy.deepcopy(block)
                self.record.pairs[slot] = L.PairValue(block, value.size, value.stored)
            if block.elements is not None:
                for element in block.elements:
                    if start <= at and at + len(data) <= start + len(element.raw):
                        element.raw[at - start:at - start + len(data)] = data
                        return
                    start += len(element.raw)
                continue
            if start <= at and at + len(data) <= start + len(block.data or b""):
                raw = bytearray(block.data)
                raw[at - start:at - start + len(data)] = data
                block.data = bytes(raw)
                return
        raise FormatError("a list element's value outside the element")

    def set_pair(self, at: int, data: bytes) -> None:
        rel = at - self.base_at
        self.record.pairs[rel] = L.PairValue(L.Block(bytes(data)) if data else None)


# --- the import (W22-1 to W22-9) ----------------------------------------------------------------------

@dataclass
class ImportResult:
    data: bytes
    changes: list  # rows {"path", "old", "new"}
    warnings: list
    nav: bytes | None = None  # a written .nav (W22-8)


def _row(change: Change) -> dict:
    if change.kind == "list":
        return {"path": path_text(change.path), "old": {"length": len(change.old)}, "new": {"length": len(change.new)}}
    return {"path": path_text(change.path), "old": change.old, "new": change.new}


LAYOUT_MESSAGE = "the layout follows the marker sets and scenes; edit those"


def _layout_check(ext, template, build, edited, old, changes, after):
    """``51`` AB3: a world's ``layout`` is derived from the scenes and marker sets. The given layout is
    accepted when it equals the layout derived from the state after the apply (``after``: the lookups
    of that state) or, untouched, the template's own export; otherwise ``derived_field``. Returns the
    changes left and the lookups the written world is exported with again."""
    if ext != "wrl":
        return changes, None
    layout = [c for c in changes if c.path[:1] == ("layout",)]
    rest = [c for c in changes if c.path[:1] != ("layout",)]
    if layout:
        scenes, sets = after if after is not None else (None, None)
        derived = _json_round(template_json("wrl", template, build, None, scenes, sets)).get("layout")
        if after is None or not _close(edited.get("layout"), derived):
            error = Refused(("layout",), "derived_field", LAYOUT_MESSAGE)
            error.refusals = [error]
            raise error
    return rest, after


def import_json(ext: str, template: bytes, build: str, edited: dict, *, nav: bytes | None = None,
                scenes=None, sets=None, known=None, after=None) -> ImportResult:
    """W22-1 to W22-9 for one entry: the template's new bytes. Raises ``Refused`` (the first refusal;
    ``refusals`` holds all) when any leaf is refused; nothing is written then. ``51`` AB3: ``after`` gives
    a world the (scenes, sets) lookups of the state after the apply. ``51`` AB2: export diagnostics are
    not compared; a diagnostic row new in the export of the written file is an ``export_warning``."""
    if ext == "phm":
        return import_collision(template, build, edited)
    if ext in effects.GROUPS:  # ``55`` W23-8
        data, rows, warnings = effects.encode_result(template, ext, build, edited, known)
        return ImportResult(data, rows, warnings, None)
    order = build_order(build)
    old = _json_round(template_json(ext, template, build, nav, scenes, sets))
    found = places(ext, template, build, nav)
    changes = compare(old, edited, found, ignore=lambda p: is_diagnostic(ext, p))
    changes, later = _layout_check(ext, template, build, edited, old, changes, after)
    if later is not None:
        scenes, sets = later
    rows = [_row(c) for c in changes]
    if not changes:
        return ImportResult(bytes(template), [], [], None)
    ctx = _Context(build, order, known)
    refusals: list[Refused] = []
    patch = TreePatch(template, ext, build)
    leaves: list = []
    derived: list[Change] = []
    checked = list(changes)  # the changes W22-5 checks on the written file (``50`` AA2: list elements too)
    for change in changes:
        try:
            if change.kind == "keys":
                raise Refused(change.path, "bad_json", "keys were added or removed")
            if change.kind == "list":
                if change.path[:1] == ("navigation",):
                    raise Refused(change.path, "navigation_geometry", "the number of navigation rows is fixed")
                for sub, record, base_at in _apply_list(ext, change, patch, ctx, old, found):
                    # ``50`` AA2: changes inside the elements of a list that changed length
                    checked.append(sub)
                    if sub.kind == "list":
                        refusals.append(Refused(sub.path, "list_length_fixed",
                                                f"{len(sub.old)} elements, {len(sub.new)} given (W22-4)"))
                    elif sub.kind == "keys":
                        refusals.append(Refused(sub.path, "bad_json", "keys were added or removed"))
                    elif sub.place is None:
                        derived.append(sub)  # AA1: no stored place, so W22-5 decides
                    else:
                        leaves.append((sub, record, base_at))
                continue
            if change.place is None:
                kind = classify(ext, change.path, found)
                if (change.path[:1] == ("navigation",) and kind == "derived"
                        and not _matches(change.path, NAV_DERIVED)):
                    raise Refused(change.path, "navigation_geometry", "only cell flags and neighbour cells change")
                derived.append(change)
                continue
            leaves.append((change, None, None))
        except Refused as refusal:
            refusals.append(refusal)
        except Exception as exc:  # ``50`` AA1: an unexpected error refuses this entry only
            refusals.append(_failed(change.path, exc))
    for change, record, base_at in leaves:
        try:
            target = patch if record is None else _Redirect(patch, record, base_at)
            _write_change(change, _Writes(target, ctx, change.path, id(record)), ctx)
        except Refused as refusal:
            refusals.append(refusal)
        except Exception as exc:  # AA1
            refusals.append(_failed(change.path, exc))
    new_nav = None
    new_data = None
    if not refusals:
        try:
            new_data = patch.bytes()
            if ctx.nav_changes:
                new_data, new_nav = _write_navigation(new_data, build, nav, ctx, changes)
        except Refused as refusal:
            refusals.append(refusal)
        except FormatError as exc:
            refusals.append(Refused((), "value_out_of_range", str(exc)))
    if not refusals:
        try:
            again = _json_round(template_json(ext, new_data, build, new_nav if new_nav is not None else nav,
                                              scenes, sets))
        except FormatError as exc:
            refusals.append(Refused((), "value_out_of_range", f"the written file does not read back: {exc}"))
        else:
            refusals.extend(_check_again(checked, derived, edited, again))
    if refusals:
        error = refusals[0]
        error.refusals = refusals
        raise error
    before = old if later is None else _json_round(template_json(ext, template, build, nav, scenes, sets))
    return ImportResult(new_data, rows, ctx.warnings + export_warnings(ext, before, again), new_nav)


def _failed(path, exc: Exception) -> Refused:
    """``50`` AA1: an unexpected error in the import of one leaf, with its type and the JSON path."""
    return Refused(path, "import_failed", f"{type(exc).__name__}: {exc}")


def _check_again(changes: list, derived: list, edited: dict, again: dict) -> list:
    """W22-5: every changed leaf of the edited JSON as a new export gives it. A derived leaf that does not
    is ``derived_field``; a stored one that does not was not representable (``value_out_of_range``)."""
    found = []
    derived_ids = {id(c) for c in derived}
    for change in changes:
        if change.kind == "list":
            continue
        try:
            value = get_path(again, change.path)
        except (KeyError, IndexError, TypeError):
            found.append(Refused(change.path, "derived_field", "not in the written file's export"))
            continue
        if change.place is not None and (change.place.kind in ("enum", "flags", "rotation", "tagmap")
                                         or change.place.compound and change.place.kind == "nav"):
            continue  # checked by its own rule (W22-3)
        if _close(value, change.new):
            continue
        if id(change) in derived_ids:
            found.append(Refused(change.path, "derived_field", f"the export computes {json.dumps(value)} from the stored "
                                                               "values"))
        else:
            found.append(Refused(change.path, "value_out_of_range", f"reads back as {json.dumps(value)}"))
    return found


# ``52`` AC2: the navigation leaf the export derives from other leaves (a neighbour's border follows its cell);
# W22-5 compares it with the export of the written file, as every derived leaf
NAV_DERIVED = ("navigation", "cells", "*", "neighbours", "*", "border")


def _write_navigation(data: bytes, build: str, nav: bytes | None, ctx: _Context, changes) -> tuple:
    """W22-8: changed cell flags and neighbour cells; a ``.nav`` (ps3, x360) is written with ``write_nav``."""
    from . import children

    navi = navigation.parse_scene(data, build, nav)
    cells = list(navi.cells)
    neighbours = list(navi.neighbours)
    for (what, index), (value, _) in sorted(ctx.nav_changes.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        if what == "cell_flags":
            if not 0 <= value <= 0xFFFF:
                raise Refused(("navigation", "cells", index, "flags"), "value_out_of_range", "cell flags are 16 bits")
            cells[index] = navigation.NavCell(cells[index].min, cells[index].max, value,
                                              cells[index].neighbour_count, cells[index].first_neighbour)
        else:
            edge = neighbours[index]
            border = edge.flags == navigation.NEIGHBOUR_BORDER
            if not isinstance(value, int) or (not border and not 0 <= value < len(cells)) or not 0 <= value <= 0xFFFF:
                raise Refused(("navigation", "neighbours", index, "cell"), "value_out_of_range",
                              f"cell {value} of {len(cells)}")
            neighbours[index] = navigation.Neighbour(edge.flags, value)
    navi.cells, navi.neighbours = cells, neighbours
    order = build_order(build)
    if build == "switch":
        view = ByteView(data, order)
        cell_pair = view.pair(0x10 + navigation.SECTION_PAIRS[1])
        edge_pair = view.pair(0x10 + navigation.SECTION_PAIRS[2])
        patch = TreePatch(data, "scn", build)
        for i, cell in enumerate(cells):
            patch.write(cell_pair.file_offset + i * navigation.CELL_SIZE + 24, struct.pack(order + "H", cell.flags))
        for k, edge in enumerate(neighbours):
            patch.write(edge_pair.file_offset + k * navigation.NEIGHBOUR_SIZE + 2, struct.pack(order + "H", edge.cell))
        return patch.bytes(), None
    written = children.write_nav(navi)
    return data, written if written is not None else b""


# --- collision (W22-9, ``51`` AB1, ``52`` AC1) -----------------------------------------------------------

COLLISION_MESH_KEYS = ("vertices", "triangles", "flags", "material")
# ``52`` AC1: the mesh keys whose given values a refit writes as given while they still enclose
COLLISION_FIT_KEYS = ("bounds", "box_scale", "tree", "stored_vertices")


def _leaf_entries(mesh, build: str) -> list[int]:
    """The table entry of each exported triangle (``collision.parse_collision``: the table on switch and
    x360, the entries in leaf order on ps3)."""
    if build != "ps3":
        return list(range(len(mesh.triangles)))
    runs = collision_write.leaf_runs(mesh.nodes, build)
    return list(dict.fromkeys(runs.get(0, [])))


def _f32_step(x: float, up: bool) -> float:
    """The next float32 after ``x`` (a float32) towards +inf (``up``) or -inf."""
    if x == 0:
        bits = 1 if up else 0x80000001
    else:
        bits = struct.unpack("<I", struct.pack("<f", x))[0]
        bits += 1 if (x > 0) == up else -1
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def _nearest_row(value: float, centre: float) -> float:
    """``52`` AC1 rule 2: the float32 vertex row (relative to ``centre``) nearest the JSON value: of
    ``f32(value - centre)`` and its two float32 neighbours, one whose position as the export writes it
    (float32 of row + centre) is the JSON value's float32 when one is, the nearest first."""
    f32 = collision_write.f32
    base = f32(value - centre)
    candidates = [base, _f32_step(base, False), _f32_step(base, True)]
    target = f32(value)
    exact = [r for r in candidates if f32(r + centre) == target]
    pool = exact or candidates
    return min(pool, key=lambda r: abs(r + centre - value))


def _mesh_rows(mesh, positions, centre, same, keep: bool, given=None) -> list:
    """``53`` AD1, per coordinate: a value of the given stored form (``given``, the JSON's
    ``stored_vertices``) is kept when, against ``centre``, it exports as the JSON value (float32 of
    value + centre); without a given row (an older export), the stored value of a coordinate whose JSON value
    equals the template's export while the template's bounds are written (``keep``, ``52`` AC1 rule 2);
    every other coordinate is quantised anew (``_nearest_row``). The fourth word of a stored vertex is kept
    (0 for a new vertex)."""
    f32 = collision_write.f32
    stored = mesh.vertices or []
    rows = []
    for k, p in enumerate(positions):
        row = []
        own = given[k] if given is not None and k < len(given) else None
        for a in range(3):
            if own is not None:
                if f32(own[a] + centre[a]) == f32(p[a]):
                    row.append(own[a])
                    continue
            elif keep and k < len(stored) and k < len(same) and same[k][a]:
                row.append(stored[k][a])
                continue
            row.append(_nearest_row(p[a], centre[a]))
        row.append(stored[k][3] if k < len(stored) else 0.0)
        rows.append(tuple(row))
    return rows


def _encloses_rows(rows, half, scale, build: str) -> bool:
    """``52`` AC1 rule 1: the half extent and box scale enclose every vertex row (and the quantised box of
    any row fits the stored 16-bit range)."""
    limit = collision_write.QUANT_MAX[collision_write.form_of(build)]
    for a in range(3):
        if not (math.isfinite(half[a]) and math.isfinite(scale[a]) and scale[a] > 0 and half[a] >= 0):
            return False
    return all(abs(r[a]) <= half[a] and abs(r[a]) <= limit * scale[a] for r in rows for a in range(3))


def _signed(word: int) -> int:
    return word - 0x10000 if word >= 0x8000 else word


def _given_boxes(boxes, nodes, build: str, path) -> list:
    """``52`` AC1: the given ``tree.boxes`` rows as node boxes (switch form: (lower, upper); ps3: (a, b) or
    None for a leaf); a row of the wrong form is ``value_out_of_range``."""
    if len(boxes) != len(nodes):
        raise Refused(path, "list_length_fixed", f"one box per tree node: {len(nodes)} nodes, {len(boxes)} given")
    out = []
    for i, (row, node) in enumerate(zip(boxes, nodes)):
        at = path + (i,)
        if build == "ps3":
            if node.is_leaf:
                out.append(None)
                continue
            if (not isinstance(row, list) or len(row) != 2
                    or not all(isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= 0xFFFF for v in row)):
                raise Refused(at, "value_out_of_range", "an inner node's box is two 16-bit words [a, b]")
            out.append(tuple(row))
        else:
            if (not isinstance(row, list) or len(row) != 6
                    or not all(isinstance(v, int) and not isinstance(v, bool) and -32768 <= v <= 32767 for v in row)):
                raise Refused(at, "value_out_of_range", "a node box is six 16-bit numbers [lower xyz, upper xyz]")
            out.append((tuple(row[:3]), tuple(row[3:])))
    return out


def fit_boxes(nodes, table, rows, scale, build: str, start=None) -> tuple[list, list[int]]:
    """``52`` AC1 rule 3: the tree boxes in the bounds and scale being written. A node keeps its box (the
    given row of ``start`` when there is one, else its stored box) while that box encloses every triangle
    below it; only a node whose box no longer encloses is fitted again, bottom-up: a switch form leaf from
    its triangles (W14-14), an inner node from its children's boxes, a ps3 inner node's split words from
    the triangles of its two children (W14-14). A switch form parent of a refitted node is fitted again
    only when it no longer encloses the new box. Returns the nodes and the indices fitted again."""
    nodes = list(nodes)
    if not nodes:
        return nodes, []
    runs = collision_write.leaf_runs(nodes, build)
    ps3 = build == "ps3"
    shift = 2 if ps3 else 6
    fitted: list[int] = []
    changed: set[int] = set()
    for i in range(len(nodes) - 1, -1, -1):
        node = nodes[i]
        given = start[i] if start is not None else None
        if ps3:
            if node.is_leaf:
                continue
            a, b = given if given is not None else (node.a, node.b)
            s = scale[node.axis]
            left = [table[e].vertices for e in runs[i + 1]]
            right = [table[e].vertices for e in runs[i + (node.data >> shift)]]
            top = max(rows[v][node.axis] for tri in left for v in tri)
            bottom = min(rows[v][node.axis] for tri in right for v in tri)
            if not (_signed(a) * s >= top and _signed(b) * s <= bottom):
                a, b = collision_write.ps3_split_values(rows, left, right, node.axis, scale)
                fitted.append(i)
            nodes[i] = collision.TreeNode(node.data, None, None, a, b)
            continue
        lower, upper = given if given is not None else (node.lower, node.upper)
        tris = [table[e].vertices for e in runs[i]]
        lo = [min(rows[v][k] for tri in tris for v in tri) for k in range(3)]
        hi = [max(rows[v][k] for tri in tris for v in tri) for k in range(3)]
        encloses = all(lower[k] * scale[k] <= lo[k] and upper[k] * scale[k] >= hi[k] for k in range(3))
        kids = [] if node.is_leaf else [i + 1, i + (node.data >> shift)]
        for kid in kids:
            if kid in changed:
                box = nodes[kid]
                encloses = encloses and all(lower[k] <= box.lower[k] and upper[k] >= box.upper[k] for k in range(3))
        if not encloses:
            if node.is_leaf:
                lower, upper = collision_write.switch_node_bounds(rows, tris, scale)
            else:
                lower = tuple(min(nodes[kid].lower[k] for kid in kids) for k in range(3))
                upper = tuple(max(nodes[kid].upper[k] for kid in kids) for k in range(3))
            fitted.append(i)
            changed.add(i)
        nodes[i] = collision.TreeNode(node.data, tuple(lower), tuple(upper), None, None)
    return nodes, sorted(fitted)


def fit_bounds(mesh, positions, build: str, *, bounds=None, same=None, given=None):
    """``52`` AC1 rules 1 and 2: (centre, half extent, box scale, vertex rows, refitted). The given
    ``bounds`` (centre, half extent, box scale; the template's when None) are kept while they enclose
    every vertex; otherwise they are derived (W14-12) and ``refitted`` is True. Vertex rows: rule 2."""
    f32 = collision_write.f32
    template = (tuple(mesh.centre), tuple(mesh.half_extent), tuple(mesh.box_scale))
    stored_rows = given
    given = template if bounds is None else tuple(tuple(f32(float(v)) for v in part) for part in bounds)
    same = same if same is not None else [(True, True, True)] * len(positions)
    rows = _mesh_rows(mesh, positions, given[0], same, given == template, stored_rows)
    if _encloses_rows(rows, given[1], given[2], build):
        return given[0], given[1], given[2], rows, False
    centre, _, half, scale = collision_write.mesh_bounds(positions, build)
    derived = (tuple(centre), tuple(half), tuple(scale))
    rows = _mesh_rows(mesh, positions, derived[0], same, derived == template, stored_rows)
    return derived[0], derived[1], derived[2], rows, True


def refit_mesh(mesh, positions, materials, flags, build: str, *, mesh_flags: int, mesh_material: int,
               bounds=None, same=None, boxes=None, given=None):
    """``51`` AB1 with ``52`` AC1: a mesh whose triangle list stays as it is, refitted with the least
    change: the stored triangle order, tree shape (nodes, children, leaf ranges) and neighbours are kept;
    the bounds and box scale (``bounds``, the template's when None) are kept while they enclose every
    vertex, else derived (W14-12); unedited vertices (``same``) keep their stored rows while the template's
    bounds are written; each node keeps its box (``boxes``: given rows, else the stored ones) while it
    encloses (``fit_boxes``). Materials and flags are given per exported triangle. Returns (mesh,
    refitted): ``refitted`` when the given bounds did not enclose."""
    entries = _leaf_entries(mesh, build)
    if len(entries) != len(materials):
        raise FormatError(f"{len(materials)} triangles for {len(entries)} stored ones")
    table = list(mesh.table)
    count = mesh.triangle_count
    last = table[count - 1] if count else None
    for i, entry in enumerate(entries):
        old = table[entry]
        table[entry] = collision.Triangle(old.vertices, old.neighbours, materials[i], flags[i])
    if build == "ps3" and count:
        for k in range(count, len(table)):  # W14-12: the tail copies the last used triangle
            if mesh.table[k] == last:
                table[k] = table[count - 1]
    used = [table[e] for e in entries]
    for tri in table[:count]:
        if not all(0 <= v < len(positions) for v in tri.vertices):
            raise FormatError(f"triangle {tri.vertices} of {len(positions)} vertices")
    centre, half, scale, rows, refitted = fit_bounds(mesh, positions, build, bounds=bounds, same=same, given=given)
    nodes, _ = fit_boxes(mesh.nodes, table, rows, scale, build, boxes)
    stored = [(r[0] + centre[0], r[1] + centre[1], r[2] + centre[2]) for r in rows]
    return collision.CollisionMesh(centre, half, scale, mesh.tree_height, mesh.triangle_count, mesh_flags,
                                   mesh_material, stored, table, nodes, used, rows), refitted


def _given_rows(value, path) -> list:
    """``53`` AD1: the JSON's ``stored_vertices``: per vertex three float32 values, each the 8 hexadecimal
    digits of its 32 bits (most significant first); a row of another form is ``value_out_of_range``."""
    if not isinstance(value, list):
        raise Refused(path, "bad_json", "stored_vertices is a list of rows")
    rows = []
    for k, row in enumerate(value):
        if (not isinstance(row, list) or len(row) != 3
                or not all(isinstance(v, str) and len(v) == 8 and all(c in "0123456789abcdefABCDEF" for c in v)
                           for v in row)):
            raise Refused(path + (k,), "value_out_of_range", "a row is three 8-digit hexadecimal float32 words")
        rows.append(tuple(struct.unpack(">f", bytes.fromhex(v))[0] for v in row))
    return rows


def _three(value, path) -> tuple:
    if (not isinstance(value, list) or len(value) != 3
            or not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in value)):
        raise Refused(path, "value_out_of_range", "three finite numbers")
    return tuple(float(v) for v in value)


def import_collision(template: bytes, build: str, edited: dict) -> ImportResult:
    """W22-9 with ``51`` AB1 and ``52`` AC1. A mesh whose vertices, triangles, flags, material, bounds, box
    scale or tree boxes changed is
    - refitted (``refit_mesh``) when its triangle list stays as it is (the same number of triangles, the same
      vertex indices in each): its stored order, tree shape and neighbours are kept, and every stored value
      that still fits (bounds, box scale, vertex rows, node boxes) is kept; a changed neighbour is
      ``derived_field``;
    - rebuilt with ``build_collision_mesh`` otherwise, in the given bounds while they enclose, with the
      warning ``triangles_reordered`` when the built order differs from the edited JSON's. Its neighbours are
      derived: only a neighbour the user changed is checked, against the neighbours W14-15 gives the edited
      triangles.
    Bounds that do not enclose are derived with the warning ``bounds_refitted: mesh <i>``. A mesh without
    ``tree`` in the JSON (an older export) keeps its stored boxes as rule 3 says. Counts are derived
    (W22-5). The other meshes keep their stored values."""
    old = _json_round(template_json("phm", template, build))
    for index, mesh_json in enumerate((edited.get("meshes") or []) if isinstance(edited, dict) else []):
        if (isinstance(mesh_json, dict) and "tree" not in mesh_json and index < len(old["meshes"])):
            old["meshes"][index].pop("tree", None)  # AC1: an older export without the boxes
        if isinstance(mesh_json, dict) and "stored_vertices" not in mesh_json and index < len(old["meshes"]):
            old["meshes"][index].pop("stored_vertices", None)  # ``53`` AD1: nor the stored vertices
    changes = compare(old, edited, {}, ignore=lambda p: is_diagnostic("phm", p))
    rows = [_row(c) for c in changes]
    if not changes:
        return ImportResult(bytes(template), [], [], None)
    phm = collision.parse_collision(template, build)
    refusals: list[Refused] = []
    touched: set[int] = set()
    neighbours: dict[int, list[Change]] = {}
    file_flags = False
    derived: list[Change] = []
    for change in changes:
        path = change.path
        if change.kind == "keys" and not (len(path) == 3 and path[:1] == ("meshes",) and path[2] == "tree"):
            refusals.append(Refused(path, "bad_json", "keys were added or removed"))
            continue
        if path[:1] == ("flags",) and change.kind == "leaf":
            file_flags = True
            continue
        if path[:1] != ("meshes",) or len(path) < 3:
            if change.kind == "list" and path == ("meshes",):
                refusals.append(Refused(path, "list_length_fixed", "the number of collision meshes is fixed"))
            else:
                derived.append(change)
            continue
        index, key = path[1], path[2]
        if key in COLLISION_FIT_KEYS:
            touched.add(index)  # AC1: written as given while they enclose; never refused for their value
            continue
        if key not in COLLISION_MESH_KEYS:
            derived.append(change)
            continue
        touched.add(index)
        if key == "triangles" and len(path) >= 5 and path[4] == "neighbours":
            if change.kind == "list" or len(path) == 5:
                refusals.append(Refused(path, "list_length_fixed", "a triangle has three neighbour slots"))
            else:
                neighbours.setdefault(index, []).append(change)
    meshes = list(phm.meshes)
    warnings: list[str] = []
    for index in sorted(touched):
        try:
            mesh_json = edited["meshes"][index]
            old_json = old["meshes"][index]
            positions = [tuple(float(c) for c in v) for v in mesh_json["vertices"]]
            if any(len(p) != 3 for p in positions):
                raise ValueError("a vertex is three numbers")
            triangles = mesh_json["triangles"]
            tris = [tuple(int(v) for v in t["vertices"]) for t in triangles]
            materials = [int(t["material"]) for t in triangles]
            flags = [int(t["flags"]) for t in triangles]
            for values, what in ((materials, "material"), (flags, "flags")):
                for value in values:
                    if not 0 <= value < (1 << 32):
                        raise ValueError(f"triangle {what} {value}")
            at = ("meshes", index)
            bounds = (_three(mesh_json["bounds"]["centre"], at + ("bounds", "centre")),
                      _three(mesh_json["bounds"]["half_extent"], at + ("bounds", "half_extent")),
                      _three(mesh_json["box_scale"], at + ("box_scale",)))
            same = [tuple(k < len(old_json["vertices"]) and a < len(v) and a < len(old_json["vertices"][k])
                          and v[a] == old_json["vertices"][k][a] for a in range(3))
                    for k, v in enumerate(mesh_json["vertices"])]  # ``53`` AD1: per coordinate
            given = None
            if "stored_vertices" in mesh_json:
                given = _given_rows(mesh_json["stored_vertices"], at + ("stored_vertices",))
            mesh = phm.meshes[index]
            stored = [tuple(t["vertices"]) for t in old_json["triangles"]]
            if tris == stored:  # AB1: the triangle list stays: refit
                boxes = None
                tree = mesh_json.get("tree")
                if tree is not None:
                    if not isinstance(tree, dict) or not isinstance(tree.get("boxes"), list):
                        raise Refused(at + ("tree",), "bad_json", "tree is {\"boxes\": [...]}")
                    boxes = _given_boxes(tree["boxes"], mesh.nodes, build, at + ("tree", "boxes"))
                mesh, refitted = refit_mesh(mesh, positions, materials, flags, build,
                                            mesh_flags=int(mesh_json["flags"]),
                                            mesh_material=int(mesh_json["material"]),
                                            bounds=bounds, same=same, boxes=boxes, given=given)
                for change in neighbours.get(index, ()):
                    refusals.append(Refused(change.path, "derived_field", "the neighbours of a refitted mesh are "
                                                                          "the stored ones (51 AB1)"))
            else:
                for t in tris:
                    if len(t) != 3 or not all(0 <= v < len(positions) for v in t):
                        raise FormatError(f"triangle {t} of {len(positions)} vertices")
                centre, half, scale, vrows, refitted = fit_bounds(mesh, positions, build, bounds=bounds, same=same,
                                                                  given=given)
                mesh = collision_write.build_collision_mesh(
                    positions, tris, materials, flags, build, mesh_flags=int(mesh_json["flags"]),
                    mesh_material=int(mesh_json["material"]), bounds=(centre, half, scale), rows=vrows)
                if [tuple(t.vertices) for t in mesh.triangles] != tris:
                    warnings.append(f"triangles_reordered: mesh {index}")
                derived_n = collision_write.triangle_neighbours(tris)
                for change in neighbours.get(index, ()):
                    t, j = change.path[3], change.path[5]
                    if t >= len(derived_n) or change.new != derived_n[t][j]:
                        refusals.append(Refused(change.path, "derived_field", "the neighbours of a rebuilt mesh are "
                                                                              "derived from its triangles (W14-15)"))
            if refitted:
                warnings.append(f"bounds_refitted: mesh {index}")
            meshes[index] = mesh
        except Refused as refusal:
            refusals.append(refusal)
        except (KeyError, TypeError, ValueError, IndexError, FormatError) as exc:
            refusals.append(Refused(("meshes", index), "value_out_of_range", str(exc)))
    if refusals:
        error = refusals[0]
        error.refusals = refusals
        raise error
    phm.meshes = meshes
    if file_flags:
        phm.flags = edited["flags"]
    try:
        data = collision_write.write_collision(phm)
        again = _json_round(template_json("phm", data, build))
    except (FormatError, struct.error) as exc:
        error = Refused((), "value_out_of_range", str(exc))
        error.refusals = [error]
        raise error
    found = []
    for change in derived:
        path = change.path
        try:
            value = get_path(again, path)
        except (KeyError, IndexError, TypeError):
            found.append(Refused(path, "derived_field", "not in the written file's export"))
            continue
        if not _close(value, change.new):
            found.append(Refused(path, "derived_field", f"the export computes {json.dumps(value)}"))
    if found:
        error = found[0]
        error.refusals = found
        raise error
    return ImportResult(data, rows, warnings + export_warnings("phm", old, again), None)
