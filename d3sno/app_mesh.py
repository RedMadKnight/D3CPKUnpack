"""The mesh-level ``.app`` writer of ``25_corrections14.md`` section 1 (W2-1 to W2-15).

The writer starts from the structure tree of the source (``sno_layout``, W11), encodes every
vertex, index, pool, skinning and ps3 draw block from a decoded mesh (an ``AppearanceFile``,
possibly changed), recomputes the counts and derived fields of what changed, keeps the values the
description lists as kept, and lays the file out with the generic writer.

A block whose new bytes equal the stored ones keeps its stored block (so shared blocks stay
shared and an unchanged mesh gives the source bytes); a block whose bytes change gets a block of
its own (W2-14). Records are changed only where something they describe changed.

Not built here (Q376): adding or removing sub-objects, draws or skinning records and changing a
palette; without the stage-13 rules the visual-mesh octree of a changed mesh is emptied (W2-13).
The ``derive`` option applies the rules of ``26_corrections15.md`` (W13-1 to W13-7): derived
work-buffer sizes, steps and ps3 bone ranges, an octree built again for a changed mesh (module
``octree``), first-use palettes split at 45 bones, compact influence lists, and empty skinning
records dropped.
"""

from __future__ import annotations

import dataclasses
import math
import re
import struct

from . import octree as O
from . import sno_layout as L
from .appearance import (
    BONE_RANGE_UNIT, FAT_VERTEX, FORMAT_RIGID, FORMAT_SKINNED, PS3_DESCRIPTOR_FIXED, PS3_PIECE, AppearanceFile,
    Draw, SubObject, parse_appearance,
)
from .binary import FormatError, build_order

# Top-record pairs (W11 body offsets): the sub-object arrays of geo set 0 and 1, the two pools.
GEO_SUB_OBJECTS = (0x64, 0x7C)
POOLS = {"switch": (0x210, 0x218), "x360": (0x210, 0x218), "ps3": (0x220, 0x228)}
# The structure is inline at body 0x10: its sphere (+0x80), octree (+0xD0) and box (+0x118).
STRUCT_SPHERE = 0x10 + 0x80
STRUCT_OCTREE = 0x10 + 0xD0
STRUCT_BOX = 0x10 + 0x118
OCTREE_COUNTS = 0x18  # six u32 counts before the node pair (Q378)
OCTREE_PAIRS = (STRUCT_OCTREE + 0x18, STRUCT_OCTREE + 0x28, STRUCT_OCTREE + 0x38)

# Sub-object record offsets (switch and x360; ps3).
SWITCH_SUB = {"fat": 0x04, "index": 0x28, "box": 0x6C, "packed_count": 0xA0, "packed": 0xB0,
              "skins": 0xB8, "influence": 0xC8, "bias": 0xD8, "scale": 0xE4}
PS3_SUB = {"fat": 0x04, "index": 0x20, "box": 0xD0, "draw_count": 0xFC, "draws": 0x100, "influence": 0x10C,
           "bias": 0x118, "scale": 0x124}
# Skinning record (switch, x360) and ps3 draw record offsets.
SKIN = {"bones": 0x00, "palette": 0x04, "vertex_count": 0x18, "vertices": 0x28,
        "index_count": 0x30, "indices": 0x40, "zero": (0x48, 0x68)}
DRAW = {"descriptor": 0x08, "index_count": 0x10, "indices": 0x18, "index_pieces": 0x20,
        "vertex_count": 0x24, "a": 0x2C, "a_words": 0x34, "b": 0x54, "b_words": 0x5C, "c": 0x6C,
        "work": 0x98, "area": 0x9C}

RAW_LIMIT = 32767  # W2-3
BASE_STEP_BONES = 1 / 512
BASE_STEP_NO_BONES = 1 / 128
PS3_WORK_LIMIT = 0xC000
SKIN_PALETTE_LIMIT = 45  # W13-5
PS3_RANGES = 0x60  # W13-4: offset0, offset1, size0, size1 (u16)
RULES = ("W13-1", "W13-2", "W13-3", "W13-4", "W13-5", "W13-6", "W13-7")
PACKED_RIGID = 28
PACKED_SKINNED = 36
INFLUENCE_RECORD = 6


def _align(value: int, step: int) -> int:
    return (value + step - 1) // step * step


# --- Word ranges (stage 12.1) -------------------------------------------------------------------
# Every integer word is checked against its width before it is packed, so a mesh that does not
# fit the format raises FormatError naming the field, never struct.error.

WORD_RANGES = {"B": (0, 0xFF), "H": (0, 0xFFFF), "h": (-0x8000, 0x7FFF),
               "I": (0, 0xFFFFFFFF), "i": (-0x80000000, 0x7FFFFFFF)}
_FLOAT32_MAX = 3.4028234663852886e38


def check_word(value, code: str, field: str) -> int:
    """``value`` as a word of struct code ``code`` (B, H, h, I, i), else FormatError naming
    ``field`` and the limit."""
    low, high = WORD_RANGES[code]
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise FormatError(f"{field}: {value!r} does not fit a {code} word ({low}..{high})")
    return value


def check_float(value, field: str) -> float:
    """``value`` as a float32: finite and within the float32 range."""
    if not isinstance(value, (int, float)) or not math.isfinite(value) or abs(value) > _FLOAT32_MAX:
        raise FormatError(f"{field}: {value!r} is not a finite float32")
    return value


def _check_values(fmt: str, values, field: str) -> None:
    codes = [code for count, code in re.findall(r"(\d*)([A-Za-z])", fmt) for _ in range(int(count or 1))]
    if len(codes) != len(values):
        raise FormatError(f"{field}: {len(values)} values for {fmt}")
    for k, (code, value) in enumerate(zip(codes, values)):
        name = field if len(codes) == 1 else f"{field}[{k}]"
        if code in WORD_RANGES:
            check_word(value, code, name)
        elif code == "f":
            check_float(value, name)


def _packed(order: str, fmt: str, field: str, *values) -> bytes:
    """``struct.pack`` after the range checks of every value."""
    _check_values(fmt, values, field)
    return struct.pack(order + fmt, *values)


# --- Encoders (W2-3, W2-5, W2-8, W2-9) -----------------------------------------------------------

def normal_bytes(normal) -> bytes:
    """W2-5: ``round(n x 128 + 128)`` per axis."""
    for n in normal:
        check_float(n, "normal")
    return bytes(max(0, min(255, round(n * 128 + 128))) for n in normal)


def uv1_words(uv) -> tuple[int, int]:
    """W2-5: ``round(u x 512 + 0x8000)``, as u16 little-endian on every build."""
    for u in uv:
        check_float(u, "UV1")
    words = tuple(round(u * 512 + 0x8000) for u in uv)
    if not all(0 <= w <= 0xFFFF for w in words):
        raise FormatError(f"UV {uv} is outside the u16 range of UV1")
    return words


def raw_coordinates(position, bias, scale) -> tuple[int, int, int]:
    """W2-3: ``round((p - bias) / step)`` per axis, within -32767..32767."""
    for k in range(3):
        check_float(position[k], "position")
        check_float(bias[k], "bias")
        check_float(scale[k], "step")
        if scale[k] == 0:
            raise FormatError(f"step 0 on axis {k}")
    raw = tuple(round((position[k] - bias[k]) / scale[k]) for k in range(3))
    if not all(-RAW_LIMIT <= r <= RAW_LIMIT for r in raw):
        raise FormatError(f"position {position} does not fit the fixed point of bias {bias}, step {scale}")
    return raw


def fixed_point_bias(positions) -> tuple[float, float, float]:
    """W2-2 for a new or moved mesh: round(box centre) per axis."""
    if not positions:
        return (0.0, 0.0, 0.0)
    return tuple(float(round((min(p[k] for p in positions) + max(p[k] for p in positions)) / 2))
                 for k in range(3))


def fixed_point_steps(positions, bias, has_bones: bool) -> tuple[float, float, float]:
    """W2-3: per axis, the smallest power of two not below the base step (1/512 with bones, 1/128
    without) for which every raw coordinate stays within -32767..32767."""
    base = BASE_STEP_BONES if has_bones else BASE_STEP_NO_BONES
    steps = []
    for k in range(3):
        reach = max((abs(p[k] - bias[k]) for p in positions), default=0.0)
        step = base
        while round(reach / step) > RAW_LIMIT:
            step *= 2
        steps.append(step)
    return tuple(steps)


def _colour(colour: bytes, flip: bool) -> bytes:
    if len(colour) != 4:
        raise FormatError(f"colour of {len(colour)} bytes")
    return bytes(colour[::-1]) if flip else bytes(colour)


def encode_packed_vertices(draw: Draw, bias, scale, order: str, slots=None) -> bytes:
    """W2-5: 28-byte rigid or 36-byte skinned packed vertices in the file byte order. On x360
    (big-endian) the colour, joint-slot and weight groups are byte-reversed. ``slots`` gives the
    stored slot bytes per vertex (W2-6); weighted slots come from the palette."""
    flip = order == ">"
    out = bytearray()
    for i, position in enumerate(draw.positions):
        out += _packed(order, "3h", f"vertex {i} position", *raw_coordinates(position, bias, scale)) + bytes(2)
        out += normal_bytes(draw.normals[i]) + b"\0"
        out += _colour(draw.colours_1[i], flip) + _colour(draw.colours_2[i], flip)
        out += _packed("<", "2H", f"vertex {i} UV1", *uv1_words(draw.uv1[i]))
        out += _packed("<", "2H", f"vertex {i} UV2", *draw.uv2_raw[i])
        if draw.skinned:
            weights = _packed("<", "4B", f"vertex {i} weights", *draw.weights[i])
            joint_slots = skin_slots(draw, i, slots[i] if slots is not None and i < len(slots) else None)
            out += (joint_slots[::-1] + weights[::-1]) if flip else (joint_slots + weights)
    return bytes(out)


def skin_slots(draw: Draw, i: int, stored: bytes | None) -> bytes:
    """W2-6: a weighted slot is the first palette position of the vertex's joint (a stored slot
    that already names the joint is kept, Q379); a slot of weight 0 keeps its stored value (0 for
    a new vertex)."""
    palette = draw.palette or []
    slots = []
    for j, (joint, weight) in enumerate(zip(draw.joints[i], draw.weights[i])):
        if weight:
            if joint not in palette:
                raise FormatError(f"vertex {i}: joint {joint} is not in the palette (W2-6)")
            if stored is not None and stored[j] < len(palette) and palette[stored[j]] == joint:
                slots.append(stored[j])
            else:
                slots.append(palette.index(joint))
        else:
            slots.append(stored[j] if stored is not None else 0)
    return _packed("<", "4B", f"vertex {i} joint slots (palette positions)", *slots)


def encode_fat_vertices(draw: Draw, order: str) -> bytes:
    """W2-8: three floats, normal bytes and a zero byte, the two colours as stored, UV1, UV2."""
    out = bytearray()
    for i, position in enumerate(draw.positions):
        out += _packed(order, "3f", f"fat vertex {i} position", *position)
        out += normal_bytes(draw.normals[i]) + b"\0"
        out += _colour(draw.colours_1[i], False) + _colour(draw.colours_2[i], False)
        out += _packed("<", "2H", f"fat vertex {i} UV1", *uv1_words(draw.uv1[i]))
        out += _packed("<", "2H", f"fat vertex {i} UV2", *draw.uv2_raw[i])
    return bytes(out)


def encode_indices(indices, order: str, pad: int = 1) -> bytes:
    raw = _packed(order, f"{len(indices)}H", "index", *indices)
    return raw + bytes(_align(len(raw), pad) - len(raw))


def ps3_command_space(fmt: int, index_count: int) -> int:
    """W2-9: descriptor byte 1. w = ceil((index count - 1) / 256), n = 28 + w (0x81) or
    36 + w (0x83), byte 1 = ceil((n + ceil(n / 32)) / 4)."""
    w = -(-(index_count - 1) // 256) if index_count > 0 else 0
    n = (28 if fmt == FORMAT_RIGID else 36) + w
    return -(-(n + -(-n // 32)) // 4)


def ps3_work_size(fmt: int, vertex_count: int, index_count: int, stored: int | None = None) -> int:
    """W2-10: the work-buffer size at +0x98. Format 0x81 by its formula; format 0x83 the larger
    of the stored value and the replacement formula (HYPOTHESIS), never above 0xC000."""
    byte1 = ps3_command_space(fmt, index_count)
    if fmt == FORMAT_RIGID:
        return _align(12 * _align(vertex_count, 8) + _align(2 * index_count, 16) + 16 * byte1 + 192, 128)
    value = _align(28 * _align(vertex_count, 8) + _align(2 * index_count, 16) + 16 * byte1 + 336, 128)
    value = max(value, stored or 0)
    if value > PS3_WORK_LIMIT:
        raise FormatError(f"skinned draw work buffer {value} is above 0xC000 (W2-10)")
    return value


# --- Stage 13 rules (26_corrections15.md) ----------------------------------------------------------

def rule_set(derive) -> frozenset[str]:
    """The W13 rules a ``derive`` option names: False / None (none), True (all) or an iterable of
    rule names from ``RULES``."""
    if derive is True:
        return frozenset(RULES)
    if not derive:
        return frozenset()
    if isinstance(derive, str):
        derive = (derive,)
    rules = frozenset(derive)
    unknown = rules - set(RULES)
    if unknown:
        raise FormatError(f"unknown rules {sorted(unknown)}; known: {', '.join(RULES)}")
    return rules


def ps3_skinned_work_size(vertex_count: int, index_count: int, command_space: int, ranges) -> int:
    """W13-1: the work-buffer size (+0x98) of a format 0x83 draw. ``ranges`` = (offset0, offset1,
    size0, size1) in bytes; span = offset1 + size1 - offset0 when size1 > 0, else size0.
    P = align(2 ni, 16) + 16 h + 28 align(nv, 8) + 208; Q = align(2 ni, 16) + 18 align(nv, 8) +
    span + 448; work = align(max(P, Q), 128)."""
    offset0, offset1, size0, size1 = ranges
    span = offset1 + size1 - offset0 if size1 > 0 else size0
    indices = _align(2 * index_count, 16)
    vertices = _align(vertex_count, 8)
    p = indices + 16 * command_space + 28 * vertices + 208
    q = indices + 18 * vertices + span + 448
    return _align(max(p, q), 128)


def quantisation_steps(centre, half, bias, has_bones: bool) -> tuple[float, float, float]:
    """W13-2: per axis, reach = half-extent + |centre - bias| + 1; the step is the smallest power
    of two not below the base step (1/512 with bones, 1/128 without) with reach / step <= 32767."""
    base = BASE_STEP_BONES if has_bones else BASE_STEP_NO_BONES
    steps = []
    for k in range(3):
        check_float(centre[k], "box centre")
        check_float(half[k], "box half-extent")
        check_float(bias[k], "bias")
        reach = abs(half[k]) + abs(centre[k] - bias[k]) + 1
        step = base
        while reach / step > RAW_LIMIT:
            step *= 2
        steps.append(step)
    return tuple(steps)


def ps3_bone_ranges(bones) -> tuple[int, int, int, int]:
    """W13-4: (offset0, offset1, size0, size1) of a set of skeleton bones: one range when no two
    sorted neighbours differ by more than 1, else two ranges split at the leftmost largest gap.
    48 bytes per bone."""
    ordered = sorted(set(bones))
    if not ordered:
        return (0, 0, 0, 0)
    gaps = [ordered[i + 1] - ordered[i] for i in range(len(ordered) - 1)]
    unit = BONE_RANGE_UNIT
    if not gaps or max(gaps) <= 1:
        ranges = (unit * ordered[0], 0, unit * (ordered[-1] - ordered[0] + 1), 0)
    else:
        cut = gaps.index(max(gaps))
        first, last = ordered[cut], ordered[cut + 1]
        ranges = (unit * ordered[0], unit * last, unit * (first - ordered[0] + 1),
                  unit * (ordered[-1] - last + 1))
    for name, value in zip(("offset0", "offset1", "size0", "size1"), ranges):
        check_word(value, "H", f"ps3 bone range {name} (W13-4)")
    return ranges


def ranges_palette(ranges) -> list[int]:
    """W13-4: the palette is range 0 followed by range 1, bone by bone."""
    offset0, offset1, size0, size1 = ranges
    unit = BONE_RANGE_UNIT
    return list(range(offset0 // unit, (offset0 + size0) // unit)) + \
        list(range(offset1 // unit, (offset1 + size1) // unit))


def first_use_palette(joints, weights) -> list[int]:
    """W13-5: skeleton bones in order of first use over the vertices and their slots in order
    (weighted slots; zero-weight slots are 0 on a changed mesh and name palette position 0)."""
    palette: list[int] = []
    for vertex_joints, vertex_weights in zip(joints, weights):
        for joint, weight in zip(vertex_joints, vertex_weights):
            if weight and joint not in palette:
                palette.append(joint)
    return palette


def _vertex_bones(draw: Draw, i: int) -> set:
    return {j for j, w in zip(draw.joints[i], draw.weights[i]) if w}


def _subset(draw: Draw, ids: list[int], indices: list[int]) -> Draw:
    def pick(values):
        return [values[i] for i in ids] if values is not None else None
    return dataclasses.replace(
        draw, positions=pick(draw.positions), normals=pick(draw.normals),
        colours_1=pick(draw.colours_1), colours_2=pick(draw.colours_2), uv1=pick(draw.uv1),
        uv2_raw=pick(draw.uv2_raw), indices=indices, joints=pick(draw.joints),
        weights=pick(draw.weights), local_joints=None)


def split_skinned_draw(draw: Draw, limit: int = SKIN_PALETTE_LIMIT) -> list[tuple[Draw, list[int]]]:
    """W13-5: the draw as skinning records of at most ``limit`` bones. Triangles are taken in
    order and a new record starts before one would exceed the limit; each record holds the
    vertices its triangles use, in vertex order. Returns (record draw, source vertex ids); a draw
    that fits is returned whole."""
    if len(first_use_palette(draw.joints, draw.weights)) <= limit:
        return [(draw, list(range(len(draw.positions))))]
    if len(draw.indices) % 3:
        raise FormatError(f"{len(draw.indices)} indices are not whole triangles; "
                          "the record cannot be split (W13-5)")
    groups: list[list[int]] = []
    current: list[int] = []
    bones: set = set()
    for t in range(0, len(draw.indices), 3):
        triangle = draw.indices[t:t + 3]
        need = set().union(*(_vertex_bones(draw, v) for v in triangle))
        if len(need) > limit:
            raise FormatError(f"a triangle uses {len(need)} bones, above {limit} (W13-5)")
        if current and len(bones | need) > limit:
            groups.append(current)
            current, bones = [], set()
        current += triangle
        bones |= need
    if current:
        groups.append(current)
    out = []
    for group in groups:
        ids = sorted(set(group))
        where = {v: n for n, v in enumerate(ids)}
        out.append((_subset(draw, ids, [where[v] for v in group]), ids))
    return out


def influence_entry(draw: Draw, i: int) -> bytes:
    """W13-6 (Switch): the vertex's weighted slots in slot order as (skeleton bone, weight), at
    most three, then (0, 0)."""
    pairs = [(j, w) for j, w in zip(draw.joints[i], draw.weights[i]) if w][:3]
    pairs += [(0, 0)] * (3 - len(pairs))
    return _packed("<", "6B", f"vertex {i} compact influence entry", *(x for pair in pairs for x in pair))


def _same_draw(a: Draw, b: Draw) -> bool:
    return all(getattr(a, name) == getattr(b, name) for name in (
        "positions", "normals", "colours_1", "colours_2", "uv1", "uv2_raw", "indices", "joints", "weights"))


def _empty(draw: Draw) -> bool:
    return not draw.positions or not draw.indices


def encode_ps3_draw(draw: Draw, bias, scale, stored_slots=None) -> dict[str, bytes]:
    """W2-9: the descriptor, index list and streams A, B and C of a ps3 draw."""
    fmt = FORMAT_SKINNED if draw.skinned else FORMAT_RIGID
    nv, nidx = len(draw.positions), len(draw.indices)
    check_word(nv, "H", "ps3 draw vertex count (descriptor bytes 8..9)")
    check_word(nidx, "H", "ps3 draw index count (descriptor bytes 10..11)")
    byte1 = check_word(ps3_command_space(fmt, nidx), "B", "ps3 descriptor byte 1 (command space)")
    a, b, c = bytearray(), bytearray(), bytearray()
    for i, position in enumerate(draw.positions):
        a += _packed(">", "3h", f"vertex {i} position", *raw_coordinates(position, bias, scale))
        uv = _packed("<", "2H", f"vertex {i} UV1", *uv1_words(draw.uv1[i])) \
            + _packed("<", "2H", f"vertex {i} UV2", *draw.uv2_raw[i])
        colours = _colour(draw.colours_1[i], False) + _colour(draw.colours_2[i], False)
        if draw.skinned:
            a += normal_bytes(draw.normals[i]) + b"\x80"
            slots = skin_slots(draw, i, stored_slots[i] if stored_slots is not None and i < len(stored_slots) else None)
            weights = _packed("<", "4B", f"vertex {i} weights", *draw.weights[i])
            b += bytes(x for pair in zip(weights, slots) for x in pair)
            c += colours + uv
        else:
            c += normal_bytes(draw.normals[i]) + b"\0" + colours + uv
    descriptor = bytes([fmt, byte1]) + PS3_DESCRIPTOR_FIXED[fmt] \
        + _packed(">", "HH", "ps3 descriptor counts", nv, nidx) + b"\xFF" * 4
    pad = lambda raw: bytes(raw) + bytes(_align(len(raw), 16) - len(raw))  # noqa: E731
    return {
        "descriptor": descriptor,
        "indices": encode_indices(draw.indices, ">", 16),
        "a": pad(a),
        "b": pad(b) if draw.skinned else b"",
        "c": bytes(c),
    }


# --- The writer -------------------------------------------------------------------------------

def _block_bytes(value: L.PairValue | None) -> bytes:
    if value is None or value.block is None:
        return b""
    block = value.block
    if block.elements is None:
        return block.data or b""
    return b"".join(bytes(e.raw) for e in block.elements) + block.extra


def _set_bytes(record: L.Record, offset: int, data: bytes) -> bool:
    """W2-14: give the pair a block of its own when its bytes change; True when they did."""
    if _block_bytes(record.pairs.get(offset)) == data:
        return False
    record.pairs[offset] = L.PairValue(L.Block(bytes(data)) if data else None)
    return True


def _pack(raw: bytearray, order: str, fmt: str, offset: int, *values, field: str = "") -> None:
    """``struct.pack_into`` after the range checks of every value (stage 12.1)."""
    _check_values(fmt, values, field or f"word +0x{offset:X}")
    struct.pack_into(order + fmt, raw, offset, *values)


class _Context:
    def __init__(self, app: AppearanceFile, tree: L.SnoTree, old: AppearanceFile, rules=frozenset(),
                 keep_skin: bool = False):
        self.build, self.tree, self.old = app.build, tree, old
        self.rules = rules
        self.keep_skin = keep_skin  # V21-2
        self.bone_count = len(old.bones)
        self.order = build_order(app.build)
        self.has_bones = bool(old.bones)
        index_at, vertex_at = POOLS[app.build]
        self.old_index_pool = _block_bytes(tree.top.pairs.get(index_at))
        self.old_vertex_pool = _block_bytes(tree.top.pairs.get(vertex_at))
        self.index_pool = bytearray()
        self.vertex_pool = bytearray()
        self.old_index_run = 0
        self.old_vertex_run = 0
        self.geo0_moved = False
        self.geo0_triangles = False

    def stored_index_slice(self, count: int) -> bytes:
        return self.old_index_pool[self.old_index_run * 2:(self.old_index_run + count) * 2]

    def check_bone(self, bone: int, label: str, rule: str) -> None:
        if not 0 <= bone < self.bone_count:
            raise FormatError(f"{label}: joint {bone} is not one of the {self.bone_count} bones ({rule})")


def _fixed_point(sub: SubObject, positions, moved: bool, has_bones: bool, rules=frozenset()):
    """W2-2, W2-3: an unchanged mesh keeps its bias and step. A changed one keeps its bias while
    it stays within 1 of the new box centre on every axis (else round(box centre)) and takes the
    step of the W2-3 rule (Q375). With W13-2 the step of every sub-object comes from its box
    (the stored box, or the vertex extremes when it is all zero; for a changed mesh the box of
    the new positions) and its bias (Q395)."""
    bias, scale = tuple(sub.bias), tuple(sub.scale)
    if not positions:
        return bias, scale
    low = [min(p[k] for p in positions) for k in range(3)]
    high = [max(p[k] for p in positions) for k in range(3)]
    if moved:
        centre = [(low[k] + high[k]) / 2 for k in range(3)]
        if not all(abs(bias[k] - centre[k]) <= 1 for k in range(3)):
            bias = fixed_point_bias(positions)
        if "W13-2" not in rules:
            return bias, fixed_point_steps(positions, bias, has_bones)
        half = [(high[k] - low[k]) / 2 for k in range(3)]
        return bias, quantisation_steps(centre, half, bias, has_bones)
    if "W13-2" not in rules:
        return bias, scale
    if any(sub.bounds_raw) and sub.bounds_centre is not None:
        centre, half = sub.bounds_centre, sub.bounds_half
    else:  # G1-1: an all-zero box stores nothing; the decoded extremes are the box
        centre = [(low[k] + high[k]) / 2 for k in range(3)]
        half = [(high[k] - low[k]) / 2 for k in range(3)]
    return bias, quantisation_steps(centre, half, bias, has_bones)


def _decoded(positions, bias, scale):
    return [tuple(raw_coordinates(p, bias, scale)[k] * scale[k] + bias[k] for k in range(3))
            for p in positions]


def _write_box(raw: bytearray, order: str, at: int, positions) -> None:
    """W2-4: centre = (min + max) / 2, half-extent = (max - min) / 2; all-zero boxes stay zero."""
    if not any(raw[at:at + 24]) or not positions:
        return
    low = [min(p[k] for p in positions) for k in range(3)]
    high = [max(p[k] for p in positions) for k in range(3)]
    _pack(raw, order, "6f", at, *[(low[k] + high[k]) / 2 for k in range(3)],
          *[(high[k] - low[k]) / 2 for k in range(3)], field="sub-object box")


def _resize_influences(record: L.Record, offset: int, old_count: int, new_count: int) -> None:
    """W2-7: keep the compact influence list; add zero entries for added vertices or drop the
    entries of removed ones (vertices are added or removed at the end, Q377)."""
    stored = _block_bytes(record.pairs.get(offset))
    if old_count == new_count or len(stored) != INFLUENCE_RECORD * old_count:
        return
    if new_count < old_count:
        data = stored[:INFLUENCE_RECORD * new_count]
    else:
        data = stored + bytes(INFLUENCE_RECORD * (new_count - old_count))
    _set_bytes(record, offset, data)


def _fat_and_pools(ctx: _Context, record: L.Record, sub: SubObject, old: SubObject, own_indices,
                   index_field: int) -> None:
    """W2-8, W2-12: the sub-object's fat vertices go to the vertex pool; its run of the index
    pool is its own list (switch rigid), else its fat indices, else the stored slice."""
    order, raw = ctx.order, record.raw
    if sub.fat_draw is not None:
        ctx.vertex_pool += encode_fat_vertices(sub.fat_draw, order)
        _pack(raw, order, "I", SWITCH_SUB["fat"], len(sub.fat_draw.positions), field="nVertCount (+0x04)")
    elif old.cloth_vertex_count:
        start = ctx.old_vertex_run * FAT_VERTEX
        ctx.vertex_pool += ctx.old_vertex_pool[start:start + old.cloth_vertex_count * FAT_VERTEX]
    ctx.old_vertex_run += old.cloth_vertex_count
    if own_indices is not None:
        run = encode_indices(own_indices, order)
    elif sub.fat_draw is not None:
        run = encode_indices(sub.fat_draw.indices, order)
    else:
        run = ctx.stored_index_slice(old.index_count)
    ctx.index_pool += run
    _pack(raw, order, "i", index_field, len(run) // 2, field="nIndexCount")
    ctx.old_index_run += old.index_count


def _positions(draws) -> list:
    return [p for d in draws for p in d.positions]


def _empty_record(ctx: _Context, sub: SubObject, what: str) -> None:
    """W13-7: an empty skinning record or skinned draw is dropped with the rule, else refused."""
    if "W13-7" not in ctx.rules:
        raise FormatError(f"{sub.label}: an empty {what} (no vertices or no indices) is not "
                          "written; drop it with rule W13-7")


def skin_edit_only(draw: Draw, before: Draw) -> bool:
    """V21-2: a skinned record whose vertex count, positions and index list are the stored ones
    (only per-vertex values such as joints and weights changed)."""
    return draw.skinned and before.skinned and draw.joints is not None and before.joints is not None \
        and draw.positions == before.positions and draw.indices == before.indices


def _changed_skin_vertices(draw: Draw, before: Draw) -> list[int]:
    return [i for i in range(len(draw.positions))
            if draw.joints[i] != before.joints[i] or draw.weights[i] != before.weights[i]]


def _renumbered_slots(before: Draw, i: int, stored, palette: list[int]) -> bytes:
    """V21-2: an unchanged vertex keeps its stored bones, as slots of a derived palette. A
    zero-weight slot names the position of the bone its stored slot named, else 0."""
    old = before.palette or []
    out = []
    for j, (joint, weight) in enumerate(zip(before.joints[i], before.weights[i])):
        if weight:
            out.append(palette.index(joint))
        else:
            bone = old[stored[j]] if stored[j] < len(old) else None
            out.append(palette.index(bone) if bone in palette else 0)
    return bytes(out)


def kept_skin_slots(draw: Draw, before: Draw, stored, palette: list[int], derived: bool) -> list:
    """V21-2: the slots to encode a skin-edited record with. An unchanged vertex (joints and
    weights as stored) keeps its stored slot bytes, zero-weight slots included, renumbered into
    the palette when it was ``derived``; a changed vertex gets None (W2-6: the first palette
    position of each weighted bone, zero-weight slots 0)."""
    changed = set(_changed_skin_vertices(draw, before))
    out = []
    for i in range(len(draw.positions)):
        slots = stored[i] if stored is not None and i < len(stored) else None
        if i in changed or slots is None:
            out.append(None)
        elif derived:
            out.append(_renumbered_slots(before, i, slots, palette))
        else:
            out.append(bytes(slots))
    return out


def patched_influences(stored: bytes, edits) -> bytes | None:
    """V21-2 (HYPOTHESIS): the stored compact influence list with, for each changed vertex
    ``(old entry, new entry)`` (W13-6 entry form), the unique stored entry equal to its old
    entry replaced by the new one. None when an entry matches no or several stored entries (or
    two vertices the same one): the list is then rebuilt by W13-6.

    ``36`` V22-1: on real switch and x360 files the old entry is almost never found exactly once,
    so a skin edit usually rewrites the whole compact list (and may change its length). For this
    list "an edit changes only what was edited" does not hold; which layout the game needs is
    OPEN. The palette, the ranges and the slots of unchanged vertices are kept (PROVEN)."""
    if len(stored) % INFLUENCE_RECORD:
        return None
    chunks = [stored[k:k + INFLUENCE_RECORD] for k in range(0, len(stored), INFLUENCE_RECORD)]
    out = list(chunks)
    taken: set = set()
    for old, new in edits:
        if old == new:
            continue
        hits = [k for k, chunk in enumerate(chunks) if chunk == old]
        if len(hits) != 1 or hits[0] in taken:
            return None
        taken.add(hits[0])
        out[hits[0]] = new
    return b"".join(out)


def _skin_record_words(target: L.Record, piece: Draw, order: str) -> None:
    """W2-1, W2-6: the count words of a changed skinning record."""
    _pack(target.raw, order, "i", SKIN["vertex_count"], len(piece.positions), field="skinning nPackVertCount (+0x18)")
    _pack(target.raw, order, "i", SKIN["index_count"], len(piece.indices), field="skinning nPackIndexCount (+0x30)")
    _pack(target.raw, order, "i", SKIN["bones"], len(_block_bytes(target.pairs.get(SKIN["palette"]))) // 4, field="skinning bone count (+0x00)")
    start, end = SKIN["zero"]
    target.raw[start:end] = bytes(end - start)


def _stored_switch_slots(skin: L.Record, order: str) -> list[bytes]:
    stored = _block_bytes(skin.pairs.get(SKIN["vertices"]))
    slots = [stored[i * PACKED_SKINNED + 28:i * PACKED_SKINNED + 32] for i in range(len(stored) // PACKED_SKINNED)]
    return [s[::-1] for s in slots] if order == ">" else slots


def _kept_switch_record(ctx: _Context, skin: L.Record, sub: SubObject, draw: Draw, before: Draw, bias, scale):
    """V21-2 for one switch / x360 skinning record of a skin-only edit: the stored palette while
    every weighted bone is in it, else a first-use palette (W13-5). None when the derived palette
    would need a split (above 45 bones): the record is then derived as before."""
    order = ctx.order
    bones = set().union(*(_vertex_bones(draw, i) for i in range(len(draw.positions))))
    palette, derived = list(before.palette or []), False
    if not bones <= set(palette):
        palette, derived = first_use_palette(draw.joints, draw.weights), True
        if not palette or len(palette) > SKIN_PALETTE_LIMIT:
            return None
        for bone in palette:
            ctx.check_bone(bone, sub.label, "W13-5")
    piece = dataclasses.replace(draw, palette=palette)
    slots = kept_skin_slots(draw, before, _stored_switch_slots(skin, order), palette, derived)
    changed = False
    if derived:
        changed |= _set_bytes(skin, SKIN["palette"], _packed(order, f"{len(palette)}i", "skinning palette (pBoneIDs)", *palette))
    changed |= _set_bytes(skin, SKIN["vertices"], encode_packed_vertices(piece, bias, scale, order, slots))
    changed |= _set_bytes(skin, SKIN["indices"], encode_indices(piece.indices, order))
    if changed:
        _skin_record_words(skin, piece, order)
    return changed


def _switch_skins(ctx: _Context, record: L.Record, sub: SubObject, skins, old_skins, bias, scale):
    """The skinning records of a switch / x360 sub-object (W2-1, W2-6; W13-5, W13-7; V21-2).
    Returns (any record changed, triangles changed, compact influence entries, compact list edits
    or None). The edits (old entry, new entry per changed vertex) are given when every changed
    record is a skin-only edit kept by V21-2."""
    order, rules = ctx.order, ctx.rules
    skin_value = record.pairs.get(SWITCH_SUB["skins"])
    skin_records = skin_value.block.elements if skin_value is not None and skin_value.block else []
    new_records: list[L.Record] = []
    entries = bytearray()
    edits: list | None = []
    any_changed = triangles = False
    for skin, draw, before in zip(skin_records, skins, old_skins):
        same = _same_draw(draw, before)
        triangles |= draw.indices != before.indices
        if _empty(draw) and not (same and _empty(before)):
            _empty_record(ctx, sub, "skinning record")
            any_changed = triangles = True
            edits = None
            continue
        if ctx.keep_skin and not same and "W13-5" in rules and skin_edit_only(draw, before):
            changed = _kept_switch_record(ctx, skin, sub, draw, before, bias, scale)
            if changed is not None:  # V21-2
                any_changed = True
                new_records.append(skin)
                for i in range(len(draw.positions)):
                    entries += influence_entry(draw, i)
                if edits is not None:
                    edits += [(influence_entry(before, i), influence_entry(draw, i))
                              for i in _changed_skin_vertices(draw, before)]
                continue
        if not same:
            edits = None
        rebuild = "W13-5" in rules and not same
        pieces = split_skinned_draw(draw) if rebuild else [(draw, list(range(len(draw.positions))))]
        used: set = set()
        for n, (piece, ids) in enumerate(pieces):
            target = skin if n == 0 else L.Record(skin.name, bytearray(skin.raw), {})
            changed = n > 0
            if rebuild:  # W13-5: palette by first use, zero-weight slots 0
                palette = first_use_palette(piece.joints, piece.weights)
                if not palette:
                    raise FormatError(f"{sub.label}: a skinning record with no weighted slot has "
                                      "an empty palette, which is not observed (W13-7)")
                for bone in palette:
                    ctx.check_bone(bone, sub.label, "W13-5")
                piece = dataclasses.replace(piece, palette=palette)
                changed |= _set_bytes(target, SKIN["palette"], _packed(
                    order, f"{len(palette)}i", "skinning palette (pBoneIDs)", *palette))
                slots = None
            else:
                slots = _stored_switch_slots(skin, order)
            changed |= _set_bytes(target, SKIN["vertices"], encode_packed_vertices(piece, bias, scale, order, slots))
            changed |= _set_bytes(target, SKIN["indices"], encode_indices(piece.indices, order))
            if changed:  # W2-1, W2-6
                _skin_record_words(target, piece, order)
            any_changed |= changed or not same
            new_records.append(target)
            used.update(ids)
        for i in sorted(used):  # W13-6: one entry per vertex, each distinct vertex once
            entries += influence_entry(draw, i)
    if len(new_records) != len(skin_records) or any(a is not b for a, b in zip(new_records, skin_records)):
        record.pairs[SWITCH_SUB["skins"]] = L.PairValue(L.Block(elements=new_records) if new_records else None)
    return any_changed, triangles, bytes(entries), edits


def _switch_sub(ctx: _Context, record: L.Record, sub: SubObject, old: SubObject) -> tuple[bool, bool]:
    order, raw, rules = ctx.order, record.raw, ctx.rules
    rigid = [d for d in sub.draws if d.kind == "packed" and not d.skinned]
    skins = [d for d in sub.draws if d.kind == "packed" and d.skinned]
    old_rigid = [d for d in old.draws if d.kind == "packed" and not d.skinned]
    old_skins = [d for d in old.draws if d.kind == "packed" and d.skinned]
    if len(rigid) != len(old_rigid) or len(skins) != len(old_skins):
        raise FormatError(f"{sub.label}: adding or removing packed lists is not described (Q376)")
    fixed, old_fixed = _positions(rigid + skins), _positions(old_rigid + old_skins)
    moved = fixed != old_fixed
    bias, scale = _fixed_point(sub, fixed, moved, ctx.has_bones, rules)
    if moved or (bias, scale) != (tuple(sub.bias), tuple(sub.scale)):
        _pack(raw, order, "3f", SWITCH_SUB["bias"], *bias, field="sub-object bias")
        _pack(raw, order, "3f", SWITCH_SUB["scale"], *scale, field="sub-object step")
    triangles = moved
    for draw, before in zip(rigid, old_rigid):
        _set_bytes(record, SWITCH_SUB["packed"], encode_packed_vertices(draw, bias, scale, order))
        _pack(raw, order, "i", SWITCH_SUB["packed_count"], len(draw.positions), field="nPackVertCount (+0xA0)")
        triangles |= draw.indices != before.indices
    skins_changed, skin_triangles, entries, edits = _switch_skins(ctx, record, sub, skins, old_skins, bias, scale)
    triangles |= skin_triangles
    fat_moved = sub.fat_draw is not None and old.fat_draw is not None and \
        sub.fat_draw.positions != old.fat_draw.positions
    if sub.fat_draw is not None and old.fat_draw is not None:
        triangles |= fat_moved or sub.fat_draw.indices != old.fat_draw.indices
    _fat_and_pools(ctx, record, sub, old, rigid[0].indices if rigid else None, SWITCH_SUB["index"])
    if "W13-6" in rules and skins and skins_changed:
        if edits is not None:  # V21-2: the stored list with the changed entries replaced
            patched = patched_influences(_block_bytes(record.pairs.get(SWITCH_SUB["influence"])), edits)
            entries = entries if patched is None else patched
        _set_bytes(record, SWITCH_SUB["influence"], entries)  # W13-6: vertex order (Q399)
    else:
        old_n = sum(len(d.positions) for d in old_skins) or (
            len(old.fat_draw.positions) if old.fat_draw is not None and ctx.has_bones else 0)
        new_n = sum(len(d.positions) for d in skins) or (
            len(sub.fat_draw.positions) if sub.fat_draw is not None and ctx.has_bones else 0)
        _resize_influences(record, SWITCH_SUB["influence"], old_n, new_n)
    if moved:
        _write_box(raw, order, SWITCH_SUB["box"], _decoded(fixed, bias, scale))
    elif fat_moved and not fixed:
        _write_box(raw, order, SWITCH_SUB["box"], sub.fat_draw.positions)
    return moved or fat_moved, triangles


def _ps3_ranges(ctx: _Context, draw: Draw, before: Draw, same: bool, label: str):
    """W13-4: the bone ranges of a skinned draw and the slots to encode it with. B is the weighted
    bones plus, for an unchanged draw, the bones its stored zero-weight slots name; a changed draw
    writes zero-weight slots as 0."""
    bones: set = set()
    for i in range(len(draw.positions)):
        bones |= _vertex_bones(draw, i)
    old_palette = before.palette or []
    if same and before.local_joints is not None:
        for weights, slots in zip(before.weights, before.local_joints):
            bones.update(old_palette[s] for w, s in zip(weights, slots) if not w and s < len(old_palette))
    for bone in bones:
        ctx.check_bone(bone, label, "W13-4")
    ranges = ps3_bone_ranges(bones)
    palette = ranges_palette(ranges)
    slots = None
    if same and before.local_joints is not None:
        slots = [tuple(palette.index(old_palette[s]) if s < len(old_palette) and old_palette[s] in palette else 0
                       for s in vertex) for vertex in before.local_joints]
    return ranges, dataclasses.replace(draw, palette=palette), slots


def _ps3_kept_ranges(ctx: _Context, draw: Draw, before: Draw, stored_ranges, label: str):
    """V21-2 for a skin-only edit of a ps3 draw: the stored bone ranges while every weighted bone
    of the draw is in them, else ranges derived by W13-4 from the weighted bones. Unchanged
    vertices keep their stored slots (renumbered when the ranges were derived)."""
    bones: set = set()
    for i in range(len(draw.positions)):
        bones |= _vertex_bones(draw, i)
    for bone in bones:
        ctx.check_bone(bone, label, "W13-4")
    palette = list(before.palette or [])
    ranges, derived = tuple(stored_ranges), False
    if not bones <= set(palette):
        ranges, derived = ps3_bone_ranges(bones), True
        palette = ranges_palette(ranges)
    slots = kept_skin_slots(draw, before, before.local_joints, palette, derived)
    return ranges, dataclasses.replace(draw, palette=palette), slots


def _ps3_sub(ctx: _Context, record: L.Record, sub: SubObject, old: SubObject) -> tuple[bool, bool]:
    order, raw, rules = ctx.order, record.raw, ctx.rules
    fat, old_fat = sub.fat_draw, old.fat_draw
    draws = [d for d in sub.draws if d.kind == "ps3"]
    old_draws = [d for d in old.draws if d.kind == "ps3"]
    value = record.pairs.get(PS3_SUB["draws"])
    records = value.block.elements if value is not None and value.block else []
    if len(draws) != len(old_draws) or len(records) != len(draws):
        raise FormatError(f"{sub.label}: adding or removing draws is not described (Q376)")
    fixed, old_fixed = _positions(draws), _positions(old_draws)
    moved = fixed != old_fixed
    bias, scale = _fixed_point(sub, fixed, moved, ctx.has_bones, rules)
    if moved or (bias, scale) != (tuple(sub.bias), tuple(sub.scale)):
        _pack(raw, order, "3f", PS3_SUB["bias"], *bias, field="sub-object bias")
        _pack(raw, order, "3f", PS3_SUB["scale"], *scale, field="sub-object step")
    triangles = moved
    kept, kept_draws = [], []
    changed_mesh = moved
    for k, (draw_record, draw, before) in enumerate(zip(records, draws, old_draws)):
        same = _same_draw(draw, before)
        triangles |= draw.indices != before.indices
        changed_mesh |= not same
        if _empty(draw) and not (same and _empty(before)):
            _empty_record(ctx, sub, "ps3 draw")
            triangles = True
            continue
        label = f"{sub.label}, draw {k}"
        stored_ranges = struct.unpack_from(">4H", draw_record.raw, PS3_RANGES)
        ranges, slots = stored_ranges, before.local_joints
        if draw.skinned and "W13-4" in rules:
            if ctx.keep_skin and not same and before.local_joints is not None and skin_edit_only(draw, before):
                ranges, draw, slots = _ps3_kept_ranges(ctx, draw, before, stored_ranges, label)  # V21-2
            else:
                ranges, draw, slots = _ps3_ranges(ctx, draw, before, same, label)
        blocks = encode_ps3_draw(draw, bias, scale, slots)
        work = None
        if draw.skinned and "W13-1" in rules:
            work = ps3_skinned_work_size(len(draw.positions), len(draw.indices), blocks["descriptor"][1], ranges)
            if work > PS3_WORK_LIMIT:
                raise FormatError(f"{label}: skinned draw work buffer {work} is above 0xC000 (W13-1)")
        kept.append(draw_record)
        kept_draws.append(draw)
        r = draw_record.raw
        if not all(_block_bytes(draw_record.pairs.get(DRAW[name])) == blocks[name]
                   for name in ("descriptor", "indices", "a", "b", "c")):
            words = _ps3_draw_words(draw_record, blocks, len(draw.positions), len(draw.indices), label, work)
            for name in ("descriptor", "indices", "a", "b", "c"):
                _set_bytes(draw_record, DRAW[name], blocks[name])
            for name, fmt, values in words:
                _pack(r, ">", fmt, DRAW[name], *values, field=name)
        elif work is not None:
            _pack(r, ">", "I", DRAW["work"], work, field=f"{label}: work-buffer size (+0x98, W13-1)")
        if tuple(ranges) != tuple(stored_ranges):
            _pack(r, ">", "4H", PS3_RANGES, *ranges, field=f"{label}: bone ranges (+0x60, W13-4)")
    if len(kept) != len(records):  # W13-7: dropped draws
        record.pairs[PS3_SUB["draws"]] = L.PairValue(L.Block(elements=kept) if kept else None)
        _pack(raw, order, "i", PS3_SUB["draw_count"], len(kept), field="draw count (+0xFC)")
    fat_moved = fat is not None and old_fat is not None and fat.positions != old_fat.positions
    if fat is not None and old_fat is not None:
        triangles |= fat_moved or fat.indices != old_fat.indices
        changed_mesh |= fat_moved or fat.indices != old_fat.indices
    _fat_and_pools(ctx, record, sub, old, None, PS3_SUB["index"])
    if "W13-6" in rules and changed_mesh:
        # W13-6: present exactly in sub-objects owning fat vertices in a file with bones; 6 x the
        # skinned draws' vertices, else 6 x the fat vertices; stored entries kept (Q400)
        fat_n = len(fat.positions) if fat is not None else 0
        skinned_n = sum(len(d.positions) for d in kept_draws if d.skinned)
        want = INFLUENCE_RECORD * (skinned_n or fat_n) if fat_n and ctx.has_bones else 0
        stored = _block_bytes(record.pairs.get(PS3_SUB["influence"]))
        _set_bytes(record, PS3_SUB["influence"], stored[:want] + bytes(max(0, want - len(stored))))
    else:
        old_n = sum(len(d.positions) for d in old_draws) or (len(old_fat.positions) if old_fat else 0)
        new_n = sum(len(d.positions) for d in draws) or (len(fat.positions) if fat else 0)
        _resize_influences(record, PS3_SUB["influence"], old_n, new_n)
    if moved:
        _write_box(raw, order, PS3_SUB["box"], _decoded(fixed, bias, scale))
    elif fat_moved and not fixed:
        _write_box(raw, order, PS3_SUB["box"], fat.positions)
    return moved or fat_moved, triangles


def _ps3_draw_words(record: L.Record, blocks: dict, nv: int, nidx: int, label: str, work: int | None = None) -> list:
    """W2-9, W2-10: the record words of a changed ps3 draw, every one checked before anything is
    written: the work buffer (0xC000) first, then the stream sizes (each at most 0x4000, G1-5),
    then the 16-bit index-pair pieces and size words. ``work`` is a value already derived and
    checked (W13-1)."""
    fmt = blocks["descriptor"][0]
    if work is None:
        stored_work = struct.unpack_from(">I", record.raw, DRAW["work"])[0]
        try:
            work = ps3_work_size(fmt, nv, nidx, stored_work)
        except FormatError as exc:
            raise FormatError(f"{label}: {exc}") from None
    a_size, b_size = len(blocks["a"]), len(blocks["b"])
    for name, size in (("stream A", a_size), ("stream B", b_size)):
        if size > PS3_PIECE:
            raise FormatError(f"{label}: {name} of {size} bytes is above 0x4000 (G1-5)")
    index_size = len(blocks["indices"])
    pieces = (index_size, 0) if index_size <= PS3_PIECE else (PS3_PIECE, index_size - PS3_PIECE)
    for k, piece in enumerate(pieces):
        check_word(piece, "H", f"{label}: index-pair piece {k} (+0x{DRAW['index_pieces'] + 2 * k:X})")
    area = _align(4 * nv, 32) if fmt == FORMAT_SKINNED else _align(2 * nv, 16)
    words = [("index_count", "I", (nidx,)), ("vertex_count", "I", (nv,)),
             ("index_pieces", "HH", pieces), ("a_words", "16H", (a_size,) + (0,) * 15),
             ("b_words", "HH", (b_size, 0)), ("work", "I", (work,)), ("area", "I", (area,))]
    for name, fmt_words, values in words:
        _check_values(fmt_words, values, f"{label}: {name} (+0x{DRAW[name]:X})")
    return words


def _structure(ctx: _Context, records: list[L.Record], box_at: int) -> None:
    """W2-11: the union of the geo-set-0 sub-object boxes and a sphere around its centre reaching
    the farthest corner of any sub-object box. W2-13: an empty octree when triangles changed
    (without W13-3; with it the octree is built again, ``_rebuild_octree``)."""
    top, order = ctx.tree.top, ctx.order
    if ctx.geo0_moved:
        boxes = []
        for record in records:
            centre_half = struct.unpack_from(order + "6f", record.raw, box_at)
            if any(record.raw[box_at:box_at + 24]):
                boxes.append(centre_half)
        if boxes:
            low = [min(b[k] - b[k + 3] for b in boxes) for k in range(3)]
            high = [max(b[k] + b[k + 3] for b in boxes) for k in range(3)]
            centre = [(low[k] + high[k]) / 2 for k in range(3)]
            _pack(top.raw, order, "6f", STRUCT_BOX, *centre, *[(high[k] - low[k]) / 2 for k in range(3)], field="structure box")
            radius = 0.0
            for b in boxes:
                for corner in range(8):
                    point = [b[k] + (b[k + 3] if corner >> k & 1 else -b[k + 3]) for k in range(3)]
                    radius = max(radius, math.dist(point, centre))
            _pack(top.raw, order, "4f", STRUCT_SPHERE, *centre, radius, field="structure sphere")
    if ctx.geo0_triangles and "W13-3" not in ctx.rules:
        top.raw[STRUCT_OCTREE:STRUCT_OCTREE + OCTREE_COUNTS] = bytes(OCTREE_COUNTS)
        for offset in OCTREE_PAIRS:
            top.pairs[offset] = L.PairValue(None)


def changed_mesh_octree(mesh: AppearanceFile, stored: O.Octree | None) -> O.Octree | None:
    """W13-3, rule for a changed mesh: keep the stored root cube when the new primitives' box
    fits inside it, else a derived root (doubled while a primitive would lie on its open high
    face, Q397); build the tree by the subdivision rule. Without primitives: no octree when none
    was stored, else the stored root with 8 empty leaves and the stored maxima, raised to the
    counts where needed (Q396, V15-4, V15-5)."""
    triangles = O.octree_triangles(mesh)
    if not triangles:
        if stored is None:
            return None
        empty = O.build_octree(mesh, stored.root, triangles)
        empty.maxima = O.raised_maxima(stored.maxima, empty)
        return empty
    if stored is not None and O.root_holds(stored.root, triangles):
        return O.build_octree(mesh, stored.root, triangles)
    low, size = O.derived_root(triangles)
    centre = [low[k] + size / 2 for k in range(3)]
    while not O.root_holds((low, size), triangles):
        size *= 2
        low = tuple(O.f32(centre[k] - size / 2) for k in range(3))
    return O.build_octree(mesh, (low, size), triangles)


def _rebuild_octree(tree: L.SnoTree, stored: O.Octree | None) -> None:
    """W13-3 in the writer: the tree is laid out, read back, and its octree built from the mesh
    as written (decoded positions), then stored as new blocks."""
    mesh = parse_appearance(L.write_tree(tree), tree.build)
    O.set_octree(tree, changed_mesh_octree(mesh, stored))


def app_mesh_tree(app: AppearanceFile, mesh: AppearanceFile | None = None, derive=False,
                  keep_skin_bytes: bool = False) -> L.SnoTree:
    """W2-1 to W2-15: the structure tree of ``app``'s source with every mesh block of ``mesh`` (a
    decoded ``.app``, by default ``app`` itself, changed by the caller) encoded again.
    ``derive`` (True, or rule names of ``RULES``) applies the stage-13 rules: W13-1, W13-2 and
    W13-4 derive the values the writer otherwise keeps; W13-3, W13-5 and W13-6 apply to changed
    meshes; W13-7 drops empty skinning records and skinned draws (else they are refused).
    ``keep_skin_bytes`` (V21-2, used by the mesh import): a skinning record or ps3 draw whose
    vertex count, positions and indices are unchanged keeps its stored palette or bone ranges
    while every weighted bone is in them, the stored slot bytes of its unchanged vertices, and
    (switch, x360) the stored compact influence list with only the changed entries replaced."""
    rules = rule_set(derive)
    mesh = app if mesh is None else mesh
    if mesh.build != app.build:
        raise FormatError(f"mesh of build {mesh.build} for an .app of build {app.build}")
    source = bytes(app.source)
    tree = L.parse_tree(source, "app", app.build)
    for path, data in app.block_edits.items():
        L.set_block(tree, path, data)
    stored_octree = O.read_octree(tree)
    old = parse_appearance(source, app.build)
    ctx = _Context(app, tree, old, rules, keep_skin_bytes)
    sub_fn = _ps3_sub if app.build == "ps3" else _switch_sub
    box_at = (PS3_SUB if app.build == "ps3" else SWITCH_SUB)["box"]
    geo0_records: list[L.Record] = []
    for geo, (subs, old_subs) in enumerate(((mesh.sub_objects, old.sub_objects),
                                            (mesh.extra_sub_objects, old.extra_sub_objects))):
        value = tree.top.pairs.get(GEO_SUB_OBJECTS[geo])
        records = value.block.elements if value is not None and value.block else []
        if not len(records) == len(subs) == len(old_subs):
            raise FormatError(f"geo set {geo}: {len(subs)} sub-objects for {len(records)} records (Q376)")
        for record, sub, before in zip(records, subs, old_subs):
            moved, triangles = sub_fn(ctx, record, sub, before)
            if geo == 0:
                ctx.geo0_moved |= moved
                ctx.geo0_triangles |= triangles
        if geo == 0:
            geo0_records = records
    index_at, vertex_at = POOLS[app.build]
    _set_bytes(tree.top, index_at, bytes(ctx.index_pool))
    _set_bytes(tree.top, vertex_at, bytes(ctx.vertex_pool))
    _structure(ctx, geo0_records, box_at)
    if ctx.geo0_triangles and "W13-3" in rules:
        _rebuild_octree(tree, stored_octree)
    return tree


def write_app_mesh(app: AppearanceFile, mesh: AppearanceFile | None = None, check: bool = True,
                   derive=False, keep_skin_bytes: bool = False) -> bytes:
    """Encode ``mesh`` (default: ``app`` as changed by the caller) into ``app``'s ``.app`` and lay
    it out (W2-1 to W2-15, W11; ``derive``: the W13 rules, ``keep_skin_bytes``: V21-2, see
    ``app_mesh_tree``). An unchanged
    mesh gives the source bytes (with ``derive`` too, for a file that follows the rules). With
    ``check`` the output is read again and must give the same counts (the same index count per
    sub-object when W13-5 or W13-7 may split or drop records)."""
    rules = rule_set(derive)
    out = L.write_tree(app_mesh_tree(app, mesh, rules, keep_skin_bytes))
    if check:
        again = parse_appearance(out, app.build)
        mesh = app if mesh is None else mesh
        for a, b in zip(again.sub_objects + again.extra_sub_objects, mesh.sub_objects + mesh.extra_sub_objects):
            if rules & {"W13-5", "W13-7"}:
                same = sum(len(d.indices) for d in a.draws) == sum(len(d.indices) for d in b.draws)
            else:
                same = [len(d.positions) for d in a.draws] == [len(d.positions) for d in b.draws] and \
                    [len(d.indices) for d in a.draws] == [len(d.indices) for d in b.draws]
            if not same:
                raise FormatError(f"{b.label}: the written mesh does not read back with its counts")
    return out


def index_pool_problems(app: AppearanceFile) -> list[str]:
    """W2-12 (note to ``05``): the switch and x360 index pool is exactly 2 x the sum of the
    sub-object index counts, with no spare tail."""
    if app.build == "ps3":
        return []
    tree = L.parse_tree(bytes(app.source), "app", app.build)
    pool = _block_bytes(tree.top.pairs.get(POOLS[app.build][0]))
    total = sum(s.index_count for s in app.sub_objects + app.extra_sub_objects)
    if len(pool) != 2 * total:
        return [f"index pool of {len(pool)} bytes for {total} indices (W2-12: no spare tail)"]
    return []


__all__ = [
    "RULES", "app_mesh_tree", "changed_mesh_octree", "encode_fat_vertices", "encode_indices",
    "encode_packed_vertices", "encode_ps3_draw", "first_use_palette", "fixed_point_bias",
    "fixed_point_steps", "index_pool_problems", "influence_entry", "kept_skin_slots", "normal_bytes",
    "patched_influences", "skin_edit_only",
    "ps3_bone_ranges", "ps3_command_space", "ps3_skinned_work_size", "ps3_work_size",
    "quantisation_steps", "ranges_palette", "raw_coordinates", "rule_set", "skin_slots",
    "split_skinned_draw", "uv1_words", "write_app_mesh",
]
