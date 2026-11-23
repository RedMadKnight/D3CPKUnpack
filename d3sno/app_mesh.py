"""The mesh-level ``.app`` writer of ``25_corrections14.md`` section 1 (W2-1 to W2-15).

The writer starts from the structure tree of the source (``sno_layout``, W11), encodes every
vertex, index, pool, skinning and ps3 draw block from a decoded mesh (an ``AppearanceFile``,
possibly changed), recomputes the counts and derived fields of what changed, keeps the values the
description lists as kept, and lays the file out with the generic writer.

A block whose new bytes equal the stored ones keeps its stored block (so shared blocks stay
shared and an unchanged mesh gives the source bytes); a block whose bytes change gets a block of
its own (W2-14). Records are changed only where something they describe changed.

Stage 17 (``37_mesh_parts.md`` W17-10, W17-11, W17-14; record rules in ``app_parts``): the mesh may
add and remove sub-objects (either geo set), ps3 draws, switch / x360 skinning records and rigid
lists; ``SubObject.origin`` and ``Draw.origin`` say which stored part each comes from (None: new).
New sub-objects are built from zero (W17-3), new ps3 draws by W17-5 / W17-5a, skinning records by
W17-7, material entries by W17-8; ``transplant`` moves a sub-object in from another file and
``rebuild_from_zero`` rebuilds every record of a file (the "Check" of ``37``). This replaces the
refusals of Q376. Without the stage-13 rules the visual-mesh octree of a changed mesh is emptied
(W2-13). The ``derive`` option applies the rules of ``26_corrections15.md`` (W13-1 to W13-7): derived
work-buffer sizes, steps and ps3 bone ranges, an octree built again for a changed mesh (module
``octree``), first-use palettes split at 45 bones, compact influence lists, and empty skinning
records dropped.
"""

from __future__ import annotations

import copy
import dataclasses
import math
import re
import struct

from . import app_parts as PP
from . import octree as O
from . import sno_layout as L
from .appearance import (
    BONE_RANGE_UNIT, FAT_VERTEX, FORMAT_RIGID, FORMAT_SKINNED, GEOSET_NAMES, PS3_DESCRIPTOR_FIXED,
    PS3_FORMAT_BLOCKS, PS3_PIECE, AppearanceFile, Draw, SubObject, parse_appearance,
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
RULES = ("W13-1", "W13-2", "W13-3", "W13-4", "W13-5", "W13-6", "W13-7", "W17-14")
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


def encode_packed_vertices(draw: Draw, bias, scale, order: str, slots=None, label: str = "") -> bytes:
    """W2-5: 28-byte rigid or 36-byte skinned packed vertices in the file byte order. On x360
    (big-endian) the colour, joint-slot and weight groups are byte-reversed. ``slots`` gives the
    stored slot bytes per vertex (W2-6); weighted slots come from the palette. ``label`` (the
    part's label, ``39`` V24-2) begins a palette refusal."""
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
            joint_slots = skin_slots(draw, i, slots[i] if slots is not None and i < len(slots) else None, label)
            out += (joint_slots[::-1] + weights[::-1]) if flip else (joint_slots + weights)
    return bytes(out)


def skin_slots(draw: Draw, i: int, stored: bytes | None, label: str = "") -> bytes:
    """W2-6: a weighted slot is the first palette position of the vertex's joint (a stored slot
    that already names the joint is kept, Q379); a slot of weight 0 keeps its stored value (0 for
    a new vertex). ``39`` V24-2: a refusal begins with ``label`` (the draw or record label) when
    given."""
    palette = draw.palette or []
    slots = []
    for j, (joint, weight) in enumerate(zip(draw.joints[i], draw.weights[i])):
        if weight:
            if joint not in palette:
                where = f"{label}: " if label else ""
                raise FormatError(f"{where}vertex {i}: joint {joint} is not in the palette (W2-6)")
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
    """W13-5: the draw as skinning records of at most ``limit`` bones. ``38`` V23-4: a written
    non-last record may hold fewer than 45 entries (the next triangle brings two or three new
    bones); stored files always have 45 there, and no check requires it of written output. Triangles are taken in
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


def encode_ps3_draw(draw: Draw, bias, scale, stored_slots=None, label: str = "") -> dict[str, bytes]:
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
            slots = skin_slots(draw, i, stored_slots[i] if stored_slots is not None and i < len(stored_slots) else None,
                               label)
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


@dataclasses.dataclass
class PartSource:
    """W17-11: the sub-object (``geo_set``, ``index``) of ``app``, another parsed ``.app`` of the
    same build, whose stored record a new sub-object copies (``SubObject.stored_from``)."""

    app: AppearanceFile
    geo_set: int
    index: int


@dataclasses.dataclass
class _Pools:
    """The stored pools a sub-object's runs are read from and where its runs start (G1-2)."""

    index_pool: bytes = b""
    vertex_pool: bytes = b""
    index_start: int = 0
    vertex_start: int = 0

    def index_slice(self, count: int) -> bytes:
        return self.index_pool[self.index_start * 2:(self.index_start + count) * 2]

    def vertex_slice(self, count: int) -> bytes:
        return self.vertex_pool[self.vertex_start * FAT_VERTEX:(self.vertex_start + count) * FAT_VERTEX]


def _pool_runs(app: AppearanceFile) -> dict[tuple[int, int], tuple[int, int]]:
    """G1-2, W2-12: where each sub-object's runs start (vertex, index), geo set 0 then 1."""
    runs, vertices, indices = {}, 0, 0
    for sub in list(app.sub_objects) + list(app.extra_sub_objects):
        runs[(sub.geo_set, sub.index)] = (vertices, indices)
        vertices += sub.cloth_vertex_count
        indices += sub.index_count
    return runs


def _file_format(app: AppearanceFile) -> int:
    """W17-5: the ps3 draw format of the file: that of its draws, else 0x83 with bones and 0x81
    without."""
    formats = {FORMAT_SKINNED if d.skinned else FORMAT_RIGID
               for sub in app.sub_objects + app.extra_sub_objects for d in sub.draws if d.kind == "ps3"}
    if len(formats) == 1:
        return formats.pop()
    return FORMAT_SKINNED if app.bones else FORMAT_RIGID


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
        self.runs = _pool_runs(old)
        self.index_pool = bytearray()
        self.vertex_pool = bytearray()
        self.geo0_moved = False
        self.geo0_triangles = False
        self.report = PP.PartsReport()  # W17-10, W17-14
        self.parts_changed = False  # a part was added or removed somewhere (W17-6 pass)
        self.ps3_format = _file_format(old) if app.build == "ps3" else None
        self.donors: dict[int, tuple] = {}
        self.written: list[tuple] = []  # (record, mesh sub-object, label, exact counts)

    def template_pools(self, geo: int, index: int) -> _Pools:
        vertices, indices = self.runs[(geo, index)]
        return _Pools(self.old_index_pool, self.old_vertex_pool, indices, vertices)

    def donor(self, app: AppearanceFile):
        """A donor file (W17-11, W17-8): its tree, parse, pool runs and pools, read once."""
        if app.build != self.build:
            raise FormatError(f"a part of a {app.build} file for a {self.build} file (W17-11)")
        key = id(app)
        if key not in self.donors:
            source = bytes(app.source)
            tree = L.parse_tree(source, "app", app.build)
            parsed = parse_appearance(source, app.build)
            index_at, vertex_at = POOLS[app.build]
            self.donors[key] = (tree, parsed, _pool_runs(parsed), _block_bytes(tree.top.pairs.get(index_at)),
                                _block_bytes(tree.top.pairs.get(vertex_at)), app)
        return self.donors[key]

    def mark(self) -> tuple[int, int]:
        return len(self.index_pool), len(self.vertex_pool)

    def rollback(self, mark: tuple[int, int]) -> None:
        del self.index_pool[mark[0]:]
        del self.vertex_pool[mark[1]:]

    def check_bone(self, bone: int, label: str, rule: str) -> None:
        if not 0 <= bone < self.bone_count:
            raise FormatError(f"{label}: joint {bone} is not one of the {self.bone_count} bones ({rule})")


def _fixed_point(sub: SubObject, positions, moved: bool, has_bones: bool, rules=frozenset(), new: bool = False):
    """W2-2, W2-3: an unchanged mesh keeps its bias and step. A changed one keeps its bias while
    it stays within 1 of the new box centre on every axis (else round(box centre)) and takes the
    step of the W2-3 rule (Q375). With W13-2 the step of every sub-object comes from its box
    (the stored box, or the vertex extremes when it is all zero; for a changed mesh the box of
    the new positions) and its bias (Q395). ``new`` (W17-3, a part built from zero): bias
    round(box centre), step by W13-2."""
    bias, scale = tuple(sub.bias), tuple(sub.scale)
    if not positions:
        return bias, scale
    low = [min(p[k] for p in positions) for k in range(3)]
    high = [max(p[k] for p in positions) for k in range(3)]
    if new:
        bias = fixed_point_bias(positions)
        centre = [(low[k] + high[k]) / 2 for k in range(3)]
        half = [(high[k] - low[k]) / 2 for k in range(3)]
        return bias, quantisation_steps(centre, half, bias, has_bones)
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


def _write_box(raw: bytearray, order: str, at: int, positions, force: bool = False) -> None:
    """W2-4: centre = (min + max) / 2, half-extent = (max - min) / 2; all-zero boxes stay zero
    (``force``: a part built from zero, W17-3, always gets its box)."""
    if (not force and not any(raw[at:at + 24])) or not positions:
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
                   index_field: int, pools: _Pools) -> None:
    """W2-8, W2-12: the sub-object's fat vertices go to the vertex pool; its run of the index
    pool is its own list (switch rigid), else its fat indices, else the stored slice (read from
    ``pools``: the template's, or a donor's for a moved part)."""
    order, raw = ctx.order, record.raw
    if sub.fat_draw is not None:
        ctx.vertex_pool += encode_fat_vertices(sub.fat_draw, order)
        _pack(raw, order, "I", SWITCH_SUB["fat"], len(sub.fat_draw.positions), field="nVertCount (+0x04)")
    elif old.cloth_vertex_count:
        ctx.vertex_pool += pools.vertex_slice(old.cloth_vertex_count)
    if own_indices is not None:
        run = encode_indices(own_indices, order)
    elif sub.fat_draw is not None:
        run = encode_indices(sub.fat_draw.indices, order)
    else:
        run = pools.index_slice(old.index_count)
    ctx.index_pool += run
    _pack(raw, order, "i", index_field, len(run) // 2, field="nIndexCount")


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


def _kept_switch_record(ctx: _Context, skin: L.Record, sub: SubObject, draw: Draw, before: Draw, bias, scale,
                        label: str = ""):
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
    changed |= _set_bytes(skin, SKIN["vertices"], encode_packed_vertices(piece, bias, scale, order, slots, label))
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
    for k, (skin, draw, before) in enumerate(zip(skin_records, skins, old_skins)):
        rlabel = f"{sub.label}, skinning record {k}"
        same = _same_draw(draw, before)
        triangles |= draw.indices != before.indices
        if _empty(draw) and not (same and _empty(before)):
            _empty_record(ctx, sub, "skinning record")
            any_changed = triangles = True
            edits = None
            continue
        if ctx.keep_skin and not same and "W13-5" in rules and skin_edit_only(draw, before):
            changed = _kept_switch_record(ctx, skin, sub, draw, before, bias, scale, rlabel)
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
            changed |= _set_bytes(target, SKIN["vertices"], encode_packed_vertices(piece, bias, scale, order, slots,
                                                                                    rlabel))
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


def _match_parts(parts, count: int, label: str, what: str, all_new: bool = False):
    """W17-10: match the mesh's parts of one kind (draws, skinning records, rigid list) to the
    ``count`` stored ones by ``Draw.origin``. Returns ([(part, stored index or None)], removed stored
    indices). When no part carries an origin, they match by position (extra parts are new, missing
    ones removed). ``all_new``: every part is new (a sub-object built from zero)."""
    if all_new:
        return [(p, None) for p in parts], []
    if all(getattr(p, "origin", None) is None for p in parts):
        return [(p, k if k < count else None) for k, p in enumerate(parts)], list(range(len(parts), count))
    seen: set = set()
    out = []
    for part in parts:
        origin = getattr(part, "origin", None)
        if origin is not None:
            if isinstance(origin, bool) or not isinstance(origin, int) or not 0 <= origin < count:
                raise FormatError(f"{label}: {what} origin {origin!r} is not one of the {count} stored "
                                  f"{what}s (W17-10)")
            if origin in seen:
                raise FormatError(f"{label}: two {what}s come from stored {what} {origin}; mark the copy "
                                  "new (origin None) (W17-10)")
            seen.add(origin)
        out.append((part, origin))
    return out, [k for k in range(count) if k not in seen]


def _check_draw(draw: Draw, label: str) -> None:
    """The per-vertex lists of a new draw have one entry per position; indices are whole triangles
    within the vertices."""
    n = len(draw.positions)
    names = ["normals", "colours_1", "colours_2", "uv1", "uv2_raw"] + (["joints", "weights"] if draw.skinned else [])
    for name in names:
        values = getattr(draw, name)
        if values is None or len(values) != n:
            raise FormatError(f"{label}: {name} has {0 if values is None else len(values)} entries for {n} vertices")
    if len(draw.indices) % 3:
        raise FormatError(f"{label}: {len(draw.indices)} indices are not whole triangles")
    if draw.indices and not 0 <= min(draw.indices) <= max(draw.indices) < n:
        raise FormatError(f"{label}: an index is past the {n} vertices")


def merge_draws(draws) -> Draw:
    """W17-14, W17-7: the geometry of several skinned records as one list (vertices in order, a
    vertex of two records twice), to be split again by W17-7."""
    out = Draw(True, [], [], [], [], [], [], [], joints=[], weights=[], kind="packed")
    for draw in draws:
        base = len(out.positions)
        for name in ("positions", "normals", "colours_1", "colours_2", "uv1", "uv2_raw", "joints", "weights"):
            getattr(out, name).extend(getattr(draw, name))
        out.indices.extend(i + base for i in draw.indices)
    return out


def _regroup_skins(ctx: _Context, record: L.Record, label: str, skins, bias, scale) -> int:
    """W17-7 (W17-14): the skinning records of a sub-object from zero: its non-empty skinned
    geometry, in order, is split before a triangle would bring the palette to a 46th bone; palette by
    first use with zero-weight slots 0, local indices, a vertex used by two records written in both;
    the compact influence list by W13-6. Returns the number of records."""
    order = ctx.order
    live = [d for d in skins if not _empty(d)]
    records: list[L.Record] = []
    entries = bytearray()
    if live:
        merged = merge_draws(live)
        for i in range(len(merged.positions)):
            for bone in _vertex_bones(merged, i):
                ctx.check_bone(bone, label, "W17-7")
        used: set = set()
        for piece, ids in split_skinned_draw(merged):
            palette = first_use_palette(piece.joints, piece.weights)
            if not palette:
                raise FormatError(f"{label}: a skinning record with no weighted slot has an empty palette (W13-7)")
            piece = dataclasses.replace(piece, palette=palette)
            skin = L.new_record(ctx.build, "app", PP.SKIN_RECORD_NAME)
            _set_bytes(skin, SKIN["palette"], _packed(order, f"{len(palette)}i", "skinning palette (pBoneIDs)", *palette))
            _set_bytes(skin, SKIN["vertices"], encode_packed_vertices(piece, bias, scale, order, None,
                                                                      f"{label}, skinning record {len(records)}"))
            _set_bytes(skin, SKIN["indices"], encode_indices(piece.indices, order))
            skin.raw[:] = PP.skin_record_words(skin, order)
            records.append(skin)
            used.update(ids)
        for i in sorted(used):  # W13-6: one entry per vertex, each distinct vertex once
            entries += influence_entry(merged, i)
    record.pairs[SWITCH_SUB["skins"]] = L.PairValue(L.Block(elements=records) if records else None)
    _set_bytes(record, SWITCH_SUB["influence"], bytes(entries))
    return len(records)


def _cloth_part(old: SubObject) -> bool:
    return bool(old.has_cloth or old.cloth_vertex_count or old.fat_draw is not None)


def _cloth_parts_unchanged(label: str, parts, origins) -> None:
    """``38`` V23-5: a V21-1 sub-object (cloth record or fat vertices) is kept unchanged or removed
    whole: any change to one of its draws or skinning records (including emptying it in place, or
    reordering) is refused with the part's label. ``39`` V24-1: the fat part counts as a part: any
    change to its positions, normals, colours, UVs or indices, or dropping it, is refused too."""
    if origins != list(range(len(origins))):
        raise FormatError(f"{label}: the parts of a cloth sub-object (cloth record or fat vertices) are not "
                          "reordered; it is only removed whole or kept unchanged (V21-1, V23-5)")
    for part_label, draw, before in parts:
        if (draw is None) != (before is None) or draw is not None and not _same_draw(draw, before):
            raise FormatError(f"{part_label}: a part of a cloth sub-object (cloth record or fat vertices) is not "
                              "changed; the sub-object is only removed whole or kept unchanged (V21-1, V23-5)")


def _fat_list_matches(label: str, sub: SubObject, old: SubObject) -> None:
    """``40`` V25-1: the fat draw of a cloth or fat sub-object is written from ``fat_draw``; a fat
    draw in the list of draws must be that draw (or hold the same values), and a fat draw the stored
    list held may not be dropped from the list alone. A mismatch is refused with the V24-1 label
    instead of being ignored."""
    listed = [d for d in sub.draws if d.kind == "fat"]
    stored = [d for d in old.draws if d.kind == "fat"]
    fat = sub.fat_draw
    bad = len(listed) > 1 or len(listed) < len(stored) or any(
        fat is None or (d is not fat and not _same_draw(d, fat)) for d in listed)
    if bad:
        raise FormatError(f"{label}, fat vertices: the fat draw in the list of draws differs from the fat-draw "
                          "field; change neither, the sub-object is only removed whole or kept unchanged "
                          "(V21-1, V23-5, V25-1)")


def _switch_sub(ctx: _Context, record: L.Record, sub: SubObject, old: SubObject, pools: _Pools,
                label: str | None = None, from_zero: bool = False, new: bool = False):
    """One switch / x360 sub-object (W2-1 to W2-8, W13-5 to W13-7, V21-2; W17-1, W17-3, W17-7,
    W17-10, W17-14). Returns (moved, triangles changed, keep): keep is False when the sub-object
    is left without geometry and goes (W17-14)."""
    order, raw, rules = ctx.order, record.raw, ctx.rules
    label = label or sub.label
    packed = [d for d in sub.draws if d.kind != "fat"]
    rigid = [d for d in packed if not d.skinned]
    skins = [d for d in packed if d.skinned]
    old_rigid = [d for d in old.draws if d.kind == "packed" and not d.skinned]
    old_skins = [d for d in old.draws if d.kind == "packed" and d.skinned]
    if len(rigid) > 1:
        raise FormatError(f"{label}: a second rigid list in a switch/x360 sub-object (W17-10)")
    rigid_pairs, rigid_removed = _match_parts(rigid, len(old_rigid), label, "rigid list", from_zero)
    skin_pairs, skin_removed = _match_parts(skins, len(old_skins), label, "skinning record", from_zero)
    added_rigid = any(k is None for _, k in rigid_pairs)
    added_skins = any(k is None for _, k in skin_pairs)
    for draw, k in rigid_pairs + skin_pairs:
        if k is None:
            _check_draw(draw, label)
    if rigid and added_rigid and ctx.has_bones:
        raise FormatError(f"{label}: rigid geometry in a switch/x360 file with bones (W17-10)")
    if skins and (added_skins or from_zero) and not ctx.has_bones:
        raise FormatError(f"{label}: skinned geometry in a file without bones (W17-10)")
    if rigid and len(rigid[0].positions) > 0x10000:
        raise FormatError(f"{label}: a rigid list of {len(rigid[0].positions)} vertices, above 65,536 (W17-10)")
    regroup = from_zero or added_skins or bool(skin_removed) or \
        [k for _, k in skin_pairs] != list(range(len(skin_pairs)))
    if _cloth_part(old) and not from_zero and (added_rigid or rigid_removed or added_skins or skin_removed):
        raise FormatError(f"{label}: a cloth sub-object (cloth record or fat vertices) is only removed whole "
                          "or kept unchanged (V21-1, W17-10)")
    if _cloth_part(old) and not from_zero:  # ``38`` V23-5: no change to its parts either
        _fat_list_matches(label, sub, old)  # ``40`` V25-1
        _cloth_parts_unchanged(label, [(f"{label}, rigid list", d, old_rigid[k]) for d, k in rigid_pairs]
                               + [(f"{label}, skinning record {k}", d, old_skins[k]) for d, k in skin_pairs]
                               + [(f"{label}, fat vertices", sub.fat_draw, old.fat_draw)],  # V24-1
                               [k for _, k in skin_pairs])
    removed_explicitly = bool(rigid_removed or skin_removed)
    if not new:  # V23-5: parts dropped by W13-7 / W17-14 are reported by the writer too
        for draw, k in skin_pairs:
            if k is not None and _empty(draw) and not _empty(old_skins[k]) and "W13-7" in rules:
                ctx.report.add("removed", "skinning record", f"{old.label}, skinning record {k}")
        if rigid and rigid_pairs[0][1] is not None and not rigid[0].positions and "W17-14" in rules:
            ctx.report.add("removed", "rigid list", f"{old.label}, rigid list")
    if rigid and not rigid[0].positions and (from_zero or "W17-14" in rules or added_rigid):
        if not added_rigid:  # W17-14: an emptied rigid list goes with its sub-object
            removed_explicitly = True
        rigid = []
    if not new:  # a new sub-object is reported whole
        for k in rigid_removed:
            ctx.report.add("removed", "rigid list", f"{old.label}, rigid list")
        for k in skin_removed:
            ctx.report.add("removed", "skinning record", f"{old.label}, skinning record {k}")
    fixed = _positions(rigid + skins)
    old_fixed = _positions(old_rigid + old_skins)
    moved = fixed != old_fixed
    bias, scale = _fixed_point(sub, fixed, moved, ctx.has_bones, rules, new=from_zero)
    if moved or (bias, scale) != (tuple(sub.bias), tuple(sub.scale)) or from_zero:
        _pack(raw, order, "3f", SWITCH_SUB["bias"], *bias, field="sub-object bias")
        _pack(raw, order, "3f", SWITCH_SUB["scale"], *scale, field="sub-object step")
    triangles = moved
    if rigid:
        draw = rigid[0]
        _set_bytes(record, SWITCH_SUB["packed"], encode_packed_vertices(draw, bias, scale, order))
        _pack(raw, order, "i", SWITCH_SUB["packed_count"], len(draw.positions), field="nPackVertCount (+0xA0)")
        before = old_rigid[0] if old_rigid and not added_rigid else None
        triangles |= before is None or draw.indices != before.indices
    elif old_rigid:  # removed or emptied (W17-14)
        _set_bytes(record, SWITCH_SUB["packed"], b"")
        _pack(raw, order, "i", SWITCH_SUB["packed_count"], 0, field="nPackVertCount (+0xA0)")
        triangles = True
    if regroup:
        skin_count = _regroup_skins(ctx, record, label, skins, bias, scale)
        triangles |= bool(skins or old_skins)
        skins_changed, entries, edits = True, None, None
    else:
        skins_changed, skin_triangles, entries, edits = _switch_skins(ctx, record, sub, skins, old_skins, bias, scale)
        triangles |= skin_triangles
        value = record.pairs.get(SWITCH_SUB["skins"])
        skin_count = len(value.block.elements) if value is not None and value.block else 0
    fat_moved = sub.fat_draw is not None and old.fat_draw is not None and \
        sub.fat_draw.positions != old.fat_draw.positions
    if sub.fat_draw is not None and old.fat_draw is not None:
        triangles |= fat_moved or sub.fat_draw.indices != old.fat_draw.indices
    # W17-3: the rigid list's indices, else the fat ones, else none (a regrouped skinned part has 0)
    own = rigid[0].indices if rigid else ([] if old_rigid or (regroup and sub.fat_draw is None) else None)
    _fat_and_pools(ctx, record, sub, old, own, SWITCH_SUB["index"], pools)
    if regroup:
        pass  # W13-6 list set with the records
    elif "W13-6" in rules and skins and skins_changed:
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
    if moved or from_zero:
        _write_box(raw, order, SWITCH_SUB["box"], _decoded(fixed, bias, scale), force=from_zero)
    elif fat_moved and not fixed:
        _write_box(raw, order, SWITCH_SUB["box"], sub.fat_draw.positions)
    structure = from_zero or added_rigid or bool(old_rigid) != bool(rigid) or regroup or skin_count != len(old_skins)
    fat_count = struct.unpack_from(order + "I", raw, SWITCH_SUB["fat"])[0]
    if structure:  # W17-1, W17-3
        ctx.parts_changed = True
        if not rigid and not skin_count:
            _pack(raw, order, "6f", SWITCH_SUB["bias"], *(0.0,) * 6, field="sub-object bias and step")
        stored = struct.unpack_from(order + "I", raw, 0)[0]
        flags = PP.sub_object_flags(
            ctx.build, stored, fat=bool(fat_count), cloth=bool(PP.pair_size(record, PP.SWITCH_FIELDS["cloth"])),
            rigid=bool(rigid), skin_records=skin_count, keep_bit8=bool(stored >> PP.BIT_FAT & 1))
        _pack(raw, order, "I", 0, flags, field="sub-object flags (W17-1)")
        if not new:
            for k, (draw, origin) in enumerate(skin_pairs):
                if origin is None and not _empty(draw):
                    ctx.report.add("added", "skinning record", f"{label}, skinning record {k}")
            if added_rigid and rigid:
                ctx.report.add("added", "rigid list", f"{label}, rigid list")
    alive = bool(rigid) or skin_count > 0 or fat_count > 0
    keep = alive or not (from_zero or removed_explicitly or "W17-14" in rules)
    return moved or fat_moved or structure, triangles or structure, keep


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


def ps3_draw_fits(fmt: int, vertex_count: int, index_count: int, bones=()) -> bool:
    """W17-5a: a ps3 draw of these counts is within the size budget (21 nv + ni <= 55,998 for 0x81,
    46 nv + ni <= 58,168 for 0x83), stream A and B at most 0x4000 bytes, the index pair pieces and
    counts within their words and the work buffer at most 0xC000 (W2-10 / W13-1 with the ranges of
    ``bones``)."""
    if not PP.within_budget(fmt, vertex_count, index_count):
        return False
    if vertex_count > 0xFFFF or index_count > 0xFFFF:
        return False
    skinned = fmt == FORMAT_SKINNED
    if _align((10 if skinned else 6) * vertex_count, 16) > PS3_PIECE:
        return False
    if skinned and _align(8 * vertex_count, 16) > PS3_PIECE:
        return False
    if _align(2 * index_count, 16) > PS3_PIECE + 0xFFFF:
        return False
    byte1 = ps3_command_space(fmt, index_count)
    if byte1 > 0xFF:
        return False
    if skinned:
        work = ps3_skinned_work_size(vertex_count, index_count, byte1, ps3_bone_ranges(bones))
    else:
        work = _align(12 * _align(vertex_count, 8) + _align(2 * index_count, 16) + 16 * byte1 + 192, 128)
    return work <= PS3_WORK_LIMIT


def cut_ps3_draw(draw: Draw, fmt: int | None = None) -> list[Draw]:
    """W17-5a: a new ps3 draw as several draws within the size budget. A draw that fits is
    returned whole. Else the triangles are walked in order and a new draw starts before one would
    break a limit (``ps3_draw_fits``); each draw holds the vertices its triangles use, renumbered by
    first use. ``38`` V23-2: every limit cuts (the budget value, stream A 0x4000 bytes, the work
    buffer 0xC000); on dense geometry the work buffer is reached first. A changed existing draw is
    never cut but refused over a limit (V23-3, ``_changed_draw_limits``)."""
    fmt = fmt if fmt is not None else (FORMAT_SKINNED if draw.skinned else FORMAT_RIGID)
    skinned = fmt == FORMAT_SKINNED

    def bones_of(ids):
        found: set = set()
        if skinned:
            for v in ids:
                found |= _vertex_bones(draw, v)
        return found

    if ps3_draw_fits(fmt, len(draw.positions), len(draw.indices), bones_of(range(len(draw.positions)))):
        return [draw]
    if len(draw.indices) % 3:
        raise FormatError(f"{len(draw.indices)} indices are not whole triangles; the draw cannot be cut (W17-5a)")
    groups: list[list[int]] = []
    current: list[int] = []
    used: dict[int, None] = {}
    bones: set = set()
    for t in range(0, len(draw.indices), 3):
        triangle = draw.indices[t:t + 3]
        grown = dict(used)
        grown.update((v, None) for v in triangle)
        need = bones | bones_of(triangle)
        if current and not ps3_draw_fits(fmt, len(grown), len(current) + 3, need):
            groups.append(current)
            current, used, bones = [], {}, set()
            grown = dict.fromkeys(triangle)
            need = bones_of(triangle)
        if not ps3_draw_fits(fmt, len(grown), len(current) + 3, need):
            raise FormatError("a triangle does not fit a ps3 draw (W17-5a)")
        current += triangle
        used, bones = grown, need
    if current:
        groups.append(current)
    out = []
    for group in groups:
        ids = list(dict.fromkeys(group))  # first use
        where = {v: n for n, v in enumerate(ids)}
        out.append(_subset(draw, ids, [where[v] for v in group]))
    return out


def _ps3_new_geometry(ctx: _Context, draw: Draw, label: str) -> Draw:
    """W17-5: a new ps3 draw takes the file's format. Rigid geometry in a file of format 0x83 is
    written skinned, one bone per vertex with weight 255 (its joints when it has them, else bone 0,
    Q-S17-6); skinned geometry needs bones and a 0x83 file."""
    _check_draw(draw, label)
    fmt = ctx.ps3_format
    if draw.skinned:
        if not ctx.has_bones:
            raise FormatError(f"{label}: skinned geometry in a file without bones (W17-10)")
        if fmt != FORMAT_SKINNED:
            raise FormatError(f"{label}: a skinned draw in a ps3 file of rigid draws: a mixed ps3 file (W17-5)")
        return dataclasses.replace(draw, kind="ps3", palette=None, local_joints=None, origin=None)
    if fmt == FORMAT_SKINNED:
        n = len(draw.positions)
        joints = [tuple(j) for j in draw.joints] if draw.joints is not None and draw.weights is not None else None
        weights = [tuple(w) for w in draw.weights] if joints is not None else None
        if joints is None or len(joints) != n:
            joints, weights = [(0, 0, 0, 0)] * n, [(255, 0, 0, 0)] * n
        return dataclasses.replace(draw, skinned=True, joints=joints, weights=weights, palette=None,
                                   local_joints=None, kind="ps3", origin=None)
    return dataclasses.replace(draw, kind="ps3", joints=None, weights=None, palette=None, local_joints=None,
                               origin=None)


def _new_ps3_draw(ctx: _Context, draw: Draw, bias, scale, label: str) -> tuple[L.Record, Draw]:
    """W17-5: a ps3 draw record from zero: W2-9 blocks, G1-5 / W2-10 / W13-1 words, W13-4 ranges
    (zero-weight slots 0), G1-6 vertex-format blocks."""
    fmt = FORMAT_SKINNED if draw.skinned else FORMAT_RIGID
    ranges = (0, 0, 0, 0)
    if draw.skinned:
        bones: set = set()
        for i in range(len(draw.positions)):
            bones |= _vertex_bones(draw, i)
        for bone in bones:
            ctx.check_bone(bone, label, "W13-4")
        ranges = ps3_bone_ranges(bones)
        draw = dataclasses.replace(draw, palette=ranges_palette(ranges))
    blocks = encode_ps3_draw(draw, bias, scale, None, label)
    work = None
    if draw.skinned:
        work = ps3_skinned_work_size(len(draw.positions), len(draw.indices), blocks["descriptor"][1], ranges)
        if work > PS3_WORK_LIMIT:
            raise FormatError(f"{label}: skinned draw work buffer {work} is above 0xC000 (W13-1)")
    record = PP.new_draw_record(fmt)
    words = _ps3_draw_words(record, blocks, len(draw.positions), len(draw.indices), label, work)
    for name in ("descriptor", "indices", "a", "b", "c"):
        _set_bytes(record, DRAW[name], blocks[name])
    for name, fmt_words, values in words:
        _pack(record.raw, ">", fmt_words, DRAW[name], *values, field=name)
    _pack(record.raw, ">", "4H", PS3_RANGES, *ranges, field=f"{label}: bone ranges (+0x60, W13-4)")
    return record, draw


def _ps3_sub(ctx: _Context, record: L.Record, sub: SubObject, old: SubObject, pools: _Pools,
             label: str | None = None, from_zero: bool = False, new: bool = False):
    """One ps3 sub-object (W2-8 to W2-10, W13-1, W13-4, W13-6, W13-7, V21-2; W17-1, W17-3, W17-5,
    W17-10, W17-14). Returns (moved, triangles changed, keep)."""
    order, raw, rules = ctx.order, record.raw, ctx.rules
    label = label or sub.label
    fat, old_fat = sub.fat_draw, old.fat_draw
    draws = [d for d in sub.draws if d.kind != "fat"]
    old_draws = [d for d in old.draws if d.kind == "ps3"]
    value = record.pairs.get(PS3_SUB["draws"])
    records = value.block.elements if value is not None and value.block else []
    if len(records) != len(old_draws):
        raise FormatError(f"{label}: {len(records)} draw records for {len(old_draws)} draws")
    pairs, removed = _match_parts(draws, len(old_draws), label, "draw", from_zero)
    added = any(k is None for _, k in pairs)
    if _cloth_part(old) and not from_zero and (added or removed):
        raise FormatError(f"{label}: a cloth sub-object (cloth record or fat vertices) is only removed whole "
                          "or kept unchanged (V21-1, W17-10)")
    if from_zero and fat is not None:
        raise FormatError(f"{label}: a new sub-object with fat vertices is only moved in whole (W17-11, V21-1)")
    if _cloth_part(old) and not from_zero:  # ``38`` V23-5
        _fat_list_matches(label, sub, old)  # ``40`` V25-1
        _cloth_parts_unchanged(label, [(f"{label}, draw {k}", d, old_draws[k]) for d, k in pairs]
                               + [(f"{label}, fat vertices", fat, old_fat)],  # V24-1
                               [k for _, k in pairs])
    for k in removed if not new else ():  # a new sub-object is reported whole
        ctx.report.add("removed", "draw", f"{old.label}, draw {k}")
    plan = []  # (stored record or None, draw, stored draw or None)
    for draw, k in pairs:
        if k is not None:
            plan.append((records[k], draw, old_draws[k]))
        elif not _empty(draw):
            for piece in cut_ps3_draw(_ps3_new_geometry(ctx, draw, label), ctx.ps3_format):
                plan.append((None, piece, None))
    fixed = _positions([d for _, d, _ in plan])
    old_fixed = _positions(old_draws)
    moved = fixed != old_fixed
    bias, scale = _fixed_point(sub, fixed, moved, ctx.has_bones, rules, new=from_zero)
    if moved or (bias, scale) != (tuple(sub.bias), tuple(sub.scale)) or from_zero:
        _pack(raw, order, "3f", PS3_SUB["bias"], *bias, field="sub-object bias")
        _pack(raw, order, "3f", PS3_SUB["scale"], *scale, field="sub-object step")
    triangles = moved or bool(removed)
    kept, kept_draws = [], []
    changed_mesh = moved or bool(removed)
    for draw_record, draw, before in plan:
        dlabel = f"{label}, draw {len(kept)}"
        if before is None:  # W17-5: a new draw
            new_record, draw = _new_ps3_draw(ctx, draw, bias, scale, dlabel)
            kept.append(new_record)
            kept_draws.append(draw)
            triangles = changed_mesh = True
            if not new:
                ctx.report.add("added", "draw", dlabel)
            continue
        same = _same_draw(draw, before)
        triangles |= draw.indices != before.indices
        changed_mesh |= not same
        if _empty(draw) and not (same and _empty(before)):
            _empty_record(ctx, sub, "ps3 draw")
            triangles = True
            if not new:  # V23-5: a draw dropped by W13-7 is reported by the writer too
                stored_k = next(i for i, r in enumerate(records) if r is draw_record)
                ctx.report.add("removed", "draw", f"{old.label}, draw {stored_k}")
            continue
        stored_ranges = struct.unpack_from(">4H", draw_record.raw, PS3_RANGES)
        ranges, slots = stored_ranges, before.local_joints
        if draw.skinned and "W13-4" in rules:
            if ctx.keep_skin and not same and before.local_joints is not None and skin_edit_only(draw, before):
                ranges, draw, slots = _ps3_kept_ranges(ctx, draw, before, stored_ranges, dlabel)  # V21-2
            else:
                ranges, draw, slots = _ps3_ranges(ctx, draw, before, same, dlabel)
        blocks = encode_ps3_draw(draw, bias, scale, slots, dlabel)
        work = None
        if draw.skinned and "W13-1" in rules:
            work = ps3_skinned_work_size(len(draw.positions), len(draw.indices), blocks["descriptor"][1], ranges)
        if not same:  # ``39`` V24-4: budget, stream A, then the work buffer (the order of V23-2)
            _changed_draw_limits(draw_record, blocks, len(draw.positions), len(draw.indices), work, dlabel)
        if work is not None:
            if work > PS3_WORK_LIMIT:
                raise FormatError(f"{dlabel}: skinned draw work buffer {work} is above 0xC000 (W13-1); "
                                  f"{CHANGED_DRAW_HINT}")
        kept.append(draw_record)
        kept_draws.append(draw)
        r = draw_record.raw
        if not all(_block_bytes(draw_record.pairs.get(DRAW[name])) == blocks[name]
                   for name in ("descriptor", "indices", "a", "b", "c")):
            words = _ps3_draw_words(draw_record, blocks, len(draw.positions), len(draw.indices), dlabel, work)
            for name in ("descriptor", "indices", "a", "b", "c"):
                _set_bytes(draw_record, DRAW[name], blocks[name])
            for name, fmt, values in words:
                _pack(r, ">", fmt, DRAW[name], *values, field=name)
        elif work is not None:
            _pack(r, ">", "I", DRAW["work"], work, field=f"{dlabel}: work-buffer size (+0x98, W13-1)")
        if tuple(ranges) != tuple(stored_ranges):
            _pack(r, ">", "4H", PS3_RANGES, *ranges, field=f"{dlabel}: bone ranges (+0x60, W13-4)")
    structure = from_zero or len(kept) != len(records) or any(a is not b for a, b in zip(kept, records))
    if structure:  # W13-7 drops, W17-14 removals, W17-5 additions
        record.pairs[PS3_SUB["draws"]] = L.PairValue(L.Block(elements=kept) if kept else None)
        _pack(raw, order, "i", PS3_SUB["draw_count"], len(kept), field="draw count (+0xFC)")
    fat_moved = fat is not None and old_fat is not None and fat.positions != old_fat.positions
    if fat is not None and old_fat is not None:
        triangles |= fat_moved or fat.indices != old_fat.indices
        changed_mesh |= fat_moved or fat.indices != old_fat.indices
    _fat_and_pools(ctx, record, sub, old, None, PS3_SUB["index"], pools)
    if ("W13-6" in rules or from_zero) and changed_mesh:
        # W13-6: present exactly in sub-objects owning fat vertices in a file with bones; 6 x the
        # skinned draws' vertices, else 6 x the fat vertices; stored entries kept (Q400)
        fat_n = len(fat.positions) if fat is not None else 0
        skinned_n = sum(len(d.positions) for d in kept_draws if d.skinned)
        want = INFLUENCE_RECORD * (skinned_n or fat_n) if fat_n and ctx.has_bones else 0
        stored = _block_bytes(record.pairs.get(PS3_SUB["influence"]))
        _set_bytes(record, PS3_SUB["influence"], stored[:want] + bytes(max(0, want - len(stored))))
    else:
        old_n = sum(len(d.positions) for d in old_draws) or (len(old_fat.positions) if old_fat else 0)
        new_n = sum(len(d.positions) for d in kept_draws) or (len(fat.positions) if fat else 0)
        _resize_influences(record, PS3_SUB["influence"], old_n, new_n)
    if moved or from_zero:
        _write_box(raw, order, PS3_SUB["box"], _decoded(fixed, bias, scale), force=from_zero)
    elif fat_moved and not fixed:
        _write_box(raw, order, PS3_SUB["box"], fat.positions)
    fat_count = struct.unpack_from(order + "i", raw, PS3_SUB["fat"])[0]
    if structure:  # W17-1, W17-3
        ctx.parts_changed = True
        if not kept_draws:
            _pack(raw, order, "6f", PS3_SUB["bias"], *(0.0,) * 6, field="sub-object bias and step")
        stored = struct.unpack_from(order + "I", raw, 0)[0]
        flags = PP.sub_object_flags("ps3", stored, fat=bool(fat_count), draws=len(kept),
                                    skinned_draws=sum(d.skinned for d in kept_draws))
        _pack(raw, order, "I", 0, flags, field="sub-object flags (W17-1)")
    alive = bool(kept) or fat_count > 0
    keep = alive or not (from_zero or bool(removed) or "W17-14" in rules)
    return moved or fat_moved or structure, triangles or structure, keep


CHANGED_DRAW_HINT = ("a changed draw is not cut (V23-3); import the geometry as a new mesh, which is cut "
                     "into draws (W17-5a)")


def _changed_draw_limits(record: L.Record, blocks: dict, nv: int, nidx: int, work: int | None, label: str) -> None:
    """``38`` V23-3: an existing ps3 draw whose geometry changed stays one draw; when it passes any
    limit of V23-2 (the W17-5a budget, stream A 0x4000 bytes, the work buffer 0xC000) it is
    refused with its label, on either format."""
    fmt = blocks["descriptor"][0]
    budget = PP.draw_budget(fmt, nv, nidx)
    if budget > PP.BUDGET[fmt][1]:
        raise FormatError(f"{label}: budget value {budget} is above {PP.BUDGET[fmt][1]} (W17-5a); {CHANGED_DRAW_HINT}")
    if len(blocks["a"]) > PS3_PIECE:  # V24-3: a safeguard; with the stored strides the budget is passed first
        raise FormatError(f"{label}: stream A of {len(blocks['a'])} bytes is above 0x4000 (G1-5); {CHANGED_DRAW_HINT}")
    if work is None:
        if fmt == FORMAT_RIGID:
            work = ps3_work_size(fmt, nv, nidx)
        else:
            stored = struct.unpack_from(">I", record.raw, DRAW["work"])[0]
            work = _align(28 * _align(nv, 8) + _align(2 * nidx, 16) + 16 * ps3_command_space(fmt, nidx) + 336, 128)
            work = max(work, stored)
    if work > PS3_WORK_LIMIT:
        raise FormatError(f"{label}: work buffer {work} is above 0xC000 (W2-10, W13-1); {CHANGED_DRAW_HINT}")


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


def _match_subs(geo: int, subs, old_subs):
    """W17-10: the mesh's sub-objects of one geo set against the template's, by ``origin``.
    Existing sub-objects keep their order; new ones (origin None) come after them. Returns
    ([(sub-object, template index or None)], removed template indices)."""
    plan, seen = [], set()
    last, new_seen = -1, False
    for sub in subs:
        origin = getattr(sub, "origin", None)
        if origin is None:
            plan.append((sub, None))
            new_seen = True
            continue
        try:
            g, i = origin
        except (TypeError, ValueError):
            raise FormatError(f"{sub.label}: origin {origin!r} is not (geo set, index) (W17-10)") from None
        if g != geo:
            raise FormatError(f"{sub.label}: origin {origin!r} is in geo set {g}, not {geo}; a part moved to "
                              "the other geo set is a new sub-object (origin None) (W17-10)")
        if isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < len(old_subs):
            raise FormatError(f"{sub.label}: origin {origin!r} is not one of the {len(old_subs)} template "
                              f"sub-objects of geo set {geo} (W17-10)")
        if i in seen:
            raise FormatError(f"{sub.label}: two sub-objects come from template sub-object {origin}; mark "
                              "the copy new (origin None) (W17-10)")
        if i < last or new_seen:
            raise FormatError(f"{sub.label}: existing sub-objects keep their order and new ones go at the "
                              f"end of their geo set (W17-10)")
        seen.add(i)
        last = i
        plan.append((sub, i))
    return plan, [i for i in range(len(old_subs)) if i not in seen]


def _stored_word(raw: bytes, order: str, at: int) -> int:
    return struct.unpack_from(order + "I", raw, at)[0]


def _effective_hash(ctx: _Context, sub: SubObject, k, geo: int, template_raws, label: str) -> tuple[int, bool]:
    """W17-2: the sub-object hash a written sub-object will have, and whether it is new or renamed
    (and so must not repeat another one of its geo set)."""
    if k is not None:
        stored = _stored_word(template_raws[(geo, k)], ctx.order, PP.sub_fields(ctx.build)["hash"])
        if ctx.build == "ps3" and sub.name is not None and sub.name != ctx.old_lists[geo][k].name:
            return PP.name_hash(sub.name), True  # a renamed ps3 sub-object (switch stores no name)
        return stored, False
    if sub.name is not None:
        return PP.name_hash(sub.name), True
    if ctx.build == "ps3" and sub.stored_from is None:
        raise FormatError(f"{label}: a new ps3 sub-object needs a name (W17-2)")
    if sub.sub_hash is None:
        raise FormatError(f"{label}: a new sub-object needs a name or a sub-object hash (W17-2)")
    return sub.sub_hash & 0xFFFFFFFF, True


def _check_names(ctx: _Context, plans, template_raws) -> None:
    """W17-10: a new or renamed sub-object may not repeat the name (hash, W17-2) of another
    sub-object of its geo set."""
    for geo, (plan, _) in enumerate(plans):
        keys = []
        for n, (sub, k) in enumerate(plan):
            label = f"{GEOSET_NAMES[geo]}, sub-object {n}"
            keys.append((*_effective_hash(ctx, sub, k, geo, template_raws, label), label, sub.name))
        for n, (key, check, label, name) in enumerate(keys):
            if check and any(other[0] == key for m, other in enumerate(keys) if m != n):
                shown = f"name {name!r}" if name is not None else "sub-object hash"
                raise FormatError(f"{label}: duplicate {shown} (hash 0x{key:08X}) in the {GEOSET_NAMES[geo]} "
                                  "(W17-10)")


def _template_part(ctx: _Context, sub: SubObject, label: str):
    """W17-10: the template part a new sub-object copies values from: the one the caller names
    (``template_part``), else the first sub-object with the same material, else the file's first."""
    old = ctx.old_lists
    if sub.template_part is not None:
        try:
            g, i = sub.template_part
            return (g, i), old[g][i]
        except (TypeError, ValueError, IndexError):
            raise FormatError(f"{label}: template part {sub.template_part!r} is not a template sub-object "
                              "(W17-10)") from None
    everything = [(s.geo_set, s.index, s) for s in old[0] + old[1]]
    if sub.material_hash is not None:
        for g, i, s in everything:
            if s.material_hash == sub.material_hash:
                return (g, i), s
    if not everything:
        raise FormatError(f"{label}: the template has no sub-object to copy values from (W17-10)")
    g, i, s = everything[0]
    return (g, i), s


def _from_zero_record(ctx: _Context, sub: SubObject, template_raws, label: str) -> L.Record:
    """W17-3: the record of a new sub-object, copied values from its template part."""
    if sub.has_cloth or sub.cloth is not None or sub.fat_draw is not None:
        raise FormatError(f"{label}: a new sub-object with cloth or fat vertices is only moved in whole "
                          "(W17-11, V21-1)")
    where, part = _template_part(ctx, sub, label)
    material = sub.material_hash if sub.material_hash is not None else part.material_hash
    key, _ = _effective_hash(ctx, sub, None, sub.geo_set, template_raws, label)
    return PP.new_sub_record(ctx.build, template_raws[where], material, key,
                             sub.name if ctx.build == "ps3" else None)


def _donor_record(ctx: _Context, sub: SubObject, label: str):
    """W17-11: a copy of the stored record a moved sub-object comes from, its parse and pools."""
    source = sub.stored_from
    if not isinstance(source, PartSource):
        raise FormatError(f"{label}: stored_from {source!r} is not a PartSource (W17-11)")
    tree, parsed, runs, index_pool, vertex_pool, _ = ctx.donor(source.app)
    value = tree.top.pairs.get(GEO_SUB_OBJECTS[source.geo_set]) if source.geo_set in (0, 1) else None
    records = value.block.elements if value is not None and value.block else []
    if not 0 <= source.index < len(records):
        raise FormatError(f"{label}: the source has no sub-object {source.index} in geo set {source.geo_set}")
    before = (parsed.sub_objects, parsed.extra_sub_objects)[source.geo_set][source.index]
    vertices, indices = runs[(source.geo_set, source.index)]
    return PP.copy_record(records[source.index]), before, _Pools(index_pool, vertex_pool, indices, vertices)


def _identity(ctx: _Context, record: L.Record, sub: SubObject, before: SubObject, from_zero: bool) -> bool:
    """The name, sub-object hash and material hash of a kept or moved sub-object (W17-2, W17-8).
    Returns True when a word changed."""
    if from_zero:
        return False
    fields, order, raw = PP.sub_fields(ctx.build), ctx.order, record.raw
    changed = False
    rename = sub.name is not None and sub.name != before.name and (ctx.build == "ps3" or sub.origin is None)
    if rename:  # ps3: a renamed sub-object; switch / x360: a moved one (no name is stored, W17-2)
        if ctx.build == "ps3":
            raw[fields["name"]:fields["name"] + PP.PS3_NAME_SIZE] = PP.ps3_name_field(sub.name)
        struct.pack_into(order + "I", raw, fields["hash"], PP.name_hash(sub.name))
        changed = True
    elif ctx.build != "ps3" and sub.name is None and sub.origin is None and sub.sub_hash is not None \
            and sub.sub_hash != _stored_word(raw, order, fields["hash"]):
        struct.pack_into(order + "I", raw, fields["hash"], sub.sub_hash & 0xFFFFFFFF)
        changed = True
    if sub.material_hash is not None and sub.material_hash != _stored_word(raw, order, fields["material"]):
        _pack(raw, order, "I", fields["material"], sub.material_hash, field="material hash")
        changed = True
    return changed


def _materials(ctx: _Context, template_entries: list[L.Record]) -> None:
    """W17-8: a hash newly used by a sub-object gets an entry, copied from its source (the same
    file or a donor) and appended; the entry of a hash no sub-object uses any more goes. A hash
    already used in the template, and stored unused or repeated entries, are left as stored."""
    tree, order = ctx.tree, ctx.order
    at = PP.sub_fields(ctx.build)["material"]
    before = {s.material_hash for s in ctx.old_lists[0] + ctx.old_lists[1]}
    entries = PP.material_entries(tree)
    have = {PP.entry_hash(e, order) for e in entries}
    after: set = set()
    changed = False
    looks = PP.look_count(tree)
    for record, sub, label, _ in ctx.written:
        value = _stored_word(record.raw, order, at)
        after.add(value)
        if value in have or value in before:
            continue
        source = PP.material_source(sub.material_source)
        if source is None:
            raise FormatError(f"{label}: material hash 0x{value:08X} has no table entry and no source for a "
                              "new entry (W17-8)")
        if source.app is None:
            candidates = template_entries
        else:
            candidates = PP.material_entries(ctx.donor(source.app)[0])
        found = next((e for e in candidates if PP.entry_hash(e, order) == source.hash), None)
        if found is None:
            where = "the file" if source.app is None else "the donor file"
            raise FormatError(f"{label}: material source 0x{source.hash:08X} is not in {where}'s table (W17-8)")
        entries.append(PP.copied_material_entry(found, value, order, looks))
        have.add(value)
        ctx.report.add("added", "material", f"material 0x{value:08X}")
        changed = True
    gone = (before - after) & have
    if gone:
        entries = [e for e in entries if PP.entry_hash(e, order) not in gone]
        for value in sorted(gone):
            ctx.report.add("removed", "material", f"material 0x{value:08X}")
        changed = True
    if changed:
        PP.set_material_entries(tree, entries)


def app_mesh_tree(app: AppearanceFile, mesh: AppearanceFile | None = None, derive=False,
                  keep_skin_bytes: bool = False, report: PP.PartsReport | None = None,
                  _written: list | None = None) -> L.SnoTree:
    """W2-1 to W2-15: the structure tree of ``app``'s source with every mesh block of ``mesh`` (a
    decoded ``.app``, by default ``app`` itself, changed by the caller) encoded again.
    ``derive`` (True, or rule names of ``RULES``) applies the stage-13 rules: W13-1, W13-2 and
    W13-4 derive the values the writer otherwise keeps; W13-3, W13-5 and W13-6 apply to changed
    meshes; W13-7 drops empty skinning records and skinned draws (else they are refused); W17-14
    removes a sub-object left without geometry by such a drop.
    ``keep_skin_bytes`` (V21-2, used by the mesh import): a skinning record or ps3 draw whose
    vertex count, positions and indices are unchanged keeps its stored palette or bone ranges
    while every weighted bone is in them, the stored slot bytes of its unchanged vertices, and
    (switch, x360) the stored compact influence list with only the changed entries replaced.

    W17-10: the mesh's sub-object lists may differ from the template's. ``SubObject.origin`` says
    which template sub-object each comes from; None marks a new one (built from zero by W17-3
    from its template part, or copied whole from ``stored_from``, W17-11). Template sub-objects
    missing from the lists are removed with their pool runs. Within a sub-object, ``Draw.origin``
    matches ps3 draws, skinning records and the rigid list the same way. ``report`` (a
    ``app_parts.PartsReport``) receives the added and removed parts."""
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
    if report is not None:
        ctx.report = report
    if _written is not None:
        ctx.written = _written
    ctx.old_lists = (old.sub_objects, old.extra_sub_objects)
    sub_fn = _ps3_sub if app.build == "ps3" else _switch_sub
    box_at = (PS3_SUB if app.build == "ps3" else SWITCH_SUB)["box"]
    geo_records, template_raws = [], {}
    for geo in (0, 1):
        value = tree.top.pairs.get(GEO_SUB_OBJECTS[geo])
        records = list(value.block.elements) if value is not None and value.block else []
        if len(records) != len(ctx.old_lists[geo]):
            raise FormatError(f"geo set {geo}: {len(records)} records for {len(ctx.old_lists[geo])} sub-objects")
        geo_records.append(records)
        template_raws.update(((geo, i), bytes(r.raw)) for i, r in enumerate(records))
    template_entries = PP.material_entries(tree)
    plans = [_match_subs(geo, subs, ctx.old_lists[geo])
             for geo, subs in enumerate((mesh.sub_objects, mesh.extra_sub_objects))]
    _check_names(ctx, plans, template_raws)
    geo0_records: list[L.Record] = []
    for geo, (plan, removed) in enumerate(plans):
        records = geo_records[geo]
        for k in removed:
            ctx.report.add("removed", "sub-object", ctx.old_lists[geo][k].label)
        out: list[L.Record] = []
        for sub, k in plan:
            label = f"{GEOSET_NAMES[geo]}, sub-object {len(out)}"
            from_zero = False
            if k is not None:
                record, before, pools = records[k], ctx.old_lists[geo][k], ctx.template_pools(geo, k)
            elif sub.stored_from is not None:
                record, before, pools = _donor_record(ctx, sub, label)
            else:
                record = _from_zero_record(ctx, sub, template_raws, label)
                before = SubObject(-1, None, None, (0.0,) * 3, (0.0,) * 3, None, None, [], geo_set=geo)
                pools, from_zero = _Pools(), True
            mark = ctx.mark()
            _identity(ctx, record, sub, before, from_zero)
            moved, triangles, keep = sub_fn(ctx, record, sub, before, pools, label=label, from_zero=from_zero,
                                            new=k is None)
            if not keep:  # W17-14: no geometry left (or a new part without geometry, not written)
                ctx.rollback(mark)
                if k is not None:
                    ctx.report.add("removed", "sub-object", before.label)
                else:
                    ctx.report.add("skipped", "sub-object", label)
                continue
            if k is None:
                ctx.report.add("added", "sub-object", label)
            out.append(record)
            draws = [d for d in sub.draws if d.kind != "fat"]
            origins = [getattr(d, "origin", None) for d in draws]
            stored_n = len([d for d in before.draws if d.kind != "fat"])
            in_place = origins == list(range(stored_n)) or all(o is None for o in origins)
            exact = k is not None and len(draws) == stored_n and in_place
            ctx.written.append((record, sub, label, exact))
            if geo == 0:
                ctx.geo0_moved |= moved
                ctx.geo0_triangles |= triangles
        if len(out) != len(records) or any(a is not b for a, b in zip(out, records)):
            tree.top.pairs[GEO_SUB_OBJECTS[geo]] = L.PairValue(L.Block(elements=out) if out else None)
            _pack(tree.top.raw, ctx.order, "i", PP.GEO_COUNTS[geo], len(out), field=f"geo set {geo} count")
            ctx.parts_changed = True
            if geo == 0:
                ctx.geo0_moved = ctx.geo0_triangles = True
        if geo == 0:
            geo0_records = out
    _materials(ctx, template_entries)
    index_at, vertex_at = POOLS[app.build]
    _set_bytes(tree.top, index_at, bytes(ctx.index_pool))
    _set_bytes(tree.top, vertex_at, bytes(ctx.vertex_pool))
    if app.build == "ps3" and ctx.parts_changed:
        PP.share_ps3_blocks(tree)  # W17-6
    _structure(ctx, geo0_records, box_at)
    if ctx.geo0_triangles and "W13-3" in rules:
        _rebuild_octree(tree, stored_octree)
    return tree


def write_app_mesh_report(app: AppearanceFile, mesh: AppearanceFile | None = None, check: bool = True,
                          derive=False, keep_skin_bytes: bool = False) -> tuple[bytes, PP.PartsReport]:
    """``write_app_mesh`` with the report of added, removed and skipped parts (W17-10, W17-14)."""
    rules = rule_set(derive)
    report = PP.PartsReport()
    written: list = []
    out = L.write_tree(app_mesh_tree(app, mesh, rules, keep_skin_bytes, report, written))
    if check:
        again = parse_appearance(out, app.build)
        found = again.sub_objects + again.extra_sub_objects
        if len(found) != len(written):
            raise FormatError(f"the written file reads back with {len(found)} sub-objects for {len(written)}")
        for a, (_, b, label, exact) in zip(found, written):
            if rules & {"W13-5", "W13-7"} or not exact:
                same = sum(len(d.indices) for d in a.draws if d.kind != "fat") ==                     sum(len(d.indices) for d in b.draws if d.kind != "fat")
            else:
                same = [len(d.positions) for d in a.draws] == [len(d.positions) for d in b.draws] and \
                    [len(d.indices) for d in a.draws] == [len(d.indices) for d in b.draws]
            if not same:
                raise FormatError(f"{label}: the written mesh does not read back with its counts")
    return out, report


def write_app_mesh(app: AppearanceFile, mesh: AppearanceFile | None = None, check: bool = True,
                   derive=False, keep_skin_bytes: bool = False) -> bytes:
    """Encode ``mesh`` (default: ``app`` as changed by the caller) into ``app``'s ``.app`` and lay
    it out (W2-1 to W2-15, W11; ``derive``: the W13 rules and W17-14, ``keep_skin_bytes``: V21-2,
    see ``app_mesh_tree``). An unchanged mesh gives the source bytes (with ``derive`` too, for a
    file that follows the rules). W17-10: the mesh may add and remove sub-objects, ps3 draws,
    skinning records and rigid lists (``write_app_mesh_report`` also returns what was added and
    removed). With ``check`` the output is read again and must give the same counts (the same
    index count per sub-object when parts may be split, dropped or added)."""
    return write_app_mesh_report(app, mesh, check, derive, keep_skin_bytes)[0]


# --- W17-11: moving a sub-object between files ----------------------------------------------------

def _bone_hashes(app: AppearanceFile) -> list[int]:
    return [bone.name_hash for bone in app.bones]


def transplant(target: AppearanceFile, source: AppearanceFile, geo_set: int, index: int, *,
               name: str | None = None) -> AppearanceFile:
    """W17-11: ``target``'s mesh with sub-object (``geo_set``, ``index``) of ``source`` added at the
    end of the same geo set. Every stored value of the part is kept (its record is copied: geometry,
    copied fields, compact list, cloth, shapes). Needs the same build and the same skeleton (equal
    bone name hashes in order, or neither has bones); on ps3 also the same draw format. The source
    material comes along when the target lacks its hash (W17-8). The name: ``name`` when given, else
    the source name (ps3) or the stored hash (switch, x360); a taken one gets ``_2``, ``_3``, ...
    (W17-12; on switch / x360 the base is the exporter's object name). Write the result with
    ``write_app_mesh(target, mesh, derive=True)``."""
    if source.build != target.build:
        raise FormatError(f"a {source.build} sub-object for a {target.build} file (W17-11)")
    if _bone_hashes(source) != _bone_hashes(target):
        raise FormatError(f"source sub-object {geo_set}:{index}: the files have different skeletons (bone name "
                          "hashes); a palette remap is out of scope (W17-11, V23-9)")
    if geo_set not in (0, 1):
        raise FormatError(f"geo set {geo_set} (W17-11)")
    subs = (source.sub_objects, source.extra_sub_objects)[geo_set]
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(subs):
        raise FormatError(f"the source has no sub-object {index} in geo set {geo_set} (W17-11)")
    part = subs[index]
    if target.build == "ps3":
        formats = {FORMAT_SKINNED if d.skinned else FORMAT_RIGID for d in part.draws if d.kind == "ps3"}
        if formats and formats != {_file_format(target)}:
            raise FormatError(f"source sub-object {geo_set}:{index}: the ps3 draw formats differ (W17-11)")
    moved = copy.deepcopy(part)
    moved.origin, moved.template_part, moved.material_source = None, None, None
    moved.stored_from = PartSource(source, geo_set, index)
    target_subs = list((target.sub_objects, target.extra_sub_objects)[geo_set])
    if target.build == "ps3":
        taken = {PP.name_hash(s.name) for s in target_subs if s.name is not None}
        base = name if name is not None else (part.name or f"subobject_{index}")
        moved.name = base if name is not None else PP.free_name(base, taken)
    else:
        taken = {s.sub_hash for s in target_subs}
        if name is not None:
            moved.name = name
        elif part.sub_hash in taken:
            base = part.name or f"subobject_{index}"
            moved.name = PP.free_name(base, taken)
    if part.material_hash is not None and target.material_data.find(part.material_hash) is None \
            and source.material_data.find(part.material_hash) is not None:
        moved.material_source = PP.MaterialSource(part.material_hash, source)
    moved.index, moved.geo_set = len(target_subs), geo_set
    lists = [list(target.sub_objects), list(target.extra_sub_objects)]
    lists[geo_set].append(moved)
    return dataclasses.replace(target, sub_objects=lists[0], extra_sub_objects=lists[1])


# --- Check: every record from zero ----------------------------------------------------------------

def _rebuilt_sub(build: str, order: str, record: L.Record, sub: SubObject, has_bones: bool) -> bytearray:
    """W17-1, W17-3: a sub-object record from zero. Copied from the stored record: the copied flag
    bits and fields (W17-3, W17-4), the material hash, the switch / x360 sub-object hash (no name is
    stored) and the values kept for an unchanged mesh (box, bias: W2-2, W2-4). Derived: the content
    flag bits, counts, ps3 name hash and the step (W13-2). Every other word is 0."""
    stored = bytes(record.raw)
    fields = PP.sub_fields(build)
    raw = bytearray(len(stored))
    for name in PP.copied_names(build) + ("material", "box"):
        at = fields[name]
        size = 24 if name == "box" else 4
        raw[at:at + size] = stored[at:at + size]
    rigid = [d for d in sub.draws if d.kind == "packed" and not d.skinned]
    skins = [d for d in sub.draws if d.kind == "packed" and d.skinned]
    ps3_draws = [d for d in sub.draws if d.kind == "ps3"]
    fat = sub.fat_draw
    fat_count = len(fat.positions) if fat is not None else _stored_word(stored, order, fields["fat"])
    fat_indices = len(fat.indices) if fat is not None else _stored_word(stored, order, fields["index"])
    fixed = _positions(rigid + skins + ps3_draws)
    if build == "ps3":
        at = fields["name"]
        raw[at:at + PP.PS3_NAME_SIZE] = PP.ps3_name_field(sub.name) if sub.name else bytes(PP.PS3_NAME_SIZE)
        struct.pack_into(order + "I", raw, fields["hash"], PP.name_hash(sub.name or ""))
        struct.pack_into(order + "i", raw, fields["index"], fat_indices if fat_count else 0)
        struct.pack_into(order + "i", raw, fields["draw_count"], len(ps3_draws))
    else:
        raw[fields["hash"]:fields["hash"] + 4] = stored[fields["hash"]:fields["hash"] + 4]
        index_count = len(rigid[0].indices) if rigid else (fat_indices if fat_count else 0)
        struct.pack_into(order + "i", raw, fields["index"], index_count)
        struct.pack_into(order + "i", raw, fields["packed_count"], len(rigid[0].positions) if rigid else 0)
    struct.pack_into(order + "I", raw, fields["fat"], fat_count)
    struct.pack_into(order + "i", raw, fields["shape_count"], PP.shape_count(build, record))
    if fixed:
        raw[fields["bias"]:fields["bias"] + 12] = stored[fields["bias"]:fields["bias"] + 12]
        _, scale = _fixed_point(sub, fixed, False, has_bones, {"W13-2"})
        struct.pack_into(order + "3f", raw, fields["scale"], *scale)
    stored_flags = _stored_word(stored, order, 0)
    skin_value = record.pairs.get(SWITCH_SUB["skins"]) if build != "ps3" else None
    skin_records = len(skin_value.block.elements) if skin_value is not None and skin_value.block else 0
    flags = PP.sub_object_flags(
        build, stored_flags, fat=bool(fat_count), cloth=bool(PP.pair_size(record, fields["cloth"])),
        draws=len(ps3_draws), skinned_draws=sum(d.skinned for d in ps3_draws), rigid=bool(rigid),
        skin_records=skin_records, keep_bit8=bool(stored_flags >> PP.BIT_FAT & 1))
    struct.pack_into(order + "I", raw, 0, flags)
    return raw


def _rebuilt_draw(record: L.Record, draw: Draw, bone_count: int, label: str) -> None:
    """W17-5: a ps3 draw record from zero around its stored index list and streams: descriptor
    (W2-9), words (G1-5, W2-10, W13-1), bone ranges (W13-4, with the bones of stored zero-weight
    slots), vertex-format blocks (G1-6)."""
    fmt = FORMAT_SKINNED if draw.skinned else FORMAT_RIGID
    nv, ni = len(draw.positions), len(draw.indices)
    ranges = (0, 0, 0, 0)
    if draw.skinned:
        bones: set = set()
        for i in range(nv):
            bones |= _vertex_bones(draw, i)
        palette = draw.palette or []
        for weights, slots in zip(draw.weights, draw.local_joints or []):
            bones.update(palette[s] for w, s in zip(weights, slots) if not w and s < len(palette))
        ranges = ps3_bone_ranges(bones)
    byte1 = ps3_command_space(fmt, ni)
    blocks = {"descriptor": PP.ps3_descriptor(fmt, byte1, nv, ni)}
    for name in ("indices", "a", "b", "c"):
        blocks[name] = _block_bytes(record.pairs.get(DRAW[name]))
    work = ps3_skinned_work_size(nv, ni, byte1, ranges) if draw.skinned else None
    zero = L.Record(record.name, bytearray(len(record.raw)), record.pairs)
    words = _ps3_draw_words(zero, blocks, nv, ni, label, work)
    for name, fmt_words, values in words:
        _pack(zero.raw, ">", fmt_words, DRAW[name], *values, field=name)
    struct.pack_into(">4H", zero.raw, PS3_RANGES, *ranges)
    record.raw[:] = zero.raw
    record.pairs[DRAW["descriptor"]] = L.PairValue(L.Block(blocks["descriptor"]))
    record.pairs[0x80] = L.PairValue(L.Block(PS3_FORMAT_BLOCKS[fmt][0]))
    record.pairs[0x90] = L.PairValue(L.Block(PS3_FORMAT_BLOCKS[fmt][1]))


def rebuild_from_zero(app: AppearanceFile) -> bytes:
    """The "Check" of ``37``: every sub-object record, ps3 draw record and switch / x360 skinning
    record of ``app``'s source rebuilt from zero bytes by W17-1 to W17-7, copying only the copied
    fields (and the values W2-2 / W2-4 keep for an unchanged mesh: box and bias); ps3 descriptor and
    vertex-format blocks shared by W17-6. A file that follows the rules comes back byte for byte.
    ``38`` V23-1: the box and bias of an existing part are kept as stored (the rebuild that produced
    the rules kept them; a W2-4 box equals the stored one in only a few sub-objects)."""
    source = bytes(app.source)
    tree = L.parse_tree(source, "app", app.build)
    parsed = parse_appearance(source, app.build)
    order = build_order(app.build)
    for geo, subs in enumerate((parsed.sub_objects, parsed.extra_sub_objects)):
        value = tree.top.pairs.get(GEO_SUB_OBJECTS[geo])
        records = value.block.elements if value is not None and value.block else []
        for record, sub in zip(records, subs):
            if app.build == "ps3":
                draws_value = record.pairs.get(PS3_SUB["draws"])
                draw_records = draws_value.block.elements if draws_value is not None and draws_value.block else []
                for k, (draw_record, draw) in enumerate(zip(draw_records, [d for d in sub.draws if d.kind == "ps3"])):
                    _rebuilt_draw(draw_record, draw, len(parsed.bones), f"{sub.label}, draw {k}")
            else:
                skins = record.pairs.get(SWITCH_SUB["skins"])
                for skin in skins.block.elements if skins is not None and skins.block else []:
                    skin.raw[:] = PP.skin_record_words(skin, order)
            record.raw[:] = _rebuilt_sub(app.build, order, record, sub, bool(parsed.bones))
    if app.build == "ps3":
        PP.share_ps3_blocks(tree)
    return L.write_tree(tree)


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
    # stage 17
    "PartSource", "cut_ps3_draw", "merge_draws", "ps3_draw_fits", "rebuild_from_zero", "transplant",
    "write_app_mesh_report",
]
