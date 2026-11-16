"""The mesh-level ``.app`` writer of ``25_corrections14.md`` section 1 (W2-1 to W2-15).

The writer starts from the structure tree of the source (``sno_layout``, W11), encodes every
vertex, index, pool, skinning and ps3 draw block from a decoded mesh (an ``AppearanceFile``,
possibly changed), recomputes the counts and derived fields of what changed, keeps the values the
description lists as kept, and lays the file out with the generic writer.

A block whose new bytes equal the stored ones keeps its stored block (so shared blocks stay
shared and an unchanged mesh gives the source bytes); a block whose bytes change gets a block of
its own (W2-14). Records are changed only where something they describe changed.

Not built here (Q376): adding or removing sub-objects, draws or skinning records, changing a
palette, and rebuilding the visual-mesh octree (an empty octree is stored instead, W2-13).
"""

from __future__ import annotations

import math
import re
import struct

from . import sno_layout as L
from .appearance import (
    FAT_VERTEX, FORMAT_RIGID, FORMAT_SKINNED, PS3_DESCRIPTOR_FIXED, PS3_PIECE, AppearanceFile,
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
PS3_SUB = {"fat": 0x04, "index": 0x20, "box": 0xD0, "draws": 0x100, "influence": 0x10C,
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
    def __init__(self, app: AppearanceFile, tree: L.SnoTree, old: AppearanceFile):
        self.build, self.tree, self.old = app.build, tree, old
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


def _fixed_point(sub: SubObject, positions, moved: bool, has_bones: bool):
    """W2-2, W2-3: an unchanged mesh keeps its bias and step. A changed one keeps its bias while
    it stays within 1 of the new box centre on every axis (else round(box centre)) and takes the
    step of the W2-3 rule (Q375)."""
    bias, scale = tuple(sub.bias), tuple(sub.scale)
    if not moved or not positions:
        return bias, scale
    centre = [(min(p[k] for p in positions) + max(p[k] for p in positions)) / 2 for k in range(3)]
    if not all(abs(bias[k] - centre[k]) <= 1 for k in range(3)):
        bias = fixed_point_bias(positions)
    return bias, fixed_point_steps(positions, bias, has_bones)


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


def _switch_sub(ctx: _Context, record: L.Record, sub: SubObject, old: SubObject) -> tuple[bool, bool]:
    order, raw = ctx.order, record.raw
    rigid = [d for d in sub.draws if d.kind == "packed" and not d.skinned]
    skins = [d for d in sub.draws if d.kind == "packed" and d.skinned]
    old_rigid = [d for d in old.draws if d.kind == "packed" and not d.skinned]
    old_skins = [d for d in old.draws if d.kind == "packed" and d.skinned]
    if len(rigid) != len(old_rigid) or len(skins) != len(old_skins):
        raise FormatError(f"{sub.label}: adding or removing packed lists is not described (Q376)")
    fixed, old_fixed = _positions(rigid + skins), _positions(old_rigid + old_skins)
    moved = fixed != old_fixed
    bias, scale = _fixed_point(sub, fixed, moved, ctx.has_bones)
    if moved:
        _pack(raw, order, "3f", SWITCH_SUB["bias"], *bias, field="sub-object bias")
        _pack(raw, order, "3f", SWITCH_SUB["scale"], *scale, field="sub-object step")
    triangles = moved
    for draw, before in zip(rigid, old_rigid):
        _set_bytes(record, SWITCH_SUB["packed"], encode_packed_vertices(draw, bias, scale, order))
        _pack(raw, order, "i", SWITCH_SUB["packed_count"], len(draw.positions), field="nPackVertCount (+0xA0)")
        triangles |= draw.indices != before.indices
    skin_value = record.pairs.get(SWITCH_SUB["skins"])
    skin_records = skin_value.block.elements if skin_value is not None and skin_value.block else []
    for skin, draw, before in zip(skin_records, skins, old_skins):
        stored = _block_bytes(skin.pairs.get(SKIN["vertices"]))
        slots = [stored[i * PACKED_SKINNED + 28:i * PACKED_SKINNED + 32] for i in range(len(stored) // PACKED_SKINNED)]
        if order == ">":
            slots = [s[::-1] for s in slots]
        changed = _set_bytes(skin, SKIN["vertices"], encode_packed_vertices(draw, bias, scale, order, slots))
        changed |= _set_bytes(skin, SKIN["indices"], encode_indices(draw.indices, order))
        if changed:  # W2-1, W2-6
            _pack(skin.raw, order, "i", SKIN["vertex_count"], len(draw.positions), field="skinning nPackVertCount (+0x18)")
            _pack(skin.raw, order, "i", SKIN["index_count"], len(draw.indices), field="skinning nPackIndexCount (+0x30)")
            _pack(skin.raw, order, "i", SKIN["bones"], len(_block_bytes(skin.pairs.get(SKIN["palette"]))) // 4, field="skinning bone count (+0x00)")
            start, end = SKIN["zero"]
            skin.raw[start:end] = bytes(end - start)
        triangles |= draw.indices != before.indices
    fat_moved = sub.fat_draw is not None and old.fat_draw is not None and \
        sub.fat_draw.positions != old.fat_draw.positions
    if sub.fat_draw is not None and old.fat_draw is not None:
        triangles |= fat_moved or sub.fat_draw.indices != old.fat_draw.indices
    _fat_and_pools(ctx, record, sub, old, rigid[0].indices if rigid else None, SWITCH_SUB["index"])
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


def _ps3_sub(ctx: _Context, record: L.Record, sub: SubObject, old: SubObject) -> tuple[bool, bool]:
    order, raw = ctx.order, record.raw
    fat, old_fat = sub.fat_draw, old.fat_draw
    draws = [d for d in sub.draws if d.kind == "ps3"]
    old_draws = [d for d in old.draws if d.kind == "ps3"]
    value = record.pairs.get(PS3_SUB["draws"])
    records = value.block.elements if value is not None and value.block else []
    if len(draws) != len(old_draws) or len(records) != len(draws):
        raise FormatError(f"{sub.label}: adding or removing draws is not described (Q376)")
    fixed, old_fixed = _positions(draws), _positions(old_draws)
    moved = fixed != old_fixed
    bias, scale = _fixed_point(sub, fixed, moved, ctx.has_bones)
    if moved:
        _pack(raw, order, "3f", PS3_SUB["bias"], *bias, field="sub-object bias")
        _pack(raw, order, "3f", PS3_SUB["scale"], *scale, field="sub-object step")
    triangles = moved
    for k, (draw_record, draw, before) in enumerate(zip(records, draws, old_draws)):
        blocks = encode_ps3_draw(draw, bias, scale, before.local_joints)
        triangles |= draw.indices != before.indices
        if all(_block_bytes(draw_record.pairs.get(DRAW[name])) == blocks[name]
               for name in ("descriptor", "indices", "a", "b", "c")):
            continue
        words = _ps3_draw_words(draw_record, blocks, len(draw.positions), len(draw.indices),
                                f"{sub.label}, draw {k}")
        for name in ("descriptor", "indices", "a", "b", "c"):
            _set_bytes(draw_record, DRAW[name], blocks[name])
        r = draw_record.raw
        for name, fmt, values in words:
            _pack(r, ">", fmt, DRAW[name], *values, field=name)
    fat_moved = fat is not None and old_fat is not None and fat.positions != old_fat.positions
    if fat is not None and old_fat is not None:
        triangles |= fat_moved or fat.indices != old_fat.indices
    _fat_and_pools(ctx, record, sub, old, None, PS3_SUB["index"])
    old_n = sum(len(d.positions) for d in old_draws) or (len(old_fat.positions) if old_fat else 0)
    new_n = sum(len(d.positions) for d in draws) or (len(fat.positions) if fat else 0)
    _resize_influences(record, PS3_SUB["influence"], old_n, new_n)
    if moved:
        _write_box(raw, order, PS3_SUB["box"], _decoded(fixed, bias, scale))
    elif fat_moved and not fixed:
        _write_box(raw, order, PS3_SUB["box"], fat.positions)
    return moved or fat_moved, triangles


def _ps3_draw_words(record: L.Record, blocks: dict, nv: int, nidx: int, label: str) -> list:
    """W2-9, W2-10: the record words of a changed ps3 draw, every one checked before anything is
    written: the work buffer (0xC000) first, then the stream sizes (each at most 0x4000, G1-5),
    then the 16-bit index-pair pieces and size words."""
    fmt = blocks["descriptor"][0]
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
    the farthest corner of any sub-object box. W2-13: an empty octree when triangles changed."""
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
    if ctx.geo0_triangles:
        top.raw[STRUCT_OCTREE:STRUCT_OCTREE + OCTREE_COUNTS] = bytes(OCTREE_COUNTS)
        for offset in OCTREE_PAIRS:
            top.pairs[offset] = L.PairValue(None)


def app_mesh_tree(app: AppearanceFile, mesh: AppearanceFile | None = None) -> L.SnoTree:
    """W2-1 to W2-15: the structure tree of ``app``'s source with every mesh block of ``mesh`` (a
    decoded ``.app``, by default ``app`` itself, changed by the caller) encoded again."""
    mesh = app if mesh is None else mesh
    if mesh.build != app.build:
        raise FormatError(f"mesh of build {mesh.build} for an .app of build {app.build}")
    source = bytes(app.source)
    tree = L.parse_tree(source, "app", app.build)
    for path, data in app.block_edits.items():
        L.set_block(tree, path, data)
    old = parse_appearance(source, app.build)
    ctx = _Context(app, tree, old)
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
    return tree


def write_app_mesh(app: AppearanceFile, mesh: AppearanceFile | None = None, check: bool = True) -> bytes:
    """Encode ``mesh`` (default: ``app`` as changed by the caller) into ``app``'s ``.app`` and lay
    it out (W2-1 to W2-15, W11). An unchanged mesh gives the source bytes. With ``check`` the
    output is read again and must give the same counts."""
    out = L.write_tree(app_mesh_tree(app, mesh))
    if check:
        again = parse_appearance(out, app.build)
        mesh = app if mesh is None else mesh
        for a, b in zip(again.sub_objects + again.extra_sub_objects, mesh.sub_objects + mesh.extra_sub_objects):
            if [len(d.positions) for d in a.draws] != [len(d.positions) for d in b.draws] or \
                    [len(d.indices) for d in a.draws] != [len(d.indices) for d in b.draws]:
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
    "app_mesh_tree", "encode_fat_vertices", "encode_indices", "encode_packed_vertices",
    "encode_ps3_draw", "fixed_point_bias", "fixed_point_steps", "index_pool_problems",
    "normal_bytes", "ps3_command_space", "ps3_work_size", "raw_coordinates", "skin_slots",
    "uv1_words", "write_app_mesh",
]
