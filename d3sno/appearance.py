"""``.app`` files for both builds, with Wavefront OBJ and glTF 2.0 export."""

from __future__ import annotations

import base64
import json
import math
import struct
from dataclasses import dataclass, field

from .binary import ByBuild, ByteView, FormatError, Pair, build_order
from .cloth import ClothRecord, cloth_problems, read_cloth
from .material import AppMaterials, read_app_materials, sub_object_texture
from .sno import SnoHeader, parse_header, resolve_build
from .transform import Transform, quat_normalise, read_transform, transforms_match

_BONE_RECORD = ByBuild({"ps3": 176, "switch": 256})
_SUB_RECORD = ByBuild({"ps3": 308, "switch": 240})
_BONE_PARENT = ByBuild({"ps3": 4, "switch": 8})
# Positions of the rest transform, its inverse and the parent-relative transform.
_BONE_TRANSFORMS = ByBuild({"ps3": (0x30, 0x50, 0x70), "switch": (0x34, 0x54, 0x74)})
_BONE_BIND = 0x94
_BONE_BIND_INVERSE = 0xB4
BONE_RANGE_UNIT = 48
HARDPOINT_RECORD = 40
DRAW_RECORD = 160
SKIN_RECORD = 104
FORMAT_RIGID = 0x81
FORMAT_SKINNED = 0x83
SWITCH_INDEX_POOL_AT = 0x220
# G1-2: serAppearanceSubObjectsIndices / ...Vertices (body 0x210/0x218 Switch, 0x220/0x228 PS3).
POOL_PAIRS = ByBuild({"switch": (0x220, 0x228), "ps3": (0x230, 0x238)})
FAT_VERTEX = 32
# G1-1: the two geo sets of the structure record.
GEOSET_HIGHPOLY = 0
GEOSET_LOWPOLY = 1
GEOSET_NAMES = {GEOSET_HIGHPOLY: "high-poly set", GEOSET_LOWPOLY: "low-poly set"}
FAT_BOUNDS_TOLERANCE = 1e-3
FAT_AXIS_TOLERANCE = 1e-4
_FLOAT32_MAX = 3.4028234663852886e38


@dataclass
class Bone:
    index: int
    name_hash: int
    parent: int
    rest: Transform
    rest_inverse: Transform
    local: Transform  # parent-relative
    bind: Transform | None  # switch only
    bind_inverse: Transform | None  # switch only
    raw: bytes

    @property
    def translation(self) -> tuple[float, float, float]:
        return self.rest.t

    @property
    def skin_inverse(self) -> Transform:
        """The bind inverse. ps3 has none, so the rest inverse stands in (*Likely*)."""
        return self.bind_inverse if self.bind_inverse is not None else self.rest_inverse


@dataclass
class Draw:
    skinned: bool
    positions: list[tuple[float, float, float]]
    normals: list[tuple[float, float, float]]
    colours_1: list[bytes]
    colours_2: list[bytes]
    uv1: list[tuple[float, float]]
    uv2_raw: list[tuple[int, int]]
    indices: list[int]
    # Each vertex holds four entries. Joints are skeleton bone ids; a joint with weight 0 is 0.
    joints: list[tuple[int, ...]] | None = None
    weights: list[tuple[int, ...]] | None = None
    palette: list[int] | None = None
    local_joints: list[tuple[int, ...]] | None = None  # ps3: as stored, indices into palette
    kind: str = "packed"  # "packed" (fixed point, skinned or rigid), "ps3" or "fat" (G1-3)
    # W17-10: the stored record this draw was read from: the ps3 draw index, the switch / x360
    # skinning record index, 0 for the switch / x360 rigid list; None for a fat draw and for a new
    # draw. Not compared (two draws with equal values are equal wherever they come from).
    origin: int | None = field(default=None, compare=False)

    def uv2(self, lightmap_v_scale: float = 0.0) -> list[tuple[float, float]]:
        """Lightmap coordinates. A non-zero V scale comes from the lightmap's ``.tex``."""
        factor = lightmap_v_scale or 1.0
        return [
            ((u - 0x8000) / 8192, (v - 0x8000) / 8192 * factor) for u, v in self.uv2_raw
        ]


@dataclass
class SubObject:
    index: int
    name: str | None
    format_bits: int | None
    bias: tuple[float, float, float]
    scale: tuple[float, float, float]
    bounds_centre: tuple[float, float, float] | None
    bounds_half: tuple[float, float, float] | None
    draws: list[Draw]
    influences: list[tuple[tuple[int, int], ...]] = field(default_factory=list)
    material_hash: int | None = None
    sub_hash: int | None = None
    flags: int = 0  # dwFlags (on Switch also in ``format_bits``)
    has_cloth: bool = False  # the cloth pair is not empty
    cloth_only: bool = False  # indices whose vertices live in the cloth record (T11)
    # +0x04 nVertCount: the number of fat vertices this sub-object owns in the vertex pool (G1-2;
    # V7-T11 on Switch and x360: non-zero exactly for cloth without packed rigid vertices).
    cloth_vertex_count: int = 0
    geo_set: int = GEOSET_HIGHPOLY  # G1-1
    index_count: int = 0  # nIndexCount (Switch +0x28, PS3 +0x20)
    fat_draw: Draw | None = None  # the fat vertices and pool indices (G1-2, G1-3)
    cloth: ClothRecord | None = None  # G1-4
    bounds_raw: bytes = b""  # aabbBounds as stored (24 bytes)
    # W17-10 identity fields (not compared). ``origin``: (geo set, index) of the template
    # sub-object this one was read from; None marks a new sub-object. For a new one,
    # ``template_part`` names the template sub-object (geo set, index) whose copied values it takes
    # (W17-3), ``material_source`` the source of a new material entry when its ``material_hash`` is
    # not in the table (an int hash of the same file, ``app_parts.MaterialSource`` or (donor file,
    # hash), W17-8), and ``stored_from`` (set by ``app_mesh.transplant``) the
    # ``app_mesh.PartSource`` whose stored record it copies (W17-11).
    origin: tuple[int, int] | None = field(default=None, compare=False)
    template_part: tuple[int, int] | None = field(default=None, compare=False)
    material_source: object = field(default=None, compare=False, repr=False)
    stored_from: object = field(default=None, compare=False, repr=False)

    @property
    def vert_count(self) -> int:
        return self.cloth_vertex_count

    @property
    def list_name(self) -> str:
        return GEOSET_NAMES[self.geo_set]

    @property
    def label(self) -> str:
        """How messages name the sub-object: its geo set and index (G1-1)."""
        return f"{self.list_name}, sub-object {self.index}"

    @property
    def bounds_stored(self) -> bool:
        """G1-1: all-zero bounds mean no box is stored."""
        return any(self.bounds_raw) if self.bounds_raw else True


@dataclass(frozen=True)
class Hardpoint:
    hash: int
    flag: int
    bone: int
    rotation: tuple[float, float, float, float]
    position: tuple[float, float, float]


@dataclass
class AppearanceFile:
    header: SnoHeader
    build: str
    bones: list[Bone]
    hardpoints: list[Hardpoint]
    sub_objects: list[SubObject]
    skipped_sub_objects: list[int]
    warnings: list[str]
    material_data: AppMaterials = field(default_factory=AppMaterials)
    extra_sub_objects: list[SubObject] = field(default_factory=list)  # geo set 1, low-poly (G1-1)
    source: bytearray = field(default_factory=bytearray, repr=False)
    notes: list[str] = field(default_factory=list)  # reported, never a strict failure
    # Section 3 of 22: block changes for the structure-level writer, by pair path
    # (``sno_layout.iter_pairs``); see ``write.set_app_block``.
    block_edits: dict[str, bytes] = field(default_factory=dict, repr=False)


# Sub-object flag bits (stage 7, T10).
SUB_CLOTH_BIT = 8
SUB_PACK_GEOMETRY_BIT = 15
SUB_PACK_SKIN_BIT = 16


def sub_object_flag_problems(sub: SubObject, build: str, packed: bool = False, skinned: bool = False) -> list[str]:
    """T10: cloth data implies bit 8 (not the reverse). On Switch, bit 16 is set exactly when
    skinning records exist, and packed rigid vertices exist exactly when bit 15 is set and
    bit 16 is clear."""
    found = []
    bit = lambda n: bool(sub.flags >> n & 1)  # noqa: E731
    if sub.has_cloth and not bit(SUB_CLOTH_BIT):
        found.append(f"{sub.label}: cloth data without flag bit 8")
    if build != "ps3":  # Switch and x360
        if bit(SUB_PACK_SKIN_BIT) != skinned:
            found.append(f"{sub.label}: flag bit 16 does not match the skinning records")
        if packed != (bit(SUB_PACK_GEOMETRY_BIT) and not bit(SUB_PACK_SKIN_BIT)):
            found.append(f"{sub.label}: flag bits 15/16 do not match the packed vertices")
        if bool(sub.cloth_vertex_count) != (sub.has_cloth and not packed):
            cloth = "present" if sub.has_cloth else "absent"
            rigid = "present" if packed else "absent"
            found.append(f"{sub.label}: word +0x04 is {sub.cloth_vertex_count}, cloth data {cloth}, "
                         f"packed rigid vertices {rigid} (V7-T11)")
    return found


def check_float32(values: tuple[float, ...], what: str) -> None:
    """Reject NaN, infinity and magnitudes a float32 export could not hold."""
    if not all(abs(v) <= _FLOAT32_MAX for v in values):
        raise FormatError(f"{what} {values} is outside the float32 range")


def _position(raw: tuple[int, int, int], scale, bias) -> tuple[float, float, float]:
    result = (
        raw[0] * scale[0] + bias[0],
        raw[1] * scale[1] + bias[1],
        raw[2] * scale[2] + bias[2],
    )
    check_float32(result, "position")
    return result


def _normal(x: int, y: int, z: int) -> tuple[float, float, float]:
    return ((x - 128) / 128, (y - 128) / 128, (z - 128) / 128)


def _uv1(u: int, v: int) -> tuple[float, float]:
    return ((u - 0x8000) / 512, (v - 0x8000) / 512)


def _read_indices(raw: bytes, order: str, count: int, vertices: int, pos: int) -> list[int]:
    if count % 3:
        raise FormatError(f"index count {count} is not divisible by 3", pos)
    indices = list(struct.unpack_from(f"{order}{count}H", raw, 0))
    if indices and max(indices) >= vertices:
        raise FormatError(f"index {max(indices)} past {vertices} vertices", pos)
    return indices


def _section(
    view: ByteView, count: int, pair: Pair, record: int, position: int, what: str,
) -> bytes:
    if count < 0 or pair.size != count * record:
        raise FormatError(f"{what}: pair size {pair.size} != {count} * {record}", position)
    return view.pair_data(pair, position, what)


def _padded_stream(view: ByteView, pair: Pair, needed: int, position: int, what: str) -> bytes:
    """Read a stream whose size may be rounded up to a multiple of 16."""
    raw = view.pair_data(pair, position, what)
    if not needed <= len(raw) < needed + 16:
        raise FormatError(f"{what}: size {len(raw)}, needed {needed}", position)
    return raw


def _read_bones(view: ByteView, build: str) -> list[Bone]:
    count = view.i32(0x24, "bone count")
    size = _BONE_RECORD[build]
    raw = _section(view, count, view.pair(0x28), size, 0x28, "bone array")
    inner = ByteView(raw, view.order)
    bones = []
    for i in range(count):
        base = i * size
        parent = inner.i32(base + _BONE_PARENT[build])
        if parent != -1 and not 0 <= parent < i:
            raise FormatError(f"bone {i}: parent {parent} is not earlier", 0x28)
        rest_at, inverse_at, local_at = _BONE_TRANSFORMS[build]
        rest = read_transform(inner, base + rest_at)
        check_float32(rest.t, f"bone {i} translation")
        bind = bind_inverse = None
        if build != "ps3":  # Switch and x360
            bind = read_transform(inner, base + _BONE_BIND)
            bind_inverse = read_transform(inner, base + _BONE_BIND_INVERSE)
        bones.append(Bone(
            i, inner.u32(base), parent, rest, read_transform(inner, base + inverse_at),
            read_transform(inner, base + local_at), bind, bind_inverse, inner.take(base, size),
        ))
    return bones


def _check_bone_chain(bones: list[Bone], warnings: list[str]) -> None:
    """``parent.rest x parentRelative = rest``."""
    for bone in bones:
        if bone.parent < 0:
            continue
        composed = bones[bone.parent].rest.compose(bone.local)
        if not transforms_match(composed, bone.rest):
            warnings.append(f"bone {bone.index}: parent rest and local do not give rest")


def _read_hardpoints(view: ByteView, bone_count: int, warnings: list[str]) -> list[Hardpoint]:
    count = view.i32(0xC8, "hardpoint count")
    raw = _section(view, count, view.pair(0xCC), HARDPOINT_RECORD, 0xCC, "hardpoint array")
    inner = ByteView(raw, view.order)
    hardpoints = []
    for i in range(count):
        at = i * HARDPOINT_RECORD
        point = Hardpoint(
            inner.u32(at), inner.u32(at + 4), inner.i32(at + 8), inner.f32s(at + 12, 4),
            inner.f32s(at + 28, 3),
        )
        if point.bone >= bone_count or point.bone < -1:
            warnings.append(f"hardpoint {i}: bone {point.bone} of {bone_count}")
        hardpoints.append(point)
    return hardpoints




def _read_palette(r: ByteView, bone_count: int, base: int) -> list[int]:
    """The bones of range 1 then range 2. Values are byte offsets, 48 per bone."""
    starts = r.unpack("HH", 0x60)
    sizes = r.unpack("HH", 0x64)
    palette: list[int] = []
    for start, size in zip(starts, sizes):
        if start % BONE_RANGE_UNIT or size % BONE_RANGE_UNIT:
            raise FormatError(f"bone range {start}+{size} is not a multiple of 48", base + 0x60)
        if size:
            first = start // BONE_RANGE_UNIT
            palette.extend(range(first, first + size // BONE_RANGE_UNIT))
    if palette and max(palette) >= bone_count:
        raise FormatError(f"bone range reaches bone {max(palette)} of {bone_count}", base + 0x60)
    return palette


def _translate_joints(
    slots: bytes, weights: tuple[int, ...], palette, base: int,
) -> tuple[int, ...]:
    joints = []
    for slot, weight in zip(slots, weights):
        if weight == 0:
            joints.append(0)
        elif slot >= len(palette):
            raise FormatError(f"palette index {slot} past {len(palette)}", base)
        else:
            joints.append(palette[slot])
    return tuple(joints)


def _align(value: int, step: int) -> int:
    return (value + step - 1) // step * step


# G1-6: the vertex-format blocks of each format.
PS3_FORMAT_BLOCKS = {
    FORMAT_RIGID: (
        bytes.fromhex("01060100 00000000 00050301 06000000".replace(" ", "")),
        bytes.fromhex("010C0100 00000000 00020301 0C000000".replace(" ", "")),
    ),
    FORMAT_SKINNED: (
        bytes.fromhex(("02 0A 02 00 00000000 00 05 03 01 06 00 00 00 06 07 04 02 04 02 00 00"
                       + " 00" * 8).replace(" ", "")),
        bytes.fromhex(("03 1C 03 00 00000000 00 02 03 01 0C 00 00 00 0C 07 04 02 04 02 00 00"
                       " 10 02 03 04 0C 0F 00 00").replace(" ", "")),
    ),
}
# G1-5: descriptor bytes 2..7 per format; bytes 12..15 are FF.
PS3_DESCRIPTOR_FIXED = {
    FORMAT_RIGID: bytes.fromhex("FF0000FF1000"),
    FORMAT_SKINNED: bytes.fromhex("FF00FFFF1300"),
}
PS3_PIECE = 0x4000
PS3_OUTPUT_STRIDE = {FORMAT_RIGID: 12, FORMAT_SKINNED: 28}


def ps3_draw_problems(view: ByteView, record: bytes, descriptor: bytes, base: int) -> list[str]:
    """G1-5 and G1-6: size words, buffer sizes, fixed descriptor bytes and format blocks of a
    PS3 draw (``s10_ps3_draw_rules.tsv``; every rule holds in all 126,957 draws). The work-buffer
    size at +0x98 is only bounded: its formula is open, so a writer keeps the stored value."""
    r = ByteView(record, view.order)
    fmt = descriptor[0]
    skinned = fmt == FORMAT_SKINNED
    a_stride, c_stride = (10, 16) if skinned else (6, 20)
    nv, nidx = r.u32(0x24), r.u32(0x10)
    found = []
    index_size, a_size = r.pair(0x18).size, r.pair(0x2C).size
    b_size, c_size = r.pair(0x54).size, r.pair(0x6C).size
    pieces = (index_size, 0) if index_size <= PS3_PIECE else (PS3_PIECE, index_size - PS3_PIECE)
    if r.unpack("HH", 0x20) != pieces:
        found.append(f"+0x20 {r.unpack('HH', 0x20)} is not the index pair size {index_size} in pieces")
    words = r.unpack("16H", 0x34)
    if words[0] != a_size or any(words[1:]):
        found.append("+0x34 is not (stream A pair size, 15 zeros)")
    if a_size > PS3_PIECE or b_size > PS3_PIECE:
        found.append("a stream pair is larger than 0x4000")
    if r.unpack("HH", 0x5C) != (b_size, 0):
        found.append("+0x5C is not (stream B pair size, 0)")
    work, area = r.unpack("II", 0x98)
    want_area = _align(4 * nv, 32) if skinned else _align(2 * nv, 16)
    if area != want_area:
        found.append(f"+0x9C is {area}, the per-vertex work area is {want_area}")
    if work % 128 or work > 0xC000 or work < PS3_OUTPUT_STRIDE[fmt] * nv + 2 * nidx + 320:
        found.append(f"+0x98 work-buffer size {work} is out of its bounds")
    if index_size != _align(2 * nidx, 16):
        found.append(f"index pair size {index_size} is not align16(2 x {nidx})")
    if a_size != _align(a_stride * nv, 16):
        found.append(f"stream A pair size {a_size} is not align16({a_stride} x {nv})")
    if b_size != (_align(8 * nv, 16) if skinned else 0):
        found.append(f"stream B pair size {b_size} does not follow the vertex count")
    if c_size != c_stride * nv:
        found.append(f"stream C pair size {c_size} is not {c_stride} x {nv}")
    for at in (0x00, 0x04, 0x14, 0x28, 0x68):
        if r.u32(at):
            found.append(f"word +0x{at:02X} is not zero")
    if any(record[0x74:0x80]) or any(record[0x88:0x90]):
        found.append("+0x74..+0x7F or +0x88..+0x8F is not zero")
    if descriptor[2:8] != PS3_DESCRIPTOR_FIXED[fmt] or descriptor[12:16] != b"\xFF" * 4:
        found.append("descriptor bytes 2..7 or 12..15 are not the fixed bytes of the format")
    blocks = PS3_FORMAT_BLOCKS[fmt]
    for name, at, want in (("A", 0x80, blocks[0]), ("B", 0x90, blocks[1])):
        got = view.pair_data(r.pair(at), base + at, f"vertex format block {name}")
        if got != want:
            found.append(f"vertex format block {name} is not the block of format 0x{fmt:02X}")
    return found


def _read_ps3_draw(
    view: ByteView, record: bytes, sub: SubObject, base: int, bone_count: int,
) -> tuple[Draw, str | None]:
    r = ByteView(record, view.order)
    descriptor_pair = r.pair(0x08)
    if descriptor_pair.size != 16:
        raise FormatError(f"draw descriptor size {descriptor_pair.size}", base + 0x08)
    descriptor = view.pair_data(descriptor_pair, base + 0x08, "draw descriptor")
    fmt = descriptor[0]
    if fmt not in (FORMAT_RIGID, FORMAT_SKINNED):
        raise FormatError(f"draw format 0x{fmt:02X}", descriptor_pair.file_offset)
    skinned = fmt == FORMAT_SKINNED
    index_count = r.u32(0x10)
    vertex_count = r.u32(0x24)
    d = ByteView(descriptor, view.order)
    if (d.u16(8), d.u16(10)) != (vertex_count, index_count):
        raise FormatError("draw descriptor counts differ from the draw", descriptor_pair.file_offset)

    a_stride, c_stride = (10, 16) if skinned else (6, 20)
    a_pair, b_pair, c_pair = r.pair(0x2C), r.pair(0x54), r.pair(0x6C)
    a_raw = _padded_stream(view, a_pair, vertex_count * a_stride, base + 0x2C, "stream A")
    c_raw = _section(view, vertex_count, c_pair, c_stride, base + 0x6C, "stream C")
    if skinned:
        b_raw = _padded_stream(view, b_pair, vertex_count * 8, base + 0x54, "stream B")
    else:
        b_raw = view.pair_data(b_pair, base + 0x54, "stream B")
        if b_raw:
            raise FormatError(f"stream B size {len(b_raw)} in a rigid draw", base + 0x54)
    index_pair = r.pair(0x18)
    index_raw = view.pair_data(index_pair, base + 0x18, "index buffer")
    if len(index_raw) < index_count * 2:
        raise FormatError(f"index buffer of {len(index_raw)} for {index_count} indices", base + 0x18)
    indices = _read_indices(index_raw, view.order, index_count, vertex_count, base + 0x10)

    draw = Draw(skinned, [], [], [], [], [], [], indices, kind="ps3")
    problems = ps3_draw_problems(view, record, descriptor, base)
    warning = f"{sub.label}: draw: {problems[0]} (G1-5)" if problems else None
    if skinned:
        draw.joints, draw.weights, draw.local_joints = [], [], []
        draw.palette = _read_palette(r, bone_count, base)
    for i in range(vertex_count):
        at = i * a_stride
        raw_pos = struct.unpack_from(">3h", a_raw, at)
        draw.positions.append(_position(raw_pos, sub.scale, sub.bias))
        c_at = i * c_stride
        if skinned:
            draw.normals.append(_normal(*a_raw[at + 6:at + 9]))
            draw.colours_1.append(c_raw[c_at:c_at + 4])
            draw.colours_2.append(c_raw[c_at + 4:c_at + 8])
            uv_at = c_at + 8
            influence = b_raw[i * 8:i * 8 + 8]
            weights = tuple(influence[0::2])
            draw.weights.append(weights)
            draw.local_joints.append(tuple(influence[1::2]))
            draw.joints.append(_translate_joints(influence[1::2], weights, draw.palette, base))
            if sum(weights) != 255 or list(weights) != sorted(weights, reverse=True):
                warning = f"{sub.label}: weights not descending or not summing to 255"
        else:
            draw.normals.append(_normal(*c_raw[c_at:c_at + 3]))
            draw.colours_1.append(c_raw[c_at + 4:c_at + 8])
            draw.colours_2.append(c_raw[c_at + 8:c_at + 12])
            uv_at = c_at + 12
        u1, v1, u2, v2 = struct.unpack_from("<4H", c_raw, uv_at)
        draw.uv1.append(_uv1(u1, v1))
        draw.uv2_raw.append((u2, v2))
    return draw, warning


def _draw_tolerance(sub: SubObject, draw: Draw) -> list[float]:
    """2 raw units for fixed-point vertices; 1e-3 for fat vertices (model space, G1-3)."""
    if draw.kind == "fat":
        return [FAT_BOUNDS_TOLERANCE] * 3
    return [2 * abs(v) for v in sub.scale]


def _check_bounds(sub: SubObject, warnings: list[str], notes: list[str] | None = None) -> None:
    """G1-1: the vertex-within-bounds rule for every sub-object of both geo sets whose bounds
    are not all zero. All-zero bounds store no box: the rule is skipped, with a note when a
    vertex lies farther than 2 raw units from the origin."""
    draws = list(sub.draws)
    if sub.fat_draw is not None and sub.fat_draw not in draws:
        draws.append(sub.fat_draw)
    if not sub.bounds_stored:
        for draw in draws:
            tolerance = _draw_tolerance(sub, draw)
            if any(abs(p[k]) > tolerance[k] for p in draw.positions for k in range(3)):
                if notes is not None:
                    notes.append(f"{sub.label}: bounds not stored (all zero), check skipped (G1-1)")
                return
        return
    for draw in draws:
        tolerance = _draw_tolerance(sub, draw)
        for position in draw.positions:
            for axis in range(3):
                reach = sub.bounds_half[axis] + tolerance[axis]
                if abs(position[axis] - sub.bounds_centre[axis]) > reach:
                    warnings.append(f"{sub.label}: vertex outside bounds")
                    return


def _read_ps3_sub_objects(
    view: ByteView, raw: bytes, count: int, bone_count: int, warnings: list[str],
    notes: list[str] | None = None, geo_set: int = GEOSET_HIGHPOLY,
) -> list[SubObject]:
    size = _SUB_RECORD["ps3"]
    subs = []
    for i in range(count):
        record = ByteView(raw[i * size:(i + 1) * size], view.order)
        sub = SubObject(
            i, record.cstring(0x48, 128) or None, None, record.f32s(0x118, 3),
            record.f32s(0x124, 3), record.f32s(0xD0, 3), record.f32s(0xDC, 3), [],
            material_hash=record.u32(0xC8), sub_hash=record.u32(0xCC),
            flags=record.u32(0x00), has_cloth=record.pair(0x30).size > 0,
            cloth_vertex_count=record.i32(0x04), geo_set=geo_set, index_count=record.i32(0x20),
            bounds_raw=record.take(0xD0, 24), origin=(geo_set, i),
        )
        if sub.cloth_vertex_count < 0 or sub.index_count < 0:
            raise FormatError(f"{sub.label}: negative count", 0x74)
        if sub.cloth_vertex_count and record.pair(0x24).size:
            warnings.append(f"{sub.label}: own index pair is not empty (G1-2)")
        if sub.has_cloth:
            _read_sub_cloth(view, record.pair(0x30), "ps3", sub, warnings)
        warnings.extend(sub_object_flag_problems(sub, "ps3"))
        draw_count = record.i32(0xFC)
        draw_pair = record.pair(0x100)
        draws_raw = _section(view, draw_count, draw_pair, DRAW_RECORD, 0x78, "draw array")
        for k in range(draw_count):
            base = draw_pair.file_offset + k * DRAW_RECORD
            draw, warning = _read_ps3_draw(
                view, draws_raw[k * DRAW_RECORD:(k + 1) * DRAW_RECORD], sub, base, bone_count,
            )
            draw.origin = k  # W17-10
            sub.draws.append(draw)
            if warning:
                warnings.append(warning)
        if sub.has_cloth and sub.cloth_vertex_count and sub.draws:
            # V9-G1-3: a PS3 cloth sub-object with fat vertices may also have one PS3 draw with
            # the fat counts. Only the draw is exported (the fat vertices join ``draws`` only when
            # there is no draw), as for the Switch skinned cloth.
            counts = [(len(d.positions), len(d.indices)) for d in sub.draws]
            if counts != [(sub.cloth_vertex_count, sub.index_count)]:
                warnings.append(f"{sub.label}: cloth draws with counts {counts} differ from one "
                                f"draw of nVertCount {sub.cloth_vertex_count}, nIndexCount "
                                f"{sub.index_count} (V9-G1-3)")
        subs.append(sub)
    return subs


def _switch_vertices(
    raw: bytes, count: int, stride: int, sub: SubObject, order: str = "<",
) -> Draw:
    """Packed vertices. On x360 (``order`` big-endian) positions are big-endian and the colour
    groups are byte-reversed (X6); colours are returned in the Switch byte order. Normals and
    texture coordinates are the same bytes on both."""
    draw = Draw(stride == 36, [], [], [], [], [], [], [])
    flip = order == ">"
    for i in range(count):
        at = i * stride
        raw_pos = struct.unpack_from(f"{order}3h", raw, at)
        draw.positions.append(_position(raw_pos, sub.scale, sub.bias))
        draw.normals.append(_normal(*raw[at + 8:at + 11]))
        colour_1, colour_2 = raw[at + 12:at + 16], raw[at + 16:at + 20]
        draw.colours_1.append(colour_1[::-1] if flip else colour_1)
        draw.colours_2.append(colour_2[::-1] if flip else colour_2)
        u1, v1, u2, v2 = struct.unpack_from("<4H", raw, at + 20)
        draw.uv1.append(_uv1(u1, v1))
        draw.uv2_raw.append((u2, v2))
    return draw


def _read_switch_skin(
    view: ByteView, record: bytes, sub: SubObject, bone_count: int, base: int,
) -> Draw:
    r = ByteView(record, view.order)
    palette_count = r.i32(0x00)
    palette_raw = _section(
        view, palette_count, r.pair(0x04), 4, base + 0x04, "palette",
    )
    palette = list(struct.unpack_from(f"{view.order}{palette_count}i", palette_raw, 0))
    vertex_count = r.i32(0x18)
    vertices = _section(view, vertex_count, r.pair(0x28), 36, base + 0x28, "skinned vertices")
    index_count = r.i32(0x30)
    index_raw = view.pair_data(r.pair(0x40), base + 0x40, "skinned indices")
    if index_count < 0 or len(index_raw) < index_count * 2:
        raise FormatError(f"skinned index pair too small for {index_count}", base + 0x40)

    draw = _switch_vertices(vertices, vertex_count, 36, sub, view.order)
    draw.indices = _read_indices(index_raw, view.order, index_count, vertex_count, base + 0x30)
    draw.joints, draw.weights, draw.palette = [], [], palette
    flip = view.order == ">"  # x360: joint and weight groups byte-reversed (X6)
    for i in range(vertex_count):
        at = i * 36 + 28
        slots, raw_weights = vertices[at:at + 4], vertices[at + 4:at + 8]
        if flip:
            slots, raw_weights = slots[::-1], raw_weights[::-1]
        weights = tuple(raw_weights)
        joints = _translate_joints(slots, weights, palette, base)
        for joint, weight in zip(joints, weights):
            if weight and not 0 <= joint < bone_count:
                raise FormatError(f"palette entry {joint} is not a bone", base)
        draw.joints.append(joints)
        draw.weights.append(weights)
    return draw


def _read_sub_cloth(view: ByteView, pair: Pair, build: str, sub: SubObject, warnings: list[str]) -> None:
    try:
        sub.cloth = read_cloth(view, pair, build, pair.file_offset)
    except FormatError as exc:
        warnings.append(f"{sub.label}: cloth record not read: {exc}")


def _read_switch_sub_objects(
    view: ByteView, raw: bytes, count: int, bone_count: int,
    skipped: list[int], warnings: list[str], running: int = 0, notes: list[str] | None = None,
    geo_set: int = GEOSET_HIGHPOLY, build: str = "switch",
) -> tuple[list[SubObject], int]:
    size = _SUB_RECORD["switch"]
    pool_pair = view.pair(SWITCH_INDEX_POOL_AT, "index pool pair")
    pool = view.pair_data(pool_pair, SWITCH_INDEX_POOL_AT, "index pool")
    subs = []
    for i in range(count):
        record = ByteView(raw[i * size:(i + 1) * size], view.order)
        sub = SubObject(
            i, None, record.u32(0x00), record.f32s(0xD8, 3), record.f32s(0xE4, 3),
            record.f32s(0x6C, 3), record.f32s(0x78, 3), [],
            material_hash=record.u32(0x64), sub_hash=record.u32(0x68),
            flags=record.u32(0x00), has_cloth=record.pair(0x40).size > 0,
            cloth_vertex_count=record.u32(0x04), geo_set=geo_set, index_count=record.i32(0x28),
            bounds_raw=record.take(0x6C, 24), origin=(geo_set, i),
        )
        if sub.has_cloth:
            _read_sub_cloth(view, record.pair(0x40), build, sub, warnings)
        index_count = record.i32(0x28)
        vertex_count = record.i32(0xA0)
        if index_count < 0 or vertex_count < 0:
            raise FormatError(f"{sub.label}: negative count", 0x78)
        start = running
        running += index_count
        skin_pair = record.pair(0xB8)
        skin_raw = view.pair_data(skin_pair, 0x78, "skinning records")
        if len(skin_raw) % SKIN_RECORD:
            raise FormatError(f"skinning pair size {len(skin_raw)}", skin_pair.file_offset)
        if vertex_count:
            vertices = _section(
                view, vertex_count, record.pair(0xB0), 28, 0x78, "rigid vertices",
            )
            if (start + index_count) * 2 > len(pool):
                raise FormatError(f"{sub.label}: indices past the pool", SWITCH_INDEX_POOL_AT)
            draw = _switch_vertices(vertices, vertex_count, 28, sub, view.order)
            draw.indices = _read_indices(
                pool[start * 2:], view.order, index_count, vertex_count, SWITCH_INDEX_POOL_AT,
            )
            draw.origin = 0  # W17-10: the rigid list
            sub.draws.append(draw)
        elif index_count and not skin_raw:
            if sub.has_cloth and record.pair(0x08).size == 0:
                # T11 / G1-3: the vertices are fat vertices in the appearance vertex pool.
                sub.cloth_only = True
            elif sub.cloth_vertex_count:
                pass  # fat vertices without a cloth record (G1-2)
            else:
                skipped.append(i)
                warnings.append(f"{sub.label}: {index_count} indices but no vertices, skipped")
        warnings.extend(sub_object_flag_problems(sub, "switch", vertex_count > 0, bool(skin_raw)))
        for k in range(len(skin_raw) // SKIN_RECORD):
            skin = skin_raw[k * SKIN_RECORD:(k + 1) * SKIN_RECORD]
            skin_draw = _read_switch_skin(
                view, skin, sub, bone_count, skin_pair.file_offset + k * SKIN_RECORD,
            )
            skin_draw.origin = k  # W17-10: skinning record k
            sub.draws.append(skin_draw)
            if sub.has_cloth and sub.cloth_vertex_count:
                # G1-3: a skinned cloth sub-object's skinning record has the fat counts.
                r = ByteView(skin, view.order)
                if (r.i32(0x18), r.i32(0x30)) != (sub.cloth_vertex_count, index_count):
                    warnings.append(f"{sub.label}: skinning record counts {r.i32(0x18)}, "
                                    f"{r.i32(0x30)} differ from nVertCount {sub.cloth_vertex_count}, "
                                    f"nIndexCount {index_count} (G1-3)")
        influence = view.pair_data(record.pair(0xC8), 0x78, "influence stream")
        if len(influence) % 6:
            raise FormatError(f"influence stream size {len(influence)}", 0x78)
        for at in range(0, len(influence), 6):
            sub.influences.append(tuple(
                (influence[at + n], influence[at + n + 1]) for n in (0, 2, 4)
            ))
        subs.append(sub)
    return subs, running


# --- Fat vertices and pools (G1-2, G1-3) -------------------------------------------------------

def _fat_vertices(raw: bytes, order: str) -> Draw:
    draw = Draw(False, [], [], [], [], [], [], [], kind="fat")
    for at in range(0, len(raw), FAT_VERTEX):
        position = struct.unpack_from(order + "3f", raw, at)
        check_float32(position, "fat position")
        draw.positions.append(position)
        draw.normals.append(_normal(*raw[at + 12:at + 15]))
        draw.colours_1.append(raw[at + 16:at + 20])
        draw.colours_2.append(raw[at + 20:at + 24])
        u1, v1, u2, v2 = struct.unpack_from("<4H", raw, at + 24)
        draw.uv1.append(_uv1(u1, v1))
        draw.uv2_raw.append((u2, v2))
    return draw


def _assign_fat(view: ByteView, build: str, subs: list[SubObject], warnings: list[str]) -> None:
    """G1-2: walk geo set 0 then geo set 1 with running vertex and index sums."""
    index_at, vertex_at = POOL_PAIRS[build]
    index_pool = view.pair_data(view.pair(index_at, "index pool pair"), index_at, "index pool")
    vertex_pool = view.pair_data(view.pair(vertex_at, "vertex pool pair"), vertex_at, "vertex pool")
    total_vertices = sum(sub.cloth_vertex_count for sub in subs)
    total_indices = sum(sub.index_count for sub in subs)
    if (vertex_pool or total_vertices) and len(vertex_pool) != FAT_VERTEX * total_vertices:
        warnings.append(f"vertex pool of {len(vertex_pool)} bytes for {total_vertices} fat vertices (G1-2)")
    if build == "ps3" and (index_pool or total_indices) and len(index_pool) != 2 * total_indices:
        warnings.append(f"index pool of {len(index_pool)} bytes for {total_indices} indices (G1-2)")
    v_run = i_run = 0
    for sub in subs:
        n, count = sub.cloth_vertex_count, sub.index_count
        if n > 0:
            vertices = vertex_pool[v_run * FAT_VERTEX:(v_run + n) * FAT_VERTEX]
            indices_raw = index_pool[i_run * 2:(i_run + count) * 2]
            if len(vertices) != n * FAT_VERTEX or len(indices_raw) != count * 2:
                warnings.append(f"{sub.label}: fat range outside its pool (G1-2)")
            else:
                draw = _fat_vertices(vertices, view.order)
                try:
                    draw.indices = _read_indices(indices_raw, view.order, count, n, index_at)
                except FormatError as exc:
                    warnings.append(f"{sub.label}: fat indices: {exc}")
                if set(draw.indices) != set(range(n)):
                    warnings.append(f"{sub.label}: a fat vertex is not used by the indices (G1-2)")
                if any(vertices[at + 15] for at in range(0, len(vertices), FAT_VERTEX)):
                    warnings.append(f"{sub.label}: fat normal byte 15 is not zero (G1-3)")
                sub.fat_draw = draw
                if not sub.draws:
                    sub.draws.append(draw)
            v_run += n
        i_run += count


def _cloth_checks(sub: SubObject, bone_count: int, warnings: list[str]) -> None:
    if sub.cloth is None:
        return
    if sub.fat_draw is not None:
        render, tolerance = sub.fat_draw.positions, (FAT_AXIS_TOLERANCE,) * 3
    else:
        packed = [d for d in sub.draws if d.kind == "packed" and not d.skinned]
        if not packed:
            return
        render, tolerance = packed[0].positions, tuple(abs(v) for v in sub.scale)
    warnings.extend(cloth_problems(sub.cloth, render, tolerance, bone_count, sub.label))


def ps3_cloth_draw_problems(sub: SubObject) -> list[str]:
    """V10-G1-3: a PS3 cloth sub-object's draw holds the fat positions, within one quantisation
    step per axis, in another vertex order (both ways)."""
    fat = sub.fat_draw
    draws = [d for d in sub.draws if d is not fat]
    if fat is None or not sub.has_cloth or len(draws) != 1:
        return []
    step = tuple(abs(v) for v in sub.scale)

    def near(a, b):
        return all(abs(a[k] - b[k]) <= step[k] * (1 + 1e-6) for k in range(3))

    drawn = draws[0].positions
    if all(any(near(p, q) for q in fat.positions) for p in drawn) and             all(any(near(q, p) for p in drawn) for q in fat.positions):
        return []
    return [f"{sub.label}: draw positions are not the fat positions within one step (V10-G1-3)"]


def parse_appearance(data: bytes, build: str | None = None, strict: bool = False) -> AppearanceFile:
    """Read an ``.app``. With ``strict``, data warnings become FormatError."""
    header = parse_header(data, build)
    resolved = resolve_build("app", header, build, strict)
    view = ByteView(data, build_order(resolved))
    bones = _read_bones(view, resolved)
    sub_count = view.i32(0x70, "sub-object count")
    subs_raw = _section(view, sub_count, view.pair(0x74), _SUB_RECORD[resolved], 0x74, "sub-objects")
    warnings: list[str] = []
    hardpoints = _read_hardpoints(view, len(bones), warnings)
    skipped: list[int] = []
    _check_bone_chain(bones, warnings)
    extra_count = view.i32(0x88, "low-poly sub-object count")
    extra_raw = _section(
        view, extra_count, view.pair(0x8C), _SUB_RECORD[resolved], 0x8C, "low-poly sub-objects",
    )
    notes: list[str] = []
    if resolved == "ps3":
        subs = _read_ps3_sub_objects(view, subs_raw, sub_count, len(bones), warnings, notes)
        extra = _read_ps3_sub_objects(
            view, extra_raw, extra_count, len(bones), warnings, notes, GEOSET_LOWPOLY,
        )
        kinds = {draw.skinned for sub in subs + extra for draw in sub.draws}
        if kinds == {True, False}:
            raise FormatError("rigid and skinned draws mixed in one file", 0x74)
        if True in kinds and not bones:
            raise FormatError("skinned draws in a file without bones", 0x24)
    else:
        subs, running = _read_switch_sub_objects(
            view, subs_raw, sub_count, len(bones), skipped, warnings, notes=notes, build=resolved,
        )
        extra, _ = _read_switch_sub_objects(
            view, extra_raw, extra_count, len(bones), skipped, warnings, running, notes,
            GEOSET_LOWPOLY, resolved,
        )
    _assign_fat(view, resolved, subs + extra, warnings)
    for sub in subs + extra:
        _cloth_checks(sub, len(bones), warnings)
        if resolved == "ps3":
            warnings.extend(ps3_cloth_draw_problems(sub))
        _check_bounds(sub, warnings, notes)
    material_data = AppMaterials()
    try:
        material_data = read_app_materials(view, resolved)
    except FormatError as exc:
        # A mesh stays usable without its materials.
        warnings.append(f"materials not read: {exc}")
    if strict and warnings:
        raise FormatError(warnings[0], 0x74)
    return AppearanceFile(
        header, resolved, bones, hardpoints, subs, skipped, warnings, material_data, extra,
        bytearray(data), notes,
    )


# --- Export --------------------------------------------------------------------------------

def _object_name(sub: SubObject) -> str:
    name = "".join(c if c.isalnum() or c in "_-." else "_" for c in sub.name or "")
    name = name or f"subobject_{sub.index}"
    return f"lowpoly_{name}" if sub.geo_set == GEOSET_LOWPOLY else name


REPEATED_NAME_SUFFIX = "__sub"  # V21-5


def export_object_names(app: AppearanceFile) -> dict[tuple[int, int], str]:
    """V21-5: the exported object name of every sub-object of the file, by (geo set, index).
    When an object name (``_object_name``, the ``lowpoly_`` prefix included) occurs more than
    once among all sub-objects of the file, every occurrence is written as
    ``<name>__sub<i>``, i the sub-object index in its geo set. Unique names stay as they are."""
    subs = list(app.sub_objects) + list(app.extra_sub_objects)
    counts: dict[str, int] = {}
    for sub in subs:
        counts[_object_name(sub)] = counts.get(_object_name(sub), 0) + 1
    out = {}
    for sub in subs:
        name = _object_name(sub)
        out[(sub.geo_set, sub.index)] = f"{name}{REPEATED_NAME_SUFFIX}{sub.index}" if counts[name] > 1 else name
    return out


def export_sub_objects(app: AppearanceFile, low_poly: bool = True) -> list[SubObject]:
    """G1-1: the high-poly set, then the low-poly set as separate objects (``low_poly`` False
    skips it). The two sets are never merged."""
    return list(app.sub_objects) + (list(app.extra_sub_objects) if low_poly else [])


def to_obj(app: AppearanceFile, flip_v: bool = False, low_poly: bool = True) -> str:
    """One object per sub-object: positions, normals, UV1. Low-poly objects are named
    ``lowpoly_...``. W16-5: each draw starts with ``g draw_<k>`` (k its index in the
    sub-object's draw list, counting the draws without positions, which are skipped). V21-5:
    repeated object names are written as ``<name>__sub<i>`` (``export_object_names``)."""
    lines = ["# d3sno appearance export"]
    base = 1
    names = export_object_names(app)
    for sub in export_sub_objects(app, low_poly):
        if not sub.draws:
            continue
        lines.append(f"o {names[(sub.geo_set, sub.index)]}")
        for k, draw in enumerate(sub.draws):
            if not draw.positions:
                continue
            lines.append(f"g draw_{k}")
            for x, y, z in draw.positions:
                lines.append(f"v {x:.6f} {y:.6f} {z:.6f}")
            for x, y, z in draw.normals:
                lines.append(f"vn {x:.6f} {y:.6f} {z:.6f}")
            for u, v in draw.uv1:
                lines.append(f"vt {u:.6f} {1 - v if flip_v else v:.6f}")
            ids = [i + base for i in draw.indices]
            for k in range(0, len(ids), 3):
                a, b, c = ids[k:k + 3]
                lines.append(f"f {a}/{a}/{a} {b}/{b}/{b} {c}/{c}/{c}")
            base += len(draw.positions)
    return "\n".join(lines) + "\n"


class _GltfBuffer:
    def __init__(self) -> None:
        self.blob = bytearray()
        self.views: list[dict] = []
        self.accessors: list[dict] = []

    def add(
        self, raw: bytes, component: int, kind: str, count: int, target: int | None = None,
        minimum: list[float] | None = None, maximum: list[float] | None = None,
    ) -> int:
        self.blob += bytes(-len(self.blob) % 4)
        view: dict = {"buffer": 0, "byteOffset": len(self.blob), "byteLength": len(raw)}
        if target is not None:
            view["target"] = target
        self.views.append(view)
        self.blob += raw
        accessor: dict = {
            "bufferView": len(self.views) - 1, "componentType": component,
            "count": count, "type": kind,
        }
        if minimum is not None:
            accessor["min"], accessor["max"] = minimum, maximum
        self.accessors.append(accessor)
        return len(self.accessors) - 1

    def add_view(self, raw: bytes) -> int:
        self.blob += bytes(-len(self.blob) % 4)
        self.views.append({"buffer": 0, "byteOffset": len(self.blob), "byteLength": len(raw)})
        self.blob += raw
        return len(self.views) - 1


_FLOAT = 5126
_USHORT = 5123
_UBYTE = 5121
_ARRAY = 34962
_REPEAT = 10497
_ELEMENTS = 34963


def _unit(normal: tuple[float, float, float]) -> tuple[float, float, float]:
    length = math.sqrt(sum(c * c for c in normal))
    return tuple(c / length for c in normal) if length else (0.0, 0.0, 1.0)


def _flat(rows) -> list:
    return [value for row in rows for value in row]


def _gltf_primitive(
    buf: _GltfBuffer, draw: Draw, flip_v: bool, lightmap_v_scale: float | None,
) -> dict:
    count = len(draw.positions)
    positions = struct.pack(f"<{count * 3}f", *_flat(draw.positions))
    stored = struct.unpack(f"<{count * 3}f", positions)
    low = [min(stored[axis::3]) for axis in range(3)]
    high = [max(stored[axis::3]) for axis in range(3)]
    uv = [(u, 1 - v if flip_v else v) for u, v in draw.uv1]
    attributes = {
        "POSITION": buf.add(positions, _FLOAT, "VEC3", count, _ARRAY, low, high),
        "NORMAL": buf.add(
            struct.pack(f"<{count * 3}f", *_flat(map(_unit, draw.normals))),
            _FLOAT, "VEC3", count, _ARRAY,
        ),
        "TEXCOORD_0": buf.add(
            struct.pack(f"<{count * 2}f", *_flat(uv)), _FLOAT, "VEC2", count, _ARRAY,
        ),
    }
    if lightmap_v_scale is not None:
        attributes["TEXCOORD_1"] = buf.add(
            struct.pack(f"<{count * 2}f", *_flat(draw.uv2(lightmap_v_scale))),
            _FLOAT, "VEC2", count, _ARRAY,
        )
    # W16-5: the stored colours and UV2 as application-specific attributes.
    for name, colours in (("_D3_COLOR1", draw.colours_1), ("_D3_COLOR2", draw.colours_2)):
        if colours:
            attributes[name] = buf.add(b"".join(bytes(c) for c in colours), _UBYTE, "VEC4", count, _ARRAY)
    attributes["_D3_UV2"] = buf.add(
        struct.pack(f"<{count * 2}H", *_flat(draw.uv2_raw)), _USHORT, "VEC2", count, _ARRAY,
    )
    primitive: dict = {"attributes": attributes, "mode": 4}
    if draw.indices:
        primitive["indices"] = buf.add(
            struct.pack(f"<{len(draw.indices)}H", *draw.indices),
            _USHORT, "SCALAR", len(draw.indices), _ELEMENTS,
        )
    if draw.joints is None:
        return primitive
    weights = []
    for row in draw.weights:
        total = sum(row)
        weights.extend([w / total for w in row] if total else [1.0, 0.0, 0.0, 0.0])
    attributes["JOINTS_0"] = buf.add(
        struct.pack(f"<{count * 4}H", *_flat(draw.joints)), _USHORT, "VEC4", count, _ARRAY,
    )
    attributes["WEIGHTS_0"] = buf.add(
        struct.pack(f"<{count * 4}f", *weights), _FLOAT, "VEC4", count, _ARRAY,
    )
    return primitive


class _GltfMaterials:
    """One glTF texture per texture handle, made from what the resolver returns (called once
    per handle). W16-5: every primitive gets a material of its own kind, named
    ``texture_<handle>_draw_<k>`` with a texture, else ``d3sno_draw_<k>``; materials with the same
    name are shared."""

    def __init__(self, buf: _GltfBuffer, resolver) -> None:
        self.buf = buf
        self.resolver = resolver
        self.images: list[dict] = []
        self.textures: list[dict] = []
        self.materials: list[dict] = []
        self.by_handle: dict[int, int | None] = {}
        self.by_name: dict[str, int] = {}

    def texture(self, handle: int | None) -> int | None:
        if self.resolver is None or handle is None or handle == -1:
            return None
        if handle not in self.by_handle:
            self.by_handle[handle] = self._make(handle)
        return self.by_handle[handle]

    def material(self, handle: int | None, draw: int) -> int:
        texture = self.texture(handle)
        name = f"texture_{handle}_draw_{draw}" if texture is not None else f"d3sno_draw_{draw}"
        if name not in self.by_name:
            material: dict = {"name": name}
            if texture is not None:
                material["pbrMetallicRoughness"] = {
                    "baseColorTexture": {"index": texture},
                    "metallicFactor": 0.0,
                }
            self.by_name[name] = len(self.materials)
            self.materials.append(material)
        return self.by_name[name]

    def _make(self, handle: int) -> int | None:
        target = self.resolver(handle)
        if target is None:
            return None
        if isinstance(target, str):
            image: dict = {"uri": target}
        else:
            image = {"bufferView": self.buf.add_view(bytes(target)), "mimeType": "image/png"}
        self.images.append(image)
        self.textures.append({"source": len(self.images) - 1, "sampler": 0})
        return len(self.textures) - 1

    def write(self, document: dict) -> None:
        if self.materials:
            document["materials"] = self.materials
        if self.textures:
            document["textures"] = self.textures
            document["images"] = self.images
            document["samplers"] = [{"wrapS": _REPEAT, "wrapT": _REPEAT}]


def _bone_node(bone: Bone) -> dict:
    local = bone.local
    check_float32((*local.q, *local.t, local.s), f"bone {bone.index} transform")
    return {
        "name": f"bone_{bone.index}_{bone.name_hash:08X}",
        "rotation": list(quat_normalise(local.q)),
        "translation": list(local.t),
        "scale": [local.s] * 3,
    }


SKIN_ROOT_NAME = "d3sno_skin_root"


def skeleton_nodes(app: AppearanceFile, skinned: bool) -> tuple[list[dict], list[int], int | None]:
    """The bone nodes (node i is bone i), the scene's root nodes and the added skin root.

    ``41`` V26-2: when a skin's joints have more than one root bone, one node without a transform
    (``d3sno_skin_root``, ``extras.d3sno_export.added``, not a joint) is added right after the
    bones, above the root bones; it is the skin's ``skeleton``. One root: no node is added."""
    nodes: list[dict] = []
    roots: list[int] = []
    for bone in app.bones:
        if bone.parent >= 0:
            nodes[bone.parent].setdefault("children", []).append(bone.index)
        else:
            roots.append(bone.index)
        nodes.append(_bone_node(bone))
    if not skinned or len(roots) < 2:
        return nodes, roots, None
    added = len(nodes)
    nodes.append({"name": SKIN_ROOT_NAME, "children": roots, "extras": {"d3sno_export": {"added": True}}})
    return nodes, [added], added


def to_gltf(
    app: AppearanceFile, flip_v: bool = False, lightmap_v_scale: float | None = None,
    extend=None, texture_resolver=None, look: int = 0, low_poly: bool = True,
) -> dict:
    """A glTF 2.0 document with one embedded buffer. Bones are nodes.

    ``lightmap_v_scale`` adds UV2 as TEXCOORD_1; pass 0 for a lightmap with no V scale.
    ``extend(buffer, document)`` runs before the buffer is sealed.
    ``texture_resolver(handle)`` gives a URI, PNG bytes to embed, or None. With it, each
    sub-object's materials use its diffuse texture for ``look``.

    W16-5: meshes carry ``extras.d3sno`` (geo set, sub-object index), primitives
    ``extras.d3sno.draw`` and a material per draw (``texture_<handle>_draw_<k>`` or
    ``d3sno_draw_<k>``), the skin ``extras.d3sno.bones``; colours are written as ``_D3_COLOR1`` /
    ``_D3_COLOR2`` (u8 VEC4) and UV2 as ``_D3_UV2`` (u16 VEC2), as stored.
    V21-4: a draw with vertices and no indices is a primitive without indices with
    ``extras.d3sno.no_indices`` true. V21-5: repeated object names (mesh and node) are written as
    ``<name>__sub<i>``. W17-12: mesh ``extras.d3sno`` also holds ``name`` (the ps3 sub-object name;
    on switch / x360, which store none, the object name) and ``material`` (the material hash).
    """
    buf = _GltfBuffer()
    materials = _GltfMaterials(buf, texture_resolver)
    skinned = any(d.joints is not None for sub in export_sub_objects(app, low_poly) for d in sub.draws)
    nodes, roots, skin_root = skeleton_nodes(app, skinned)
    skins = []
    if skinned:
        matrices = bytearray()
        for bone in app.bones:
            matrix = bone.skin_inverse.matrix()
            check_float32(matrix, f"bone {bone.index} inverse bind matrix")
            matrices += struct.pack("<16f", *matrix)
        skin: dict = {
            "joints": list(range(len(app.bones))),
            "inverseBindMatrices": buf.add(bytes(matrices), _FLOAT, "MAT4", len(app.bones)),
        }
        if skin_root is not None:
            skin["skeleton"] = skin_root  # ``41`` V26-2
        extras: dict = {"bind_from_rest_inverse": True} if app.build == "ps3" else {}
        extras["d3sno"] = {"bones": len(app.bones)}  # W16-5
        skin["extras"] = extras
        skins.append(skin)

    meshes = []
    names = export_object_names(app)
    for sub in export_sub_objects(app, low_poly):
        handle = sub_object_texture(app, sub, look) if texture_resolver is not None else None
        primitives = []
        for k, draw in enumerate(sub.draws):
            if not draw.positions:
                continue  # skipped, but counted in k (W16-5)
            primitive = _gltf_primitive(buf, draw, flip_v, lightmap_v_scale)
            primitive["material"] = materials.material(handle, k)
            info: dict = {"draw": k}
            if not draw.indices:
                info["no_indices"] = True  # V21-4: vertices without triangles
            primitive["extras"] = {"d3sno": info}
            primitives.append(primitive)
        if not primitives:
            continue
        name = names[(sub.geo_set, sub.index)]
        node = {"name": name, "mesh": len(meshes)}
        if skinned and any("JOINTS_0" in p["attributes"] for p in primitives):
            node["skin"] = 0
        # W17-12: the sub-object's name (switch / x360 store none: the object name) and material
        mesh_info: dict = {"geo_set": sub.geo_set, "sub_object": sub.index,
                           "name": sub.name if app.build == "ps3" and sub.name else name}
        if sub.material_hash is not None:
            mesh_info["material"] = sub.material_hash
        meshes.append({"name": name, "primitives": primitives, "extras": {"d3sno": mesh_info}})
        roots.append(len(nodes))
        nodes.append(node)

    document: dict = {
        "asset": {"version": "2.0", "generator": "d3sno"},
        "scene": 0,
        "scenes": [{"nodes": roots}],
    }
    if nodes:
        document["nodes"] = nodes
    if meshes:
        document["meshes"] = meshes
    if skins:
        document["skins"] = skins
    materials.write(document)
    if extend is not None:
        extend(buf, document)
    if buf.blob:
        encoded = base64.b64encode(bytes(buf.blob)).decode("ascii")
        document["buffers"] = [{
            "byteLength": len(buf.blob),
            "uri": "data:application/octet-stream;base64," + encoded,
        }]
        document["bufferViews"] = buf.views
        document["accessors"] = buf.accessors
    return document


def to_gltf_skeleton(app: AppearanceFile, extend=None, low_poly: bool = True) -> dict:
    """``41`` V26-3: the joint nodes of ``to_gltf`` only (same names, order and V26-2 root node),
    no mesh, skin or images. ``extend(buffer, document)`` runs before the buffer is sealed."""
    buf = _GltfBuffer()
    skinned = any(d.joints is not None for sub in export_sub_objects(app, low_poly) for d in sub.draws)
    nodes, roots, _ = skeleton_nodes(app, skinned)
    document: dict = {"asset": {"version": "2.0", "generator": "d3sno"}, "scene": 0, "scenes": [{"nodes": roots}]}
    if nodes:
        document["nodes"] = nodes
    if extend is not None:
        extend(buf, document)
    if buf.blob:
        encoded = base64.b64encode(bytes(buf.blob)).decode("ascii")
        document["buffers"] = [{"byteLength": len(buf.blob),
                                "uri": "data:application/octet-stream;base64," + encoded}]
        document["bufferViews"] = buf.views
        document["accessors"] = buf.accessors
    return document


def gltf_dumps(document: dict) -> str:
    return json.dumps(document, separators=(",", ":"), allow_nan=False)


def to_gltf_json(
    app: AppearanceFile, flip_v: bool = False, lightmap_v_scale: float | None = None,
    texture_resolver=None, look: int = 0, low_poly: bool = True,
) -> str:
    return gltf_dumps(to_gltf(
        app, flip_v, lightmap_v_scale, texture_resolver=texture_resolver, look=look,
        low_poly=low_poly,
    ))
