"""``.app`` files for both builds, with Wavefront OBJ and glTF 2.0 export."""

from __future__ import annotations

import base64
import json
import math
import struct
from dataclasses import dataclass, field

from .binary import ByteView, FormatError, Pair, build_order
from .material import AppMaterials, read_app_materials, sub_object_texture
from .sno import SnoHeader, parse_header, resolve_build
from .transform import Transform, quat_normalise, read_transform, transforms_match

_BONE_RECORD = {"ps3": 176, "switch": 256}
_SUB_RECORD = {"ps3": 308, "switch": 240}
_BONE_PARENT = {"ps3": 4, "switch": 8}
# Positions of the rest transform, its inverse and the parent-relative transform.
_BONE_TRANSFORMS = {"ps3": (0x30, 0x50, 0x70), "switch": (0x34, 0x54, 0x74)}
_BONE_BIND = 0x94
_BONE_BIND_INVERSE = 0xB4
BONE_RANGE_UNIT = 48
HARDPOINT_RECORD = 40
DRAW_RECORD = 160
SKIN_RECORD = 104
FORMAT_RIGID = 0x81
FORMAT_SKINNED = 0x83
SWITCH_INDEX_POOL_AT = 0x220
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
    extra_sub_objects: list[SubObject] = field(default_factory=list)  # low-polygon meshes


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
        if build == "switch":
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

    draw = Draw(skinned, [], [], [], [], [], [], indices)
    warning = None
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
                warning = f"sub-object {sub.index}: weights not descending or not summing to 255"
        else:
            draw.normals.append(_normal(*c_raw[c_at:c_at + 3]))
            draw.colours_1.append(c_raw[c_at + 4:c_at + 8])
            draw.colours_2.append(c_raw[c_at + 8:c_at + 12])
            uv_at = c_at + 12
        u1, v1, u2, v2 = struct.unpack_from("<4H", c_raw, uv_at)
        draw.uv1.append(_uv1(u1, v1))
        draw.uv2_raw.append((u2, v2))
    return draw, warning


def _check_bounds(sub: SubObject, warnings: list[str]) -> None:
    tolerance = [2 * abs(s) for s in sub.scale]
    for draw in sub.draws:
        for position in draw.positions:
            for axis in range(3):
                reach = sub.bounds_half[axis] + tolerance[axis]
                if abs(position[axis] - sub.bounds_centre[axis]) > reach:
                    warnings.append(f"sub-object {sub.index}: vertex outside bounds")
                    return


def _read_ps3_sub_objects(
    view: ByteView, raw: bytes, count: int, bone_count: int, warnings: list[str],
) -> list[SubObject]:
    size = _SUB_RECORD["ps3"]
    subs = []
    for i in range(count):
        record = ByteView(raw[i * size:(i + 1) * size], view.order)
        sub = SubObject(
            i, record.cstring(0x48, 128) or None, None, record.f32s(0x118, 3),
            record.f32s(0x124, 3), record.f32s(0xD0, 3), record.f32s(0xDC, 3), [],
            material_hash=record.u32(0xC8), sub_hash=record.u32(0xCC),
        )
        draw_count = record.i32(0xFC)
        draw_pair = record.pair(0x100)
        draws_raw = _section(view, draw_count, draw_pair, DRAW_RECORD, 0x78, "draw array")
        for k in range(draw_count):
            base = draw_pair.file_offset + k * DRAW_RECORD
            draw, warning = _read_ps3_draw(
                view, draws_raw[k * DRAW_RECORD:(k + 1) * DRAW_RECORD], sub, base, bone_count,
            )
            sub.draws.append(draw)
            if warning:
                warnings.append(warning)
        _check_bounds(sub, warnings)
        subs.append(sub)
    return subs


def _switch_vertices(
    raw: bytes, count: int, stride: int, sub: SubObject,
) -> Draw:
    draw = Draw(stride == 36, [], [], [], [], [], [], [])
    for i in range(count):
        at = i * stride
        raw_pos = struct.unpack_from("<3h", raw, at)
        draw.positions.append(_position(raw_pos, sub.scale, sub.bias))
        draw.normals.append(_normal(*raw[at + 8:at + 11]))
        draw.colours_1.append(raw[at + 12:at + 16])
        draw.colours_2.append(raw[at + 16:at + 20])
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
    palette = list(struct.unpack_from(f"<{palette_count}i", palette_raw, 0))
    vertex_count = r.i32(0x18)
    vertices = _section(view, vertex_count, r.pair(0x28), 36, base + 0x28, "skinned vertices")
    index_count = r.i32(0x30)
    index_raw = view.pair_data(r.pair(0x40), base + 0x40, "skinned indices")
    if index_count < 0 or len(index_raw) < index_count * 2:
        raise FormatError(f"skinned index pair too small for {index_count}", base + 0x40)

    draw = _switch_vertices(vertices, vertex_count, 36, sub)
    draw.indices = _read_indices(index_raw, "<", index_count, vertex_count, base + 0x30)
    draw.joints, draw.weights, draw.palette = [], [], palette
    for i in range(vertex_count):
        at = i * 36 + 28
        weights = tuple(vertices[at + 4:at + 8])
        joints = _translate_joints(vertices[at:at + 4], weights, palette, base)
        for joint, weight in zip(joints, weights):
            if weight and not 0 <= joint < bone_count:
                raise FormatError(f"palette entry {joint} is not a bone", base)
        draw.joints.append(joints)
        draw.weights.append(weights)
    return draw


def _read_switch_sub_objects(
    view: ByteView, raw: bytes, count: int, bone_count: int,
    skipped: list[int], warnings: list[str], running: int = 0,
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
        )
        index_count = record.i32(0x28)
        vertex_count = record.i32(0xA0)
        if index_count < 0 or vertex_count < 0:
            raise FormatError(f"sub-object {i}: negative count", 0x78)
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
                raise FormatError(f"sub-object {i}: indices past the pool", SWITCH_INDEX_POOL_AT)
            draw = _switch_vertices(vertices, vertex_count, 28, sub)
            draw.indices = _read_indices(
                pool[start * 2:], view.order, index_count, vertex_count, SWITCH_INDEX_POOL_AT,
            )
            sub.draws.append(draw)
        elif index_count and not skin_raw:
            skipped.append(i)
            warnings.append(f"sub-object {i}: {index_count} indices but no vertices, skipped")
        for k in range(len(skin_raw) // SKIN_RECORD):
            sub.draws.append(_read_switch_skin(
                view, skin_raw[k * SKIN_RECORD:(k + 1) * SKIN_RECORD], sub, bone_count,
                skin_pair.file_offset + k * SKIN_RECORD,
            ))
        influence = view.pair_data(record.pair(0xC8), 0x78, "influence stream")
        if len(influence) % 6:
            raise FormatError(f"influence stream size {len(influence)}", 0x78)
        for at in range(0, len(influence), 6):
            sub.influences.append(tuple(
                (influence[at + n], influence[at + n + 1]) for n in (0, 2, 4)
            ))
        _check_bounds(sub, warnings)
        subs.append(sub)
    return subs, running


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
    extra_count = view.i32(0x88, "second sub-object count")
    extra_raw = _section(
        view, extra_count, view.pair(0x8C), _SUB_RECORD[resolved], 0x8C, "second sub-object list",
    )
    if resolved == "ps3":
        subs = _read_ps3_sub_objects(view, subs_raw, sub_count, len(bones), warnings)
        extra = _read_ps3_sub_objects(view, extra_raw, extra_count, len(bones), warnings)
        kinds = {draw.skinned for sub in subs + extra for draw in sub.draws}
        if kinds == {True, False}:
            raise FormatError("rigid and skinned draws mixed in one file", 0x74)
        if True in kinds and not bones:
            raise FormatError("skinned draws in a file without bones", 0x24)
    else:
        subs, running = _read_switch_sub_objects(
            view, subs_raw, sub_count, len(bones), skipped, warnings,
        )
        extra, _ = _read_switch_sub_objects(
            view, extra_raw, extra_count, len(bones), skipped, warnings, running,
        )
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
    )


# --- Export --------------------------------------------------------------------------------

def _object_name(sub: SubObject) -> str:
    name = "".join(c if c.isalnum() or c in "_-." else "_" for c in sub.name or "")
    return name or f"subobject_{sub.index}"


def to_obj(app: AppearanceFile, flip_v: bool = False) -> str:
    """One object per sub-object: positions, normals, UV1."""
    lines = ["# d3sno appearance export"]
    base = 1
    for sub in app.sub_objects:
        if not sub.draws:
            continue
        lines.append(f"o {_object_name(sub)}")
        for draw in sub.draws:
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
    """One glTF material per texture handle, made from what the resolver returns."""

    def __init__(self, buf: _GltfBuffer, resolver) -> None:
        self.buf = buf
        self.resolver = resolver
        self.images: list[dict] = []
        self.textures: list[dict] = []
        self.materials: list[dict] = []
        self.by_handle: dict[int, int | None] = {}

    def index(self, handle: int | None) -> int | None:
        if handle is None or handle == -1:
            return None
        if handle not in self.by_handle:
            self.by_handle[handle] = self._make(handle)
        return self.by_handle[handle]

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
        self.materials.append({
            "name": f"texture_{handle}",
            "pbrMetallicRoughness": {
                "baseColorTexture": {"index": len(self.textures) - 1},
                "metallicFactor": 0.0,
            },
        })
        return len(self.materials) - 1

    def write(self, document: dict) -> None:
        if self.materials:
            document["materials"] = self.materials
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


def to_gltf(
    app: AppearanceFile, flip_v: bool = False, lightmap_v_scale: float | None = None,
    extend=None, texture_resolver=None, look: int = 0,
) -> dict:
    """A glTF 2.0 document with one embedded buffer. Bones are nodes.

    ``lightmap_v_scale`` adds UV2 as TEXCOORD_1; pass 0 for a lightmap with no V scale.
    ``extend(buffer, document)`` runs before the buffer is sealed.
    ``texture_resolver(handle)`` gives a URI, PNG bytes to embed, or None. With it, each
    sub-object gets a material from its diffuse texture for ``look``.
    """
    buf = _GltfBuffer()
    materials = _GltfMaterials(buf, texture_resolver)
    nodes: list[dict] = []
    roots: list[int] = []
    for bone in app.bones:
        if bone.parent >= 0:
            nodes[bone.parent].setdefault("children", []).append(bone.index)
        else:
            roots.append(bone.index)
        nodes.append(_bone_node(bone))

    skinned = any(d.joints is not None for sub in app.sub_objects for d in sub.draws)
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
        if app.build == "ps3":
            skin["extras"] = {"bind_from_rest_inverse": True}
        skins.append(skin)

    meshes = []
    for sub in app.sub_objects:
        primitives = [
            _gltf_primitive(buf, draw, flip_v, lightmap_v_scale)
            for draw in sub.draws if draw.positions
        ]
        if not primitives:
            continue
        if texture_resolver is not None:
            material = materials.index(sub_object_texture(app, sub, look))
            if material is not None:
                for primitive in primitives:
                    primitive["material"] = material
        node = {"name": _object_name(sub), "mesh": len(meshes)}
        if skinned and any("JOINTS_0" in p["attributes"] for p in primitives):
            node["skin"] = 0
        meshes.append({"name": _object_name(sub), "primitives": primitives})
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


def gltf_dumps(document: dict) -> str:
    return json.dumps(document, separators=(",", ":"), allow_nan=False)


def to_gltf_json(
    app: AppearanceFile, flip_v: bool = False, lightmap_v_scale: float | None = None,
    texture_resolver=None, look: int = 0,
) -> str:
    return gltf_dumps(to_gltf(
        app, flip_v, lightmap_v_scale, texture_resolver=texture_resolver, look=look,
    ))
