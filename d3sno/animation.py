"""Animation: ``.ani``, ps3 ``.can``, curve decoding, sampling, pose and glTF channels."""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field

from .appearance import AppearanceFile, check_float32, gltf_dumps, to_gltf
from .binary import ByteView, FormatError, Pair, build_order, to_f32
from .sno import SnoHeader, parse_header, resolve_build
from .transform import Quaternion, Transform, Vector, quat_normalise

PERMUTATION_SIZE = 360
ATTACHMENT_SIZE = 192
CURVE_ENTRY_SIZE = 16
BLEND_WEIGHT_SIZE = 8
NONLINEAR_SIZE = 12
_TABLE_AT = {"ps3": 0x40, "switch": 0x38}
_PS3_ANI_HEAD = 0x50
_ROTATION_UNIT = to_f32(1 / 32767)
MODE_STOP = 0
MODE_WRAP = 2
_EVEN_FRAME_FPS = 20
_FRAME_FORMATS = {1: "B", 2: "H", 4: "I"}
SCALE_MIN = 0.01
SCALE_MAX = 100.0
IDENTITY_ROTATION = (0.0, 0.0, 0.0, 1.0)


@dataclass
class Permutation:
    build: str
    flags: int
    name: str
    frame_rate: float
    tool_values: tuple[float, ...]
    blend_times: tuple[int, int]
    weight: int
    speed_min: float
    speed_delta: float
    ragdoll_factors: tuple[float, float]
    bone_count: int
    blend_weight_count: int
    keyframes: int
    contact_keyframes: tuple[float, ...]
    contact_offsets: tuple[tuple[float, float, float], ...]
    earliest_interrupt: float
    attachment_count: int
    average_velocity: tuple[float, float, float]
    hardpoint_link: int
    raw: bytes
    has_data: bool = False  # False on ps3 until the .can is merged
    track_hashes: list[int] = field(default_factory=list)
    blend_weights: list[tuple[float, float]] = field(default_factory=list)
    translation_blobs: list[bytes] = field(default_factory=list)
    rotation_blobs: list[bytes] = field(default_factory=list)
    scale_blobs: list[bytes] = field(default_factory=list)
    attachments: list[bytes] = field(default_factory=list)
    nonlinear_offsets: list[tuple[float, float, float]] = field(default_factory=list)

    @property
    def fps(self) -> float:
        return self.frame_rate * 60

    @property
    def duration(self) -> float:
        """Seconds, ``(K - 1) / fps``. Zero if the frame rate is unusable."""
        return (self.keyframes - 1) / self.fps if self.fps > 0 else 0.0


@dataclass
class AniFile:
    header: SnoHeader
    build: str
    flags: int
    playback_mode: int
    appearance_handle: int
    table_pair: Pair
    permutations: list[Permutation]
    warnings: list[str] = field(default_factory=list)


def _read_permutation(raw: bytes, order: str, build: str) -> Permutation:
    r = ByteView(raw, order)
    return Permutation(
        build, r.u32(0), r.cstring(0x04, 68), r.f32(0x48), r.f32s(0x4C, 4), r.unpack("II", 0x5C),
        r.u32(0x64), r.f32(0x68), r.f32(0x6C), r.f32s(0x70, 2), r.u32(0x78), r.u32(0x90),
        r.u32(0xA8), r.f32s(0xE0, 4),
        tuple(r.f32s(0xF0 + i * 12, 3) for i in range(4)),
        r.f32(0x120), r.u32(0x12C), r.f32s(0x148, 3), r.u32(0x154), raw,
    )


def _sized(view: ByteView, record: ByteView, at: int, base: int, size: int, what: str) -> bytes:
    pair = record.pair(at, what)
    if pair.size != size:
        raise FormatError(f"{what}: pair size {pair.size}, expected {size}", base + at)
    return view.pair_data(pair, base + at, what)


def _read_rows(raw: bytes, order: str, row: str, width: int) -> list[tuple]:
    view = ByteView(raw, order)
    return [view.unpack(row, pos) for pos in range(0, len(raw), width)]


def _split(raw: bytes, width: int) -> list[bytes]:
    return [raw[pos:pos + width] for pos in range(0, len(raw), width)]


def _read_switch_data(view: ByteView, perm: Permutation, base: int) -> None:
    r = ByteView(perm.raw, view.order)
    bones = perm.bone_count
    if perm.blend_weight_count not in (0, bones):
        raise FormatError(f"blend weight count {perm.blend_weight_count}", base + 0x90)
    perm.track_hashes = [
        row[0] for row in _read_rows(_sized(view, r, 0x7C, base, 4 * bones, "bone hashes"),
                                     view.order, "I", 4)
    ]
    blend = _sized(view, r, 0x98, base, BLEND_WEIGHT_SIZE * perm.blend_weight_count, "blend weights")
    perm.blend_weights = _read_rows(blend, view.order, "2f", BLEND_WEIGHT_SIZE)
    tables = []
    for at, name in ((0xAC, "translation"), (0xC0, "rotation"), (0xD0, "scale")):
        if r.pair(at).size == 0:
            tables.append([b""] * bones)  # no table at all
            continue
        table = _sized(view, r, at, base, CURVE_ENTRY_SIZE * bones, f"{name} table")
        entries = ByteView(table, view.order)
        blobs = []
        for track in range(bones):
            pair = entries.pair(track * CURVE_ENTRY_SIZE)
            blobs.append(view.pair_data(pair, base + at, f"{name} blob {track}"))
        tables.append(blobs)
    perm.translation_blobs, perm.rotation_blobs, perm.scale_blobs = tables
    attachments = _sized(
        view, r, 0x124, base, ATTACHMENT_SIZE * perm.attachment_count, "attachments",
    )
    perm.attachments = _split(attachments, ATTACHMENT_SIZE)
    nonlinear = _sized(view, r, 0x138, base, NONLINEAR_SIZE * perm.keyframes, "nonlinear offsets")
    perm.nonlinear_offsets = _read_rows(nonlinear, view.order, "3f", NONLINEAR_SIZE)
    perm.has_data = True


def parse_ani(
    data: bytes, build: str | None = None, can: bytes | None = None, strict: bool = False,
) -> AniFile:
    """Read an ``.ani``. On ps3, pass the ``.can`` bytes to merge the animation data."""
    header = parse_header(data, build)
    resolved = resolve_build("ani", header, build, strict)
    view = ByteView(data, build_order(resolved))
    table_pair = view.pair(0x28, "permutation table pair")
    warnings = []
    if resolved == "ps3":
        count, excess = divmod(len(data) - _PS3_ANI_HEAD, PERMUTATION_SIZE)
        if count < 0 or excess:
            raise FormatError(f"ps3 .ani of {len(data)} bytes is not 0x50 + 360 * n", 0)
        if table_pair.size not in (0, count * PERMUTATION_SIZE):
            warnings.append(f"permutation table size {table_pair.size} for {count} records")
        table_at = 0x10 + _TABLE_AT["ps3"]
    else:
        count, excess = divmod(table_pair.size, PERMUTATION_SIZE)
        if excess:
            raise FormatError(f"permutation table size {table_pair.size}", 0x2C)
        table_at = table_pair.file_offset
        if table_pair.offset != _TABLE_AT["switch"]:
            warnings.append(f"permutation table at body 0x{table_pair.offset:X}")
    view.need(table_at, count * PERMUTATION_SIZE, "permutation table")
    permutations = []
    for i in range(count):
        base = table_at + i * PERMUTATION_SIZE
        perm = _read_permutation(view.take(base, PERMUTATION_SIZE), view.order, resolved)
        if resolved == "switch":
            _read_switch_data(view, perm, base)
        permutations.append(perm)
    if strict and warnings:
        raise FormatError(warnings[0], 0x28)
    ani = AniFile(
        header, resolved, view.u32(0x1C), view.u32(0x20), view.i32(0x24), table_pair,
        permutations, warnings,
    )
    if can is not None:
        merge_can(ani, parse_can(can))
    return ani


# --- .can ----------------------------------------------------------------------------------

@dataclass
class CanPermutation:
    track_hashes: list[int]
    blend_weights: list[tuple[float, float]]
    translation_blobs: list[bytes]
    rotation_blobs: list[bytes]
    scale_blobs: list[bytes]
    attachments: list[bytes]
    nonlinear_offsets: list[tuple[float, float, float]]


@dataclass
class CanFile:
    permutations: list[CanPermutation]


def _read_can_section(view: ByteView, pos: int, width: int, what: str) -> tuple[bytes, int]:
    size = view.u32(pos, f"{what} size")
    if size % width:
        raise FormatError(f"{what}: size {size} is not a multiple of {width}", pos)
    return view.take(pos + 4, size, what), pos + 4 + size


def _read_can_curves(view: ByteView, pos: int, tracks: int, what: str) -> tuple[list[bytes], int]:
    table_raw, table_end = _read_can_section(view, pos, CURVE_ENTRY_SIZE, what)
    if not table_raw:
        return [b""] * tracks, table_end  # no table at all
    if len(table_raw) != CURVE_ENTRY_SIZE * tracks:
        raise FormatError(f"{what}: {len(table_raw)} table bytes for {tracks} tracks", pos)
    table = pos + 4
    entries = ByteView(table_raw, ">")
    cursor = len(table_raw)
    blobs = []
    for track in range(tracks):
        offset, size, zero_1, zero_2 = entries.unpack("4I", track * CURVE_ENTRY_SIZE)
        if zero_1 or zero_2:
            raise FormatError(f"{what}: entry {track} has non-zero reserved words", table)
        if size == 0:
            blobs.append(b"")
            continue
        expected = (cursor + 7) & ~7
        if offset != expected:
            raise FormatError(
                f"{what}: blob {track} at {offset}, expected {expected}",
                table + track * CURVE_ENTRY_SIZE,
            )
        blobs.append(view.take(table + offset, size, f"{what} blob {track}"))
        cursor = offset + size
    return blobs, table + cursor


def parse_can(data: bytes) -> CanFile:
    """Walk a ps3 ``.can``: permutations in order, seven sections each, to the end of file."""
    view = ByteView(data, ">")
    permutations = []
    pos = 0
    while pos < len(data):
        hashes, pos = _read_can_section(view, pos, 4, "bone hashes")
        tracks = len(hashes) // 4
        blend, pos = _read_can_section(view, pos, BLEND_WEIGHT_SIZE, "blend weights")
        translation, pos = _read_can_curves(view, pos, tracks, "translation")
        rotation, pos = _read_can_curves(view, pos, tracks, "rotation")
        scale, pos = _read_can_curves(view, pos, tracks, "scale")
        attachments, pos = _read_can_section(view, pos, ATTACHMENT_SIZE, "attachments")
        nonlinear, pos = _read_can_section(view, pos, NONLINEAR_SIZE, "nonlinear offsets")
        permutations.append(CanPermutation(
            [row[0] for row in _read_rows(hashes, ">", "I", 4)],
            _read_rows(blend, ">", "2f", BLEND_WEIGHT_SIZE),
            translation, rotation, scale, _split(attachments, ATTACHMENT_SIZE),
            _read_rows(nonlinear, ">", "3f", NONLINEAR_SIZE),
        ))
    return CanFile(permutations)


def merge_can(ani: AniFile, can: CanFile) -> None:
    """Attach ``.can`` data to a ps3 ``.ani``. The counts in both must agree."""
    if len(can.permutations) != len(ani.permutations):
        raise FormatError(
            f".can has {len(can.permutations)} permutations, .ani {len(ani.permutations)}",
        )
    for perm, part in zip(ani.permutations, can.permutations):
        found = (
            len(part.track_hashes), len(part.blend_weights), len(part.attachments),
            len(part.nonlinear_offsets),
        )
        wanted = (
            perm.bone_count, perm.blend_weight_count, perm.attachment_count, perm.keyframes,
        )
        if found != wanted:
            raise FormatError(f".can counts {found} differ from .ani counts {wanted}")
        perm.track_hashes = part.track_hashes
        perm.blend_weights = part.blend_weights
        perm.translation_blobs = part.translation_blobs
        perm.rotation_blobs = part.rotation_blobs
        perm.scale_blobs = part.scale_blobs
        perm.attachments = part.attachments
        perm.nonlinear_offsets = part.nonlinear_offsets
        perm.has_data = True


# --- Curves --------------------------------------------------------------------------------

@dataclass
class Curve:
    frames: list[int]
    values: list[tuple[float, ...]]
    codec: int | None = None
    base: Vector = (0.0, 0.0, 0.0)


@dataclass
class DecodedPermutation:
    translation: list[Curve]
    rotation: list[Curve]
    scale: list[Curve]


def _key_width(keyframes: int) -> int:
    if keyframes < 256:
        return 1
    return 2 if keyframes <= 65535 else 4


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _read_frames(
    view: ByteView, pos: int, count: int, width: int, keyframes: int,
) -> list[int]:
    frames = list(view.unpack(f"{count}{_FRAME_FORMATS[width]}", pos, "curve frames"))
    if frames[0] != 0:
        raise FormatError(f"first key is at frame {frames[0]}", pos)
    if any(a >= b for a, b in zip(frames, frames[1:])):
        raise FormatError("key frames are not strictly ascending", pos)
    if count >= 2 and frames[-1] != keyframes - 1:
        raise FormatError(f"last key at {frames[-1]}, expected {keyframes - 1}", pos)
    return frames


def _key_count(view: ByteView, width: int) -> int:
    count = view.unpack(_FRAME_FORMATS[width], 0, "key count")[0]
    if count == 0:
        raise FormatError("non-empty curve with no keys", 0)
    return count


def _check_blob_end(blob: bytes, needed: int) -> None:
    if not needed <= len(blob) <= _align(needed, 4):
        raise FormatError(f"curve blob of {len(blob)} bytes, needed {needed}", needed)


def _read_components(view: ByteView, pos: int, count: int, codec: int) -> tuple:
    code = {1: "B", 2: "H", 4: "f"}[codec]
    return view.unpack(f"{count}{code}", pos, "curve values")


def _codec(view: ByteView, width: int) -> int:
    """Bytes per component. A value other than 1 or 2 reads as 4-byte floats."""
    codec = view.u8(width, "curve codec")
    return codec if codec in (1, 2) else 4


def decode_translation(blob: bytes, keyframes: int, order: str) -> Curve:
    if not blob:
        return Curve([], [])
    view = ByteView(blob, order)
    width = _key_width(keyframes)
    count = _key_count(view, width)
    codec = _codec(view, width)
    base = (0.0, 0.0, 0.0)
    frames_at = 2 * width
    if count >= 2:
        base_at = _align(width + 1, 4)
        base = view.f32s(base_at, 3, "curve base")
        frames_at = base_at + 12
    frames = _read_frames(view, frames_at, count, width, keyframes)
    values_at = _align(frames_at + count * width, codec)
    raw = _read_components(view, values_at, count * 3, codec)
    _check_blob_end(blob, values_at + count * 3 * codec)
    if codec == 1:
        decoded = [base[i % 3] + x / 127 - 1 for i, x in enumerate(raw)]
    elif codec == 2:
        decoded = [base[i % 3] + x / 32767 - 1 for i, x in enumerate(raw)]
    else:
        decoded = [base[i % 3] + x for i, x in enumerate(raw)]
    return Curve(frames, [tuple(decoded[i:i + 3]) for i in range(0, count * 3, 3)], codec, base)


def decode_rotation(blob: bytes, keyframes: int, order: str) -> Curve:
    if not blob:
        return Curve([], [])
    view = ByteView(blob, order)
    width = _key_width(keyframes)
    count = _key_count(view, width)
    frames = _read_frames(view, width, count, width, keyframes)
    values_at = _align(width * (count + 1), 2)
    raw = view.unpack(f"{count * 4}h", values_at, "curve values")
    _check_blob_end(blob, values_at + count * 8)
    decoded = [to_f32(x * _ROTATION_UNIT) for x in raw]
    return Curve(frames, [tuple(decoded[i:i + 4]) for i in range(0, count * 4, 4)])


def decode_scale(blob: bytes, keyframes: int, order: str) -> Curve:
    if not blob:
        return Curve([], [])
    view = ByteView(blob, order)
    width = _key_width(keyframes)
    count = _key_count(view, width)
    codec = _codec(view, width)
    frames_at = _align(width + 1, width)
    frames = _read_frames(view, frames_at, count, width, keyframes)
    values_at = _align(frames_at + count * width, codec)
    raw = _read_components(view, values_at, count, codec)
    _check_blob_end(blob, values_at + count * codec)
    divisor = {1: 127, 2: 32767, 4: 1}[codec]
    return Curve(frames, [(x / divisor,) for x in raw], codec)


def decode_permutation(perm: Permutation) -> DecodedPermutation:
    if not perm.has_data:
        raise FormatError(f"permutation {perm.name!r} has no data: its .can is missing")
    order = build_order(perm.build)
    k = perm.keyframes
    return DecodedPermutation(
        [decode_translation(blob, k, order) for blob in perm.translation_blobs],
        [decode_rotation(blob, k, order) for blob in perm.rotation_blobs],
        [decode_scale(blob, k, order) for blob in perm.scale_blobs],
    )


def sample_curve(curve: Curve, frame: float, default: tuple[float, ...]) -> tuple[float, ...]:
    """Lerp component-wise. Frames outside the keys take the nearest key."""
    frames = curve.frames
    if not frames:
        return default
    if len(frames) == 1 or frame <= frames[0]:
        return curve.values[0]
    if frame >= frames[-1]:
        return curve.values[-1]
    low, high = 0, len(frames) - 1
    while high - low > 1:
        middle = (low + high) // 2
        if frames[middle] <= frame:
            low = middle
        else:
            high = middle
    t = (frame - frames[low]) / (frames[high] - frames[low])
    return tuple(a + (b - a) * t for a, b in zip(curve.values[low], curve.values[high]))


def _sample_rotation(curve: Curve, frame: float) -> Quaternion:
    """Normalise each key, lerp, then normalise, with no shortest-arc flip."""
    if not curve.frames:
        return IDENTITY_ROTATION
    keyed = Curve(curve.frames, [quat_normalise(key) for key in curve.values])
    return quat_normalise(sample_curve(keyed, frame, IDENTITY_ROTATION))


def _sample_scale(curve: Curve, frame: float) -> float:
    """A single key stays as stored. A result between keys is clamped."""
    if not curve.frames:
        return 1.0
    value = sample_curve(curve, frame, (1.0,))[0]
    if len(curve.frames) == 1:
        return value
    if not math.isfinite(value):
        raise FormatError(f"non-finite interpolated scale {value}")
    return min(max(value, SCALE_MIN), SCALE_MAX)


def sample_track(curves: DecodedPermutation, track: int, frame: float) -> Transform:
    """A local transform."""
    translation = sample_curve(curves.translation[track], frame, (0.0, 0.0, 0.0))
    return Transform(
        _sample_rotation(curves.rotation[track], frame), translation,
        _sample_scale(curves.scale[track], frame),
    )


def play_time(perm: Permutation, time: float, mode: int) -> float:
    """Apply a playback mode to a time. Any mode but stop or wrap is fatal."""
    end = perm.duration
    if mode == MODE_STOP:
        return min(time, end)
    if mode == MODE_WRAP:
        if time <= end or end <= 0:
            return time
        wrapped = math.fmod(time, end)
        return wrapped if wrapped > 0 else end
    raise FormatError(f"playback mode {mode}")


def frame_at_time(perm: Permutation, time: float, even_frames: bool = False) -> float:
    """The frame of a time. ``even_frames`` is the special world mode, from 20 fps up."""
    frame = time * perm.fps
    if even_frames and perm.fps >= _EVEN_FRAME_FPS:
        return float(int(frame) // 2 * 2)
    return frame


# --- Pose ----------------------------------------------------------------------------------

@dataclass
class Pose:
    local: list[Transform]
    world: list[Transform]
    skin: list[Transform]
    bind_is_rest_inverse: bool  # ps3: *Likely* only


def _first_bone_by_hash(bones: list) -> dict[int, int]:
    """A track binds to the first bone with its hash."""
    found: dict[int, int] = {}
    for bone in bones:
        found.setdefault(bone.name_hash, bone.index)
    return found


def pose_at(bones: list, perm: Permutation, curves: DecodedPermutation, frame: float) -> Pose:
    """Pose a skeleton (``AppearanceFile.bones``) at a frame."""
    by_hash = _first_bone_by_hash(bones)
    local = [bone.local for bone in bones]
    for track, name_hash in enumerate(perm.track_hashes):
        index = by_hash.get(name_hash)
        if index is not None:
            local[index] = sample_track(curves, track, frame)
    world: list[Transform] = []
    for bone in bones:
        if bone.parent < 0:
            world.append(local[bone.index])
        else:
            world.append(world[bone.parent].compose(local[bone.index]))
    skin = [w.compose(bone.skin_inverse) for w, bone in zip(world, bones)]
    return Pose(local, world, skin, any(bone.bind_inverse is None for bone in bones))


def skin_vertex(position: Vector, joints, weights, skin: list[Transform]) -> Vector:
    """The sum of ``weight / 255 * skin[bone] * position``. Weight 0 needs no joint."""
    x = y = z = 0.0
    for joint, weight in zip(joints, weights):
        if weight:
            px, py, pz = skin[joint].apply(position)
            share = weight / 255
            x += share * px
            y += share * py
            z += share * pz
    return (x, y, z)


# --- glTF ----------------------------------------------------------------------------------

_FLOAT = 5126


def _add_channel(buf, animation: dict, node: int, path: str, times, values, kind: str) -> None:
    flat = [component for row in values for component in row]
    check_float32(tuple(flat), f"{path} key")
    size = len(flat)
    time_data = struct.pack(f"<{len(times)}f", *times)
    stored = struct.unpack(f"<{len(times)}f", time_data)
    sampler = {
        "input": buf.add(
            time_data, _FLOAT, "SCALAR", len(times), minimum=[min(stored)], maximum=[max(stored)],
        ),
        "output": buf.add(struct.pack(f"<{size}f", *flat), _FLOAT, kind, len(times)),
        "interpolation": "LINEAR",
    }
    animation["samplers"].append(sampler)
    animation["channels"].append({
        "sampler": len(animation["samplers"]) - 1,
        "target": {"node": node, "path": path},
    })


def _track_channels(buf, animation: dict, node: int, curves: DecodedPermutation, track: int,
                    fps: float) -> None:
    translation = curves.translation[track]
    rotation = curves.rotation[track]
    scale = curves.scale[track]
    _add_channel(
        buf, animation, node, "translation",
        [f / fps for f in translation.frames] or [0.0],
        translation.values or [(0.0, 0.0, 0.0)], "VEC3",
    )
    _add_channel(
        buf, animation, node, "rotation",
        [f / fps for f in rotation.frames] or [0.0],
        [quat_normalise(v) for v in rotation.values] or [IDENTITY_ROTATION], "VEC4",
    )
    values = scale.values
    if len(values) > 1:
        values = [(min(max(v[0], SCALE_MIN), SCALE_MAX),) for v in values]
    _add_channel(
        buf, animation, node, "scale",
        [f / fps for f in scale.frames] or [0.0],
        [v * 3 for v in values] or [(1.0, 1.0, 1.0)], "VEC3",
    )


def to_gltf_animated(
    app: AppearanceFile, ani: AniFile, permutation: int = 0, flip_v: bool = False,
    texture_resolver=None, look: int = 0,
) -> dict:
    """The appearance plus one permutation as a glTF animation.

    glTF interpolates rotations spherically, where the game lerps and normalises, so
    in-between frames differ slightly from ``pose_at``.
    """
    if not 0 <= permutation < len(ani.permutations):
        raise FormatError(f"no permutation {permutation}: the file has {len(ani.permutations)}")
    perm = ani.permutations[permutation]
    curves = decode_permutation(perm)
    if not perm.fps > 0:
        raise FormatError(f"permutation {perm.name!r} has frame rate {perm.frame_rate}")
    by_hash = _first_bone_by_hash(app.bones)

    def extend(buf, document: dict) -> None:
        animation: dict = {"name": perm.name, "samplers": [], "channels": []}
        for track, name_hash in enumerate(perm.track_hashes):
            node = by_hash.get(name_hash)
            if node is not None:
                _track_channels(buf, animation, node, curves, track, perm.fps)
        if animation["channels"]:
            document["animations"] = [animation]

    return to_gltf(app, flip_v, extend=extend, texture_resolver=texture_resolver, look=look)


def to_gltf_animated_json(
    app: AppearanceFile, ani: AniFile, permutation: int = 0, flip_v: bool = False,
    texture_resolver=None, look: int = 0,
) -> str:
    return gltf_dumps(to_gltf_animated(app, ani, permutation, flip_v, texture_resolver, look))
