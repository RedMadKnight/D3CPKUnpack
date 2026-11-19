"""Creating animation content (``30_create_content.md`` W15-B1 to W15-B3).

- W15-B1: curve blobs from key frames and values (``encode_curve_values``), whole permutations
  from decoded or new curves (``encode_animation``). A curve read from a blob keeps the decoder's
  base and raw values while its values still decode from them.
- W15-B2: switch ``.ani`` assembly through the generic writer (``write_ani``), identical blobs
  written once and shared, and (31 V18-3) equal per-permutation blocks of one field written
  once and shared, unchanged blocks where they were stored; ps3 / x360 use ``children.write_can`` and ``children.update_ani_counts``
  (``update_ani_records`` also writes changed record fields in place).
- W15-B3: inputs and defaults (``new_permutation``, ``set_in_place``), rotation unit and hemisphere
  checks (``rotation_problems``, ``align_hemispheres``), input checks (``permutation_problems``)
  and the HYPOTHESIS translation codec choice (``hypothesis_codec``; used only with
  ``codec_rule="hypothesis"``).
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import Iterable

from .animation import (
    ATTACHMENT_SIZE, BLOCK_FIELDS, PERMUTATION_SIZE, AniFile, Curve, DecodedPermutation,
    Permutation, _ROTATION_UNIT, _read_permutation, block_keys,
)
from .binary import FormatError, build_order, to_f32
from .children import CurveParts, curve_parts, encode_curve, update_ani_counts
from .sno import SNO_SIGNATURE, expected_version
from .sno_layout import (
    HEADER_SIZE, Block, PairValue, Record, SnoTree, record_layout, top_record, write_tree,
)

KINDS = ("translation", "rotation", "scale")
PER_KEY = {"translation": 3, "rotation": 4, "scale": 1}
CODECS = (1, 2, 4)
CODEC_RULES = ("float", "hypothesis")
IN_PLACE_CONTACT = (-1.0, 1.0, 0.0, 1.0)  # W15-B3
UNIT_TOLERANCE = 2e-3  # |norm - 1| allowed for a rotation key (int16 storage, Q)
BYTE_CODEC_ERROR = 1 / 254  # W15-B3: the error of codec 1
_PERM_COUNT_AT = 0x20  # top record: permutation count (W14-20 link of the permutation list)
_CHILD_TABLE = 0x50  # ps3 / x360 .ani: the records at file 0x50
_CURVE_RECORDS = {"translation": "TranslationCurve", "rotation": "RotationCurve",
                  "scale": "ScaleCurve"}


def _round(x: float) -> int:
    """Round half away from zero (Q: the rule's ``round``)."""
    if not math.isfinite(x):
        raise FormatError(f"cannot store the non-finite value {x}")
    return int(math.floor(x + 0.5)) if x >= 0 else -int(math.floor(-x + 0.5))


def _check_frames(frames: list[int], keyframes: int) -> None:
    if keyframes < 1:
        raise FormatError(f"keyframe count {keyframes}")
    if frames[0] != 0:
        raise FormatError(f"first key at frame {frames[0]}, must be 0 (W15-B1)")
    if any(a >= b for a, b in zip(frames, frames[1:])):
        raise FormatError("key frames are not strictly ascending (W15-B1)")
    if len(frames) >= 2 and frames[-1] != keyframes - 1:
        raise FormatError(f"last key at {frames[-1]}, must be K - 1 = {keyframes - 1} (W15-B1)")
    if frames[-1] > keyframes - 1:
        raise FormatError(f"key frame {frames[-1]} past K - 1 = {keyframes - 1}")


def _stored_targets(frames: list[int], values: list[tuple], base) -> list[tuple]:
    """The values a translation stores: offsets from the base with 2 or more keys, else as is."""
    if len(frames) >= 2:
        return [tuple(v[i] - base[i] for i in range(3)) for v in values]
    return [tuple(v) for v in values]


def hypothesis_codec(frames, values, base=None, max_error: float | None = None) -> int:
    """W15-B3 (HYPOTHESIS): the codec for new ps3 / x360 translation data: 4 if any stored value
    (offset from the base, or the value of a single key) is outside [-1, 1], else 2; 1 only when
    ``max_error`` accepts an error of 1/254."""
    frames = list(frames)
    values = [tuple(float(c) for c in v) for v in values]
    if len(frames) >= 2:
        base = tuple(to_f32(b) for b in (base if base is not None else values[0]))
    stored = _stored_targets(frames, values, base)
    if any(abs(c) > 1 for row in stored for c in row):
        return 4
    if max_error is not None and max_error >= BYTE_CODEC_ERROR:
        return 1
    return 2


def _pick_codec(kind, build, codec, frames, values, base, codec_rule, max_error) -> int:
    if codec is not None:
        if codec not in CODECS:
            raise FormatError(f"codec {codec}: 1, 2 or 4 bytes per component (W15-B1)")
        return codec
    if codec_rule not in CODEC_RULES:
        raise FormatError(f"codec rule {codec_rule!r}: one of {CODEC_RULES}")
    if build == "switch":
        return 4  # W15-B3: switch always 4
    if codec_rule == "hypothesis" and kind == "translation":
        return hypothesis_codec(frames, values, base, max_error)
    return 4


def _quantise(x: float, codec: int, offset_one: bool) -> int | float:
    """Codec 1: round((x + 1) * 127) / round(x * 127); codec 2 the same with 32767; codec 4 f32."""
    if codec == 4:
        return to_f32(x)
    scale, limit = (127, 0xFF) if codec == 1 else (32767, 0xFFFF)
    q = _round(((x + 1) if offset_one else x) * scale)
    if not 0 <= q <= limit:
        raise FormatError(f"value {x} does not fit codec {codec} (W15-B1)")
    return q


def _decoded_translation(raw: tuple, codec: int, base) -> tuple:
    """The decoder's arithmetic (``animation.decode_translation``)."""
    if codec == 1:
        return tuple(base[i] + x / 127 - 1 for i, x in enumerate(raw))
    if codec == 2:
        return tuple(base[i] + x / 32767 - 1 for i, x in enumerate(raw))
    return tuple(base[i] + x for i, x in enumerate(raw))


def _raw_fits(raw, codec: int | None) -> bool:
    if codec == 4:
        return all(isinstance(x, float) or isinstance(x, int) for x in raw)
    low, high = {None: (-32768, 32767), 1: (0, 0xFF), 2: (0, 0xFFFF)}[codec]
    return all(isinstance(x, int) and low <= x <= high for x in raw)


def _same(a, b) -> bool:
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


def _keep_raw(kind, raw, values, codec, base) -> bool:
    """W15-B1: the decoder's raw values are kept while the values still decode from them."""
    if raw is None or len(raw) != len(values):
        return False
    for stored, value in zip(raw, values):
        if len(stored) != PER_KEY[kind] or not _raw_fits(stored, codec if kind != "rotation" else None):
            return False
        if kind == "rotation":
            decoded = tuple(to_f32(x * _ROTATION_UNIT) for x in stored)
        elif kind == "translation":
            decoded = _decoded_translation(stored, codec, base)
        else:
            decoded = (stored[0] / {1: 127, 2: 32767, 4: 1}[codec],)
        if not _same(decoded, tuple(value)):
            return False
    return True


def encode_curve_values(
    kind: str, frames, values, keyframes: int, build: str, codec: int | None = None,
    base=None, *, raw=None, codec_rule: str = "float", max_error: float | None = None,
    unit_tolerance: float = UNIT_TOLERANCE,
) -> bytes:
    """W15-B1: a curve blob from integer key frames and values (3 per key translation, 4
    rotation, 1 scale), with frame width S from the permutation's keyframe count K (W3-3).

    Translation and scale take ``codec`` (1, 2 or 4 bytes per component); without it switch
    uses 4 and ps3 / x360 use 4, or with ``codec_rule="hypothesis"`` the W15-B3 choice for
    translation. A translation of 2 or more keys stores ``base`` (default: the first key's value)
    as three f32 and every key as its offset; one key stores its value with no base.
    ``raw`` (``Curve.raw`` of a decoded curve) is written as is while ``values`` still decode
    from it with ``base`` and ``codec``; otherwise the values are quantised. New rotation keys
    must be unit quaternions within ``unit_tolerance``. No keys: an empty curve (b"")."""
    if kind not in PER_KEY:
        raise FormatError(f"curve kind {kind!r}")
    frames = [int(f) for f in frames]
    values = [tuple(float(c) for c in v) for v in values]
    if not frames:
        if values:
            raise FormatError(f"{len(values)} values for no key frames")
        return b""
    if len(values) != len(frames):
        raise FormatError(f"{len(values)} values for {len(frames)} key frames")
    width = PER_KEY[kind]
    if any(len(v) != width for v in values):
        raise FormatError(f"{kind} keys take {width} components (W15-B1)")
    _check_frames(frames, keyframes)
    order = build_order(build)
    if kind == "rotation":
        if _keep_raw(kind, raw, values, None, None):
            stored = [tuple(int(x) for x in row) for row in raw]
        else:
            for i, v in enumerate(values):
                if abs(math.sqrt(sum(c * c for c in v)) - 1) > unit_tolerance:
                    raise FormatError(f"rotation key {i} is not a unit quaternion (W15-B3)")
            stored = [tuple(_round(c * 32767) for c in v) for v in values]
            if any(not -32768 <= x <= 32767 for row in stored for x in row):
                raise FormatError("rotation component outside int16 (W15-B1)")
        return encode_curve(CurveParts(kind, frames, stored), keyframes, order)
    if kind == "translation":
        if len(frames) >= 2:
            base = tuple(to_f32(b) for b in (base if base is not None else values[0]))
            if len(base) != 3:
                raise FormatError("a translation base is three floats")
        else:
            if base is not None and any(base):
                raise FormatError("a translation of one key has no base (W15-B1)")
            base = (0.0, 0.0, 0.0)
    codec = _pick_codec(kind, build, codec, frames, values, base, codec_rule, max_error)
    if _keep_raw(kind, raw, values, codec, base):
        stored = [tuple(row) for row in raw]
    elif kind == "translation":
        stored = [tuple(_quantise(c, codec, True) for c in row)
                  for row in _stored_targets(frames, values, base)]
    else:
        stored = [(_quantise(v[0], codec, False),) for v in values]
    parts = CurveParts(kind, frames, stored, codec, base if kind == "translation" else (0.0, 0.0, 0.0))
    return encode_curve(parts, keyframes, order)


def encode_curve_object(curve: Curve, kind: str, keyframes: int, build: str,
                        codec_rule: str = "float", max_error: float | None = None) -> bytes:
    """W15-B1 for a ``Curve``: a decoded curve (``raw`` set) passes its codec, base and raw values;
    a new curve (``raw`` None) passes its codec only, so the base is its first key."""
    decoded = curve.raw is not None
    return encode_curve_values(
        kind, curve.frames, curve.values, keyframes, build,
        codec=None if kind == "rotation" else curve.codec,
        base=curve.base if decoded and kind == "translation" and len(curve.frames) >= 2 else None,
        raw=curve.raw, codec_rule=codec_rule, max_error=max_error,
    )


def encode_animation(permutations: Iterable, build: str, codec_rule: str = "float",
                     max_error: float | None = None) -> list[Permutation]:
    """W15-B1: encode every curve of every permutation. Each item is ``(Permutation,
    DecodedPermutation)``; the permutation's blobs are replaced (one list per kind, one curve per
    bone) and its build set. Returns the permutations."""
    done = []
    for perm, curves in permutations:
        bones = len(perm.track_hashes)
        for kind in KINDS:
            track_curves = getattr(curves, kind)
            if len(track_curves) != bones:
                raise FormatError(f"{perm.name!r}: {len(track_curves)} {kind} curves for {bones} bones")
            setattr(perm, f"{kind}_blobs", [
                encode_curve_object(c, kind, perm.keyframes, build, codec_rule, max_error)
                for c in track_curves
            ])
        perm.build = build
        perm.has_data = True
        done.append(perm)
    return done


# --- Rotations (W15-B3) --------------------------------------------------------------------------

def rotation_problems(values, tolerance: float = UNIT_TOLERANCE) -> list[str]:
    """W15-B3: every key a unit quaternion, and each key in the hemisphere of the previous one
    (dot product >= 0), since sampling lerps without a sign flip."""
    found = []
    previous = None
    for i, v in enumerate(values):
        norm = math.sqrt(sum(c * c for c in v))
        if abs(norm - 1) > tolerance:
            found.append(f"rotation key {i}: norm {norm:.6f} (W15-B3: unit)")
        if previous is not None and sum(a * b for a, b in zip(previous, v)) < 0:
            found.append(f"rotation key {i}: opposite hemisphere to key {i - 1} (W15-B3)")
        previous = v
    return found


def align_hemispheres(values) -> list[tuple[float, ...]]:
    """Negate each key that is in the opposite hemisphere of the (aligned) previous key."""
    out: list[tuple[float, ...]] = []
    for v in values:
        v = tuple(float(c) for c in v)
        if out and sum(a * b for a, b in zip(out[-1], v)) < 0:
            v = tuple(-c for c in v)
        out.append(v)
    return out


# --- Permutation records (W15-B3) ----------------------------------------------------------------

# (attribute, record offset, struct code); counts are taken from the data lists
_FIELDS = (
    ("flags", 0x00, "I"), ("frame_rate", 0x48, "f"), ("tool_values", 0x4C, "4f"),
    ("blend_times", 0x5C, "2I"), ("weight", 0x64, "I"), ("speed_min", 0x68, "f"),
    ("speed_delta", 0x6C, "f"), ("ragdoll_factors", 0x70, "2f"), ("bone_count", 0x78, "I"),
    ("blend_weight_count", 0x90, "I"), ("keyframes", 0xA8, "I"), ("contact_keyframes", 0xE0, "4f"),
    ("contact_offsets", 0xF0, "12f"), ("earliest_interrupt", 0x120, "f"),
    ("attachment_count", 0x12C, "I"), ("average_velocity", 0x148, "3f"),
    ("hardpoint_link", 0x154, "I"),
)
_NAME_AT, _NAME_SIZE = 0x04, 68


def sync_counts(perm: Permutation) -> None:
    """The record counts from the data lists (W3-1, W14-20)."""
    if perm.has_data:
        perm.bone_count = len(perm.track_hashes)
        perm.blend_weight_count = len(perm.blend_weights)
        perm.attachment_count = len(perm.attachments)


def pack_permutation(perm: Permutation) -> bytes:
    """The 360-byte record: ``perm.raw`` with every field whose value differs from the raw bytes
    written (unchanged fields keep their bytes, NaN payloads and signed zeros included). Pairs
    are left to the writer."""
    sync_counts(perm)
    order = build_order(perm.build)
    raw = bytearray(perm.raw if len(perm.raw) == PERMUTATION_SIZE else bytes(PERMUTATION_SIZE))
    current = _read_permutation(bytes(raw), order, perm.build)
    for attr, at, code in _FIELDS:
        value = getattr(perm, attr)
        if _same(value, getattr(current, attr)):
            continue
        flat = [c for row in value for c in row] if attr == "contact_offsets" else (
            list(value) if isinstance(value, (tuple, list)) else [value])
        struct.pack_into(order + code, raw, at, *flat)
    if perm.name != current.name:
        name = perm.name.encode("latin-1")
        if len(name) >= _NAME_SIZE:
            raise FormatError(f"permutation name of {len(name)} bytes; at most {_NAME_SIZE - 1}")
        raw[_NAME_AT:_NAME_AT + _NAME_SIZE] = name.ljust(_NAME_SIZE, b"\0")
    perm.raw = bytes(raw)
    return perm.raw


def new_permutation(
    build: str, name: str, keyframes: int, frame_rate: float, track_hashes, *, flags: int = 0,
    tool_values=(0.0, 0.0, 0.0, 0.0), blend_times=(0, 0), weight: int = 0,
    speed_min: float = 0.0, speed_delta: float = 0.0, ragdoll_factors=(0.0, 0.0),
    blend_weights=(), attachments=(), average_velocity=(0.0, 0.0, 0.0), nonlinear_offsets=None,
    contact_keyframes=IN_PLACE_CONTACT, contact_offsets=None, earliest_interrupt: float = 0.0,
    hardpoint_link: int = 0,
) -> Permutation:
    """W15-B3: a permutation from its inputs, with the in-place defaults: average velocity 0,
    nonlinear offsets zero (K rows), contact keyframes (-1, 1, 0, 1), contact offsets zero.
    ``frame_rate`` is fps / 60. Curves are empty until ``encode_animation``."""
    hashes = [int(h) & 0xFFFFFFFF for h in track_hashes]
    bones = len(hashes)
    perm = Permutation(
        build, flags, name, float(frame_rate), tuple(tool_values), tuple(blend_times), weight,
        speed_min, speed_delta, tuple(ragdoll_factors), bones, len(blend_weights), keyframes,
        tuple(contact_keyframes),
        tuple(tuple(r) for r in (contact_offsets or [(0.0, 0.0, 0.0)] * 4)),
        earliest_interrupt, len(attachments), tuple(average_velocity), hardpoint_link,
        bytes(PERMUTATION_SIZE), True, hashes, [tuple(w) for w in blend_weights],
        [b""] * bones, [b""] * bones, [b""] * bones, [bytes(a) for a in attachments],
        [tuple(r) for r in (nonlinear_offsets or [(0.0, 0.0, 0.0)] * keyframes)],
    )
    pack_permutation(perm)
    return perm


def set_in_place(perm: Permutation) -> None:
    """W15-B3: the in-place defaults: no root motion (velocity 0, offsets zero), contact
    keyframes (-1, 1, 0, 1), contact offsets zero."""
    perm.average_velocity = (0.0, 0.0, 0.0)
    perm.nonlinear_offsets = [(0.0, 0.0, 0.0)] * perm.keyframes
    perm.contact_keyframes = IN_PLACE_CONTACT
    perm.contact_offsets = ((0.0, 0.0, 0.0),) * 4


def _structure_problems(perm: Permutation) -> list[str]:
    """The counts a writer needs: K nonlinear rows, blend weights none or one per bone, one curve
    per bone and kind, 192-byte attachments."""
    found = []
    k = perm.keyframes
    bones = len(perm.track_hashes)
    if k < 1:
        found.append(f"keyframe count {k}")
    if len(perm.nonlinear_offsets) != k:
        found.append(f"{len(perm.nonlinear_offsets)} nonlinear offset rows for K = {k}")
    if len(perm.blend_weights) not in (0, bones):
        found.append(f"{len(perm.blend_weights)} blend weights for {bones} bones")
    if any(len(a) != ATTACHMENT_SIZE for a in perm.attachments):
        found.append(f"an attachment is not {ATTACHMENT_SIZE} bytes")
    for kind in KINDS:
        blobs = getattr(perm, f"{kind}_blobs")
        if len(blobs) != bones:
            found.append(f"{len(blobs)} {kind} curves for {bones} bones")
    return found


def permutation_problems(perm: Permutation, curves: DecodedPermutation | None = None,
                         tolerance: float = UNIT_TOLERANCE) -> list[str]:
    """W15-B3 input checks: K >= 1, a positive frame rate, K nonlinear rows with row 0 zero,
    blend weights none or one per bone, one curve per bone and kind, 192-byte attachments, codec
    4 on switch, and (with ``curves``) unit rotation keys each in the previous key's hemisphere."""
    found = _structure_problems(perm)
    if not perm.frame_rate > 0:
        found.append(f"frame rate {perm.frame_rate}")
    if perm.nonlinear_offsets and any(perm.nonlinear_offsets[0]):
        found.append("nonlinear offset row 0 is not zero (W15-B3)")
    order = build_order(perm.build)
    if perm.build == "switch":
        for kind in ("translation", "scale"):
            blobs = getattr(perm, f"{kind}_blobs")
            for track, blob in enumerate(blobs):
                parts = curve_parts(blob, kind, perm.keyframes, order) if blob else None
                if parts is not None and parts.codec_byte != 4:
                    found.append(f"{kind} curve {track}: codec {parts.codec_byte}, switch uses 4 (W15-B3)")
    if curves is not None:
        for track, curve in enumerate(curves.rotation):
            found += [f"track {track}: {p}" for p in rotation_problems(curve.values, tolerance)]
    return found


# --- Switch .ani (W15-B2) ------------------------------------------------------------------------

def _head(head, build: str, top_size: int) -> tuple[bytes, bytearray]:
    order = build_order(build)
    if isinstance(head, (bytes, bytearray)):
        if len(head) < HEADER_SIZE + top_size:
            raise FormatError(f"head of {len(head)} bytes; the header and top record need "
                              f"{HEADER_SIZE + top_size}")
        return bytes(head[:HEADER_SIZE]), bytearray(head[HEADER_SIZE:HEADER_SIZE + top_size])
    top = bytearray(top_size)
    if isinstance(head, AniFile):
        h = head.header
        header = struct.pack(order + "4I", h.signature, h.version, h.revision, h.source_hash)
        struct.pack_into(order + "3I", top, 0, h.handle, h.runtime_1, h.runtime_2)
        struct.pack_into(order + "IIi", top, 0x0C, head.flags, head.playback_mode,
                         head.appearance_handle)
        return header, top
    if head is not None:
        raise FormatError("head: the source file's bytes, an AniFile or None")
    version = expected_version("ani", build) or 0
    header = struct.pack(order + "4I", SNO_SIGNATURE, version, 0, 0)
    struct.pack_into(order + "i", top, 0x14, -1)  # no appearance
    return header, top


def _value_block(raw: bytes) -> PairValue | None:
    return PairValue(Block(bytes(raw))) if raw else None


class _BlockPool:
    """31 V18-3: the per-permutation blocks placed so far, by field: (key, stored offset or
    None, Block). The stored offset is set only for a block that still has its parsed content."""

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.placed: dict[str, list[tuple[object, int | None, Block]]] = {}

    @staticmethod
    def _stored(perm: Permutation, name: str, key) -> int | None:
        """The parsed offset of the permutation's block while its content is unchanged."""
        stored = perm.stored_blocks.get(name)
        return stored[0] if stored is not None and stored[1] == key else None

    def find(self, perm: Permutation, name: str, key) -> Block | None:
        """An earlier block of the same field to point at, or None to store the block.

        - New or changed content: the first earlier block with equal bytes.
        - Unchanged content (``stored_blocks``): an earlier equal block stored at the same
          offset, else an earlier equal block that is new or changed; an earlier equal block
          that was stored elsewhere and is unchanged stays apart (stored twice, as parsed)."""
        if not self.enabled or key is None:
            return None
        mine = self._stored(perm, name, key)
        loose = None
        for other_key, other_at, block in self.placed.get(name, ()):
            if other_key != key:
                continue
            if mine is None or other_at == mine:
                return block
            if other_at is None and loose is None:
                loose = block
        return loose

    def add(self, perm: Permutation, name: str, key, block: Block) -> None:
        if key is not None:
            self.placed.setdefault(name, []).append((key, self._stored(perm, name, key), block))


def write_ani(permutations, build: str = "switch", head=None, share_blobs: bool = True,
              share_blocks: bool = True) -> bytes:
    """W15-B2 (switch): the ``.ani`` from its permutations. ``head`` gives the SNO header and
    the top record (the source file's bytes, an ``AniFile`` or None for a new file); the
    permutation count, record counts and every pair are set. Blobs are inline; identical blobs
    of the file are written once and every curve entry with the same bytes points at that copy
    (``share_blobs``). The generic writer places the blocks.

    31 V18-3 (``share_blocks``): a per-permutation block (track hashes, blend weights, each
    curve-entry table, attachments, nonlinear offsets) whose bytes equal the same field of an
    earlier permutation is stored once and the later pair points at it. Curve tables are shared
    only with ``share_blobs`` (their bytes are equal only when their blobs are shared).

    Unchanged blocks keep the stored layout: a permutation read by ``parse_ani`` carries each
    block's parsed offset and content (``Permutation.stored_blocks``). A block whose content is
    unchanged shares an earlier equal block that was stored at the same offset, and is stored
    again where the file stored an equal block twice (seen for blend weights). So an unchanged
    file is written back as parsed; a new or changed block follows the sharing rule."""
    if build != "switch":
        raise FormatError(f"{build} .ani data lives in the .can: use children.write_can and "
                          "update_ani_counts / update_ani_records (W15-B2)")
    order = build_order(build)
    top_name = top_record(build, "ani")
    top_size = record_layout(build, "ani", top_name).size
    header, top_raw = _head(head, build, top_size)
    perms = list(permutations)
    struct.pack_into(order + "I", top_raw, _PERM_COUNT_AT, len(perms))
    shared: dict[bytes, Block] = {}
    pool = _BlockPool(share_blocks)
    records = []
    for perm in perms:
        if not perm.has_data:
            raise FormatError(f"permutation {perm.name!r} has no curve data")
        perm.build = build
        problems = _structure_problems(perm)
        if problems:
            raise FormatError(f"permutation {perm.name!r}: {problems[0]}")
        record = Record("AnimPermutation", bytearray(pack_permutation(perm)))
        keys = block_keys(perm)

        def place(name: str, make, keys=keys, perm=perm, record=record) -> None:
            """V18-3: point at an earlier equal block of the field, or store a new one."""
            key = keys[name]
            if name in KINDS and not share_blobs:
                key = None  # without shared blobs equal tables have different bytes
            block = pool.find(perm, name, key)
            if block is not None:
                record.pairs[BLOCK_FIELDS[name]] = PairValue(block, shared=True)
                return
            value = make()
            record.pairs[BLOCK_FIELDS[name]] = value
            if value is not None:
                pool.add(perm, name, key, value.block)

        place("track_hashes", lambda: _value_block(keys["track_hashes"] or b""))
        place("blend_weights", lambda: _value_block(keys["blend_weights"] or b""))
        for kind in KINDS:
            place(kind, lambda kind=kind: _curve_table(perm, kind, shared, share_blobs))
        place("attachments", lambda: PairValue(Block(elements=[
            Record("KeyframedAttachment", bytearray(a)) for a in perm.attachments]))
            if perm.attachments else None)
        place("nonlinear_offsets", lambda: _value_block(keys["nonlinear_offsets"] or b""))
        records.append(record)
    top = Record(top_name, top_raw)
    top.pairs[0x18] = PairValue(Block(elements=records)) if records else None
    return write_tree(SnoTree(build, "ani", header, top))


def _curve_table(perm: Permutation, kind: str, shared: dict[bytes, Block],
                 share_blobs: bool) -> PairValue | None:
    """A curve-entry table with its blobs; equal blobs of the file share one block."""
    tails = perm.curve_tails.get(kind, [])
    entries = []
    for track, blob in enumerate(getattr(perm, f"{kind}_blobs")):
        entry = Record(_CURVE_RECORDS[kind], bytearray(16))
        if track < len(tails) and len(tails[track]) == 8:
            entry.raw[8:] = tails[track]
        if blob:
            block = shared.get(bytes(blob)) if share_blobs else None
            if block is None:
                block = Block(bytes(blob))
                shared[bytes(blob)] = block
                entry.pairs[0] = PairValue(block)
            else:
                entry.pairs[0] = PairValue(block, shared=True)
        entries.append(entry)
    return PairValue(Block(elements=entries)) if entries else None


# --- ps3 / x360 parent (W15-B2, W3-5) ------------------------------------------------------------

def update_ani_records(ani_data: bytes, ani: AniFile) -> bytes:
    """ps3 / x360: the parent ``.ani`` with every permutation record's changed fields written in
    place (``pack_permutation``) and its counts set (``update_ani_counts``); pairs stay zero.
    The child is ``children.write_can``. The permutation count cannot change here."""
    count = (len(ani_data) - _CHILD_TABLE) // PERMUTATION_SIZE
    if count != len(ani.permutations):
        raise FormatError(f"{len(ani.permutations)} permutations for a parent of {count} records")
    out = bytearray(ani_data)
    for i, perm in enumerate(ani.permutations):
        if not perm.has_data:
            raise FormatError(f"permutation {perm.name!r} has no data: merge its .can first")
        problems = _structure_problems(perm)
        if problems:
            raise FormatError(f"permutation {perm.name!r}: {problems[0]}")
        at = _CHILD_TABLE + i * PERMUTATION_SIZE
        if len(perm.raw) != PERMUTATION_SIZE:
            perm.raw = bytes(out[at:at + PERMUTATION_SIZE])
        out[at:at + PERMUTATION_SIZE] = pack_permutation(perm)
    return update_ani_counts(bytes(out), ani)


__all__ = [
    "CODEC_RULES", "IN_PLACE_CONTACT", "KINDS", "align_hemispheres", "encode_animation",
    "encode_curve_object", "encode_curve_values", "hypothesis_codec", "new_permutation",
    "pack_permutation", "permutation_problems", "rotation_problems", "set_in_place", "sync_counts",
    "update_ani_records", "write_ani",
]
