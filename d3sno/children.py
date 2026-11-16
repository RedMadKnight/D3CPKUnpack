"""Writers of the ps3 / x360 child files of ``25_corrections14.md`` section 2 (W3-1 to W3-5): the
animation child ``.can`` of an ``.ani`` and the navigation child ``.nav`` of a ``.scn``.

The parent records hold counts only, never offsets into the child (W3-5). A change in a child is
written by laying the child out again and updating the matching parent counts in place; the
parent is otherwise unchanged.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

from .animation import (
    ATTACHMENT_SIZE, BLEND_WEIGHT_SIZE, CURVE_ENTRY_SIZE, NONLINEAR_SIZE, PERMUTATION_SIZE,
    AniFile, CanFile, CanPermutation, Permutation, _TABLE_AT,
)
from .binary import FormatError, build_order
from .navigation import Navigation

CHILD_BUILDS = ("ps3", "x360")
CAN_ORDER = ">"  # W3-1: big-endian on both builds
BLOB_ALIGN = 8  # W3-2: from the table start
# W3-1: the parent permutation record's counts.
ANI_COUNTS = {"bones": 0x78, "blend": 0x90, "keyframes": 0xA8, "attachments": 0x12C}
# W3-4: the parent scene's counts (body offsets).
SCN_COUNTS = {"cells": 0x180, "neighbours": 0x198, "lookups": 0x1D8, "borders": 0x1F0}
CURVE_KINDS = ("translation", "rotation", "scale")
_FRAME_CODES = {1: "B", 2: "H", 4: "I"}
_VALUE_CODES = {1: "B", 2: "H", 4: "f"}


def _align(value: int, step: int) -> int:
    return (value + step - 1) // step * step


def key_width(keyframes: int) -> int:
    """W3-3: frame width S: 1 below 256 keyframes, 2 below 65,536, else 4."""
    if keyframes < 256:
        return 1
    return 2 if keyframes < 65536 else 4


# --- Curve blobs (W3-3) -------------------------------------------------------------------------

@dataclass
class CurveParts:
    """A curve blob by its parts: key count, codec byte (translation, scale), base (translation
    with 2 or more keys), key frames and raw values (3, 4 or 1 per key)."""

    kind: str
    frames: list[int]
    values: list[tuple]
    codec_byte: int = 0
    base: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def codec(self) -> int:
        return self.codec_byte if self.codec_byte in (1, 2) else 4


def curve_parts(blob: bytes, kind: str, keyframes: int, order: str = CAN_ORDER) -> CurveParts | None:
    """Split a curve blob into its parts (None for an empty curve)."""
    if not blob:
        return None
    s = key_width(keyframes)
    count = struct.unpack_from(order + _FRAME_CODES[s], blob, 0)[0]
    if kind == "rotation":
        frames = list(struct.unpack_from(f"{order}{count}{_FRAME_CODES[s]}", blob, s))
        at = _align(s * (count + 1), 2)
        raw = struct.unpack_from(f"{order}{count * 4}h", blob, at)
        return CurveParts(kind, frames, [tuple(raw[i:i + 4]) for i in range(0, count * 4, 4)])
    codec_byte = blob[s]
    codec = codec_byte if codec_byte in (1, 2) else 4
    base = (0.0, 0.0, 0.0)
    if kind == "translation":
        frames_at = 2 * s
        if count >= 2:
            base_at = _align(s + 1, 4)
            base = struct.unpack_from(order + "3f", blob, base_at)
            frames_at = base_at + 12
        per_key = 3
    elif kind == "scale":
        frames_at = _align(s + 1, s)
        per_key = 1
    else:
        raise FormatError(f"curve kind {kind!r}")
    frames = list(struct.unpack_from(f"{order}{count}{_FRAME_CODES[s]}", blob, frames_at))
    at = _align(frames_at + count * s, codec)
    raw = struct.unpack_from(f"{order}{count * per_key}{_VALUE_CODES[codec]}", blob, at)
    return CurveParts(kind, frames, [tuple(raw[i:i + per_key]) for i in range(0, count * per_key, per_key)],
                      codec_byte, tuple(base))


def encode_curve(parts: CurveParts | None, keyframes: int, order: str = CAN_ORDER) -> bytes:
    """W3-3: a curve blob from its parts, with frame width S from the permutation's keyframes.
    Translation: count, codec, [zeros to 4, base | zeros to 2S], frames, zeros to the codec,
    3 values per key. Rotation: count, frames, zeros to 2, 4 x int16 per key. Scale: count,
    codec, zeros to S, frames, zeros to the codec, 1 value per key. Zeros to 4 at the end."""
    if parts is None or not parts.frames:
        return b""
    s = key_width(keyframes)
    count = len(parts.frames)
    out = bytearray(struct.pack(order + _FRAME_CODES[s], count))

    def pad(step: int) -> None:
        out.extend(bytes(_align(len(out), step) - len(out)))

    if parts.kind == "rotation":
        out += struct.pack(f"{order}{count}{_FRAME_CODES[s]}", *parts.frames)
        pad(2)
        out += struct.pack(f"{order}{count * 4}h", *[x for v in parts.values for x in v])
    else:
        out.append(parts.codec_byte)
        if parts.kind == "translation":
            if count >= 2:
                pad(4)
                out += struct.pack(order + "3f", *parts.base)
            else:
                pad(2 * s)
        else:
            pad(s)
        out += struct.pack(f"{order}{count}{_FRAME_CODES[s]}", *parts.frames)
        pad(parts.codec)
        values = [x for v in parts.values for x in v]
        out += struct.pack(f"{order}{len(values)}{_VALUE_CODES[parts.codec]}", *values)
    pad(4)
    return bytes(out)


# --- .can (W3-1, W3-2) ----------------------------------------------------------------------------

def _section(raw: bytes) -> bytes:
    return struct.pack(">I", len(raw)) + raw


def _curve_section(blobs: list[bytes], tracks: int) -> bytes:
    """W3-2: the size word counts the table only (16 x bones); the table is always written; blobs
    follow in track order, each at the next multiple of 8 from the table start; gaps are zero;
    the section ends at the end of the last blob."""
    if len(blobs) != tracks:
        raise FormatError(f"{len(blobs)} curves for {tracks} bones")
    table = bytearray()
    body = bytearray()
    cursor = CURVE_ENTRY_SIZE * tracks
    for blob in blobs:
        if not blob:
            table += bytes(CURVE_ENTRY_SIZE)
            continue
        offset = _align(cursor, BLOB_ALIGN)
        body += bytes(offset - cursor) + blob
        table += struct.pack(">4I", offset, len(blob), 0, 0)
        cursor = offset + len(blob)
    return struct.pack(">I", len(table)) + bytes(table) + bytes(body)


def encode_can_permutation(perm: Permutation | CanPermutation) -> bytes:
    """W3-1: the seven sections of one permutation."""
    bones = len(perm.track_hashes)
    return b"".join((
        _section(struct.pack(f">{bones}I", *perm.track_hashes)),
        _section(b"".join(struct.pack(">2f", *w) for w in perm.blend_weights)),
        _curve_section(list(perm.translation_blobs), bones),
        _curve_section(list(perm.rotation_blobs), bones),
        _curve_section(list(perm.scale_blobs), bones),
        _section(b"".join(_attachment(a) for a in perm.attachments)),
        _section(b"".join(struct.pack(">3f", *row) for row in perm.nonlinear_offsets)),
    ))


def _attachment(raw: bytes) -> bytes:
    if len(raw) != ATTACHMENT_SIZE:
        raise FormatError(f"attachment of {len(raw)} bytes, {ATTACHMENT_SIZE} expected")
    return bytes(raw)


def write_can(source: AniFile | CanFile | list) -> bytes:
    """W3-1: the child file: the permutations in record order with nothing before, between or
    after them."""
    perms = source.permutations if isinstance(source, (AniFile, CanFile)) else source
    return b"".join(encode_can_permutation(p) for p in perms)


def can_count_problems(ani: AniFile, can: CanFile | None = None) -> list[str]:
    """W3-1: every permutation's counts (bones, blend weights, attachments, keyframes) equal its
    parent record's; with ``can`` None the data merged into ``ani`` is checked."""
    found = []
    parts = can.permutations if can is not None else ani.permutations
    if len(parts) != len(ani.permutations):
        return [f"{len(parts)} child permutations for {len(ani.permutations)} records (W3-1)"]
    for i, (perm, part) in enumerate(zip(ani.permutations, parts)):
        have = (len(part.track_hashes), len(part.blend_weights), len(part.attachments),
                len(part.nonlinear_offsets))
        want = (perm.bone_count, perm.blend_weight_count, perm.attachment_count, perm.keyframes)
        if have != want:
            found.append(f"permutation {i}: child counts {have}, parent counts {want} (W3-1)")
    return found


def update_ani_counts(ani_data: bytes, ani: AniFile) -> bytes:
    """W3-5: the parent ``.ani`` with each permutation record's bone, blend, keyframe and
    attachment counts set from the permutation's data; nothing else changes."""
    if ani.build not in CHILD_BUILDS:
        raise FormatError(f"{ani.build} .ani holds its own data (no child)")
    out = bytearray(ani_data)
    table = 0x10 + _TABLE_AT["ps3"]
    order = build_order(ani.build)
    for i, perm in enumerate(ani.permutations):
        base = table + i * PERMUTATION_SIZE
        counts = {"bones": len(perm.track_hashes), "blend": len(perm.blend_weights),
                  "keyframes": perm.keyframes, "attachments": len(perm.attachments)}
        for name, at in ANI_COUNTS.items():
            struct.pack_into(order + "I", out, base + at, counts[name])
        perm.bone_count, perm.blend_weight_count = counts["bones"], counts["blend"]
        perm.attachment_count = counts["attachments"]
    return bytes(out)


def extend_keyframes(perm: Permutation, extra: int = 1) -> None:
    """W3-3, W3-5: ``extra`` more keyframes at the end, holding the last pose: every curve of 2
    or more keys gets one key at the new last frame with its last value, the nonlinear offsets
    repeat their last row, and every blob is encoded again with the new frame width."""
    if extra < 1:
        raise FormatError("extend_keyframes needs at least one keyframe")
    old_k = perm.keyframes
    new_k = old_k + extra
    for kind in CURVE_KINDS:
        attr = f"{kind}_blobs"
        blobs = []
        for blob in getattr(perm, attr):
            parts = curve_parts(blob, kind, old_k)
            if parts is not None and len(parts.frames) >= 2:
                parts.frames.append(new_k - 1)
                parts.values.append(parts.values[-1])
            blobs.append(encode_curve(parts, new_k))
        setattr(perm, attr, blobs)
    last = perm.nonlinear_offsets[-1] if perm.nonlinear_offsets else (0.0, 0.0, 0.0)
    perm.nonlinear_offsets = list(perm.nonlinear_offsets) + [last] * extra
    perm.keyframes = new_k


# --- .nav (W3-4) ---------------------------------------------------------------------------------

def nav_sections(nav: Navigation) -> list[bytes]:
    """W3-4: grid, cells, neighbours, coarse grid, lookups and borders, rows in the file order."""
    order = build_order(nav.build)
    return [
        b"".join(struct.pack(order + "HH", *row) for row in nav.grid),
        b"".join(struct.pack(order + "6fHHI", *c.min, *c.max, c.flags, c.neighbour_count, c.first_neighbour)
                 for c in nav.cells),
        b"".join(struct.pack(order + "HH", n.flags, n.cell) for n in nav.neighbours),
        b"".join(struct.pack(order + "HHH", s.flags, s.count, s.first_lookup) for s in nav.coarse),
        b"".join(struct.pack(order + "HH", x.flags, x.cell) for x in nav.lookups),
        b"".join(struct.pack(order + "HH", b.cell, b.sides) for b in nav.borders),
    ]


def write_nav(nav: Navigation) -> bytes | None:
    """W3-4: the six sections back to back, with no header, size words or padding. A scene whose
    total is 0 has no child on ps3 (None) and a 0-byte child on x360."""
    if nav.build not in CHILD_BUILDS:
        raise FormatError(f"{nav.build} scenes hold their sections (no .nav)")
    problems = nav_count_problems(nav)
    if problems:
        raise FormatError(problems[0])
    data = b"".join(nav_sections(nav))
    if not data and nav.build == "ps3":
        return None
    return data


def nav_count_problems(nav: Navigation) -> list[str]:
    """W3-4: every section's row count against the scene's counts."""
    h = nav.header
    found = []
    inner = h.inner_width * h.inner_height
    if len(nav.grid) not in ((inner,) if inner else (0,)):
        found.append(f"grid of {len(nav.grid)} rows for {h.inner_width} x {h.inner_height} (W3-4)")
    if len(nav.coarse) != h.coarse_width * h.coarse_height:
        found.append(f"coarse grid of {len(nav.coarse)} rows for {h.coarse_width} x {h.coarse_height} (W3-4)")
    for name, rows, count in (("cells", nav.cells, h.cell_count), ("neighbours", nav.neighbours, h.neighbour_count),
                              ("lookups", nav.lookups, h.lookup_count), ("borders", nav.borders, h.border_count)):
        if len(rows) != count:
            found.append(f"{name}: {len(rows)} rows, the scene counts {count} (W3-4)")
    return found


def update_scene_counts(scn_data: bytes, nav: Navigation) -> bytes:
    """W3-5: the parent ``.scn`` with its cell, neighbour, lookup and border counts set from the
    rows (the grid and coarse dimensions stay; their rows must fit them). The special flags stay
    in the scene."""
    if nav.build not in CHILD_BUILDS:
        raise FormatError(f"{nav.build} scenes hold their sections (no .nav)")
    out = bytearray(scn_data)
    order = build_order(nav.build)
    h = nav.header
    counts = {"cells": len(nav.cells), "neighbours": len(nav.neighbours),
              "lookups": len(nav.lookups), "borders": len(nav.borders)}
    for name, body in SCN_COUNTS.items():
        struct.pack_into(order + "I", out, 0x10 + body, counts[name])
    h.cell_count, h.neighbour_count = counts["cells"], counts["neighbours"]
    h.lookup_count, h.border_count = counts["lookups"], counts["borders"]
    problems = nav_count_problems(nav)
    if problems:
        raise FormatError(problems[0])
    return bytes(out)


__all__ = [
    "CurveParts", "can_count_problems", "curve_parts", "encode_can_permutation", "encode_curve",
    "extend_keyframes", "key_width", "nav_count_problems", "nav_sections", "update_ani_counts",
    "update_scene_counts", "write_can", "write_nav",
]
