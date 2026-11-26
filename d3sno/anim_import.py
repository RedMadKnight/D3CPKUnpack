"""Stage 22 (``49_stage22_json_import.md``) W22-10: importing an edited ``anims/<h>.glb`` onto its ``.ani``.

The template is exported again in memory (``export._animation_glb``, the same rules and options) and
compared with the edited file channel by channel. A (joint, path) whose sampled keys differ from the
template's by more than ``TOLERANCE`` is changed; its curve is encoded from the new keys with W15-B1 to
W15-B3 (``anim_write.encode_curve_values``) in the template's codec and keyframe count. Every other curve
keeps its stored blob, and keys that equal the template's keep their stored values. Switch files are
written with ``anim_write.write_ani`` (unchanged permutations keep their bytes, ``share_blobs``); ps3 and
x360 keep the animation data in the ``.can``, which is written with ``children.write_can``.
"""

from __future__ import annotations

import json
import math

from . import animation, anim_write, appearance, children, export
from .binary import FormatError
from .json_import import ImportResult, Refused, path_text
from .mesh_import import read_gltf
from .transform import quat_normalise

__all__ = ["TOLERANCE", "import_animation"]

TOLERANCE = 1e-5  # W22-10
FRAME_TOLERANCE = 1e-3  # a key time on the template's frame grid
_KINDS = {"translation": "translation", "rotation": "rotation", "scale": "scale"}


def _close(a, b, tolerance: float = TOLERANCE) -> bool:
    return len(a) == len(b) and all(abs(x - y) <= tolerance for x, y in zip(a, b))


def _channels(doc, anim: dict) -> dict:
    out = {}
    for channel in anim.get("channels", []):
        target = channel.get("target", {})
        sampler = anim["samplers"][channel["sampler"]]
        times = [row[0] for row in doc.accessor(sampler["input"])]
        values = [tuple(row) for row in doc.accessor(sampler["output"])]
        out[(target.get("node"), target.get("path"))] = (times, values)
    return out


def import_animation(template: bytes, build: str, app_data: bytes, app_build: str, edited: bytes, handle: int,
                     app_handle: int, can: bytes | None = None, travel: bool = False,
                     force: bool = False) -> ImportResult:
    """W22-10. Returns the new ``.ani`` bytes (``data``) and, on ps3 / x360, the new ``.can`` (``nav``
    field: the companion file). ``force`` treats every channel as changed (a test of the encoder: an
    unedited file then gives the stored bytes)."""
    ani = animation.parse_ani(template, build, can)
    app = appearance.parse_appearance(app_data, app_build)
    old = read_gltf(export._animation_glb(app, ani, handle, app_handle, travel))
    try:
        new = read_gltf(edited)
    except FormatError as exc:
        raise _one((), "bad_json", f"the glTF does not read: {exc}")
    refusals: list[Refused] = []
    old_nodes = [n.get("name") for n in old.json.get("nodes", [])]
    new_nodes = [n.get("name") for n in new.json.get("nodes", [])]
    if old_nodes != new_nodes:
        raise _one(("nodes",), "skeleton_mismatch", f"{len(new_nodes)} joints; the mesh file's skeleton has "
                                                    f"{len(old_nodes)} with other names" if len(old_nodes) != len(new_nodes)
                   else "the joint names differ from the mesh file's skeleton")
    for i, (a, b) in enumerate(zip(old.json.get("nodes", []), new.json.get("nodes", []))):
        for key in sorted(set(a) | set(b)):
            if json.dumps(a.get(key), sort_keys=True) != json.dumps(b.get(key), sort_keys=True):
                refusals.append(Refused(("nodes", i, key), "derived_field", "the skeleton is the mesh file's"))
    old_anims, new_anims = old.json.get("animations", []), new.json.get("animations", [])
    if len(old_anims) != len(new_anims):
        raise _one(("animations",), "permutation_count", f"{len(new_anims)} animations for {len(old_anims)} "
                                                         "permutations with data")
    old_extras = (old.json.get("extras") or {}).get("d3sno_export", {})
    new_extras = (new.json.get("extras") or {}).get("d3sno_export", {})
    for key in sorted(set(old_extras) | set(new_extras)):
        if json.dumps(old_extras.get(key), sort_keys=True) != json.dumps(new_extras.get(key), sort_keys=True):
            refusals.append(Refused(("extras", "d3sno_export", key), "derived_field",
                                    "average_velocity, nonlinear_offsets and ticks_per_second keep their stored values"))
    listed = old_extras.get("permutations", [])
    perm_of = {entry["animation"]: entry["index"] for entry in listed if entry.get("animation") is not None}
    by_hash = animation._first_bone_by_hash(app.bones)
    rows, changed_perms = [], set()
    for a, (oa, na) in enumerate(zip(old_anims, new_anims)):
        perm_index = perm_of[a]
        perm = ani.permutations[perm_index]
        if oa.get("name") != na.get("name"):
            refusals.append(Refused(("animations", a, "name"), "derived_field", "the permutation name is not changed"))
        old_ch, new_ch = _channels(old, oa), _channels(new, na)
        if set(old_ch) != set(new_ch):
            refusals.append(Refused(("animations", a, "channels"), "skeleton_mismatch",
                                    "the animated joints and paths differ from the template's"))
            continue
        curves = animation.decode_permutation(perm)
        track_of = {}
        for track, name_hash in enumerate(perm.track_hashes):
            node = by_hash.get(name_hash)
            if node is not None:
                track_of.setdefault(node, track)
        for (node, kind), (times, values) in sorted(old_ch.items(), key=lambda kv: (kv[0][0], kv[0][1])):
            new_times, new_values = new_ch[(node, kind)]
            same = len(times) == len(new_times) and _close(times, new_times) and all(
                _close(x, y) for x, y in zip(values, new_values))
            if same and not force:
                continue
            path = ("animations", a, "channels", node, kind)
            try:
                blob = _encode(perm, curves, track_of[node], kind, times, values, new_times, new_values, build, path)
            except Refused as refusal:
                refusals.append(refusal)
                continue
            setattr_list = getattr(perm, f"{kind}_blobs")
            if blob != setattr_list[track_of[node]]:
                setattr_list[track_of[node]] = blob
                changed_perms.add(perm_index)
            if not same:
                rows.append({"path": path_text(path), "old": {"keys": len(times)}, "new": {"keys": len(new_times)}})
    if refusals:
        raise _many(refusals)
    if not changed_perms:
        return ImportResult(bytes(template), rows, [], None)
    if build == "switch":
        data = anim_write.write_ani(ani.permutations, build, head=template)
        return ImportResult(data, rows, [], None)
    return ImportResult(bytes(template), rows, [], children.write_can(ani))


def _encode(perm, curves, track: int, kind: str, times, values, new_times, new_values, build: str, path) -> bytes:
    fps = perm.fps
    curve = getattr(curves, kind)[track]
    duration = perm.duration
    end = new_times[-1] if new_times else None
    # ``51`` AB5: the range of the template's own export of this channel (0 to the duration when it has none)
    first, last = (times[0], times[-1]) if times else (0.0, duration)
    if not new_times or abs(new_times[0] - first) > TOLERANCE or (
            len(new_times) >= 2 and abs(max(new_times) - last) > TOLERANCE):
        raise Refused(path, "duration_changed", f"keys from {new_times[0] if new_times else None} to {end} s; the "
                                                f"template's channel runs {first} to {last} s")
    frames = []
    for t in new_times:
        frame = t * fps
        if abs(frame - round(frame)) > FRAME_TOLERANCE:
            raise Refused(path, "keyframe_count_mismatch", f"a key at {t} s is not on a frame of the permutation's "
                                                           f"{perm.keyframes} keyframes")
        frames.append(int(round(frame)))
    if any(a >= b for a, b in zip(frames, frames[1:])):
        raise Refused(path, "keyframe_count_mismatch", "two keys on one frame, or keys out of order")
    last_frame = int(round(last * fps)) if times else perm.keyframes - 1  # ``51`` AB5: the template's last key
    if len(frames) >= 2 and frames[-1] != last_frame:
        raise Refused(path, "keyframe_count_mismatch", f"the last key is frame {frames[-1]}, the template's channel "
                                                       f"ends at frame {last_frame} of {perm.keyframes} keyframes")
    if not curve.frames and len(frames) == 1 and _close(new_values[0], values[0]):
        return b""  # an empty curve stays empty (its default key)
    by_frame = {f: (i, v) for i, (f, v) in enumerate(zip(curve.frames, curve.values))}
    shown = dict(zip([int(round(t * fps)) for t in times], values))
    out = []
    for frame, value in zip(frames, new_values):
        stored = by_frame.get(frame)
        if stored is not None and frame in shown and _close(shown[frame], value):
            out.append(stored[1])  # the stored key, so its stored bytes are kept
            continue
        if kind == "rotation":
            norm = math.sqrt(sum(c * c for c in value))
            if not norm:
                raise Refused(path, "value_out_of_range", "a zero rotation key")
            out.append(tuple(quat_normalise(value)))
        elif kind == "scale":
            if max(value) - min(value) > TOLERANCE:
                raise Refused(path, "value_out_of_range", "the game stores one scale per key: x, y and z must agree")
            out.append((value[0],))
        else:
            out.append(tuple(value))
    keep = len(frames) == len(curve.frames) and frames == curve.frames
    try:
        blob = anim_write.encode_curve_values(
            kind, frames, out, perm.keyframes, build,
            codec=None if kind == "rotation" else curve.codec,
            base=curve.base if kind == "translation" and keep and len(frames) >= 2 else None,
            raw=curve.raw if keep else None)
    except FormatError as exc:
        raise Refused(path, "value_out_of_range", str(exc))
    stored = getattr(perm, f"{kind}_blobs")[track]
    if blob != stored and _decoded(kind, blob, perm) == _decoded(kind, stored, perm):
        return stored  # the same keys and stored values: the stored bytes (padding as stored) are kept
    return blob


def _decoded(kind: str, blob: bytes, perm):
    from .binary import build_order

    decode = {"translation": animation.decode_translation, "rotation": animation.decode_rotation,
              "scale": animation.decode_scale}[kind]
    try:
        curve = decode(blob, perm.keyframes, build_order(perm.build))
    except FormatError:
        return None
    return curve.frames, curve.raw, curve.codec, curve.base


def _one(path, reason: str, message: str) -> Refused:
    error = Refused(path, reason, message)
    error.refusals = [error]
    return error


def _many(refusals: list) -> Refused:
    error = refusals[0]
    error.refusals = refusals
    return error
