"""Stage 18 (``40_export.md``): ``export`` writes the user's archives as glTF 2.0 binary, PNG and
JSON in a local output directory, with ``manifest.json`` and ``coverage.json`` (W18-1 to W18-10).

Export, don't interpret: every file holds what the library's existing readers and exporters already
decode. Open questions are written as flags (``"open"``), never guessed; ``43`` (stage 19, W19-1
to W19-7) writes the settled ones as statements. One input gives the same
output every time: JSON keys sorted, indent 1, a newline at the end, floats as the shortest text
that reads back to the same float32; only ``created`` in the manifest depends on the time.

The output directory is game content: it is made on the user's computer only and must never be put
into a repository, a test fixture or a release. ``export_game`` refuses an output directory inside
a git working tree unless ``force`` is given.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import math
import mmap
import os
import re
import struct
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from . import (
    animation, anims, appearance, archive, collision, index, material, navigation, records, shader, text,
    texture, uv2_use, world,
)
from .binary import FormatError
from .mesh_import import gltf_to_glb
from .power import TagMap, TagRecord
from .sno import group_by_extension, group_by_id, parse_header
from .transform import quat_normalise

__all__ = [
    "COVERAGE_FORMAT", "COVERAGE_FORMAT_VERSION", "EXIT_OK", "EXIT_REFUSED", "EXIT_STOPPED", "EXPORT_GROUPS", "ExportResult",
    "FORMAT", "FORMAT_VERSION", "REASONS", "ROTATION", "TICKS_PER_SECOND", "TRAVEL_NODE", "dumps", "export_game", "f32_text",
    "in_git_tree", "plain", "prune_empty",
]

FORMAT = "d3sno-export"
COVERAGE_FORMAT = "d3sno-export-coverage"
FORMAT_VERSION = 4  # the manifest (``43`` W19-1, ``44`` W20-5, ``46`` W21-1)
COVERAGE_FORMAT_VERSION = 1
EXIT_OK, EXIT_STOPPED, EXIT_REFUSED = 0, 1, 2
# W18-4 to W18-8, in the order the export runs them (textures before the meshes that name them,
# meshes before the animations that need them).
EXPORT_GROUPS = ("tex", "app", "ani", "acr", "mon", "mat", "scn", "wrl", "lvl", "mrk", "phm", "stl")
REASONS = ("parse_error", "not_decoded", "missing_dependency", "filtered", "unsupported_build",
           "duplicate_differs")
OPEN = "open"
# ``43`` W19-1: the settled questions, written as statements
V_DIRECTION = "top_left_origin"
TANGENTS = "not_stored"
ROOT_MOTION = "in_place"
# ``44`` W20-4: every stored quaternion [x, y, z, w] rotates as v' = q v q* (Hamilton product, active
# rotation, the glTF reading); a parent applies to a child as q_parent * q_child
ROTATION = "active_xyzw"
COORDINATES = {"source": "stored", "up": "+z", "unit": "stored", "handedness": "right", "forward": "+x",
               "winding": "ccw", "rotation": ROTATION}
WORLD_ROTATION_NOTE = "world_rotation_untested"
TANGENT_NOTE = ("no tangents are stored; a viewer computes them from TEXCOORD_0 (V as stored) and the stored "
                "normals, e.g. by the MikkTSpace method; vertices of mirrored UV islands get bitangent sign -1")
TICKS_PER_SECOND = 60
TRAVEL_NODE = "d3sno_travel"
_READ_ONLY = ("ans", "shm")  # read for other groups, never exported (``44`` W20-1: shader maps)
_INDEX_NAMES = re.compile(r"^(coretoc|commontoc|act[1-5]toc)\.dat$", re.IGNORECASE)
_X360_LOCALE = re.compile(r"^([a-z]{2}[A-Z]{2})_")
_TAG_TYPES = {0: "int", 1: "float", 2: "sno", 3: "balance", 4: "formula", 5: "actor_group",
              6: "attribute", 7: "start_location"}
_REF = re.compile(r"file_ref\(group (\d+)\)")


class ExportRefused(Exception):
    """A W18-1 refusal (exit code 2)."""


class _Stopped(Exception):
    def __init__(self, info: dict) -> None:
        super().__init__(info.get("error"))
        self.info = info


@dataclass
class ExportResult:
    """W18-10: the manifest and coverage as dicts (None after a refusal) and the W18-1 exit code."""

    manifest: dict | None
    coverage: dict | None
    exit_code: int
    message: str | None = None  # the refusal, or why the export stopped
    summary: list[str] = field(default_factory=list)  # one line per group, then the totals


# --- deterministic JSON -------------------------------------------------------------------------

def f32_text(value: float) -> str:
    """The shortest decimal text that reads back to the same float32 as ``value``."""
    target = struct.pack("<f", value)
    single = struct.unpack("<f", target)[0]  # the float32 itself: equal float32 values give equal text
    for digits in range(1, 10):
        text_ = f"{single:.{digits}g}"
        try:
            if struct.pack("<f", float(text_)) == target:
                break
        except OverflowError:  # ``41`` V26-4: a text just above FLT_MAX is no match; try more digits
            continue
    if not any(c in text_ for c in ".en"):
        text_ += ".0"
    return text_


def _float(value: float):
    if not math.isfinite(value):
        return "nan" if math.isnan(value) else ("inf" if value > 0 else "-inf")
    try:
        return float(f32_text(value))
    except OverflowError:  # above the float32 range: kept as the double it is
        return value


def plain(value):
    """A JSON value of a decoded object: dataclasses as dicts of their fields, tuples as lists,
    bytes as ``{"bytes": "<hex>"}``, floats as float32 text, non-finite floats as strings."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return _float(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"bytes": bytes(value).hex()}
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        items = sorted(value) if isinstance(value, (set, frozenset)) else value
        return [plain(v) for v in items]
    if isinstance(value, TagMap):
        return _tag_map(value)
    if isinstance(value, records.Record):
        return _record(value)
    if is_dataclass(value):
        return {f.name: plain(getattr(value, f.name)) for f in fields(value)}
    return str(value)


def dumps(document) -> bytes:
    """Sorted keys, indent 1, one newline at the end (W18 principle)."""
    return (json.dumps(plain(document), sort_keys=True, indent=1, ensure_ascii=False, allow_nan=False)
            + "\n").encode("utf-8")


# --- shared JSON forms (W18-8) -------------------------------------------------------------------

def _link(group_id: int, handle) -> dict | None:
    """``{"group": "<ext>", "handle": n}``; a link of value -1 (none) is null."""
    if handle is None or handle in (-1, 0xFFFFFFFF):
        return None
    group = group_by_id(group_id & 0xFFFFFFFF) if isinstance(group_id, int) else None
    return {"group": group.extension if group else str(group_id), "handle": int(handle)}


def _enum(value: int, name) -> dict:
    return {"value": value, "name": name}


def _flags(value: int, names) -> dict:
    return {"value": value, "names": list(names)}


def _rotation(q) -> dict:
    """A unit quaternion ``[x, y, z, w]``; the stored form under ``raw``."""
    return {"quaternion": plain(list(quat_normalise(tuple(q)))), "raw": plain(list(q))}


def _header(header) -> dict:
    return {"handle": header.handle, "version": header.version, "revision": header.revision,
            "source_hash": header.source_hash, "byte_order": "big" if header.order == ">" else "little"}


def _tag_record(record: TagRecord) -> dict:
    return {"tag": record.tag, "type": _enum(record.type, _TAG_TYPES.get(record.type)),
            "value": plain(record.value)}


def _tag_map(tag_map: TagMap) -> dict:
    out = {"records": [_tag_record(r) for r in tag_map.records]}
    if tag_map.error:
        out["error"] = tag_map.error
    return out


def _record(record: records.Record) -> dict:
    """A layout record by its field names: links, enums and flags in their W18-8 forms."""
    try:
        spec = records.layout(record.build or "switch", record.type)
        kinds = {}
        for item in spec.fields:
            for name in item.name.split(","):
                kinds[name] = item.type if "," not in item.name else "i32"
    except FormatError:
        kinds = {}
    out = {}
    for name, value in record.values.items():
        kind = kinds.get(name, "")
        ref = _REF.match(kind)
        if ref and isinstance(value, int):
            out[name] = _link(int(ref.group(1)), value)
        elif kind.startswith("typed_ref") and isinstance(value, (tuple, list)):
            out[name] = _link(value[0], value[1])
        elif "enum" in kind and isinstance(value, int):
            row = record.enum(name)
            out[name] = _enum(value, row.name if row else None)
        elif "flags" in kind and isinstance(value, int):
            found = record.flags(name)
            entry = _flags(value, found.names)
            if found.unknown_bits:
                entry["unknown_bits"] = list(found.unknown_bits)
            out[name] = entry
        else:
            out[name] = plain(value)
    return out


# --- per group --------------------------------------------------------------------------------------

def _actor_json(data: bytes, build: str) -> dict:
    found = records.parse_record_file(data, "acr", build)
    rec = found.record
    return {
        "header": _header(found.header), "build": found.build, "fields": _record(rec),
        "links": {"appearance": _link(9, rec.get("appearance_ref")), "anim_set": _link(8, rec.get("anim_set_ref")),
                  "collision": _link(61, rec.get("physics_mesh_ref")), "monster": _link(25, rec.get("monster_ref"))},
        "bounds": {name: plain(rec.get(name)) for name in (
            "bound_cylinder_base", "bound_cylinder_height", "bound_cylinder_radius", "bound_sphere_center",
            "bound_sphere_radius", "bound_box_min", "bound_box_extent")},
        "actor_type": _record(rec).get("actor_kind"),
        "warnings": found.warnings,
    }


def _monster_json(data: bytes, build: str) -> dict:
    found = records.parse_record_file(data, "mon", build)
    rec = found.record
    order = found.header.order
    return {"header": _header(found.header), "build": found.build, "fields": _record(rec),
            "links": {"actor": _link(1, rec.get("actor_ref"))},
            "attributes": plain(records.monster_attributes(rec, order)), "warnings": found.warnings}


def _material_json(data: bytes, build: str) -> dict:
    mat = material.parse_material(data, build)
    block = mat.block
    return {
        "header": _header(mat.header), "build": mat.build, "flags": mat.flags,
        "tag_map": {"bytes": mat.tag_map_raw.hex()},
        "shader": _link(36, block.shader_map),  # a link only (W18-8)
        "textures": [{"slot": _enum(t.type, t.name), "texture": _link(44, t.handle),
                      "animation_raw": {"bytes": t.animation_raw.hex()}} for t in block.textures],
        "parameters": {"ambient": plain(block.ambient), "diffuse": plain(block.diffuse),
                       "specular": plain(block.specular), "emissive": plain(block.emissive),
                       "power": plain(block.power), "sort_priority_raw": block.sort_priority_raw},
    }


def _navigation_json(nav: navigation.Navigation) -> dict:
    cells = []
    for cell in nav.cells:
        linked = nav.neighbours[cell.first_neighbour:cell.first_neighbour + cell.neighbour_count]
        cells.append({
            "min": plain(cell.min), "max": plain(cell.max),
            "flags": _flags(cell.flags, navigation.flag_names(cell.flags)),
            "neighbour_count": cell.neighbour_count, "first_neighbour": cell.first_neighbour,
            "neighbours": [{"flags": n.flags, "cell": n.cell, "border": n.border_index} for n in linked],
        })
    return {
        "has_sections": nav.has_sections, "scene_header": plain(nav.header), "cells": cells,
        "neighbours": plain(nav.neighbours), "grid": plain(nav.grid), "coarse": plain(nav.coarse),
        "lookups": plain(nav.lookups), "borders": plain(nav.borders),
        "special_flags": {"bytes": bytes(nav.special_flags).hex()}, "warnings": nav.warnings,
    }


def _scene_json(data: bytes, build: str, nav: bytes | None) -> tuple[dict, bool]:
    """The scene and its navigation; the flag says that a needed ``.nav`` was missing."""
    info = world.parse_scene_info(data, build)
    navi = navigation.parse_scene(data, info.build, nav)
    return {
        "header": _header(info.header), "build": info.build,
        "bounds": plain(info.bounds), "marker_bounds": plain(info.marker_bounds),
        "exclusions": info.exclusions, "inclusions": info.inclusions,
        "marker_sets": [_link(24, h) for h in info.marker_sets], "look_hash": info.look_hash,
        "xp_density": plain(info.xp_density), "appearance": _link(9, info.appearance),
        "collision": _link(61, info.collision_mesh), "navigation": _navigation_json(navi),
        "warnings": info.warnings,
    }, not navi.has_sections


def _marker_json(marker: world.Marker) -> dict:
    return {
        "name_hash": marker.name_hash, "type": _enum(marker.type, marker.type_name),
        "link": _link(marker.group, marker.handle), "position": plain(marker.position),
        "rotation": _rotation(marker.rotation), "scale": plain(marker.scale), "tags": _tag_map(marker.tags),
        "links": plain(marker.links),
    }


def _marker_set_json(data: bytes, build: str) -> dict:
    found = world.parse_marker_set(data, build)
    return {"header": _header(found.header), "build": found.build,
            "markers": [_marker_json(m) for m in found.markers], "circles": plain(found.circles),
            "bounds": plain(found.bounds), "flags": found.flags, "label": found.label,
            "always_active": found.always_active, "special_indices": found.special_indices}


def _entry_json(entry: world.SceneEntry) -> dict:
    out = {"scene": _link(entry.group, entry.handle), "position": plain(entry.position),
           "rotation": _rotation(entry.rotation), "level_areas": list(entry.level_areas),
           "cluster_id": entry.cluster_id}
    if not _identity(entry.rotation):  # ``44`` W20-4: no shipped file has one
        out["note"] = WORLD_ROTATION_NOTE
    return out


def _identity(q) -> bool:
    """A rotation of none: x, y and z zero (any w; a zero quaternion is read as the identity)."""
    return q[0] == 0 and q[1] == 0 and q[2] == 0


def _world_json(data: bytes, build: str, scenes, sets) -> tuple[dict, list]:
    """The world, its generator blocks as stored and the layout of ``lay_out_world``; the list
    names the scenes and marker sets that the selected archives do not hold."""
    wrl = world.parse_world(data, build)
    server = None
    if wrl.server is not None:
        s = wrl.server
        server = {"generators": plain(s.generators),
                  "scene_lists": [[_entry_json(e) for e in entries] for entries in s.scene_lists],
                  "next_cluster_id": s.next_cluster_id, "cluster_count": s.cluster_count,
                  "clusters_raw": plain(s.clusters_raw), "navigation_powers": list(s.navigation_powers),
                  "script": _link(35, s.script_handle), "seed": s.seed}
    wanted = {e.handle for e in wrl.scene_entries}
    scene_map = {h: scenes(h) for h in sorted(wanted)}
    scene_map = {h: v for h, v in scene_map.items() if v is not None}
    set_handles = sorted({h for v in scene_map.values() for h in v.marker_sets} | set(wrl.marker_sets))
    set_map = {h: sets(h) for h in set_handles}
    set_map = {h: v for h, v in set_map.items() if v is not None}
    layout = world.lay_out_world(wrl, scene_map, set_map)
    placed = [{
        "marker_set": _link(24, p.marker_set), "scene": None if p.scene is None else _link(p.scene.group, p.scene.handle),
        "position": plain(p.position), "rotation": _rotation(p.rotation), "conditional": p.conditional,
        "marker": _marker_json(p.marker),
    } for p in layout.markers]
    missing = [("scn", h) for h in layout.missing_scenes] + [("mrk", h) for h in layout.missing_marker_sets]
    return {
        "header": _header(wrl.header), "build": wrl.build, "is_generated": wrl.is_generated, "server": server,
        "scene_entries": [_entry_json(e) for e in wrl.scene_entries],
        "marker_sets": [_link(24, h) for h in wrl.marker_sets],
        "skybox_actor": _link(1, wrl.skybox_actor), "skybox_word": wrl.skybox_word,
        "music": _link(62, wrl.music), "ambient": _link(5, wrl.ambient), "reverb": _link(60, wrl.reverb),
        "weather": _link(47, wrl.weather),
        "layout": {"markers": placed, "sub_scene_slots": sum(1 for p in layout.markers if p.marker.type == world.TYPE_SUB_SCENE),
                   "missing_scenes": layout.missing_scenes, "missing_marker_sets": layout.missing_marker_sets},
        "warnings": wrl.warnings,
    }, missing


def _collision_json(data: bytes, build: str) -> dict:
    """Per mesh: vertices, triangles, flags and bounds. The tree is not exported (W18-8)."""
    phm = collision.parse_collision(data, build)
    meshes = [{
        "vertices": plain(mesh.positions),
        "triangles": [{"vertices": list(t.vertices), "neighbours": list(t.neighbours), "material": t.material,
                       "flags": t.flags} for t in mesh.triangles],
        "flags": mesh.flags, "material": mesh.material, "triangle_count": mesh.triangle_count,
        "bounds": {"centre": plain(mesh.centre), "half_extent": plain(mesh.half_extent)},
        "box_scale": plain(mesh.box_scale),
    } for mesh in phm.meshes]
    return {"header": _header(phm.header), "build": phm.build, "flags": phm.flags, "meshes": meshes,
            "warnings": phm.warnings}


def _level_area_json(data: bytes, build: str) -> dict:
    """``.lvl`` has no decoder beyond the SNO header: the header, and the body as an open flag."""
    header = parse_header(data)
    return {"header": _header(header), "build": build, "body": OPEN, "size": len(data)}


def _strings_json(data: bytes, build: str) -> dict:
    found = text.parse_string_list(data, build)
    return {"header": _header(found.header), "build": found.build,
            "strings": {f"{h:08X}": t for h, t in found.by_hash.items()}, "warnings": found.warnings}


def _locale(name: str, archive_stem: str) -> str:
    """The locale of a string table: the ``|xxYY`` name suffix (switch), the archive prefix (ps3,
    x360 ``enUS_*``), else ``default``."""
    found = text.locale_of(name) or text.locale_of_archive(archive_stem)
    if found is None:
        match = _X360_LOCALE.match(archive_stem)
        found = match.group(1) if match else None
    return found or "default"


# --- git working tree check (W18-1) -----------------------------------------------------------------

def in_git_tree(path) -> Path | None:
    """The directory holding a ``.git`` entry at ``path`` or above it, or None."""
    target = Path(os.path.abspath(path))
    for candidate in (target, *target.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def _inside(child: Path, parent: Path) -> bool:
    c, p = os.path.normcase(str(child)), os.path.normcase(str(parent))
    return c == p or c.startswith(p.rstrip("\\/") + os.sep)


# --- the run ------------------------------------------------------------------------------------------

@dataclass
class _Copy:
    archive: str
    cpk: archive.CpkArchive
    entry: archive.CpkFile

    @property
    def locale(self) -> str | None:
        """The ``|locale`` part of the entry's name, or None (``42`` V27-2)."""
        return (self.entry.name or "").partition("|")[2] or None


def _locale_rank(copy: _Copy):
    """V27-2: the entry without a locale part first, then ``enUS``, then the others sorted."""
    locale = copy.locale
    return (0, "") if locale is None else (1, "") if locale == "enUS" else (2, locale)


@dataclass
class _Item:
    ext: str
    name: str  # the archive name without the locale
    locale: str | None
    full: str
    copies: list[_Copy] = field(default_factory=list)
    data: bytes | None = None
    build: str | None = None
    handle: int | None = None
    copy: _Copy | None = None
    read_error: str | None = None
    hash: str | None = None
    differs: list = field(default_factory=list)


def _open_archive(path: Path):
    handle = open(path, "rb")
    try:
        if os.fstat(handle.fileno()).st_size < 4:
            raise FormatError("file too short for an archive")
        view = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
    except (OSError, ValueError, FormatError):
        handle.close()
        raise
    try:
        return archive.read_archive(view, strict=False), (view, handle)
    except Exception:
        view.close()
        handle.close()
        raise


def _groups(groups) -> list[str]:
    if groups is None:
        return sorted(EXPORT_GROUPS)
    if isinstance(groups, str):
        groups = [g for g in groups.split(",")]
    chosen = []
    for g in groups:
        g = g.strip().lstrip(".").lower()
        if not g:
            continue
        if g not in EXPORT_GROUPS:
            raise ExportRefused(f"--groups: {g!r} is not an exported group ({', '.join(sorted(EXPORT_GROUPS))})")
        if g not in chosen:
            chosen.append(g)
    if not chosen:
        raise ExportRefused("--groups names no group")
    return sorted(chosen)


class _Run:
    def __init__(self, game: Path, out: Path, archives, groups, all_levels, skip_existing, build, travel=False):
        self.game, self.out = game, out
        self.travel = bool(travel)
        self.patterns = list(archives) if archives else None
        self.groups = groups
        self.all_levels = all_levels
        self.skip_existing = skip_existing
        self.given_build = build
        self.handles: list = []
        self.entries: list[dict] = []
        self.coverage: dict[str, dict] = {g: {"found": 0, "unique": 0, "exported": 0, "skipped": 0, "files": []}
                                          for g in EXPORT_GROUPS}
        self.unread: list[dict] = []
        self.archives: list[tuple[str, archive.CpkArchive]] = []
        self.names: dict[tuple[int, int], str] = {}
        self.name_handles: dict[tuple[int, str], int] = {}
        self.items: dict[str, dict[tuple, _Item]] = {g: {} for g in (*EXPORT_GROUPS, *_READ_ONLY)}
        self.exported: dict[str, set] = {g: set() for g in EXPORT_GROUPS}
        self.old: dict[str, dict] = {}
        self.build = None
        self.memory: dict[str, bytes] | None = None  # ``46`` W21-2: an export in memory
        self.archive_paths: dict[str, Path] = {}  # archive base name -> its file (``46`` W21-6)

    # --- setup -----------------------------------------------------------------------------
    def close(self) -> None:
        for view, handle in self.handles:
            view.close()
            handle.close()
        self.handles.clear()

    def open_archives(self) -> None:
        found = sorted((p for p in self.game.rglob("*") if p.is_file() and p.suffix.lower() == ".cpk"),
                       key=lambda p: (p.stem.lower(), str(p.relative_to(self.game)).lower()))
        recognised: dict[str, archive.CpkArchive] = {}
        for path in found:
            stem = path.stem
            if stem.lower() in {s.lower() for s in recognised}:
                self.unread.append({"archive": stem, "message": "a second archive with this base name"})
                continue
            try:
                cpk, held = _open_archive(path)
            except (FormatError, OSError, ValueError, struct.error) as exc:
                self.unread.append({"archive": stem, "message": str(exc)})
                continue
            self.handles.append(held)
            recognised[stem] = cpk
            self.archive_paths[stem] = path
        if not recognised:
            raise ExportRefused(f"{self.game.name or self.game}: no recognised archive")
        builds = sorted({cpk.header.build for cpk in recognised.values()})
        if len(builds) > 1:
            raise ExportRefused(f"the archives are of several builds: {', '.join(builds)}")
        self.build = builds[0]
        if self.given_build is not None and self.given_build != self.build:
            raise ExportRefused(f"--build {self.given_build}: the archives are {self.build}")
        self.all_archives = recognised
        for stem in sorted(recognised, key=str.lower):
            if "cutscene" in stem.lower():
                continue  # never exported in this stage (W18-1)
            if self.patterns is not None and not any(
                    fnmatch.fnmatch(stem.lower(), p.lower()) or fnmatch.fnmatch(f"{stem}.cpk".lower(), p.lower())
                    for p in self.patterns):
                continue
            self.archives.append((stem, recognised[stem]))
        if not self.archives:
            raise ExportRefused("no archive matches --archives (cutscene archives are never exported)")

    def read_indexes(self) -> None:
        """CoreTOC (or the six x360 TOC files) beside the archives and inside any recognised
        archive; the first name of each (group, handle) wins, in sorted order."""
        sources = []
        for path in sorted(p for p in self.game.rglob("*") if p.is_file() and _INDEX_NAMES.match(p.name)):
            sources.append((path.name, lambda path=path: path.read_bytes()))
        for stem in sorted(self.all_archives, key=str.lower):
            cpk = self.all_archives[stem]
            for entry in sorted((e for e in cpk.files if e.name), key=lambda e: e.name.lower()):
                base = entry.name.replace("/", "\\").rpartition("\\")[2]
                if _INDEX_NAMES.match(base):
                    sources.append((f"{stem}:{entry.name}", lambda c=cpk, e=entry: c.read_file(e)))
        for label, read in sources:
            try:
                toc = index.parse_core_toc(read(), self.build)
            except (FormatError, OSError, struct.error) as exc:
                self.unread.append({"index": label.rpartition(":")[2], "message": str(exc)})
                continue
            for e in toc.entries:
                key = (e.group & 0xFFFFFFFF, e.handle & 0xFFFFFFFF)
                self.names.setdefault(key, e.name)
                self.name_handles.setdefault((key[0], e.name.lower()), key[1])

    def list_files(self) -> None:
        wanted = set(EXPORT_GROUPS) | set(_READ_ONLY)
        for stem, cpk in self.archives:
            for entry in cpk.files:
                if entry.name is None:
                    continue
                full = entry.name
                name, _, locale = full.partition("|")
                ext = name.rpartition(".")[2].lower() if "." in name.rpartition("\\")[2] else ""
                if ext not in wanted or group_by_extension(ext) is None:
                    continue
                where = _locale(full, stem) if ext == "stl" else None
                key = (name.lower(), where)
                item = self.items[ext].get(key)
                if item is None:
                    item = self.items[ext][key] = _Item(ext, name, where, full)
                item.copies.append(_Copy(stem, cpk, entry))
                if ext in self.coverage:
                    self.coverage[ext]["found"] += 1

    def load_old_manifest(self) -> None:
        path = self.out / "manifest.json"
        if not self.skip_existing or not path.is_file():
            return
        try:
            old = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if not isinstance(old, dict) or old.get("format") != FORMAT:
            return
        # ``43``: files of another manifest version, or of another ``--travel``, are written again
        options = old.get("options")
        if old.get("format_version") != FORMAT_VERSION or not isinstance(options, dict) \
                or bool(options.get("travel", False)) != self.travel:
            return
        for entry in old.get("entries", []):
            if isinstance(entry, dict) and isinstance(entry.get("path"), str):
                self.old[entry["path"]] = entry

    # --- reading ---------------------------------------------------------------------------
    def stem_handle(self, item: _Item) -> int | None:
        group = group_by_extension(item.ext)
        stem = item.name.replace("/", "\\").partition("\\")[2].rpartition(".")[0]
        return self.name_handles.get((group.id, stem.lower())) if group else None

    def load(self, item: _Item) -> bool:
        """Read the first copy (sorted archive order) that reads; note copies that differ."""
        if item.data is not None or item.read_error is not None:
            return item.data is not None
        hashes = []  # (archive entry, copy index, hash, message): every copy of every entry (V26-5)
        item.copies.sort(key=_locale_rank)  # stable: archive order within one locale (V27-2)
        for copy in item.copies:
            for index_ in range(max(copy.entry.count, 1)):
                try:
                    data = copy.cpk.read_file(copy.entry, index_)
                except FormatError as exc:
                    hashes.append((copy, index_, None, str(exc)))
                    continue
                hashes.append((copy, index_, hashlib.sha256(data).hexdigest(), None))
                if item.data is None:
                    item.data, item.copy = data, copy
        if item.data is None:
            item.read_error = hashes[0][3] if hashes else "no copy"
            item.handle = self.stem_handle(item)
            return False
        item.hash = hashlib.sha256(item.data).hexdigest()
        try:
            item.build = archive.file_build(item.copy.cpk.header, item.data, item.ext)
            item.handle = parse_header(item.data).handle
        except FormatError as exc:
            item.read_error = str(exc)
            item.handle = self.stem_handle(item)
            item.build = item.copy.cpk.header.build
        differ = len({h for _, _, h, _ in hashes}) > 1
        item.differs = [{"archive": c.archive, "copy": i, "hash": h, "locale": self.row_locale(item, c),
                         **({"message": m} if m else {})} for c, i, h, m in hashes] if differ else []
        return item.read_error is None

    @staticmethod
    def row_locale(item: _Item, copy: _Copy | None) -> str | None:
        """V27-2: the locale of a ``duplicate_differs`` row or manifest entry: a string table's own
        locale (W18-8), else the entry's ``|locale`` part, else None."""
        if item.ext == "stl":
            return item.locale
        return copy.locale if copy is not None else None

    def companion(self, item: _Item, folder_ext: str) -> bytes | None:
        """``Anim\\<stem>\\0000.can`` beside ``Anim\\<stem>.ani``, ``Scene\\<stem>\\0000.nav`` beside
        ``Scene\\<stem>.scn``: from the archive of the copy used, else any selected archive."""
        name = item.name.rpartition(".")[0] + f"\\0000.{folder_ext}"
        order = [item.copy.cpk] + [c for _, c in self.archives if c is not item.copy.cpk]
        for cpk in order:
            entry = cpk.find(name)
            if entry is not None:
                try:
                    return cpk.read_file(entry)
                except FormatError:
                    return None
        return None

    def by_handle(self, ext: str):
        """(handle -> item) of a group, first by sorted name order; read on demand."""
        cache: dict[int, _Item] = {}
        for key in sorted(self.items[ext]):
            item = self.items[ext][key]
            if self.load(item) and item.handle not in cache:
                cache[item.handle] = item
        return cache

    # --- coverage and output -------------------------------------------------------------------
    def note(self, item: _Item, reason: str, exported: bool, message: str | None = None, **extra) -> None:
        entry = {"handle": item.handle, "archive": item.copy.archive if item.copy else item.copies[0].archive,
                 "reason": reason, "exported": exported}
        if item.locale:
            entry["locale"] = item.locale
        if message:
            entry["message"] = message
        entry.update(extra)
        self.coverage[item.ext if item.ext in self.coverage else "stl"]["files"].append(entry)

    def emit(self, item: _Item, outputs: list) -> None:
        """``outputs``: (path, producer, extra entry fields). Every producer runs before any file
        is written, so a failed file leaves no output; ``skip_existing`` keeps a file whose earlier
        manifest entry has the same source archive and hash and an ``output_hash`` (``46`` W21-1:
        the SHA-256 of the file as written, kept from that entry). ``memory``: nothing is written,
        the files are kept in ``self.memory`` by path (``46`` W21-2)."""
        source = {"archive": item.copy.archive, "hash": item.hash}
        group = group_by_extension(item.ext)
        name = self.names.get((group.id, item.handle & 0xFFFFFFFF)) if item.handle is not None else None
        entries = []
        for path, _, extra in outputs:
            entry = {"group": item.ext, "handle": item.handle, "name": name, "path": path, "source": source,
                     "locale": self.row_locale(item, item.copy)}  # V27-2
            entry.update(extra)
            entries.append(entry)
        keep = self.memory is None and self.skip_existing and all(
            self.old.get(path, {}).get("source") == source and isinstance(self.old[path].get("output_hash"), str)
            and (self.out / path).is_file() for path, _, _ in outputs)
        if keep:
            for entry in entries:
                entry["output_hash"] = self.old[entry["path"]]["output_hash"]
        else:
            blobs = [(path, producer()) for path, producer, _ in outputs]
            for entry, (path, blob) in zip(entries, blobs):
                entry["output_hash"] = hashlib.sha256(blob).hexdigest()
            for path, blob in blobs:
                if self.memory is not None:
                    self.memory[path] = blob
                else:
                    _write(self.out / path, blob)
        self.entries.extend(entries)

    def export_item(self, item: _Item, job) -> None:
        """Run one file: FormatError is ``parse_error``; any other exception stops the export."""
        cov = self.coverage[item.ext]
        cov["unique"] += 1
        if item.ext not in self.groups:
            cov["skipped"] += 1
            if item.handle is None:
                item.handle = self.stem_handle(item)
            self.note(item, "filtered", False)
            return
        if not self.load(item):
            cov["skipped"] += 1
            self.note(item, "parse_error", False, item.read_error)
            return
        key = (item.handle, item.locale if item.ext == "stl" else None)
        if key in self.exported[item.ext]:
            cov["skipped"] += 1
            self.note(item, "duplicate_differs", False, "another file of this group has the same handle",
                      locale=self.row_locale(item, item.copy))
            return
        try:
            result = job(item)
        except FormatError as exc:
            cov["skipped"] += 1
            self.note(item, "parse_error", False, str(exc))
            return
        except Exception as exc:  # a reader fault: stop the whole export (exit code 1)
            raise _Stopped({"group": item.ext, "handle": item.handle, "archive": item.copy.archive,
                            "error": f"{type(exc).__name__}: {exc}"}) from exc
        if result is None:
            return  # the job noted the skip
        self.exported[item.ext].add(key)
        cov["exported"] += 1
        for differ in item.differs:
            self.note(item, "duplicate_differs", True, differ.get("message"), archive=differ["archive"],
                      copy=differ["copy"], copy_hash=differ["hash"], locale=differ["locale"])

    def skip(self, item: _Item, reason: str, message: str | None = None, **extra) -> None:
        self.coverage[item.ext]["skipped"] += 1
        self.note(item, reason, False, message, **extra)

    # --- groups --------------------------------------------------------------------------------
    def items_of(self, ext: str):
        return [self.items[ext][k] for k in sorted(self.items[ext])]

    def run(self) -> None:
        self.tex_handles = set()
        self.tex_scales: dict[int, float] = {}
        for item in self.items_of("tex"):
            self.export_item(item, self.job_texture)
        for item in self.items_of("app"):
            self.export_item(item, self.job_mesh)
        self.link_animations()
        for item in self.items_of("ani"):
            self.export_item(item, self.job_animation)
        json_jobs = {"acr": lambda d, b, i: _actor_json(d, b), "mon": lambda d, b, i: _monster_json(d, b),
                     "mat": lambda d, b, i: _material_json(d, b), "lvl": lambda d, b, i: _level_area_json(d, b),
                     "mrk": lambda d, b, i: _marker_set_json(d, b), "phm": lambda d, b, i: _collision_json(d, b)}
        for ext in ("acr", "mon", "mat"):
            for item in self.items_of(ext):
                self.export_item(item, self.json_job(json_jobs[ext]))
        for item in self.items_of("scn"):
            self.export_item(item, self.job_scene)
        for item in self.items_of("wrl"):
            self.export_item(item, self.job_world)
        for ext in ("lvl", "mrk", "phm"):
            for item in self.items_of(ext):
                self.export_item(item, self.json_job(json_jobs[ext]))
        for item in self.items_of("stl"):
            self.export_item(item, self.job_strings)

    def json_job(self, make):
        def job(item: _Item):
            document = make(item.data, item.build, item)
            self.emit(item, [(f"data/{item.ext}/{item.handle}.json", lambda: dumps(document), {})])
            return True
        return job

    def job_texture(self, item: _Item):
        tex = texture.parse_texture(item.data, item.build)
        self.tex_scales[item.handle] = tex.lightmap_v_scale  # ``43`` W19-2
        if item.build == "x360":
            from . import x360_texture
            decode = x360_texture.decode_image
        else:
            decode = texture.decode_image
        outputs = []
        extra_all = {"normal_map": True} if tex.format == texture.FORMAT_NORMAL else {}
        for level in (range(tex.levels) if self.all_levels else (0,)):
            depth = tex.level_depth(level)
            for face in range(tex.faces):
                for z in range(depth):
                    suffix = (f"_l{level}" if level else "") + (f"_f{face}" if tex.faces > 1 else "") + \
                             (f"_s{z}" if depth > 1 else "")
                    extra = {"level": level, "face": face, "slice": z, **extra_all}
                    outputs.append((f"textures/{item.handle}{suffix}.png",
                                    lambda level=level, face=face, z=z: texture.encode_png(decode(tex, level, face, z)),
                                    extra))
        self.emit(item, outputs)
        self.tex_handles.add(item.handle)
        return True

    def job_mesh(self, item: _Item):
        app = appearance.parse_appearance(item.data, item.build)
        # W18-4 diffuse textures and ``43`` W19-2 lightmaps of the sub-objects that are written
        wanted = sorted({h for sub in appearance.export_sub_objects(app, True)
                         for h in [material.sub_object_texture(app, sub, 0)]
                         + ([material.sub_object_texture(app, sub, 0, material.SLOT_LIGHTMAP)]
                            if any(d.positions for d in sub.draws) else [])
                         if h is not None and h != -1})
        # ``44`` W20-3: the layer textures that the materials name
        wanted = sorted(set(wanted) | {h for sub in appearance.export_sub_objects(app, True)
                                       if any(d.positions for d in sub.draws)
                                       for h in self.layer_textures(app, sub)})
        self.emit(item, [(f"meshes/{item.handle}.glb",
                          lambda: _mesh_glb(app, item.handle, self.lightmap, self.uv2_rows), {})])
        missing = [h for h in wanted if h not in self.tex_handles]
        if missing:
            self.note(item, "missing_dependency", True, None, needs=[_link(44, h) for h in missing])
        return True

    def shader_names(self, shader_map: int):
        """``44`` W20-1: the shader names of every use of a shader map (None for a shader without a
        name in the index), or None when the handle is -1 or the shader map does not resolve."""
        if not hasattr(self, "_shader_maps"):
            self._shader_maps = self.by_handle("shm")
            self._shader_names: dict[int, tuple | None] = {}
        if shader_map == -1 or shader_map not in self._shader_maps:
            return None
        if shader_map not in self._shader_names:
            item = self._shader_maps[shader_map]
            try:
                links = shader.shader_map_links(shader.read_shader_map(item.data, item.build))
            except (FormatError, struct.error, ValueError):
                self._shader_names[shader_map] = None
            else:
                self._shader_names[shader_map] = tuple(
                    self.names.get((shader.GROUP_SHADERS, h & 0xFFFFFFFF))
                    for _, h in sorted(links.items()) if h != -1)
        return self._shader_names[shader_map]

    def uv2_rows(self, block):
        """``44`` W20-1: the table rows of a material block's shaders, those of the archives' build (``45`` X6)."""
        names = None if block is None else self.shader_names(block.shader_map)
        return uv2_use.rows_for(names, uv2_use.tables(), self.build)

    def layer_textures(self, app, sub) -> list[int]:
        block = material.sub_object_block(app, sub, 0)
        return uv2_use.layer_texture_handles(self.uv2_rows(block), block)

    def lightmap(self, handle: int):
        """``43`` W19-2: (URI, V scale) of an exported lightmap texture, else None."""
        if handle not in self.tex_handles:
            return None
        return _texture_uri(handle), self.tex_scales.get(handle, 0.0)

    def link_animations(self) -> None:
        """W18-5, W18-6: animation handle -> the first exported appearance (by handle) whose
        actor's animation set (with its parents) links it. ``.acr`` and ``.ans`` are read even
        when ``--groups`` leaves out their JSON."""
        self.ani_app: dict[int, int] = {}
        linked: dict[int, set[int]] = {}
        exported_apps = {h for h, _ in self.exported["app"]}
        sets = {}
        for handle, item in self.by_handle("ans").items():
            try:
                sets[handle] = anims.parse_anim_set(item.data, item.build)
            except FormatError:
                continue

        def animations(set_handle: int) -> set[int]:
            found, seen = set(), set()
            while set_handle in sets and set_handle not in seen:
                seen.add(set_handle)
                anim_set = sets[set_handle]
                for tag_map in (anim_set.default, *anim_set.weapons):
                    for record in tag_map.records:
                        if isinstance(record.value, int) and record.value != -1:
                            found.add(record.value)
                set_handle = anim_set.parent
            return found

        for _, actor in sorted(self.by_handle("acr").items()):
            try:
                rec = records.parse_record_file(actor.data, "acr", actor.build).record
            except FormatError:
                continue
            app, anim_set = rec.get("appearance_ref"), rec.get("anim_set_ref")
            if app not in exported_apps or anim_set in (None, -1):
                continue
            for ani in animations(anim_set):
                linked.setdefault(ani, set()).add(app)
        self.app_items = self.by_handle("app")
        bones: dict[int, bool] = {}

        def has_bones(app: int) -> bool:
            if app not in bones:
                item = self.app_items[app]
                try:
                    bones[app] = bool(appearance.parse_appearance(item.data, item.build).bones)
                except FormatError:
                    bones[app] = False
            return bones[app]

        # ``42`` V27-1: the first linked appearance, by handle, that has at least one bone
        self.ani_unboned = set()
        for ani, apps in linked.items():
            chosen = next((app for app in sorted(apps) if has_bones(app)), None)
            if chosen is None:
                self.ani_unboned.add(ani)
            else:
                self.ani_app[ani] = chosen

    def job_animation(self, item: _Item):
        app_handle = self.ani_app.get(item.handle)
        if app_handle is None and item.handle in self.ani_unboned:
            self.skip(item, "missing_dependency", "no linked appearance has a skeleton (V27-1)")
            return None
        if app_handle is None:
            self.skip(item, "missing_dependency", "no exported appearance links this animation (W18-5)")
            return None
        can = self.companion(item, "can") if item.build in ("ps3", "x360") else None
        ani = animation.parse_ani(item.data, item.build, can)
        if not any(p.has_data for p in ani.permutations):
            self.skip(item, "missing_dependency", "the animation data (.can) is not in the selected archives"
                      if item.build in ("ps3", "x360") else "no permutation holds animation data")
            return None
        source = self.app_items[app_handle]
        app = appearance.parse_appearance(source.data, source.build)
        self.emit(item, [(f"anims/{item.handle}.glb", lambda: _animation_glb(app, ani, item.handle, app_handle, self.travel), {})])
        return True

    def job_scene(self, item: _Item):
        nav = self.companion(item, "nav") if item.build in ("ps3", "x360") else None
        document, lacking = _scene_json(item.data, item.build, nav)
        self.emit(item, [(f"data/scn/{item.handle}.json", lambda: dumps(document), {})])
        if lacking:
            self.note(item, "missing_dependency", True, "the scene's .nav is not in the selected archives")
        return True

    def job_world(self, item: _Item):
        if not hasattr(self, "_scene_items"):
            self._scene_items, self._set_items = self.by_handle("scn"), self.by_handle("mrk")
            self._scene_cache, self._set_cache = {}, {}

        def lookup(items, cache, parse):
            def get(handle):
                if handle not in cache:
                    found = items.get(handle)
                    try:
                        cache[handle] = None if found is None else parse(found.data, found.build)
                    except FormatError:
                        cache[handle] = None
                return cache[handle]
            return get

        document, missing = _world_json(
            item.data, item.build,
            lookup(self._scene_items, self._scene_cache, world.parse_scene_info),
            lookup(self._set_items, self._set_cache, world.parse_marker_set))
        self.emit(item, [(f"data/wrl/{item.handle}.json", lambda: dumps(document), {})])
        if missing:
            self.note(item, "missing_dependency", True, None, needs=[{"group": g, "handle": h} for g, h in missing])
        return True

    def job_strings(self, item: _Item):
        document = _strings_json(item.data, item.build)
        locale = item.locale
        self.emit(item, [(f"data/text/{locale}/{item.handle}.json", lambda: dumps(document), {"locale": locale})])
        return True

    # --- results -----------------------------------------------------------------------------
    def manifest(self, created: str) -> dict:
        entries = sorted(self.entries, key=lambda e: (e["group"], -1 if e["handle"] is None else e["handle"], e["path"]))
        return {
            "build": self.build, "coordinates": dict(COORDINATES),  # ``43`` W19-1
            "created": created, "entries": entries, "format": FORMAT, "format_version": FORMAT_VERSION,
            "library_version": __version__,
            "options": {"archives": self.patterns, "groups": list(self.groups), "all_levels": bool(self.all_levels),
                        "travel": self.travel},
        }

    def coverage_dict(self, stopped: dict | None) -> dict:
        groups = {}
        for ext, cov in self.coverage.items():
            files = sorted(cov["files"], key=lambda f: (-1 if f["handle"] is None else f["handle"], f["archive"],
                                                        f.get("locale") or "", f["reason"], f.get("copy", -1)))
            groups[ext] = {**cov, "files": files}
        totals = {k: sum(c[k] for c in self.coverage.values()) for k in ("found", "unique", "exported", "skipped")}
        return {"archives": [{"archive": stem, "build": cpk.header.build, "files": len(cpk.files)}
                             for stem, cpk in self.archives],
                "format": COVERAGE_FORMAT, "format_version": COVERAGE_FORMAT_VERSION, "groups": groups,
                "reasons": list(REASONS), "stopped": stopped, "totals": totals, "unread": self.unread}

    def summary(self) -> list[str]:
        lines = []
        for ext in EXPORT_GROUPS:
            c = self.coverage[ext]
            lines.append(f"{ext}: found {c['found']}, exported {c['exported']}, skipped {c['skipped']}")
        t = {k: sum(c[k] for c in self.coverage.values()) for k in ("found", "exported", "skipped")}
        lines.append(f"total: found {t['found']}, exported {t['exported']}, skipped {t['skipped']}, "
                     f"files written {len(self.entries)}")
        return lines


def prune_empty(value):
    """``42`` V27-1: drop every key whose value is an empty array, at any depth (an exported glTF
    file holds no empty array)."""
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            item = prune_empty(item)
            if not (isinstance(item, list) and not item):
                out[key] = item
        return out
    if isinstance(value, list):
        return [prune_empty(item) for item in value]
    return value


def _texture_uri(handle: int) -> str:
    return f"../textures/{handle}.png"


def _export_extras(document: dict, handle: int, app: appearance.AppearanceFile, extra: dict) -> None:
    """``43`` W19-1, W19-4: ``uv2_scale`` per primitive (from the primitive's own extras), the V
    direction and tangents as statements."""
    primitives = []
    for m, mesh in enumerate(document.get("meshes", [])):
        info = mesh.get("extras", {}).get("d3sno", {})
        for p, primitive in enumerate(mesh["primitives"]):
            own = primitive.get("extras", {}).get("d3sno_export", {})
            entry = {"mesh": m, "primitive": p, "material": info.get("material"),
                     "uv2_scale": own.get("uv2_scale", OPEN)}
            if "uv2_use" in own:  # ``44`` W20-2
                entry["uv2_use"] = own["uv2_use"]
            primitives.append(entry)
    document.setdefault("extras", {})["d3sno_export"] = {
        "handle": handle, "primitives": primitives, "v_direction": V_DIRECTION, "tangents": TANGENTS,
        "tangent_note": TANGENT_NOTE, "rotation": ROTATION, **extra}


def _no_lightmap(handle: int):
    return None


def _no_rows(block):
    return uv2_use.rows_for(None, None)


def _mesh_glb(app, handle: int, lightmap=None, uv2_rows=None) -> bytes:
    """W18-4: ``appearance.to_gltf`` with default options, diffuse textures by URI, packed as .glb.
    ``43`` W19-2: ``lightmap(handle)`` gives (URI, V scale) of an exported lightmap, else None.
    ``44`` W20-1: ``uv2_rows(block)`` gives the table rows of a material block's shaders."""
    document = appearance.to_gltf(app, texture_resolver=_texture_uri, lightmap_resolver=lightmap or _no_lightmap,
                                  uv2_rows=uv2_rows or _no_rows)
    _export_extras(document, handle, app, {})
    return gltf_to_glb(prune_empty(document))


def _animation_glb(app, ani, handle: int, app_handle: int, travel: bool = False) -> bytes:
    """W18-5 with ``41`` V26-1, V26-3: the joint nodes of the linked mesh file (names, order, the
    V26-2 root node) and every permutation that holds data as one glTF animation each, sampled as
    ``animation.to_gltf_animated`` samples one; no mesh, skin or images. Bones as stored: the
    animations are in place (``43`` W19-5), each permutation's average velocity is in the extras.
    Each (node, path) has one channel; the tracks left out are listed.

    ``travel`` (W19-5, off by default): one added node ``d3sno_travel`` above the scene's root
    nodes, whose translation moves by ``average_velocity * 60 * t``."""
    by_hash = animation._first_bone_by_hash(app.bones)
    listed, dropped = [], []

    def extend(buf, document: dict) -> None:
        out, moving = [], []
        for index_, perm in enumerate(ani.permutations):
            entry = {"index": index_, "name": perm.name, "animation": None,
                     "average_velocity": list(perm.average_velocity)}
            if any(any(c != 0 for c in row) for row in perm.nonlinear_offsets):
                entry["nonlinear_offsets"] = [list(row) for row in perm.nonlinear_offsets]
            listed.append(entry)
            if not perm.has_data or not perm.fps > 0:
                continue
            curves = animation.decode_permutation(perm)
            anim: dict = {"name": perm.name, "samplers": [], "channels": []}
            for drop in animation._permutation_channels(buf, anim, perm, curves, by_hash):
                dropped.append({"permutation": index_, **drop})
            if anim["channels"]:
                entry["animation"] = len(out)
                out.append(anim)
                moving.append(perm)
        if travel:
            _travel_node(buf, document, out, moving)
        if out:
            document["animations"] = out

    document = appearance.to_gltf_skeleton(app, extend=extend)
    document.setdefault("extras", {})["d3sno_export"] = {
        "handle": handle, "appearance": app_handle, "mesh": f"meshes/{app_handle}.glb", "root_motion": ROOT_MOTION,
        "rotation": ROTATION, "ticks_per_second": TICKS_PER_SECOND, "permutations": listed, "dropped_tracks": dropped,
        **({"travel": True} if travel else {})}
    return gltf_to_glb(prune_empty(document))


def _travel_node(buf, document: dict, animations: list, perms: list) -> None:
    """``43`` W19-5: the ``d3sno_travel`` node (no transform, ``extras.d3sno_export.added``) as the
    parent of the scene's root nodes, and per animation one linear translation channel from 0 at
    time 0 to ``average_velocity * 60 * T`` at the animation's last key time T."""
    nodes = document.setdefault("nodes", [])
    scene = document["scenes"][0]
    travel = len(nodes)
    nodes.append({"name": TRAVEL_NODE, "children": list(scene.get("nodes", [])),
                  "extras": {"d3sno_export": {"added": True}}})
    scene["nodes"] = [travel]
    for anim, perm in zip(animations, perms):
        end = max((buf.accessors[s["input"]].get("max", [0.0])[0] for s in anim["samplers"]), default=0.0)
        times = [0.0, end] if end > 0 else [0.0]
        values = [tuple(c * TICKS_PER_SECOND * t for c in perm.average_velocity) for t in times]
        animation._add_channel(buf, anim, travel, "translation", times, values, "VEC3")


def _write(target: Path, blob: bytes) -> None:
    """W18-2: written under a temporary name in the same directory, renamed when complete."""
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.part")
    try:
        with open(temporary, "wb") as handle:
            handle.write(blob)
        os.replace(temporary, target)
    except BaseException:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise


def export_game(game_dir, out_dir, *, build=None, archives=None, groups=None, all_levels=False,
                skip_existing=False, force=False, created: str | None = None, travel=False) -> ExportResult:
    """W18-10: export the archives of ``game_dir`` into ``out_dir`` (W18-1 to W18-9, ``43`` W19-1
    to W19-5; ``travel`` adds the W19-5 travel node to every animation file)."""
    game = Path(os.path.abspath(game_dir))
    out = Path(os.path.abspath(out_dir))
    run = None
    try:
        if not game.is_dir():
            raise ExportRefused(f"{game_dir}: not a directory")
        if _inside(out, game):
            raise ExportRefused("the output directory may not be the game directory or inside it")
        if out.exists() and not out.is_dir():
            raise ExportRefused(f"{out_dir}: not a directory")
        tree = in_git_tree(out)
        if tree is not None and not force:
            raise ExportRefused("the output directory is inside a git working tree; game content is never put "
                                "into a repository (pass --force to export there anyway)")
        if build is not None and build not in ("ps3", "switch", "x360"):
            raise ExportRefused(f"--build {build!r}: give ps3, switch or x360")
        chosen = _groups(groups)
        run = _Run(game, out, archives, chosen, all_levels, skip_existing, build, travel)
        run.open_archives()
        run.read_indexes()
        run.list_files()
    except ExportRefused as exc:
        if run is not None:
            run.close()
        return ExportResult(None, None, EXIT_REFUSED, str(exc))
    stopped = None
    try:
        out.mkdir(parents=True, exist_ok=True)
        run.load_old_manifest()
        try:
            run.run()
        except _Stopped as exc:
            stopped = exc.info
        except OSError as exc:
            stopped = {"error": f"{type(exc).__name__}: {exc}"}
        stamp = created or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        manifest = run.manifest(stamp)
        coverage = run.coverage_dict(stopped)
        _write(out / "manifest.json", dumps(manifest))
        _write(out / "coverage.json", dumps(coverage))
    finally:
        run.close()
    code = EXIT_STOPPED if stopped else EXIT_OK
    message = None if not stopped else f"export stopped: {stopped.get('error')}"
    return ExportResult(manifest, coverage, code, message, run.summary())
