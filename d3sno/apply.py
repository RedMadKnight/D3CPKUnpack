"""Stage 21 (``46_stage21_apply.md``): ``apply`` takes the game folder and an export folder that a person
has edited (stage 18, ``40``), finds what changed (W21-1, W21-2), imports each changed mesh, texture and
string table onto its template (W21-3 to W21-5, through the importers of ``34`` and the ``.stl`` helpers
of ``12``), and writes new archives (W21-6) with a report (W21-7). What did not change is kept byte for
byte. Stage 22 (``49``): the JSON records, world, collision and animation files too (W22-1 to W22-12,
``json_import`` and ``anim_import``).

The output is game content, as in ``40``: it is made on the user's computer only and must never be put
into a repository, a test fixture or a release; an output folder inside a git working tree is refused
unless ``force`` is given.

``47`` Y1: archives are written one at a time, each streamed into its file as it is built (files that
are not replaced are read from the memory-mapped source in pieces); ``--copy-unchanged`` copies in
chunks. The report names each written archive with its size and SHA-256 (``archives``).
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from . import __version__
from . import appearance, export, texture
from .binary import FormatError
from .effects import GROUPS as EFFECT_GROUPS

__all__ = [
    "ApplyResult", "EXIT_OK", "EXIT_REFUSED", "EXIT_STOPPED", "KINDS", "REPORT_FORMAT", "REPORT_FORMAT_VERSION",
    "REPORT_NAME", "apply_export", "kind_of",
]

EXIT_OK, EXIT_STOPPED, EXIT_REFUSED = export.EXIT_OK, export.EXIT_STOPPED, export.EXIT_REFUSED
REPORT_NAME = "apply_report.json"
REPORT_FORMAT = "d3sno-apply-report"
REPORT_FORMAT_VERSION = 1
KINDS = ("meshes", "textures", "text", "records", "world", "collision", "anims")  # W21-4, W22-11: ``--only``
# W22-11: the JSON groups of each kind (records are applied after meshes and textures)
JSON_KINDS = {"acr": "records", "mon": "records", "mat": "records", "lvl": "records", "scn": "world",
              "wrl": "world", "mrk": "world", "phm": "collision",
              **{ext: "records" for ext in EFFECT_GROUPS}}  # ``55`` W23-8: the effect groups are records
NEVER_APPLIED = ("manifest.json", "coverage.json")  # W21-2
# results of a changed entry (W21-7)
RESULTS = ("applied", "no_effect", "not_importable", "filtered", "refused")
FIELDS_LISTED = 200  # W22-11
MESH_OPTIONS = {"flip_v": False, "lightmap_v_scale": None, "missing": "keep", "skin": "refuse", "low_poly": True}


class ApplyRefused(Exception):
    """W21-8: a refusal (exit code 2)."""


@dataclass
class ApplyResult:
    report: dict | None
    exit_code: int
    message: str | None = None
    summary: list = field(default_factory=list)


def kind_of(entry: dict) -> str | None:
    """W21-4: ``meshes``, ``textures`` or ``text`` for an importable manifest entry, else None."""
    path, group = entry.get("path", ""), entry.get("group")
    if group == "app" and path.startswith("meshes/") and path.endswith(".glb"):
        return "meshes"
    if group == "tex" and path.startswith("textures/") and path.endswith(".png"):
        return "textures"
    if group == "stl" and path.startswith("data/text/") and path.endswith(".json"):
        return "text"
    if group in JSON_KINDS and path == f"data/{group}/{entry.get('handle')}.json":  # W22-11
        return JSON_KINDS[group]
    if group == "ani" and path.startswith("anims/") and path.endswith(".glb"):
        return "anims"
    return None


def _only(only) -> tuple:
    if only is None:
        return KINDS
    if isinstance(only, str):
        only = only.split(",")
    chosen = []
    for kind in only:
        kind = kind.strip().lower()
        if not kind:
            continue
        if kind not in KINDS:
            raise ApplyRefused(f"--only: {kind!r} is not one of {', '.join(KINDS)}")
        if kind not in chosen:
            chosen.append(kind)
    if not chosen:
        raise ApplyRefused("--only names no kind of file")
    return tuple(k for k in KINDS if k in chosen)


def _mesh_options(given) -> dict:
    options = dict(MESH_OPTIONS)
    for key, value in (given or {}).items():
        if key not in options:
            raise ApplyRefused(f"unknown mesh option {key!r}")
        options[key] = value
    if options["missing"] not in ("keep", "empty"):
        raise ApplyRefused(f"--missing {options['missing']!r}: give keep or empty")
    if options["skin"] not in ("refuse", "nearest"):
        raise ApplyRefused(f"--skin {options['skin']!r}: give refuse or nearest")
    return options


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _export_file(root: Path, path) -> Path | None:
    """The file of a manifest ``path`` below the export folder, or None when the path leaves it."""
    if not isinstance(path, str) or not path:
        return None
    pure = PurePosixPath(path)
    if pure.is_absolute() or any(part in ("", ".", "..") for part in pure.parts) or ":" in path or "\\" in path:
        return None
    return root.joinpath(*pure.parts)


def _rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _refused_rows(rows: list) -> dict:
    """``51`` AB6: at most ``FIELDS_LISTED`` refused paths: the first row (the one the entry's reason and
    message come from), then the others in path order; with the total and the count of each reason."""
    rows = list(rows)
    by_reason: dict[str, int] = {}
    for row in rows:
        by_reason[row.get("reason")] = by_reason.get(row.get("reason"), 0) + 1
    listed = rows[:1] + sorted(rows[1:], key=lambda r: str(r.get("path")))
    return {"refused_fields": listed[:FIELDS_LISTED], "refused_fields_total": len(rows),
            "refused_by_reason": dict(sorted(by_reason.items()))}


class _Refusal(Exception):
    """One entry is refused (W21-3, W21-4); the others go on. ``fields``: W22-11, every refused path."""

    def __init__(self, reason: str, message: str | None = None, fields=None):
        super().__init__(message or reason)
        self.reason, self.message, self.fields = reason, message, fields


@dataclass
class _Change:
    """One applied template: its new bytes, the archive entries to replace and the warnings."""
    data: bytes
    targets: list  # (archive base name, archive entry name)
    warnings: list
    fields: list | None = None  # W22-11: the changed paths with old and new values
    also: list = field(default_factory=list)  # W22-8, W22-10: other files written for this entry


class _Apply:
    def __init__(self, game: Path, root: Path, out: Path, manifest: dict, build, dry_run, copy_unchanged, only,
                 mesh, jobs=None, segment=False, profile=False):
        self.game, self.root, self.out = game, root, out
        self.jobs = jobs  # ``48`` Z4: workers per archive written (None: the number of processors)
        self.segment, self.profile = bool(segment), bool(profile)  # ``51`` AB7
        self.manifest, self.given_build = manifest, build
        self.dry_run, self.copy_unchanged, self.only, self.mesh = dry_run, copy_unchanged, only, mesh
        options = manifest.get("options") if isinstance(manifest.get("options"), dict) else {}
        self.options = options
        groups = options.get("groups") or None
        try:
            chosen = export._groups(groups)
        except export.ExportRefused as exc:
            raise ApplyRefused(f"manifest.json: {exc}") from None
        self.run = export._Run(game, out, options.get("archives"), chosen, bool(options.get("all_levels")),
                               False, build, bool(options.get("travel")))
        self.rows: list[dict] = []
        self.counts = {k: 0 for k in ("unchanged", "changed", "applied", "missing", "unmanaged", "not_importable",
                                      "refused", "filtered", "no_effect")}
        self.missing: list[str] = []
        self.unmanaged: list[str] = []
        self.replacements: dict[str, dict[str, bytes]] = {}  # archive base name -> entry name -> bytes
        self.templates: dict[tuple, _Change | _Refusal] = {}
        # ``51`` AB3, ``52`` AC6: (group, handle) -> the parsed scene info or marker set of this run's new bytes,
        # only for the scenes and marker sets the run's worlds refer to (``layout_refs``)
        self.layout_parts: dict[tuple, object] = {}
        self.layout_refs: set[tuple] = set()

    # --- W21-2 ---------------------------------------------------------------------------------
    def open(self) -> None:
        try:
            self.run.open_archives()
        except export.ExportRefused as exc:
            raise ApplyRefused(str(exc)) from None
        if self.manifest.get("build") != self.run.build:
            raise ApplyRefused(f"the export is of build {self.manifest.get('build')!r}, the game folder of "
                               f"{self.run.build!r}")
        self.run.read_indexes()
        self.run.list_files()

    def expected_hashes(self) -> dict[str, str]:
        """W21-1, W21-2: the output hash of each manifest path; for a manifest before format
        version 4, what this library's export writes for the same source file, in memory."""
        entries = [e for e in self.manifest.get("entries", []) if isinstance(e, dict)]
        version = self.manifest.get("format_version")
        if isinstance(version, int) and version >= 4:
            return {e["path"]: e.get("output_hash") for e in entries if isinstance(e.get("path"), str)}
        if self.manifest.get("library_version") != __version__:
            raise ApplyRefused(f"manifest.json of format version {version} has no output hashes and was written "
                               f"by library version {self.manifest.get('library_version')!r}; this is {__version__}: "
                               "export again with this version")
        memory = export._Run(self.game, self.out, self.options.get("archives"), self.run.groups,
                             self.run.all_levels, False, self.given_build, self.run.travel)
        memory.memory = {}
        try:
            memory.open_archives()
            memory.read_indexes()
            memory.list_files()
            memory.run()
        except export._Stopped as exc:
            raise ApplyRefused(f"the export in memory stopped: {exc.info.get('error')}") from None
        finally:
            memory.close()
        return {e["path"]: e["output_hash"] for e in memory.entries}

    def classify(self) -> list[tuple[dict, Path]]:
        hashes = self.expected_hashes()
        changed = []
        managed = set()
        for entry in sorted((e for e in self.manifest.get("entries", []) if isinstance(e, dict)),
                            key=lambda e: str(e.get("path"))):
            path = entry.get("path")
            target = _export_file(self.root, path)
            if target is None:
                self.counts["changed"] += 1
                self.row(entry, "refused", reason="bad_path", message="the path leaves the export folder")
                continue
            managed.add(path)
            if not target.is_file():
                self.counts["missing"] += 1
                self.missing.append(path)
                continue
            if _sha(target.read_bytes()) == hashes.get(path):
                self.counts["unchanged"] += 1
                continue
            self.counts["changed"] += 1
            changed.append((entry, target))
        for found in sorted(p for p in self.root.rglob("*") if p.is_file()):
            rel = _rel(found, self.root)
            if rel in NEVER_APPLIED or rel in managed:
                continue
            self.counts["unmanaged"] += 1
            self.unmanaged.append(rel)
        return changed

    # --- W21-3, W21-5 --------------------------------------------------------------------------
    def template(self, entry: dict):
        """(item, template bytes): the stored file named by ``source`` whose SHA-256 is ``source.hash``."""
        ext, handle, locale = entry.get("group"), entry.get("handle"), entry.get("locale")
        source = entry.get("source") if isinstance(entry.get("source"), dict) else {}
        archive_name, want = source.get("archive"), source.get("hash")
        for key in sorted(self.run.items.get(ext, {})):
            item = self.run.items[ext][key]
            if ext == "stl" and item.locale != locale:
                continue
            self.run.load(item)
            if item.handle != handle:
                continue
            for copy in sorted(item.copies, key=export._locale_rank):
                if copy.archive != archive_name or (ext != "stl" and copy.locale != locale):
                    continue
                for index in range(max(copy.entry.count, 1)):
                    try:
                        data = copy.cpk.read_file(copy.entry, index)
                    except FormatError:
                        continue
                    if _sha(data) == want:
                        return item, data
        raise _Refusal("template_changed", f"no file of group {ext} with handle {handle} in archive "
                                           f"{archive_name} has the exported hash: the game folder is not the one "
                                           "that was exported")

    def targets(self, item, entry: dict, template: bytes):
        """W21-5 with ``50`` AA3: every archive entry of the exported variant that holds the template. The
        repeated copies of one entry are one file: if any of them equals the template the entry is a target
        (the writer replaces all its copies, W14-6), with the warning ``repeated_copies_differed`` when not all
        of them did. An entry none of whose copies equals the template is not changed
        (``copy_differs_not_changed``)."""
        locale = entry.get("locale")
        targets, warnings = [], []
        for copy in sorted(item.copies, key=lambda c: (c.archive.lower(), c.entry.name or "")):
            if item.ext != "stl" and copy.locale != locale:
                continue  # another locale variant (V27-2): not changed
            equal = []
            for index in range(max(copy.entry.count, 1)):
                try:
                    equal.append(copy.cpk.read_file(copy.entry, index) == template)
                except FormatError:
                    equal.append(False)
            if any(equal):
                targets.append((copy.archive, copy.entry.name))
                if not all(equal):
                    warnings.append(f"repeated_copies_differed: archive {copy.archive}")
            else:
                warnings.append(f"copy_differs_not_changed: archive {copy.archive}")
        return targets, warnings

    # --- W21-4 ---------------------------------------------------------------------------------
    def import_mesh(self, template: bytes, target: Path) -> tuple[bytes, list]:
        from . import mesh_import

        app = appearance.parse_appearance(template, self.run.build)
        document = mesh_import.read_gltf(target.read_bytes())
        out, report = mesh_import.import_mesh_report(app, document, **self.mesh)
        return out, list(report.warnings)

    def import_texture(self, template: bytes, entry: dict) -> tuple[bytes, list]:
        from . import png, texture_import

        tex = texture.parse_texture(template, self.run.build)
        rows = sorted((e for e in self.manifest.get("entries", []) if isinstance(e, dict) and kind_of(e) == "textures"
                       and e.get("handle") == entry.get("handle") and e.get("source") == entry.get("source")),
                      key=lambda e: (e.get("level", 0), e.get("face", 0), e.get("slice", 0)))
        images = {}
        for row in rows:
            path = _export_file(self.root, row.get("path"))
            if path is None or not path.is_file():
                raise _Refusal("image_missing", f"{row.get('path')} is not in the export folder; every image of a "
                                                "texture is imported together")
            images[(row.get("level", 0), row.get("face", 0), row.get("slice", 0))] = png.read_png(path.read_bytes())
        levels = sorted({k[0] for k in images})
        if levels != list(range(len(levels))):
            raise _Refusal("image_missing", "the levels of the exported images are not 0, 1, ...")
        if tex.depth > 1:
            given = [images[(0, 0, z)] for z in range(tex.depth)]
        elif tex.faces > 1:
            given = [[images[(m, f, 0)] for m in levels] for f in range(tex.faces)]
        else:
            given = [images[(m, 0, 0)] for m in levels]
        return texture_import.import_texture(tex, given), []

    def import_text(self, template: bytes, target: Path) -> tuple[bytes, list]:
        from . import text, write

        try:
            document = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            raise _Refusal("bad_json", str(exc)) from None
        strings = document.get("strings") if isinstance(document, dict) else None
        if not isinstance(strings, dict):
            raise _Refusal("bad_json", "no \"strings\" object")
        wanted: dict[int, str] = {}
        for key, value in strings.items():
            if not (isinstance(key, str) and len(key) == 8 and all(c in "0123456789abcdefABCDEF" for c in key)) \
                    or not isinstance(value, str):
                raise _Refusal("bad_json", f"string key {key!r}: 8 hex digits and a text are needed")
            wanted[int(key, 16)] = value
        table = text.parse_string_list(template, self.run.build)
        for entry in list(table.entries):
            label = entry.label_hash
            if label not in wanted:
                if label in table.by_hash:
                    write.remove_entry(table, label)
            elif table.by_hash.get(label) != wanted[label]:
                write.set_text(table, label, wanted[label])
        for label in wanted:  # sorted keys, as the export writes them
            if label not in table.by_hash:
                write.add_entry(table, label, wanted[label])
        return write.write_stl(table), list(table.warnings)

    # --- W22 -------------------------------------------------------------------------------------
    def known(self, group_id: int, handle: int) -> bool:
        """W22-3: a handle that the game folder's index names."""
        return (group_id, handle & 0xFFFFFFFF) in self.run.names

    def companion_copies(self, item, folder_ext: str):
        """(archive base name, entry name, bytes) of the ``.nav`` / ``.can`` beside each copy of ``item``."""
        name = item.name.rpartition(".")[0] + f"\\0000.{folder_ext}"
        found = []
        for copy in item.copies:
            entry = copy.cpk.find(name)
            if entry is None:
                continue
            try:
                found.append((copy.archive, entry.name or name, copy.cpk.read_file(entry)))
            except FormatError:
                continue
        return found

    def scenes_and_sets(self, after: bool = False):
        """The scene and marker set lookups of a world's layout: the stored files, or (``after``, ``51``
        AB3) the new bytes of each scene and marker set this run writes and the stored bytes otherwise."""
        from . import world

        def lookup(ext, parse):
            items = self.run.by_handle(ext)
            cache = {}

            def get(handle):
                if handle not in cache:
                    found = items.get(handle)
                    held = self.layout_parts.get((ext, handle)) if after else None
                    try:
                        if held is not None:
                            cache[handle] = held
                        elif found is None:
                            cache[handle] = None
                        else:
                            cache[handle] = parse(found.data, found.build)
                    except FormatError:
                        cache[handle] = None
                return cache[handle]
            return get

        return lookup("scn", world.parse_scene_info), lookup("mrk", world.parse_marker_set)

    def import_json(self, item, template: bytes, entry: dict, target: Path):
        from . import json_import, sno_layout

        ext = entry.get("group")
        build = self.run.build
        if ext not in sno_layout.covered_types(item.build or build):
            raise _Refusal("not_importable", f"no tree layout for {item.build} .{ext} (W22-6)")
        try:
            edited = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            raise _Refusal("bad_json", str(exc)) from None
        nav = self.run.companion(item, "nav") if ext == "scn" and item.build in ("ps3", "x360") else None
        scenes, sets = self.scenes_and_sets() if ext == "wrl" else (None, None)
        after = self.scenes_and_sets(after=True) if ext == "wrl" else None
        try:
            result = json_import.import_json(ext, template, item.build, edited, nav=nav, scenes=scenes, sets=sets,
                                             known=self.known, after=after)
        except json_import.Refused as refused:
            rows = [r.as_row() for r in getattr(refused, "refusals", [refused])]
            raise _Refusal(refused.reason, refused.message, rows) from None
        also = []
        if result.nav is not None and nav is not None and result.nav != nav:
            also = [(stem, name, result.nav) for stem, name, data in self.companion_copies(item, "nav") if data == nav]
            if not also:  # ``50`` AA4
                raise _Refusal("no_copy_matches", "the scene's .nav changes, but no stored copy equals its template")
        return result.data, list(result.warnings), result.changes, also

    def import_anim(self, item, template: bytes, entry: dict, target: Path):
        from . import anim_import, json_import

        if not hasattr(self, "_ani_app"):
            self.run.exported["app"] = {(e.get("handle"), None) for e in self.manifest.get("entries", [])
                                        if isinstance(e, dict) and e.get("group") == "app"}
            self.run.link_animations()
            self._ani_app = dict(self.run.ani_app)
        app_handle = self._ani_app.get(item.handle)
        if app_handle is None:
            raise _Refusal("not_importable", "no exported appearance links this animation (W18-5)")
        app = self.run.app_items[app_handle]
        can = self.run.companion(item, "can") if item.build in ("ps3", "x360") else None
        try:
            result = anim_import.import_animation(template, item.build, app.data, app.build, target.read_bytes(),
                                                  item.handle, app_handle, can, self.run.travel)
        except json_import.Refused as refused:
            rows = [r.as_row() for r in getattr(refused, "refusals", [refused])]
            raise _Refusal(refused.reason, refused.message, rows) from None
        also = []
        if result.nav is not None and can is not None and result.nav != can:
            also = [(stem, name, result.nav) for stem, name, data in self.companion_copies(item, "can") if data == can]
            if not also:  # ``50`` AA4
                raise _Refusal("no_copy_matches", "the animation's .can changes, but no stored copy equals its template")
        return result.data, list(result.warnings), result.changes, also

    def change(self, entry: dict, target: Path, kind: str) -> _Change:
        key = (entry.get("group"), entry.get("handle"), entry.get("locale"), json.dumps(entry.get("source"), sort_keys=True))
        if kind == "textures" and key in self.templates:  # every image of a handle at once
            found = self.templates[key]
            if isinstance(found, _Refusal):
                raise found
            return found
        try:
            item, template = self.template(entry)
            fields, also = None, []
            try:
                if kind == "meshes":
                    data, warnings = self.import_mesh(template, target)
                elif kind == "textures":
                    data, warnings = self.import_texture(template, entry)
                elif kind == "text":
                    data, warnings = self.import_text(template, target)
                elif kind == "anims":
                    data, warnings, fields, also = self.import_anim(item, template, entry, target)
                else:
                    data, warnings, fields, also = self.import_json(item, template, entry, target)
            except (FormatError, ValueError, struct.error, KeyError) as exc:  # any other error: ``apply`` (AA1)
                raise _Refusal("import_failed", f"{type(exc).__name__}: {exc}") from None
            targets, copy_warnings = self.targets(item, entry, template)
            if data != template and not targets:  # ``50`` AA4: a new file that no copy can take
                raise _Refusal("no_copy_matches", "the import gives a new file, but every stored copy differs from "
                                                  "the template: " + "; ".join(copy_warnings))
            found = _Change(data, targets if data != template else [], [*warnings, *copy_warnings], fields, also)
        except _Refusal as refusal:
            if kind == "textures":
                self.templates[key] = refusal
            raise
        if kind == "textures":
            self.templates[key] = found
        if data != template:
            for stem, name in targets:
                self.replacements.setdefault(stem, {})[name] = data
            self.hold_for_layout(entry, item, data)  # ``51`` AB3, ``52`` AC6
        for stem, name, data_also in found.also:  # W22-8, W22-10: the .nav / .can of the entry
            self.replacements.setdefault(stem, {})[name] = data_also
        return found

    def layout_references(self, changed) -> None:
        """``52`` AC6: before anything is written, the scenes and marker sets the run's worlds refer to: the
        scenes of each world's scene entries and its marker sets (stored and edited), and the marker sets of
        those scenes (stored)."""
        from . import world

        scenes, _ = self.scenes_and_sets()
        refs: set[tuple] = set()
        for entry, target in changed:
            if entry.get("group") != "wrl" or kind_of(entry) not in self.only:
                continue
            try:
                _, template = self.template(entry)
                wrl = world.parse_world(template, self.run.build)
                refs |= {("scn", e.handle) for e in wrl.scene_entries} | {("mrk", h) for h in wrl.marker_sets}
            except (_Refusal, FormatError):
                pass
            try:
                edited = json.loads(Path(target).read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, ValueError):
                edited = None
            if isinstance(edited, dict):
                for row in edited.get("scene_entries") or []:
                    link = row.get("scene") if isinstance(row, dict) else None
                    if isinstance(link, dict) and isinstance(link.get("handle"), int):
                        refs.add(("scn", link["handle"]))
                for link in edited.get("marker_sets") or []:
                    if isinstance(link, dict) and isinstance(link.get("handle"), int):
                        refs.add(("mrk", link["handle"]))
        for ext, handle in sorted(refs):
            if ext == "scn":
                info = scenes(handle)
                if info is not None:
                    refs |= {("mrk", h) for h in info.marker_sets}
        self.layout_refs = refs

    def hold_for_layout(self, entry: dict, item, data: bytes) -> None:
        """``52`` AC6: of a scene or marker set the run writes, only what a world's layout needs (its parsed
        scene info or marker set, not its bytes), and only when a world of the run refers to it. A held scene
        adds its marker sets to the references."""
        from . import world

        key = (entry.get("group"), entry.get("handle"))
        if key not in self.layout_refs:
            return
        try:
            if key[0] == "scn":
                info = world.parse_scene_info(data, item.build)
                self.layout_refs |= {("mrk", h) for h in info.marker_sets}
                self.layout_parts[key] = info
            elif key[0] == "mrk":
                self.layout_parts[key] = world.parse_marker_set(data, item.build)
        except FormatError:
            self.layout_parts.pop(key, None)

    def row(self, entry: dict, result: str, **extra) -> None:
        if result in ("applied", "no_effect", "not_importable", "refused", "filtered"):
            self.counts[result] += 1
        row = {"path": entry.get("path"), "group": entry.get("group"), "handle": entry.get("handle"),
               "result": result, "archives": [], "warnings": []}
        row.update(extra)
        self.rows.append(row)

    def apply(self, changed) -> None:
        # W22-11: records after meshes and textures (every entry is a file of its own; the report keeps the
        # order); ``51`` AB3: worlds last, after every other entry of the run
        changed = sorted(changed, key=lambda pair: (pair[0].get("group") == "wrl",
                                                    KINDS.index(kind_of(pair[0])) if kind_of(pair[0]) in KINDS
                                                    else len(KINDS), str(pair[0].get("path"))))
        self.layout_references(changed)  # ``52`` AC6
        for entry, target in changed:
            kind = kind_of(entry)
            if kind is None:
                self.row(entry, "not_importable", reason="not_importable")
                continue
            if kind not in self.only:
                self.row(entry, "filtered", reason="filtered", message=f"--only leaves out {kind}")
                continue
            try:
                try:
                    found = self.change(entry, target, kind)
                except _Refusal:
                    raise
                except Exception as exc:  # ``50`` AA1: a fault in one entry refuses that entry only
                    raise _Refusal("import_failed", f"{type(exc).__name__}: {exc}") from None
            except _Refusal as refusal:
                extra = {"reason": refusal.reason}
                if refusal.message:
                    extra["message"] = refusal.message
                if refusal.fields is not None:  # W22-11, ``51`` AB6: the refused paths, bounded
                    extra.update(_refused_rows(refusal.fields))
                self.row(entry, "refused", **extra)
                continue
            archives = sorted({stem for stem, _ in found.targets} | {stem for stem, _, _ in found.also}, key=str.lower)
            extra = {}
            if found.fields is not None:  # W22-11, ``51`` AB6
                fields = sorted(found.fields, key=lambda r: str(r.get("path")))
                extra = {"fields_changed": len(fields), "fields": fields[:FIELDS_LISTED]}
                if len(fields) > FIELDS_LISTED:
                    extra["fields_total"] = len(fields)
            if found.also:
                extra["also_written"] = sorted({name for _, name, _ in found.also})
            self.row(entry, "applied" if found.targets or found.also else "no_effect", archives=archives,
                     warnings=list(found.warnings), **extra)

    # --- W21-6, Y1 ------------------------------------------------------------------------------
    def archive_jobs(self) -> list:
        """(output path relative to the game folder, source archive, replacements), by path."""
        jobs = [(_rel(self.run.archive_paths[stem], self.game), self.run.all_archives[stem], self.replacements[stem])
                for stem in self.replacements]
        return sorted(jobs, key=lambda job: job[0].lower())

    def write_archives(self) -> dict[str, dict]:
        """Y1: each changed archive streamed into its output file (a temporary name, renamed when
        every archive is complete; the dry run only counts and hashes). Returns path -> size and
        SHA-256. On a fault no archive is left."""
        from . import index_write
        from .archive import part_name

        written: dict[str, dict] = {}
        parts: list[tuple[str, Path]] = []
        try:
            for rel, cpk, replacements in self.archive_jobs():
                if self.dry_run:
                    done = index_write.write_changed(cpk, None, replacements, jobs=self.jobs, segment=self.segment)
                else:
                    target = self.out.joinpath(*PurePosixPath(rel).parts)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    temporary = part_name(target)
                    parts.append((temporary, target))
                    with open(temporary, "wb") as handle:
                        done = index_write.write_changed(cpk, handle, replacements, jobs=self.jobs,
                                                         segment=self.segment)
                written[rel] = {"layout": done.layout, "sha256": done.sha256, "size": done.size}  # ``51`` AB7
                replacements.clear()  # ``52`` AC6: the entries of this archive are let go once it is written
                if self.profile and done.workers is not None:
                    written[rel]["workers"] = done.workers
            for temporary, target in parts:
                os.replace(temporary, target)
        except BaseException:
            for temporary, _ in parts:
                try:
                    os.unlink(temporary)
                except OSError:
                    pass
            raise
        return written

    def report(self, written: list, copied: list, stopped, archives: dict | None = None) -> dict:
        return {
            "archives": archives or {},
            "archives_written": written, "build": self.run.build, "changed": sorted(self.rows, key=lambda r: str(r["path"])),
            "copied": copied, "counts": dict(self.counts), "dry_run": bool(self.dry_run), "format": REPORT_FORMAT,
            "format_version": REPORT_FORMAT_VERSION, "index_files_written": [], "library_version": __version__,
            "missing": self.missing, "options": {
                "copy_unchanged": bool(self.copy_unchanged), "mesh": dict(self.mesh), "only": list(self.only),
                "segment": self.segment},
            "stopped": stopped, "unmanaged": self.unmanaged,
            **({"profile": self.profile_stats()} if self.profile else {}),
        }

    def profile_stats(self) -> dict:
        """``52`` AC6: the scenes and marker sets held for the worlds' layouts and the peak memory of the
        process (``--profile``)."""
        return {"layout_held": len(self.layout_parts), "layout_referenced": len(self.layout_refs),
                "peak_memory_bytes": peak_memory()}


def peak_memory() -> int | None:
    """``52`` AC6: the peak memory of this process in bytes (the peak working set on Windows, the maximum
    resident set elsewhere), or None where it cannot be read."""
    try:
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            class Counters(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                            ("PeakPagefileUsage", ctypes.c_size_t)]

            counters = Counters()
            counters.cb = ctypes.sizeof(Counters)
            process = ctypes.windll.kernel32.GetCurrentProcess()
            if not ctypes.windll.psapi.GetProcessMemoryInfo(process, ctypes.byref(counters), counters.cb):
                return None
            return int(counters.PeakWorkingSetSize)
        import resource
        import sys

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(peak if sys.platform == "darwin" else peak * 1024)
    except Exception:  # no way to read it here
        return None


def _copy_file(source: Path, target: Path) -> None:
    """Y1: a byte-for-byte copy in chunks of at most ``archive.COPY_CHUNK`` bytes, under a temporary
    name renamed when complete."""
    from . import archive

    target.parent.mkdir(parents=True, exist_ok=True)

    def produce(out):
        with open(source, "rb") as handle:
            while True:
                chunk = handle.read(archive.COPY_CHUNK)
                if not chunk:
                    break
                out.write(chunk)

    archive.stream_to(target, produce)


def _refuse_paths(game: Path, root: Path, out: Path, force: bool) -> None:
    if not game.is_dir():
        raise ApplyRefused(f"{game}: the game folder is not a directory")
    if not root.is_dir():
        raise ApplyRefused(f"{root}: the export folder is not a directory")
    for other, what in ((game, "game"), (root, "export")):
        if export._inside(out, other):
            raise ApplyRefused(f"the output folder may not be the {what} folder or inside it")
    if out.exists() and not out.is_dir():
        raise ApplyRefused(f"{out}: not a directory")
    if export.in_git_tree(out) is not None and not force:
        raise ApplyRefused("the output folder is inside a git working tree; game content is never put into a "
                           "repository (pass --force to write there anyway)")


def _read_manifest(root: Path) -> dict:
    path = root / "manifest.json"
    if not path.is_file():
        raise ApplyRefused(f"{root}: no manifest.json; give a folder written by export")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise ApplyRefused(f"manifest.json: {exc}") from None
    if not isinstance(manifest, dict) or manifest.get("format") != export.FORMAT:
        raise ApplyRefused(f"manifest.json is not a {export.FORMAT} manifest")
    if not isinstance(manifest.get("entries"), list):
        raise ApplyRefused("manifest.json has no entries")
    return manifest


def apply_export(game_dir, export_dir, out_dir, *, build=None, dry_run=False, copy_unchanged=False, only=None,
                 mesh_options=None, force=False, jobs=None, segment=False, profile=False) -> ApplyResult:
    """W21-8: apply the changes of ``export_dir`` (an edited export of ``game_dir``) and write the
    changed archives and ``apply_report.json`` into ``out_dir``. Refusals give exit code 2 and no
    report; a stop (an unexpected fault) exit code 1 with the report. ``48`` Z4: ``jobs`` worker
    threads write each archive (None: the number of processors; 1: one thread); the archives are the
    same for any number. ``51`` AB7: ``segment`` writes the segmented layout; ``profile`` puts the use of
    the workers into the report (``archives``)."""
    game, root, out = (Path(os.path.abspath(p)) for p in (game_dir, export_dir, out_dir))
    job = None
    try:
        if build is not None and build not in ("ps3", "switch", "x360"):
            raise ApplyRefused(f"--build {build!r}: give ps3, switch or x360")
        _refuse_paths(game, root, out, force)
        manifest = _read_manifest(root)
        if jobs is not None and (isinstance(jobs, bool) or not isinstance(jobs, int) or jobs < 1):
            raise ApplyRefused(f"--jobs {jobs!r}: give a number of at least 1")
        job = _Apply(game, root, out, manifest, build, dry_run, copy_unchanged, _only(only),
                     _mesh_options(mesh_options), jobs, segment, profile)
        job.open()
        changed = job.classify()
    except ApplyRefused as exc:
        if job is not None:
            job.run.close()
        return ApplyResult(None, EXIT_REFUSED, str(exc))
    stopped = None
    written: list[str] = []
    copied: list[str] = []
    try:
        out.mkdir(parents=True, exist_ok=True)
        try:
            job.apply(changed)
            archives = job.write_archives()
        except Exception as exc:  # an importer or writer fault: stop (exit code 1)
            stopped = {"error": f"{type(exc).__name__}: {exc}"}
            archives = {}
        written = sorted(archives, key=str.lower)
        if job.copy_unchanged and stopped is None:
            copied = sorted((_rel(p, game) for p in game.rglob("*") if p.is_file() and _rel(p, game) not in archives),
                            key=str.lower)
        if not dry_run and stopped is None:
            for rel in copied:
                _copy_file(game.joinpath(*PurePosixPath(rel).parts), out.joinpath(*PurePosixPath(rel).parts))
        report = job.report(written, copied, stopped, archives)
        export._write(out / REPORT_NAME, export.dumps(report))
    finally:
        job.run.close()
    counts = report["counts"]
    summary = [f"{k}: {counts[k]}" for k in sorted(counts)] + [f"archives written: {len(written)}"
                                                               + (" (dry run, none written)" if dry_run else "")]
    summary.append(f"layout: {'segmented' if segment else 'default'}")  # ``51`` AB7
    if profile:
        from .archive import workers_text

        for rel in written:
            stats = report["archives"].get(rel, {}).get("workers")
            if stats:
                summary.append(f"{rel} workers: " + workers_text(stats))
        held = report.get("profile") or {}  # ``52`` AC6
        summary.append(f"layout held: {held.get('layout_held')} of {held.get('layout_referenced')} referenced; "
                       f"peak memory: {held.get('peak_memory_bytes')} bytes")
    code = EXIT_STOPPED if stopped else EXIT_OK
    return ApplyResult(report, code, None if not stopped else f"apply stopped: {stopped['error']}", summary)
