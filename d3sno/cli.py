"""Command line interface: ``python -m d3sno <command>``."""

from __future__ import annotations

import argparse
import fnmatch
import importlib
import os
import re
import sys
from pathlib import Path

from . import (
    animation, appearance, archive, balance, collision, formula, index, power, text, texture,
    world,
)
from .binary import FormatError
from .sno import group_by_id


def _read(path: str) -> bytes:
    return Path(path).read_bytes()


class _MappedArchive:
    """``47`` Y1: an archive file parsed from a read-only memory map (not read into memory), for the
    commands that write a changed archive; closed when the ``with`` block ends."""

    def __init__(self, path) -> None:
        self.path = path

    def __enter__(self):
        import mmap

        self.handle = open(self.path, "rb")
        try:
            self.view = mmap.mmap(self.handle.fileno(), 0, access=mmap.ACCESS_READ)
        except ValueError:  # an empty file cannot be mapped
            self.view = None
            return archive.read_archive(self.handle.read(), strict=False)
        return archive.read_archive(self.view, strict=False)

    def __exit__(self, *exc) -> None:
        if self.view is not None:
            self.view.close()
        self.handle.close()


def _print(text: str = "") -> None:
    """Print one line of output, replacing what the console cannot show."""
    stream = sys.stdout
    encoding = getattr(stream, "encoding", None) or "utf-8"
    stream.write(text.encode(encoding, "backslashreplace").decode(encoding) + "\n")


def cmd_info(args) -> int:
    cpk = archive.read_archive(_read(args.archive), strict=False)
    h = cpk.header
    order = "big-endian" if h.order == ">" else "little-endian"
    _print(f"version        {h.version} ({order})")
    _print(f"flags          0x{h.flags:X}")
    _print(f"files          {h.file_count}")
    _print(f"locations      {h.location_count}")
    _print(f"stream size    {h.stream_size}")
    _print(f"sector region  0x{h.start:X}")
    _print(f"read sector    0x{h.read_sector:X}")
    _print(f"comp sector    0x{h.comp_sector:X}")
    _print(f"comp sectors   {h.comp_sector_count}")
    _print(f"decomp sectors {h.decomp_sector_count}")
    _print(f"widths         size {h.size_bits}, copies {h.copies_bits}, "
           f"first {h.first_location_bits}, location {h.location_bits}, "
           f"sector index {h.sector_index_bits}")
    _print(f"crc            0x{h.crc:08X}")
    return 0


def cmd_verify(args) -> int:
    report = archive.verify_archive(_read(args.archive), allow_gaps=args.allow_gaps)
    _print(report.format())
    return 0 if report.ok else 1


def cmd_list(args) -> int:
    cpk = archive.read_archive(_read(args.archive), strict=False)
    for entry in cpk.files:
        name = entry.name if entry.name is not None else "?"
        if args.pattern and not fnmatch.fnmatch(name.lower(), args.pattern.lower()):
            continue
        _print(f"{entry.hash:016X} {entry.size:>10} {entry.count:>3} {name}")
    return 0


def output_path(root: Path, name: str) -> Path:
    """Map an archive name to a path below ``root``. ``|locale`` becomes ``@locale``."""
    parts = name.replace("|", "@").replace("\\", "/").split("/")
    if any(part in ("", ".", "..") or ":" in part for part in parts):
        raise FormatError(f"unsafe name in archive: {name!r}")
    return root.joinpath(*parts)


def cmd_extract(args) -> int:
    cpk = archive.read_archive(_read(args.archive), strict=False)
    root = Path(args.output)
    done = 0
    failed = 0
    for entry in cpk.files:
        if entry.name is None:
            failed += 1
            _print(f"skipped 0x{entry.hash:016X}: no name")
            continue
        if args.pattern and not fnmatch.fnmatch(entry.name.lower(), args.pattern.lower()):
            continue
        try:
            content = cpk.read_file(entry)
            target = output_path(root, entry.name)
        except FormatError as exc:
            failed += 1
            _print(f"failed {entry.name}: {exc}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        done += 1
    _print(f"extracted {done}, failed {failed}")
    return 1 if failed else 0


def cmd_toc(args) -> int:
    toc = index.parse_core_toc(_read(args.file), args.build)
    _print(f"build {toc.build}, {len(toc.entries)} entries")
    if toc.unknown is not None:
        _print(f"unknown word 0x{toc.unknown:08X}")
    if args.entries:
        for entry in toc.entries:
            _print(f"{entry.group:>3} {entry.handle:>11} {entry.name}")
    return 0


def cmd_prefetch(args) -> int:
    fetch = index.parse_prefetch(_read(args.file), args.build)
    _print(f"build {fetch.build}, {len(fetch.parents)} parents, {len(fetch.deps)} dependencies")
    if fetch.build == "switch":
        _print(f"parents sorted by low handle bits: {fetch.parents_sorted}")
    if args.toc:
        toc = index.parse_core_toc(_read(args.toc), fetch.build)
        _print(f"unresolved in CoreTOC: {fetch.unresolved(toc)}")
    return 0


def cmd_power(args) -> int:
    pow_file = power.parse_power(_read(args.file), args.build)
    _print(f"build {pow_file.build}, version {pow_file.header.version}, "
           f"handle {pow_file.header.handle}")
    _print(f"powerBits 0x{pow_file.power_bits:X}: {', '.join(pow_file.power_bit_names())}")
    _print(f"scripted {pow_file.scripted}, secondary {pow_file.uses_secondary_resource}, "
           f"formulaCount {pow_file.formula_count}, "
           f"formula records {len(pow_file.formula_records)}")
    _print(f"script bytes {len(pow_file.script)}")
    for tag_map in pow_file.maps.values():
        if not tag_map.records:
            continue
        _print(f"[{tag_map.name}] {len(tag_map.records)} records")
        for record in tag_map.records:
            if record.type == power.TAG_FORMULA:
                _print(f"  0x{record.tag:X} formula {record.value.text!r}")
                if args.disasm:
                    for line in formula.disassemble(record.value.code, pow_file.build).splitlines():
                        _print(f"    {line}")
            else:
                _print(f"  0x{record.tag:X} type {record.type}: {record.value}")
    return 0


def cmd_mesh(args) -> int:
    app = appearance.parse_appearance(_read(args.file), args.build)
    if args.format == "obj":
        text = appearance.to_obj(app, args.flip_v, not args.skip_low_poly)
    else:
        resolver = None
        if args.texture_uri:
            resolver = lambda handle: args.texture_uri.format(handle=handle)
        text = appearance.to_gltf_json(
            app, args.flip_v, args.lightmap_v_scale, resolver, args.look, not args.skip_low_poly,
        )
    Path(args.output).write_text(text, encoding="utf-8", newline="\n")
    for warning in app.warnings:
        _print(f"warning: {warning}")
    for note in app.notes:
        _print(f"note: {note}")
    _print(f"build {app.build}, {len(app.sub_objects)} sub-objects, {len(app.bones)} bones")
    return 0


def _texture_summary(tex: texture.TextureFile) -> None:
    _print(f"build {tex.build}, version {tex.header.version}, format {tex.format}")
    _print(f"size {tex.width}x{tex.height}, faces {tex.faces}, depth {tex.depth}, "
           f"levels {tex.levels}")
    _print(f"frames {len(tex.frames)}, image ids {len(tex.image_ids)}, "
           f"lightmap V scale {tex.lightmap_v_scale}")
    for warning in tex.warnings:
        _print(f"warning: {warning}")


def cmd_texture(args) -> int:
    tex = texture.parse_texture(_read(args.file), args.build)
    if not args.output:
        _texture_summary(tex)
        return 0
    target = Path(args.output)
    # ``35`` V21-7: one path for every build; an option that cannot be honoured is refused,
    # never silently ignored
    modes = [name for name, on in (("--all-levels", args.all_levels), ("--strip", args.strip),
                                   ("--frame", args.frame is not None), ("--normal", args.normal)) if on]
    if len(modes) > 1:
        raise FormatError(f"{' and '.join(modes)} cannot be combined")
    if args.strip and (args.face or args.slice):
        raise FormatError("--strip takes every face or slice: --face and --slice do not apply")
    if args.frame is not None and (args.face or args.slice):
        raise FormatError("--frame crops level images of face 0: --face and --slice do not apply")
    if args.normal and args.slice:
        raise FormatError("--normal does not take --slice")
    if tex.build == "x360":
        if not args.x360_pixels:
            raise FormatError("x360 pixels are decoded only with --x360-pixels (G2-T)")
        from . import x360_texture

        decode = x360_texture.decode_image
    else:
        decode = texture.decode_image
    if args.all_levels:
        for level in range(tex.levels):
            image = decode(tex, level, args.face, args.slice)
            path = target.with_name(f"{target.stem}_level{level}{target.suffix}")
            path.write_bytes(texture.encode_png(image))
        return 0
    if args.strip:
        image = texture.vertical_strip([
            decode(tex, args.level, face, z)
            for face in range(tex.faces) for z in range(tex.level_depth(args.level))
        ])
    elif args.frame is not None:
        if not 0 <= args.frame < len(tex.frames):
            raise FormatError(f"no frame {args.frame}: the file has {len(tex.frames)}", 0x220)
        rect = tex.frames[args.frame].rect(tex.width, tex.height)
        image = texture.crop(decode(tex, args.level), *(value >> args.level for value in rect))
    elif args.normal:
        if tex.format != texture.FORMAT_NORMAL:
            raise FormatError(f"format {tex.format} does not hold a normal map", 0x20)
        plain = decode(tex, args.level, args.face)
        out = bytearray(len(plain.rgba))
        out[0::4] = plain.rgba[3::4]
        out[1::4] = plain.rgba[1::4]
        out[3::4] = b"\xFF" * (len(out) // 4)
        image = texture.Image(plain.width, plain.height, bytes(out))
    else:
        image = decode(tex, args.level, args.face, args.slice)
    target.write_bytes(texture.encode_png(image))
    return 0


def cmd_anim(args) -> int:
    can = _read(args.can) if args.can else None
    ani = animation.parse_ani(_read(args.file), args.build, can)
    _print(f"build {ani.build}, version {ani.header.version}, playback {ani.playback_mode}, "
           f"appearance {ani.appearance_handle}, permutations {len(ani.permutations)}")
    for perm in ani.permutations:
        _print(f"  {perm.name!r}: {perm.bone_count} bones, {perm.keyframes} keyframes, "
               f"{perm.fps:g} fps, {perm.duration:g} s, {perm.attachment_count} attachments")
        if perm.has_data:
            animation.decode_permutation(perm)
    if args.app:
        if not args.output:
            raise FormatError("--app needs -o")
        app = appearance.parse_appearance(_read(args.app), args.build)
        text = animation.to_gltf_animated_json(app, ani, args.permutation, args.flip_v)
        Path(args.output).write_text(text, encoding="utf-8", newline="\n")
    return 0


def cmd_strings(args) -> int:
    strings = text.parse_string_list(_read(args.file), args.build)
    if args.label:
        found = text.lookup(strings, args.label)
        if found is None:
            _print("not found")
            return 1
        _print(found)
        return 0
    for entry in strings.entries:
        _print(f"{entry.label_hash:08X} {entry.text!r}")
    return 0


def cmd_balance(args) -> int:
    if args.gbid:
        _print(str(balance.gbid(args.gbid)))
        return 0
    if not args.file:
        raise FormatError("a file or --gbid is needed")
    table = balance.parse_balance(_read(args.file), args.build)
    _print(f"build {table.build}, type {table.type}, table {table.table}, "
           f"{len(table.rows)} rows of {table.row_size}, other slots {len(table.raw_slots)}")
    for row in table.rows:
        if row.name is not None:
            _print(f"{row.index:>5} {row.id:>12} {row.name}")
    return 0


class _SnoStore:
    """Finds SNO files in archives by (group, handle), through CoreTOC names."""

    def __init__(self, archive_paths: list[str], toc_path: str | None) -> None:
        self.archives = [archive.read_archive(_read(p), strict=False) for p in archive_paths]
        toc = index.parse_core_toc(_read(toc_path)) if toc_path else None
        self.names = {} if toc is None else {
            (e.group & 0xFFFFFFFF, e.handle & 0xFFFFFFFF): e.name for e in toc.entries
        }

    def read(self, group_id: int, handle: int) -> bytes | None:
        name = self.names.get((group_id, handle))
        group = group_by_id(group_id)
        if name is None or group is None:
            return None
        path = f"{group.name}\\{name}.{group.extension}"
        for cpk in self.archives:
            entry = cpk.find(path)
            if entry is not None:
                return cpk.read_file(entry)
        return None


def _load_layout(wrl: world.WorldFile, store: _SnoStore) -> world.WorldLayout:
    scenes: dict[int, world.SceneInfo] = {}
    sets: dict[int, world.MarkerSet] = {}
    for entry in wrl.scene_entries:
        if entry.handle in scenes:
            continue
        data = store.read(world.SCENE_GROUP, entry.handle)
        if data is not None:
            scenes[entry.handle] = world.parse_scene_info(data, wrl.build)
    wanted = [h for scene in scenes.values() for h in scene.marker_sets] + wrl.marker_sets
    for handle in wanted:
        if handle not in sets:
            data = store.read(24, handle)
            if data is not None:
                sets[handle] = world.parse_marker_set(data, wrl.build)
    return world.lay_out_world(wrl, scenes, sets)


def cmd_level(args) -> int:
    wrl = world.parse_world(_read(args.file), args.build)
    generators = len(wrl.server.generators) if wrl.server else 0
    _print(f"build {wrl.build}, generated {wrl.is_generated}, "
           f"scenes {len(wrl.scene_entries)}, generators {generators}, "
           f"world marker sets {len(wrl.marker_sets)}")
    if not args.archive:
        for entry in wrl.scene_entries:
            _print(f"scene {entry.handle} at {entry.position}")
        return 0
    layout = _load_layout(wrl, _SnoStore(args.archive, args.toc))
    _print(f"markers {len(layout.markers)}, sub-scene slots {len(layout.sub_scene_slots)}, "
           f"missing scenes {len(layout.missing_scenes)}, "
           f"missing marker sets {len(layout.missing_marker_sets)}")
    for placed in layout.markers:
        state = " conditional" if placed.conditional else ""
        _print(f"{placed.marker.type_name or placed.marker.type} {placed.marker.group}:"
               f"{placed.marker.handle} at {placed.position}{state}")
    return 0


def cmd_collision(args) -> int:
    mesh = collision.parse_collision(_read(args.file), args.build)
    triangles = sum(len(m.triangles) for m in mesh.meshes)
    _print(f"build {mesh.build}, {len(mesh.meshes)} meshes, {triangles} triangles")
    for warning in mesh.warnings:
        _print(f"warning: {warning}")
    if args.output:
        Path(args.output).write_text(collision.to_obj(mesh), encoding="utf-8", newline="\n")
    return 0


# --- stage 16: import and archive write commands (W16-12) -------------------------------------

def _module(name: str):
    """Import a d3sno module when a command needs it (the import modules are optional)."""
    return importlib.import_module(f"{__package__}.{name}")


def _same_path(a, b) -> bool:
    pa, pb = Path(a), Path(b)
    try:
        if pa.exists() and pb.exists():
            return os.path.samefile(pa, pb)
    except OSError:
        pass
    return os.path.normcase(os.path.abspath(pa)) == os.path.normcase(os.path.abspath(pb))


def refuse_overwrite(outputs, inputs) -> None:
    """W16-12: write commands never overwrite their input."""
    for out in outputs:
        for given in inputs:
            if _same_path(out, given):
                raise ValueError(f"output {out} is the input {given}: write commands never overwrite "
                                 "their input")


def archive_name_of(relative: Path) -> str:
    """The inverse of ``output_path``: a path below the extract folder -> archive name
    (``/`` -> ``\\``, ``@locale`` -> ``|locale``)."""
    parts = relative.parts
    if not parts or any(part in ("", ".", "..") for part in parts):
        raise FormatError(f"no archive name for {relative}")
    return "\\".join(parts).replace("@", "|")


_LEVEL_FILE = re.compile(r"_level(\d+)\.png$", re.IGNORECASE)


def level_order(paths: list[str]) -> list[str]:
    """W16-12: files named ``*_level<n>.png`` (as ``texture --all-levels`` writes them) are
    sorted by n; any other list keeps the given order."""
    found = [_LEVEL_FILE.search(Path(p).name) for p in paths]
    if len(paths) < 2 or not all(found):
        return list(paths)
    levels = [int(m.group(1)) for m in found]
    if len(set(levels)) != len(levels):
        raise ValueError("two images have the same level number")
    return [p for _, p in sorted(zip(levels, paths))]


def cmd_texture_import(args) -> int:
    paths = level_order(args.images)
    refuse_overwrite([args.output], [args.template, *paths])
    png = _module("png")
    importer = _module("texture_import")
    template = texture.parse_texture(_read(args.template), args.build)
    images = [png.read_png(_read(p)) for p in paths]
    surfaces = template.faces * template.depth
    if args.strip:
        if len(images) != 1:
            raise ValueError("--strip takes one strip image")
        if surfaces == 1:
            raise ValueError("--strip needs a cube or volume template")
        given = importer.split_strip(images[0], surfaces)
    elif surfaces > 1:
        if len(images) != surfaces:
            kind = "cube faces" if template.faces > 1 else "volume slices"
            raise ValueError(f"the template needs {surfaces} {kind}, {len(images)} images given")
        given = images
    else:
        given = images[0] if len(images) == 1 else images
    out = importer.import_texture(template, given, levels=args.levels, format=args.format)
    Path(args.output).write_bytes(out)
    _print(f"wrote {args.output}: {len(out)} bytes")
    return 0


def read_model(path: str):
    """W16-12: a glTF (``.gltf`` / ``.glb``) or OBJ model, by its extension."""
    importer = _module("mesh_import")
    suffix = Path(path).suffix.lower()
    if suffix in (".gltf", ".glb"):
        return importer.read_gltf(_read(path), base_path=str(Path(path).resolve().parent))
    if suffix == ".obj":
        return importer.read_obj(_read(path).decode("utf-8"))
    raise ValueError(f"{path}: the model must be .gltf, .glb or .obj")


def cmd_mesh_import(args) -> int:
    refuse_overwrite([args.output], [args.template, args.model])
    importer = _module("mesh_import")
    template = appearance.parse_appearance(_read(args.template), args.build)
    document = read_model(args.model)
    out, report = importer.import_mesh_report(
        template, document, flip_v=args.flip_v, lightmap_v_scale=args.lightmap_v_scale,
        missing=args.missing, skin=args.skin, low_poly=not args.skip_low_poly)
    Path(args.output).write_bytes(out)
    added, removed = getattr(report, "added", []), getattr(report, "removed", [])
    for label in report.changed:
        _print(f"changed {label}")
    for label in report.kept:
        _print(f"kept {label}")
    for label in added:  # W17-14: every added and removed part, with its label
        _print(f"added {label}")
    for label in removed:
        _print(f"removed {label}")
    for warning in report.warnings:
        _print(f"warning: {warning}")
    parts = f", {len(added)} added, {len(removed)} removed" if added or removed else ""
    _print(f"wrote {args.output}: {len(report.changed)} changed, {len(report.kept)} kept{parts}")
    return 0


def cmd_export(args) -> int:
    """W18-1: the user's archives as glTF, PNG and JSON in OUT_DIR, with a manifest and coverage."""
    export = _module("export")
    result = export.export_game(
        args.game_dir, args.out, build=args.build, archives=args.archives, groups=args.groups,
        all_levels=args.all_levels, skip_existing=args.skip_existing, force=args.force, travel=args.travel)
    if result.exit_code == export.EXIT_REFUSED:
        print(f"error: {result.message}", file=sys.stderr)
        return result.exit_code
    for line in result.summary:
        _print(line)
    if result.message:
        print(f"error: {result.message}", file=sys.stderr)
    return result.exit_code


def _jobs(text: str) -> int:
    value = int(text)
    if value < 1:
        raise ValueError(f"--jobs {value}: give at least 1")
    return value


def _jobs_option(sub) -> None:
    """``48`` Z4: ``--jobs N`` of ``apply``, ``replace`` and ``add``."""
    from .archive import default_jobs

    sub.add_argument("--jobs", type=_jobs, default=default_jobs(), metavar="N",
                     help="worker threads (default: the number of processors; 1: one thread): they read and "
                          "inflate the source, and with --segment pack the segments; the archive is the same "
                          "for any number")
    sub.add_argument("--segment", action="store_true",
                     help="the segmented layout (51 AB7): the stream in segments of 8 MiB, each starting a new "
                          "slot, packed in parallel. Off by default: whether the game reads a slot ended early "
                          "by a stored size of 0 is not known")
    sub.add_argument("--profile", action="store_true",
                     help="print the use of the worker threads (work submitted, taken, dropped; the time "
                          "the packer waited)")


def _profile_lines(written) -> list[str]:
    """``51`` AB7: the layout and the ``--profile`` statistics of one written archive."""
    lines = [f"layout: {getattr(written, 'layout', 'default')}"]
    stats = getattr(written, "workers", None)
    if stats:
        from .archive import workers_text

        lines.append("workers: " + workers_text(stats))
    return lines


def cmd_apply(args) -> int:
    """``46`` W21-8: apply an edited export folder to the game folder, writing new archives into OUT_DIR."""
    apply = _module("apply")
    mesh = {"flip_v": args.flip_v, "lightmap_v_scale": args.lightmap_v_scale, "missing": args.missing,
            "skin": args.skin, "low_poly": not args.skip_low_poly}
    result = apply.apply_export(args.game_dir, args.export_dir, args.out, build=args.build, dry_run=args.dry_run,
                                copy_unchanged=args.copy_unchanged, only=args.only, mesh_options=mesh,
                                force=args.force, jobs=args.jobs, segment=args.segment, profile=args.profile)
    if result.exit_code == apply.EXIT_REFUSED:
        print(f"error: {result.message}", file=sys.stderr)
        return result.exit_code
    for line in result.summary:
        _print(line)
    if result.message:
        print(f"error: {result.message}", file=sys.stderr)
    return result.exit_code


APPLY_NOTE = (
    "Compares each file of EXPORT_DIR with the output_hash of its manifest entry (W21-1, W21-2), imports each "
    "changed mesh (meshes/<h>.glb), texture (all textures/<h>*.png of the handle) and string table "
    "(data/text/<locale>/<h>.json) onto its stored template in GAME_DIR, and writes every archive that holds a "
    "change into OUT_DIR, with apply_report.json (W21-3 to W21-7). Unchanged files stay byte for byte; missing "
    "files are no deletion; files without a manifest entry and other kinds of file are reported, not applied. "
    "The output is game content: an OUT_DIR inside a git working tree is refused without --force. Exit code 0: "
    "every changed file applied or reported with a reason; 1: stopped; 2: refused."
)


EXPORT_NOTE = (
    "Writes meshes/<handle>.glb, anims/<handle>.glb, textures/<handle>.png, data/<group>/<handle>.json, "
    "data/text/<locale>/<handle>.json, manifest.json and coverage.json (W18-2, W18-3). The output is game "
    "content: it stays on this computer and never goes into a repository; an OUT_DIR inside a git working "
    "tree is refused without --force. Coordinates are as stored (right-handed, z up, actors face +x, triangles "
    "counter-clockwise), not converted (W18-9, W19-1); this holds for the meshes, animations and the scn, wrl and "
    "mrk records alike, as manifest.json states once (W19-6). Quaternions are [x, y, z, w] as stored and "
    "rotate as v' = q v q* (Hamilton product, the glTF reading; W20-4). Animations are in place; --travel adds a "
    "d3sno_travel node moved by the average velocity (W19-5). Exit code 0: "
    "every file exported or skipped with a reason in coverage.json; 1: the export stopped; 2: refused."
)


def sub_object_place(text: str) -> tuple[int, int]:
    """``--sub-object G:I``: geo set G (0 or 1) and sub-object index I."""
    match = re.fullmatch(r"([01]):(\d+)", text.strip())
    if match is None:
        raise ValueError(f"--sub-object {text!r}: give G:I, geo set 0 or 1 and a sub-object index")
    return int(match.group(1)), int(match.group(2))


def cmd_mesh_transplant(args) -> int:
    """W17-11 as a command: TARGET with sub-object G:I of SOURCE added at the end of geo set G."""
    refuse_overwrite([args.output], [args.target, args.source])
    geo_set, index = sub_object_place(args.sub_object)
    app_mesh = _module("app_mesh")
    target = appearance.parse_appearance(_read(args.target), args.build)
    source = appearance.parse_appearance(_read(args.source), args.build)
    mesh = app_mesh.transplant(target, source, geo_set, index, name=args.name)
    out, report = app_mesh.write_app_mesh_report(target, mesh, derive=True)
    Path(args.output).write_bytes(out)
    for label in report.added:
        _print(f"added {label}")
    for label in report.removed:
        _print(f"removed {label}")
    _print(f"wrote {args.output}: {len(out)} bytes")
    return 0


def cmd_replace(args) -> int:
    from . import index_write

    if args.from_dir is None:
        if args.name is None or args.file is None:
            raise ValueError("replace needs NAME FILE, or --from-dir DIR")
        if args.add_new:
            raise ValueError("--add-new goes with --from-dir")
        refuse_overwrite([args.output], [args.archive, args.file])
        with _MappedArchive(args.archive) as cpk:  # Y1: streamed, nothing of the archive kept
            written = index_write.write_changed(cpk, args.output, {args.name: _read(args.file)}, jobs=args.jobs,
                                                segment=args.segment)
        _print("replaced 1, added 0")
        _report_written(args, written)
        return 0
    if args.name is not None or args.file is not None:
        raise ValueError("pass NAME FILE or --from-dir, not both")
    root = Path(args.from_dir)
    if not root.is_dir():
        raise ValueError(f"{root} is not a folder")
    files = sorted(p for p in root.rglob("*") if p.is_file())
    refuse_overwrite([args.output], [args.archive, *files])
    with _MappedArchive(args.archive) as cpk:
        return _replace_from_dir(args, cpk, root, files)


def _replace_from_dir(args, cpk, root: Path, files) -> int:
    from . import index_write

    replace: dict[str, bytes] = {}
    add: dict[str, bytes] = {}
    unknown = []
    seen: dict[int, Path] = {}
    for path in files:
        name = archive_name_of(path.relative_to(root))
        key = archive.name_hash(name)
        if key in seen:
            raise FormatError(f"{path} and {seen[key]} map to the same archive name {name!r}")
        seen[key] = path
        entry = cpk.find(name)
        if entry is not None:
            replace[entry.name if entry.name is not None else name] = _read(path)
        elif args.add_new:
            add[name] = _read(path)
        else:
            unknown.append(name)
    if unknown:
        raise FormatError(f"{len(unknown)} files map to no archive name (pass --add-new to add them): "
                          + ", ".join(unknown[:10]))
    if not replace and not add:
        raise ValueError(f"{root} holds no file")
    written = index_write.write_changed(cpk, args.output, replace, add, jobs=args.jobs,
                                        segment=args.segment)  # Y1: streamed
    _print(f"replaced {len(replace)}, added {len(add)}")
    _report_written(args, written)
    return 0


def _report_written(args, written) -> None:
    """``51`` AB7: the layout used; with ``--profile`` the use of the workers."""
    lines = _profile_lines(written)
    for line in lines if args.profile else lines[:1]:
        _print(line)


def cmd_add(args) -> int:
    from . import index_write

    tocs = list(args.toc or [])
    inputs = [args.archive, args.file, *tocs] + ([args.prefetch] if args.prefetch else [])
    refuse_overwrite([args.output], inputs)
    if args.depends and not args.prefetch:
        raise ValueError("--depends needs --prefetch")
    names = [Path(t).name for t in tocs] + ([Path(args.prefetch).name] if args.depends else [])
    if len({n.lower() for n in names}) != len(names):
        raise ValueError("two index files share a base name; --index-out would write both to one file")
    if names:
        if not args.index_out:
            raise ValueError("--toc and --prefetch need --index-out")
        refuse_overwrite([Path(args.index_out) / n for n in names], inputs)
        refuse_overwrite([args.output], [Path(args.index_out) / n for n in names])
    with _MappedArchive(args.archive) as cpk:
        pack = args.pack
        if cpk.header.build == "x360" and pack is None and tocs:
            pack = index_write.x360_pack(Path(args.archive).name, [Path(t).name for t in tocs])
        toc_map = {Path(t).name: _read(t) for t in tocs}
        prefetch = {Path(args.prefetch).name: _read(args.prefetch)} if args.prefetch else None
        result = index_write.add_sno_file(cpk, args.name, _read(args.file), toc_map, prefetch,
                                          args.depends or (), pack=pack, out=args.output,
                                          jobs=args.jobs, segment=args.segment)  # Y1: streamed; Z4, AB7
    if result.index_files:
        target = Path(args.index_out)
        target.mkdir(parents=True, exist_ok=True)
        for name, data in result.index_files.items():
            (target / name).write_bytes(data)
            _print(f"wrote {target / name}")
    ident = result.identity
    if ident is not None:
        _print(f"added {args.name}: group {ident.group}, handle {ident.handle}")
    else:
        _print(f"added {args.name} (no CoreTOC entry)")
    _report_written(args, result.archive)
    return 0


MESH_IMPORT_NOTE = (
    "To empty a draw from an editor, remove its primitive (or OBJ group) and import with "
    "--missing empty; a glTF primitive without indices and without extras.d3sno.no_indices is "
    "read as triangles 0..count-1 (V21-4). --missing remove removes every template sub-object, "
    "draw or skinning record the model does not hold; a sub-object left without geometry goes "
    "(W17-13). A mesh (OBJ object) that matches no sub-object, or an editor copy of one, is added "
    "as a new sub-object; a primitive (OBJ group) that matches no draw is new geometry (W17-12). "
    "Every added and removed part is printed (W17-14)."
)

MESH_TRANSPLANT_NOTE = (
    "Copies sub-object G:I of SOURCE (same build, same skeleton; on ps3 the same draw format) to the "
    "end of geo set G of TARGET, with every stored value; its material entry comes along when TARGET "
    "lacks it; a taken name gets _2, _3, ... (W17-11, W17-12). The output may not be an input."
)

_WRITE_COMMANDS = ("export", "apply", "texture-import", "mesh-import", "mesh-transplant", "replace", "add")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="d3sno")
    commands = parser.add_subparsers(dest="command", required=True)

    def add(name: str, handler) -> argparse.ArgumentParser:
        sub = commands.add_parser(name)
        sub.set_defaults(handler=handler)
        return sub

    def add_build(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--build", choices=("ps3", "switch", "x360"))

    add("info", cmd_info).add_argument("archive")
    sub = add("verify", cmd_verify)
    sub.add_argument("archive")
    sub.add_argument("--allow-gaps", action="store_true")
    sub = add("list", cmd_list)
    sub.add_argument("archive")
    sub.add_argument("pattern", nargs="?")
    sub = add("extract", cmd_extract)
    sub.add_argument("archive")
    sub.add_argument("pattern", nargs="?")
    sub.add_argument("-o", "--output", default=".")
    sub = add("toc", cmd_toc)
    sub.add_argument("file")
    sub.add_argument("--entries", action="store_true")
    add_build(sub)
    sub = add("prefetch", cmd_prefetch)
    sub.add_argument("file")
    sub.add_argument("--toc", help="CoreTOC.dat, to count unresolved dependencies")
    add_build(sub)
    sub = add("power", cmd_power)
    sub.add_argument("file")
    sub.add_argument("--disasm", action="store_true")
    add_build(sub)
    sub = add("mesh", cmd_mesh)
    sub.add_argument("file")
    sub.add_argument("-o", "--output", required=True)
    sub.add_argument("--format", choices=("obj", "gltf"), default="obj")
    sub.add_argument("--flip-v", action="store_true")
    sub.add_argument("--lightmap-v-scale", type=float, help="adds UV2 to glTF output")
    sub.add_argument("--texture-uri", help="glTF image URI, with {handle}, e.g. tex/{handle}.png")
    sub.add_argument("--look", type=int, default=0)
    sub.add_argument("--skip-low-poly", action="store_true", help="leave out geo set 1 (G1-1)")
    add_build(sub)
    sub = add("texture", cmd_texture)
    sub.add_argument("file")
    sub.add_argument("-o", "--output", help="PNG to write; without it, print a summary")
    sub.add_argument("--level", type=int, default=0)
    sub.add_argument("--face", type=int, default=0)
    sub.add_argument("--slice", type=int, default=0)
    sub.add_argument("--frame", type=int, help="crop this frame")
    sub.add_argument("--all-levels", action="store_true")
    sub.add_argument("--strip", action="store_true", help="all faces or slices, stacked")
    sub.add_argument("--normal", action="store_true", help="format 43: X in red, Y in green")
    sub.add_argument("--x360-pixels", action="store_true",
                     help="decode an x360 payload by the G2-T layout (optional)")
    add_build(sub)
    sub = add("anim", cmd_anim)
    sub.add_argument("file")
    sub.add_argument("--can", help="the matching .can, for ps3")
    sub.add_argument("--app", help="an .app, to export one permutation as glTF")
    sub.add_argument("-o", "--output")
    sub.add_argument("--permutation", type=int, default=0)
    sub.add_argument("--flip-v", action="store_true")
    add_build(sub)
    sub = add("strings", cmd_strings)
    sub.add_argument("file")
    sub.add_argument("--label", help="print the text of this label")
    add_build(sub)
    sub = add("balance", cmd_balance)
    sub.add_argument("file", nargs="?")
    sub.add_argument("--gbid", help="print the game balance id of a name")
    add_build(sub)
    sub = add("level", cmd_level)
    sub.add_argument("file")
    sub.add_argument("--archive", action="append", help="repeat for each archive")
    sub.add_argument("--toc", help="CoreTOC.dat, to find scenes and marker sets by handle")
    add_build(sub)
    sub = add("collision", cmd_collision)
    sub.add_argument("file")
    sub.add_argument("-o", "--output", help="OBJ to write")
    add_build(sub)
    sub = add("texture-import", cmd_texture_import)
    sub.add_argument("template", help="the .tex to replace")
    sub.add_argument("images", nargs="+", help="PNG files: a mip chain, or cube faces / volume slices")
    sub.add_argument("-o", "--output", required=True)
    sub.add_argument("--format", type=int)
    sub.add_argument("--levels", type=int)
    sub.add_argument("--strip", action="store_true", help="one vertical strip of faces or slices")
    add_build(sub)
    sub = add("mesh-import", cmd_mesh_import)
    # ``35`` V21-4
    sub.epilog = MESH_IMPORT_NOTE
    sub.add_argument("template", help="the .app to replace")
    sub.add_argument("model", help=".gltf, .glb or .obj")
    sub.add_argument("-o", "--output", required=True)
    sub.add_argument("--flip-v", action="store_true")
    sub.add_argument("--lightmap-v-scale", type=float)
    sub.add_argument("--missing", choices=("keep", "empty", "remove"), default="keep")
    sub.add_argument("--skin", choices=("refuse", "nearest"), default="refuse")
    sub.add_argument("--skip-low-poly", action="store_true")
    add_build(sub)
    sub = add("mesh-transplant", cmd_mesh_transplant)
    sub.epilog = MESH_TRANSPLANT_NOTE
    sub.add_argument("target", help="the .app to add the sub-object to")
    sub.add_argument("source", help="the .app to copy it from")
    sub.add_argument("--sub-object", required=True, metavar="G:I", help="geo set and index in SOURCE")
    sub.add_argument("-o", "--output", required=True)
    sub.add_argument("--name", help="the new sub-object's name (W17-2)")
    add_build(sub)
    sub = add("export", cmd_export)
    sub.epilog = EXPORT_NOTE
    sub.add_argument("game_dir", metavar="GAME_DIR", help="the folder of the .cpk archives and index files")
    sub.add_argument("--out", required=True, metavar="OUT_DIR")
    sub.add_argument("--archives", nargs="+", metavar="PATTERN", help="archive base names, e.g. Act1* Common*")
    sub.add_argument("--groups", help="comma-separated SNO groups, e.g. app,tex,ani")
    sub.add_argument("--all-levels", action="store_true", help="also the texture levels after level 0")
    sub.add_argument("--skip-existing", action="store_true",
                     help="keep files whose earlier manifest entry has the same source and hash")
    sub.add_argument("--force", action="store_true", help="export inside a git working tree anyway")
    sub.add_argument("--travel", action="store_true",
                     help="add a d3sno_travel node to each animation, moved by its average velocity (W19-5)")
    add_build(sub)
    sub = add("apply", cmd_apply)
    sub.epilog = APPLY_NOTE
    sub.add_argument("game_dir", metavar="GAME_DIR", help="the game folder that was exported")
    sub.add_argument("export_dir", metavar="EXPORT_DIR", help="the edited export folder")
    sub.add_argument("--out", required=True, metavar="OUT_DIR")
    sub.add_argument("--dry-run", action="store_true", help="import in memory, write only apply_report.json")
    sub.add_argument("--copy-unchanged", action="store_true",
                     help="also copy every other file of GAME_DIR, so that OUT_DIR is a complete game folder")
    sub.add_argument("--only", help="comma-separated kinds: meshes,textures,text,records,world,collision,anims")
    sub.add_argument("--flip-v", action="store_true")
    sub.add_argument("--lightmap-v-scale", type=float)
    sub.add_argument("--missing", choices=("keep", "empty"), default="keep")
    sub.add_argument("--skin", choices=("refuse", "nearest"), default="refuse")
    sub.add_argument("--skip-low-poly", action="store_true")
    sub.add_argument("--force", action="store_true", help="write inside a git working tree anyway")
    _jobs_option(sub)
    add_build(sub)
    sub = add("replace", cmd_replace)
    sub.add_argument("archive")
    sub.add_argument("name", nargs="?", help="archive name as list prints it")
    sub.add_argument("file", nargs="?")
    sub.add_argument("--from-dir", help="a folder laid out as extract writes it")
    sub.add_argument("--add-new", action="store_true", help="with --from-dir: add unknown files")
    sub.add_argument("-o", "--output", required=True)
    _jobs_option(sub)
    sub = add("add", cmd_add)
    sub.add_argument("archive")
    sub.add_argument("name")
    sub.add_argument("file")
    sub.add_argument("-o", "--output", required=True)
    sub.add_argument("--toc", nargs="+", help="CoreTOC.dat, or on x360 all six TOC files")
    sub.add_argument("--prefetch")
    sub.add_argument("--depends", nargs="+", help="SNO names (Group\\name.ext) of the dependencies")
    sub.add_argument("--index-out", help="folder for the written index files")
    _jobs_option(sub)
    sub.add_argument("--pack", help="x360: the pack (Common, Act1 .. Act5) when the archive name "
                                    "does not tell")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    try:
        return args.handler(args)
    except FormatError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        if args.command not in _WRITE_COMMANDS:
            raise
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
