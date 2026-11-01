"""Command line interface: ``python -m d3sno <command>``."""

from __future__ import annotations

import argparse
import fnmatch
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
        text = appearance.to_obj(app, args.flip_v)
    else:
        resolver = None
        if args.texture_uri:
            resolver = lambda handle: args.texture_uri.format(handle=handle)
        text = appearance.to_gltf_json(
            app, args.flip_v, args.lightmap_v_scale, resolver, args.look,
        )
    Path(args.output).write_text(text, encoding="utf-8", newline="\n")
    for warning in app.warnings:
        _print(f"warning: {warning}")
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
    if args.all_levels:
        for level, image in enumerate(texture.decode_levels(tex, args.face, args.slice)):
            path = target.with_name(f"{target.stem}_level{level}{target.suffix}")
            path.write_bytes(texture.encode_png(image))
        return 0
    if args.strip:
        image = texture.vertical_strip(texture.decode_surfaces(tex, args.level))
    elif args.frame is not None:
        image = texture.frame_image(tex, args.frame, args.level)
    elif args.normal:
        image = texture.decode_normal(tex, args.level, args.face)
    else:
        image = texture.decode_image(tex, args.level, args.face, args.slice)
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


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="d3sno")
    commands = parser.add_subparsers(dest="command", required=True)

    def add(name: str, handler) -> argparse.ArgumentParser:
        sub = commands.add_parser(name)
        sub.set_defaults(handler=handler)
        return sub

    def add_build(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--build", choices=("ps3", "switch"))

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


if __name__ == "__main__":
    sys.exit(main())
