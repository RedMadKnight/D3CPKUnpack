#!/usr/bin/env python3
"""
cpk_tool - command-line frontend for cpklib + cpkrebuild.

Subcommands:
    info     <archive>
    list     <archive> [filters]
    extract  <archive> [filters] --out DIR
    replace  <archive> <internal-name> <src> --out OUT [--resize] [--level N]
    delete   <archive> <internal-name>       --out OUT          [--level N]
    add      <archive> <internal-name> <src> --out OUT          [--level N]
    create   <dir>                           --out OUT [--version 6|7]
                                             [--endian big|little] [--level N]
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cpklib  # noqa: E402
import cpkrebuild  # noqa: E402


_SIZE_SUFFIXES = {"": 1, "k": 1024, "m": 1024**2, "g": 1024**3}


def _parse_size(s):
    if s is None:
        return None
    m = re.match(r"^(\d+(?:\.\d+)?)\s*([kmg]?)b?$", s.strip(), re.I)
    if not m:
        raise argparse.ArgumentTypeError(f"bad size: {s!r}")
    return int(float(m.group(1)) * _SIZE_SUFFIXES[m.group(2).lower()])


def _build_filter(args):
    exts = set(e.lower().lstrip(".") for e in (args.ext or []))
    name_re = re.compile(args.iname, re.I) if args.iname else (
        re.compile(args.name) if args.name else None
    )
    min_size = _parse_size(args.min_size) if args.min_size else None
    max_size = _parse_size(args.max_size) if args.max_size else None
    locale = args.locale.lower() if args.locale else None

    def keep(e):
        if exts and e.ext not in exts:
            return False
        if name_re and not name_re.search(e.name):
            return False
        if min_size is not None and e.size < min_size:
            return False
        if max_size is not None and e.size > max_size:
            return False
        if locale is not None:
            if "|" not in e.name:
                return False
            if e.name.split("|", 1)[1].lower() != locale:
                return False
        return True

    return keep


def _sort_entries(entries, field, reverse):
    keys = {
        "name":   lambda e: e.name.lower(),
        "size":   lambda e: e.size,
        "ext":    lambda e: (e.ext, e.name.lower()),
        "offset": lambda e: e.offset,
    }
    return sorted(entries, key=keys[field], reverse=reverse)


def _check_out_path(args):
    if os.path.abspath(args.out) == os.path.abspath(args.archive):
        print("  refusing to overwrite source; use --out something else",
              file=sys.stderr)
        sys.exit(2)


def _maybe_set_level(w, args):
    if getattr(args, "level", None) is not None:
        w.set_zlib_level(args.level)


def cmd_info(args):
    ar = cpklib.CPKArchive.open(args.archive)
    s = ar.stats()
    print(f"  archive:        {args.archive}")
    print(f"  CPK version:    {s['version']}  ({'big' if s['endian']=='>' else 'little'}-endian)")
    print(f"  files:          {s['file_count']:,}")
    print(f"  decomp size:    {s['decomp_fs']:,} B")
    print(f"  comp sector:    0x{s['comp_sector']:X} ({s['comp_sector_count']:,} sectors)")
    print(f"  top extensions:")
    for ext, n in s["ext_count_top10"]:
        print(f"    .{ext or '<none>':<8s} {n:>8,}")


def cmd_list(args):
    ar = cpklib.CPKArchive.open(args.archive)
    keep = _build_filter(args)
    rows = [e for e in ar if keep(e)]
    rows = _sort_entries(rows, args.sort, args.reverse)
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        print("  (no matching entries)")
        return
    total = sum(e.size for e in rows)
    if args.long:
        print(f"  {'size':>10}  {'offset':>10}  {'ext':<5}  name")
        for e in rows:
            print(f"  {e.size:>10,}  {e.offset:>10,}  {e.ext:<5}  {e.name}")
    else:
        for e in rows:
            print(f"  {e.size:>10,}  {e.name}")
    print(f"--- {len(rows):,} entries, {total:,} B total ---")


def cmd_extract(args):
    ar = cpklib.CPKArchive.open(args.archive)
    keep = _build_filter(args)
    selected = [e for e in ar if keep(e)]
    selected = _sort_entries(selected, args.sort, args.reverse)
    if args.limit:
        selected = selected[:args.limit]
    if not selected:
        print("  (no matching entries)")
        return
    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)
    n = len(selected)
    total_size = sum(e.size for e in selected)
    print(f"  extracting {n:,} files ({total_size:,} B) -> {out_dir}/")
    t0 = time.time()
    written = zeros = failed = 0
    for i, e in enumerate(selected, 1):
        try:
            payload = ar.read(e)
        except IOError as ex:
            failed += 1
            if args.verbose:
                print(f"    FAIL  {e.name}: {ex}")
            continue
        if not payload or all(b == 0 for b in payload[:64]):
            zeros += 1
        cpklib.CPKArchive._write_entry(
            e, payload, out_dir, flat=args.flat, sanitize_pipe=True,
        )
        written += 1
        if args.progress and (i % 1000 == 0 or i == n):
            print(f"    {i}/{n}  elapsed {time.time()-t0:.1f}s")
    print(f"  done: {written:,} written, {zeros:,} mostly-zero, "
          f"{failed:,} failed, {time.time()-t0:.1f}s")


def cmd_replace(args):
    _check_out_path(args)
    ar = cpklib.CPKArchive.open(args.archive)
    e = ar.get(args.name)
    if e is None:
        print(f"  not found: {args.name}", file=sys.stderr)
        sys.exit(2)
    with open(args.source, "rb") as f:
        new_bytes = f.read()
    if len(new_bytes) != e.size and not args.resize:
        print(f"  size mismatch: source is {len(new_bytes):,} B, "
              f"target is {e.size:,} B (use --resize to allow)",
              file=sys.stderr)
        sys.exit(2)
    print(f"  replacing {args.name} ({e.size:,} -> {len(new_bytes):,} B)")
    t0 = time.time()
    w = cpklib.CPKWriter.from_archive(ar)
    if len(new_bytes) == e.size:
        w.replace_file_same_size(args.name, new_bytes)
    else:
        w.replace_file(args.name, new_bytes)
    _maybe_set_level(w, args)
    w.save(args.out)
    print(f"  saved in {time.time()-t0:.1f}s -> {args.out}")
    re_ar = cpklib.CPKArchive.open(args.out)
    if re_ar.read(args.name) != new_bytes:
        print("  WARNING: round-trip mismatch", file=sys.stderr)
        sys.exit(1)
    print("  verified: round-trip OK")


def cmd_delete(args):
    _check_out_path(args)
    ar = cpklib.CPKArchive.open(args.archive)
    if ar.get(args.name) is None:
        print(f"  not found: {args.name}", file=sys.stderr)
        sys.exit(2)
    print(f"  deleting {args.name}")
    t0 = time.time()
    w = cpklib.CPKWriter.from_archive(ar)
    w.delete_file(args.name)
    _maybe_set_level(w, args)
    w.save(args.out)
    print(f"  saved in {time.time()-t0:.1f}s -> {args.out}")
    re_ar = cpklib.CPKArchive.open(args.out)
    if re_ar.get(args.name) is not None:
        print("  WARNING: file still present", file=sys.stderr)
        sys.exit(1)
    print(f"  verified: {len(re_ar):,} files (was {len(ar):,})")


def cmd_add(args):
    _check_out_path(args)
    ar = cpklib.CPKArchive.open(args.archive)
    if ar.get(args.name) is not None:
        print(f"  already exists: {args.name}", file=sys.stderr)
        sys.exit(2)
    with open(args.source, "rb") as f:
        new_bytes = f.read()
    print(f"  adding {args.name} ({len(new_bytes):,} B)")
    t0 = time.time()
    w = cpklib.CPKWriter.from_archive(ar)
    w.add_file(args.name, new_bytes)
    _maybe_set_level(w, args)
    w.save(args.out)
    print(f"  saved in {time.time()-t0:.1f}s -> {args.out}")
    re_ar = cpklib.CPKArchive.open(args.out)
    if re_ar.get(args.name) is None or re_ar.read(args.name) != new_bytes:
        print("  WARNING: add round-trip failed", file=sys.stderr)
        sys.exit(1)
    print(f"  verified: {len(re_ar):,} files (was {len(ar):,})")


def cmd_create(args):
    if os.path.exists(args.out):
        print(f"  refusing to overwrite existing: {args.out}", file=sys.stderr)
        sys.exit(2)
    root = os.path.abspath(args.dir)
    if not os.path.isdir(root):
        print(f"  not a directory: {root}", file=sys.stderr)
        sys.exit(2)
    files = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            full = os.path.join(dp, fn)
            rel = os.path.relpath(full, root)
            internal = rel.replace("/", "\\")
            files.append((internal, full))
    files.sort()
    if not files:
        print(f"  no files under {root}", file=sys.stderr)
        sys.exit(2)
    print(f"  creating v{args.version} ({args.endian}-endian) archive "
          f"with {len(files):,} files from {root}")
    t0 = time.time()
    w = cpklib.CPKWriter.create_new(version=args.version, endian=args.endian)
    total = 0
    for internal, full in files:
        with open(full, "rb") as f:
            data = f.read()
        w.add_file(internal, data)
        total += len(data)
    _maybe_set_level(w, args)
    w.save(args.out)
    print(f"  saved in {time.time()-t0:.1f}s ({total:,} B input) -> {args.out}")
    re_ar = cpklib.CPKArchive.open(args.out)
    if len(re_ar) != len(files):
        print(f"  WARNING: file count mismatch ({len(re_ar)} vs {len(files)})",
              file=sys.stderr)
        sys.exit(1)
    for idx in (0, len(files) // 2, len(files) - 1):
        internal, full = files[idx]
        if re_ar.read(internal) != open(full, "rb").read():
            print(f"  WARNING: round-trip mismatch on {internal}",
                  file=sys.stderr)
            sys.exit(1)
    print(f"  verified: {len(re_ar):,} files round-trip OK")


def _add_filter_args(p):
    p.add_argument("--ext", nargs="+", help="filter by extension(s)")
    p.add_argument("--name", help="regex against full path")
    p.add_argument("--iname", help="case-insensitive regex")
    p.add_argument("--min-size", help="min size, e.g. 1k, 5M")
    p.add_argument("--max-size", help="max size")
    p.add_argument("--locale", help="match localized variant suffix")
    p.add_argument("--sort", choices=("name","size","ext","offset"), default="name")
    p.add_argument("--reverse", action="store_true")
    p.add_argument("--limit", type=int)


_LEVEL_HELP = ("zlib level 0..9 (default 6). 0=none, 1=fastest, "
               "6=balanced, 9=smallest. Only affects rebuild paths.")


def main():
    ap = argparse.ArgumentParser(prog="cpk_tool", description="D3 CPK CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("info"); p.add_argument("archive")
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("list"); p.add_argument("archive")
    _add_filter_args(p)
    p.add_argument("--long", "-l", action="store_true")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("extract"); p.add_argument("archive")
    p.add_argument("--out", "-o", required=True)
    p.add_argument("--flat", action="store_true")
    p.add_argument("--progress", action="store_true")
    p.add_argument("--verbose", "-v", action="store_true")
    _add_filter_args(p)
    p.set_defaults(func=cmd_extract)

    p = sub.add_parser("replace", help="replace one file")
    p.add_argument("archive"); p.add_argument("name"); p.add_argument("source")
    p.add_argument("--out", "-o", required=True)
    p.add_argument("--resize", action="store_true",
                   help="allow size change (full rebuild)")
    p.add_argument("--level", type=int, choices=range(0, 10), help=_LEVEL_HELP)
    p.set_defaults(func=cmd_replace)

    p = sub.add_parser("delete", help="remove a file (full rebuild)")
    p.add_argument("archive"); p.add_argument("name")
    p.add_argument("--out", "-o", required=True)
    p.add_argument("--level", type=int, choices=range(0, 10), help=_LEVEL_HELP)
    p.set_defaults(func=cmd_delete)

    p = sub.add_parser("add", help="add a new file (full rebuild)")
    p.add_argument("archive"); p.add_argument("name"); p.add_argument("source")
    p.add_argument("--out", "-o", required=True)
    p.add_argument("--level", type=int, choices=range(0, 10), help=_LEVEL_HELP)
    p.set_defaults(func=cmd_add)

    p = sub.add_parser("create", help="build a fresh archive from a directory")
    p.add_argument("dir", help="directory tree to pack")
    p.add_argument("--out", "-o", required=True)
    p.add_argument("--version", type=int, choices=(6, 7), default=6)
    p.add_argument("--endian", choices=("big", "little"), default="big")
    p.add_argument("--level", type=int, choices=range(0, 10), help=_LEVEL_HELP)
    p.set_defaults(func=cmd_create)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
