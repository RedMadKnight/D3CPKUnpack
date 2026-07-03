#!/usr/bin/env python3
"""
Command-line frontend for Diablo III raw cutscene CPK archives.

Examples:
    python3 d3cutscene_tool.py info enUS_Cutscene.cpk
    python3 d3cutscene_tool.py list enUS_Cutscene.cpk --long
    python3 d3cutscene_tool.py validate enUS_Cutscene.cpk
    python3 d3cutscene_tool.py extract enUS_Cutscene.cpk --out ./cutscenes
    python3 d3cutscene_tool.py extract enUS_Cutscene.cpk --flat --name Hero_Create --out ./hero
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d3cutscene  # noqa: E402


def _entry_filter(args):
    name_re = None
    if getattr(args, "iname", None):
        name_re = re.compile(args.iname, re.I)
    elif getattr(args, "name", None):
        name_re = re.compile(args.name)

    def keep(entry: d3cutscene.CutsceneEntry) -> bool:
        if name_re and not name_re.search(entry.name):
            return False
        return True

    return keep


def _sort_entries(entries, field, reverse):
    keys = {
        "index": lambda e: e.index,
        "name": lambda e: e.name.lower(),
        "size": lambda e: e.size,
        "offset": lambda e: e.offset,
        "loc": lambda e: e.loc_index,
    }
    return sorted(entries, key=keys[field], reverse=reverse)


def _selected_entries(args, archive):
    keep = _entry_filter(args)
    entries = [entry for entry in archive.entries if keep(entry)]
    entries = _sort_entries(entries, args.sort, args.reverse)
    if args.limit:
        entries = entries[: args.limit]
    return entries


def _add_filter_args(parser):
    parser.add_argument("--name", help="regex against full internal path")
    parser.add_argument("--iname", help="case-insensitive regex against path")
    parser.add_argument(
        "--sort",
        choices=("index", "name", "size", "offset", "loc"),
        default="index",
    )
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--limit", type=int)


def cmd_info(args):
    archive = d3cutscene.CutsceneCPK(args.archive)
    total = sum(entry.size for entry in archive.entries)
    print(f"archive:       {args.archive}")
    print(
        f"version:       {archive.version} "
        f"({'big' if archive.endian == '>' else 'little'}-endian)"
    )
    print(f"fmt_const:     {archive.fmt_const}")
    print(f"files:         {archive.file_count}")
    print(f"locations:     {archive.loc_count}")
    print(f"data_start:    {archive.data_start:#x}")
    print(f"payload size:  {archive.file_size - archive.data_start:,} B")
    print(f"declared size: {archive.decomp_size:,} B")
    print(f"entry bytes:   {total:,} B")
    if args.scan_ogg:
        groups = archive.scan_ogg_groups()
        print(f"ogg groups:    {len(groups)}")
        print(
            "entry/group:   "
            + ("1:1" if len(groups) == len(archive.entries) else "not 1:1")
        )


def cmd_list(args):
    archive = d3cutscene.CutsceneCPK(args.archive)
    entries = _selected_entries(args, archive)
    if not entries:
        print("(no matching entries)")
        return
    groups_by_start = {}
    if args.groups:
        groups_by_start = {
            group.start: group for group in archive.scan_ogg_groups()
        }
    if args.long:
        if args.groups:
            print(f"{'idx':>3} {'loc':>3} {'group':>5} {'offset':>10} {'size':>10}  name")
        else:
            print(f"{'idx':>3} {'loc':>3} {'offset':>10} {'size':>10}  name")
    for entry in entries:
        if args.long:
            if args.groups:
                group = groups_by_start.get(entry.offset)
                group_text = "--" if group is None else str(group.index)
                print(
                    f"{entry.index:>3} {entry.loc_index:>3} {group_text:>5} "
                    f"{entry.offset:#010x} {entry.size:>10}  {entry.name}"
                )
            else:
                print(
                    f"{entry.index:>3} {entry.loc_index:>3} "
                    f"{entry.offset:#010x} {entry.size:>10}  {entry.name}"
                )
        else:
            print(entry.name)
    print(f"--- {len(entries)} entries, {sum(e.size for e in entries):,} B total ---")


def cmd_validate(args):
    archive = d3cutscene.CutsceneCPK(args.archive)
    entries = _selected_entries(args, archive)
    if not entries:
        print("(no matching entries)")
        return
    failures = 0
    t0 = time.time()
    for i, entry in enumerate(entries, 1):
        validation = archive.validate_entry(entry)
        if args.verbose or not validation.ok:
            status = "OK" if validation.ok else "FAIL"
            print(
                f"{status:4} idx={entry.index:02d} pages={validation.pages:5d} "
                f"streams={len(validation.streams):2d} {entry.name} - "
                f"{validation.status}"
            )
        if not validation.ok:
            failures += 1
        if args.progress and (i % 10 == 0 or i == len(entries)):
            print(f"progress: {i}/{len(entries)}")
    elapsed = time.time() - t0
    print(
        f"validated {len(entries)} entries in {elapsed:.1f}s; "
        f"failures={failures}"
    )
    if failures:
        sys.exit(1)


def cmd_extract(args):
    archive = d3cutscene.CutsceneCPK(args.archive)
    entries = _selected_entries(args, archive)
    if not entries:
        print("(no matching entries)")
        return
    os.makedirs(args.out, exist_ok=True)
    total = sum(entry.size for entry in entries)
    print(f"extracting {len(entries)} files ({total:,} B) -> {args.out}")
    t0 = time.time()
    failures = 0
    for i, entry in enumerate(entries, 1):
        if args.validate:
            validation = archive.validate_entry(entry)
            if not validation.ok:
                failures += 1
                print(f"FAIL {entry.name}: {validation.status}", file=sys.stderr)
                if not args.keep_going:
                    break
                continue
        out_path = archive.extract_entry(entry, args.out, flat=args.flat)
        if args.verbose:
            print(f"wrote {out_path} ({entry.size:,} B)")
        if args.progress and (i % 10 == 0 or i == len(entries)):
            print(f"progress: {i}/{len(entries)}")
    elapsed = time.time() - t0
    print(f"done in {elapsed:.1f}s; failures={failures}")
    if failures:
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        prog="d3cutscene_tool",
        description="Inspect and extract Diablo III raw cutscene CPK archives.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("info")
    p.add_argument("archive")
    p.add_argument("--scan-ogg", action="store_true", help="scan Ogg groups")
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("list")
    p.add_argument("archive")
    p.add_argument("--long", "-l", action="store_true")
    p.add_argument("--groups", action="store_true", help="show Ogg group index")
    _add_filter_args(p)
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("validate")
    p.add_argument("archive")
    p.add_argument("--verbose", "-v", action="store_true")
    p.add_argument("--progress", action="store_true")
    _add_filter_args(p)
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("extract")
    p.add_argument("archive")
    p.add_argument("--out", "-o", required=True)
    p.add_argument("--flat", action="store_true", help="write basenames only")
    p.add_argument("--validate", action="store_true", help="validate before writing")
    p.add_argument("--keep-going", action="store_true")
    p.add_argument("--progress", action="store_true")
    p.add_argument("--verbose", "-v", action="store_true")
    _add_filter_args(p)
    p.set_defaults(func=cmd_extract)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
