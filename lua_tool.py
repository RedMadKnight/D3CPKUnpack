#!/usr/bin/env python3
"""
lua_tool - recover the Lua 5.1 scripts embedded in Diablo III CPK archives.

    info     <archive>...                    count scripts per file type
    extract  <archive>... --out DIR          write pseudo-Lua for every script
                                             [--ext pow con] [--iname REGEX]
                                             [--exact] [--asm] [--raw]
                                             [--keep-empty]

Output paths follow the internal names, so Power\\Foo.pow becomes
DIR/Power/Foo.pow.lua. --asm adds a .lst disassembly and --raw the original
.luac chunk. --exact writes source that compiles back to the same instructions,
with plainer names, in place of the readable pseudo-Lua. Scripts compiled from
an empty source are skipped by default.
"""
import argparse
import os
import re
import sys
from collections import Counter

import cpklib
import lua51
import luadec
import relua

SCRIPT_EXTENSIONS = ("pow", "cnv", "scr", "con")


def iter_scripts(ar, exts, pattern=None):
    """Yield (entry, offset, chunk) for each script in matching files."""
    for e in ar:
        if e.ext not in exts or (pattern and not pattern.search(e.name)):
            continue
        for offset, chunk in lua51.find_chunks(ar.read(e)):
            yield e, offset, chunk


def is_empty(chunk):
    return len(chunk.main.code) == 1 and not chunk.main.protos


def output_stem(out_dir, name):
    parts = [p for p in name.replace("\\", "/").split("/") if p]
    if not parts or any(p in (".", "..") or ":" in p for p in parts):
        raise ValueError(f"unsafe path in {name!r}")
    return os.path.join(out_dir, *parts)


def exact_source(chunk):
    """Compilable source, or None if the code fits no known compiler pattern."""
    try:
        return relua.render(relua.decompile(chunk))
    except relua.Fail:
        return None


def cmd_info(args):
    for path in args.archive:
        with cpklib.CPKArchive.open(path) as ar:
            scripts, empty, functions, instructions = Counter(), Counter(), 0, 0
            for e, _, chunk in iter_scripts(ar, SCRIPT_EXTENSIONS):
                (empty if is_empty(chunk) else scripts)[e.ext] += 1
                for fn in chunk.main.walk():
                    functions += 1
                    instructions += len(fn.code)
        print(f"  archive:      {path}")
        for ext in SCRIPT_EXTENSIONS:
            print(f"    .{ext}  {scripts[ext]:>6,} scripts, {empty[ext]:>6,} empty")
        print(f"  functions:    {functions:,}")
        print(f"  instructions: {instructions:,}")


def cmd_extract(args):
    pattern = re.compile(args.iname, re.IGNORECASE) if args.iname else None
    written = skipped = fallback = inexact = 0
    for path in args.archive:
        with cpklib.CPKArchive.open(path) as ar:
            for e, offset, chunk in iter_scripts(ar, args.ext, pattern):
                if is_empty(chunk) and not args.keep_empty:
                    skipped += 1
                    continue
                stem = output_stem(args.out, e.name)
                os.makedirs(os.path.dirname(stem), exist_ok=True)
                text = exact_source(chunk) if args.exact else None
                inexact += args.exact and text is None
                text = text or luadec.decompile(chunk)
                fallback += text.count("-- could not decompile")
                with open(stem + ".lua", "w", encoding="utf-8", newline="\n") as f:
                    f.write(text)
                if args.asm:
                    with open(stem + ".lst", "w", encoding="utf-8", newline="\n") as f:
                        f.write("\n".join(lua51.disassemble(chunk.main)) + "\n")
                if args.raw:
                    with open(stem + ".luac", "wb") as f:
                        f.write(ar.read(e)[offset:offset + chunk.size])
                written += 1
    print(f"  written: {written:,} scripts -> {args.out}")
    print(f"  skipped: {skipped:,} empty")
    if fallback:
        print(f"  {fallback:,} function(s) left as disassembly", file=sys.stderr)
    if inexact:
        print(f"  {inexact:,} script(s) written as pseudo-Lua, not exact", file=sys.stderr)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="lua_tool", description=__doc__.split("\n")[1])
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("info", help="count scripts per file type")
    p.add_argument("archive", nargs="+")
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("extract", help="write pseudo-Lua for every script")
    p.add_argument("archive", nargs="+")
    p.add_argument("--out", "-o", required=True)
    p.add_argument("--ext", nargs="+", default=list(SCRIPT_EXTENSIONS),
                   help="file types to search")
    p.add_argument("--iname", help="case-insensitive regex against full path")
    p.add_argument("--exact", action="store_true",
                   help="write source that recompiles to the same instructions")
    p.add_argument("--asm", action="store_true", help="also write a .lst listing")
    p.add_argument("--raw", action="store_true", help="also write the .luac chunk")
    p.add_argument("--keep-empty", action="store_true")
    p.set_defaults(func=cmd_extract)

    args = parser.parse_args(argv)
    try:
        args.func(args)
    except (ValueError, OSError) as ex:
        sys.exit(str(ex))


if __name__ == "__main__":
    main()
