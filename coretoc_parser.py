#!/usr/bin/env python3
"""
coretoc_parser.py — parser for Diablo III console CoreTOC.dat

Format (discovered empirically + confirmed via file size math):
  Big-endian, fixed record layout.

  uint32 BE:  count                        (number of SNO entries)
  for each of <count> entries (136 bytes):
    uint32 BE:  snoGroupID                 (index into SnoExtensions table)
    uint32 BE:  snoID                      (globally unique asset id)
    char[128]:  name (null-padded ASCII)

This differs from the PC CASC variant documented in jybp/casc, which uses
little-endian plus a 844-byte header with EntryCounts / EntryOffsets / Unks.
The console variant is flatter and byte-swapped.

Outputs:
  coretoc.csv       — groupID, snoID, extension, groupName, name
  coretoc.json      — same data, grouped by snoGroupID
  groups_summary.txt — human-readable group statistics

Usage:
  python3 coretoc_parser.py <CoreTOC.dat> [<output_dir>]
"""

import struct
import sys
import os
import json
import csv
from collections import defaultdict, Counter
from pathlib import Path


# SnoExtensions from jybp/casc (PC Diablo III) augmented with console-specific
# groups empirically identified from the file itself.
#
# Original PC groups (confirmed from Ladislav Zezula's / jybp's work):
SNO_EXTENSIONS_PC = {
    0:  ("", ""),
    1:  ("Actor", "acr"),
    2:  ("Adventure", "adv"),
    5:  ("AmbientSound", "ams"),
    6:  ("Anim", "ani"),
    7:  ("Anim2D", "an2"),
    8:  ("AnimSet", "ans"),
    9:  ("Appearance", "app"),
    11: ("Cloth", "clt"),
    12: ("Conversation", "cnv"),
    14: ("EffectGroup", "efg"),
    15: ("Encounter", "enc"),
    17: ("Explosion", "xpl"),
    19: ("Font", "fnt"),
    20: ("GameBalance", "gam"),
    21: ("Globals", "glo"),
    22: ("LevelArea", "lvl"),
    23: ("Light", "lit"),
    24: ("MarkerSet", "mrk"),
    25: ("Monster", "mon"),
    26: ("Observer", "obs"),
    27: ("Particle", "prt"),
    28: ("Physics", "phy"),
    29: ("Power", "pow"),
    31: ("Quest", "qst"),
    32: ("Rope", "rop"),
    33: ("Scene", "scn"),
    34: ("SceneGroup", "scg"),
    36: ("ShaderMap", "shm"),
    37: ("Shaders", "shd"),
    38: ("Shakes", "shk"),
    39: ("SkillKit", "skl"),
    40: ("Sound", "snd"),
    41: ("SoundBank", "sbk"),
    42: ("StringList", "stl"),
    43: ("Surface", "srf"),
    44: ("Textures", "tex"),
    45: ("Trail", "trl"),
    46: ("UI", "ui"),
    47: ("Weather", "wth"),
    48: ("Worlds", "wrl"),
    49: ("Recipe", "rcp"),
    51: ("Condition", "cnd"),
    56: ("Act", "act"),
    57: ("Material", "mat"),
    58: ("QuestRange", "qsr"),
    59: ("Lore", "lor"),
    60: ("Reverb", "rev"),
    61: ("PhysMesh", "phm"),
    62: ("Music", "mus"),
    63: ("Tutorial", "tut"),
    64: ("BossEncounter", "bos"),
    66: ("Accolade", "aco"),
}

# Console-only groups, identified by inspecting sample entries:
SNO_EXTENSIONS_CONSOLE_EXTRA = {
    3:  ("AiBehavior", "aib"),      # ConstantAttack, PoltahrEscortFollow, SkeletonKing
    4:  ("AiState", "ais"),          # *_Attack, *_Idle, etc.
    13: ("NpcRole", "npc"),          # A1C1DeckardCain, A1C1LostAdventurersLeader
    18: ("FlagSet", "flg"),          # GameFlags, PlayerFlags
    35: ("Encounter2", "en2"),       # Alternate encounter definitions (console-specific)
    52: ("TreasureClass", "trs"),    # MonsterDropSpellRune_Normal, LootClicky_Type1 ← CONFIRMED
    54: ("NpcRole2", "np2"),         # A2C2GreedyMiner, A2C1Poltahr
    55: ("Dungeon", "dun"),          # ZK Random Dungeon, Jar Of Souls, AlcarnusRitual
    65: ("Unknown65", "u65"),
    67: ("NpcExtra", "npx"),         # NPC, Player, NPC_AdditiveFlinch
    68: ("Misc68", "m68"),           # Sound-related "StaticShortHigh", Barbarian_WeaponThrow
}

# Merged map
SNO_EXTENSIONS = {**SNO_EXTENSIONS_PC, **SNO_EXTENSIONS_CONSOLE_EXTRA}


def parse_coretoc(data: bytes):
    """Return list of (snoGroupID, snoID, name) tuples."""
    if len(data) < 4:
        raise ValueError("file too small")

    count = struct.unpack('>I', data[:4])[0]
    expected_size = 4 + count * 136
    if expected_size != len(data):
        raise ValueError(
            f"File size mismatch: expected {expected_size} for count={count} "
            f"with 136-byte records, got {len(data)}. "
            f"This doesn't look like a console CoreTOC.dat."
        )

    entries = []
    for i in range(count):
        off = 4 + i * 136
        gid, sid = struct.unpack('>II', data[off:off+8])
        # Name is null-padded ASCII in the next 128 bytes
        name_bytes = data[off+8:off+136]
        null_pos = name_bytes.find(b'\x00')
        if null_pos == -1:
            null_pos = 128
        name = name_bytes[:null_pos].decode('latin-1', errors='replace')
        entries.append((gid, sid, name))
    return entries


def get_group_info(group_id):
    """Return (name, extension) for a given group ID."""
    return SNO_EXTENSIONS.get(group_id, (f"Unknown{group_id}", f"g{group_id:02d}"))


def export_csv(entries, out_path):
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['snoGroupID', 'snoID', 'extension', 'groupName', 'name'])
        for gid, sid, name in entries:
            gname, ext = get_group_info(gid)
            w.writerow([gid, sid, ext, gname, name])


def export_json(entries, out_path):
    groups = defaultdict(list)
    for gid, sid, name in entries:
        groups[gid].append({'snoID': sid, 'name': name})

    out = {
        'total_entries': len(entries),
        'groups': {},
    }
    for gid in sorted(groups.keys()):
        gname, ext = get_group_info(gid)
        out['groups'][str(gid)] = {
            'group_id': gid,
            'group_name': gname,
            'extension': ext,
            'count': len(groups[gid]),
            'entries': sorted(groups[gid], key=lambda e: e['snoID']),
        }

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


def export_summary(entries, out_path):
    group_counter = Counter(gid for gid, _, _ in entries)
    total = sum(group_counter.values())

    lines = []
    lines.append(f"=== CoreTOC.dat summary ===")
    lines.append(f"Total SNO entries: {total:,}")
    lines.append(f"Distinct groups:   {len(group_counter)}")
    lines.append("")
    lines.append(f"{'groupID':>8}  {'extension':>10}  {'groupName':<20}  {'count':>8}  {'%':>6}")
    lines.append("-" * 60)
    for gid, cnt in sorted(group_counter.items()):
        gname, ext = get_group_info(gid)
        pct = 100 * cnt / total
        marker = ""
        if gid in SNO_EXTENSIONS_CONSOLE_EXTRA:
            marker = " ← console-specific"
        elif gid not in SNO_EXTENSIONS_PC:
            marker = " ← UNKNOWN"
        lines.append(
            f"{gid:>8}  {'.'+ext:>10}  {gname:<20}  {cnt:>8}  {pct:>5.2f}%{marker}"
        )

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 coretoc_parser.py <CoreTOC.dat> [<output_dir>]")
        sys.exit(1)

    src = Path(sys.argv[1])
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("coretoc_out")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading {src}...")
    with open(src, 'rb') as f:
        data = f.read()
    print(f"  {len(data):,} B")

    print(f"Parsing CoreTOC format (console big-endian)...")
    entries = parse_coretoc(data)
    print(f"  Parsed {len(entries):,} SNO entries")

    csv_path = out_dir / "coretoc.csv"
    json_path = out_dir / "coretoc.json"
    summary_path = out_dir / "groups_summary.txt"

    print(f"Writing {csv_path}")
    export_csv(entries, csv_path)

    print(f"Writing {json_path}")
    export_json(entries, json_path)

    print(f"Writing {summary_path}")
    export_summary(entries, summary_path)

    # Quick stats on console
    group_counter = Counter(gid for gid, _, _ in entries)
    print(f"\nTop 10 groups:")
    for gid, cnt in group_counter.most_common(10):
        gname, ext = get_group_info(gid)
        print(f"  [{gid:3}] {gname:18s} (.{ext}): {cnt:,}")

    print(f"\nDone. Output directory: {out_dir}/")


if __name__ == "__main__":
    main()
