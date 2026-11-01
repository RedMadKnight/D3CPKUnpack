#!/usr/bin/env python3
r"""
prefetch_parser.py — parser for Diablo III console Prefetch.dat

Format, in the build's byte order (big-endian on PS3, little on Switch):

  uint32: parent_count
  <parent_count> 16-byte parent records:
    uint32: snoGroupID             (type of the parent asset)
    uint32: snoID                  (ID of the parent asset)
    uint32: dep_count              (number of dependencies of this parent)
    uint32: first_dep              (index of its first dependency)
  uint32: total_deps_count         (sum of all dep_counts)
  <total_deps_count> 8-byte dependency records:
    uint32: snoGroupID
    uint32: snoID

A parent's dependencies are records [first_dep, first_dep + dep_count). The
ranges partition the list without gaps or overlaps, but not in parent order.

Semantically this is a prefetch dependency graph: "when the engine loads
asset X, it should also preload assets Y, Z, W...". For example,
AnimSet\Monk_Male pulls its 406 animations.

Verified on the PS3 and Switch files: every byte is accounted for, and the
ranges partition the list. Dependencies need not resolve in CoreTOC.dat: 420
of 376,385 do not on PS3, nor do 24,872 of 844,045 on Switch.

Outputs:
  prefetch.csv          — one row per dependency edge (parent → child)
  prefetch_parents.json — each parent with its full list of dependencies
  prefetch_summary.txt  — human-readable statistics

Usage:
  python3 prefetch_parser.py <Prefetch.dat> <CoreTOC.dat> [<output_dir>]

CoreTOC.dat is optional — without it you get IDs only (no names).
"""

import struct
import sys
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import coretoc_parser


def parse_coretoc(path):
    """Parse CoreTOC.dat → dict[(groupID, snoID)] = (groupID, name).

    Handles are unique only within a group, so the group is part of the key.
    """
    with open(path, 'rb') as f:
        entries = coretoc_parser.parse_coretoc(f.read())
    return {(gid, sid): (gid, name) for gid, sid, name in entries}


def parse_prefetch(path):
    """Parse Prefetch.dat → (parent records with their deps, totals)."""
    with open(path, 'rb') as f:
        data = f.read()

    for endian in '><':
        parent_count = struct.unpack_from(endian + 'I', data, 0)[0]
        deps_pos = 4 + parent_count * 16 + 4
        if deps_pos > len(data):
            continue
        total = struct.unpack_from(endian + 'I', data, deps_pos - 4)[0]
        if deps_pos + total * 8 == len(data):
            break
    else:
        raise ValueError("size does not fit a Prefetch.dat in either byte order")

    parents = []
    for i in range(parent_count):
        gid, sno, dep_count, first_dep = struct.unpack_from(
            endian + 'IIII', data, 4 + i * 16)
        parents.append({
            'snoGroupID': gid,
            'snoID': sno,
            'dep_count': dep_count,
            'first_dep': first_dep,
        })

    deps_flat = list(struct.iter_unpack(endian + 'II', data[deps_pos:]))

    declared = sum(p['dep_count'] for p in parents)
    if declared != total:
        raise ValueError(
            f"Dependency count mismatch: parents declare {declared} deps, "
            f"the list has {total} records"
        )

    position = 0
    for p in sorted(parents, key=lambda p: p['first_dep']):
        if not p['dep_count']:
            continue
        if p['first_dep'] != position:
            raise ValueError(
                f"Dependency ranges do not partition the list at {position}")
        position += p['dep_count']

    graph = [{
        **p,
        'deps': deps_flat[p['first_dep']:p['first_dep'] + p['dep_count']],
    } for p in parents]

    return graph, {
        'parent_count': parent_count,
        'total_deps': total,
    }


def export_csv(graph, sno_map, out_path):
    """Export flat edge list: one row per (parent, dep) pair."""
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow([
            'parent_snoID', 'parent_groupID', 'parent_name',
            'dep_snoID', 'dep_groupID', 'dep_name',
        ])
        for p in graph:
            pname = sno_map.get((p['snoGroupID'], p['snoID']), (None, '???'))[1]
            for dgid, dsno in p['deps']:
                dname = sno_map.get((dgid, dsno), (None, '???'))[1]
                w.writerow([
                    p['snoID'], p['snoGroupID'], pname,
                    dsno, dgid, dname,
                ])


def export_parents_json(graph, sno_map, out_path):
    out = []
    for p in graph:
        pname = sno_map.get((p['snoGroupID'], p['snoID']), (None, '???'))[1] if sno_map else ''
        deps_out = []
        for dgid, dsno in p['deps']:
            dname = sno_map.get((dgid, dsno), (None, '???'))[1] if sno_map else ''
            deps_out.append({
                'snoGroupID': dgid,
                'snoID': dsno,
                'name': dname,
            })
        out.append({
            'snoID': p['snoID'],
            'snoGroupID': p['snoGroupID'],
            'name': pname,
            'dep_count': p['dep_count'],
            'first_dep': p['first_dep'],
            'deps': deps_out,
        })
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({'parents': out}, f, indent=2, ensure_ascii=False)


def export_summary(graph, meta, sno_map, out_path):
    # Top parents by dep count
    sorted_by_deps = sorted(graph, key=lambda p: -p['dep_count'])

    # Group-level stats
    parent_groups = Counter(p['snoGroupID'] for p in graph)
    dep_groups = Counter(dgid for p in graph for dgid, _ in p['deps'])
    deps_per_parent_dist = Counter(p['dep_count'] for p in graph)

    # Unique deps
    unique_dep_snos = set(dsno for p in graph for _, dsno in p['deps'])

    lines = []
    lines.append("=== Prefetch.dat summary ===")
    lines.append(f"Parent entries:            {meta['parent_count']:,}")
    lines.append(f"Total dependency edges:    {meta['total_deps']:,}")
    lines.append(f"Unique dep snoIDs:         {len(unique_dep_snos):,}")
    lines.append("")
    lines.append(f"Average deps per parent:   {meta['total_deps']/meta['parent_count']:.2f}")
    lines.append(f"Max deps single parent:    {max(p['dep_count'] for p in graph)}")
    lines.append(f"Parents with 1 dep only:   {deps_per_parent_dist.get(1, 0):,}")
    lines.append("")

    lines.append("Top 15 parents by dep count:")
    for p in sorted_by_deps[:15]:
        pname = sno_map.get((p['snoGroupID'], p['snoID']), (None, '???'))[1]
        gname = get_group_name(p['snoGroupID'])
        lines.append(f"  {p['dep_count']:>4}  {gname:>14}  {pname}")
    lines.append("")

    lines.append("Parent distribution by group:")
    for gid, cnt in sorted(parent_groups.items(), key=lambda x: -x[1]):
        gname = get_group_name(gid)
        pct = 100 * cnt / meta['parent_count']
        lines.append(f"  [{gid:>3}] {gname:>16}: {cnt:>6} ({pct:>5.2f}%)")
    lines.append("")

    lines.append("Dep distribution by group:")
    for gid, cnt in sorted(dep_groups.items(), key=lambda x: -x[1])[:20]:
        gname = get_group_name(gid)
        pct = 100 * cnt / meta['total_deps']
        lines.append(f"  [{gid:>3}] {gname:>16}: {cnt:>7} ({pct:>5.2f}%)")
    lines.append("")

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


# Group name mapping (merged PC + console)
GROUP_NAMES = {
    1: "Actor", 2: "Adventure", 3: "AiBehavior", 4: "AiState",
    5: "AmbientSound", 6: "Anim", 7: "Anim2D", 8: "AnimSet",
    9: "Appearance", 11: "Cloth", 12: "Conversation", 13: "ConversationList",
    14: "EffectGroup", 15: "Encounter", 17: "Explosion", 18: "FlagSet",
    19: "Font", 20: "GameBalance", 21: "Globals", 22: "LevelArea",
    23: "Light", 24: "MarkerSet", 25: "Monster", 26: "Observer",
    27: "Particle", 28: "Physics", 29: "Power", 31: "Quest",
    32: "Rope", 33: "Scene", 34: "SceneGroup", 35: "Script",
    36: "ShaderMap", 37: "Shaders", 38: "Shakes", 39: "SkillKit",
    40: "Sound", 41: "SoundBank", 42: "StringList", 43: "Surface",
    44: "Textures", 45: "Trail", 46: "UI", 47: "Weather",
    48: "Worlds", 49: "Recipe", 51: "Condition", 52: "TreasureClass",
    54: "Conductor", 55: "TimedEvent", 56: "Act", 57: "Material",
    58: "QuestRange", 59: "Lore", 60: "Reverb", 61: "PhysMesh",
    62: "Music", 63: "Tutorial", 64: "BossEncounter", 65: "ControlScheme",
    66: "Accolade", 67: "AnimTree", 68: "Vibrations", 69: "DungeonFinder",
}


def get_group_name(gid):
    return GROUP_NAMES.get(gid, f"Group{gid}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 prefetch_parser.py <Prefetch.dat> "
              "[<CoreTOC.dat>] [<output_dir>]")
        sys.exit(1)

    prefetch_path = Path(sys.argv[1])
    coretoc_path = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    out_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("prefetch_out")
    out_dir.mkdir(parents=True, exist_ok=True)

    sno_map = {}
    if coretoc_path and coretoc_path.exists():
        print(f"Loading CoreTOC from {coretoc_path}...")
        sno_map = parse_coretoc(coretoc_path)
        print(f"  Loaded {len(sno_map):,} SNO entries")
    else:
        print("(No CoreTOC provided — names will be blank)")

    print(f"Parsing {prefetch_path}...")
    graph, meta = parse_prefetch(prefetch_path)
    print(f"  {meta['parent_count']:,} parents, {meta['total_deps']:,} deps")

    csv_path = out_dir / "prefetch.csv"
    json_path = out_dir / "prefetch_parents.json"
    summary_path = out_dir / "prefetch_summary.txt"

    print(f"Writing {csv_path}")
    export_csv(graph, sno_map, csv_path)

    print(f"Writing {json_path}")
    export_parents_json(graph, sno_map, json_path)

    print(f"Writing {summary_path}")
    export_summary(graph, meta, sno_map, summary_path)

    print(f"\nDone. Output directory: {out_dir}/")


if __name__ == "__main__":
    main()
