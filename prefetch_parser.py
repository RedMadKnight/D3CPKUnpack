#!/usr/bin/env python3
r"""
prefetch_parser.py — parser for Diablo III console Prefetch.dat

Format (reverse-engineered empirically, big-endian throughout):

  SECTION 1: 16-byte parent records
    uint32 BE: unknown_32          (engine metric — size/priority/handle?)
    uint32 BE: snoGroupID          (type of the parent asset)
    uint32 BE: snoID               (ID of the parent asset)
    uint32 BE: dep_count           (number of dependencies this parent triggers)

    The first uint32 of the file (= unknown_32 of record 0) also happens to
    equal the count of SECTION 1 records.

  SECTION 2 HEADER: 8 bytes
    uint32 BE: unknown_32          (possibly sum/count of something)
    uint32 BE: total_deps_count    (total # of 8-byte records that follow,
                                    equals sum of all dep_counts from Section 1)

  SECTION 2 BODY: 8-byte dependency records
    uint32 BE: snoGroupID
    uint32 BE: snoID

    Dependencies are consumed sequentially per Section 1 record, i.e. record
    i's deps are at positions [sum(dep_counts[0..i-1]), sum(dep_counts[0..i])).

Semantically this is a prefetch dependency graph: "when the engine loads
asset X, it should also preload assets Y, Z, W...". Confirmed by spot-checks:
  - AnimSet\Monk_Male pulls 406 deps (textures, actors, sounds for the
    full monk character package)
  - Power\EmoteWait pulls 1 dep: Textures\Overlay_sandDeath
  - Globals\globals pulls 371 deps (level lighting, player lights, textures)

Verified invariants on the input file:
  - b (groupID field) matches CoreTOC.groupID for 77,186/77,186 parent records
    (100%)
  - sum(dep_count) == section 2 header count (376,385 == 376,385)
  - All 113,380-range snoIDs resolve to valid names when cross-referenced
    with CoreTOC.dat

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


def parse_coretoc(path):
    """Parse console CoreTOC.dat → dict[snoID] = (groupID, name)."""
    with open(path, 'rb') as f:
        data = f.read()
    count = struct.unpack('>I', data[:4])[0]
    if 4 + count * 136 != len(data):
        raise ValueError(f"CoreTOC size mismatch: expected {4+count*136}, got {len(data)}")
    out = {}
    for i in range(count):
        off = 4 + i * 136
        gid, sid = struct.unpack('>II', data[off:off+8])
        name_bytes = data[off+8:off+136]
        null_pos = name_bytes.find(b'\x00')
        if null_pos == -1: null_pos = 128
        name = name_bytes[:null_pos].decode('latin-1', errors='replace')
        out[sid] = (gid, name)
    return out


def parse_prefetch(path):
    """Parse console Prefetch.dat → list of (parent_sno, parent_gid, deps)."""
    with open(path, 'rb') as f:
        data = f.read()

    parent_count = struct.unpack('>I', data[:4])[0]
    section1_end = parent_count * 16
    if section1_end + 8 > len(data):
        raise ValueError("file too small for declared parent count")

    # Parse section 1
    parents = []
    for i in range(parent_count):
        off = i * 16
        unk, gid, sno, dep_count = struct.unpack('>IIII', data[off:off+16])
        parents.append({
            'unknown32': unk,
            'snoGroupID': gid,
            'snoID': sno,
            'dep_count': dep_count,
        })

    # Parse section 2 header + pairs
    sec2_unk, sec2_count = struct.unpack('>II', data[section1_end:section1_end+8])
    expected_sec2_bytes = 8 + sec2_count * 8
    if section1_end + expected_sec2_bytes != len(data):
        raise ValueError(
            f"Section 2 size mismatch: "
            f"expected {section1_end+expected_sec2_bytes}, got {len(data)}"
        )

    deps_flat = []
    for i in range(sec2_count):
        off = section1_end + 8 + i * 8
        g, s = struct.unpack('>II', data[off:off+8])
        deps_flat.append((g, s))

    # Validate: sum of dep_counts must equal section 2 count
    total_declared = sum(p['dep_count'] for p in parents)
    if total_declared != sec2_count:
        raise ValueError(
            f"Dependency count mismatch: parents declare {total_declared} deps, "
            f"section 2 has {sec2_count} records"
        )

    # Consume deps sequentially per parent
    cursor = 0
    graph = []
    for p in parents:
        dc = p['dep_count']
        graph.append({
            **p,
            'deps': deps_flat[cursor:cursor+dc],
        })
        cursor += dc

    return graph, {
        'parent_count': parent_count,
        'total_deps': sec2_count,
        'section2_unknown': sec2_unk,
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
            pname = sno_map.get(p['snoID'], (None, '???'))[1]
            for dgid, dsno in p['deps']:
                dname = sno_map.get(dsno, (None, '???'))[1]
                w.writerow([
                    p['snoID'], p['snoGroupID'], pname,
                    dsno, dgid, dname,
                ])


def export_parents_json(graph, sno_map, out_path):
    out = []
    for p in graph:
        pname = sno_map.get(p['snoID'], (None, '???'))[1] if sno_map else ''
        deps_out = []
        for dgid, dsno in p['deps']:
            dname = sno_map.get(dsno, (None, '???'))[1] if sno_map else ''
            deps_out.append({
                'snoGroupID': dgid,
                'snoID': dsno,
                'name': dname,
            })
        out.append({
            'snoID': p['snoID'],
            'snoGroupID': p['snoGroupID'],
            'name': pname,
            'unknown32': p['unknown32'],
            'dep_count': p['dep_count'],
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
    lines.append(f"Mystery sec2 header value: {meta['section2_unknown']:,}")
    lines.append("")
    lines.append(f"Average deps per parent:   {meta['total_deps']/meta['parent_count']:.2f}")
    lines.append(f"Max deps single parent:    {max(p['dep_count'] for p in graph)}")
    lines.append(f"Parents with 1 dep only:   {deps_per_parent_dist.get(1, 0):,}")
    lines.append("")

    lines.append("Top 15 parents by dep count:")
    for p in sorted_by_deps[:15]:
        pname = sno_map.get(p['snoID'], (None, '???'))[1]
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
    9: "Appearance", 11: "Cloth", 12: "Conversation", 13: "NpcRole",
    14: "EffectGroup", 15: "Encounter", 17: "Explosion", 18: "FlagSet",
    19: "Font", 20: "GameBalance", 21: "Globals", 22: "LevelArea",
    23: "Light", 24: "MarkerSet", 25: "Monster", 26: "Observer",
    27: "Particle", 28: "Physics", 29: "Power", 31: "Quest",
    32: "Rope", 33: "Scene", 34: "SceneGroup", 35: "Encounter2",
    36: "ShaderMap", 37: "Shaders", 38: "Shakes", 39: "SkillKit",
    40: "Sound", 41: "SoundBank", 42: "StringList", 43: "Surface",
    44: "Textures", 45: "Trail", 46: "UI", 47: "Weather",
    48: "Worlds", 49: "Recipe", 51: "Condition", 52: "TreasureClass",
    54: "NpcRole2", 55: "Dungeon", 56: "Act", 57: "Material",
    58: "QuestRange", 59: "Lore", 60: "Reverb", 61: "PhysMesh",
    62: "Music", 63: "Tutorial", 64: "BossEncounter", 66: "Accolade",
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
