"""CoreTOC, Prefetch, descriptor sizes, manifests, sprite tables and subtitles."""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass, field

from .binary import ByteView, FormatError, build_order
from .sno import hash33

GROUP_SLOTS = 70
SWITCH_TOC_HEADER = GROUP_SLOTS * 3 * 4 + 4
DESCRIPTOR_MAGIC = 0xAABBCCDD
MANIFEST_MAGIC = 0xAABB0004
SPRITE_MAGIC = 0xBBCC0005


@dataclass(frozen=True)
class TocEntry:
    group: int
    handle: int
    name: str


@dataclass
class CoreToc:
    build: str
    entries: list[TocEntry]
    unknown: int | None = None  # switch: the open trailing word of the header

    def keys(self) -> set[tuple[int, int]]:
        return {(e.group, e.handle) for e in self.entries}


def _is_ps3_toc(data: bytes) -> bool:
    if len(data) < 4:
        return False
    return len(data) == 4 + struct.unpack_from(">I", data, 0)[0] * 136


def parse_core_toc(data: bytes, build: str | None = None) -> CoreToc:
    """Read either form. Without ``build`` the file size decides."""
    if build is None:
        build = "ps3" if _is_ps3_toc(data) else "switch"
    else:
        build_order(build)
    return _parse_toc_ps3(data) if build == "ps3" else _parse_toc_switch(data)


def _parse_toc_ps3(data: bytes) -> CoreToc:
    view = ByteView(data, ">")
    count = view.u32(0, "CoreTOC count")
    if len(data) != 4 + count * 136:
        raise FormatError(f"CoreTOC size {len(data)} does not fit count {count}", 0)
    entries = []
    for i in range(count):
        pos = 4 + i * 136
        group, handle = view.unpack("II", pos)
        entries.append(TocEntry(group, handle, view.cstring(pos + 8, 128)))
    return CoreToc("ps3", entries)


def _parse_toc_switch(data: bytes) -> CoreToc:
    view = ByteView(data, "<")
    view.need(0, SWITCH_TOC_HEADER, "CoreTOC header")
    counts = view.unpack(f"{GROUP_SLOTS}i", 0)
    offsets = view.unpack(f"{GROUP_SLOTS}i", GROUP_SLOTS * 4)
    counts2 = view.unpack(f"{GROUP_SLOTS}i", GROUP_SLOTS * 8)
    unknown = view.u32(GROUP_SLOTS * 12)
    entries = []
    for g in range(GROUP_SLOTS):
        if counts[g] != counts2[g]:
            raise FormatError(f"group {g}: count {counts[g]} != count2 {counts2[g]}", g * 4)
        if counts[g] < 0:
            raise FormatError(f"group {g}: negative count", g * 4)
        if counts[g] == 0:
            continue
        if offsets[g] < 0:
            raise FormatError(f"group {g}: negative offset", (GROUP_SLOTS + g) * 4)
        base = SWITCH_TOC_HEADER + offsets[g]
        names_base = base + counts[g] * 12
        view.need(base, counts[g] * 12, "CoreTOC records")
        for i in range(counts[g]):
            pos = base + i * 12
            group, handle, name_offset = view.unpack("iii", pos)
            if name_offset < 0:
                raise FormatError("negative name offset", pos + 8)
            start = names_base + name_offset
            end = data.find(b"\0", start)
            if start >= len(data) or end < 0:
                raise FormatError("CoreTOC name has no terminator", pos + 8)
            entries.append(TocEntry(group, handle, bytes(data[start:end]).decode("latin-1")))
    return CoreToc("switch", entries, unknown)


@dataclass(frozen=True)
class PrefetchParent:
    group: int
    handle: int
    dep_count: int
    first_dep: int


@dataclass
class Prefetch:
    build: str
    parents: list[PrefetchParent]
    deps: list[tuple[int, int]]

    def dependencies(self, parent: PrefetchParent) -> list[tuple[int, int]]:
        return self.deps[parent.first_dep:parent.first_dep + parent.dep_count]

    @property
    def parents_sorted(self) -> bool:
        """Switch parents are sorted by the low 16 bits of the handle."""
        keys = [p.handle & 0xFFFF for p in self.parents]
        return all(a <= b for a, b in zip(keys, keys[1:]))

    def unresolved(self, toc: CoreToc) -> int:
        """Dependencies missing from CoreTOC. Some are expected on both builds."""
        known = toc.keys()
        return sum(1 for key in self.deps if key not in known)


def _parse_prefetch_order(data: bytes, build: str) -> Prefetch:
    view = ByteView(data, build_order(build))
    parent_count = view.u32(0, "parent count")
    view.need(4, parent_count * 16, "Prefetch parents")
    parents = [
        PrefetchParent(*view.unpack("4I", 4 + i * 16)) for i in range(parent_count)
    ]
    pos = 4 + parent_count * 16
    dep_total = view.u32(pos, "dependency total")
    pos += 4
    if len(data) != pos + dep_total * 8:
        raise FormatError(f"size {len(data)} does not fit {dep_total} dependencies", pos - 4)
    declared = sum(p.dep_count for p in parents)
    if declared != dep_total:
        raise FormatError(f"depCount sum {declared} != depTotal {dep_total}", pos - 4)
    deps = [view.unpack("II", pos + i * 8) for i in range(dep_total)]
    cursor = 0
    for parent in sorted(parents, key=lambda p: p.first_dep):
        if parent.dep_count and parent.first_dep != cursor:
            kind = "gap" if parent.first_dep > cursor else "overlap"
            raise FormatError(f"dependency ranges have a {kind} at {cursor}", 4)
        cursor += parent.dep_count
    return Prefetch(build, parents, deps)


def parse_prefetch(data: bytes, build: str | None = None) -> Prefetch:
    """Read Prefetch. Without ``build``, the byte order that fits the file size is used."""
    if build is not None:
        return _parse_prefetch_order(data, build)
    failures = []
    results = []
    for candidate in ("ps3", "switch"):
        try:
            results.append(_parse_prefetch_order(data, candidate))
        except FormatError as exc:
            failures.append(exc)
    if len(results) == 1:
        return results[0]
    if results:
        raise FormatError("Prefetch fits both byte orders; pass a build", 0)
    raise failures[0]


@dataclass(frozen=True)
class DescriptorSize:
    name: str
    size: int
    words: tuple[int, ...]  # open: exposed raw

    def fits_smallest_file(self, smallest_file: int) -> bool:
        """For an SNO type, the smallest file is at least ``size + 16``."""
        return smallest_file >= self.size + 16


def parse_descriptor_sizes(data: bytes) -> list[DescriptorSize]:
    view = ByteView(data, "<")
    magic = view.u32(0, "descriptor magic")
    if magic != DESCRIPTOR_MAGIC:
        raise FormatError(f"bad descriptor magic 0x{magic:08X}", 0)
    body = len(data) - 4
    if body % 1092:
        raise FormatError(f"descriptor body of {body} bytes is not a multiple of 1092", 4)
    entries = []
    for pos in range(4, len(data), 1092):
        name = view.cstring(pos, 64)
        size = view.u32(pos + 64)
        entries.append(DescriptorSize(name, size, view.unpack("256I", pos + 68)))
    return entries


def _magic_order(data: bytes, magic: int, what: str) -> ByteView:
    if len(data) < 4:
        raise FormatError(f"{what} too short", 0)
    if bytes(data[:4]) == magic.to_bytes(4, "big"):
        return ByteView(data, ">")
    if bytes(data[:4]) == magic.to_bytes(4, "little"):
        return ByteView(data, "<")
    raise FormatError(f"bad {what} magic", 0)


@dataclass
class Manifest:
    order: str
    entries: list[tuple[int, int]]


def parse_manifest(data: bytes) -> Manifest:
    view = _magic_order(data, MANIFEST_MAGIC, "manifest")
    if (len(data) - 4) % 8:
        raise FormatError("manifest body is not a multiple of 8", 4)
    entries = [view.unpack("II", pos) for pos in range(4, len(data), 8)]
    return Manifest(view.order, entries)


@dataclass(frozen=True)
class Sprite:
    name_hash: int
    texture_handle: int
    frame: int


@dataclass
class SpriteTable:
    order: str
    sprites: list[Sprite]

    def problems(self) -> list[str]:
        """Check that handles are sorted and frames run from 0 per texture."""
        found = []
        handles = [s.texture_handle for s in self.sprites]
        if handles != sorted(handles):
            found.append("texture handles are not sorted")
        frames: dict[int, list[int]] = {}
        for sprite in self.sprites:
            frames.setdefault(sprite.texture_handle, []).append(sprite.frame)
        for handle, values in frames.items():
            if sorted(values) != list(range(len(values))):
                found.append(f"texture {handle}: frames do not run from 0")
        return found

    def find(self, name: str) -> Sprite | None:
        target = hash33(name)
        return next((s for s in self.sprites if s.name_hash == target), None)


def parse_sprite_table(data: bytes) -> SpriteTable:
    view = _magic_order(data, SPRITE_MAGIC, "sprite table")
    if (len(data) - 4) % 12:
        raise FormatError("sprite body is not a multiple of 12", 4)
    sprites = [Sprite(*view.unpack("III", pos)) for pos in range(4, len(data), 12)]
    return SpriteTable(view.order, sprites)


def dat_kind(data: bytes) -> str | None:
    """Classify a ``.dat`` by its first word, in either byte order."""
    head = bytes(data[:4])
    for magic, kind in (
        (MANIFEST_MAGIC, "manifest"),
        (SPRITE_MAGIC, "sprite"),
        (DESCRIPTOR_MAGIC, "descriptor"),
    ):
        if head in (magic.to_bytes(4, "big"), magic.to_bytes(4, "little")):
            return kind
    return None


@dataclass
class Subtitle:
    start: str | None
    end: str | None
    lines: list[str] = field(default_factory=list)


_CUE = re.compile(r"^(\d\d:\d\d:\d\d:\d\d) - (\d\d:\d\d:\d\d:\d\d)$")


def parse_subtitles(data: bytes) -> list[Subtitle]:
    """Read ``.sbt``. A block with no time code has ``start`` and ``end`` of None."""
    if bytes(data[:3]) == b"\xEF\xBB\xBF":
        data = data[3:]
    try:
        text = bytes(data).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise FormatError("subtitle text is not UTF-8", exc.start) from None
    blocks: list[Subtitle] = []
    current: Subtitle | None = None
    for line in re.split(r"\r?\n", text):
        if not line.strip():
            current = None
            continue
        match = _CUE.match(line)
        if match:
            current = Subtitle(match.group(1), match.group(2))
            blocks.append(current)
        elif current is None:
            current = Subtitle(None, None, [line])
            blocks.append(current)
        else:
            current.lines.append(line)
    return blocks
