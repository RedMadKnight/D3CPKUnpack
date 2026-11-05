"""Table-driven record layouts for eleven file types (``cov_layouts.tsv``) and the checks R1 to R6.

``parse_structure`` reads a whole file as nested dicts, field names from the table. Raw values
are kept; ``StructuredFile.named()`` gives the geometry enum and flag names (stage 7, T8). ``verify_structure`` runs the coverage
checks: pairs inside the file and whole elements (R2), run-time slots, reserved ranges and
uncovered bytes zero (R3), and the variable area tiled by the pair blocks (R4).
"""

from __future__ import annotations

import csv
import struct
from dataclasses import dataclass, field
from functools import lru_cache

from .binary import ByteView, FormatError, Pair
from .datafiles import DATA
from .power import TagMap, read_tag_map
from .sno import SnoHeader, parse_header, resolve_build

EXTENSIONS = ("app", "ani", "ans", "scn", "phm", "tex", "snd", "sbk", "wrl", "mrk", "mat")
TOP_RECORDS = {
    "app": "AppearanceDefinition", "ani": "AnimationDefinition", "ans": "AnimSetDefinition",
    "scn": "SceneDefinition", "phm": "PhysMeshDefinition", "tex": "TextureDefinition",
    "snd": "SoundDefinition", "sbk": "SoundBankDefinition", "wrl": "WorldDefinition",
    "mrk": "MarkerSetDefinition", "mat": "MaterialDefinition",
}
# Value arrays a pair may point at: element format.
VALUE_ARRAYS = {"BYTE": "B", "WORD": "H", "INT": "i", "SNO": "i", "VECTOR3D": "3f"}
# Primitive types of the table: struct format of one element.
_PRIM_FORMATS = {
    "i": "i", "I": "I", "f": "f", "H": "H", "B": "B", "2H": "2H", "2f": "2f", "2i": "2i",
    "3f": "3f", "4B": "4B", "4f": "4f", "h": "h", "b": "b", "q": "q",
}
# Rows of kind ``raw`` that hold known values (stage 7, T12): read them.
KIND_FIXES = {
    ("dmMeshNode", "lowerX"): ("prim", "h"), ("dmMeshNode", "lowerY"): ("prim", "h"),
    ("dmMeshNode", "lowerZ"): ("prim", "h"), ("dmMeshNode", "upperX"): ("prim", "h"),
    ("dmMeshNode", "upperY"): ("prim", "h"), ("dmMeshNode", "upperZ"): ("prim", "h"),
    ("dmSubEdge", "twinOffset"): ("prim", "b"),
    ("AppearanceDefinition", "tUnwrapTimeStamp"): ("prim", "q"),
    ("DspEffect", "u8"): ("dsp", ""),
}
DSP_SLOTS = 16
DSP_SLOT_SIZE = 16
TILE_GAP = 16


@dataclass(frozen=True)
class LayoutField:
    offset: int
    size: int
    kind: str  # prim, struct, pair, ptr, zero, raw
    type: str
    points_to: str
    name: str


@dataclass(frozen=True)
class RecordLayout:
    build: str
    name: str
    size: int
    fields: tuple[LayoutField, ...]
    root_ext: str


@lru_cache(maxsize=None)
def layouts() -> dict[tuple[str, str], RecordLayout]:
    """Every record of ``cov_layouts.tsv`` and ``x360_cov_layouts.tsv`` by ``(build, record)``."""
    sizes: dict[tuple[str, str], tuple[int, str]] = {}
    fields: dict[tuple[str, str], list[LayoutField]] = {}
    rows: list[dict] = []
    for table in ("cov_layouts.tsv", "x360_cov_layouts.tsv"):
        with open(DATA / table, encoding="utf-8", newline="") as handle:
            rows += csv.DictReader(handle, delimiter="\t")
    for row in rows:
        key = (row["build"], row["record"])
        if row["kind"] == "RECORD SIZE":
            sizes[key] = (int(row["bytes"]), row["root_ext"])
            continue
        kind, kind_type = KIND_FIXES.get((row["record"], row["name"]), (row["kind"], row["type"]))
        fields.setdefault(key, []).append(LayoutField(
            int(row["offset"], 16), int(row["bytes"]), kind, kind_type,
            row["points_to"], row["name"],
        ))
    return {
        key: RecordLayout(key[0], key[1], size, tuple(sorted(fields.get(key, ()), key=lambda f: f.offset)), ext)
        for key, (size, ext) in sizes.items()
    }


@lru_cache(maxsize=None)
def _ps3_form_records() -> frozenset[str]:
    """Records of the ``.ani`` tree: x360 reads ``.ani`` in the ps3 form (X5)."""
    return frozenset(name for (build, name), lay in layouts().items() if build == "ps3" and lay.root_ext == "ani")


def layout(build: str, record: str) -> RecordLayout:
    """A record's layout. x360 has its own rows where ``x360_cov_layouts.tsv`` gives them (the
    scene, X4), the ps3 rows for the ``.ani`` tree (X5) and the Switch rows otherwise (X1)."""
    found = layouts().get((build, record))
    if found is None and build == "x360":
        found = layouts().get(("ps3" if record in _ps3_form_records() else "switch", record))
    if found is None:
        raise FormatError(f"no layout for {record} on {build}")
    return found


@dataclass(frozen=True)
class BitMember:
    name: str
    bits: int
    shift: int

    def extract(self, word: int) -> int:
        return (word >> self.shift) & ((1 << self.bits) - 1)


@lru_cache(maxsize=None)
def bitfields() -> dict[str, tuple[BitMember, ...]]:
    """Members of packed words by record (``cov_bitfields.tsv``)."""
    found: dict[str, list[BitMember]] = {}
    with open(DATA / "cov_bitfields.tsv", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            found.setdefault(row["record"], []).append(
                BitMember(row["member"], int(row["bits"]), int(row["shift_from_lsb"]))
            )
    return {key: tuple(sorted(value, key=lambda m: m.shift)) for key, value in found.items()}


# Which packed members belong to which word. The table lists the members of a record's union;
# for the three-word appearance sample the description names the words in order.
PACKED_WORDS = {
    ("AppearanceSample", "u0"): ("dwZFraction", "snoSurface"),
    ("AppearanceSample", "u4"): ("dwZFraction", "snoParticle"),
    ("AppearanceSample", "u8"): ("dwClutterDensity", "dwMask", "dwColor"),
    ("CompAndFormatUnion", "u0"): None,  # None: every member of the record
    ("SubObjectNameInfo", "u0"): None,
    ("dmMeshNode", "uc"): None,
    ("dmMeshNodeP3", "data"): "ps3",  # its own packing (stage 7, T7), see ``ps3_mesh_node``
}
# A handle field of an appearance sample with one of these values means none (T7).
NO_HANDLE = (0, 0xFFFFFF)


def sample_handle(value: int) -> int | None:
    """A surface or particle handle from an appearance sample word; None for none."""
    return None if value in NO_HANDLE else value


def ps3_mesh_node(word: int) -> dict[str, int]:
    """The PS3 collision node word: bits 0 to 1 axis (3 = leaf), bits 2 to 31 index.

    A leaf's triangle count is the node's ``a``; an internal node's ``a`` and ``b`` are its
    quantised split positions. The Switch packing (4-bit count, 26-bit index) does not apply.
    """
    return {"axis": word & 3, "index": word >> 2}


@lru_cache(maxsize=None)
def dsp_params() -> dict[int, tuple[str, tuple[str, ...]]]:
    """DSP effect type: its name and the parameter of each used slot (``dsp_params.tsv``)."""
    found: dict[int, tuple[str, list[str]]] = {}
    with open(DATA / "dsp_params.tsv", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            kind = int(row["dsp_type"])
            name, params = found.setdefault(kind, (row["type_name"], []))
            if int(row["slot"]) != len(params) or int(row["offset_in_effect"], 16) != 8 + 16 * len(params):
                raise ValueError(f"dsp_params.tsv: slot order for type {kind}")
            params.append(row["parameter"])
    return {kind: (name, tuple(params)) for kind, (name, params) in found.items()}


def unpack_bits(record: str, word: int, members: tuple[str, ...] | None = None) -> dict[str, int]:
    """Split a packed 32-bit word by the members ``cov_bitfields.tsv`` lists for ``record``."""
    found = {}
    for member in bitfields().get(record, ()):
        if members is None or member.name in members:
            found[member.name] = member.extract(word)
    return found


def _packed(record: str, name: str, word: int) -> dict[str, int] | None:
    key = (record, name)
    if key not in PACKED_WORDS:
        return None
    spec = PACKED_WORDS[key]
    if spec == "ps3":
        return ps3_mesh_node(word)
    if isinstance(spec, str):
        return unpack_bits(spec, word)
    return unpack_bits(record, word, spec)


@dataclass
class StructuredFile:
    header: SnoHeader
    build: str
    extension: str
    record: str
    root: dict
    blocks: list[tuple[int, int, str]] = field(default_factory=list)  # file offset, size, where
    records: list[tuple[str, str, dict]] = field(default_factory=list)  # every record read

    def named(self) -> list[tuple[str, str, dict[str, object]]]:
        """Geometry enum and flag names of every record that has any (stage 7, T8)."""
        found = []
        for record, where, values in self.records:
            names = geometry_names(record, values)
            if names:
                found.append((record, where, names))
        return found


class _Reader:
    def __init__(self, data: bytes, build: str, order: str) -> None:
        self.view = ByteView(data, order)
        self.build = build
        self.order = order
        self.blocks: list[tuple[int, int, str]] = []
        self.problems: list[str] = []
        self.depth = 0
        self.records: list[tuple[str, str, dict]] = []  # record name, where, values

    # --- values ---------------------------------------------------------------------------
    def prim(self, f: LayoutField, at: int):
        if f.type == "c":
            return self.view.cstring(at, f.size)
        fmt = _PRIM_FORMATS.get(f.type)
        if fmt is None:
            return self.view.take(at, f.size)
        one = struct.calcsize(self.order + fmt)
        count = f.size // one
        values = [self.view.unpack(fmt, at + i * one) for i in range(count)]
        values = [v[0] if len(v) == 1 else v for v in values]
        return values[0] if count == 1 else tuple(values)

    def record(self, name: str, at: int, where: str) -> dict:
        lay = layout(self.build, name)
        self.view.need(at, lay.size, name)
        self.depth += 1
        if self.depth > 64:
            raise FormatError("records nest too deep", at)
        result: dict = {}
        covered = bytearray(lay.size)
        for f in lay.fields:
            pos = at + f.offset
            covered[f.offset:f.offset + f.size] = b"\1" * min(f.size, lay.size - f.offset)
            label = f"{where}.{f.name}"
            if f.kind == "prim":
                value = self.prim(f, pos)
                result[f.name] = value
                packed = _packed(name, f.name, value) if isinstance(value, int) else None
                if packed is not None:
                    result[f.name + "_bits"] = packed
            elif f.kind == "struct":
                inner = layout(self.build, f.type)
                count = max(1, f.size // inner.size)
                items = [self.record(f.type, pos + i * inner.size, label) for i in range(count)]
                result[f.name] = items[0] if count == 1 else items
            elif f.kind == "pair":
                result[f.name] = self.pair(f, pos, label)
            elif f.kind == "dsp":
                result[f.name] = self.view.take(pos, f.size)
                result["parameters"] = self.dsp(result.get("eDspType"), pos, label)
            else:  # ptr, zero, raw: kept as bytes
                raw = self.view.take(pos, f.size)
                result[f.name] = raw
                if f.kind in ("ptr", "zero") and any(raw):
                    self.problems.append(f"R3: {label} ({f.kind}) at 0x{pos:X} is not zero")
        for offset, mark in enumerate(covered):
            if not mark and self.view.data[at + offset]:
                self.problems.append(
                    f"R3: {where} byte +0x{offset:X} at 0x{at + offset:X} is not zero"
                )
                break
        self.depth -= 1
        self.records.append((name, where, result))
        return result

    def dsp(self, kind, pos: int, where: str) -> dict[str, dict]:
        """The 16 parameter slots of a DSP effect, named by the type; unused slots must be zero."""
        known = isinstance(kind, int) and (kind in dsp_params() or kind == -1)
        used = dsp_params().get(kind, ("", ()))[1] if known else ()
        found = {}
        for slot in range(DSP_SLOTS):
            at = pos + slot * DSP_SLOT_SIZE
            if slot >= len(used):
                if known and any(self.view.take(at, DSP_SLOT_SIZE)):
                    self.problems.append(f"R3: {where} slot {slot} past the type's count is not zero")
                continue
            minimum, delta, flags, scaler = self.view.unpack("ffIi", at)
            found[used[slot]] = {"minimum": minimum, "delta": delta, "flags": flags, "source_scaler": scaler}
        return found

    def pair(self, f: LayoutField, pos: int, where: str):
        pair = self.view.pair(pos, where)
        target = f.points_to
        if pair.size == 0:
            if pair.offset and pair.file_offset > len(self.view.data):
                raise FormatError(f"{where}: offset past the end", pos)
            return [] if target.startswith("array:") else (TagMap(f.name, pair) if target == "tagmap:" else b"")
        raw = self.view.pair_data(pair, pos, where)
        self.blocks.append((pair.file_offset, pair.size, where))
        if target == "tagmap:":
            return read_tag_map(self.view, pair, f.name, pos, type_7=True)
        if target.startswith("array:"):
            element = target[6:]
            if element in VALUE_ARRAYS:
                fmt = VALUE_ARRAYS[element]
                size = struct.calcsize(self.order + fmt)
                if pair.size % size:
                    raise FormatError(f"R2: {where} size {pair.size} is not whole {element}", pos)
                values = [self.view.unpack(fmt, pair.file_offset + i) for i in range(0, pair.size, size)]
                return [v[0] if len(v) == 1 else v for v in values]
            inner = layout(self.build, element)
            if pair.size % inner.size:
                raise FormatError(
                    f"R2: {where} size {pair.size} is not a multiple of {inner.size}", pos,
                )
            return [
                self.record(element, pair.file_offset + i, f"{where}[{i // inner.size}]")
                for i in range(0, pair.size, inner.size)
            ]
        return raw


def _resolve(data: bytes, extension: str, build: str | None) -> tuple[SnoHeader, str, str]:
    extension = extension.lstrip(".").lower()
    if extension not in TOP_RECORDS:
        raise FormatError(f"no table layout for .{extension}")
    header = parse_header(data, build)
    return header, resolve_build(extension, header, build), extension


def _read(data: bytes, extension: str, build: str | None) -> tuple[StructuredFile, _Reader]:
    header, resolved, extension = _resolve(data, extension, build)
    reader = _Reader(data, resolved, header.order)
    name = TOP_RECORDS[extension]
    top = layout(resolved, name)
    if len(data) < 0x10 + top.size:
        raise FormatError(f"R1: {name} of {top.size} bytes does not fit", 0x10)
    root = reader.record(name, 0x10, name)
    return StructuredFile(header, resolved, extension, name, root, reader.blocks, reader.records), reader


def geometry_names(record: str, values: dict) -> dict[str, object]:
    """Names of a record's enum fields (the row's name, None without one) and flag words
    (``FlagNames``). Members of packed words (``*_bits``) are looked up too."""
    from .enums import geometry_enums, geometry_flags, geometry_value, is_geometry_flags

    tables = {name for (rec, name) in geometry_enums() if rec == record}
    if not tables:
        return {}
    flat = dict(values)
    for key, value in values.items():
        if key.endswith("_bits") and isinstance(value, dict):
            flat.update(value)
    found: dict[str, object] = {}
    for name in sorted(tables):
        value = flat.get(name)
        if not isinstance(value, int):
            continue
        if is_geometry_flags(record, name):
            found[name] = geometry_flags(record, name, value)
        else:
            row = geometry_value(record, name, value)
            found[name] = row.name if row else None
    return found


def parse_structure(data: bytes, extension: str, build: str | None = None) -> StructuredFile:
    """Read a whole file by its table layout. Offsets in pairs are file offset - 0x10."""
    return _read(data, extension, build)[0]


def tiling_problems(data_size: int, start: int, blocks: list[tuple[int, int, str]], data: bytes) -> list[str]:
    """R4: blocks cover ``start`` to the end exactly, up to zero gaps under 16 bytes."""
    problems = []
    unique = sorted({(offset, size) for offset, size, _ in blocks})
    cursor = start
    for offset, size in unique:
        if offset < cursor:
            problems.append(f"R4: block at 0x{offset:X} overlaps the one before (ends 0x{cursor:X})")
        else:
            gap = offset - cursor
            if gap >= TILE_GAP:
                problems.append(f"R4: gap of {gap} bytes at 0x{cursor:X}")
            elif any(data[cursor:offset]):
                problems.append(f"R4: gap at 0x{cursor:X} is not zero")
        cursor = max(cursor, offset + size)
    tail = data_size - cursor
    if tail < 0:
        problems.append("R4: a block runs past the end")
    elif tail >= TILE_GAP:
        problems.append(f"R4: {tail} bytes after the last block")
    elif any(data[cursor:]):
        problems.append("R4: bytes after the last block are not zero")
    return problems


@dataclass
class StructureReport:
    build: str
    record: str
    problems: list[str]

    @property
    def ok(self) -> bool:
        return not self.problems


def verify_structure(data: bytes, extension: str, build: str | None = None) -> StructureReport:
    """Run R1 to R4 on a file. A layout error (R1, R2) is reported, not raised."""
    try:
        parsed, reader = _read(data, extension, build)
    except FormatError as exc:
        header, resolved, extension = _resolve(data, extension, build)
        return StructureReport(resolved, TOP_RECORDS[extension], [str(exc)])
    top = layout(parsed.build, parsed.record)
    problems = list(reader.problems)
    problems += tiling_problems(len(data), 0x10 + top.size, reader.blocks, data)
    return StructureReport(parsed.build, parsed.record, problems)


def field_names(build: str, record: str) -> list[str]:
    return [f.name for f in layout(build, record).fields]


def overlapping_fields(build: str, record: str) -> list[tuple[str, str]]:
    """Pairs of fields of one record whose bytes overlap (none expected)."""
    found = []
    fields = layout(build, record).fields
    for a, b in zip(fields, fields[1:]):
        if b.offset < a.offset + a.size:
            found.append((a.name, b.name))
    return found
